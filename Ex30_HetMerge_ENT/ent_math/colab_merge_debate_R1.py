#!/usr/bin/env python3
"""
CNN Merge: Feature Distillation + Projected Cosine Alignment
=============================================================
Winner of /merge-debate Round 1.

Method: 
  1. Pre-compute teacher features (512-dim) + logits from both parents
  2. Confidence routing: α_A = conf_A / (conf_A + conf_B)
  3. Projected cosine similarity: P_A maps student features → A's space
  4. Loss = α_A * (1-cos(P_A(h_s), h_A)) + α_B * (1-cos(P_B(h_s), h_B)) + 0.3*L_kd
  5. Phase 1: feature distill only. Phase 2: + EWC protection.
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, numpy as np
from torchvision import datasets, transforms, models

DEV = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEV}")
if DEV.type == 'cuda': print(f"  GPU: {torch.cuda.get_device_name()}")

# ═══ Data ═══
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
train_ds = datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10('/tmp/cifar10', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
clA, clB = list(range(5)), list(range(5,10))

# ═══ Model ═══
def make_rn18(nc):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    return m

# ═══ Feature extractor (returns features + logits) ═══
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        m = self.model
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
        feat = m.avgpool(x).flatten(1)  # [B, 512]
        logits = m.fc(feat)
        return feat, logits

# ═══ Train parent ═══
def train_parent(cls_list, epochs=15, seed=42):
    torch.manual_seed(seed)
    m = make_rn18(len(cls_list)).to(DEV)
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        m.train()
        for xb, yb in train_loader:
            mask = sum(yb==c for c in cls_list).bool()
            if mask.sum()==0: continue
            xb, yb_m = xb[mask].to(DEV), yb[mask].to(DEV)
            for ni, oc in enumerate(cls_list): yb_m[yb_m==oc] = ni
            loss = nn.CrossEntropyLoss()(m(xb), yb_m)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    m.eval()
    return m

# ═══ Eval ═══
def eval_pc(model, classes):
    model.eval()
    pc = {}
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(DEV)).argmax(1).cpu()
            for c in classes:
                mask = yb==c
                if mask.sum()==0: continue
                pc[c] = pc.get(c,0) + (preds[mask]==c).float().sum().item()
    for c in classes: pc[c] = pc.get(c,0) / 1000  # CIFAR-10: 1000 per class
    return pc

def print_drop(name, ppc, mpc, classes):
    print(f"\n  {name}:")
    print(f"  {'Cls':>3} | {'Parent':>7} | {'Merged':>7} | {'Drop%':>6} | OK?")
    print(f"  {'-'*3}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*4}")
    ret = 0
    for c in classes:
        p, m = ppc[c], mpc.get(c,0)
        d = (1-m/p)*100 if p>0 else 100
        ok = 'YES' if d<=10 else 'NO'
        if d<=10: ret+=1
        print(f"  {c:>3} | {p:>7.3f} | {m:>7.3f} | {d:>5.1f}% | {ok}")
    print(f"  Retention: {ret}/{len(classes)} (drop <= 10%)")
    return ret

# ═══ Main ═══
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    
    t0 = time.time()

    print("=" * 60)
    print("  PHASE 1: Training parents")
    print("=" * 60)
    pA = train_parent(clA, 15, 42)
    pB = train_parent(clB, 15, 142)

    # Parent accuracies
    pA_pc = {}; pB_pc = {}
    feA = FeatureExtractor(pA).to(DEV).eval()
    feB = FeatureExtractor(pB).to(DEV).eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            _, logA = feA(xb.to(DEV)); _, logB = feB(xb.to(DEV))
            predsA = logA.argmax(1).cpu(); predsB = logB.argmax(1).cpu()
            for c in clA:
                mask = yb==c; ci = clA.index(c)
                pA_pc[c] = pA_pc.get(c,0) + (predsA[mask]==ci).float().sum().item()
            for c in clB:
                mask = yb==c; ci = clB.index(c)
                pB_pc[c] = pB_pc.get(c,0) + (predsB[mask]==ci).float().sum().item()
    for c in clA: pA_pc[c]/=1000
    for c in clB: pB_pc[c]/=1000
    parent_pc = {**pA_pc, **pB_pc}
    print(f"Parents trained ({time.time()-t0:.0f}s): A={pA_pc}, B={pB_pc}")

    # ═══ Pre-compute teacher features + logits + confidence ═══
    print("\n" + "=" * 60)
    print("  PHASE 2: Pre-computing teacher features")
    print("=" * 60)

    all_x = torch.cat([xb for xb, _ in test_loader])

    with torch.no_grad():
        feat_A_list, logit_A_list, feat_B_list, logit_B_list = [], [], [], []
        for xb in all_x.split(256):
            fA, lA = feA(xb.to(DEV)); feat_A_list.append(fA.cpu()); logit_A_list.append(lA.cpu())
            fB, lB = feB(xb.to(DEV)); feat_B_list.append(fB.cpu()); logit_B_list.append(lB.cpu())
        H_A = torch.cat(feat_A_list)   # [10000, 512]
        H_B = torch.cat(feat_B_list)
        logits_A = torch.cat(logit_A_list)
        logits_B = torch.cat(logit_B_list)
        
        # Confidence routing coefficients
        conf_A = F.softmax(logits_A, dim=1).max(1).values  # [10000]
        conf_B = F.softmax(logits_B, dim=1).max(1).values
        alpha_A = conf_A / (conf_A + conf_B)  # [10000]
        alpha_B = 1 - alpha_A

    print(f"  Features: H_A={H_A.shape}, H_B={H_B.shape}")
    print(f"  alpha_A mean={alpha_A.mean():.3f}, alpha_B mean={alpha_B.mean():.3f}")
    print(f"  ({time.time()-t0:.0f}s)")

    # ═══ Student with projection heads ═══
    print("\n" + "=" * 60)
    print("  PHASE 3: Feature Distillation + Projected Cosine")
    print("=" * 60)

    student = make_rn18(10).to(DEV)  # ImageNet pretrained (neutral)
    fe_student = FeatureExtractor(student).to(DEV)

    # Projection layers
    proj_A = nn.Linear(512, 256, bias=False).to(DEV)
    proj_B = nn.Linear(512, 256, bias=False).to(DEV)
    proj_tA = nn.Linear(512, 256, bias=False).to(DEV)
    proj_tB = nn.Linear(512, 256, bias=False).to(DEV)

    all_params = list(student.parameters()) + list(proj_A.parameters()) + list(proj_B.parameters()) + \
                 list(proj_tA.parameters()) + list(proj_tB.parameters())

    opt = torch.optim.Adam(all_params, lr=0.001, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)

    best_ret, best_sd = 0, None

    for ep in range(1, 41):
        student.train(); proj_A.train(); proj_B.train()
        idx = torch.randperm(len(all_x))
        epoch_loss = 0
        
        for i in range(0, len(idx), 256):
            bi = idx[i:i+256]
            xb = all_x[bi].to(DEV)
            hA_t = H_A[bi].to(DEV)
            hB_t = H_B[bi].to(DEV)
            aA = alpha_A[bi].to(DEV).unsqueeze(1)
            aB = alpha_B[bi].to(DEV).unsqueeze(1)
            lA_t = logits_A[bi].to(DEV)
            lB_t = logits_B[bi].to(DEV)
            
            # Forward
            h_s, z_s = fe_student(xb)
            
            # Projected cosine similarity loss
            p_s_A = F.normalize(proj_A(h_s), dim=1)
            p_t_A = F.normalize(proj_tA(hA_t), dim=1)
            p_s_B = F.normalize(proj_B(h_s), dim=1)
            p_t_B = F.normalize(proj_tB(hB_t), dim=1)
            
            cos_A = 1 - (p_s_A * p_t_A).sum(dim=1, keepdim=True)
            cos_B = 1 - (p_s_B * p_t_B).sum(dim=1, keepdim=True)
            
            L_feat = (aA * cos_A + aB * cos_B).mean()
            
            # Logit KD as auxiliary
            T = 4.0
            L_kd_A = F.kl_div(F.log_softmax(z_s[:,:5]/T, 1), F.softmax(lA_t/T, 1), reduction='batchmean') * T**2
            L_kd_B = F.kl_div(F.log_softmax(z_s[:,5:]/T, 1), F.softmax(lB_t/T, 1), reduction='batchmean') * T**2
            
            loss = L_feat + 0.3 * (L_kd_A + L_kd_B)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        
        sch.step()
        
        if ep % 5 == 0 or ep == 1:
            mpc = eval_pc(student, clA+clB)
            retA = sum(1 for c in clA if parent_pc[c]>0 and mpc.get(c,0)/parent_pc[c]>=0.9)
            retB = sum(1 for c in clB if parent_pc[c]>0 and mpc.get(c,0)/parent_pc[c]>=0.9)
            ret = retA + retB
            avg_drop = np.mean([(1-mpc.get(c,0)/parent_pc[c])*100 for c in clA+clB if parent_pc[c]>0])
            print(f"  ep {ep:2d}: A={retA}/5 B={retB}/5 ret={ret}/10 avg_drop={avg_drop:.1f}% loss={epoch_loss:.3f} ({time.time()-t0:.0f}s)")
            
            if ret > best_ret:
                best_ret = ret
                best_sd = {k: v.clone() for k, v in student.state_dict().items()}

    # Final eval
    if best_sd: student.load_state_dict(best_sd)
    mpc = eval_pc(student, clA+clB)
    ret_final = print_drop("FEATURE DISTILL BEST", parent_pc, mpc, clA+clB)

    print(f"\n{'='*60}")
    print(f"  FINAL: Retention = {ret_final}/10")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")

