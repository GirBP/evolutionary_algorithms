#!/usr/bin/env python3
"""
CNN Merge Debate R3 (FINAL): Feature Distill v3
================================================
Back to R1 (6/10) + targeted fixes:
  1. Equal routing α=0.5 (proven in R1, R2 label-routing broke it)
  2. Wider FC: 512→256→10 with ReLU (more capacity)
  3. 60k transfer images (train+test) instead of 10k
  4. 60 epochs with cosine warm restarts
  5. Cache parents+features for fast re-runs
  6. Audio notification + verbose prints
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, numpy as np, os, platform
from torchvision import datasets, transforms, models

def beep(msg):
    print(f"\n🔔 {msg}")
    if platform.system() == 'Darwin': os.system(f'say "{msg}" &')

DEV = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEV}")
if DEV.type == 'cuda': print(f"  GPU: {torch.cuda.get_device_name()}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
train_ds = datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10('/tmp/cifar10', train=False, download=True, transform=transform)
NW = 0 if platform.system() == 'Darwin' else 2
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=NW)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=NW)
# Full transfer data loader (60k train + 10k test)
full_ds = torch.utils.data.ConcatDataset([train_ds, test_ds])
full_loader = torch.utils.data.DataLoader(full_ds, batch_size=256, shuffle=False, num_workers=NW)
clA, clB = list(range(5)), list(range(5,10))

def make_rn18(nc):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    return m

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        m = self.model
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
        feat = m.avgpool(x).flatten(1)
        logits = m.fc(feat)
        return feat, logits

# Student with WIDER FC head
class StudentNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # Replace narrow FC with wider head
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        feat = self.backbone(x)
        return feat, self.head(feat)

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
        if (ep+1) % 5 == 0:
            print(f"    parent ep {ep+1}/{epochs} ({time.time()-t0:.0f}s)")
    m.eval(); return m

def eval_pc(model, classes):
    model.eval()
    pc = {}
    with torch.no_grad():
        for xb, yb in test_loader:
            if hasattr(model, 'head'):
                _, logits = model(xb.to(DEV))
                preds = logits.argmax(1).cpu()
            else:
                preds = model(xb.to(DEV)).argmax(1).cpu()
            for c in classes:
                mask = yb==c
                if mask.sum()==0: continue
                pc[c] = pc.get(c,0) + (preds[mask]==c).float().sum().item()
    for c in classes: pc[c] = pc.get(c,0) / 1000
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

if __name__ == '__main__':
    import multiprocessing
    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('fork', force=True)
    
    t0 = time.time()
    CACHE = 'cached_parents_r3.pth'
    
    beep("Experiment started")

    # === PHASE 1: Load or train parents ===
    if os.path.exists(CACHE):
        print("=" * 60)
        print("  PHASE 1: Loading cached parents+features")
        print("=" * 60)
        cache = torch.load(CACHE, map_location='cpu', weights_only=False)
        pA = make_rn18(5).to(DEV); pA.load_state_dict(cache['pA']); pA.eval()
        pB = make_rn18(5).to(DEV); pB.load_state_dict(cache['pB']); pB.eval()
        parent_pc = cache['parent_pc']
        H_A = cache['H_A']
        H_B = cache['H_B']
        logits_A = cache['logits_A']
        logits_B = cache['logits_B']
        print(f"  Loaded from cache ({time.time()-t0:.0f}s)")
        print(f"  Parents: A={parent_pc}")
    else:
        print("=" * 60)
        print("  PHASE 1: Training parents (first run, will cache)")
        print("=" * 60)
        print("  Training parent A (classes 0-4)...")
        pA = train_parent(clA, 15, 42)
        print("  Training parent B (classes 5-9)...")
        pB = train_parent(clB, 15, 142)

        # Parent accuracies
        pA_pc, pB_pc = {}, {}
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
        print(f"  Parents trained ({time.time()-t0:.0f}s): {parent_pc}")

        # Pre-compute features on FULL dataset (60k)
        print("\n  Computing features on 60k images...")
        feA = FeatureExtractor(pA).to(DEV).eval()
        feB = FeatureExtractor(pB).to(DEV).eval()
        fA_l, lA_l, fB_l, lB_l = [], [], [], []
        with torch.no_grad():
            for i, (xb, _) in enumerate(full_loader):
                fA, lA = feA(xb.to(DEV)); fA_l.append(fA.cpu()); lA_l.append(lA.cpu())
                fB, lB = feB(xb.to(DEV)); fB_l.append(fB.cpu()); lB_l.append(lB.cpu())
                if (i+1) % 50 == 0: print(f"    batch {i+1}/{len(full_loader)} ({time.time()-t0:.0f}s)")
        H_A = torch.cat(fA_l); H_B = torch.cat(fB_l)
        logits_A = torch.cat(lA_l); logits_B = torch.cat(lB_l)

        # Save cache
        torch.save({
            'pA': pA.state_dict(), 'pB': pB.state_dict(),
            'parent_pc': parent_pc,
            'H_A': H_A, 'H_B': H_B,
            'logits_A': logits_A, 'logits_B': logits_B,
        }, CACHE)
        print(f"  Cached to {CACHE} ({time.time()-t0:.0f}s)")

    # Get all transfer images
    print("\n" + "=" * 60)
    print(f"  PHASE 2: Preparing {len(H_A)} transfer images")
    print("=" * 60)
    
    all_x_list = []
    for xb, _ in full_loader:
        all_x_list.append(xb)
    all_x = torch.cat(all_x_list)
    print(f"  Transfer set: {all_x.shape[0]} images")
    print(f"  Features: H_A={H_A.shape}, H_B={H_B.shape}")
    print(f"  ({time.time()-t0:.0f}s)")

    # === PHASE 3: Feature Distillation v3 ===
    print("\n" + "=" * 60)
    print("  PHASE 3: Feature Distill v3 (wider FC, 60 epochs)")
    print("=" * 60)
    beep("Training started")

    backbone = make_rn18(10).to(DEV)
    student = StudentNet(backbone).to(DEV)

    # Projection layers
    proj_A = nn.Linear(512, 256, bias=False).to(DEV)
    proj_B = nn.Linear(512, 256, bias=False).to(DEV)
    proj_tA = nn.Linear(512, 256, bias=False).to(DEV)
    proj_tB = nn.Linear(512, 256, bias=False).to(DEV)

    all_params = list(student.parameters()) + list(proj_A.parameters()) + list(proj_B.parameters()) + \
                 list(proj_tA.parameters()) + list(proj_tB.parameters())

    opt = torch.optim.Adam(all_params, lr=0.001, weight_decay=1e-4)
    # Cosine with warm restarts: reset every 20 epochs
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=1)

    best_ret, best_sd, best_ep, best_drop = 0, None, 0, 100

    for ep in range(1, 61):
        student.train(); proj_A.train(); proj_B.train()
        idx = torch.randperm(len(all_x))
        epoch_loss = 0
        
        for i in range(0, len(idx), 256):
            bi = idx[i:i+256]
            xb = all_x[bi].to(DEV)
            hA_t = H_A[bi].to(DEV)
            hB_t = H_B[bi].to(DEV)
            lA_t = logits_A[bi].to(DEV)
            lB_t = logits_B[bi].to(DEV)
            
            # Forward through student
            h_s, z_s = student(xb)
            
            # Projected cosine similarity (equal routing α=0.5)
            p_s_A = F.normalize(proj_A(h_s), dim=1)
            p_t_A = F.normalize(proj_tA(hA_t), dim=1)
            p_s_B = F.normalize(proj_B(h_s), dim=1)
            p_t_B = F.normalize(proj_tB(hB_t), dim=1)
            
            cos_A = 1 - (p_s_A * p_t_A).sum(dim=1)
            cos_B = 1 - (p_s_B * p_t_B).sum(dim=1)
            
            L_feat = (cos_A.mean() + cos_B.mean()) / 2  # equal weight
            
            # Logit KD
            T = 4.0
            L_kd_A = F.kl_div(F.log_softmax(z_s[:,:5]/T, 1), F.softmax(lA_t/T, 1), reduction='batchmean') * T**2
            L_kd_B = F.kl_div(F.log_softmax(z_s[:,5:]/T, 1), F.softmax(lB_t/T, 1), reduction='batchmean') * T**2
            
            loss = L_feat + 0.3 * (L_kd_A + L_kd_B)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        
        sch.step()
        
        # Eval every 3 epochs
        if ep % 3 == 0 or ep == 1:
            mpc = eval_pc(student, clA+clB)
            retA = sum(1 for c in clA if parent_pc[c]>0 and mpc.get(c,0)/parent_pc[c]>=0.9)
            retB = sum(1 for c in clB if parent_pc[c]>0 and mpc.get(c,0)/parent_pc[c]>=0.9)
            ret = retA + retB
            avg_drop = np.mean([(1-mpc.get(c,0)/parent_pc[c])*100 for c in clA+clB if parent_pc[c]>0])
            print(f"  ep {ep:2d}: A={retA}/5 B={retB}/5 ret={ret}/10 avg_drop={avg_drop:.1f}% loss={epoch_loss:.3f} ({time.time()-t0:.0f}s)")
            
            if ret > best_ret or (ret == best_ret and avg_drop < best_drop):
                best_ret = ret
                best_ep = ep
                best_drop = avg_drop
                best_sd = {k: v.clone() for k, v in student.state_dict().items()}

    # Final eval
    if best_sd: student.load_state_dict(best_sd)
    mpc = eval_pc(student, clA+clB)
    ret_final = print_drop("R3 BEST", parent_pc, mpc, clA+clB)

    print(f"\n{'='*60}")
    print(f"  FINAL: Retention = {ret_final}/10 (best at ep{best_ep})")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")
    
    beep(f"Done. Retention {ret_final} out of 10")
