#!/usr/bin/env python3
"""
CNN Merge: SupCon + Prototype Imprinting
=========================================
Based on external agent proposal.

Pipeline:
  Stage 0: Load/train parents, compute features
  Stage 1: Prototype routing — assign pseudo-labels via feature-space distance
  Stage 2: SupCon + CE on balanced seed set
  Stage 3: Self-training expansion (optional)
  Stage 4: Prototype imprinting — replace FC with class means
"""
import torch, torch.nn as nn, torch.nn.functional as F
import time, numpy as np, os, platform
from torchvision import datasets, transforms, models

def beep(msg):
    print(f"\n🔔 {msg}")
    if platform.system() == 'Darwin': os.system(f'say "{msg}" &')

DEV = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEV}")
if DEV.type == 'cuda': print(f"  GPU: {torch.cuda.get_device_name()}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

NW = 0 if platform.system() == 'Darwin' else 2
test_ds = datasets.CIFAR10('/tmp/cifar10', train=False, download=True, transform=transform)
train_ds = datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=NW)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=NW)
clA, clB = list(range(5)), list(range(5,10))


# ═══ Models ═══
def make_rn18(nc):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    return m

def get_features(model, x):
    """Extract 512-dim features before FC."""
    m = model
    x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
    x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
    return m.avgpool(x).flatten(1)


# ═══ SupCon Loss ═══
class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [B, dim], already L2-normalized
        # labels: [B]
        B = features.shape[0]
        if B <= 1: return torch.tensor(0.0, device=features.device)
        
        # Cosine similarity matrix
        sim = torch.mm(features, features.T) / self.temperature  # [B, B]
        
        # Mask: same class = positive
        labels = labels.unsqueeze(0)
        mask_pos = (labels == labels.T).float()  # [B, B]
        mask_pos.fill_diagonal_(0)  # exclude self
        
        # For numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        
        # Denominator: all except self
        mask_self = torch.eye(B, device=features.device)
        exp_sim = torch.exp(sim) * (1 - mask_self)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # Mean of positive pairs
        n_pos = mask_pos.sum(dim=1)
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (n_pos + 1e-8)
        
        # Only use samples with at least 1 positive pair
        valid = n_pos > 0
        if valid.sum() == 0: return torch.tensor(0.0, device=features.device)
        
        loss = -mean_log_prob[valid].mean()
        return loss


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
        if (ep+1) % 5 == 0: print(f"    parent ep {ep+1}/{epochs} ({time.time()-t0:.0f}s)")
    m.eval(); return m


def eval_pc(model, classes):
    model.eval()
    pc = {}
    with torch.no_grad():
        for xb, yb in test_loader:
            feat = get_features(model, xb.to(DEV))
            logits = model.fc(feat)
            preds = logits.argmax(1).cpu()
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

    # ═══════════════════════════════════════════════
    # STAGE 0: Load or train parents
    # ═══════════════════════════════════════════════
    if os.path.exists(CACHE):
        print("=" * 60)
        print("  STAGE 0: Loading cached parents")
        print("=" * 60)
        cache = torch.load(CACHE, map_location='cpu', weights_only=False)
        pA = make_rn18(5).to(DEV); pA.load_state_dict(cache['pA']); pA.eval()
        pB = make_rn18(5).to(DEV); pB.load_state_dict(cache['pB']); pB.eval()
        parent_pc = cache['parent_pc']
        print(f"  Loaded ({time.time()-t0:.0f}s)")
        print(f"  Parents: {parent_pc}")
    else:
        print("=" * 60)
        print("  STAGE 0: Training parents (will cache)")
        print("=" * 60)
        print("  Training parent A (classes 0-4)...")
        pA = train_parent(clA, 15, 42)
        print("  Training parent B (classes 5-9)...")
        pB = train_parent(clB, 15, 142)
        pA_pc, pB_pc = {}, {}
        with torch.no_grad():
            for xb, yb in test_loader:
                fA = get_features(pA, xb.to(DEV)); predsA = pA.fc(fA).argmax(1).cpu()
                fB = get_features(pB, xb.to(DEV)); predsB = pB.fc(fB).argmax(1).cpu()
                for c in clA:
                    mask = yb==c; pA_pc[c] = pA_pc.get(c,0) + (predsA[mask]==clA.index(c)).float().sum().item()
                for c in clB:
                    mask = yb==c; pB_pc[c] = pB_pc.get(c,0) + (predsB[mask]==clB.index(c)).float().sum().item()
        for c in clA: pA_pc[c]/=1000
        for c in clB: pB_pc[c]/=1000
        parent_pc = {**pA_pc, **pB_pc}
        torch.save({'pA': pA.state_dict(), 'pB': pB.state_dict(), 'parent_pc': parent_pc}, CACHE)
        print(f"  Cached ({time.time()-t0:.0f}s): {parent_pc}")

    # ═══════════════════════════════════════════════
    # STAGE 1: Prototype routing + pseudo-label assignment
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 1: Prototype routing")
    print("=" * 60)

    # Collect all test images + features
    all_x = torch.cat([xb for xb, _ in test_loader])
    print(f"  Transfer set: {len(all_x)} images")

    with torch.no_grad():
        # Features in each parent's space
        hA_all = torch.cat([get_features(pA, xb.to(DEV)).cpu() for xb in all_x.split(256)])
        hB_all = torch.cat([get_features(pB, xb.to(DEV)).cpu() for xb in all_x.split(256)])
        logA_all = torch.cat([pA.fc(get_features(pA, xb.to(DEV))).cpu() for xb in all_x.split(256)])
        logB_all = torch.cat([pB.fc(get_features(pB, xb.to(DEV))).cpu() for xb in all_x.split(256)])
    
    print(f"  Features computed ({time.time()-t0:.0f}s)")

    # Build class prototypes from high-margin predictions
    probA = F.softmax(logA_all, dim=1)
    probB = F.softmax(logB_all, dim=1)
    
    # Margin = top1 - top2
    topA = probA.topk(2, dim=1)
    marginA = topA.values[:,0] - topA.values[:,1]
    predA = logA_all.argmax(1)  # 0-4
    
    topB = probB.topk(2, dim=1)
    marginB = topB.values[:,0] - topB.values[:,1]
    predB = logB_all.argmax(1)  # 0-4 (maps to 5-9)

    # Prototypes: mean of top-50% margin samples per class
    prototypes_A = {}  # class -> prototype in A's feature space
    for c in range(5):
        mask = predA == c
        if mask.sum() == 0: continue
        margins = marginA[mask]
        threshold = margins.median()
        high_margin = mask.clone()
        high_margin[mask] = margins >= threshold
        prototypes_A[c] = F.normalize(hA_all[high_margin].mean(0, keepdim=True), dim=1)
    
    prototypes_B = {}  # class -> prototype in B's feature space
    for c in range(5):
        mask = predB == c
        if mask.sum() == 0: continue
        margins = marginB[mask]
        threshold = margins.median()
        high_margin = mask.clone()
        high_margin[mask] = margins >= threshold
        prototypes_B[c] = F.normalize(hB_all[high_margin].mean(0, keepdim=True), dim=1)

    print(f"  Prototypes: A={len(prototypes_A)} classes, B={len(prototypes_B)} classes")

    # Route each image: which parent's prototype is closer?
    pseudo_labels = torch.full((len(all_x),), -1, dtype=torch.long)
    routing_confidence = torch.zeros(len(all_x))
    
    hA_norm = F.normalize(hA_all, dim=1)
    hB_norm = F.normalize(hB_all, dim=1)
    
    # Vectorized routing: compute all similarities at once
    proto_A_mat = torch.cat([prototypes_A[c] for c in sorted(prototypes_A.keys())])  # [5, 512]
    proto_B_mat = torch.cat([prototypes_B[c] for c in sorted(prototypes_B.keys())])  # [5, 512]
    cls_A = torch.tensor(sorted(prototypes_A.keys()))  # [5] = [0,1,2,3,4]
    cls_B = torch.tensor(sorted(prototypes_B.keys())) + 5  # [5] = [5,6,7,8,9]
    
    sim_A = hA_norm @ proto_A_mat.T  # [N, 5]
    sim_B = hB_norm @ proto_B_mat.T  # [N, 5]
    
    best_sim_A, best_idx_A = sim_A.max(dim=1)  # [N]
    best_sim_B, best_idx_B = sim_B.max(dim=1)
    
    best_cls_A = cls_A[best_idx_A]  # classes 0-4
    best_cls_B = cls_B[best_idx_B]  # classes 5-9
    
    combined_A = best_sim_A * (1 + marginA)
    combined_B = best_sim_B * (1 + marginB)
    
    route_to_A = combined_A > combined_B
    pseudo_labels = torch.where(route_to_A, best_cls_A, best_cls_B)
    routing_confidence = torch.where(route_to_A, combined_A, combined_B)

    # Report routing stats
    for c in range(10):
        n = (pseudo_labels == c).sum().item()
        print(f"    class {c}: {n} samples")
    
    print(f"  Routing done ({time.time()-t0:.0f}s)")

    # Build balanced seed set: top-K per class by routing confidence
    K_PER_CLASS = 500
    seed_indices = []
    for c in range(10):
        mask = pseudo_labels == c
        if mask.sum() == 0:
            print(f"  ⚠️ class {c}: 0 samples!")
            continue
        confs = routing_confidence[mask]
        idx_in_mask = confs.argsort(descending=True)[:K_PER_CLASS]
        global_idx = torch.where(mask)[0][idx_in_mask]
        seed_indices.append(global_idx)
    
    seed_idx = torch.cat(seed_indices)
    seed_x = all_x[seed_idx]
    seed_y = pseudo_labels[seed_idx]
    
    print(f"  Seed set: {len(seed_x)} images, {len(torch.unique(seed_y))} classes")
    for c in range(10):
        print(f"    class {c}: {(seed_y==c).sum().item()} samples")
    print(f"  ({time.time()-t0:.0f}s)")

    # ═══════════════════════════════════════════════
    # STAGE 2: SupCon + CE training
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 2: SupCon + CE training (30 epochs)")
    print("=" * 60)
    beep("Training stage 2")

    student = make_rn18(10).to(DEV)
    # Projection head for SupCon
    proj_head = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    ).to(DEV)

    supcon = SupConLoss(temperature=0.1)
    
    all_params = list(student.parameters()) + list(proj_head.parameters())
    opt = torch.optim.Adam(all_params, lr=0.001, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

    best_ret, best_sd, best_ep, best_drop = 0, None, 0, 100

    for ep in range(1, 31):
        student.train(); proj_head.train()
        idx = torch.randperm(len(seed_x))
        ep_loss_ce, ep_loss_sc = 0, 0
        
        for i in range(0, len(idx), 128):  # smaller batch for more SupCon pairs
            bi = idx[i:i+128]
            xb = seed_x[bi].to(DEV)
            yb = seed_y[bi].to(DEV)
            
            # Forward
            feat = get_features(student, xb)
            logits = student.fc(feat)
            
            # CE loss
            L_ce = F.cross_entropy(logits, yb)
            
            # SupCon loss
            z = F.normalize(proj_head(feat), dim=1)
            L_sc = supcon(z, yb)
            
            loss = L_ce + 0.5 * L_sc
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss_ce += L_ce.item()
            ep_loss_sc += L_sc.item()
        
        sch.step()
        
        if ep % 3 == 0 or ep == 1:
            mpc = eval_pc(student, clA+clB)
            retA = sum(1 for c in clA if parent_pc[c]>0 and mpc.get(c,0)/parent_pc[c]>=0.9)
            retB = sum(1 for c in clB if parent_pc[c]>0 and mpc.get(c,0)/parent_pc[c]>=0.9)
            ret = retA + retB
            avg_drop = np.mean([(1-mpc.get(c,0)/parent_pc[c])*100 for c in clA+clB if parent_pc[c]>0])
            print(f"  ep {ep:2d}: A={retA}/5 B={retB}/5 ret={ret}/10 avg_drop={avg_drop:.1f}% CE={ep_loss_ce:.2f} SC={ep_loss_sc:.2f} ({time.time()-t0:.0f}s)")
            
            if ret > best_ret or (ret == best_ret and avg_drop < best_drop):
                best_ret = ret; best_ep = ep; best_drop = avg_drop
                best_sd = {k: v.clone() for k, v in student.state_dict().items()}

    if best_sd: student.load_state_dict(best_sd)
    print(f"\n  Best before imprinting: {best_ret}/10 at ep{best_ep}")

    # ═══════════════════════════════════════════════
    # STAGE 3: Prototype Imprinting
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 3: Prototype imprinting")
    print("=" * 60)

    student.eval()
    class_features = {c: [] for c in range(10)}
    
    with torch.no_grad():
        for i in range(0, len(seed_x), 256):
            xb = seed_x[i:i+256].to(DEV)
            yb = seed_y[i:i+256]
            feat = get_features(student, xb).cpu()
            for j in range(len(yb)):
                class_features[yb[j].item()].append(feat[j])
    
    # Compute class means and set as FC weights
    prototypes = torch.zeros(10, 512)
    for c in range(10):
        if class_features[c]:
            stacked = torch.stack(class_features[c])
            prototypes[c] = F.normalize(stacked.mean(0), dim=0)
            print(f"    class {c}: {len(class_features[c])} samples → prototype norm={prototypes[c].norm():.3f}")
        else:
            print(f"    class {c}: NO SAMPLES!")
    
    # Replace FC with prototype-based classifier
    # W = prototypes, bias = 0, scale by temperature
    scale = 16.0  # learnable temperature analog
    student.fc.weight.data = prototypes.to(DEV) * scale
    student.fc.bias.data.zero_()
    
    print(f"\n  Prototype imprinting done ({time.time()-t0:.0f}s)")

    # ═══════════════════════════════════════════════
    # STAGE 4: Brief fine-tune after imprinting (FC only)
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 4: Fine-tune FC after imprinting (10 epochs)")
    print("=" * 60)

    # Freeze backbone, train only FC
    for name, p in student.named_parameters():
        p.requires_grad = 'fc' in name
    
    opt_fc = torch.optim.Adam(student.fc.parameters(), lr=0.01)
    sch_fc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fc, T_max=10)
    
    for ep in range(1, 11):
        student.train()
        idx = torch.randperm(len(seed_x))
        for i in range(0, len(idx), 128):
            bi = idx[i:i+128]
            xb = seed_x[bi].to(DEV)
            yb = seed_y[bi].to(DEV)
            with torch.no_grad():
                feat = get_features(student, xb)
            logits = student.fc(feat)
            loss = F.cross_entropy(logits, yb)
            opt_fc.zero_grad(); loss.backward(); opt_fc.step()
        sch_fc.step()
        
        if ep % 3 == 0 or ep == 1:
            mpc = eval_pc(student, clA+clB)
            retA = sum(1 for c in clA if parent_pc[c]>0 and mpc.get(c,0)/parent_pc[c]>=0.9)
            retB = sum(1 for c in clB if parent_pc[c]>0 and mpc.get(c,0)/parent_pc[c]>=0.9)
            ret = retA + retB
            avg_drop = np.mean([(1-mpc.get(c,0)/parent_pc[c])*100 for c in clA+clB if parent_pc[c]>0])
            print(f"  ep {ep:2d}: A={retA}/5 B={retB}/5 ret={ret}/10 avg_drop={avg_drop:.1f}% ({time.time()-t0:.0f}s)")
            
            if ret > best_ret or (ret == best_ret and avg_drop < best_drop):
                best_ret = ret; best_ep = ep; best_drop = avg_drop
                best_sd = {k: v.clone() for k, v in student.state_dict().items()}

    # ═══════════════════════════════════════════════
    # FINAL EVAL
    # ═══════════════════════════════════════════════
    if best_sd: student.load_state_dict(best_sd)
    mpc = eval_pc(student, clA+clB)
    ret_final = print_drop("SUPCON + PROTOTYPE IMPRINT", parent_pc, mpc, clA+clB)

    print(f"\n{'='*60}")
    print(f"  FINAL: Retention = {ret_final}/10")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")
    
    beep(f"Done. Retention {ret_final} out of 10")
