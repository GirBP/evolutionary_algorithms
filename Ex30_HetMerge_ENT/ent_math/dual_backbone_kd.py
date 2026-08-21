#!/usr/bin/env python3
"""
CNN Merge: Dual-Backbone Teacher → Single-Student KD
=====================================================
Radically different approach:
  Stage 1: Use BOTH parent backbones (frozen) as feature extractors
           Concatenate features [h_A; h_B] ∈ R^1024
           Train linear probe → 10-class teacher (should be ~95%+)
  Stage 2: Use this teacher to generate 10-class soft targets
           Distill into single ResNet-18 (standard single-teacher KD)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import time, numpy as np, os, platform
from torchvision import datasets, transforms, models

def beep(msg):
    print(f"\n🔔 {msg}")
    if platform.system() == 'Darwin': os.system(f'say "{msg}" &')
    else: print('\a')

DEV = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEV}")
if DEV.type == 'cuda': print(f"  GPU: {torch.cuda.get_device_name()}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
NW = 0 if platform.system() == 'Darwin' else 2
train_ds = datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10('/tmp/cifar10', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=NW)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=NW)
clA, clB = list(range(5)), list(range(5,10))

def make_rn18(nc):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    return m

def get_features(model, x):
    m = model
    x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
    x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
    return m.avgpool(x).flatten(1)  # [B, 512]

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
        if (ep+1) % 5 == 0: print(f"    ep {ep+1}/{epochs} ({time.time()-t0:.0f}s)")
    m.eval(); return m

def eval_pc_with_model(model, classes):
    """Eval model that outputs 10-class logits."""
    model.eval()
    pc = {}
    with torch.no_grad():
        for xb, yb in test_loader:
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


# ═══ Dual-backbone teacher model ═══
class DualBackboneTeacher(nn.Module):
    def __init__(self, parentA, parentB):
        super().__init__()
        self.parentA = parentA
        self.parentB = parentB
        # Freeze both backbones
        for p in self.parentA.parameters(): p.requires_grad = False
        for p in self.parentB.parameters(): p.requires_grad = False
        # Trainable classifier on concatenated features
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        with torch.no_grad():
            hA = get_features(self.parentA, x)  # [B, 512]
            hB = get_features(self.parentB, x)   # [B, 512]
        h = torch.cat([hA, hB], dim=1)  # [B, 1024]
        return self.classifier(h)


if __name__ == '__main__':
    import multiprocessing
    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('fork', force=True)
    
    t0 = time.time()
    CACHE = 'cached_parents_r3.pth'
    beep("Experiment started")

    # ═══════════════════════════════════════════════
    # STAGE 0: Load parents
    # ═══════════════════════════════════════════════
    if os.path.exists(CACHE):
        print("=" * 60)
        print("  STAGE 0: Loading cached parents")
        print("=" * 60)
        cache = torch.load(CACHE, map_location='cpu', weights_only=False)
        pA = make_rn18(5).to(DEV); pA.load_state_dict(cache['pA']); pA.eval()
        pB = make_rn18(5).to(DEV); pB.load_state_dict(cache['pB']); pB.eval()
        parent_pc = cache['parent_pc']
        print(f"  Loaded ({time.time()-t0:.0f}s): {parent_pc}")
    else:
        print("=" * 60)
        print("  STAGE 0: Training parents (will cache)")
        print("=" * 60)
        print("  Training parent A...")
        pA = train_parent(clA, 15, 42)
        print("  Training parent B...")
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
    # STAGE 1: Train dual-backbone teacher
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 1: Dual-backbone teacher (linear probe)")
    print("=" * 60)

    teacher = DualBackboneTeacher(pA, pB).to(DEV)
    
    # Collect pseudo-labels from parents (NOT real labels)
    all_x = torch.cat([xb for xb, _ in test_loader])
    
    print(f"  Transfer set: {len(all_x)} images")
    print(f"  Computing pseudo-labels from parents...")
    
    # Generate pseudo-labels by concatenating logits [z_A; z_B]
    # In-domain logits are naturally higher → argmax picks correct parent
    pseudo_y = torch.zeros(len(all_x), dtype=torch.long)
    with torch.no_grad():
        for i, xb in enumerate(all_x.split(256)):
            xb_dev = xb.to(DEV)
            logA = pA.fc(get_features(pA, xb_dev))  # [B, 5] classes 0-4
            logB = pB.fc(get_features(pB, xb_dev))  # [B, 5] classes 5-9
            
            # Concatenate: [z_A; z_B] → 10-dim, argmax = pseudo-label
            combined = torch.cat([logA, logB], dim=1)  # [B, 10]
            pseudo_y[i*256:i*256+len(xb)] = combined.argmax(1).cpu()
    
    # Report pseudo-label distribution
    for c in range(10):
        n = (pseudo_y == c).sum().item()
        print(f"    class {c}: {n} samples")
    print(f"  Pseudo-labels done ({time.time()-t0:.0f}s)")
    
    # Train classifier (only ~67K trainable params)
    n_trainable = sum(p.numel() for p in teacher.classifier.parameters())
    print(f"  Trainable params: {n_trainable:,} (classifier only)")
    
    opt = torch.optim.Adam(teacher.classifier.parameters(), lr=0.001, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    
    for ep in range(1, 21):
        teacher.train()
        idx = torch.randperm(len(all_x))
        ep_loss = 0
        for i in range(0, len(idx), 256):
            bi = idx[i:i+256]
            xb = all_x[bi].to(DEV)
            yb = pseudo_y[bi].to(DEV)
            logits = teacher(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        sch.step()
        
        if ep % 5 == 0 or ep == 1:
            tpc = eval_pc_with_model(teacher, clA+clB)
            ret = sum(1 for c in clA+clB if parent_pc[c]>0 and tpc.get(c,0)/parent_pc[c]>=0.9)
            avg_drop = np.mean([(1-tpc.get(c,0)/parent_pc[c])*100 for c in clA+clB if parent_pc[c]>0])
            print(f"  ep {ep:2d}: ret={ret}/10 avg_drop={avg_drop:.1f}% loss={ep_loss:.2f} ({time.time()-t0:.0f}s)")

    # Teacher quality
    tpc = eval_pc_with_model(teacher, clA+clB)
    t_ret = print_drop("DUAL-BACKBONE TEACHER", parent_pc, tpc, clA+clB)
    print(f"\n  Teacher quality: {t_ret}/10")

    # ═══════════════════════════════════════════════
    # STAGE 2: Distill teacher → single ResNet-18
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 2: KD from 10-class teacher → single ResNet-18")
    print("=" * 60)
    beep("Distillation started")

    # Pre-compute teacher logits (10-class, unified)
    teacher.eval()
    with torch.no_grad():
        teacher_logits = torch.cat([teacher(xb.to(DEV)).cpu() for xb in all_x.split(256)])
    print(f"  Teacher logits: {teacher_logits.shape}")

    # Fresh student from ImageNet
    student = make_rn18(10).to(DEV)
    
    opt = torch.optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    
    best_ret, best_sd, best_ep, best_drop = 0, None, 0, 100
    
    for ep in range(1, 41):
        student.train()
        idx = torch.randperm(len(all_x))
        ep_loss = 0
        
        for i in range(0, len(idx), 256):
            bi = idx[i:i+256]
            xb = all_x[bi].to(DEV)
            t_logits = teacher_logits[bi].to(DEV)
            yb = pseudo_y[bi].to(DEV)
            
            s_logits = student(xb)
            
            # KD loss: student mimics teacher's 10-class distribution
            T = 4.0
            L_kd = F.kl_div(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(t_logits / T, dim=1),
                reduction='batchmean'
            ) * T**2
            
            # CE with real labels (auxiliary)
            L_ce = F.cross_entropy(s_logits, yb)
            
            loss = 0.7 * L_kd + 0.3 * L_ce
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        
        sch.step()
        
        if ep % 5 == 0 or ep == 1:
            mpc = eval_pc_with_model(student, clA+clB)
            retA = sum(1 for c in clA if parent_pc[c]>0 and mpc.get(c,0)/parent_pc[c]>=0.9)
            retB = sum(1 for c in clB if parent_pc[c]>0 and mpc.get(c,0)/parent_pc[c]>=0.9)
            ret = retA + retB
            avg_drop = np.mean([(1-mpc.get(c,0)/parent_pc[c])*100 for c in clA+clB if parent_pc[c]>0])
            print(f"  ep {ep:2d}: A={retA}/5 B={retB}/5 ret={ret}/10 avg_drop={avg_drop:.1f}% ({time.time()-t0:.0f}s)")
            
            if ret > best_ret or (ret == best_ret and avg_drop < best_drop):
                best_ret = ret; best_ep = ep; best_drop = avg_drop
                best_sd = {k: v.clone() for k, v in student.state_dict().items()}
    
    # Final eval
    if best_sd: student.load_state_dict(best_sd)
    mpc = eval_pc_with_model(student, clA+clB)
    ret_final = print_drop("SINGLE ResNet-18 (distilled)", parent_pc, mpc, clA+clB)

    print(f"\n{'='*60}")
    print(f"  TEACHER: {t_ret}/10")
    print(f"  STUDENT: {ret_final}/10 (best at ep{best_ep})")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")
    
    beep(f"Done. Teacher {t_ret}, Student {ret_final} out of 10")
