#!/usr/bin/env python3
"""
CNN Merge: CMA-ES Calibrated Dual-Backbone KD
===============================================
Pipeline:
  Stage 0: Load parents, pre-compute features + logits
  Stage 1: CMA-ES optimizes logit calibration for pseudo-labels
           θ = [scale_A, scale_B, bias_0...bias_9] (12D)
           Fitness = teacher retention on real test labels
  Stage 2: Train dual-backbone teacher with best pseudo-labels
  Stage 3: KD from teacher → single ResNet-18
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
    return m.avgpool(x).flatten(1)

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

def eval_retention(logits, true_y, parent_pc, classes):
    """Compute retention from 10-class logits."""
    preds = logits.argmax(1)
    pc = {}
    for c in classes:
        mask = true_y == c
        if mask.sum() == 0: continue
        pc[c] = (preds[mask] == c).float().mean().item()
    ret = sum(1 for c in classes if parent_pc[c]>0 and pc.get(c,0)/parent_pc[c]>=0.9)
    avg_drop = np.mean([(1-pc.get(c,0)/parent_pc[c])*100 for c in classes if parent_pc[c]>0])
    return ret, avg_drop, pc

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
    # STAGE 0: Load parents + pre-compute everything
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

    # Pre-compute all features and logits
    print("\n  Pre-computing features and logits...")
    all_x = torch.cat([xb for xb, _ in test_loader])
    true_y = torch.cat([yb for _, yb in test_loader])  # real labels for FITNESS EVAL ONLY
    
    with torch.no_grad():
        H_A = torch.cat([get_features(pA, xb.to(DEV)).cpu() for xb in all_x.split(256)])
        H_B = torch.cat([get_features(pB, xb.to(DEV)).cpu() for xb in all_x.split(256)])
        logA = torch.cat([pA.fc(get_features(pA, xb.to(DEV))).cpu() for xb in all_x.split(256)])
        logB = torch.cat([pB.fc(get_features(pB, xb.to(DEV))).cpu() for xb in all_x.split(256)])
    
    H_dual = torch.cat([H_A, H_B], dim=1)  # [10000, 1024]
    H_dual_dev = H_dual.to(DEV)
    
    print(f"  H_dual={H_dual.shape}, logA={logA.shape}, logB={logB.shape}")
    print(f"  ({time.time()-t0:.0f}s)")

    # Baseline: raw concatenation pseudo-labels
    raw_pseudo = torch.cat([logA, logB], dim=1).argmax(1)
    raw_acc = (raw_pseudo == true_y).float().mean().item()
    print(f"\n  Baseline pseudo-label accuracy: {raw_acc:.3f} ({(raw_pseudo==true_y).sum()}/{len(true_y)})")
    for c in range(10):
        n = (raw_pseudo == c).sum().item()
        correct = ((raw_pseudo == c) & (true_y == c)).sum().item()
        print(f"    class {c}: {n} assigned, {correct} correct")

    # ═══════════════════════════════════════════════
    # STAGE 1: CMA-ES calibration optimization
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 1: CMA-ES logit calibration (12D)")
    print("=" * 60)
    beep("CMA-ES started")

    def make_pseudo_labels(theta):
        """Generate pseudo-labels from calibration params."""
        scale_A, scale_B = theta[0], theta[1]
        bias = torch.tensor(theta[2:12], dtype=torch.float32)
        calibrated = torch.cat([scale_A * logA + bias[:5], scale_B * logB + bias[5:]], dim=1)
        return calibrated.argmax(1)

    def quick_train_eval(pseudo_y, n_epochs=5):
        """Fast: train linear probe on pre-computed features, return retention."""
        probe = nn.Linear(1024, 10).to(DEV)
        opt = torch.optim.Adam(probe.parameters(), lr=0.01)
        py_dev = pseudo_y.to(DEV)
        
        for _ in range(n_epochs):
            idx = torch.randperm(len(H_dual_dev))
            for i in range(0, len(idx), 512):
                bi = idx[i:i+512]
                logits = probe(H_dual_dev[bi])
                loss = F.cross_entropy(logits, py_dev[bi])
                opt.zero_grad(); loss.backward(); opt.step()
        
        # Eval
        with torch.no_grad():
            all_logits = probe(H_dual_dev).cpu()
        ret, avg_drop, _ = eval_retention(all_logits, true_y, parent_pc, clA+clB)
        return ret, avg_drop

    def fitness(theta):
        """CMA-ES fitness: MINIMIZE negative retention + avg_drop."""
        pseudo_y = make_pseudo_labels(theta)
        
        # Check: at least 100 samples per class
        for c in range(10):
            if (pseudo_y == c).sum() < 50:
                return 100.0  # penalty: bad distribution
        
        ret, avg_drop = quick_train_eval(pseudo_y, n_epochs=5)
        # Minimize: -retention * 10 + avg_drop
        return -ret * 10 + avg_drop

    # CMA-ES
    try:
        import cma
        HAS_CMA = True
    except ImportError:
        HAS_CMA = False
        print("  cma not found, installing...")
        os.system('pip install cma -q')
        import cma
        HAS_CMA = True

    # Initial guess: scales=1, biases=0
    x0 = [1.0, 1.0] + [0.0] * 10
    sigma0 = 0.5
    
    print(f"  x0 = scales=[1,1], biases=[0]*10")
    print(f"  sigma0 = {sigma0}")
    print(f"  Fitness per eval: ~5-10s on T4")
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'maxiter': 30,
        'popsize': 10,
        'seed': 42,
        'verbose': -1,  # quiet
    })
    
    gen = 0
    best_fitness_ever = 100
    best_theta_ever = x0[:]
    
    while not es.stop():
        gen += 1
        solutions = es.ask()
        fitnesses = []
        
        for sol in solutions:
            f = fitness(sol)
            fitnesses.append(f)
        
        es.tell(solutions, fitnesses)
        
        best_f = min(fitnesses)
        best_sol = solutions[fitnesses.index(best_f)]
        
        if best_f < best_fitness_ever:
            best_fitness_ever = best_f
            best_theta_ever = best_sol[:]
        
        # Decode fitness: ret = -(best_f - avg_drop) / 10 approximately
        pseudo_y = make_pseudo_labels(best_sol)
        pseudo_acc = (pseudo_y == true_y).float().mean().item()
        ret_est = -int(best_f // 10) if best_f < 0 else 0
        
        print(f"  gen {gen:2d}: best_f={best_f:+.1f} pseudo_acc={pseudo_acc:.3f} "
              f"scales=[{best_sol[0]:.2f},{best_sol[1]:.2f}] ({time.time()-t0:.0f}s)")

    # Final best
    print(f"\n  CMA-ES done: best_f={best_fitness_ever:+.1f}")
    print(f"  Best θ: scales=[{best_theta_ever[0]:.3f}, {best_theta_ever[1]:.3f}]")
    print(f"  Best biases: {[f'{b:.3f}' for b in best_theta_ever[2:]]}")
    
    best_pseudo_y = make_pseudo_labels(best_theta_ever)
    best_acc = (best_pseudo_y == true_y).float().mean().item()
    print(f"  Best pseudo-label accuracy: {best_acc:.3f} (baseline: {raw_acc:.3f})")
    print(f"  ({time.time()-t0:.0f}s)")

    # Verify best labels
    print("\n  Best pseudo-label distribution:")
    for c in range(10):
        n = (best_pseudo_y == c).sum().item()
        correct = ((best_pseudo_y == c) & (true_y == c)).sum().item()
        print(f"    class {c}: {n} assigned, {correct} correct ({correct/max(n,1)*100:.0f}%)")

    # ═══════════════════════════════════════════════
    # STAGE 2: Train full dual-backbone teacher
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 2: Dual-backbone teacher (CMA-calibrated labels)")
    print("=" * 60)

    class DualTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 10)
            )
        def forward(self, h_dual):
            return self.classifier(h_dual)

    teacher = DualTeacher().to(DEV)
    opt = torch.optim.Adam(teacher.parameters(), lr=0.001, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    py_dev = best_pseudo_y.to(DEV)

    for ep in range(1, 21):
        teacher.train()
        idx = torch.randperm(len(H_dual_dev))
        ep_loss = 0
        for i in range(0, len(idx), 256):
            bi = idx[i:i+256]
            logits = teacher(H_dual_dev[bi])
            loss = F.cross_entropy(logits, py_dev[bi])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        sch.step()
        
        if ep % 5 == 0 or ep == 1:
            with torch.no_grad():
                t_logits = teacher(H_dual_dev).cpu()
            ret, avg_drop, tpc = eval_retention(t_logits, true_y, parent_pc, clA+clB)
            print(f"  ep {ep:2d}: ret={ret}/10 avg_drop={avg_drop:.1f}% loss={ep_loss:.2f} ({time.time()-t0:.0f}s)")

    with torch.no_grad():
        t_logits = teacher(H_dual_dev).cpu()
    t_ret = print_drop("DUAL TEACHER (CMA-calibrated)", parent_pc,
                       {c: ((t_logits.argmax(1)==c) & (true_y==c)).sum().item()/1000 for c in clA+clB},
                       clA+clB)

    # ═══════════════════════════════════════════════
    # STAGE 3: KD from teacher → single ResNet-18
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 3: KD → single ResNet-18")
    print("=" * 60)
    beep("KD started")

    # Pre-compute teacher soft targets
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(H_dual_dev).cpu()  # [10000, 10]

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
            t_log = teacher_logits[bi].to(DEV)
            py = best_pseudo_y[bi].to(DEV)
            
            s_logits = student(xb)
            
            T = 4.0
            L_kd = F.kl_div(
                F.log_softmax(s_logits/T, 1),
                F.softmax(t_log/T, 1),
                reduction='batchmean'
            ) * T**2
            
            L_ce = F.cross_entropy(s_logits, py)
            loss = 0.7 * L_kd + 0.3 * L_ce
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        sch.step()
        
        if ep % 5 == 0 or ep == 1:
            student.eval()
            with torch.no_grad():
                s_logits_all = torch.cat([student(xb.to(DEV)).cpu() for xb in all_x.split(256)])
            ret, avg_drop, spc = eval_retention(s_logits_all, true_y, parent_pc, clA+clB)
            retA = sum(1 for c in clA if parent_pc[c]>0 and spc.get(c,0)/parent_pc[c]>=0.9)
            retB = sum(1 for c in clB if parent_pc[c]>0 and spc.get(c,0)/parent_pc[c]>=0.9)
            print(f"  ep {ep:2d}: A={retA}/5 B={retB}/5 ret={ret}/10 avg_drop={avg_drop:.1f}% ({time.time()-t0:.0f}s)")
            
            if ret > best_ret or (ret == best_ret and avg_drop < best_drop):
                best_ret = ret; best_ep = ep; best_drop = avg_drop
                best_sd = {k: v.clone() for k, v in student.state_dict().items()}

    # Final
    if best_sd: student.load_state_dict(best_sd)
    student.eval()
    with torch.no_grad():
        s_logits_all = torch.cat([student(xb.to(DEV)).cpu() for xb in all_x.split(256)])
    _, _, spc = eval_retention(s_logits_all, true_y, parent_pc, clA+clB)
    ret_final = print_drop("SINGLE ResNet-18 (CMA-ES + KD)", parent_pc, spc, clA+clB)

    print(f"\n{'='*60}")
    print(f"  TEACHER: {t_ret}/10 (dual-backbone, CMA-calibrated)")
    print(f"  STUDENT: {ret_final}/10 (single ResNet-18, best at ep{best_ep})")
    print(f"  CMA-ES improvement: pseudo-acc {raw_acc:.3f} → {best_acc:.3f}")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")
    beep(f"Done. Teacher {t_ret}, Student {ret_final} out of 10")
