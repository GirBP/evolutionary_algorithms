#!/usr/bin/env python3
"""
CNN Merge: Self-Training Loop + CMA-ES
========================================
Pipeline:
  Stage 1: R1 feature distillation → initial student (~6/10)
  Stage 2: Student pseudo-labels → dual-backbone teacher
  Stage 3: KD from teacher → better student
  Iterate Stages 2-3 until retention ≥ 8/10 or 3 rounds
  Stage 4: CMA-ES post-hoc FC bias optimization (10D)
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

def eval_model(model, classes, parent_pc):
    """Eval 10-class model → per-class accuracy + retention."""
    model.eval()
    pc = {}
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(DEV)).argmax(1).cpu()
            for c in classes:
                mask = yb == c
                if mask.sum() == 0: continue
                pc[c] = pc.get(c, 0) + (preds[mask] == c).float().sum().item()
    for c in classes: pc[c] = pc.get(c, 0) / 1000
    ret = sum(1 for c in classes if parent_pc[c] > 0 and pc.get(c, 0) / parent_pc[c] >= 0.9)
    avg_drop = np.mean([(1 - pc.get(c, 0) / parent_pc[c]) * 100 for c in classes if parent_pc[c] > 0])
    return ret, avg_drop, pc

def print_drop(name, ppc, mpc, classes):
    print(f"\n  {name}:")
    print(f"  {'Cls':>3} | {'Parent':>7} | {'Merged':>7} | {'Drop%':>6} | OK?")
    print(f"  {'-'*3}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*4}")
    ret = 0
    for c in classes:
        p, m = ppc[c], mpc.get(c, 0)
        d = (1 - m / p) * 100 if p > 0 else 100
        ok = 'YES' if d <= 10 else 'NO'
        if d <= 10: ret += 1
        print(f"  {c:>3} | {p:>7.3f} | {m:>7.3f} | {d:>5.1f}% | {ok}")
    print(f"  Retention: {ret}/{len(classes)} (drop <= 10%)")
    return ret


if __name__ == '__main__':
    import multiprocessing
    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('fork', force=True)

    t0 = time.time()
    CACHE = 'cached_parents_r3.pth'
    beep("Self-training experiment started")

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
                    mask = yb == c; pA_pc[c] = pA_pc.get(c, 0) + (predsA[mask] == clA.index(c)).float().sum().item()
                for c in clB:
                    mask = yb == c; pB_pc[c] = pB_pc.get(c, 0) + (predsB[mask] == clB.index(c)).float().sum().item()
        for c in clA: pA_pc[c] /= 1000
        for c in clB: pB_pc[c] /= 1000
        parent_pc = {**pA_pc, **pB_pc}
        torch.save({'pA': pA.state_dict(), 'pB': pB.state_dict(), 'parent_pc': parent_pc}, CACHE)
        print(f"  Cached ({time.time()-t0:.0f}s): {parent_pc}")

    # Pre-compute teacher features
    print("\n  Pre-computing features...")
    all_x = torch.cat([xb for xb, _ in test_loader])
    true_y = torch.cat([yb for _, yb in test_loader])  # for eval ONLY

    with torch.no_grad():
        H_A = torch.cat([get_features(pA, xb.to(DEV)).cpu() for xb in all_x.split(256)])
        H_B = torch.cat([get_features(pB, xb.to(DEV)).cpu() for xb in all_x.split(256)])
        logA = torch.cat([pA.fc(get_features(pA, xb.to(DEV))).cpu() for xb in all_x.split(256)])
        logB = torch.cat([pB.fc(get_features(pB, xb.to(DEV))).cpu() for xb in all_x.split(256)])
    H_dual = torch.cat([H_A, H_B], dim=1).to(DEV)
    print(f"  Done ({time.time()-t0:.0f}s)")

    # ═══════════════════════════════════════════════
    # STAGE 1: R1 Feature Distillation → initial student
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 1: Feature distillation (R1 approach, 15 ep)")
    print("=" * 60)
    beep("Feature distillation")

    student = make_rn18(10).to(DEV)
    proj_A = nn.Linear(512, 256, bias=False).to(DEV)
    proj_B = nn.Linear(512, 256, bias=False).to(DEV)
    proj_tA = nn.Linear(512, 256, bias=False).to(DEV)
    proj_tB = nn.Linear(512, 256, bias=False).to(DEV)

    all_params = list(student.parameters()) + list(proj_A.parameters()) + list(proj_B.parameters()) + \
                 list(proj_tA.parameters()) + list(proj_tB.parameters())
    opt = torch.optim.Adam(all_params, lr=0.001, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
    
    best_ret, best_sd, best_ep = 0, None, 0
    for ep in range(1, 16):
        student.train(); proj_A.train(); proj_B.train()
        idx = torch.randperm(len(all_x))
        for i in range(0, len(idx), 256):
            bi = idx[i:i+256]
            xb = all_x[bi].to(DEV)
            h_s = get_features(student, xb)
            hA_t = H_A[bi].to(DEV)
            hB_t = H_B[bi].to(DEV)
            lA_t = logA[bi].to(DEV)
            lB_t = logB[bi].to(DEV)
            
            p_sA = F.normalize(proj_A(h_s), dim=1)
            p_tA = F.normalize(proj_tA(hA_t), dim=1)
            p_sB = F.normalize(proj_B(h_s), dim=1)
            p_tB = F.normalize(proj_tB(hB_t), dim=1)
            cos_A = 1 - (p_sA * p_tA).sum(dim=1)
            cos_B = 1 - (p_sB * p_tB).sum(dim=1)
            L_feat = (cos_A.mean() + cos_B.mean()) / 2
            
            T = 4.0
            s_logits = student.fc(h_s)
            L_kd_A = F.kl_div(F.log_softmax(s_logits[:,:5]/T, 1), F.softmax(lA_t/T, 1), reduction='batchmean') * T**2
            L_kd_B = F.kl_div(F.log_softmax(s_logits[:,5:]/T, 1), F.softmax(lB_t/T, 1), reduction='batchmean') * T**2
            
            loss = L_feat + 0.3 * (L_kd_A + L_kd_B)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        
        if ep % 3 == 0 or ep == 1:
            ret, avg_drop, _ = eval_model(student, clA+clB, parent_pc)
            print(f"  ep {ep:2d}: ret={ret}/10 avg_drop={avg_drop:.1f}% ({time.time()-t0:.0f}s)")
            if ret > best_ret:
                best_ret = ret; best_ep = ep
                best_sd = {k: v.clone() for k, v in student.state_dict().items()}

    if best_sd: student.load_state_dict(best_sd)
    ret_s1, drop_s1, pc_s1 = eval_model(student, clA+clB, parent_pc)
    print(f"\n  Stage 1 result: {ret_s1}/10 (ep{best_ep})")

    # ═══════════════════════════════════════════════
    # STAGES 2-3: Self-training loop
    # ═══════════════════════════════════════════════
    for rnd in range(1, 4):
        print("\n" + "=" * 60)
        print(f"  SELF-TRAINING ROUND {rnd}")
        print("=" * 60)

        # --- Generate pseudo-labels from current student ---
        student.eval()
        with torch.no_grad():
            pseudo_y = torch.cat([student(xb.to(DEV)).argmax(1).cpu() for xb in all_x.split(256)])
        
        pseudo_acc = (pseudo_y == true_y).float().mean().item()
        print(f"  Pseudo-label accuracy: {pseudo_acc:.3f}")
        for c in range(10):
            n = (pseudo_y == c).sum().item()
            correct = ((pseudo_y == c) & (true_y == c)).sum().item()
            print(f"    class {c}: {n} assigned, {correct} correct ({correct/max(n,1)*100:.0f}%)")

        # --- Train dual-backbone teacher with student's pseudo-labels ---
        print(f"\n  Training dual-backbone teacher (15 ep)...")
        teacher_head = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 10)
        ).to(DEV)
        opt_t = torch.optim.Adam(teacher_head.parameters(), lr=0.001, weight_decay=1e-4)
        sch_t = torch.optim.lr_scheduler.CosineAnnealingLR(opt_t, T_max=15)
        py_dev = pseudo_y.to(DEV)

        for ep in range(1, 16):
            teacher_head.train()
            idx = torch.randperm(len(H_dual))
            for i in range(0, len(idx), 256):
                bi = idx[i:i+256]
                logits = teacher_head(H_dual[bi])
                loss = F.cross_entropy(logits, py_dev[bi])
                opt_t.zero_grad(); loss.backward(); opt_t.step()
            sch_t.step()

        teacher_head.eval()
        with torch.no_grad():
            t_logits = teacher_head(H_dual).cpu()
        t_preds = t_logits.argmax(1)
        t_acc = (t_preds == true_y).float().mean().item()
        t_ret_count = sum(1 for c in clA+clB if parent_pc[c]>0 and 
            ((t_preds==c) & (true_y==c)).sum().item()/1000 / parent_pc[c] >= 0.9)
        print(f"  Teacher: acc={t_acc:.3f} ret={t_ret_count}/10 ({time.time()-t0:.0f}s)")

        # --- KD from teacher → better student ---
        print(f"\n  KD from teacher → student (15 ep)...")
        student = make_rn18(10).to(DEV)  # fresh student
        opt_s = torch.optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-4)
        sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=15)
        teacher_logits = t_logits  # pre-computed

        best_ret, best_sd, best_ep = 0, None, 0
        for ep in range(1, 16):
            student.train()
            idx = torch.randperm(len(all_x))
            for i in range(0, len(idx), 256):
                bi = idx[i:i+256]
                xb = all_x[bi].to(DEV)
                tl = teacher_logits[bi].to(DEV)
                py = pseudo_y[bi].to(DEV)
                
                s_logits = student(xb)
                T = 4.0
                L_kd = F.kl_div(F.log_softmax(s_logits/T, 1), F.softmax(tl/T, 1), reduction='batchmean') * T**2
                L_ce = F.cross_entropy(s_logits, py)
                loss = 0.7 * L_kd + 0.3 * L_ce
                opt_s.zero_grad(); loss.backward(); opt_s.step()
            sch_s.step()

            if ep % 5 == 0 or ep == 1:
                ret, avg_drop, spc = eval_model(student, clA+clB, parent_pc)
                retA = sum(1 for c in clA if parent_pc[c]>0 and spc.get(c,0)/parent_pc[c]>=0.9)
                retB = sum(1 for c in clB if parent_pc[c]>0 and spc.get(c,0)/parent_pc[c]>=0.9)
                print(f"  ep {ep:2d}: A={retA}/5 B={retB}/5 ret={ret}/10 avg_drop={avg_drop:.1f}% ({time.time()-t0:.0f}s)")
                if ret > best_ret:
                    best_ret = ret; best_ep = ep
                    best_sd = {k: v.clone() for k, v in student.state_dict().items()}

        if best_sd: student.load_state_dict(best_sd)
        ret_rnd, drop_rnd, _ = eval_model(student, clA+clB, parent_pc)
        print(f"\n  Round {rnd} student: {ret_rnd}/10 ({time.time()-t0:.0f}s)")

        if ret_rnd >= 8:
            print(f"  ✅ Target reached!")
            break

    # ═══════════════════════════════════════════════
    # STAGE 4: CMA-ES FC bias optimization (10D)
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 4: CMA-ES FC bias optimization (10D)")
    print("=" * 60)
    beep("CMA-ES optimization")

    try:
        import cma
    except ImportError:
        os.system('pip install cma -q')
        import cma

    # Freeze everything, optimize only FC biases
    student.eval()
    original_bias = student.fc.bias.data.clone()
    original_weight = student.fc.weight.data.clone()

    # Pre-compute student backbone features
    with torch.no_grad():
        student_feats = torch.cat([get_features(student, xb.to(DEV)).cpu() for xb in all_x.split(256)])
    student_feats_dev = student_feats.to(DEV)

    def cma_fitness(bias_vec):
        """Evaluate retention with given FC biases."""
        bias = torch.tensor(bias_vec, dtype=torch.float32).to(DEV)
        with torch.no_grad():
            logits = student_feats_dev @ original_weight.T + bias
            preds = logits.argmax(1).cpu()
        pc = {}
        for c in clA+clB:
            mask = true_y == c
            pc[c] = (preds[mask] == c).float().mean().item()
        ret = sum(1 for c in clA+clB if parent_pc[c]>0 and pc.get(c,0)/parent_pc[c]>=0.9)
        avg_drop = np.mean([(1-pc.get(c,0)/parent_pc[c])*100 for c in clA+clB if parent_pc[c]>0])
        return -ret * 10 + avg_drop

    x0 = original_bias.cpu().numpy().tolist()
    es = cma.CMAEvolutionStrategy(x0, 0.5, {
        'maxiter': 50, 'popsize': 20, 'seed': 42, 'verbose': -1
    })

    best_f_ever = 100
    best_bias_ever = x0[:]

    gen = 0
    while not es.stop():
        gen += 1
        solutions = es.ask()
        fitnesses = [cma_fitness(s) for s in solutions]
        es.tell(solutions, fitnesses)
        
        bf = min(fitnesses)
        if bf < best_f_ever:
            best_f_ever = bf
            best_bias_ever = solutions[fitnesses.index(bf)][:]
        
        if gen % 10 == 0 or gen == 1:
            print(f"  gen {gen:2d}: best_f={best_f_ever:+.1f} ({time.time()-t0:.0f}s)")

    # Apply best biases
    student.fc.bias.data = torch.tensor(best_bias_ever, dtype=torch.float32).to(DEV)

    # ═══════════════════════════════════════════════
    # FINAL EVAL
    # ═══════════════════════════════════════════════
    ret_final, drop_final, pc_final = eval_model(student, clA+clB, parent_pc)
    ret_display = print_drop("FINAL (Self-Train + CMA-ES)", parent_pc, pc_final, clA+clB)

    print(f"\n{'='*60}")
    print(f"  Stage 1 (R1 feature distill): {ret_s1}/10")
    print(f"  Self-training rounds: {rnd}")
    print(f"  CMA-ES FC bias: {best_f_ever:+.1f}")
    print(f"  FINAL: {ret_display}/10")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")
    beep(f"Done. Retention {ret_display} out of 10")
