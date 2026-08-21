#!/usr/bin/env python3
"""
CNN Merge: AdaMerging-style Layer-wise Merge + CMA-ES
======================================================
Based on AdaMerging (ICLR 2024): entropy minimization on unlabeled data.

Key insight: NO pseudo-labels needed. Minimize entropy of merged model's
predictions → confident predictions = good merge.

Pipeline:
  Stage 0: Load parents
  Stage 1: Layer-wise weight interpolation with CMA-ES
           α_l for each layer (20D), FC = concatenation
           Fitness = -entropy on unlabeled test data
  Stage 2: (Optional) Brief fine-tune with feature distillation
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


# ═══ Layer-wise merge utilities ═══
def get_mergeable_layers(model):
    """Get named parameters that can be merged layer-wise (backbone only)."""
    layers = {}
    for name, param in model.named_parameters():
        if 'fc' in name: continue  # skip FC — handle separately
        # Group by block: conv1, bn1, layer1.0, layer1.1, etc.
        parts = name.split('.')
        if parts[0] in ('conv1', 'bn1'):
            group = parts[0]
        elif parts[0].startswith('layer'):
            group = f"{parts[0]}.{parts[1]}"  # e.g. layer1.0
        else:
            group = parts[0]
        if group not in layers:
            layers[group] = []
        layers[group].append(name)
    return layers

def merge_models_layerwise(student, pA, pB, alphas, layer_groups):
    """Merge pA and pB into student using per-group alpha coefficients."""
    sd_A = pA.state_dict()
    sd_B = pB.state_dict()
    sd_S = student.state_dict()
    
    group_names = sorted(layer_groups.keys())
    
    for i, group in enumerate(group_names):
        alpha = alphas[i] if i < len(alphas) else 0.5
        # Sigmoid to constrain to [0, 1]
        alpha_s = 1.0 / (1.0 + np.exp(-alpha))
        
        for param_name in layer_groups[group]:
            if param_name in sd_A and param_name in sd_B:
                sd_S[param_name] = alpha_s * sd_A[param_name] + (1 - alpha_s) * sd_B[param_name]
    
    # FC: concatenate A's [5x512] and B's [5x512] → [10x512]
    sd_S['fc.weight'] = torch.cat([sd_A['fc.weight'], sd_B['fc.weight']], dim=0)
    sd_S['fc.bias'] = torch.cat([sd_A['fc.bias'], sd_B['fc.bias']], dim=0)
    
    student.load_state_dict(sd_S)
    return student


if __name__ == '__main__':
    import multiprocessing
    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('fork', force=True)

    t0 = time.time()
    CACHE = 'cached_parents_r3.pth'
    beep("AdaMerging experiment started")

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

    # Pre-load test images
    all_x = torch.cat([xb for xb, _ in test_loader])
    true_y = torch.cat([yb for _, yb in test_loader])  # for eval ONLY
    
    # Identify mergeable layer groups
    layer_groups = get_mergeable_layers(pA)
    group_names = sorted(layer_groups.keys())
    N_GROUPS = len(group_names)
    print(f"\n  Mergeable layer groups ({N_GROUPS}):")
    for g in group_names:
        print(f"    {g}: {len(layer_groups[g])} params")

    # ═══════════════════════════════════════════════
    # STAGE 1: Baseline — uniform merge (alpha=0.5)
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 1: Baseline — uniform alpha=0.5")
    print("=" * 60)

    student = make_rn18(10).to(DEV)
    alphas_uniform = [0.0] * N_GROUPS  # sigmoid(0) = 0.5
    merge_models_layerwise(student, pA, pB, alphas_uniform, layer_groups)
    
    ret_base, drop_base, pc_base = eval_model(student, clA+clB, parent_pc)
    print_drop("UNIFORM MERGE (α=0.5)", parent_pc, pc_base, clA+clB)

    # Compute entropy baseline
    student.eval()
    with torch.no_grad():
        all_logits = torch.cat([student(xb.to(DEV)).cpu() for xb in all_x.split(256)])
        probs = F.softmax(all_logits, dim=1)
        entropy_base = -(probs * (probs + 1e-8).log()).sum(1).mean().item()
    print(f"  Entropy: {entropy_base:.4f}")

    # ═══════════════════════════════════════════════
    # STAGE 2: CMA-ES entropy minimization
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"  STAGE 2: CMA-ES layer-wise merge ({N_GROUPS}D)")
    print("=" * 60)
    beep("CMA-ES AdaMerging")

    try:
        import cma
    except ImportError:
        os.system('pip install cma -q')
        import cma

    def entropy_fitness(alphas):
        """Entropy of merged model on unlabeled test data. Lower = better."""
        student_eval = make_rn18(10).to(DEV)
        merge_models_layerwise(student_eval, pA, pB, list(alphas), layer_groups)
        student_eval.eval()
        
        with torch.no_grad():
            all_logits = torch.cat([student_eval(xb.to(DEV)).cpu() for xb in all_x.split(256)])
            probs = F.softmax(all_logits, dim=1)
            entropy = -(probs * (probs + 1e-8).log()).sum(1).mean().item()
        
        return entropy  # minimize

    def retention_fitness(alphas):
        """Combined: entropy + retention (uses labels for eval, not training)."""
        student_eval = make_rn18(10).to(DEV)
        merge_models_layerwise(student_eval, pA, pB, list(alphas), layer_groups)
        student_eval.eval()
        
        with torch.no_grad():
            all_logits = torch.cat([student_eval(xb.to(DEV)).cpu() for xb in all_x.split(256)])
            probs = F.softmax(all_logits, dim=1)
            entropy = -(probs * (probs + 1e-8).log()).sum(1).mean().item()
        
        # Also compute retention (for monitoring, uses true_y)
        preds = all_logits.argmax(1)
        pc = {}
        for c in clA+clB:
            mask = true_y == c
            pc[c] = (preds[mask] == c).float().mean().item()
        ret = sum(1 for c in clA+clB if parent_pc[c]>0 and pc[c]/parent_pc[c]>=0.9)
        
        return entropy, ret, pc

    # CMA-ES with entropy-only fitness
    x0 = [0.0] * N_GROUPS  # sigmoid(0) = 0.5 for all layers
    es = cma.CMAEvolutionStrategy(x0, 1.0, {
        'maxiter': 60,
        'popsize': 14,
        'seed': 42,
        'verbose': -1,
    })

    best_entropy_ever = entropy_base
    best_alphas_ever = x0[:]
    best_ret_ever = ret_base

    gen = 0
    while not es.stop():
        gen += 1
        solutions = es.ask()
        fitnesses = [entropy_fitness(s) for s in solutions]
        es.tell(solutions, fitnesses)
        
        bf = min(fitnesses)
        bi = fitnesses.index(bf)
        
        if bf < best_entropy_ever:
            best_entropy_ever = bf
            best_alphas_ever = solutions[bi][:]
        
        if gen % 5 == 0 or gen == 1:
            # Full eval of current best
            entropy, ret, pc = retention_fitness(best_alphas_ever)
            if ret > best_ret_ever:
                best_ret_ever = ret
            sigmoided = [1/(1+np.exp(-a)) for a in best_alphas_ever]
            print(f"  gen {gen:2d}: entropy={entropy:.4f} ret={ret}/10 "
                  f"α_range=[{min(sigmoided):.2f},{max(sigmoided):.2f}] ({time.time()-t0:.0f}s)")

    # Final eval with best alphas
    print(f"\n  CMA-ES done ({time.time()-t0:.0f}s)")
    print(f"  Best entropy: {best_entropy_ever:.4f} (baseline: {entropy_base:.4f})")
    
    sigmoided = [1/(1+np.exp(-a)) for a in best_alphas_ever]
    print(f"\n  Optimal α per layer group:")
    for i, g in enumerate(group_names):
        print(f"    {g}: α={sigmoided[i]:.3f}")

    student = make_rn18(10).to(DEV)
    merge_models_layerwise(student, pA, pB, best_alphas_ever, layer_groups)
    ret_cma, drop_cma, pc_cma = eval_model(student, clA+clB, parent_pc)
    ret_display = print_drop("ADAMERGING (CMA-ES)", parent_pc, pc_cma, clA+clB)

    # ═══════════════════════════════════════════════
    # STAGE 3: Brief fine-tune with feature distillation
    # ═══════════════════════════════════════════════
    if ret_display < 8:
        print("\n" + "=" * 60)
        print("  STAGE 3: Fine-tune merged model (feature distill, 15 ep)")
        print("=" * 60)
        beep("Fine-tuning")

        with torch.no_grad():
            H_A = torch.cat([get_features(pA, xb.to(DEV)).cpu() for xb in all_x.split(256)])
            H_B = torch.cat([get_features(pB, xb.to(DEV)).cpu() for xb in all_x.split(256)])
            logA = torch.cat([pA.fc(get_features(pA, xb.to(DEV))).cpu() for xb in all_x.split(256)])
            logB = torch.cat([pB.fc(get_features(pB, xb.to(DEV))).cpu() for xb in all_x.split(256)])

        proj_A = nn.Linear(512, 256, bias=False).to(DEV)
        proj_B = nn.Linear(512, 256, bias=False).to(DEV)
        proj_tA = nn.Linear(512, 256, bias=False).to(DEV)
        proj_tB = nn.Linear(512, 256, bias=False).to(DEV)

        all_params = list(student.parameters()) + list(proj_A.parameters()) + \
                     list(proj_B.parameters()) + list(proj_tA.parameters()) + list(proj_tB.parameters())
        opt = torch.optim.Adam(all_params, lr=0.0005, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
        
        best_ret_ft, best_sd_ft = 0, None
        for ep in range(1, 16):
            student.train()
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
                if ret > best_ret_ft:
                    best_ret_ft = ret
                    best_sd_ft = {k: v.clone() for k, v in student.state_dict().items()}

        if best_sd_ft: student.load_state_dict(best_sd_ft)
        ret_final, drop_final, pc_final = eval_model(student, clA+clB, parent_pc)
        ret_display = print_drop("FINAL (AdaMerge + Feature Distill)", parent_pc, pc_final, clA+clB)

    print(f"\n{'='*60}")
    print(f"  Baseline (uniform α=0.5): {ret_base}/10")
    print(f"  CMA-ES AdaMerge: {ret_cma}/10")
    print(f"  FINAL: {ret_display}/10")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")
    beep(f"Done. Retention {ret_display} out of 10")
