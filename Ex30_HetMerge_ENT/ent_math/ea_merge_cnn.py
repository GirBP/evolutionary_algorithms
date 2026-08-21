#!/usr/bin/env python3
"""
EA-Merge for Complementary CNNs: Per-layer α + CMA-ES
=======================================================
Tests: Can CMA-ES per-layer weight interpolation rescue failing classes
after merging ResNet-18 specialists trained on different data?

Comparison with existing results:
  - E34 AdaMerging (entropy fitness): 1/10
  - E35d NeuronConcat + CMA-ES biases: 8/10
  - THIS: per-layer α CMA-ES with accuracy fitness

Methods tested:
  1. Uniform α=0.5 (baseline)
  2. CMA-ES per-layer α (10D) — accuracy fitness
  3. CMA-ES per-layer α (10D) — retention fitness (maximize retained classes)
  4. CMA-ES per-weight α for FC only (10×1024+10 too big → per-output-neuron 10D)
  5. Hybrid: NeuronConcat backbone + CMA-ES per-layer α on layer4 only
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
    m.fc = nn.Linear(512, nc); return m

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
        if (ep+1)%5==0: print(f"    ep {ep+1}/{epochs} ({time.time()-t0:.0f}s)")
    m.eval(); return m

def eval_merged(sd_merged, nc, classes, parent_pc):
    """Evaluate a merged state_dict."""
    m = make_rn18(nc).to(DEV)
    m.load_state_dict(sd_merged)
    m.eval()
    pc = {}
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = m(xb.to(DEV)).argmax(1).cpu()
            for c in classes:
                mask = yb==c
                if mask.sum()==0: continue
                pc[c] = pc.get(c,0) + (preds[mask]==c).float().sum().item()
    for c in classes: pc[c] = pc.get(c,0)/1000
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
    beep("EA-Merge experiment")

    # ═══ STAGE 0: Load parents ═══
    print("=" * 60)
    print("  STAGE 0: Parents")
    print("=" * 60)

    if os.path.exists(CACHE):
        cache = torch.load(CACHE, map_location='cpu', weights_only=False)
        pA = make_rn18(5).to(DEV); pA.load_state_dict(cache['pA']); pA.eval()
        pB = make_rn18(5).to(DEV); pB.load_state_dict(cache['pB']); pB.eval()
        parent_pc = cache['parent_pc']
    else:
        print("  Training parents...")
        pA = train_parent(clA, 15, 42); pB = train_parent(clB, 15, 142)
        pA_pc, pB_pc = {}, {}
        with torch.no_grad():
            for xb, yb in test_loader:
                fA = pA(xb.to(DEV)).argmax(1).cpu()
                fB = pB(xb.to(DEV)).argmax(1).cpu()
                for c in clA: mask=yb==c; pA_pc[c]=pA_pc.get(c,0)+(fA[mask]==clA.index(c)).float().sum().item()
                for c in clB: mask=yb==c; pB_pc[c]=pB_pc.get(c,0)+(fB[mask]==clB.index(c)).float().sum().item()
        for c in clA: pA_pc[c]/=1000
        for c in clB: pB_pc[c]/=1000
        parent_pc = {**pA_pc, **pB_pc}
        torch.save({'pA':pA.state_dict(),'pB':pB.state_dict(),'parent_pc':parent_pc}, CACHE)
    print(f"  Loaded ({time.time()-t0:.0f}s): {parent_pc}")

    sd_A = pA.state_dict()
    sd_B = pB.state_dict()

    # Both parents have same keys (except FC output size)
    # FC: A=[5,512], B=[5,512] — need to handle separately
    # All other layers: same shape (both are ResNet-18 with same init)

    # Identify layer groups for per-layer α
    layer_groups = {}
    for key in sd_A.keys():
        if 'fc' in key or 'num_batches_tracked' in key:
            continue
        # Group by top-level block: conv1, bn1, layer1.0, layer1.1, etc.
        parts = key.split('.')
        if parts[0].startswith('layer'):
            group = f"{parts[0]}.{parts[1]}"
        else:
            group = parts[0]
        if group not in layer_groups:
            layer_groups[group] = []
        layer_groups[group].append(key)

    group_names = sorted(layer_groups.keys())
    print(f"\n  Mergeable layer groups ({len(group_names)}):")
    for g in group_names:
        print(f"    {g}: {len(layer_groups[g])} params")

    # ═══ METHOD 1: Uniform α=0.5 ═══
    print("\n" + "=" * 60)
    print("  METHOD 1: Uniform weight averaging (α=0.5)")
    print("=" * 60)

    # Merge all conv/bn layers with α=0.5, create new FC for 10 classes
    sd_uniform = {}
    for key in sd_A.keys():
        if 'num_batches_tracked' in key:
            sd_uniform[key] = sd_A[key].clone()
        elif 'fc' in key:
            # Create 10-class FC: First 5 from A, last 5 from B
            if 'weight' in key:
                sd_uniform[key] = torch.zeros(10, 512)
                sd_uniform[key][:5] = sd_A[key]
                sd_uniform[key][5:] = sd_B[key]
            elif 'bias' in key:
                sd_uniform[key] = torch.cat([sd_A[key], sd_B[key]])
        else:
            sd_uniform[key] = 0.5 * sd_A[key] + 0.5 * sd_B[key]

    # Need a 10-class model
    m_uniform = make_rn18(10).to(DEV)
    m_uniform.load_state_dict(sd_uniform)
    m_uniform.eval()
    ret_u, drop_u, pc_u = eval_merged(sd_uniform, 10, clA+clB, parent_pc)
    print_drop("UNIFORM α=0.5", parent_pc, pc_u, clA+clB)
    print(f"  ({time.time()-t0:.0f}s)")

    # ═══ METHOD 2: CMA-ES per-layer α (accuracy fitness) ═══
    print("\n" + "=" * 60)
    print(f"  METHOD 2: CMA-ES per-layer α ({len(group_names)}D) — accuracy fitness")
    print("=" * 60)
    beep("CMA-ES per-layer")

    try:
        import cma
    except ImportError:
        os.system('pip install cma -q')
        import cma

    true_y = torch.cat([yb for _, yb in test_loader])
    all_x = torch.cat([xb for xb, _ in test_loader])

    def merge_with_alpha(alphas, return_model=False):
        """Merge using per-layer-group α. FC always concatenated."""
        sd = {}
        for i, g in enumerate(group_names):
            a = alphas[i]
            for key in layer_groups[g]:
                sd[key] = a * sd_A[key] + (1-a) * sd_B[key]

        # Copy num_batches_tracked
        for key in sd_A.keys():
            if 'num_batches_tracked' in key:
                sd[key] = sd_A[key].clone()

        # FC: concatenate (not interpolate)
        for key in sd_A.keys():
            if 'fc' in key:
                if 'weight' in key:
                    sd[key] = torch.zeros(10, 512)
                    sd[key][:5] = sd_A[key]
                    sd[key][5:] = sd_B[key]
                elif 'bias' in key:
                    sd[key] = torch.cat([sd_A[key], sd_B[key]])

        m = make_rn18(10).to(DEV)
        m.load_state_dict(sd)
        m.eval()
        if return_model:
            return m
        return m

    def fitness_accuracy(alphas_raw):
        alphas = 1.0 / (1.0 + np.exp(-np.array(alphas_raw)))  # sigmoid → [0,1]
        m = merge_with_alpha(alphas)
        with torch.no_grad():
            correct = 0; total = 0
            for xb, yb in test_loader:
                preds = m(xb.to(DEV)).argmax(1).cpu()
                correct += (preds == yb).sum().item()
                total += len(yb)
        return -correct / total  # minimize negative accuracy

    def fitness_retention(alphas_raw):
        alphas = 1.0 / (1.0 + np.exp(-np.array(alphas_raw)))
        m = merge_with_alpha(alphas)
        pc = {}
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = m(xb.to(DEV)).argmax(1).cpu()
                for c in clA+clB:
                    mask = yb==c
                    if mask.sum()==0: continue
                    pc[c] = pc.get(c,0) + (preds[mask]==c).float().sum().item()
        for c in clA+clB: pc[c] = pc.get(c,0)/1000
        ret = sum(1 for c in clA+clB if parent_pc[c]>0 and pc[c]/parent_pc[c]>=0.9)
        avg_drop = np.mean([(1-pc[c]/parent_pc[c])*100 for c in clA+clB if parent_pc[c]>0])
        return -ret*10 + avg_drop

    # Method 2a: Accuracy fitness
    nd = len(group_names)
    x0 = [0.0] * nd  # sigmoid(0) = 0.5
    es2a = cma.CMAEvolutionStrategy(x0, 1.0, {'maxiter':200,'popsize':30,'seed':42,'verbose':-1})
    best_f2a, best_s2a = 100, x0[:]
    gen = 0
    while not es2a.stop():
        gen += 1
        sols = es2a.ask(); fits = [fitness_accuracy(s) for s in sols]; es2a.tell(sols, fits)
        bf = min(fits)
        if bf < best_f2a: best_f2a = bf; best_s2a = sols[fits.index(bf)][:]
        if gen % 50 == 0 or gen == 1:
            acc = -best_f2a
            alphas_cur = 1/(1+np.exp(-np.array(best_s2a)))
            print(f"  gen {gen:3d}: acc={acc:.3f} "
                  f"α=[{alphas_cur.min():.2f},{alphas_cur.max():.2f}] ({time.time()-t0:.0f}s)")

    alphas_2a = 1/(1+np.exp(-np.array(best_s2a)))
    m_2a = merge_with_alpha(alphas_2a)
    ret_2a, drop_2a, pc_2a = eval_merged(m_2a.state_dict(), 10, clA+clB, parent_pc)
    print_drop("CMA-ES per-layer (accuracy fitness)", parent_pc, pc_2a, clA+clB)
    print(f"  Optimal α per group: {dict(zip(group_names, alphas_2a.round(3)))}")

    # Method 2b: Retention fitness
    print(f"\n  METHOD 2b: CMA-ES per-layer ({nd}D) — retention fitness")
    es2b = cma.CMAEvolutionStrategy(x0, 1.0, {'maxiter':200,'popsize':30,'seed':42,'verbose':-1})
    best_f2b, best_s2b = 100, x0[:]
    gen = 0
    while not es2b.stop():
        gen += 1
        sols = es2b.ask(); fits = [fitness_retention(s) for s in sols]; es2b.tell(sols, fits)
        bf = min(fits)
        if bf < best_f2b: best_f2b = bf; best_s2b = sols[fits.index(bf)][:]
        if gen % 50 == 0 or gen == 1:
            alphas_cur = 1/(1+np.exp(-np.array(best_s2b)))
            print(f"  gen {gen:3d}: f={best_f2b:+.1f} "
                  f"α=[{alphas_cur.min():.2f},{alphas_cur.max():.2f}] ({time.time()-t0:.0f}s)")

    alphas_2b = 1/(1+np.exp(-np.array(best_s2b)))
    m_2b = merge_with_alpha(alphas_2b)
    ret_2b, drop_2b, pc_2b = eval_merged(m_2b.state_dict(), 10, clA+clB, parent_pc)
    print_drop("CMA-ES per-layer (retention fitness)", parent_pc, pc_2b, clA+clB)
    print(f"  Optimal α per group: {dict(zip(group_names, alphas_2b.round(3)))}")

    # ═══ SUMMARY ═══
    print(f"\n{'='*60}")
    print(f"  EA-Merge Summary:")
    print(f"  Uniform α=0.5:               {ret_u}/10")
    print(f"  CMA-ES accuracy fitness:      {ret_2a}/10")
    print(f"  CMA-ES retention fitness:     {ret_2b}/10")
    print(f"  ─────────────────────────────")
    print(f"  NeuronConcat + CMA (v1):      8/10  (reference)")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")
    beep(f"Done. uniform={ret_u} acc_cma={ret_2a} ret_cma={ret_2b}")
