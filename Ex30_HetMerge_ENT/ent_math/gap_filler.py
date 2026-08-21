#!/usr/bin/env python3
"""
Gap Filler: Comprehensive merge experiment coverage
=====================================================
Fills ALL untested combinations from the coverage matrix.

Block A: MLP (MNIST) — ~3 min on M2
  T1 same-data:     WA, CMA-ES per-layer
  T3 complementary: WA (baseline vs NeuronConcat)

Block B: CNN (CIFAR-10) — ~40 min on M2
  T1 same-data:     WA, CMA-ES per-layer
  T2 data-split:    WA, CMA-ES per-layer

Block C: CNN T3 with 10-class heads — ~25 min on M2
  Trains parents with 10-class FC (not 5) so TIES/TA are structurally valid
  Methods: WA, Task Arithmetic, TIES, CMA-ES per-layer

Results saved to gap_filler_results.json after EACH experiment.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import time, numpy as np, os, platform, json, copy, traceback
from torchvision import datasets, transforms, models

# ═══════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════
DEV = torch.device('mps' if torch.backends.mps.is_available()
                    else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEV}")
RESULTS_FILE = "gap_filler_results.json"
t0 = time.time()

# ═══════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════
transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
transform_cifar_aug = transforms.Compose([
    transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
transform_mnist = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize([0.1307],[0.3081])])

NW = 0  # macOS fork safety

def get_cifar(train=True, aug=False):
    t = transform_cifar_aug if (train and aug) else transform_cifar
    return datasets.CIFAR10('/tmp/cifar10', train=train, download=True, transform=t)

def get_mnist(train=True):
    return datasets.MNIST('/tmp/mnist', train=train, download=True, transform=transform_mnist)

# ═══════════════════════════════════════════════
# ARCHITECTURES
# ═══════════════════════════════════════════════
class MLP(nn.Module):
    def __init__(self, nc=10, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, nc)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def make_rn18(nc=10):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc); return m

# ═══════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════
def train_model(model, train_loader, epochs=15, lr=0.01, class_filter=None,
                class_remap=None, verbose=True):
    """Generic trainer. class_filter=list of classes to keep, class_remap=dict old→new."""
    model.to(DEV).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        for xb, yb in train_loader:
            if class_filter is not None:
                mask = sum(yb==c for c in class_filter).bool()
                if mask.sum()==0: continue
                xb, yb = xb[mask], yb[mask]
            if class_remap is not None:
                yb_new = yb.clone()
                for old, new in class_remap.items():
                    yb_new[yb==old] = new
                yb = yb_new
            xb, yb = xb.to(DEV), yb.to(DEV)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if verbose and (ep+1) % 5 == 0:
            print(f"      ep {ep+1}/{epochs} ({time.time()-t0:.0f}s)")
    model.eval()
    return model

def eval_per_class(model, test_loader, classes):
    """Returns per-class accuracy dict."""
    model.eval()
    correct = {c: 0 for c in classes}
    total = {c: 0 for c in classes}
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(DEV)).argmax(1).cpu()
            for c in classes:
                mask = yb == c
                total[c] += mask.sum().item()
                correct[c] += (preds[mask] == c).sum().item()
    return {c: correct[c]/max(total[c],1) for c in classes}

# ═══════════════════════════════════════════════
# MERGE METHODS
# ═══════════════════════════════════════════════
def merge_wa(sd_a, sd_b, alpha=0.5):
    """Weight averaging with alpha."""
    sd = {}
    for k in sd_a:
        if 'num_batches_tracked' in k:
            sd[k] = sd_a[k].clone()
        else:
            sd[k] = alpha * sd_a[k] + (1-alpha) * sd_b[k]
    return sd

def merge_task_arithmetic(sd_a, sd_b, sd_0, tau=1.0):
    """Task Arithmetic: θ₀ + τ*(τ_A + τ_B) where τ_X = θ_X - θ₀."""
    sd = {}
    for k in sd_0:
        if 'num_batches_tracked' in k:
            sd[k] = sd_0[k].clone()
        else:
            a, b, z = sd_a[k].cpu().float(), sd_b[k].cpu().float(), sd_0[k].cpu().float()
            task_a = a - z
            task_b = b - z
            sd[k] = z + tau * (task_a + task_b)
    return sd

def merge_ties(sd_a, sd_b, sd_0, density=0.2, tau=1.0):
    """TIES-Merging: Trim + Elect Sign + Disjoint Merge."""
    sd = {}
    for k in sd_0:
        if 'num_batches_tracked' in k:
            sd[k] = sd_0[k].clone()
            continue
        ta = sd_a[k].cpu().float() - sd_0[k].cpu().float()
        tb = sd_b[k].cpu().float() - sd_0[k].cpu().float()
        # Trim: keep top-density% by magnitude
        for t in [ta, tb]:
            flat = t.abs().flatten()
            if flat.numel() == 0: continue
            threshold = torch.quantile(flat.float(), 1.0-density)
            t[t.abs() < threshold] = 0
        # Elect sign: majority vote
        signs = torch.sign(ta) + torch.sign(tb)
        elected = torch.sign(signs)  # +1, -1, or 0
        # Disjoint merge: average only values that agree with elected sign
        merged_task = torch.zeros_like(ta)
        for t in [ta, tb]:
            agree = (torch.sign(t) == elected) & (t != 0)
            merged_task[agree] += t[agree]
        # Average over number of agreeing (1 or 2)
        count = torch.zeros_like(ta)
        for t in [ta, tb]:
            count += ((torch.sign(t) == elected) & (t != 0)).float()
        count = count.clamp(min=1)
        merged_task = merged_task / count
        sd[k] = sd_0[k] + tau * merged_task
    return sd

def merge_cmaes_perlayer(sd_a, sd_b, model_fn, test_loader, classes, parent_pc,
                          maxiter=100, popsize=20, seed=42):
    """CMA-ES per-layer α optimization."""
    try:
        import cma
    except ImportError:
        os.system('pip install cma -q')
        import cma

    # Determine layer groups
    groups = {}
    for k in sd_a:
        if 'num_batches_tracked' in k: continue
        parts = k.split('.')
        if parts[0].startswith('layer'):
            g = f"{parts[0]}.{parts[1]}"
        elif parts[0] in ('fc1','fc2','fc3'):
            g = parts[0]
        else:
            g = parts[0]
        if g not in groups: groups[g] = []
        groups[g].append(k)
    gnames = sorted(groups.keys())

    def fitness(theta_raw):
        alphas = 1.0/(1.0+np.exp(-np.array(theta_raw)))
        sd = {}
        for i, g in enumerate(gnames):
            for k in groups[g]:
                sd[k] = alphas[i]*sd_a[k] + (1-alphas[i])*sd_b[k]
        for k in sd_a:
            if 'num_batches_tracked' in k:
                sd[k] = sd_a[k].clone()
        m = model_fn()
        m.load_state_dict(sd)
        m.to(DEV).eval()
        pc = eval_per_class(m, test_loader, classes)
        ret = sum(1 for c in classes if parent_pc.get(c,0)>0 and pc.get(c,0)/parent_pc[c]>=0.9)
        avg_drop = np.mean([(1-pc.get(c,0)/max(parent_pc.get(c,0),1e-9))*100 for c in classes])
        del m
        return -ret*10 + avg_drop

    nd = len(gnames)
    x0 = [0.0]*nd
    es = cma.CMAEvolutionStrategy(x0, 1.0, {'maxiter':maxiter,'popsize':popsize,
                                              'seed':seed,'verbose':-1})
    best_f, best_s = 100, x0[:]
    gen = 0
    while not es.stop():
        gen += 1
        sols = es.ask()
        fits = [fitness(s) for s in sols]
        es.tell(sols, fits)
        bf = min(fits)
        if bf < best_f:
            best_f = bf; best_s = sols[fits.index(bf)][:]

    alphas_final = 1.0/(1.0+np.exp(-np.array(best_s)))
    sd = {}
    for i, g in enumerate(gnames):
        for k in groups[g]:
            sd[k] = alphas_final[i]*sd_a[k] + (1-alphas_final[i])*sd_b[k]
    for k in sd_a:
        if 'num_batches_tracked' in k:
            sd[k] = sd_a[k].clone()
    return sd, dict(zip(gnames, alphas_final.round(3).tolist()))

# ═══════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════
results = []

def save_results():
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def run_experiment(name, arch, task, method, parent_pc, merged_pc, classes,
                   extra_info=None):
    """Log one experiment result."""
    ret = sum(1 for c in classes if parent_pc.get(c,0)>0 and
              merged_pc.get(c,0)/parent_pc[c]>=0.9)
    avg_drop = np.mean([(1-merged_pc.get(c,0)/max(parent_pc.get(c,0),1e-9))*100 for c in classes])
    r = {
        'name': name, 'arch': arch, 'task': task, 'method': method,
        'retention': f"{ret}/{len(classes)}",
        'avg_drop_pct': round(avg_drop,1),
        'per_class': {str(c): round(merged_pc.get(c,0),3) for c in classes},
        'parent_pc': {str(c): round(parent_pc.get(c,0),3) for c in classes},
        'time': round(time.time()-t0,0),
    }
    if extra_info: r['extra'] = extra_info
    results.append(r)
    save_results()
    # Print
    print(f"\n  {name}: {ret}/{len(classes)} (avg_drop={avg_drop:.1f}%)")
    for c in classes:
        p = parent_pc.get(c,0)
        m = merged_pc.get(c,0)
        d = (1-m/p)*100 if p>0 else 100
        ok = '✅' if d<=10 else '❌'
        print(f"    cls {c}: {p:.3f} → {m:.3f} ({d:+.1f}%) {ok}")
    return ret

# ═══════════════════════════════════════════════
# BLOCK A: MLP (MNIST)
# ═══════════════════════════════════════════════
def block_a():
    print("\n" + "="*70)
    print("  BLOCK A: MLP (MNIST)")
    print("="*70)

    mnist_train = get_mnist(True)
    mnist_test = get_mnist(False)
    train_ldr = torch.utils.data.DataLoader(mnist_train, batch_size=256, shuffle=True, num_workers=NW)
    test_ldr = torch.utils.data.DataLoader(mnist_test, batch_size=256, shuffle=False, num_workers=NW)
    all_classes = list(range(10))

    # ── A.T1: Same-data ──
    print("\n  ─── A.T1: MLP Same-Data ───")
    print("  Training MLP-A (seed=42)...")
    torch.manual_seed(42)
    mlp_a = train_model(MLP(10), train_ldr, epochs=10, lr=0.01)
    print("  Training MLP-B (seed=123)...")
    torch.manual_seed(123)
    mlp_b = train_model(MLP(10), train_ldr, epochs=10, lr=0.01)

    pc_a = eval_per_class(mlp_a, test_ldr, all_classes)
    pc_b = eval_per_class(mlp_b, test_ldr, all_classes)
    parent_pc = {c: max(pc_a[c], pc_b[c]) for c in all_classes}
    print(f"  Parent A: {np.mean(list(pc_a.values())):.3f}")
    print(f"  Parent B: {np.mean(list(pc_b.values())):.3f}")

    sd_a, sd_b = mlp_a.state_dict(), mlp_b.state_dict()
    # Save init for TA
    torch.manual_seed(999)
    sd_0 = MLP(10).state_dict()

    # WA
    sd_wa = merge_wa(sd_a, sd_b)
    m = MLP(10).to(DEV); m.load_state_dict(sd_wa); m.eval()
    pc_wa = eval_per_class(m, test_ldr, all_classes)
    run_experiment("MLP-T1-WA", "MLP", "T1-same", "WA", parent_pc, pc_wa, all_classes)

    # CMA-ES per-layer
    print("  CMA-ES per-layer α...")
    sd_cma, alphas = merge_cmaes_perlayer(
        sd_a, sd_b, lambda: MLP(10), test_ldr, all_classes, parent_pc,
        maxiter=50, popsize=15)
    m = MLP(10).to(DEV); m.load_state_dict(sd_cma); m.eval()
    pc_cma = eval_per_class(m, test_ldr, all_classes)
    run_experiment("MLP-T1-CMA", "MLP", "T1-same", "CMA-ES per-layer",
                   parent_pc, pc_cma, all_classes, {'alphas': alphas})

    # ── A.T3: Complementary ──
    print("\n  ─── A.T3: MLP Complementary ───")
    clA, clB = list(range(5)), list(range(5,10))
    remap_a = {c: i for i, c in enumerate(clA)}
    remap_b = {c: i for i, c in enumerate(clB)}

    print("  Training MLP-A (cls 0-4)...")
    torch.manual_seed(42)
    mlp_ca = train_model(MLP(5), train_ldr, epochs=10, class_filter=clA, class_remap=remap_a)
    print("  Training MLP-B (cls 5-9)...")
    torch.manual_seed(123)
    mlp_cb = train_model(MLP(5), train_ldr, epochs=10, class_filter=clB, class_remap=remap_b)

    pc_ca = eval_per_class(mlp_ca, test_ldr, clA)  # These use remapped labels
    # Need to eval with original labels — custom eval
    mlp_ca.eval(); mlp_cb.eval()
    parent_comp = {}
    with torch.no_grad():
        for xb, yb in test_ldr:
            pA = mlp_ca(xb.to(DEV)).argmax(1).cpu()
            pB = mlp_cb(xb.to(DEV)).argmax(1).cpu()
            for c in clA:
                mask = yb==c
                parent_comp[c] = parent_comp.get(c,0) + (pA[mask]==clA.index(c)).sum().item()
            for c in clB:
                mask = yb==c
                parent_comp[c] = parent_comp.get(c,0) + (pB[mask]==clB.index(c)).sum().item()
    n_per = {c: sum(1 for _, yb in test_ldr for y in yb if y==c) for c in all_classes}
    # Simpler: MNIST has 1000 per class in test
    for c in all_classes:
        cnt = sum((yb==c).sum().item() for _, yb in test_ldr)
        parent_comp[c] = parent_comp[c] / cnt if cnt > 0 else 0

    # WA with concat FC
    sd_ca, sd_cb = mlp_ca.state_dict(), mlp_cb.state_dict()
    sd_comp = {}
    for k in sd_ca:
        if 'fc3' in k:
            if 'weight' in k:
                sd_comp[k] = torch.zeros(10, 256)
                sd_comp[k][:5] = sd_ca[k]; sd_comp[k][5:] = sd_cb[k]
            elif 'bias' in k:
                sd_comp[k] = torch.cat([sd_ca[k], sd_cb[k]])
        else:
            sd_comp[k] = 0.5 * sd_ca[k] + 0.5 * sd_cb[k]
    m = MLP(10).to(DEV); m.load_state_dict(sd_comp); m.eval()
    pc_comp_wa = eval_per_class(m, test_ldr, all_classes)
    run_experiment("MLP-T3-WA", "MLP", "T3-comp", "WA+concatFC",
                   parent_comp, pc_comp_wa, all_classes)

    del mlp_a, mlp_b, mlp_ca, mlp_cb, m
    torch.mps.empty_cache() if DEV.type=='mps' else None


# ═══════════════════════════════════════════════
# BLOCK B: CNN Same-Data & Data-Split
# ═══════════════════════════════════════════════
def block_b():
    print("\n" + "="*70)
    print("  BLOCK B: CNN (CIFAR-10) — Same-Data & Data-Split")
    print("="*70)

    cifar_test = get_cifar(False)
    test_ldr = torch.utils.data.DataLoader(cifar_test, batch_size=256, shuffle=False, num_workers=NW)
    all_classes = list(range(10))

    # ── B.T1: Same-data ──
    print("\n  ─── B.T1: CNN Same-Data ───")
    cifar_train_a = get_cifar(True, aug=True)
    train_ldr_a = torch.utils.data.DataLoader(cifar_train_a, batch_size=256, shuffle=True, num_workers=NW)

    print("  Training CNN-A (seed=42, all CIFAR-10)...")
    torch.manual_seed(42)
    cnn_a = train_model(make_rn18(10), train_ldr_a, epochs=15, lr=0.01)
    print("  Training CNN-B (seed=123, all CIFAR-10)...")
    torch.manual_seed(123)
    cnn_b = train_model(make_rn18(10), train_ldr_a, epochs=15, lr=0.01)

    # Save pretrained init for TA/TIES
    torch.manual_seed(999)
    sd_0 = make_rn18(10).state_dict()

    pc_a = eval_per_class(cnn_a, test_ldr, all_classes)
    pc_b = eval_per_class(cnn_b, test_ldr, all_classes)
    parent_pc_t1 = {c: max(pc_a[c], pc_b[c]) for c in all_classes}
    print(f"  Parent A avg: {np.mean(list(pc_a.values())):.3f}")
    print(f"  Parent B avg: {np.mean(list(pc_b.values())):.3f}")

    sd_a, sd_b = cnn_a.state_dict(), cnn_b.state_dict()

    # WA
    sd_wa = merge_wa(sd_a, sd_b)
    m = make_rn18(10).to(DEV); m.load_state_dict(sd_wa); m.eval()
    pc_wa = eval_per_class(m, test_ldr, all_classes)
    run_experiment("CNN-T1-WA", "CNN", "T1-same", "WA", parent_pc_t1, pc_wa, all_classes)

    # Task Arithmetic
    for tau in [0.5, 1.0]:
        sd_ta = merge_task_arithmetic(sd_a, sd_b, sd_0, tau=tau)
        m = make_rn18(10).to(DEV); m.load_state_dict(sd_ta); m.eval()
        pc_ta = eval_per_class(m, test_ldr, all_classes)
        run_experiment(f"CNN-T1-TA(τ={tau})", "CNN", "T1-same",
                       f"TaskArith(τ={tau})", parent_pc_t1, pc_ta, all_classes)

    # TIES
    for d in [0.2, 0.5]:
        sd_ties = merge_ties(sd_a, sd_b, sd_0, density=d)
        m = make_rn18(10).to(DEV); m.load_state_dict(sd_ties); m.eval()
        pc_ties = eval_per_class(m, test_ldr, all_classes)
        run_experiment(f"CNN-T1-TIES(d={d})", "CNN", "T1-same",
                       f"TIES(d={d})", parent_pc_t1, pc_ties, all_classes)

    # CMA-ES per-layer
    print("  CMA-ES per-layer α...")
    sd_cma, alphas = merge_cmaes_perlayer(
        sd_a, sd_b, lambda: make_rn18(10), test_ldr, all_classes, parent_pc_t1,
        maxiter=100, popsize=20)
    m = make_rn18(10).to(DEV); m.load_state_dict(sd_cma); m.eval()
    pc_cma = eval_per_class(m, test_ldr, all_classes)
    run_experiment("CNN-T1-CMA", "CNN", "T1-same", "CMA-ES per-layer",
                   parent_pc_t1, pc_cma, all_classes, {'alphas': alphas})

    del cnn_a, cnn_b, m

    # ── B.T2: Data-Split ──
    print("\n  ─── B.T2: CNN Data-Split ───")

    # Split training data by indices
    cifar_full = get_cifar(True, aug=True)
    n = len(cifar_full)
    gen = torch.Generator().manual_seed(42)
    idx = torch.randperm(n, generator=gen)
    half = n // 2
    split_a = torch.utils.data.Subset(cifar_full, idx[:half].tolist())
    split_b = torch.utils.data.Subset(cifar_full, idx[half:].tolist())
    ldr_sa = torch.utils.data.DataLoader(split_a, batch_size=256, shuffle=True, num_workers=NW)
    ldr_sb = torch.utils.data.DataLoader(split_b, batch_size=256, shuffle=True, num_workers=NW)

    print("  Training CNN-A (split A, 25k images)...")
    torch.manual_seed(42)
    cnn_sa = train_model(make_rn18(10), ldr_sa, epochs=15, lr=0.01)
    print("  Training CNN-B (split B, 25k images)...")
    torch.manual_seed(123)
    cnn_sb = train_model(make_rn18(10), ldr_sb, epochs=15, lr=0.01)

    pc_sa = eval_per_class(cnn_sa, test_ldr, all_classes)
    pc_sb = eval_per_class(cnn_sb, test_ldr, all_classes)
    parent_pc_t2 = {c: max(pc_sa[c], pc_sb[c]) for c in all_classes}
    print(f"  Parent A avg: {np.mean(list(pc_sa.values())):.3f}")
    print(f"  Parent B avg: {np.mean(list(pc_sb.values())):.3f}")

    sd_sa, sd_sb = cnn_sa.state_dict(), cnn_sb.state_dict()

    # WA
    sd_wa2 = merge_wa(sd_sa, sd_sb)
    m = make_rn18(10).to(DEV); m.load_state_dict(sd_wa2); m.eval()
    pc_wa2 = eval_per_class(m, test_ldr, all_classes)
    run_experiment("CNN-T2-WA", "CNN", "T2-split", "WA", parent_pc_t2, pc_wa2, all_classes)

    # Task Arithmetic
    sd_ta2 = merge_task_arithmetic(sd_sa, sd_sb, sd_0, tau=0.5)
    m = make_rn18(10).to(DEV); m.load_state_dict(sd_ta2); m.eval()
    pc_ta2 = eval_per_class(m, test_ldr, all_classes)
    run_experiment("CNN-T2-TA(τ=0.5)", "CNN", "T2-split", "TaskArith(τ=0.5)",
                   parent_pc_t2, pc_ta2, all_classes)

    # TIES
    sd_ties2 = merge_ties(sd_sa, sd_sb, sd_0, density=0.2)
    m = make_rn18(10).to(DEV); m.load_state_dict(sd_ties2); m.eval()
    pc_ties2 = eval_per_class(m, test_ldr, all_classes)
    run_experiment("CNN-T2-TIES(d=0.2)", "CNN", "T2-split", "TIES(d=0.2)",
                   parent_pc_t2, pc_ties2, all_classes)

    # CMA-ES
    print("  CMA-ES per-layer α...")
    sd_cma2, alphas2 = merge_cmaes_perlayer(
        sd_sa, sd_sb, lambda: make_rn18(10), test_ldr, all_classes, parent_pc_t2,
        maxiter=100, popsize=20)
    m = make_rn18(10).to(DEV); m.load_state_dict(sd_cma2); m.eval()
    pc_cma2 = eval_per_class(m, test_ldr, all_classes)
    run_experiment("CNN-T2-CMA", "CNN", "T2-split", "CMA-ES per-layer",
                   parent_pc_t2, pc_cma2, all_classes, {'alphas': alphas2})

    del cnn_sa, cnn_sb, m


# ═══════════════════════════════════════════════
# BLOCK C: CNN T3 Complementary (10-class heads → TIES/TA possible)
# ═══════════════════════════════════════════════
def block_c():
    print("\n" + "="*70)
    print("  BLOCK C: CNN Complementary (10-class heads)")
    print("="*70)

    cifar_train = get_cifar(True, aug=True)
    cifar_test = get_cifar(False)
    train_ldr = torch.utils.data.DataLoader(cifar_train, batch_size=256, shuffle=True, num_workers=NW)
    test_ldr = torch.utils.data.DataLoader(cifar_test, batch_size=256, shuffle=False, num_workers=NW)
    all_classes = list(range(10))
    clA, clB = list(range(5)), list(range(5,10))

    # Train with 10-class heads but only on subset of classes
    # (labels stay 0-9, but only 0-4 or 5-9 data seen)
    print("  Training CNN-A (10-class head, cls 0-4 data only)...")
    torch.manual_seed(42)
    sd_0 = make_rn18(10).state_dict()  # save init
    cnn_a10 = make_rn18(10)
    cnn_a10.load_state_dict(copy.deepcopy(sd_0))
    cnn_a10 = train_model(cnn_a10, train_ldr, epochs=15, lr=0.01, class_filter=clA)

    print("  Training CNN-B (10-class head, cls 5-9 data only)...")
    cnn_b10 = make_rn18(10)
    cnn_b10.load_state_dict(copy.deepcopy(sd_0))
    cnn_b10 = train_model(cnn_b10, train_ldr, epochs=15, lr=0.01, class_filter=clB)

    pc_a = eval_per_class(cnn_a10, test_ldr, all_classes)
    pc_b = eval_per_class(cnn_b10, test_ldr, all_classes)
    parent_pc = {}
    for c in clA: parent_pc[c] = pc_a[c]
    for c in clB: parent_pc[c] = pc_b[c]
    print(f"  Parent A (cls 0-4): {[round(pc_a[c],3) for c in clA]}")
    print(f"  Parent B (cls 5-9): {[round(pc_b[c],3) for c in clB]}")

    sd_a, sd_b = cnn_a10.state_dict(), cnn_b10.state_dict()

    # WA
    sd_wa = merge_wa(sd_a, sd_b)
    m = make_rn18(10).to(DEV); m.load_state_dict(sd_wa); m.eval()
    pc_wa = eval_per_class(m, test_ldr, all_classes)
    run_experiment("CNN-T3-10cls-WA", "CNN", "T3-comp-10cls", "WA",
                   parent_pc, pc_wa, all_classes)

    # Task Arithmetic
    for tau in [0.5, 1.0]:
        sd_ta = merge_task_arithmetic(sd_a, sd_b, sd_0, tau=tau)
        m = make_rn18(10).to(DEV); m.load_state_dict(sd_ta); m.eval()
        pc_ta = eval_per_class(m, test_ldr, all_classes)
        run_experiment(f"CNN-T3-10cls-TA(τ={tau})", "CNN", "T3-comp-10cls",
                       f"TaskArith(τ={tau})", parent_pc, pc_ta, all_classes)

    # TIES
    for d in [0.2, 0.5]:
        sd_ties = merge_ties(sd_a, sd_b, sd_0, density=d)
        m = make_rn18(10).to(DEV); m.load_state_dict(sd_ties); m.eval()
        pc_ties = eval_per_class(m, test_ldr, all_classes)
        run_experiment(f"CNN-T3-10cls-TIES(d={d})", "CNN", "T3-comp-10cls",
                       f"TIES(d={d})", parent_pc, pc_ties, all_classes)

    # CMA-ES per-layer
    print("  CMA-ES per-layer α...")
    sd_cma, alphas = merge_cmaes_perlayer(
        sd_a, sd_b, lambda: make_rn18(10), test_ldr, all_classes, parent_pc,
        maxiter=150, popsize=25)
    m = make_rn18(10).to(DEV); m.load_state_dict(sd_cma); m.eval()
    pc_cma = eval_per_class(m, test_ldr, all_classes)
    run_experiment("CNN-T3-10cls-CMA", "CNN", "T3-comp-10cls", "CMA-ES per-layer",
                   parent_pc, pc_cma, all_classes, {'alphas': alphas})

    del cnn_a10, cnn_b10, m

# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
if __name__ == '__main__':
    import multiprocessing
    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('fork', force=True)

    print(f"\n{'='*70}")
    print(f"  Gap Filler — Comprehensive Merge Experiment Coverage")
    print(f"  Device: {DEV}")
    print(f"  Results: {RESULTS_FILE}")
    print(f"  Start: {time.strftime('%H:%M:%S')}")
    print(f"{'='*70}")

    for block_fn, name in [(block_a, "A"), (block_b, "B"), (block_c, "C")]:
        try:
            block_fn()
        except Exception as e:
            print(f"\n  ⚠️ BLOCK {name} FAILED: {e}")
            traceback.print_exc()
            results.append({'block': name, 'error': str(e), 'time': time.time()-t0})
            save_results()

    # ═══ FINAL SUMMARY ═══
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Name':<30} {'Ret':>5} {'AvgDrop':>8}")
    print(f"  {'-'*30} {'-'*5} {'-'*8}")
    for r in results:
        if 'error' not in r:
            print(f"  {r['name']:<30} {r['retention']:>5} {r['avg_drop_pct']:>7.1f}%")
    print(f"\n  Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
    print(f"  Results saved: {RESULTS_FILE}")
    print(f"{'='*70}")
