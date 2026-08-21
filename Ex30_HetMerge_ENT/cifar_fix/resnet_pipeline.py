#!/usr/bin/env python3
"""Phase 1+2+3: ResNet-18 pretrained parents + ENT merge + significance.
All phases in one script for efficiency.
"""
import numpy as np, torch, torch.nn as nn, random, copy, json, time, sys
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import torchvision, torchvision.models as models
from torchvision import datasets, transforms
from scipy import stats
import cma

DEVICE = 'cpu'  # M2 Air
t0 = time.time()

# ═══════════════════════════════════════════
# Data — full CIFAR-10
# ═══════════════════════════════════════════
print("Loading CIFAR-10...", flush=True)
raw_tr = datasets.CIFAR10('/tmp/cifar10', train=True, download=True)
raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)

mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)  # ImageNet mean for pretrained
std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

X_tr_all = torch.tensor(raw_tr.data).permute(0,3,1,2).float() / 255.0
X_tr_all = (X_tr_all - mean) / std
y_tr_all = torch.tensor(raw_tr.targets)

X_te_all = torch.tensor(raw_te.data).permute(0,3,1,2).float() / 255.0
X_te_all = (X_te_all - mean) / std
y_te_all = torch.tensor(raw_te.targets)

clA, clB = list(range(5)), list(range(5,10))
ALL_CLS = list(range(10))

def get_subset(X, y, cls, n_per_class=None):
    mask = sum(y==c for c in cls).bool()
    Xs, ys = X[mask], y[mask]
    if n_per_class:
        idx = torch.cat([torch.where(ys==c)[0][:n_per_class] for c in cls])
        Xs, ys = Xs[idx], ys[idx]
    return Xs, ys

print(f"Data loaded ({time.time()-t0:.1f}s)", flush=True)

# ═══════════════════════════════════════════
# ResNet-18 adapted for 32×32 CIFAR
# ═══════════════════════════════════════════
def make_resnet18(n_classes=5):
    """ResNet-18 pretrained, adapted for CIFAR-10 (32×32)."""
    m = models.resnet18(weights='IMAGENET1K_V1')
    # Adapt for 32×32: replace 7×7 stride 2 conv with 3×3 stride 1
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, n_classes)
    # Init new layers
    nn.init.kaiming_normal_(m.conv1.weight, mode='fan_out', nonlinearity='relu')
    nn.init.kaiming_normal_(m.fc.weight)
    nn.init.zeros_(m.fc.bias)
    return m

def get_features(model, X, batch_size=256):
    """Extract 512-dim features from penultimate layer."""
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size]
            h = model.conv1(xb)
            h = model.bn1(h)
            h = model.relu(h)
            h = model.maxpool(h)
            h = model.layer1(h)
            h = model.layer2(h)
            h = model.layer3(h)
            h = model.layer4(h)
            h = model.avgpool(h)
            h = torch.flatten(h, 1)  # [B, 512]
            feats.append(h)
    return torch.cat(feats, 0)

def train_parent(n_classes, cls_list, X_tr, y_tr, X_te, y_te, epochs=15, lr=0.001, seed=42):
    """Train a parent ResNet-18 on cls_list."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    
    model = make_resnet18(n_classes)
    
    # Remap labels to 0..n_classes-1
    cls_map = {c: i for i, c in enumerate(cls_list)}
    
    Xs, ys = get_subset(X_tr, y_tr, cls_list, n_per_class=5000)  # 25k total
    ys_mapped = torch.tensor([cls_map[y.item()] for y in ys])
    
    Xt, yt = get_subset(X_te, y_te, cls_list)
    yt_mapped = torch.tensor([cls_map[y.item()] for y in yt])
    
    # Fine-tune: freeze early layers, train later layers + fc
    for name, param in model.named_parameters():
        if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
    
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xs))
        total_loss = 0; n = 0
        for i in range(0, len(Xs), 128):
            idx = perm[i:i+128]
            xb, yb = Xs[idx], ys_mapped[idx]
            # Simple augmentation: random flip
            if torch.rand(1) > 0.5:
                xb = xb.flip(-1)
            out = model(xb)
            loss = nn.CrossEntropyLoss()(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1
        scheduler.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xt), 256):
            preds.append(model(Xt[i:i+256]).argmax(1))
        preds = torch.cat(preds)
        acc = (preds == yt_mapped).float().mean().item()
        per_class = {}
        for orig_c in cls_list:
            mapped_c = cls_map[orig_c]
            mask = yt_mapped == mapped_c
            if mask.sum() > 0:
                per_class[orig_c] = (preds[mask] == mapped_c).float().mean().item()
    
    return model, acc, per_class, cls_map

# ═══════════════════════════════════════════
# Phase 1: Train Parents
# ═══════════════════════════════════════════
def run_full_pipeline(seed):
    """Run full pipeline for one seed."""
    print(f"\n{'='*60}")
    print(f"  SEED={seed}")
    print(f"{'='*60}", flush=True)
    
    st = time.time()
    
    # Train Parent A (cls 0-4)
    modelA, accA, pcA, mapA = train_parent(5, clA, X_tr_all, y_tr_all, X_te_all, y_te_all, 
                                            epochs=15, lr=0.001, seed=seed)
    print(f"  Parent A (cls 0-4): acc={accA:.3f} per_class={[round(pcA[c],2) for c in clA]}", flush=True)
    
    # Train Parent B (cls 5-9)
    modelB, accB, pcB, mapB = train_parent(5, clB, X_tr_all, y_tr_all, X_te_all, y_te_all,
                                            epochs=15, lr=0.001, seed=seed+10000)
    print(f"  Parent B (cls 5-9): acc={accB:.3f} per_class={[round(pcB[c],2) for c in clB]}", flush=True)
    
    results = {'seed': seed, 'parentA_acc': round(accA, 4), 'parentB_acc': round(accB, 4),
               'parentA_pc': {c: round(v, 4) for c, v in pcA.items()},
               'parentB_pc': {c: round(v, 4) for c, v in pcB.items()}}
    
    # ═══════════════════════════════════════════
    # Phase 2: Merge Methods
    # ═══════════════════════════════════════════
    
    # Extract features for all test data
    featA = get_features(modelA, X_te_all)  # [N, 512]
    featB = get_features(modelB, X_te_all)  # [N, 512]
    
    # Also extract val features for training merge head
    X_val, y_val = X_tr_all[45000:], y_tr_all[45000:]  # 5k val
    featA_val = get_features(modelA, X_val)
    featB_val = get_features(modelB, X_val)
    
    # For calibration data (used in CMA-ES fitness)
    X_cal, y_cal = X_tr_all[40000:45000], y_tr_all[40000:45000]  # 5k cal
    featA_cal = get_features(modelA, X_cal)
    featB_cal = get_features(modelB, X_cal)
    
    def eval_10class(preds_10, y):
        """Evaluate 10-class predictions."""
        acc = (preds_10 == y).float().mean().item()
        pc = {}
        for c in ALL_CLS:
            mask = y == c
            if mask.sum() > 0:
                pc[c] = (preds_10[mask] == c).float().mean().item()
        ok = sum(1 for c in ALL_CLS if pc.get(c,0) > 0.3)
        mn = min(pc.get(c,0) for c in ALL_CLS)
        aM = np.mean([pc.get(c,0) for c in clA])
        bM = np.mean([pc.get(c,0) for c in clB])
        bal = min(aM,bM)/(max(aM,bM)+1e-10)
        return {'acc': round(acc,4), 'ok': ok, 'min': round(mn,4), 'bal': round(bal,4),
                'pc': {c: round(pc.get(c,0),3) for c in ALL_CLS}}
    
    # --- Method 1: Dual-backbone concat + Linear Probe ---
    print("  Merge: DualBackbone+Probe...", flush=True)
    feat_cat_val = torch.cat([featA_val, featB_val], dim=1)  # [N, 1024]
    feat_cat_te = torch.cat([featA, featB], dim=1)
    
    torch.manual_seed(seed)
    probe = nn.Linear(1024, 10)
    opt_p = torch.optim.Adam(probe.parameters(), lr=0.01)
    probe.train()
    for ep in range(100):
        perm = torch.randperm(len(feat_cat_val))[:512]
        logits = probe(feat_cat_val[perm])
        loss = nn.CrossEntropyLoss()(logits, y_val[perm])
        opt_p.zero_grad(); loss.backward(); opt_p.step()
    probe.eval()
    with torch.no_grad():
        preds_probe = probe(feat_cat_te).argmax(1)
    r_probe = eval_10class(preds_probe, y_te_all)
    r_probe['name'] = 'DualBackbone+Probe'
    print(f"    Probe: acc={r_probe['acc']:.3f} ok={r_probe['ok']}/10 bal={r_probe['bal']:.3f}", flush=True)
    
    # --- Method 2: Max-confidence routing ---
    print("  Merge: MaxConf routing...", flush=True)
    modelA.eval(); modelB.eval()
    with torch.no_grad():
        # Get 5-class logits from each parent
        logitsA_all = []; logitsB_all = []
        for i in range(0, len(X_te_all), 256):
            logitsA_all.append(modelA(X_te_all[i:i+256]))
            logitsB_all.append(modelB(X_te_all[i:i+256]))
        logitsA_5 = torch.cat(logitsA_all)  # [N, 5] — classes 0-4
        logitsB_5 = torch.cat(logitsB_all)  # [N, 5] — classes 5-9
        
        # Combine: create 10-class logits by concat [A_logits, B_logits]
        logits_10 = torch.cat([logitsA_5, logitsB_5], dim=1)  # [N, 10]
        preds_concat = logits_10.argmax(1)
    r_concat = eval_10class(preds_concat, y_te_all)
    r_concat['name'] = 'LogitConcat'
    print(f"    LogitConcat: acc={r_concat['acc']:.3f} ok={r_concat['ok']}/10 bal={r_concat['bal']:.3f}", flush=True)
    
    # --- Method 3: Calibrated logit concat (scale per parent) ---
    print("  Merge: CalibratedConcat...", flush=True)
    # Use CMA-ES to find optimal scaling
    logitsA_cal = []; logitsB_cal = []
    with torch.no_grad():
        for i in range(0, len(X_cal), 256):
            logitsA_cal.append(modelA(X_cal[i:i+256]))
            logitsB_cal.append(modelB(X_cal[i:i+256]))
    logA_cal = torch.cat(logitsA_cal)
    logB_cal = torch.cat(logitsB_cal)
    
    def fitness_scale(x):
        sA, sB, bA, bB = x
        logits = torch.cat([logA_cal * sA + bA, logB_cal * sB + bB], dim=1)
        preds = logits.argmax(1)
        acc = (preds == y_cal).float().mean().item()
        pc = {c: (preds[y_cal==c]==c).float().mean().item() for c in ALL_CLS if (y_cal==c).sum()>0}
        mn = min(pc.values()) if pc else 0
        return -(0.5*acc + 0.5*mn)
    
    es = cma.CMAEvolutionStrategy([1.0, 1.0, 0.0, 0.0], 0.5,
                                   {'maxiter': 15, 'popsize': 8, 'seed': seed, 'verbose': -1})
    best_s = float('inf'); best_x = None
    while not es.stop():
        sols = es.ask()
        scores = [fitness_scale(x) for x in sols]
        es.tell(sols, scores)
        if min(scores) < best_s:
            best_s = min(scores)
            best_x = sols[np.argmin(scores)]
    
    with torch.no_grad():
        sA, sB, bA, bB = best_x
        logits_cal = torch.cat([logitsA_5 * sA + bA, logitsB_5 * sB + bB], dim=1)
        preds_cal = logits_cal.argmax(1)
    r_cal = eval_10class(preds_cal, y_te_all)
    r_cal['name'] = 'CalibratedConcat'
    print(f"    Calibrated: acc={r_cal['acc']:.3f} ok={r_cal['ok']}/10 bal={r_cal['bal']:.3f}", flush=True)
    
    # --- Method 4: Weight Average of backbone + merged fc ---
    print("  Merge: WeightAvg...", flush=True)
    # Can't just average because fc heads are 5-class for different classes
    # Instead: average backbone, use dual fc heads mapped to 10 classes
    avg_model = make_resnet18(10)  # 10-class output
    sdA = modelA.state_dict()
    sdB = modelB.state_dict()
    sd_avg = {}
    for k in sdA:
        if 'fc' not in k:
            sd_avg[k] = 0.5 * sdA[k] + 0.5 * sdB[k]
    
    # For fc: map A's 5 outputs to classes 0-4, B's 5 outputs to classes 5-9
    fc_w = torch.zeros(10, 512)
    fc_b = torch.zeros(10)
    wA = sdA['fc.weight']  # [5, 512]
    bA = sdA['fc.bias']    # [5]
    wB = sdB['fc.weight']  # [5, 512]
    bB = sdB['fc.bias']    # [5]
    for i, c in enumerate(clA):
        fc_w[c] = wA[i]
        fc_b[c] = bA[i]
    for i, c in enumerate(clB):
        fc_w[c] = wB[i]
        fc_b[c] = bB[i]
    sd_avg['fc.weight'] = fc_w
    sd_avg['fc.bias'] = fc_b
    avg_model.load_state_dict(sd_avg)
    avg_model.eval()
    
    with torch.no_grad():
        preds_avg = []
        for i in range(0, len(X_te_all), 256):
            preds_avg.append(avg_model(X_te_all[i:i+256]).argmax(1))
        preds_avg = torch.cat(preds_avg)
    r_avg = eval_10class(preds_avg, y_te_all)
    r_avg['name'] = 'WeightAvg+FCMap'
    print(f"    WeightAvg: acc={r_avg['acc']:.3f} ok={r_avg['ok']}/10 bal={r_avg['bal']:.3f}", flush=True)
    
    # --- Method 5: ENT per-layer α (CMA-ES) ---
    print("  Merge: ENT (CMA-ES per-layer α)...", flush=True)
    
    # Identify mergeable layer groups
    layer_groups = []
    for name in sdA:
        if 'fc' not in name and name in sdB:
            layer_groups.append(name)
    n_groups = len(layer_groups)
    
    def build_ent_merge(x):
        """Build merged model from per-layer alphas + fc routing."""
        alphas = 1 / (1 + np.exp(-np.array(x[:n_groups])))  # sigmoid
        route = x[n_groups:]  # 10 routing values
        
        merged = make_resnet18(10)
        sd = {}
        for i, k in enumerate(layer_groups):
            sd[k] = alphas[i] * sdA[k] + (1-alphas[i]) * sdB[k]
        
        # FC: weighted combination based on routing
        fc_w = torch.zeros(10, 512)
        fc_b = torch.zeros(10)
        for ci, c in enumerate(clA):
            a = 1 / (1 + np.exp(-route[c]))
            fc_w[c] = a * wA[ci]
            fc_b[c] = a * bA[ci]
        for ci, c in enumerate(clB):
            a = 1 / (1 + np.exp(-route[c]))
            fc_w[c] = (1-a) * wB[ci]
            fc_b[c] = (1-a) * bB[ci]
        sd['fc.weight'] = fc_w
        sd['fc.bias'] = fc_b
        merged.load_state_dict(sd)
        return merged
    
    def fitness_ent(x):
        m = build_ent_merge(x)
        m.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(X_cal), 256):
                preds.append(m(X_cal[i:i+256]).argmax(1))
            preds = torch.cat(preds)
        acc = (preds == y_cal).float().mean().item()
        pc = {c: (preds[y_cal==c]==c).float().mean().item() for c in ALL_CLS if (y_cal==c).sum()>0}
        mn = min(pc.values()) if pc else 0
        ok = sum(1 for v in pc.values() if v > 0.3)
        return -(0.3*acc + 0.4*mn + 0.2*ok/10 + 0.1*np.mean(list(pc.values())))
    
    # Init: backbone from A (since A+B share pretrained), route A→0-4, B→5-9
    x0 = np.zeros(n_groups + 10)
    x0[:n_groups] = 0  # sigmoid(0)=0.5 → equal blend
    x0[n_groups:n_groups+5] = 3.0   # route cls 0-4 towards A
    x0[n_groups+5:] = -3.0           # route cls 5-9 towards B
    
    es = cma.CMAEvolutionStrategy(x0, 0.5, 
                                   {'maxiter': 12, 'popsize': 8, 'seed': seed, 'verbose': -1})
    best_s = float('inf'); best_x = None
    while not es.stop():
        sols = es.ask()
        scores = [fitness_ent(x) for x in sols]
        es.tell(sols, scores)
        if min(scores) < best_s:
            best_s = min(scores)
            best_x = sols[np.argmin(scores)]
    
    merged_ent = build_ent_merge(best_x)
    merged_ent.eval()
    with torch.no_grad():
        preds_ent = []
        for i in range(0, len(X_te_all), 256):
            preds_ent.append(merged_ent(X_te_all[i:i+256]).argmax(1))
        preds_ent = torch.cat(preds_ent)
    r_ent = eval_10class(preds_ent, y_te_all)
    r_ent['name'] = 'ENT(CMA-ES)'
    print(f"    ENT: acc={r_ent['acc']:.3f} ok={r_ent['ok']}/10 bal={r_ent['bal']:.3f}", flush=True)
    
    results['methods'] = {
        'DualProbe': {k:v for k,v in r_probe.items() if k!='pc'},
        'LogitConcat': {k:v for k,v in r_concat.items() if k!='pc'},
        'Calibrated': {k:v for k,v in r_cal.items() if k!='pc'},
        'WeightAvg': {k:v for k,v in r_avg.items() if k!='pc'},
        'ENT': {k:v for k,v in r_ent.items() if k!='pc'},
    }
    results['per_class'] = {
        'DualProbe': r_probe['pc'],
        'LogitConcat': r_concat['pc'],
        'Calibrated': r_cal['pc'],
        'WeightAvg': r_avg['pc'],
        'ENT': r_ent['pc'],
    }
    results['time_s'] = round(time.time() - st, 1)
    
    # Summary line
    print(f"\n  Summary (seed={seed}):")
    for name in ['WeightAvg', 'LogitConcat', 'Calibrated', 'DualProbe', 'ENT']:
        r = results['methods'][name]
        print(f"    {name:20s}: acc={r['acc']:.3f} ok={r['ok']}/10 bal={r['bal']:.3f} min={r['min']:.3f}")
    
    return results

# ═══════════════════════════════════════════
# Run multiple seeds
# ═══════════════════════════════════════════
SEEDS = [42, 123, 456, 789, 1000]
all_results = {}

for seed in SEEDS:
    r = run_full_pipeline(seed)
    all_results[seed] = r
    # Save incrementally
    with open('results/all_seeds.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved seed={seed} ({r['time_s']:.0f}s)", flush=True)
    sys.stdout.flush()

# ═══════════════════════════════════════════
# Statistical Analysis
# ═══════════════════════════════════════════
print(f"\n{'='*70}", flush=True)
print("STATISTICAL ANALYSIS (5 seeds)", flush=True)

methods = ['WeightAvg', 'LogitConcat', 'Calibrated', 'DualProbe', 'ENT']
metrics = {m: {'acc':[], 'ok':[], 'bal':[], 'min':[]} for m in methods}
for seed, r in sorted(all_results.items()):
    for m in methods:
        if m in r['methods']:
            for k in ['acc','ok','bal','min']:
                metrics[m][k].append(r['methods'][m][k])

print(f"\n  {'Method':20s} {'Acc':>14} {'OK':>10} {'Balance':>14} {'Min':>10}")
for m in methods:
    d = metrics[m]
    if d['acc']:
        print(f"  {m:20s} {np.mean(d['acc']):.3f}±{np.std(d['acc']):.3f} "
              f"{np.mean(d['ok']):>5.1f}±{np.std(d['ok']):.1f} "
              f"{np.mean(d['bal']):.3f}±{np.std(d['bal']):.3f} "
              f"{np.mean(d['min']):.3f}")

# P-values
best_method = max(methods, key=lambda m: np.mean(metrics[m]['ok']) if metrics[m]['ok'] else 0)
print(f"\n  Best method by OK: {best_method} ({np.mean(metrics[best_method]['ok']):.1f})")

for baseline in methods:
    if baseline == best_method: continue
    bm_ok = np.array(metrics[best_method]['ok'], dtype=float)
    bl_ok = np.array(metrics[baseline]['ok'], dtype=float)
    n = min(len(bm_ok), len(bl_ok))
    if n >= 2:
        _, p = stats.ttest_rel(bm_ok[:n], bl_ok[:n])
        print(f"  {best_method} vs {baseline}: ok p={p:.4f}")

# Parent quality
parA = [all_results[s]['parentA_acc'] for s in SEEDS]
parB = [all_results[s]['parentB_acc'] for s in SEEDS]
print(f"\n  Parent A acc: {np.mean(parA):.3f}±{np.std(parA):.3f}")
print(f"  Parent B acc: {np.mean(parB):.3f}±{np.std(parB):.3f}")

elapsed = time.time() - t0
print(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f}min)")

# Save final
with open('results/cifar_fix_results.json', 'w') as f:
    json.dump({
        'all_results': {str(s): r for s, r in all_results.items()},
        'summary': {m: {k: round(float(np.mean(v)),4) for k,v in metrics[m].items()} for m in methods if metrics[m]['acc']},
        'parent_acc': {'A': round(float(np.mean(parA)),4), 'B': round(float(np.mean(parB)),4)},
        'time_s': round(elapsed, 1),
    }, f, indent=2)

# §K5
print(f"\nmetric_parentA_acc: {np.mean(parA):.4f}")
print(f"metric_parentB_acc: {np.mean(parB):.4f}")
for m in methods:
    if metrics[m]['ok']:
        print(f"metric_{m}_ok: {np.mean(metrics[m]['ok']):.1f}")
        print(f"metric_{m}_acc: {np.mean(metrics[m]['acc']):.4f}")
print("Done!", flush=True)
