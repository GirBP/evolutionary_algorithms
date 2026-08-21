#!/usr/bin/env python3
"""Task 5: Extended Baselines — Ensemble, Majority Voting, Linear Probe.
MNIST disjoint, seed=42. Compare with ENT results from results_e34.json.
"""
import numpy as np, torch, torch.nn as nn, random, json, time
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets, transforms

t0 = time.time()

# ═══════════════════════════════════════════
# Data (same as e34_benchmark.py)
# ═══════════════════════════════════════════
tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
tr = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=tf)
te = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=tf)
X_tr = torch.stack([tr[i][0] for i in range(20000)]); y_tr = torch.tensor([tr[i][1] for i in range(20000)])
X_te = torch.stack([te[i][0] for i in range(2000)]); y_te = torch.tensor([te[i][1] for i in range(2000)])
idx = torch.randperm(20000, generator=torch.Generator().manual_seed(0))
Xv, yv = X_tr[idx[15000:18000]], y_tr[idx[15000:18000]]
Xc = X_tr[idx[:2000]]

class MLP(nn.Module):
    def __init__(s, a):
        super().__init__(); l = []
        for i in range(len(a)-1):
            l.append(nn.Linear(a[i], a[i+1]))
            if i < len(a)-2: l.append(nn.ReLU())
        s.net = nn.Sequential(*l); s.arch = a
    def forward(s, x): return s.net(x)

def ev(m, X, y):
    m.eval()
    with torch.no_grad(): return (m(X).argmax(1) == y).float().mean().item()

def pc(m, X, y):
    m.eval()
    with torch.no_grad(): p = m(X).argmax(1)
    return {c: (p[y==c]==c).float().mean().item() if (y==c).sum() > 0 else 0 for c in range(10)}

def evaluate_preds(preds, y, name):
    acc = (preds == y).float().mean().item()
    clA, clB = list(range(5)), list(range(5, 10))
    pcM = {}
    for c in range(10):
        mask = y == c
        if mask.sum() > 0:
            pcM[c] = (preds[mask] == c).float().mean().item()
        else:
            pcM[c] = 0.0
    aM = np.mean([pcM[c] for c in clA]); bM = np.mean([pcM[c] for c in clB])
    bal = min(aM, bM) / (max(aM, bM) + 1e-10)
    mn = min(pcM[c] for c in range(10))
    ok = sum(1 for c in range(10) if pcM[c] > 0.3)
    return {'name': name, 'acc': round(acc, 4), 'bal': round(bal, 4),
            'min': round(mn, 4), 'ok': ok, 'A': round(aM, 4), 'B': round(bM, 4),
            'per_class': {c: round(pcM[c], 3) for c in range(10)}}

def train_model(arch, X, y, cls, epochs=15):
    m = MLP(arch)
    mask = sum(y == c for c in cls).bool()
    Xs, ys = X[mask][:5000], y[mask][:5000]
    opt = torch.optim.Adam(m.parameters(), lr=0.003); m.train()
    for _ in range(epochs):
        l = nn.CrossEntropyLoss()(m(Xs), ys); opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m

clA, clB = list(range(5)), list(range(5, 10))
arch = [784, 128, 64, 10]

print("Training parents (seed=42)...")
torch.manual_seed(SEED)
modelA = train_model(arch, X_tr, y_tr, clA)
modelB = train_model(arch, X_tr, y_tr, clB)
print(f"  Model A (cls 0-4): {ev(modelA, X_te, y_te):.3f}")
print(f"  Model B (cls 5-9): {ev(modelB, X_te, y_te):.3f}")

results = []

# ═══════════════════════════════════════════
# METHOD 1: Logit Ensemble (average logits, take argmax)
# ═══════════════════════════════════════════
print("\n--- Method 1: Logit Ensemble ---")
modelA.eval(); modelB.eval()
with torch.no_grad():
    logitsA = modelA(X_te)
    logitsB = modelB(X_te)
    # Try different blending
    for alpha in [0.5, 0.3, 0.7]:
        combined = alpha * logitsA + (1-alpha) * logitsB
        preds = combined.argmax(1)
        r = evaluate_preds(preds, y_te, f"Ensemble(α={alpha})")
        print(f"  α={alpha}: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")
        results.append(r)

# ═══════════════════════════════════════════
# METHOD 2: Majority Voting (hard vote from both parents)
# ═══════════════════════════════════════════
print("\n--- Method 2: Majority Voting ---")
with torch.no_grad():
    predsA = modelA(X_te).argmax(1)
    predsB = modelB(X_te).argmax(1)
    confA = modelA(X_te).softmax(1).max(1).values
    confB = modelB(X_te).softmax(1).max(1).values
    
    # Simple: when they agree, use that; when not, pick higher confidence
    preds_vote = torch.where(confA >= confB, predsA, predsB)
    r = evaluate_preds(preds_vote, y_te, "MajorityVote")
    print(f"  Confidence-vote: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")
    results.append(r)

    # Alternative: max-logit selection per sample
    maxA = modelA(X_te).max(1).values
    maxB = modelB(X_te).max(1).values
    preds_max = torch.where(maxA >= maxB, predsA, predsB)
    r2 = evaluate_preds(preds_max, y_te, "MaxLogitSelect")
    print(f"  Max-logit: acc={r2['acc']:.3f} bal={r2['bal']:.3f} ok={r2['ok']}/10 min={r2['min']:.3f}")
    results.append(r2)

# ═══════════════════════════════════════════
# METHOD 3: Feature Concat + Linear Probe
# ═══════════════════════════════════════════
print("\n--- Method 3: Feature Concat + Linear Probe ---")

def get_features(model, X):
    """Get penultimate layer features."""
    model.eval()
    with torch.no_grad():
        h = X
        for m in list(model.net)[:-1]:  # all but last linear
            h = m(h)
    return h

featA_val = get_features(modelA, Xv)
featB_val = get_features(modelB, Xv)
feat_val = torch.cat([featA_val, featB_val], dim=1)

featA_te = get_features(modelA, X_te)
featB_te = get_features(modelB, X_te)
feat_te = torch.cat([featA_te, featB_te], dim=1)

# Train a simple linear probe on validation data
probe = nn.Linear(feat_val.shape[1], 10)
opt = torch.optim.Adam(probe.parameters(), lr=0.01)
probe.train()
for ep in range(50):
    logits = probe(feat_val)
    loss = nn.CrossEntropyLoss()(logits, yv)
    opt.zero_grad(); loss.backward(); opt.step()
probe.eval()
with torch.no_grad():
    preds_probe = probe(feat_te).argmax(1)
r = evaluate_preds(preds_probe, y_te, "LinearProbe")
print(f"  LinearProbe: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")
results.append(r)

# Also try with more capacity: 2-layer probe
probe2 = nn.Sequential(nn.Linear(feat_val.shape[1], 64), nn.ReLU(), nn.Linear(64, 10))
opt = torch.optim.Adam(probe2.parameters(), lr=0.005)
probe2.train()
for ep in range(100):
    logits = probe2(feat_val)
    loss = nn.CrossEntropyLoss()(logits, yv)
    opt.zero_grad(); loss.backward(); opt.step()
probe2.eval()
with torch.no_grad():
    preds_probe2 = probe2(feat_te).argmax(1)
r = evaluate_preds(preds_probe2, y_te, "MLPProbe")
print(f"  MLPProbe: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")
results.append(r)

# ═══════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════
# Load ENT results from seed=42
with open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/results_e34.json') as f:
    e34 = json.load(f)
ent_r = [x for x in e34 if x['name'] == 'ENT'][0]

print(f"\n{'='*70}")
print("Task 5: Extended Baselines vs ENT (MNIST disjoint, seed=42)")
print(f"{'='*70}")
print(f"  {'Method':<22} {'Acc':>6} {'Balance':>8} {'Min_cls':>8} {'OK':>5}")
print(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*8} {'-'*5}")

# Best from each category
best_ens = max([r for r in results if 'Ensemble' in r['name']], key=lambda x: x['bal'])
best_vote = max([r for r in results if 'Vote' in r['name'] or 'MaxLogit' in r['name']], key=lambda x: x['bal'])
best_probe = max([r for r in results if 'Probe' in r['name']], key=lambda x: x['bal'])
all_summary = [best_ens, best_vote, best_probe]

for r in all_summary:
    print(f"  {r['name']:<22} {r['acc']:>6.3f} {r['bal']:>8.3f} {r['min']:>8.3f} {r['ok']:>3}/10")

print(f"  {'ENT (from e34)':<22} {ent_r['acc']:>6.3f} {ent_r['bal']:>8.3f} {ent_r['min']:>8.3f} {ent_r['ok']:>3}/10 ✅")

print(f"\n  Per-class breakdown:")
print(f"  {'Cls':>3} {'Parent':>7} {'BestEns':>8} {'Vote':>8} {'Probe':>8} {'ENT':>8}")
for c in range(10):
    bp = max(pc(modelA, X_te, y_te)[c], pc(modelB, X_te, y_te)[c])
    print(f"  {c:>3} {bp:>7.3f} {best_ens['per_class'][c]:>8.3f} {best_vote['per_class'][c]:>8.3f} {best_probe['per_class'][c]:>8.3f} {ent_r['per_class'][str(c)]:>8.3f}")

elapsed = time.time() - t0
print(f"\n  Time: {elapsed:.1f}s")

# Save
with open('results/task5_baselines.json', 'w') as f:
    json.dump({
        'all_results': results,
        'best_ensemble': best_ens,
        'best_voting': best_vote,
        'best_probe': best_probe,
        'ent_reference': ent_r,
        'time_s': round(elapsed, 1),
    }, f, indent=2)

# Print key metrics for grep (§K5)
print(f"\nmetric_ensemble_bal: {best_ens['bal']}")
print(f"metric_ensemble_ok: {best_ens['ok']}")
print(f"metric_vote_bal: {best_vote['bal']}")
print(f"metric_vote_ok: {best_vote['ok']}")
print(f"metric_probe_bal: {best_probe['bal']}")
print(f"metric_probe_ok: {best_probe['ok']}")
print(f"metric_ent_bal: {ent_r['bal']}")
print(f"metric_ent_ok: {ent_r['ok']}")

print("Done!")
