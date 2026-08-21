#!/usr/bin/env python3
"""Task 3: Het-arch ENT — 6 configs × 3 seeds.
Adapted from e23_ent.py with fixes for het-width/depth crashes.
"""
import numpy as np, torch, torch.nn as nn, random, json, time
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets, transforms

t0 = time.time()
SEEDS = [42, 123, 456]

tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
tr = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=tf)
te = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=tf)
X_tr = torch.stack([tr[i][0] for i in range(20000)]); y_tr = torch.tensor([tr[i][1] for i in range(20000)])
X_te = torch.stack([te[i][0] for i in range(2000)]); y_te = torch.tensor([te[i][1] for i in range(2000)])
idx = torch.randperm(20000, generator=torch.Generator().manual_seed(0))
Xv, yv = X_tr[idx[15000:18000]], y_tr[idx[15000:18000]]
Xc = X_tr[idx[:2000]]

clA, clB = list(range(5)), list(range(5, 10))

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

def train_model(arch, X, y, cls, epochs=15, seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    m = MLP(arch)
    mask = sum(y == c for c in cls).bool()
    Xs, ys = X[mask][:5000], y[mask][:5000]
    opt = torch.optim.Adam(m.parameters(), lr=0.003); m.train()
    for _ in range(epochs):
        l = nn.CrossEntropyLoss()(m(Xs), ys); opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m

def virt2L(model, Xc_):
    """Collapse ANY depth MLP to virtual 2-layer via LSQ."""
    model.eval()
    with torch.no_grad():
        h = Xc_
        for m_ in list(model.net)[:-1]: h = m_(h)
        hid = h.numpy()
    ps = list(model.parameters())
    Wo = ps[-2].detach().numpy(); bo = ps[-1].detach().numpy()
    fd = hid.shape[1]
    x = Xc_.numpy(); N = x.shape[0]
    xb = np.hstack([x, np.ones((N,1), dtype=np.float32)])
    Wb = np.linalg.lstsq(xb, np.maximum(hid, 0), rcond=None)[0].T
    W1 = Wb[:,:-1].astype(np.float32); b1 = Wb[:,-1].astype(np.float32)
    W2 = np.eye(fd, dtype=np.float32); b2 = np.zeros(fd, dtype=np.float32)
    return [W1, b1, W2, b2, Wo.copy(), bo.copy()], [fd, fd]

def bld_ent(ch, WA_, WB_, sA_, sB_, rA_, rB_):
    sizes = [784]
    for i in range(2):
        n = int(ch['mA'][i].sum()) + int(ch['mB'][i].sum())
        if n < 2: return None
        sizes.append(n)
    sizes.append(10)
    iA = np.where(ch['mA'][0])[0]; iB = np.where(ch['mB'][0])[0]
    W0 = np.vstack([WA_[0][iA], WB_[0][iB]]); b0 = np.concatenate([WA_[1][iA], WB_[1][iB]])
    icA = np.where(ch['mA'][1])[0]; icB = np.where(ch['mB'][1])[0]
    W1 = np.zeros((len(icA)+len(icB), len(iA)+len(iB)), dtype=np.float32)
    b1 = np.zeros(len(icA)+len(icB), dtype=np.float32)
    W1[:len(icA),:len(iA)] = WA_[2][np.ix_(icA, iA)]; b1[:len(icA)] = WA_[3][icA]
    W1[len(icA):,len(iA):] = WB_[2][np.ix_(icB, iB)]; b1[len(icA):] = WB_[3][icB]
    ilA = icA; ilB = icB
    Wo_ = np.zeros((10, len(ilA)+len(ilB)), dtype=np.float32); bo_ = np.zeros(10, dtype=np.float32)
    for c in range(10):
        a = 1.0 / (1.0 + np.exp(-ch['route'][c]))
        if len(ilA) > 0: Wo_[c,:len(ilA)] = a * rA_ * WA_[4][c][ilA]
        if len(ilB) > 0: Wo_[c,len(ilA):] = (1-a) * rB_ * WB_[4][c][ilB]
        bo_[c] = a * rA_ * WA_[5][c] + (1-a) * rB_ * WB_[5][c]
    m_ = MLP(sizes)
    with torch.no_grad():
        ps = list(m_.parameters())
        ps[0].copy_(torch.tensor(W0)); ps[1].copy_(torch.tensor(b0))
        ps[2].copy_(torch.tensor(W1)); ps[3].copy_(torch.tensor(b1))
        ps[4].copy_(torch.tensor(Wo_)); ps[5].copy_(torch.tensor(bo_))
    return m_

def run_ent(modelA, modelB, seed):
    """Run ENT merge on two models."""
    WA, sA = virt2L(modelA, Xc); WB, sB = virt2L(modelB, Xc)
    with torch.no_grad(): sl = modelA(Xc).numpy().std(); sr = modelB(Xc).numpy().std()
    t_ = (sl+sr)/2; rA = t_/(sl+1e-10); rB = t_/(sr+1e-10)
    
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    pop = []
    for _ in range(20):
        ch = {'mA': [np.random.random(d) > 0.3 for d in sA],
              'mB': [np.random.random(d) > 0.3 for d in sB],
              'route': np.random.randn(10) * 1.5}
        for ms in [ch['mA'], ch['mB']]:
            for m_ in ms:
                if m_.sum() == 0: m_[0] = True
        pop.append(ch)
    pop[0] = {'mA': [np.ones(d, dtype=bool) for d in sA],
              'mB': [np.ones(d, dtype=bool) for d in sB],
              'route': np.array([2.0]*5 + [-2.0]*5)}
    
    bf = -1; bc = None
    for gen in range(30):
        fs = []
        for ch in pop:
            m_ = bld_ent(ch, WA, WB, sA, sB, rA, rB)
            if m_ is None: fs.append(-1); continue
            d = pc(m_, Xv, yv); acc_ = ev(m_, Xv, yv)
            mn_ = min(d[c] for c in range(10))
            fs.append(0.4*acc_ + 0.4*mn_ + 0.1*np.mean([d[c] for c in range(10)]) + 0.1*(1-sum(ch['mA'][i].sum()+ch['mB'][i].sum() for i in range(2))/(sum(sA)+sum(sB))))
        gi = np.argmax(fs)
        if fs[gi] > bf:
            bf = fs[gi]
            bc = {'mA': [m_.copy() for m_ in pop[gi]['mA']],
                  'mB': [m_.copy() for m_ in pop[gi]['mB']],
                  'route': pop[gi]['route'].copy()}
        new = [{'mA': [m_.copy() for m_ in bc['mA']], 'mB': [m_.copy() for m_ in bc['mB']], 'route': bc['route'].copy()}]
        while len(new) < 20:
            ti = random.sample(range(len(pop)), 3)
            p1 = pop[ti[np.argmax([fs[i] for i in ti])]]
            ch = {'mA': [m_.copy() for m_ in p1['mA']], 'mB': [m_.copy() for m_ in p1['mB']], 'route': p1['route'] + np.random.randn(10)*0.3}
            pf = max(0.02, 0.06 - gen*0.001)
            for ms in [ch['mA'], ch['mB']]:
                for m_ in ms:
                    f_ = np.random.random(len(m_)) < pf; m_[f_] = ~m_[f_]
                    if m_.sum() == 0: m_[np.random.randint(len(m_))] = True
            new.append(ch)
        pop = new
    
    m_ = bld_ent(bc, WA, WB, sA, sB, rA, rB)
    if m_ is None:
        return {'acc': 0, 'bal': 0, 'min': 0, 'ok': 0, 'pc': {c: 0 for c in range(10)}}
    m_.eval()
    pcM = pc(m_, X_te, y_te); acc = ev(m_, X_te, y_te)
    aM = np.mean([pcM[c] for c in clA]); bM = np.mean([pcM[c] for c in clB])
    bal = min(aM, bM) / (max(aM, bM) + 1e-10)
    mn = min(pcM[c] for c in range(10))
    ok = sum(1 for c in range(10) if pcM[c] > 0.3)
    return {'acc': round(acc, 4), 'bal': round(bal, 4), 'min': round(mn, 4), 'ok': ok}

# ═══════════════════════════════════════════
# 6 Configs
# ═══════════════════════════════════════════
CONFIGS = [
    ('same-arch', [784, 128, 64, 10], [784, 128, 64, 10]),
    ('het-width', [784, 64, 32, 10], [784, 256, 128, 10]),
    ('het-depth-2v3', [784, 128, 10], [784, 128, 64, 10]),
    ('het-depth-3v2', [784, 128, 64, 10], [784, 128, 10]),
    ('extreme-2v4', [784, 128, 10], [784, 256, 128, 64, 10]),
    ('cross-width', [784, 64, 10], [784, 256, 128, 10]),
]

all_results = {}
for config_name, archA, archB in CONFIGS:
    print(f"\n{'='*50}")
    print(f"  {config_name}: A={archA} B={archB}")
    config_results = {}
    
    for seed in SEEDS:
        print(f"  seed={seed}...", end=' ', flush=True)
        try:
            modelA = train_model(archA, X_tr, y_tr, clA, seed=seed)
            modelB = train_model(archB, X_tr, y_tr, clB, seed=seed+10000)
            r = run_ent(modelA, modelB, seed)
            config_results[seed] = r
            print(f"ok={r['ok']}/10 bal={r['bal']:.3f} acc={r['acc']:.3f}", flush=True)
        except Exception as e:
            print(f"CRASH: {e}", flush=True)
            config_results[seed] = {'acc': 0, 'bal': 0, 'min': 0, 'ok': 0, 'error': str(e)}
    
    all_results[config_name] = config_results

# ═══════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════
print(f"\n{'='*70}")
print("Task 3: Het-arch Results (6 configs × 3 seeds)")
print(f"{'='*70}")
print(f"  {'Config':<18} {'OK':>10} {'Balance':>14} {'Acc':>14}")
for config_name in [c[0] for c in CONFIGS]:
    data = all_results[config_name]
    oks = [data[s]['ok'] for s in SEEDS if s in data]
    bals = [data[s]['bal'] for s in SEEDS if s in data and 'error' not in data[s]]
    accs = [data[s]['acc'] for s in SEEDS if s in data and 'error' not in data[s]]
    if bals:
        print(f"  {config_name:<18} {np.mean(oks):.1f}±{np.std(oks):.1f} {np.mean(bals):.3f}±{np.std(bals):.3f} {np.mean(accs):.3f}±{np.std(accs):.3f}")
    else:
        print(f"  {config_name:<18} FAILED")

# Count success: configs with mean ok ≥ 8
n_success = 0
for config_name in [c[0] for c in CONFIGS]:
    data = all_results[config_name]
    oks = [data[s]['ok'] for s in SEEDS if s in data]
    if np.mean(oks) >= 8: n_success += 1
print(f"\n  Configs with mean ok ≥ 8: {n_success}/{len(CONFIGS)}")

elapsed = time.time() - t0
print(f"  Total: {elapsed:.1f}s ({elapsed/60:.1f} min)")

with open('results/task3_hetarch.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

for config_name in [c[0] for c in CONFIGS]:
    oks = [all_results[config_name][s]['ok'] for s in SEEDS if s in all_results[config_name]]
    print(f"\nmetric_{config_name}_ok: {np.mean(oks):.1f}")
print("Done!")
