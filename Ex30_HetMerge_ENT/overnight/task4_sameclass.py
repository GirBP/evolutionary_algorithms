#!/usr/bin/env python3
"""Task 4: Same-class / Overlapping Merge — 3 scenarios × 3 seeds.
Tests ENT when models share classes (not complementary).
"""
import numpy as np, torch, torch.nn as nn, random, json, time, copy
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets, transforms
from scipy import stats

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

def evaluate(m, name, all_cls):
    pcM = pc(m, X_te, y_te); acc = ev(m, X_te, y_te)
    # For same-class: retention = mean accuracy on classes in union(A_cls, B_cls)
    union_cls = list(set(all_cls))
    valid_accs = [pcM[c] for c in union_cls if pcM[c] is not None]
    mn = min(valid_accs) if valid_accs else 0
    ok = sum(1 for c in range(10) if pcM[c] > 0.3)
    mean_acc = np.mean(valid_accs)
    return {'name': name, 'acc': round(acc, 4), 'mean_target': round(mean_acc, 4),
            'min': round(mn, 4), 'ok': ok, 'pc': pcM}

arch = [784, 128, 64, 10]

SCENARIOS = [
    ('same-data', list(range(10)), list(range(10))),  # both trained on all 10
    ('overlapping', list(range(7)), list(range(4, 10))),  # A: 0-6, B: 4-9 (overlap: 4-6)
    ('imbalanced', list(range(3)), list(range(3, 10))),  # A: 0-2, B: 3-9
]

all_results = {}

for scenario_name, clsA, clsB in SCENARIOS:
    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario_name}")
    print(f"  A classes: {clsA}, B classes: {clsB}")
    all_cls = list(set(clsA + clsB))
    
    scenario_results = {seed: {} for seed in SEEDS}
    
    for seed in SEEDS:
        print(f"\n  --- seed={seed} ---")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        
        modelA = train_model(arch, X_tr, y_tr, clsA, seed=seed)
        modelB = train_model(arch, X_tr, y_tr, clsB, seed=seed+10000)
        sdA = modelA.state_dict(); sdB = modelB.state_dict()
        
        base = MLP(arch)
        torch.manual_seed(0); nn.init.xavier_uniform_(list(base.parameters())[0])
        base_sd = copy.deepcopy(base.state_dict())
        
        # Average
        m = MLP(arch)
        sd = {k: 0.5 * sdA[k] + 0.5 * sdB[k] for k in sdA}
        m.load_state_dict(sd); m.eval()
        r = evaluate(m, 'Average', all_cls)
        scenario_results[seed]['Average'] = r
        
        # TIES (best density)
        best_ties = None; best_ties_v = -1
        for density in [0.1, 0.3]:
            m = MLP(arch); sd = {}
            for k in sdA:
                dA = sdA[k] - base_sd[k]; dB = sdB[k] - base_sd[k]
                for delta in [dA, dB]:
                    flat = delta.flatten()
                    n_keep = max(1, int(len(flat) * density))
                    thr = flat.abs().topk(n_keep).values[-1]
                    delta[delta.abs() < thr] = 0
                sign_mask = dA.sign() + dB.sign()
                merged_delta = torch.where(sign_mask > 0,
                    torch.maximum(dA, torch.zeros_like(dA)) + torch.maximum(dB, torch.zeros_like(dB)),
                    torch.minimum(dA, torch.zeros_like(dA)) + torch.minimum(dB, torch.zeros_like(dB)))
                sd[k] = base_sd[k] + merged_delta * 0.5
            m.load_state_dict(sd); m.eval()
            r = evaluate(m, f'TIES(d={density})', all_cls)
            if r['mean_target'] > best_ties_v: best_ties_v = r['mean_target']; best_ties = r
        scenario_results[seed]['TIES'] = best_ties
        
        # ENT
        def virt2L(model, Xc_):
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
        
        WA, sA = virt2L(modelA, Xc); WB, sB = virt2L(modelB, Xc)
        with torch.no_grad(): sl = modelA(Xc).numpy().std(); sr = modelB(Xc).numpy().std()
        t_ = (sl+sr)/2; rA = t_/(sl+1e-10); rB = t_/(sr+1e-10)
        
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        # For same-class: initial routing could be 0 (equal)
        if scenario_name == 'same-data':
            init_route = np.zeros(10)
        elif scenario_name == 'overlapping':
            init_route = np.array([2.]*4 + [0.]*3 + [-2.]*3)  # A-only, shared, B-only
        else:  # imbalanced
            init_route = np.array([2.]*3 + [-2.]*7)
        
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
                  'route': init_route.copy()}
        
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
        if m_ is not None:
            m_.eval()
            r = evaluate(m_, 'ENT', all_cls)
        else:
            r = {'name': 'ENT', 'acc': 0, 'mean_target': 0, 'min': 0, 'ok': 0, 'pc': {c: 0 for c in range(10)}}
        scenario_results[seed]['ENT'] = r
        
        # Also test simple parent-only (best parent accuracy)
        r_parentA = evaluate(modelA, 'ParentA', all_cls)
        r_parentB = evaluate(modelB, 'ParentB', all_cls)
        scenario_results[seed]['BestParent'] = r_parentA if r_parentA['acc'] > r_parentB['acc'] else r_parentB
        
        print(f"    Avg: acc={scenario_results[seed]['Average']['acc']:.3f} ok={scenario_results[seed]['Average']['ok']}/10")
        print(f"    TIES: acc={scenario_results[seed]['TIES']['acc']:.3f} ok={scenario_results[seed]['TIES']['ok']}/10")
        print(f"    ENT: acc={r['acc']:.3f} ok={r['ok']}/10 mean_target={r['mean_target']:.3f}")
    
    all_results[scenario_name] = scenario_results

# ═══════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════
print(f"\n{'='*70}")
print("Task 4: Same-class Merge Results")
print(f"{'='*70}")

for scenario, data in all_results.items():
    print(f"\n  Scenario: {scenario}")
    methods = ['Average', 'TIES', 'ENT', 'BestParent']
    for method in methods:
        accs = [data[s][method]['acc'] for s in SEEDS if method in data[s]]
        oks = [data[s][method]['ok'] for s in SEEDS if method in data[s]]
        mts = [data[s][method].get('mean_target', 0) for s in SEEDS if method in data[s]]
        print(f"    {method:<12}: acc={np.mean(accs):.3f}±{np.std(accs):.3f} ok={np.mean(oks):.1f} target_mean={np.mean(mts):.3f}")
    
    # ENT retention vs BestParent
    ent_accs = [data[s]['ENT']['acc'] for s in SEEDS]
    bp_accs = [data[s]['BestParent']['acc'] for s in SEEDS]
    retention = [e/b if b > 0 else 0 for e, b in zip(ent_accs, bp_accs)]
    print(f"    ENT retention vs best parent: {np.mean(retention):.3f}±{np.std(retention):.3f}")

elapsed = time.time() - t0
print(f"\n  Total: {elapsed:.1f}s ({elapsed/60:.1f} min)")

# Save
save_data = {}
for scenario, data in all_results.items():
    save_data[scenario] = {}
    for seed in SEEDS:
        save_data[scenario][seed] = {}
        for method in ['Average', 'TIES', 'ENT', 'BestParent']:
            if method in data[seed]:
                r = data[seed][method]
                save_data[scenario][seed][method] = {k: v for k, v in r.items() if k != 'pc'}

with open('results/task4_sameclass.json', 'w') as f:
    json.dump(save_data, f, indent=2, default=str)

# §K5
for scenario in all_results:
    ent_accs = [all_results[scenario][s]['ENT']['acc'] for s in SEEDS]
    ent_oks = [all_results[scenario][s]['ENT']['ok'] for s in SEEDS]
    print(f"\nmetric_{scenario}_ent_acc: {np.mean(ent_accs):.4f}")
    print(f"metric_{scenario}_ent_ok: {np.mean(ent_oks):.1f}")

print("Done!")
