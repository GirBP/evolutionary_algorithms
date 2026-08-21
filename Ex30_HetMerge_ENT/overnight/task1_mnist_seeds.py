#!/usr/bin/env python3
"""Task 1: MNIST Multi-seed Significance.
seed=42 already exists in results_e34.json — DO NOT re-run.
9 new seeds × 4 key methods (Average, TIES, Fisher, ENT).
Output: mean ± std, paired t-test p-values.
"""
import numpy as np, torch, torch.nn as nn, random, json, time, copy
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets, transforms
from scipy import stats

t0 = time.time()

# ═══════════════════════════════════════════
# Data
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

def train_model(arch, X, y, cls, epochs=15, seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    m = MLP(arch)
    mask = sum(y == c for c in cls).bool()
    Xs, ys = X[mask][:5000], y[mask][:5000]
    opt = torch.optim.Adam(m.parameters(), lr=0.003); m.train()
    for _ in range(epochs):
        l = nn.CrossEntropyLoss()(m(Xs), ys); opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m

clA, clB = list(range(5)), list(range(5, 10))
arch = [784, 128, 64, 10]

def run_all_methods(seed):
    """Run 4 key methods for a given seed. Returns results dict."""
    print(f"\n{'='*60}")
    print(f"  SEED={seed}")
    print(f"{'='*60}")
    
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    
    # Train parents
    modelA = train_model(arch, X_tr, y_tr, clA, seed=seed)
    modelB = train_model(arch, X_tr, y_tr, clB, seed=seed+10000)  # different seed for B
    
    sdA = modelA.state_dict(); sdB = modelB.state_dict()
    accA = ev(modelA, X_te, y_te); accB = ev(modelB, X_te, y_te)
    print(f"  Parents: A={accA:.3f}, B={accB:.3f}")
    
    # Base model for TA/TIES  
    torch.manual_seed(0)
    base = MLP(arch)
    nn.init.xavier_uniform_(list(base.parameters())[0])
    base_sd = copy.deepcopy(base.state_dict())
    
    results = {}
    
    # 1. Average (best α)
    best_bal = -1; best_r = None
    for alpha in [0.3, 0.5, 0.7]:
        m = MLP(arch)
        sd = {k: alpha * sdA[k] + (1-alpha) * sdB[k] for k in sdA}
        m.load_state_dict(sd); m.eval()
        pcM = pc(m, X_te, y_te); acc = ev(m, X_te, y_te)
        aM = np.mean([pcM[c] for c in clA]); bM = np.mean([pcM[c] for c in clB])
        bal = min(aM, bM) / (max(aM, bM) + 1e-10)
        mn = min(pcM[c] for c in range(10))
        ok = sum(1 for c in range(10) if pcM[c] > 0.3)
        if bal > best_bal:
            best_bal = bal
            best_r = {'acc': acc, 'bal': bal, 'min': mn, 'ok': ok, 'pc': pcM}
    results['Average'] = best_r
    print(f"  Average: acc={best_r['acc']:.3f} bal={best_r['bal']:.3f} ok={best_r['ok']}/10")
    
    # 2. TIES (best density)
    best_bal = -1; best_r = None
    for density in [0.1, 0.3, 0.5]:
        m = MLP(arch); sd = {}
        for k in sdA:
            dA = sdA[k] - base_sd[k]; dB = sdB[k] - base_sd[k]
            for delta in [dA, dB]:
                flat = delta.flatten()
                n_keep = max(1, int(len(flat) * density))
                threshold = flat.abs().topk(n_keep).values[-1]
                delta[delta.abs() < threshold] = 0
            sign_mask = (dA.sign() + dB.sign())
            merged_delta = torch.where(sign_mask > 0,
                torch.maximum(dA, torch.zeros_like(dA)) + torch.maximum(dB, torch.zeros_like(dB)),
                torch.minimum(dA, torch.zeros_like(dA)) + torch.minimum(dB, torch.zeros_like(dB)))
            n_nonzero = (merged_delta != 0).float().sum()
            if n_nonzero > 0:
                merged_delta = merged_delta * ((dA != 0).float() + (dB != 0).float()).clamp(max=1)
            sd[k] = base_sd[k] + merged_delta * 0.5
        m.load_state_dict(sd); m.eval()
        pcM = pc(m, X_te, y_te); acc = ev(m, X_te, y_te)
        aM = np.mean([pcM[c] for c in clA]); bM = np.mean([pcM[c] for c in clB])
        bal = min(aM, bM) / (max(aM, bM) + 1e-10)
        mn = min(pcM[c] for c in range(10))
        ok = sum(1 for c in range(10) if pcM[c] > 0.3)
        if bal > best_bal:
            best_bal = bal
            best_r = {'acc': acc, 'bal': bal, 'min': mn, 'ok': ok, 'pc': pcM}
    results['TIES'] = best_r
    print(f"  TIES: acc={best_r['acc']:.3f} bal={best_r['bal']:.3f} ok={best_r['ok']}/10")
    
    # 3. Fisher
    def compute_fisher(model, X, y, n=300):
        model.eval()
        fisher = {n_: torch.zeros_like(p) for n_, p in model.named_parameters()}
        model.train()
        for i in range(min(n, len(X))):
            model.zero_grad()
            out = model(X[i:i+1])
            log_prob = nn.LogSoftmax(dim=1)(out)
            target = out.argmax(1)
            loss = nn.NLLLoss()(log_prob, target)
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.data ** 2
        for name in fisher: fisher[name] /= n
        model.eval()
        return fisher
    
    fisherA = compute_fisher(modelA, X_tr, y_tr, n=300)
    fisherB = compute_fisher(modelB, X_tr, y_tr, n=300)
    m = MLP(arch); sd = {}
    for k in sdA:
        found = False
        fA = torch.ones_like(sdA[k]); fB = torch.ones_like(sdB[k])
        for fk in fisherA:
            if fk in k or k in fk:
                fA = fisherA[fk]; fB = fisherB[fk]; found = True; break
        denom = fA + fB + 1e-8
        sd[k] = (fA * sdA[k] + fB * sdB[k]) / denom
    m.load_state_dict(sd); m.eval()
    pcM = pc(m, X_te, y_te); acc = ev(m, X_te, y_te)
    aM = np.mean([pcM[c] for c in clA]); bM = np.mean([pcM[c] for c in clB])
    bal = min(aM, bM) / (max(aM, bM) + 1e-10)
    mn = min(pcM[c] for c in range(10))
    ok = sum(1 for c in range(10) if pcM[c] > 0.3)
    results['Fisher'] = {'acc': acc, 'bal': bal, 'min': mn, 'ok': ok, 'pc': pcM}
    print(f"  Fisher: acc={acc:.3f} bal={bal:.3f} ok={ok}/10")
    
    # 4. ENT
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
        ipA, ipB = iA, iB
        icA = np.where(ch['mA'][1])[0]; icB = np.where(ch['mB'][1])[0]
        W1 = np.zeros((len(icA)+len(icB), len(ipA)+len(ipB)), dtype=np.float32)
        b1 = np.zeros(len(icA)+len(icB), dtype=np.float32)
        W1[:len(icA),:len(ipA)] = WA_[2][np.ix_(icA, ipA)]; b1[:len(icA)] = WA_[3][icA]
        W1[len(icA):,len(ipA):] = WB_[2][np.ix_(icB, ipB)]; b1[len(icA):] = WB_[3][icB]
        ilA = np.where(ch['mA'][1])[0]; ilB = np.where(ch['mB'][1])[0]
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
    
    m_ = bld_ent(bc, WA, WB, sA, sB, rA, rB); m_.eval()
    pcM = pc(m_, X_te, y_te); acc = ev(m_, X_te, y_te)
    aM = np.mean([pcM[c] for c in clA]); bM = np.mean([pcM[c] for c in clB])
    bal = min(aM, bM) / (max(aM, bM) + 1e-10)
    mn = min(pcM[c] for c in range(10))
    ok = sum(1 for c in range(10) if pcM[c] > 0.3)
    n_ent = sum(bc['mA'][i].sum() + bc['mB'][i].sum() for i in range(2))
    n_max = sum(sA) + sum(sB)
    compression = round(1 - n_ent / n_max, 3)
    results['ENT'] = {'acc': acc, 'bal': bal, 'min': mn, 'ok': ok, 'pc': pcM, 'compression': compression}
    print(f"  ENT: acc={acc:.3f} bal={bal:.3f} ok={ok}/10 min={mn:.3f}")
    
    return results

# ═══════════════════════════════════════════
# Run all seeds
# ═══════════════════════════════════════════
NEW_SEEDS = [123, 456, 789, 1000, 2024, 3141, 7777, 9999, 31415]

# Load existing seed=42 results
with open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/results_e34.json') as f:
    e34 = json.load(f)

seed42_results = {}
for r in e34:
    if r['name'] == 'Average(α=0.5)':
        seed42_results['Average'] = {'acc': r['acc'], 'bal': r['bal'], 'min': r['min'], 'ok': r['ok'], 'pc': {int(k): v for k, v in r['per_class'].items()}}
    elif r['name'] == 'TIES(d=0.3)':
        seed42_results['TIES'] = {'acc': r['acc'], 'bal': r['bal'], 'min': r['min'], 'ok': r['ok'], 'pc': {int(k): v for k, v in r['per_class'].items()}}
    elif r['name'] == 'Fisher':
        seed42_results['Fisher'] = {'acc': r['acc'], 'bal': r['bal'], 'min': r['min'], 'ok': r['ok'], 'pc': {int(k): v for k, v in r['per_class'].items()}}
    elif r['name'] == 'ENT':
        seed42_results['ENT'] = {'acc': r['acc'], 'bal': r['bal'], 'min': r['min'], 'ok': r['ok'], 'pc': {int(k): v for k, v in r['per_class'].items()}, 'compression': r.get('compression', 0)}

all_seed_results = {42: seed42_results}
seed_times = {}

for seed in NEW_SEEDS:
    st = time.time()
    try:
        res = run_all_methods(seed)
        all_seed_results[seed] = res
        seed_times[seed] = round(time.time() - st, 1)
        print(f"  Seed {seed} done in {seed_times[seed]:.1f}s")
    except Exception as e:
        print(f"  SEED {seed} CRASHED: {e}")
        seed_times[seed] = round(time.time() - st, 1)

# ═══════════════════════════════════════════
# Statistical Analysis
# ═══════════════════════════════════════════
print(f"\n{'='*70}")
print("STATISTICAL ANALYSIS: 10 seeds")
print(f"{'='*70}")

methods = ['Average', 'TIES', 'Fisher', 'ENT']
metrics_by_method = {m: {'acc': [], 'bal': [], 'min': [], 'ok': []} for m in methods}

for seed, res in sorted(all_seed_results.items()):
    for method in methods:
        if method in res:
            for k in ['acc', 'bal', 'min', 'ok']:
                metrics_by_method[method][k].append(res[method][k])

print(f"\n  {'Method':<12} {'Acc':>14} {'Balance':>14} {'Min_cls':>14} {'OK':>10}")
print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*14} {'-'*10}")

for method in methods:
    d = metrics_by_method[method]
    n = len(d['acc'])
    if n == 0: continue
    print(f"  {method:<12} "
          f"{np.mean(d['acc']):.3f}±{np.std(d['acc']):.3f} "
          f"{np.mean(d['bal']):.3f}±{np.std(d['bal']):.3f} "
          f"{np.mean(d['min']):.3f}±{np.std(d['min']):.3f} "
          f"{np.mean(d['ok']):.1f}±{np.std(d['ok']):.1f}")

# Paired t-tests: ENT vs each baseline
print(f"\n  Paired t-tests (ENT vs baseline):")
ent_accs = np.array(metrics_by_method['ENT']['acc'])
ent_bals = np.array(metrics_by_method['ENT']['bal'])
ent_oks = np.array(metrics_by_method['ENT']['ok'])
ent_mins = np.array(metrics_by_method['ENT']['min'])

for baseline in ['Average', 'TIES', 'Fisher']:
    b_accs = np.array(metrics_by_method[baseline]['acc'])
    b_bals = np.array(metrics_by_method[baseline]['bal'])
    b_oks = np.array(metrics_by_method[baseline]['ok'])
    b_mins = np.array(metrics_by_method[baseline]['min'])
    
    n_ = min(len(ent_accs), len(b_accs))
    if n_ < 2:
        print(f"  vs {baseline}: insufficient data (n={n_})")
        continue
    
    t_acc, p_acc = stats.ttest_rel(ent_accs[:n_], b_accs[:n_])
    t_bal, p_bal = stats.ttest_rel(ent_bals[:n_], b_bals[:n_])
    t_ok, p_ok = stats.ttest_rel(ent_oks[:n_].astype(float), b_oks[:n_].astype(float))
    t_min, p_min = stats.ttest_rel(ent_mins[:n_], b_mins[:n_])
    
    print(f"  vs {baseline:<8}: acc p={p_acc:.4f} {'***' if p_acc<0.001 else '**' if p_acc<0.01 else '*' if p_acc<0.05 else 'ns'} | "
          f"bal p={p_bal:.4f} {'***' if p_bal<0.001 else '**' if p_bal<0.01 else '*' if p_bal<0.05 else 'ns'} | "
          f"ok p={p_ok:.4f} {'***' if p_ok<0.001 else '**' if p_ok<0.01 else '*' if p_ok<0.05 else 'ns'} | "
          f"min p={p_min:.4f} {'***' if p_min<0.001 else '**' if p_min<0.01 else '*' if p_min<0.05 else 'ns'}")

# Effect sizes (Cohen's d)
print(f"\n  Effect sizes (Cohen's d):")
for baseline in ['Average', 'TIES', 'Fisher']:
    b_accs = np.array(metrics_by_method[baseline]['acc'])
    n_ = min(len(ent_accs), len(b_accs))
    if n_ < 2: continue
    diff = ent_accs[:n_] - b_accs[:n_]
    cohens_d = diff.mean() / (diff.std() + 1e-10)
    print(f"  vs {baseline:<8}: d={cohens_d:.2f} ({'large' if abs(cohens_d)>0.8 else 'medium' if abs(cohens_d)>0.5 else 'small'})")

# Print raw data for each seed
print(f"\n  Raw data per seed:")
print(f"  {'Seed':>6} {'Avg_acc':>8} {'TIES_acc':>9} {'Fish_acc':>9} {'ENT_acc':>8} {'ENT_bal':>8} {'ENT_ok':>7}")
for seed in sorted(all_seed_results.keys()):
    r = all_seed_results[seed]
    print(f"  {seed:>6} "
          f"{r.get('Average',{}).get('acc',0):>8.3f} "
          f"{r.get('TIES',{}).get('acc',0):>9.3f} "
          f"{r.get('Fisher',{}).get('acc',0):>9.3f} "
          f"{r.get('ENT',{}).get('acc',0):>8.3f} "
          f"{r.get('ENT',{}).get('bal',0):>8.3f} "
          f"{r.get('ENT',{}).get('ok',0):>5}/10")

elapsed = time.time() - t0
print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

# Save full results
save_data = {
    'seeds': sorted(all_seed_results.keys()),
    'per_seed': {},
    'summary': {},
    'significance': {},
    'time_s': round(elapsed, 1),
}

for seed in sorted(all_seed_results.keys()):
    save_data['per_seed'][seed] = {}
    for m_ in methods:
        if m_ in all_seed_results[seed]:
            r = all_seed_results[seed][m_]
            save_data['per_seed'][seed][m_] = {
                'acc': round(r['acc'], 4), 'bal': round(r['bal'], 4),
                'min': round(r['min'], 4), 'ok': r['ok']
            }

for method in methods:
    d = metrics_by_method[method]
    if len(d['acc']) > 0:
        save_data['summary'][method] = {
            'acc_mean': round(float(np.mean(d['acc'])), 4),
            'acc_std': round(float(np.std(d['acc'])), 4),
            'bal_mean': round(float(np.mean(d['bal'])), 4),
            'bal_std': round(float(np.std(d['bal'])), 4),
            'min_mean': round(float(np.mean(d['min'])), 4),
            'min_std': round(float(np.std(d['min'])), 4),
            'ok_mean': round(float(np.mean(d['ok'])), 4),
            'ok_std': round(float(np.std(d['ok'])), 4),
            'n_seeds': len(d['acc']),
        }

for baseline in ['Average', 'TIES', 'Fisher']:
    b_accs = np.array(metrics_by_method[baseline]['acc'])
    b_bals = np.array(metrics_by_method[baseline]['bal'])
    n_ = min(len(ent_accs), len(b_accs))
    if n_ >= 2:
        t_acc, p_acc = stats.ttest_rel(ent_accs[:n_], b_accs[:n_])
        t_bal, p_bal = stats.ttest_rel(ent_bals[:n_], b_bals[:n_])
        save_data['significance'][f'ENT_vs_{baseline}'] = {
            'n': int(n_),
            'acc_tstat': round(float(t_acc), 4),
            'acc_pvalue': round(float(p_acc), 6),
            'bal_tstat': round(float(t_bal), 4),
            'bal_pvalue': round(float(p_bal), 6),
        }

with open('results/task1_mnist_seeds.json', 'w') as f:
    json.dump(save_data, f, indent=2, default=str)

# Key metrics for grep (§K5)
print(f"\nmetric_ent_acc_mean: {np.mean(metrics_by_method['ENT']['acc']):.4f}")
print(f"metric_ent_acc_std: {np.std(metrics_by_method['ENT']['acc']):.4f}")
print(f"metric_ent_bal_mean: {np.mean(metrics_by_method['ENT']['bal']):.4f}")
print(f"metric_ent_ok_mean: {np.mean(metrics_by_method['ENT']['ok']):.1f}")
if 'TIES' in save_data.get('significance', {}):
    print(f"metric_pvalue_vs_ties_acc: {save_data['significance']['ENT_vs_TIES']['acc_pvalue']:.6f}")
if 'Average' in save_data.get('significance', {}):
    print(f"metric_pvalue_vs_avg_acc: {save_data['significance']['ENT_vs_Average']['acc_pvalue']:.6f}")

print("Done!")
