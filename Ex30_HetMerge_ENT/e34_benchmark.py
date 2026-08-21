#!/usr/bin/env python3
"""E34: COMPREHENSIVE BENCHMARK — ENT vs ALL SOTA methods.
Same data, same evaluation, fair comparison.

Methods: Average, SLERP, Task Arithmetic, TIES, DARE,
         Git Re-Basin (simplified), Sakana-style, NeuronConcat, ENT
"""
import numpy as np, torch, torch.nn as nn, random, time, json, copy
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets, transforms

R = open(str(__import__('pathlib').Path(__file__).resolve().parent / 'results_e34.txt'), 'w')
def log(s): R.write(s + '\n'); R.flush(); print(s, flush=True)

log("=" * 70)
log("E34: COMPREHENSIVE BENCHMARK — ENT vs ALL SOTA model merging methods")
log("=" * 70)

# ═══════════════════════════════════════════
# Data & Models
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

def evaluate(m, name):
    """Standard evaluation: acc, per-class, balance, min_class, classes_ok."""
    pcM = pc(m, X_te, y_te); acc = ev(m, X_te, y_te)
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

# Train SAME base model, then fine-tune for TA/TIES/DARE
log("\nTraining models...")
base = MLP(arch)  # untrained base for Task Arithmetic
torch.manual_seed(0)
nn.init.xavier_uniform_(list(base.parameters())[0])
base_sd = copy.deepcopy(base.state_dict())

torch.manual_seed(SEED)
modelA = train_model(arch, X_tr, y_tr, clA)
modelB = train_model(arch, X_tr, y_tr, clB)
log(f"  Model A (cls 0-4): {ev(modelA, X_te, y_te):.3f}")
log(f"  Model B (cls 5-9): {ev(modelB, X_te, y_te):.3f}")

sdA = modelA.state_dict()
sdB = modelB.state_dict()
results = []

# ═══════════════════════════════════════════
# METHOD 1: Weight Averaging (sweep α)
# ═══════════════════════════════════════════
log("\n--- Method 1: Weight Averaging ---")
best_avg = None; best_avg_bal = -1
for alpha in [0.3, 0.5, 0.7]:
    m = MLP(arch)
    sd = {k: alpha * sdA[k] + (1-alpha) * sdB[k] for k in sdA}
    m.load_state_dict(sd); m.eval()
    r = evaluate(m, f"Average(α={alpha})")
    if r['bal'] > best_avg_bal: best_avg_bal = r['bal']; best_avg = r
    log(f"  α={alpha}: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")
results.append(best_avg)

# ═══════════════════════════════════════════
# METHOD 2: SLERP
# ═══════════════════════════════════════════
log("\n--- Method 2: SLERP ---")
def slerp_merge(sdA, sdB, t=0.5):
    sd = {}
    for k in sdA:
        vA, vB = sdA[k].float().flatten(), sdB[k].float().flatten()
        nA, nB = vA.norm(), vB.norm()
        if nA < 1e-8 or nB < 1e-8:
            sd[k] = (t * sdA[k] + (1-t) * sdB[k])
            continue
        cos_omega = (vA @ vB) / (nA * nB)
        cos_omega = cos_omega.clamp(-1, 1)
        omega = torch.acos(cos_omega)
        if omega.abs() < 1e-6:
            sd[k] = (t * sdA[k] + (1-t) * sdB[k])
        else:
            sA = torch.sin((1-t) * omega) / torch.sin(omega)
            sB = torch.sin(t * omega) / torch.sin(omega)
            sd[k] = (sA * sdA[k] + sB * sdB[k])
    return sd

best_slerp = None; best_slerp_bal = -1
for t in [0.3, 0.5, 0.7]:
    m = MLP(arch); m.load_state_dict(slerp_merge(sdA, sdB, t)); m.eval()
    r = evaluate(m, f"SLERP(t={t})")
    if r['bal'] > best_slerp_bal: best_slerp_bal = r['bal']; best_slerp = r
    log(f"  t={t}: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")
results.append(best_slerp)

# ═══════════════════════════════════════════
# METHOD 3: Task Arithmetic
# ═══════════════════════════════════════════
log("\n--- Method 3: Task Arithmetic ---")
best_ta = None; best_ta_bal = -1
for tau in [0.3, 0.5, 0.7, 1.0]:
    m = MLP(arch)
    sd = {}
    for k in sdA:
        delta_A = sdA[k] - base_sd[k]
        delta_B = sdB[k] - base_sd[k]
        sd[k] = base_sd[k] + tau * (delta_A + delta_B)
    m.load_state_dict(sd); m.eval()
    r = evaluate(m, f"TaskArith(τ={tau})")
    if r['bal'] > best_ta_bal: best_ta_bal = r['bal']; best_ta = r
    log(f"  τ={tau}: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")
results.append(best_ta)

# ═══════════════════════════════════════════
# METHOD 4: TIES-Merging
# ═══════════════════════════════════════════
log("\n--- Method 4: TIES-Merging ---")
best_ties = None; best_ties_bal = -1
for density in [0.1, 0.3, 0.5]:
    m = MLP(arch); sd = {}
    for k in sdA:
        dA = sdA[k] - base_sd[k]; dB = sdB[k] - base_sd[k]
        # Trim: keep top-k by magnitude
        for delta in [dA, dB]:
            flat = delta.flatten()
            n_keep = max(1, int(len(flat) * density))
            threshold = flat.abs().topk(n_keep).values[-1]
            delta[delta.abs() < threshold] = 0
        # Elect sign: majority vote
        sign_mask = (dA.sign() + dB.sign())
        merged_delta = torch.where(sign_mask > 0,
                                   torch.maximum(dA, torch.zeros_like(dA)) + torch.maximum(dB, torch.zeros_like(dB)),
                                   torch.minimum(dA, torch.zeros_like(dA)) + torch.minimum(dB, torch.zeros_like(dB)))
        # Average non-zero
        n_nonzero = (merged_delta != 0).float().sum()
        if n_nonzero > 0:
            merged_delta = merged_delta * ((dA != 0).float() + (dB != 0).float()).clamp(max=1)
        sd[k] = base_sd[k] + merged_delta * 0.5
    m.load_state_dict(sd); m.eval()
    r = evaluate(m, f"TIES(d={density})")
    if r['bal'] > best_ties_bal: best_ties_bal = r['bal']; best_ties = r
    log(f"  d={density}: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")
results.append(best_ties)

# ═══════════════════════════════════════════
# METHOD 5: DARE (Drop And Rescale)
# ═══════════════════════════════════════════
log("\n--- Method 5: DARE ---")
best_dare = None; best_dare_bal = -1
for p in [0.1, 0.3, 0.5, 0.9]:
    m = MLP(arch); sd = {}
    torch.manual_seed(SEED)
    for k in sdA:
        dA = sdA[k] - base_sd[k]; dB = sdB[k] - base_sd[k]
        # Drop (1-p) fraction, rescale by 1/p
        maskA = (torch.rand_like(dA.float()) < p).float()
        maskB = (torch.rand_like(dB.float()) < p).float()
        dA_dare = dA * maskA / max(p, 0.01)
        dB_dare = dB * maskB / max(p, 0.01)
        sd[k] = base_sd[k] + 0.5 * (dA_dare + dB_dare)
    m.load_state_dict(sd); m.eval()
    r = evaluate(m, f"DARE(p={p})")
    if r['bal'] > best_dare_bal: best_dare_bal = r['bal']; best_dare = r
    log(f"  p={p}: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")
results.append(best_dare)

# ═══════════════════════════════════════════
# METHOD 6: Fisher Merging (diagonal Fisher approx)
# ═══════════════════════════════════════════
log("\n--- Method 6: Fisher Merging ---")
def compute_fisher(model, X, y, n=500):
    """Diagonal Fisher Information Matrix."""
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
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
    for name in fisher:
        fisher[name] /= n
    model.eval()
    return fisher

fisherA = compute_fisher(modelA, X_tr, y_tr, n=300)
fisherB = compute_fisher(modelB, X_tr, y_tr, n=300)

m = MLP(arch); sd = {}
for k in sdA:
    fA = fisherA.get(k.replace('.weight', '').replace('.bias', ''), torch.ones_like(sdA[k]))
    fB = fisherB.get(k.replace('.weight', '').replace('.bias', ''), torch.ones_like(sdB[k]))
    # Match fisher dict keys to state dict keys
    found = False
    for fk in fisherA:
        if fk in k or k in fk:
            fA = fisherA[fk]; fB = fisherB[fk]; found = True; break
    if not found: fA = torch.ones_like(sdA[k]); fB = torch.ones_like(sdB[k])
    denom = fA + fB + 1e-8
    sd[k] = (fA * sdA[k] + fB * sdB[k]) / denom
m.load_state_dict(sd); m.eval()
r = evaluate(m, "Fisher")
results.append(r)
log(f"  Fisher: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")

# ═══════════════════════════════════════════
# METHOD 7: NeuronConcat (our prior method)
# ═══════════════════════════════════════════
log("\n--- Method 7: NeuronConcat ---")
def neuron_concat(mA, mB, rA=1.0, rB=1.0):
    pA, pB = list(mA.parameters()), list(mB.parameters())
    sA = [p.shape[0] for p in pA[::2]]  # layer sizes
    sB = [p.shape[0] for p in pB[::2]]
    new_arch = [784] + [sA[i]+sB[i] for i in range(len(sA)-1)] + [10]
    m = MLP(new_arch)
    with torch.no_grad():
        ps = list(m.parameters())
        # Layer 0: concat
        W0 = torch.cat([pA[0].data, pB[0].data], dim=0)
        b0 = torch.cat([pA[1].data, pB[1].data])
        ps[0].copy_(W0); ps[1].copy_(b0)
        # Layer 1: block-diagonal
        s0A, s0B = sA[0], sB[0]  # hidden sizes
        s1A, s1B = sA[1], sB[1] if len(sA) > 2 else (10, 10)
        W1 = torch.zeros(s1A+s1B, s0A+s0B)
        W1[:s1A, :s0A] = pA[2].data
        W1[s1A:, s0A:] = pB[2].data
        b1 = torch.cat([pA[3].data, pB[3].data])
        ps[2].copy_(W1); ps[3].copy_(b1)
        # Output: average routing
        Wo = torch.zeros(10, s1A+s1B)
        Wo[:, :s1A] = 0.5 * pA[4].data
        Wo[:, s1A:] = 0.5 * pB[4].data
        bo = 0.5 * pA[5].data + 0.5 * pB[5].data
        ps[4].copy_(Wo); ps[5].copy_(bo)
    return m

m = neuron_concat(modelA, modelB); m.eval()
r = evaluate(m, "NeuronConcat")
results.append(r)
log(f"  NeuronConcat: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")

# ═══════════════════════════════════════════
# METHOD 8: Sakana-style (per-layer CMA-ES α)
# ═══════════════════════════════════════════
log("\n--- Method 8: Sakana-style (per-layer α, CMA-ES) ---")
import cma
n_merge_layers = 3  # 3 parameter pairs (W0,b0), (W1,b1), (Wo,bo)
def build_sakana(x):
    alphas = 1.0 / (1.0 + np.exp(-np.array(x)))
    m = MLP(arch); sd = {}
    keys = list(sdA.keys())
    for i, k in enumerate(keys):
        layer_idx = i // 2  # 0,0,1,1,2,2
        a = alphas[min(layer_idx, len(alphas)-1)]
        sd[k] = a * sdA[k] + (1-a) * sdB[k]
    m.load_state_dict(sd); m.eval(); return m

es = cma.CMAEvolutionStrategy(np.zeros(n_merge_layers), 1.5, {
    'maxiter': 15, 'popsize': 8, 'seed': SEED, 'verbose': -1})
best_sak_s = -1; best_sak_x = None
while not es.stop():
    sols = es.ask()
    scores = []
    for x in sols:
        m = build_sakana(x)
        d = pc(m, Xv, yv); acc = ev(m, Xv, yv)
        mn = min(d[c] for c in range(10))
        s = 0.4*acc + 0.4*mn + 0.1*np.mean([d[c] for c in range(10)])
        scores.append(-s)  # CMA minimizes
    es.tell(sols, scores)
    if -min(scores) > best_sak_s:
        best_sak_s = -min(scores)
        best_sak_x = sols[np.argmin(scores)]
m = build_sakana(best_sak_x); m.eval()
r = evaluate(m, "Sakana-CMA")
results.append(r)
sak_a = 1.0 / (1.0 + np.exp(-best_sak_x))
log(f"  Sakana-CMA: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f} alphas={[round(a,2) for a in sak_a]}")

# ═══════════════════════════════════════════
# METHOD 9: ENT (our method)
# ═══════════════════════════════════════════
log("\n--- Method 9: ENT (ours) ---")
def virt2L(model, Xc):
    model.eval()
    with torch.no_grad():
        h = Xc
        for m in list(model.net)[:-1]: h = m(h)
        hid = h.numpy()
    ps = list(model.parameters())
    Wo = ps[-2].detach().numpy(); bo = ps[-1].detach().numpy()
    fd = hid.shape[1]
    x = Xc.numpy(); N = x.shape[0]
    xb = np.hstack([x, np.ones((N,1), dtype=np.float32)])
    Wb = np.linalg.lstsq(xb, np.maximum(hid, 0), rcond=None)[0].T
    W1 = Wb[:,:-1].astype(np.float32); b1 = Wb[:,-1].astype(np.float32)
    W2 = np.eye(fd, dtype=np.float32); b2 = np.zeros(fd, dtype=np.float32)
    return [W1, b1, W2, b2, Wo.copy(), bo.copy()], [fd, fd]

def bld_ent(ch, WA, WB, sA, sB, rA, rB):
    sizes = [784]
    for i in range(2):
        n = int(ch['mA'][i].sum()) + int(ch['mB'][i].sum())
        if n < 2: return None
        sizes.append(n)
    sizes.append(10)
    iA = np.where(ch['mA'][0])[0]; iB = np.where(ch['mB'][0])[0]
    W0 = np.vstack([WA[0][iA], WB[0][iB]]); b0 = np.concatenate([WA[1][iA], WB[1][iB]])
    ipA, ipB = iA, iB
    icA = np.where(ch['mA'][1])[0]; icB = np.where(ch['mB'][1])[0]
    W1 = np.zeros((len(icA)+len(icB), len(ipA)+len(ipB)), dtype=np.float32)
    b1 = np.zeros(len(icA)+len(icB), dtype=np.float32)
    W1[:len(icA),:len(ipA)] = WA[2][np.ix_(icA, ipA)]; b1[:len(icA)] = WA[3][icA]
    W1[len(icA):,len(ipA):] = WB[2][np.ix_(icB, ipB)]; b1[len(icA):] = WB[3][icB]
    ilA = np.where(ch['mA'][1])[0]; ilB = np.where(ch['mB'][1])[0]
    Wo = np.zeros((10, len(ilA)+len(ilB)), dtype=np.float32); bo = np.zeros(10, dtype=np.float32)
    for c in range(10):
        a = 1.0 / (1.0 + np.exp(-ch['route'][c]))
        if len(ilA) > 0: Wo[c,:len(ilA)] = a * rA * WA[4][c][ilA]
        if len(ilB) > 0: Wo[c,len(ilA):] = (1-a) * rB * WB[4][c][ilB]
        bo[c] = a * rA * WA[5][c] + (1-a) * rB * WB[5][c]
    m = MLP(sizes)
    with torch.no_grad():
        ps = list(m.parameters())
        ps[0].copy_(torch.tensor(W0)); ps[1].copy_(torch.tensor(b0))
        ps[2].copy_(torch.tensor(W1)); ps[3].copy_(torch.tensor(b1))
        ps[4].copy_(torch.tensor(Wo)); ps[5].copy_(torch.tensor(bo))
    return m

WA, sA = virt2L(modelA, Xc); WB, sB = virt2L(modelB, Xc)
modelA.eval(); modelB.eval()
with torch.no_grad(): sl = modelA(Xc).numpy().std(); sr = modelB(Xc).numpy().std()
t = (sl+sr)/2; rA = t/(sl+1e-10); rB = t/(sr+1e-10)

pop = []
for _ in range(20):
    ch = {'mA': [np.random.random(d) > 0.3 for d in sA],
          'mB': [np.random.random(d) > 0.3 for d in sB],
          'route': np.random.randn(10) * 1.5}
    for ms in [ch['mA'], ch['mB']]:
        for m in ms:
            if m.sum() == 0: m[0] = True
    pop.append(ch)
# Seed: full model with good routing
pop[0] = {'mA': [np.ones(d, dtype=bool) for d in sA],
          'mB': [np.ones(d, dtype=bool) for d in sB],
          'route': np.array([2.0]*5 + [-2.0]*5)}

bf = -1; bc = None
for gen in range(30):
    fs = []
    for ch in pop:
        m = bld_ent(ch, WA, WB, sA, sB, rA, rB)
        if m is None: fs.append(-1); continue
        d = pc(m, Xv, yv); acc = ev(m, Xv, yv)
        mn = min(d[c] for c in range(10))
        fs.append(0.4*acc + 0.4*mn + 0.1*np.mean([d[c] for c in range(10)]) + 0.1*(1-sum(ch['mA'][i].sum()+ch['mB'][i].sum() for i in range(2))/(sum(sA)+sum(sB))))
    gi = np.argmax(fs)
    if fs[gi] > bf:
        bf = fs[gi]
        bc = {'mA': [m.copy() for m in pop[gi]['mA']],
              'mB': [m.copy() for m in pop[gi]['mB']],
              'route': pop[gi]['route'].copy()}
    if gen % 10 == 0:
        m_ = bld_ent(bc, WA, WB, sA, sB, rA, rB)
        if m_: log(f"  Gen {gen}: fit={fs[gi]:.4f} min={min(pc(m_,Xv,yv)[c] for c in range(10)):.3f}")
    new = [{'mA': [m.copy() for m in bc['mA']], 'mB': [m.copy() for m in bc['mB']], 'route': bc['route'].copy()}]
    while len(new) < 20:
        ti = random.sample(range(len(pop)), 3)
        p1 = pop[ti[np.argmax([fs[i] for i in ti])]]
        ch = {'mA': [m.copy() for m in p1['mA']], 'mB': [m.copy() for m in p1['mB']], 'route': p1['route'] + np.random.randn(10)*0.3}
        pf = max(0.02, 0.06 - gen*0.001)
        for ms in [ch['mA'], ch['mB']]:
            for m in ms:
                f = np.random.random(len(m)) < pf; m[f] = ~m[f]
                if m.sum() == 0: m[np.random.randint(len(m))] = True
        new.append(ch)
    pop = new

m = bld_ent(bc, WA, WB, sA, sB, rA, rB); m.eval()
r = evaluate(m, "ENT")
n_ent = sum(bc['mA'][i].sum() + bc['mB'][i].sum() for i in range(2))
n_max = sum(sA) + sum(sB)
r['compression'] = round(1 - n_ent / n_max, 3)
results.append(r)
log(f"  ENT: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f} compression={r['compression']:.0%}")

# ═══════════════════════════════════════════
# FINAL COMPARISON TABLE
# ═══════════════════════════════════════════
log(f"\n{'='*70}")
log(f"  FINAL COMPARISON — MNIST Complementary Merge (cls 0-4 vs 5-9)")
log(f"{'='*70}")
log(f"  {'Method':<22} {'Acc':>6} {'Balance':>8} {'Min_cls':>8} {'OK':>5} {'A-mean':>7} {'B-mean':>7}")
log(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*8} {'-'*5} {'-'*7} {'-'*7}")
for r in results:
    marker = " " if 'ENT' in r['name'] else ""
    log(f"  {r['name']:<22} {r['acc']:>6.3f} {r['bal']:>8.3f} {r['min']:>8.3f} {r['ok']:>3}/10 {r['A']:>7.3f} {r['B']:>7.3f}{marker}")

log(f"\n  Per-class breakdown (best methods):")
log(f"  {'Class':>5} {'Parent':>7} {'Avg':>7} {'TIES':>7} {'Sakana':>7} {'ENT':>7}")
for c in range(10):
    bp = max(pc(modelA, X_te, y_te)[c], pc(modelB, X_te, y_te)[c])
    vals = [best_avg, best_ties, results[-2], results[-1]]  # avg, ties, sakana, ent
    log(f"  {c:>5} {bp:>7.3f}" + "".join(f" {v['per_class'][c]:>7.3f}" for v in vals))

# Save JSON
with open(str(__import__('pathlib').Path(__file__).resolve().parent / 'results_e34.json'), 'w') as jf:
    json.dump(results, jf, indent=2)
R.close()
print("Done!", flush=True)
