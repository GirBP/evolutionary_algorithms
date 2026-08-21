#!/usr/bin/env python3
"""E05: Balance proxy — output symmetry on noise as data-free fitness.
Key insight: a good merge should produce BALANCED class predictions on random noise.
If one parent dominates, noise predictions will cluster on that parent's classes.
Fitness = entropy of mean class probabilities on noise + logit magnitude balance.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, time
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets
import cma

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
t0 = time.time()

raw_tr = datasets.CIFAR10('/tmp/cifar10', train=True, download=True)
raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mn = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
sd = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_tr = (torch.tensor(raw_tr.data).permute(0,3,1,2).float()/255-mn)/sd
y_tr = torch.tensor(raw_tr.targets)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255-mn)/sd
y_te = torch.tensor(raw_te.targets)
clA, clB = [0,1], [2,3]; ALL = clA+clB

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch,ch,3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch,ch,3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))+x)

class TinyResNet(nn.Module):
    def __init__(self, nc=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.block1 = ResBlock(16)
        self.block2 = ResBlock(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, nc)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x); x = self.block2(x)
        return self.fc(self.pool(x).flatten(1))

def train_tiny(cls_list, seed_p, epochs=15, base_sd=None):
    torch.manual_seed(seed_p); np.random.seed(seed_p); random.seed(seed_p)
    m = TinyResNet(len(cls_list))
    if base_sd:
        sdd = m.state_dict()
        for k in base_sd:
            if 'fc' not in k and k in sdd: sdd[k] = base_sd[k].clone()
        m.load_state_dict(sdd)
    cmap = {c:i for i,c in enumerate(cls_list)}
    mask = sum(y_tr==c for c in cls_list).bool()
    Xs = X_tr[mask]; ys = torch.tensor([cmap[y.item()] for y in y_tr[mask]])
    idx = torch.cat([torch.where(ys==i)[0][:1000] for i in range(len(cls_list))])
    Xs, ys = Xs[idx], ys[idx]
    m.to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=0.003)
    m.train()
    for ep in range(epochs):
        pm = torch.randperm(len(Xs))
        for i in range(0, len(Xs), 128):
            ix = pm[i:i+128]
            loss = F.cross_entropy(m(Xs[ix].to(DEVICE)), ys[ix].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    m.to('cpu').eval()
    mask_te = sum(y_te==c for c in cls_list).bool()
    Xt = X_te[mask_te]; yt = torch.tensor([cmap[y.item()] for y in y_te[mask_te]])
    with torch.no_grad():
        pc = {cls_list[i]:(m(Xt).argmax(1)[yt==i]==i).float().mean().item() for i in range(len(cls_list))}
    return m, pc

# Train base + parents
mBase = TinyResNet(4); torch.manual_seed(SEED)
mask_all = sum(y_tr==c for c in ALL).bool()
Xs_all = X_tr[mask_all]; ys_all = y_tr[mask_all]
idx_all = torch.cat([torch.where(ys_all==c)[0][:500] for c in ALL])
mBase.to(DEVICE); opt = torch.optim.Adam(mBase.parameters(), lr=0.003)
mBase.train()
for ep in range(5):
    pm = torch.randperm(len(idx_all))
    for i in range(0, len(idx_all), 128):
        ix = idx_all[pm[i:i+128]]
        loss = F.cross_entropy(mBase(Xs_all[ix].to(DEVICE)), ys_all[ix].to(DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
mBase.to('cpu').eval()
sd_base = {k:v.clone() for k,v in mBase.state_dict().items()}

mA, pcA = train_tiny(clA, SEED, 15, sd_base)
mB, pcB = train_tiny(clB, SEED+100, 15, sd_base)
sdA, sdB = mA.state_dict(), mB.state_dict()
parent_pc = {}; parent_pc.update(pcA); parent_pc.update(pcB)
print(f"Parents ({time.time()-t0:.1f}s) A={pcA} B={pcB}", flush=True)

# ═══ Model builder ═══
conv_keys = [k for k in sdA if 'weight' in k and 'bn' not in k and 'fc' not in k]
dim = len(conv_keys)*2 + 2

def build_merged(x):
    lA = 1/(1+np.exp(-x[:len(conv_keys)]))
    lB = 1/(1+np.exp(-x[len(conv_keys):2*len(conv_keys)]))
    sA, sB = x[-2], x[-1]
    sd = {}
    for ki, k in enumerate(conv_keys):
        tA = sdA[k]-sd_base[k]; tB = sdB[k]-sd_base[k]
        sd[k] = sd_base[k] + lA[ki]*tA + lB[ki]*tB
    for k in sdA:
        if 'fc' in k or 'num_batches_tracked' in k: continue
        if k not in sd:
            sd[k] = sd_base[k] + 0.5*(sdA[k]-sd_base[k]) + 0.5*(sdB[k]-sd_base[k])
    for k in sdA:
        if 'num_batches_tracked' in k: sd[k] = torch.tensor(0)
    fw = torch.zeros(4,16); fb = torch.zeros(4)
    for ci,c in enumerate(clA): fw[c]=sdA['fc.weight'][ci]*sA; fb[c]=sdA['fc.bias'][ci]*sA
    for ci,c in enumerate(clB): fw[c]=sdB['fc.weight'][ci]*sB; fb[c]=sdB['fc.bias'][ci]*sB
    sd['fc.weight']=fw; sd['fc.bias']=fb
    m = TinyResNet(4); m.load_state_dict(sd); return m

def reset_bn_noise(m, n=512):
    m.train()
    ns = torch.randn(n,3,32,32)
    with torch.no_grad():
        for i in range(0,n,64): m(ns[i:i+64])
    m.eval()

# ═══ Balance proxy (data-free!) ═══
noise_eval = torch.randn(512, 3, 32, 32)

def balance_fitness(x):
    """Data-free fitness: balance of predictions on noise + constrained scales."""
    m = build_merged(x); reset_bn_noise(m, 256)
    m.eval()
    with torch.no_grad():
        logits = m(noise_eval)  # [512, 4]
        probs = F.softmax(logits, dim=1)
    
    # 1. Mean class probability should be uniform (1/4 each)
    mean_probs = probs.mean(0)  # [4]
    # KL from uniform
    uniform = torch.ones(4)/4
    kl = (mean_probs * (mean_probs/(uniform+1e-10)+1e-10).log()).sum().item()
    
    # 2. Per-sample entropy (higher = more uncertain = better)
    entropy = -(probs * (probs+1e-10).log()).sum(1).mean().item()
    
    # 3. Prediction diversity (how many classes used?)
    pred_classes = logits.argmax(1)
    n_unique = len(pred_classes.unique())
    
    # 4. Scale balance penalty (prevent collapse)
    sA, sB = abs(x[-2]), abs(x[-1])
    scale_penalty = abs(np.log(sA+1e-5) - np.log(sB+1e-5))
    
    # Minimize: KL + scale_penalty - entropy_bonus - diversity_bonus
    return kl + 0.5*scale_penalty - 0.3*entropy - 0.2*(n_unique/4)

# ═══ Correlation study ═══
print(f"\n--- Balance Proxy Correlation ---", flush=True)
samples = []
for trial in range(30):
    x = np.random.randn(dim)*0.5
    x[-2] = 0.8 + np.random.rand()*0.4
    x[-1] = 0.8 + np.random.rand()*0.4
    m = build_merged(x); reset_bn_noise(m)
    bf = balance_fitness(x)
    mask = sum(y_te==c for c in ALL).bool()
    Xt = X_te[mask]; yt = y_te[mask]
    with torch.no_grad():
        preds = m(Xt).argmax(1)
    acc = (preds==yt).float().mean().item()
    pc = {c:(preds[yt==c]==c).float().mean().item() for c in ALL}
    mn_ = min(pc.values())
    samples.append({'balance':-bf, 'acc':acc, 'min':mn_})

from scipy.stats import spearmanr
bv = np.array([s['balance'] for s in samples])
av = np.array([s['acc'] for s in samples])
mv = np.array([s['min'] for s in samples])
r_acc, p_acc = spearmanr(bv, av)
r_min, p_min = spearmanr(bv, mv)
print(f"  Balance↔acc:  r={r_acc:+.3f} p={p_acc:.4f}")
print(f"  Balance↔min:  r={r_min:+.3f} p={p_min:.4f}")

# ═══ EA + Balance proxy ═══
print(f"\n--- EA + Balance Proxy (data-free!) ---", flush=True)
x0 = np.zeros(dim); x0[-2]=1.0; x0[-1]=1.0
es = cma.CMAEvolutionStrategy(x0, 0.3,
    {'maxiter': 30, 'popsize': 16, 'seed': SEED, 'verbose': -1, 'timeout': 25,
     'bounds': [[-3]*len(conv_keys)*2 + [0.3, 0.3], [3]*len(conv_keys)*2 + [3.0, 3.0]]})
bf_best = float('inf'); bx = None; gen = 0
while not es.stop():
    sols = es.ask(); sc = [balance_fitness(x) for x in sols]
    es.tell(sols, sc)
    if min(sc) < bf_best: bf_best = min(sc); bx = sols[np.argmin(sc)]
    gen += 1

m_ea = build_merged(bx); reset_bn_noise(m_ea)
mask = sum(y_te==c for c in ALL).bool()
Xt = X_te[mask]; yt = y_te[mask]
with torch.no_grad(): preds = m_ea(Xt).argmax(1)
pc = {c:(preds[yt==c]==c).float().mean().item() for c in ALL}

print(f"\n  EA+Balance (data-free):")
print(f"  {'Class':>5} | {'Parent':>8} | {'Merged':>8} | {'Drop%':>7} | Retained?")
print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}")
retained = 0
for c in ALL:
    p = parent_pc[c]; m_ = pc[c]
    drop = (1-m_/p)*100 if p>0 else 100
    ok = '✅' if drop<=10 else '❌'
    if drop<=10: retained+=1
    print(f"  {c:>5} | {p:>8.3f} | {m_:>8.3f} | {drop:>6.1f}% | {ok}")
print(f"  Retention: {retained}/{len(ALL)} (drop ≤ 10%)")

lA = 1/(1+np.exp(-bx[:len(conv_keys)]))
lB = 1/(1+np.exp(-bx[len(conv_keys):2*len(conv_keys)]))
print(f"\n  λ_A: {[round(l,3) for l in lA]}")
print(f"  λ_B: {[round(l,3) for l in lB]}")
print(f"  sA={bx[-2]:.3f} sB={bx[-1]:.3f}")
print(f"  CMA: {gen} gens")

elapsed = time.time()-t0
print(f"\n  Time: {elapsed:.1f}s")
print(f"\nmetric_balance_acc_corr: {r_acc:.4f}")
print(f"metric_balance_min_corr: {r_min:.4f}")
print(f"metric_ea_balance_retained: {retained}")
print("Done!", flush=True)
