#!/usr/bin/env python3
"""E04: DAVE proxy validation + EA optimization on toy.
Q1: Does DAVE score correlate with per-class accuracy?
Q2: Can EA + DAVE find good λ_A, λ_B data-free?

DAVE = feature variance + output entropy on random noise.
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
    def features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x); x = self.block2(x)
        return self.pool(x).flatten(1)

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
mBase = TinyResNet(4)
torch.manual_seed(SEED)
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
print(f"Parents trained ({time.time()-t0:.1f}s) A={pcA} B={pcB}", flush=True)

# ═══ DAVE proxy ═══
noise = torch.randn(256, 3, 32, 32)

def dave_score(model):
    """DAVE zero-cost proxy on random noise."""
    model.eval()
    with torch.no_grad():
        feats = model.features(noise)  # [256, 16]
        logits = model(noise)          # [256, 4]
    # Feature variance across channels
    var_score = feats.var(dim=0).mean().item()
    # Entropy of softmax
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * (probs+1e-10).log()).sum(1).mean().item()
    # Output diversity (variance of logit means per class)
    logit_div = logits.mean(0).var().item()
    return var_score + entropy + logit_div

# ═══ Build merged model ═══
conv_keys = [k for k in sdA if 'weight' in k and 'bn' not in k and 'fc' not in k]
all_keys = [k for k in sdA if 'fc' not in k and 'num_batches_tracked' not in k]
dim = len(conv_keys)*2 + 2  # λ_A + λ_B per conv layer + 2 scales

def build_merged(x):
    lambdas_A = 1/(1+np.exp(-x[:len(conv_keys)]))
    lambdas_B = 1/(1+np.exp(-x[len(conv_keys):2*len(conv_keys)]))
    sA, sB = x[-2], x[-1]
    sd = {}
    for ki, k in enumerate(conv_keys):
        tA = sdA[k]-sd_base[k]; tB = sdB[k]-sd_base[k]
        sd[k] = sd_base[k] + lambdas_A[ki]*tA + lambdas_B[ki]*tB
    # BN: average parent stats
    for k in sdA:
        if 'fc' in k or 'num_batches_tracked' in k: continue
        if k not in sd:
            tA = sdA[k]-sd_base[k]; tB = sdB[k]-sd_base[k]
            sd[k] = sd_base[k] + 0.5*tA + 0.5*tB
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

# ═══ Q1: DAVE correlation ═══
print(f"\n--- DAVE Correlation Study ---", flush=True)
samples = []
for _ in range(20):
    x = np.random.randn(dim)
    x[-2]=1.0+np.random.randn()*0.2; x[-1]=1.0+np.random.randn()*0.2
    m = build_merged(x); reset_bn_noise(m)
    ds = dave_score(m)
    # True eval
    mask = sum(y_te==c for c in ALL).bool()
    Xt = X_te[mask]; yt = y_te[mask]
    with torch.no_grad():
        preds = m(Xt).argmax(1)
    acc = (preds==yt).float().mean().item()
    pc = {c:(preds[yt==c]==c).float().mean().item() for c in ALL}
    mn = min(pc.values())
    rets = sum(1 for c in ALL if pc[c]>=0.9*parent_pc[c])
    samples.append({'dave':ds, 'acc':acc, 'min':mn, 'retained':rets})

dave_vals = np.array([s['dave'] for s in samples])
acc_vals = np.array([s['acc'] for s in samples])
min_vals = np.array([s['min'] for s in samples])
ret_vals = np.array([s['retained'] for s in samples])

from scipy.stats import spearmanr
r_acc, p_acc = spearmanr(dave_vals, acc_vals)
r_min, p_min = spearmanr(dave_vals, min_vals)
r_ret, p_ret = spearmanr(dave_vals, ret_vals)
print(f"  Spearman DAVE↔acc:      r={r_acc:+.3f} p={p_acc:.4f}")
print(f"  Spearman DAVE↔min_class: r={r_min:+.3f} p={p_min:.4f}")
print(f"  Spearman DAVE↔retained: r={r_ret:+.3f} p={p_ret:.4f}")

# ═══ Q2: EA + DAVE ═══
print(f"\n--- EA + DAVE (data-free!) ---", flush=True)

def fitness(x):
    m = build_merged(x); reset_bn_noise(m, 256)
    return -dave_score(m)  # maximize DAVE

x0 = np.zeros(dim); x0[-2]=1.0; x0[-1]=1.0
es = cma.CMAEvolutionStrategy(x0, 0.5,
    {'maxiter': 25, 'popsize': 14, 'seed': SEED, 'verbose': -1, 'timeout': 30})
bf = float('inf'); bx = None; gen = 0
while not es.stop():
    sols = es.ask(); sc = [fitness(x) for x in sols]
    es.tell(sols, sc)
    if min(sc)<bf: bf=min(sc); bx=sols[np.argmin(sc)]
    gen += 1

m_ea = build_merged(bx); reset_bn_noise(m_ea)
# Eval on REAL test data
mask = sum(y_te==c for c in ALL).bool()
Xt = X_te[mask]; yt = y_te[mask]
with torch.no_grad():
    preds = m_ea(Xt).argmax(1)
pc = {c:(preds[yt==c]==c).float().mean().item() for c in ALL}

print(f"\n  EA+DAVE (data-free):")
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

lambdas_A = 1/(1+np.exp(-bx[:len(conv_keys)]))
lambdas_B = 1/(1+np.exp(-bx[len(conv_keys):2*len(conv_keys)]))
print(f"\n  λ_A: {[round(l,3) for l in lambdas_A]}")
print(f"  λ_B: {[round(l,3) for l in lambdas_B]}")
print(f"  sA={bx[-2]:.3f} sB={bx[-1]:.3f}")
print(f"  CMA: {gen} gens DAVE={-bf:.4f}")

elapsed = time.time()-t0
print(f"\n  Time: {elapsed:.1f}s")
print(f"\nmetric_dave_acc_corr: {r_acc:.4f}")
print(f"metric_dave_min_corr: {r_min:.4f}")
print(f"metric_ea_dave_retained: {retained}")
print("Done!", flush=True)
