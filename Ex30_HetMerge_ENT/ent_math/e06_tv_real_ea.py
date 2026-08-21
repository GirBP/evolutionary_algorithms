#!/usr/bin/env python3
"""E06: Task vectors + real data BN reset + CMA-ES per-layer λ.
Combines the task vector framework (W = W_base + λ_A·τ_A + λ_B·τ_B)
with REAL data for BN reset and fitness evaluation.
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

# Key groups for per-layer λ
GROUPS = ['conv1', 'block1.conv1', 'block1.conv2', 'block2.conv1', 'block2.conv2']
key_group = {}
all_keys = [k for k in sdA if 'fc' not in k and 'num_batches_tracked' not in k]
for k in all_keys:
    for gi, g in enumerate(GROUPS):
        if k.startswith(g+'.'): key_group[k] = gi; break
    else:
        # BN keys that don't match any group — use nearest conv
        for gi, g in enumerate(GROUPS):
            prefix = g.rsplit('.', 1)[0] if '.' in g else g
            if k.startswith(prefix):
                key_group[k] = gi; break

dim = len(GROUPS)*2 + 2  # λ_A + λ_B per group + sA + sB

def build_merged(x):
    lA = 1/(1+np.exp(-x[:len(GROUPS)]))
    lB = 1/(1+np.exp(-x[len(GROUPS):2*len(GROUPS)]))
    sA, sB = x[-2], x[-1]
    sd = {}
    for k in all_keys:
        gi = key_group.get(k, 0)
        tA = sdA[k]-sd_base[k]; tB = sdB[k]-sd_base[k]
        sd[k] = sd_base[k] + lA[gi]*tA + lB[gi]*tB
    for k in sdA:
        if 'num_batches_tracked' in k: sd[k] = torch.tensor(0)
    fw = torch.zeros(4,16); fb = torch.zeros(4)
    for ci,c in enumerate(clA): fw[c]=sdA['fc.weight'][ci]*sA; fb[c]=sdA['fc.bias'][ci]*sA
    for ci,c in enumerate(clB): fw[c]=sdB['fc.weight'][ci]*sB; fb[c]=sdB['fc.bias'][ci]*sB
    sd['fc.weight']=fw; sd['fc.bias']=fb
    m = TinyResNet(4); m.load_state_dict(sd); return m

X_cal = X_tr[:3000]; y_cal_full = y_tr[:3000]
mask_cal = sum(y_cal_full==c for c in ALL).bool()
X_c = X_cal[mask_cal]; y_c = y_cal_full[mask_cal]

def fitness(x):
    m = build_merged(x)
    # BN reset on REAL data
    m.train()
    with torch.no_grad():
        for i in range(0, len(X_cal), 256): m(X_cal[i:i+256])
    m.eval()
    with torch.no_grad():
        preds = m(X_c).argmax(1)
    acc = (preds==y_c).float().mean().item()
    pc = {c:(preds[y_c==c]==c).float().mean().item() for c in ALL if (y_c==c).sum()>0}
    mn = min(pc.values()) if pc else 0
    rets = sum(1 for c in ALL if pc.get(c,0) >= 0.9*parent_pc.get(c,0))
    return -(0.3*acc + 0.3*(rets/4) + 0.3*mn + 0.1*np.mean(list(pc.values())))

# CMA-ES
print(f"CMA-ES (dim={dim})...", flush=True)
x0 = np.zeros(dim); x0[-2]=1.0; x0[-1]=1.0
es = cma.CMAEvolutionStrategy(x0, 0.5,
    {'maxiter': 25, 'popsize': 14, 'seed': SEED, 'verbose': -1, 'timeout': 35})
bf = float('inf'); bx = None; gen = 0
while not es.stop():
    sols = es.ask(); sc = [fitness(x) for x in sols]
    es.tell(sols, sc)
    if min(sc)<bf: bf=min(sc); bx=sols[np.argmin(sc)]
    gen += 1
    if gen % 5 == 0: print(f"  Gen {gen}: fitness={-bf:.4f}", flush=True)

# Final eval on TEST
m_final = build_merged(bx)
m_final.train()
with torch.no_grad():
    for i in range(0, 5000, 256): m_final(X_tr[i:i+256])
m_final.eval()

mask = sum(y_te==c for c in ALL).bool()
Xt = X_te[mask]; yt = y_te[mask]
with torch.no_grad(): preds = m_final(Xt).argmax(1)
pc = {c:(preds[yt==c]==c).float().mean().item() for c in ALL}

print(f"\n  TV + Real BN + CMA-ES:")
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

lA = 1/(1+np.exp(-bx[:len(GROUPS)]))
lB = 1/(1+np.exp(-bx[len(GROUPS):2*len(GROUPS)]))
print(f"\n  λ_A: {[round(l,3) for l in lA]}")
print(f"  λ_B: {[round(l,3) for l in lB]}")
print(f"  sA={bx[-2]:.3f} sB={bx[-1]:.3f}")
print(f"  CMA: {gen} gens")

elapsed = time.time()-t0
print(f"\n  Time: {elapsed:.1f}s")
print(f"\nmetric_ea_tv_retained: {retained}")
print(f"metric_ea_tv_min: {min(pc.values()):.4f}")
print("Done!", flush=True)
