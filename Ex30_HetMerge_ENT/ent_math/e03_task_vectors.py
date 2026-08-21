#!/usr/bin/env python3
"""E03: Task vector analysis on toy TinyResNet.
Q: Are task vectors τ_A, τ_B approximately orthogonal for complementary classes?
Also: naive task arithmetic merge + BN noise reset + drop% table.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, time, sys
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
t0 = time.time()

# Data
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
    if base_sd is not None:
        # Load base weights (except fc)
        sdd = m.state_dict()
        for k in base_sd:
            if 'fc' not in k and k in sdd:
                sdd[k] = base_sd[k].clone()
        m.load_state_dict(sdd)
    cmap = {c:i for i,c in enumerate(cls_list)}
    mask = sum(y_tr==c for c in cls_list).bool()
    Xs, ys = X_tr[mask], torch.tensor([cmap[y.item()] for y in y_tr[mask]])
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
    # Eval
    mask_te = sum(y_te==c for c in cls_list).bool()
    Xt = X_te[mask_te]; yt = torch.tensor([cmap[y.item()] for y in y_te[mask_te]])
    with torch.no_grad():
        acc = (m(Xt).argmax(1)==yt).float().mean().item()
        pc = {cls_list[i]:(m(Xt).argmax(1)[yt==i]==i).float().mean().item() for i in range(len(cls_list))}
    return m, acc, pc

# ═══ Train base (on ALL classes briefly) then fine-tune parents ═══
print("Training base + parents...", flush=True)
mBase = TinyResNet(4)
# Train base on ALL 4 classes for 5 epochs (minimal shared representation)
torch.manual_seed(SEED)
mask_all = sum(y_tr==c for c in ALL).bool()
Xs_all = X_tr[mask_all]; ys_all = y_tr[mask_all]
idx_all = torch.cat([torch.where(ys_all==c)[0][:500] for c in ALL])
Xs_all, ys_all = Xs_all[idx_all], ys_all[idx_all]
mBase.to(DEVICE)
opt = torch.optim.Adam(mBase.parameters(), lr=0.003)
mBase.train()
for ep in range(5):
    pm = torch.randperm(len(Xs_all))
    for i in range(0, len(Xs_all), 128):
        ix = pm[i:i+128]
        loss = F.cross_entropy(mBase(Xs_all[ix].to(DEVICE)), ys_all[ix].to(DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
mBase.to('cpu').eval()
sd_base = {k:v.clone() for k,v in mBase.state_dict().items()}

# Fine-tune parents FROM BASE
mA, accA, pcA = train_tiny(clA, SEED, epochs=15, base_sd=sd_base)
mB, accB, pcB = train_tiny(clB, SEED+100, epochs=15, base_sd=sd_base)
sdA, sdB = mA.state_dict(), mB.state_dict()
print(f"Parents: A={accA:.3f} pc={pcA}  B={accB:.3f} pc={pcB} ({time.time()-t0:.1f}s)", flush=True)

# ═══ Task Vector Analysis ═══
print(f"\n--- Task Vector Analysis ---", flush=True)
conv_keys = [k for k in sdA if 'weight' in k and 'bn' not in k and 'fc' not in k]

for k in conv_keys:
    tA = (sdA[k] - sd_base[k]).flatten().float()
    tB = (sdB[k] - sd_base[k]).flatten().float()
    cos = F.cosine_similarity(tA.unsqueeze(0), tB.unsqueeze(0)).item()
    nA = torch.norm(tA).item()
    nB = torch.norm(tB).item()
    print(f"  {k:25s}: cos={cos:+.4f}  |τ_A|={nA:.3f}  |τ_B|={nB:.3f}", flush=True)

# Also BN parameters
bn_keys = [k for k in sdA if 'bn' in k and ('weight' in k or 'bias' in k)]
for k in bn_keys:
    tA = (sdA[k] - sd_base[k]).flatten().float()
    tB = (sdB[k] - sd_base[k]).flatten().float()
    cos = F.cosine_similarity(tA.unsqueeze(0), tB.unsqueeze(0)).item()
    print(f"  {k:25s}: cos={cos:+.4f}", flush=True)

# ═══ Naive Task Arithmetic: W = W_base + τ_A + τ_B ═══
print(f"\n--- Merge Methods ---", flush=True)

def build_ta_merge(sd_base, sdA, sdB, lamA=1.0, lamB=1.0, scA=1.0, scB=1.0):
    """Task arithmetic merge into 4-class model."""
    sd = {}
    for k in sd_base:
        if 'fc' in k or 'num_batches_tracked' in k: continue
        tA = sdA[k] - sd_base[k]
        tB = sdB[k] - sd_base[k]
        sd[k] = sd_base[k] + lamA*tA + lamB*tB
    for k in sd_base:
        if 'num_batches_tracked' in k: sd[k] = torch.tensor(0)
    # FC: direct map
    wA, bA = sdA['fc.weight'], sdA['fc.bias']
    wB, bB = sdB['fc.weight'], sdB['fc.bias']
    fw = torch.zeros(4, 16); fb = torch.zeros(4)
    for ci,c in enumerate(clA): fw[c]=wA[ci]*scA; fb[c]=bA[ci]*scA
    for ci,c in enumerate(clB): fw[c]=wB[ci]*scB; fb[c]=bB[ci]*scB
    sd['fc.weight']=fw; sd['fc.bias']=fb
    return sd

def reset_bn_noise(model, n=512):
    """BN reset with random noise — DATA FREE."""
    model.train()
    noise = torch.randn(n, 3, 32, 32)
    with torch.no_grad():
        for i in range(0, n, 64): model(noise[i:i+64])
    model.eval()
    return model

def reset_bn_real(model, X, bs=256):
    """BN reset with REAL data (for comparison)."""
    model.train()
    with torch.no_grad():
        for i in range(0, len(X), bs): model(X[i:i+bs])
    model.eval()
    return model

def eval_merged(model, name, parent_pc):
    mask = sum(y_te==c for c in ALL).bool()
    Xt = X_te[mask]; yt = y_te[mask]
    with torch.no_grad():
        preds = model(Xt).argmax(1)
    pc = {c:(preds[yt==c]==c).float().mean().item() for c in ALL}
    # Drop% table
    print(f"\n  {name}:")
    print(f"  {'Class':>5} | {'Parent':>8} | {'Merged':>8} | {'Drop%':>7} | Retained?")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}")
    retained = 0
    for c in ALL:
        p = parent_pc[c]; m = pc[c]
        drop = (1-m/p)*100 if p>0 else 100
        ok = '✅' if drop<=10 else '❌'
        if drop<=10: retained+=1
        print(f"  {c:>5} | {p:>8.3f} | {m:>8.3f} | {drop:>6.1f}% | {ok}")
    print(f"  Retention: {retained}/{len(ALL)} (drop ≤ 10%)")
    return {'name':name,'retained':retained,'pc':pc}

# Build parent_pc dict
parent_pc = {}
parent_pc.update(pcA); parent_pc.update(pcB)
print(f"  Parent per-class: {parent_pc}")

# Method 1: Naive TA (λ=1), no BN reset
m1 = TinyResNet(4); m1.load_state_dict(build_ta_merge(sd_base, sdA, sdB)); m1.eval()
r1 = eval_merged(m1, "TaskArith λ=1 (no BN)", parent_pc)

# Method 2: Naive TA, BN noise reset
m2 = TinyResNet(4); m2.load_state_dict(build_ta_merge(sd_base, sdA, sdB))
reset_bn_noise(m2)
r2 = eval_merged(m2, "TaskArith λ=1 + BN noise", parent_pc)

# Method 3: TA, BN real data reset (for comparison)
m3 = TinyResNet(4); m3.load_state_dict(build_ta_merge(sd_base, sdA, sdB))
reset_bn_real(m3, X_tr[:5000])
r3 = eval_merged(m3, "TaskArith λ=1 + BN real", parent_pc)

# Method 4: TA with λ=0.5
m4 = TinyResNet(4); m4.load_state_dict(build_ta_merge(sd_base, sdA, sdB, 0.5, 0.5))
reset_bn_noise(m4)
r4 = eval_merged(m4, "TaskArith λ=0.5 + BN noise", parent_pc)

elapsed = time.time()-t0
print(f"\nTotal: {elapsed:.1f}s")
print(f"\nmetric_ta1_retained: {r1['retained']}")
print(f"metric_ta1_bn_noise_retained: {r2['retained']}")
print(f"metric_ta1_bn_real_retained: {r3['retained']}")
print(f"metric_ta05_bn_noise_retained: {r4['retained']}")
print("Done!", flush=True)
