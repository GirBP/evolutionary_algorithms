#!/usr/bin/env python3
"""H2: Full KD + Progressive + multiple variants.

Last chance — full budget KD. Progressive from H4 worked for A (2/4).
Now try: more epochs, better curricula, EWC regularization.

Also: test if using LABELED public data (supervised training, not KD)
gives ceiling reference — what's achievable with the same data?
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, time
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
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
X_pub = X_tr[:5000]; y_pub = y_tr[:5000]

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

data = torch.load('results/toy_parents.pth', weights_only=False)
mA = TinyResNet(2); mA.load_state_dict(data['sdA']); mA.eval()
mB = TinyResNet(2); mB.load_state_dict(data['sdB']); mB.eval()
pcA, pcB = data['pcA'], data['pcB']
parent_pc = {}; parent_pc.update(pcA); parent_pc.update(pcB)

mA.eval(); mB.eval()
with torch.no_grad():
    tA = torch.cat([mA(X_pub[i:i+512]) for i in range(0,len(X_pub),512)])
    tB = torch.cat([mB(X_pub[i:i+512]) for i in range(0,len(X_pub),512)])
print(f"Parents ({time.time()-t0:.1f}s): A={pcA} B={pcB}", flush=True)

def kd_loss(s, t, T=4.0):
    return F.kl_div(F.log_softmax(s/T,1), F.softmax(t/T,1), reduction='batchmean')*(T*T)

def print_results(name, ppc, mpc, cls):
    print(f"\n  {name}:")
    print(f"  {'Class':>5} | {'Parent':>8} | {'Merged':>8} | {'Drop%':>7} | Retained?")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}")
    ret = 0
    for c in cls:
        p = ppc[c]; m = mpc[c]
        drop = (1-m/p)*100 if p>0 else 100
        ok = '✅' if drop<=10 else '❌'
        if drop<=10: ret+=1
        print(f"  {c:>5} | {p:>8.3f} | {m:>8.3f} | {drop:>6.1f}% | {ok}")
    print(f"  Retention: {ret}/{len(cls)} (drop ≤ 10%)")
    return ret

def eval_student(student, name):
    mask = sum(y_te==c for c in ALL).bool()
    Xt = X_te[mask]; yt = y_te[mask]
    with torch.no_grad():
        preds = student(Xt).argmax(1)
    mpc = {c:(preds[yt==c]==c).float().mean().item() for c in ALL}
    return print_results(name, parent_pc, mpc, ALL)

# ═══ Method 1: Progressive KD, 20+50 epochs, class-selective ═══
print("\n--- Progressive KD: 20+50 ep, class-selective ---", flush=True)
torch.manual_seed(SEED)
s1 = TinyResNet(4).to(DEVICE)
opt = torch.optim.Adam(s1.parameters(), lr=0.01)

# Phase 1: A only (20 ep)
s1.train()
for ep in range(20):
    pm = torch.randperm(len(X_pub))
    for i in range(0, len(X_pub), 128):
        ix = pm[i:i+128]
        loss = kd_loss(s1(X_pub[ix].to(DEVICE))[:,:2], tA[ix].to(DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()

# Phase 2: A+B, w_B=3, lower lr (50 ep)
opt2 = torch.optim.Adam(s1.parameters(), lr=0.001)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=50)
for ep in range(50):
    pm = torch.randperm(len(X_pub))
    for i in range(0, len(X_pub), 128):
        ix = pm[i:i+128]
        s = s1(X_pub[ix].to(DEVICE))
        lossA = kd_loss(s[:,:2], tA[ix].to(DEVICE))
        lossB = kd_loss(s[:,2:], tB[ix].to(DEVICE))
        loss = 0.5*lossA + 3.0*lossB
        opt2.zero_grad(); loss.backward(); opt2.step()
    sch.step()

s1.to('cpu').eval()
r1 = eval_student(s1, "Progressive 20+50, w_B=3")

# ═══ Method 2: Supervised ceiling (public labeled data) ═══
print(f"\n--- Supervised ceiling (real labels, 4 classes) ---", flush=True)
mask_pub = sum(y_pub==c for c in ALL).bool()
Xs = X_pub[mask_pub]; ys = y_pub[mask_pub]
print(f"  Labeled data: {len(Xs)} images", flush=True)

torch.manual_seed(SEED)
s2 = TinyResNet(4).to(DEVICE)
opt3 = torch.optim.Adam(s2.parameters(), lr=0.003)
s2.train()
for ep in range(50):
    pm = torch.randperm(len(Xs))
    for i in range(0, len(Xs), 128):
        ix = pm[i:i+128]
        loss = F.cross_entropy(s2(Xs[ix].to(DEVICE)), ys[ix].to(DEVICE))
        opt3.zero_grad(); loss.backward(); opt3.step()

s2.to('cpu').eval()
r2 = eval_student(s2, "Supervised ceiling (real labels)")

# ═══ Method 3: KD simultaneous, T=10 (softer), 70 epochs ═══
print(f"\n--- Simultaneous KD: T=10, 70ep, w_B=2 ---", flush=True)
torch.manual_seed(SEED)
s3 = TinyResNet(4).to(DEVICE)
opt4 = torch.optim.Adam(s3.parameters(), lr=0.005)
sch4 = torch.optim.lr_scheduler.CosineAnnealingLR(opt4, T_max=70)
s3.train()
for ep in range(70):
    pm = torch.randperm(len(X_pub))
    for i in range(0, len(X_pub), 128):
        ix = pm[i:i+128]
        s = s3(X_pub[ix].to(DEVICE))
        loss = kd_loss(s[:,:2], tA[ix].to(DEVICE), T=10) + \
               2.0*kd_loss(s[:,2:], tB[ix].to(DEVICE), T=10)
        opt4.zero_grad(); loss.backward(); opt4.step()
    sch4.step()

s3.to('cpu').eval()
r3 = eval_student(s3, "Simultaneous KD T=10 70ep w_B=2")

best = max(r1, r2, r3)
elapsed = time.time()-t0
print(f"\nTotal: {elapsed:.1f}s")
print(f"\nmetric_h2_prog_retained: {r1}")
print(f"metric_h2_supervised: {r2}")
print(f"metric_h2_simult_retained: {r3}")
print(f"metric_h2_best: {best}")
print("Done!", flush=True)
