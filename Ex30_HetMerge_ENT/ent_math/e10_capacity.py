#!/usr/bin/env python3
"""E04: Capacity test — wider TinyResNet (32ch) + jump to ResNet-18 KD.
Hypothesis: 16ch model can't hold both parents. More capacity → both retained.
Also: ResNet-18 student directly (real target scenario).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, time
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import torchvision.models as models
from torchvision import datasets

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

X_transfer = X_te  # public

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
    def __init__(self, nc=2, ch=16):
        super().__init__()
        self.conv1 = nn.Conv2d(3,ch,3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.block1 = ResBlock(ch)
        self.block2 = ResBlock(ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch, nc)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x); x = self.block2(x)
        return self.fc(self.pool(x).flatten(1))

def train_parent(cls_list, seed_p, ch=16, epochs=15):
    torch.manual_seed(seed_p); np.random.seed(seed_p); random.seed(seed_p)
    m = TinyResNet(len(cls_list), ch=ch)
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

def kd_loss(s, t, T):
    return F.kl_div(F.log_softmax(s/T,1), F.softmax(t/T,1), reduction='batchmean')*(T*T)

def eval_student(student, name, parent_pc, classes):
    mask = sum(y_te==c for c in classes).bool()
    Xt = X_te[mask]; yt = y_te[mask]
    with torch.no_grad():
        preds = student(Xt).argmax(1)
    pc = {c:(preds[yt==c]==c).float().mean().item() for c in classes}
    print(f"\n  {name}:")
    print(f"  {'Class':>5} | {'Parent':>8} | {'Merged':>8} | {'Drop%':>7} | Retained?")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}")
    retained = 0
    for c in classes:
        p = parent_pc[c]; m_ = pc[c]
        drop = (1-m_/p)*100 if p>0 else 100
        ok = '✅' if drop<=10 else '❌'
        if drop<=10: retained+=1
        print(f"  {c:>5} | {p:>8.3f} | {m_:>8.3f} | {drop:>6.1f}% | {ok}")
    print(f"  Retention: {retained}/{len(classes)} (drop ≤ 10%)")
    return retained

# ═══ Part A: Toy 32ch ═══
print("=== TOY 32ch ===", flush=True)
mA32, pcA32 = train_parent(clA, SEED, ch=32)
mB32, pcB32 = train_parent(clB, SEED+100, ch=32)
ppc32 = {}; ppc32.update(pcA32); ppc32.update(pcB32)
print(f"  Parents 32ch: A={pcA32} B={pcB32}", flush=True)

mA32.eval(); mB32.eval()
with torch.no_grad():
    tA32 = torch.cat([mA32(X_transfer[i:i+512]) for i in range(0,len(X_transfer),512)])
    tB32 = torch.cat([mB32(X_transfer[i:i+512]) for i in range(0,len(X_transfer),512)])

for w_B in [1.0, 1.5, 2.0]:
    torch.manual_seed(SEED)
    student = TinyResNet(4, ch=32).to(DEVICE)
    opt = torch.optim.Adam(student.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    student.train()
    T = 4.0
    for ep in range(20):
        perm = torch.randperm(len(X_transfer))
        for i in range(0, len(X_transfer), 128):
            ix = perm[i:i+128]
            xb = X_transfer[ix].to(DEVICE)
            s = student(xb)
            loss = kd_loss(s[:,:2], tA32[ix].to(DEVICE), T) + \
                   w_B * kd_loss(s[:,2:], tB32[ix].to(DEVICE), T)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    student.to('cpu').eval()
    r = eval_student(student, f"32ch KD w_B={w_B}", ppc32, ALL)

# ═══ Part B: ResNet-18 student, ResNet-18 parents (real target) ═══
print(f"\n\n=== RESNET-18 KD (from saved parents) ===", flush=True)
print(f"  ({time.time()-t0:.0f}s elapsed)", flush=True)

def make_rn(nc=5):
    m = models.resnet18(weights='IMAGENET1K_V1')
    m.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    return m

# Load saved parents
import json
clA10, clB10 = list(range(5)), list(range(5,10))
ALL10 = list(range(10))

pA = make_rn(5)
pA.load_state_dict(torch.load('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/cifar_fix/results/parentA_s42.pth',
                               weights_only=True, map_location='cpu'))
pB = make_rn(5)
pB.load_state_dict(torch.load('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/cifar_fix/results/parentB_s42.pth',
                               weights_only=True, map_location='cpu'))

with open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/cifar_fix/results/parents_strong.json') as f:
    pd = json.load(f)['42']
ppc10 = {}
for c,a in pd['pcA'].items(): ppc10[int(c)] = a
for c,a in pd['pcB'].items(): ppc10[int(c)] = a
print(f"  Parents: A={pd['A']:.3f} B={pd['B']:.3f}", flush=True)

# Compute teacher logits on transfer data (use first 5000 for speed)
X_kd10 = X_te[:5000]
pA.eval(); pB.eval()
with torch.no_grad():
    tA10 = torch.cat([pA(X_kd10[i:i+256]) for i in range(0,len(X_kd10),256)])  # [5k, 5]
    tB10 = torch.cat([pB(X_kd10[i:i+256]) for i in range(0,len(X_kd10),256)])  # [5k, 5]
print(f"  Teacher logits computed ({time.time()-t0:.0f}s)", flush=True)

# Student = ResNet-18(10 classes) init from pretrained
student10 = make_rn(10).to(DEVICE)
opt = torch.optim.Adam(student10.parameters(), lr=0.005)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
T = 4.0; w_B = 1.5
student10.train()
for ep in range(10):
    perm = torch.randperm(len(X_kd10))
    for i in range(0, len(X_kd10), 64):
        ix = perm[i:i+64]
        xb = X_kd10[ix].to(DEVICE)
        s = student10(xb)
        # Student[0:5] ← parentA, Student[5:10] ← parentB
        loss = kd_loss(s[:,:5], tA10[ix].to(DEVICE), T) + \
               w_B * kd_loss(s[:,5:], tB10[ix].to(DEVICE), T)
        opt.zero_grad(); loss.backward(); opt.step()
    sch.step()
    if (ep+1) % 5 == 0:
        print(f"    Epoch {ep+1}/10 ({time.time()-t0:.0f}s)", flush=True)

student10.to('cpu').eval()
r10 = eval_student(student10, "ResNet-18 KD w_B=1.5 T=4", ppc10, ALL10)

elapsed = time.time()-t0
print(f"\nTotal: {elapsed:.1f}s")
print(f"\nmetric_rn18_retained: {r10}")
print("Done!", flush=True)
