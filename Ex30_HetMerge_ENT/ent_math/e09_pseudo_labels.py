#!/usr/bin/env python3
"""E03: Pseudo-label training — parents generate hard targets for student.
Key insight: instead of soft KD, use HARD pseudo-labels.
Each input image gets TWO predictions: parentA→cls∈{0,1}, parentB→cls∈{2,3}.
Student learns via weighted CE loss from both pseudo-label signals.

Also tries: confidence-weighted training (only use high-confidence predictions).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, time
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
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

def train_parent(cls_list, seed_p, epochs=15):
    torch.manual_seed(seed_p); np.random.seed(seed_p); random.seed(seed_p)
    m = TinyResNet(len(cls_list))
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

mA, pcA = train_parent(clA, SEED)
mB, pcB = train_parent(clB, SEED+100)
parent_pc = {}; parent_pc.update(pcA); parent_pc.update(pcB)

# ═══ Generate pseudo-labels ═══
X_transfer = X_te  # public images
mA.eval(); mB.eval()
with torch.no_grad():
    logA = torch.cat([mA(X_transfer[i:i+512]) for i in range(0,len(X_transfer),512)])
    logB = torch.cat([mB(X_transfer[i:i+512]) for i in range(0,len(X_transfer),512)])
    confA = F.softmax(logA, dim=1).max(1)[0]  # confidence
    confB = F.softmax(logB, dim=1).max(1)[0]
    predA = logA.argmax(1)  # 0 or 1 (maps to class 0 or 1)
    predB = logB.argmax(1) + 2  # 0 or 1 → 2 or 3

print(f"Setup ({time.time()-t0:.1f}s): A={pcA} B={pcB}", flush=True)
print(f"  A confidence: mean={confA.mean():.3f} min={confA.min():.3f}")
print(f"  B confidence: mean={confB.mean():.3f} min={confB.min():.3f}")

def eval_student(student, name):
    mask = sum(y_te==c for c in ALL).bool()
    Xt = X_te[mask]; yt = y_te[mask]
    with torch.no_grad():
        preds = student(Xt).argmax(1)
    pc = {c:(preds[yt==c]==c).float().mean().item() for c in ALL}
    print(f"\n  {name}:")
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
    return retained, pc

# ═══ Method 1: Multi-task CE with BOTH pseudo-labels ═══
def train_multitask(w_B=1.0, lr=0.01, epochs=20, conf_threshold=0.0):
    torch.manual_seed(SEED)
    student = TinyResNet(4).to(DEVICE)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    student.train()
    for ep in range(epochs):
        perm = torch.randperm(len(X_transfer))
        for i in range(0, len(X_transfer), 128):
            ix = perm[i:i+128]
            xb = X_transfer[ix].to(DEVICE)
            s_logits = student(xb)  # [B, 4]
            
            # Loss A: student should predict predA for these images
            lossA = F.cross_entropy(s_logits, predA[ix].to(DEVICE))
            
            # Loss B: student should predict predB for these images
            lossB = F.cross_entropy(s_logits, predB[ix].to(DEVICE))
            
            loss = lossA + w_B * lossB
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    student.to('cpu').eval()
    return student

# ═══ Method 2: Alternating training ═══
def train_alternating(w_B=1.0, lr=0.01, epochs=20):
    torch.manual_seed(SEED)
    student = TinyResNet(4).to(DEVICE)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    
    student.train()
    for ep in range(epochs):
        perm = torch.randperm(len(X_transfer))
        for i in range(0, len(X_transfer), 128):
            ix = perm[i:i+128]
            xb = X_transfer[ix].to(DEVICE)
            s_logits = student(xb)
            
            # Alternate: even steps → A, odd steps → B
            if (i//128) % 2 == 0:
                loss = F.cross_entropy(s_logits, predA[ix].to(DEVICE))
            else:
                loss = w_B * F.cross_entropy(s_logits, predB[ix].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    student.to('cpu').eval()
    return student

# ═══ Method 3: Soft KD + hard CE combined ═══
def train_hybrid(w_B=2.0, T=4.0, lr=0.01, epochs=20):
    torch.manual_seed(SEED)
    student = TinyResNet(4).to(DEVICE)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    
    student.train()
    for ep in range(epochs):
        perm = torch.randperm(len(X_transfer))
        for i in range(0, len(X_transfer), 128):
            ix = perm[i:i+128]
            xb = X_transfer[ix].to(DEVICE)
            s_logits = student(xb)
            
            # Soft KD
            s_soft = F.log_softmax(s_logits[:,:2]/T, dim=1)
            t_soft_A = F.softmax(logA[ix].to(DEVICE)/T, dim=1)
            kd_A = F.kl_div(s_soft, t_soft_A, reduction='batchmean')*(T*T)
            
            s_soft_B = F.log_softmax(s_logits[:,2:]/T, dim=1)
            t_soft_B = F.softmax(logB[ix].to(DEVICE)/T, dim=1)
            kd_B = F.kl_div(s_soft_B, t_soft_B, reduction='batchmean')*(T*T)
            
            # Hard pseudo-label CE
            ce_A = F.cross_entropy(s_logits, predA[ix].to(DEVICE))
            ce_B = F.cross_entropy(s_logits, predB[ix].to(DEVICE))
            
            loss = kd_A + w_B*kd_B + 0.5*(ce_A + w_B*ce_B)
            opt.zero_grad(); loss.backward(); opt.step()
    student.to('cpu').eval()
    return student

# ═══ Run all methods ═══
print(f"\n--- Method 1: Multi-task CE (w_B=1) ---", flush=True)
s1 = train_multitask(w_B=1.0, lr=0.01, epochs=20)
r1,_ = eval_student(s1, "Multi-task CE w_B=1")

print(f"\n--- Method 1b: Multi-task CE (w_B=3) ---", flush=True)
s1b = train_multitask(w_B=3.0, lr=0.01, epochs=20)
r1b,_ = eval_student(s1b, "Multi-task CE w_B=3")

print(f"\n--- Method 2: Alternating ---", flush=True)
s2 = train_alternating(w_B=2.0, lr=0.01, epochs=20)
r2,_ = eval_student(s2, "Alternating w_B=2")

print(f"\n--- Method 3: Hybrid KD+CE ---", flush=True)
s3 = train_hybrid(w_B=2.0, T=4.0, lr=0.01, epochs=20)
r3,_ = eval_student(s3, "Hybrid KD+CE w_B=2")

elapsed = time.time()-t0
best = max(r1, r1b, r2, r3)
print(f"\nTotal: {elapsed:.1f}s, best={best}/4")
print(f"\nmetric_mt_wB1: {r1}")
print(f"metric_mt_wB3: {r1b}")
print(f"metric_alt: {r2}")
print(f"metric_hybrid: {r3}")
print(f"metric_best: {best}")
print("Done!", flush=True)
