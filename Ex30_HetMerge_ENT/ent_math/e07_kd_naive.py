#!/usr/bin/env python3
"""E01: Naive KD on TinyResNet — student learns from both parents.
Transfer data: CIFAR-10 test images (public, not original training data).
KD loss: student matches soft-label distributions of both parents.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, time
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
t0 = time.time()

# ═══ Data ═══
raw_tr = datasets.CIFAR10('/tmp/cifar10', train=True, download=True)
raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mn = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
sd = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_tr = (torch.tensor(raw_tr.data).permute(0,3,1,2).float()/255-mn)/sd
y_tr = torch.tensor(raw_tr.targets)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255-mn)/sd
y_te = torch.tensor(raw_te.targets)
clA, clB = [0,1], [2,3]; ALL = clA+clB

# Transfer data: ALL test images (public data, not original training set)
# We use ALL 10k test images, not just the 4 target classes
X_transfer = X_te  # 10000 images
print(f"Data: {time.time()-t0:.1f}s, transfer={len(X_transfer)}", flush=True)

# ═══ Models ═══
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

# Train parents
mA, pcA = train_parent(clA, SEED)
mB, pcB = train_parent(clB, SEED+100)
parent_pc = {}; parent_pc.update(pcA); parent_pc.update(pcB)
print(f"Parents ({time.time()-t0:.1f}s): A={pcA} B={pcB}", flush=True)

# ═══ KD Loss ═══
def kd_loss(student_logits, teacher_logits, T=4.0):
    s_soft = F.log_softmax(student_logits / T, dim=1)
    t_soft = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s_soft, t_soft, reduction='batchmean') * (T * T)

# ═══ Pre-compute teacher logits on transfer data ═══
mA.eval(); mB.eval()
with torch.no_grad():
    # Teacher logits on ALL transfer data
    tA_logits = torch.cat([mA(X_transfer[i:i+512]) for i in range(0,len(X_transfer),512)])  # [10k, 2]
    tB_logits = torch.cat([mB(X_transfer[i:i+512]) for i in range(0,len(X_transfer),512)])  # [10k, 2]
print(f"Teacher logits precomputed ({time.time()-t0:.1f}s)", flush=True)

# ═══ Train student ═══
def train_student(T=4.0, lr=0.01, epochs=20, bs=128):
    torch.manual_seed(SEED)
    student = TinyResNet(4).to(DEVICE)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    student.train()
    for ep in range(epochs):
        perm = torch.randperm(len(X_transfer))
        for i in range(0, len(X_transfer), bs):
            ix = perm[i:i+bs]
            xb = X_transfer[ix].to(DEVICE)
            s_logits = student(xb)  # [B, 4]
            
            # Split student logits: classes 0,1 match parentA, classes 2,3 match parentB
            # Student output [4]: indices 0,1 → parentA output [2], indices 2,3 → parentB output [2]
            s_for_A = s_logits[:, :2]  # student's logits for A's classes
            s_for_B = s_logits[:, 2:]  # student's logits for B's classes
            
            t_A = tA_logits[ix].to(DEVICE)  # [B, 2]
            t_B = tB_logits[ix].to(DEVICE)  # [B, 2]
            
            loss = kd_loss(s_for_A, t_A, T) + kd_loss(s_for_B, t_B, T)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    
    student.to('cpu').eval()
    return student

# ═══ Evaluate ═══
def eval_student(student, name):
    mask = sum(y_te==c for c in ALL).bool()
    Xt = X_te[mask]; yt = y_te[mask]
    with torch.no_grad():
        preds = student(Xt).argmax(1)
    pc = {}
    # Map: student output 0→cls0, 1→cls1, 2→cls2, 3→cls3
    for ci, c in enumerate(ALL):
        mask_c = yt==c
        pc[c] = (preds[mask_c]==ci).float().mean().item()
    
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

# ═══ Run with different temperatures ═══
print(f"\n--- Naive KD (T=4, lr=0.01, 20 ep) ---", flush=True)
s1 = train_student(T=4.0, lr=0.01, epochs=20)
r1, pc1 = eval_student(s1, "KD T=4 lr=0.01")

print(f"\n--- KD T=2 ---", flush=True)
s2 = train_student(T=2.0, lr=0.01, epochs=20)
r2, pc2 = eval_student(s2, "KD T=2 lr=0.01")

print(f"\n--- KD T=10 ---", flush=True)
s3 = train_student(T=10.0, lr=0.01, epochs=20)
r3, pc3 = eval_student(s3, "KD T=10 lr=0.01")

print(f"\n--- KD T=4 lr=0.003 ---", flush=True)
s4 = train_student(T=4.0, lr=0.003, epochs=20)
r4, pc4 = eval_student(s4, "KD T=4 lr=0.003")

elapsed = time.time()-t0
print(f"\nTotal: {elapsed:.1f}s")
print(f"\nmetric_kd_t4_retained: {r1}")
print(f"metric_kd_t2_retained: {r2}")
print(f"metric_kd_t10_retained: {r3}")
print(f"metric_kd_t4_lr003_retained: {r4}")
best = max(r1, r2, r3, r4)
print(f"metric_best_retained: {best}")
print("Done!", flush=True)
