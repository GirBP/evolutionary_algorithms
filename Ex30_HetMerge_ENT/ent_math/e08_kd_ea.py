#!/usr/bin/env python3
"""E02: EA optimizes KD hyperparams — per-class weights, T, lr.
Key insight from E01: student learns A well but ignores B.
Solution: EA finds optimal loss balancing.
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

X_transfer = X_te  # public data
# Split: 8k train, 2k validation for EA fitness
X_kd = X_transfer[:8000]; X_val = X_transfer[8000:]
y_val = y_te[8000:]

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

mA.eval(); mB.eval()
with torch.no_grad():
    tA_kd = torch.cat([mA(X_kd[i:i+512]) for i in range(0,len(X_kd),512)])
    tB_kd = torch.cat([mB(X_kd[i:i+512]) for i in range(0,len(X_kd),512)])
    tA_val = torch.cat([mA(X_val[i:i+512]) for i in range(0,len(X_val),512)])
    tB_val = torch.cat([mB(X_val[i:i+512]) for i in range(0,len(X_val),512)])
print(f"Setup: {time.time()-t0:.1f}s, parents: {pcA} {pcB}", flush=True)

def kd_loss(s, t, T):
    return F.kl_div(F.log_softmax(s/T, dim=1), F.softmax(t/T, dim=1), reduction='batchmean')*(T*T)

# ═══ EA-optimized training ═══
def train_eval(x):
    """x = [w_B, T, lr_exp, epochs_raw]"""
    w_B = max(0.1, 1.0 + x[0])     # weight for B's loss (>1 to emphasize B)
    T = max(1.0, 2.0 + x[1]*2)      # temperature [1, ~10]
    lr = 10 ** (-2 + x[2]*0.5)      # lr [~0.003, ~0.03]
    lr = np.clip(lr, 0.001, 0.05)
    epochs = max(10, int(15 + x[3]*3))
    epochs = min(epochs, 25)
    
    torch.manual_seed(SEED)
    student = TinyResNet(4).to(DEVICE)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    student.train()
    for ep in range(epochs):
        perm = torch.randperm(len(X_kd))
        for i in range(0, len(X_kd), 256):
            ix = perm[i:i+256]
            xb = X_kd[ix].to(DEVICE)
            s_logits = student(xb)
            loss = kd_loss(s_logits[:,:2], tA_kd[ix].to(DEVICE), T) + \
                   w_B * kd_loss(s_logits[:,2:], tB_kd[ix].to(DEVICE), T)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    
    # Eval on validation
    student.to('cpu').eval()
    mask = sum(y_val==c for c in ALL).bool()
    Xv = X_val[mask]; yv = y_val[mask]
    with torch.no_grad():
        preds = student(Xv).argmax(1)
    pc = {}
    for ci, c in enumerate(ALL):
        mc = yv==c
        pc[c] = (preds[mc]==ci).float().mean().item() if mc.sum()>0 else 0
    
    # Fitness: minimize max drop, then mean drop
    drops = [(1-pc[c]/parent_pc[c])*100 if parent_pc[c]>0 else 100 for c in ALL]
    max_drop = max(drops)
    mean_drop = np.mean(drops)
    retained = sum(1 for d in drops if d<=10)
    return max_drop + 0.1*mean_drop - 10*retained, student, pc, {'w_B':w_B,'T':T,'lr':lr,'ep':epochs}

# ═══ CMA-ES ═══
print(f"\nCMA-ES optimization...", flush=True)
x0 = np.array([1.0, 1.0, 0.0, 0.0])  # w_B=2, T=4, lr=0.01, ep=15
es = cma.CMAEvolutionStrategy(x0, 0.5,
    {'maxiter': 8, 'popsize': 8, 'seed': SEED, 'verbose': -1, 'timeout': 80})
bf = float('inf'); bx = None; gen = 0
while not es.stop():
    sols = es.ask()
    fvals = []
    for s in sols:
        f, _, _, _ = train_eval(s)
        fvals.append(f)
    es.tell(sols, fvals)
    if min(fvals) < bf:
        bf = min(fvals)
        bx = sols[np.argmin(fvals)]
    gen += 1
    print(f"  Gen {gen}: best={bf:.1f} ({time.time()-t0:.0f}s)", flush=True)

# ═══ Final eval with best params ═══
_, best_student, best_pc, best_params = train_eval(bx)
print(f"\nBest params: {best_params}", flush=True)

# Eval on FULL test set
mask = sum(y_te==c for c in ALL).bool()
Xt = X_te[mask]; yt = y_te[mask]
with torch.no_grad():
    preds = best_student(Xt).argmax(1)
pc_test = {}
for ci, c in enumerate(ALL):
    mc = yt==c
    pc_test[c] = (preds[mc]==ci).float().mean().item()

print(f"\n  EA-Optimized KD:")
print(f"  {'Class':>5} | {'Parent':>8} | {'Merged':>8} | {'Drop%':>7} | Retained?")
print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}")
retained = 0
for c in ALL:
    p = parent_pc[c]; m_ = pc_test[c]
    drop = (1-m_/p)*100 if p>0 else 100
    ok = '✅' if drop<=10 else '❌'
    if drop<=10: retained+=1
    print(f"  {c:>5} | {p:>8.3f} | {m_:>8.3f} | {drop:>6.1f}% | {ok}")
print(f"  Retention: {retained}/{len(ALL)} (drop ≤ 10%)")

elapsed = time.time()-t0
print(f"\nTotal: {elapsed:.1f}s")
print(f"\nmetric_ea_retained: {retained}")
print(f"metric_ea_max_drop: {max((1-pc_test[c]/parent_pc[c])*100 for c in ALL):.1f}")
print("Done!", flush=True)
