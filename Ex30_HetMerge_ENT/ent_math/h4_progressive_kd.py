#!/usr/bin/env python3
"""H4 Kill Test: Progressive Distillation.

Step 1: Train student from scratch on A's KD → verify A retained
Step 2: Freeze conv1+block1, continue training on A+B KD → check all 4

Key insight: seesaw is caused by simultaneous A+B training.
Fix: establish A first, then carefully add B.
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
X_pub = X_tr[:5000]

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
print(f"Parents ({time.time()-t0:.1f}s): A={pcA} B={pcB}", flush=True)

# Pre-compute teacher logits
mA.eval(); mB.eval()
with torch.no_grad():
    tA = torch.cat([mA(X_pub[i:i+512]) for i in range(0,len(X_pub),512)])
    tB = torch.cat([mB(X_pub[i:i+512]) for i in range(0,len(X_pub),512)])

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
    merged_pc = {c:(preds[yt==c]==c).float().mean().item() for c in ALL}
    return print_results(name, parent_pc, merged_pc, ALL)

# ═══ Step 1: Train on A only (20 epochs) ═══
print("\n--- Step 1: KD from Parent A only ---", flush=True)
torch.manual_seed(SEED)
student = TinyResNet(4).to(DEVICE)
optS = torch.optim.Adam(student.parameters(), lr=0.01)

student.train()
for ep in range(20):
    pm = torch.randperm(len(X_pub))
    for i in range(0, len(X_pub), 128):
        ix = pm[i:i+128]
        xb = X_pub[ix].to(DEVICE)
        s_logits = student(xb)
        # Only train on A's classes (0,1)
        loss = kd_loss(s_logits[:,:2], tA[ix].to(DEVICE), T=4.0)
        optS.zero_grad(); loss.backward(); optS.step()

student.to('cpu').eval()
r_step1 = eval_student(student, "Step 1: A-only KD (20ep)")

# ═══ Step 2: Progressive — freeze early, add B ═══
print("\n--- Step 2: Add B (freeze conv1+block1) ---", flush=True)
student.to(DEVICE)

# Freeze conv1 and block1
for name, p in student.named_parameters():
    if 'conv1' in name or 'bn1' in name or 'block1' in name:
        p.requires_grad = False

# Lower LR for fine-tuning
optS2 = torch.optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=0.003)

student.train()
for ep in range(30):
    pm = torch.randperm(len(X_pub))
    for i in range(0, len(X_pub), 128):
        ix = pm[i:i+128]
        xb = X_pub[ix].to(DEVICE)
        s_logits = student(xb)
        # KD from BOTH parents, but weight B more (since A is already learned)
        lA = kd_loss(s_logits[:,:2], tA[ix].to(DEVICE), T=4.0)
        lB = kd_loss(s_logits[:,2:], tB[ix].to(DEVICE), T=4.0)
        loss = 0.3*lA + 1.0*lB  # lower weight on A (maintaining), higher on B (learning)
        optS2.zero_grad(); loss.backward(); optS2.step()

student.to('cpu').eval()
r_step2 = eval_student(student, "Step 2: Progressive (freeze+B KD)")

# ═══ Step 2b: Freeze NOTHING, just lower rate ═══
print("\n--- Step 2b: Add B (no freeze, low lr) ---", flush=True)
# Reload step 1 state
torch.manual_seed(SEED)
student2 = TinyResNet(4).to(DEVICE)
opt2 = torch.optim.Adam(student2.parameters(), lr=0.01)
student2.train()
for ep in range(20):
    pm = torch.randperm(len(X_pub))
    for i in range(0, len(X_pub), 128):
        ix = pm[i:i+128]
        loss = kd_loss(student2(X_pub[ix].to(DEVICE))[:,:2], tA[ix].to(DEVICE))
        opt2.zero_grad(); loss.backward(); opt2.step()

# Now add B with combined loss, very low LR
opt2b = torch.optim.Adam(student2.parameters(), lr=0.001)
student2.train()
for ep in range(30):
    pm = torch.randperm(len(X_pub))
    for i in range(0, len(X_pub), 128):
        ix = pm[i:i+128]
        s = student2(X_pub[ix].to(DEVICE))
        lA = kd_loss(s[:,:2], tA[ix].to(DEVICE))
        lB = kd_loss(s[:,2:], tB[ix].to(DEVICE))
        loss = lA + 2.0*lB
        opt2b.zero_grad(); loss.backward(); opt2b.step()

student2.to('cpu').eval()
r_step2b = eval_student(student2, "Step 2b: Progressive (no freeze, lr=0.001)")

best = max(r_step2, r_step2b)
elapsed = time.time()-t0
print(f"\nTotal: {elapsed:.1f}s")
print(f"\nmetric_h4_step1: {r_step1}")
print(f"metric_h4_step2_freeze: {r_step2}")
print(f"metric_h4_step2b_nofr: {r_step2b}")
print(f"metric_h4_best: {best}")

if best == 0:
    print("\n🔴 KILL H4: retention=0.")
else:
    print(f"\n🟢 H4 PASS: retention={best}/4.")
print("Done!", flush=True)
