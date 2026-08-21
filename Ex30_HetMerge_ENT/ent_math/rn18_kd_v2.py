#!/usr/bin/env python3
"""E09: ResNet-18 KD — full progressive with pre-computed teacher logits.

Teacher logits loaded from disk (12s MPS, not 95s CPU).
Budget: 600s → ~580s for training.
Progressive: 10ep A → 30ep A+B (w_B=2, cosine LR).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
import random, time, json
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
t0 = time.time()
MAX_TIME = 570

raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mn = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
sd = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255-mn)/sd
y_te = torch.tensor(raw_te.targets)
ALL = list(range(10))

# Load pre-computed teacher logits
tl = torch.load('results/teacher_logits.pth', weights_only=True)
tA, tB = tl['tA'], tl['tB']

with open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/cifar_fix/results/parents_strong.json') as f:
    pd = json.load(f)['42']
parent_pc = {}
for c,a in pd['pcA'].items(): parent_pc[int(c)] = a
for c,a in pd['pcB'].items(): parent_pc[int(c)] = a
print(f"Setup ({time.time()-t0:.1f}s): A={pd['A']:.3f} B={pd['B']:.3f}", flush=True)

def make_rn(nc=10):
    m = models.resnet18(weights='IMAGENET1K_V1')
    m.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    return m

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

def quick_eval(student, label=""):
    student.to('cpu').eval()
    with torch.no_grad():
        preds = torch.cat([student(X_te[i:i+256]).argmax(1) for i in range(0,len(X_te),256)])
    mpc = {c:(preds[y_te==c]==c).float().mean().item() for c in ALL}
    ret = sum(1 for c in ALL if (1-mpc[c]/parent_pc[c])*100 <= 10)
    drops = [(1-mpc[c]/parent_pc[c])*100 for c in ALL]
    if label:
        print(f"  {label}: ret={ret}/10 max_drop={max(drops):.1f}% ({time.time()-t0:.0f}s)", flush=True)
    student.to(DEVICE).train()
    return ret, mpc

# ═══ Student ═══
torch.manual_seed(SEED)
student = make_rn(10).to(DEVICE)
BS = 64
N = len(X_te)

# ═══ Phase 1: A-only (10 ep) ═══
print(f"\n--- Phase 1: A-only (10 ep) ---", flush=True)
opt1 = torch.optim.Adam(student.parameters(), lr=0.003)
sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=10)
student.train()
for ep in range(10):
    if time.time()-t0 > MAX_TIME: print(f"  TIME at ep {ep}", flush=True); break
    pm = torch.randperm(N)
    for i in range(0, N, BS):
        ix = pm[i:i+BS]
        s = student(X_te[ix].to(DEVICE))
        loss = kd_loss(s[:,:5], tA[ix].to(DEVICE), T=4.0)
        opt1.zero_grad(); loss.backward(); opt1.step()
    sch1.step()
    if (ep+1)%5==0: quick_eval(student, f"Phase1 ep {ep+1}")

# ═══ Phase 2: A+B (30 ep, w_B=2) ═══
print(f"\n--- Phase 2: A+B (30 ep, w_B=2) ---", flush=True)
opt2 = torch.optim.Adam(student.parameters(), lr=0.001)
sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=30)
student.to(DEVICE).train()
for ep in range(30):
    if time.time()-t0 > MAX_TIME: print(f"  TIME at ep {ep}", flush=True); break
    pm = torch.randperm(N)
    for i in range(0, N, BS):
        ix = pm[i:i+BS]
        s = student(X_te[ix].to(DEVICE))
        lA = kd_loss(s[:,:5], tA[ix].to(DEVICE), T=4.0)
        lB = kd_loss(s[:,5:], tB[ix].to(DEVICE), T=4.0)
        loss = 0.5*lA + 2.0*lB
        opt2.zero_grad(); loss.backward(); opt2.step()
    sch2.step()
    if (ep+1)%5==0: quick_eval(student, f"Phase2 ep {ep+1}")

# ═══ Final eval ═══
student.to('cpu').eval()
with torch.no_grad():
    preds_f = torch.cat([student(X_te[i:i+256]).argmax(1) for i in range(0,len(X_te),256)])
mpc_f = {c:(preds_f[y_te==c]==c).float().mean().item() for c in ALL}
r_final = print_results("Final: Progressive KD 10+30ep (precomp)", parent_pc, mpc_f, ALL)

elapsed = time.time()-t0
print(f"\nTotal: {elapsed:.1f}s")
print(f"\nmetric_rn18_retained: {r_final}")
print(f"metric_rn18_mean_drop: {np.mean([(1-mpc_f[c]/parent_pc[c])*100 for c in ALL]):.1f}")
print("Done!", flush=True)
