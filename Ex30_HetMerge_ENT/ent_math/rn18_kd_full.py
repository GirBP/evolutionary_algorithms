#!/usr/bin/env python3
"""E08: ResNet-18 KD — full budget (600s).

Progressive KD: 20ep A → 50ep A+B.
Transfer: 10k CIFAR-10 test images (public).
Budget: 600s.

Key improvements over E07:
- 10k images (was 3k)
- 70 total epochs (was 15)
- Cosine LR schedule
- w_B=2.0 for balanced learning
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
import random, time, json, sys
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
t0 = time.time()
MAX_TIME = 550  # leave margin for eval

raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mn = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
sd = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255-mn)/sd
y_te = torch.tensor(raw_te.targets)
ALL = list(range(10))
X_pub = X_te  # full 10k

def make_rn(nc=5):
    m = models.resnet18(weights='IMAGENET1K_V1')
    m.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    return m

# Load parents
pA = make_rn(5)
pA.load_state_dict(torch.load('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/cifar_fix/results/parentA_s42.pth',
                               weights_only=True, map_location='cpu'))
pB = make_rn(5)
pB.load_state_dict(torch.load('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/cifar_fix/results/parentB_s42.pth',
                               weights_only=True, map_location='cpu'))

with open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/cifar_fix/results/parents_strong.json') as f:
    pd = json.load(f)['42']
parent_pc = {}
for c,a in pd['pcA'].items(): parent_pc[int(c)] = a
for c,a in pd['pcB'].items(): parent_pc[int(c)] = a
print(f"Parents ({time.time()-t0:.1f}s): A={pd['A']:.3f} B={pd['B']:.3f}", flush=True)

# Pre-compute teacher logits on CPU (save MPS memory for student)
print("Computing teacher logits...", flush=True)
pA.eval(); pB.eval()
with torch.no_grad():
    tA = torch.cat([pA(X_pub[i:i+256]) for i in range(0,len(X_pub),256)])  # [10k, 5]
    tB = torch.cat([pB(X_pub[i:i+256]) for i in range(0,len(X_pub),256)])  # [10k, 5]
del pA, pB  # free memory
print(f"Teacher logits ({time.time()-t0:.1f}s): tA={tA.shape}", flush=True)

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

# ═══ Progressive KD ═══
torch.manual_seed(SEED)
student = make_rn(10).to(DEVICE)

# Phase 1: A-only (20 epochs)
print(f"\n--- Phase 1: A-only KD (20 ep) ---", flush=True)
opt = torch.optim.Adam(student.parameters(), lr=0.003)
sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
student.train()
BS = 64
for ep in range(20):
    if time.time()-t0 > MAX_TIME:
        print(f"  TIME LIMIT at ep {ep}", flush=True); break
    pm = torch.randperm(len(X_pub))
    for i in range(0, len(X_pub), BS):
        ix = pm[i:i+BS]
        s = student(X_pub[ix].to(DEVICE))
        loss = kd_loss(s[:,:5], tA[ix].to(DEVICE), T=4.0)
        opt.zero_grad(); loss.backward(); opt.step()
    sch1.step()
    if (ep+1)%5==0: print(f"  ep {ep+1}/20 ({time.time()-t0:.0f}s)", flush=True)

# Quick check after phase 1
student.to('cpu').eval()
with torch.no_grad():
    preds = torch.cat([student(X_te[i:i+256]).argmax(1) for i in range(0,len(X_te),256)])
mpc1 = {c:(preds[y_te==c]==c).float().mean().item() for c in ALL}
r_p1 = print_results("After Phase 1 (A-only)", parent_pc, mpc1, ALL)
student.to(DEVICE)

# Phase 2: A+B (50 epochs, w_B=2)
print(f"\n--- Phase 2: A+B KD (50 ep, w_B=2) ---", flush=True)
opt2 = torch.optim.Adam(student.parameters(), lr=0.001)
sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=50)
student.train()
for ep in range(50):
    if time.time()-t0 > MAX_TIME:
        print(f"  TIME LIMIT at ep {ep}", flush=True); break
    pm = torch.randperm(len(X_pub))
    for i in range(0, len(X_pub), BS):
        ix = pm[i:i+BS]
        s = student(X_pub[ix].to(DEVICE))
        lA = kd_loss(s[:,:5], tA[ix].to(DEVICE), T=4.0)
        lB = kd_loss(s[:,5:], tB[ix].to(DEVICE), T=4.0)
        loss = 0.5*lA + 2.0*lB
        opt2.zero_grad(); loss.backward(); opt2.step()
    sch2.step()
    if (ep+1)%10==0:
        # Quick eval every 10 epochs
        student.to('cpu').eval()
        with torch.no_grad():
            preds_q = torch.cat([student(X_te[i:i+256]).argmax(1) for i in range(0,len(X_te),256)])
        mpc_q = {c:(preds_q[y_te==c]==c).float().mean().item() for c in ALL}
        ret_q = sum(1 for c in ALL if (1-mpc_q[c]/parent_pc[c])*100 <= 10)
        min_d = min((1-mpc_q[c]/parent_pc[c])*100 for c in ALL)
        print(f"  ep {ep+1}/50: ret={ret_q}/10 min_drop={min_d:.1f}% ({time.time()-t0:.0f}s)", flush=True)
        student.to(DEVICE).train()

# Final eval
student.to('cpu').eval()
with torch.no_grad():
    preds_f = torch.cat([student(X_te[i:i+256]).argmax(1) for i in range(0,len(X_te),256)])
mpc_f = {c:(preds_f[y_te==c]==c).float().mean().item() for c in ALL}
r_final = print_results("Final: Progressive KD 20+50ep", parent_pc, mpc_f, ALL)

elapsed = time.time()-t0
print(f"\nTotal: {elapsed:.1f}s")
print(f"\nmetric_rn18_final_retained: {r_final}")
print(f"metric_rn18_mean_drop: {np.mean([(1-mpc_f[c]/parent_pc[c])*100 for c in ALL]):.1f}")
print("Done!", flush=True)
