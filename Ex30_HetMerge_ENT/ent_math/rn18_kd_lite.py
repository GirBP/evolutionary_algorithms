#!/usr/bin/env python3
"""E06b: ResNet-18 KD — lightweight version. 
Budget-aware: 3k transfer images, 5+10 epochs, batch_size=64.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
import random, time, json
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
t0 = time.time()

raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mn = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
sd = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255-mn)/sd
y_te = torch.tensor(raw_te.targets)
ALL = list(range(10))
X_pub = X_te[:3000]  # reduced transfer set

def make_rn(nc=5):
    m = models.resnet18(weights='IMAGENET1K_V1')
    m.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    return m

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

# Pre-compute teacher logits (on CPU to save MPS memory)
pA.eval(); pB.eval()
with torch.no_grad():
    tA = torch.cat([pA(X_pub[i:i+256]) for i in range(0,len(X_pub),256)])
    tB = torch.cat([pB(X_pub[i:i+256]) for i in range(0,len(X_pub),256)])
print(f"Teacher logits ({time.time()-t0:.1f}s)", flush=True)

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
print(f"\n--- Progressive KD: 5+10 ep ---", flush=True)
torch.manual_seed(SEED)
student = make_rn(10).to(DEVICE)

# Phase 1: A only (5 ep, fast)
opt = torch.optim.Adam(student.parameters(), lr=0.005)
student.train()
for ep in range(5):
    pm = torch.randperm(len(X_pub))
    for i in range(0, len(X_pub), 64):
        ix = pm[i:i+64]
        s = student(X_pub[ix].to(DEVICE))
        loss = kd_loss(s[:,:5], tA[ix].to(DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"  Phase1 ep {ep+1} ({time.time()-t0:.0f}s)", flush=True)

# Phase 2: A+B (10 ep)
opt2 = torch.optim.Adam(student.parameters(), lr=0.002)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10)
for ep in range(10):
    pm = torch.randperm(len(X_pub))
    for i in range(0, len(X_pub), 64):
        ix = pm[i:i+64]
        s = student(X_pub[ix].to(DEVICE))
        lossA = kd_loss(s[:,:5], tA[ix].to(DEVICE))
        lossB = kd_loss(s[:,5:], tB[ix].to(DEVICE))
        loss = 0.5*lossA + 2.0*lossB
        opt2.zero_grad(); loss.backward(); opt2.step()
    sch.step()
    print(f"  Phase2 ep {ep+1} ({time.time()-t0:.0f}s)", flush=True)

student.to('cpu').eval()
# Eval on FULL test set (10k)
with torch.no_grad():
    preds = torch.cat([student(X_te[i:i+256]).argmax(1) for i in range(0,len(X_te),256)])
mpc = {c:(preds[y_te==c]==c).float().mean().item() for c in ALL}
r1 = print_results("Progressive KD 5+10ep (3k transfer)", parent_pc, mpc, ALL)

elapsed = time.time()-t0
print(f"\nTotal: {elapsed:.1f}s")
print(f"\nmetric_rn18_kd_retained: {r1}")
print("Done!", flush=True)
