#!/usr/bin/env python3
"""E10: ResNet-18 KD — simultaneous A+B, balanced weights.

E09 revealed: progressive KD causes SEESAW.
Phase 1 → 5/10. Phase 2 → 2/10 (A degrades as B learns).

Fix: Train A+B simultaneously from start with EQUAL weights.
w_B=1.0 (not 2.0). lr=0.002. 20 epochs.
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

tl = torch.load('results/teacher_logits.pth', weights_only=True)
tA, tB = tl['tA'], tl['tB']

with open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/cifar_fix/results/parents_strong.json') as f:
    pd = json.load(f)['42']
parent_pc = {}
for c,a in pd['pcA'].items(): parent_pc[int(c)] = a
for c,a in pd['pcB'].items(): parent_pc[int(c)] = a
print(f"Setup ({time.time()-t0:.1f}s)", flush=True)

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

N = len(X_te); BS = 64

# ═══ Simultaneous A+B, w=1,1 ═══
print(f"\n--- Simultaneous KD (w_A=1, w_B=1, 20ep) ---", flush=True)
torch.manual_seed(SEED)
student = make_rn(10).to(DEVICE)
opt = torch.optim.Adam(student.parameters(), lr=0.002)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
student.train()
for ep in range(20):
    if time.time()-t0 > MAX_TIME: print(f"  TIME at ep {ep}", flush=True); break
    pm = torch.randperm(N)
    for i in range(0, N, BS):
        ix = pm[i:i+BS]
        s = student(X_te[ix].to(DEVICE))
        lA = kd_loss(s[:,:5], tA[ix].to(DEVICE), T=4.0)
        lB = kd_loss(s[:,5:], tB[ix].to(DEVICE), T=4.0)
        loss = lA + lB  # EQUAL weights
        opt.zero_grad(); loss.backward(); opt.step()
    sch.step()
    if (ep+1)%5==0:
        student.to('cpu').eval()
        with torch.no_grad():
            preds = torch.cat([student(X_te[i:i+256]).argmax(1) for i in range(0,N,256)])
        mpc = {c:(preds[y_te==c]==c).float().mean().item() for c in ALL}
        ret = sum(1 for c in ALL if (1-mpc[c]/parent_pc[c])*100 <= 10)
        print(f"  ep {ep+1}: ret={ret}/10 ({time.time()-t0:.0f}s)", flush=True)
        student.to(DEVICE).train()

student.to('cpu').eval()
with torch.no_grad():
    preds_f = torch.cat([student(X_te[i:i+256]).argmax(1) for i in range(0,N,256)])
mpc_f = {c:(preds_f[y_te==c]==c).float().mean().item() for c in ALL}
r1 = print_results("Simultaneous KD w=1,1 20ep", parent_pc, mpc_f, ALL)

elapsed = time.time()-t0
print(f"\nTotal: {elapsed:.1f}s")
print(f"\nmetric_rn18_simult_retained: {r1}")
mins = [(1-mpc_f[c]/parent_pc[c])*100 for c in ALL]
print(f"metric_rn18_mean_drop: {np.mean(mins):.1f}")
print(f"metric_rn18_max_drop: {max(mins):.1f}")
print("Done!", flush=True)
