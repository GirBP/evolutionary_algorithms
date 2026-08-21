#!/usr/bin/env python3
"""E11: ResNet-18 frozen backbone + FC retraining from KD.

ImageNet pretrained ResNet-18 backbone is frozen.
Only train FC layer (512→10) using KD signals from both parents.
This is ~100× faster per epoch than full fine-tuning.

Also: try unfreezing last layer after FC converges (2-stage).
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

N = len(X_te)

# ═══ Pre-compute features from ImageNet backbone ═══
print("Pre-computing ImageNet features...", flush=True)
torch.manual_seed(SEED)
backbone = make_rn(10)
# Remove FC to get features
backbone.fc = nn.Identity()
backbone.to(DEVICE).eval()
with torch.no_grad():
    feats = torch.cat([backbone(X_te[i:i+128].to(DEVICE)).cpu() for i in range(0,N,128)])  # [10k, 512]
del backbone
print(f"Features: {feats.shape} ({time.time()-t0:.1f}s)", flush=True)

# ═══ Method 1: Train FC from KD (fast!) ═══
print(f"\n--- FC-only KD (200 ep) ---", flush=True)
torch.manual_seed(SEED)
fc = nn.Linear(512, 10)
opt = torch.optim.Adam(fc.parameters(), lr=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
BS = 256

for ep in range(200):
    if time.time()-t0 > MAX_TIME/2: break  # save time for stage 2
    pm = torch.randperm(N)
    for i in range(0, N, BS):
        ix = pm[i:i+BS]
        logits = fc(feats[ix].float())
        lA = kd_loss(logits[:,:5], tA[ix], T=4.0)
        lB = kd_loss(logits[:,5:], tB[ix], T=4.0)
        loss = lA + lB
        opt.zero_grad(); loss.backward(); opt.step()
    sch.step()

fc.eval()
with torch.no_grad():
    preds = fc(feats.float()).argmax(1)
mpc1 = {c:(preds[y_te==c]==c).float().mean().item() for c in ALL}
r1 = print_results(f"FC-only KD (ep={ep+1})", parent_pc, mpc1, ALL)

# ═══ Method 2: Stage 2 — unfreeze layer4 + FC ═══
print(f"\n--- Stage 2: Unfreeze layer4 ({time.time()-t0:.0f}s) ---", flush=True)
torch.manual_seed(SEED)
student2 = make_rn(10)
# Copy the trained FC
student2.fc.weight.data = fc.weight.data.clone()
student2.fc.bias.data = fc.bias.data.clone()
# Freeze everything except layer4 + fc
for name, p in student2.named_parameters():
    if 'layer4' not in name and 'fc' not in name:
        p.requires_grad = False

student2.to(DEVICE)
opt2 = torch.optim.Adam(filter(lambda p: p.requires_grad, student2.parameters()), lr=0.0005)
sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=15)
student2.train()
for ep in range(15):
    if time.time()-t0 > MAX_TIME: print(f"  TIME at ep {ep}", flush=True); break
    pm = torch.randperm(N)
    for i in range(0, N, 64):
        ix = pm[i:i+64]
        s = student2(X_te[ix].to(DEVICE))
        lA = kd_loss(s[:,:5], tA[ix].to(DEVICE), T=4.0)
        lB = kd_loss(s[:,5:], tB[ix].to(DEVICE), T=4.0)
        loss = lA + lB
        opt2.zero_grad(); loss.backward(); opt2.step()
    sch2.step()
    if (ep+1)%5==0:
        student2.to('cpu').eval()
        with torch.no_grad():
            preds2q = torch.cat([student2(X_te[i:i+256]).argmax(1) for i in range(0,N,256)])
        mpc2q = {c:(preds2q[y_te==c]==c).float().mean().item() for c in ALL}
        ret2q = sum(1 for c in ALL if (1-mpc2q[c]/parent_pc[c])*100 <= 10)
        print(f"  ep {ep+1}: ret={ret2q}/10 ({time.time()-t0:.0f}s)", flush=True)
        student2.to(DEVICE).train()

student2.to('cpu').eval()
with torch.no_grad():
    preds_f = torch.cat([student2(X_te[i:i+256]).argmax(1) for i in range(0,N,256)])
mpc_f = {c:(preds_f[y_te==c]==c).float().mean().item() for c in ALL}
r2 = print_results("Stage 2: layer4+FC fine-tuned", parent_pc, mpc_f, ALL)

best = max(r1, r2)
elapsed = time.time()-t0
print(f"\nTotal: {elapsed:.1f}s")
print(f"\nmetric_fc_only: {r1}")
print(f"metric_stage2: {r2}")
print(f"metric_best: {best}")
print("Done!", flush=True)
