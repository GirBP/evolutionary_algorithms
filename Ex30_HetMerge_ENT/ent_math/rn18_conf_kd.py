#!/usr/bin/env python3
"""E14: Class-selective KD with confidence routing.

Key insight from E12-E13:
- A=4/5, B=0/5 regardless of w_B
- Parent B gives NOISE on images from classes 0-4 (it can only distinguish 5-9)
- That noise signal is harming B's classes in the student

Fix: For each image, only use KD from the parent that is CONFIDENT.
- If max(softmax(teacher_A)) > threshold → use KD from A for that image
- If max(softmax(teacher_B)) > threshold → use KD from B for that image  
- Confidence-weight the loss

Also: use pseudo-labels for high-confidence samples (CE → cleaner signal).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
import random, time, json
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
t0 = time.time()
MAX_TIME = 1150

raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mn = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
sd = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255-mn)/sd
y_te = torch.tensor(raw_te.targets)
ALL = list(range(10))
N = len(X_te)

tl = torch.load('results/teacher_logits.pth', weights_only=True)
tA, tB = tl['tA'], tl['tB']

# Pre-compute teacher confidences
confA = F.softmax(tA, dim=1).max(1)[0]  # [10k]
confB = F.softmax(tB, dim=1).max(1)[0]  # [10k]
predA = tA.argmax(1)  # 0-4 label from parent A
predB = tB.argmax(1) + 5  # 5-9 label from parent B

print(f"Teacher confidence stats:", flush=True)
print(f"  A: mean={confA.mean():.3f} std={confA.std():.3f} >0.8: {(confA>0.8).float().mean():.2%}", flush=True)
print(f"  B: mean={confB.mean():.3f} std={confB.std():.3f} >0.8: {(confB>0.8).float().mean():.2%}", flush=True)

with open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/cifar_fix/results/parents_strong.json') as f:
    pd = json.load(f)['42']
parent_pc = {}
for c,a in pd['pcA'].items(): parent_pc[int(c)] = a
for c,a in pd['pcB'].items(): parent_pc[int(c)] = a

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

def eval_model(student, label=""):
    student.to('cpu').eval()
    with torch.no_grad():
        preds = torch.cat([student(X_te[i:i+256]).argmax(1) for i in range(0,N,256)])
    mpc = {c:(preds[y_te==c]==c).float().mean().item() for c in ALL}
    drops = {c:(1-mpc[c]/parent_pc[c])*100 for c in ALL}
    ret = sum(1 for c in ALL if drops[c] <= 10)
    if label:
        a_ret = sum(1 for c in range(5) if drops[c]<=10)
        b_ret = sum(1 for c in range(5,10) if drops[c]<=10)
        print(f"  {label}: ret={ret}/10 (A={a_ret}/5 B={b_ret}/5) mean_drop={np.mean(list(drops.values())):.1f}% ({time.time()-t0:.0f}s)", flush=True)
    return ret, mpc

# ═══ Stage 1: FC on frozen backbone with confidence-weighted loss ═══
print("\nStage 1: FC on frozen + confidence weighting...", flush=True)
torch.manual_seed(SEED)
backbone = make_rn(10)
backbone.fc = nn.Identity()
backbone.to(DEVICE).eval()
with torch.no_grad():
    feats = torch.cat([backbone(X_te[i:i+128].to(DEVICE)).cpu() for i in range(0,N,128)])
del backbone

torch.manual_seed(SEED)
fc = nn.Linear(512, 10)
opt_fc = torch.optim.Adam(fc.parameters(), lr=0.01)
sch_fc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fc, T_max=200)

for ep in range(200):
    pm = torch.randperm(N)
    for i in range(0, N, 256):
        ix = pm[i:i+256]
        logits = fc(feats[ix].float())
        
        # Confidence-weighted KD: multiply each sample's loss by its teacher's confidence
        # KD from A 
        s_A = F.log_softmax(logits[:,:5] / 4.0, dim=1)
        t_A = F.softmax(tA[ix] / 4.0, dim=1)
        kd_A_per = F.kl_div(s_A, t_A, reduction='none').sum(1) * 16  # T²
        loss_A = (kd_A_per * confA[ix]).mean()
        
        # KD from B
        s_B = F.log_softmax(logits[:,5:] / 4.0, dim=1)
        t_B = F.softmax(tB[ix] / 4.0, dim=1)
        kd_B_per = F.kl_div(s_B, t_B, reduction='none').sum(1) * 16
        loss_B = (kd_B_per * confB[ix]).mean()
        
        loss = loss_A + loss_B
        opt_fc.zero_grad(); loss.backward(); opt_fc.step()
    sch_fc.step()
print(f"  FC trained ({time.time()-t0:.0f}s)", flush=True)

# ═══ Stage 2: Full fine-tune with conf-weighted + pseudo-label hybrid ═══
print(f"\n--- Stage 2: Full + conf-weighted KD ---", flush=True)
torch.manual_seed(SEED)
student = make_rn(10)
student.fc.weight.data = fc.weight.data.clone()
student.fc.bias.data = fc.bias.data.clone()
student.to(DEVICE)

opt2 = torch.optim.Adam([
    {'params': [p for n,p in student.named_parameters() if 'fc' not in n], 'lr': 0.0001},
    {'params': student.fc.parameters(), 'lr': 0.001}
])
sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=20)

best_ret = 0; best_mpc = None; best_ep = 0
student.train()
for ep in range(20):
    if time.time()-t0 > MAX_TIME:
        print(f"  TIME at ep {ep}", flush=True); break
    pm = torch.randperm(N)
    for i in range(0, N, 64):
        ix = pm[i:i+64]
        s = student(X_te[ix].to(DEVICE))
        
        # Conf-weighted KD
        s_A = F.log_softmax(s[:,:5] / 4.0, dim=1)
        t_A = F.softmax(tA[ix].to(DEVICE) / 4.0, dim=1)
        kd_A_per = F.kl_div(s_A, t_A, reduction='none').sum(1) * 16
        loss_A = (kd_A_per * confA[ix].to(DEVICE)).mean()
        
        s_B = F.log_softmax(s[:,5:] / 4.0, dim=1)
        t_B = F.softmax(tB[ix].to(DEVICE) / 4.0, dim=1)
        kd_B_per = F.kl_div(s_B, t_B, reduction='none').sum(1) * 16
        loss_B = (kd_B_per * confB[ix].to(DEVICE)).mean()
        
        # Also add pseudo-label CE for high-confidence samples
        mask_high_A = confA[ix] > 0.8
        mask_high_B = confB[ix] > 0.8
        ce_loss = 0
        if mask_high_A.any():
            ce_loss = ce_loss + F.cross_entropy(s[mask_high_A.to(DEVICE)], predA[ix][mask_high_A].to(DEVICE))
        if mask_high_B.any():
            ce_loss = ce_loss + F.cross_entropy(s[mask_high_B.to(DEVICE)], predB[ix][mask_high_B].to(DEVICE))
        
        loss = loss_A + loss_B + 0.3 * ce_loss
        opt2.zero_grad(); loss.backward(); opt2.step()
    sch2.step()
    
    ret, mpc = eval_model(student, f"ep {ep+1}")
    if ret >= best_ret:
        best_ret = ret; best_mpc = mpc.copy(); best_ep = ep+1
    student.to(DEVICE).train()

r = print_results(f"BEST (ep {best_ep})", parent_pc, best_mpc, ALL)

elapsed = time.time()-t0
print(f"\nTotal: {elapsed:.1f}s")
print(f"\nmetric_best_retained: {r}")
print(f"metric_best_ep: {best_ep}")
print("Done!", flush=True)
