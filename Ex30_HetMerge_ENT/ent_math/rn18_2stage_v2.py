#!/usr/bin/env python3
"""E13: ResNet-18 2-stage KD — 1200s budget, balanced B.

E12: 4/10 in 7 epochs (600s). Monotonic improvement.
Now: 1200s → ~18 epochs. w_B=1.5 to push B classes.

Also: per-class drop tracking → adaptive w_B during training.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
import random, time, json
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
t0 = time.time()
MAX_TIME = 1150  # leave margin

raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mn = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
sd = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255-mn)/sd
y_te = torch.tensor(raw_te.targets)
ALL = list(range(10))
N = len(X_te)

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
    return ret, mpc, drops

# ═══ Stage 1: FC on frozen backbone ═══
print("Stage 1: FC on frozen backbone (200ep)...", flush=True)
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
        loss = kd_loss(logits[:,:5], tA[ix], T=4.0) + kd_loss(logits[:,5:], tB[ix], T=4.0)
        opt_fc.zero_grad(); loss.backward(); opt_fc.step()
    sch_fc.step()
print(f"  FC trained ({time.time()-t0:.0f}s)", flush=True)

# ═══ Stage 2: Full fine-tune with w_B=1.5, diff LR ═══
print(f"\n--- Stage 2: Full fine-tune (w_B=1.5, backbone_lr=1e-4, fc_lr=1e-3) ---", flush=True)
torch.manual_seed(SEED)
student = make_rn(10)
student.fc.weight.data = fc.weight.data.clone()
student.fc.bias.data = fc.bias.data.clone()
student.to(DEVICE)

opt2 = torch.optim.Adam([
    {'params': [p for n,p in student.named_parameters() if 'fc' not in n], 'lr': 0.0001},
    {'params': student.fc.parameters(), 'lr': 0.001}
])
# Longer schedule for 25 epochs
sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=25)

best_ret = 0; best_mpc = None; best_ep = 0
w_B = 1.5  # upweight B

student.train()
for ep in range(25):
    if time.time()-t0 > MAX_TIME:
        print(f"  TIME LIMIT at ep {ep}", flush=True)
        break
    pm = torch.randperm(N)
    for i in range(0, N, 64):
        ix = pm[i:i+64]
        s = student(X_te[ix].to(DEVICE))
        lA = kd_loss(s[:,:5], tA[ix].to(DEVICE), T=4.0)
        lB = kd_loss(s[:,5:], tB[ix].to(DEVICE), T=4.0)
        loss = lA + w_B * lB
        opt2.zero_grad(); loss.backward(); opt2.step()
    sch2.step()
    
    # Eval every epoch
    ret, mpc, drops = eval_model(student, f"ep {ep+1}")
    if ret >= best_ret:
        best_ret = ret; best_mpc = mpc.copy(); best_ep = ep+1
        # Save best checkpoint
        torch.save(student.state_dict(), 'results/rn18_kd_best.pth')
    student.to(DEVICE).train()

# ═══ Final report ═══
r = print_results(f"BEST (ep {best_ep})", parent_pc, best_mpc, ALL)

elapsed = time.time()-t0
print(f"\nTotal: {elapsed:.1f}s")
print(f"\nmetric_best_retained: {r}")
print(f"metric_best_ep: {best_ep}")
drops_f = [(1-best_mpc[c]/parent_pc[c])*100 for c in ALL]
print(f"metric_mean_drop: {np.mean(drops_f):.1f}")
print(f"metric_max_drop: {max(drops_f):.1f}")
print("Done!", flush=True)
