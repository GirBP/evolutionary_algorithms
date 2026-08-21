#!/usr/bin/env python3
"""ENT logit-space merge for strong parents.
Key insight: backbone merge fails when backbones diverge (fine-tune ≠ shared).
Instead: run BOTH backbones, merge at the logit/feature level.

Usage: python3 ent_logit_merge.py <seed>
"""
import numpy as np, torch, torch.nn as nn, random, json, time, sys
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import torchvision.models as models
from torchvision import datasets
import cma

SEED = int(sys.argv[1])
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
t0 = time.time()

raw_tr = datasets.CIFAR10('/tmp/cifar10', train=True, download=True)
raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_tr = (torch.tensor(raw_tr.data).permute(0,3,1,2).float()/255-mean)/std
y_tr = torch.tensor(raw_tr.targets)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255-mean)/std
y_te = torch.tensor(raw_te.targets)
clA, clB = list(range(5)), list(range(5,10))
ALL = list(range(10))

def make_rn(nc=5):
    m = models.resnet18(weights='IMAGENET1K_V1')
    m.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    return m

# Load parents
mA = make_rn(5); mA.load_state_dict(torch.load(f'results/parentA_s{SEED}.pth', weights_only=True, map_location='cpu')); mA.eval()
mB = make_rn(5); mB.load_state_dict(torch.load(f'results/parentB_s{SEED}.pth', weights_only=True, map_location='cpu')); mB.eval()

with open('results/parents_strong.json') as f: pd = json.load(f)[str(SEED)]
parent_pc = {}
for c, a in pd['pcA'].items(): parent_pc[int(c)] = a
for c, a in pd['pcB'].items(): parent_pc[int(c)] = a
print(f"Parents: A={pd['A']:.3f} B={pd['B']:.3f}, dev={DEV}", flush=True)

# Pre-compute ALL logits on MPS (fast, do once)
def get_logits(model, X, bs=512):
    model.to(DEV).eval()
    with torch.no_grad():
        logits = torch.cat([model(X[i:i+bs].to(DEV)).cpu() for i in range(0,len(X),bs)])
    model.cpu()
    return logits

print("Computing logits on MPS...", flush=True)
lA_te = get_logits(mA, X_te)  # [10000, 5]
lB_te = get_logits(mB, X_te)  # [10000, 5]
lA_cal = get_logits(mA, X_tr[40000:45000])
lB_cal = get_logits(mB, X_tr[40000:45000])
y_cal = y_tr[40000:45000]

# Features for DualProbe
def get_feat(model, X, bs=512):
    model.to(DEV).eval()
    feats = []
    with torch.no_grad():
        for i in range(0,len(X),bs):
            xb = X[i:i+bs].to(DEV)
            h = model.conv1(xb);h=model.bn1(h);h=model.relu(h);h=model.maxpool(h)
            h=model.layer1(h);h=model.layer2(h);h=model.layer3(h);h=model.layer4(h)
            feats.append(torch.flatten(model.avgpool(h),1).cpu())
    model.cpu()
    return torch.cat(feats)

fA_te = get_feat(mA, X_te); fB_te = get_feat(mB, X_te)
fA_v = get_feat(mA, X_tr[45000:]); fB_v = get_feat(mB, X_tr[45000:])
yv = y_tr[45000:]
print(f"Logits+features computed ({time.time()-t0:.0f}s)", flush=True)

def eval_10(preds, y):
    acc = (preds==y).float().mean().item()
    pc = {c:(preds[y==c]==c).float().mean().item() if (y==c).sum()>0 else 0 for c in ALL}
    retained = sum(1 for c in ALL if pc[c] >= 0.9*parent_pc.get(c,0))
    ok = sum(1 for c in ALL if pc[c]>0.3)
    mn = min(pc[c] for c in ALL)
    aM = np.mean([pc[c] for c in clA]); bM = np.mean([pc[c] for c in clB])
    bal = min(aM,bM)/(max(aM,bM)+1e-10)
    return {'acc':round(acc,4),'retained':retained,'ok':ok,'min':round(mn,4),
            'bal':round(bal,4),'pc':{c:round(pc[c],4) for c in ALL}}

# ═══ Method 1: LogitConcat (zero-shot) ═══
pr = torch.cat([lA_te, lB_te], dim=1).argmax(1)
r1 = eval_10(pr, y_te); r1['name'] = 'LogitConcat'
print(f"  LogitConcat: ret={r1['retained']}/10 acc={r1['acc']:.3f}", flush=True)

# ═══ Method 2: DualProbe ═══
torch.manual_seed(SEED)
f_cat_v = torch.cat([fA_v, fB_v], 1); f_cat_te = torch.cat([fA_te, fB_te], 1)
probe = nn.Linear(1024, 10)
op = torch.optim.Adam(probe.parameters(), lr=0.01)
probe.train()
for ep in range(300):
    pm = torch.randperm(len(f_cat_v))[:512]
    loss = nn.CrossEntropyLoss()(probe(f_cat_v[pm]), yv[pm])
    op.zero_grad(); loss.backward(); op.step()
probe.eval()
with torch.no_grad(): pr = probe(f_cat_te).argmax(1)
r2 = eval_10(pr, y_te); r2['name'] = 'DualProbe'
print(f"  DualProbe: ret={r2['retained']}/10 acc={r2['acc']:.3f}", flush=True)

# ═══ Method 3: ENT Logit-Space CMA-ES ═══
print(f"  ENT CMA-ES...", flush=True)

def fitness_ent(x):
    """x=[10 routes, sA, sB, tempA, tempB]"""
    route = x[:10]; sA = x[10]; sB = x[11]; tA = max(x[12], 0.1); tB = max(x[13], 0.1)
    logits = torch.zeros(len(lA_cal), 10)
    for ci,c in enumerate(clA):
        a = 1/(1+np.exp(-route[c]))
        logits[:,c] = a * sA * lA_cal[:,ci] / tA
    for ci,c in enumerate(clB):
        a = 1/(1+np.exp(-route[c]))
        logits[:,c] = (1-a) * sB * lB_cal[:,ci] / tB
    preds = logits.argmax(1)
    acc = (preds==y_cal).float().mean().item()
    pc_ = {c:(preds[y_cal==c]==c).float().mean().item() for c in ALL if (y_cal==c).sum()>0}
    ret = sum(1 for c in ALL if pc_.get(c,0) >= 0.9*parent_pc.get(c,0))
    mn = min(pc_.values()) if pc_ else 0
    return -(0.3*acc + 0.3*(ret/10) + 0.3*mn + 0.1*np.mean(list(pc_.values())))

x0 = np.zeros(14)
x0[:5] = 3.0; x0[5:10] = -3.0  # route A→high, B→low
x0[10] = 1.0; x0[11] = 1.0     # scales
x0[12] = 1.0; x0[13] = 1.0     # temperatures

es = cma.CMAEvolutionStrategy(x0, 0.5,
    {'maxiter': 30, 'popsize': 16, 'seed': SEED, 'verbose': -1})
bf = float('inf'); bx = None; gen = 0
while not es.stop():
    sols = es.ask(); sc = [fitness_ent(x) for x in sols]
    es.tell(sols, sc)
    if min(sc) < bf: bf = min(sc); bx = sols[np.argmin(sc)]
    gen += 1

with torch.no_grad():
    route = bx[:10]; sA = bx[10]; sB = bx[11]; tA = max(bx[12],0.1); tB = max(bx[13],0.1)
    logits = torch.zeros(len(lA_te), 10)
    for ci,c in enumerate(clA):
        a = 1/(1+np.exp(-route[c])); logits[:,c] = a*sA*lA_te[:,ci]/tA
    for ci,c in enumerate(clB):
        a = 1/(1+np.exp(-route[c])); logits[:,c] = (1-a)*sB*lB_te[:,ci]/tB
    pr = logits.argmax(1)
r3 = eval_10(pr, y_te); r3['name'] = 'ENT-Logit'
r3['gen'] = gen
print(f"  ENT-Logit: ret={r3['retained']}/10 acc={r3['acc']:.3f} bal={r3['bal']:.3f}", flush=True)
print(f"    pc: {[r3['pc'][c] for c in ALL]}", flush=True)

# ═══ Summary ═══
elapsed = time.time()-t0
all_m = [r1, r2, r3]
print(f"\n{'='*60}")
print(f"SEED={SEED} Parents: A={pd['A']:.3f} B={pd['B']:.3f} ({elapsed:.0f}s)")
print(f"{'Method':20s} {'Ret':>6} {'Acc':>8} {'Bal':>8} {'Min':>8}")
for r in all_m:
    f = '✅' if r['retained']>=8 else ''
    print(f"{r['name']:20s} {r['retained']:>2}/10   {r['acc']:.4f}   {r['bal']:.4f}   {r['min']:.4f}  {f}")

result = {'seed':SEED, 'parentA':pd['A'], 'parentB':pd['B'],
          'methods':{r['name']:{k:v for k,v in r.items() if k!='name'} for r in all_m},
          'time_s':round(elapsed,1)}
fpath = 'results/merge_results.json'
try:
    with open(fpath) as f: acc = json.load(f)
except: acc = {}
acc[str(SEED)] = result
with open(fpath,'w') as f: json.dump(acc, f, indent=2)

print(f"\nmetric_ent_retained: {r3['retained']}")
print(f"metric_ent_acc: {r3['acc']}")
print(f"metric_ent_bal: {r3['bal']}")
print("Done!", flush=True)
