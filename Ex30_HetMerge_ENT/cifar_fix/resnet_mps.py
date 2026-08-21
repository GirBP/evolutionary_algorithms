#!/usr/bin/env python3
"""Per-seed ResNet-18 pipeline — MPS accelerated.
Usage: python3 resnet_mps.py <seed>
Training on MPS, eval on CPU.
"""
import numpy as np, torch, torch.nn as nn, random, json, time, sys, os
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
import cma

SEED = int(sys.argv[1])
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEV = torch.device('mps')
CPU = torch.device('cpu')
t0 = time.time()

# ═══════════════════════════════════════════
# Data
# ═══════════════════════════════════════════
raw_tr = datasets.CIFAR10('/tmp/cifar10', train=True, download=True)
raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_tr = (torch.tensor(raw_tr.data).permute(0,3,1,2).float()/255.0-mean)/std
y_tr = torch.tensor(raw_tr.targets)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255.0-mean)/std
y_te = torch.tensor(raw_te.targets)
clA, clB = list(range(5)), list(range(5,10))
ALL = list(range(10))
print(f"Data loaded ({time.time()-t0:.1f}s)", flush=True)

def make_rn(nc=5):
    m = models.resnet18(weights='IMAGENET1K_V1')
    m.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512,nc)
    nn.init.kaiming_normal_(m.conv1.weight, mode='fan_out', nonlinearity='relu')
    nn.init.kaiming_normal_(m.fc.weight); nn.init.zeros_(m.fc.bias)
    return m

def train_parent_mps(cls_list, seed_p, epochs=15, lr=0.001):
    """Train parent on MPS GPU."""
    torch.manual_seed(seed_p); np.random.seed(seed_p); random.seed(seed_p)
    m = make_rn(len(cls_list)).to(DEV)
    cls_map = {c:i for i,c in enumerate(cls_list)}
    
    # Get data subset
    mask = sum(y_tr==c for c in cls_list).bool()
    Xs, ys = X_tr[mask], y_tr[mask]
    ys_mapped = torch.tensor([cls_map[y.item()] for y in ys])
    # Limit per class
    idx = torch.cat([torch.where(ys_mapped==i)[0][:5000] for i in range(len(cls_list))])
    Xs, ys_mapped = Xs[idx], ys_mapped[idx]
    
    # Freeze early layers
    for n,p in m.named_parameters():
        if 'layer3' not in n and 'layer4' not in n and 'fc' not in n:
            p.requires_grad = False
    
    opt = torch.optim.Adam(filter(lambda p:p.requires_grad, m.parameters()), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    # DataLoader
    ds = TensorDataset(Xs, ys_mapped)
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    
    m.train()
    for ep in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            # Random flip
            if torch.rand(1).item() > 0.5:
                xb = xb.flip(-1)
            loss = nn.CrossEntropyLoss()(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    
    # Eval on CPU
    m = m.to(CPU).eval()
    mask_te = sum(y_te==c for c in cls_list).bool()
    Xt = X_te[mask_te]
    yt = torch.tensor([cls_map[y.item()] for y in y_te[mask_te]])
    with torch.no_grad():
        preds = torch.cat([m(Xt[i:i+512]).argmax(1) for i in range(0,len(Xt),512)])
        acc = (preds==yt).float().mean().item()
        pc = {cls_list[i]:(preds[yt==i]==i).float().mean().item() for i in range(len(cls_list))}
    return m, acc, pc, cls_map

def get_feat(model, X, bs=512):
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i+bs]
            h = model.conv1(xb); h = model.bn1(h); h = model.relu(h); h = model.maxpool(h)
            h = model.layer1(h); h = model.layer2(h); h = model.layer3(h); h = model.layer4(h)
            feats.append(torch.flatten(model.avgpool(h), 1))
    return torch.cat(feats)

def eval_10(preds, y):
    acc = (preds==y).float().mean().item()
    pc = {c:(preds[y==c]==c).float().mean().item() if (y==c).sum()>0 else 0 for c in ALL}
    ok = sum(1 for c in ALL if pc[c]>0.3)
    mn = min(pc[c] for c in ALL)
    aM = np.mean([pc[c] for c in clA]); bM = np.mean([pc[c] for c in clB])
    bal = min(aM,bM)/(max(aM,bM)+1e-10)
    return {'acc':round(acc,4),'ok':ok,'min':round(mn,4),'bal':round(bal,4),
            'pc':{c:round(pc[c],3) for c in ALL}}

print(f"--- Seed {SEED} ---", flush=True)

# ═══ Phase 1: Train Parents on MPS ═══
t1 = time.time()
mA, accA, pcA, mapA = train_parent_mps(clA, SEED)
print(f"  Parent A: {accA:.3f} ({time.time()-t1:.0f}s) {[round(pcA[c],2) for c in clA]}", flush=True)

t1 = time.time()
mB, accB, pcB, mapB = train_parent_mps(clB, SEED+10000)
print(f"  Parent B: {accB:.3f} ({time.time()-t1:.0f}s) {[round(pcB[c],2) for c in clB]}", flush=True)

sdA, sdB = mA.state_dict(), mB.state_dict()
wA, bA = sdA['fc.weight'], sdA['fc.bias']  # [5,512], [5]
wB, bB = sdB['fc.weight'], sdB['fc.bias']

res = {'parentA':round(accA,4),'parentB':round(accB,4),
       'pcA':{c:round(v,3) for c,v in pcA.items()},
       'pcB':{c:round(v,3) for c,v in pcB.items()}}

# ═══ Phase 2: Merge Methods (all on CPU) ═══

# Method 1: LogitConcat (zero-shot — just concatenate 5+5 class logits)
mA.eval(); mB.eval()
with torch.no_grad():
    lA = torch.cat([mA(X_te[i:i+512]) for i in range(0,len(X_te),512)])
    lB = torch.cat([mB(X_te[i:i+512]) for i in range(0,len(X_te),512)])
    pr = torch.cat([lA, lB], dim=1).argmax(1)
r1 = eval_10(pr, y_te); r1['name'] = 'LogitConcat'
print(f"  LogitConcat: ok={r1['ok']}/10 acc={r1['acc']:.3f} bal={r1['bal']:.3f}", flush=True)

# Method 2: WeightAvg backbone + fc mapping
avgM = make_rn(10)
sd_avg = {}
for k in sdA:
    if 'fc' not in k: sd_avg[k] = 0.5*sdA[k] + 0.5*sdB[k]
fc_w = torch.zeros(10, 512); fc_b = torch.zeros(10)
for i,c in enumerate(clA): fc_w[c] = wA[i]; fc_b[c] = bA[i]
for i,c in enumerate(clB): fc_w[c] = wB[i]; fc_b[c] = bB[i]
sd_avg['fc.weight'] = fc_w; sd_avg['fc.bias'] = fc_b
avgM.load_state_dict(sd_avg); avgM.eval()
with torch.no_grad():
    pr = torch.cat([avgM(X_te[i:i+512]).argmax(1) for i in range(0,len(X_te),512)])
r2 = eval_10(pr, y_te); r2['name'] = 'WeightAvg'
print(f"  WeightAvg: ok={r2['ok']}/10 acc={r2['acc']:.3f} bal={r2['bal']:.3f}", flush=True)

# Method 3: DualBackbone + LinearProbe (needs training data)
fA_te = get_feat(mA, X_te); fB_te = get_feat(mB, X_te)
fA_v = get_feat(mA, X_tr[45000:]); fB_v = get_feat(mB, X_tr[45000:])
yv = y_tr[45000:]
f_cat_v = torch.cat([fA_v, fB_v], 1)  # [5000, 1024]
f_cat_te = torch.cat([fA_te, fB_te], 1)

torch.manual_seed(SEED)
# Train probe on MPS for speed
probe = nn.Linear(1024, 10).to(DEV)
op = torch.optim.Adam(probe.parameters(), lr=0.01)
probe.train()
for ep in range(100):
    pm = torch.randperm(len(f_cat_v))[:512]
    xb = f_cat_v[pm].to(DEV); yb = yv[pm].to(DEV)
    loss = nn.CrossEntropyLoss()(probe(xb), yb)
    op.zero_grad(); loss.backward(); op.step()
probe = probe.to(CPU).eval()
with torch.no_grad():
    pr = probe(f_cat_te).argmax(1)
r3 = eval_10(pr, y_te); r3['name'] = 'DualProbe'
print(f"  DualProbe: ok={r3['ok']}/10 acc={r3['acc']:.3f} bal={r3['bal']:.3f}", flush=True)

# Method 4: ENT CMA-ES per-layer α
layer_keys = [k for k in sdA if 'fc' not in k and k in sdB]
n_lk = len(layer_keys)
X_cal, y_cal = X_tr[40000:45000], y_tr[40000:45000]

def build_ent(x):
    a = 1/(1+np.exp(-np.array(x[:n_lk]))); rt = x[n_lk:]
    m = make_rn(10); sd = {}
    for i,k in enumerate(layer_keys):
        sd[k] = a[i]*sdA[k] + (1-a[i])*sdB[k]
    fw = torch.zeros(10,512); fb = torch.zeros(10)
    for ci,c in enumerate(clA):
        s = 1/(1+np.exp(-rt[c])); fw[c] = s*wA[ci]; fb[c] = s*bA[ci]
    for ci,c in enumerate(clB):
        s = 1/(1+np.exp(-rt[c])); fw[c] = (1-s)*wB[ci]; fb[c] = (1-s)*bB[ci]
    sd['fc.weight'] = fw; sd['fc.bias'] = fb
    m.load_state_dict(sd); return m

def fit_ent(x):
    m = build_ent(x); m.eval()
    with torch.no_grad():
        pr = torch.cat([m(X_cal[i:i+512]).argmax(1) for i in range(0,len(X_cal),512)])
    acc = (pr==y_cal).float().mean().item()
    pc = {c:(pr[y_cal==c]==c).float().mean().item() for c in ALL if (y_cal==c).sum()>0}
    mn = min(pc.values()) if pc else 0
    ok = sum(1 for v in pc.values() if v > 0.3)
    return -(0.3*acc + 0.4*mn + 0.2*ok/10 + 0.1*np.mean(list(pc.values())))

x0 = np.zeros(n_lk + 10)
x0[n_lk:n_lk+5] = 3.0    # route cls 0-4 → A
x0[n_lk+5:] = -3.0        # route cls 5-9 → B

es = cma.CMAEvolutionStrategy(x0, 0.5,
    {'maxiter': 15, 'popsize': 10, 'seed': SEED, 'verbose': -1})
bs_f = float('inf'); bx = None
while not es.stop():
    sols = es.ask()
    sc = [fit_ent(x) for x in sols]
    es.tell(sols, sc)
    if min(sc) < bs_f:
        bs_f = min(sc); bx = sols[np.argmin(sc)]

mE = build_ent(bx); mE.eval()
with torch.no_grad():
    pr = torch.cat([mE(X_te[i:i+512]).argmax(1) for i in range(0,len(X_te),512)])
r4 = eval_10(pr, y_te); r4['name'] = 'ENT'
print(f"  ENT: ok={r4['ok']}/10 acc={r4['acc']:.3f} bal={r4['bal']:.3f}", flush=True)

# Method 5: Task Arithmetic
# TA: base + τ*(A-base) + τ*(B-base) — but we don't have shared base
# Instead: interpolate backbone with per-layer τ optimized by CMA-ES
# Already covered by ENT above. Use fixed α=0.5 as TA baseline.
taM = make_rn(10)
sd_ta = {}
for k in sdA:
    if 'fc' not in k: sd_ta[k] = 0.5*sdA[k] + 0.5*sdB[k]
sd_ta['fc.weight'] = fc_w; sd_ta['fc.bias'] = fc_b
taM.load_state_dict(sd_ta); taM.eval()
# This is identical to WeightAvg — skip duplicate

# ═══ Save ═══
elapsed = time.time() - t0
all_methods = [r1, r2, r3, r4]
res['methods'] = {r['name']: {k:v for k,v in r.items() if k not in ('name','pc')} for r in all_methods}
res['per_class'] = {r['name']: r['pc'] for r in all_methods}
res['time_s'] = round(elapsed, 1)

fpath = 'results/all_seeds.json'
try:
    with open(fpath) as f: accum = json.load(f)
except: accum = {}
accum[str(SEED)] = res
with open(fpath, 'w') as f: json.dump(accum, f, indent=2)

print(f"\n  Summary (seed={SEED}, {elapsed:.0f}s):")
print(f"  Parents: A={accA:.3f} B={accB:.3f}")
for r in all_methods:
    s = '✅' if r['ok']>=8 else ''
    print(f"  {r['name']:15s}: ok={r['ok']}/10 acc={r['acc']:.3f} bal={r['bal']:.3f} min={r['min']:.3f} {s}")
    print(f"    pc: {[r['pc'][c] for c in ALL]}")

for r in all_methods:
    print(f"metric_{r['name']}_ok: {r['ok']}")
    print(f"metric_{r['name']}_acc: {r['acc']}")
print(f"metric_parentA: {accA:.4f}")
print(f"metric_parentB: {accB:.4f}")
print("Done!", flush=True)
