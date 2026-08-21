#!/usr/bin/env python3
"""Per-seed ResNet-18 pipeline — MPS, MINIMAL training.
Freeze backbone entirely. Only train fc. 10k samples, 10 epochs.
Usage: python3 resnet_fast.py <seed>
"""
import numpy as np, torch, torch.nn as nn, random, json, time, sys
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import torchvision.models as models
from torchvision import datasets
import cma

SEED = int(sys.argv[1])
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEV = torch.device('mps')
t0 = time.time()

# Data
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
print(f"Data: {time.time()-t0:.1f}s", flush=True)

def make_rn(nc=5):
    m = models.resnet18(weights='IMAGENET1K_V1')
    m.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    nn.init.kaiming_normal_(m.conv1.weight, mode='fan_out', nonlinearity='relu')
    nn.init.kaiming_normal_(m.fc.weight); nn.init.zeros_(m.fc.bias)
    return m

def extract_features_mps(model, X, bs=512):
    """Extract features using MPS for speed."""
    model = model.to(DEV).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i+bs].to(DEV)
            h = model.conv1(xb); h = model.bn1(h); h = model.relu(h); h = model.maxpool(h)
            h = model.layer1(h); h = model.layer2(h); h = model.layer3(h); h = model.layer4(h)
            feats.append(torch.flatten(model.avgpool(h), 1).cpu())
    model = model.cpu()
    return torch.cat(feats)

def train_parent_features(cls_list, seed_p, epochs=20, lr=0.01):
    """Two-stage: extract features on MPS, then train fc on CPU (instant)."""
    torch.manual_seed(seed_p); np.random.seed(seed_p); random.seed(seed_p)
    cls_map = {c:i for i,c in enumerate(cls_list)}
    
    # Get base ResNet features
    base = make_rn(len(cls_list))
    # Freeze everything except fc
    for p in base.parameters(): p.requires_grad = False
    base.fc.weight.requires_grad = True
    base.fc.bias.requires_grad = True
    
    # Extract features with base backbone (pretrained)
    mask_tr = sum(y_tr==c for c in cls_list).bool()
    Xs = X_tr[mask_tr]
    ys = torch.tensor([cls_map[y.item()] for y in y_tr[mask_tr]])
    # Limit
    idx = torch.cat([torch.where(ys==i)[0][:2000] for i in range(len(cls_list))])
    Xs, ys = Xs[idx], ys[idx]
    
    print(f"    Extracting features ({len(Xs)} samples)...", end='', flush=True)
    feats = extract_features_mps(base, Xs)  # [N, 512]
    print(f" done", flush=True)
    
    # Train fc on CPU (very fast — just 512→5 linear)
    fc = nn.Linear(512, len(cls_list))
    nn.init.kaiming_normal_(fc.weight); nn.init.zeros_(fc.bias)
    opt = torch.optim.Adam(fc.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    fc.train()
    for ep in range(epochs):
        perm = torch.randperm(len(feats))
        for i in range(0, len(feats), 256):
            ix = perm[i:i+256]
            loss = nn.CrossEntropyLoss()(fc(feats[ix]), ys[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    
    # Assemble full model
    base.fc = fc
    base.eval()
    
    # Eval
    mask_te = sum(y_te==c for c in cls_list).bool()
    Xt = X_te[mask_te]
    yt = torch.tensor([cls_map[y.item()] for y in y_te[mask_te]])
    feats_te = extract_features_mps(base, Xt)
    fc.eval()
    with torch.no_grad():
        preds = fc(feats_te).argmax(1)
        acc = (preds==yt).float().mean().item()
        pc = {cls_list[i]:(preds[yt==i]==i).float().mean().item() for i in range(len(cls_list))}
    
    return base, acc, pc, cls_map

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

# ═══ Phase 1: Train Parents ═══
t1 = time.time()
mA, accA, pcA, mapA = train_parent_features(clA, SEED)
print(f"  Parent A: {accA:.3f} ({time.time()-t1:.0f}s)", flush=True)

t1 = time.time()
mB, accB, pcB, mapB = train_parent_features(clB, SEED+10000)
print(f"  Parent B: {accB:.3f} ({time.time()-t1:.0f}s)", flush=True)

sdA, sdB = mA.state_dict(), mB.state_dict()
wA, bA = sdA['fc.weight'], sdA['fc.bias']
wB, bB = sdB['fc.weight'], sdB['fc.bias']
res = {'parentA':round(accA,4),'parentB':round(accB,4),
       'pcA':{c:round(v,3) for c,v in pcA.items()},
       'pcB':{c:round(v,3) for c,v in pcB.items()}}

# ═══ Phase 2: Merge Methods ═══

# Pre-extract features for all test data (once)
print("  Extracting test features...", flush=True)
fA_te = extract_features_mps(mA, X_te)  # [10000, 512]
fB_te = extract_features_mps(mB, X_te)  # [10000, 512]

# Method 1: LogitConcat
fcA = mA.fc; fcB = mB.fc
fcA.eval(); fcB.eval()
with torch.no_grad():
    lA = fcA(fA_te)  # [N, 5] for cls 0-4
    lB = fcB(fB_te)  # [N, 5] for cls 5-9
    pr = torch.cat([lA, lB], dim=1).argmax(1)
r1 = eval_10(pr, y_te); r1['name'] = 'LogitConcat'
print(f"  LogitConcat: ok={r1['ok']}/10 acc={r1['acc']:.3f} bal={r1['bal']:.3f}", flush=True)

# Method 2: WeightAvg backbone + fc map
avgM = make_rn(10)
sd_avg = {}
for k in sdA:
    if 'fc' not in k: sd_avg[k] = 0.5*sdA[k] + 0.5*sdB[k]
fc_w = torch.zeros(10,512); fc_b = torch.zeros(10)
for i,c in enumerate(clA): fc_w[c] = wA[i]; fc_b[c] = bA[i]
for i,c in enumerate(clB): fc_w[c] = wB[i]; fc_b[c] = bB[i]
sd_avg['fc.weight'] = fc_w; sd_avg['fc.bias'] = fc_b
avgM.load_state_dict(sd_avg); avgM.eval()
# Eval with backbone avg — need full forward pass
feat_avg = extract_features_mps(avgM, X_te)
with torch.no_grad():
    pr = avgM.fc(feat_avg).argmax(1)
r2 = eval_10(pr, y_te); r2['name'] = 'WeightAvg'
print(f"  WeightAvg: ok={r2['ok']}/10 acc={r2['acc']:.3f} bal={r2['bal']:.3f}", flush=True)

# Method 3: DualBackbone + Probe
f_cat_te = torch.cat([fA_te, fB_te], 1)  # [10000, 1024]
# Val data features
fA_v = extract_features_mps(mA, X_tr[45000:])
fB_v = extract_features_mps(mB, X_tr[45000:])
yv = y_tr[45000:]
f_cat_v = torch.cat([fA_v, fB_v], 1)

torch.manual_seed(SEED)
probe = nn.Linear(1024, 10)
op = torch.optim.Adam(probe.parameters(), lr=0.01)
probe.train()
for ep in range(200):
    pm = torch.randperm(len(f_cat_v))[:512]
    loss = nn.CrossEntropyLoss()(probe(f_cat_v[pm]), yv[pm])
    op.zero_grad(); loss.backward(); op.step()
probe.eval()
with torch.no_grad(): pr = probe(f_cat_te).argmax(1)
r3 = eval_10(pr, y_te); r3['name'] = 'DualProbe'
print(f"  DualProbe: ok={r3['ok']}/10 acc={r3['acc']:.3f} bal={r3['bal']:.3f}", flush=True)

# Method 4: ENT CMA-ES
# Since backbone is frozen (pretrained), ENT operates on feature→fc space
# Optimize: per-class routing coefficients for combining the fc outputs
X_cal, y_cal = X_tr[40000:45000], y_tr[40000:45000]
fA_cal = extract_features_mps(mA, X_cal)
fB_cal = extract_features_mps(mB, X_cal)

def fitness_ent(x):
    """x = [10 route values, sA, sB]."""
    route = x[:10]; sA = x[10]; sB = x[11]
    # Build 10-class logits from A (cls 0-4) and B (cls 5-9)
    with torch.no_grad():
        lA = fcA(fA_cal) * sA  # [N, 5]
        lB = fcB(fB_cal) * sB  # [N, 5]
        logits_10 = torch.zeros(len(fA_cal), 10)
        for ci, c in enumerate(clA):
            alpha = 1/(1+np.exp(-route[c]))
            logits_10[:, c] = alpha * lA[:, ci]
        for ci, c in enumerate(clB):
            alpha = 1/(1+np.exp(-route[c]))
            logits_10[:, c] = (1-alpha) * lB[:, ci]
        preds = logits_10.argmax(1)
    acc = (preds==y_cal).float().mean().item()
    pc = {c:(preds[y_cal==c]==c).float().mean().item() for c in ALL if (y_cal==c).sum()>0}
    mn = min(pc.values()) if pc else 0
    ok = sum(1 for v in pc.values() if v>0.3)
    return -(0.3*acc + 0.4*mn + 0.2*ok/10 + 0.1*np.mean(list(pc.values())))

x0 = np.zeros(12)
x0[:5] = 3.0; x0[5:10] = -3.0; x0[10] = 1.0; x0[11] = 1.0
es = cma.CMAEvolutionStrategy(x0, 0.5,
    {'maxiter': 20, 'popsize': 12, 'seed': SEED, 'verbose': -1})
bs_f = float('inf'); bx = None
while not es.stop():
    sols = es.ask()
    sc = [fitness_ent(x) for x in sols]
    es.tell(sols, sc)
    if min(sc) < bs_f: bs_f = min(sc); bx = sols[np.argmin(sc)]

with torch.no_grad():
    route = bx[:10]; sA_v = bx[10]; sB_v = bx[11]
    lA_t = fcA(fA_te) * sA_v
    lB_t = fcB(fB_te) * sB_v
    logits_10 = torch.zeros(len(fA_te), 10)
    for ci, c in enumerate(clA):
        alpha = 1/(1+np.exp(-route[c]))
        logits_10[:, c] = alpha * lA_t[:, ci]
    for ci, c in enumerate(clB):
        alpha = 1/(1+np.exp(-route[c]))
        logits_10[:, c] = (1-alpha) * lB_t[:, ci]
    pr = logits_10.argmax(1)
r4 = eval_10(pr, y_te); r4['name'] = 'ENT'
print(f"  ENT: ok={r4['ok']}/10 acc={r4['acc']:.3f} bal={r4['bal']:.3f}", flush=True)

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

print(f"\n  === seed={SEED} ({elapsed:.0f}s) ===")
print(f"  Parents: A={accA:.3f} B={accB:.3f}")
for r in all_methods:
    s = '✅' if r['ok']>=8 else ''
    print(f"  {r['name']:15s}: ok={r['ok']:>2}/10 acc={r['acc']:.3f} bal={r['bal']:.3f} min={r['min']:.3f} {s}")
    print(f"    pc: {[r['pc'][c] for c in ALL]}")

for r in all_methods:
    print(f"metric_{r['name']}_ok: {r['ok']}")
print(f"metric_parentA: {accA:.4f}")
print(f"metric_parentB: {accB:.4f}")
print("Done!", flush=True)
