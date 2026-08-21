#!/usr/bin/env python3
"""CIFAR-10 ENT: 5 seeds × (parents + 4 merge methods + stats).
ResNet-18 pretrained, fc-only fine-tune, ~45s/seed.
"""
import numpy as np, torch, torch.nn as nn, random, json, time, sys
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import torchvision.models as models
from torchvision import datasets
from scipy import stats
import cma

DEVICE_TRAIN = 'mps' if torch.backends.mps.is_available() else 'cpu'
DEVICE_EVAL = 'cpu'
SEEDS = [42, 123, 456, 789, 1000]
t0 = time.time()

# ═══ Data ═══
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
print(f"Data loaded ({time.time()-t0:.1f}s), MPS={DEVICE_TRAIN}", flush=True)

def make_rn(nc=5):
    m = models.resnet18(weights='IMAGENET1K_V1')
    m.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    nn.init.kaiming_normal_(m.conv1.weight, mode='fan_out', nonlinearity='relu')
    nn.init.kaiming_normal_(m.fc.weight); nn.init.zeros_(m.fc.bias)
    return m

def extract_feat(model, X, bs=512):
    """Extract 512-d features using MPS for speed."""
    dev = torch.device(DEVICE_TRAIN)
    model = model.to(dev).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i+bs].to(dev)
            h = model.conv1(xb); h = model.bn1(h); h = model.relu(h); h = model.maxpool(h)
            h = model.layer1(h); h = model.layer2(h); h = model.layer3(h); h = model.layer4(h)
            feats.append(torch.flatten(model.avgpool(h), 1).cpu())
    model = model.to('cpu')
    return torch.cat(feats)

def train_parent(cls_list, seed_p, epochs=20, lr=0.01):
    """Extract backbone features on MPS, train fc on CPU."""
    torch.manual_seed(seed_p); np.random.seed(seed_p); random.seed(seed_p)
    cls_map = {c:i for i,c in enumerate(cls_list)}
    base = make_rn(len(cls_list))
    for p in base.parameters(): p.requires_grad = False
    base.fc.weight.requires_grad = True; base.fc.bias.requires_grad = True
    
    mask = sum(y_tr==c for c in cls_list).bool()
    Xs = X_tr[mask]; ys = torch.tensor([cls_map[y.item()] for y in y_tr[mask]])
    idx = torch.cat([torch.where(ys==i)[0][:2000] for i in range(len(cls_list))])
    Xs, ys = Xs[idx], ys[idx]
    
    feats = extract_feat(base, Xs)
    fc = nn.Linear(512, len(cls_list))
    nn.init.kaiming_normal_(fc.weight); nn.init.zeros_(fc.bias)
    opt = torch.optim.Adam(fc.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    fc.train()
    for ep in range(epochs):
        pm = torch.randperm(len(feats))
        for i in range(0, len(feats), 256):
            ix = pm[i:i+256]
            loss = nn.CrossEntropyLoss()(fc(feats[ix]), ys[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    base.fc = fc; base.eval()
    
    mask_te = sum(y_te==c for c in cls_list).bool()
    Xt = X_te[mask_te]; yt = torch.tensor([cls_map[y.item()] for y in y_te[mask_te]])
    feats_te = extract_feat(base, Xt)
    fc.eval()
    with torch.no_grad():
        preds = fc(feats_te).argmax(1)
        acc = (preds==yt).float().mean().item()
        pc = {cls_list[i]:(preds[yt==i]==i).float().mean().item() for i in range(len(cls_list))}
    return base, acc, pc

def eval_10(preds, y):
    acc = (preds==y).float().mean().item()
    pc = {c:(preds[y==c]==c).float().mean().item() if (y==c).sum()>0 else 0 for c in ALL}
    ok = sum(1 for c in ALL if pc[c]>0.3)
    mn = min(pc[c] for c in ALL)
    aM = np.mean([pc[c] for c in clA]); bM = np.mean([pc[c] for c in clB])
    bal = min(aM,bM)/(max(aM,bM)+1e-10)
    return {'acc':round(acc,4),'ok':ok,'min':round(mn,4),'bal':round(bal,4),
            'pc':{c:round(pc[c],3) for c in ALL}}

# ═══ Run per seed ═══
all_results = {}
for seed in SEEDS:
    st = time.time()
    print(f"\n--- Seed {seed} ---", flush=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    
    mA, accA, pcA = train_parent(clA, seed)
    mB, accB, pcB = train_parent(clB, seed+10000)
    print(f"  Parents: A={accA:.3f} B={accB:.3f}", flush=True)
    
    # Get logits for test data
    mA.eval(); mB.eval()
    with torch.no_grad():
        lA = torch.cat([mA(X_te[i:i+512]) for i in range(0,len(X_te),512)])
        lB = torch.cat([mB(X_te[i:i+512]) for i in range(0,len(X_te),512)])
    
    fcA, fcB = mA.fc, mB.fc
    
    # Method 1: LogitConcat
    pr = torch.cat([lA, lB], dim=1).argmax(1)
    r1 = eval_10(pr, y_te); r1['name'] = 'LogitConcat'
    
    # Method 2: WeightAvg + fc map
    sdA, sdB = mA.state_dict(), mB.state_dict()
    wA, bA = sdA['fc.weight'], sdA['fc.bias']
    wB, bB = sdB['fc.weight'], sdB['fc.bias']
    avgM = make_rn(10); sd_avg = {}
    for k in sdA:
        if 'fc' not in k: sd_avg[k] = 0.5*sdA[k]+0.5*sdB[k]
    fc_w = torch.zeros(10,512); fc_b = torch.zeros(10)
    for i,c in enumerate(clA): fc_w[c]=wA[i]; fc_b[c]=bA[i]
    for i,c in enumerate(clB): fc_w[c]=wB[i]; fc_b[c]=bB[i]
    sd_avg['fc.weight']=fc_w; sd_avg['fc.bias']=fc_b
    avgM.load_state_dict(sd_avg); avgM.eval()
    with torch.no_grad():
        pr = torch.cat([avgM(X_te[i:i+512]).argmax(1) for i in range(0,len(X_te),512)])
    r2 = eval_10(pr, y_te); r2['name'] = 'WeightAvg'
    
    # Method 3: DualProbe
    fA_te = extract_feat(mA, X_te); fB_te = extract_feat(mB, X_te)
    fA_v = extract_feat(mA, X_tr[45000:]); fB_v = extract_feat(mB, X_tr[45000:])
    yv = y_tr[45000:]
    f_cat_v = torch.cat([fA_v,fB_v],1); f_cat_te = torch.cat([fA_te,fB_te],1)
    torch.manual_seed(seed)
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
    
    # Method 4: ENT CMA-ES logit routing
    X_cal, y_cal = X_tr[40000:45000], y_tr[40000:45000]
    with torch.no_grad():
        lA_cal = torch.cat([mA(X_cal[i:i+512]) for i in range(0,len(X_cal),512)])
        lB_cal = torch.cat([mB(X_cal[i:i+512]) for i in range(0,len(X_cal),512)])
    
    def fit_ent(x):
        route = x[:10]; sA_ = x[10]; sB_ = x[11]
        logits = torch.zeros(len(lA_cal), 10)
        for ci,c in enumerate(clA):
            a = 1/(1+np.exp(-route[c])); logits[:,c] = a*sA_*lA_cal[:,ci]
        for ci,c in enumerate(clB):
            a = 1/(1+np.exp(-route[c])); logits[:,c] = (1-a)*sB_*lB_cal[:,ci]
        preds = logits.argmax(1)
        acc = (preds==y_cal).float().mean().item()
        pc_ = {c:(preds[y_cal==c]==c).float().mean().item() for c in ALL if (y_cal==c).sum()>0}
        mn = min(pc_.values()) if pc_ else 0
        ok = sum(1 for v in pc_.values() if v>0.3)
        return -(0.3*acc+0.4*mn+0.2*ok/10+0.1*np.mean(list(pc_.values())))
    
    x0 = np.zeros(12); x0[:5]=3.0; x0[5:10]=-3.0; x0[10]=1.0; x0[11]=1.0
    es = cma.CMAEvolutionStrategy(x0, 0.5,
        {'maxiter':20, 'popsize':12, 'seed':seed, 'verbose':-1})
    bs_f = float('inf'); bx = None
    while not es.stop():
        sols = es.ask(); sc = [fit_ent(x) for x in sols]
        es.tell(sols, sc)
        if min(sc) < bs_f: bs_f = min(sc); bx = sols[np.argmin(sc)]
    
    with torch.no_grad():
        route = bx[:10]; sA_ = bx[10]; sB_ = bx[11]
        logits = torch.zeros(len(lA), 10)
        for ci,c in enumerate(clA):
            a = 1/(1+np.exp(-route[c])); logits[:,c] = a*sA_*lA[:,ci]
        for ci,c in enumerate(clB):
            a = 1/(1+np.exp(-route[c])); logits[:,c] = (1-a)*sB_*lB[:,ci]
        pr = logits.argmax(1)
    r4 = eval_10(pr, y_te); r4['name'] = 'ENT'
    
    elapsed = time.time()-st
    all_m = [r1, r2, r3, r4]
    for r in all_m:
        s = '✅' if r['ok']>=8 else ''
        print(f"  {r['name']:15s}: ok={r['ok']:>2}/10 acc={r['acc']:.3f} bal={r['bal']:.3f} {s}", flush=True)
    
    all_results[seed] = {
        'parentA': round(accA,4), 'parentB': round(accB,4),
        'methods': {r['name']: {k:v for k,v in r.items() if k not in ('name','pc')} for r in all_m},
        'per_class': {r['name']: r['pc'] for r in all_m},
        'time_s': round(elapsed,1)
    }

# ═══ Statistical Analysis ═══
print(f"\n{'='*70}", flush=True)
print("STATISTICAL ANALYSIS (5 seeds)", flush=True)

methods = ['LogitConcat', 'WeightAvg', 'DualProbe', 'ENT']
metrics = {m: {'acc':[],'ok':[],'bal':[],'min':[]} for m in methods}
for seed in SEEDS:
    for m in methods:
        if m in all_results[seed]['methods']:
            for k in ['acc','ok','bal','min']:
                metrics[m][k].append(all_results[seed]['methods'][m][k])

print(f"\n  {'Method':15s} {'Acc':>14} {'OK':>10} {'Balance':>14} {'Min':>10}")
for m in methods:
    d = metrics[m]
    print(f"  {m:15s} {np.mean(d['acc']):.3f}±{np.std(d['acc']):.3f} "
          f"{np.mean(d['ok']):>5.1f}±{np.std(d['ok']):.1f} "
          f"{np.mean(d['bal']):.3f}±{np.std(d['bal']):.3f} "
          f"{np.mean(d['min']):.3f}")

# P-values: ENT vs each baseline
ent_ok = np.array(metrics['ENT']['ok'], dtype=float)
ent_bal = np.array(metrics['ENT']['bal'], dtype=float)
for bl in ['LogitConcat', 'WeightAvg', 'DualProbe']:
    bl_ok = np.array(metrics[bl]['ok'], dtype=float)
    bl_bal = np.array(metrics[bl]['bal'], dtype=float)
    _, p_ok = stats.ttest_rel(ent_ok, bl_ok)
    _, p_bal = stats.ttest_rel(ent_bal, bl_bal)
    print(f"  ENT vs {bl:15s}: ok p={p_ok:.4f} bal p={p_bal:.4f}")

# Parent quality
parA = [all_results[s]['parentA'] for s in SEEDS]
parB = [all_results[s]['parentB'] for s in SEEDS]
print(f"\n  Parent A: {np.mean(parA):.3f}±{np.std(parA):.3f}")
print(f"  Parent B: {np.mean(parB):.3f}±{np.std(parB):.3f}")

elapsed_total = time.time()-t0
print(f"\n  Total: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")

# Save
with open('results/cifar_fix_results.json', 'w') as f:
    json.dump({str(s): r for s, r in all_results.items()}, f, indent=2)

# §K5
for m in methods:
    print(f"\nmetric_{m}_ok_mean: {np.mean(metrics[m]['ok']):.1f}")
    print(f"metric_{m}_acc_mean: {np.mean(metrics[m]['acc']):.4f}")
print(f"metric_parentA_mean: {np.mean(parA):.4f}")
print(f"metric_parentB_mean: {np.mean(parB):.4f}")
print("Done!", flush=True)
