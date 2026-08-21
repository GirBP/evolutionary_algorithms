#!/usr/bin/env python3
"""Phase 2+3: Load strong parents, run merge methods, statistical analysis.
All 5 seeds × 4 methods. Parents loaded from .pth files.
"""
import numpy as np, torch, torch.nn as nn, random, json, time, sys
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import torchvision.models as models
from torchvision import datasets
from scipy import stats
import cma

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SEEDS = [42, 123, 456, 789, 1000]
t0 = time.time()

raw_tr = datasets.CIFAR10('/tmp/cifar10', train=True, download=True)
raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mn = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
sd_ = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_tr = (torch.tensor(raw_tr.data).permute(0,3,1,2).float()/255.0-mn)/sd_
y_tr = torch.tensor(raw_tr.targets)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255.0-mn)/sd_
y_te = torch.tensor(raw_te.targets)
clA, clB = list(range(5)), list(range(5,10))
ALL = list(range(10))
print(f"Data: {time.time()-t0:.1f}s", flush=True)

def make_rn(nc=5):
    m = models.resnet18(weights='IMAGENET1K_V1')
    m.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    return m

def get_feat(model, X, bs=512):
    model.eval(); feats = []
    with torch.no_grad():
        for i in range(0,len(X),bs):
            xb=X[i:i+bs]
            h=model.conv1(xb);h=model.bn1(h);h=model.relu(h);h=model.maxpool(h)
            h=model.layer1(h);h=model.layer2(h);h=model.layer3(h);h=model.layer4(h)
            feats.append(torch.flatten(model.avgpool(h),1))
    return torch.cat(feats)

def eval_10(preds, y):
    acc=(preds==y).float().mean().item()
    pc={c:(preds[y==c]==c).float().mean().item() if (y==c).sum()>0 else 0 for c in ALL}
    ok=sum(1 for c in ALL if pc[c]>0.3);mn=min(pc[c] for c in ALL)
    aM=np.mean([pc[c] for c in clA]);bM=np.mean([pc[c] for c in clB])
    bal=min(aM,bM)/(max(aM,bM)+1e-10)
    return {'acc':round(acc,4),'ok':ok,'min':round(mn,4),'bal':round(bal,4),
            'pc':{c:round(pc[c],3) for c in ALL}}

all_results = {}
for seed in SEEDS:
    st = time.time()
    print(f"\n--- Seed {seed} ---", flush=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    
    mA = make_rn(5); mA.load_state_dict(torch.load(f'results/parentA_s{seed}.pth', weights_only=True)); mA.eval()
    mB = make_rn(5); mB.load_state_dict(torch.load(f'results/parentB_s{seed}.pth', weights_only=True)); mB.eval()
    
    sdA, sdB = mA.state_dict(), mB.state_dict()
    wA, bA = sdA['fc.weight'], sdA['fc.bias']
    wB, bB = sdB['fc.weight'], sdB['fc.bias']
    
    # Get logits
    with torch.no_grad():
        lA = torch.cat([mA(X_te[i:i+512]) for i in range(0,len(X_te),512)])
        lB = torch.cat([mB(X_te[i:i+512]) for i in range(0,len(X_te),512)])
    
    # M1: LogitConcat
    pr = torch.cat([lA, lB], dim=1).argmax(1)
    r1 = eval_10(pr, y_te); r1['name'] = 'LogitConcat'
    
    # M2: WeightAvg + fc map
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
    
    # M3: DualProbe
    fA_te = get_feat(mA, X_te); fB_te = get_feat(mB, X_te)
    fA_v = get_feat(mA, X_tr[45000:]); fB_v = get_feat(mB, X_tr[45000:])
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
    
    # M4: ENT CMA-ES logit routing
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
        mn_ = min(pc_.values()) if pc_ else 0
        ok_ = sum(1 for v in pc_.values() if v>0.3)
        return -(0.3*acc+0.4*mn_+0.2*ok_/10+0.1*np.mean(list(pc_.values())))
    
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
        print(f"  {r['name']:15s}: ok={r['ok']:>2}/10 acc={r['acc']:.3f} bal={r['bal']:.3f} min={r['min']:.3f} {s}", flush=True)
    
    all_results[seed] = {
        'methods': {r['name']: {k:v for k,v in r.items() if k not in ('name','pc')} for r in all_m},
        'per_class': {r['name']: r['pc'] for r in all_m},
        'time_s': round(elapsed,1)
    }

# ═══ Statistics ═══
print(f"\n{'='*70}", flush=True)
print("STATISTICAL ANALYSIS (5 seeds, STRONG parents)", flush=True)

methods = ['LogitConcat', 'WeightAvg', 'DualProbe', 'ENT']
metrics = {m: {'acc':[],'ok':[],'bal':[],'min':[]} for m in methods}
for seed in SEEDS:
    for m in methods:
        for k in ['acc','ok','bal','min']:
            metrics[m][k].append(all_results[seed]['methods'][m][k])

print(f"\n  {'Method':15s} {'Acc':>14} {'OK':>10} {'Balance':>14} {'Min':>10}")
for m in methods:
    d = metrics[m]
    print(f"  {m:15s} {np.mean(d['acc']):.3f}±{np.std(d['acc']):.3f} "
          f"{np.mean(d['ok']):>5.1f}±{np.std(d['ok']):.1f} "
          f"{np.mean(d['bal']):.3f}±{np.std(d['bal']):.3f} "
          f"{np.mean(d['min']):.3f}")

ent_ok = np.array(metrics['ENT']['ok'], dtype=float)
ent_bal = np.array(metrics['ENT']['bal'], dtype=float)
ent_acc = np.array(metrics['ENT']['acc'], dtype=float)
for bl in ['LogitConcat', 'WeightAvg', 'DualProbe']:
    bl_ok = np.array(metrics[bl]['ok'], dtype=float)
    bl_bal = np.array(metrics[bl]['bal'], dtype=float)
    bl_acc = np.array(metrics[bl]['acc'], dtype=float)
    _, p_ok = stats.ttest_rel(ent_ok, bl_ok)
    _, p_bal = stats.ttest_rel(ent_bal, bl_bal)
    _, p_acc = stats.ttest_rel(ent_acc, bl_acc)
    print(f"  ENT vs {bl:15s}: ok p={p_ok:.4f} bal p={p_bal:.4f} acc p={p_acc:.4f}")

# Load parent data
with open('results/parents_strong.json') as f: pd = json.load(f)
pA = [pd[str(s)]['A'] for s in SEEDS]
pB = [pd[str(s)]['B'] for s in SEEDS]
print(f"\n  Parent A: {np.mean(pA):.3f}±{np.std(pA):.3f}")
print(f"  Parent B: {np.mean(pB):.3f}±{np.std(pB):.3f}")

elapsed_total = time.time()-t0
print(f"\n  Total merge: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")

with open('results/merge_strong_results.json', 'w') as f:
    json.dump({str(s): r for s, r in all_results.items()}, f, indent=2)

for m in methods:
    print(f"\nmetric_{m}_ok: {np.mean(metrics[m]['ok']):.1f}")
    print(f"metric_{m}_acc: {np.mean(metrics[m]['acc']):.4f}")
    print(f"metric_{m}_bal: {np.mean(metrics[m]['bal']):.4f}")
print("Done!", flush=True)
