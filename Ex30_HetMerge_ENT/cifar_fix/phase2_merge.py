#!/usr/bin/env python3
"""Phase 2: Load trained parents + run all merge methods.
Usage: python3 phase2_merge.py <seed>
"""
import numpy as np, torch, torch.nn as nn, random, json, time, sys
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import torchvision.models as models
from torchvision import datasets
import cma

SEED = int(sys.argv[1])
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
t0 = time.time()

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

def make_rn(nc=5):
    m = models.resnet18(weights='IMAGENET1K_V1')
    m.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    return m

# Load parents
mA = make_rn(5); mA.load_state_dict(torch.load(f'results/parentA_s{SEED}.pth', weights_only=True)); mA.eval()
mB = make_rn(5); mB.load_state_dict(torch.load(f'results/parentB_s{SEED}.pth', weights_only=True)); mB.eval()
print(f"Parents loaded ({time.time()-t0:.1f}s)", flush=True)

sdA, sdB = mA.state_dict(), mB.state_dict()
wA, bA = sdA['fc.weight'], sdA['fc.bias']
wB, bB = sdB['fc.weight'], sdB['fc.bias']

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
    ok=sum(1 for c in ALL if pc[c]>0.3); mn=min(pc[c] for c in ALL)
    aM=np.mean([pc[c] for c in clA]); bM=np.mean([pc[c] for c in clB])
    bal=min(aM,bM)/(max(aM,bM)+1e-10)
    return {'acc':round(acc,4),'ok':ok,'min':round(mn,4),'bal':round(bal,4),
            'pc':{c:round(pc[c],3) for c in ALL}}

print(f"--- Merge Methods (seed={SEED}) ---", flush=True)

# Get logits for all test data
with torch.no_grad():
    lA=torch.cat([mA(X_te[i:i+512]) for i in range(0,len(X_te),512)])
    lB=torch.cat([mB(X_te[i:i+512]) for i in range(0,len(X_te),512)])
print(f"  Logits computed ({time.time()-t0:.0f}s)", flush=True)

# Method 1: LogitConcat
pr = torch.cat([lA, lB], dim=1).argmax(1)
r1 = eval_10(pr, y_te); r1['name'] = 'LogitConcat'
print(f"  LogitConcat: ok={r1['ok']}/10 acc={r1['acc']:.3f} bal={r1['bal']:.3f}", flush=True)

# Method 2: WeightAvg + fc map
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
print(f"  WeightAvg: ok={r2['ok']}/10 acc={r2['acc']:.3f} bal={r2['bal']:.3f}", flush=True)

# Method 3: DualProbe
fA_te = get_feat(mA, X_te); fB_te = get_feat(mB, X_te)
fA_v = get_feat(mA, X_tr[45000:]); fB_v = get_feat(mB, X_tr[45000:])
yv = y_tr[45000:]
f_cat_v = torch.cat([fA_v, fB_v], 1)
f_cat_te = torch.cat([fA_te, fB_te], 1)

torch.manual_seed(SEED)
probe = nn.Linear(1024, 10)
op = torch.optim.Adam(probe.parameters(), lr=0.01)
probe.train()
for ep in range(300):
    pm = torch.randperm(len(f_cat_v))[:512]
    loss = nn.CrossEntropyLoss()(probe(f_cat_v[pm]), yv[pm])
    op.zero_grad(); loss.backward(); op.step()
probe.eval()
with torch.no_grad(): pr = probe(f_cat_te).argmax(1)
r3 = eval_10(pr, y_te); r3['name'] = 'DualProbe'
print(f"  DualProbe: ok={r3['ok']}/10 acc={r3['acc']:.3f} bal={r3['bal']:.3f}", flush=True)

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
    pc = {c:(preds[y_cal==c]==c).float().mean().item() for c in ALL if (y_cal==c).sum()>0}
    mn = min(pc.values()) if pc else 0
    ok = sum(1 for v in pc.values() if v>0.3)
    return -(0.3*acc+0.4*mn+0.2*ok/10+0.1*np.mean(list(pc.values())))

x0 = np.zeros(12); x0[:5]=3.0; x0[5:10]=-3.0; x0[10]=1.0; x0[11]=1.0
es = cma.CMAEvolutionStrategy(x0, 0.5, {'maxiter':25, 'popsize':14, 'seed':SEED, 'verbose':-1})
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
print(f"  ENT: ok={r4['ok']}/10 acc={r4['acc']:.3f} bal={r4['bal']:.3f}", flush=True)

# Method 5: ENT per-layer backbone α + logit routing
layer_keys = [k for k in sdA if 'fc' not in k and k in sdB]
n_lk = len(layer_keys)

def build_ent_backbone(x):
    alphas = 1/(1+np.exp(-np.array(x[:n_lk])))
    route = x[n_lk:n_lk+10]; sA_ = x[n_lk+10]; sB_ = x[n_lk+11]
    m = make_rn(10); sd = {}
    for i,k in enumerate(layer_keys):
        sd[k] = alphas[i]*sdA[k]+(1-alphas[i])*sdB[k]
    fw = torch.zeros(10,512); fb = torch.zeros(10)
    for ci,c in enumerate(clA): fw[c]=wA[ci]*sA_; fb[c]=bA[ci]*sA_
    for ci,c in enumerate(clB): fw[c]=wB[ci]*sB_; fb[c]=bB[ci]*sB_
    sd['fc.weight']=fw; sd['fc.bias']=fb
    m.load_state_dict(sd); return m

def fit_ent_bb(x):
    m = build_ent_backbone(x); m.eval()
    with torch.no_grad():
        pr = torch.cat([m(X_cal[i:i+512]).argmax(1) for i in range(0,len(X_cal),512)])
    acc = (pr==y_cal).float().mean().item()
    pc = {c:(pr[y_cal==c]==c).float().mean().item() for c in ALL if (y_cal==c).sum()>0}
    mn = min(pc.values()) if pc else 0
    ok = sum(1 for v in pc.values() if v>0.3)
    return -(0.3*acc+0.4*mn+0.2*ok/10+0.1*np.mean(list(pc.values())))

x0_bb = np.zeros(n_lk+12)
x0_bb[n_lk:n_lk+5] = 3.0; x0_bb[n_lk+5:n_lk+10] = -3.0
x0_bb[n_lk+10] = 1.0; x0_bb[n_lk+11] = 1.0
es2 = cma.CMAEvolutionStrategy(x0_bb, 0.3,
    {'maxiter':8, 'popsize':8, 'seed':SEED, 'verbose':-1})
bs2 = float('inf'); bx2 = None
while not es2.stop():
    sols = es2.ask(); sc = [fit_ent_bb(x) for x in sols]
    es2.tell(sols, sc)
    if min(sc) < bs2: bs2 = min(sc); bx2 = sols[np.argmin(sc)]

m5 = build_ent_backbone(bx2); m5.eval()
with torch.no_grad():
    pr = torch.cat([m5(X_te[i:i+512]).argmax(1) for i in range(0,len(X_te),512)])
r5 = eval_10(pr, y_te); r5['name'] = 'ENT-Backbone'
print(f"  ENT-BB: ok={r5['ok']}/10 acc={r5['acc']:.3f} bal={r5['bal']:.3f}", flush=True)

# ═══ Save ═══
elapsed = time.time()-t0
all_m = [r1,r2,r3,r4,r5]

# Load parent accuracies
try:
    with open('results/parents_acc.json') as f: pdata = json.load(f)[str(SEED)]
    accA, accB = pdata['parentA'], pdata['parentB']
except: accA, accB = 0, 0

res = {'seed':SEED, 'parentA':accA, 'parentB':accB,
       'methods':{r['name']:{k:v for k,v in r.items() if k not in ('name','pc')} for r in all_m},
       'per_class':{r['name']:r['pc'] for r in all_m},
       'time_merge': round(elapsed,1)}

fpath = 'results/all_seeds.json'
try:
    with open(fpath) as f: accum = json.load(f)
except: accum = {}
accum[str(SEED)] = res
with open(fpath,'w') as f: json.dump(accum, f, indent=2)

print(f"\n  === seed={SEED} ({elapsed:.0f}s) ===")
print(f"  Parents: A={accA:.3f} B={accB:.3f}")
for r in all_m:
    s = '✅' if r['ok']>=8 else ''
    print(f"  {r['name']:15s}: ok={r['ok']:>2}/10 acc={r['acc']:.3f} bal={r['bal']:.3f} min={r['min']:.3f} {s}")
    print(f"    pc: {[r['pc'][c] for c in ALL]}")

for r in all_m:
    print(f"metric_{r['name']}_ok: {r['ok']}")
print("Done!", flush=True)
