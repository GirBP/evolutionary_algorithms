#!/usr/bin/env python3
"""ENT merge for CIFAR-10 — CORRECT implementation.

Key design:
- Parents pre-trained, loaded from .pth (NO training here)
- Per-BLOCK α interpolation (13 groups, not 120 individual keys)
- MPS for fitness eval forward pass
- Adaptive CMA convergence
- Retention = drop ≤ 10% from parent per-class accuracy

Usage: python3 ent_merge_correct.py <seed>
"""
import numpy as np, torch, torch.nn as nn, random, json, time, sys, copy
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import torchvision.models as models
from torchvision import datasets
import cma

SEED = int(sys.argv[1])
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
t0 = time.time()

DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEV}", flush=True)

# ═══ Data ═══
raw_tr = datasets.CIFAR10('/tmp/cifar10', train=True, download=True)
raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
X_tr = (torch.tensor(raw_tr.data).permute(0,3,1,2).float()/255 - mean)/std
y_tr = torch.tensor(raw_tr.targets)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255 - mean)/std
y_te = torch.tensor(raw_te.targets)

X_cal, y_cal = X_tr[40000:45000].to(DEV), y_tr[40000:45000].to(DEV)

clA, clB = list(range(5)), list(range(5,10))
ALL = list(range(10))
print(f"Data: {time.time()-t0:.1f}s", flush=True)

# ═══ Model ═══
def make_rn(nc=5):
    m = models.resnet18(weights='IMAGENET1K_V1')
    m.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    return m

# ═══ Load parents ═══
mA = make_rn(5); mA.load_state_dict(torch.load(f'results/parentA_s{SEED}.pth', weights_only=True, map_location='cpu')); mA.eval()
mB = make_rn(5); mB.load_state_dict(torch.load(f'results/parentB_s{SEED}.pth', weights_only=True, map_location='cpu')); mB.eval()

with open('results/parents_strong.json') as f:
    pd = json.load(f)[str(SEED)]
parent_pc = {}
for c, a in pd['pcA'].items(): parent_pc[int(c)] = a
for c, a in pd['pcB'].items(): parent_pc[int(c)] = a
print(f"Parents: A={pd['A']:.3f} B={pd['B']:.3f}", flush=True)

sdA, sdB = mA.state_dict(), mB.state_dict()
wA, bA = sdA['fc.weight'], sdA['fc.bias']
wB, bB = sdB['fc.weight'], sdB['fc.bias']

# ═══ BLOCK GROUPS (13, not 120) ═══
GROUPS = ['conv1', 'bn1',
          'layer1.0', 'layer1.1', 'layer2.0', 'layer2.1',
          'layer3.0', 'layer3.1', 'layer4.0', 'layer4.1']
# Map each key to its group index
key_to_group = {}
all_keys = [k for k in sdA if 'fc' not in k and k in sdB]
for k in all_keys:
    for gi, g in enumerate(GROUPS):
        if k.startswith(g):
            key_to_group[k] = gi
            break

n_groups = len(GROUPS)
dim = n_groups + 2  # 10 block alphas + scale_a + scale_b
print(f"ENT: {n_groups} block groups, dim={dim}", flush=True)

# ═══ Build merged model ═══
def build_merged(x):
    """Build ONE ResNet-18(10 classes). x = [n_groups sigmoid alphas, sA, sB]"""
    alphas = 1.0 / (1.0 + np.exp(-x[:n_groups]))
    sA, sB = x[n_groups], x[n_groups+1]
    
    m = make_rn(10)
    sd = {}
    for k in all_keys:
        gi = key_to_group.get(k, 0)
        a = float(alphas[gi])
        sd[k] = (1-a)*sdA[k] + a*sdB[k]
    
    fw = torch.zeros(10, 512); fb = torch.zeros(10)
    for ci,c in enumerate(clA): fw[c]=wA[ci]*sA; fb[c]=bA[ci]*sA
    for ci,c in enumerate(clB): fw[c]=wB[ci]*sB; fb[c]=bB[ci]*sB
    sd['fc.weight'] = fw; sd['fc.bias'] = fb
    m.load_state_dict(sd)
    return m

# ═══ Eval ═══
def eval_model(model, X, y, dev, bs=1024):
    model.eval().to(dev)
    with torch.no_grad():
        preds = torch.cat([model(X[i:i+bs]).argmax(1) for i in range(0,len(X),bs)])
    acc = (preds == y).float().mean().item()
    pc = {}; retained = 0
    for c in ALL:
        mask = y == c
        if mask.sum() == 0: continue
        ca = (preds[mask]==c).float().mean().item()
        pc[c] = ca
        if ca >= 0.9 * parent_pc.get(c, 0):
            retained += 1
    mn = min(pc.values()) if pc else 0
    aM = np.mean([pc.get(c,0) for c in clA])
    bM = np.mean([pc.get(c,0) for c in clB])
    bal = min(aM,bM)/(max(aM,bM)+1e-10)
    return {'acc':round(acc,4), 'retained':retained, 'min':round(mn,4),
            'bal':round(bal,4), 'pc':{c:round(v,4) for c,v in pc.items()}}

# ═══ CMA-ES fitness (on MPS!) ═══
def fitness(x):
    m = build_merged(x)
    m.to(DEV)
    m.eval()
    with torch.no_grad():
        preds = torch.cat([m(X_cal[i:i+1024]).argmax(1) for i in range(0,len(X_cal),1024)])
    acc = (preds == y_cal).float().mean().item()
    pc_vals = []
    ret = 0
    for c in ALL:
        mask = y_cal == c
        if mask.sum() == 0: continue
        ca = (preds[mask]==c).float().mean().item()
        pc_vals.append(ca)
        if ca >= 0.9 * parent_pc.get(c, 0):
            ret += 1
    mn = min(pc_vals) if pc_vals else 0
    return -(0.3*acc + 0.3*(ret/10) + 0.3*mn + 0.1*np.mean(pc_vals))

# ═══ ENT CMA-ES ═══
print(f"\n--- ENT (seed={SEED}) ---", flush=True)
x0 = np.zeros(dim); x0[n_groups]=1.0; x0[n_groups+1]=1.0
popsize = max(10, 4 + int(3*np.log(dim)))
print(f"  CMA: dim={dim}, pop={popsize}", flush=True)

es = cma.CMAEvolutionStrategy(x0, 0.5, {
    'popsize': popsize, 'maxiter': dim*5, 'tolx': 1e-5,
    'tolfun': 1e-8, 'timeout': 180, 'seed': SEED, 'verbose': -1,
})

bf = float('inf'); bx = None; gen = 0
while not es.stop():
    sols = es.ask()
    fvals = [fitness(x) for x in sols]
    es.tell(sols, fvals)
    gbest = min(fvals)
    if gbest < bf: bf = gbest; bx = sols[np.argmin(fvals)].copy()
    gen += 1
    if gen % 5 == 0:
        print(f"    Gen {gen}: fitness={-bf:.4f} ({gen*popsize} evals, {time.time()-t0:.0f}s)", flush=True)

print(f"  Converged: {gen} gens, {gen*popsize} evals, {time.time()-t0:.0f}s", flush=True)

# Final eval on TEST (CPU for consistency)
merged_ent = build_merged(bx)
r_ent = eval_model(merged_ent, X_te, y_te, torch.device('cpu'))
r_ent['name'] = 'ENT-Full'; r_ent['gen'] = gen; r_ent['evals'] = gen*popsize
print(f"  ENT: retained={r_ent['retained']}/10 acc={r_ent['acc']:.3f} bal={r_ent['bal']:.3f} min={r_ent['min']:.3f}", flush=True)
print(f"    pc: {[r_ent['pc'][c] for c in ALL]}", flush=True)

alphas_final = 1.0/(1.0+np.exp(-bx[:n_groups]))
for gi, g in enumerate(GROUPS):
    bar = '█'*int(alphas_final[gi]*20) + '░'*(20-int(alphas_final[gi]*20))
    print(f"    {g:12s} α={alphas_final[gi]:.3f} |{bar}|", flush=True)

# ═══ Baselines (all ONE model) ═══
print(f"\n--- Baselines ---", flush=True)

# 1. Weight Average
avg_m = make_rn(10); sd_a = {}
for k in all_keys: sd_a[k] = 0.5*sdA[k]+0.5*sdB[k]
fw=torch.zeros(10,512); fb=torch.zeros(10)
for ci,c in enumerate(clA): fw[c]=wA[ci]; fb[c]=bA[ci]
for ci,c in enumerate(clB): fw[c]=wB[ci]; fb[c]=bB[ci]
sd_a['fc.weight']=fw; sd_a['fc.bias']=fb
avg_m.load_state_dict(sd_a)
r_avg = eval_model(avg_m, X_te, y_te, torch.device('cpu'))
r_avg['name'] = 'WeightAvg'
print(f"  WeightAvg: ret={r_avg['retained']}/10 acc={r_avg['acc']:.3f}", flush=True)

# 2. Task Arithmetic
base_full = models.resnet18(weights='IMAGENET1K_V1')
base_full.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
base_full.maxpool = nn.Identity()
sd_pre = base_full.state_dict()

best_ta = None
for lam in [0.3, 0.5, 0.7, 1.0]:
    ta_m = make_rn(10); sd_t = {}
    for k in all_keys:
        if k in sd_pre:
            sd_t[k] = sd_pre[k] + lam*((sdA[k]-sd_pre[k])+(sdB[k]-sd_pre[k]))
        else:
            sd_t[k] = 0.5*sdA[k]+0.5*sdB[k]
    sd_t['fc.weight']=fw; sd_t['fc.bias']=fb
    ta_m.load_state_dict(sd_t)
    r = eval_model(ta_m, X_te, y_te, torch.device('cpu'))
    if best_ta is None or r['retained'] > best_ta['retained'] or \
       (r['retained']==best_ta['retained'] and r['acc']>best_ta['acc']):
        best_ta = r; best_ta['name'] = f'TA(λ={lam})'
r_ta = best_ta
print(f"  {r_ta['name']}: ret={r_ta['retained']}/10 acc={r_ta['acc']:.3f}", flush=True)

# 3. TIES
ties_m = make_rn(10); sd_ties = {}
for k in all_keys:
    if k in sd_pre:
        ta_a = sdA[k]-sd_pre[k]; ta_b = sdB[k]-sd_pre[k]
        for t in [ta_a, ta_b]:
            th = torch.quantile(t.abs().float(), 0.7)
            t[t.abs()<th] = 0
        ss = torch.sign(ta_a)+torch.sign(ta_b)
        mask = ss != 0
        merged_t = torch.where(mask, (ta_a+ta_b)/2, torch.zeros_like(ta_a))
        sd_ties[k] = sd_pre[k] + merged_t
    else:
        sd_ties[k] = 0.5*sdA[k]+0.5*sdB[k]
sd_ties['fc.weight']=fw; sd_ties['fc.bias']=fb
ties_m.load_state_dict(sd_ties)
r_ties = eval_model(ties_m, X_te, y_te, torch.device('cpu'))
r_ties['name'] = 'TIES'
print(f"  TIES: ret={r_ties['retained']}/10 acc={r_ties['acc']:.3f}", flush=True)

# ═══ Summary ═══
elapsed = time.time()-t0
all_m = [r_ent, r_avg, r_ta, r_ties]
print(f"\n{'='*60}")
print(f"SEED={SEED} Parents: A={pd['A']:.3f} B={pd['B']:.3f} ({elapsed:.0f}s)")
print(f"{'Method':20s} {'Ret':>6} {'Acc':>8} {'Bal':>8} {'Min':>8}")
for r in all_m:
    f = '✅' if r['retained']>=8 else ''
    print(f"{r['name']:20s} {r['retained']:>2}/10   {r['acc']:.4f}   {r['bal']:.4f}   {r['min']:.4f}  {f}")

# Save
result = {'seed':SEED, 'parentA':pd['A'], 'parentB':pd['B'], 'parent_pc':parent_pc,
          'methods':{r['name']:{k:v for k,v in r.items() if k!='name'} for r in all_m},
          'ent_alphas':alphas_final.tolist(), 'ent_scales':[float(bx[n_groups]),float(bx[n_groups+1])],
          'time_s':round(elapsed,1)}
fpath = 'results/merge_results.json'
try:
    with open(fpath) as f: acc = json.load(f)
except: acc = {}
acc[str(SEED)] = result
with open(fpath,'w') as f: json.dump(acc, f, indent=2)

print(f"\nmetric_ent_retained: {r_ent['retained']}")
print(f"metric_ent_acc: {r_ent['acc']}")
print(f"metric_ent_bal: {r_ent['bal']}")
print("Done!", flush=True)
