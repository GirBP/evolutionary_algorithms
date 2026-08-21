#!/usr/bin/env python3
"""All 5 seeds: logit-space merge + statistics.
Each seed ~33s → ~165s total. Well within 420s.
"""
import numpy as np, torch, torch.nn as nn, random, json, time, sys
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import torchvision.models as models
from torchvision import datasets
from scipy import stats
import cma

DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1000]
t0 = time.time()

raw_tr = datasets.CIFAR10('/tmp/cifar10', train=True, download=True)
raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
std_ = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_tr = (torch.tensor(raw_tr.data).permute(0,3,1,2).float()/255-mean)/std_
y_tr = torch.tensor(raw_tr.targets)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255-mean)/std_
y_te = torch.tensor(raw_te.targets)
clA, clB = list(range(5)), list(range(5,10))
ALL = list(range(10))
X_cal, y_cal = X_tr[40000:45000], y_tr[40000:45000]
print(f"Data: {time.time()-t0:.1f}s MPS={DEV}", flush=True)

def make_rn(nc=5):
    m = models.resnet18(weights='IMAGENET1K_V1')
    m.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    return m

with open('results/parents_strong.json') as f: pdata = json.load(f)

def get_logits(model, X, bs=512):
    model.to(DEV).eval()
    with torch.no_grad():
        l = torch.cat([model(X[i:i+bs].to(DEV)).cpu() for i in range(0,len(X),bs)])
    model.cpu()
    return l

def get_feat(model, X, bs=512):
    model.to(DEV).eval()
    feats = []
    with torch.no_grad():
        for i in range(0,len(X),bs):
            xb = X[i:i+bs].to(DEV)
            h=model.conv1(xb);h=model.bn1(h);h=model.relu(h);h=model.maxpool(h)
            h=model.layer1(h);h=model.layer2(h);h=model.layer3(h);h=model.layer4(h)
            feats.append(torch.flatten(model.avgpool(h),1).cpu())
    model.cpu()
    return torch.cat(feats)

def eval_10(preds, y, ppc):
    acc=(preds==y).float().mean().item()
    pc={c:(preds[y==c]==c).float().mean().item() if (y==c).sum()>0 else 0 for c in ALL}
    retained = sum(1 for c in ALL if pc[c] >= 0.9*ppc.get(c,0))
    ok = sum(1 for c in ALL if pc[c]>0.3)
    mn = min(pc[c] for c in ALL)
    aM=np.mean([pc[c] for c in clA]);bM=np.mean([pc[c] for c in clB])
    bal=min(aM,bM)/(max(aM,bM)+1e-10)
    return {'acc':round(acc,4),'retained':retained,'ok':ok,'min':round(mn,4),
            'bal':round(bal,4),'pc':{c:round(pc[c],4) for c in ALL}}

all_results = {}
for seed in SEEDS:
    st = time.time()
    print(f"\n--- Seed {seed} ---", flush=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    
    pd = pdata[str(seed)]
    ppc = {}
    for c,a in pd['pcA'].items(): ppc[int(c)] = a
    for c,a in pd['pcB'].items(): ppc[int(c)] = a
    
    mA = make_rn(5); mA.load_state_dict(torch.load(f'results/parentA_s{seed}.pth', weights_only=True, map_location='cpu'))
    mB = make_rn(5); mB.load_state_dict(torch.load(f'results/parentB_s{seed}.pth', weights_only=True, map_location='cpu'))
    
    lA_te = get_logits(mA, X_te); lB_te = get_logits(mB, X_te)
    lA_c = get_logits(mA, X_cal); lB_c = get_logits(mB, X_cal)
    
    # M1: LogitConcat
    pr = torch.cat([lA_te, lB_te], dim=1).argmax(1)
    r1 = eval_10(pr, y_te, ppc); r1['name'] = 'LogitConcat'
    
    # M2: DualProbe
    fA_te = get_feat(mA, X_te); fB_te = get_feat(mB, X_te)
    fA_v = get_feat(mA, X_tr[45000:]); fB_v = get_feat(mB, X_tr[45000:]); yv = y_tr[45000:]
    fcv = torch.cat([fA_v,fB_v],1); fct = torch.cat([fA_te,fB_te],1)
    torch.manual_seed(seed)
    probe = nn.Linear(1024,10); op = torch.optim.Adam(probe.parameters(), lr=0.01)
    probe.train()
    for ep in range(300):
        pm = torch.randperm(len(fcv))[:512]
        loss = nn.CrossEntropyLoss()(probe(fcv[pm]), yv[pm])
        op.zero_grad(); loss.backward(); op.step()
    probe.eval()
    with torch.no_grad(): pr = probe(fct).argmax(1)
    r2 = eval_10(pr, y_te, ppc); r2['name'] = 'DualProbe'
    
    # M3: ENT logit CMA-ES
    def fit_ent(x):
        route=x[:10]; sA=x[10]; sB=x[11]; tA=max(x[12],0.1); tB=max(x[13],0.1)
        logits=torch.zeros(len(lA_c),10)
        for ci,c in enumerate(clA):
            a=1/(1+np.exp(-route[c])); logits[:,c]=a*sA*lA_c[:,ci]/tA
        for ci,c in enumerate(clB):
            a=1/(1+np.exp(-route[c])); logits[:,c]=(1-a)*sB*lB_c[:,ci]/tB
        preds=logits.argmax(1)
        acc=(preds==y_cal).float().mean().item()
        pc_={c:(preds[y_cal==c]==c).float().mean().item() for c in ALL if (y_cal==c).sum()>0}
        ret=sum(1 for c in ALL if pc_.get(c,0)>=0.9*ppc.get(c,0))
        mn=min(pc_.values()) if pc_ else 0
        return -(0.3*acc+0.3*(ret/10)+0.3*mn+0.1*np.mean(list(pc_.values())))
    
    x0=np.zeros(14);x0[:5]=3.0;x0[5:10]=-3.0;x0[10]=1.0;x0[11]=1.0;x0[12]=1.0;x0[13]=1.0
    es=cma.CMAEvolutionStrategy(x0,0.5,{'maxiter':30,'popsize':16,'seed':seed,'verbose':-1})
    bf=float('inf');bx=None
    while not es.stop():
        sols=es.ask();sc=[fit_ent(x) for x in sols]
        es.tell(sols,sc)
        if min(sc)<bf: bf=min(sc);bx=sols[np.argmin(sc)]
    
    with torch.no_grad():
        route=bx[:10];sA=bx[10];sB=bx[11];tA=max(bx[12],0.1);tB=max(bx[13],0.1)
        logits=torch.zeros(len(lA_te),10)
        for ci,c in enumerate(clA):
            a=1/(1+np.exp(-route[c]));logits[:,c]=a*sA*lA_te[:,ci]/tA
        for ci,c in enumerate(clB):
            a=1/(1+np.exp(-route[c]));logits[:,c]=(1-a)*sB*lB_te[:,ci]/tB
        pr=logits.argmax(1)
    r3 = eval_10(pr, y_te, ppc); r3['name'] = 'ENT-Logit'
    
    all_m = [r1,r2,r3]
    for r in all_m:
        s='✅' if r['retained']>=8 else ''
        print(f"  {r['name']:15s}: ret={r['retained']:>2}/10 ok={r['ok']:>2}/10 acc={r['acc']:.3f} {s}", flush=True)
    
    all_results[seed] = {
        'parentA': pd['A'], 'parentB': pd['B'],
        'methods': {r['name']:{k:v for k,v in r.items() if k!='name'} for r in all_m},
        'per_class': {r['name']: r['pc'] for r in all_m}
    }

# ═══ Stats ═══
print(f"\n{'='*70}")
print("STATISTICAL ANALYSIS — STRONG PARENTS (5 seeds)")
methods = ['LogitConcat', 'DualProbe', 'ENT-Logit']
M = {m: {'acc':[],'retained':[],'ok':[],'bal':[],'min':[]} for m in methods}
for s in SEEDS:
    for m in methods:
        for k in ['acc','retained','ok','bal','min']:
            M[m][k].append(all_results[s]['methods'][m][k])

print(f"\n  {'Method':15s} {'Ret':>10} {'OK':>10} {'Acc':>14} {'Bal':>14} {'Min':>10}")
for m in methods:
    d=M[m]
    print(f"  {m:15s} {np.mean(d['retained']):5.1f}±{np.std(d['retained']):.1f} "
          f"{np.mean(d['ok']):5.1f}±{np.std(d['ok']):.1f} "
          f"{np.mean(d['acc']):.3f}±{np.std(d['acc']):.3f} "
          f"{np.mean(d['bal']):.3f}±{np.std(d['bal']):.3f} "
          f"{np.mean(d['min']):.3f}")

ent = {k: np.array(M['ENT-Logit'][k], dtype=float) for k in ['acc','retained','ok','bal']}
for bl in ['LogitConcat', 'DualProbe']:
    bld = {k: np.array(M[bl][k], dtype=float) for k in ['acc','retained','ok','bal']}
    _, p_ret = stats.ttest_rel(ent['retained'], bld['retained'])
    _, p_acc = stats.ttest_rel(ent['acc'], bld['acc'])
    _, p_bal = stats.ttest_rel(ent['bal'], bld['bal'])
    print(f"  ENT vs {bl:15s}: ret p={p_ret:.4f} acc p={p_acc:.4f} bal p={p_bal:.4f}")

pA=[pdata[str(s)]['A'] for s in SEEDS]; pB=[pdata[str(s)]['B'] for s in SEEDS]
print(f"\n  Parent A: {np.mean(pA):.3f}±{np.std(pA):.3f}")
print(f"  Parent B: {np.mean(pB):.3f}±{np.std(pB):.3f}")
print(f"\n  Total: {time.time()-t0:.0f}s")

with open('results/merge_results.json', 'w') as f:
    json.dump({str(s):r for s,r in all_results.items()}, f, indent=2)

for m in methods:
    print(f"\nmetric_{m}_ret: {np.mean(M[m]['retained']):.1f}")
    print(f"metric_{m}_ok: {np.mean(M[m]['ok']):.1f}")
    print(f"metric_{m}_acc: {np.mean(M[m]['acc']):.4f}")
print("Done!", flush=True)
