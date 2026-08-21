#!/usr/bin/env python3
"""Task 2: CIFAR-10 ENT — 5 seeds (v3: maximally optimized).
Key changes: tiny val set for EA (500 samples), fewer CMA iters.
"""
import numpy as np, torch, torch.nn as nn, random, copy, time, json, sys
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from scipy import stats
import cma

t0 = time.time()
SEEDS = [42, 123, 456, 789, 1000]

print("Loading CIFAR-10...", flush=True)
from torchvision import datasets

raw_tr = datasets.CIFAR10('/tmp/cifar10', train=True, download=True)
raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)

mean = torch.tensor([0.4914,0.4822,0.4465]).view(3,1,1)
std = torch.tensor([0.247,0.243,0.261]).view(3,1,1)

N_TR, N_TE = 8000, 2000
X_tr = torch.tensor(raw_tr.data[:N_TR]).permute(0,3,1,2).float() / 255.0
X_tr = (X_tr - mean) / std
y_tr = torch.tensor(raw_tr.targets[:N_TR])
X_te = torch.tensor(raw_te.data[:N_TE]).permute(0,3,1,2).float() / 255.0
X_te = (X_te - mean) / std
y_te = torch.tensor(raw_te.targets[:N_TE])

# Small val/cal sets for EA speed
Xv, yv = X_tr[6000:7000], y_tr[6000:7000]  # 1000 for val
Xc = X_tr[:1000]  # 1000 for calibration
clA, clB = list(range(5)), list(range(5,10))
print(f"Data: {time.time()-t0:.1f}s", flush=True)

class CNN(nn.Module):
    def __init__(s, ch1=32, ch2=64, fc=128):
        super().__init__()
        s.features = nn.Sequential(
            nn.Conv2d(3, ch1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(ch1, ch2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        s.classifier = nn.Sequential(
            nn.Linear(ch2*8*8, fc), nn.ReLU(), nn.Linear(fc, 10))
        s.ch1, s.ch2, s.fc_dim = ch1, ch2, fc
    def forward(s, x):
        return s.classifier(s.features(x).view(x.size(0), -1))

def ev(m, X, y):
    m.eval()
    with torch.no_grad(): return (m(X).argmax(1)==y).float().mean().item()
def pc(m, X, y):
    m.eval()
    with torch.no_grad(): p = m(X).argmax(1)
    return {c:(p[y==c]==c).float().mean().item() if (y==c).sum()>0 else 0 for c in range(10)}
def evaluate(m, name):
    pcM = pc(m,X_te,y_te); acc = ev(m,X_te,y_te)
    aM=np.mean([pcM[c] for c in clA]); bM=np.mean([pcM[c] for c in clB])
    bal=min(aM,bM)/(max(aM,bM)+1e-10); mn=min(pcM[c] for c in range(10))
    ok=sum(1 for c in range(10) if pcM[c]>0.2)
    return {'name':name,'acc':round(acc,4),'bal':round(bal,4),'min':round(mn,4),
            'ok':ok,'A':round(aM,4),'B':round(bM,4),'pc':{c:round(pcM[c],3) for c in range(10)}}

def train_cnn(model, X, y, cls, epochs=12):
    mask = sum(y==c for c in cls).bool()
    Xs, ys = X[mask][:2000], y[mask][:2000]
    opt = torch.optim.Adam(model.parameters(), lr=0.003); model.train()
    for ep in range(epochs):
        idx_ = torch.randperm(len(Xs))[:256]
        l = nn.CrossEntropyLoss()(model(Xs[idx_]), ys[idx_])
        opt.zero_grad(); l.backward(); opt.step()
    model.eval(); return model

def prune_cnn(model, tch1, tch2, tfc):
    W1=model.features[0].weight.data; b1=model.features[0].bias.data
    idx1=W1.abs().view(W1.shape[0],-1).sum(1).argsort(descending=True)[:tch1].sort().values
    W2=model.features[3].weight.data; b2=model.features[3].bias.data
    idx2=W2.abs().view(W2.shape[0],-1).sum(1).argsort(descending=True)[:tch2].sort().values
    Wf=model.classifier[0].weight.data; bf=model.classifier[0].bias.data
    idx_fc=Wf.abs().sum(1).argsort(descending=True)[:tfc].sort().values
    p=CNN(tch1,tch2,tfc)
    with torch.no_grad():
        p.features[0].weight.copy_(W1[idx1]); p.features[0].bias.copy_(b1[idx1])
        p.features[3].weight.copy_(W2[idx2][:,idx1]); p.features[3].bias.copy_(b2[idx2])
        flat_idx=torch.cat([torch.arange(i*64,(i+1)*64) for i in idx2])
        p.classifier[0].weight.copy_(Wf[idx_fc][:,flat_idx]); p.classifier[0].bias.copy_(bf[idx_fc])
        p.classifier[2].weight.copy_(model.classifier[2].weight.data[:,idx_fc])
        p.classifier[2].bias.copy_(model.classifier[2].bias.data)
    return p

def merge_cnns(cA, cB, route, rA, rB):
    ch1,ch2,fc=cA.ch1,cA.ch2,cA.fc_dim
    m=CNN(ch1*2,ch2*2,fc*2)
    with torch.no_grad():
        m.features[0].weight.zero_();m.features[0].bias.zero_()
        m.features[0].weight[:ch1]=cA.features[0].weight; m.features[0].weight[ch1:]=cB.features[0].weight
        m.features[0].bias[:ch1]=cA.features[0].bias; m.features[0].bias[ch1:]=cB.features[0].bias
        m.features[3].weight.zero_();m.features[3].bias.zero_()
        m.features[3].weight[:ch2,:ch1]=cA.features[3].weight; m.features[3].weight[ch2:,ch1:]=cB.features[3].weight
        m.features[3].bias[:ch2]=cA.features[3].bias; m.features[3].bias[ch2:]=cB.features[3].bias
        m.classifier[0].weight.zero_();m.classifier[0].bias.zero_()
        m.classifier[0].weight[:fc,:ch2*64]=cA.classifier[0].weight; m.classifier[0].bias[:fc]=cA.classifier[0].bias
        m.classifier[0].weight[fc:,ch2*64:]=cB.classifier[0].weight; m.classifier[0].bias[fc:]=cB.classifier[0].bias
        m.classifier[2].weight.zero_();m.classifier[2].bias.zero_()
        for c in range(10):
            a=1/(1+np.exp(-route[c]))
            m.classifier[2].weight[c,:fc]=a*rA*cA.classifier[2].weight[c]
            m.classifier[2].weight[c,fc:]=(1-a)*rB*cB.classifier[2].weight[c]
            m.classifier[2].bias[c]=a*rA*cA.classifier[2].bias[c]+(1-a)*rB*cB.classifier[2].bias[c]
    return m

all_results = {}
for seed in SEEDS:
    st = time.time()
    print(f"\n--- SEED={seed} ---", flush=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    
    base = CNN()
    opt = torch.optim.Adam(base.parameters(), lr=0.003); base.train()
    for ep in range(6):
        idx_ = torch.randperm(6000)[:256]
        l = nn.CrossEntropyLoss()(base(X_tr[idx_]), y_tr[idx_])
        opt.zero_grad(); l.backward(); opt.step()
    base.eval(); base_sd = copy.deepcopy(base.state_dict())
    
    ftA = copy.deepcopy(base); ftA = train_cnn(ftA, X_tr, y_tr, clA)
    ftB = copy.deepcopy(base); ftB = train_cnn(ftB, X_tr, y_tr, clB)
    sdA = ftA.state_dict(); sdB = ftB.state_dict()
    
    results = {}
    # Average
    m=CNN(); sd={k:0.5*sdA[k]+0.5*sdB[k] for k in sdA}; m.load_state_dict(sd); m.eval()
    results['Average'] = evaluate(m,"Avg")
    # TA
    best_ta = None; best_bal = -1
    for tau in [0.3, 0.5]:
        m=CNN(); sd={}
        for k in base_sd: sd[k]=base_sd[k]+tau*((sdA[k]-base_sd[k])+(sdB[k]-base_sd[k]))
        m.load_state_dict(sd); m.eval(); r=evaluate(m,f"TA")
        if r['bal']>best_bal: best_bal=r['bal']; best_ta=r
    results['TA'] = best_ta
    # Sakana
    def build_sak(x):
        a=1/(1+np.exp(-np.array(x))); m=CNN(); sd={}
        for k in sdA.keys():
            gi = 0 if 'features.0' in k else (1 if 'features.3' in k else (2 if 'classifier.0' in k else 3))
            sd[k]=a[gi]*sdA[k]+(1-a[gi])*sdB[k]
        m.load_state_dict(sd); m.eval(); return m
    es=cma.CMAEvolutionStrategy(np.zeros(4),1.5,{'maxiter':8,'popsize':6,'seed':seed,'verbose':-1})
    best_s=-1;best_x=None
    while not es.stop():
        sols=es.ask();sc=[]
        for x in sols:
            m=build_sak(x);d=pc(m,Xv,yv);a_=ev(m,Xv,yv)
            sc.append(-(0.4*a_+0.4*min(d[c] for c in range(10))+0.1*np.mean([d[c] for c in range(10)])))
        es.tell(sols,sc)
        if -min(sc)>best_s: best_s=-min(sc);best_x=sols[np.argmin(sc)]
    results['Sakana'] = evaluate(build_sak(best_x),"Sak")
    
    # ENT
    tch1,tch2,tfc = 24,48,96
    pA=prune_cnn(ftA,tch1,tch2,tfc); pB=prune_cnn(ftB,tch1,tch2,tfc)
    pA.eval();pB.eval()
    with torch.no_grad(): sA_=pA(Xc).numpy().std();sB_=pB(Xc).numpy().std()
    t_=(sA_+sB_)/2;rA_=t_/(sA_+1e-10);rB_=t_/(sB_+1e-10)
    
    torch.manual_seed(seed)
    pop=[np.array([2.]*5+[-2.]*5)]+[np.random.randn(10)*1.5 for _ in range(11)]
    bf=-1;bc=None
    for gen in range(20):
        fs=[]
        for r_ in pop:
            m=merge_cnns(pA,pB,r_,rA_,rB_); d=pc(m,Xv,yv); a_=ev(m,Xv,yv)
            mn_=min(d[c] for c in range(10))
            fs.append(0.5*mn_ + 0.3*np.exp(np.mean(np.log([max(d[c],1e-6) for c in range(10)]))) + 0.2*a_)
        gi=np.argmax(fs)
        if fs[gi]>bf: bf=fs[gi];bc=pop[gi].copy()
        new=[bc.copy()]+[pop[random.randint(0,len(pop)-1)]+np.random.randn(10)*0.3 for _ in range(11)]
        pop=new
    
    results['ENT'] = evaluate(merge_cnns(pA,pB,bc,rA_,rB_),"ENT")
    all_results[seed] = results
    r = results['ENT']
    print(f"  ENT: ok={r['ok']}/10 bal={r['bal']:.3f} min={r['min']:.3f} | Avg ok={results['Average']['ok']} | TA ok={results['TA']['ok']} | {time.time()-st:.0f}s", flush=True)

# ═══════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
methods = ['Average', 'TA', 'Sakana', 'ENT']
metrics = {m: {'acc':[],'bal':[],'min':[],'ok':[]} for m in methods}
for seed, res in sorted(all_results.items()):
    for m in methods:
        if m in res:
            for k in ['acc','bal','min','ok']: metrics[m][k].append(res[m][k])

for m in methods:
    d=metrics[m]
    if d['acc']:
        print(f"  {m:<8}: acc={np.mean(d['acc']):.3f}±{np.std(d['acc']):.3f} bal={np.mean(d['bal']):.3f}±{np.std(d['bal']):.3f} ok={np.mean(d['ok']):.1f}±{np.std(d['ok']):.1f} min={np.mean(d['min']):.3f}±{np.std(d['min']):.3f}")

ent_ok = np.array(metrics['ENT']['ok'],dtype=float)
for bl in ['Average','TA','Sakana']:
    b_ok = np.array(metrics[bl]['ok'],dtype=float)
    n=min(len(ent_ok),len(b_ok))
    if n>=2:
        _,p=stats.ttest_rel(ent_ok[:n],b_ok[:n])
        print(f"  ENT vs {bl}: ok p={p:.4f}")

elapsed = time.time()-t0
print(f"\n  Total: {elapsed:.0f}s")

with open('results/task2_cifar_seeds.json','w') as f:
    json.dump({
        'per_seed': {str(s): {m: {k:v for k,v in r.items() if k!='pc'} for m,r in res.items()} for s,res in all_results.items()},
        'summary': {m: {'acc':round(float(np.mean(metrics[m]['acc'])),4),'bal':round(float(np.mean(metrics[m]['bal'])),4),'ok':round(float(np.mean(metrics[m]['ok'])),1),'min':round(float(np.mean(metrics[m]['min'])),4)} for m in methods if metrics[m]['acc']},
        'time_s': round(elapsed,1)
    }, f, indent=2)

print(f"\nmetric_ent_ok_mean: {np.mean(metrics['ENT']['ok']):.1f}")
print(f"metric_ent_bal_mean: {np.mean(metrics['ENT']['bal']):.4f}")
print(f"metric_ent_min_mean: {np.mean(metrics['ENT']['min']):.4f}")
print("Done!", flush=True)
