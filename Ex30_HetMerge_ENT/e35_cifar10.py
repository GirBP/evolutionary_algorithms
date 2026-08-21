#!/usr/bin/env python3
"""E35: CIFAR-10 BENCHMARK — ENT vs ALL SOTA on CNN.

Shared pre-trained base → fine-tune → merge (fair comparison).
Architecture: small CNN (Conv32→Conv64→FC128→FC10)
Split: classes 0-4 vs 5-9
"""
import numpy as np, torch, torch.nn as nn, random, copy, time
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets, transforms
import cma

R = open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/results_e35.txt', 'w')
def log(s): R.write(s+'\n'); R.flush(); print(s, flush=True)

log("="*70)
log("E35: CIFAR-10 BENCHMARK — ENT vs ALL SOTA (CNN)")
log("="*70)

# Data
tf_tr = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
tf_te = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
tr = datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=tf_tr)
te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True, transform=tf_te)

N_TR = 15000; N_TE = 2000
X_tr = torch.stack([tr[i][0] for i in range(N_TR)]); y_tr = torch.tensor([tr[i][1] for i in range(N_TR)])
X_te = torch.stack([te[i][0] for i in range(N_TE)]); y_te = torch.tensor([te[i][1] for i in range(N_TE)])
Xv, yv = X_tr[12000:15000], y_tr[12000:15000]
Xc = X_tr[:2000]
log(f"Data: train={N_TR}, test={N_TE}, val=3000, calib=2000")

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
    def head_features(s, x):
        h = s.features(x).view(x.size(0), -1)
        for m in list(s.classifier)[:-1]: h = m(h)
        return h

clA, clB = list(range(5)), list(range(5, 10))
# CIFAR-10: 0=airplane,1=auto,2=bird,3=cat,4=deer | 5=dog,6=frog,7=horse,8=ship,9=truck

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

def train_cnn(model, X, y, cls, epochs=20, lr=0.003):
    mask = sum(y==c for c in cls).bool()
    Xs, ys = X[mask][:5000], y[mask][:5000]
    opt = torch.optim.Adam(model.parameters(), lr=lr); model.train()
    for ep in range(epochs):
        idx = torch.randperm(len(Xs))[:256]
        l = nn.CrossEntropyLoss()(model(Xs[idx]), ys[idx])
        opt.zero_grad(); l.backward(); opt.step()
    model.eval(); return model

# ═══════════════════════════════════════════
# Pre-train base → fine-tune
# ═══════════════════════════════════════════
log("\n1. Pre-training base CNN on ALL classes (10 epochs)...")
t0 = time.time()
base = CNN()
opt = torch.optim.Adam(base.parameters(), lr=0.003); base.train()
for ep in range(10):
    idx = torch.randperm(10000)[:512]
    l = nn.CrossEntropyLoss()(base(X_tr[idx]), y_tr[idx])
    opt.zero_grad(); l.backward(); opt.step()
base.eval()
base_sd = copy.deepcopy(base.state_dict())
log(f"  Base accuracy: {ev(base,X_te,y_te):.3f}")

log("2. Fine-tuning specialists (20 epochs each)...")
ftA = copy.deepcopy(base); ftA = train_cnn(ftA, X_tr, y_tr, clA, epochs=20)
ftB = copy.deepcopy(base); ftB = train_cnn(ftB, X_tr, y_tr, clB, epochs=20)
log(f"  FT-A (cls 0-4): {ev(ftA,X_te,y_te):.3f}")
log(f"  FT-B (cls 5-9): {ev(ftB,X_te,y_te):.3f}")
pcA_full = pc(ftA, X_te, y_te); pcB_full = pc(ftB, X_te, y_te)
for c in range(10):
    bp = max(pcA_full[c], pcB_full[c])
    log(f"  Class {c}: best_parent={bp:.3f}")

sdA = ftA.state_dict(); sdB = ftB.state_dict()
results = []

# ═══════════════════════════════════════════
# METHOD 1: Average
# ═══════════════════════════════════════════
log("\n--- Average ---")
for a in [0.3, 0.5, 0.7]:
    m=CNN(); sd={k:a*sdA[k]+(1-a)*sdB[k] for k in sdA}; m.load_state_dict(sd); m.eval()
    r=evaluate(m, f"Avg(α={a})"); log(f"  α={a}: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")
    results.append(r)

# ═══════════════════════════════════════════
# METHOD 2: SLERP
# ═══════════════════════════════════════════
log("\n--- SLERP ---")
def slerp(sA,sB,t):
    sd={}
    for k in sA:
        vA,vB=sA[k].float().flatten(),sB[k].float().flatten()
        nA,nB=vA.norm(),vB.norm()
        if nA<1e-8 or nB<1e-8: sd[k]=t*sA[k]+(1-t)*sB[k]; continue
        co=(vA@vB)/(nA*nB); co=co.clamp(-1,1); o=torch.acos(co)
        if o.abs()<1e-6: sd[k]=t*sA[k]+(1-t)*sB[k]
        else: sd[k]=(torch.sin((1-t)*o)/torch.sin(o))*sA[k]+(torch.sin(t*o)/torch.sin(o))*sB[k]
    return sd
m=CNN(); m.load_state_dict(slerp(sdA,sdB,0.5)); m.eval()
r=evaluate(m,"SLERP(0.5)"); results.append(r)
log(f"  SLERP: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")

# ═══════════════════════════════════════════
# METHOD 3: Task Arithmetic
# ═══════════════════════════════════════════
log("\n--- Task Arithmetic ---")
for tau in [0.3, 0.5, 0.7, 1.0, 1.5]:
    m=CNN(); sd={}
    for k in base_sd: sd[k]=base_sd[k]+tau*((sdA[k]-base_sd[k])+(sdB[k]-base_sd[k]))
    m.load_state_dict(sd); m.eval()
    r=evaluate(m,f"TA(τ={tau})"); results.append(r)
    log(f"  τ={tau}: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")

# ═══════════════════════════════════════════
# METHOD 4: TIES-Merging
# ═══════════════════════════════════════════
log("\n--- TIES-Merging ---")
for density in [0.1, 0.2, 0.3, 0.5]:
    m=CNN(); sd={}
    for k in base_sd:
        tvA=sdA[k]-base_sd[k]; tvB=sdB[k]-base_sd[k]
        for tv in [tvA,tvB]:
            flat=tv.flatten(); n_keep=max(1,int(len(flat)*density))
            if n_keep<len(flat):
                thr=flat.abs().topk(n_keep).values[-1]; tv[tv.abs()<thr]=0
        elect=torch.where(tvA.abs()>=tvB.abs(),tvA.sign(),tvB.sign())
        tvAf=torch.where(tvA.sign()==elect,tvA,torch.zeros_like(tvA))
        tvBf=torch.where(tvB.sign()==elect,tvB,torch.zeros_like(tvB))
        n_nz=((tvAf!=0).float()+(tvBf!=0).float()).clamp(min=1)
        sd[k]=base_sd[k]+(tvAf+tvBf)/n_nz
    m.load_state_dict(sd); m.eval()
    r=evaluate(m,f"TIES(d={density})"); results.append(r)
    log(f"  d={density}: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")

# ═══════════════════════════════════════════
# METHOD 5: DARE-TIES
# ═══════════════════════════════════════════
log("\n--- DARE-TIES ---")
for p in [0.1, 0.3, 0.5]:
    m=CNN(); sd={}; torch.manual_seed(SEED)
    for k in base_sd:
        tvA=sdA[k]-base_sd[k]; tvB=sdB[k]-base_sd[k]
        mA=(torch.rand_like(tvA.float())<p).float(); mB=(torch.rand_like(tvB.float())<p).float()
        tvA=tvA*mA/max(p,0.01); tvB=tvB*mB/max(p,0.01)
        elect=torch.where(tvA.abs()>=tvB.abs(),tvA.sign(),tvB.sign())
        tvA=torch.where(tvA.sign()==elect,tvA,torch.zeros_like(tvA))
        tvB=torch.where(tvB.sign()==elect,tvB,torch.zeros_like(tvB))
        n_nz=((tvA!=0).float()+(tvB!=0).float()).clamp(min=1)
        sd[k]=base_sd[k]+(tvA+tvB)/n_nz
    m.load_state_dict(sd); m.eval()
    r=evaluate(m,f"DARE-TIES(p={p})"); results.append(r)
    log(f"  p={p}: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")

# ═══════════════════════════════════════════
# METHOD 6: Sakana-CMA
# ═══════════════════════════════════════════
log("\n--- Sakana-CMA ---")
# Group params: conv1(2) + conv2(2) + fc1(2) + fc2(2) = 4 param groups
def build_sak(x):
    a=1/(1+np.exp(-np.array(x))); m=CNN(); sd={}; keys=list(sdA.keys())
    groups = {}
    for k in keys:
        if 'features.0' in k: gi=0
        elif 'features.3' in k: gi=1
        elif 'classifier.0' in k: gi=2
        else: gi=3
        sd[k]=a[gi]*sdA[k]+(1-a[gi])*sdB[k]
    m.load_state_dict(sd); m.eval(); return m

es=cma.CMAEvolutionStrategy(np.zeros(4),1.5,{'maxiter':15,'popsize':8,'seed':SEED,'verbose':-1})
best_s=-1;best_x=None
while not es.stop():
    sols=es.ask();scores=[]
    for x in sols:
        m=build_sak(x);d=pc(m,Xv,yv);acc=ev(m,Xv,yv)
        mn=min(d[c] for c in range(10));s=0.4*acc+0.4*mn+0.1*np.mean([d[c] for c in range(10)])
        scores.append(-s)
    es.tell(sols,scores)
    if -min(scores)>best_s: best_s=-min(scores);best_x=sols[np.argmin(scores)]
m=build_sak(best_x); r=evaluate(m,"Sakana-CMA"); results.append(r)
log(f"  Sakana-CMA: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")

# ═══════════════════════════════════════════
# METHOD 7: ENT (ZCP prune + block-diagonal + EA routing)
# ═══════════════════════════════════════════
log("\n--- ENT (ours) ---")

def prune_cnn(model, tch1, tch2, tfc):
    """Structured pruning to target dims."""
    W1=model.features[0].weight.data; b1=model.features[0].bias.data
    imp1=W1.abs().view(W1.shape[0],-1).sum(1)
    idx1=imp1.argsort(descending=True)[:tch1].sort().values

    W2=model.features[3].weight.data; b2=model.features[3].bias.data
    imp2=W2.abs().view(W2.shape[0],-1).sum(1)
    idx2=imp2.argsort(descending=True)[:tch2].sort().values

    Wf=model.classifier[0].weight.data; bf=model.classifier[0].bias.data
    fimp=Wf.abs().sum(1)
    idx_fc=fimp.argsort(descending=True)[:tfc].sort().values

    p=CNN(tch1,tch2,tfc)
    with torch.no_grad():
        p.features[0].weight.copy_(W1[idx1]); p.features[0].bias.copy_(b1[idx1])
        p.features[3].weight.copy_(W2[idx2][:,idx1]); p.features[3].bias.copy_(b2[idx2])
        flat_idx=torch.cat([torch.arange(i*64,(i+1)*64) for i in idx2])  # 8*8=64
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

# Prune both to 24/48/96 (75% of 32/64/128)
tch1,tch2,tfc = 24,48,96
log(f"  Pruning to {tch1}/{tch2}/{tfc}...")
pA=prune_cnn(ftA,tch1,tch2,tfc); pB=prune_cnn(ftB,tch1,tch2,tfc)
log(f"  Pruned A: {ev(pA,X_te,y_te):.3f}  Pruned B: {ev(pB,X_te,y_te):.3f}")

pA.eval();pB.eval()
with torch.no_grad(): sA_=pA(Xc).numpy().std();sB_=pB(Xc).numpy().std()
t_=(sA_+sB_)/2;rA_=t_/(sA_+1e-10);rB_=t_/(sB_+1e-10)

pop=[np.array([2.]*5+[-2.]*5)]+[np.random.randn(10)*1.5 for _ in range(11)]
bf=-1;bc=None
for gen in range(20):
    fs=[]
    for r in pop:
        m=merge_cnns(pA,pB,r,rA_,rB_); d=pc(m,Xv,yv); acc=ev(m,Xv,yv)
        fs.append(0.4*acc+0.4*min(d[c] for c in range(10))+0.1*np.mean([d[c] for c in range(10)]))
    gi=np.argmax(fs)
    if fs[gi]>bf: bf=fs[gi];bc=pop[gi].copy()
    if gen%10==0: log(f"  Gen {gen}: fit={fs[gi]:.4f}")
    new=[bc.copy()]+[pop[random.randint(0,len(pop)-1)]+np.random.randn(10)*0.3 for _ in range(11)]
    pop=new

m=merge_cnns(pA,pB,bc,rA_,rB_)
r=evaluate(m,"ENT-CNN"); results.append(r)
log(f"  ENT-CNN: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")

# ═══════════════════════════════════════════
# FINAL TABLE
# ═══════════════════════════════════════════
log(f"\n{'='*70}")
log("CIFAR-10 RESULTS: Shared Base → Fine-tune → Merge (CNN)")
log(f"{'='*70}")
log(f"  {'Method':<22} {'Acc':>6} {'Bal':>6} {'Min':>6} {'OK':>5} {'A':>6} {'B':>6}")
log(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*6} {'-'*6}")
for r in sorted(results, key=lambda x: (-x['ok'], -x['bal'])):
    mk = " ✅" if 'ENT' in r['name'] else ""
    log(f"  {r['name']:<22} {r['acc']:>6.3f} {r['bal']:>6.3f} {r['min']:>6.3f} {r['ok']:>3}/10 {r['A']:>6.3f} {r['B']:>6.3f}{mk}")

log(f"\n  Per-class (sorted by balance):")
top5 = sorted(results, key=lambda x: -x['bal'])[:5]
log(f"  {'C':>2}" + "".join(f" {r['name'][:10]:>10}" for r in top5))
for c in range(10):
    log(f"  {c:>2}" + "".join(f" {r['pc'][c]:>10.3f}" for r in top5))

log(f"\n  Time: {time.time()-t0:.0f}s")
R.close()
print("Done!", flush=True)
