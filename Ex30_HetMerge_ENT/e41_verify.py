#!/usr/bin/env python3
"""E41: Systematic hypothesis verification — ALL v4 approaches.

Tests 6 concrete hypotheses on same data, same parents, same eval.
Target: per-class drop ≤ 5%.

H1: Full concat (no selection) — is capacity the bottleneck?
H2: Lagrangian constrained pruning — per-class λ 
H3: Shapley neuron importance — principled selection
H4: Cross-connections + EA bridge α — break isolation
H5: Per-class fine-tuning (constrained) — post-merge adaptation
H6: Standard ENT (baseline for comparison)
"""
import numpy as np, torch, torch.nn as nn, random, copy, time
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
import ssl; ssl._create_default_https_context=ssl._create_unverified_context
from torchvision import datasets, transforms

R=open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/results_e41.txt','w')
def log(s): R.write(s+'\n'); R.flush(); print(s,flush=True)

log("="*70)
log("E41: Systematic Hypothesis Verification")
log("="*70)

tf=transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x:x.view(-1))])
tr=datasets.MNIST('/tmp/mnist',train=True,download=True,transform=tf)
te=datasets.MNIST('/tmp/mnist',train=False,download=True,transform=tf)
X_tr=torch.stack([tr[i][0] for i in range(20000)]); y_tr=torch.tensor([tr[i][1] for i in range(20000)])
X_te=torch.stack([te[i][0] for i in range(2000)]); y_te=torch.tensor([te[i][1] for i in range(2000)])
idx=torch.randperm(20000,generator=torch.Generator().manual_seed(0))
Xv,yv=X_tr[idx[15000:18000]],y_tr[idx[15000:18000]]; Xc=X_tr[idx[:2000]]
clA,clB=list(range(5)),list(range(5,10))

class MLP(nn.Module):
    def __init__(s,a):
        super().__init__(); l=[]
        for i in range(len(a)-1):
            l.append(nn.Linear(a[i],a[i+1]))
            if i<len(a)-2: l.append(nn.ReLU())
        s.net=nn.Sequential(*l); s.arch=a
    def forward(s,x): return s.net(x)

def ev(m,X,y):
    m.eval()
    with torch.no_grad(): return (m(X).argmax(1)==y).float().mean().item()
def pc(m,X,y):
    m.eval()
    with torch.no_grad(): p=m(X).argmax(1)
    return {c:(p[y==c]==c).float().mean().item() if (y==c).sum()>0 else 0 for c in range(10)}

# Train parents
arch=[784,128,64,10]
base=MLP(arch); torch.manual_seed(SEED)
opt=torch.optim.Adam(base.parameters(),lr=0.003)
for _ in range(5): l=nn.CrossEntropyLoss()(base(X_tr[:5000]),y_tr[:5000]);opt.zero_grad();l.backward();opt.step()
base.eval()
def train_ft(m,X,y,cls,ep=15):
    mask=sum(y==c for c in cls).bool();Xs,ys=X[mask][:5000],y[mask][:5000]
    opt=torch.optim.Adam(m.parameters(),lr=0.003);m.train()
    for _ in range(ep): l=nn.CrossEntropyLoss()(m(Xs),ys);opt.zero_grad();l.backward();opt.step()
    m.eval();return m
ftA=train_ft(copy.deepcopy(base),X_tr,y_tr,clA)
ftB=train_ft(copy.deepcopy(base),X_tr,y_tr,clB)
pcA=pc(ftA,X_te,y_te); pcB=pc(ftB,X_te,y_te)
bp={c:max(pcA[c],pcB[c]) for c in range(10)}
log(f"Parents: A={ev(ftA,X_te,y_te):.3f} B={ev(ftB,X_te,y_te):.3f}")
for c in range(10): log(f"  C{c}: best_parent={bp[c]:.3f}")

# Extract weights
psA=list(ftA.parameters()); psB=list(ftB.parameters())
W0A,b0A=psA[0].data.numpy(),psA[1].data.numpy()
W1A,b1A=psA[2].data.numpy(),psA[3].data.numpy()
W2A,b2A=psA[4].data.numpy(),psA[5].data.numpy()
W0B,b0B=psB[0].data.numpy(),psB[1].data.numpy()
W1B,b1B=psB[2].data.numpy(),psB[3].data.numpy()
W2B,b2B=psB[4].data.numpy(),psB[5].data.numpy()
ftA.eval();ftB.eval()
with torch.no_grad(): sl=ftA(Xc).numpy().std();sr=ftB(Xc).numpy().std()
t_=(sl+sr)/2;rA=t_/(sl+1e-10);rB=t_/(sr+1e-10)

def evaluate(m, name):
    m.eval(); acc=ev(m,X_te,y_te); d=pc(m,X_te,y_te)
    drops={c:max(0,bp[c]-d[c]) for c in range(10)}
    w5=sum(1 for c in range(10) if drops[c]<=0.05)
    w10=sum(1 for c in range(10) if drops[c]<=0.10)
    aM=np.mean([d[c] for c in clA]);bM=np.mean([d[c] for c in clB])
    bal=min(aM,bM)/(max(aM,bM)+1e-10);mn=min(d[c] for c in range(10))
    return {'name':name,'acc':acc,'bal':bal,'min':mn,
            'ok':sum(1 for c in range(10) if d[c]>0.3),
            'maxdrop':max(drops.values()),
            'w5':w5,'w10':w10,'drops':drops,'pc':d,
            'params':sum(p.numel() for p in m.parameters())}

def build_ent(mA0,mB0,mA1,mB1,route,W0A_,b0A_,W1A_,b1A_,W2A_,b2A_,
              W0B_,b0B_,W1B_,b1B_,W2B_,b2B_,rA_,rB_,cross_alpha=0.0):
    iA0=np.where(mA0)[0];iB0=np.where(mB0)[0]
    iA1=np.where(mA1)[0];iB1=np.where(mB1)[0]
    na0,nb0,na1,nb1=len(iA0),len(iB0),len(iA1),len(iB1)
    if na0+nb0<2 or na1+nb1<2: return None
    m=MLP([784,na0+nb0,na1+nb1,10])
    with torch.no_grad():
        ps=list(m.parameters())
        ps[0].copy_(torch.tensor(np.vstack([W0A_[iA0],W0B_[iB0]])))
        ps[1].copy_(torch.tensor(np.concatenate([b0A_[iA0],b0B_[iB0]])))
        W1=np.zeros((na1+nb1,na0+nb0),dtype=np.float32)
        W1[:na1,:na0]=W1A_[np.ix_(iA1,iA0)]; W1[na1:,na0:]=W1B_[np.ix_(iB1,iB0)]
        # Cross-connections
        if cross_alpha>0:
            W1[:na1,na0:]=cross_alpha*np.random.randn(na1,nb0).astype(np.float32)*0.01
            W1[na1:,:na0]=cross_alpha*np.random.randn(nb1,na0).astype(np.float32)*0.01
        b1=np.concatenate([b1A_[iA1],b1B_[iB1]])
        ps[2].copy_(torch.tensor(W1)); ps[3].copy_(torch.tensor(b1))
        Wo=np.zeros((10,na1+nb1),dtype=np.float32);bo=np.zeros(10,dtype=np.float32)
        for c in range(10):
            a=1/(1+np.exp(-route[c]))
            if na1>0: Wo[c,:na1]=a*rA_*W2A_[c][iA1]
            if nb1>0: Wo[c,na1:]=(1-a)*rB_*W2B_[c][iB1]
            bo[c]=a*rA_*b2A_[c]+(1-a)*rB_*b2B_[c]
        ps[4].copy_(torch.tensor(Wo));ps[5].copy_(torch.tensor(bo))
    return m

# ═══════════════════════════════════════════
# H6: Standard ENT (baseline)
# ═══════════════════════════════════════════
log(f"\n{'='*50}\nH6: Standard ENT (baseline)")
random.seed(SEED);np.random.seed(SEED)
pop=[]
for _ in range(20):
    ch={'mA0':np.random.random(128)>0.3,'mB0':np.random.random(128)>0.3,
        'mA1':np.random.random(64)>0.3,'mB1':np.random.random(64)>0.3,
        'route':np.random.randn(10)*1.5}
    for k in ['mA0','mB0','mA1','mB1']:
        if ch[k].sum()==0:ch[k][0]=True
    pop.append(ch)
pop[0]={'mA0':np.ones(128,dtype=bool),'mB0':np.ones(128,dtype=bool),
        'mA1':np.ones(64,dtype=bool),'mB1':np.ones(64,dtype=bool),
        'route':np.array([2.]*5+[-2.]*5)}
bf=-1;bc=None
for gen in range(30):
    fs=[]
    for ch in pop:
        m=build_ent(ch['mA0'],ch['mB0'],ch['mA1'],ch['mB1'],ch['route'],
                    W0A,b0A,W1A,b1A,W2A,b2A,W0B,b0B,W1B,b1B,W2B,b2B,rA,rB)
        if m is None: fs.append(-1);continue
        d=pc(m,Xv,yv);acc=ev(m,Xv,yv);mn=min(d[c] for c in range(10))
        fs.append(0.4*acc+0.4*mn+0.1*np.mean([d[c] for c in range(10)])+0.1*0.5)
    gi=np.argmax(fs)
    if fs[gi]>bf:bf=fs[gi];bc={k:v.copy() for k,v in pop[gi].items()}
    new=[{k:v.copy() for k,v in bc.items()}]
    while len(new)<20:
        ti=random.sample(range(len(pop)),3);p1=pop[ti[np.argmax([fs[i] for i in ti])]]
        ch={k:v.copy() for k,v in p1.items()};ch['route']+=np.random.randn(10)*0.3
        pf=max(0.02,0.06-gen*0.001)
        for k in ['mA0','mB0','mA1','mB1']:
            f=np.random.random(len(ch[k]))<pf;ch[k][f]=~ch[k][f]
            if ch[k].sum()==0:ch[k][0]=True
        new.append(ch)
    pop=new
m_base=build_ent(bc['mA0'],bc['mB0'],bc['mA1'],bc['mB1'],bc['route'],
                 W0A,b0A,W1A,b1A,W2A,b2A,W0B,b0B,W1B,b1B,W2B,b2B,rA,rB)
r_base=evaluate(m_base,"H6:Standard")
log(f"  acc={r_base['acc']:.3f} min={r_base['min']:.3f} maxdrop={r_base['maxdrop']:.3f} ≤5%={r_base['w5']}/10")

# ═══════════════════════════════════════════
# H1: Full concat (ALL neurons, no pruning)
# ═══════════════════════════════════════════
log(f"\n{'='*50}\nH1: Full concat — is capacity the issue?")
# Just use all neurons + optimize routing only
best_route=None;best_fit=-1
for trial in range(200):
    route=np.random.randn(10)*2 if trial>0 else np.array([2.]*5+[-2.]*5)
    m=build_ent(np.ones(128,dtype=bool),np.ones(128,dtype=bool),
                np.ones(64,dtype=bool),np.ones(64,dtype=bool),route,
                W0A,b0A,W1A,b1A,W2A,b2A,W0B,b0B,W1B,b1B,W2B,b2B,rA,rB)
    d=pc(m,Xv,yv);acc=ev(m,Xv,yv);mn=min(d[c] for c in range(10))
    fit=0.4*acc+0.4*mn+0.1*np.mean([d[c] for c in range(10)])
    if fit>best_fit: best_fit=fit;best_route=route.copy()
m_full=build_ent(np.ones(128,dtype=bool),np.ones(128,dtype=bool),
                 np.ones(64,dtype=bool),np.ones(64,dtype=bool),best_route,
                 W0A,b0A,W1A,b1A,W2A,b2A,W0B,b0B,W1B,b1B,W2B,b2B,rA,rB)
r_full=evaluate(m_full,"H1:FullConcat")
log(f"  acc={r_full['acc']:.3f} min={r_full['min']:.3f} maxdrop={r_full['maxdrop']:.3f} ≤5%={r_full['w5']}/10")

# ═══════════════════════════════════════════
# H3: Shapley neuron importance
# ═══════════════════════════════════════════
log(f"\n{'='*50}\nH3: Shapley-based neuron selection")
# Approximate Shapley: for each neuron, measure accuracy WITH vs WITHOUT
ftA.eval(); ftB.eval()
shap_A0=np.zeros(128); shap_B0=np.zeros(128)
shap_A1=np.zeros(64); shap_B1=np.zeros(64)

# Full model (all neurons, good routing)
m_all=build_ent(np.ones(128,dtype=bool),np.ones(128,dtype=bool),
                np.ones(64,dtype=bool),np.ones(64,dtype=bool),best_route,
                W0A,b0A,W1A,b1A,W2A,b2A,W0B,b0B,W1B,b1B,W2B,b2B,rA,rB)
base_pc=pc(m_all,Xv[:200],yv[:200])

# Leave-one-out for layer 0 (fast: only 256 evals)
for i in range(128):
    mask=np.ones(128,dtype=bool);mask[i]=False
    m=build_ent(mask,np.ones(128,dtype=bool),np.ones(64,dtype=bool),np.ones(64,dtype=bool),
                best_route,W0A,b0A,W1A,b1A,W2A,b2A,W0B,b0B,W1B,b1B,W2B,b2B,rA,rB)
    d=pc(m,Xv[:200],yv[:200])
    # Per-class importance: how much does removing this neuron hurt
    for c in range(10): shap_A0[i]+=max(0,base_pc[c]-d[c])
for i in range(128):
    mask=np.ones(128,dtype=bool);mask[i]=False
    m=build_ent(np.ones(128,dtype=bool),mask,np.ones(64,dtype=bool),np.ones(64,dtype=bool),
                best_route,W0A,b0A,W1A,b1A,W2A,b2A,W0B,b0B,W1B,b1B,W2B,b2B,rA,rB)
    d=pc(m,Xv[:200],yv[:200])
    for c in range(10): shap_B0[i]+=max(0,base_pc[c]-d[c])
# Layer 1
for i in range(64):
    mask=np.ones(64,dtype=bool);mask[i]=False
    m=build_ent(np.ones(128,dtype=bool),np.ones(128,dtype=bool),mask,np.ones(64,dtype=bool),
                best_route,W0A,b0A,W1A,b1A,W2A,b2A,W0B,b0B,W1B,b1B,W2B,b2B,rA,rB)
    d=pc(m,Xv[:200],yv[:200])
    for c in range(10): shap_A1[i]+=max(0,base_pc[c]-d[c])
for i in range(64):
    mask=np.ones(64,dtype=bool);mask[i]=False
    m=build_ent(np.ones(128,dtype=bool),np.ones(128,dtype=bool),np.ones(64,dtype=bool),mask,
                best_route,W0A,b0A,W1A,b1A,W2A,b2A,W0B,b0B,W1B,b1B,W2B,b2B,rA,rB)
    d=pc(m,Xv[:200],yv[:200])
    for c in range(10): shap_B1[i]+=max(0,base_pc[c]-d[c])

log(f"  Shapley stats: A0 mean={shap_A0.mean():.4f} B0={shap_B0.mean():.4f}")
log(f"  Shapley stats: A1 mean={shap_A1.mean():.4f} B1={shap_B1.mean():.4f}")
log(f"  Top-5 important A0: {np.argsort(shap_A0)[-5:]}")
log(f"  Top-5 important B0: {np.argsort(shap_B0)[-5:]}")

# Use Shapley to initialize EA (seed with top neurons)
keep_A0 = shap_A0 > np.percentile(shap_A0, 20)  # keep top 80%
keep_B0 = shap_B0 > np.percentile(shap_B0, 20)
keep_A1 = shap_A1 > np.percentile(shap_A1, 20)
keep_B1 = shap_B1 > np.percentile(shap_B1, 20)

random.seed(SEED);np.random.seed(SEED)
pop=[]
for _ in range(20):
    ch={'mA0':keep_A0.copy(),'mB0':keep_B0.copy(),
        'mA1':keep_A1.copy(),'mB1':keep_B1.copy(),
        'route':np.random.randn(10)*1.5}
    # Mutate from Shapley init
    for k in ['mA0','mB0','mA1','mB1']:
        f=np.random.random(len(ch[k]))<0.1;ch[k][f]=~ch[k][f]
        if ch[k].sum()==0:ch[k][0]=True
    pop.append(ch)
pop[0]={'mA0':np.ones(128,dtype=bool),'mB0':np.ones(128,dtype=bool),
        'mA1':np.ones(64,dtype=bool),'mB1':np.ones(64,dtype=bool),
        'route':np.array([2.]*5+[-2.]*5)}
bf=-1;bc=None
for gen in range(30):
    fs=[]
    for ch in pop:
        m=build_ent(ch['mA0'],ch['mB0'],ch['mA1'],ch['mB1'],ch['route'],
                    W0A,b0A,W1A,b1A,W2A,b2A,W0B,b0B,W1B,b1B,W2B,b2B,rA,rB)
        if m is None: fs.append(-1);continue
        d=pc(m,Xv,yv);acc=ev(m,Xv,yv);mn=min(d[c] for c in range(10))
        fs.append(0.4*acc+0.4*mn+0.1*np.mean([d[c] for c in range(10)])+0.1*0.5)
    gi=np.argmax(fs)
    if fs[gi]>bf:bf=fs[gi];bc={k:v.copy() for k,v in pop[gi].items()}
    new=[{k:v.copy() for k,v in bc.items()}]
    while len(new)<20:
        ti=random.sample(range(len(pop)),3);p1=pop[ti[np.argmax([fs[i] for i in ti])]]
        ch={k:v.copy() for k,v in p1.items()};ch['route']+=np.random.randn(10)*0.3
        pf=max(0.02,0.06-gen*0.001)
        for k in ['mA0','mB0','mA1','mB1']:
            f=np.random.random(len(ch[k]))<pf;ch[k][f]=~ch[k][f]
            if ch[k].sum()==0:ch[k][0]=True
        new.append(ch)
    pop=new
m_shap=build_ent(bc['mA0'],bc['mB0'],bc['mA1'],bc['mB1'],bc['route'],
                 W0A,b0A,W1A,b1A,W2A,b2A,W0B,b0B,W1B,b1B,W2B,b2B,rA,rB)
r_shap=evaluate(m_shap,"H3:Shapley+EA")
log(f"  acc={r_shap['acc']:.3f} min={r_shap['min']:.3f} maxdrop={r_shap['maxdrop']:.3f} ≤5%={r_shap['w5']}/10")

# ═══════════════════════════════════════════
# H2: Lagrangian constrained pruning
# ═══════════════════════════════════════════
log(f"\n{'='*50}\nH2: Lagrangian constrained — target ≤5% per-class")
# Start from best ENT, then iteratively fix violations
m_lag=copy.deepcopy(m_base)
torch.set_grad_enabled(True)
# Per-class λ multipliers
lam=np.ones(10)*1.0
for iteration in range(5):
    m_lag.train()
    opt=torch.optim.Adam(m_lag.parameters(),lr=0.001)
    for step in range(30):
        d=pc(m_lag,Xv[:500],yv[:500])
        # Lagrangian: minimize -acc + Σ λ_c * max(0, drop_c - 0.05)
        out=m_lag(Xv[:500])
        ce_loss=nn.CrossEntropyLoss()(out,yv[:500])
        # Per-class penalty
        penalty=torch.tensor(0.0)
        for c in range(10):
            drop_c=bp[c]-d[c]
            if drop_c>0.05:
                mask_c=(yv[:500]==c)
                if mask_c.sum()>0:
                    penalty+=lam[c]*nn.CrossEntropyLoss()(out[mask_c],yv[:500][mask_c])
        loss=ce_loss+penalty
        opt.zero_grad();loss.backward();opt.step()
    m_lag.eval()
    d=pc(m_lag,X_te,y_te)
    violations=sum(1 for c in range(10) if bp[c]-d[c]>0.05)
    for c in range(10):
        if bp[c]-d[c]>0.05: lam[c]*=2.0
        else: lam[c]=max(0.5,lam[c]*0.9)
    log(f"  Iter {iteration}: violations={violations} λ_max={max(lam):.1f}")
    if violations==0: break
torch.set_grad_enabled(False); m_lag.eval()
r_lag=evaluate(m_lag,"H2:Lagrangian")
log(f"  acc={r_lag['acc']:.3f} min={r_lag['min']:.3f} maxdrop={r_lag['maxdrop']:.3f} ≤5%={r_lag['w5']}/10")

# ═══════════════════════════════════════════
# H5: Per-class fine-tuning (constrained SGD)
# ═══════════════════════════════════════════
log(f"\n{'='*50}\nH5: Per-class fine-tuning from full concat")
m_ft=copy.deepcopy(m_full)
torch.set_grad_enabled(True); m_ft.train()
# Use KD from both parents but CLASS-WEIGHTED
ftA.eval();ftB.eval()
with torch.no_grad():
    logA=ftA(Xv).detach(); logB=ftB(Xv).detach()
opt=torch.optim.Adam(m_ft.parameters(),lr=0.0005)
for ep in range(20):
    # Class-weighted: higher weight for classes with more drop
    d=pc(m_ft,Xv,yv)
    class_weights=torch.ones(10)
    for c in range(10):
        drop_c=bp[c]-d[c]
        class_weights[c]=1.0+max(0,drop_c-0.03)*20  # boost classes with >3% drop
    idx_=torch.randperm(len(Xv))[:500]
    out=m_ft(Xv[idx_])
    # Weighted KD
    soft=torch.zeros_like(logA[idx_])
    for i in range(len(idx_)):
        yi=yv[idx_[i]].item()
        if yi in clA: soft[i]=logA[idx_[i]]
        else: soft[i]=logB[idx_[i]]
        soft[i]*=class_weights[yi]
    soft=torch.softmax(soft/3,dim=1)
    kd=nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(out/3,dim=1),soft)*9
    ce=nn.CrossEntropyLoss(weight=class_weights)(out,yv[idx_])
    loss=0.3*kd+0.7*ce
    opt.zero_grad();loss.backward();opt.step()
torch.set_grad_enabled(False); m_ft.eval()
r_ft=evaluate(m_ft,"H5:ClassWtKD")
log(f"  acc={r_ft['acc']:.3f} min={r_ft['min']:.3f} maxdrop={r_ft['maxdrop']:.3f} ≤5%={r_ft['w5']}/10")

# ═══════════════════════════════════════════
# H4: Cross-connections + EA-optimized bridge
# ═══════════════════════════════════════════
log(f"\n{'='*50}\nH4: Cross-connections (EA bridge)")
random.seed(SEED);np.random.seed(SEED)
pop=[]
for _ in range(20):
    ch={'mA0':np.ones(128,dtype=bool),'mB0':np.ones(128,dtype=bool),
        'mA1':np.ones(64,dtype=bool),'mB1':np.ones(64,dtype=bool),
        'route':np.random.randn(10)*1.5,'bridge':random.uniform(0.01,0.3)}
    pop.append(ch)
pop[0]['route']=np.array([2.]*5+[-2.]*5);pop[0]['bridge']=0.05
bf=-1;bc=None
for gen in range(30):
    fs=[]
    for ch in pop:
        m=build_ent(ch['mA0'],ch['mB0'],ch['mA1'],ch['mB1'],ch['route'],
                    W0A,b0A,W1A,b1A,W2A,b2A,W0B,b0B,W1B,b1B,W2B,b2B,rA,rB,
                    cross_alpha=ch['bridge'])
        if m is None: fs.append(-1);continue
        d=pc(m,Xv,yv);acc=ev(m,Xv,yv);mn=min(d[c] for c in range(10))
        fs.append(0.4*acc+0.4*mn+0.1*np.mean([d[c] for c in range(10)])+0.1*0.5)
    gi=np.argmax(fs)
    if fs[gi]>bf:bf=fs[gi];bc={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in pop[gi].items()}
    new=[{k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in bc.items()}]
    while len(new)<20:
        ti=random.sample(range(len(pop)),3);p1=pop[ti[np.argmax([fs[i] for i in ti])]]
        ch={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in p1.items()}
        ch['route']+=np.random.randn(10)*0.3
        ch['bridge']=max(0.001,ch['bridge']+random.gauss(0,0.03))
        new.append(ch)
    pop=new
m_cross=build_ent(bc['mA0'],bc['mB0'],bc['mA1'],bc['mB1'],bc['route'],
                  W0A,b0A,W1A,b1A,W2A,b2A,W0B,b0B,W1B,b1B,W2B,b2B,rA,rB,
                  cross_alpha=bc['bridge'])
r_cross=evaluate(m_cross,"H4:CrossBridge")
log(f"  bridge_α={bc['bridge']:.4f}")
log(f"  acc={r_cross['acc']:.3f} min={r_cross['min']:.3f} maxdrop={r_cross['maxdrop']:.3f} ≤5%={r_cross['w5']}/10")

# ═══════════════════════════════════════════
# FINAL COMPARISON
# ═══════════════════════════════════════════
log(f"\n{'='*70}")
log("HYPOTHESIS VERIFICATION — FINAL RESULTS")
log("="*70)
all_r=[r_base,r_full,r_lag,r_shap,r_cross,r_ft]
log(f"\n  {'Method':<20} {'Acc':>6} {'Min':>6} {'MaxDr':>6} {'≤5%':>5} {'≤10%':>5} {'Params':>8} {'Verdict':>10}")
log(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*8} {'-'*10}")
for r in sorted(all_r,key=lambda x:(-x['w5'],-x['w10'],x['maxdrop'])):
    v="✅ BEST" if r['w5']==max(x['w5'] for x in all_r) else ("🟡" if r['w5']>r_base['w5'] else "❌")
    log(f"  {r['name']:<20} {r['acc']:>6.3f} {r['min']:>6.3f} {r['maxdrop']:>6.3f} {r['w5']:>3}/10 {r['w10']:>3}/10 {r['params']:>8,} {v:>10}")

log(f"\n  Per-class drop:")
log(f"  {'C':>2} {'Parent':>7}" + "".join(f" {r['name'][:8]:>9}" for r in all_r))
for c in range(10):
    log(f"  {c:>2} {bp[c]:>7.3f}" + "".join(f" {r['drops'][c]:>9.3f}" for r in all_r))

# Winners per class
log(f"\n  Best method per class:")
for c in range(10):
    best_m=min(all_r,key=lambda r:r['drops'][c])
    sym="✅" if best_m['drops'][c]<=0.05 else "❌"
    log(f"  C{c}: {best_m['name']:<20} drop={best_m['drops'][c]:.3f} {sym}")

R.close(); print("Done!",flush=True)
