#!/usr/bin/env python3
"""E42: Ensemble Upper Bound + Distillation Baseline.

The missing experiment: what is the THEORETICAL MAXIMUM 
that any merging method can achieve?

Hierarchy:
  Ensemble (0% drop, 2× cost)  — UPPER BOUND
  > Distillation (WITH data)    — best single model possible
  > ENT (WITHOUT data retrain)  — our method
  > Average                     — naive baseline

Gap analysis:
  Gap1 = Ensemble − Distillation = information loss from compression
  Gap2 = Distillation − ENT      = ROOM FOR IMPROVEMENT in ENT
  Gap3 = ENT − Average           = ENT's contribution (what we've proven)
"""
import numpy as np, torch, torch.nn as nn, random, copy, time
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
import ssl; ssl._create_default_https_context=ssl._create_unverified_context
from torchvision import datasets, transforms

R=open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/results_e42.txt','w')
def log(s): R.write(s+'\n'); R.flush(); print(s,flush=True)

log("="*70)
log("E42: Theoretical Bounds — Ensemble | Distillation | ENT | Average")
log("="*70)

tf=transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x:x.view(-1))])
tr=datasets.MNIST('/tmp/mnist',train=True,download=True,transform=tf)
te=datasets.MNIST('/tmp/mnist',train=False,download=True,transform=tf)
X_tr=torch.stack([tr[i][0] for i in range(20000)]); y_tr=torch.tensor([tr[i][1] for i in range(20000)])
X_te=torch.stack([te[i][0] for i in range(2000)]); y_te=torch.tensor([te[i][1] for i in range(2000)])
idx=torch.randperm(20000,generator=torch.Generator().manual_seed(0))
Xv=X_tr[idx[15000:18000]]; yv=y_tr[idx[15000:18000]]
Xc=X_tr[idx[:2000]]
clA,clB=list(range(5)),list(range(5,10)); arch=[784,128,64,10]

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

# ═══════════════════════════════════════════
# STEP 1: Train parents (shared base)
# ═══════════════════════════════════════════
base=MLP(arch); torch.manual_seed(SEED)
opt=torch.optim.Adam(base.parameters(),lr=0.003)
for _ in range(5): l=nn.CrossEntropyLoss()(base(X_tr[:5000]),y_tr[:5000]);opt.zero_grad();l.backward();opt.step()
base.eval(); base_sd=copy.deepcopy(base.state_dict())

def train_ft(m,X,y,cls,ep=15):
    mask=sum(y==c for c in cls).bool();Xs,ys=X[mask][:5000],y[mask][:5000]
    opt=torch.optim.Adam(m.parameters(),lr=0.003);m.train()
    for _ in range(ep): l=nn.CrossEntropyLoss()(m(Xs),ys);opt.zero_grad();l.backward();opt.step()
    m.eval();return m

ftA=train_ft(copy.deepcopy(base),X_tr,y_tr,clA)
ftB=train_ft(copy.deepcopy(base),X_tr,y_tr,clB)
pcA=pc(ftA,X_te,y_te); pcB=pc(ftB,X_te,y_te)
bp={c:max(pcA[c],pcB[c]) for c in range(10)}
log(f"\nParents: A={ev(ftA,X_te,y_te):.3f} B={ev(ftB,X_te,y_te):.3f}")
for c in range(10): log(f"  C{c}: A={pcA[c]:.3f} B={pcB[c]:.3f} best={bp[c]:.3f}")

# ═══════════════════════════════════════════
# STEP 2: ENSEMBLE — theoretical upper bound
# ═══════════════════════════════════════════
log(f"\n{'='*50}")
log("ENSEMBLE (upper bound — 0% drop expected)")

class Ensemble(nn.Module):
    def __init__(s, mA, mB, clA, clB):
        super().__init__()
        s.mA=mA; s.mB=mB; s.clA=clA; s.clB=clB
    def forward(s, x):
        lA=s.mA(x); lB=s.mB(x)
        out=torch.zeros_like(lA)
        for c in s.clA: out[:,c]=lA[:,c]
        for c in s.clB: out[:,c]=lB[:,c]
        return out

ensemble=Ensemble(ftA,ftB,clA,clB); ensemble.eval()
acc_ens=ev(ensemble,X_te,y_te); pc_ens=pc(ensemble,X_te,y_te)
drops_ens={c:max(0,bp[c]-pc_ens[c]) for c in range(10)}
log(f"  Accuracy: {acc_ens:.4f}")
log(f"  Per-class:")
for c in range(10):
    log(f"    C{c}: {pc_ens[c]:.3f} (parent={bp[c]:.3f}, drop={drops_ens[c]:.3f})")
log(f"  Max drop: {max(drops_ens.values()):.4f}")
log(f"  Classes ≤5%: {sum(1 for d in drops_ens.values() if d<=0.05)}/10")
w5_ens=sum(1 for d in drops_ens.values() if d<=0.05)

# ═══════════════════════════════════════════
# STEP 3: DISTILLATION — best single model (WITH data)
# ═══════════════════════════════════════════
log(f"\n{'='*50}")
log("DISTILLATION (best single model, WITH training data)")

# Generate soft labels from ensemble on ALL training data
ensemble.eval()
with torch.no_grad():
    soft_labels = torch.softmax(ensemble(X_tr) / 3.0, dim=1)  # temperature=3
    hard_labels = ensemble(X_tr).argmax(1)

# Train student (same architecture as parents)
for student_arch_name, student_arch in [("same [128,64]", [784,128,64,10]),
                                         ("wide [256,128]", [784,256,128,10])]:
    student=MLP(student_arch)
    opt=torch.optim.Adam(student.parameters(),lr=0.002)
    student.train()
    for ep in range(50):
        idx_=torch.randperm(15000)[:2000]
        out=student(X_tr[idx_])
        kd=nn.KLDivLoss(reduction='batchmean')(
            torch.log_softmax(out/3.0,dim=1), soft_labels[idx_]) * 9
        ce=nn.CrossEntropyLoss()(out, hard_labels[idx_])
        loss=0.5*kd+0.5*ce
        opt.zero_grad();loss.backward();opt.step()
        if ep%10==0:
            student.eval()
            log(f"  [{student_arch_name}] ep={ep}: acc={ev(student,X_te,y_te):.3f}")
            student.train()
    student.eval()
    acc_dist=ev(student,X_te,y_te); pc_dist=pc(student,X_te,y_te)
    drops_dist={c:max(0,bp[c]-pc_dist[c]) for c in range(10)}
    w5_dist=sum(1 for d in drops_dist.values() if d<=0.05)
    w10_dist=sum(1 for d in drops_dist.values() if d<=0.10)
    log(f"  DISTILLED {student_arch_name}: acc={acc_dist:.4f} max_drop={max(drops_dist.values()):.3f} ≤5%={w5_dist}/10 ≤10%={w10_dist}/10")
    for c in range(10):
        log(f"    C{c}: {pc_dist[c]:.3f} (drop={drops_dist[c]:.3f})")

# ═══════════════════════════════════════════
# STEP 4: ENT — our method (WITHOUT training data access)
# ═══════════════════════════════════════════
log(f"\n{'='*50}")
log("ENT (zero-shot merge, no training data)")

W0A,b0A=list(ftA.parameters())[0].data.numpy(),list(ftA.parameters())[1].data.numpy()
W1A,b1A=list(ftA.parameters())[2].data.numpy(),list(ftA.parameters())[3].data.numpy()
W2A,b2A=list(ftA.parameters())[4].data.numpy(),list(ftA.parameters())[5].data.numpy()
W0B,b0B=list(ftB.parameters())[0].data.numpy(),list(ftB.parameters())[1].data.numpy()
W1B,b1B=list(ftB.parameters())[2].data.numpy(),list(ftB.parameters())[3].data.numpy()
W2B,b2B=list(ftB.parameters())[4].data.numpy(),list(ftB.parameters())[5].data.numpy()
ftA.eval();ftB.eval()
with torch.no_grad(): sl=ftA(Xc).numpy().std();sr=ftB(Xc).numpy().std()
t_=(sl+sr)/2;rA=t_/(sl+1e-10);rB=t_/(sr+1e-10)

def build_ent(mA0,mB0,mA1,mB1,route):
    iA0=np.where(mA0)[0];iB0=np.where(mB0)[0]
    iA1=np.where(mA1)[0];iB1=np.where(mB1)[0]
    na0,nb0,na1,nb1=len(iA0),len(iB0),len(iA1),len(iB1)
    if na0+nb0<2 or na1+nb1<2: return None
    m=MLP([784,na0+nb0,na1+nb1,10])
    with torch.no_grad():
        ps=list(m.parameters())
        ps[0].copy_(torch.tensor(np.vstack([W0A[iA0],W0B[iB0]])))
        ps[1].copy_(torch.tensor(np.concatenate([b0A[iA0],b0B[iB0]])))
        W1=np.zeros((na1+nb1,na0+nb0),dtype=np.float32)
        W1[:na1,:na0]=W1A[np.ix_(iA1,iA0)];W1[na1:,na0:]=W1B[np.ix_(iB1,iB0)]
        ps[2].copy_(torch.tensor(W1))
        ps[3].copy_(torch.tensor(np.concatenate([b1A[iA1],b1B[iB1]])))
        Wo=np.zeros((10,na1+nb1),dtype=np.float32);bo=np.zeros(10,dtype=np.float32)
        for c in range(10):
            a=1/(1+np.exp(-route[c]))
            if na1>0:Wo[c,:na1]=a*rA*W2A[c][iA1]
            if nb1>0:Wo[c,na1:]=(1-a)*rB*W2B[c][iB1]
            bo[c]=a*rA*b2A[c]+(1-a)*rB*b2B[c]
        ps[4].copy_(torch.tensor(Wo));ps[5].copy_(torch.tensor(bo))
    return m

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
        m=build_ent(ch['mA0'],ch['mB0'],ch['mA1'],ch['mB1'],ch['route'])
        if m is None:fs.append(-1);continue
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
m_ent=build_ent(bc['mA0'],bc['mB0'],bc['mA1'],bc['mB1'],bc['route'])
m_ent.eval()
acc_ent=ev(m_ent,X_te,y_te); pc_ent=pc(m_ent,X_te,y_te)
drops_ent={c:max(0,bp[c]-pc_ent[c]) for c in range(10)}
w5_ent=sum(1 for d in drops_ent.values() if d<=0.05)
w10_ent=sum(1 for d in drops_ent.values() if d<=0.10)
log(f"  ENT: acc={acc_ent:.4f} max_drop={max(drops_ent.values()):.3f} ≤5%={w5_ent}/10 ≤10%={w10_ent}/10")
for c in range(10):
    log(f"    C{c}: {pc_ent[c]:.3f} (drop={drops_ent[c]:.3f})")

# ═══════════════════════════════════════════
# STEP 5: AVERAGE — naive baseline
# ═══════════════════════════════════════════
log(f"\n{'='*50}")
log("AVERAGE (naive baseline)")
sdA=ftA.state_dict();sdB=ftB.state_dict()
m_avg=MLP(arch);m_avg.load_state_dict({k:0.5*sdA[k]+0.5*sdB[k] for k in sdA});m_avg.eval()
acc_avg=ev(m_avg,X_te,y_te); pc_avg=pc(m_avg,X_te,y_te)
drops_avg={c:max(0,bp[c]-pc_avg[c]) for c in range(10)}
w5_avg=sum(1 for d in drops_avg.values() if d<=0.05)
log(f"  Average: acc={acc_avg:.4f} max_drop={max(drops_avg.values()):.3f} ≤5%={w5_avg}/10")

# ═══════════════════════════════════════════
# STEP 6: TIES baseline
# ═══════════════════════════════════════════
log(f"\n{'='*50}")
log("TIES (best SOTA)")
m_ties=MLP(arch);sd={}
for k in base_sd:
    tvA=sdA[k]-base_sd[k];tvB=sdB[k]-base_sd[k]
    for tv in [tvA,tvB]:
        flat=tv.flatten();n_keep=max(1,int(len(flat)*0.1))
        if n_keep<len(flat):thr=flat.abs().topk(n_keep).values[-1];tv[tv.abs()<thr]=0
    elect=torch.where(tvA.abs()>=tvB.abs(),tvA.sign(),tvB.sign())
    tvAf=torch.where(tvA.sign()==elect,tvA,torch.zeros_like(tvA))
    tvBf=torch.where(tvB.sign()==elect,tvB,torch.zeros_like(tvB))
    n_nz=((tvAf!=0).float()+(tvBf!=0).float()).clamp(min=1)
    sd[k]=base_sd[k]+(tvAf+tvBf)/n_nz
m_ties.load_state_dict(sd);m_ties.eval()
acc_ties=ev(m_ties,X_te,y_te);pc_ties=pc(m_ties,X_te,y_te)
drops_ties={c:max(0,bp[c]-pc_ties[c]) for c in range(10)}
w5_ties=sum(1 for d in drops_ties.values() if d<=0.05)
log(f"  TIES: acc={acc_ties:.4f} max_drop={max(drops_ties.values()):.3f} ≤5%={w5_ties}/10")

# ═══════════════════════════════════════════
# FINAL: GAP ANALYSIS
# ═══════════════════════════════════════════
log(f"\n{'='*70}")
log("GAP ANALYSIS — Where is the room for improvement?")
log("="*70)

log(f"\n  HIERARCHY (accuracy):")
log(f"  ┌─ Ensemble (upper bound):  {acc_ens:.3f}  ≤5%={w5_ens}/10")
log(f"  │")
log(f"  │  Gap1 = {acc_ens-acc_dist:.3f} (information loss from compression)")
log(f"  │")
log(f"  ├─ Distillation (WITH data): {acc_dist:.3f}  ≤5%={w5_dist}/10")
log(f"  │")
log(f"  │  Gap2 = {acc_dist-acc_ent:.3f} ← ROOM FOR ENT IMPROVEMENT")
log(f"  │")
log(f"  ├─ ENT (zero-shot merge):    {acc_ent:.3f}  ≤5%={w5_ent}/10")
log(f"  │")
log(f"  │  Gap3 = {acc_ent-acc_ties:.3f} ← ENT's CONTRIBUTION over TIES")
log(f"  │")
log(f"  ├─ TIES:                      {acc_ties:.3f}  ≤5%={w5_ties}/10")
log(f"  │")
log(f"  │  Gap4 = {acc_ties-acc_avg:.3f}")
log(f"  │")
log(f"  └─ Average (naive):           {acc_avg:.3f}  ≤5%={w5_avg}/10")

log(f"\n  Per-class hierarchy:")
log(f"  {'C':>2} {'Parent':>7} {'Ens':>7} {'Dist':>7} {'ENT':>7} {'TIES':>7} {'Avg':>7}")
for c in range(10):
    log(f"  {c:>2} {bp[c]:>7.3f} {pc_ens[c]:>7.3f} {pc_dist[c]:>7.3f} {pc_ent[c]:>7.3f} {pc_ties[c]:>7.3f} {pc_avg[c]:>7.3f}")

log(f"\n  Per-class DROPS:")
log(f"  {'C':>2} {'Parent':>7} {'Ens':>7} {'Dist':>7} {'ENT':>7} {'TIES':>7} {'Avg':>7}")
for c in range(10):
    log(f"  {c:>2} {bp[c]:>7.3f} {drops_ens[c]:>7.3f} {drops_dist[c]:>7.3f} {drops_ent[c]:>7.3f} {drops_ties[c]:>7.3f} {drops_avg[c]:>7.3f}")

# Per-class gap analysis
log(f"\n  Per-class Gap2 (Distill − ENT = room for improvement):")
for c in range(10):
    gap2 = pc_dist[c] - pc_ent[c]
    gap_pct = gap2 / bp[c] * 100 if bp[c] > 0 else 0
    sym = "🔴" if gap2 > 0.1 else ("🟡" if gap2 > 0.03 else "✅")
    log(f"  C{c}: gap={gap2:+.3f} ({gap_pct:+.1f}%) {sym}")

R.close(); print("Done!", flush=True)
