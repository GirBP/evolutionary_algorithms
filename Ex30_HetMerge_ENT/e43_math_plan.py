#!/usr/bin/env python3
"""E43: Complete the mathematician's plan — ALL missing points.

1. FIXED ensemble (logit normalization) — true upper bound
2. Lower bound on |θ_M| — minimum model size scan
3. Redundancy analysis (MI, CKA, correlation)
4. OT neuron alignment
5. Structural analysis: student vs ENT neuron usage
6. Convergence analysis of EA
"""
import numpy as np, torch, torch.nn as nn, random, copy, time
from scipy.optimize import linear_sum_assignment
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
import ssl; ssl._create_default_https_context=ssl._create_unverified_context
from torchvision import datasets, transforms

R=open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/results_e43.txt','w')
def log(s): R.write(s+'\n'); R.flush(); print(s,flush=True)

log("="*70)
log("E43: Mathematician's Plan — ALL Missing Points")
log("="*70)

tf=transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x:x.view(-1))])
tr=datasets.MNIST('/tmp/mnist',train=True,download=True,transform=tf)
te=datasets.MNIST('/tmp/mnist',train=False,download=True,transform=tf)
X_tr=torch.stack([tr[i][0] for i in range(20000)]); y_tr=torch.tensor([tr[i][1] for i in range(20000)])
X_te=torch.stack([te[i][0] for i in range(2000)]); y_te=torch.tensor([te[i][1] for i in range(2000)])
idx=torch.randperm(20000,generator=torch.Generator().manual_seed(0))
Xv=X_tr[idx[15000:18000]]; yv=y_tr[idx[15000:18000]]
Xc=X_tr[idx[:2000]]
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

# ═══════════════════════════════════════════
# 1. FIXED ENSEMBLE — multiple variants
# ═══════════════════════════════════════════
log(f"\n{'='*60}")
log("1. FIXED ENSEMBLE — True upper bound")
log("="*60)

ftA.eval(); ftB.eval()
with torch.no_grad():
    logA=ftA(X_te); logB=ftB(X_te)

# v1: Naive (E42 version — take raw logits)
out_v1=torch.zeros_like(logA)
for c in clA: out_v1[:,c]=logA[:,c]
for c in clB: out_v1[:,c]=logB[:,c]
pc_v1={c:(out_v1.argmax(1)[y_te==c]==c).float().mean().item() for c in range(10)}
log(f"  v1 (naive):        acc={sum(pc_v1[c]*((y_te==c).sum().item()/len(y_te)) for c in range(10)):.3f}")

# v2: Softmax-normalized per model, then combine
sA=torch.softmax(logA,dim=1); sB=torch.softmax(logB,dim=1)
out_v2=torch.zeros_like(logA)
for c in clA: out_v2[:,c]=sA[:,c]
for c in clB: out_v2[:,c]=sB[:,c]
pc_v2={c:(out_v2.argmax(1)[y_te==c]==c).float().mean().item() for c in range(10)}
acc_v2=sum(pc_v2[c]*((y_te==c).sum().item()/len(y_te)) for c in range(10))
log(f"  v2 (softmax norm): acc={acc_v2:.3f}")

# v3: Temperature-scaled softmax
sA3=torch.softmax(logA/0.5,dim=1); sB3=torch.softmax(logB/0.5,dim=1)
out_v3=torch.zeros_like(logA)
for c in clA: out_v3[:,c]=sA3[:,c]
for c in clB: out_v3[:,c]=sB3[:,c]
pc_v3={c:(out_v3.argmax(1)[y_te==c]==c).float().mean().item() for c in range(10)}
acc_v3=sum(pc_v3[c]*((y_te==c).sum().item()/len(y_te)) for c in range(10))
log(f"  v3 (temp=0.5):     acc={acc_v3:.3f}")

# v4: Oracle — always pick the correct model's prediction
out_v4=torch.zeros_like(logA)
for i in range(len(y_te)):
    if y_te[i].item() in clA: out_v4[i]=logA[i]
    else: out_v4[i]=logB[i]
pc_v4={c:(out_v4.argmax(1)[y_te==c]==c).float().mean().item() for c in range(10)}
acc_v4=sum(pc_v4[c]*((y_te==c).sum().item()/len(y_te)) for c in range(10))
log(f"  v4 (ORACLE):       acc={acc_v4:.3f}  ← TRUE upper bound")

# v5: Mask-based — zero out "wrong" model's logits
out_v5=torch.full_like(logA, -1e10)
for c in clA: out_v5[:,c]=logA[:,c]
for c in clB: out_v5[:,c]=logB[:,c]
pc_v5={c:(out_v5.argmax(1)[y_te==c]==c).float().mean().item() for c in range(10)}
acc_v5=sum(pc_v5[c]*((y_te==c).sum().item()/len(y_te)) for c in range(10))
log(f"  v5 (mask -inf):    acc={acc_v5:.3f}")

log(f"\n  Per-class comparison:")
log(f"  {'C':>2} {'Parent':>7} {'Naive':>7} {'Sfmx':>7} {'T=0.5':>7} {'Oracle':>7} {'Mask':>7}")
for c in range(10):
    log(f"  {c:>2} {bp[c]:>7.3f} {pc_v1[c]:>7.3f} {pc_v2[c]:>7.3f} {pc_v3[c]:>7.3f} {pc_v4[c]:>7.3f} {pc_v5[c]:>7.3f}")

# Use best ensemble as true upper bound
best_ens_acc = max(acc_v2, acc_v3, acc_v4, acc_v5)
best_ens_name = ["v2","v3","v4","v5"][[acc_v2,acc_v3,acc_v4,acc_v5].index(best_ens_acc)]
best_ens_pc = [pc_v2,pc_v3,pc_v4,pc_v5][[acc_v2,acc_v3,acc_v4,acc_v5].index(best_ens_acc)]
log(f"\n  TRUE upper bound: {best_ens_name} acc={best_ens_acc:.3f}")

# ═══════════════════════════════════════════
# 2. LOWER BOUND — minimum model size scan
# ═══════════════════════════════════════════
log(f"\n{'='*60}")
log("2. LOWER BOUND — Minimum model size for ε-drop")
log("="*60)

# Distill into models of different sizes
for h1,h2 in [(16,8),(32,16),(64,32),(128,64),(256,128)]:
    student=MLP([784,h1,h2,10])
    with torch.no_grad():
        soft=torch.softmax(out_v4/3,dim=1)  # oracle-based soft labels
        hard=out_v4.argmax(1)
    # Use training data
    with torch.no_grad():
        soft_tr=torch.zeros(len(X_tr),10)
        lA_tr=ftA(X_tr);lB_tr=ftB(X_tr)
        for i in range(len(X_tr)):
            if y_tr[i].item() in clA: soft_tr[i]=torch.softmax(lA_tr[i]/3,dim=0)
            else: soft_tr[i]=torch.softmax(lB_tr[i]/3,dim=0)
        hard_tr=soft_tr.argmax(1)
    opt=torch.optim.Adam(student.parameters(),lr=0.002)
    student.train()
    for ep in range(40):
        ix=torch.randperm(15000)[:2000]
        out=student(X_tr[ix])
        kd=nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(out/3,dim=1),soft_tr[ix])*9
        ce=nn.CrossEntropyLoss()(out,hard_tr[ix])
        loss=0.5*kd+0.5*ce; opt.zero_grad();loss.backward();opt.step()
    student.eval()
    d=pc(student,X_te,y_te); acc=ev(student,X_te,y_te)
    drops={c:max(0,bp[c]-d[c]) for c in range(10)}
    w5=sum(1 for dr in drops.values() if dr<=0.05)
    w10=sum(1 for dr in drops.values() if dr<=0.10)
    params=sum(p.numel() for p in student.parameters())
    log(f"  [{h1:>3},{h2:>3}] params={params:>8,} acc={acc:.3f} maxdrop={max(drops.values()):.3f} ≤5%={w5}/10 ≤10%={w10}/10")

# ═══════════════════════════════════════════
# 3. REDUNDANCY ANALYSIS
# ═══════════════════════════════════════════
log(f"\n{'='*60}")
log("3. REDUNDANCY — MI, CKA, Correlation between A and B")
log("="*60)

with torch.no_grad():
    hA0=torch.relu(Xc@list(ftA.parameters())[0].data.T+list(ftA.parameters())[1].data).numpy()
    hB0=torch.relu(Xc@list(ftB.parameters())[0].data.T+list(ftB.parameters())[1].data).numpy()

# Per-neuron correlation matrix
corr_matrix=np.zeros((128,128))
for i in range(128):
    for j in range(128):
        if hA0[:,i].std()>1e-8 and hB0[:,j].std()>1e-8:
            corr_matrix[i,j]=abs(np.corrcoef(hA0[:,i],hB0[:,j])[0,1])
        else:
            corr_matrix[i,j]=0

# Statistics
log(f"  Correlation matrix stats:")
log(f"    Mean: {corr_matrix.mean():.4f}")
log(f"    Max:  {corr_matrix.max():.4f}")
log(f"    >0.7: {(corr_matrix>0.7).sum()} pairs")
log(f"    >0.5: {(corr_matrix>0.5).sum()} pairs")
log(f"    >0.3: {(corr_matrix>0.3).sum()} pairs")

# Layer-level CKA
def linear_cka(X, Y):
    X=X-X.mean(0); Y=Y-Y.mean(0)
    hsic=np.linalg.norm(X.T@Y,'fro')**2
    return hsic/(np.sqrt(np.linalg.norm(X.T@X,'fro')**2*np.linalg.norm(Y.T@Y,'fro')**2)+1e-10)
cka_l0=linear_cka(hA0,hB0)
log(f"  Layer 0 CKA: {cka_l0:.4f}")

W1A_=list(ftA.parameters())[2].data.numpy()
W1B_=list(ftB.parameters())[2].data.numpy()
hA1=np.maximum(0,hA0@W1A_.T+list(ftA.parameters())[3].data.numpy())
hB1=np.maximum(0,hB0@W1B_.T+list(ftB.parameters())[3].data.numpy())
cka_l1=linear_cka(hA1,hB1)
log(f"  Layer 1 CKA: {cka_l1:.4f}")

# Mutual information (binned approximation)
def mi_approx(x,y,bins=20):
    h_xy,_,_=np.histogram2d(x,y,bins=bins)
    h_xy=h_xy/h_xy.sum()+1e-12
    h_x=h_xy.sum(1); h_y=h_xy.sum(0)
    return np.sum(h_xy*np.log(h_xy/(h_x[:,None]*h_y[None,:]+1e-12)))

# Average MI between top-10 neurons of each model
mi_vals=[]
top_A=np.argsort(-hA0.var(0))[:10]
top_B=np.argsort(-hB0.var(0))[:10]
for i in top_A:
    for j in top_B:
        mi_vals.append(mi_approx(hA0[:,i],hB0[:,j]))
log(f"  MI (top-10 neurons): mean={np.mean(mi_vals):.4f} max={np.max(mi_vals):.4f}")
log(f"  Interpretation: {'LOW redundancy' if np.mean(mi_vals)<0.1 else 'MODERATE redundancy' if np.mean(mi_vals)<0.5 else 'HIGH redundancy'}")

# ═══════════════════════════════════════════
# 4. OT NEURON ALIGNMENT
# ═══════════════════════════════════════════
log(f"\n{'='*60}")
log("4. OT NEURON ALIGNMENT (Sinkhorn)")
log("="*60)

# Cost matrix: functional distance between neurons
# c(i,j) = 1 - |corr(hA_i, hB_j)|
cost_matrix = 1 - corr_matrix

# Hungarian algorithm (exact OT for discrete case)
row_ind, col_ind = linear_sum_assignment(cost_matrix)
total_cost = cost_matrix[row_ind, col_ind].sum()
log(f"  OT total cost: {total_cost:.2f} (max possible: {128:.0f})")
log(f"  Average match quality: {1-total_cost/128:.4f}")

# Show top matches
matches = [(i,j,1-cost_matrix[i,j]) for i,j in zip(row_ind,col_ind)]
matches.sort(key=lambda x:-x[2])
log(f"  Top-5 matches:")
for i,j,q in matches[:5]:
    log(f"    A[{i:>3}] ↔ B[{j:>3}]: quality={q:.3f}")
log(f"  Worst-5 matches:")
for i,j,q in matches[-5:]:
    log(f"    A[{i:>3}] ↔ B[{j:>3}]: quality={q:.3f}")

# Build OT-aligned merge
log(f"\n  OT-aligned merge:")
good_matches = [(i,j,q) for i,j,q in matches if q>0.3]
log(f"  Matched pairs (quality>0.3): {len(good_matches)}")
unmatched_A = [i for i in range(128) if i not in [m[0] for m in good_matches]]
unmatched_B = [j for j in range(128) if j not in [m[1] for m in good_matches]]

# Build merged layer 0: matched(avg) + unmatched_A + unmatched_B
W0A_=list(ftA.parameters())[0].data.numpy(); b0A_=list(ftA.parameters())[1].data.numpy()
W0B_=list(ftB.parameters())[0].data.numpy(); b0B_=list(ftB.parameters())[1].data.numpy()
W2A_=list(ftA.parameters())[4].data.numpy(); b2A_=list(ftA.parameters())[5].data.numpy()
W2B_=list(ftB.parameters())[4].data.numpy(); b2B_=list(ftB.parameters())[5].data.numpy()

n_m=len(good_matches); n_uA=len(unmatched_A); n_uB=len(unmatched_B)
n0=n_m+n_uA+n_uB
W0=np.zeros((n0,784),dtype=np.float32); b0=np.zeros(n0,dtype=np.float32)
for idx,(ai,bi,q) in enumerate(good_matches):
    W0[idx]=0.5*W0A_[ai]+0.5*W0B_[bi]; b0[idx]=0.5*b0A_[ai]+0.5*b0B_[bi]
for idx,ai in enumerate(unmatched_A):
    W0[n_m+idx]=W0A_[ai]; b0[n_m+idx]=b0A_[ai]
for idx,bi in enumerate(unmatched_B):
    W0[n_m+n_uA+idx]=W0B_[bi]; b0[n_m+n_uA+idx]=b0B_[bi]

# Layer 1: need remapping
map_A={}; map_B={}
for idx,(ai,bi,q) in enumerate(good_matches): map_A[ai]=idx; map_B[bi]=idx
for idx,ai in enumerate(unmatched_A): map_A[ai]=n_m+idx
for idx,bi in enumerate(unmatched_B): map_B[bi]=n_m+n_uA+idx

# Use all 64 neurons from each + routing
n1=128  # 64+64
W1=np.zeros((n1,n0),dtype=np.float32); b1=np.zeros(n1,dtype=np.float32)
for i in range(64):
    for j in range(128):
        if j in map_A: W1[i,map_A[j]]=W1A_[i,j]
    b1[i]=list(ftA.parameters())[3].data.numpy()[i]
for i in range(64):
    for j in range(128):
        if j in map_B: W1[64+i,map_B[j]]=W1B_[i,j]
    b1[64+i]=list(ftB.parameters())[3].data.numpy()[i]

# Output with routing (optimize routing only)
with torch.no_grad():
    sl=ftA(Xc).numpy().std();sr=ftB(Xc).numpy().std()
t_=(sl+sr)/2;rA_=t_/(sl+1e-10);rB_=t_/(sr+1e-10)

best_acc=-1; best_route=None
for trial in range(200):
    route=np.random.randn(10)*2 if trial>0 else np.array([2.]*5+[-2.]*5)
    Wo=np.zeros((10,n1),dtype=np.float32); bo=np.zeros(10,dtype=np.float32)
    for c in range(10):
        a=1/(1+np.exp(-route[c]))
        Wo[c,:64]=a*rA_*W2A_[c]; Wo[c,64:]=(1-a)*rB_*W2B_[c]
        bo[c]=a*rA_*b2A_[c]+(1-a)*rB_*b2B_[c]
    m=MLP([784,n0,n1,10])
    with torch.no_grad():
        ps=list(m.parameters())
        ps[0].copy_(torch.tensor(W0));ps[1].copy_(torch.tensor(b0))
        ps[2].copy_(torch.tensor(W1));ps[3].copy_(torch.tensor(b1))
        ps[4].copy_(torch.tensor(Wo));ps[5].copy_(torch.tensor(bo))
    m.eval(); acc=ev(m,Xv,yv)
    if acc>best_acc: best_acc=acc;best_route=route.copy()

# Build final OT model  
Wo=np.zeros((10,n1),dtype=np.float32); bo=np.zeros(10,dtype=np.float32)
for c in range(10):
    a=1/(1+np.exp(-best_route[c]))
    Wo[c,:64]=a*rA_*W2A_[c]; Wo[c,64:]=(1-a)*rB_*W2B_[c]
    bo[c]=a*rA_*b2A_[c]+(1-a)*rB_*b2B_[c]
m_ot=MLP([784,n0,n1,10])
with torch.no_grad():
    ps=list(m_ot.parameters())
    ps[0].copy_(torch.tensor(W0));ps[1].copy_(torch.tensor(b0))
    ps[2].copy_(torch.tensor(W1));ps[3].copy_(torch.tensor(b1))
    ps[4].copy_(torch.tensor(Wo));ps[5].copy_(torch.tensor(bo))
m_ot.eval()
acc_ot=ev(m_ot,X_te,y_te); pc_ot=pc(m_ot,X_te,y_te)
drops_ot={c:max(0,bp[c]-pc_ot[c]) for c in range(10)}
w5_ot=sum(1 for d in drops_ot.values() if d<=0.05)
log(f"  OT merge: acc={acc_ot:.3f} maxdrop={max(drops_ot.values()):.3f} ≤5%={w5_ot}/10 params={sum(p.numel() for p in m_ot.parameters()):,}")

# ═══════════════════════════════════════════
# 5. ENT for comparison
# ═══════════════════════════════════════════
log(f"\n{'='*60}")
log("5. ENT baseline (for comparison)")
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
        iA0=np.where(ch['mA0'])[0];iB0=np.where(ch['mB0'])[0]
        iA1=np.where(ch['mA1'])[0];iB1=np.where(ch['mB1'])[0]
        if len(iA0)+len(iB0)<2 or len(iA1)+len(iB1)<2:fs.append(-1);continue
        m=MLP([784,len(iA0)+len(iB0),len(iA1)+len(iB1),10])
        with torch.no_grad():
            ps=list(m.parameters())
            ps[0].copy_(torch.tensor(np.vstack([W0A_[iA0],W0B_[iB0]])))
            ps[1].copy_(torch.tensor(np.concatenate([b0A_[iA0],b0B_[iB0]])))
            W1t=np.zeros((len(iA1)+len(iB1),len(iA0)+len(iB0)),dtype=np.float32)
            W1t[:len(iA1),:len(iA0)]=W1A_[np.ix_(iA1,iA0)]
            W1t[len(iA1):,len(iA0):]=W1B_[np.ix_(iB1,iB0)]
            ps[2].copy_(torch.tensor(W1t))
            ps[3].copy_(torch.tensor(np.concatenate([list(ftA.parameters())[3].data.numpy()[iA1],list(ftB.parameters())[3].data.numpy()[iB1]])))
            Wo=np.zeros((10,len(iA1)+len(iB1)),dtype=np.float32);bo=np.zeros(10,dtype=np.float32)
            for c in range(10):
                a=1/(1+np.exp(-ch['route'][c]))
                if len(iA1)>0:Wo[c,:len(iA1)]=a*rA_*W2A_[c][iA1]
                if len(iB1)>0:Wo[c,len(iA1):]=(1-a)*rB_*W2B_[c][iB1]
                bo[c]=a*rA_*b2A_[c]+(1-a)*rB_*b2B_[c]
            ps[4].copy_(torch.tensor(Wo));ps[5].copy_(torch.tensor(bo))
        m.eval();d=pc(m,Xv,yv);acc=ev(m,Xv,yv);mn=min(d[c] for c in range(10))
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
# Build final ENT
iA0=np.where(bc['mA0'])[0];iB0=np.where(bc['mB0'])[0]
iA1=np.where(bc['mA1'])[0];iB1=np.where(bc['mB1'])[0]
m_ent=MLP([784,len(iA0)+len(iB0),len(iA1)+len(iB1),10])
with torch.no_grad():
    ps=list(m_ent.parameters())
    ps[0].copy_(torch.tensor(np.vstack([W0A_[iA0],W0B_[iB0]])))
    ps[1].copy_(torch.tensor(np.concatenate([b0A_[iA0],b0B_[iB0]])))
    W1e=np.zeros((len(iA1)+len(iB1),len(iA0)+len(iB0)),dtype=np.float32)
    W1e[:len(iA1),:len(iA0)]=W1A_[np.ix_(iA1,iA0)];W1e[len(iA1):,len(iA0):]=W1B_[np.ix_(iB1,iB0)]
    ps[2].copy_(torch.tensor(W1e))
    ps[3].copy_(torch.tensor(np.concatenate([list(ftA.parameters())[3].data.numpy()[iA1],list(ftB.parameters())[3].data.numpy()[iB1]])))
    Wo=np.zeros((10,len(iA1)+len(iB1)),dtype=np.float32);bo=np.zeros(10,dtype=np.float32)
    for c in range(10):
        a=1/(1+np.exp(-bc['route'][c]))
        if len(iA1)>0:Wo[c,:len(iA1)]=a*rA_*W2A_[c][iA1]
        if len(iB1)>0:Wo[c,len(iA1):]=(1-a)*rB_*W2B_[c][iB1]
        bo[c]=a*rA_*b2A_[c]+(1-a)*rB_*b2B_[c]
    ps[4].copy_(torch.tensor(Wo));ps[5].copy_(torch.tensor(bo))
m_ent.eval()
acc_ent=ev(m_ent,X_te,y_te);pc_ent=pc(m_ent,X_te,y_te)
drops_ent={c:max(0,bp[c]-pc_ent[c]) for c in range(10)}
w5_ent=sum(1 for d in drops_ent.values() if d<=0.05)
log(f"  ENT: acc={acc_ent:.3f} maxdrop={max(drops_ent.values()):.3f} ≤5%={w5_ent}/10")
log(f"  ENT neurons used: A0={bc['mA0'].sum()}/128 B0={bc['mB0'].sum()}/128 A1={bc['mA1'].sum()}/64 B1={bc['mB1'].sum()}/64")

# ═══════════════════════════════════════════
# 6. EA CONVERGENCE ANALYSIS
# ═══════════════════════════════════════════
log(f"\n{'='*60}")
log("6. EA CONVERGENCE — fitness over generations")
log("="*60)
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
convergence=[]
bf=-1;bc=None
for gen in range(50):  # 50 gen for convergence study
    fs=[]
    for ch in pop:
        iA0=np.where(ch['mA0'])[0];iB0=np.where(ch['mB0'])[0]
        iA1=np.where(ch['mA1'])[0];iB1=np.where(ch['mB1'])[0]
        if len(iA0)+len(iB0)<2 or len(iA1)+len(iB1)<2:fs.append(-1);continue
        m=MLP([784,len(iA0)+len(iB0),len(iA1)+len(iB1),10])
        with torch.no_grad():
            ps=list(m.parameters())
            ps[0].copy_(torch.tensor(np.vstack([W0A_[iA0],W0B_[iB0]])))
            ps[1].copy_(torch.tensor(np.concatenate([b0A_[iA0],b0B_[iB0]])))
            W1t=np.zeros((len(iA1)+len(iB1),len(iA0)+len(iB0)),dtype=np.float32)
            W1t[:len(iA1),:len(iA0)]=W1A_[np.ix_(iA1,iA0)];W1t[len(iA1):,len(iA0):]=W1B_[np.ix_(iB1,iB0)]
            ps[2].copy_(torch.tensor(W1t))
            ps[3].copy_(torch.tensor(np.concatenate([list(ftA.parameters())[3].data.numpy()[iA1],list(ftB.parameters())[3].data.numpy()[iB1]])))
            Wo=np.zeros((10,len(iA1)+len(iB1)),dtype=np.float32);bo=np.zeros(10,dtype=np.float32)
            for c in range(10):
                a=1/(1+np.exp(-ch['route'][c]))
                if len(iA1)>0:Wo[c,:len(iA1)]=a*rA_*W2A_[c][iA1]
                if len(iB1)>0:Wo[c,len(iA1):]=(1-a)*rB_*W2B_[c][iB1]
                bo[c]=a*rA_*b2A_[c]+(1-a)*rB_*b2B_[c]
            ps[4].copy_(torch.tensor(Wo));ps[5].copy_(torch.tensor(bo))
        m.eval();d=pc(m,Xv,yv);acc=ev(m,Xv,yv);mn=min(d[c] for c in range(10))
        fs.append(0.4*acc+0.4*mn+0.1*np.mean([d[c] for c in range(10)])+0.1*0.5)
    gi=np.argmax(fs)
    if fs[gi]>bf:bf=fs[gi];bc={k:v.copy() for k,v in pop[gi].items()}
    convergence.append(bf)
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

log(f"  Convergence curve:")
for g in [0,5,10,15,20,25,30,35,40,45,49]:
    if g<len(convergence):
        log(f"    Gen {g:>2}: fitness={convergence[g]:.4f}")
improvement = convergence[-1]-convergence[0]
plateau_gen = next((g for g in range(1,len(convergence)) if convergence[g]-convergence[g-1]<0.001),len(convergence))
log(f"  Total improvement: {improvement:.4f}")
log(f"  Plateau at gen: {plateau_gen}")
log(f"  80% of improvement by gen: {next((g for g in range(len(convergence)) if convergence[g]>=convergence[0]+0.8*improvement),len(convergence))}")

# ═══════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════
log(f"\n{'='*70}")
log("MATHEMATICIAN'S PLAN — COMPLETE RESULTS")
log("="*70)
log(f"\n  1. TRUE UPPER BOUND: Oracle ensemble = {acc_v4:.3f} (v4)")
log(f"     Mask ensemble = {acc_v5:.3f} (v5)")
log(f"     ENT vs true UB: {'ENT BEATS UB' if acc_ent>acc_v5 else f'UB wins by {acc_v5-acc_ent:.3f}'}")
log(f"\n  2. LOWER BOUND (distillation scan): see size table above")
log(f"\n  3. REDUNDANCY: CKA_L0={cka_l0:.3f} CKA_L1={cka_l1:.3f} MI={np.mean(mi_vals):.4f}")
log(f"     → {'LOW' if cka_l0<0.3 else 'MODERATE' if cka_l0<0.6 else 'HIGH'} redundancy → concat is justified")
log(f"\n  4. OT ALIGNMENT: acc={acc_ot:.3f} ({len(good_matches)} matched pairs)")
log(f"     OT vs ENT: {'OT wins' if acc_ot>acc_ent else f'ENT wins by {acc_ent-acc_ot:.3f}'}")
log(f"\n  5. ENT: acc={acc_ent:.3f} ≤5%={w5_ent}/10")
log(f"\n  6. EA CONVERGENCE: plateau at gen {plateau_gen}, 80% by gen {next((g for g in range(len(convergence)) if convergence[g]>=convergence[0]+0.8*improvement),len(convergence))}")

R.close(); print("Done!",flush=True)
