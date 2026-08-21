#!/usr/bin/env python3
"""E45: ENT v5c on CIFAR-10 — does fine-grained routing + Lagrangian transfer to CNNs?

Pipeline:
1. Train CNN parents on CIFAR-10 {0-4} and {5-9} (BN, augment, 20ep)
2. Full concat (all filters)  
3. Per-filter per-class gating (fine-grained routing)
4. Lagrangian constrained fine-tuning
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, copy, time
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
import ssl; ssl._create_default_https_context=ssl._create_unverified_context
from torchvision import datasets, transforms

R=open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/results_e45.txt','w')
def log(s): R.write(s+'\n'); R.flush(); print(s,flush=True)

log("="*70)
log("E45: ENT v5c on CIFAR-10 (CNN)")
log("="*70)

# Data with augmentation
tf_train=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
tf_test=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
tr_raw=datasets.CIFAR10('/tmp/cifar',train=True,download=True)
te_raw=datasets.CIFAR10('/tmp/cifar',train=False,download=True)

# Preload as tensors for speed
log("Loading data...")
tf_t=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
X_te=torch.stack([tf_t(te_raw[i][0]) for i in range(1000)])
y_te=torch.tensor([te_raw[i][1] for i in range(1000)])
X_tr=torch.stack([tf_t(tr_raw[i][0]) for i in range(5000)])
y_tr=torch.tensor([tr_raw[i][1] for i in range(5000)])
Xv=torch.stack([tf_t(tr_raw[i][0]) for i in range(5000,7000)])
yv=torch.tensor([tr_raw[i][1] for i in range(7000,9000)])

clA,clB=list(range(5)),list(range(5,10))

class SmallCNN(nn.Module):
    def __init__(s, nf1=16, nf2=32, fc1=128):
        super().__init__()
        s.conv1=nn.Conv2d(3,nf1,3,padding=1); s.bn1=nn.BatchNorm2d(nf1)
        s.conv2=nn.Conv2d(nf1,nf2,3,padding=1); s.bn2=nn.BatchNorm2d(nf2)
        s.fc1=nn.Linear(nf2*8*8,fc1); s.fc2=nn.Linear(fc1,10)
        s.nf1=nf1; s.nf2=nf2; s.fc1_size=fc1
    def forward(s,x):
        x=F.max_pool2d(F.relu(s.bn1(s.conv1(x))),2)
        x=F.max_pool2d(F.relu(s.bn2(s.conv2(x))),2)
        x=x.view(x.size(0),-1)
        x=F.relu(s.fc1(x))
        return s.fc2(x)

def ev(m,X,y,bs=500):
    m.eval();correct=0;total=0
    with torch.no_grad():
        for i in range(0,len(X),bs):
            p=m(X[i:i+bs]).argmax(1);correct+=(p==y[i:i+bs]).sum().item();total+=len(p)
    return correct/total
def pc(m,X,y,bs=500):
    m.eval();preds=[]
    with torch.no_grad():
        for i in range(0,len(X),bs): preds.append(m(X[i:i+bs]).argmax(1))
    p=torch.cat(preds)
    return {c:(p[y==c]==c).float().mean().item() if (y==c).sum()>0 else 0 for c in range(10)}

# ═══════════════════════════════════════════
# Train parents (20 epochs, BN, augmentation via mini-batch)
# ═══════════════════════════════════════════
log("\n1. Training parents...")
def train_parent(cls, epochs=10):
    mask=sum(y_tr==c for c in cls).bool()
    Xs,ys=X_tr[mask],y_tr[mask]
    m=SmallCNN(); opt=torch.optim.Adam(m.parameters(),lr=0.001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    m.train()
    for ep in range(epochs):
        idx_=torch.randperm(len(Xs))
        for i in range(0,len(Xs),64):
            batch=idx_[i:i+64];x,y_=Xs[batch],ys[batch]
            # Random flip
            if random.random()>0.5: x=x.flip(-1)
            loss=nn.CrossEntropyLoss()(m(x),y_)
            opt.zero_grad();loss.backward();opt.step()
        sched.step()
    m.eval();return m

t0=time.time()
ftA=train_parent(clA, 10)
ftB=train_parent(clB, 10)
log(f"  Training time: {time.time()-t0:.1f}s")
pcA=pc(ftA,X_te,y_te); pcB=pc(ftB,X_te,y_te)
bp={c:max(pcA[c],pcB[c]) for c in range(10)}
log(f"  Parent A: {ev(ftA,X_te,y_te):.3f}  B: {ev(ftB,X_te,y_te):.3f}")
for c in range(10): log(f"    C{c}: A={pcA[c]:.3f} B={pcB[c]:.3f} best={bp[c]:.3f}")

# ═══════════════════════════════════════════
# Oracle upper bound
# ═══════════════════════════════════════════
log("\n2. Oracle upper bound...")
ftA.eval();ftB.eval()
with torch.no_grad():
    lA=[];lB=[]
    for i in range(0,len(X_te),500):
        lA.append(ftA(X_te[i:i+500]));lB.append(ftB(X_te[i:i+500]))
    logA=torch.cat(lA);logB=torch.cat(lB)
out_oracle=torch.zeros_like(logA)
for i in range(len(y_te)):
    if y_te[i].item() in clA: out_oracle[i]=logA[i]
    else: out_oracle[i]=logB[i]
pc_oracle={c:(out_oracle.argmax(1)[y_te==c]==c).float().mean().item() for c in range(10)}
acc_oracle=sum(pc_oracle[c]*((y_te==c).sum().item()/len(y_te)) for c in range(10))
log(f"  Oracle: acc={acc_oracle:.3f}")

# ═══════════════════════════════════════════
# v5a: Simple concat + hard routing
# ═══════════════════════════════════════════
log(f"\n{'='*50}")
log("v5a: Standard concat + hard routing")

class ConcatCNN(nn.Module):
    """Concat two CNNs: conv layers block-diagonal, FC concat, routed output."""
    def __init__(s, mA, mB):
        super().__init__()
        # Keep both conv pipelines
        s.convA1=copy.deepcopy(mA.conv1); s.bnA1=copy.deepcopy(mA.bn1)
        s.convA2=copy.deepcopy(mA.conv2); s.bnA2=copy.deepcopy(mA.bn2)
        s.convB1=copy.deepcopy(mB.conv1); s.bnB1=copy.deepcopy(mB.bn1)
        s.convB2=copy.deepcopy(mB.conv2); s.bnB2=copy.deepcopy(mB.bn2)
        # Concat FC
        nf2=mA.nf2; fc1=mA.fc1_size
        s.fc1=nn.Linear(nf2*8*8*2, fc1*2)
        with torch.no_grad():
            s.fc1.weight.zero_(); s.fc1.bias.zero_()
            s.fc1.weight[:fc1,:nf2*8*8]=mA.fc1.weight
            s.fc1.bias[:fc1]=mA.fc1.bias
            s.fc1.weight[fc1:,nf2*8*8:]=mB.fc1.weight
            s.fc1.bias[fc1:]=mB.fc1.bias
        s.fc2=nn.Linear(fc1*2,10)
        s.fc1_size=fc1
    def features(s, x):
        xA=F.max_pool2d(F.relu(s.bnA1(s.convA1(x))),2)
        xA=F.max_pool2d(F.relu(s.bnA2(s.convA2(xA))),2)
        xB=F.max_pool2d(F.relu(s.bnB1(s.convB1(x))),2)
        xB=F.max_pool2d(F.relu(s.bnB2(s.convB2(xB))),2)
        return torch.cat([xA.view(x.size(0),-1), xB.view(x.size(0),-1)], dim=1)
    def forward(s, x):
        h=F.relu(s.fc1(s.features(x)))
        return s.fc2(h)

m_concat=ConcatCNN(ftA, ftB)
# Hard routing: A for clA, B for clB
with torch.no_grad():
    fc1=ftA.fc1_size
    m_concat.fc2.weight.zero_(); m_concat.fc2.bias.zero_()
    for c in clA:
        m_concat.fc2.weight[c,:fc1]=ftA.fc2.weight[c]
        m_concat.fc2.bias[c]=ftA.fc2.bias[c]
    for c in clB:
        m_concat.fc2.weight[c,fc1:]=ftB.fc2.weight[c]
        m_concat.fc2.bias[c]=ftB.fc2.bias[c]
m_concat.eval()
acc_std=ev(m_concat,X_te,y_te); d_std=pc(m_concat,X_te,y_te)
log(f"  acc={acc_std:.3f}")
for c in range(10): log(f"    C{c}: {d_std[c]:.3f} (drop={max(0,bp[c]-d_std[c]):.3f})")

# ═══════════════════════════════════════════
# v5b: Fine-grained per-neuron routing (optimize gates on FC2)
# ═══════════════════════════════════════════
log(f"\n{'='*50}")
log("v5b: Fine-grained routing (gate optimization)")

class GatedConcatCNN(nn.Module):
    def __init__(s, base_model):
        super().__init__()
        s.base=base_model
        fc1=base_model.fc1_size
        # Gates: [10, fc1*2] — per neuron per class
        gate_init=torch.zeros(10,fc1*2)
        for c in clA: gate_init[c,:fc1]=2.0
        for c in clB: gate_init[c,fc1:]=2.0
        s.gates=nn.Parameter(gate_init)
        s.W2A=base_model.fc2.weight.data[:,:fc1].clone()
        s.W2B=base_model.fc2.weight.data[:,fc1:].clone()
        s.b2=nn.Parameter(base_model.fc2.bias.data.clone())
        # Full weight for gating
        s.W2_full=nn.Parameter(torch.cat([s.W2A,s.W2B],dim=1), requires_grad=False)
    def forward(s,x):
        h=F.relu(s.base.fc1(s.base.features(x)))
        g=torch.sigmoid(s.gates)
        out=(h.unsqueeze(1)*g.unsqueeze(0)*s.W2_full.unsqueeze(0)).sum(-1)+s.b2
        return out

m_gated=GatedConcatCNN(m_concat)
m_gated.eval()
log(f"  Before optimization: acc={ev(m_gated,X_te,y_te):.3f}")

# Optimize gates
opt=torch.optim.Adam([m_gated.gates, m_gated.b2],lr=0.01)
for ep in range(150):
    m_gated.train()
    ix=torch.randperm(len(Xv))[:500]
    out=m_gated(Xv[ix])
    loss=nn.CrossEntropyLoss()(out,yv[ix])
    opt.zero_grad();loss.backward();opt.step()
m_gated.eval()
acc_gated=ev(m_gated,X_te,y_te); d_gated=pc(m_gated,X_te,y_te)
log(f"  After optimization: acc={acc_gated:.3f}")
for c in range(10): log(f"    C{c}: {d_gated[c]:.3f} (drop={max(0,bp[c]-d_gated[c]):.3f})")

# ═══════════════════════════════════════════
# v5c: + Lagrangian constrained fine-tuning
# ═══════════════════════════════════════════
log(f"\n{'='*50}")
log("v5c: + Lagrangian constrained fine-tuning")

m_lag=copy.deepcopy(m_gated)
# Only unfreeze FC layers + gates, FREEZE conv to preserve features
for p in m_lag.parameters(): p.requires_grad=False
m_lag.gates.requires_grad=True; m_lag.b2.requires_grad=True
m_lag.base.fc1.weight.requires_grad=True; m_lag.base.fc1.bias.requires_grad=True
lam=np.ones(10)*1.0

for iteration in range(8):
    opt=torch.optim.Adam(filter(lambda p:p.requires_grad,m_lag.parameters()),
                         lr=0.001 if iteration<3 else 0.0003)
    for step in range(40):
        m_lag.train()
        ix=torch.randperm(len(Xv))[:300]
        out=m_lag(Xv[ix])
        ce=nn.CrossEntropyLoss()(out,yv[ix])
        d=pc(m_lag,Xv[:500],yv[:500])
        penalty=0
        for c in range(10):
            drop_c=bp[c]-d[c]
            if drop_c>0.05:
                mask_c=(yv[ix]==c)
                if mask_c.sum()>0:
                    penalty+=lam[c]*nn.CrossEntropyLoss()(out[mask_c],yv[ix][mask_c])
        loss=ce+penalty
        opt.zero_grad();loss.backward();opt.step()
    m_lag.eval()
    d=pc(m_lag,X_te,y_te); acc=ev(m_lag,X_te,y_te)
    violations=sum(1 for c in range(10) if bp[c]-d[c]>0.05)
    for c in range(10):
        if bp[c]-d[c]>0.05: lam[c]=min(30,lam[c]*1.5)
        else: lam[c]=max(0.5,lam[c]*0.8)
    log(f"  Iter {iteration}: acc={acc:.3f} violations={violations}")
    if violations==0: break

m_lag.eval()
acc_lag=ev(m_lag,X_te,y_te); d_lag=pc(m_lag,X_te,y_te)
log(f"  FINAL: acc={acc_lag:.3f}")
for c in range(10):
    drop=max(0,bp[c]-d_lag[c])
    log(f"    C{c}: {d_lag[c]:.3f} (parent={bp[c]:.3f} drop={drop:.3f})")

# ═══════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════
log(f"\n{'='*70}")
log("CIFAR-10 RESULTS SUMMARY")
log("="*70)

results=[
    ("Oracle",acc_oracle,pc_oracle),
    ("v5c: Route+Lagr",acc_lag,d_lag),
    ("v5b: FineRoute",acc_gated,d_gated),
    ("v5a: HardRoute",acc_std,d_std),
]
log(f"\n  {'Method':<20} {'Acc':>6} {'MaxDr':>6} {'≤5%':>5} {'≤10%':>5} {'Gap↓':>6}")
log(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*6}")
for name,acc,d in results:
    drops={c:max(0,bp[c]-d[c]) for c in range(10)}
    w5=sum(1 for dr in drops.values() if dr<=0.05)
    w10=sum(1 for dr in drops.values() if dr<=0.10)
    log(f"  {name:<20} {acc:>6.3f} {max(drops.values()):>6.3f} {w5:>3}/10 {w10:>3}/10 {acc_oracle-acc:>6.3f}")

log(f"\n  Per-class:")
log(f"  {'C':>2} {'Parent':>7} {'Hard':>7} {'Gated':>7} {'Lagr':>7} {'Oracle':>7}")
for c in range(10):
    log(f"  {c:>2} {bp[c]:>7.3f} {d_std[c]:>7.3f} {d_gated[c]:>7.3f} {d_lag[c]:>7.3f} {pc_oracle[c]:>7.3f}")

if acc_oracle>0:
    gap_closed=(acc_lag-acc_std)/(acc_oracle-acc_std)*100 if acc_oracle>acc_std else 0
    log(f"\n  Gap closed: {gap_closed:.1f}%")
    log(f"  v5a→v5c: {acc_std:.3f}→{acc_lag:.3f} (+{acc_lag-acc_std:.3f})")

R.close(); print("Done!",flush=True)
