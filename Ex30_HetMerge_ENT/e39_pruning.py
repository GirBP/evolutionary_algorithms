#!/usr/bin/env python3
"""E39: Does pre-merge pruning improve ENT quality?

Hypothesis: Pruning low-importance neurons BEFORE merge reduces noise
and improves per-class accuracy retention.

Test: ENT with 0%, 20%, 40%, 60% pruning of each parent.
Pruning criteria: L1 norm of outgoing weights + activation magnitude.
"""
import numpy as np, torch, torch.nn as nn, random, copy, time
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
import ssl; ssl._create_default_https_context=ssl._create_unverified_context
from torchvision import datasets, transforms

R=open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/results_e39.txt','w')
def log(s): R.write(s+'\n'); R.flush(); print(s,flush=True)

log("="*70)
log("E39: Pre-merge Pruning — Does it help?")
log("="*70)

tf=transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x:x.view(-1))])
tr=datasets.MNIST('/tmp/mnist',train=True,download=True,transform=tf)
te=datasets.MNIST('/tmp/mnist',train=False,download=True,transform=tf)
X_tr=torch.stack([tr[i][0] for i in range(20000)]); y_tr=torch.tensor([tr[i][1] for i in range(20000)])
X_te=torch.stack([te[i][0] for i in range(2000)]); y_te=torch.tensor([te[i][1] for i in range(2000)])
idx=torch.randperm(20000,generator=torch.Generator().manual_seed(0))
Xv,yv=X_tr[idx[15000:18000]],y_tr[idx[15000:18000]]; Xc=X_tr[idx[:2000]]
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

# Train parents (shared base)
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
log(f"Parents: A={ev(ftA,X_te,y_te):.3f} B={ev(ftB,X_te,y_te):.3f}")
for c in range(10):
    bp=max(pcA[c],pcB[c]); log(f"  Class {c}: best_parent={bp:.3f}")

# ═══════════════════════════════════════════
# Pruning function
# ═══════════════════════════════════════════
def prune_mlp(model, keep_frac, Xcal):
    """Structured pruning: remove neurons by combined importance.
    Importance = L1_out * activation_magnitude on calibration data."""
    model.eval()
    ps = list(model.parameters())
    # Layer 0: 784→128 (W0, b0)
    # Layer 1: 128→64  (W1, b1)
    # Layer 2: 64→10   (W2, b2)
    W0,b0 = ps[0].data, ps[1].data  # [128, 784], [128]
    W1,b1 = ps[2].data, ps[3].data  # [64, 128], [64]
    W2,b2 = ps[4].data, ps[5].data  # [10, 64], [10]

    # Get activations
    with torch.no_grad():
        h0 = torch.relu(Xcal @ W0.T + b0)  # [N, 128]
        h1 = torch.relu(h0 @ W1.T + b1)     # [N, 64]

    # Importance for layer 0 neurons (128)
    l1_out_0 = W1.abs().sum(0)  # [128] — how much layer 1 uses each neuron
    act_mag_0 = h0.abs().mean(0)  # [128]
    imp_0 = l1_out_0 * act_mag_0

    # Importance for layer 1 neurons (64)
    l1_out_1 = W2.abs().sum(0)  # [64]
    act_mag_1 = h1.abs().mean(0)  # [64]
    imp_1 = l1_out_1 * act_mag_1

    # Select top neurons
    n0 = max(4, int(128 * keep_frac))
    n1 = max(4, int(64 * keep_frac))
    idx0 = imp_0.argsort(descending=True)[:n0].sort().values
    idx1 = imp_1.argsort(descending=True)[:n1].sort().values

    # Build pruned model
    pm = MLP([784, n0, n1, 10])
    with torch.no_grad():
        pps = list(pm.parameters())
        pps[0].copy_(W0[idx0])
        pps[1].copy_(b0[idx0])
        pps[2].copy_(W1[idx1][:, idx0])
        pps[3].copy_(b1[idx1])
        pps[4].copy_(W2[:, idx1])
        pps[5].copy_(b2)
    pm.eval()
    return pm, n0, n1

# ═══════════════════════════════════════════
# ENT core (from pruned models)
# ═══════════════════════════════════════════
def run_ent_from_models(mA, mB, Xc, Xv, yv, pop_sz=20, n_gen=30):
    """Run ENT directly on model weights (no virt2L — use actual weights)."""
    psA = list(mA.parameters())
    psB = list(mB.parameters())
    # Extract weights
    W0A,b0A = psA[0].data.numpy(), psA[1].data.numpy()
    W1A,b1A = psA[2].data.numpy(), psA[3].data.numpy()
    W2A,b2A = psA[4].data.numpy(), psA[5].data.numpy()
    W0B,b0B = psB[0].data.numpy(), psB[1].data.numpy()
    W1B,b1B = psB[2].data.numpy(), psB[3].data.numpy()
    W2B,b2B = psB[4].data.numpy(), psB[5].data.numpy()

    sA = [W0A.shape[0], W1A.shape[0]]  # [n0_A, n1_A]
    sB = [W0B.shape[0], W1B.shape[0]]

    # Scale normalization
    mA.eval(); mB.eval()
    with torch.no_grad():
        sl=mA(Xc).numpy().std(); sr=mB(Xc).numpy().std()
    t_=(sl+sr)/2; rA=t_/(sl+1e-10); rB=t_/(sr+1e-10)

    def build(ch):
        iA0=np.where(ch['mA'][0])[0]; iB0=np.where(ch['mB'][0])[0]
        iA1=np.where(ch['mA'][1])[0]; iB1=np.where(ch['mB'][1])[0]
        if len(iA0)+len(iB0)<2 or len(iA1)+len(iB1)<2: return None
        sizes=[784,len(iA0)+len(iB0),len(iA1)+len(iB1),10]
        # Layer 0: concat
        nW0=np.vstack([W0A[iA0],W0B[iB0]]); nb0=np.concatenate([b0A[iA0],b0B[iB0]])
        # Layer 1: block-diagonal
        nW1=np.zeros((len(iA1)+len(iB1),len(iA0)+len(iB0)),dtype=np.float32)
        nb1=np.zeros(len(iA1)+len(iB1),dtype=np.float32)
        nW1[:len(iA1),:len(iA0)]=W1A[np.ix_(iA1,iA0)]; nb1[:len(iA1)]=b1A[iA1]
        nW1[len(iA1):,len(iA0):]=W1B[np.ix_(iB1,iB0)]; nb1[len(iA1):]=b1B[iB1]
        # Output: routing
        nW2=np.zeros((10,len(iA1)+len(iB1)),dtype=np.float32); nb2=np.zeros(10,dtype=np.float32)
        for c in range(10):
            a=1/(1+np.exp(-ch['route'][c]))
            if len(iA1)>0: nW2[c,:len(iA1)]=a*rA*W2A[c][iA1]
            if len(iB1)>0: nW2[c,len(iA1):]=(1-a)*rB*W2B[c][iB1]
            nb2[c]=a*rA*b2A[c]+(1-a)*rB*b2B[c]
        m=MLP(sizes)
        with torch.no_grad():
            ps=list(m.parameters())
            ps[0].copy_(torch.tensor(nW0));ps[1].copy_(torch.tensor(nb0))
            ps[2].copy_(torch.tensor(nW1));ps[3].copy_(torch.tensor(nb1))
            ps[4].copy_(torch.tensor(nW2));ps[5].copy_(torch.tensor(nb2))
        return m

    # EA
    pop=[]
    for _ in range(pop_sz):
        ch={'mA':[np.random.random(d)>0.3 for d in sA],'mB':[np.random.random(d)>0.3 for d in sB],'route':np.random.randn(10)*1.5}
        for ms in [ch['mA'],ch['mB']]:
            for m_ in ms:
                if m_.sum()==0: m_[0]=True
        pop.append(ch)
    pop[0]={'mA':[np.ones(d,dtype=bool) for d in sA],'mB':[np.ones(d,dtype=bool) for d in sB],'route':np.array([2.]*5+[-2.]*5)}

    bf=-1;bc=None
    for gen in range(n_gen):
        fs=[]
        for ch in pop:
            m=build(ch)
            if m is None: fs.append(-1);continue
            d=pc(m,Xv,yv);acc=ev(m,Xv,yv);mn=min(d[c] for c in range(10))
            compr=1-sum(ch['mA'][i].sum()+ch['mB'][i].sum() for i in range(2))/(sum(sA)+sum(sB))
            fs.append(0.4*acc+0.4*mn+0.1*np.mean([d[c] for c in range(10)])+0.1*compr)
        gi=np.argmax(fs)
        if fs[gi]>bf: bf=fs[gi];bc={'mA':[m.copy() for m in pop[gi]['mA']],'mB':[m.copy() for m in pop[gi]['mB']],'route':pop[gi]['route'].copy()}
        new=[{'mA':[m.copy() for m in bc['mA']],'mB':[m.copy() for m in bc['mB']],'route':bc['route'].copy()}]
        while len(new)<pop_sz:
            ti=random.sample(range(len(pop)),3);p1=pop[ti[np.argmax([fs[i] for i in ti])]]
            ch={'mA':[m.copy() for m in p1['mA']],'mB':[m.copy() for m in p1['mB']],'route':p1['route']+np.random.randn(10)*0.3}
            pf=max(0.02,0.06-gen*0.001)
            for ms in [ch['mA'],ch['mB']]:
                for m_ in ms: f=np.random.random(len(m_))<pf;m_[f]=~m_[f];(m_.__setitem__(np.random.randint(len(m_)),True) if m_.sum()==0 else None)
            new.append(ch)
        pop=new
    return build(bc), bc

# ═══════════════════════════════════════════
# EXPERIMENT: Different pruning levels
# ═══════════════════════════════════════════
log(f"\n{'='*70}")
log("PRUNING ABLATION")
log("="*70)

results = []
for keep in [1.0, 0.8, 0.6, 0.4]:
    prune_pct = int((1-keep)*100)
    log(f"\n--- Pruning {prune_pct}% (keep {int(keep*100)}%) ---")

    if keep < 1.0:
        pA, n0A, n1A = prune_mlp(ftA, keep, Xc)
        pB, n0B, n1B = prune_mlp(ftB, keep, Xc)
        log(f"  Pruned A: [{784},{n0A},{n1A},10] acc={ev(pA,X_te,y_te):.3f}")
        log(f"  Pruned B: [{784},{n0B},{n1B},10] acc={ev(pB,X_te,y_te):.3f}")
        pcPA = pc(pA, X_te, y_te); pcPB = pc(pB, X_te, y_te)
        # Per-class retention after pruning
        drop_prune_A = [max(0, pcA[c]-pcPA[c]) for c in range(10)]
        drop_prune_B = [max(0, pcB[c]-pcPB[c]) for c in range(10)]
        log(f"  Pruning drop A: max={max(drop_prune_A):.3f} mean={np.mean(drop_prune_A):.3f}")
        log(f"  Pruning drop B: max={max(drop_prune_B):.3f} mean={np.mean(drop_prune_B):.3f}")
    else:
        pA, pB = ftA, ftB

    # Run ENT
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    m_ent, bc = run_ent_from_models(pA, pB, Xc, Xv, yv, pop_sz=20, n_gen=30)
    m_ent.eval()
    acc = ev(m_ent, X_te, y_te)
    pcM = pc(m_ent, X_te, y_te)
    bal_A = np.mean([pcM[c] for c in clA])
    bal_B = np.mean([pcM[c] for c in clB])
    bal = min(bal_A, bal_B)/(max(bal_A, bal_B)+1e-10)
    mn = min(pcM[c] for c in range(10))
    ok = sum(1 for c in range(10) if pcM[c]>0.3)
    params = sum(p.numel() for p in m_ent.parameters())

    # Per-class drop vs best parent
    drops = []
    for c in range(10):
        bp = max(pcA[c], pcB[c])
        drop = max(0, bp - pcM[c])
        drops.append(drop)

    r = {'keep': keep, 'prune_pct': prune_pct, 'acc': acc, 'bal': bal,
         'min': mn, 'ok': ok, 'params': params, 'pcM': pcM, 'drops': drops,
         'max_drop': max(drops), 'mean_drop': np.mean(drops),
         'within_5': sum(1 for d in drops if d <= 0.05),
         'within_10': sum(1 for d in drops if d <= 0.10)}
    results.append(r)

    log(f"  ENT: acc={acc:.3f} bal={bal:.3f} ok={ok}/10 min={mn:.3f} params={params:,}")
    log(f"  Drop vs best parent: max={max(drops):.3f} mean={np.mean(drops):.3f}")
    log(f"  Classes within 5% drop: {r['within_5']}/10")
    log(f"  Classes within 10% drop: {r['within_10']}/10")
    log(f"  Per-class:")
    for c in range(10):
        bp = max(pcA[c], pcB[c])
        sym = "✅" if drops[c]<=0.05 else ("🟡" if drops[c]<=0.10 else "❌")
        log(f"    {c}: parent={bp:.3f} merged={pcM[c]:.3f} drop={drops[c]:.3f} {sym}")

# ═══════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════
log(f"\n{'='*70}")
log("PRUNING ABLATION SUMMARY")
log("="*70)
log(f"  {'Pruned':>7} {'Acc':>6} {'Bal':>6} {'Min':>6} {'OK':>5} {'MaxDrop':>8} {'≤5%':>5} {'≤10%':>5} {'Params':>8}")
log(f"  {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*8} {'-'*5} {'-'*5} {'-'*8}")
for r in results:
    log(f"  {r['prune_pct']:>5}%  {r['acc']:>6.3f} {r['bal']:>6.3f} {r['min']:>6.3f} {r['ok']:>3}/10 {r['max_drop']:>8.3f} {r['within_5']:>3}/10 {r['within_10']:>3}/10 {r['params']:>8,}")

# Per-class comparison
log(f"\n  Per-class drop (← lower is better):")
log(f"  {'C':>2}" + "".join(f"  {'P'+str(r['prune_pct'])+'%':>7}" for r in results))
for c in range(10):
    bp = max(pcA[c], pcB[c])
    log(f"  {c:>2}" + "".join(f"  {r['drops'][c]:>7.3f}" for r in results) + f"  (parent={bp:.3f})")

R.close(); print("Done!",flush=True)
