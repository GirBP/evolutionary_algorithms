#!/usr/bin/env python3
"""E40: ENT with CROSS-CONNECTIONS — breaking block-diagonal isolation.

Current ENT problem:
  H1 = [A_neurons | B_neurons]
  H2 = block_diag(W1_A, W1_B) @ H1  — A_h2 ONLY sees A_h1, B_h2 ONLY sees B_h1

Fix: Allow cross-connections (bridge weights) between A and B subnetworks.
  H2 = [block_diag(W1_A, W1_B) + Bridge] @ H1
  Bridge is sparse, selected by EA.

Also test: CKA neuron matching — merge functionally similar neurons.
"""
import numpy as np, torch, torch.nn as nn, random, copy, time
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
import ssl; ssl._create_default_https_context=ssl._create_unverified_context
from torchvision import datasets, transforms

R=open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/results_e40.txt','w')
def log(s): R.write(s+'\n'); R.flush(); print(s,flush=True)

log("="*70)
log("E40: Breaking Block-Diagonal — Cross-Connections + CKA Matching")
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

# Train parents
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
log(f"Parents: A={ev(ftA,X_te,y_te):.3f} B={ev(ftB,X_te,y_te):.3f}")

# Extract weights
psA=list(ftA.parameters()); psB=list(ftB.parameters())
W0A,b0A=psA[0].data.numpy(),psA[1].data.numpy()
W1A,b1A=psA[2].data.numpy(),psA[3].data.numpy()
W2A,b2A=psA[4].data.numpy(),psA[5].data.numpy()
W0B,b0B=psB[0].data.numpy(),psB[1].data.numpy()
W1B,b1B=psB[2].data.numpy(),psB[3].data.numpy()
W2B,b2B=psB[4].data.numpy(),psB[5].data.numpy()

# Scale normalization
ftA.eval();ftB.eval()
with torch.no_grad(): sl=ftA(Xc).numpy().std();sr=ftB(Xc).numpy().std()
t_=(sl+sr)/2;rA=t_/(sl+1e-10);rB=t_/(sr+1e-10)

# ═══════════════════════════════════════════
# CKA: Find functionally similar neurons
# ═══════════════════════════════════════════
log("\n1. CKA neuron matching...")
ftA.eval(); ftB.eval()
with torch.no_grad():
    h0A = torch.relu(Xc @ psA[0].data.T + psA[1].data).numpy()  # [N, 128]
    h0B = torch.relu(Xc @ psB[0].data.T + psB[1].data).numpy()  # [N, 128]
    h1A = np.maximum(0, h0A @ W1A.T + b1A)  # [N, 64]
    h1B = np.maximum(0, h0B @ W1B.T + b1B)  # [N, 64]

def linear_cka(X, Y):
    """CKA between two sets of activations."""
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro')**2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro')**2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro')**2
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)

# Per-neuron CKA: correlation between each A-neuron and each B-neuron
cka0 = np.zeros((128, 128))  # layer 0
for i in range(128):
    for j in range(128):
        cka0[i,j] = abs(np.corrcoef(h0A[:,i], h0B[:,j])[0,1])

cka1 = np.zeros((64, 64))  # layer 1
for i in range(64):
    for j in range(64):
        cka1[i,j] = abs(np.corrcoef(h1A[:,i], h1B[:,j])[0,1])

# Find matched pairs (high correlation)
threshold = 0.7
matches_0 = []
for i in range(128):
    j = cka0[i].argmax()
    if cka0[i,j] > threshold and cka0[:,j].argmax() == i:  # mutual best match
        matches_0.append((i, j, cka0[i,j]))
matches_1 = []
for i in range(64):
    j = cka1[i].argmax()
    if cka1[i,j] > threshold and cka1[:,j].argmax() == i:
        matches_1.append((i, j, cka1[i,j]))

log(f"  Layer 0: {len(matches_0)} matched pairs (threshold={threshold})")
log(f"  Layer 1: {len(matches_1)} matched pairs")
if matches_0: log(f"    Best match: A[{matches_0[0][0]}]-B[{matches_0[0][1]}] corr={matches_0[0][2]:.3f}")

# ═══════════════════════════════════════════
# METHOD 1: Standard ENT (baseline)
# ═══════════════════════════════════════════
log(f"\n{'='*50}")
log("METHOD 1: Standard ENT (block-diagonal, no cross)")

def build_standard(ch):
    iA0=np.where(ch['mA0'])[0]; iB0=np.where(ch['mB0'])[0]
    iA1=np.where(ch['mA1'])[0]; iB1=np.where(ch['mB1'])[0]
    na0,nb0,na1,nb1=len(iA0),len(iB0),len(iA1),len(iB1)
    if na0+nb0<2 or na1+nb1<2: return None
    sizes=[784,na0+nb0,na1+nb1,10]
    m=MLP(sizes)
    with torch.no_grad():
        ps=list(m.parameters())
        ps[0].copy_(torch.tensor(np.vstack([W0A[iA0],W0B[iB0]])))
        ps[1].copy_(torch.tensor(np.concatenate([b0A[iA0],b0B[iB0]])))
        # Block-diagonal layer 1
        W1=np.zeros((na1+nb1,na0+nb0),dtype=np.float32)
        W1[:na1,:na0]=W1A[np.ix_(iA1,iA0)]; W1[na1:,na0:]=W1B[np.ix_(iB1,iB0)]
        b1=np.concatenate([b1A[iA1],b1B[iB1]])
        ps[2].copy_(torch.tensor(W1)); ps[3].copy_(torch.tensor(b1))
        # Output routing
        Wo=np.zeros((10,na1+nb1),dtype=np.float32); bo=np.zeros(10,dtype=np.float32)
        for c in range(10):
            a=1/(1+np.exp(-ch['route'][c]))
            if na1>0: Wo[c,:na1]=a*rA*W2A[c][iA1]
            if nb1>0: Wo[c,na1:]=(1-a)*rB*W2B[c][iB1]
            bo[c]=a*rA*b2A[c]+(1-a)*rB*b2B[c]
        ps[4].copy_(torch.tensor(Wo)); ps[5].copy_(torch.tensor(bo))
    return m

# ═══════════════════════════════════════════
# METHOD 2: ENT + Cross-Connections (bridge)
# ═══════════════════════════════════════════
log(f"\n{'='*50}")
log("METHOD 2: ENT + Cross-Connections")

def build_cross(ch):
    """Like standard but with cross-connections in layer 1."""
    iA0=np.where(ch['mA0'])[0]; iB0=np.where(ch['mB0'])[0]
    iA1=np.where(ch['mA1'])[0]; iB1=np.where(ch['mB1'])[0]
    na0,nb0,na1,nb1=len(iA0),len(iB0),len(iA1),len(iB1)
    if na0+nb0<2 or na1+nb1<2: return None
    sizes=[784,na0+nb0,na1+nb1,10]
    m=MLP(sizes)
    with torch.no_grad():
        ps=list(m.parameters())
        ps[0].copy_(torch.tensor(np.vstack([W0A[iA0],W0B[iB0]])))
        ps[1].copy_(torch.tensor(np.concatenate([b0A[iA0],b0B[iB0]])))
        # Block-diagonal WITH cross-connections
        W1=np.zeros((na1+nb1,na0+nb0),dtype=np.float32)
        W1[:na1,:na0]=W1A[np.ix_(iA1,iA0)]; W1[na1:,na0:]=W1B[np.ix_(iB1,iB0)]
        # BRIDGE: cross-connections scaled by bridge_alpha
        bridge_alpha = ch.get('bridge', 0.1)
        # A_h1 neurons can see B_h0 features (scaled)
        # Use matched CKA pairs for informed cross-connections
        for mi, mj, corr in matches_0:
            if mi in iA0 and mj in iB0:
                ai = np.where(iA0==mi)[0]
                bj = np.where(iB0==mj)[0]
                if len(ai)>0 and len(bj)>0:
                    # A's layer1 neurons also receive from matched B's layer0 neuron
                    for ki in range(na1):
                        W1[ki, na0+bj[0]] = bridge_alpha * W1A[iA1[ki], mi] * corr
                    for ki in range(nb1):
                        W1[na1+ki, ai[0]] = bridge_alpha * W1B[iB1[ki], mj] * corr
        b1=np.concatenate([b1A[iA1],b1B[iB1]])
        ps[2].copy_(torch.tensor(W1)); ps[3].copy_(torch.tensor(b1))
        # Output routing (same)
        Wo=np.zeros((10,na1+nb1),dtype=np.float32); bo=np.zeros(10,dtype=np.float32)
        for c in range(10):
            a=1/(1+np.exp(-ch['route'][c]))
            if na1>0: Wo[c,:na1]=a*rA*W2A[c][iA1]
            if nb1>0: Wo[c,na1:]=(1-a)*rB*W2B[c][iB1]
            bo[c]=a*rA*b2A[c]+(1-a)*rB*b2B[c]
        ps[4].copy_(torch.tensor(Wo)); ps[5].copy_(torch.tensor(bo))
    return m

# ═══════════════════════════════════════════
# METHOD 3: CKA Merge — average matched, concat unmatched
# ═══════════════════════════════════════════
log(f"\n{'='*50}")
log("METHOD 3: CKA Merge (average matched neurons)")

def build_cka_merge(ch):
    """Merge matched neurons (average), concat unmatched."""
    matched_A0 = set(m[0] for m in matches_0)
    matched_B0 = set(m[1] for m in matches_0)
    matched_A1 = set(m[0] for m in matches_1)
    matched_B1 = set(m[1] for m in matches_1)

    # Unmatched neurons
    unm_A0 = [i for i in range(128) if i not in matched_A0]
    unm_B0 = [i for i in range(128) if i not in matched_B0]
    unm_A1 = [i for i in range(64) if i not in matched_A1]
    unm_B1 = [i for i in range(64) if i not in matched_B1]

    # Layer 0: merged_matched + unmatched_A + unmatched_B
    n_merged_0 = len(matches_0)
    n_unmA_0 = len(unm_A0); n_unmB_0 = len(unm_B0)
    n0_total = n_merged_0 + n_unmA_0 + n_unmB_0

    n_merged_1 = len(matches_1)
    n_unmA_1 = len(unm_A1); n_unmB_1 = len(unm_B1)
    n1_total = n_merged_1 + n_unmA_1 + n_unmB_1

    if n0_total < 2 or n1_total < 2: return None

    # Build layer 0 weights
    W0_merged = np.zeros((n0_total, 784), dtype=np.float32)
    b0_merged = np.zeros(n0_total, dtype=np.float32)
    # Merged pairs (average)
    for idx, (ai, bi, corr) in enumerate(matches_0):
        alpha = 0.5  # could optimize
        W0_merged[idx] = alpha * W0A[ai] + (1-alpha) * W0B[bi]
        b0_merged[idx] = alpha * b0A[ai] + (1-alpha) * b0B[bi]
    # Unmatched A
    for idx, ai in enumerate(unm_A0):
        W0_merged[n_merged_0 + idx] = W0A[ai]
        b0_merged[n_merged_0 + idx] = b0A[ai]
    # Unmatched B
    for idx, bi in enumerate(unm_B0):
        W0_merged[n_merged_0 + n_unmA_0 + idx] = W0B[bi]
        b0_merged[n_merged_0 + n_unmA_0 + idx] = b0B[bi]

    # Build layer 1 — more complex: need to remap indices
    # Create index mapping: original neuron → position in merged layer 0
    map_A0 = {}; map_B0 = {}
    for idx, (ai, bi, _) in enumerate(matches_0):
        map_A0[ai] = idx; map_B0[bi] = idx
    for idx, ai in enumerate(unm_A0): map_A0[ai] = n_merged_0 + idx
    for idx, bi in enumerate(unm_B0): map_B0[bi] = n_merged_0 + n_unmA_0 + idx

    W1_merged = np.zeros((n1_total, n0_total), dtype=np.float32)
    b1_merged = np.zeros(n1_total, dtype=np.float32)
    # Merged layer 1 pairs
    for idx, (ai, bi, corr) in enumerate(matches_1):
        for src_a in range(128):
            if src_a in map_A0: W1_merged[idx, map_A0[src_a]] += 0.5 * W1A[ai, src_a]
        for src_b in range(128):
            if src_b in map_B0: W1_merged[idx, map_B0[src_b]] += 0.5 * W1B[bi, src_b]
        b1_merged[idx] = 0.5 * b1A[ai] + 0.5 * b1B[bi]
    # Unmatched A layer 1
    for idx, ai in enumerate(unm_A1):
        for src_a in range(128):
            if src_a in map_A0: W1_merged[n_merged_1 + idx, map_A0[src_a]] = W1A[ai, src_a]
        b1_merged[n_merged_1 + idx] = b1A[ai]
    # Unmatched B layer 1
    for idx, bi in enumerate(unm_B1):
        for src_b in range(128):
            if src_b in map_B0: W1_merged[n_merged_1 + n_unmA_1 + idx, map_B0[src_b]] = W1B[bi, src_b]
        b1_merged[n_merged_1 + n_unmA_1 + idx] = b1B[bi]

    # Output layer with routing
    W2_merged = np.zeros((10, n1_total), dtype=np.float32)
    b2_merged = np.zeros(10, dtype=np.float32)
    # Create layer 1 index mapping
    map_A1 = {}; map_B1 = {}
    for idx, (ai, bi, _) in enumerate(matches_1):
        map_A1[ai] = idx; map_B1[bi] = idx
    for idx, ai in enumerate(unm_A1): map_A1[ai] = n_merged_1 + idx
    for idx, bi in enumerate(unm_B1): map_B1[bi] = n_merged_1 + n_unmA_1 + idx

    for c in range(10):
        a = 1/(1+np.exp(-ch['route'][c]))
        for src_a in range(64):
            if src_a in map_A1: W2_merged[c, map_A1[src_a]] += a * rA * W2A[c, src_a]
        for src_b in range(64):
            if src_b in map_B1: W2_merged[c, map_B1[src_b]] += (1-a) * rB * W2B[c, src_b]
        b2_merged[c] = a*rA*b2A[c] + (1-a)*rB*b2B[c]

    sizes = [784, n0_total, n1_total, 10]
    m = MLP(sizes)
    with torch.no_grad():
        ps=list(m.parameters())
        ps[0].copy_(torch.tensor(W0_merged)); ps[1].copy_(torch.tensor(b0_merged))
        ps[2].copy_(torch.tensor(W1_merged)); ps[3].copy_(torch.tensor(b1_merged))
        ps[4].copy_(torch.tensor(W2_merged)); ps[5].copy_(torch.tensor(b2_merged))
    return m

# ═══════════════════════════════════════════
# Run all methods
# ═══════════════════════════════════════════

def run_ea(build_fn, pop_sz=20, n_gen=30, label="", extra_genes=None):
    pop = []
    for _ in range(pop_sz):
        ch = {'mA0':np.random.random(128)>0.3, 'mB0':np.random.random(128)>0.3,
              'mA1':np.random.random(64)>0.3, 'mB1':np.random.random(64)>0.3,
              'route':np.random.randn(10)*1.5}
        if extra_genes: ch.update({k:v() for k,v in extra_genes.items()})
        for k in ['mA0','mB0','mA1','mB1']:
            if ch[k].sum()==0: ch[k][0]=True
        pop.append(ch)
    pop[0] = {'mA0':np.ones(128,dtype=bool),'mB0':np.ones(128,dtype=bool),
              'mA1':np.ones(64,dtype=bool),'mB1':np.ones(64,dtype=bool),
              'route':np.array([2.]*5+[-2.]*5)}
    if extra_genes: pop[0].update({k:v() for k,v in extra_genes.items()})

    bf=-1;bc=None
    for gen in range(n_gen):
        fs=[]
        for ch in pop:
            m=build_fn(ch)
            if m is None: fs.append(-1);continue
            m.eval();d=pc(m,Xv,yv);acc=ev(m,Xv,yv);mn=min(d[c] for c in range(10))
            fs.append(0.4*acc+0.4*mn+0.1*np.mean([d[c] for c in range(10)])+0.1*0.5)
        gi=np.argmax(fs)
        if fs[gi]>bf: bf=fs[gi]; bc={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in pop[gi].items()}
        if gen%10==0:
            m_=build_fn(bc)
            if m_:
                m_.eval();d=pc(m_,Xv,yv)
                log(f"  [{label}] Gen {gen}: fit={fs[gi]:.4f} acc={ev(m_,Xv,yv):.3f} min={min(d[c] for c in range(10)):.3f}")
        new=[{k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in bc.items()}]
        while len(new)<pop_sz:
            ti=random.sample(range(len(pop)),3);p1=pop[ti[np.argmax([fs[i] for i in ti])]]
            ch={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in p1.items()}
            ch['route']+=np.random.randn(10)*0.3
            if 'bridge' in ch: ch['bridge']=max(0.01,ch['bridge']+random.gauss(0,0.05))
            pf=max(0.02,0.06-gen*0.001)
            for k in ['mA0','mB0','mA1','mB1']:
                f=np.random.random(len(ch[k]))<pf;ch[k][f]=~ch[k][f]
                if ch[k].sum()==0: ch[k][0]=True
            new.append(ch)
        pop=new
    return build_fn(bc), bc

# Standard ENT
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
m1,bc1 = run_ea(build_standard, label="Standard")
m1.eval(); r1=pc(m1,X_te,y_te); a1=ev(m1,X_te,y_te)

# Cross-connections
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
m2,bc2 = run_ea(build_cross, label="Cross", extra_genes={'bridge':lambda:0.1})
m2.eval(); r2=pc(m2,X_te,y_te); a2=ev(m2,X_te,y_te)

# CKA merge
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
m3,bc3 = run_ea(build_cka_merge, label="CKA")
m3.eval(); r3=pc(m3,X_te,y_te); a3=ev(m3,X_te,y_te)

# ═══════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════
log(f"\n{'='*70}")
log("RESULTS: Breaking Block-Diagonal Isolation")
log("="*70)

methods = [("Standard ENT", a1, r1, m1),
           ("ENT+Cross", a2, r2, m2),
           ("CKA Merge", a3, r3, m3)]

log(f"\n  {'Method':<18} {'Acc':>6} {'Bal':>6} {'Min':>6} {'OK':>5} {'MaxDrop':>8} {'≤5%':>5} {'≤10%':>5} {'Params':>8}")
log(f"  {'-'*18} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*8} {'-'*5} {'-'*5} {'-'*8}")

for name, acc, pcM, model in methods:
    aM=np.mean([pcM[c] for c in clA]); bM=np.mean([pcM[c] for c in clB])
    bal=min(aM,bM)/(max(aM,bM)+1e-10); mn=min(pcM[c] for c in range(10))
    ok=sum(1 for c in range(10) if pcM[c]>0.3)
    drops=[max(0,max(pcA[c],pcB[c])-pcM[c]) for c in range(10)]
    w5=sum(1 for d in drops if d<=0.05); w10=sum(1 for d in drops if d<=0.10)
    params=sum(p.numel() for p in model.parameters())
    log(f"  {name:<18} {acc:>6.3f} {bal:>6.3f} {mn:>6.3f} {ok:>3}/10 {max(drops):>8.3f} {w5:>3}/10 {w10:>3}/10 {params:>8,}")

log(f"\n  Per-class drop vs best parent:")
log(f"  {'C':>2} {'Parent':>7} {'Standard':>9} {'Cross':>9} {'CKA':>9}")
for c in range(10):
    bp=max(pcA[c],pcB[c])
    d1=max(0,bp-r1[c]); d2=max(0,bp-r2[c]); d3=max(0,bp-r3[c])
    best=min(d1,d2,d3)
    log(f"  {c:>2} {bp:>7.3f} {d1:>9.3f}{' ✓' if d1==best else '  '} {d2:>9.3f}{' ✓' if d2==best else '  '} {d3:>9.3f}{' ✓' if d3==best else '  '}")

R.close(); print("Done!",flush=True)
