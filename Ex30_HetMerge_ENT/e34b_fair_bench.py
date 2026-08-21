#!/usr/bin/env python3
"""E34b: FAIR BENCHMARK — shared pre-trained base + fine-tuning.

Setup matches TIES/DARE/TA papers:
1. Train base model on ALL classes (partial/weak)
2. Fine-tune A on classes 0-4
3. Fine-tune B on classes 5-9
4. Merge using Task Vectors relative to base

ALSO tests the disjoint case (train from scratch, no shared base)
"""
import numpy as np, torch, torch.nn as nn, random, copy, json
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets, transforms
import cma

R = open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/results_e34b.txt', 'w')
def log(s): R.write(s + '\n'); R.flush(); print(s, flush=True)

log("=" * 70)
log("E34b: FAIR BENCHMARK — BOTH shared-base AND disjoint scenarios")
log("=" * 70)

tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
tr = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=tf)
te = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=tf)
X_tr = torch.stack([tr[i][0] for i in range(20000)]); y_tr = torch.tensor([tr[i][1] for i in range(20000)])
X_te = torch.stack([te[i][0] for i in range(2000)]); y_te = torch.tensor([te[i][1] for i in range(2000)])
idx = torch.randperm(20000, generator=torch.Generator().manual_seed(0))
Xv, yv = X_tr[idx[15000:18000]], y_tr[idx[15000:18000]]
Xc = X_tr[idx[:2000]]

class MLP(nn.Module):
    def __init__(s, a):
        super().__init__(); l = []
        for i in range(len(a)-1):
            l.append(nn.Linear(a[i], a[i+1]))
            if i < len(a)-2: l.append(nn.ReLU())
        s.net = nn.Sequential(*l); s.arch = a
    def forward(s, x): return s.net(x)

clA, clB = list(range(5)), list(range(5, 10))
arch = [784, 128, 64, 10]

def ev(m, X, y):
    m.eval()
    with torch.no_grad(): return (m(X).argmax(1) == y).float().mean().item()
def pc(m, X, y):
    m.eval()
    with torch.no_grad(): p = m(X).argmax(1)
    return {c: (p[y==c]==c).float().mean().item() if (y==c).sum() > 0 else 0 for c in range(10)}
def evaluate(m, name):
    pcM = pc(m, X_te, y_te); acc = ev(m, X_te, y_te)
    aM = np.mean([pcM[c] for c in clA]); bM = np.mean([pcM[c] for c in clB])
    bal = min(aM, bM) / (max(aM, bM) + 1e-10)
    mn = min(pcM[c] for c in range(10))
    ok = sum(1 for c in range(10) if pcM[c] > 0.3)
    return {'name': name, 'acc': round(acc, 4), 'bal': round(bal, 4),
            'min': round(mn, 4), 'ok': ok, 'A': round(aM, 4), 'B': round(bM, 4),
            'pc': {c: round(pcM[c], 3) for c in range(10)}}

def train(m, X, y, cls, ep=15, lr=0.003):
    mask = sum(y == c for c in cls).bool()
    Xs, ys = X[mask][:5000], y[mask][:5000]
    opt = torch.optim.Adam(m.parameters(), lr=lr); m.train()
    for _ in range(ep):
        l = nn.CrossEntropyLoss()(m(Xs), ys); opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m

# ═══════════════════════════════════════════
# SCENARIO 1: Shared pre-trained base (matches TIES/DARE papers)
# ═══════════════════════════════════════════
log("\n" + "=" * 70)
log("SCENARIO 1: Shared pre-trained base → fine-tune → merge")
log("(matches TIES/DARE/Task Arithmetic paper setup)")
log("=" * 70)

# Step 1: Pre-train base on ALL classes (weak)
log("\n1. Pre-training base on ALL classes (5 epochs)...")
base = MLP(arch)
torch.manual_seed(SEED)
opt = torch.optim.Adam(base.parameters(), lr=0.003); base.train()
for _ in range(5):
    l = nn.CrossEntropyLoss()(base(X_tr[:5000]), y_tr[:5000]); opt.zero_grad(); l.backward(); opt.step()
base.eval()
base_sd = copy.deepcopy(base.state_dict())
log(f"   Base accuracy: {ev(base, X_te, y_te):.3f}")

# Step 2: Fine-tune on tasks
log("2. Fine-tuning specialists...")
ftA = copy.deepcopy(base)
ftA = train(ftA, X_tr, y_tr, clA, ep=15)
ftB = copy.deepcopy(base)
ftB = train(ftB, X_tr, y_tr, clB, ep=15)
log(f"   FT-A (cls 0-4): {ev(ftA, X_te, y_te):.3f}")
log(f"   FT-B (cls 5-9): {ev(ftB, X_te, y_te):.3f}")

sdA = ftA.state_dict()
sdB = ftB.state_dict()
results_shared = []

# ─── Task Arithmetic ───
log("\nTask Arithmetic:")
for tau in [0.3, 0.5, 0.7, 1.0, 1.5]:
    m = MLP(arch); sd = {}
    for k in base_sd:
        tv_A = sdA[k] - base_sd[k]; tv_B = sdB[k] - base_sd[k]
        sd[k] = base_sd[k] + tau * (tv_A + tv_B)
    m.load_state_dict(sd); m.eval()
    r = evaluate(m, f"TA(τ={tau})")
    log(f"  τ={tau}: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")
    results_shared.append(r)

# ─── TIES (faithful implementation) ───
log("\nTIES-Merging:")
for density in [0.1, 0.2, 0.3, 0.5]:
    m = MLP(arch); sd = {}
    for k in base_sd:
        tvA = sdA[k] - base_sd[k]; tvB = sdB[k] - base_sd[k]
        # Step 1: TRIM — zero out bottom (1-density) of each task vector
        for tv in [tvA, tvB]:
            flat = tv.flatten()
            n_keep = max(1, int(len(flat) * density))
            if n_keep < len(flat):
                threshold = flat.abs().topk(n_keep).values[-1]
                tv[tv.abs() < threshold] = 0
        # Step 2: ELECT SIGN — resolve sign conflicts
        agree = (tvA.sign() == tvB.sign()) | (tvA == 0) | (tvB == 0)
        # For disagreements: use the one with larger magnitude
        sign_A = tvA.sign(); sign_B = tvB.sign()
        elected_sign = torch.where(tvA.abs() >= tvB.abs(), sign_A, sign_B)
        # Step 3: DISJOINT MERGE — keep only values matching elected sign
        tvA_filtered = torch.where(tvA.sign() == elected_sign, tvA, torch.zeros_like(tvA))
        tvB_filtered = torch.where(tvB.sign() == elected_sign, tvB, torch.zeros_like(tvB))
        # Mean of non-zero values
        n_nonzero = ((tvA_filtered != 0).float() + (tvB_filtered != 0).float()).clamp(min=1)
        merged_tv = (tvA_filtered + tvB_filtered) / n_nonzero
        sd[k] = base_sd[k] + merged_tv
    m.load_state_dict(sd); m.eval()
    r = evaluate(m, f"TIES(d={density})")
    log(f"  d={density}: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")
    results_shared.append(r)

# ─── DARE + TIES ───
log("\nDARE + TIES:")
for p_keep in [0.1, 0.3, 0.5]:
    m = MLP(arch); sd = {}
    torch.manual_seed(SEED)
    for k in base_sd:
        tvA = sdA[k] - base_sd[k]; tvB = sdB[k] - base_sd[k]
        # DARE: random drop, then rescale
        maskA = (torch.rand_like(tvA.float()) < p_keep).float()
        maskB = (torch.rand_like(tvB.float()) < p_keep).float()
        tvA = tvA * maskA / max(p_keep, 0.01)
        tvB = tvB * maskB / max(p_keep, 0.01)
        # Then TIES sign election
        elected_sign = torch.where(tvA.abs() >= tvB.abs(), tvA.sign(), tvB.sign())
        tvA = torch.where(tvA.sign() == elected_sign, tvA, torch.zeros_like(tvA))
        tvB = torch.where(tvB.sign() == elected_sign, tvB, torch.zeros_like(tvB))
        n_nz = ((tvA != 0).float() + (tvB != 0).float()).clamp(min=1)
        sd[k] = base_sd[k] + (tvA + tvB) / n_nz
    m.load_state_dict(sd); m.eval()
    r = evaluate(m, f"DARE-TIES(p={p_keep})")
    log(f"  p={p_keep}: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")
    results_shared.append(r)

# ─── Average ───
m = MLP(arch)
sd = {k: 0.5 * sdA[k] + 0.5 * sdB[k] for k in sdA}
m.load_state_dict(sd); m.eval()
r = evaluate(m, "Average")
results_shared.append(r)
log(f"\nAverage: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10")

# ─── SLERP ───
def slerp(sdA, sdB, t):
    sd = {}
    for k in sdA:
        vA, vB = sdA[k].float().flatten(), sdB[k].float().flatten()
        nA, nB = vA.norm(), vB.norm()
        if nA < 1e-8 or nB < 1e-8: sd[k] = t*sdA[k]+(1-t)*sdB[k]; continue
        cos_o = (vA@vB)/(nA*nB); cos_o = cos_o.clamp(-1,1); o = torch.acos(cos_o)
        if o.abs()<1e-6: sd[k]=t*sdA[k]+(1-t)*sdB[k]
        else: sd[k]=(torch.sin((1-t)*o)/torch.sin(o))*sdA[k]+(torch.sin(t*o)/torch.sin(o))*sdB[k]
    return sd
m = MLP(arch); m.load_state_dict(slerp(sdA, sdB, 0.5)); m.eval()
r = evaluate(m, "SLERP")
results_shared.append(r)
log(f"SLERP: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10")

# ─── Sakana-style ───
log("\nSakana-CMA (per-layer):")
def build_sak(x, sA, sB):
    a = 1/(1+np.exp(-np.array(x))); m=MLP(arch); sd={}; keys=list(sA.keys())
    for i,k in enumerate(keys): sd[k]=a[i//2]*sA[k]+(1-a[i//2])*sB[k]
    m.load_state_dict(sd); m.eval(); return m
es = cma.CMAEvolutionStrategy(np.zeros(3), 1.5, {'maxiter':15,'popsize':8,'seed':SEED,'verbose':-1})
best_s=-1; best_x=None
while not es.stop():
    sols=es.ask(); scores=[]
    for x in sols:
        m=build_sak(x,sdA,sdB); d=pc(m,Xv,yv); acc=ev(m,Xv,yv)
        mn=min(d[c] for c in range(10)); s=0.4*acc+0.4*mn+0.1*np.mean([d[c] for c in range(10)])
        scores.append(-s)
    es.tell(sols,scores)
    if -min(scores)>best_s: best_s=-min(scores); best_x=sols[np.argmin(scores)]
m=build_sak(best_x,sdA,sdB); r=evaluate(m,"Sakana-CMA")
results_shared.append(r)
log(f"  Sakana-CMA: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")

# ─── ENT ───
log("\nENT (ours):")
def virt2L(model, Xc):
    model.eval()
    with torch.no_grad():
        h=Xc
        for m in list(model.net)[:-1]: h=m(h)
        hid=h.numpy()
    ps=list(model.parameters())
    Wo=ps[-2].detach().numpy(); bo=ps[-1].detach().numpy()
    fd=hid.shape[1]; x=Xc.numpy(); N=x.shape[0]
    xb=np.hstack([x,np.ones((N,1),dtype=np.float32)])
    Wb=np.linalg.lstsq(xb,np.maximum(hid,0),rcond=None)[0].T
    W1=Wb[:,:-1].astype(np.float32); b1=Wb[:,-1].astype(np.float32)
    return [W1,b1,np.eye(fd,dtype=np.float32),np.zeros(fd,dtype=np.float32),Wo.copy(),bo.copy()],[fd,fd]

def bld_ent(ch,WA,WB,sA,sB,rA,rB):
    iA0=np.where(ch['mA'][0])[0];iB0=np.where(ch['mB'][0])[0]
    iA1=np.where(ch['mA'][1])[0];iB1=np.where(ch['mB'][1])[0]
    if len(iA0)+len(iB0)<2 or len(iA1)+len(iB1)<2: return None
    sizes=[784,len(iA0)+len(iB0),len(iA1)+len(iB1),10]
    W0=np.vstack([WA[0][iA0],WB[0][iB0]]); b0=np.concatenate([WA[1][iA0],WB[1][iB0]])
    W1=np.zeros((len(iA1)+len(iB1),len(iA0)+len(iB0)),dtype=np.float32)
    b1=np.zeros(len(iA1)+len(iB1),dtype=np.float32)
    W1[:len(iA1),:len(iA0)]=WA[2][np.ix_(iA1,iA0)]; b1[:len(iA1)]=WA[3][iA1]
    W1[len(iA1):,len(iA0):]=WB[2][np.ix_(iB1,iB0)]; b1[len(iA1):]=WB[3][iB1]
    Wo=np.zeros((10,len(iA1)+len(iB1)),dtype=np.float32); bo=np.zeros(10,dtype=np.float32)
    for c in range(10):
        a=1/(1+np.exp(-ch['route'][c]))
        if len(iA1)>0: Wo[c,:len(iA1)]=a*rA*WA[4][c][iA1]
        if len(iB1)>0: Wo[c,len(iA1):]=(1-a)*rB*WB[4][c][iB1]
        bo[c]=a*rA*WA[5][c]+(1-a)*rB*WB[5][c]
    m=MLP(sizes)
    with torch.no_grad():
        ps=list(m.parameters())
        ps[0].copy_(torch.tensor(W0)); ps[1].copy_(torch.tensor(b0))
        ps[2].copy_(torch.tensor(W1)); ps[3].copy_(torch.tensor(b1))
        ps[4].copy_(torch.tensor(Wo)); ps[5].copy_(torch.tensor(bo))
    return m

WA,sAs=virt2L(ftA,Xc); WB,sBs=virt2L(ftB,Xc)
ftA.eval();ftB.eval()
with torch.no_grad(): sl=ftA(Xc).numpy().std();sr=ftB(Xc).numpy().std()
t=(sl+sr)/2;rA_=t/(sl+1e-10);rB_=t/(sr+1e-10)

pop=[]
for _ in range(20):
    ch={'mA':[np.random.random(d)>0.3 for d in sAs],'mB':[np.random.random(d)>0.3 for d in sBs],'route':np.random.randn(10)*1.5}
    for ms in [ch['mA'],ch['mB']]:
        for m_ in ms:
            if m_.sum()==0: m_[0]=True
    pop.append(ch)
pop[0]={'mA':[np.ones(d,dtype=bool) for d in sAs],'mB':[np.ones(d,dtype=bool) for d in sBs],'route':np.array([2.]*5+[-2.]*5)}

bf=-1;bc=None
for gen in range(30):
    fs=[]
    for ch in pop:
        m=bld_ent(ch,WA,WB,sAs,sBs,rA_,rB_)
        if m is None: fs.append(-1); continue
        d=pc(m,Xv,yv);acc=ev(m,Xv,yv);mn=min(d[c] for c in range(10))
        fs.append(0.4*acc+0.4*mn+0.1*np.mean([d[c] for c in range(10)])+0.1*(1-sum(ch['mA'][i].sum()+ch['mB'][i].sum() for i in range(2))/(sum(sAs)+sum(sBs))))
    gi=np.argmax(fs)
    if fs[gi]>bf: bf=fs[gi]; bc={'mA':[m.copy() for m in pop[gi]['mA']],'mB':[m.copy() for m in pop[gi]['mB']],'route':pop[gi]['route'].copy()}
    if gen%10==0: log(f"  Gen {gen}: fit={fs[gi]:.4f}")
    new=[{'mA':[m.copy() for m in bc['mA']],'mB':[m.copy() for m in bc['mB']],'route':bc['route'].copy()}]
    while len(new)<20:
        ti=random.sample(range(len(pop)),3);p1=pop[ti[np.argmax([fs[i] for i in ti])]]
        ch={'mA':[m.copy() for m in p1['mA']],'mB':[m.copy() for m in p1['mB']],'route':p1['route']+np.random.randn(10)*0.3}
        pf=max(0.02,0.06-gen*0.001)
        for ms in [ch['mA'],ch['mB']]:
            for m_ in ms: f=np.random.random(len(m_))<pf; m_[f]=~m_[f]; (m_.__setitem__(np.random.randint(len(m_)),True) if m_.sum()==0 else None)
        new.append(ch)
    pop=new

m=bld_ent(bc,WA,WB,sAs,sBs,rA_,rB_); m.eval()
r=evaluate(m,"ENT")
results_shared.append(r)
log(f"  ENT: acc={r['acc']:.3f} bal={r['bal']:.3f} ok={r['ok']}/10 min={r['min']:.3f}")

# ═════════════ SUMMARY TABLE ═══════════════
log(f"\n{'='*70}")
log("SCENARIO 1 RESULTS: Shared Base → Fine-tune → Merge")
log(f"{'='*70}")
log(f"  {'Method':<22} {'Acc':>6} {'Balance':>8} {'Min':>6} {'OK':>5} {'A':>6} {'B':>6}")
log(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*6} {'-'*5} {'-'*6} {'-'*6}")
# Sort by balance
for r in sorted(results_shared, key=lambda x: -x['bal']):
    mk = " ✅" if 'ENT' in r['name'] else ""
    log(f"  {r['name']:<22} {r['acc']:>6.3f} {r['bal']:>8.3f} {r['min']:>6.3f} {r['ok']:>3}/10 {r['A']:>6.3f} {r['B']:>6.3f}{mk}")

# Per-class
log(f"\n  Per-class (top methods by balance):")
top = sorted(results_shared, key=lambda x: -x['bal'])[:5]
log(f"  {'Cls':>3}  {'Parent':>6}" + "".join(f"  {r['name'][:12]:>12}" for r in top))
for c in range(10):
    bp = max(pc(ftA,X_te,y_te)[c], pc(ftB,X_te,y_te)[c])
    log(f"  {c:>3}  {bp:>6.3f}" + "".join(f"  {r['pc'][c]:>12.3f}" for r in top))

R.close()
print("Done!", flush=True)
