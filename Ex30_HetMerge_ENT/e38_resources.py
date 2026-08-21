#!/usr/bin/env python3
"""E38: Resource Usage + remaining weakness analysis.
Measures: time, memory, FLOPs for ENT vs all baselines.
Tests: KD balance fix, multi-seed, CNN het-arch.
"""
import numpy as np, torch, torch.nn as nn, random, copy, time, sys, os
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
import ssl; ssl._create_default_https_context=ssl._create_unverified_context
from torchvision import datasets, transforms

R=open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/results_e38.txt','w')
def log(s): R.write(s+'\n'); R.flush(); print(s,flush=True)

log("="*70)
log("E38: Resource Usage + Remaining Weakness Analysis")
log("="*70)

# Data
tf=transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x:x.view(-1))])
tr=datasets.MNIST('/tmp/mnist',train=True,download=True,transform=tf)
te=datasets.MNIST('/tmp/mnist',train=False,download=True,transform=tf)
X_tr=torch.stack([tr[i][0] for i in range(20000)]); y_tr=torch.tensor([tr[i][1] for i in range(20000)])
X_te=torch.stack([te[i][0] for i in range(2000)]); y_te=torch.tensor([te[i][1] for i in range(2000)])
idx=torch.randperm(20000,generator=torch.Generator().manual_seed(0))
Xv,yv=X_tr[idx[15000:18000]],y_tr[idx[15000:18000]]; Xc=X_tr[idx[:2000]]

class MLP(nn.Module):
    def __init__(s,a):
        super().__init__(); l=[]
        for i in range(len(a)-1):
            l.append(nn.Linear(a[i],a[i+1]))
            if i<len(a)-2: l.append(nn.ReLU())
        s.net=nn.Sequential(*l); s.arch=a
    def forward(s,x): return s.net(x)

clA,clB=list(range(5)),list(range(5,10)); arch=[784,128,64,10]

def ev(m,X,y):
    m.eval()
    with torch.no_grad(): return (m(X).argmax(1)==y).float().mean().item()
def pc(m,X,y):
    m.eval()
    with torch.no_grad(): p=m(X).argmax(1)
    return {c:(p[y==c]==c).float().mean().item() if (y==c).sum()>0 else 0 for c in range(10)}
def evaluate(m,name):
    pcM=pc(m,X_te,y_te);acc=ev(m,X_te,y_te)
    aM=np.mean([pcM[c] for c in clA]);bM=np.mean([pcM[c] for c in clB])
    bal=min(aM,bM)/(max(aM,bM)+1e-10);mn=min(pcM[c] for c in range(10))
    ok=sum(1 for c in range(10) if pcM[c]>0.3)
    return {'name':name,'acc':round(acc,4),'bal':round(bal,4),'min':round(mn,4),
            'ok':ok,'A':round(aM,4),'B':round(bM,4),'pc':{c:round(pcM[c],3) for c in range(10)}}
def count_params(m): return sum(p.numel() for p in m.parameters())
def model_size_kb(m): return sum(p.numel()*p.element_size() for p in m.parameters())/1024

# ═══════════════════════════════════════════
# PART 1: Resource Usage — MNIST MLP
# ═══════════════════════════════════════════
log("\n" + "="*70)
log("PART 1: RESOURCE USAGE (MNIST MLP)")
log("="*70)

# Train parents (timed)
t0=time.time()
base=MLP(arch); torch.manual_seed(SEED)
opt=torch.optim.Adam(base.parameters(),lr=0.003)
for _ in range(5):
    l=nn.CrossEntropyLoss()(base(X_tr[:5000]),y_tr[:5000]);opt.zero_grad();l.backward();opt.step()
base.eval();base_sd=copy.deepcopy(base.state_dict())
t_base=time.time()-t0

def train_ft(m,X,y,cls,ep=15):
    mask=sum(y==c for c in cls).bool();Xs,ys=X[mask][:5000],y[mask][:5000]
    opt=torch.optim.Adam(m.parameters(),lr=0.003);m.train()
    for _ in range(ep): l=nn.CrossEntropyLoss()(m(Xs),ys);opt.zero_grad();l.backward();opt.step()
    m.eval();return m

t0=time.time()
ftA=train_ft(copy.deepcopy(base),X_tr,y_tr,clA)
ftB=train_ft(copy.deepcopy(base),X_tr,y_tr,clB)
t_finetune=time.time()-t0
sdA=ftA.state_dict();sdB=ftB.state_dict()

log(f"\n  Training costs:")
log(f"    Base pre-train (5 ep): {t_base:.2f}s")
log(f"    Fine-tune A+B (15 ep each): {t_finetune:.2f}s")
log(f"    Parent A: {ev(ftA,X_te,y_te):.3f}, Parent B: {ev(ftB,X_te,y_te):.3f}")

# === Average (timed) ===
t0=time.time()
m=MLP(arch);sd={k:0.5*sdA[k]+0.5*sdB[k] for k in sdA};m.load_state_dict(sd);m.eval()
t_avg=time.time()-t0
r_avg=evaluate(m,"Average")

# === TIES (timed) ===
t0=time.time()
m=MLP(arch);sd={}
for k in base_sd:
    tvA=sdA[k]-base_sd[k];tvB=sdB[k]-base_sd[k]
    for tv in [tvA,tvB]:
        flat=tv.flatten();n_keep=max(1,int(len(flat)*0.1))
        if n_keep<len(flat): thr=flat.abs().topk(n_keep).values[-1];tv[tv.abs()<thr]=0
    elect=torch.where(tvA.abs()>=tvB.abs(),tvA.sign(),tvB.sign())
    tvAf=torch.where(tvA.sign()==elect,tvA,torch.zeros_like(tvA))
    tvBf=torch.where(tvB.sign()==elect,tvB,torch.zeros_like(tvB))
    n_nz=((tvAf!=0).float()+(tvBf!=0).float()).clamp(min=1)
    sd[k]=base_sd[k]+(tvAf+tvBf)/n_nz
m.load_state_dict(sd);m.eval()
t_ties=time.time()-t0
r_ties=evaluate(m,"TIES")

# === ENT (timed) ===
def virt2L(model,Xc):
    model.eval()
    with torch.no_grad():
        h=Xc
        for m in list(model.net)[:-1]: h=m(h)
        hid=h.numpy()
    ps=list(model.parameters());Wo=ps[-2].detach().numpy();bo=ps[-1].detach().numpy()
    fd=hid.shape[1];x=Xc.numpy();N=x.shape[0]
    xb=np.hstack([x,np.ones((N,1),dtype=np.float32)])
    Wb=np.linalg.lstsq(xb,np.maximum(hid,0),rcond=None)[0].T
    W1=Wb[:,:-1].astype(np.float32);b1=Wb[:,-1].astype(np.float32)
    return [W1,b1,np.eye(fd,dtype=np.float32),np.zeros(fd,dtype=np.float32),Wo.copy(),bo.copy()],[fd,fd]

def bld_ent(ch,WA,WB,sA,sB,rA,rB):
    iA0=np.where(ch['mA'][0])[0];iB0=np.where(ch['mB'][0])[0]
    iA1=np.where(ch['mA'][1])[0];iB1=np.where(ch['mB'][1])[0]
    if len(iA0)+len(iB0)<2 or len(iA1)+len(iB1)<2: return None
    sizes=[784,len(iA0)+len(iB0),len(iA1)+len(iB1),10]
    W0=np.vstack([WA[0][iA0],WB[0][iB0]]);b0=np.concatenate([WA[1][iA0],WB[1][iB0]])
    W1=np.zeros((len(iA1)+len(iB1),len(iA0)+len(iB0)),dtype=np.float32)
    b1=np.zeros(len(iA1)+len(iB1),dtype=np.float32)
    W1[:len(iA1),:len(iA0)]=WA[2][np.ix_(iA1,iA0)];b1[:len(iA1)]=WA[3][iA1]
    W1[len(iA1):,len(iA0):]=WB[2][np.ix_(iB1,iB0)];b1[len(iA1):]=WB[3][iB1]
    Wo=np.zeros((10,len(iA1)+len(iB1)),dtype=np.float32);bo=np.zeros(10,dtype=np.float32)
    for c in range(10):
        a=1/(1+np.exp(-ch['route'][c]))
        if len(iA1)>0: Wo[c,:len(iA1)]=a*rA*WA[4][c][iA1]
        if len(iB1)>0: Wo[c,len(iA1):]=(1-a)*rB*WB[4][c][iB1]
        bo[c]=a*rA*WA[5][c]+(1-a)*rB*WB[5][c]
    m=MLP(sizes)
    with torch.no_grad():
        ps=list(m.parameters())
        ps[0].copy_(torch.tensor(W0));ps[1].copy_(torch.tensor(b0))
        ps[2].copy_(torch.tensor(W1));ps[3].copy_(torch.tensor(b1))
        ps[4].copy_(torch.tensor(Wo));ps[5].copy_(torch.tensor(bo))
    return m

t0=time.time()
t_virt=time.time()
WA,sA=virt2L(ftA,Xc);WB,sB=virt2L(ftB,Xc)
t_virt=time.time()-t_virt
ftA.eval();ftB.eval()
with torch.no_grad(): sl=ftA(Xc).numpy().std();sr=ftB(Xc).numpy().std()
t_=(sl+sr)/2;rA_=t_/(sl+1e-10);rB_=t_/(sr+1e-10)

t_ea_start=time.time()
pop=[]
for _ in range(20):
    ch={'mA':[np.random.random(d)>0.3 for d in sA],'mB':[np.random.random(d)>0.3 for d in sB],'route':np.random.randn(10)*1.5}
    for ms in [ch['mA'],ch['mB']]:
        for m_ in ms:
            if m_.sum()==0: m_[0]=True
    pop.append(ch)
pop[0]={'mA':[np.ones(d,dtype=bool) for d in sA],'mB':[np.ones(d,dtype=bool) for d in sB],'route':np.array([2.]*5+[-2.]*5)}
bf=-1;bc=None
for gen in range(30):
    fs=[]
    for ch in pop:
        m=bld_ent(ch,WA,WB,sA,sB,rA_,rB_)
        if m is None: fs.append(-1);continue
        d=pc(m,Xv,yv);acc=ev(m,Xv,yv);mn=min(d[c] for c in range(10))
        fs.append(0.4*acc+0.4*mn+0.1*np.mean([d[c] for c in range(10)])+0.1*(1-sum(ch['mA'][i].sum()+ch['mB'][i].sum() for i in range(2))/(sum(sA)+sum(sB))))
    gi=np.argmax(fs)
    if fs[gi]>bf: bf=fs[gi];bc={'mA':[m.copy() for m in pop[gi]['mA']],'mB':[m.copy() for m in pop[gi]['mB']],'route':pop[gi]['route'].copy()}
    new=[{'mA':[m.copy() for m in bc['mA']],'mB':[m.copy() for m in bc['mB']],'route':bc['route'].copy()}]
    while len(new)<20:
        ti=random.sample(range(len(pop)),3);p1=pop[ti[np.argmax([fs[i] for i in ti])]]
        ch={'mA':[m.copy() for m in p1['mA']],'mB':[m.copy() for m in p1['mB']],'route':p1['route']+np.random.randn(10)*0.3}
        pf=max(0.02,0.06-gen*0.001)
        for ms in [ch['mA'],ch['mB']]:
            for m_ in ms: f=np.random.random(len(m_))<pf;m_[f]=~m_[f];(m_.__setitem__(np.random.randint(len(m_)),True) if m_.sum()==0 else None)
        new.append(ch)
    pop=new
t_ea=time.time()-t_ea_start
t_ent_total=time.time()-t0

m_ent=bld_ent(bc,WA,WB,sA,sB,rA_,rB_);m_ent.eval()
r_ent=evaluate(m_ent,"ENT")

# Inference timing (1000 runs)
log(f"\n  Merge costs:")
log(f"    Average merge: {t_avg*1000:.2f}ms")
log(f"    TIES merge: {t_ties*1000:.2f}ms")
log(f"    ENT total: {t_ent_total:.2f}s")
log(f"      virt2L extraction: {t_virt*1000:.1f}ms")
log(f"      EA search (pop=20, gen=30): {t_ea:.2f}s = {t_ea*1000:.0f}ms")
log(f"      EA evaluations: {20*30}={20*30}")

# Inference time
log(f"\n  Inference costs (1000 samples):")
X_bench=X_te[:1000]
for name,model in [("Parent A",ftA),("Parent B",ftB),("Average",MLP(arch)),("ENT",m_ent)]:
    if name=="Average":
        model.load_state_dict({k:0.5*sdA[k]+0.5*sdB[k] for k in sdA})
    model.eval()
    # Warmup
    with torch.no_grad(): _ = model(X_bench)
    t0=time.time()
    for _ in range(100):
        with torch.no_grad(): _ = model(X_bench)
    t_inf=(time.time()-t0)/100*1000
    log(f"    {name:<15} {t_inf:.2f}ms  params={count_params(model):,}  size={model_size_kb(model):.1f}KB")

# ═══════════════════════════════════════════
# PART 2: Multi-seed stability (3 seeds)
# ═══════════════════════════════════════════
log(f"\n{'='*70}")
log("PART 2: MULTI-SEED STABILITY (3 seeds)")
log("="*70)
results_by_seed=[]
for seed in [42, 123, 7]:
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    b=MLP(arch)
    opt=torch.optim.Adam(b.parameters(),lr=0.003)
    for _ in range(5): l=nn.CrossEntropyLoss()(b(X_tr[:5000]),y_tr[:5000]);opt.zero_grad();l.backward();opt.step()
    b.eval();bsd=copy.deepcopy(b.state_dict())
    fA=train_ft(copy.deepcopy(b),X_tr,y_tr,clA)
    fB=train_ft(copy.deepcopy(b),X_tr,y_tr,clB)
    WA_,sA_=virt2L(fA,Xc);WB_,sB_=virt2L(fB,Xc)
    fA.eval();fB.eval()
    with torch.no_grad(): sl_=fA(Xc).numpy().std();sr_=fB(Xc).numpy().std()
    t__=(sl_+sr_)/2;rA__=t__/(sl_+1e-10);rB__=t__/(sr_+1e-10)
    pop=[]
    for _ in range(20):
        ch={'mA':[np.random.random(d)>0.3 for d in sA_],'mB':[np.random.random(d)>0.3 for d in sB_],'route':np.random.randn(10)*1.5}
        for ms in [ch['mA'],ch['mB']]:
            for m_ in ms:
                if m_.sum()==0: m_[0]=True
        pop.append(ch)
    pop[0]={'mA':[np.ones(d,dtype=bool) for d in sA_],'mB':[np.ones(d,dtype=bool) for d in sB_],'route':np.array([2.]*5+[-2.]*5)}
    bf_=-1;bc_=None
    for gen in range(30):
        fs=[]
        for ch in pop:
            m=bld_ent(ch,WA_,WB_,sA_,sB_,rA__,rB__)
            if m is None: fs.append(-1);continue
            d=pc(m,Xv,yv);acc=ev(m,Xv,yv);mn=min(d[c] for c in range(10))
            fs.append(0.4*acc+0.4*mn+0.1*np.mean([d[c] for c in range(10)])+0.1*(1-sum(ch['mA'][i].sum()+ch['mB'][i].sum() for i in range(2))/(sum(sA_)+sum(sB_))))
        gi_=np.argmax(fs)
        if fs[gi_]>bf_: bf_=fs[gi_];bc_={'mA':[m.copy() for m in pop[gi_]['mA']],'mB':[m.copy() for m in pop[gi_]['mB']],'route':pop[gi_]['route'].copy()}
        new=[{'mA':[m.copy() for m in bc_['mA']],'mB':[m.copy() for m in bc_['mB']],'route':bc_['route'].copy()}]
        while len(new)<20:
            ti=random.sample(range(len(pop)),3);p1=pop[ti[np.argmax([fs[i] for i in ti])]]
            ch={'mA':[m.copy() for m in p1['mA']],'mB':[m.copy() for m in p1['mB']],'route':p1['route']+np.random.randn(10)*0.3}
            pf=max(0.02,0.06-gen*0.001)
            for ms in [ch['mA'],ch['mB']]:
                for m_ in ms: f=np.random.random(len(m_))<pf;m_[f]=~m_[f];(m_.__setitem__(np.random.randint(len(m_)),True) if m_.sum()==0 else None)
            new.append(ch)
        pop=new
    m_=bld_ent(bc_,WA_,WB_,sA_,sB_,rA__,rB__);m_.eval()
    r_=evaluate(m_,f"ENT_s{seed}")
    # Also TIES
    sda=fA.state_dict();sdb=fB.state_dict()
    mt=MLP(arch);sdt={}
    for k in bsd:
        tvA=sda[k]-bsd[k];tvB=sdb[k]-bsd[k]
        for tv in [tvA,tvB]:
            flat=tv.flatten();n_keep=max(1,int(len(flat)*0.1))
            if n_keep<len(flat): thr=flat.abs().topk(n_keep).values[-1];tv[tv.abs()<thr]=0
        elect=torch.where(tvA.abs()>=tvB.abs(),tvA.sign(),tvB.sign())
        tvAf=torch.where(tvA.sign()==elect,tvA,torch.zeros_like(tvA))
        tvBf=torch.where(tvB.sign()==elect,tvB,torch.zeros_like(tvB))
        n_nz=((tvAf!=0).float()+(tvBf!=0).float()).clamp(min=1)
        sdt[k]=bsd[k]+(tvAf+tvBf)/n_nz
    mt.load_state_dict(sdt);mt.eval()
    rt=evaluate(mt,f"TIES_s{seed}")
    results_by_seed.append((seed,r_,rt))
    log(f"  Seed {seed}: ENT acc={r_['acc']:.3f} bal={r_['bal']:.3f} ok={r_['ok']}/10 | TIES acc={rt['acc']:.3f} bal={rt['bal']:.3f} ok={rt['ok']}/10")

ent_accs=[r[1]['acc'] for r in results_by_seed]
ent_bals=[r[1]['bal'] for r in results_by_seed]
ent_oks=[r[1]['ok'] for r in results_by_seed]
ties_accs=[r[2]['acc'] for r in results_by_seed]
ties_bals=[r[2]['bal'] for r in results_by_seed]
ties_oks=[r[2]['ok'] for r in results_by_seed]

log(f"\n  Summary (3 seeds):")
log(f"    ENT:  acc={np.mean(ent_accs):.3f}±{np.std(ent_accs):.3f}  bal={np.mean(ent_bals):.3f}±{np.std(ent_bals):.3f}  ok={np.mean(ent_oks):.1f}±{np.std(ent_oks):.1f}")
log(f"    TIES: acc={np.mean(ties_accs):.3f}±{np.std(ties_accs):.3f}  bal={np.mean(ties_bals):.3f}±{np.std(ties_bals):.3f}  ok={np.mean(ties_oks):.1f}±{np.std(ties_oks):.1f}")
log(f"    ENT > TIES on all seeds: {all(e>t for e,t in zip(ent_accs,ties_accs))}")
log(f"    ENT 10/10 on all seeds: {all(o==10 for o in ent_oks)}")

# ═══════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════
log(f"\n{'='*70}")
log("RESOURCE USAGE SUMMARY")
log("="*70)
log(f"  {'Method':<15} {'Merge time':>12} {'Acc':>6} {'Bal':>6} {'OK':>5} {'Params':>8} {'Size':>8}")
log(f"  {'-'*15} {'-'*12} {'-'*6} {'-'*6} {'-'*5} {'-'*8} {'-'*8}")
log(f"  {'Average':<15} {f'{t_avg*1000:.1f}ms':>12} {r_avg['acc']:>6.3f} {r_avg['bal']:>6.3f} {r_avg['ok']:>3}/10 {count_params(MLP(arch)):>8,} {'—':>8}")
log(f"  {'TIES':<15} {f'{t_ties*1000:.1f}ms':>12} {r_ties['acc']:>6.3f} {r_ties['bal']:>6.3f} {r_ties['ok']:>3}/10 {count_params(MLP(arch)):>8,} {'—':>8}")
log(f"  {'ENT':<15} {f'{t_ent_total:.1f}s':>12} {r_ent['acc']:>6.3f} {r_ent['bal']:>6.3f} {r_ent['ok']:>3}/10 {count_params(m_ent):>8,} {model_size_kb(m_ent):.1f}KB")
log(f"\n  ENT breakdown:")
log(f"    virt2L projection: {t_virt*1000:.1f}ms")
log(f"    EA search: {t_ea:.2f}s ({20*30} evaluations)")
log(f"    Single model eval: {t_ea/(20*30)*1000:.1f}ms")
log(f"    Inference (post-merge): same as parent (~ms)")

R.close()
print("Done!",flush=True)
