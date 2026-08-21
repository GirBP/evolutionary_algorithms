#!/usr/bin/env python3
"""E37: ENT v3 — CIFAR-10 with stronger parents + conv filter masks.
Optimized for M2: pre-load all data, mini-batch from tensor.
"""
import numpy as np, torch, torch.nn as nn, random, copy, time, gc
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
import ssl; ssl._create_default_https_context=ssl._create_unverified_context
from torchvision import datasets, transforms

R=open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/results_e37.txt','w')
def log(s): R.write(s+'\n'); R.flush(); print(s, flush=True)

log("="*70)
log("E37: ENT v3 — CIFAR-10 stronger parents + conv filter EA")
log("="*70)

tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
tr=datasets.CIFAR10('/tmp/cifar10',train=True,download=True,transform=tf)
te=datasets.CIFAR10('/tmp/cifar10',train=False,download=True,transform=tf)

# Pre-load into tensors
X_all=torch.stack([tr[i][0] for i in range(25000)]); y_all=torch.tensor([tr[i][1] for i in range(25000)])
X_te=torch.stack([te[i][0] for i in range(2000)]); y_te=torch.tensor([te[i][1] for i in range(2000)])
Xv=X_te[:1000]; yv=y_te[:1000]
clA,clB=list(range(5)),list(range(5,10))
t0=time.time()

class CNN(nn.Module):
    def __init__(s,ch1=32,ch2=64,fc=128):
        super().__init__()
        s.features=nn.Sequential(
            nn.Conv2d(3,ch1,3,padding=1),nn.BatchNorm2d(ch1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(ch1,ch2,3,padding=1),nn.BatchNorm2d(ch2),nn.ReLU(),nn.MaxPool2d(2))
        s.classifier=nn.Sequential(nn.Linear(ch2*8*8,fc),nn.ReLU(),nn.Dropout(0.3),nn.Linear(fc,10))
        s.ch1,s.ch2,s.fc_dim=ch1,ch2,fc
    def forward(s,x): return s.classifier(s.features(x).view(x.size(0),-1))

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
    ok=sum(1 for c in range(10) if pcM[c]>0.15)
    return {'name':name,'acc':round(acc,4),'bal':round(bal,4),'min':round(mn,4),
            'ok':ok,'A':round(aM,4),'B':round(bM,4),'pc':{c:round(pcM[c],3) for c in range(10)}}

# Train strong parents (mini-batch from tensor, with BN)
log("\n1. Training parents (30 epochs, BN, mini-batch)...")
def train_parent(cls, epochs=30):
    m=CNN(); mask=sum(y_all==c for c in cls).bool()
    Xs,ys=X_all[mask][:5000],y_all[mask][:5000]
    opt=torch.optim.Adam(m.parameters(),lr=0.001,weight_decay=1e-4)
    m.train()
    for ep in range(epochs):
        idx=torch.randperm(len(Xs)); total_l=0; n=0
        for i in range(0,len(Xs),64):
            batch=idx[i:i+64]; xb,yb=Xs[batch],ys[batch]
            out=m(xb); loss=nn.CrossEntropyLoss()(out,yb)
            opt.zero_grad();loss.backward();opt.step()
            total_l+=loss.item();n+=1
        if ep%10==0: log(f"  ep={ep} loss={total_l/n:.3f}")
    m.eval(); return m

modelA=train_parent(clA, epochs=30)
modelB=train_parent(clB, epochs=30)
log(f"\n  Parent A: {ev(modelA,X_te,y_te):.3f}")
log(f"  Parent B: {ev(modelB,X_te,y_te):.3f}")
pcA=pc(modelA,X_te,y_te); pcB=pc(modelB,X_te,y_te)
for c in range(10):
    bp=max(pcA[c],pcB[c]); log(f"  Class {c}: best_parent={bp:.3f}")

# Calibration
modelA.eval(); modelB.eval()
with torch.no_grad():
    sA_=modelA(Xv).numpy().std(); sB_=modelB(Xv).numpy().std()
t_=(sA_+sB_)/2; rA_=t_/(sA_+1e-10); rB_=t_/(sB_+1e-10)

# ═══════════════════════════════════════════
# ENT v3: per-filter masks
# ═══════════════════════════════════════════
log("\n2. ENT v3 — per-filter EA...")
ch1,ch2,fc=32,64,128

def merge_v3(mA, mB, masks, route, rA, rB):
    c1A=np.where(masks['c1A'])[0]; c1B=np.where(masks['c1B'])[0]
    c2A=np.where(masks['c2A'])[0]; c2B=np.where(masks['c2B'])[0]
    fA=np.where(masks['fA'])[0]; fB=np.where(masks['fB'])[0]
    n1=len(c1A)+len(c1B); n2=len(c2A)+len(c2B); nf=len(fA)+len(fB)
    if n1<2 or n2<2 or nf<2: return None

    m=CNN(n1,n2,nf)
    with torch.no_grad():
        # Conv1
        m.features[0].weight.copy_(torch.cat([mA.features[0].weight[c1A],mB.features[0].weight[c1B]]))
        m.features[0].bias.copy_(torch.cat([mA.features[0].bias[c1A],mB.features[0].bias[c1B]]))
        m.features[1].weight.copy_(torch.cat([mA.features[1].weight[c1A],mB.features[1].weight[c1B]]))
        m.features[1].bias.copy_(torch.cat([mA.features[1].bias[c1A],mB.features[1].bias[c1B]]))
        m.features[1].running_mean.copy_(torch.cat([mA.features[1].running_mean[c1A],mB.features[1].running_mean[c1B]]))
        m.features[1].running_var.copy_(torch.cat([mA.features[1].running_var[c1A],mB.features[1].running_var[c1B]]))
        # Conv2: block-diagonal
        m.features[4].weight.zero_()
        m.features[4].weight[:len(c2A),:len(c1A)]=mA.features[4].weight[c2A][:,c1A]
        m.features[4].weight[len(c2A):,len(c1A):]=mB.features[4].weight[c2B][:,c1B]
        m.features[4].bias.copy_(torch.cat([mA.features[4].bias[c2A],mB.features[4].bias[c2B]]))
        m.features[5].weight.copy_(torch.cat([mA.features[5].weight[c2A],mB.features[5].weight[c2B]]))
        m.features[5].bias.copy_(torch.cat([mA.features[5].bias[c2A],mB.features[5].bias[c2B]]))
        m.features[5].running_mean.copy_(torch.cat([mA.features[5].running_mean[c2A],mB.features[5].running_mean[c2B]]))
        m.features[5].running_var.copy_(torch.cat([mA.features[5].running_var[c2A],mB.features[5].running_var[c2B]]))
        # FC0: block-diagonal
        m.classifier[0].weight.zero_()
        idxA=torch.cat([torch.arange(i*64,(i+1)*64) for i in c2A])
        idxB=torch.cat([torch.arange(i*64,(i+1)*64) for i in c2B])
        m.classifier[0].weight[:len(fA),:len(c2A)*64]=mA.classifier[0].weight[fA][:,idxA]
        m.classifier[0].weight[len(fA):,len(c2A)*64:]=mB.classifier[0].weight[fB][:,idxB]
        m.classifier[0].bias.copy_(torch.cat([mA.classifier[0].bias[fA],mB.classifier[0].bias[fB]]))
        # Output: routing
        m.classifier[3].weight.zero_(); m.classifier[3].bias.zero_()
        for c in range(10):
            a=1/(1+np.exp(-route[c]))
            m.classifier[3].weight[c,:len(fA)]=a*rA*mA.classifier[3].weight[c,fA]
            m.classifier[3].weight[c,len(fA):]=(1-a)*rB*mB.classifier[3].weight[c,fB]
            m.classifier[3].bias[c]=a*rA*mA.classifier[3].bias[c]+(1-a)*rB*mB.classifier[3].bias[c]
    return m

def make_ch():
    return {'c1A':np.random.random(ch1)>0.2,'c1B':np.random.random(ch1)>0.2,
            'c2A':np.random.random(ch2)>0.2,'c2B':np.random.random(ch2)>0.2,
            'fA':np.random.random(fc)>0.3,'fB':np.random.random(fc)>0.3,
            'route':np.random.randn(10)*1.5}
def ensure(ch):
    for k in ch:
        if k=='route': continue
        if ch[k].sum()<2: ch[k][:2]=True

pop=[make_ch() for _ in range(15)]
pop[0]={k:(np.ones(v.shape,dtype=bool) if k!='route' else np.array([3.]*5+[-3.]*5)) for k,v in pop[0].items()}
for c in pop: ensure(c)

bf=-1; bc=None
for gen in range(25):
    fs=[]
    for ch in pop:
        m=merge_v3(modelA,modelB,ch,ch['route'],rA_,rB_)
        if m is None: fs.append(-1); continue
        m.eval(); d=pc(m,Xv,yv); acc=ev(m,Xv,yv)
        mn=min(d[c] for c in range(10))
        n_ok=sum(1 for c in range(10) if d[c]>0.15)/10
        total=sum(ch[k].sum() for k in ch if k!='route')
        fs.append(0.3*acc+0.3*mn+0.25*n_ok+0.15*(1-total/(ch1*2+ch2*2+fc*2)))
        del m
    gi=np.argmax(fs)
    if fs[gi]>bf: bf=fs[gi]; bc={k:v.copy() for k,v in pop[gi].items()}
    if gen%5==0:
        m_=merge_v3(modelA,modelB,bc,bc['route'],rA_,rB_)
        if m_:
            d=pc(m_,Xv,yv);log(f"  Gen {gen}: fit={fs[gi]:.4f} acc={ev(m_,Xv,yv):.3f} min={min(d[c] for c in range(10)):.3f} ok={sum(1 for c in range(10) if d[c]>0.15)}/10")
    new=[{k:v.copy() for k,v in bc.items()}]
    while len(new)<15:
        ti=random.sample(range(len(pop)),3);p1=pop[ti[np.argmax([fs[i] for i in ti])]]
        ch={k:v.copy() for k,v in p1.items()}; ch['route']+=np.random.randn(10)*0.3
        pf=max(0.02,0.05-gen*0.001)
        for k in ch:
            if k=='route': continue
            f=np.random.random(len(ch[k]))<pf; ch[k][f]=~ch[k][f]
        ensure(ch); new.append(ch)
    pop=new

m_ent=merge_v3(modelA,modelB,bc,bc['route'],rA_,rB_); m_ent.eval()
r_ent=evaluate(m_ent,"ENT-v3")
log(f"\n  ENT-v3: acc={r_ent['acc']:.3f} bal={r_ent['bal']:.3f} ok={r_ent['ok']}/10 min={r_ent['min']:.3f}")

# Post-merge KD
log("\n3. KD fine-tuning...")
m_kd=copy.deepcopy(m_ent); torch.set_grad_enabled(True); m_kd.train()
Xkd=X_all[:3000]; ykd=y_all[:3000]
modelA.eval();modelB.eval()
with torch.no_grad():
    logA=modelA(Xkd);logB=modelB(Xkd)
    soft=torch.zeros_like(logA)
    for i in range(len(ykd)):
        if ykd[i].item() in clA: soft[i]=logA[i]
        else: soft[i]=logB[i]
    soft=torch.softmax(soft/3,dim=1); hard=soft.argmax(1)
opt=torch.optim.Adam(m_kd.parameters(),lr=0.0005)
for ep in range(10):
    idx=torch.randperm(3000)[:512]
    out=m_kd(Xkd[idx])
    kd=nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(out/3,dim=1),soft[idx])*9
    ce=nn.CrossEntropyLoss()(out,hard[idx])
    loss=0.3*kd+0.7*ce; opt.zero_grad();loss.backward();opt.step()
    if ep%3==0: log(f"  KD ep {ep}: loss={loss.item():.3f}")
torch.set_grad_enabled(False); m_kd.eval()
r_kd=evaluate(m_kd,"ENT-v3+KD")
log(f"  ENT-v3+KD: acc={r_kd['acc']:.3f} bal={r_kd['bal']:.3f} ok={r_kd['ok']}/10 min={r_kd['min']:.3f}")

# Baselines
log("\n4. Baselines...")
sdA=modelA.state_dict();sdB=modelB.state_dict()
m_avg=CNN();m_avg.load_state_dict({k:0.5*sdA[k]+0.5*sdB[k] for k in sdA});m_avg.eval()
r_avg=evaluate(m_avg,"Average")
log(f"  Avg: acc={r_avg['acc']:.3f} bal={r_avg['bal']:.3f} ok={r_avg['ok']}/10")

log(f"\n{'='*70}")
log("E37 RESULTS")
log(f"{'='*70}")
all_r=[r_avg,r_ent,r_kd]
log(f"  {'Method':<22} {'Acc':>6} {'Bal':>6} {'Min':>6} {'OK':>5}")
for r in sorted(all_r,key=lambda x:(-x['ok'],-x['bal'])):
    mk=" ✅" if 'ENT' in r['name'] else ""
    log(f"  {r['name']:<22} {r['acc']:>6.3f} {r['bal']:>6.3f} {r['min']:>6.3f} {r['ok']:>3}/10{mk}")
log(f"\n  Per-class:")
for c in range(10):
    bp=max(pcA[c],pcB[c])
    log(f"  {c}: parent={bp:.3f} " + " ".join(f"{r['name'][:8]}={r['pc'][c]:.3f}" for r in all_r))
log(f"\n  Time: {time.time()-t0:.0f}s")
R.close(); print("Done!",flush=True)
