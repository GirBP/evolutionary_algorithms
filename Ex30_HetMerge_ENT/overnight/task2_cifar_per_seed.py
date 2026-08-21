#!/usr/bin/env python3
"""Task 2: CIFAR-10 — run ONE seed at a time, accumulate in JSON.
Usage: python3 task2_cifar_per_seed.py <seed>
"""
import numpy as np, torch, torch.nn as nn, random, copy, time, json, sys
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import cma

SEED = int(sys.argv[1])
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
t0 = time.time()

from torchvision import datasets
raw_tr = datasets.CIFAR10('/tmp/cifar10', train=True, download=True)
raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mean = torch.tensor([0.4914,0.4822,0.4465]).view(3,1,1)
std = torch.tensor([0.247,0.243,0.261]).view(3,1,1)
N_TR, N_TE = 8000, 2000
X_tr = (torch.tensor(raw_tr.data[:N_TR]).permute(0,3,1,2).float()/255.0 - mean) / std
y_tr = torch.tensor(raw_tr.targets[:N_TR])
X_te = (torch.tensor(raw_te.data[:N_TE]).permute(0,3,1,2).float()/255.0 - mean) / std
y_te = torch.tensor(raw_te.targets[:N_TE])
Xv, yv = X_tr[6000:7000], y_tr[6000:7000]
Xc = X_tr[:1000]
clA, clB = list(range(5)), list(range(5,10))

class CNN(nn.Module):
    def __init__(s, ch1=32, ch2=64, fc=128):
        super().__init__()
        s.features = nn.Sequential(nn.Conv2d(3,ch1,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                   nn.Conv2d(ch1,ch2,3,padding=1), nn.ReLU(), nn.MaxPool2d(2))
        s.classifier = nn.Sequential(nn.Linear(ch2*8*8,fc), nn.ReLU(), nn.Linear(fc,10))
        s.ch1,s.ch2,s.fc_dim = ch1,ch2,fc
    def forward(s,x): return s.classifier(s.features(x).view(x.size(0),-1))

def ev(m,X,y):
    m.eval()
    with torch.no_grad(): return (m(X).argmax(1)==y).float().mean().item()
def pc(m,X,y):
    m.eval()
    with torch.no_grad(): p=m(X).argmax(1)
    return {c:(p[y==c]==c).float().mean().item() if (y==c).sum()>0 else 0 for c in range(10)}
def evaluate(m):
    pcM=pc(m,X_te,y_te);acc=ev(m,X_te,y_te)
    aM=np.mean([pcM[c] for c in clA]);bM=np.mean([pcM[c] for c in clB])
    bal=min(aM,bM)/(max(aM,bM)+1e-10);mn=min(pcM[c] for c in range(10))
    ok=sum(1 for c in range(10) if pcM[c]>0.2)
    return {'acc':round(acc,4),'bal':round(bal,4),'min':round(mn,4),'ok':ok,
            'pc':{c:round(pcM[c],3) for c in range(10)}}

def train_cnn(model,X,y,cls,ep=12):
    mask=sum(y==c for c in cls).bool();Xs,ys=X[mask][:2000],y[mask][:2000]
    opt=torch.optim.Adam(model.parameters(),lr=0.003);model.train()
    for _ in range(ep):
        i=torch.randperm(len(Xs))[:256];l=nn.CrossEntropyLoss()(model(Xs[i]),ys[i]);opt.zero_grad();l.backward();opt.step()
    model.eval();return model

def prune_cnn(model,t1,t2,tf):
    W1=model.features[0].weight.data;b1=model.features[0].bias.data
    i1=W1.abs().view(W1.shape[0],-1).sum(1).argsort(descending=True)[:t1].sort().values
    W2=model.features[3].weight.data;b2=model.features[3].bias.data
    i2=W2.abs().view(W2.shape[0],-1).sum(1).argsort(descending=True)[:t2].sort().values
    Wf=model.classifier[0].weight.data;bf=model.classifier[0].bias.data
    ifc=Wf.abs().sum(1).argsort(descending=True)[:tf].sort().values
    p=CNN(t1,t2,tf)
    with torch.no_grad():
        p.features[0].weight.copy_(W1[i1]);p.features[0].bias.copy_(b1[i1])
        p.features[3].weight.copy_(W2[i2][:,i1]);p.features[3].bias.copy_(b2[i2])
        fi=torch.cat([torch.arange(i*64,(i+1)*64) for i in i2])
        p.classifier[0].weight.copy_(Wf[ifc][:,fi]);p.classifier[0].bias.copy_(bf[ifc])
        p.classifier[2].weight.copy_(model.classifier[2].weight.data[:,ifc])
        p.classifier[2].bias.copy_(model.classifier[2].bias.data)
    return p

def merge_cnns(cA,cB,route,rA,rB):
    ch1,ch2,fc=cA.ch1,cA.ch2,cA.fc_dim;m=CNN(ch1*2,ch2*2,fc*2)
    with torch.no_grad():
        m.features[0].weight.zero_();m.features[0].bias.zero_()
        m.features[0].weight[:ch1]=cA.features[0].weight;m.features[0].weight[ch1:]=cB.features[0].weight
        m.features[0].bias[:ch1]=cA.features[0].bias;m.features[0].bias[ch1:]=cB.features[0].bias
        m.features[3].weight.zero_();m.features[3].bias.zero_()
        m.features[3].weight[:ch2,:ch1]=cA.features[3].weight;m.features[3].weight[ch2:,ch1:]=cB.features[3].weight
        m.features[3].bias[:ch2]=cA.features[3].bias;m.features[3].bias[ch2:]=cB.features[3].bias
        m.classifier[0].weight.zero_();m.classifier[0].bias.zero_()
        m.classifier[0].weight[:fc,:ch2*64]=cA.classifier[0].weight;m.classifier[0].bias[:fc]=cA.classifier[0].bias
        m.classifier[0].weight[fc:,ch2*64:]=cB.classifier[0].weight;m.classifier[0].bias[fc:]=cB.classifier[0].bias
        m.classifier[2].weight.zero_();m.classifier[2].bias.zero_()
        for c in range(10):
            a=1/(1+np.exp(-route[c]))
            m.classifier[2].weight[c,:fc]=a*rA*cA.classifier[2].weight[c]
            m.classifier[2].weight[c,fc:]=(1-a)*rB*cB.classifier[2].weight[c]
            m.classifier[2].bias[c]=a*rA*cA.classifier[2].bias[c]+(1-a)*rB*cB.classifier[2].bias[c]
    return m

# Train
base=CNN();opt=torch.optim.Adam(base.parameters(),lr=0.003);base.train()
for _ in range(6):
    i=torch.randperm(6000)[:256];l=nn.CrossEntropyLoss()(base(X_tr[i]),y_tr[i]);opt.zero_grad();l.backward();opt.step()
base.eval();base_sd=copy.deepcopy(base.state_dict())
ftA=copy.deepcopy(base);ftA=train_cnn(ftA,X_tr,y_tr,clA)
ftB=copy.deepcopy(base);ftB=train_cnn(ftB,X_tr,y_tr,clB)
sdA=ftA.state_dict();sdB=ftB.state_dict()
print(f"Parents: A={ev(ftA,X_te,y_te):.3f}, B={ev(ftB,X_te,y_te):.3f}")

res = {}
# Average
m=CNN();sd={k:0.5*sdA[k]+0.5*sdB[k] for k in sdA};m.load_state_dict(sd);m.eval()
res['Average']=evaluate(m)
# TA
best_ta=None;best_b=-1
for tau in [0.3,0.5]:
    m=CNN();sd={};
    for k in base_sd: sd[k]=base_sd[k]+tau*((sdA[k]-base_sd[k])+(sdB[k]-base_sd[k]))
    m.load_state_dict(sd);m.eval();r=evaluate(m)
    if r['bal']>best_b: best_b=r['bal'];best_ta=r
res['TA']=best_ta
# Sakana
def build_sak(x):
    a=1/(1+np.exp(-np.array(x)));m=CNN();sd={}
    for k in sdA.keys():
        gi=0 if 'features.0' in k else (1 if 'features.3' in k else (2 if 'classifier.0' in k else 3))
        sd[k]=a[gi]*sdA[k]+(1-a[gi])*sdB[k]
    m.load_state_dict(sd);m.eval();return m
es=cma.CMAEvolutionStrategy(np.zeros(4),1.5,{'maxiter':8,'popsize':6,'seed':SEED,'verbose':-1})
bs=-1;bx=None
while not es.stop():
    sols=es.ask();sc=[]
    for x in sols:
        m=build_sak(x);d=pc(m,Xv,yv);a_=ev(m,Xv,yv)
        sc.append(-(0.4*a_+0.4*min(d[c] for c in range(10))+0.1*np.mean([d[c] for c in range(10)])))
    es.tell(sols,sc)
    if -min(sc)>bs: bs=-min(sc);bx=sols[np.argmin(sc)]
res['Sakana']=evaluate(build_sak(bx))
# ENT
pA=prune_cnn(ftA,24,48,96);pB=prune_cnn(ftB,24,48,96)
pA.eval();pB.eval()
with torch.no_grad(): sA_=pA(Xc).numpy().std();sB_=pB(Xc).numpy().std()
t_=(sA_+sB_)/2;rA_=t_/(sA_+1e-10);rB_=t_/(sB_+1e-10)
torch.manual_seed(SEED)
pop=[np.array([2.]*5+[-2.]*5)]+[np.random.randn(10)*1.5 for _ in range(11)]
bf=-1;bc=None
for gen in range(20):
    fs=[]
    for r_ in pop:
        m=merge_cnns(pA,pB,r_,rA_,rB_);d=pc(m,Xv,yv);a_=ev(m,Xv,yv)
        mn_=min(d[c] for c in range(10))
        fs.append(0.5*mn_+0.3*np.exp(np.mean(np.log([max(d[c],1e-6) for c in range(10)])))+0.2*a_)
    gi=np.argmax(fs)
    if fs[gi]>bf: bf=fs[gi];bc=pop[gi].copy()
    new=[bc.copy()]+[pop[random.randint(0,len(pop)-1)]+np.random.randn(10)*0.3 for _ in range(11)]
    pop=new
res['ENT']=evaluate(merge_cnns(pA,pB,bc,rA_,rB_))
elapsed=time.time()-t0

# Accumulate
fpath = 'results/task2_cifar_accum.json'
try:
    with open(fpath) as f: accum = json.load(f)
except: accum = {}
accum[str(SEED)] = {m: {k:v for k,v in r.items() if k!='pc'} for m,r in res.items()}
accum[str(SEED)]['_per_class_ent'] = res['ENT']['pc']
accum[str(SEED)]['_time_s'] = round(elapsed,1)
with open(fpath,'w') as f: json.dump(accum, f, indent=2)

r=res['ENT']
print(f"seed={SEED}: ENT ok={r['ok']}/10 bal={r['bal']:.3f} min={r['min']:.3f} acc={r['acc']:.3f} | Avg ok={res['Average']['ok']} TA ok={res['TA']['ok']} Sak ok={res['Sakana']['ok']} | {elapsed:.0f}s")
print(f"metric_ent_ok: {r['ok']}")
print(f"metric_ent_bal: {r['bal']}")
print(f"metric_ent_min: {r['min']}")
print("Done!")
