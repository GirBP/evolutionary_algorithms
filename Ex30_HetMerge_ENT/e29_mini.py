import numpy as np, torch, torch.nn as nn, random, sys
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets, transforms
print("Loading...", flush=True)
tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
tr = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=tf)
te = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=tf)
X_tr = torch.stack([tr[i][0] for i in range(10000)])
y_tr = torch.tensor([tr[i][1] for i in range(10000)])
X_te = torch.stack([te[i][0] for i in range(1000)])
y_te = torch.tensor([te[i][1] for i in range(1000)])
Xv, yv = X_tr[8000:10000], y_tr[8000:10000]
Xc = X_tr[:1000]
print("Data loaded", flush=True)

class MLP(nn.Module):
    def __init__(s, a):
        super().__init__(); l = []
        for i in range(len(a)-1):
            l.append(nn.Linear(a[i], a[i+1]))
            if i < len(a)-2: l.append(nn.ReLU())
        s.net = nn.Sequential(*l)
    def forward(s, x): return s.net(x)
    def features(s, x):
        for m in list(s.net)[:-1]: x = m(x)
        return x

class CNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2))
        s.fc = nn.Sequential(nn.Linear(32*7*7, 64), nn.ReLU(), nn.Linear(64, 10))
    def forward(s, x):
        if x.dim()==2: x = x.view(-1,1,28,28)
        return s.fc(s.conv(x).view(x.size(0),-1))
    def features(s, x):
        if x.dim()==2: x = x.view(-1,1,28,28)
        h = s.conv(x).view(x.size(0),-1)
        for m in list(s.fc)[:-1]: h = m(h)
        return h

def ev(m,X,y):
    m.eval()
    with torch.no_grad(): return (m(X).argmax(1)==y).float().mean().item()
def pc(m,X,y):
    m.eval()
    with torch.no_grad(): p = m(X).argmax(1)
    return {c:(p[y==c]==c).float().mean().item() if (y==c).sum()>0 else 0 for c in range(10)}
def trn(m, X, y, cls):
    mask = sum(y==c for c in cls).bool()
    Xs, ys = X[mask][:3000], y[mask][:3000]
    opt = torch.optim.Adam(m.parameters(), lr=0.003); m.train()
    for _ in range(10):
        l = nn.CrossEntropyLoss()(m(Xs), ys); opt.zero_grad(); l.backward(); opt.step()
    return m

clA, clB = list(range(5)), list(range(5,10))
print("Training...", flush=True)
mlpA = trn(MLP([784,128,64,10]), X_tr, y_tr, clA)
mlpB = trn(MLP([784,128,64,10]), X_tr, y_tr, clB)
cnnA = trn(CNN(), X_tr, y_tr, clA)
cnnB = trn(CNN(), X_tr, y_tr, clB)
print(f"MLP-A:{ev(mlpA,X_te,y_te):.3f} MLP-B:{ev(mlpB,X_te,y_te):.3f} CNN-A:{ev(cnnA,X_te,y_te):.3f} CNN-B:{ev(cnnB,X_te,y_te):.3f}", flush=True)

class DualModel(nn.Module):
    def __init__(s, mA, mB, W, b, iA, iB):
        super().__init__(); s.mA=mA; s.mB=mB
        s.W=torch.tensor(W); s.b=torch.tensor(b); s.iA=iA; s.iB=iB
    def forward(s, x):
        with torch.no_grad():
            fA = s.mA.features(x)[:, s.iA]
            fB = s.mB.features(x)[:, s.iB]
        return torch.cat([fA,fB],1) @ s.W.T + s.b

F = open('results_e29.txt', 'w')
for nm, mA, mB in [('MLP+MLP',mlpA,mlpB),('CNN+CNN',cnnA,cnnB),('MLP+CNN',mlpA,cnnB)]:
    print(f"\n{nm}:", flush=True)
    mA.eval(); mB.eval()
    with torch.no_grad():
        dA = mA.features(Xc).shape[1]; dB = mB.features(Xc).shape[1]
        sA = mA(Xc).numpy().std(); sB = mB(Xc).numpy().std()
    t = (sA+sB)/2; rA = t/(sA+1e-10); rB = t/(sB+1e-10)
    WoA = list(mA.parameters())[-2].detach().numpy()
    boA = list(mA.parameters())[-1].detach().numpy()
    WoB = list(mB.parameters())[-2].detach().numpy()
    boB = list(mB.parameters())[-1].detach().numpy()

    def fitness(route, mskA, mskB):
        iA = np.where(mskA)[0]; iB = np.where(mskB)[0]
        if len(iA)+len(iB)<2: return -1
        W = np.zeros((10,len(iA)+len(iB)),dtype=np.float32)
        b = np.zeros(10,dtype=np.float32)
        for c in range(10):
            a = 1/(1+np.exp(-route[c]))
            if len(iA)>0: W[c,:len(iA)] = a*rA*WoA[c][iA]
            if len(iB)>0: W[c,len(iA):] = (1-a)*rB*WoB[c][iB]
            b[c] = a*rA*boA[c]+(1-a)*rB*boB[c]
        m = DualModel(mA,mB,W,b,iA,iB)
        d = pc(m,Xv,yv); acc = ev(m,Xv,yv)
        mn = min(d[c] for c in range(10))
        return 0.4*acc+0.4*mn+0.1*np.mean([d[c] for c in range(10)])

    # EA: pop=12, gen=20
    pop = [(np.array([2.]*5+[-2.]*5), np.ones(dA,bool), np.ones(dB,bool))]
    for _ in range(11):
        r = np.random.randn(10)*1.5
        ma = np.random.random(dA)>0.3; mb = np.random.random(dB)>0.3
        ma[0]=True; mb[0]=True
        pop.append((r,ma,mb))
    bf=-1; bc=None
    for gen in range(20):
        fs = [fitness(r,ma,mb) for r,ma,mb in pop]
        gi = np.argmax(fs)
        if fs[gi]>bf: bf=fs[gi]; bc=(pop[gi][0].copy(),pop[gi][1].copy(),pop[gi][2].copy())
        if gen%10==0: print(f"  Gen {gen}: fit={fs[gi]:.4f}", flush=True)
        new = [(bc[0].copy(),bc[1].copy(),bc[2].copy())]
        while len(new)<12:
            i = random.randint(0,len(pop)-1); p = pop[i]
            r = p[0]+np.random.randn(10)*0.3; ma=p[1].copy(); mb=p[2].copy()
            f = np.random.random(dA)<0.05; ma[f]=~ma[f]; ma[0]=True
            f = np.random.random(dB)<0.05; mb[f]=~mb[f]; mb[0]=True
            new.append((r,ma,mb))
        pop = new

    # Build best
    iA=np.where(bc[1])[0]; iB=np.where(bc[2])[0]
    W=np.zeros((10,len(iA)+len(iB)),dtype=np.float32); b=np.zeros(10,dtype=np.float32)
    for c in range(10):
        a=1/(1+np.exp(-bc[0][c]))
        if len(iA)>0: W[c,:len(iA)]=a*rA*WoA[c][iA]
        if len(iB)>0: W[c,len(iA):]=(1-a)*rB*WoB[c][iB]
        b[c]=a*rA*boA[c]+(1-a)*rB*boB[c]
    final = DualModel(mA,mB,W,b,iA,iB)
    pcA,pcB = pc(mA,X_te,y_te),pc(mB,X_te,y_te)
    pcM = pc(final,X_te,y_te); acc = ev(final,X_te,y_te)
    aM = np.mean([pcM[c] for c in clA]); bM = np.mean([pcM[c] for c in clB])
    bal = min(aM,bM)/(max(aM,bM)+1e-10)
    nok = sum(1 for c in range(10) if pcM[c]>0.4)
    mn = min(pcM[c] for c in range(10))
    line = f"{nm}: acc={acc:.3f} A={aM:.3f} B={bM:.3f} bal={bal:.3f} ok={nok}/10 min={mn:.3f} feat={len(iA)}+{len(iB)}"
    print(line, flush=True); F.write(line+'\n')
    for c in range(10):
        bp = max(pcA[c],pcB[c]); mk='✅' if pcM[c]>=0.5*bp else '❌'
        cl = f"  {c}: par={bp:.3f} ent={pcM[c]:.3f} {mk}"
        F.write(cl+'\n')
F.close()
print("Done!", flush=True)
