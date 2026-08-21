#!/usr/bin/env python3
"""E29: Dual-Backbone ENT — keep both feature extractors, merge only heads."""
import numpy as np, torch, torch.nn as nn, random, copy, sys
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets, transforms

tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
tr = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=tf)
te = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=tf)
N_TR=20000
X_tr = torch.stack([tr[i][0] for i in range(N_TR)]); y_tr = torch.tensor([tr[i][1] for i in range(N_TR)])
X_te = torch.stack([te[i][0] for i in range(1000)]); y_te = torch.tensor([te[i][1] for i in range(1000)])
idx = torch.randperm(N_TR, generator=torch.Generator().manual_seed(0))
Xv, yv = X_tr[idx[15000:17000]], y_tr[idx[15000:17000]]
Xc = X_tr[idx[:1500]]; Xd = X_tr[:5000]

class MLP(nn.Module):
    def __init__(s, a):
        super().__init__(); l = []
        for i in range(len(a)-1): l.append(nn.Linear(a[i], a[i+1])); (l.append(nn.ReLU()) if i < len(a)-2 else None)
        s.net = nn.Sequential(*l); s.arch = a
    def forward(s, x): return s.net(x)
    def features(s, x):
        x = x.view(x.size(0), -1)
        for m in list(s.net)[:-1]: x = m(x)
        return x

class CNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2))
        s.fc = nn.Sequential(nn.Linear(32*7*7, 64), nn.ReLU(), nn.Linear(64, 10))
    def forward(s, x):
        if x.dim() == 2: x = x.view(-1, 1, 28, 28)
        return s.fc(s.conv(x).view(x.size(0), -1))
    def features(s, x):
        if x.dim() == 2: x = x.view(-1, 1, 28, 28)
        h = s.conv(x).view(x.size(0), -1)
        for m in list(s.fc)[:-1]: h = m(h)
        return h

def ev(m, X, y):
    m.eval()
    with torch.no_grad(): return (m(X).argmax(1) == y).float().mean().item()

def pc(m, X, y):
    m.eval()
    with torch.no_grad(): p = m(X).argmax(1)
    return {c: (p[y==c]==c).float().mean().item() if (y==c).sum() > 0 else 0 for c in range(10)}

def trn(model, X, y, cls):
    mask = torch.zeros(len(y), dtype=torch.bool)
    for c in cls: mask |= (y == c)
    Xs, ys = X[mask][:5000], y[mask][:5000]
    opt = torch.optim.Adam(model.parameters(), lr=0.003); model.train()
    for _ in range(15):
        l = nn.CrossEntropyLoss()(model(Xs), ys); opt.zero_grad(); l.backward(); opt.step()
    return model

class DualModel(nn.Module):
    def __init__(s, mA, mB, Wo, bo, idxA, idxB):
        super().__init__()
        s.mA, s.mB = mA, mB
        s.register_buffer('Wo', torch.tensor(Wo, dtype=torch.float32))
        s.register_buffer('bo', torch.tensor(bo, dtype=torch.float32))
        s.idxA, s.idxB = idxA, idxB
    def forward(s, x):
        with torch.no_grad():
            fA = s.mA.features(x)[:, s.idxA]
            fB = s.mB.features(x)[:, s.idxB]
        f = torch.cat([fA, fB], dim=1)
        return f @ s.Wo.T + s.bo

clA, clB = list(range(5)), list(range(5, 10))
F = open('results_e29.txt', 'w')

mlpA = trn(MLP([784,128,64,10]), X_tr, y_tr, clA)
mlpB = trn(MLP([784,128,64,10]), X_tr, y_tr, clB)
cnnA = trn(CNN(), X_tr, y_tr, clA)
cnnB = trn(CNN(), X_tr, y_tr, clB)

hdr = f'Parents: MLP-A:{ev(mlpA,X_te,y_te):.3f} MLP-B:{ev(mlpB,X_te,y_te):.3f} CNN-A:{ev(cnnA,X_te,y_te):.3f} CNN-B:{ev(cnnB,X_te,y_te):.3f}'
F.write(hdr + '\n'); print(hdr, flush=True)

for nm, mA, mB in [('MLP+MLP', mlpA, mlpB), ('CNN+CNN', cnnA, cnnB),
                     ('MLP+CNN', mlpA, cnnB), ('CNN+MLP', cnnA, mlpB)]:
    pcA, pcB = pc(mA, X_te, y_te), pc(mB, X_te, y_te)
    mA.eval(); mB.eval()
    with torch.no_grad():
        dA = mA.features(Xc).shape[1]; dB = mB.features(Xc).shape[1]
        sA = mA(Xc).numpy().std(); sB = mB(Xc).numpy().std()
    t = (sA+sB)/2; rA = t/(sA+1e-10); rB = t/(sB+1e-10)
    pA = list(mA.parameters()); pB = list(mB.parameters())
    WoA, boA = pA[-2].detach().numpy(), pA[-1].detach().numpy()
    WoB, boB = pB[-2].detach().numpy(), pB[-1].detach().numpy()

    def fit(route, maskA, maskB):
        iA = np.where(maskA)[0]; iB = np.where(maskB)[0]
        if len(iA)+len(iB) < 2: return -1.0
        Wo = np.zeros((10, len(iA)+len(iB)), dtype=np.float32)
        bo = np.zeros(10, dtype=np.float32)
        for c in range(10):
            a = 1.0/(1.0+np.exp(-route[c]))
            if len(iA) > 0: Wo[c, :len(iA)] = a*rA*WoA[c][iA]
            if len(iB) > 0: Wo[c, len(iA):] = (1-a)*rB*WoB[c][iB]
            bo[c] = a*rA*boA[c] + (1-a)*rB*boB[c]
        m = DualModel(mA, mB, Wo, bo, iA, iB)
        d = pc(m, Xv, yv); acc = ev(m, Xv, yv)
        mn = min(d[c] for c in range(10))
        return 0.4*acc + 0.4*mn + 0.1*np.mean([d[c] for c in range(10)]) + 0.1*(1-(maskA.sum()+maskB.sum())/(dA+dB))

    # EA
    pop = [(np.array([2.0]*5+[-2.0]*5), np.ones(dA, dtype=bool), np.ones(dB, dtype=bool))]
    for _ in range(15):
        r = np.random.randn(10)*1.5
        ma = np.random.random(dA) > 0.3; mb = np.random.random(dB) > 0.3
        if ma.sum() == 0: ma[0] = True
        if mb.sum() == 0: mb[0] = True
        pop.append((r, ma, mb))

    bf = -1; bc = None
    for gen in range(25):
        fs = [fit(r, ma, mb) for r, ma, mb in pop]
        gi = np.argmax(fs)
        if fs[gi] > bf: bf = fs[gi]; bc = (pop[gi][0].copy(), pop[gi][1].copy(), pop[gi][2].copy())
        new = [(bc[0].copy(), bc[1].copy(), bc[2].copy())]
        while len(new) < 16:
            ti = random.sample(range(len(pop)), 3); p1 = pop[ti[np.argmax([fs[i] for i in ti])]]
            r = p1[0] + np.random.randn(10)*0.3; ma = p1[1].copy(); mb = p1[2].copy()
            pf = max(0.02, 0.06-gen*0.001)
            f = np.random.random(dA) < pf; ma[f] = ~ma[f]
            f = np.random.random(dB) < pf; mb[f] = ~mb[f]
            if not ma.any(): ma[np.random.randint(dA)] = True
            if not mb.any(): mb[np.random.randint(dB)] = True
            new.append((r, ma, mb))
        pop = new

    iA = np.where(bc[1])[0]; iB = np.where(bc[2])[0]
    Wo = np.zeros((10, len(iA)+len(iB)), dtype=np.float32); bo_f = np.zeros(10, dtype=np.float32)
    for c in range(10):
        a = 1.0/(1.0+np.exp(-bc[0][c]))
        if len(iA) > 0: Wo[c, :len(iA)] = a*rA*WoA[c][iA]
        if len(iB) > 0: Wo[c, len(iA):] = (1-a)*rB*WoB[c][iB]
        bo_f[c] = a*rA*boA[c] + (1-a)*rB*boB[c]

    # KD refinement
    with torch.no_grad():
        teacher = DualModel(mA, mB, Wo, bo_f, iA, iB); teacher.eval()
        fA_d = mA.features(Xd)[:, iA]; fB_d = mB.features(Xd)[:, iB]
        feat = torch.cat([fA_d, fB_d], dim=1)
        soft = torch.softmax(teacher(Xd)/3, dim=1); hard = teacher(Xd).argmax(1)
    sh = nn.Linear(len(iA)+len(iB), 10)
    opt = torch.optim.Adam(sh.parameters(), lr=0.002)
    for e in range(20):
        sl = torch.log_softmax(sh(feat)/3, dim=1)
        kd = nn.KLDivLoss(reduction='batchmean')(sl, soft)*9
        ce = nn.CrossEntropyLoss()(sh(feat), hard)
        loss = 0.7*kd + 0.3*ce; opt.zero_grad(); loss.backward(); opt.step()

    final = DualModel(mA, mB, sh.weight.detach().numpy(), sh.bias.detach().numpy(), iA, iB)
    pcM = pc(final, X_te, y_te); acc = ev(final, X_te, y_te)
    aM = np.mean([pcM[c] for c in clA]); bM = np.mean([pcM[c] for c in clB])
    bal = min(aM, bM)/(max(aM, bM)+1e-10)
    nok = sum(1 for c in range(10) if pcM[c] > 0.4)
    mn = min(pcM[c] for c in range(10))
    line = f'{nm}: acc={acc:.3f} A={aM:.3f} B={bM:.3f} bal={bal:.3f} ok={nok}/10 min={mn:.3f} feat=({len(iA)}+{len(iB)})/{dA+dB}'
    F.write(line + '\n'); print(line, flush=True)
    for c in range(10):
        bp = max(pcA[c], pcB[c]); mk = '✅' if pcM[c] >= 0.5*bp else '❌'
        cl = f'  {c}: par={bp:.3f} ent={pcM[c]:.3f} {mk}'
        F.write(cl + '\n')

F.close()
print('Done', flush=True)
