#!/usr/bin/env python3
"""E02: EA (CMA-ES) per-layer λ on toy TinyResNet.
Build on E01: alignment + BN reset + CMA-ES λ optimization.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, json, time, sys
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets
from scipy.optimize import linear_sum_assignment
import cma

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
t0 = time.time()

raw_tr = datasets.CIFAR10('/tmp/cifar10', train=True, download=True)
raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mn = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
sd = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_tr = (torch.tensor(raw_tr.data).permute(0,3,1,2).float()/255-mn)/sd
y_tr = torch.tensor(raw_tr.targets)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255-mn)/sd
y_te = torch.tensor(raw_te.targets)
clA, clB = [0,1], [2,3]; ALL = clA+clB
X_cal = X_tr[:5000]; y_cal = y_tr[:5000]

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch,ch,3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch,ch,3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))+x)

class TinyResNet(nn.Module):
    def __init__(self, nc=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.block1 = ResBlock(16)
        self.block2 = ResBlock(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, nc)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x); x = self.block2(x)
        return self.fc(self.pool(x).flatten(1))

def train_tiny(cls_list, seed_p, epochs=15):
    torch.manual_seed(seed_p); np.random.seed(seed_p); random.seed(seed_p)
    m = TinyResNet(len(cls_list))
    cmap = {c:i for i,c in enumerate(cls_list)}
    mask = sum(y_tr==c for c in cls_list).bool()
    Xs, ys = X_tr[mask], torch.tensor([cmap[y.item()] for y in y_tr[mask]])
    idx = torch.cat([torch.where(ys==i)[0][:1000] for i in range(len(cls_list))])
    Xs, ys = Xs[idx], ys[idx]
    m.to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=0.003)
    m.train()
    for ep in range(epochs):
        pm = torch.randperm(len(Xs))
        for i in range(0, len(Xs), 128):
            ix = pm[i:i+128]
            loss = F.cross_entropy(m(Xs[ix].to(DEVICE)), ys[ix].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    m.to('cpu').eval()
    return m

mA = train_tiny(clA, SEED)
mB = train_tiny(clB, SEED+100)
sdA, sdB = mA.state_dict(), mB.state_dict()

# ═══ Alignment (from E01) ═══
def align_channels(W_A, W_B):
    n = W_A.shape[0]
    A_f = W_A.reshape(n,-1).numpy(); B_f = W_B.reshape(n,-1).numpy()
    An = A_f/(np.linalg.norm(A_f,axis=1,keepdims=True)+1e-10)
    Bn = B_f/(np.linalg.norm(B_f,axis=1,keepdims=True)+1e-10)
    _, col = linear_sum_assignment(-An @ Bn.T)
    return col

def permute_bn(sd, prefix, p):
    for s in ['weight','bias','running_mean','running_var']:
        k = f"{prefix}.{s}"
        if k in sd: sd[k] = sd[k][p]

perm_c1 = align_channels(sdA['conv1.weight'], sdB['conv1.weight'])
sdB_a = {k:v.clone() for k,v in sdB.items()}
sdB_a['conv1.weight'] = sdB_a['conv1.weight'][perm_c1]
permute_bn(sdB_a, 'bn1', perm_c1)
sdB_a['block1.conv1.weight'] = sdB_a['block1.conv1.weight'][:, perm_c1, :, :]
p1 = align_channels(sdA['block1.conv1.weight'], sdB_a['block1.conv1.weight'])
sdB_a['block1.conv1.weight'] = sdB_a['block1.conv1.weight'][p1]
permute_bn(sdB_a, 'block1.bn1', p1)
sdB_a['block1.conv2.weight'] = sdB_a['block1.conv2.weight'][:, p1, :, :]
sdB_a['block1.conv2.weight'] = sdB_a['block1.conv2.weight'][perm_c1]
permute_bn(sdB_a, 'block1.bn2', perm_c1)
sdB_a['block2.conv1.weight'] = sdB_a['block2.conv1.weight'][:, perm_c1, :, :]
p2 = align_channels(sdA['block2.conv1.weight'], sdB_a['block2.conv1.weight'])
sdB_a['block2.conv1.weight'] = sdB_a['block2.conv1.weight'][p2]
permute_bn(sdB_a, 'block2.bn1', p2)
sdB_a['block2.conv2.weight'] = sdB_a['block2.conv2.weight'][:, p2, :, :]
sdB_a['block2.conv2.weight'] = sdB_a['block2.conv2.weight'][perm_c1]
permute_bn(sdB_a, 'block2.bn2', perm_c1)
sdB_a['fc.weight'] = sdB_a['fc.weight'][:, perm_c1]
print(f"Parents trained + aligned ({time.time()-t0:.1f}s)", flush=True)

# ═══ Layer groups for EA ═══
LAYER_GROUPS = ['conv1.weight', 'block1.conv1.weight', 'block1.conv2.weight',
                'block2.conv1.weight', 'block2.conv2.weight']
# Build key → group map
key_group = {}
for k in sdA:
    if 'fc' in k or 'num_batches_tracked' in k: continue
    for gi, g in enumerate(LAYER_GROUPS):
        prefix = g.replace('.weight','')
        if k.startswith(prefix):
            key_group[k] = gi
            break

dim = len(LAYER_GROUPS) + 2  # 5 lambdas + sA + sB

def build_merged(x):
    lambdas = 1/(1+np.exp(-x[:len(LAYER_GROUPS)]))
    sA, sB = x[-2], x[-1]
    sd = {}
    for k in sdA:
        if 'fc' in k or 'num_batches_tracked' in k: continue
        gi = key_group.get(k, 0)
        lam = float(lambdas[gi])
        sd[k] = (1-lam)*sdA[k] + lam*sdB_a[k]
    for k in sdA:
        if 'num_batches_tracked' in k: sd[k] = torch.tensor(0)
    wA_fc, bA_fc = sdA['fc.weight'], sdA['fc.bias']
    wB_fc, bB_fc = sdB_a['fc.weight'], sdB_a['fc.bias']
    fw = torch.zeros(4, 16); fb = torch.zeros(4)
    for ci,c in enumerate(clA): fw[c]=wA_fc[ci]*sA; fb[c]=bA_fc[ci]*sA
    for ci,c in enumerate(clB): fw[c]=wB_fc[ci]*sB; fb[c]=bB_fc[ci]*sB
    sd['fc.weight']=fw; sd['fc.bias']=fb
    m = TinyResNet(4); m.load_state_dict(sd)
    return m

def reset_bn(m, X, bs=256):
    m.train()
    with torch.no_grad():
        for i in range(0, len(X), bs): m(X[i:i+bs])
    m.eval()
    return m

# Fitness
mask_cal = sum(y_cal==c for c in ALL).bool()
X_c = X_cal[mask_cal]; y_c = y_cal[mask_cal]

def fitness(x):
    m = build_merged(x)
    reset_bn(m, X_cal[:2000])
    with torch.no_grad():
        preds = m(X_c).argmax(1)
    acc = (preds==y_c).float().mean().item()
    pc = {c:(preds[y_c==c]==c).float().mean().item() for c in ALL if (y_c==c).sum()>0}
    mn = min(pc.values()) if pc else 0
    ok = sum(1 for v in pc.values() if v>0.3)
    return -(0.3*acc + 0.4*mn + 0.2*ok/4 + 0.1*np.mean(list(pc.values())))

# CMA-ES
print(f"CMA-ES (dim={dim})...", flush=True)
x0 = np.zeros(dim); x0[-2]=1.0; x0[-1]=1.0
es = cma.CMAEvolutionStrategy(x0, 0.5,
    {'maxiter': 30, 'popsize': 14, 'seed': SEED, 'verbose': -1, 'timeout': 40})
bf = float('inf'); bx = None; gen = 0
while not es.stop():
    sols = es.ask(); sc = [fitness(x) for x in sols]
    es.tell(sols, sc)
    if min(sc)<bf: bf=min(sc); bx=sols[np.argmin(sc)]
    gen += 1
    if gen % 5 == 0:
        print(f"  Gen {gen}: fitness={-bf:.4f}", flush=True)

# Final eval on test
m_final = build_merged(bx)
reset_bn(m_final, X_cal)
mask_te = sum(y_te==c for c in ALL).bool()
Xt = X_te[mask_te]; yt = y_te[mask_te]
with torch.no_grad():
    preds = m_final(Xt).argmax(1)
acc = (preds==yt).float().mean().item()
pc = {c:round((preds[yt==c]==c).float().mean().item(),3) for c in ALL}
ok = sum(1 for c in ALL if pc[c]>0.3)

# Also eval aligned + BN (no EA) for comparison
m_fixed = build_merged(np.array([0,0,0,0,0,1.0,1.0]))
reset_bn(m_fixed, X_cal)
with torch.no_grad():
    preds_f = m_fixed(Xt).argmax(1)
acc_f = (preds_f==yt).float().mean().item()
pc_f = {c:round((preds_f[yt==c]==c).float().mean().item(),3) for c in ALL}
ok_f = sum(1 for c in ALL if pc_f[c]>0.3)

lambdas = 1/(1+np.exp(-bx[:len(LAYER_GROUPS)]))
print(f"\n--- Results ---")
print(f"  Fixed α=0.5: acc={acc_f:.3f} ok={ok_f}/4 pc={[pc_f[c] for c in ALL]}")
print(f"  EA optimized: acc={acc:.3f} ok={ok}/4 pc={[pc[c] for c in ALL]}")
print(f"  Lambdas: {[round(l,3) for l in lambdas]}")
print(f"  Scales: sA={bx[-2]:.3f} sB={bx[-1]:.3f}")
print(f"  CMA-ES: {gen} gens, {gen*14} evals")
print(f"  Time: {time.time()-t0:.1f}s")
print(f"\nmetric_fixed_ok: {ok_f}")
print(f"metric_ea_ok: {ok}")
print(f"metric_ea_acc: {acc}")
print(f"metric_improvement_over_fixed: {ok - ok_f}")
print("Done!", flush=True)
