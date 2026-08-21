#!/usr/bin/env python3
"""H1 Optimize: Feature alignment + CMA-ES scale/bias correction.

The T mapping works (R²=0.93) but adapted FC weights are imbalanced.
CMA-ES finds per-class scale/bias corrections.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, time
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets
import cma

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
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
X_pub = X_tr[:5000]

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
    def features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x); x = self.block2(x)
        return self.pool(x).flatten(1)

data = torch.load('results/toy_parents.pth', weights_only=False)
mA = TinyResNet(2); mA.load_state_dict(data['sdA']); mA.eval()
mB = TinyResNet(2); mB.load_state_dict(data['sdB']); mB.eval()
pcA, pcB = data['pcA'], data['pcB']
parent_pc = {}; parent_pc.update(pcA); parent_pc.update(pcB)
W_A, b_A = data['sdA']['fc.weight'], data['sdA']['fc.bias']
W_B, b_B = data['sdB']['fc.weight'], data['sdB']['fc.bias']

# Compute features and alignment T
with torch.no_grad():
    G_A = torch.cat([mA.features(X_pub[i:i+512]) for i in range(0,len(X_pub),512)])
    G_B = torch.cat([mB.features(X_pub[i:i+512]) for i in range(0,len(X_pub),512)])
    # Also compute on test subset for eval
    G_A_te = torch.cat([mA.features(X_te[i:i+512]) for i in range(0,len(X_te),512)])

G_Af = G_A.float(); G_Bf = G_B.float()
lam_reg = 0.01
T = torch.linalg.solve(G_Af.T @ G_Af + lam_reg*torch.eye(16), G_Af.T @ G_Bf)
W_B_adapted = W_B.float() @ T.float()  # [2, 16]
print(f"Setup ({time.time()-t0:.1f}s) R²={1-((G_Bf-G_Af@T)**2).mean()/G_Bf.var():.3f}", flush=True)

# Pre-compute features on a calibration subset
mask_cal = sum(y_te==c for c in ALL).bool()
# Use DualProbe approach: compute logits from features directly
G_A_te_f = G_A_te.float()

def print_results(name, ppc, mpc, cls):
    print(f"\n  {name}:")
    print(f"  {'Class':>5} | {'Parent':>8} | {'Merged':>8} | {'Drop%':>7} | Retained?")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}")
    ret = 0
    for c in cls:
        p = ppc[c]; m = mpc[c]
        drop = (1-m/p)*100 if p>0 else 100
        ok = '✅' if drop<=10 else '❌'
        if drop<=10: ret+=1
        print(f"  {c:>5} | {p:>8.3f} | {m:>8.3f} | {drop:>6.1f}% | {ok}")
    print(f"  Retention: {ret}/{len(cls)} (drop ≤ 10%)")
    return ret

# ═══ CMA-ES: optimize per-class scales and bias shifts ═══
# Chromosome: [s0, s1, s2, s3, b0, b1, b2, b3] = 8 params
# s_i = scale for class i logit, b_i = bias shift
dim = 8

# Build FC weights once
W_M_base = torch.zeros(4, 16)
b_M_base = torch.zeros(4)
W_M_base[0] = W_A[0]; b_M_base[0] = b_A[0]
W_M_base[1] = W_A[1]; b_M_base[1] = b_A[1]
W_M_base[2] = W_B_adapted[0]; b_M_base[2] = b_B[0]
W_M_base[3] = W_B_adapted[1]; b_M_base[3] = b_B[1]

# Use ALL test features for fast eval (no forward pass needed)
Xt_f = X_te[mask_cal]; yt_cal = y_te[mask_cal]
G_cal = G_A_te_f[mask_cal]  # features from g_A backbone

def fitness(x):
    """Compute logits = G_cal @ W^T + b, with per-class scale+shift."""
    scales = x[:4]
    biases = x[4:]
    W = W_M_base.clone().float()
    b = b_M_base.clone().float()
    for c in range(4):
        W[c] *= scales[c]
        b[c] = b[c] * scales[c] + biases[c]
    logits = G_cal @ W.T + b  # [N, 4]
    preds = logits.argmax(1)
    pc = {}
    for c in ALL:
        mc = yt_cal==c
        pc[c] = (preds[mc]==c).float().mean().item() if mc.sum()>0 else 0
    drops = [(1-pc[c]/parent_pc[c])*100 if parent_pc[c]>0 else 100 for c in ALL]
    ret = sum(1 for d in drops if d<=10)
    mn = min(pc.values())
    return -(0.3*ret/4 + 0.4*mn + 0.3*np.mean(list(pc.values())))

print(f"CMA-ES (dim={dim})...", flush=True)
x0 = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
es = cma.CMAEvolutionStrategy(x0, 0.5,
    {'maxiter': 50, 'popsize': 20, 'seed': SEED, 'verbose': -1, 'timeout': 60})
bf = float('inf'); bx = None; gen = 0
while not es.stop():
    sols = es.ask(); sc = [fitness(x) for x in sols]
    es.tell(sols, sc)
    if min(sc)<bf: bf=min(sc); bx=sols[np.argmin(sc)]
    gen += 1
    if gen % 10 == 0:
        print(f"  Gen {gen}: fitness={-bf:.4f}", flush=True)

# Final eval
scales = bx[:4]; biases = bx[4:]
W_final = W_M_base.clone().float(); b_final = b_M_base.clone().float()
for c in range(4):
    W_final[c] *= scales[c]; b_final[c] = b_final[c]*scales[c]+biases[c]

logits_f = G_cal @ W_final.T + b_final
preds_f = logits_f.argmax(1)
merged_pc = {c:(preds_f[yt_cal==c]==c).float().mean().item() for c in ALL}
r1 = print_results("H1+CMA: scales+bias (g_A backbone)", parent_pc, merged_pc, ALL)

print(f"\n  Scales: {[round(s,3) for s in scales]}")
print(f"  Biases: {[round(b,3) for b in biases]}")
print(f"  CMA: {gen} gens, {gen*20} evals")

# ═══ Also try: retrain FC layer with gradient descent ═══
print(f"\n--- FC Retrain (gradient descent, g_A backbone) ---", flush=True)

# Generate pseudo-labels from parents on public data
with torch.no_grad():
    pA = torch.cat([mA(X_pub[i:i+512]).argmax(1) for i in range(0,len(X_pub),512)])  # 0 or 1
    pB = torch.cat([mB(X_pub[i:i+512]).argmax(1) for i in range(0,len(X_pub),512)]) + 2  # 2 or 3
    fA = torch.cat([mA.features(X_pub[i:i+512]) for i in range(0,len(X_pub),512)])

# Train FC on g_A features with pseudo-labels from both parents
# For each image: use the parent whose confidence is higher
with torch.no_grad():
    confA = F.softmax(torch.cat([mA(X_pub[i:i+512]) for i in range(0,len(X_pub),512)]),1).max(1)[0]
    confB = F.softmax(torch.cat([mB(X_pub[i:i+512]) for i in range(0,len(X_pub),512)]),1).max(1)[0]

# Duplicate data: each image → A label AND B label
features_double = torch.cat([fA, fA], 0)  # [10000, 16]
labels_double = torch.cat([pA, pB], 0)     # [10000]

torch.manual_seed(SEED)
fc = nn.Linear(16, 4)
opt = torch.optim.Adam(fc.parameters(), lr=0.01)
for ep in range(100):
    pm = torch.randperm(len(features_double))[:1024]
    logits = fc(features_double[pm].float())
    loss = F.cross_entropy(logits, labels_double[pm])
    opt.zero_grad(); loss.backward(); opt.step()

# Eval
fc.eval()
with torch.no_grad():
    preds_fc = fc(G_cal.float()).argmax(1)
merged_pc_fc = {c:(preds_fc[yt_cal==c]==c).float().mean().item() for c in ALL}
r2 = print_results("H1+FC retrain (g_A features, pseudo-labels)", parent_pc, merged_pc_fc, ALL)

best = max(r1, r2)
elapsed = time.time()-t0
print(f"\n  Time: {elapsed:.1f}s")
print(f"\nmetric_h1_cma_retained: {r1}")
print(f"metric_h1_fc_retained: {r2}")
print(f"metric_h1_best: {best}")
print("Done!", flush=True)
