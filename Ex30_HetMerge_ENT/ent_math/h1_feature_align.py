#!/usr/bin/env python3
"""H1 Kill Test: Feature Space Alignment (closed-form).

T = (G_A^T G_A)^{-1} G_A^T G_B
Maps g_B features → g_A feature space.
Then h_M[0:2] = h_A, h_M[2:4] = h_B · T (adapted FC).

KILL COND: retention = 0/4 → KILL
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import time
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
t0 = time.time()

# Data
raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mn = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
sd = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255-mn)/sd
y_te = torch.tensor(raw_te.targets)
clA, clB = [0,1], [2,3]; ALL = clA+clB

# Also load some public transfer data for computing T
raw_tr = datasets.CIFAR10('/tmp/cifar10', train=True, download=True)
X_tr = (torch.tensor(raw_tr.data).permute(0,3,1,2).float()/255-mn)/sd
X_pub = X_tr[:5000]  # public subset for computing alignment

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

# Load saved parents
data = torch.load('results/toy_parents.pth', weights_only=False)
mA = TinyResNet(2); mA.load_state_dict(data['sdA']); mA.eval()
mB = TinyResNet(2); mB.load_state_dict(data['sdB']); mB.eval()
pcA, pcB = data['pcA'], data['pcB']
parent_pc = {}; parent_pc.update(pcA); parent_pc.update(pcB)
print(f"Parents loaded ({time.time()-t0:.1f}s): A={pcA} B={pcB}", flush=True)

def print_results(method_name, parent_pc, merged_pc, classes):
    print(f"\n  {method_name}:")
    print(f"  {'Class':>5} | {'Parent':>8} | {'Merged':>8} | {'Drop%':>7} | Retained?")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}")
    retained = 0
    for c in classes:
        p = parent_pc[c]
        m = merged_pc[c]
        drop = (1 - m/p) * 100 if p > 0 else 100
        ok = '✅' if drop <= 10 else '❌'
        if drop <= 10: retained += 1
        print(f"  {c:>5} | {p:>8.3f} | {m:>8.3f} | {drop:>6.1f}% | {ok}")
    print(f"  Retention: {retained}/{len(classes)} (drop ≤ 10%)")
    return retained

# ═══ Compute features on public data ═══
with torch.no_grad():
    G_A = torch.cat([mA.features(X_pub[i:i+512]) for i in range(0,len(X_pub),512)])  # [5000, 16]
    G_B = torch.cat([mB.features(X_pub[i:i+512]) for i in range(0,len(X_pub),512)])  # [5000, 16]
print(f"Features: G_A={G_A.shape} G_B={G_B.shape}", flush=True)

# ═══ Compute alignment T = (G_A^T G_A)^{-1} G_A^T G_B ═══
# T maps B's features → A's feature space
# So h_B(g_B(x)) ≈ h_B(T^T g_A(x)) -- NO, we need it the other way:
# We keep g_A as the shared backbone.
# For B's classes: h_B originally takes g_B features.
# We need: h_B_adapted(g_A(x)) = h_B(T * g_A(x)) where T maps A→B space.
# T_AB = (G_B^T G_B)^{-1} G_B^T G_A  -- this maps A→B
# But let's solve directly: find T such that G_B ≈ G_A · T (A→B mapping)
# T = (G_A^T G_A)^{-1} G_A^T G_B
G_A_f = G_A.float(); G_B_f = G_B.float()

# Ridge regression for numerical stability
lam = 0.01
ATA = G_A_f.T @ G_A_f + lam * torch.eye(16)
ATB = G_A_f.T @ G_B_f
T = torch.linalg.solve(ATA, ATB)  # [16, 16]

# Check reconstruction quality
G_B_pred = G_A_f @ T
recon_err = ((G_B_f - G_B_pred)**2).mean().item()
recon_norm = (G_B_f**2).mean().item()
r2 = 1 - recon_err/recon_norm
print(f"Alignment T: shape={T.shape}, R²={r2:.4f} recon_err={recon_err:.4f}", flush=True)

# ═══ Build merged model ═══
# Use g_A as shared backbone.
# h_M[0:2] = h_A (direct, no change)
# h_M[2:4] = h_B(T * ·) = W_B @ T @ · + b_B
# So: W_M[2:4] = W_B @ T, b_M[2:4] = b_B

W_A = data['sdA']['fc.weight']  # [2, 16]
b_A = data['sdA']['fc.bias']    # [2]
W_B = data['sdB']['fc.weight']  # [2, 16]
b_B = data['sdB']['fc.bias']    # [2]

W_B_adapted = W_B.float() @ T.float()  # [2, 16] -- B's FC in A's feature space

# Create 4-class model with A's backbone
merged = TinyResNet(4)
sd_merged = {k: v.clone() for k, v in data['sdA'].items() if 'fc' not in k}

W_M = torch.zeros(4, 16)
b_M = torch.zeros(4)
W_M[0] = W_A[0]; b_M[0] = b_A[0]  # class 0
W_M[1] = W_A[1]; b_M[1] = b_A[1]  # class 1
W_M[2] = W_B_adapted[0]; b_M[2] = b_B[0]  # class 2
W_M[3] = W_B_adapted[1]; b_M[3] = b_B[1]  # class 3
sd_merged['fc.weight'] = W_M
sd_merged['fc.bias'] = b_M
merged.load_state_dict(sd_merged)
merged.eval()

# ═══ Eval ═══
mask = sum(y_te==c for c in ALL).bool()
Xt = X_te[mask]; yt = y_te[mask]
with torch.no_grad():
    preds = merged(Xt).argmax(1)
merged_pc = {c: (preds[yt==c]==c).float().mean().item() for c in ALL}

r1 = print_results("H1: Feature Align (T=ridge, g_A backbone)", parent_pc, merged_pc, ALL)

# ═══ Try also with B's backbone ═══
# T_BA: maps B→A, so h_A(T_BA * g_B(x))
ATA_B = G_B_f.T @ G_B_f + lam * torch.eye(16)
ATB_B = G_B_f.T @ G_A_f
T_BA = torch.linalg.solve(ATA_B, ATB_B)  # maps B features → A features

W_A_adapted = W_A.float() @ T_BA.float()  # A's FC in B's feature space

merged_B = TinyResNet(4)
sd_mB = {k: v.clone() for k, v in data['sdB'].items() if 'fc' not in k}
W_M2 = torch.zeros(4, 16); b_M2 = torch.zeros(4)
W_M2[0] = W_A_adapted[0]; b_M2[0] = b_A[0]
W_M2[1] = W_A_adapted[1]; b_M2[1] = b_A[1]
W_M2[2] = W_B[0]; b_M2[2] = b_B[0]
W_M2[3] = W_B[1]; b_M2[3] = b_B[1]
sd_mB['fc.weight'] = W_M2; sd_mB['fc.bias'] = b_M2
merged_B.load_state_dict(sd_mB); merged_B.eval()

with torch.no_grad():
    preds2 = merged_B(Xt).argmax(1)
merged_pc2 = {c: (preds2[yt==c]==c).float().mean().item() for c in ALL}
r2 = print_results("H1: Feature Align (T=ridge, g_B backbone)", parent_pc, merged_pc2, ALL)

# ═══ Try ensemble of both ═══
with torch.no_grad():
    logits1 = merged(Xt)
    logits2 = merged_B(Xt)
    preds_ens = ((logits1 + logits2)/2).argmax(1)
merged_pc3 = {c: (preds_ens[yt==c]==c).float().mean().item() for c in ALL}
r3 = print_results("H1: Ensemble (g_A + g_B)", parent_pc, merged_pc3, ALL)

best = max(r1, r2, r3)
elapsed = time.time()-t0
print(f"\n  R² (G_B ≈ G_A·T): {r2:.4f}")
print(f"  Time: {elapsed:.1f}s")
print(f"\nmetric_h1_gA_retained: {r1}")
print(f"metric_h1_gB_retained: {r2}")
print(f"metric_h1_ensemble_retained: {r3}")
print(f"metric_h1_best: {best}")

if best == 0:
    print("\n🔴 KILL H1: retention=0. Features not linearly related.")
else:
    print(f"\n🟢 H1 PASS: retention={best}/4. Proceed to optimize.")
print("Done!", flush=True)
