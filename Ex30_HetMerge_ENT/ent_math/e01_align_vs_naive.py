#!/usr/bin/env python3
"""E01: Toy TinyResNet — aligned vs unaligned merge.
Train 2 parents on CIFAR-10 (A=cls0,1; B=cls2,3), merge with/without alignment.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, json, time, sys
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets
from scipy.optimize import linear_sum_assignment

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
t0 = time.time()

# ═══ Data ═══
raw_tr = datasets.CIFAR10('/tmp/cifar10', train=True, download=True)
raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mn = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
sd = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_tr = (torch.tensor(raw_tr.data).permute(0,3,1,2).float()/255-mn)/sd
y_tr = torch.tensor(raw_tr.targets)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255-mn)/sd
y_te = torch.tensor(raw_te.targets)

clA, clB = [0,1], [2,3]
ALL = clA + clB
print(f"Data: {time.time()-t0:.1f}s", flush=True)

# ═══ Model ═══
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + x)

class TinyResNet(nn.Module):
    def __init__(self, nc=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.block1 = ResBlock(16)
        self.block2 = ResBlock(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, nc)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        return self.fc(self.pool(x).flatten(1))

# ═══ Train ═══
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
            xb = Xs[ix].to(DEVICE); yb = ys[ix].to(DEVICE)
            loss = F.cross_entropy(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    m.to('cpu').eval()
    mask_te = sum(y_te==c for c in cls_list).bool()
    Xt = X_te[mask_te]; yt = torch.tensor([cmap[y.item()] for y in y_te[mask_te]])
    with torch.no_grad():
        preds = m(Xt).argmax(1)
        acc = (preds==yt).float().mean().item()
    return m, acc

mA, accA = train_tiny(clA, SEED)
mB, accB = train_tiny(clB, SEED+100)
print(f"Parents: A={accA:.3f} B={accB:.3f} ({time.time()-t0:.1f}s)", flush=True)

# ═══ Alignment ═══
def align_channels(W_A, W_B):
    """Find permutation aligning B's output channels to A's."""
    n = W_A.shape[0]
    A_flat = W_A.reshape(n, -1).numpy()
    B_flat = W_B.reshape(n, -1).numpy()
    # Cosine similarity cost matrix
    A_norm = A_flat / (np.linalg.norm(A_flat, axis=1, keepdims=True) + 1e-10)
    B_norm = B_flat / (np.linalg.norm(B_flat, axis=1, keepdims=True) + 1e-10)
    cost = -A_norm @ B_norm.T
    _, col_idx = linear_sum_assignment(cost)
    sim = -cost[np.arange(n), col_idx].mean()
    return col_idx, sim

sdA, sdB = mA.state_dict(), mB.state_dict()

# Align conv1 output channels
perm_conv1, sim_conv1 = align_channels(sdA['conv1.weight'], sdB['conv1.weight'])
print(f"conv1 alignment: sim={sim_conv1:.3f}, perm={perm_conv1[:5]}...", flush=True)

# Apply permutation chain
def permute_bn(sd_B, prefix, perm):
    """Permute BN parameters."""
    for suffix in ['weight', 'bias', 'running_mean', 'running_var']:
        k = f"{prefix}.{suffix}"
        if k in sd_B:
            sd_B[k] = sd_B[k][perm]
    return sd_B

def build_aligned_B(sdB_orig, perm_conv1):
    """Build fully aligned copy of B's state_dict."""
    sdB_a = {k: v.clone() for k, v in sdB_orig.items()}
    p = perm_conv1
    
    # conv1 output → permute conv1 weights + bn1
    sdB_a['conv1.weight'] = sdB_a['conv1.weight'][p]
    permute_bn(sdB_a, 'bn1', p)
    
    # block1.conv1: input channels from conv1 → permute input
    sdB_a['block1.conv1.weight'] = sdB_a['block1.conv1.weight'][:, p, :, :]
    # block1.conv1 output: need new permutation
    p1, sim1 = align_channels(sdA['block1.conv1.weight'][:, :, :, :], sdB_a['block1.conv1.weight'])
    sdB_a['block1.conv1.weight'] = sdB_a['block1.conv1.weight'][p1]
    permute_bn(sdB_a, 'block1.bn1', p1)
    
    # block1.conv2: input from block1.conv1 output
    sdB_a['block1.conv2.weight'] = sdB_a['block1.conv2.weight'][:, p1, :, :]
    # block1.conv2 output must match conv1 output (residual!) → use perm_conv1
    # Actually for residual: output of block must match input of block
    # block1 input = conv1 output (permuted by p), block1 output must also be in p order
    # So block1.conv2 output permutation = p (to match residual skip)
    p1out = perm_conv1  # residual constraint
    sdB_a['block1.conv2.weight'] = sdB_a['block1.conv2.weight'][p1out]
    permute_bn(sdB_a, 'block1.bn2', p1out)
    
    # block2 input = block1 output = p order
    sdB_a['block2.conv1.weight'] = sdB_a['block2.conv1.weight'][:, p1out, :, :]
    p2, sim2 = align_channels(sdA['block2.conv1.weight'], sdB_a['block2.conv1.weight'])
    sdB_a['block2.conv1.weight'] = sdB_a['block2.conv1.weight'][p2]
    permute_bn(sdB_a, 'block2.bn1', p2)
    
    sdB_a['block2.conv2.weight'] = sdB_a['block2.conv2.weight'][:, p2, :, :]
    p2out = p1out  # residual constraint
    sdB_a['block2.conv2.weight'] = sdB_a['block2.conv2.weight'][p2out]
    permute_bn(sdB_a, 'block2.bn2', p2out)
    
    # fc: input from pool(block2 output) = p1out order
    sdB_a['fc.weight'] = sdB_a['fc.weight'][:, p1out]
    
    return sdB_a

sdB_aligned = build_aligned_B(sdB.copy(), perm_conv1)

# ═══ Merge (4-class model) ═══
def merge_models(sdA, sdB, alpha=0.5):
    """Merge state dicts into 4-class model."""
    sd = {}
    wA_fc = sdA['fc.weight']; bA_fc = sdA['fc.bias']
    wB_fc = sdB['fc.weight']; bB_fc = sdB['fc.bias']
    
    for k in sdA:
        if 'fc' in k: continue
        if 'num_batches_tracked' in k:
            sd[k] = sdA[k]
            continue
        sd[k] = (1-alpha)*sdA[k] + alpha*sdB[k]
    
    # Map fc: A's 2 classes → 0,1; B's 2 classes → 2,3
    fc_w = torch.zeros(4, wA_fc.shape[1])
    fc_b = torch.zeros(4)
    for ci, c in enumerate(clA):
        fc_w[c] = wA_fc[ci]; fc_b[c] = bA_fc[ci]
    for ci, c in enumerate(clB):
        fc_w[c] = wB_fc[ci]; fc_b[c] = bB_fc[ci]
    sd['fc.weight'] = fc_w; sd['fc.bias'] = fc_b
    return sd

# Build merged models
m_unaligned = TinyResNet(4)
sd_unaligned = merge_models(sdA, sdB)
m_unaligned.load_state_dict(sd_unaligned); m_unaligned.eval()

m_aligned = TinyResNet(4)
sd_aligned = merge_models(sdA, sdB_aligned)
m_aligned.load_state_dict(sd_aligned); m_aligned.eval()

# ═══ BN Reset ═══
def reset_bn(model, X, bs=256):
    model.train()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            model(X[i:i+bs])
    model.eval()
    return model

X_cal = X_tr[:5000]
m_unaligned_bn = TinyResNet(4)
m_unaligned_bn.load_state_dict(merge_models(sdA, sdB))
reset_bn(m_unaligned_bn, X_cal)

m_aligned_bn = TinyResNet(4)
m_aligned_bn.load_state_dict(merge_models(sdA, sdB_aligned))
reset_bn(m_aligned_bn, X_cal)

# ═══ Evaluation ═══
def eval_merged(model, name):
    mask = sum(y_te==c for c in ALL).bool()
    Xt = X_te[mask]; yt = y_te[mask]
    with torch.no_grad():
        preds = model(Xt).argmax(1)
    acc = (preds==yt).float().mean().item()
    pc = {c:(preds[yt==c]==c).float().mean().item() for c in ALL}
    ok = sum(1 for c in ALL if pc[c]>0.3)
    print(f"  {name:25s}: acc={acc:.3f} ok={ok}/4 pc={[round(pc[c],3) for c in ALL]}", flush=True)
    return {'name':name,'acc':round(acc,4),'ok':ok,'pc':{c:round(pc[c],3) for c in ALL}}

print(f"\n--- Results ---", flush=True)
r1 = eval_merged(m_unaligned, "Unaligned merge")
r2 = eval_merged(m_aligned, "Aligned merge")
r3 = eval_merged(m_unaligned_bn, "Unaligned + BN reset")
r4 = eval_merged(m_aligned_bn, "Aligned + BN reset")

elapsed = time.time()-t0
print(f"\nTotal: {elapsed:.1f}s")
print(f"\nmetric_unaligned_ok: {r1['ok']}")
print(f"metric_aligned_ok: {r2['ok']}")
print(f"metric_unaligned_bn_ok: {r3['ok']}")
print(f"metric_aligned_bn_ok: {r4['ok']}")
print(f"metric_improvement: {r4['ok'] - r1['ok']}")
print("Done!", flush=True)
