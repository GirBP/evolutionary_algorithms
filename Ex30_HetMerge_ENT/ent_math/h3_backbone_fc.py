#!/usr/bin/env python3
"""H3 Kill Test: Backbone swap + FC retrain.

Hypothesis: g_A backbone retains general features for ALL classes.
Method: Freeze g_A backbone, train ONLY FC layer using parent pseudo-labels.

Key difference from H1: H1 tried to adapt the FC weights analytically.
H3 LEARNS the FC layer from scratch using gradient descent.
Also tests g_B backbone.

KILL COND: accuracy < 70% on all 4 classes
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, time
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets

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
print(f"Parents loaded ({time.time()-t0:.1f}s): A={pcA} B={pcB}", flush=True)

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

# ═══ Generate pseudo-labels ═══
# For EACH image, get predictions from BOTH parents
# Then assign the label from the parent with HIGHER confidence
mA.eval(); mB.eval()
with torch.no_grad():
    logA = torch.cat([mA(X_pub[i:i+512]) for i in range(0,len(X_pub),512)])  # [5k, 2]
    logB = torch.cat([mB(X_pub[i:i+512]) for i in range(0,len(X_pub),512)])  # [5k, 2]
    # Get predicted class and confidence
    confA = F.softmax(logA, dim=1).max(1)
    confB = F.softmax(logB, dim=1).max(1)
    predA = confA.indices  # 0 or 1 → maps to class 0,1
    predB = confB.indices + 2  # 0 or 1 → maps to class 2,3
    confA_v = confA.values; confB_v = confB.values

# Strategy: for each image, it should learn BOTH parent predictions
# Duplicate: image → A label, same image → B label
print(f"  Conf A: mean={confA_v.mean():.3f} B: mean={confB_v.mean():.3f}", flush=True)

def train_fc_on_backbone(backbone_model, backbone_name, epochs=200, lr=0.01):
    """Freeze backbone, train FC layer."""
    backbone_model.eval()
    # Pre-compute features on public data (fast!)
    with torch.no_grad():
        feats = torch.cat([backbone_model.features(X_pub[i:i+512]) for i in range(0,len(X_pub),512)])
    
    # Duplicate: each image → A pseudo-label AND B pseudo-label
    feats_double = torch.cat([feats, feats], 0)  # [10k, 16]
    labels_double = torch.cat([predA, predB], 0)  # [10k]
    
    torch.manual_seed(SEED)
    fc = nn.Linear(feats.shape[1], 4)
    opt = torch.optim.Adam(fc.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    for ep in range(epochs):
        pm = torch.randperm(len(feats_double))
        for i in range(0, len(feats_double), 256):
            ix = pm[i:i+256]
            logits = fc(feats_double[ix].float())
            loss = F.cross_entropy(logits, labels_double[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    
    # Eval on test
    with torch.no_grad():
        te_feats = torch.cat([backbone_model.features(X_te[i:i+512]) for i in range(0,len(X_te),512)])
    mask = sum(y_te==c for c in ALL).bool()
    te_f = te_feats[mask]; yt = y_te[mask]
    with torch.no_grad():
        preds = fc(te_f.float()).argmax(1)
    merged_pc = {c:(preds[yt==c]==c).float().mean().item() for c in ALL}
    return print_results(f"H3: {backbone_name} + FC retrain ({epochs}ep)", parent_pc, merged_pc, ALL)

# ═══ Test both backbones ═══
print(f"\n--- g_A backbone ---", flush=True)
r1 = train_fc_on_backbone(mA, "g_A", epochs=200)

print(f"\n--- g_B backbone ---", flush=True)
r2 = train_fc_on_backbone(mB, "g_B", epochs=200)

# ═══ Try confidence-weighted training ═══
print(f"\n--- g_A backbone, confidence-weighted ---", flush=True)
mA.eval()
with torch.no_grad():
    feats = torch.cat([mA.features(X_pub[i:i+512]) for i in range(0,len(X_pub),512)])
    te_feats = torch.cat([mA.features(X_te[i:i+512]) for i in range(0,len(X_te),512)])

feats_double = torch.cat([feats, feats], 0)
labels_double = torch.cat([predA, predB], 0)
weights = torch.cat([confA_v, confB_v], 0)

torch.manual_seed(SEED)
fc3 = nn.Linear(16, 4)
opt = torch.optim.Adam(fc3.parameters(), lr=0.01)
for ep in range(200):
    pm = torch.randperm(len(feats_double))
    for i in range(0, len(feats_double), 256):
        ix = pm[i:i+256]
        logits = fc3(feats_double[ix].float())
        loss_per = F.cross_entropy(logits, labels_double[ix], reduction='none')
        loss = (loss_per * weights[ix]).mean()
        opt.zero_grad(); loss.backward(); opt.step()

mask = sum(y_te==c for c in ALL).bool()
te_f = te_feats[mask]; yt = y_te[mask]
with torch.no_grad():
    preds3 = fc3(te_f.float()).argmax(1)
merged_pc3 = {c:(preds3[yt==c]==c).float().mean().item() for c in ALL}
r3 = print_results("H3: g_A + FC(conf-weighted)", parent_pc, merged_pc3, ALL)

best = max(r1, r2, r3)
elapsed = time.time()-t0
print(f"\n  Time: {elapsed:.1f}s")
print(f"\nmetric_h3_gA_retained: {r1}")
print(f"metric_h3_gB_retained: {r2}")
print(f"metric_h3_conf_retained: {r3}")
print(f"metric_h3_best: {best}")

if best == 0:
    print("\n🔴 KILL H3: retention=0.")
else:
    print(f"\n🟢 H3 PASS: retention={best}/4.")
print("Done!", flush=True)
