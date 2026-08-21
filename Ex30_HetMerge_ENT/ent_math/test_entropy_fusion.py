#!/usr/bin/env python3
"""Method 1: Bayesian Entropy Fusion on ResNet-18 parents.
No retraining. Just forward both models + entropy gating."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, time
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets, models

DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
t0 = time.time()

# Load CIFAR-10 test
raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255 - mean)/std
y_te = torch.tensor(raw_te.targets)

# Load parents
clA = list(range(5))  # 0-4
clB = list(range(5,10))  # 5-9

def load_parent(path, n_classes):
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, n_classes)
    sd = torch.load(path, map_location='cpu', weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    return m

pA = load_parent('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/cifar_fix/results/parentA_s42.pth', 5)
pB = load_parent('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/cifar_fix/results/parentB_s42.pth', 5)

# Parent accuracies
import json
with open('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/cifar_fix/results/parents_strong.json') as f:
    pinfo = json.load(f)['42']
parent_pc = {}
for c in clA:
    parent_pc[c] = pinfo['pcA'][str(c)]
for c in clB:
    parent_pc[c] = pinfo['pcB'][str(c)]

print(f"Parents loaded ({time.time()-t0:.1f}s)")
print(f"  A classes {clA}: overall {pinfo['A']:.3f}")
print(f"  B classes {clB}: overall {pinfo['B']:.3f}")

# ═══ Method 1: Entropy Fusion ═══
def normalized_entropy(logits, n_classes):
    """Normalized Shannon entropy [0, 1]."""
    p = F.softmax(logits, dim=1)
    log_p = torch.log(p + 1e-10)
    H = -(p * log_p).sum(dim=1)  # [batch]
    return H / np.log(n_classes)

def entropy_fusion(pA, pB, X, clA, clB, T=1.0, batch_size=256):
    """Fuse two models using entropy-based gating."""
    pA_dev = pA.to(DEV); pB_dev = pB.to(DEV)
    pA_dev.eval(); pB_dev.eval()
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(DEV)
            
            logits_A = pA_dev(xb)  # [B, 5]
            logits_B = pB_dev(xb)  # [B, 5]
            
            H_A = normalized_entropy(logits_A, len(clA))  # [B]
            H_B = normalized_entropy(logits_B, len(clB))  # [B]
            
            # Gating weights via softmax(-H/T)
            scores = torch.stack([-H_A/T, -H_B/T], dim=1)  # [B, 2]
            W = F.softmax(scores, dim=1)  # [B, 2]
            W_A = W[:, 0:1]  # [B, 1]
            W_B = W[:, 1:2]  # [B, 1]
            
            # Weighted probabilities
            P_A = F.softmax(logits_A, dim=1) * W_A  # [B, 5]
            P_B = F.softmax(logits_B, dim=1) * W_B  # [B, 5]
            
            # Fused: [B, 10]
            P_fused = torch.cat([P_A, P_B], dim=1)
            
            preds = P_fused.argmax(dim=1)
            all_preds.append(preds)
    
    return torch.cat(all_preds)

# ═══ Eval ═══
def print_results(method_name, parent_pc, preds, y_true, classes):
    print(f"\n  {method_name}:")
    print(f"  {'Class':>5} | {'Parent':>8} | {'Merged':>8} | {'Drop%':>7} | Retained?")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}")
    retained = 0
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0: continue
        p = parent_pc[c]
        m = (preds[mask] == c).float().mean().item()
        drop = (1 - m/p) * 100 if p > 0 else 100
        ok = '✅' if drop <= 10 else '❌'
        if drop <= 10: retained += 1
        print(f"  {c:>5} | {p:>8.3f} | {m:>8.3f} | {drop:>6.1f}% | {ok}")
    total_correct = sum((preds == c).sum().item() for c in classes if (y_te == c).sum() > 0)
    total_samples = sum((y_te == c).sum().item() for c in classes)
    print(f"  Retention: {retained}/{len(classes)} (drop ≤ 10%)")
    print(f"  Overall acc: {total_correct/total_samples:.3f}")

all_classes = clA + clB

# Test with different temperatures
for T in [0.1, 0.5, 1.0, 2.0, 5.0]:
    preds = entropy_fusion(pA, pB, X_te, clA, clB, T=T)
    print_results(f"Entropy Fusion T={T}", parent_pc, preds, y_te, all_classes)

print(f"\nTotal: {time.time()-t0:.1f}s")
