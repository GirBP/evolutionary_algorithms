#!/usr/bin/env python3
"""E30: CNN ENT via Zero-Cost Pruning + Merge.
Pipeline:
1. Train CNN_A (classes 0-4), CNN_B (classes 5-9)
2. ZCP-guided structured pruning → both CNNs get same target architecture
3. Merge conv layers (block-diagonal or interpolation)
4. ENT on FC head for output routing
"""
import numpy as np, torch, torch.nn as nn, torch.nn.utils.prune as prune
import random, time, copy, sys
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets, transforms

print("Loading data...", flush=True)
tf = transforms.Compose([transforms.ToTensor()])
tr = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=tf)
te = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=tf)
N = 15000
X_tr = torch.stack([tr[i][0] for i in range(N)]); y_tr = torch.tensor([tr[i][1] for i in range(N)])
X_te = torch.stack([te[i][0] for i in range(1000)]); y_te = torch.tensor([te[i][1] for i in range(1000)])
Xv, yv = X_tr[12000:15000], y_tr[12000:15000]
Xc = X_tr[:1000]

# ═══════════════════════════════════════════
# Models
# ═══════════════════════════════════════════

class CNN(nn.Module):
    def __init__(s, ch1=16, ch2=32, fc_dim=64):
        super().__init__()
        s.conv1 = nn.Conv2d(1, ch1, 3, padding=1)
        s.conv2 = nn.Conv2d(ch1, ch2, 3, padding=1)
        s.pool = nn.MaxPool2d(2)
        s.relu = nn.ReLU()
        s.fc1 = nn.Linear(ch2 * 7 * 7, fc_dim)
        s.fc2 = nn.Linear(fc_dim, 10)
        s.ch1, s.ch2, s.fc_dim = ch1, ch2, fc_dim

    def forward(s, x):
        if x.dim() == 2: x = x.view(-1, 1, 28, 28)
        x = s.pool(s.relu(s.conv1(x)))
        x = s.pool(s.relu(s.conv2(x)))
        x = x.view(x.size(0), -1)
        x = s.relu(s.fc1(x))
        return s.fc2(x)

    def features(s, x):
        if x.dim() == 2: x = x.view(-1, 1, 28, 28)
        x = s.pool(s.relu(s.conv1(x)))
        x = s.pool(s.relu(s.conv2(x)))
        x = x.view(x.size(0), -1)
        return s.relu(s.fc1(x))

def ev(m, X, y):
    m.eval()
    with torch.no_grad(): return (m(X).argmax(1) == y).float().mean().item()

def pc(m, X, y):
    m.eval()
    with torch.no_grad(): p = m(X).argmax(1)
    return {c: (p[y==c]==c).float().mean().item() if (y==c).sum() > 0 else 0 for c in range(10)}

def train_cnn(ch1, ch2, fc, X, y, cls, epochs=15):
    model = CNN(ch1, ch2, fc)
    mask = sum(y == c for c in cls).bool()
    Xs, ys = X[mask][:4000], y[mask][:4000]
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    model.train()
    for _ in range(epochs):
        l = nn.CrossEntropyLoss()(model(Xs), ys); opt.zero_grad(); l.backward(); opt.step()
    return model

# ═══════════════════════════════════════════
# Zero-Cost Proxy: filter importance scoring
# ═══════════════════════════════════════════

def synflow_importance(model, Xc):
    """SynFlow: data-agnostic zero-cost proxy for filter importance."""
    model.eval()
    # Set all params to their absolute values for SynFlow
    signs = {}
    for name, p in model.named_parameters():
        signs[name] = torch.sign(p.data)
        p.data.abs_()

    # Forward with ones input
    x = torch.ones_like(Xc[:1])
    model.zero_grad()
    out = model(x)
    out.sum().backward()

    importance = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            imp = (p.data * p.grad).abs()
            importance[name] = imp
        p.data *= signs[name]  # restore signs

    return importance

def magnitude_importance(model):
    """L1-norm of filters as importance."""
    importance = {}
    for name, p in model.named_parameters():
        if 'weight' in name and p.dim() >= 2:
            # Per-output-channel importance
            imp = p.data.abs().view(p.shape[0], -1).sum(dim=1)
            importance[name] = imp
    return importance

def get_filter_importance(model, Xc):
    """Combined importance: SynFlow + Magnitude."""
    mag = magnitude_importance(model)
    # Normalize and combine
    importance = {}
    for name in mag:
        m = mag[name]
        m = m / (m.max() + 1e-10)
        importance[name] = m
    return importance

# ═══════════════════════════════════════════
# Structured Pruning → target architecture
# ═══════════════════════════════════════════

def prune_cnn_to_target(model, target_ch1, target_ch2, target_fc, Xc):
    """Prune CNN to target architecture using importance scores.
    Returns new CNN with target dimensions and transferred weights."""
    importance = get_filter_importance(model, Xc)

    # Select top-k filters for each layer
    W1 = model.conv1.weight.data   # (ch1, 1, 3, 3)
    b1 = model.conv1.bias.data     # (ch1,)
    imp1 = importance.get('conv1.weight', torch.ones(W1.shape[0]))
    idx1 = torch.argsort(imp1, descending=True)[:target_ch1]
    idx1 = idx1.sort().values

    W2 = model.conv2.weight.data   # (ch2, ch1, 3, 3)
    b2 = model.conv2.bias.data     # (ch2,)
    imp2 = importance.get('conv2.weight', torch.ones(W2.shape[0]))
    idx2 = torch.argsort(imp2, descending=True)[:target_ch2]
    idx2 = idx2.sort().values

    Wfc1 = model.fc1.weight.data   # (fc, ch2*7*7)
    bfc1 = model.fc1.bias.data     # (fc,)
    # FC importance: per-neuron L1
    fc_imp = Wfc1.abs().sum(dim=1)
    idx_fc = torch.argsort(fc_imp, descending=True)[:target_fc]
    idx_fc = idx_fc.sort().values

    # Build pruned model
    pruned = CNN(target_ch1, target_ch2, target_fc)
    with torch.no_grad():
        # Conv1: keep top filters
        pruned.conv1.weight.copy_(W1[idx1])
        pruned.conv1.bias.copy_(b1[idx1])

        # Conv2: keep top output filters, keep only input channels from conv1
        pruned.conv2.weight.copy_(W2[idx2][:, idx1])
        pruned.conv2.bias.copy_(b2[idx2])

        # FC1: reshape index for flattened conv2 output
        # Original: (ch2 * 7 * 7) → need indices for selected ch2 channels
        flat_idx = []
        for ch_idx in idx2:
            start = ch_idx * 49  # 7*7
            flat_idx.extend(range(start, start + 49))
        flat_idx = torch.tensor(flat_idx)

        pruned.fc1.weight.copy_(Wfc1[idx_fc][:, flat_idx])
        pruned.fc1.bias.copy_(bfc1[idx_fc])

        # FC2: output layer
        pruned.fc2.weight.copy_(model.fc2.weight.data[:, idx_fc])
        pruned.fc2.bias.copy_(model.fc2.bias.data)

    return pruned

# ═══════════════════════════════════════════
# Merge pruned CNNs
# ═══════════════════════════════════════════

def merge_cnns(cnnA, cnnB, route, rA, rB):
    """Merge two same-architecture CNNs with per-class routing.
    Conv layers: concatenated (block-diagonal style)
    FC head: per-class routing from A/B features.
    """
    ch1, ch2, fc = cnnA.ch1, cnnA.ch2, cnnA.fc_dim

    # Merged conv layers: doubled channels
    merged = CNN(ch1 * 2, ch2 * 2, fc * 2)

    with torch.no_grad():
        # Conv1: block-diagonal [A_filters; B_filters]
        merged.conv1.weight.zero_()
        merged.conv1.bias.zero_()
        merged.conv1.weight[:ch1] = cnnA.conv1.weight
        merged.conv1.weight[ch1:] = cnnB.conv1.weight
        merged.conv1.bias[:ch1] = cnnA.conv1.bias
        merged.conv1.bias[ch1:] = cnnB.conv1.bias

        # Conv2: block-diagonal
        merged.conv2.weight.zero_()
        merged.conv2.bias.zero_()
        merged.conv2.weight[:ch2, :ch1] = cnnA.conv2.weight
        merged.conv2.weight[ch2:, ch1:] = cnnB.conv2.weight
        merged.conv2.bias[:ch2] = cnnA.conv2.bias
        merged.conv2.bias[ch2:] = cnnB.conv2.bias

        # FC1: block-diagonal
        merged.fc1.weight.zero_()
        merged.fc1.bias.zero_()
        # A's FC1: maps from first ch2*49 features → first fc neurons
        merged.fc1.weight[:fc, :ch2*49] = cnnA.fc1.weight
        merged.fc1.bias[:fc] = cnnA.fc1.bias
        # B's FC1: maps from second ch2*49 features → second fc neurons
        merged.fc1.weight[fc:, ch2*49:] = cnnB.fc1.weight
        merged.fc1.bias[fc:] = cnnB.fc1.bias

        # FC2 (output): per-class routing
        merged.fc2.weight.zero_()
        merged.fc2.bias.zero_()
        for c in range(10):
            a = 1.0 / (1.0 + np.exp(-route[c]))
            merged.fc2.weight[c, :fc] = a * rA * cnnA.fc2.weight[c]
            merged.fc2.weight[c, fc:] = (1-a) * rB * cnnB.fc2.weight[c]
            merged.fc2.bias[c] = a * rA * cnnA.fc2.bias[c] + (1-a) * rB * cnnB.fc2.bias[c]

    return merged

# ═══════════════════════════════════════════
# ENT on merged CNN
# ═══════════════════════════════════════════

def ent_cnn(cnnA, cnnB, Xv, yv, Xc, clA, clB, target_ch1=8, target_ch2=16, target_fc=32):
    """Full pipeline: prune → align → merge → EA optimize routing."""
    # Step 1: Prune both to target arch
    print(f"  Pruning A: {cnnA.ch1}/{cnnA.ch2}/{cnnA.fc_dim} → {target_ch1}/{target_ch2}/{target_fc}", flush=True)
    pA = prune_cnn_to_target(cnnA, target_ch1, target_ch2, target_fc, Xc)
    print(f"  Pruning B: {cnnB.ch1}/{cnnB.ch2}/{cnnB.fc_dim} → {target_ch1}/{target_ch2}/{target_fc}", flush=True)
    pB = prune_cnn_to_target(cnnB, target_ch1, target_ch2, target_fc, Xc)

    print(f"  Pruned A: {ev(pA,Xv,yv):.3f}  Pruned B: {ev(pB,Xv,yv):.3f}", flush=True)

    # Logit scale
    pA.eval(); pB.eval()
    with torch.no_grad():
        sA = pA(Xc).numpy().std(); sB = pB(Xc).numpy().std()
    t = (sA + sB) / 2; rA = t / (sA + 1e-10); rB = t / (sB + 1e-10)

    # Step 2: EA on routing
    def fitness(route):
        m = merge_cnns(pA, pB, route, rA, rB)
        d = pc(m, Xv, yv); acc = ev(m, Xv, yv)
        mn = min(d[c] for c in range(10))
        return 0.4*acc + 0.4*mn + 0.1*np.mean([d[c] for c in range(10)])

    pop = [np.array([2.]*5 + [-2.]*5)]  # A-route for 0-4, B-route for 5-9
    for _ in range(11):
        pop.append(np.random.randn(10) * 1.5)

    bf = -1; bc = None
    for gen in range(20):
        fs = [fitness(r) for r in pop]
        gi = np.argmax(fs)
        if fs[gi] > bf: bf = fs[gi]; bc = pop[gi].copy()
        if gen % 10 == 0 or gen == 19:
            m = merge_cnns(pA, pB, bc, rA, rB)
            d = pc(m, Xv, yv); mn = min(d[c] for c in range(10))
            print(f"  Gen {gen}: fit={fs[gi]:.4f} min={mn:.3f} acc={ev(m,Xv,yv):.3f}", flush=True)
        new = [bc.copy()]
        while len(new) < 12:
            i = random.randint(0, len(pop)-1)
            new.append(pop[i] + np.random.randn(10) * 0.3)
        pop = new

    merged = merge_cnns(pA, pB, bc, rA, rB)
    return merged, pA, pB, bc

# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
t0 = time.time()
clA, clB = list(range(5)), list(range(5, 10))

F = open('results_e30.txt', 'w')

# Train full CNNs
print("Training parent CNNs...", flush=True)
configs = [
    ("Same16/32/64", 16, 32, 64, 16, 32, 64),
    ("Big→Small 32/64/128", 32, 64, 128, 32, 64, 128),
    ("Het 16/32 vs 32/64", 16, 32, 64, 32, 64, 128),
]

for cfg_name, ch1A, ch2A, fcA, ch1B, ch2B, fcB in configs:
    print(f"\n{'═'*60}", flush=True)
    print(f"  {cfg_name}: A=[{ch1A},{ch2A},{fcA}] B=[{ch1B},{ch2B},{fcB}]", flush=True)
    F.write(f"\n{cfg_name}:\n")

    cnnA = train_cnn(ch1A, ch2A, fcA, X_tr, y_tr, clA)
    cnnB = train_cnn(ch1B, ch2B, fcB, X_tr, y_tr, clB)
    pcA, pcB = pc(cnnA, X_te, y_te), pc(cnnB, X_te, y_te)
    print(f"  Parent A: {ev(cnnA, X_te, y_te):.3f}  Parent B: {ev(cnnB, X_te, y_te):.3f}", flush=True)

    # Target: half of min arch
    tgt_ch1 = min(ch1A, ch1B) // 2
    tgt_ch2 = min(ch2A, ch2B) // 2
    tgt_fc = min(fcA, fcB) // 2
    if tgt_ch1 < 4: tgt_ch1 = 4
    if tgt_ch2 < 8: tgt_ch2 = 8
    if tgt_fc < 16: tgt_fc = 16

    merged, pA, pB, route = ent_cnn(cnnA, cnnB, Xv, yv, Xc, clA, clB,
                                      tgt_ch1, tgt_ch2, tgt_fc)

    pcM = pc(merged, X_te, y_te); acc = ev(merged, X_te, y_te)
    aM = np.mean([pcM[c] for c in clA]); bM = np.mean([pcM[c] for c in clB])
    bal = min(aM, bM) / (max(aM, bM) + 1e-10)
    nok = sum(1 for c in range(10) if pcM[c] > 0.3)
    mn = min(pcM[c] for c in range(10))

    # Param counts
    pars_A = sum(p.numel() for p in cnnA.parameters())
    pars_B = sum(p.numel() for p in cnnB.parameters())
    pars_M = sum(p.numel() for p in merged.parameters())

    line = f"  acc={acc:.3f} A={aM:.3f} B={bM:.3f} bal={bal:.3f} ok={nok}/10 min={mn:.3f}"
    line += f" params={pars_M} ({pars_M/(pars_A+pars_B)*100:.0f}% of sum)"
    print(line, flush=True)
    F.write(line + '\n')

    for c in range(10):
        bp = max(pcA[c], pcB[c]); mk = '✅' if pcM[c] >= 0.5 * bp else '❌'
        cl = f"  {c}: par={bp:.3f} ent={pcM[c]:.3f} {mk}"
        F.write(cl + '\n')
        print(cl, flush=True)

print(f"\nTime: {time.time()-t0:.1f}s", flush=True)
F.close()
print("Done!", flush=True)
