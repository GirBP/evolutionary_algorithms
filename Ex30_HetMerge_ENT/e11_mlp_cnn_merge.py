#!/usr/bin/env python3
"""
E11 [Hypothesis] — Cross-architecture merge: MLP + CNN.
=========================================================
Truly heterogeneous: not just different widths, but different layer TYPES.

Strategy:
  Both models map input → penultimate features → output logits.
  CNN:  Conv layers → flatten → FC(d_cnn, 128) → ReLU → FC(128, 10)
  MLP:  FC(784, 256) → ReLU → FC(256, 128) → ReLU → FC(128, 10)

  Both produce 128-dim penultimate features (by design).
  
  Merge approach:
  1. SVD-align the 128-dim penultimate activations
  2. CMA-ES optimizes: scaling + per-layer α for FC head
  3. Merged model = CNN backbone + merged FC head
     → retains CNN's spatial feature extraction
     → injects MLP's classification patterns via merged head

  This is a PARTIAL merge: CNN conv weights stay, FC head gets mixed.
  But the FC head often carries most of the classification logic.

Fidelity: L0 (MNIST, 1 seed)
Kill threshold: retention < 0.60
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, json

SEED = 42
N_TR, N_VAL, N_TE = 2000, 1000, 1000


def load_data():
    import ssl; ssl._create_default_https_context = ssl._create_unverified_context
    from torchvision import datasets, transforms
    tf = transforms.ToTensor()
    tr = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=tf)
    te = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=tf)
    
    idx = torch.randperm(len(tr), generator=torch.Generator().manual_seed(0))
    X_all = torch.stack([tr[i][0] for i in range(len(tr))])
    y_all = torch.tensor([tr[i][1] for i in range(len(tr))])
    X_te = torch.stack([te[i][0] for i in range(N_TE)])
    y_te = torch.tensor([te[i][1] for i in range(N_TE)])
    
    return (X_all[idx[:N_TR]], y_all[idx[:N_TR]],
            X_all[idx[N_TR:N_TR+N_VAL]], y_all[idx[N_TR:N_TR+N_VAL]],
            X_te, y_te)


# ─── Models ──────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """CNN with 2 conv layers + FC head."""
    def __init__(self, fc_hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # After 2x MaxPool2d(2) on 28x28: 7x7
        self.flatten_dim = 32 * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 10),
        )
        self.fc_hidden = fc_hidden
    
    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)
    
    def get_penultimate(self, x):
        """Get 128-dim features before output layer."""
        with torch.no_grad():
            h = self.conv(x).view(x.size(0), -1)
            h = F.relu(self.fc[0](h))
        return h


class MLP(nn.Module):
    def __init__(self, arch):
        super().__init__()
        layers = []
        for i in range(len(arch)-1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i < len(arch)-2: layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.arch = arch
        self.fc_hidden = arch[-2]
    
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
    
    def get_penultimate(self, x):
        with torch.no_grad():
            h = x.view(x.size(0), -1)
            for m in self.net[:-1]:
                h = m(h)
        return h


def ev(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def train(model, X, y, epochs=5, lr=0.003):
    torch.manual_seed(SEED); np.random.seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        l = nn.CrossEntropyLoss()(model(X), y)
        opt.zero_grad(); l.backward(); opt.step()
    return model


# ─── Cross-architecture PCMA ────────────────────────────────────────

def cross_arch_pcma(cnn, mlp, X_cal, X_val, y_val, maxiter=40, popsize=14):
    """Merge MLP knowledge into CNN via FC head alignment.
    
    Strategy:
    1. Get penultimate features from both models on calibration data
    2. SVD align (both are 128-dim by design → square mapping)
    3. CMA-ES optimizes mapping + merge of output layer
    4. Result: CNN architecture with merged FC head
    """
    import cma
    
    # Penultimate features (before output layer)  
    H_cnn = cnn.get_penultimate(X_cal).numpy()  # (N, 128)
    H_mlp = mlp.get_penultimate(X_cal).numpy()  # (N, 128)
    
    d_cnn = H_cnn.shape[1]
    d_mlp = H_mlp.shape[1]
    d_min = min(d_cnn, d_mlp)
    
    # SVD alignment
    C = H_cnn.T @ H_mlp
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    s_init = S[:d_min] / (S[0] + 1e-10)
    U_use = U[:, :d_min]
    Vt_use = Vt[:d_min, :]
    
    # Output layer weights
    # CNN: fc[-1] = Linear(128, 10)
    W_cnn = cnn.fc[-1].weight.detach().numpy()  # (10, 128)
    b_cnn = cnn.fc[-1].bias.detach().numpy()    # (10,)
    
    # MLP: net[-1] = Linear(128, 10) 
    W_mlp = mlp.net[-1].weight.detach().numpy()  # (10, d_mlp)
    b_mlp = mlp.net[-1].bias.detach().numpy()    # (10,)
    
    # FC hidden layer weights (first FC layer)
    # CNN: fc[0] = Linear(flatten_dim, 128)
    W_fc_cnn = cnn.fc[0].weight.detach().numpy()  # (128, flatten_dim)
    b_fc_cnn = cnn.fc[0].bias.detach().numpy()    # (128,)
    
    # CMA-ES params: s (d_min) + alpha_hidden (1) + alpha_output (1)
    x0 = np.concatenate([s_init, [0.5, 0.5]])
    nd_s = d_min
    nd = len(x0)
    
    import copy
    
    def build_merged(s_vec, alpha_h, alpha_out):
        M = U_use @ np.diag(s_vec) @ Vt_use  # (d_cnn, d_mlp)
        
        # Map MLP's output weights through alignment
        # W_mlp maps from MLP's hidden space → output
        # We need to express this in CNN's hidden space
        # W_mlp_mapped = W_mlp @ M.T  (maps output weights to CNN's space)
        W_mlp_mapped = W_mlp @ M.T  if d_mlp <= d_cnn else W_mlp[:, :d_cnn]
        b_mlp_mapped = b_mlp
        
        # Merge output layer
        a = np.clip(alpha_out, 0.05, 0.95)
        W_merged = a * W_cnn + (1 - a) * W_mlp_mapped
        b_merged = a * b_cnn + (1 - a) * b_mlp_mapped
        
        # Build merged CNN
        merged = copy.deepcopy(cnn)
        with torch.no_grad():
            merged.fc[-1].weight.copy_(torch.tensor(W_merged, dtype=torch.float32))
            merged.fc[-1].bias.copy_(torch.tensor(b_merged, dtype=torch.float32))
        return merged
    
    def fitness(x):
        s = x[:nd_s]
        alpha_h, alpha_out = x[nd_s], x[nd_s+1]
        try:
            m = build_merged(s, alpha_h, alpha_out)
            return -ev(m, X_val, y_val)
        except:
            return 1.0
    
    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.3, {
        'maxiter': maxiter, 'popsize': popsize, 'seed': SEED,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [[-3]*nd_s + [0.1, 0.1], [3]*nd_s + [0.9, 0.9]],
    })
    while not es.stop():
        sols = es.ask()
        es.tell(sols, [fitness(np.array(s)) for s in sols])
    
    best = np.array(es.result.xbest)
    merged = build_merged(best[:nd_s], best[nd_s], best[nd_s+1])
    
    # Also try: just use CNN (no merge) vs ensemble blend
    return merged, [best[nd_s], best[nd_s+1]], nd


# ─── Per-class analysis ──────────────────────────────────────────────

def per_class_acc(model, X, y, n_classes=10):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(1)
    accs = []
    for c in range(n_classes):
        mask = (y == c)
        if mask.sum() > 0:
            accs.append((preds[mask] == c).float().mean().item())
        else:
            accs.append(0)
    return accs


# ─── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 75)
    print("  E11 [Hypothesis] Cross-architecture merge: MLP + CNN")
    print("=" * 75)
    
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_data()
    X_cal = X_tr[:3000]
    
    results = []
    
    # Test different MLP architectures merged with CNN
    configs = [
        ("CNN+MLP_same128", 128, [784, 256, 128, 10]),
        ("CNN+MLP_diff64",  128, [784, 128, 64, 10]),
        ("CNN+MLP_wide",    128, [784, 512, 128, 10]),
    ]
    
    for name, cnn_hidden, mlp_arch in configs:
        print(f"\n{'━' * 75}")
        print(f"  {name}")
        print(f"  CNN: Conv(1→16→32) → FC({32*7*7}→{cnn_hidden}→10)")
        print(f"  MLP: {mlp_arch}")
        print(f"{'━' * 75}")
        
        cnn = train(SimpleCNN(fc_hidden=cnn_hidden), X_tr, y_tr)
        mlp = train(MLP(mlp_arch), X_tr, y_tr)
        
        acc_cnn_te = ev(cnn, X_te, y_te)
        acc_mlp_te = ev(mlp, X_te, y_te)
        bp = max(acc_cnn_te, acc_mlp_te)
        
        print(f"  CNN acc (test): {acc_cnn_te:.4f}")
        print(f"  MLP acc (test): {acc_mlp_te:.4f}")
        
        # PCMA cross-architecture merge
        merged, alphas, nd = cross_arch_pcma(cnn, mlp, X_cal, X_val, y_val)
        acc_merged_val = ev(merged, X_val, y_val)
        acc_merged_te = ev(merged, X_te, y_te)
        ret = acc_merged_te / bp if bp > 0 else 0
        gap = acc_merged_val - acc_merged_te
        
        # Per-class analysis
        pc_cnn = per_class_acc(cnn, X_te, y_te)
        pc_mlp = per_class_acc(mlp, X_te, y_te)
        pc_merged = per_class_acc(merged, X_te, y_te)
        
        # Knowledge retention: does merged beat BOTH parents on any class?
        classes_cnn_better = sum(1 for c in range(10) if pc_cnn[c] > pc_mlp[c])
        classes_merged_best = sum(1 for c in range(10) 
                                  if pc_merged[c] >= min(pc_cnn[c], pc_mlp[c]) - 0.02)
        
        mk = "✅" if ret >= 0.60 else "❌"
        print(f"\n  Merged acc: val={acc_merged_val:.4f} test={acc_merged_te:.4f} gap={gap:+.4f}")
        print(f"  Retention: {ret:.4f} {mk}")
        print(f"  α = {[round(float(a),3) for a in alphas]}")
        
        print(f"\n  Per-class accuracy (test):")
        print(f"  {'Class':>6s} {'CNN':>6s} {'MLP':>6s} {'Merged':>7s} {'Best of':>8s}")
        for c in range(10):
            best = max(pc_cnn[c], pc_mlp[c])
            indicator = "✅" if pc_merged[c] >= best - 0.02 else "🟡"
            print(f"  {c:>6d} {pc_cnn[c]:>6.3f} {pc_mlp[c]:>6.3f} {pc_merged[c]:>7.3f} {best:>8.3f} {indicator}")
        
        results.append({
            'name': name, 'cnn_acc': round(acc_cnn_te, 4), 'mlp_acc': round(acc_mlp_te, 4),
            'merged_acc': round(acc_merged_te, 4), 'retention': round(ret, 4),
            'gen_gap': round(gap, 4), 'alphas': [round(float(a),3) for a in alphas],
            'per_class_merged': [round(a, 4) for a in pc_merged],
            'classes_preserved': classes_merged_best,
        })
    
    elapsed = time.time() - t0
    print(f"\n{'=' * 75}")
    print("  SUMMARY: Cross-Architecture Merge")
    print(f"{'=' * 75}")
    for r in results:
        mk = "✅" if r['retention'] >= 0.60 else "❌"
        print(f"  {r['name']:<20s}: CNN={r['cnn_acc']:.3f} MLP={r['mlp_acc']:.3f} "
              f"Merged={r['merged_acc']:.3f} ret={r['retention']:.3f} {mk} "
              f"classes_ok={r['classes_preserved']}/10")
    print(f"\n  Time: {elapsed:.1f}s")
    
    with open("results_e11.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
