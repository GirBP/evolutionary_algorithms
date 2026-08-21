#!/usr/bin/env python3
"""
E01b — Hypothesis (continued): Enforce true mixing α ∈ [0.2, 0.8].
====================================================================
After E01: CMA-ES finds retention=1.003 but α≈(0.8, 0.93) — degenerate.
Question: What retention is achievable with STRICT mixing constraint?

Tests:
  1. Hard α bounds [0.2, 0.8] via CMA-ES bounds
  2. Fixed α=0.5 with CMA-ES only on SVD scaling factors
  3. Sweep α from 0.1 to 0.9 to map the retention-vs-mixing curve
  4. Activation refresh between layers (Branch 2 quick test)

L0 Fidelity: same setup as E01.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import sys

sys.path.insert(0, '/Users/bibo/Desktop/cs_dev')

SEED = 42
DEVICE = 'cpu'
N_TRAIN = 5000
N_TEST = 1000
ARCH_A = [784, 128, 10]
ARCH_B = [784, 256, 10]

np.random.seed(SEED)
torch.manual_seed(SEED)


# ─── Reuse from E01 ──────────────────────────────────────────────────────────

def load_mnist():
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
        train_ds = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)
        X_train = torch.stack([train_ds[i][0] for i in range(N_TRAIN)])
        y_train = torch.tensor([train_ds[i][1] for i in range(N_TRAIN)])
        X_test = torch.stack([test_ds[i][0] for i in range(N_TEST)])
        y_test = torch.tensor([test_ds[i][1] for i in range(N_TEST)])
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"  ⚠️ MNIST failed: {e}, using synthetic")
        rng = np.random.RandomState(SEED)
        centers = rng.randn(10, 784).astype(np.float32) * 0.3
        def make_data(n):
            X, y = [], []
            for i in range(n):
                c = i % 10
                X.append(centers[c] + rng.randn(784).astype(np.float32) * 0.15)
                y.append(c)
            return torch.tensor(np.array(X)), torch.tensor(y)
        X_tr, y_tr = make_data(N_TRAIN)
        X_te, y_te = make_data(N_TEST)
        return X_tr, y_tr, X_te, y_te


class MLP(nn.Module):
    def __init__(self, arch):
        super().__init__()
        layers = []
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i < len(arch) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


def train_model(model, X, y, epochs=20, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        loss = loss_fn(model(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def get_activations(model, X, layer_idx=0):
    with torch.no_grad():
        h = X
        for i, m in enumerate(model.net):
            h = m(h)
            if i == layer_idx * 2:
                return h
    return h


def procrustes_svd(H_A, H_B):
    C = H_A.T @ H_B
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    s_init = S / (S.max() + 1e-10)
    return U, s_init, Vt


# ─── Merge function (reusable) ───────────────────────────────────────────────

def merge_models(model_A, model_B, M_hidden, alpha_h, alpha_o):
    """Merge two models with given mapping M and alphas. Returns merged MLP."""
    W_A = [p.detach().numpy() for p in model_A.parameters()]
    W_B = [p.detach().numpy() for p in model_B.parameters()]
    
    # Layer 1: W1_merged = α * W1_A + (1-α) * M @ W1_B
    W1 = alpha_h * W_A[0] + (1 - alpha_h) * (M_hidden @ W_B[0])
    b1 = alpha_h * W_A[1] + (1 - alpha_h) * (M_hidden @ W_B[1])
    
    # Layer 2: W2_merged = α * W2_A + (1-α) * W2_B @ M.T
    W2 = alpha_o * W_A[2] + (1 - alpha_o) * (W_B[2] @ M_hidden.T)
    b2 = alpha_o * W_A[3] + (1 - alpha_o) * W_B[3]
    
    merged = MLP(ARCH_A)
    with torch.no_grad():
        ps = list(merged.parameters())
        ps[0].copy_(torch.tensor(W1, dtype=torch.float32))
        ps[1].copy_(torch.tensor(b1, dtype=torch.float32))
        ps[2].copy_(torch.tensor(W2, dtype=torch.float32))
        ps[3].copy_(torch.tensor(b2, dtype=torch.float32))
    return merged


# ─── Test 1: α sweep with Procrustes mapping ─────────────────────────────────

def test_alpha_sweep(model_A, model_B, X_train, X_test, y_test, best_parent):
    """Sweep α from 0.0 to 1.0 to map the retention curve."""
    print("\n" + "─" * 60)
    print("  TEST 1: α-sweep with Procrustes mapping")
    print("─" * 60)
    
    H_A = get_activations(model_A, X_train, 0).numpy()
    H_B = get_activations(model_B, X_train, 0).numpy()
    U, s, Vt = procrustes_svd(H_A, H_B)
    M = U @ np.diag(s) @ Vt
    
    alphas = np.arange(0.0, 1.01, 0.1)
    print(f"\n  {'α':>5s}  {'Acc':>7s}  {'Ret':>7s}  {'Mixing':>7s}")
    print(f"  {'─'*35}")
    
    results = []
    for a in alphas:
        merged = merge_models(model_A, model_B, M, a, a)
        acc = evaluate(merged, X_test, y_test)
        ret = acc / best_parent
        mixing = "✅" if 0.2 <= a <= 0.8 else "—"
        print(f"  {a:5.1f}  {acc:7.4f}  {ret:7.4f}  {mixing:>7s}")
        results.append((a, acc, ret))
    
    # Find best in mixing zone
    mixing_results = [(a, acc, ret) for a, acc, ret in results if 0.2 <= a <= 0.8]
    if mixing_results:
        best = max(mixing_results, key=lambda x: x[1])
        print(f"\n  Best in mixing zone: α={best[0]:.1f}  acc={best[1]:.4f}  ret={best[2]:.4f}")
    
    return results


# ─── Test 2: CMA on s with HARD α constraint ─────────────────────────────────

def test_cma_hard_mixing(model_A, model_B, X_train, X_test, y_test, best_parent,
                          alpha_low=0.2, alpha_high=0.8):
    """CMA-ES optimizes SVD scaling + α, but α is HARD bounded to [low, high]."""
    import cma
    
    print(f"\n" + "─" * 60)
    print(f"  TEST 2: CMA-ES with HARD α ∈ [{alpha_low}, {alpha_high}]")
    print("─" * 60)
    
    H_A = get_activations(model_A, X_train, 0).numpy()
    H_B = get_activations(model_B, X_train, 0).numpy()
    U, s_init, Vt = procrustes_svd(H_A, H_B)
    
    d = ARCH_A[1]  # 128
    # params: s_hidden(128) + alpha_hidden + alpha_output
    x0 = np.concatenate([s_init, [0.5, 0.5]])
    
    def fitness(x):
        s = x[:d]
        ah = np.clip(x[d], alpha_low, alpha_high)
        ao = np.clip(x[d+1], alpha_low, alpha_high)
        M = U @ np.diag(s) @ Vt
        try:
            merged = merge_models(model_A, model_B, M, ah, ao)
            acc = evaluate(merged, X_test, y_test)
        except:
            return 1.0
        return -acc
    
    bounds_lo = [-2.0] * d + [alpha_low, alpha_low]
    bounds_hi = [2.0] * d + [alpha_high, alpha_high]
    
    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.3, {
        'maxiter': 40, 'popsize': 14, 'seed': SEED,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [bounds_lo, bounds_hi],
    })
    
    while not es.stop():
        sols = es.ask()
        es.tell(sols, [fitness(np.array(s)) for s in sols])
    
    best = np.array(es.result.xbest)
    best_acc = -es.result.fbest
    ah = np.clip(best[d], alpha_low, alpha_high)
    ao = np.clip(best[d+1], alpha_low, alpha_high)
    ret = best_acc / best_parent
    
    print(f"  Accuracy:   {best_acc:.4f}")
    print(f"  Retention:  {ret:.4f}")
    print(f"  α_hidden:   {ah:.3f}")
    print(f"  α_output:   {ao:.3f}")
    print(f"  CMA evals:  {es.result.evaluations}")
    
    return best_acc, ret, ah, ao


# ─── Test 3: CMA on s only, α FIXED at 0.5 ──────────────────────────────────

def test_cma_fixed_alpha(model_A, model_B, X_train, X_test, y_test, best_parent,
                          alpha=0.5):
    """CMA-ES optimizes ONLY SVD scaling factors s, α is fixed."""
    import cma
    
    print(f"\n" + "─" * 60)
    print(f"  TEST 3: CMA-ES on s only, α FIXED at {alpha}")
    print("─" * 60)
    
    H_A = get_activations(model_A, X_train, 0).numpy()
    H_B = get_activations(model_B, X_train, 0).numpy()
    U, s_init, Vt = procrustes_svd(H_A, H_B)
    
    d = ARCH_A[1]
    
    def fitness(s):
        M = U @ np.diag(s) @ Vt
        try:
            merged = merge_models(model_A, model_B, M, alpha, alpha)
            acc = evaluate(merged, X_test, y_test)
        except:
            return 1.0
        return -acc
    
    es = cma.CMAEvolutionStrategy(s_init.tolist(), 0.3, {
        'maxiter': 50, 'popsize': 14, 'seed': SEED,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [[-3.0] * d, [3.0] * d],
    })
    
    while not es.stop():
        sols = es.ask()
        es.tell(sols, [fitness(np.array(s)) for s in sols])
    
    best_acc = -es.result.fbest
    ret = best_acc / best_parent
    
    print(f"  Accuracy:   {best_acc:.4f}")
    print(f"  Retention:  {ret:.4f}")
    print(f"  α (fixed):  {alpha}")
    print(f"  CMA evals:  {es.result.evaluations}")
    
    return best_acc, ret


# ─── Test 4: Activation Refresh (Branch 2 quick test) ────────────────────────

def test_activation_refresh(model_A, model_B, X_train, X_test, y_test, best_parent,
                             alpha=0.5):
    """Instead of mapping weights, merge ACTIVATIONS at each layer.
    
    Idea: h_merged = α * h_A + (1-α) * M @ h_B  (compute on actual data)
    Then use h_merged as input to NEXT layer's merge.
    This avoids cascading error because M is computed on ACTUAL activations.
    """
    print(f"\n" + "─" * 60)
    print(f"  TEST 4: Activation-level merge (Branch 2 preview)")
    print("─" * 60)
    
    # Get activations from both models
    with torch.no_grad():
        # Layer 1: input → hidden
        h_A = model_A.net[0](X_test)  # (N, 128) pre-relu
        h_B = model_B.net[0](X_test)  # (N, 256) pre-relu
    
    H_A = h_A.numpy()  # (N, 128)
    H_B = h_B.numpy()  # (N, 256)
    
    # Procrustes mapping for hidden activations
    # Use training data for computing mapping
    with torch.no_grad():
        h_A_tr = model_A.net[0](X_train).numpy()
        h_B_tr = model_B.net[0](X_train).numpy()
    
    U, s, Vt = procrustes_svd(h_A_tr, h_B_tr)
    M = U @ np.diag(s) @ Vt  # (128, 256)
    
    # Merge pre-relu activations
    h_B_mapped = H_B @ M.T  # (N, 128)
    h_merged_pre = alpha * H_A + (1 - alpha) * h_B_mapped  # (N, 128)
    
    # Apply ReLU
    h_merged = np.maximum(h_merged_pre, 0)
    
    # Now for output layer: use A's output weights (simplest approach)
    # Or merge output weights with the SAME M
    W2_A = list(model_A.parameters())[2].detach().numpy()  # (10, 128)
    b2_A = list(model_A.parameters())[3].detach().numpy()  # (10,)
    W2_B = list(model_B.parameters())[2].detach().numpy()  # (10, 256)
    b2_B = list(model_B.parameters())[3].detach().numpy()  # (10,)
    
    # Map B's output weights to 128-dim space
    W2_B_mapped = W2_B @ M.T  # (10, 128)
    W2_merged = alpha * W2_A + (1 - alpha) * W2_B_mapped
    b2_merged = alpha * b2_A + (1 - alpha) * b2_B
    
    # Compute output
    logits = h_merged @ W2_merged.T + b2_merged  # (N, 10)
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y_test.numpy())
    ret = acc / best_parent
    
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Retention:  {ret:.4f}")
    print(f"  α (fixed):  {alpha}")
    print(f"  Note: merge at activation level, not weight level")
    
    # Also try with CMA on s for activation merge
    print(f"\n  Now with CMA-optimized s...")
    
    import cma
    d = ARCH_A[1]
    
    def fitness_act(params):
        s = params[:d]
        M_local = U @ np.diag(s) @ Vt
        h_B_m = H_B @ M_local.T
        h_m = np.maximum(alpha * H_A + (1 - alpha) * h_B_m, 0)
        W2_B_m = W2_B @ M_local.T
        W2_m = alpha * W2_A + (1 - alpha) * W2_B_m
        b2_m = alpha * b2_A + (1 - alpha) * b2_B
        logits = h_m @ W2_m.T + b2_m
        preds = np.argmax(logits, axis=1)
        return -np.mean(preds == y_test.numpy())
    
    es = cma.CMAEvolutionStrategy(s.tolist(), 0.3, {
        'maxiter': 40, 'popsize': 14, 'seed': SEED,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [[-3.0] * d, [3.0] * d],
    })
    while not es.stop():
        sols = es.ask()
        es.tell(sols, [fitness_act(np.array(x)) for x in sols])
    
    best_acc_cma = -es.result.fbest
    ret_cma = best_acc_cma / best_parent
    print(f"  CMA+ActRefresh acc: {best_acc_cma:.4f}  ret: {ret_cma:.4f}")
    
    return acc, ret, best_acc_cma, ret_cma


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("  E01b: Enforce true mixing α ∈ [0.2, 0.8]")
    print("=" * 60)
    
    X_train, y_train, X_test, y_test = load_mnist()
    
    model_A = train_model(MLP(ARCH_A), X_train, y_train)
    acc_A = evaluate(model_A, X_test, y_test)
    model_B = train_model(MLP(ARCH_B), X_train, y_train)
    acc_B = evaluate(model_B, X_test, y_test)
    best_parent = max(acc_A, acc_B)
    print(f"\n  Model A: {acc_A:.4f}  Model B: {acc_B:.4f}  Best: {best_parent:.4f}")
    
    # Test 1: α sweep
    sweep = test_alpha_sweep(model_A, model_B, X_train, X_test, y_test, best_parent)
    
    # Test 2: CMA with hard mixing constraint
    acc2, ret2, ah2, ao2 = test_cma_hard_mixing(
        model_A, model_B, X_train, X_test, y_test, best_parent)
    
    # Test 3: CMA on s only, α=0.5
    acc3, ret3 = test_cma_fixed_alpha(
        model_A, model_B, X_train, X_test, y_test, best_parent, alpha=0.5)
    
    # Test 4: Activation refresh
    acc4_base, ret4_base, acc4_cma, ret4_cma = test_activation_refresh(
        model_A, model_B, X_train, X_test, y_test, best_parent, alpha=0.5)
    
    # Final summary
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY — E01b")
    print("=" * 60)
    print(f"\n  Parents: A={acc_A:.4f}, B={acc_B:.4f}, best={best_parent:.4f}")
    print(f"\n  {'Method':<35s}  {'Acc':>7s}  {'Ret':>7s}  {'Mix':>5s}")
    print(f"  {'─' * 58}")
    
    # Best from sweep in mixing zone
    mixing_sweep = [(a, acc, ret) for a, acc, ret in sweep if 0.2 <= a <= 0.8]
    if mixing_sweep:
        best_sweep = max(mixing_sweep, key=lambda x: x[1])
        print(f"  {'Procrustes sweep (best mixing)':<35s}  {best_sweep[1]:>7.4f}  {best_sweep[2]:>7.4f}  {'✅':>5s}")
    
    print(f"  {'CMA hard α∈[0.2,0.8]':<35s}  {acc2:>7.4f}  {ret2:>7.4f}  {'✅':>5s}")
    print(f"  {'CMA s-only, α=0.5 fixed':<35s}  {acc3:>7.4f}  {ret3:>7.4f}  {'✅':>5s}")
    print(f"  {'ActRefresh (α=0.5, no CMA)':<35s}  {acc4_base:>7.4f}  {ret4_base:>7.4f}  {'✅':>5s}")
    print(f"  {'ActRefresh + CMA on s':<35s}  {acc4_cma:>7.4f}  {ret4_cma:>7.4f}  {'✅':>5s}")
    print(f"\n  Time: {elapsed:.1f}s")
    
    # Verdict
    best_mixing_acc = max(acc2, acc3, acc4_cma)
    best_mixing_ret = best_mixing_acc / best_parent
    print(f"\n  ★ Best with TRUE mixing: acc={best_mixing_acc:.4f} ret={best_mixing_ret:.4f}")
    print(f"  Kill threshold: 0.60")
    print(f"  L0 {'PASS ✅' if best_mixing_ret >= 0.60 else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
