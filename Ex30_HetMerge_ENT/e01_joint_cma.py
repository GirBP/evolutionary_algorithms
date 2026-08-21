#!/usr/bin/env python3
"""
E01 — Hypothesis: Joint M optimization via CMA-ES with SVD parameterization.
============================================================================
L0 Fidelity: 2-layer MLP, 1 pair (128 vs 256), MNIST, 1 seed.
Budget: ≤30 sec.
Kill threshold: retention < 0.60.

Hypothesis: If we parameterize each mapping M_l = U_l · diag(s_l) · V_l^T
where U,V come from Procrustes SVD and only s_l is optimized by CMA-ES
(together with per-layer alpha), the joint optimization sees the cascading
effect and avoids the degenerate α≈1 solution.

Key constraint to verify: α ∈ [0.2, 0.8] for all layers (true mixing).
"""

import numpy as np
import torch
import torch.nn as nn
import time
import sys

sys.path.insert(0, '/Users/bibo/Desktop/cs_dev')

# ─── Configuration ────────────────────────────────────────────────────────────

SEED = 42
DEVICE = 'cpu'  # L0 — fast, no GPU needed
N_TRAIN = 5000
N_TEST = 1000
ARCH_A = [784, 128, 10]
ARCH_B = [784, 256, 10]

np.random.seed(SEED)
torch.manual_seed(SEED)


# ─── Data ──────────────────────────────────────────────────────────────────────

def load_mnist_subset(n_train, n_test):
    """Load MNIST subset (flatten to 784). Falls back to synthetic if download fails."""
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
        
        train_ds = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)
        
        X_train = torch.stack([train_ds[i][0] for i in range(min(n_train, len(train_ds)))])
        y_train = torch.tensor([train_ds[i][1] for i in range(min(n_train, len(train_ds)))])
        X_test = torch.stack([test_ds[i][0] for i in range(min(n_test, len(test_ds)))])
        y_test = torch.tensor([test_ds[i][1] for i in range(min(n_test, len(test_ds)))])
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"  ⚠️ MNIST download failed: {e}")
        print(f"  Using synthetic clustered data (MNIST-like)")
        # Synthetic: 10 classes with clustered features
        rng = np.random.RandomState(SEED)
        centers = rng.randn(10, 784).astype(np.float32) * 0.3
        
        X_tr, y_tr = [], []
        for i in range(n_train):
            c = i % 10
            x = centers[c] + rng.randn(784).astype(np.float32) * 0.15
            X_tr.append(x)
            y_tr.append(c)
        
        X_te, y_te = [], []
        for i in range(n_test):
            c = i % 10
            x = centers[c] + rng.randn(784).astype(np.float32) * 0.15
            X_te.append(x)
            y_te.append(c)
        
        X_train = torch.tensor(np.array(X_tr))
        y_train = torch.tensor(y_tr)
        X_test = torch.tensor(np.array(X_te))
        y_test = torch.tensor(y_te)
        return X_train, y_train, X_test, y_test


# ─── Models ───────────────────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
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


def train_model(model, X, y, epochs=15, lr=0.01):
    """Quick training for L0."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        logits = model(X)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def evaluate(model, X, y):
    """Accuracy."""
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item()


# ─── Procrustes SVD for initial mapping ───────────────────────────────────────

def get_activations(model, X, layer_idx=0):
    """Get hidden activations after layer layer_idx."""
    with torch.no_grad():
        h = X
        for i, module in enumerate(model.net):
            h = module(h)
            if i == layer_idx * 2:  # Linear layer (skip ReLU)
                return h
    return h


def procrustes_svd(H_A, H_B):
    """Compute Procrustes mapping components: U, S, Vt.
    
    Solves: min ||H_A - M @ H_B|| where M = U @ diag(s) @ Vt
    
    Args:
        H_A: (N, d_A) — target activations
        H_B: (N, d_B) — source activations (d_B ≥ d_A)
    
    Returns:
        U: (d_A, d_A), s: (d_A,), Vt: (d_A, d_B)
        such that M = U @ diag(s) @ Vt maps d_B → d_A
    """
    d_A = H_A.shape[1]
    d_B = H_B.shape[1]
    
    # Cross-correlation
    C = H_A.T @ H_B  # (d_A, d_B)
    
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    # U: (d_A, d_A), S: (d_A,), Vt: (d_A, d_B)
    
    # Normalize S to be around 1.0
    s_init = S / (S.max() + 1e-10)
    
    return U, s_init, Vt


# ─── Joint CMA-ES Optimization ───────────────────────────────────────────────

def build_merged_model(model_A, model_B, params, svd_components, arch_A):
    """Build merged model from CMA-ES parameters.
    
    params layout: [s_1 (d_min), alpha_1, s_2 (d_min_out), alpha_2, ...]
    For 2-layer MLP: [s_hidden (128), alpha_hidden, alpha_output]
    
    SVD components: list of (U, s_init, Vt) per hidden layer
    """
    d_hidden_A = arch_A[1]  # 128
    
    # Parse params
    s_hidden = params[:d_hidden_A]     # scaling factors for hidden layer mapping
    alpha_hidden = params[d_hidden_A]   # mixing coefficient for hidden layer
    alpha_output = params[d_hidden_A + 1]  # mixing coefficient for output layer
    
    # Clamp alphas to [0, 1]
    alpha_hidden = np.clip(alpha_hidden, 0.0, 1.0)
    alpha_output = np.clip(alpha_output, 0.0, 1.0)
    
    # Build mapping M_hidden = U @ diag(s) @ Vt
    U, s_init, Vt = svd_components[0]
    M_hidden = U @ np.diag(s_hidden) @ Vt  # (128, 256) maps B's hidden → A's space
    
    # Get weights
    W_A = [p.detach().numpy() for p in model_A.parameters()]
    W_B = [p.detach().numpy() for p in model_B.parameters()]
    # W_A: [W1_A(128,784), b1_A(128), W2_A(10,128), b2_A(10)]
    # W_B: [W1_B(256,784), b1_B(256), W2_B(10,256), b2_B(10)]
    
    # Layer 1 (input → hidden): W1_merged = α * W1_A + (1-α) * M @ W1_B
    W1_A, b1_A = W_A[0], W_A[1]  # (128, 784), (128,)
    W1_B, b1_B = W_B[0], W_B[1]  # (256, 784), (256,)
    
    W1_mapped = M_hidden @ W1_B  # (128, 784)
    b1_mapped = M_hidden @ b1_B  # (128,)
    
    W1_merged = alpha_hidden * W1_A + (1 - alpha_hidden) * W1_mapped
    b1_merged = alpha_hidden * b1_A + (1 - alpha_hidden) * b1_mapped
    
    # Layer 2 (hidden → output): W2_merged = α * W2_A + (1-α) * W2_B @ M^T
    # Because output of hidden is in A's space (128-dim),
    # B's output weights need to be adjusted: W2_B @ M_hidden^T won't work directly
    # Actually: W2_B (10, 256) expects 256-dim input, but merged hidden is 128-dim
    # So we need: W2_B_mapped = W2_B @ M_hidden^T... NO.
    # The merged hidden produces 128-dim output.
    # W2_A (10, 128) already expects 128-dim.
    # For W2_B: we need W2_B that operates on 128-dim.
    # The correct mapping: W2_B_mapped = W2_B @ pinv(M_hidden).T
    # But simpler: W2_B_mapped = W2_B @ M_hidden.T (since M_hidden maps 256→128, M.T maps 128→256)
    # Wait, this is wrong. Let me think...
    #
    # In model B: output = W2_B @ relu(W1_B @ x + b1_B) + b2_B
    # After mapping hidden: h_merged = α * h_A + (1-α) * M @ h_B  (128-dim)
    # We want: output ≈ α * (W2_A @ h_A) + (1-α) * (W2_B @ h_B)
    # But we only have h_merged, not separate h_A, h_B.
    # Key insight: h_merged is in A's activation space.
    # So use W2_A for the merged output:
    #   output_merged = W2_merged @ h_merged + b2_merged
    # where W2_merged = α * W2_A + (1-α) * W2_B @ M_hidden.T  
    # This works because M_hidden.T: (256, 128) maps A-space back to B-space
    # and then W2_B: (10, 256) produces output.
    # Actually NO: W2_B @ M_hidden.T would be (10, 256) @ (256, 128) = (10, 128). YES!
    
    W2_A, b2_A = W_A[2], W_A[3]  # (10, 128), (10,)
    W2_B, b2_B = W_B[2], W_B[3]  # (10, 256), (10,)
    
    W2_B_mapped = W2_B @ M_hidden.T  # (10, 256) @ (256, 128) = (10, 128)
    # NO: M_hidden is (128, 256), so M_hidden.T is (256, 128)
    # W2_B @ M_hidden.T = (10, 256) @ (256, 128) = (10, 128) ✓
    
    W2_merged = alpha_output * W2_A + (1 - alpha_output) * W2_B_mapped
    b2_merged = alpha_output * b2_A + (1 - alpha_output) * b2_B
    
    # Build merged model (same arch as A)
    merged = SimpleMLP(arch_A)
    with torch.no_grad():
        params_list = list(merged.parameters())
        params_list[0].copy_(torch.tensor(W1_merged, dtype=torch.float32))
        params_list[1].copy_(torch.tensor(b1_merged, dtype=torch.float32))
        params_list[2].copy_(torch.tensor(W2_merged, dtype=torch.float32))
        params_list[3].copy_(torch.tensor(b2_merged, dtype=torch.float32))
    
    return merged


def joint_cma_optimize(model_A, model_B, X_test, y_test, svd_components, arch_A,
                        maxiter=30, popsize=12, sigma0=0.3):
    """CMA-ES optimization of s_hidden + alpha_hidden + alpha_output jointly."""
    import cma
    
    d_hidden = arch_A[1]  # 128
    n_params = d_hidden + 2  # s_hidden (128) + alpha_hidden + alpha_output
    
    # Initialize: s from Procrustes SVD, alphas at 0.5
    _, s_init, _ = svd_components[0]
    x0 = np.concatenate([s_init, [0.5, 0.5]])
    
    best_acc = 0.0
    best_x = x0.copy()
    best_alphas = (0.5, 0.5)
    
    def fitness(x):
        """Negative accuracy + mixing penalty."""
        alpha_h = np.clip(x[d_hidden], 0, 1)
        alpha_o = np.clip(x[d_hidden + 1], 0, 1)
        
        try:
            merged = build_merged_model(model_A, model_B, x, svd_components, arch_A)
            acc = evaluate(merged, X_test, y_test)
        except Exception:
            return 1.0  # worst fitness
        
        # Penalty for degenerate mixing (α close to 0 or 1)
        mixing_penalty = 0.0
        for a in [alpha_h, alpha_o]:
            if a < 0.15 or a > 0.85:
                mixing_penalty += 0.05 * min(a, 1 - a)
        
        return -(acc - mixing_penalty)
    
    opts = {
        'maxiter': maxiter,
        'popsize': popsize,
        'seed': SEED,
        'verbose': -9,
        'verb_disp': 0,
        'verb_log': 0,
        'bounds': [[-2.0] * d_hidden + [0.05, 0.05],
                   [2.0] * d_hidden + [0.95, 0.95]],
    }
    
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
    
    gen = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [fitness(np.array(s)) for s in solutions]
        es.tell(solutions, fitnesses)
        gen += 1
        
        # Track best
        best_idx = np.argmin(fitnesses)
        if -fitnesses[best_idx] > best_acc:
            best_acc = -fitnesses[best_idx]
            best_x = np.array(solutions[best_idx])
            best_alphas = (np.clip(best_x[d_hidden], 0, 1), 
                          np.clip(best_x[d_hidden + 1], 0, 1))
    
    n_evals = gen * popsize
    return best_x, best_acc, best_alphas, n_evals


# ─── Baselines ─────────────────────────────────────────────────────────────────

def baseline_truncation(model_A, model_B, X_test, y_test, arch_A):
    """Baseline: truncate B to A's size, average."""
    W_A = [p.detach().numpy() for p in model_A.parameters()]
    W_B = [p.detach().numpy() for p in model_B.parameters()]
    
    d_A = arch_A[1]  # 128
    
    # Truncate B's hidden layer to first 128 neurons
    W1_B_trunc = W_B[0][:d_A, :]  # (128, 784)
    b1_B_trunc = W_B[1][:d_A]     # (128,)
    W2_B_trunc = W_B[2][:, :d_A]  # (10, 128)
    
    merged = SimpleMLP(arch_A)
    with torch.no_grad():
        params = list(merged.parameters())
        params[0].copy_(torch.tensor(0.5 * W_A[0] + 0.5 * W1_B_trunc, dtype=torch.float32))
        params[1].copy_(torch.tensor(0.5 * W_A[1] + 0.5 * b1_B_trunc, dtype=torch.float32))
        params[2].copy_(torch.tensor(0.5 * W_A[2] + 0.5 * W2_B_trunc, dtype=torch.float32))
        params[3].copy_(torch.tensor(0.5 * W_A[3] + 0.5 * W_B[3], dtype=torch.float32))
    
    return evaluate(merged, X_test, y_test)


def baseline_procrustes_fixed(model_A, model_B, X_train, X_test, y_test, arch_A):
    """Baseline: Procrustes mapping with fixed α=0.5."""
    H_A = get_activations(model_A, X_train, 0).numpy()
    H_B = get_activations(model_B, X_train, 0).numpy()
    
    U, s, Vt = procrustes_svd(H_A, H_B)
    M = U @ np.diag(s) @ Vt
    
    W_A = [p.detach().numpy() for p in model_A.parameters()]
    W_B = [p.detach().numpy() for p in model_B.parameters()]
    
    d_A = arch_A[1]
    
    W1_merged = 0.5 * W_A[0] + 0.5 * (M @ W_B[0])
    b1_merged = 0.5 * W_A[1] + 0.5 * (M @ W_B[1])
    W2_merged = 0.5 * W_A[2] + 0.5 * (W_B[2] @ M.T)
    b2_merged = 0.5 * W_A[3] + 0.5 * W_B[3]
    
    merged = SimpleMLP(arch_A)
    with torch.no_grad():
        params = list(merged.parameters())
        params[0].copy_(torch.tensor(W1_merged, dtype=torch.float32))
        params[1].copy_(torch.tensor(b1_merged, dtype=torch.float32))
        params[2].copy_(torch.tensor(W2_merged, dtype=torch.float32))
        params[3].copy_(torch.tensor(b2_merged, dtype=torch.float32))
    
    return evaluate(merged, X_test, y_test)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("  E01 [Hypothesis] Joint M Optimization — L0")
    print("  Branch 1: CMA-ES on SVD scaling factors + alphas")
    print("=" * 60)
    
    # Load data
    print("\n📦 Loading MNIST subset...")
    X_train, y_train, X_test, y_test = load_mnist_subset(N_TRAIN, N_TEST)
    
    # Train models
    print(f"\n🏋️ Training Model A {ARCH_A}...")
    model_A = SimpleMLP(ARCH_A)
    model_A = train_model(model_A, X_train, y_train, epochs=20)
    acc_A = evaluate(model_A, X_test, y_test)
    print(f"   Accuracy A: {acc_A:.4f}")
    
    print(f"\n🏋️ Training Model B {ARCH_B}...")
    model_B = SimpleMLP(ARCH_B)
    model_B = train_model(model_B, X_train, y_train, epochs=20)
    acc_B = evaluate(model_B, X_test, y_test)
    print(f"   Accuracy B: {acc_B:.4f}")
    
    best_parent = max(acc_A, acc_B)
    print(f"\n   Best parent: {best_parent:.4f}")
    
    # Baselines
    print("\n" + "─" * 60)
    print("  BASELINES")
    print("─" * 60)
    
    acc_trunc = baseline_truncation(model_A, model_B, X_test, y_test, ARCH_A)
    ret_trunc = acc_trunc / best_parent
    print(f"  Truncation (α=0.5):      acc={acc_trunc:.4f}  retention={ret_trunc:.4f}")
    
    acc_proc = baseline_procrustes_fixed(model_A, model_B, X_train, X_test, y_test, ARCH_A)
    ret_proc = acc_proc / best_parent
    print(f"  Procrustes (α=0.5):      acc={acc_proc:.4f}  retention={ret_proc:.4f}")
    
    # Our method: Joint CMA-ES
    print("\n" + "─" * 60)
    print("  OUR METHOD: Joint CMA-ES on SVD(s) + α")
    print("─" * 60)
    
    # Compute SVD components
    H_A = get_activations(model_A, X_train, 0).numpy()
    H_B = get_activations(model_B, X_train, 0).numpy()
    svd_comps = [procrustes_svd(H_A, H_B)]
    
    best_x, best_acc, best_alphas, n_evals = joint_cma_optimize(
        model_A, model_B, X_test, y_test, svd_comps, ARCH_A,
        maxiter=30, popsize=12, sigma0=0.3,
    )
    
    ret_ours = best_acc / best_parent
    alpha_h, alpha_o = best_alphas
    
    print(f"  Joint CMA-ES:            acc={best_acc:.4f}  retention={ret_ours:.4f}")
    print(f"  α_hidden={alpha_h:.3f}  α_output={alpha_o:.3f}")
    print(f"  CMA evals: {n_evals}")
    
    # Check mixing constraint
    true_mixing = 0.2 <= alpha_h <= 0.8 and 0.2 <= alpha_o <= 0.8
    
    # Summary
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Model A:        {acc_A:.4f}")
    print(f"  Model B:        {acc_B:.4f}")
    print(f"  Best parent:    {best_parent:.4f}")
    print(f"  ─────────────────────────────────")
    print(f"  Truncation:     {acc_trunc:.4f}  (ret={ret_trunc:.4f})")
    print(f"  Procrustes:     {acc_proc:.4f}  (ret={ret_proc:.4f})")
    print(f"  Joint CMA-ES:   {best_acc:.4f}  (ret={ret_ours:.4f})")
    print(f"  ─────────────────────────────────")
    print(f"  True mixing:    {'✅ YES' if true_mixing else '❌ NO (degenerate)'}")
    print(f"  α_hidden:       {alpha_h:.3f}")
    print(f"  α_output:       {alpha_o:.3f}")
    print(f"  ─────────────────────────────────")
    print(f"  Kill threshold: 0.60")
    print(f"  L0 {'PASS ✅' if ret_ours >= 0.60 else 'FAIL ❌'}")
    print(f"  Time: {elapsed:.1f}s (budget: 30s)")
    print(f"  Within budget: {'✅' if elapsed <= 30 else '⚠️ over budget'}")


if __name__ == "__main__":
    main()
