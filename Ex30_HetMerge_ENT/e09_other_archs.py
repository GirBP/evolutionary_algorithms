#!/usr/bin/env python3
"""
E09 [Hypothesis] — PCMA on non-standard architectures.
========================================================
Following Knuth protocol: new branch exploration.

Test PCMA on architectures beyond vanilla MLP:
  Arch 1: MLP + BatchNorm (нормалізація між шарами)
  Arch 2: MLP + different activations (LeakyReLU, GELU)
  Arch 3: MLP + asymmetric widths (не geometric decay, а random-ish)
  Arch 4: Simple CNN (Conv + FC head) — requires PCMA extension

Fidelity: L0 (MNIST subset, 1 seed per arch)
Kill threshold: retention < 0.60
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, sys, json

DEVICE = 'cpu'
SEED = 42
N_TR, N_VAL, N_TE = 5000, 2000, 2000

np.random.seed(SEED)
torch.manual_seed(SEED)


def load_data():
    import ssl; ssl._create_default_https_context = ssl._create_unverified_context
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
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


def ev(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def train(model, X, y, epochs=15, lr=0.003):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        loss = loss_fn(model(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


# ─── Architecture 1: MLP + BatchNorm ─────────────────────────────────

class MLP_BN(nn.Module):
    """MLP with BatchNorm after each hidden layer."""
    def __init__(self, arch, act='relu'):
        super().__init__()
        self.arch = arch
        layers = []
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i < len(arch) - 2:
                layers.append(nn.BatchNorm1d(arch[i+1]))
                if act == 'relu': layers.append(nn.ReLU())
                elif act == 'leaky': layers.append(nn.LeakyReLU(0.1))
                elif act == 'gelu': layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# ─── Generalized PCMA for arbitrary architectures ────────────────────

def extract_linear_layers(model):
    """Extract Linear layers and their indices in the model."""
    linears = []
    for i, m in enumerate(model.net):
        if isinstance(m, nn.Linear):
            linears.append((i, m))
    return linears


def get_hidden_activations(model, X, after_nonlinearity=True):
    """Get activations after each hidden layer's nonlinearity.
    Works with any architecture: skips BatchNorm, finds activations
    after ReLU/LeakyReLU/GELU (or after Linear if no nonlinearity).
    """
    acts = []
    linears = extract_linear_layers(model)
    n_linear = len(linears)
    
    with torch.no_grad():
        h = X
        last_linear_output = None
        for i, m in enumerate(model.net):
            h = m(h)
            if isinstance(m, nn.Linear):
                last_linear_output = h.clone()
            # After nonlinearity (not last layer)
            if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU)):
                acts.append(h.numpy().copy())
    
    return acts


def compute_svd_general(model_A, model_B, X_cal):
    """Compute SVD components for any MLP-like architecture."""
    acts_A = get_hidden_activations(model_A, X_cal)
    acts_B = get_hidden_activations(model_B, X_cal)
    
    linears_A = extract_linear_layers(model_A)
    linears_B = extract_linear_layers(model_B)
    n_hidden = len(linears_A) - 1  # exclude output layer
    
    svd_list, s_inits = [], []
    for i in range(n_hidden):
        d_a = linears_A[i][1].out_features
        d_b = linears_B[i][1].out_features
        d_min = min(d_a, d_b)
        
        HA = acts_A[i] if i < len(acts_A) else np.random.randn(len(X_cal), d_a).astype(np.float32) * 0.01
        HB = acts_B[i] if i < len(acts_B) else np.random.randn(len(X_cal), d_b).astype(np.float32) * 0.01
        
        C = HA.T @ HB
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        s_init = S[:d_min] / (S[0] + 1e-10)
        svd_list.append((U[:, :d_min], Vt[:d_min, :]))
        s_inits.append(s_init)
    
    return svd_list, s_inits, linears_A, linears_B


def merge_general(model_A, model_B, svd_list, s_vec, alphas,
                  linears_A, linears_B):
    """Merge two models by mapping Linear layer weights.
    Handles BatchNorm by copying from model A (since merged activations are in A's space).
    """
    n_hidden = len(svd_list)
    n_linear = len(linears_A)
    
    # Build mappings
    mappings = []
    offset = 0
    for i in range(n_hidden):
        U, Vt = svd_list[i]
        d = U.shape[1]
        s = s_vec[offset:offset + d]
        offset += d
        mappings.append(U @ np.diag(s) @ Vt)
    
    # Get linear weights
    WA = [(l[1].weight.detach().numpy(), l[1].bias.detach().numpy()) for l in linears_A]
    WB = [(l[1].weight.detach().numpy(), l[1].bias.detach().numpy()) for l in linears_B]
    
    # Merge linear layers
    merged_linear_params = []
    for li in range(n_linear):
        a = np.clip(alphas[li] if li < len(alphas) else 0.5, 0.05, 0.95)
        wa, ba = WA[li]
        wb, bb = WB[li]
        
        if li == 0:
            wbm = mappings[0] @ wb; bbm = mappings[0] @ bb
        elif li < n_hidden:
            wbm = mappings[li] @ wb @ mappings[li-1].T; bbm = mappings[li] @ bb
        else:
            wbm = wb @ mappings[-1].T; bbm = bb
        
        merged_linear_params.append((a*wa + (1-a)*wbm, a*ba + (1-a)*bbm))
    
    # Build merged model (clone A's architecture)
    import copy
    merged = copy.deepcopy(model_A)
    
    # Set merged linear weights
    linear_idx = 0
    for m in merged.net:
        if isinstance(m, nn.Linear):
            w, b = merged_linear_params[linear_idx]
            with torch.no_grad():
                m.weight.copy_(torch.tensor(w, dtype=torch.float32))
                m.bias.copy_(torch.tensor(b, dtype=torch.float32))
            linear_idx += 1
    
    return merged


def run_pcma_general(model_A, model_B, X_cal, X_val, y_val, seed=42,
                     maxiter=30, popsize=12):
    """Generalized PCMA for any MLP-like architecture."""
    import cma
    
    svd_list, s_inits, linears_A, linears_B = compute_svd_general(model_A, model_B, X_cal)
    s0 = np.concatenate(s_inits) if s_inits else np.array([])
    nd_s = len(s0)
    n_linear = len(linears_A)
    
    x0 = np.concatenate([s0, [0.5] * n_linear])
    nd = len(x0)
    
    def fitness(x):
        s, al = x[:nd_s], x[nd_s:].tolist()
        try:
            m = merge_general(model_A, model_B, svd_list, s, al, linears_A, linears_B)
            return -ev(m, X_val, y_val)
        except:
            return 1.0
    
    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.3, {
        'maxiter': maxiter, 'popsize': popsize, 'seed': seed,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [[-3]*nd_s + [0.1]*n_linear, [3]*nd_s + [0.9]*n_linear],
        'CMA_diagonal': nd > 80,
    })
    while not es.stop():
        sols = es.ask()
        es.tell(sols, [fitness(np.array(s)) for s in sols])
    
    best = np.array(es.result.xbest)
    s_opt, alphas = best[:nd_s], best[nd_s:].tolist()
    
    merged = merge_general(model_A, model_B, svd_list, s_opt, alphas, linears_A, linears_B)
    
    # Procrustes baseline
    proc = merge_general(model_A, model_B, svd_list, s0, [0.5]*n_linear, linears_A, linears_B)
    
    return merged, proc, alphas, nd


# ─── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 75)
    print("  E09 [Hypothesis] PCMA on non-standard architectures")
    print("=" * 75)
    
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_data()
    X_cal = X_tr[:3000]
    
    tests = []
    
    # ─── Arch 1: MLP + BatchNorm (ReLU) ──────────────────────────────
    print(f"\n{'━' * 75}")
    print("  ARCH 1: MLP + BatchNorm (ReLU)")
    print(f"{'━' * 75}")
    
    for archA, archB, label in [
        ([784,128,64,10], [784,256,128,10], "3L_BN"),
        ([784,128,64,32,10], [784,256,128,64,10], "4L_BN"),
    ]:
        mA = train(MLP_BN(archA, 'relu'), X_tr, y_tr)
        mB = train(MLP_BN(archB, 'relu'), X_tr, y_tr)
        accA, accB = ev(mA, X_te, y_te), ev(mB, X_te, y_te)
        bp = max(accA, accB)
        
        merged, proc, alphas, nd = run_pcma_general(mA, mB, X_cal, X_val, y_val)
        acc_m = ev(merged, X_te, y_te)
        acc_p = ev(proc, X_te, y_te)
        ret = acc_m / bp if bp > 0 else 0
        ret_p = acc_p / bp if bp > 0 else 0
        
        m = "✅" if ret >= 0.60 else "❌"
        print(f"  {label}: A={accA:.3f} B={accB:.3f} Proc={ret_p:.3f} PCMA={ret:.3f} {m} dims={nd}")
        tests.append({"arch": label, "accA": accA, "accB": accB, 
                       "proc_ret": round(ret_p,4), "pcma_ret": round(ret,4), "dims": nd})
    
    # ─── Arch 2: MLP + LeakyReLU ─────────────────────────────────────
    print(f"\n{'━' * 75}")
    print("  ARCH 2: MLP + LeakyReLU")
    print(f"{'━' * 75}")
    
    for archA, archB, label in [
        ([784,128,64,10], [784,256,128,10], "3L_Leaky"),
        ([784,128,64,32,10], [784,256,128,64,10], "4L_Leaky"),
    ]:
        mA = train(MLP_BN(archA, 'leaky'), X_tr, y_tr)
        mB = train(MLP_BN(archB, 'leaky'), X_tr, y_tr)
        accA, accB = ev(mA, X_te, y_te), ev(mB, X_te, y_te)
        bp = max(accA, accB)
        
        merged, proc, alphas, nd = run_pcma_general(mA, mB, X_cal, X_val, y_val)
        acc_m = ev(merged, X_te, y_te)
        acc_p = ev(proc, X_te, y_te)
        ret = acc_m / bp if bp > 0 else 0
        ret_p = acc_p / bp if bp > 0 else 0
        
        m = "✅" if ret >= 0.60 else "❌"
        print(f"  {label}: A={accA:.3f} B={accB:.3f} Proc={ret_p:.3f} PCMA={ret:.3f} {m} dims={nd}")
        tests.append({"arch": label, "accA": accA, "accB": accB,
                       "proc_ret": round(ret_p,4), "pcma_ret": round(ret,4), "dims": nd})
    
    # ─── Arch 3: MLP + GELU ──────────────────────────────────────────
    print(f"\n{'━' * 75}")
    print("  ARCH 3: MLP + GELU")
    print(f"{'━' * 75}")
    
    for archA, archB, label in [
        ([784,128,64,10], [784,256,128,10], "3L_GELU"),
        ([784,128,64,32,10], [784,256,128,64,10], "4L_GELU"),
    ]:
        mA = train(MLP_BN(archA, 'gelu'), X_tr, y_tr)
        mB = train(MLP_BN(archB, 'gelu'), X_tr, y_tr)
        accA, accB = ev(mA, X_te, y_te), ev(mB, X_te, y_te)
        bp = max(accA, accB)
        
        merged, proc, alphas, nd = run_pcma_general(mA, mB, X_cal, X_val, y_val)
        acc_m = ev(merged, X_te, y_te)
        acc_p = ev(proc, X_te, y_te)
        ret = acc_m / bp if bp > 0 else 0
        ret_p = acc_p / bp if bp > 0 else 0
        
        m = "✅" if ret >= 0.60 else "❌"
        print(f"  {label}: A={accA:.3f} B={accB:.3f} Proc={ret_p:.3f} PCMA={ret:.3f} {m} dims={nd}")
        tests.append({"arch": label, "accA": accA, "accB": accB,
                       "proc_ret": round(ret_p,4), "pcma_ret": round(ret,4), "dims": nd})
    
    # ─── Arch 4: Asymmetric widths ───────────────────────────────────
    print(f"\n{'━' * 75}")
    print("  ARCH 4: Asymmetric widths (non-geometric)")
    print(f"{'━' * 75}")
    
    for archA, archB, label in [
        ([784,100,50,10], [784,300,80,10], "3L_asym"),
        ([784,200,30,60,10], [784,150,90,40,10], "4L_asym_mixed"),
    ]:
        mA = train(MLP_BN(archA, 'relu'), X_tr, y_tr)
        mB = train(MLP_BN(archB, 'relu'), X_tr, y_tr)
        accA, accB = ev(mA, X_te, y_te), ev(mB, X_te, y_te)
        bp = max(accA, accB)
        
        merged, proc, alphas, nd = run_pcma_general(mA, mB, X_cal, X_val, y_val)
        acc_m = ev(merged, X_te, y_te)
        acc_p = ev(proc, X_te, y_te)
        ret = acc_m / bp if bp > 0 else 0
        ret_p = acc_p / bp if bp > 0 else 0
        
        m = "✅" if ret >= 0.60 else "❌"
        print(f"  {label}: A={accA:.3f} B={accB:.3f} Proc={ret_p:.3f} PCMA={ret:.3f} {m} dims={nd}")
        tests.append({"arch": label, "accA": accA, "accB": accB,
                       "proc_ret": round(ret_p,4), "pcma_ret": round(ret,4), "dims": nd})
    
    # ─── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 75}")
    print("  SUMMARY")
    print(f"{'=' * 75}")
    print(f"\n  {'Arch':<16s} {'Proc':>7s} {'PCMA':>7s} {'Δ':>7s} {'Status':>7s}")
    print(f"  {'─'*50}")
    for t in tests:
        delta = t['pcma_ret'] - t['proc_ret']
        st = "✅" if t['pcma_ret'] >= 0.60 else "❌"
        print(f"  {t['arch']:<16s} {t['proc_ret']:>7.3f} {t['pcma_ret']:>7.3f} {delta:>+7.3f} {st:>7s}")
    
    avg_pcma = np.mean([t['pcma_ret'] for t in tests])
    avg_proc = np.mean([t['proc_ret'] for t in tests])
    n_pass = sum(1 for t in tests if t['pcma_ret'] >= 0.60)
    
    print(f"\n  Avg Proc: {avg_proc:.4f}  Avg PCMA: {avg_pcma:.4f}")
    print(f"  Pass: {n_pass}/{len(tests)}")
    print(f"  Time: {elapsed:.1f}s")
    
    with open("results_e09.json", "w") as f:
        json.dump(tests, f, indent=2)


if __name__ == "__main__":
    main()
