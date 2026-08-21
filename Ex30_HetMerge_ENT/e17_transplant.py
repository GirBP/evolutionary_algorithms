#!/usr/bin/env python3
"""
E17 [Reframe] — Neuron transplantation merge.
================================================
Instead of weight interpolation (α·A + (1-α)·B), we CONCATENATE neurons.

Merged = wider network containing BOTH A and B as subnetworks.

Architecture:
  A: [784, 128, 64, 10]  → knows classes 0-4
  B: [784, 256, 128, 10] → knows classes 5-9
  
  Merged: [784, 128+d_min1, 64+d_min2, 10]
  Where d_min = min(d_A, d_B) at each layer (SVD-mapped B neurons)

Weight construction:
  Hidden layer l:
    W_merged = [W_A    ;  0     ]   ← A's neurons, unconnected to B's prev
               [0      ; M·W_B ]   ← B's mapped neurons
  
  Output layer:
    Row c∈A_classes: [W_out_A[c], zeros]
    Row c∈B_classes: [zeros, M·W_out_B[c]]

Then CMA-ES optimizes CROSS-CONNECTIONS (off-diagonal blocks).
"""

import numpy as np
import torch
import torch.nn as nn
import time, json, copy

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


def load_mnist():
    import ssl; ssl._create_default_https_context = ssl._create_unverified_context
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=tf)
    te = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=tf)
    X_tr = torch.stack([tr[i][0] for i in range(len(tr))])
    y_tr = torch.tensor([tr[i][1] for i in range(len(tr))])
    X_te = torch.stack([te[i][0] for i in range(2000)])
    y_te = torch.tensor([te[i][1] for i in range(2000)])
    return X_tr, y_tr, X_te, y_te


class MLP(nn.Module):
    def __init__(s, arch):
        super().__init__()
        layers = []
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i + 1]))
            if i < len(arch) - 2: layers.append(nn.ReLU())
        s.net = nn.Sequential(*layers); s.arch = arch
    def forward(s, x): return s.net(x)


def ev(m, X, y):
    m.eval()
    with torch.no_grad(): return (m(X).argmax(1) == y).float().mean().item()

def pc(m, X, y):
    m.eval()
    with torch.no_grad(): p = m(X).argmax(1)
    return {c: (p[y == c] == c).float().mean().item() if (y == c).sum() > 0 else 0 for c in range(10)}

def train_on(arch, X, y, classes):
    mask = torch.zeros(len(y), dtype=torch.bool)
    for c in classes: mask |= (y == c)
    Xs, ys = X[mask][:5000], y[mask][:5000]
    m = MLP(arch); opt = torch.optim.Adam(m.parameters(), lr=0.003); m.train()
    for _ in range(15):
        l = nn.CrossEntropyLoss()(m(Xs), ys); opt.zero_grad(); l.backward(); opt.step()
    return m


# ─── Neuron transplantation merge ────────────────────────────────────

def transplant_merge(mA, mB, classesA, classesB, X_cal):
    """Create a wider merged model by concatenating neurons from A and B.
    
    Returns: merged model with architecture [784, dA1+dB1_mapped, ..., 10]
    """
    archA, archB = mA.arch, mB.arch
    n_hidden = len(archA) - 2
    
    # SVD for mapping B's representations
    actsA, actsB = [], []
    with torch.no_grad():
        hA, hB = X_cal, X_cal
        for ma, mb in zip(mA.net, mB.net):
            hA, hB = ma(hA), mb(hB)
            if isinstance(ma, nn.ReLU):
                actsA.append(hA.numpy().copy())
                actsB.append(hB.numpy().copy())
    
    svd_list = []
    d_mapped = []  # how many B neurons we transplant per layer
    for i in range(n_hidden):
        dA, dB = archA[i + 1], archB[i + 1]
        d_min = min(dA, dB)
        C = actsA[i].T @ actsB[i]
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        s_norm = S[:d_min] / (S[0] + 1e-10)
        # Mapping: (dA, dB) — maps B's neurons into A's space
        M = U[:, :d_min] @ np.diag(s_norm) @ Vt[:d_min, :]
        svd_list.append(M)
        d_mapped.append(d_min)
    
    # Build merged architecture
    merged_arch = [archA[0]]  # input dim
    for i in range(n_hidden):
        merged_arch.append(archA[i + 1] + d_mapped[i])
    merged_arch.append(10)  # output
    
    print(f"    Merged architecture: {merged_arch}")
    print(f"    A: {archA}, B: {archB}")
    
    WA = [p.detach().numpy() for p in mA.parameters()]
    WB = [p.detach().numpy() for p in mB.parameters()]
    
    # Build weights for each layer
    merged_params = []
    for li in range(n_hidden + 1):
        dA_out = archA[li + 1]
        dA_in = archA[li]
        
        if li < n_hidden:
            dM_out = d_mapped[li]  # B's mapped output neurons
        
        if li == 0:
            # First hidden layer: input is shared (784)
            # A block: (dA_out, 784)
            # B block: (dM_out, 784) — mapped B weights
            wa, ba = WA[0], WA[1]
            wb, bb = WB[0], WB[1]
            wb_mapped = svd_list[0] @ wb  # (dA_out, 784) — but actually (d_min, d_B_in)
            # svd_list[0] is (dA_out, dB_out), wb is (dB_out, 784)
            # So wb_mapped = M @ wb → (dA_out, 784)... no, M is (dA, dB)
            # wb shape: (dB_out, 784), M shape: (dA, dB)
            # M @ wb → (dA, 784)... but we want d_min neurons, not dA
            # Actually, let's just take the first d_min components
            M = svd_list[0]  # (dA, dB)
            wb_m = M @ wb    # (dA, 784) → but we want d_mapped[0] neurons
            bb_m = M @ bb    # (dA,)
            # Take d_mapped neurons from this mapping
            wb_m = wb_m[:d_mapped[0]]
            bb_m = bb_m[:d_mapped[0]]
            
            W = np.vstack([wa, wb_m])  # (dA + d_mapped, 784)
            b = np.concatenate([ba, bb_m])
            
        elif li < n_hidden:
            # Middle hidden: input from previous merged layer, output to next
            prev_dA = archA[li]
            prev_dM = d_mapped[li - 1]
            
            wa, ba = WA[li * 2], WA[li * 2 + 1]  # (dA_out, dA_in)
            wb, bb = WB[li * 2], WB[li * 2 + 1]  # (dB_out, dB_in)
            
            M_curr = svd_list[li]  # (dA_curr, dB_curr)
            M_prev = svd_list[li - 1]  # (dA_prev, dB_prev)
            
            # Map B's weights: M_curr @ W_B @ M_prev^T
            wb_m = M_curr @ wb @ M_prev.T  # (dA_curr, dA_prev)
            bb_m = M_curr @ bb             # (dA_curr,)
            wb_m = wb_m[:d_mapped[li], :d_mapped[li - 1]]
            bb_m = bb_m[:d_mapped[li]]
            
            # Build block matrix:
            # [W_A,     zeros   ]   size: (dA_out, prev_dA + prev_dM)
            # [zeros,   wb_m    ]   size: (d_mapped, prev_dA + prev_dM)
            total_in = prev_dA + prev_dM
            total_out = dA_out + d_mapped[li]
            
            W = np.zeros((total_out, total_in), dtype=np.float32)
            b_vec = np.zeros(total_out, dtype=np.float32)
            
            # A block: top-left
            W[:dA_out, :prev_dA] = wa
            b_vec[:dA_out] = ba
            
            # B block: bottom-right
            W[dA_out:, prev_dA:] = wb_m
            b_vec[dA_out:] = bb_m
            
            b = b_vec
            
        else:
            # Output layer: class-selective
            prev_dA = archA[li]
            prev_dM = d_mapped[-1]
            total_in = prev_dA + prev_dM
            
            wa, ba = WA[li * 2], WA[li * 2 + 1]  # (10, dA_last)
            wb, bb = WB[li * 2], WB[li * 2 + 1]  # (10, dB_last)
            
            M_prev = svd_list[-1]
            wb_m = wb @ M_prev.T  # (10, dA_last)
            wb_m = wb_m[:, :d_mapped[-1]]
            
            W = np.zeros((10, total_in), dtype=np.float32)
            b_vec = np.zeros(10, dtype=np.float32)
            
            for c in range(10):
                if c in classesA:
                    W[c, :prev_dA] = wa[c]
                    b_vec[c] = ba[c]
                elif c in classesB:
                    W[c, prev_dA:] = wb_m[c]
                    b_vec[c] = bb[c]
                else:
                    # Unknown class: average
                    W[c, :prev_dA] = 0.5 * wa[c]
                    W[c, prev_dA:] = 0.5 * wb_m[c]
                    b_vec[c] = 0.5 * (ba[c] + bb[c])
            
            b = b_vec
        
        merged_params.append(W)
        merged_params.append(b)
    
    # Build model
    merged = MLP(merged_arch)
    with torch.no_grad():
        for p, v in zip(merged.parameters(), merged_params):
            p.copy_(torch.tensor(v, dtype=torch.float32))
    
    return merged, merged_arch


def cma_cross_connections(merged, merged_arch, X_val, y_val, maxiter=25, popsize=12):
    """Optimize cross-connections (off-diagonal blocks) with CMA-ES.
    
    The diagonal blocks (A→A, B→B) stay fixed.
    CMA optimizes a small scale factor for each off-diagonal block.
    """
    import cma
    
    n_hidden = len(merged_arch) - 2
    
    # For each hidden layer, we have 2 off-diagonal blocks
    # scale_AB[l] = how much A's neurons feed into B's path
    # scale_BA[l] = how much B's neurons feed into A's path
    # Plus output cross-connections
    
    n_cross = 2 * (n_hidden - 1) + 2  # hidden cross + output cross
    x0 = [0.0] * n_cross  # start with zero cross-connections
    
    base_params = [p.detach().numpy().copy() for p in merged.parameters()]
    
    def build_with_cross(x):
        m = copy.deepcopy(merged)
        params = list(m.parameters())
        xi = 0
        
        for li in range(1, n_hidden):  # skip first layer (no cross possible)
            # Get block sizes
            prev_dA = merged_arch[li] // 2  # approximate
            curr_dA = merged_arch[li + 1] // 2
            w = params[li * 2]
            
            # Scale for off-diagonal blocks
            s_ab = x[xi]; xi += 1  # A→B cross
            s_ba = x[xi]; xi += 1  # B→A cross
            
            with torch.no_grad():
                # Add cross-connections as scaled identity-like matrices
                w_np = base_params[li * 2].copy()
                d_cross = min(prev_dA, curr_dA, w.shape[0] - curr_dA, w.shape[1] - prev_dA)
                if d_cross > 0:
                    # A→B: top-right block
                    for j in range(min(d_cross, w_np.shape[0] - curr_dA)):
                        if prev_dA + j < w_np.shape[1] and j < w_np.shape[0]:
                            w_np[j, prev_dA + j] = s_ab * 0.1
                    # B→A: bottom-left block  
                    for j in range(min(d_cross, w_np.shape[1])):
                        if curr_dA + j < w_np.shape[0] and j < w_np.shape[1]:
                            w_np[curr_dA + j, j] = s_ba * 0.1
                w.copy_(torch.tensor(w_np, dtype=torch.float32))
        
        return m
    
    def fitness(x):
        try:
            m = build_with_cross(x)
            return -ev(m, X_val, y_val)
        except:
            return 1.0
    
    if n_cross > 0:
        es = cma.CMAEvolutionStrategy(x0, 0.5, {
            'maxiter': maxiter, 'popsize': popsize, 'seed': SEED,
            'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
            'bounds': [[-2] * n_cross, [2] * n_cross],
        })
        while not es.stop():
            sols = es.ask()
            es.tell(sols, [fitness(np.array(s)) for s in sols])
        best = np.array(es.result.xbest)
        return build_with_cross(best)
    return merged


# ─── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 75)
    print("  E17 [Reframe] Neuron transplantation merge")
    print("  A trains: 0-4, B trains: 5-9")
    print("=" * 75)
    
    X_tr, y_tr, X_te, y_te = load_mnist()
    idx = torch.randperm(len(X_tr), generator=torch.Generator().manual_seed(0))
    X_val, y_val = X_tr[idx[50000:55000]], y_tr[idx[50000:55000]]
    X_cal = X_tr[idx[:3000]]
    
    scenarios = [
        ("Disjoint 0-4 vs 5-9", [784,128,64,10], [784,256,128,10], list(range(5)), list(range(5,10))),
        ("Overlap 0-6 vs 3-9", [784,128,64,10], [784,256,128,10], list(range(7)), list(range(3,10))),
    ]
    
    for name, archA, archB, clsA, clsB in scenarios:
        print(f"\n{'━'*75}")
        print(f"  {name}")
        print(f"{'━'*75}")
        
        mA = train_on(archA, X_tr, y_tr, clsA)
        mB = train_on(archB, X_tr, y_tr, clsB)
        
        pcA, pcB = pc(mA, X_te, y_te), pc(mB, X_te, y_te)
        print(f"  A: {ev(mA,X_te,y_te):.3f} per-class: {[round(pcA[c],2) for c in range(10)]}")
        print(f"  B: {ev(mB,X_te,y_te):.3f} per-class: {[round(pcB[c],2) for c in range(10)]}")
        
        # Stage 1: Transplant merge
        print(f"\n  Stage 1: Neuron transplantation...")
        merged, m_arch = transplant_merge(mA, mB, clsA, clsB, X_cal)
        pcM = pc(merged, X_te, y_te)
        print(f"  Transplant: {ev(merged,X_te,y_te):.3f}")
        print(f"  Per-class: {[round(pcM[c],2) for c in range(10)]}")
        
        # Stage 2: CMA cross-connections
        print(f"\n  Stage 2: CMA cross-connection optimization...")
        refined = cma_cross_connections(merged, m_arch, X_val, y_val)
        pcR = pc(refined, X_te, y_te)
        print(f"  Refined: {ev(refined,X_te,y_te):.3f}")
        print(f"  Per-class: {[round(pcR[c],2) for c in range(10)]}")
        
        # Per-class detail
        print(f"\n  {'Class':>6s} {'Parent':>7s} {'Transplant':>11s} {'Refined':>8s}")
        a_trans = np.mean([pcM[c] for c in clsA])
        b_trans = np.mean([pcM[c] for c in clsB])
        a_ref = np.mean([pcR[c] for c in clsA])
        b_ref = np.mean([pcR[c] for c in clsB])
        
        for c in range(10):
            bp = max(pcA[c], pcB[c])
            src = "A" if pcA[c] >= pcB[c] else "B"
            mk_t = "✅" if pcM[c] >= 0.5*bp else "❌"
            mk_r = "✅" if pcR[c] >= 0.5*bp else "❌"
            print(f"  {c:>6d} {bp:>7.3f} {pcM[c]:>11.3f} {mk_t} {pcR[c]:>6.3f} {mk_r}  ({src})")
        
        print(f"\n  A-classes: trans={a_trans:.3f} refined={a_ref:.3f}")
        print(f"  B-classes: trans={b_trans:.3f} refined={b_ref:.3f}")
        bal = min(a_ref, b_ref) / (max(a_ref, b_ref) + 1e-10)
        print(f"  Balance: {bal:.3f}")
        print(f"  vs E14 best (S2 ensemble): A=0.37, B=0.66, bal=0.56")
        print(f"  vs E15 best (grad-select):  A=0.75, B=0.46, bal=0.62")
    
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
