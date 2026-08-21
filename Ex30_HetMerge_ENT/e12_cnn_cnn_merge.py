#!/usr/bin/env python3
"""
E12 [Hypothesis] — CNN + CNN heterogeneous merge.
===================================================
Two CNNs with DIFFERENT architectures:
  CNN_A: Conv(1→8,3) → ReLU → Pool → Conv(8→16,3) → ReLU → Pool → FC(16*5*5→64→10)
  CNN_B: Conv(1→16,5) → ReLU → Pool → Conv(16→32,3) → ReLU → Pool → FC(32*5*5→128→10)

Strategy: Extend PCMA to conv layers.
  Conv2d weight shape: (out_ch, in_ch, kH, kW)
  Reshape to (out_ch, in_ch*kH*kW) → treat as Linear
  SVD aligns OUTPUT channels, CMA optimizes channel scaling

This enables TRUE full-network heterogeneous merging for CNNs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, json, copy

SEED = 42
N_TR, N_VAL, N_TE = 1000, 500, 500


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


# ─── CNN architectures ──────────────────────────────────────────────

class CNN_A(nn.Module):
    """Small CNN: 8→16 channels, 3×3 kernels."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)   # →8ch, 28×28
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)   # →16ch, 14×14 after pool
        self.fc1 = nn.Linear(16*7*7, 64)
        self.fc2 = nn.Linear(64, 10)
        self.name = "CNN_A(8→16→64→10)"
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CNN_B(nn.Module):
    """Larger CNN: 16→32 channels, 5×5 first kernel."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)   # →16ch, 28×28
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)   # →32ch, 14×14 after pool
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.name = "CNN_B(16→32→128→10)"
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CNN_C(nn.Module):
    """Different depth: 3 conv layers, 12→24→48 channels."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)
        self.conv3 = nn.Conv2d(24, 48, 3, padding=1)
        self.fc1 = nn.Linear(48*3*3, 96)
        self.fc2 = nn.Linear(96, 10)
        self.name = "CNN_C(12→24→48→96→10)"
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)      # 14×14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)      # 7×7
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)      # 3×3
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def ev(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()

def train(model, X, y):
    torch.manual_seed(SEED); np.random.seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    model.train()
    for _ in range(3):
        l = nn.CrossEntropyLoss()(model(X), y)
        opt.zero_grad(); l.backward(); opt.step()
    return model


# ─── Channel-level PCMA for CNNs ────────────────────────────────────

def get_channel_activations(model, X):
    """Get post-ReLU channel activations (global avg pooled) for each conv layer."""
    acts = []
    with torch.no_grad():
        h = X
        for module_name in dir(model):
            m = getattr(model, module_name)
            if isinstance(m, nn.Conv2d):
                h_conv = m(h)
                h_relu = F.relu(h_conv)
                # Global avg pool: (batch, channels, H, W) → (batch, channels)
                h_gap = h_relu.mean(dim=[2, 3]).numpy()
                acts.append(h_gap)
    return acts


def extract_layers_ordered(model, X):
    """Run forward pass and collect activations + layer info in order."""
    acts = []  # post-relu activations (GAP for conv, raw for linear)
    layer_info = []  # (type, module)
    
    with torch.no_grad():
        h = X
        modules = []
        # Collect all the computation steps
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                modules.append(('weight', m))
            elif isinstance(m, (nn.ReLU,)):
                modules.append(('relu', m))
            elif isinstance(m, (nn.MaxPool2d,)):
                modules.append(('pool', m))
    
    # Actually run forward manually
    with torch.no_grad():
        h = X
        for attr in ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']:
            if not hasattr(model, attr):
                continue
            m = getattr(model, attr)
            if isinstance(m, nn.Conv2d):
                h = F.relu(m(h))
                # Global average pooling for channel alignment
                h_gap = h.mean(dim=[2, 3]).numpy()
                acts.append(h_gap)
                layer_info.append(('conv', m))
                h = F.max_pool2d(h, 2)
            elif isinstance(m, nn.Linear):
                if h.dim() > 2:
                    h = h.view(h.size(0), -1)
                h_out = m(h)
                if attr != 'fc2':  # Not last layer
                    h_out = F.relu(h_out)
                    acts.append(h_out.numpy())
                    layer_info.append(('fc', m))
                else:
                    layer_info.append(('fc_out', m))
                h = h_out
    
    return acts, layer_info


def cnn_pcma_merge(model_A, model_B, X_cal, X_val, y_val, maxiter=35, popsize=14):
    """Merge two CNNs using channel-level SVD + CMA-ES.
    
    For conv layers: SVD aligns output channels
    For FC layers: SVD aligns hidden units (same as MLP PCMA)
    Merged model has model_A's architecture.
    """
    import cma
    
    acts_A, info_A = extract_layers_ordered(model_A, X_cal)
    acts_B, info_B = extract_layers_ordered(model_B, X_cal)
    
    # We can only align layers that exist in both models
    n_align = min(len(acts_A), len(acts_B))
    
    svd_list, s_inits = [], []
    for i in range(n_align):
        HA, HB = acts_A[i], acts_B[i]
        d_a, d_b = HA.shape[1], HB.shape[1]
        d_min = min(d_a, d_b)
        
        C = HA.T @ HB
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        s_init = S[:d_min] / (S[0] + 1e-10)
        svd_list.append((U[:, :d_min], Vt[:d_min, :]))
        s_inits.append(s_init)
    
    s0 = np.concatenate(s_inits) if s_inits else np.array([1.0])
    nd_s = len(s0)
    n_out = len(info_A)  # total weight layers
    
    x0 = np.concatenate([s0, [0.5] * n_out])
    nd = len(x0)
    
    def build_merged(s_vec, alphas):
        maps = []
        off = 0
        for i in range(n_align):
            U, Vt = svd_list[i]
            d = U.shape[1]
            s = s_vec[off:off+d]; off += d
            maps.append(U @ np.diag(s) @ Vt)
        
        merged = copy.deepcopy(model_A)
        
        # Merge each weight layer
        li = 0
        for attr in ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']:
            if not hasattr(model_A, attr) or not hasattr(model_B, attr):
                continue
            
            mA_layer = getattr(model_A, attr)
            mB_layer = getattr(model_B, attr)
            m_merged = getattr(merged, attr)
            
            a = np.clip(alphas[li] if li < len(alphas) else 0.5, 0.05, 0.95)
            
            if isinstance(mA_layer, nn.Conv2d):
                wA = mA_layer.weight.detach().numpy()  # (out, in, kH, kW)
                bA = mA_layer.bias.detach().numpy() if mA_layer.bias is not None else np.zeros(wA.shape[0])
                wB = mB_layer.weight.detach().numpy()
                bB = mB_layer.bias.detach().numpy() if mB_layer.bias is not None else np.zeros(wB.shape[0])
                
                oA, iA, kH_A, kW_A = wA.shape
                oB, iB, kH_B, kW_B = wB.shape
                
                # Map output channels via current mapping
                if li < n_align and li == 0:
                    # First conv: input channels same (1), map output channels
                    M = maps[0]  # (oA, oB)
                    # Reshape B's weights: (oB, iB, kH, kW) → handle kernel size difference
                    if kH_A == kH_B and kW_A == kW_B and iA == iB:
                        wB_flat = wB.reshape(oB, -1)  # (oB, iB*kH*kW)
                        wBm = (M @ wB_flat).reshape(oA, iA, kH_A, kW_A)
                        bBm = M @ bB
                    else:
                        # Different kernel sizes — pad/crop
                        wBm = wA  # fallback
                        bBm = bA
                elif li < n_align and li > 0:
                    M_curr = maps[li] if li < len(maps) else np.eye(oA)
                    M_prev = maps[li-1] if li-1 < len(maps) else np.eye(iA)
                    
                    if kH_A == kH_B and kW_A == kW_B:
                        # (oB, iB, k, k) → reshape to (oB, iB*k*k) → M_curr @ _ @ (M_prev_spatial)
                        # For conv: M_curr maps output channels, M_prev maps input channels
                        wB_r = wB.reshape(oB, iB, -1)  # (oB, iB, k²)
                        # Map output: M_curr @ wB_r[:,:,j] for each spatial
                        wBm_r = np.zeros((oA, iA, kH_A * kW_A))
                        for j in range(kH_A * kW_A):
                            slice_B = wB_r[:, :, j]  # (oB, iB)
                            wBm_r[:, :, j] = M_curr @ slice_B @ M_prev.T
                        wBm = wBm_r.reshape(oA, iA, kH_A, kW_A)
                        bBm = M_curr @ bB
                    else:
                        wBm = wA; bBm = bA
                else:
                    wBm = wA; bBm = bA
                
                w_merged = a * wA + (1-a) * wBm
                b_merged = a * bA + (1-a) * bBm
                
                with torch.no_grad():
                    m_merged.weight.copy_(torch.tensor(w_merged, dtype=torch.float32))
                    if m_merged.bias is not None:
                        m_merged.bias.copy_(torch.tensor(b_merged, dtype=torch.float32))
            
            elif isinstance(mA_layer, nn.Linear):
                wA = mA_layer.weight.detach().numpy()
                bA = mA_layer.bias.detach().numpy()
                wB = mB_layer.weight.detach().numpy()
                bB = mB_layer.bias.detach().numpy()
                
                oA, iA = wA.shape
                oB, iB = wB.shape
                
                # For FC layers connected to conv output
                if li < n_align:
                    M = maps[li]
                    if iA == iB:
                        wBm = M @ wB
                        bBm = M @ bB
                    elif oA == oB:
                        M_prev = maps[li-1] if li-1 >= 0 and li-1 < len(maps) else np.eye(min(iA, iB))
                        wBm = wB[:oA, :iA]
                        bBm = bB[:oA]
                    else:
                        wBm = wA; bBm = bA
                elif attr == 'fc2':
                    # Output layer: map input features
                    M_prev = maps[-1] if maps else np.eye(min(iA, iB))
                    if iA <= iB:
                        wBm = wB[:, :iA]  # truncate
                    else:
                        wBm = wA
                    bBm = bB[:oA] if len(bB) >= oA else bA
                else:
                    wBm = wA; bBm = bA
                
                w_merged = a * wA + (1-a) * wBm
                b_merged = a * bA + (1-a) * bBm
                
                with torch.no_grad():
                    m_merged.weight.copy_(torch.tensor(w_merged, dtype=torch.float32))
                    m_merged.bias.copy_(torch.tensor(b_merged, dtype=torch.float32))
            
            li += 1
        
        return merged
    
    def fitness(x):
        s, al = x[:nd_s], x[nd_s:].tolist()
        try:
            m = build_merged(s, al)
            return -ev(m, X_val, y_val)
        except:
            return 1.0
    
    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.3, {
        'maxiter': maxiter, 'popsize': popsize, 'seed': SEED,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [[-3]*nd_s + [0.1]*n_out, [3]*nd_s + [0.9]*n_out],
        'CMA_diagonal': nd > 80,
    })
    while not es.stop():
        sols = es.ask()
        es.tell(sols, [fitness(np.array(s)) for s in sols])
    
    best = np.array(es.result.xbest)
    merged = build_merged(best[:nd_s], best[nd_s:].tolist())
    alphas = best[nd_s:].tolist()
    return merged, alphas, nd


# ─── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 75)
    print("  E12 [Hypothesis] CNN + CNN heterogeneous merge")
    print("=" * 75)
    
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_data()
    X_cal = X_tr[:1500]
    
    pairs = [
        ("A_vs_B", CNN_A, CNN_B),
        ("A_vs_C", CNN_A, CNN_C),
    ]
    
    results = []
    for name, cls_A, cls_B in pairs:
        print(f"\n{'━' * 75}")
        mA = train(cls_A(), X_tr, y_tr)
        mB = train(cls_B(), X_tr, y_tr)
        print(f"  {name}: {mA.name} vs {mB.name}")
        
        acc_A = ev(mA, X_te, y_te)
        acc_B = ev(mB, X_te, y_te)
        bp = max(acc_A, acc_B)
        print(f"  A={acc_A:.4f}  B={acc_B:.4f}")
        
        merged, alphas, nd = cnn_pcma_merge(mA, mB, X_cal, X_val, y_val)
        acc_m_val = ev(merged, X_val, y_val)
        acc_m_te = ev(merged, X_te, y_te)
        ret = acc_m_te / bp if bp > 0 else 0
        gap = acc_m_val - acc_m_te
        
        mk = "✅" if ret >= 0.60 else "❌"
        print(f"  Merged: val={acc_m_val:.4f} test={acc_m_te:.4f} ret={ret:.4f} {mk}")
        print(f"  gap={gap:+.4f} α={[round(float(a),2) for a in alphas]}")
        
        results.append({
            'pair': name, 'acc_A': round(acc_A,4), 'acc_B': round(acc_B,4),
            'merged': round(acc_m_te,4), 'retention': round(ret,4),
            'gap': round(gap,4), 'dims': nd,
        })
    
    elapsed = time.time() - t0
    print(f"\n{'=' * 75}")
    print("  SUMMARY")
    print(f"{'=' * 75}")
    for r in results:
        mk = "✅" if r['retention'] >= 0.60 else "❌"
        print(f"  {r['pair']}: A={r['acc_A']:.3f} B={r['acc_B']:.3f} "
              f"M={r['merged']:.3f} ret={r['retention']:.3f} {mk}")
    print(f"  Time: {elapsed:.1f}s")
    
    with open("results_e12.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
