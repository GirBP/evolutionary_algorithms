#!/usr/bin/env python3
"""
E15 — Class-aware selective merge via neuron importance scoring.
=================================================================
KEY IDEA (from user): Use gradient/activation-based importance to
identify which neurons serve which classes, then selectively merge.

Approach:
  1. Class-importance scoring: for each hidden neuron, compute its
     importance for each class (activation magnitude per class)
  2. Class assignment: each neuron "belongs" to the class it serves most
  3. Selective output layer: row c from the model that trained on class c
  4. Selective hidden layers: neuron-level α based on class ownership
  5. CMA-ES optimizes only the scaling factors (directions from SVD)

This should solve E13/E14's problem: B-knowledge is NOT lost because
B's class-specific neurons are explicitly preserved.
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
        for i in range(len(arch)-1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i < len(arch)-2: layers.append(nn.ReLU())
        s.net = nn.Sequential(*layers); s.arch = arch
    def forward(s, x): return s.net(x)


def ev(m, X, y):
    m.eval()
    with torch.no_grad(): return (m(X).argmax(1)==y).float().mean().item()

def per_class(m, X, y):
    m.eval()
    with torch.no_grad(): preds = m(X).argmax(1)
    return {c: (preds[y==c]==c).float().mean().item() if (y==c).sum()>0 else 0 for c in range(10)}

def train_on(arch, X, y, classes):
    mask = torch.zeros(len(y), dtype=torch.bool)
    for c in classes: mask |= (y==c)
    Xs, ys = X[mask][:5000], y[mask][:5000]
    m = MLP(arch); opt = torch.optim.Adam(m.parameters(), lr=0.003); m.train()
    for _ in range(15):
        l = nn.CrossEntropyLoss()(m(Xs), ys); opt.zero_grad(); l.backward(); opt.step()
    return m


# ─── Neuron importance scoring ───────────────────────────────────────

def compute_class_importance(model, X, y, n_classes=10):
    """For each hidden neuron, compute importance per class.
    
    Method: Mean absolute activation on class-c samples.
    Returns: list of (n_neurons_l, n_classes) arrays, one per hidden layer.
    """
    model.eval()
    arch = model.arch
    n_hidden = len(arch) - 2
    
    # Collect per-class activations
    importance_per_layer = []
    
    for layer_idx in range(n_hidden):
        n_neurons = arch[layer_idx + 1]
        imp = np.zeros((n_neurons, n_classes))
        
        with torch.no_grad():
            h = X
            for i, m in enumerate(model.net):
                h = m(h)
                if isinstance(m, nn.ReLU):
                    current_layer = sum(1 for j in range(i+1) if isinstance(model.net[j], nn.ReLU)) - 1
                    if current_layer == layer_idx:
                        acts = h.numpy()  # (N, n_neurons)
                        for c in range(n_classes):
                            mask = (y == c).numpy()
                            if mask.sum() > 0:
                                imp[:, c] = np.abs(acts[mask]).mean(axis=0)
                        break
        
        importance_per_layer.append(imp)
    
    return importance_per_layer


def compute_gradient_importance(model, X, y, n_classes=10):
    """Gradient-based importance: |∂L_c/∂h_i| averaged over class-c samples.
    More precise than activation magnitude.
    """
    model.eval()
    arch = model.arch
    n_hidden = len(arch) - 2
    
    importance_per_layer = []
    
    for layer_idx in range(n_hidden):
        n_neurons = arch[layer_idx + 1]
        imp = np.zeros((n_neurons, n_classes))
        
        for c in range(n_classes):
            mask = (y == c)
            if mask.sum() == 0:
                continue
            Xc = X[mask][:200]  # limit for speed
            
            # Forward with gradient tracking on activations
            model.zero_grad()
            h = Xc
            target_act = None
            for i, m in enumerate(model.net):
                h = m(h)
                if isinstance(m, nn.ReLU):
                    current_layer = sum(1 for j in range(i+1) if isinstance(model.net[j], nn.ReLU)) - 1
                    if current_layer == layer_idx:
                        h = h.detach().requires_grad_(True)
                        target_act = h
                        # Continue forward
                        for m2 in list(model.net)[i+1:]:
                            h = m2(h)
                        break
            
            if target_act is None:
                continue
            
            # Gradient of class-c logit w.r.t. hidden activations
            logits = h
            class_logits = logits[:, c].sum()
            class_logits.backward()
            
            # Importance = average |gradient| over class-c samples
            imp[:, c] = target_act.grad.abs().mean(dim=0).numpy()
        
        importance_per_layer.append(imp)
    
    return importance_per_layer


# ─── Class-selective merge ───────────────────────────────────────────

def class_selective_merge(mA, mB, X_cal, y_cal, classesA, classesB,
                          X_val, y_val, use_gradient=True):
    """Merge using class-specific neuron selection.
    
    1. Compute importance of each neuron for each class (in both models)
    2. For output layer: take row c from the expert that trained on class c
    3. For hidden layers: per-neuron α based on which classes the neuron serves
    4. CMA-ES refines the SVD scaling factors
    """
    import cma
    
    archA, archB = mA.arch, mB.arch
    n_hidden = len(archA) - 2
    
    # Step 1: Compute importance
    if use_gradient:
        impA = compute_gradient_importance(mA, X_cal, y_cal)
        impB = compute_gradient_importance(mB, X_cal, y_cal)
    else:
        impA = compute_class_importance(mA, X_cal, y_cal)
        impB = compute_class_importance(mB, X_cal, y_cal)
    
    # Step 2: SVD alignment (same as PCMA)
    actsA, actsB = [], []
    with torch.no_grad():
        hA, hB = X_cal, X_cal
        for ma, mb in zip(mA.net, mB.net):
            hA, hB = ma(hA), mb(hB)
            if isinstance(ma, nn.ReLU):
                actsA.append(hA.numpy().copy())
                actsB.append(hB.numpy().copy())
    
    svd_list, s_inits = [], []
    for i in range(n_hidden):
        dmin = min(archA[i+1], archB[i+1])
        C = actsA[i].T @ actsB[i]
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        s_init = S[:dmin] / (S[0]+1e-10)
        svd_list.append((U[:,:dmin], Vt[:dmin,:]))
        s_inits.append(s_init)
    
    # Step 3: Compute per-neuron α based on class importance
    # For neuron i in layer l: 
    #   α_i = 1.0 if neuron mainly serves A-classes
    #   α_i = 0.0 if neuron mainly serves B-classes
    #   α_i = 0.5 if ambiguous
    
    per_neuron_alphas = []
    for l in range(n_hidden):
        n_neurons = archA[l+1]
        alphas_l = np.zeros(n_neurons)
        
        for i in range(n_neurons):
            # How much does neuron i serve A-classes vs B-classes?
            imp_for_A = sum(impA[l][i, c] for c in classesA)
            imp_for_B = sum(impA[l][i, c] for c in classesA)  # importance in A's model
            
            # Also check B-model's neuron importance (mapped)
            # For now, simple heuristic: if neuron mainly fires for A-classes → α=1
            total_imp = sum(impA[l][i, c] for c in range(10))
            if total_imp > 0:
                a_ratio = sum(impA[l][i, c] for c in classesA) / total_imp
            else:
                a_ratio = 0.5
            
            alphas_l[i] = a_ratio  # 1.0 = fully A, 0.0 = fully B
        
        per_neuron_alphas.append(alphas_l)
        print(f"    Layer {l}: avg_α={alphas_l.mean():.3f} "
              f"A-dominant={sum(alphas_l > 0.6)}/{n_neurons} "
              f"B-dominant={sum(alphas_l < 0.4)}/{n_neurons}")
    
    # Step 4: Build merge function with per-neuron α
    WA = [p.detach().numpy() for p in mA.parameters()]
    WB = [p.detach().numpy() for p in mB.parameters()]
    s0 = np.concatenate(s_inits)
    nd_s = len(s0)
    
    def build_merged(s_vec):
        maps, off = [], 0
        for i in range(n_hidden):
            U, Vt = svd_list[i]; d = U.shape[1]
            maps.append(U @ np.diag(s_vec[off:off+d]) @ Vt); off += d
        
        params = []
        n_lay = n_hidden + 1
        for li in range(n_lay):
            wa, ba = WA[li*2], WA[li*2+1]
            wb, bb = WB[li*2], WB[li*2+1]
            
            # Map B weights through SVD
            if li == 0: wm = maps[0]@wb; bm = maps[0]@bb
            elif li < n_hidden: wm = maps[li]@wb@maps[li-1].T; bm = maps[li]@bb
            else: wm = wb@maps[-1].T; bm = bb
            
            if li < n_hidden:
                # Per-neuron α for hidden layers
                alpha_vec = per_neuron_alphas[li]  # (n_neurons,)
                # w shape: (out_neurons, in_features)
                # Apply per-output-neuron alpha
                alpha_w = alpha_vec[:, np.newaxis]  # (out, 1)
                w_merged = alpha_w * wa + (1 - alpha_w) * wm
                b_merged = alpha_vec * ba + (1 - alpha_vec) * bm
            else:
                # OUTPUT LAYER: per-class selection!
                w_merged = np.zeros_like(wa)
                b_merged = np.zeros_like(ba)
                for c in range(10):
                    if c in classesA:
                        w_merged[c] = wa[c]
                        b_merged[c] = ba[c]
                    elif c in classesB:
                        w_merged[c] = wm[c]  # mapped B weights
                        b_merged[c] = bm[c]
                    else:
                        w_merged[c] = 0.5 * wa[c] + 0.5 * wm[c]
                        b_merged[c] = 0.5 * ba[c] + 0.5 * bm[c]
            
            params.append(w_merged)
            params.append(b_merged)
        
        merged = MLP(archA)
        with torch.no_grad():
            for p, v in zip(merged.parameters(), params):
                p.copy_(torch.tensor(v, dtype=torch.float32))
        return merged
    
    # Step 5: CMA-ES on scaling factors only
    def fitness(s):
        try:
            m = build_merged(s)
            return -ev(m, X_val, y_val)
        except: return 1.0
    
    es = cma.CMAEvolutionStrategy(s0.tolist(), 0.3, {
        'maxiter': 35, 'popsize': 14, 'seed': SEED,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [[-3]*nd_s, [3]*nd_s],
    })
    while not es.stop():
        sols = es.ask()
        es.tell(sols, [fitness(np.array(s)) for s in sols])
    
    best_s = np.array(es.result.xbest)
    merged = build_merged(best_s)
    return merged


# ─── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 75)
    print("  E15: Class-aware selective merge via importance scoring")
    print("  A trains: 0-4, B trains: 5-9")
    print("=" * 75)
    
    X_tr, y_tr, X_te, y_te = load_mnist()
    idx = torch.randperm(len(X_tr), generator=torch.Generator().manual_seed(0))
    X_val, y_val = X_tr[idx[50000:55000]], y_tr[idx[50000:55000]]
    X_cal = X_tr[idx[:3000]]
    y_cal = y_tr[idx[:3000]]
    
    archA = [784, 128, 64, 10]
    archB = [784, 256, 128, 10]
    classesA, classesB = list(range(5)), list(range(5,10))
    
    mA = train_on(archA, X_tr, y_tr, classesA)
    mB = train_on(archB, X_tr, y_tr, classesB)
    
    pcA, pcB = per_class(mA, X_te, y_te), per_class(mB, X_te, y_te)
    print(f"\n  Parents:")
    print(f"  A (0-4): {ev(mA,X_te,y_te):.3f}  per-class: {[round(pcA[c],2) for c in range(10)]}")
    print(f"  B (5-9): {ev(mB,X_te,y_te):.3f}  per-class: {[round(pcB[c],2) for c in range(10)]}")
    
    # ─── Method 1: Activation-based importance ───────────────────────
    print(f"\n{'━'*75}")
    print("  Method 1: Activation-magnitude importance")
    merged_act = class_selective_merge(mA, mB, X_cal, y_cal, classesA, classesB,
                                        X_val, y_val, use_gradient=False)
    pc_act = per_class(merged_act, X_te, y_te)
    a_acc = np.mean([pc_act[c] for c in range(5)])
    b_acc = np.mean([pc_act[c] for c in range(5,10)])
    print(f"  Overall: {ev(merged_act,X_te,y_te):.3f}  A={a_acc:.3f}  B={b_acc:.3f}")
    
    # ─── Method 2: Gradient-based importance ─────────────────────────
    print(f"\n{'━'*75}")
    print("  Method 2: Gradient-based importance")
    merged_grad = class_selective_merge(mA, mB, X_cal, y_cal, classesA, classesB,
                                         X_val, y_val, use_gradient=True)
    pc_grad = per_class(merged_grad, X_te, y_te)
    a_acc_g = np.mean([pc_grad[c] for c in range(5)])
    b_acc_g = np.mean([pc_grad[c] for c in range(5,10)])
    print(f"  Overall: {ev(merged_grad,X_te,y_te):.3f}  A={a_acc_g:.3f}  B={b_acc_g:.3f}")
    
    # ─── Final comparison ────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("  COMPARISON")
    print(f"{'='*75}")
    
    print(f"\n  {'Class':>6s} {'Parent':>7s} {'Act-imp':>8s} {'Grad-imp':>9s}")
    for c in range(10):
        p = max(pcA[c], pcB[c])
        a = pc_act[c]; g = pc_grad[c]
        src = "A" if c < 5 else "B"
        mk_a = "✅" if a >= 0.5*p else "❌"
        mk_g = "✅" if g >= 0.5*p else "❌"
        print(f"  {c:>6d} {p:>7.3f} {a:>8.3f} {mk_a} {g:>7.3f} {mk_g}  ({src})")
    
    a1 = np.mean([pc_act[c] for c in range(5)])
    b1 = np.mean([pc_act[c] for c in range(5,10)])
    a2 = np.mean([pc_grad[c] for c in range(5)])
    b2 = np.mean([pc_grad[c] for c in range(5,10)])
    bal1 = min(a1,b1)/(max(a1,b1)+1e-10)
    bal2 = min(a2,b2)/(max(a2,b2)+1e-10)
    
    print(f"\n  Act-importance:  A={a1:.3f} B={b1:.3f} balance={bal1:.3f}")
    print(f"  Grad-importance: A={a2:.3f} B={b2:.3f} balance={bal2:.3f}")
    print(f"  vs E14 baseline: A=0.943 B=0.000 balance=0.000")
    print(f"  vs E14 S2(ens):  A=0.366 B=0.656 balance=0.557")
    
    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.1f}s")
    
    results = {
        'activation': {'overall': ev(merged_act,X_te,y_te), 'A':a1, 'B':b1, 'balance':bal1,
                       'per_class': {str(c): round(pc_act[c],4) for c in range(10)}},
        'gradient': {'overall': ev(merged_grad,X_te,y_te), 'A':a2, 'B':b2, 'balance':bal2,
                     'per_class': {str(c): round(pc_grad[c],4) for c in range(10)}},
    }
    with open("results_e15.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
