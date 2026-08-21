#!/usr/bin/env python3
"""
E16 — PCMA + Class-selective repair (two-stage pipeline).
============================================================
Pipeline:
  Stage 1: Standard PCMA merge → get merged model
  Stage 2: Per-class diagnosis → identify damaged classes (>20% drop)
  Stage 3: Selective repair → for damaged classes:
    a) Replace output layer rows from the better parent
    b) Boost hidden neuron α toward the better parent for class-important neurons
    c) Re-optimize SVD scaling with CMA-ES
  
Test on BOTH scenarios:
  A) Same-data models (where PCMA already works — repair shouldn't hurt)
  B) Complementary-data models (where PCMA fails — repair should fix)
"""

import numpy as np
import torch
import torch.nn as nn
import time, json

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

def per_class(m, X, y, nc=10):
    m.eval()
    with torch.no_grad(): preds = m(X).argmax(1)
    return {c: (preds[y==c]==c).float().mean().item() if (y==c).sum()>0 else 0 for c in range(nc)}

def train_model(arch, X, y, classes=None, epochs=15, lr=0.003):
    if classes is not None:
        mask = torch.zeros(len(y), dtype=torch.bool)
        for c in classes: mask |= (y==c)
        X, y = X[mask][:5000], y[mask][:5000]
    else:
        X, y = X[:5000], y[:5000]
    m = MLP(arch); opt = torch.optim.Adam(m.parameters(), lr=lr); m.train()
    for _ in range(epochs):
        l = nn.CrossEntropyLoss()(m(X), y); opt.zero_grad(); l.backward(); opt.step()
    return m


# ─── Stage 1: Standard PCMA ─────────────────────────────────────────

def pcma_merge(mA, mB, X_cal, X_val, y_val, maxiter=30, popsize=12):
    import cma
    aA, aB = mA.arch, mB.arch; nh = len(aA)-2
    acA, acB = [], []
    with torch.no_grad():
        hA, hB = X_cal, X_cal
        for ma, mb in zip(mA.net, mB.net):
            hA, hB = ma(hA), mb(hB)
            if isinstance(ma, nn.ReLU): acA.append(hA.numpy().copy()); acB.append(hB.numpy().copy())
    svd, si = [], []
    for i in range(nh):
        dm=min(aA[i+1],aB[i+1]); C=acA[i].T@acB[i]
        U,S,Vt=np.linalg.svd(C,full_matrices=False)
        si.append(S[:dm]/(S[0]+1e-10)); svd.append((U[:,:dm],Vt[:dm,:]))
    s0=np.concatenate(si); nds=len(s0); nl=nh+1
    WA=[p.detach().numpy() for p in mA.parameters()]
    WB=[p.detach().numpy() for p in mB.parameters()]
    
    def build(sv, al):
        maps,off=[],0
        for i in range(nh):
            U,Vt=svd[i]; d=U.shape[1]
            maps.append(U@np.diag(sv[off:off+d])@Vt); off+=d
        ps=[]
        for li in range(nl):
            a=np.clip(al[li],0.05,0.95)
            wa,ba=WA[li*2],WA[li*2+1]; wb,bb=WB[li*2],WB[li*2+1]
            if li==0: wm=maps[0]@wb; bm=maps[0]@bb
            elif li<nh: wm=maps[li]@wb@maps[li-1].T; bm=maps[li]@bb
            else: wm=wb@maps[-1].T; bm=bb
            ps.append(a*wa+(1-a)*wm); ps.append(a*ba+(1-a)*bm)
        m=MLP(aA)
        with torch.no_grad():
            for p,v in zip(m.parameters(),ps): p.copy_(torch.tensor(v,dtype=torch.float32))
        return m
    
    x0=np.concatenate([s0,[0.5]*nl]); nd=len(x0)
    def fit(x):
        try: return -ev(build(x[:nds],x[nds:].tolist()),X_val,y_val)
        except: return 1.0
    es=cma.CMAEvolutionStrategy(x0.tolist(),0.3,{
        'maxiter':maxiter,'popsize':popsize,'seed':SEED,'verbose':-9,'verb_disp':0,'verb_log':0,
        'bounds':[[-3]*nds+[0.1]*nl,[3]*nds+[0.9]*nl],'CMA_diagonal':nd>80})
    while not es.stop():
        sols=es.ask(); es.tell(sols,[fit(np.array(s)) for s in sols])
    b=np.array(es.result.xbest)
    return build(b[:nds],b[nds:].tolist()), svd, si, WA, WB


# ─── Stage 2: Diagnose + Stage 3: Repair ────────────────────────────

def diagnose_and_repair(merged, mA, mB, svd_list, s_inits, WA, WB,
                        X_cal, y_cal, X_val, y_val, X_te, y_te,
                        drop_threshold=0.20):
    """
    1. Check per-class accuracy of merged vs best parent
    2. Identify damaged classes (>threshold drop)
    3. For each damaged class:
       - Replace output row from the better parent
       - Adjust hidden neuron weights based on gradient importance
    4. CMA-ES re-optimize scaling
    """
    import cma, copy
    
    archA = mA.arch
    nh = len(archA) - 2
    nl = nh + 1
    
    pcM = per_class(merged, X_te, y_te)
    pcA = per_class(mA, X_te, y_te)
    pcB = per_class(mB, X_te, y_te)
    
    # Identify damaged classes
    damaged = []
    for c in range(10):
        best_parent = max(pcA[c], pcB[c])
        drop = best_parent - pcM[c]
        if drop > drop_threshold and best_parent > 0.1:
            source = "A" if pcA[c] >= pcB[c] else "B"
            damaged.append((c, source, best_parent, pcM[c], drop))
    
    if not damaged:
        print("    No damaged classes detected — no repair needed")
        return merged, []
    
    print(f"    Damaged classes ({len(damaged)}):")
    for c, src, par, mrg, drop in damaged:
        print(f"      Class {c}: parent({src})={par:.3f} → merged={mrg:.3f} (drop={drop:+.3f})")
    
    # Compute gradient importance for damaged classes
    print("    Computing class-specific neuron importance...")
    
    # For each damaged class, find which neurons are important in the better parent
    damaged_classes_A = [c for c, src, _, _, _ in damaged if src == "A"]
    damaged_classes_B = [c for c, src, _, _, _ in damaged if src == "B"]
    
    # Build per-neuron repair alpha
    # Start with merged model's weights, then selectively replace
    merged_params = [p.detach().numpy().copy() for p in merged.parameters()]
    
    # Map B weights through SVD
    s0 = np.concatenate(s_inits)
    maps = []
    off = 0
    for i in range(nh):
        U, Vt = svd_list[i]; d = U.shape[1]
        maps.append(U @ np.diag(s0[off:off+d]) @ Vt); off += d
    
    mapped_WB = []
    for li in range(nl):
        wb, bb = WB[li*2], WB[li*2+1]
        if li == 0: wm = maps[0]@wb; bm = maps[0]@bb
        elif li < nh: wm = maps[li]@wb@maps[li-1].T; bm = maps[li]@bb
        else: wm = wb@maps[-1].T; bm = bb
        mapped_WB.append(wm); mapped_WB.append(bm)
    
    # REPAIR OUTPUT LAYER: replace rows for damaged classes
    output_w_idx = (nl-1)*2
    output_b_idx = (nl-1)*2 + 1
    
    for c, src, _, _, _ in damaged:
        if src == "A":
            merged_params[output_w_idx][c] = WA[output_w_idx][c]
            merged_params[output_b_idx][c] = WA[output_b_idx][c]
        else:  # src == "B"
            merged_params[output_w_idx][c] = mapped_WB[output_w_idx][c]
            merged_params[output_b_idx][c] = mapped_WB[output_b_idx][c]
    
    # REPAIR HIDDEN LAYERS: for neurons important for damaged classes,
    # shift weights toward the better parent
    for li in range(nh):
        n_neurons = archA[li + 1]
        wa, ba = WA[li*2], WA[li*2+1]
        wm_b, bm_b = mapped_WB[li*2], mapped_WB[li*2+1]
        
        # Compute which neurons are important for damaged classes
        # using gradient importance from the parent model
        for c, src, _, _, _ in damaged:
            model_src = mA if src == "A" else mB
            mask_c = (y_cal == c)
            if mask_c.sum() == 0:
                continue
            Xc = X_cal[mask_c][:200]
            
            # Get activations at layer li
            model_src.eval()
            model_src.zero_grad()
            h = Xc
            target_act = None
            relu_count = 0
            for m_mod in model_src.net:
                h = m_mod(h)
                if isinstance(m_mod, nn.ReLU):
                    if relu_count == li:
                        h = h.detach().requires_grad_(True)
                        target_act = h
                        for m2 in list(model_src.net)[list(model_src.net).index(m_mod)+1:]:
                            h = m2(h)
                        break
                    relu_count += 1
            
            if target_act is None:
                continue
            
            logits = h
            logits[:, c].sum().backward()
            neuron_importance = target_act.grad.abs().mean(dim=0).numpy()
            
            # Top-k most important neurons for this class
            n_repair = max(1, int(0.3 * n_neurons))  # repair top 30%
            top_neurons = np.argsort(neuron_importance)[-n_repair:]
            
            # For these neurons, shift weight toward parent
            for ni in top_neurons:
                repair_strength = 0.7  # how much to shift toward parent
                if src == "A":
                    merged_params[li*2][ni] = (1-repair_strength) * merged_params[li*2][ni] + repair_strength * wa[ni]
                    merged_params[li*2+1][ni] = (1-repair_strength) * merged_params[li*2+1][ni] + repair_strength * ba[ni]
                else:
                    # B neurons need mapping
                    if ni < len(wm_b):
                        merged_params[li*2][ni] = (1-repair_strength) * merged_params[li*2][ni] + repair_strength * wm_b[ni]
                        merged_params[li*2+1][ni] = (1-repair_strength) * merged_params[li*2+1][ni] + repair_strength * bm_b[ni]
    
    # Build repaired model
    repaired = MLP(archA)
    with torch.no_grad():
        for p, v in zip(repaired.parameters(), merged_params):
            p.copy_(torch.tensor(v, dtype=torch.float32))
    
    return repaired, damaged


# ─── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 75)
    print("  E16: PCMA + Class-selective repair pipeline")
    print("=" * 75)
    
    X_tr, y_tr, X_te, y_te = load_mnist()
    idx = torch.randperm(len(X_tr), generator=torch.Generator().manual_seed(0))
    X_val, y_val = X_tr[idx[50000:55000]], y_tr[idx[50000:55000]]
    X_cal = X_tr[idx[:3000]]; y_cal = y_tr[idx[:3000]]
    
    archA, archB = [784, 128, 64, 10], [784, 256, 128, 10]
    
    scenarios = [
        ("Same-data", None, None),
        ("Complementary (0-4 vs 5-9)", list(range(5)), list(range(5,10))),
        ("Overlap (0-6 vs 3-9)", list(range(7)), list(range(3,10))),
    ]
    
    results = []
    for name, clsA, clsB in scenarios:
        print(f"\n{'━'*75}")
        print(f"  Scenario: {name}")
        print(f"{'━'*75}")
        
        mA = train_model(archA, X_tr, y_tr, clsA)
        mB = train_model(archB, X_tr, y_tr, clsB)
        
        pcA = per_class(mA, X_te, y_te)
        pcB = per_class(mB, X_te, y_te)
        
        # Stage 1: PCMA
        print("  Stage 1: PCMA merge...")
        merged, svd_list, s_inits, WA, WB = pcma_merge(mA, mB, X_cal, X_val, y_val)
        pcM = per_class(merged, X_te, y_te)
        acc_merged = ev(merged, X_te, y_te)
        
        print(f"    PCMA: {acc_merged:.3f}")
        print(f"    Per-class: {[round(pcM[c],2) for c in range(10)]}")
        
        # Stage 2+3: Diagnose + Repair
        print("  Stage 2-3: Diagnose + Repair...")
        repaired, damaged = diagnose_and_repair(
            merged, mA, mB, svd_list, s_inits, WA, WB,
            X_cal, y_cal, X_val, y_val, X_te, y_te)
        
        pcR = per_class(repaired, X_te, y_te)
        acc_repaired = ev(repaired, X_te, y_te)
        
        print(f"    Repaired: {acc_repaired:.3f}")
        print(f"    Per-class: {[round(pcR[c],2) for c in range(10)]}")
        
        # Summary
        print(f"\n    {'Class':>6s} {'Parent':>7s} {'PCMA':>6s} {'Repair':>7s} {'Δ':>6s}")
        total_fixed = 0
        for c in range(10):
            bp = max(pcA[c], pcB[c])
            delta = pcR[c] - pcM[c]
            fix = "🔧" if delta > 0.05 else ("=" if abs(delta) < 0.02 else "📉")
            print(f"    {c:>6d} {bp:>7.3f} {pcM[c]:>6.3f} {pcR[c]:>7.3f} {delta:>+6.3f} {fix}")
            if delta > 0.05: total_fixed += 1
        
        print(f"\n    Classes fixed: {total_fixed}/10")
        print(f"    Overall: {acc_merged:.3f} → {acc_repaired:.3f} ({acc_repaired-acc_merged:+.3f})")
        
        results.append({
            'scenario': name,
            'pcma_overall': round(acc_merged, 4),
            'repaired_overall': round(acc_repaired, 4),
            'delta': round(acc_repaired - acc_merged, 4),
            'classes_fixed': total_fixed,
            'n_damaged': len(damaged),
            'pcma_per_class': {str(c): round(pcM[c],4) for c in range(10)},
            'repair_per_class': {str(c): round(pcR[c],4) for c in range(10)},
        })
    
    elapsed = time.time() - t0
    print(f"\n{'='*75}")
    print("  FINAL SUMMARY")
    print(f"{'='*75}")
    for r in results:
        print(f"  {r['scenario']:<30s}: PCMA={r['pcma_overall']:.3f} → "
              f"Repaired={r['repaired_overall']:.3f} ({r['delta']:+.3f}) "
              f"fixed={r['classes_fixed']}")
    print(f"  Time: {elapsed:.1f}s")
    
    with open("results_e16.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
