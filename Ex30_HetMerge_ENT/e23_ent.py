#!/usr/bin/env python3
"""
E23 — Evolutionary Neuro-Transplantation (ENT).
=================================================
A GENUINELY NOVEL METHOD: EA searches over the space of merged
network topologies assembled from pre-trained neural components.

Chromosome encodes:
  - Binary masks: which neurons from A and B to keep
  - Float genes: cross-connection scales, per-class output routing
  - Compression target: penalty for oversized networks

This is NOT:
  - NAS (doesn't search from scratch — uses pre-trained weights)
  - Model merging (doesn't just interpolate — selects structure)
  - Pruning (doesn't prune one model — assembles from two)

This IS: evolutionary optimization in the space of sub-networks
constructible from pre-trained components.
"""

import numpy as np
import torch
import torch.nn as nn
import time, json, copy, random

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)


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
    def __init__(s, a):
        super().__init__(); l = []
        for i in range(len(a)-1):
            l.append(nn.Linear(a[i], a[i+1]))
            if i < len(a)-2: l.append(nn.ReLU())
        s.net = nn.Sequential(*l); s.arch = a
    def forward(s, x): return s.net(x)


def ev(m, X, y):
    m.eval()
    with torch.no_grad(): return (m(X).argmax(1) == y).float().mean().item()

def pc(m, X, y):
    m.eval()
    with torch.no_grad(): p = m(X).argmax(1)
    return {c: (p[y==c]==c).float().mean().item() if (y==c).sum() > 0 else 0 for c in range(10)}

def trn(a, X, y, cls):
    mask = torch.zeros(len(y), dtype=torch.bool)
    for c in cls: mask |= (y == c)
    Xs, ys = X[mask][:5000], y[mask][:5000]
    m = MLP(a); opt = torch.optim.Adam(m.parameters(), lr=0.003); m.train()
    for _ in range(15):
        l = nn.CrossEntropyLoss()(m(Xs), ys)
        opt.zero_grad(); l.backward(); opt.step()
    return m


# ═══════════════════════════════════════════════════════════════════
#  ENT: Evolutionary Neuro-Transplantation
# ═══════════════════════════════════════════════════════════════════

class ENTChromosome:
    """Encodes a merged network topology from two parent models."""
    
    def __init__(self, n_layers, layer_sizes_A, layer_sizes_B, n_classes=10):
        self.n_layers = n_layers  # hidden layers
        self.sizes_A = layer_sizes_A  # neurons per hidden layer in A
        self.sizes_B = layer_sizes_B
        self.n_classes = n_classes
        
        # Binary masks: which neurons to keep
        self.masks_A = [np.ones(s, dtype=bool) for s in layer_sizes_A]
        self.masks_B = [np.ones(s, dtype=bool) for s in layer_sizes_B]
        
        # Float genes: per-class routing (sigmoid applied)
        self.routing = np.zeros(n_classes)  # >0 → A, <0 → B
        
        # Float genes: cross-connection scale per layer
        self.cross_scales = np.zeros(max(1, n_layers - 1))
    
    def copy(self):
        c = ENTChromosome(self.n_layers, self.sizes_A, self.sizes_B, self.n_classes)
        c.masks_A = [m.copy() for m in self.masks_A]
        c.masks_B = [m.copy() for m in self.masks_B]
        c.routing = self.routing.copy()
        c.cross_scales = self.cross_scales.copy()
        return c
    
    def total_neurons(self):
        return sum(m.sum() for m in self.masks_A) + sum(m.sum() for m in self.masks_B)
    
    def max_neurons(self):
        return sum(self.sizes_A) + sum(self.sizes_B)


def mutate(chrom, p_flip=0.05, sigma_float=0.3):
    """Mutation operator for ENT chromosome."""
    c = chrom.copy()
    
    # Bit-flip mutation on masks
    for masks in [c.masks_A, c.masks_B]:
        for m in masks:
            flips = np.random.random(len(m)) < p_flip
            m[flips] = ~m[flips]
            # Ensure at least 1 neuron per layer
            if m.sum() == 0:
                m[np.random.randint(len(m))] = True
    
    # Gaussian mutation on routing
    c.routing += np.random.randn(len(c.routing)) * sigma_float
    
    # Gaussian mutation on cross-connections
    c.cross_scales += np.random.randn(len(c.cross_scales)) * sigma_float
    
    return c


def crossover(parent1, parent2):
    """Uniform crossover for ENT chromosomes."""
    child = parent1.copy()
    
    # Swap mask bits
    for i in range(len(child.masks_A)):
        swap = np.random.random(len(child.masks_A[i])) > 0.5
        child.masks_A[i][swap] = parent2.masks_A[i][swap]
    for i in range(len(child.masks_B)):
        swap = np.random.random(len(child.masks_B[i])) > 0.5
        child.masks_B[i][swap] = parent2.masks_B[i][swap]
    
    # Blend routing
    alpha = np.random.random()
    child.routing = alpha * parent1.routing + (1 - alpha) * parent2.routing
    child.cross_scales = alpha * parent1.cross_scales + (1 - alpha) * parent2.cross_scales
    
    return child


def build_model_from_chromosome(chrom, WA, WB, logit_rA=1.0, logit_rB=1.0):
    """Construct a neural network from ENT chromosome + parent weights.
    
    This is the PHENOTYPE: actual runnable network assembled from
    selected neurons of both parents.
    """
    nh = chrom.n_layers
    
    # Determine merged layer sizes
    merged_sizes = [784]  # input
    for i in range(nh):
        nA = int(chrom.masks_A[i].sum())
        nB = int(chrom.masks_B[i].sum())
        merged_sizes.append(nA + nB)
    merged_sizes.append(10)
    
    # Check minimum sizes
    for s in merged_sizes[1:-1]:
        if s < 2: return None  # degenerate
    
    params = []
    
    for li in range(nh + 1):
        if li == 0:
            # First hidden: select rows from A and B
            idxA = np.where(chrom.masks_A[0])[0]
            idxB = np.where(chrom.masks_B[0])[0]
            wa, ba = WA[0], WA[1]  # (dA, 784), (dA,)
            wb, bb = WB[0], WB[1]
            
            W = np.vstack([wa[idxA], wb[idxB]])
            b = np.concatenate([ba[idxA], bb[idxB]])
        
        elif li < nh:
            # Middle hidden: block-diagonal + cross-connections
            idxA_prev = np.where(chrom.masks_A[li-1])[0]
            idxB_prev = np.where(chrom.masks_B[li-1])[0]
            idxA_curr = np.where(chrom.masks_A[li])[0]
            idxB_curr = np.where(chrom.masks_B[li])[0]
            
            nA_prev, nB_prev = len(idxA_prev), len(idxB_prev)
            nA_curr, nB_curr = len(idxA_curr), len(idxB_curr)
            
            wa, ba = WA[li*2], WA[li*2+1]
            wb, bb = WB[li*2], WB[li*2+1]
            
            # Select sub-matrices
            wa_sub = wa[np.ix_(idxA_curr, idxA_prev)]
            wb_sub = wb[np.ix_(idxB_curr, idxB_prev)]
            
            W = np.zeros((nA_curr + nB_curr, nA_prev + nB_prev), dtype=np.float32)
            b_vec = np.zeros(nA_curr + nB_curr, dtype=np.float32)
            
            # Block diagonal
            W[:nA_curr, :nA_prev] = wa_sub
            b_vec[:nA_curr] = ba[idxA_curr]
            W[nA_curr:, nA_prev:] = wb_sub
            b_vec[nA_curr:] = bb[idxB_curr]
            
            # Cross-connections (evolved scale)
            cs = chrom.cross_scales[min(li-1, len(chrom.cross_scales)-1)] * 0.05
            d_cross = min(nA_curr, nB_prev, nB_curr, nA_prev)
            for j in range(d_cross):
                if j < nA_curr and nA_prev + j < W.shape[1]:
                    W[j, nA_prev + j] = cs
                if nA_curr + j < W.shape[0] and j < nA_prev:
                    W[nA_curr + j, j] = cs
            
            b = b_vec
        
        else:
            # Output layer: per-class assembly
            idxA_last = np.where(chrom.masks_A[-1])[0]
            idxB_last = np.where(chrom.masks_B[-1])[0]
            nA_last, nB_last = len(idxA_last), len(idxB_last)
            
            wa, ba = WA[-2], WA[-1]  # (10, dA_last)
            wb, bb = WB[-2], WB[-1]
            
            W = np.zeros((10, nA_last + nB_last), dtype=np.float32)
            b_vec = np.zeros(10, dtype=np.float32)
            
            for c in range(10):
                alpha = 1.0 / (1.0 + np.exp(-chrom.routing[c]))
                
                if nA_last > 0:
                    W[c, :nA_last] = alpha * logit_rA * wa[c][idxA_last]
                if nB_last > 0:
                    W[c, nA_last:] = (1-alpha) * logit_rB * wb[c][idxB_last]
                
                b_vec[c] = alpha * logit_rA * ba[c] + (1-alpha) * logit_rB * bb[c]
            
            b = b_vec
        
        params.append(W)
        params.append(b)
    
    model = MLP(merged_sizes)
    try:
        with torch.no_grad():
            for p, v in zip(model.parameters(), params):
                p.copy_(torch.tensor(v, dtype=torch.float32))
    except:
        return None
    
    return model


def ent_evolve(mA, mB, X_val, y_val, X_cal,
               pop_size=30, n_gen=40, compression_target=0.6,
               lambda_compress=0.1):
    """
    ENT: Evolutionary Neuro-Transplantation.
    
    Evolves the structure of a merged neural network by selecting
    which neurons from each parent to keep and how to connect them.
    """
    aA, aB = mA.arch, mB.arch
    nh = len(aA) - 2
    sizes_A = [aA[i+1] for i in range(nh)]
    sizes_B = [aB[i+1] for i in range(nh)]
    
    WA = [p.detach().numpy() for p in mA.parameters()]
    WB = [p.detach().numpy() for p in mB.parameters()]
    
    # Logit normalization
    mA.eval(); mB.eval()
    with torch.no_grad():
        sA = mA(X_cal).numpy().std(); sB = mB(X_cal).numpy().std()
    tgt = (sA + sB) / 2
    rA, rB = tgt/(sA+1e-10), tgt/(sB+1e-10)
    
    max_neurons = sum(sizes_A) + sum(sizes_B)
    
    def fitness(chrom):
        model = build_model_from_chromosome(chrom, WA, WB, rA, rB)
        if model is None: return -1.0
        
        acc = ev(model, X_val, y_val)
        
        # Compression reward: fewer neurons = bonus
        compression = 1.0 - chrom.total_neurons() / max_neurons
        
        return acc + lambda_compress * compression
    
    # ─── Initialize population ────────────────────────────────
    population = []
    
    # Seed 1: full concat (all neurons)
    c_full = ENTChromosome(nh, sizes_A, sizes_B)
    c_full.routing[:5] = 2.0   # A-classes
    c_full.routing[5:] = -2.0  # B-classes
    population.append(c_full)
    
    # Seed 2: A-only
    c_a = ENTChromosome(nh, sizes_A, sizes_B)
    for m in c_a.masks_B: m[:] = False; m[0] = True
    c_a.routing[:] = 2.0
    population.append(c_a)
    
    # Seed 3: B-only
    c_b = ENTChromosome(nh, sizes_A, sizes_B)
    for m in c_b.masks_A: m[:] = False; m[0] = True
    c_b.routing[:] = -2.0
    population.append(c_b)
    
    # Rest: random
    while len(population) < pop_size:
        c = ENTChromosome(nh, sizes_A, sizes_B)
        for m in c.masks_A: m[:] = np.random.random(len(m)) > 0.3
        for m in c.masks_B: m[:] = np.random.random(len(m)) > 0.3
        for masks in [c.masks_A, c.masks_B]:
            for m in masks:
                if m.sum() == 0: m[np.random.randint(len(m))] = True
        c.routing = np.random.randn(10) * 1.5
        c.cross_scales = np.random.randn(len(c.cross_scales)) * 0.5
        population.append(c)
    
    # ─── Evolution loop ───────────────────────────────────────
    best_fitness = -1
    best_chrom = None
    
    for gen in range(n_gen):
        # Evaluate
        fitnesses = [fitness(c) for c in population]
        
        # Track best
        gen_best = max(fitnesses)
        gen_best_idx = np.argmax(fitnesses)
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_chrom = population[gen_best_idx].copy()
        
        avg_neurons = np.mean([c.total_neurons() for c in population])
        
        if gen % 10 == 0 or gen == n_gen - 1:
            print(f"    Gen {gen:>3d}: best={gen_best:.4f} "
                  f"avg_neurons={avg_neurons:.0f}/{max_neurons} "
                  f"({avg_neurons/max_neurons*100:.0f}%)")
        
        # Selection: tournament (size 3)
        new_pop = [best_chrom.copy()]  # elitism
        while len(new_pop) < pop_size:
            # Tournament
            t_idx = random.sample(range(pop_size), min(3, pop_size))
            t_fit = [fitnesses[i] for i in t_idx]
            parent1 = population[t_idx[np.argmax(t_fit)]]
            
            t_idx = random.sample(range(pop_size), min(3, pop_size))
            t_fit = [fitnesses[i] for i in t_idx]
            parent2 = population[t_idx[np.argmax(t_fit)]]
            
            # Crossover + mutation
            if random.random() < 0.7:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            child = mutate(child, p_flip=max(0.02, 0.1 - gen * 0.002))
            new_pop.append(child)
        
        population = new_pop
    
    return best_chrom, best_fitness


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("  E23: Evolutionary Neuro-Transplantation (ENT)")
    print("=" * 70)
    
    X_tr, y_tr, X_te, y_te = load_mnist()
    idx = torch.randperm(len(X_tr), generator=torch.Generator().manual_seed(0))
    X_val, y_val = X_tr[idx[50000:55000]], y_tr[idx[50000:55000]]
    X_cal = X_tr[idx[:3000]]
    
    configs = [
        ("Same 128/64", [784,128,64,10], [784,128,64,10],
         list(range(5)), list(range(5,10))),
        ("Het 64/32 vs 192/96", [784,64,32,10], [784,192,96,10],
         list(range(5)), list(range(5,10))),
    ]
    
    results = []
    for name, aA, aB, clA, clB in configs:
        print(f"\n{'━' * 70}")
        print(f"  {name}: A={aA} B={aB}")
        
        mA = trn(aA, X_tr, y_tr, clA)
        mB = trn(aB, X_tr, y_tr, clB)
        pcA, pcB = pc(mA, X_te, y_te), pc(mB, X_te, y_te)
        
        print(f"  A: {ev(mA,X_te,y_te):.3f} {[round(pcA[c],2) for c in range(10)]}")
        print(f"  B: {ev(mB,X_te,y_te):.3f} {[round(pcB[c],2) for c in range(10)]}")
        
        # ENT evolution
        print(f"\n  ENT evolving (pop=30, gen=40)...")
        WA = [p.detach().numpy() for p in mA.parameters()]
        WB = [p.detach().numpy() for p in mB.parameters()]
        
        # Logit scales
        mA.eval(); mB.eval()
        with torch.no_grad():
            sA = mA(X_cal).numpy().std(); sB = mB(X_cal).numpy().std()
        tgt = (sA+sB)/2; rA = tgt/(sA+1e-10); rB = tgt/(sB+1e-10)
        
        best_chrom, best_fit = ent_evolve(mA, mB, X_val, y_val, X_cal,
                                           pop_size=30, n_gen=40)
        
        # Build final model
        merged = build_model_from_chromosome(best_chrom, WA, WB, rA, rB)
        pcM = pc(merged, X_te, y_te)
        accM = ev(merged, X_te, y_te)
        
        # Compression stats
        total_n = best_chrom.total_neurons()
        max_n = best_chrom.max_neurons()
        
        nA_kept = [int(m.sum()) for m in best_chrom.masks_A]
        nB_kept = [int(m.sum()) for m in best_chrom.masks_B]
        routing_sig = [round(1.0/(1.0+np.exp(-r)), 2) for r in best_chrom.routing]
        
        a_acc = np.mean([pcM[c] for c in clA])
        b_acc = np.mean([pcM[c] for c in clB])
        bal = min(a_acc, b_acc) / (max(a_acc, b_acc) + 1e-10)
        
        print(f"\n  ENT Result:")
        print(f"    Overall: {accM:.3f}  A={a_acc:.3f}  B={b_acc:.3f}  bal={bal:.3f}")
        print(f"    Architecture: {merged.arch}")
        print(f"    Neurons: {total_n}/{max_n} ({total_n/max_n*100:.0f}%)")
        print(f"    A kept: {nA_kept}, B kept: {nB_kept}")
        print(f"    Routing: {routing_sig}")
        print(f"    Params: {sum(p.numel() for p in merged.parameters()):,}")
        
        for c in range(10):
            bp = max(pcA[c], pcB[c])
            ret = pcM[c]/bp if bp > 0 else 0
            mk = '✅' if ret >= 0.6 else ('🟡' if ret >= 0.3 else '❌')
            print(f"      {c}: par={bp:.3f} ent={pcM[c]:.3f} ret={ret:.3f} {mk}")
        
        nP = sum(p.numel() for p in mA.parameters())
        nM = sum(p.numel() for p in merged.parameters())
        
        results.append({
            'name': name, 'acc': round(accM, 4),
            'A': round(a_acc, 4), 'B': round(b_acc, 4),
            'balance': round(bal, 4),
            'neurons_used': total_n, 'neurons_max': max_n,
            'compression': round(total_n/max_n, 3),
            'arch': str(merged.arch),
            'params_merged': nM, 'params_parent': nP,
        })
    
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for r in results:
        print(f"  {r['name']:<25s}: acc={r['acc']:.3f} bal={r['balance']:.3f} "
              f"neurons={r['compression']:.0%} arch={r['arch']}")
    print(f"  Time: {elapsed:.1f}s")
    
    with open("results_e23.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
