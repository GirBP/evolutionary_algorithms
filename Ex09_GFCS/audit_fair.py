#!/usr/bin/env python3
"""
AUDIT: Was the GFCS vs EAIB comparison fair?

Hypothesis 1: GFCS uses MORE EA compute (pop=20, gens=30 vs pop=10, gens=15)
Hypothesis 2: EAIB's Pareto fitness pushes toward aggressive compression (more neurons removed)
Hypothesis 3: Importance scoring matters (flow-only vs blended)

Test: Equalize EVERYTHING except importance scoring.
All methods get: same EA (pop=20, gens=30, scalar fitness), same build (selection), same finetune.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, copy, time
from pathlib import Path
torch.set_num_threads(1)

DENSE_DIR = Path(__file__).resolve().parent.parent / "dense_autosearch"
sys.path.insert(0, str(DENSE_DIR.parent / "common"))
sys.path.insert(0, str(DENSE_DIR))
from compress_prepare import (
    GenericCNN, GenericResNet, CompactCNN, CompactResNet,
    get_loaders, eval_f1, micro_finetune_compact,
)

SPARSE_DIR = DENSE_DIR / "data" / "sparse"
SPATIAL = 49

def load_cached(arch, seed):
    teacher_state = torch.load(SPARSE_DIR / f"teacher_generic_{arch}_s{seed}.pt", weights_only=True)
    sparse_state = torch.load(SPARSE_DIR / f"sparse_{arch}_s{seed}.pt", weights_only=True)
    if arch == 'cnn':
        teacher = GenericCNN([32, 64], [128], 10)
        sparse = GenericCNN([32, 64], [128], 10)
    else:
        teacher = GenericResNet(32, 64, [256, 128], 10)
        sparse = GenericResNet(32, 64, [256, 128], 10)
    teacher.load_state_dict(teacher_state)
    sparse.load_state_dict(sparse_state)
    return teacher, sparse


def _copy_bn(src, dst, idx):
    if src.weight is not None: dst.weight.data.copy_(src.weight.data[idx])
    if src.bias is not None: dst.bias.data.copy_(src.bias.data[idx])
    if hasattr(src, 'running_mean') and src.running_mean is not None:
        dst.running_mean.copy_(src.running_mean[idx])
        dst.running_var.copy_(src.running_var[idx])


def _build(model, selections, arch):
    if arch == 'cnn':
        idx1, idx2 = selections[0], selections[1]
        fc_sels = selections[2:]
        compact = CompactCNN([len(idx1), len(idx2)], [len(s) for s in fc_sels], model.n_classes)
        with torch.no_grad():
            compact.conv1.weight.data.copy_(model.conv1.weight.data[idx1])
            if model.conv1.bias is not None: compact.conv1.bias.data.copy_(model.conv1.bias.data[idx1])
            compact.conv2.weight.data.copy_(model.conv2.weight.data[idx2][:, idx1])
            if model.conv2.bias is not None: compact.conv2.bias.data.copy_(model.conv2.bias.data[idx2])
            cidx = []
            for j in idx2: cidx.extend(range(j*SPATIAL, (j+1)*SPATIAL))
            prev = torch.tensor(cidx, dtype=torch.long)
            for i in range(len(model.fcs)):
                if i < len(fc_sels):
                    compact.fcs[i].weight.data.copy_(model.fcs[i].weight.data[fc_sels[i]][:, prev])
                    if model.fcs[i].bias is not None: compact.fcs[i].bias.data.copy_(model.fcs[i].bias.data[fc_sels[i]])
                    prev = fc_sels[i]
                else:
                    compact.fcs[i].weight.data.copy_(model.fcs[i].weight.data[:, prev])
                    if model.fcs[i].bias is not None: compact.fcs[i].bias.data.copy_(model.fcs[i].bias.data)
        return compact
    else:
        idx1, idx2 = selections[0], selections[1]
        fc_sels = selections[2:]
        compact = CompactResNet(len(idx1), len(idx2), [len(s) for s in fc_sels], model.n_classes)
        with torch.no_grad():
            compact.conv1.weight.data.copy_(model.conv1.weight.data[idx1])
            _copy_bn(model.bn1, compact.bn1, idx1)
            compact.conv2.weight.data.copy_(model.conv2.weight.data[idx1][:, idx1])
            _copy_bn(model.bn2, compact.bn2, idx1)
            compact.conv3.weight.data.copy_(model.conv3.weight.data[idx1][:, idx1])
            _copy_bn(model.bn3, compact.bn3, idx1)
            compact.conv_skip.weight.data.copy_(model.conv_skip.weight.data[idx2][:, idx1])
            _copy_bn(model.bn_skip, compact.bn_skip, idx2)
            compact.conv4.weight.data.copy_(model.conv4.weight.data[idx2][:, idx1])
            _copy_bn(model.bn4, compact.bn4, idx2)
            compact.conv5.weight.data.copy_(model.conv5.weight.data[idx2][:, idx2])
            _copy_bn(model.bn5, compact.bn5, idx2)
            cidx = []
            for j in idx2: cidx.extend(range(j*SPATIAL, (j+1)*SPATIAL))
            prev = torch.tensor(cidx, dtype=torch.long)
            for i in range(len(model.fcs)):
                if i < len(fc_sels):
                    compact.fcs[i].weight.data.copy_(model.fcs[i].weight.data[fc_sels[i]][:, prev])
                    compact.fcs[i].bias.data.copy_(model.fcs[i].bias.data[fc_sels[i]])
                    prev = fc_sels[i]
                else:
                    compact.fcs[i].weight.data.copy_(model.fcs[i].weight.data[:, prev])
                    compact.fcs[i].bias.data.copy_(model.fcs[i].bias.data)
        return compact


# ═══════════════════════════════════════════
#  Importance scoring functions (THE ONLY DIFFERENCE)
# ═══════════════════════════════════════════

def _get_layer_data(model, arch):
    """Get per-layer (W_in_raw, W_out_raw, spatial) for importance computation."""
    layers = []
    if arch == 'cnn':
        layers.append((model.conv1.weight.data, model.conv2.weight.data, 1))
        layers.append((model.conv2.weight.data, model.fcs[0].weight.data, SPATIAL))
        for i in range(len(model.fcs) - 1):
            layers.append((model.fcs[i].weight.data, model.fcs[i+1].weight.data, 1))
    else:
        # ResNet: aggregate flows for conv blocks
        c1 = model.conv1.weight.shape[0]
        c2 = model.conv4.weight.shape[0]

        # Block 1: conv1 outputs
        W_out_b1 = torch.zeros(1, c1)
        for i in range(c1):
            W_out_b1[0, i] = (model.conv2.weight.data[:, i].abs().sum() +
                              model.conv3.weight.data[:, i].abs().sum() +
                              model.conv_skip.weight.data[:, i].abs().sum() +
                              model.conv4.weight.data[:, i].abs().sum())
        layers.append((model.conv1.weight.data, W_out_b1, 1))

        # Block 2
        W_in_b2 = torch.zeros(c2, 1)
        W_out_b2 = torch.zeros(1, c2)
        for i in range(c2):
            W_in_b2[i, 0] = (model.conv_skip.weight.data[i].abs().sum() +
                             model.conv4.weight.data[i].abs().sum())
            W_out_b2[0, i] = (model.conv5.weight.data[:, i].abs().sum() +
                              model.fcs[0].weight.data[:, i*SPATIAL:(i+1)*SPATIAL].abs().sum())
        layers.append((W_in_b2, W_out_b2, 1))

        # FC layers
        for i in range(len(model.fcs) - 1):
            layers.append((model.fcs[i].weight.data, model.fcs[i+1].weight.data, 1))
    return layers


def score_flow(W_in, W_out, sp, alpha=None):
    """Pure flow importance: φ = ||W_in||₁ · ||W_out||₁"""
    n = W_in.shape[0]
    phi_in = W_in.abs().reshape(n, -1).sum(dim=1)
    if W_out.dim() == 4:
        phi_out = W_out.abs().sum(dim=(0,2,3))
    elif sp > 1:
        phi_out = torch.stack([W_out[:, j*sp:(j+1)*sp].abs().sum() for j in range(n)])
    else:
        phi_out = W_out.abs().sum(dim=0)
        if phi_out.shape[0] > n: phi_out = phi_out[:n]
    return phi_in * phi_out


def score_eaib(W_in, W_out, sp, alpha=None):
    """EAIB: α·(φ_in·φ_out) + (1-α)·(φ_in+φ_out) with normalization"""
    if alpha is None: alpha = 0.5
    n = W_in.shape[0]
    phi_in = W_in.abs().reshape(n, -1).sum(dim=1)
    if W_out.dim() == 4:
        phi_out = W_out.abs().sum(dim=(0,2,3))
    elif sp > 1:
        phi_out = torch.stack([W_out[:, j*sp:(j+1)*sp].abs().sum() for j in range(n)])
    else:
        phi_out = W_out.abs().sum(dim=0)
        if phi_out.shape[0] > n: phi_out = phi_out[:n]
    fi = phi_in * phi_out
    phi_in_n = phi_in / (phi_in.max() + 1e-12)
    phi_out_n = phi_out / (phi_out.max() + 1e-12)
    mi = phi_in_n + phi_out_n
    fi_n = fi / (fi.max() + 1e-12)
    mi_n = mi / (mi.max() + 1e-12)
    return alpha * fi_n + (1 - alpha) * mi_n


def score_eaib_unnorm(W_in, W_out, sp, alpha=None):
    """EAIB without normalization (as in original verify_ex09v2.py — potentially buggy)."""
    if alpha is None: alpha = 0.5
    n = W_in.shape[0]
    phi_in = W_in.abs().reshape(n, -1).sum(dim=1)
    if W_out.dim() == 4:
        phi_out = W_out.abs().sum(dim=(0,2,3))
    elif sp > 1:
        phi_out = torch.stack([W_out[:, j*sp:(j+1)*sp].abs().sum() for j in range(n)])
    else:
        phi_out = W_out.abs().sum(dim=0)
        if phi_out.shape[0] > n: phi_out = phi_out[:n]
    return alpha * (phi_in * phi_out) + (1 - alpha) * (phi_in + phi_out)


# ═══════════════════════════════════════════
#  Unified EA: SAME for all methods (pop=20, gens=30, scalar fitness)
# ═══════════════════════════════════════════

def unified_compress(sparse_model, arch, score_fn, n_extra_params=0, method_name=""):
    """All methods use this. Only 'score_fn' differs."""
    layer_data = _get_layer_data(sparse_model, arch)
    n_layers = len(layer_data)

    def get_alives():
        als = []
        for W_in, W_out, sp in layer_data:
            phi = score_flow(W_in, W_out, sp)
            als.append(max(int((phi > 1e-8).sum().item()), 4))
        return als

    alives = get_alives()
    chrom_len = n_layers + n_extra_params  # ratios + optional alphas

    def fitness(chrom):
        ratios = chrom[:n_layers]
        alphas = chrom[n_layers:] if n_extra_params > 0 else [None] * n_layers
        err, ta, tk = 0.0, 0, 0
        for i, (W_in, W_out, sp) in enumerate(layer_data):
            imp = score_fn(W_in, W_out, sp, alphas[i] if n_extra_params > 0 else None)
            n = imp.shape[0]; a = alives[i]
            k = max(4, int(a * ratios[i])); ta += a; tk += k
            ti = imp.sum().item()
            if ti > 1e-12:
                _, topk = imp.topk(min(k, n))
                err += 1.0 - imp[topk].sum().item() / ti
        return -err - 0.5 * (tk / max(ta, 1))

    # SAME EA for everyone: pop=20, gens=30
    pop_size, gens = 20, 30
    pop = []
    for _ in range(pop_size):
        r = np.random.uniform(0.1, 0.8, n_layers)
        if n_extra_params > 0:
            a = np.random.uniform(0.0, 1.0, n_extra_params)
            pop.append(np.concatenate([r, a]))
        else:
            pop.append(r)

    best_g, best_f = pop[0].copy(), -1e9
    for gen in range(gens):
        scored = sorted([(g, fitness(g)) for g in pop], key=lambda x: x[1], reverse=True)
        if scored[0][1] > best_f:
            best_g, best_f = scored[0][0].copy(), scored[0][1]
        parents = [s[0] for s in scored[:5]]
        new_pop = list(parents)
        sig = 0.12 * (1 - gen / gens * 0.5)
        while len(new_pop) < pop_size:
            p = parents[np.random.randint(5)]
            ch = p + np.random.randn(chrom_len) * sig
            ch[:n_layers] = np.clip(ch[:n_layers], 0.1, 0.8)
            if n_extra_params > 0:
                ch[n_layers:] = np.clip(ch[n_layers:], 0.0, 1.0)
            new_pop.append(ch)
        pop = new_pop

    # Build selections from best chromosome
    sels = []
    alphas_best = best_g[n_layers:] if n_extra_params > 0 else [None] * n_layers
    for i, (W_in, W_out, sp) in enumerate(layer_data):
        imp = score_fn(W_in, W_out, sp, alphas_best[i] if n_extra_params > 0 else None)
        a = alives[i]
        k = max(4, int(a * best_g[i]))
        _, idx = imp.topk(min(k, imp.shape[0]))
        sels.append(sorted(idx.tolist()))

    compact = _build(sparse_model, sels, arch)
    print(f"    {method_name:20s} sels={[len(s) for s in sels]}", flush=True)
    return compact


# ═══════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════
def main():
    print("=" * 70, flush=True)
    print("  FAIR AUDIT: Same EA, same build — only importance scoring differs", flush=True)
    print("  pop=20, gens=30, scalar fitness for ALL methods", flush=True)
    print("=" * 70, flush=True)

    methods = [
        ("GFCS (flow only)",      score_flow,         0),
        ("EAIB (blend+norm)",     score_eaib,         0),  # n_extra=0: use fixed alpha=0.5
        ("EAIB_EA (blend+norm)",  score_eaib,         None),  # n_extra=n_layers: evolve alpha
        ("EAIB_raw (no norm)",    score_eaib_unnorm,  None),  # original buggy version
    ]

    for arch in ['cnn', 'resnet']:
        for seed in [42, 123]:
            print(f"\n{'═'*60}", flush=True)
            print(f"  {arch.upper()} seed={seed}", flush=True)
            print(f"{'═'*60}", flush=True)

            teacher, sparse = load_cached(arch, seed)
            tr_dl, te_dl = get_loaders(arch, seed)
            f1_teacher = eval_f1(teacher, te_dl)
            print(f"  Teacher F1={f1_teacher:.4f}  params={teacher.count_params():,}", flush=True)

            layer_data = _get_layer_data(sparse, arch)
            n_layers = len(layer_data)

            # Measure inference RCU for sparse model (once)
            X_batch = next(iter(te_dl))[0][:64]
            sparse_model = copy.deepcopy(sparse)
            sparse_model.eval()
            # warmup
            for _ in range(3): sparse_model(X_batch)
            t0 = time.thread_time_ns()
            for _ in range(10): sparse_model(X_batch)
            rcu_sparse = (time.thread_time_ns() - t0) / 1e6 / 10

            for name, sfn, n_extra in methods:
                np.random.seed(seed); torch.manual_seed(seed)
                actual_extra = n_layers if n_extra is None else n_extra

                # RCU of compression
                t0 = time.thread_time_ns()
                compact = unified_compress(copy.deepcopy(sparse), arch, sfn, actual_extra, name)
                rcu_compress = (time.thread_time_ns() - t0) / 1e6

                # Inference RCU of compact
                compact.eval()
                for _ in range(3): compact(X_batch)
                t0 = time.thread_time_ns()
                for _ in range(10): compact(X_batch)
                rcu_compact = (time.thread_time_ns() - t0) / 1e6 / 10
                infer_speedup = rcu_sparse / max(rcu_compact, 0.001)

                f1_pre = eval_f1(compact, te_dl)
                micro_finetune_compact(compact, tr_dl, batches=25)
                f1_post = eval_f1(compact, te_dl)
                rpr = f1_post / f1_teacher
                comp = teacher.count_params() / compact.count_params()

                print(f"    {name:20s}  RPR={rpr:.4f}  comp={comp:.1f}×  "
                      f"infer={infer_speedup:.2f}×  RCU={rcu_compress:.0f}ms  "
                      f"params={compact.count_params():>8,}  F1={f1_post:.4f}", flush=True)

    print("\n=== DONE ===", flush=True)


if __name__ == '__main__':
    main()
