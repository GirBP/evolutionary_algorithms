#!/usr/bin/env python3
"""
Ex09v2 Part 2: CNN & ResNet Compression Benchmark
===================================================
5 methods × 2 architectures (CNN, ResNet) × 2 seeds = 20 measurements
All at 95% sparsity (TESA-26 iter60 pruning → compression)
FashionMNIST dataset

Uses dense_autosearch infrastructure (compress_prepare, cached sparse models).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys, json, time, copy
import torch
torch.set_num_threads(1)
import numpy as np
from pathlib import Path

# Add dense_autosearch to path for compress_prepare
DENSE_DIR = Path(__file__).resolve().parent.parent / "dense_autosearch"
COMMON_DIR = Path(__file__).resolve().parent.parent / "common"
sys.path.insert(0, str(COMMON_DIR))
sys.path.insert(0, str(DENSE_DIR))

from compress_prepare import (
    eval_compression_task, warm_compression_cache,
    GenericMLP, GenericCNN, GenericResNet,
    CompactMLP, CompactCNN, CompactResNet,
    COMPRESS_ARCHS, SEEDS,
)

# ═══════════════════════════════════════════
#  COMPRESSION METHODS (same as ex09v2_benchmark.py, but for Generic* models)
# ═══════════════════════════════════════════

# ────── Shared infrastructure ──────
def _get_layer_info(sparse_model):
    """Extract per-layer info for compress_fn. Handles MLP, CNN, ResNet."""
    layers = []
    if isinstance(sparse_model, GenericMLP):
        all_layers = list(sparse_model.layers)
        for i in range(len(all_layers) - 1):
            W_in = all_layers[i].weight.data
            W_out = all_layers[i + 1].weight.data
            alive = max((W_in.abs().sum(dim=1) > 1e-8).sum().item(), 4)
            layers.append({'W_in': W_in, 'W_out': W_out, 'spatial': 1, 'n_alive': int(alive)})
    elif isinstance(sparse_model, GenericCNN):
        layers.append({
            'W_in': sparse_model.conv1.weight.data,
            'W_out': sparse_model.conv2.weight.data,
            'spatial': 1,
            'n_alive': max(int((sparse_model.conv1.weight.data.abs().sum(dim=(1,2,3)) > 1e-8).sum().item()), 4),
        })
        layers.append({
            'W_in': sparse_model.conv2.weight.data,
            'W_out': sparse_model.fcs[0].weight.data,
            'spatial': 49,
            'n_alive': max(int((sparse_model.conv2.weight.data.abs().sum(dim=(1,2,3)) > 1e-8).sum().item()), 4),
        })
        for i in range(len(sparse_model.fcs) - 1):
            W_in = sparse_model.fcs[i].weight.data
            W_out = sparse_model.fcs[i + 1].weight.data
            alive = max(int((W_in.abs().sum(dim=1) > 1e-8).sum().item()), 4)
            layers.append({'W_in': W_in, 'W_out': W_out, 'spatial': 1, 'n_alive': alive})
    elif isinstance(sparse_model, GenericResNet):
        c1 = sparse_model.conv1.weight.shape[0]
        W_out_b1 = torch.zeros(1, c1)
        for i in range(c1):
            W_out_b1[0, i] = (
                sparse_model.conv2.weight.data[:, i].abs().sum() +
                sparse_model.conv3.weight.data[:, i].abs().sum() +
                sparse_model.conv_skip.weight.data[:, i].abs().sum() +
                sparse_model.conv4.weight.data[:, i].abs().sum())
        layers.append({'W_in': sparse_model.conv1.weight.data, 'W_out': W_out_b1, 'spatial': 1, 'n_alive': c1})
        c2 = sparse_model.conv4.weight.shape[0]
        W_in_b2 = torch.zeros(c2, 1)
        W_out_b2 = torch.zeros(1, c2)
        for i in range(c2):
            W_in_b2[i, 0] = (sparse_model.conv_skip.weight.data[i].abs().sum() +
                              sparse_model.conv4.weight.data[i].abs().sum())
            W_out_b2[0, i] = (sparse_model.conv5.weight.data[:, i].abs().sum() +
                               sparse_model.fcs[0].weight.data[:, i * 49:(i + 1) * 49].abs().sum())
        layers.append({'W_in': W_in_b2, 'W_out': W_out_b2, 'spatial': 1, 'n_alive': c2})
        for i in range(len(sparse_model.fcs) - 1):
            W_in = sparse_model.fcs[i].weight.data
            W_out = sparse_model.fcs[i + 1].weight.data
            alive = max(int((W_in.abs().sum(dim=1) > 1e-8).sum().item()), 4)
            layers.append({'W_in': W_in, 'W_out': W_out, 'spatial': 1, 'n_alive': alive})
    return layers


def _build_compact(sparse_model, selections):
    """Build compact model from selections. Same as dense_autosearch infrastructure."""
    if isinstance(sparse_model, GenericMLP):
        return _build_compact_mlp(sparse_model, selections)
    elif isinstance(sparse_model, GenericResNet):
        return _build_compact_resnet(sparse_model, selections)
    else:
        return _build_compact_cnn(sparse_model, selections)


def _build_compact_mlp(sparse_model, selections):
    all_layers = list(sparse_model.layers)
    hiddens = [len(s) for s in selections]
    compact = CompactMLP(input_dim=sparse_model.input_dim, hidden_sizes=hiddens, n_classes=sparse_model.n_classes)
    with torch.no_grad():
        prev_idx = None
        for i, sel in enumerate(selections):
            W = all_layers[i].weight.data
            if prev_idx is not None:
                W = W[:, prev_idx]
            compact.layers[i].weight.data.copy_(W[sel])
            if all_layers[i].bias is not None:
                compact.layers[i].bias.data.copy_(all_layers[i].bias.data[sel])
            prev_idx = sel
        compact.layers[-1].weight.data.copy_(all_layers[-1].weight.data[:, prev_idx])
        if all_layers[-1].bias is not None:
            compact.layers[-1].bias.data.copy_(all_layers[-1].bias.data)
    return compact


def _copy_bn(src, dst, idx):
    if src.weight is not None:
        dst.weight.data.copy_(src.weight.data[idx])
    if src.bias is not None:
        dst.bias.data.copy_(src.bias.data[idx])
    if hasattr(src, 'running_mean') and src.running_mean is not None:
        dst.running_mean.copy_(src.running_mean[idx])
        dst.running_var.copy_(src.running_var[idx])


def _build_compact_cnn(sparse_model, selections):
    idx1, idx2 = selections[0], selections[1]
    fc_sels = selections[2:]
    spatial = 49
    compact = CompactCNN(channels=[len(idx1), len(idx2)],
                         hiddens=[len(s) for s in fc_sels],
                         n_classes=sparse_model.n_classes)
    with torch.no_grad():
        compact.conv1.weight.data.copy_(sparse_model.conv1.weight.data[idx1])
        if sparse_model.conv1.bias is not None:
            compact.conv1.bias.data.copy_(sparse_model.conv1.bias.data[idx1])
        compact.conv2.weight.data.copy_(sparse_model.conv2.weight.data[idx2][:, idx1])
        if sparse_model.conv2.bias is not None:
            compact.conv2.bias.data.copy_(sparse_model.conv2.bias.data[idx2])
        cidx = []
        for j in idx2:
            cidx.extend(range(j * spatial, (j + 1) * spatial))
        prev_idx = torch.tensor(cidx, dtype=torch.long)
        for i in range(len(sparse_model.fcs)):
            if i < len(fc_sels):
                W = sparse_model.fcs[i].weight.data[:, prev_idx]
                compact.fcs[i].weight.data.copy_(W[fc_sels[i]])
                if sparse_model.fcs[i].bias is not None:
                    compact.fcs[i].bias.data.copy_(sparse_model.fcs[i].bias.data[fc_sels[i]])
                prev_idx = fc_sels[i]
            else:
                compact.fcs[i].weight.data.copy_(sparse_model.fcs[i].weight.data[:, prev_idx])
                if sparse_model.fcs[i].bias is not None:
                    compact.fcs[i].bias.data.copy_(sparse_model.fcs[i].bias.data)
    return compact


def _build_compact_resnet(sparse_model, selections):
    idx1, idx2 = selections[0], selections[1]
    fc_sels = selections[2:]
    spatial = 49
    compact = CompactResNet(c1=len(idx1), c2=len(idx2),
                            hiddens=[len(s) for s in fc_sels],
                            n_classes=sparse_model.n_classes)
    with torch.no_grad():
        compact.conv1.weight.data.copy_(sparse_model.conv1.weight.data[idx1])
        _copy_bn(sparse_model.bn1, compact.bn1, idx1)
        compact.conv2.weight.data.copy_(sparse_model.conv2.weight.data[idx1][:, idx1])
        _copy_bn(sparse_model.bn2, compact.bn2, idx1)
        compact.conv3.weight.data.copy_(sparse_model.conv3.weight.data[idx1][:, idx1])
        _copy_bn(sparse_model.bn3, compact.bn3, idx1)
        compact.conv_skip.weight.data.copy_(sparse_model.conv_skip.weight.data[idx2][:, idx1])
        _copy_bn(sparse_model.bn_skip, compact.bn_skip, idx2)
        compact.conv4.weight.data.copy_(sparse_model.conv4.weight.data[idx2][:, idx1])
        _copy_bn(sparse_model.bn4, compact.bn4, idx2)
        compact.conv5.weight.data.copy_(sparse_model.conv5.weight.data[idx2][:, idx2])
        _copy_bn(sparse_model.bn5, compact.bn5, idx2)
        cidx = []
        for j in idx2:
            cidx.extend(range(j * spatial, (j + 1) * spatial))
        prev_idx = torch.tensor(cidx, dtype=torch.long)
        for i in range(len(sparse_model.fcs)):
            if i < len(fc_sels):
                W = sparse_model.fcs[i].weight.data[:, prev_idx]
                compact.fcs[i].weight.data.copy_(W[fc_sels[i]])
                compact.fcs[i].bias.data.copy_(sparse_model.fcs[i].bias.data[fc_sels[i]])
                prev_idx = fc_sels[i]
            else:
                compact.fcs[i].weight.data.copy_(sparse_model.fcs[i].weight.data[:, prev_idx])
                compact.fcs[i].bias.data.copy_(sparse_model.fcs[i].bias.data)
    return compact


# ────── GFCS baseline (explore00) — uses model-level dispatch ──────
def _gfcs_baseline(sparse_model, teacher_state, sparsity):
    """GFCS baseline from explore00 — full greedy merge + EA."""
    # Import the original GFCS baseline
    sys.path.insert(0, str(DENSE_DIR / "results" / "explorations" / "explore00"))
    import importlib
    spec = importlib.util.spec_from_file_location(
        "gfcs_baseline",
        str(DENSE_DIR / "results" / "explorations" / "explore00" / "experiment.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compress_one(sparse_model, teacher_state, sparsity)


# ────── NeuronRemoval (selection-only: keep all alive neurons) ──────
def _neuron_removal(sparse_model, teacher_state, sparsity):
    """Keep all alive neurons — no merge, no EA. Simplest data-free method."""
    layer_info = _get_layer_info(sparse_model)
    selections = []
    for info in layer_info:
        W_in = info['W_in']
        n = W_in.shape[0]
        if W_in.dim() == 4:
            alive = (W_in.abs().sum(dim=(1, 2, 3)) > 1e-8)
        else:
            alive = (W_in.abs().reshape(n, -1).sum(dim=1) > 1e-8)
        idx = torch.where(alive)[0].tolist()
        if not idx:
            idx = [0]
        selections.append(sorted(idx))
    return _build_compact(sparse_model, selections)


# ────── Generic compress_fn wrapper ──────
def _make_compress_one(compress_fn_ref):
    """Wrap a compress_fn(layer_info) → selections into compress_one(model, ts, sp) → compact."""
    def compress_one(sparse_model, teacher_state, sparsity):
        layer_info = _get_layer_info(sparse_model)
        selections = compress_fn_ref(layer_info)
        return _build_compact(sparse_model, selections)
    return compress_one


# ────── explore25: EAIB ──────
def _compress_fn_eaib(layer_info):
    n_layers = len(layer_info)
    if n_layers == 0:
        return []
    def compute_importance(info, alpha):
        W_in, W_out, spatial = info['W_in'], info['W_out'], info['spatial']
        n = W_in.shape[0]
        phi_in = W_in.abs().reshape(n, -1).sum(dim=1)
        if W_out.dim() == 4:
            phi_out = W_out.abs().sum(dim=(0, 2, 3))
        elif spatial > 1:
            phi_out = torch.stack([W_out[:, j*spatial:(j+1)*spatial].abs().sum() for j in range(n)])
        else:
            phi_out = W_out.abs().sum(dim=0)
            if phi_out.shape[0] > n: phi_out = phi_out[:n]
        return alpha * (phi_in * phi_out) + (1 - alpha) * (phi_in + phi_out)
    def fitness(c):
        ratios, alphas = c[:n_layers], c[n_layers:]
        fe, ta, tk = 0.0, 0, 0
        for i, info in enumerate(layer_info):
            imp = compute_importance(info, alphas[i])
            n = imp.shape[0]; k = max(4, int(info['n_alive'] * ratios[i]))
            ti = imp.sum().item()
            if ti > 1e-12:
                _, topk = imp.topk(min(k, n))
                fe += 1.0 - imp[topk].sum().item() / ti
            ta += info['n_alive']; tk += k
        return (1.0 - fe, 1.0 - tk / max(ta, 1))
    pop_size, gens, sigma = 10, 15, 0.12
    cl = 2 * n_layers
    pop = [np.concatenate([np.random.uniform(0.1, 0.8, n_layers), np.random.uniform(0.0, 1.0, n_layers)]) for _ in range(pop_size)]
    bg, bf = pop[0].copy(), -np.inf
    for gen in range(gens):
        sc = [(g, fitness(g)) for g in pop]
        f0 = [(g, f) for i, (g, f) in enumerate(sc) if not any(all(sc[j][1][k] >= f[k] for k in range(2)) and any(sc[j][1][k] > f[k] for k in range(2)) for j in range(len(sc)) if j != i)]
        for g, f in f0:
            if f[0] > bf: bf = f[0]; bg = g.copy()
        parents = [g for g, _ in f0]
        if len(parents) < 2:
            ss = sorted(sc, key=lambda x: x[1][0], reverse=True)
            parents = [ss[0][0], ss[min(1, len(ss)-1)][0]]
        st = sigma * (1 - gen / gens * 0.5)
        np_ = list(parents)
        while len(np_) < pop_size:
            p = parents[np.random.randint(len(parents))]
            ch = p + np.random.randn(cl) * st
            ch[:n_layers] = np.clip(ch[:n_layers], 0.1, 0.8)
            ch[n_layers:] = np.clip(ch[n_layers:], 0.0, 1.0)
            np_.append(ch)
        pop = np_
    sels = []
    for i, info in enumerate(layer_info):
        imp = compute_importance(info, bg[n_layers + i])
        n = imp.shape[0]; k = max(4, int(info['n_alive'] * bg[i]))
        _, idx = imp.topk(min(k, n))
        sels.append(sorted(idx.cpu().tolist()))
    return sels


# ────── explore48: EPSS ──────
def _compress_fn_epss(layer_info):
    n_layers = len(layer_info)
    if n_layers == 0:
        return []
    def compute_importance(info, alpha, lam):
        W_in, W_out, spatial = info['W_in'], info['W_out'], info['spatial']
        n = W_in.shape[0]
        if n == 0: return torch.tensor([])
        phi_in = W_in.abs().reshape(n, -1).sum(dim=1)
        wof = []; D = 0
        if W_out.dim() == 4:
            phi_out = W_out.abs().sum(dim=(0, 2, 3)); D = W_out.shape[0]*W_out.shape[2]*W_out.shape[3]
            for j in range(n): wof.append(W_out[:, j, :, :].reshape(-1))
        elif spatial > 1:
            phi_out = torch.stack([W_out[:, j*spatial:(j+1)*spatial].abs().sum() for j in range(n)])
            D = W_out.shape[0] * spatial
            for j in range(n): wof.append(W_out[:, j*spatial:(j+1)*spatial].reshape(-1))
        else:
            phi_out = W_out.abs().sum(dim=0)
            if phi_out.shape[0] > n: phi_out = phi_out[:n]
            D = W_out.shape[0]
            for j in range(n): wof.append(W_out[:, j])
        base = alpha * (phi_in * phi_out) + (1 - alpha) * (phi_in + phi_out)
        scs = torch.zeros(n)
        if D > 0 and wof:
            W_flat = torch.stack(wof); M = min(64, n)
            si = torch.arange(n) if n <= M else torch.randperm(n)[:M]
            _, _, VT = torch.linalg.svd(W_flat[si].float(), full_matrices=False)
            V = VT.T; nc = V.shape[1]; p = min(4, M, D, nc)
            if p < nc:
                scs = (W_flat.float() @ V[:, p:].float()).pow(2).sum(dim=1)
        return base + lam * scs
    def fitness(c):
        r, a, l = c[:n_layers], c[n_layers:2*n_layers], c[2*n_layers:]
        fe, ta, tk = 0.0, 0, 0
        for i, info in enumerate(layer_info):
            imp = compute_importance(info, a[i], l[i]); n = imp.shape[0]
            if n == 0: continue
            k = max(4, int(info['n_alive'] * r[i]))
            ti = imp.sum().item()
            if ti > 1e-12:
                _, topk = imp.topk(min(k, n))
                fe += 1.0 - imp[topk].sum().item() / ti
            ta += info['n_alive']; tk += k
        return (1.0 - fe, 1.0 - tk / max(ta, 1))
    pop_size, gens, sigma, cl = 10, 15, 0.12, 3 * n_layers
    pop = [np.concatenate([np.random.uniform(0.1, 0.8, n_layers), np.random.uniform(0.0, 1.0, n_layers), np.random.uniform(0.0, 1.0, n_layers)]) for _ in range(pop_size)]
    bg, bf = pop[0].copy(), -np.inf
    for gen in range(gens):
        sc = [(g, fitness(g)) for g in pop]
        f0 = [(g, f) for i, (g, f) in enumerate(sc) if not any(all(sc[j][1][k] >= f[k] for k in range(2)) and any(sc[j][1][k] > f[k] for k in range(2)) for j in range(len(sc)) if j != i)]
        for g, f in f0:
            if f[0] > bf: bf = f[0]; bg = g.copy()
        parents = [g for g, _ in f0]
        if len(parents) < 2:
            ss = sorted(sc, key=lambda x: x[1][0], reverse=True)
            parents = [ss[0][0], ss[min(1, len(ss)-1)][0]]
        st = sigma * (1 - gen / gens * 0.5)
        np_ = list(parents)
        while len(np_) < pop_size:
            p = parents[np.random.randint(len(parents))]
            ch = p + np.random.randn(cl) * st
            ch[:n_layers] = np.clip(ch[:n_layers], 0.1, 0.8)
            ch[n_layers:2*n_layers] = np.clip(ch[n_layers:2*n_layers], 0.0, 1.0)
            ch[2*n_layers:] = np.clip(ch[2*n_layers:], 0.0, 1.0)
            np_.append(ch)
        pop = np_
    sels = []
    for i, info in enumerate(layer_info):
        imp = compute_importance(info, bg[n_layers+i], bg[2*n_layers+i])
        n = imp.shape[0]
        if n == 0: sels.append([]); continue
        k = max(4, int(info['n_alive'] * bg[i]))
        _, idx = imp.topk(min(k, n))
        sels.append(sorted(idx.cpu().tolist()))
    return sels


# ────── explore07: MOEA ──────
def _compress_fn_moea(layer_info):
    n_layers = len(layer_info)
    if n_layers == 0:
        return []
    def compute_phi(info):
        W_in, W_out, spatial = info['W_in'], info['W_out'], info['spatial']
        n = W_in.shape[0]
        phi_in = W_in.abs().reshape(n, -1).sum(dim=1)
        if W_out.dim() == 4:
            phi_out = W_out.abs().sum(dim=(0, 2, 3))
        elif spatial > 1:
            phi_out = torch.stack([W_out[:, j*spatial:(j+1)*spatial].abs().sum() for j in range(n)])
        else:
            phi_out = W_out.abs().sum(dim=0)
            if phi_out.shape[0] > n: phi_out = phi_out[:n]
        return phi_in * phi_out
    def fitness(r):
        fe, ta, tk = 0.0, 0, 0
        for i, info in enumerate(layer_info):
            phi = compute_phi(info); n = phi.shape[0]
            k = max(4, int(info['n_alive'] * r[i]))
            tf = phi.sum().item()
            if tf > 1e-12:
                _, topk = phi.topk(min(k, n))
                fe += 1.0 - phi[topk].sum().item() / tf
            ta += info['n_alive']; tk += k
        return (1.0 - fe, 1.0 - tk / max(ta, 1))
    def dom(a, b):
        return all(b[k] >= a[k] for k in range(2)) and any(b[k] > a[k] for k in range(2))
    pop_size, gens, sigma = 10, 15, 0.12
    pop = [np.random.uniform(0.1, 0.8, n_layers) for _ in range(pop_size)]
    bg, bf = pop[0].copy(), -np.inf
    for gen in range(gens):
        sc = [(g, fitness(g)) for g in pop]
        f0 = [(g, f) for i, (g, f) in enumerate(sc) if not any(dom(f, sc[j][1]) for j in range(len(sc)) if j != i)]
        for g, f in f0:
            if f[0] > bf: bf = f[0]; bg = g.copy()
        parents = [g for g, _ in f0]
        if len(parents) < 2:
            ss = sorted(sc, key=lambda x: x[1][0], reverse=True)
            parents = [ss[0][0], ss[min(1, len(ss)-1)][0]]
        st = sigma * (1 - gen / gens * 0.5)
        np_ = list(parents)
        while len(np_) < pop_size:
            p = parents[np.random.randint(len(parents))]
            np_.append(np.clip(p + np.random.randn(n_layers) * st, 0.1, 0.8))
        pop = np_
    sels = []
    for i, info in enumerate(layer_info):
        phi = compute_phi(info); n = phi.shape[0]
        k = max(4, int(info['n_alive'] * bg[i]))
        _, idx = phi.topk(min(k, n))
        sels.append(sorted(idx.cpu().tolist()))
    return sels


# ═══════════════════════════════════════════
#  Method registry
# ═══════════════════════════════════════════

METHODS = {
    'GFCS':      _gfcs_baseline,
    'NeuronRem': _neuron_removal,
    'EAIB_e25':  _make_compress_one(_compress_fn_eaib),
    'EPSS_e48':  _make_compress_one(_compress_fn_epss),
    'MOEA_e07':  _make_compress_one(_compress_fn_moea),
}

METHOD_LABELS = {
    'GFCS':      'GFCS (baseline)',
    'NeuronRem': 'Neuron Removal',
    'EAIB_e25':  'EAIB (explore25)',
    'EPSS_e48':  'EPSS (explore48)',
    'MOEA_e07':  'MOEA (explore07)',
}

# Only CNN and ResNet
TARGET_ARCHS = ['cnn', 'resnet']


# ═══════════════════════════════════════════
#  Main benchmark
# ═══════════════════════════════════════════
def main():
    print("=" * 80)
    print("  Ex09v2 Part 2: CNN & ResNet Compression Benchmark")
    print("  5 methods × 2 architectures × 2 seeds = 20 measurements")
    print("  Pruning: TESA-26 iter60 @ 95% sparsity (FashionMNIST)")
    print("=" * 80)

    # Warmup caches
    print("\n[Phase 1] Warming caches...", flush=True)
    warm_compression_cache()

    all_results = []
    method_order = ['GFCS', 'NeuronRem', 'EAIB_e25', 'EPSS_e48', 'MOEA_e07']

    for arch_key in TARGET_ARCHS:
        for seed in SEEDS:
            print(f"\n{'═'*70}")
            print(f"  {arch_key.upper()}, seed={seed}, sparsity=95%")
            print(f"{'═'*70}")

            for mk in method_order:
                compress_fn = METHODS[mk]
                t0 = time.time()

                result = eval_compression_task(compress_fn, arch_key, seed)
                elapsed = time.time() - t0

                result['method'] = mk
                all_results.append(result)

                if result['error']:
                    print(f"  ✗ {METHOD_LABELS[mk]:25s} ERROR: {result['error'][:80]}")
                else:
                    print(f"  ✓ {METHOD_LABELS[mk]:25s} "
                          f"RPR={result['rpr']:.4f}  "
                          f"RCU={result['rcu_method']:>7.0f}  "
                          f"Size={result['model_size']:>7d}/{result['teacher_params']:>7d}  "
                          f"Comp={result['compression_ratio']:.1f}x  "
                          f"Speed={result['speedup']:.2f}x  "
                          f"F1={result['f1']:.4f}  "
                          f"({elapsed:.1f}s)")

    # ═══════════════════════════════════════════
    #  Summary Tables
    # ═══════════════════════════════════════════
    from collections import defaultdict

    print(f"\n\n{'='*120}")
    print("  SUMMARY: CNN & ResNet — 5 methods × 2 seeds (per architecture)")
    print(f"{'='*120}")

    for arch_key in TARGET_ARCHS:
        print(f"\n  ── {arch_key.upper()} (FashionMNIST, 95% sparsity) ──\n")
        print(f"  {'Method':25s} {'RPR':>6s} {'F1':>6s} {'Comp×':>6s} {'Speed×':>7s} "
              f"{'Params':>8s} {'RCU_meth':>8s}")
        print(f"  {'─'*75}")

        for mk in method_order:
            arch_results = [r for r in all_results
                           if r.get('method') == mk and r.get('arch') == arch_key
                           and r.get('error') is None]
            if not arch_results:
                print(f"  {METHOD_LABELS[mk]:25s} FAILED")
                continue

            rpr = np.mean([r['rpr'] for r in arch_results])
            f1 = np.mean([r['f1'] for r in arch_results])
            comp = np.mean([r['compression_ratio'] for r in arch_results])
            speed = np.mean([r['speedup'] for r in arch_results])
            params = int(np.mean([r['model_size'] for r in arch_results]))
            rcu = np.mean([r['rcu_method'] for r in arch_results])
            marker = ' ◄' if mk == 'GFCS' else ''
            print(f"  {METHOD_LABELS[mk]:25s} {rpr:.4f} {f1:.4f} {comp:6.1f} {speed:7.2f} "
                  f"{params:>8,} {rcu:8.1f}{marker}")

    # ── Combined MLP + CNN + ResNet summary ──
    print(f"\n\n{'='*120}")
    print("  EFFICIENCY COMPARISON: EAIB vs GFCS (per architecture)")
    print(f"{'='*120}\n")

    for arch_key in TARGET_ARCHS:
        gfcs = [r for r in all_results if r.get('method') == 'GFCS' and r.get('arch') == arch_key and not r.get('error')]
        eaib = [r for r in all_results if r.get('method') == 'EAIB_e25' and r.get('arch') == arch_key and not r.get('error')]
        if gfcs and eaib:
            g_rpr = np.mean([r['rpr'] for r in gfcs])
            e_rpr = np.mean([r['rpr'] for r in eaib])
            g_rcu = np.mean([r['rcu_method'] for r in gfcs])
            e_rcu = np.mean([r['rcu_method'] for r in eaib])
            g_sp = np.mean([r['speedup'] for r in gfcs])
            e_sp = np.mean([r['speedup'] for r in eaib])
            g_comp = np.mean([r['compression_ratio'] for r in gfcs])
            e_comp = np.mean([r['compression_ratio'] for r in eaib])

            rcu_saving = (1 - e_rcu / g_rcu) * 100 if g_rcu > 0 else 0
            rpr_diff = (e_rpr - g_rpr) / g_rpr * 100

            print(f"  {arch_key.upper():8s}  RPR: {g_rpr:.4f}→{e_rpr:.4f} ({rpr_diff:+.1f}%)  "
                  f"RCU: {g_rcu:.0f}→{e_rcu:.0f} ({rcu_saving:+.0f}%)  "
                  f"Comp: {g_comp:.1f}→{e_comp:.1f}×  "
                  f"Speed: {g_sp:.2f}→{e_sp:.2f}×")

    # Save
    os.makedirs('results', exist_ok=True)
    with open('results/ex09v2_cnn_resnet.json', 'w') as f:
        json.dump({'results': all_results, 'method_labels': METHOD_LABELS}, f, indent=2, default=str)
    print(f"\n  Saved: results/ex09v2_cnn_resnet.json")


if __name__ == '__main__':
    main()
