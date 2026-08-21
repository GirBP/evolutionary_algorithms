#!/usr/bin/env python3
"""
Ex09v2 Standalone: GFCS vs EAIB — Fair comparison on CNN & ResNet.
Compression methods written from scratch in one file.
Uses pre-trained cached teachers/sparse from dense_autosearch.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, copy, time
from sklearn.metrics import f1_score
from pathlib import Path
torch.set_num_threads(1)

# Import ONLY model classes + data loaders from dense_autosearch (NO warm_compression_cache)
DENSE_DIR = Path(__file__).resolve().parent.parent / "dense_autosearch"
sys.path.insert(0, str(DENSE_DIR.parent / "common"))
sys.path.insert(0, str(DENSE_DIR))
from compress_prepare import (
    GenericCNN, GenericResNet, CompactCNN, CompactResNet,
    get_loaders, eval_f1 as _eval_f1, micro_finetune_compact,
)

SPARSE_DIR = DENSE_DIR / "data" / "sparse"
SPATIAL = 49  # 7*7

# ═══════════════════════════════════════════
#  Load cached models (skip training!)
# ═══════════════════════════════════════════
def load_cached(arch, seed):
    teacher_path = SPARSE_DIR / f"teacher_generic_{arch}_s{seed}.pt"
    sparse_path = SPARSE_DIR / f"sparse_{arch}_s{seed}.pt"
    assert teacher_path.exists(), f"Missing {teacher_path}"
    assert sparse_path.exists(), f"Missing {sparse_path}"
    teacher_state = torch.load(teacher_path, weights_only=True)
    sparse_state = torch.load(sparse_path, weights_only=True)

    if arch == 'cnn':
        teacher = GenericCNN([32, 64], [128], 10)
        sparse = GenericCNN([32, 64], [128], 10)
    else:
        teacher = GenericResNet(32, 64, [256, 128], 10)
        sparse = GenericResNet(32, 64, [256, 128], 10)

    teacher.load_state_dict(teacher_state)
    sparse.load_state_dict(sparse_state)
    return teacher, sparse


# ═══════════════════════════════════════════
#  GFCS (from explore00 — rewritten from scratch)
#  Core: φ = ||W_in||₁ · ||W_out||₁
#  EA: scalar fitness, pop=20, gens=30
# ═══════════════════════════════════════════

def _compute_phi(W_in, W_out, spatial=1):
    """Flow importance for each neuron/channel."""
    n = W_in.shape[0]
    if W_in.dim() == 4:
        phi_in = W_in.abs().sum(dim=(1, 2, 3))
    else:
        phi_in = W_in.abs().sum(dim=1)

    if W_out.dim() == 4:
        phi_out = W_out.abs().sum(dim=(0, 2, 3))
    elif spatial > 1:
        phi_out = torch.stack([
            W_out[:, j * spatial:(j + 1) * spatial].abs().sum() for j in range(n)
        ])
    else:
        phi_out = W_out.abs().sum(dim=0)
        if phi_out.shape[0] > n:
            phi_out = phi_out[:n]
    return phi_in * phi_out


def _get_layers_cnn(model):
    """Extract (W_in, W_out, spatial) tuples for CNN."""
    return [
        (model.conv1.weight.data, model.conv2.weight.data, 1),
        (model.conv2.weight.data, model.fcs[0].weight.data, SPATIAL),
    ] + [
        (model.fcs[i].weight.data, model.fcs[i + 1].weight.data, 1)
        for i in range(len(model.fcs) - 1)
    ]


def _get_layers_resnet(model):
    """Extract (phi, n_alive) for ResNet block channels + FC layers."""
    c1 = model.conv1.weight.shape[0]
    phi_c1 = torch.zeros(c1)
    for i in range(c1):
        phi_in = model.conv1.weight.data[i].abs().sum()
        phi_out = (model.conv2.weight.data[:, i].abs().sum() +
                   model.conv3.weight.data[:, i].abs().sum() +
                   model.conv_skip.weight.data[:, i].abs().sum() +
                   model.conv4.weight.data[:, i].abs().sum())
        phi_c1[i] = phi_in * phi_out

    c2 = model.conv4.weight.shape[0]
    phi_c2 = torch.zeros(c2)
    for i in range(c2):
        phi_in = (model.conv_skip.weight.data[i].abs().sum() +
                  model.conv4.weight.data[i].abs().sum())
        phi_out = (model.conv5.weight.data[:, i].abs().sum() +
                   model.fcs[0].weight.data[:, i * SPATIAL:(i + 1) * SPATIAL].abs().sum())
        phi_c2[i] = phi_in * phi_out

    layers = [(phi_c1, c1), (phi_c2, c2)]
    for i in range(len(model.fcs) - 1):
        phi = _compute_phi(model.fcs[i].weight.data, model.fcs[i + 1].weight.data)
        layers.append((phi, model.fcs[i].out_features))
    return layers


def gfcs_compress(sparse_model, arch):
    """GFCS: flow importance, scalar EA, pop=20 gens=30."""
    if arch == 'cnn':
        layers = _get_layers_cnn(sparse_model)
        phis = [_compute_phi(W_in, W_out, sp) for W_in, W_out, sp in layers]
    else:
        resnet_layers = _get_layers_resnet(sparse_model)
        phis = [phi for phi, _ in resnet_layers]

    alives = [max(int((p > 1e-8).sum().item()), 4) for p in phis]
    n_layers = len(phis)

    def fitness(ratios):
        err, ta, tk = 0.0, 0, 0
        for phi, r, a in zip(phis, ratios, alives):
            n = phi.shape[0]; k = max(4, int(a * r)); ta += a; tk += k
            tf = phi.sum().item()
            if tf > 1e-12:
                _, topk = phi.topk(min(k, n))
                err += 1.0 - phi[topk].sum().item() / tf
        return -err - 0.5 * (tk / max(ta, 1))

    pop = [np.random.uniform(0.1, 0.8, n_layers) for _ in range(20)]
    best_g, best_f = pop[0].copy(), -1e9
    for gen in range(30):
        scored = sorted([(g, fitness(g)) for g in pop], key=lambda x: x[1], reverse=True)
        if scored[0][1] > best_f:
            best_g, best_f = scored[0][0].copy(), scored[0][1]
        parents = [s[0] for s in scored[:5]]
        new_pop = list(parents)
        sig = 0.12 * (1 - gen / 30 * 0.5)
        while len(new_pop) < 20:
            new_pop.append(np.clip(parents[np.random.randint(5)] + np.random.randn(n_layers) * sig, 0.1, 0.8))
        pop = new_pop

    sels = []
    for i, (phi, a) in enumerate(zip(phis, alives)):
        k = max(4, int(a * best_g[i]))
        _, idx = phi.topk(min(k, phi.shape[0]))
        sels.append(sorted(idx.tolist()))

    return _build_compact(sparse_model, sels, arch)


# ═══════════════════════════════════════════
#  EAIB (from explore25 — rewritten from scratch)
#  Core: imp = α·(φ_in·φ_out) + (1-α)·(φ_in+φ_out)
#  EA: bi-objective Pareto, pop=10, gens=15
# ═══════════════════════════════════════════

def _compute_blended_phi(W_in, W_out, alpha, spatial=1):
    """Blended importance: α·FI + (1-α)·MI."""
    n = W_in.shape[0]
    if W_in.dim() == 4:
        phi_in = W_in.abs().sum(dim=(1, 2, 3))
    else:
        phi_in = W_in.abs().sum(dim=1)

    if W_out.dim() == 4:
        phi_out = W_out.abs().sum(dim=(0, 2, 3))
    elif spatial > 1:
        phi_out = torch.stack([
            W_out[:, j * spatial:(j + 1) * spatial].abs().sum() for j in range(n)
        ])
    else:
        phi_out = W_out.abs().sum(dim=0)
        if phi_out.shape[0] > n:
            phi_out = phi_out[:n]

    fi = phi_in * phi_out
    mi = phi_in + phi_out
    return alpha * fi + (1 - alpha) * mi


def eaib_compress(sparse_model, arch):
    """EAIB: blended importance, Pareto EA, pop=10 gens=15."""
    if arch == 'cnn':
        layers = _get_layers_cnn(sparse_model)
    else:
        # For ResNet, need disaggregated layers for blended importance
        layers = _get_resnet_layer_tuples(sparse_model)

    n_layers = len(layers)
    # Get alives from pure flow importance
    alives = []
    for W_in, W_out, sp in layers:
        phi = _compute_phi(W_in, W_out, sp)
        alives.append(max(int((phi > 1e-8).sum().item()), 4))

    def fitness(chrom):
        ratios = chrom[:n_layers]
        alphas = chrom[n_layers:]
        err, ta, tk = 0.0, 0, 0
        for i, (W_in, W_out, sp) in enumerate(layers):
            imp = _compute_blended_phi(W_in, W_out, alphas[i], sp)
            n = imp.shape[0]; k = max(4, int(alives[i] * ratios[i]))
            ta += alives[i]; tk += k
            ti = imp.sum().item()
            if ti > 1e-12:
                _, topk = imp.topk(min(k, n))
                err += 1.0 - imp[topk].sum().item() / ti
        return (1.0 - err, 1.0 - tk / max(ta, 1))

    cl = 2 * n_layers
    pop = [np.concatenate([np.random.uniform(0.1, 0.8, n_layers),
                           np.random.uniform(0.0, 1.0, n_layers)]) for _ in range(10)]
    bg, bf = pop[0].copy(), -np.inf

    for gen in range(15):
        sc = [(g, fitness(g)) for g in pop]
        f0 = []
        for i, (gi, fi) in enumerate(sc):
            dominated = False
            for j, (gj, fj) in enumerate(sc):
                if i != j and all(fj[k] >= fi[k] for k in range(2)) and \
                        any(fj[k] > fi[k] for k in range(2)):
                    dominated = True; break
            if not dominated:
                f0.append((gi, fi))
        for g, f in f0:
            if f[0] > bf: bf = f[0]; bg = g.copy()

        parents = [g for g, _ in f0]
        if len(parents) < 2:
            ss = sorted(sc, key=lambda x: x[1][0], reverse=True)
            parents = [ss[0][0], ss[min(1, len(ss) - 1)][0]]

        sig = 0.12 * (1 - gen / 15 * 0.5)
        new_pop = list(parents)
        while len(new_pop) < 10:
            p = parents[np.random.randint(len(parents))]
            ch = p + np.random.randn(cl) * sig
            ch[:n_layers] = np.clip(ch[:n_layers], 0.1, 0.8)
            ch[n_layers:] = np.clip(ch[n_layers:], 0.0, 1.0)
            new_pop.append(ch)
        pop = new_pop

    # Final selection
    best_alphas = bg[n_layers:]
    sels = []
    for i, (W_in, W_out, sp) in enumerate(layers):
        imp = _compute_blended_phi(W_in, W_out, best_alphas[i], sp)
        k = max(4, int(alives[i] * bg[i]))
        _, idx = imp.topk(min(k, imp.shape[0]))
        sels.append(sorted(idx.tolist()))

    return _build_compact(sparse_model, sels, arch)


def _get_resnet_layer_tuples(model):
    """Get (W_in, W_out, spatial) tuples for ResNet for blended importance."""
    c1 = model.conv1.weight.shape[0]
    # Block1: aggregate W_out for conv1
    W_out_b1 = torch.zeros(1, c1)
    for i in range(c1):
        W_out_b1[0, i] = (model.conv2.weight.data[:, i].abs().sum() +
                          model.conv3.weight.data[:, i].abs().sum() +
                          model.conv_skip.weight.data[:, i].abs().sum() +
                          model.conv4.weight.data[:, i].abs().sum())

    c2 = model.conv4.weight.shape[0]
    W_in_b2 = torch.zeros(c2, 1)
    W_out_b2 = torch.zeros(1, c2)
    for i in range(c2):
        W_in_b2[i, 0] = (model.conv_skip.weight.data[i].abs().sum() +
                         model.conv4.weight.data[i].abs().sum())
        W_out_b2[0, i] = (model.conv5.weight.data[:, i].abs().sum() +
                          model.fcs[0].weight.data[:, i * SPATIAL:(i + 1) * SPATIAL].abs().sum())

    layers = [
        (model.conv1.weight.data, W_out_b1, 1),
        (W_in_b2, W_out_b2, 1),
    ]
    for i in range(len(model.fcs) - 1):
        layers.append((model.fcs[i].weight.data, model.fcs[i + 1].weight.data, 1))
    return layers


# ═══════════════════════════════════════════
#  Build compact (shared by both methods)
# ═══════════════════════════════════════════

def _copy_bn(src, dst, idx):
    if src.weight is not None: dst.weight.data.copy_(src.weight.data[idx])
    if src.bias is not None: dst.bias.data.copy_(src.bias.data[idx])
    if hasattr(src, 'running_mean') and src.running_mean is not None:
        dst.running_mean.copy_(src.running_mean[idx])
        dst.running_var.copy_(src.running_var[idx])


def _build_compact(sparse_model, selections, arch):
    if arch == 'cnn':
        return _build_compact_cnn(sparse_model, selections)
    else:
        return _build_compact_resnet(sparse_model, selections)


def _build_compact_cnn(model, selections):
    idx1, idx2 = selections[0], selections[1]
    fc_sels = selections[2:]
    compact = CompactCNN(channels=[len(idx1), len(idx2)],
                         hiddens=[len(s) for s in fc_sels],
                         n_classes=model.n_classes)
    with torch.no_grad():
        compact.conv1.weight.data.copy_(model.conv1.weight.data[idx1])
        if model.conv1.bias is not None:
            compact.conv1.bias.data.copy_(model.conv1.bias.data[idx1])
        compact.conv2.weight.data.copy_(model.conv2.weight.data[idx2][:, idx1])
        if model.conv2.bias is not None:
            compact.conv2.bias.data.copy_(model.conv2.bias.data[idx2])
        cidx = []
        for j in idx2: cidx.extend(range(j * SPATIAL, (j + 1) * SPATIAL))
        prev = torch.tensor(cidx, dtype=torch.long)
        for i in range(len(model.fcs)):
            if i < len(fc_sels):
                compact.fcs[i].weight.data.copy_(model.fcs[i].weight.data[fc_sels[i]][:, prev])
                if model.fcs[i].bias is not None:
                    compact.fcs[i].bias.data.copy_(model.fcs[i].bias.data[fc_sels[i]])
                prev = fc_sels[i]
            else:
                compact.fcs[i].weight.data.copy_(model.fcs[i].weight.data[:, prev])
                if model.fcs[i].bias is not None:
                    compact.fcs[i].bias.data.copy_(model.fcs[i].bias.data)
    return compact


def _build_compact_resnet(model, selections):
    idx1, idx2 = selections[0], selections[1]
    fc_sels = selections[2:]
    compact = CompactResNet(c1=len(idx1), c2=len(idx2),
                            hiddens=[len(s) for s in fc_sels],
                            n_classes=model.n_classes)
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
        for j in idx2: cidx.extend(range(j * SPATIAL, (j + 1) * SPATIAL))
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
#  Benchmark
# ═══════════════════════════════════════════
def main():
    print("=" * 70, flush=True)
    print("  STANDALONE FAIR: GFCS vs EAIB on CNN & ResNet", flush=True)
    print("  Pre-trained teachers from cache. Compression from scratch.", flush=True)
    print("=" * 70, flush=True)

    for arch in ['cnn', 'resnet']:
        print(f"\n\n{'═' * 60}", flush=True)
        print(f"  Architecture: {arch.upper()}", flush=True)
        print(f"{'═' * 60}", flush=True)

        for seed in [42, 123]:
            print(f"\n  ── seed={seed} ──", flush=True)
            teacher, sparse = load_cached(arch, seed)
            tr_dl, te_dl = get_loaders(arch, seed)

            f1_teacher = _eval_f1(teacher, te_dl)
            f1_sparse = _eval_f1(sparse, te_dl)
            print(f"  Teacher: F1={f1_teacher:.4f}  params={teacher.count_params():,}", flush=True)
            print(f"  Sparse:  F1={f1_sparse:.4f}", flush=True)

            for method_name, method_fn in [('GFCS', gfcs_compress), ('EAIB', eaib_compress)]:
                np.random.seed(seed); torch.manual_seed(seed)
                t0 = time.thread_time_ns()
                compact = method_fn(copy.deepcopy(sparse), arch)
                t_ms = (time.thread_time_ns() - t0) / 1e6

                f1_pre = _eval_f1(compact, te_dl)
                micro_finetune_compact(compact, tr_dl, batches=25)
                f1_post = _eval_f1(compact, te_dl)
                rpr = f1_post / f1_teacher
                comp = teacher.count_params() / compact.count_params()

                print(f"    {method_name:5s}: pre={f1_pre:.4f} post={f1_post:.4f} "
                      f"RPR={rpr:.4f} comp={comp:.1f}× "
                      f"params={compact.count_params():>8,}  time={t_ms:.0f}ms", flush=True)

    print("\n=== DONE ===", flush=True)


if __name__ == '__main__':
    main()
