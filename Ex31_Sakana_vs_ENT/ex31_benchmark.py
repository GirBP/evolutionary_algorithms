#!/usr/bin/env python3
"""Beat Sakana: ENT-FT vs Sakana-CMA on complementary CIFAR-10 merge.

Scenario: Shared base → fine-tune → complementary merge (0-4 vs 5-9).
This is Sakana's BEST CASE (shared base exists).

Methods:
  1. Weight Average (baseline)
  2. Task Arithmetic (θ_base + λ_A·τ_A + λ_B·τ_B)
  3. TIES-Merging (trim + elect sign + merge aligned)
  4. DARE-TIES (drop & rescale + TIES)
  5. Sakana-CMA (CMA-ES per-layer α for DARE-TIES)
  6. ENT (topology selection + per-class routing)
  7. ENT-FT (ENT + LogReg calibration)
"""

import sys, os, time, json, copy, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from pathlib import Path

warnings.filterwarnings('ignore')
torch.set_num_threads(4)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════
# MODEL: CIFAR-10 CNN (~94K params)
# ══════════════════════════════════════════

class CifarCNN(nn.Module):
    def __init__(self, n_out=10, ch1=32, ch2=64, ch3=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch1, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch1)
        self.conv2 = nn.Conv2d(ch1, ch2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch2)
        self.conv3 = nn.Conv2d(ch2, ch3, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(ch3)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch3, n_out)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 32→16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))    # 16→8
        x = F.relu(self.bn3(self.conv3(x)))                # 8
        x = self.gap(x).flatten(1)                         # 128
        return self.fc(x)

    def extract_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.gap(x).flatten(1)


# ══════════════════════════════════════════
# DATA: CIFAR-10
# ══════════════════════════════════════════

def load_cifar10():
    """Load CIFAR-10 with torchvision."""
    import gc
    from torchvision import datasets, transforms
    np.random.seed(0)  # reproducible subset
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train = datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=tf)
    test = datasets.CIFAR10('/tmp/cifar10', train=False, download=True, transform=tf)

    # Use subsets to avoid OOM
    n_train = min(10000, len(train))
    n_test = min(2000, len(test))
    idx_tr = np.random.permutation(len(train))[:n_train]
    idx_te = np.random.permutation(len(test))[:n_test]
    train_xs = torch.stack([train[i][0] for i in idx_tr])
    train_ys = torch.tensor([train[i][1] for i in idx_tr])
    test_xs = torch.stack([test[i][0] for i in idx_te])
    test_ys = torch.tensor([test[i][1] for i in idx_te])
    del train, test
    gc.collect()
    return train_xs, train_ys, test_xs, test_ys


def filter_classes(xs, ys, classes):
    mask = torch.zeros(len(ys), dtype=torch.bool)
    for c in classes:
        mask |= (ys == c)
    return xs[mask], ys[mask]


def make_val_set(train_xs, train_ys, per_class=25):
    """Extract small validation set from training data."""
    val_idx, rest_idx = [], []
    for c in range(10):
        c_idx = (train_ys == c).nonzero(as_tuple=True)[0].tolist()
        np.random.shuffle(c_idx)
        val_idx.extend(c_idx[:per_class])
        rest_idx.extend(c_idx[per_class:])
    return (train_xs[rest_idx], train_ys[rest_idx],
            train_xs[val_idx], train_ys[val_idx])


# ══════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════

def train_model(model, xs, ys, epochs=30, lr=0.001, bs=128, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(xs)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            logits = model(xs[idx])
            loss = F.cross_entropy(logits, ys[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()
        if verbose and (ep+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(xs[:2000]).argmax(1) == ys[:2000]).float().mean().item()
            print(f"    ep {ep+1}/{epochs}: loss={total_loss/(n/bs):.3f} acc={acc:.3f}")
    model.eval()
    return model


# ══════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════

def eval_model(model, xs, ys, all_classes=list(range(10))):
    model.eval()
    with torch.no_grad():
        # Batch inference to avoid OOM
        preds_list = []
        for i in range(0, len(xs), 512):
            preds_list.append(model(xs[i:i+512]).argmax(1))
        preds = torch.cat(preds_list)
    per_class = {}
    for c in all_classes:
        mask = ys == c
        if mask.sum() > 0:
            per_class[c] = (preds[mask] == c).float().mean().item()
        else:
            per_class[c] = 0.0
    vals = [per_class[c] for c in all_classes]
    mn, mx = min(vals), max(vals)
    return {
        'accuracy': round(float(np.mean(vals)), 4),
        'balance': round(float(mn/mx) if mx > 0 else 0, 4),
        'min_class': round(float(mn), 4),
        'n_ok': sum(1 for v in vals if v > 0.1),
        'n_total': len(all_classes),
        'n_params': sum(p.numel() for p in model.parameters()),
        'per_class': {c: round(float(per_class[c]), 4) for c in all_classes},
    }


# ══════════════════════════════════════════
# MERGE METHODS
# ══════════════════════════════════════════

# --- 1. Weight Average ---
def merge_weight_average(sd_A, sd_B, alpha=0.5):
    merged = OrderedDict()
    for k in sd_A:
        merged[k] = alpha * sd_A[k] + (1 - alpha) * sd_B[k]
    return merged


# --- 2. Task Arithmetic ---
def merge_task_arithmetic(sd_base, sd_A, sd_B, lam_A=0.5, lam_B=0.5):
    merged = OrderedDict()
    for k in sd_base:
        tau_A = sd_A[k] - sd_base[k]
        tau_B = sd_B[k] - sd_base[k]
        merged[k] = sd_base[k] + lam_A * tau_A + lam_B * tau_B
    return merged


# --- 3. TIES-Merging ---
def merge_ties(sd_base, sd_A, sd_B, density=0.3):
    merged = OrderedDict()
    for k in sd_base:
        tau_A = (sd_A[k] - sd_base[k]).float()
        tau_B = (sd_B[k] - sd_base[k]).float()

        # Step 1: Trim — keep top-density% by magnitude
        for tau in [tau_A, tau_B]:
            flat = tau.abs().flatten()
            if len(flat) > 0:
                threshold = torch.quantile(flat, 1 - density)
                tau[tau.abs() < threshold] = 0

        # Step 2: Elect sign — majority vote
        signs = torch.sign(tau_A) + torch.sign(tau_B)
        elected = torch.sign(signs)  # tie → 0

        # Step 3: Merge aligned only
        sum_tau = torch.zeros_like(tau_A)
        count = torch.zeros_like(tau_A)
        for tau in [tau_A, tau_B]:
            aligned = (torch.sign(tau) == elected) & (elected != 0)
            sum_tau[aligned] += tau[aligned]
            count[aligned] += 1
        count = count.clamp(min=1)
        merged_tau = sum_tau / count
        merged[k] = sd_base[k] + merged_tau
    return merged


# --- 4. DARE-TIES ---
def merge_dare_ties(sd_base, sd_A, sd_B, density=0.3, dare_p=0.5):
    merged = OrderedDict()
    for k in sd_base:
        tau_A = (sd_A[k] - sd_base[k]).float()
        tau_B = (sd_B[k] - sd_base[k]).float()

        # DARE: random drop + rescale
        for tau in [tau_A, tau_B]:
            mask = torch.bernoulli(torch.ones_like(tau) * (1 - dare_p))
            tau.mul_(mask / (1 - dare_p + 1e-8))

        # TIES on DARE'd task vectors
        for tau in [tau_A, tau_B]:
            flat = tau.abs().flatten()
            if len(flat) > 0 and flat.max() > 0:
                nonzero = flat[flat > 0]
                if len(nonzero) > 0:
                    threshold = torch.quantile(nonzero, max(0, 1 - density))
                    tau[tau.abs() < threshold] = 0

        signs = torch.sign(tau_A) + torch.sign(tau_B)
        elected = torch.sign(signs)
        sum_tau = torch.zeros_like(tau_A)
        count = torch.zeros_like(tau_A)
        for tau in [tau_A, tau_B]:
            aligned = (torch.sign(tau) == elected) & (elected != 0)
            sum_tau[aligned] += tau[aligned]
            count[aligned] += 1
        count = count.clamp(min=1)
        merged[k] = sd_base[k] + sum_tau / count
    return merged


# --- 5. Sakana-CMA (per-layer α with CMA-ES) ---
def merge_sakana_cma(sd_base, sd_A, sd_B, val_xs, val_ys, n_out=10,
                     pop_size=16, n_gens=30):
    """Sakana-style: CMA-ES optimizes per-layer λ for task arithmetic."""
    # Group params by layer prefix
    layer_prefixes = []
    seen = set()
    for k in sd_base:
        prefix = k.rsplit('.', 1)[0]  # e.g., conv1, bn1, fc
        if prefix not in seen:
            seen.add(prefix)
            layer_prefixes.append(prefix)

    n_layers = len(layer_prefixes)
    dim = 2 * n_layers  # λ_A and λ_B per layer

    def fitness(params):
        lam_A = params[:n_layers]
        lam_B = params[n_layers:]
        merged = OrderedDict()
        for i, prefix in enumerate(layer_prefixes):
            for k in sd_base:
                if k.startswith(prefix + '.') or k == prefix:
                    tau_A = sd_A[k] - sd_base[k]
                    tau_B = sd_B[k] - sd_base[k]
                    merged[k] = sd_base[k] + lam_A[i] * tau_A + lam_B[i] * tau_B
        model = CifarCNN(n_out=n_out)
        model.load_state_dict(merged)
        model.eval()
        with torch.no_grad():
            preds = model(val_xs).argmax(1)
        acc = (preds == val_ys).float().mean().item()
        per_cls = []
        for c in range(n_out):
            msk = val_ys == c
            if msk.sum() > 0:
                per_cls.append((preds[msk] == c).float().mean().item())
        min_c = min(per_cls) if per_cls else 0
        return 0.5 * acc + 0.3 * min_c + 0.2 * np.mean(per_cls)

    # Simple CMA-ES
    mean = np.ones(dim) * 0.5
    sigma = 0.3
    cov = np.eye(dim)
    best_fit, best_params = -1, mean.copy()

    for gen in range(n_gens):
        samples = np.random.multivariate_normal(mean, sigma**2 * cov, pop_size)
        fits = [fitness(s) for s in samples]
        # Select top half
        elite_idx = np.argsort(fits)[-pop_size//2:]
        elite = samples[elite_idx]
        if max(fits) > best_fit:
            best_fit = max(fits)
            best_params = samples[np.argmax(fits)].copy()
        # Update
        mean = elite.mean(axis=0)
        diff = elite - mean
        cov = (diff.T @ diff) / len(elite) + 1e-4 * np.eye(dim)
        sigma = max(0.05, sigma * 0.95)

    # Build final merged model
    lam_A = best_params[:n_layers]
    lam_B = best_params[n_layers:]
    merged = OrderedDict()
    for i, prefix in enumerate(layer_prefixes):
        for k in sd_base:
            if k.startswith(prefix + '.') or k == prefix:
                tau_A = sd_A[k] - sd_base[k]
                tau_B = sd_B[k] - sd_base[k]
                merged[k] = sd_base[k] + lam_A[i] * tau_A + lam_B[i] * tau_B
    print(f"    Sakana-CMA: best fitness={best_fit:.4f}, λ_A={lam_A.round(2)}, λ_B={lam_B.round(2)}")
    return merged


# --- 6. ENT (topology selection + routing) ---
def merge_ent(sd_A, sd_B, val_xs, val_ys, cls_A, cls_B, n_out=10, do_ft=False):
    """ENT: channel mask EA + per-class routing + optional LogReg FT."""
    ALL_CLS = list(range(n_out))
    ch1 = sd_A['conv1.weight'].shape[0]
    ch2 = sd_A['conv2.weight'].shape[0]
    ch3 = sd_A['conv3.weight'].shape[0]

    def build_model(ch):
        mA1, mB1 = ch['m1A'], ch['m1B']
        mA2, mB2 = ch['m2A'], ch['m2B']
        mA3, mB3 = ch['m3A'], ch['m3B']

        iA1, iB1 = np.where(mA1)[0], np.where(mB1)[0]
        iA2, iB2 = np.where(mA2)[0], np.where(mB2)[0]
        iA3, iB3 = np.where(mA3)[0], np.where(mB3)[0]
        nA1, nB1 = len(iA1), len(iB1)
        nA2, nB2 = len(iA2), len(iB2)
        nA3, nB3 = len(iA3), len(iB3)
        if nA1+nB1<1 or nA2+nB2<1 or nA3+nB3<1:
            return None

        sd = OrderedDict()
        # Conv1: concat both (both see RGB input)
        sd['conv1.weight'] = torch.cat([sd_A['conv1.weight'][iA1],
                                         sd_B['conv1.weight'][iB1]], 0)
        sd['conv1.bias'] = torch.cat([sd_A['conv1.bias'][iA1],
                                       sd_B['conv1.bias'][iB1]])
        # BN1
        for p in ['weight','bias','running_mean','running_var']:
            k = f'bn1.{p}'
            sd[k] = torch.cat([sd_A[k][iA1], sd_B[k][iB1]])
        sd['bn1.num_batches_tracked'] = torch.tensor(0)

        # Conv2: block-diagonal
        c2_out, c2_in = nA2+nB2, nA1+nB1
        w2 = torch.zeros(c2_out, c2_in, 3, 3)
        w2[:nA2, :nA1] = sd_A['conv2.weight'][np.ix_(iA2, iA1)]
        w2[nA2:, nA1:] = sd_B['conv2.weight'][np.ix_(iB2, iB1)]
        sd['conv2.weight'] = w2
        sd['conv2.bias'] = torch.cat([sd_A['conv2.bias'][iA2],
                                       sd_B['conv2.bias'][iB2]])
        for p in ['weight','bias','running_mean','running_var']:
            k = f'bn2.{p}'
            sd[k] = torch.cat([sd_A[k][iA2], sd_B[k][iB2]])
        sd['bn2.num_batches_tracked'] = torch.tensor(0)

        # Conv3: block-diagonal
        c3_out, c3_in = nA3+nB3, nA2+nB2
        w3 = torch.zeros(c3_out, c3_in, 3, 3)
        w3[:nA3, :nA2] = sd_A['conv3.weight'][np.ix_(iA3, iA2)]
        w3[nA3:, nA2:] = sd_B['conv3.weight'][np.ix_(iB3, iB2)]
        sd['conv3.weight'] = w3
        sd['conv3.bias'] = torch.cat([sd_A['conv3.bias'][iA3],
                                       sd_B['conv3.bias'][iB3]])
        for p in ['weight','bias','running_mean','running_var']:
            k = f'bn3.{p}'
            sd[k] = torch.cat([sd_A[k][iA3], sd_B[k][iB3]])
        sd['bn3.num_batches_tracked'] = torch.tensor(0)

        # FC: routing
        route = ch.get('route', np.zeros(n_out))
        fc_w = torch.zeros(n_out, nA3+nB3)
        fc_b = torch.zeros(n_out)
        for c in ALL_CLS:
            alpha = 1.0 / (1.0 + np.exp(-route[c]))
            if c in cls_A and c in cls_B:
                fc_w[c, :nA3] = alpha * sd_A['fc.weight'][c][iA3]
                fc_w[c, nA3:] = (1-alpha) * sd_B['fc.weight'][c][iB3]
                fc_b[c] = alpha * sd_A['fc.bias'][c] + (1-alpha) * sd_B['fc.bias'][c]
            elif c in cls_A:
                fc_w[c, :nA3] = sd_A['fc.weight'][c][iA3]
                fc_b[c] = sd_A['fc.bias'][c]
            elif c in cls_B:
                fc_w[c, nA3:] = sd_B['fc.weight'][c][iB3]
                fc_b[c] = sd_B['fc.bias'][c]
        sd['fc.weight'] = fc_w
        sd['fc.bias'] = fc_b

        model = CifarCNN(n_out=n_out, ch1=nA1+nB1, ch2=nA2+nB2, ch3=nA3+nB3)
        model.load_state_dict(sd)
        model.eval()
        return model

    def fitness(ch):
        model = build_model(ch)
        if model is None: return -1.0
        with torch.no_grad():
            preds = model(val_xs).argmax(1)
        acc = (preds == val_ys).float().mean().item()
        per_cls = []
        for c in ALL_CLS:
            msk = val_ys == c
            if msk.sum() > 0:
                per_cls.append((preds[msk] == c).float().mean().item())
        min_c = min(per_cls) if per_cls else 0
        total = 2*(ch1+ch2+ch3)
        used = sum(ch[k].sum() for k in ['m1A','m1B','m2A','m2B','m3A','m3B'])
        return 0.4*acc + 0.4*min_c + 0.1*np.mean(per_cls) + 0.1*(1-used/total)

    # EA
    POP, GENS = 20, 40
    pop = []
    for _ in range(POP):
        ch = {
            'm1A': np.random.random(ch1)>0.3, 'm1B': np.random.random(ch1)>0.3,
            'm2A': np.random.random(ch2)>0.3, 'm2B': np.random.random(ch2)>0.3,
            'm3A': np.random.random(ch3)>0.3, 'm3B': np.random.random(ch3)>0.3,
            'route': np.random.randn(n_out)*1.5
        }
        for mk in ['m1A','m1B','m2A','m2B','m3A','m3B']:
            if ch[mk].sum()==0: ch[mk][0]=True
        pop.append(ch)

    # Seed: all channels, smart routing
    pop[0] = {
        'm1A': np.ones(ch1,bool), 'm1B': np.ones(ch1,bool),
        'm2A': np.ones(ch2,bool), 'm2B': np.ones(ch2,bool),
        'm3A': np.ones(ch3,bool), 'm3B': np.ones(ch3,bool),
        'route': np.zeros(n_out)
    }
    route_s = np.zeros(n_out)
    for c in cls_A: route_s[c] = 3.0
    for c in cls_B: route_s[c] = -3.0
    pop[1] = {**{k: np.ones_like(pop[0][k]) for k in pop[0] if k!='route'}, 'route': route_s}

    best_fit, best_ch = -1, None
    for gen in range(GENS):
        fits = [fitness(ch) for ch in pop]
        gi = np.argmax(fits)
        if fits[gi] > best_fit:
            best_fit = fits[gi]
            best_ch = {k: v.copy() for k,v in pop[gi].items()}
        new_pop = [{k: v.copy() for k,v in best_ch.items()}]
        while len(new_pop) < POP:
            ti = np.random.choice(len(pop), 3, replace=False)
            parent = pop[ti[np.argmax([fits[i] for i in ti])]]
            child = {k: v.copy() for k,v in parent.items()}
            child['route'] += np.random.randn(n_out)*0.3
            pf = max(0.02, 0.06 - gen*0.001)
            for mk in ['m1A','m1B','m2A','m2B','m3A','m3B']:
                flip = np.random.random(len(child[mk])) < pf
                child[mk][flip] = ~child[mk][flip]
                if child[mk].sum()==0: child[mk][np.random.randint(len(child[mk]))]=True
            new_pop.append(child)
        pop = new_pop

    print(f"    ENT EA: best fitness={best_fit:.4f}")
    model = build_model(best_ch)

    if not do_ft:
        return model

    # Phase 2: LogReg calibration
    model.eval()
    with torch.no_grad():
        features = model.extract_features(val_xs)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(features.numpy())
    y = val_ys.numpy()

    clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs', class_weight='balanced')
    clf.fit(X, y)

    # Fuse scaler into FC weights
    scale = torch.tensor(scaler.scale_, dtype=torch.float32)
    mean = torch.tensor(scaler.mean_, dtype=torch.float32)
    w_lr = torch.tensor(clf.coef_, dtype=torch.float32)
    b_lr = torch.tensor(clf.intercept_, dtype=torch.float32)
    w_fused = w_lr / scale.unsqueeze(0)
    b_fused = b_lr - (w_lr * mean.unsqueeze(0) / scale.unsqueeze(0)).sum(1)

    with torch.no_grad():
        model.fc.weight.copy_(w_fused)
        model.fc.bias.copy_(b_fused)
    print(f"    ENT-FT: LogReg calibrated on {len(val_xs)} samples")
    return model


# ══════════════════════════════════════════
# MAIN BENCHMARK
# ══════════════════════════════════════════

def run_benchmark():
    print("=" * 80)
    print("  Sakana-CMA vs ENT-FT — CIFAR-10 Complementary Merge")
    print("=" * 80)

    # Load data
    print("\n[1] Loading CIFAR-10...")
    train_xs, train_ys, test_xs, test_ys = load_cifar10()
    print(f"    Train: {train_xs.shape}, Test: {test_xs.shape}")

    CLS_A = [0, 1, 2, 3, 4]
    CLS_B = [5, 6, 7, 8, 9]
    ALL_CLS = list(range(10))
    SEEDS = [42]

    all_results = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  SEED: {seed}")
        print(f"{'='*60}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Separate val from train
        rest_xs, rest_ys, val_xs, val_ys = make_val_set(train_xs, train_ys, per_class=25)

        # ── Phase 1: Train shared base (10 epochs on ALL classes) ──
        print("\n[2] Training shared BASE model (10 epochs, all classes)...")
        base_model = CifarCNN(n_out=10)
        train_model(base_model, rest_xs, rest_ys, epochs=10, verbose=True)
        sd_base = copy.deepcopy(base_model.state_dict())

        # Eval base
        base_result = eval_model(base_model, test_xs, test_ys)
        print(f"    Base accuracy: {base_result['accuracy']:.3f}")

        # ── Phase 2: Fine-tune A on cls 0-4, B on cls 5-9 ──
        print("\n[3] Fine-tuning Parent A (cls 0-4)...")
        xs_A, ys_A = filter_classes(rest_xs, rest_ys, CLS_A)
        model_A = CifarCNN(n_out=10)
        model_A.load_state_dict(copy.deepcopy(sd_base))
        train_model(model_A, xs_A, ys_A, epochs=25, verbose=True)
        sd_A = copy.deepcopy(model_A.state_dict())

        # Eval parent A
        pA_result = eval_model(model_A, test_xs, test_ys)
        print(f"    Parent A: acc={pA_result['accuracy']:.3f} ok={pA_result['n_ok']}/10")
        for c in CLS_A:
            print(f"      c{c}: {pA_result['per_class'][c]:.3f}", end="")
        print()

        print("\n[4] Fine-tuning Parent B (cls 5-9)...")
        xs_B, ys_B = filter_classes(rest_xs, rest_ys, CLS_B)
        model_B = CifarCNN(n_out=10)
        model_B.load_state_dict(copy.deepcopy(sd_base))
        train_model(model_B, xs_B, ys_B, epochs=25, verbose=True)
        sd_B = copy.deepcopy(model_B.state_dict())

        pB_result = eval_model(model_B, test_xs, test_ys)
        print(f"    Parent B: acc={pB_result['accuracy']:.3f} ok={pB_result['n_ok']}/10")
        for c in CLS_B:
            print(f"      c{c}: {pB_result['per_class'][c]:.3f}", end="")
        print()

        parent_accs = {}
        for c in CLS_A: parent_accs[c] = pA_result['per_class'][c]
        for c in CLS_B: parent_accs[c] = pB_result['per_class'][c]

        # ── Phase 3: Run all merge methods ──
        print("\n[5] Running merge methods...")

        methods = {}

        # 1. Weight Average
        print("\n  >> Weight Average")
        methods['WA'] = merge_weight_average(sd_A, sd_B, alpha=0.5)

        # 2. Task Arithmetic (λ=0.5)
        print("  >> Task Arithmetic")
        methods['TaskArith'] = merge_task_arithmetic(sd_base, sd_A, sd_B, 0.5, 0.5)

        # 3. TIES
        print("  >> TIES-Merging")
        methods['TIES'] = merge_ties(sd_base, sd_A, sd_B, density=0.3)

        # 4. DARE-TIES
        print("  >> DARE-TIES")
        methods['DARE-TIES'] = merge_dare_ties(sd_base, sd_A, sd_B, density=0.3, dare_p=0.5)

        # 5. Sakana-CMA
        print("  >> Sakana-CMA (per-layer α, CMA-ES)")
        t0 = time.time()
        methods['Sakana-CMA'] = merge_sakana_cma(sd_base, sd_A, sd_B, val_xs, val_ys,
                                                   n_out=10, pop_size=16, n_gens=40)
        sakana_time = time.time() - t0
        print(f"    Time: {sakana_time:.1f}s")

        # 6. ENT
        print("  >> ENT (topology + routing)")
        t0 = time.time()
        ent_model = merge_ent(sd_A, sd_B, val_xs, val_ys, CLS_A, CLS_B, n_out=10, do_ft=False)
        ent_time = time.time() - t0
        print(f"    Time: {ent_time:.1f}s")

        # 7. ENT-FT
        print("  >> ENT-FT (+ LogReg)")
        t0 = time.time()
        ent_ft_model = merge_ent(sd_A, sd_B, val_xs, val_ys, CLS_A, CLS_B, n_out=10, do_ft=True)
        ent_ft_time = time.time() - t0
        print(f"    Time: {ent_ft_time:.1f}s")

        # ── Phase 4: Evaluate all methods ──
        print("\n[6] Evaluating...")
        print(f"\n  {'Method':<16} {'Acc':>6} {'Bal':>6} {'Min':>6} {'OK':>5} {'Params':>7} {'Time':>6}")
        print(f"  {'-'*16} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*7} {'-'*6}")

        for method_name, sd_or_model in [
            ('WA', methods['WA']),
            ('TaskArith', methods['TaskArith']),
            ('TIES', methods['TIES']),
            ('DARE-TIES', methods['DARE-TIES']),
            ('Sakana-CMA', methods['Sakana-CMA']),
            ('ENT', ent_model),
            ('ENT-FT', ent_ft_model),
        ]:
            if isinstance(sd_or_model, (dict, OrderedDict)):
                model = CifarCNN(n_out=10)
                model.load_state_dict(sd_or_model)
            else:
                model = sd_or_model

            t0 = time.time()
            result = eval_model(model, test_xs, test_ys)
            eval_time = time.time() - t0

            # Add timing
            if method_name == 'Sakana-CMA': result['time_s'] = round(sakana_time, 1)
            elif method_name == 'ENT': result['time_s'] = round(ent_time, 1)
            elif method_name == 'ENT-FT': result['time_s'] = round(ent_ft_time, 1)
            else: result['time_s'] = 0.1

            result['method'] = method_name
            result['seed'] = seed
            all_results.append(result)

            # Print with per-class retention
            print(f"  {method_name:<16} {result['accuracy']:>6.3f} {result['balance']:>6.3f} "
                  f"{result['min_class']:>6.3f} {result['n_ok']:>3}/{result['n_total']} "
                  f"{result['n_params']:>7d} {result['time_s']:>5.1f}s")

            # Per-class
            print(f"    ", end="")
            for c in ALL_CLS:
                a = result['per_class'][c]
                pa = parent_accs.get(c, 0)
                ret = f"({a/pa*100:.0f}%)" if pa > 0.01 else ""
                print(f"c{c}={a:.2f}{ret} ", end="")
            print()

    # ── Summary ──
    print("\n\n" + "=" * 90)
    print("  SUMMARY — Average across seeds")
    print("=" * 90)

    import pandas as pd
    df = pd.DataFrame(all_results)
    agg = df.groupby('method').agg({
        'accuracy': 'mean', 'balance': 'mean', 'min_class': 'mean',
        'n_ok': 'mean', 'n_params': 'first', 'time_s': 'mean',
    })
    agg = agg.sort_values('accuracy', ascending=False)

    print(f"\n  {'Method':<16} {'Acc':>6} {'Bal':>8} {'Min':>7} {'OK':>5} {'Params':>7} {'Time':>6}")
    print(f"  {'-'*16} {'-'*6} {'-'*8} {'-'*7} {'-'*5} {'-'*7} {'-'*6}")
    for method, row in agg.iterrows():
        print(f"  {method:<16} {row['accuracy']:>6.3f} {row['balance']:>8.3f} "
              f"{row['min_class']:>7.3f} {row['n_ok']:>4.0f}/10 "
              f"{int(row['n_params']):>7d} {row['time_s']:>5.1f}s")

    # Save
    df.to_csv(RESULTS_DIR / "sakana_vs_ent.tsv", sep='\t', index=False)
    print(f"\n  Results saved: {RESULTS_DIR / 'sakana_vs_ent.tsv'}")


if __name__ == "__main__":
    run_benchmark()
