#!/usr/bin/env python3
"""Ex30 — ENT-FT: комплементарне злиття MNIST (0-4 / 5-9) з калібрацією.

Комплементарний сценарій ENT: дві MLP навчаються з нуля на неперетинних
підмножинах класів MNIST (модель A — цифри 0-4, модель B — цифри 5-9),
точно як у e34_benchmark.py (архітектура, дані, тренування запозичено звідти).

Далі виконується:
  1. ENT — еволюційне злиття (формули 4.2-4.6, підрозд. 4.2 дисертації):
     об'єднана мережа (4.2), хромосома масок + маршрутизаторів (4.3),
     покласне зважування виходів субмереж (4.4), мульти-об'єктна цільова
     функція (4.5), еволюційний цикл Pop=20/Gen=30/p_flip (4.6).
  2. ENT-FT — калібрація вихідного шару на замороженому екстракторі
     (формули 4.7-4.9, підрозд. 4.3.1): витягування ознак (4.7),
     багатокласова логістична регресія (4.8), злиття масштабувача
     з вагами вихідного шару (4.9).

Самодостатній скрипт — без імпортів з інших файлів репозиторію.

Вивід: results_ent_ft_complementary.json (точність до/після калібрації,
покласові показники, баланс двома способами — покласовий min/max і
міжгруповий, для порівняння з таблицями 4.3-4.4 дисертації).

Використання:
    python ent_ft_complementary.py --smoke   # швидка перевірка (2 епохи, Gen=5)
    python ent_ft_complementary.py           # повний прогін (15 епох, Gen=30)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

SCRIPT_DIR = Path(__file__).resolve().parent
N_CLASSES = 10
GROUP_A = list(range(5))   # клас A: цифри 0-4
GROUP_B = list(range(5, 10))  # клас B: цифри 5-9
ARCH = [784, 128, 64, 10]  # архітектура MLP — та сама, що в e34_benchmark.py


# ══════════════════════════════════════════════════════════
# Дані та модель (запозичено з e34_benchmark.py)
# ══════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, arch: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i + 1]))
            if i < len(arch) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.arch = arch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_mnist(n_train: int, n_test: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST("/tmp/mnist", train=True, download=True, transform=tf)
    te = datasets.MNIST("/tmp/mnist", train=False, download=True, transform=tf)
    X_tr = torch.stack([tr[i][0] for i in range(n_train)])
    y_tr = torch.tensor([tr[i][1] for i in range(n_train)])
    X_te = torch.stack([te[i][0] for i in range(n_test)])
    y_te = torch.tensor([te[i][1] for i in range(n_test)])
    return X_tr, y_tr, X_te, y_te


def train_model(arch: list[int], X: torch.Tensor, y: torch.Tensor, classes: list[int],
                 epochs: int, cap: int = 5000) -> MLP:
    """Тренування MLP з нуля на підмножині класів (як train_model у e34_benchmark.py)."""
    model = MLP(arch)
    mask = sum(y == c for c in classes).bool()
    Xs, ys = X[mask][:cap], y[mask][:cap]
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    model.train()
    for _ in range(epochs):
        loss = nn.CrossEntropyLoss()(model(Xs), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def eval_per_class(model: MLP, X: torch.Tensor, y: torch.Tensor) -> dict[int, float]:
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(1)
    out = {}
    for c in range(N_CLASSES):
        m = y == c
        out[c] = (preds[m] == c).float().mean().item() if m.sum() > 0 else 0.0
    return out


def eval_acc(model: MLP, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def compute_balance(per_class: dict[int, float]) -> dict[str, float]:
    """Баланс двома способами: покласовий min/max і міжгруповий (група A vs B)."""
    vals = list(per_class.values())
    min_c, max_c = min(vals), max(vals)
    mean_a = float(np.mean([per_class[c] for c in GROUP_A]))
    mean_b = float(np.mean([per_class[c] for c in GROUP_B]))
    return {
        "min_class_acc": min_c,
        "max_class_acc": max_c,
        "per_class_balance_min_over_max": (min_c / max_c) if max_c > 0 else 0.0,
        "group_a_mean_acc": mean_a,
        "group_b_mean_acc": mean_b,
        "group_balance_min_over_max": (min(mean_a, mean_b) / max(mean_a, mean_b))
        if max(mean_a, mean_b) > 0 else 0.0,
        "classes_recognized": sum(1 for v in vals if v > 0.1),
    }


# ══════════════════════════════════════════════════════════
# ENT: об'єднана мережа + хромосома + еволюційний пошук
# Формули 4.2-4.6, підрозд. 4.2 дисертації
# ══════════════════════════════════════════════════════════

LayerWeights = list[tuple[np.ndarray, np.ndarray]]


def get_layer_weights(model: MLP) -> LayerWeights:
    """Ваги MLP по шарах: [(W0,b0) прих.1, (W1,b1) прих.2, (Wo,bo) вихід]."""
    params = list(model.parameters())
    return [(params[i].detach().numpy(), params[i + 1].detach().numpy())
            for i in range(0, len(params), 2)]


def random_chromosome(h1: int, h2: int, rng: np.random.Generator) -> dict[str, Any]:
    """Хромосома C_ENT (формула 4.3): маски m_A, m_B + маршрутизатори R_c."""
    ch = {
        "mA": [rng.random(h1) > 0.3, rng.random(h2) > 0.3],
        "mB": [rng.random(h1) > 0.3, rng.random(h2) > 0.3],
        "route": rng.standard_normal(N_CLASSES) * 1.5,
    }
    for masks in (ch["mA"], ch["mB"]):
        for m in masks:
            if m.sum() == 0:
                m[0] = True
    return ch


def mutate(parent: dict[str, Any], gen: int, rng: np.random.Generator) -> dict[str, Any]:
    """Bit-Flip (p_flip(g), формула 4.6) для масок + Гаусів шум (σ=0,3) для R_c.
    Кросинговер не застосовується (лише мутація переможця турніру)."""
    child = {
        "mA": [m.copy() for m in parent["mA"]],
        "mB": [m.copy() for m in parent["mB"]],
        "route": parent["route"] + rng.standard_normal(N_CLASSES) * 0.3,
    }
    p_flip = max(0.02, 0.06 - 0.001 * gen)  # формула 4.6
    for masks in (child["mA"], child["mB"]):
        for m in masks:
            flip = rng.random(len(m)) < p_flip
            m[flip] = ~m[flip]
            if m.sum() == 0:
                m[int(rng.integers(len(m)))] = True
    return child


def build_merged_model(chromosome: dict[str, Any], layersA: LayerWeights,
                        layersB: LayerWeights) -> MLP | None:
    """Формула 4.2: W_union^(l) = [W_A^(l); W_B^(l)] по прихованих шарах;
    вихідний шар — покласне зважування виходів субмереж (формула 4.4)."""
    mA1, mA2 = chromosome["mA"]
    mB1, mB2 = chromosome["mB"]
    iA1, iB1 = np.where(mA1)[0], np.where(mB1)[0]
    iA2, iB2 = np.where(mA2)[0], np.where(mB2)[0]
    n1, n2 = len(iA1) + len(iB1), len(iA2) + len(iB2)
    if n1 < 1 or n2 < 1:
        return None

    (WA0, bA0), (WA1, bA1), (WAo, boA) = layersA
    (WB0, bB0), (WB1, bB1), (WBo, boB) = layersB

    # Прихований шар 1: конкатенація нейронів (формула 4.2)
    W0 = np.vstack([WA0[iA1], WB0[iB1]]).astype(np.float32)
    b0 = np.concatenate([bA0[iA1], bB0[iB1]]).astype(np.float32)

    # Прихований шар 2: блок-діагональ — підмережі не інтерферують між собою
    W1 = np.zeros((n2, n1), dtype=np.float32)
    b1 = np.zeros(n2, dtype=np.float32)
    W1[:len(iA2), :len(iA1)] = WA1[np.ix_(iA2, iA1)]
    b1[:len(iA2)] = bA1[iA2]
    W1[len(iA2):, len(iA1):] = WB1[np.ix_(iB2, iB1)]
    b1[len(iA2):] = bB1[iB2]

    # Вихідний шар: O(x,c) = R_c,A * σ_A(x) + R_c,B * σ_B(x)  (формула 4.4)
    Wo = np.zeros((N_CLASSES, n2), dtype=np.float32)
    bo = np.zeros(N_CLASSES, dtype=np.float32)
    route = chromosome["route"]
    for c in range(N_CLASSES):
        alpha = 1.0 / (1.0 + np.exp(-route[c]))
        if len(iA2) > 0:
            Wo[c, :len(iA2)] = alpha * WAo[c][iA2]
        if len(iB2) > 0:
            Wo[c, len(iA2):] = (1 - alpha) * WBo[c][iB2]
        bo[c] = alpha * boA[c] + (1 - alpha) * boB[c]

    model = MLP([784, n1, n2, N_CLASSES])
    with torch.no_grad():
        p = list(model.parameters())
        p[0].copy_(torch.tensor(W0)); p[1].copy_(torch.tensor(b0))
        p[2].copy_(torch.tensor(W1)); p[3].copy_(torch.tensor(b1))
        p[4].copy_(torch.tensor(Wo)); p[5].copy_(torch.tensor(bo))
    model.eval()
    return model


def ent_fitness(chromosome: dict[str, Any], layersA: LayerWeights, layersB: LayerWeights,
                 Xv: torch.Tensor, yv: torch.Tensor, h1: int, h2: int) -> float:
    """Формула 4.5: F_ENT = w1*Acc_global + w2*min_c Acc^(c) + w3*mean(Acc_local) + w4*CR."""
    model = build_merged_model(chromosome, layersA, layersB)
    if model is None:
        return -1.0
    per_class = eval_per_class(model, Xv, yv)
    acc_global = eval_acc(model, Xv, yv)
    min_c = min(per_class.values())
    mean_c = float(np.mean(list(per_class.values())))
    n_kept = (chromosome["mA"][0].sum() + chromosome["mB"][0].sum()
              + chromosome["mA"][1].sum() + chromosome["mB"][1].sum())
    cr = 1.0 - n_kept / (2 * h1 + 2 * h2)
    w1, w2, w3, w4 = 0.4, 0.4, 0.1, 0.1  # ваги формули 4.5 (підрозд. 4.2.2)
    return w1 * acc_global + w2 * min_c + w3 * mean_c + w4 * cr


def evolve_ent(layersA: LayerWeights, layersB: LayerWeights, Xv: torch.Tensor, yv: torch.Tensor,
               h1: int, h2: int, pop_size: int, n_gen: int, seed: int,
               verbose: bool = True) -> tuple[dict[str, Any], float]:
    """Алгоритм 5 (ENT): турнірна селекція k=3, Pop=20/Gen=30 (формула 4.6),
    елітизм (найкраща хромосома переходить у наступне покоління без змін)."""
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    population = [random_chromosome(h1, h2, rng) for _ in range(pop_size)]
    # Seed-хромосома: повна мережа з "гострою" маршрутизацією A->1, B->0
    population[0] = {
        "mA": [np.ones(h1, dtype=bool), np.ones(h2, dtype=bool)],
        "mB": [np.ones(h1, dtype=bool), np.ones(h2, dtype=bool)],
        "route": np.array([2.0] * 5 + [-2.0] * 5, dtype=np.float64),
    }

    best_fit, best_ch = -1.0, population[0]
    for gen in range(n_gen):
        fits = [ent_fitness(ch, layersA, layersB, Xv, yv, h1, h2) for ch in population]
        gi = int(np.argmax(fits))
        if fits[gi] > best_fit:
            best_fit = fits[gi]
            best_ch = {"mA": [m.copy() for m in population[gi]["mA"]],
                       "mB": [m.copy() for m in population[gi]["mB"]],
                       "route": population[gi]["route"].copy()}
        if verbose and (gen % max(1, n_gen // 5) == 0 or gen == n_gen - 1):
            print(f"    Покоління {gen + 1}/{n_gen}: найкращий fitness = {best_fit:.4f}")
        new_pop = [{"mA": [m.copy() for m in best_ch["mA"]],
                    "mB": [m.copy() for m in best_ch["mB"]],
                    "route": best_ch["route"].copy()}]  # елітизм
        while len(new_pop) < pop_size:
            idx = py_rng.sample(range(len(population)), min(3, len(population)))
            parent = population[max(idx, key=lambda i: fits[i])]
            new_pop.append(mutate(parent, gen, rng))
        population = new_pop

    return best_ch, best_fit


# ══════════════════════════════════════════════════════════
# ENT-FT: калібрація вихідного шару на замороженому екстракторі
# Формули 4.7-4.9, підрозд. 4.3.1 дисертації
# ══════════════════════════════════════════════════════════

def calibrate_ent_ft(model: MLP, Xv: torch.Tensor, yv: torch.Tensor) -> MLP:
    """Заморожує екстрактор (усі шари, крім вихідного), калібрує вихідний
    шар багатокласовою логістичною регресією на стандартизованих ознаках."""
    model.eval()
    with torch.no_grad():
        h = Xv
        for layer in list(model.net)[:-1]:  # усе, крім останнього Linear
            h = layer(h)
        features = h.numpy()  # формула 4.7: Z_val = Φ_ENT(X_val)

    scaler = StandardScaler()
    Z = scaler.fit_transform(features)  # Z' = (Z - μ) / σ

    y_np = yv.numpy()
    counts = {int(c): int((y_np == c).sum()) for c in np.unique(y_np)}
    sample_weight = np.array([1.0 / counts[int(y)] for y in y_np])
    sample_weight = sample_weight / sample_weight.sum() * len(sample_weight)

    # Формула 4.8: (W_lr, B_lr) = argmin Σ ρ_yi * L_CE(...) + λ||W||²
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", class_weight="balanced")
    clf.fit(Z, y_np, sample_weight=sample_weight)

    scale = torch.tensor(scaler.scale_, dtype=torch.float32)
    mean = torch.tensor(scaler.mean_, dtype=torch.float32)
    w_lr = torch.tensor(clf.coef_, dtype=torch.float32)
    b_lr = torch.tensor(clf.intercept_, dtype=torch.float32)

    # Формула 4.9: W*_FC = W_lr / σ,  B*_FC = B_lr - Σ_j W_lr^(j) * μ^(j)/σ^(j)
    w_fused = w_lr / scale.unsqueeze(0)
    b_fused = b_lr - (w_lr * mean.unsqueeze(0) / scale.unsqueeze(0)).sum(1)

    out_layer = list(model.net)[-1]
    with torch.no_grad():
        out_layer.weight.copy_(w_fused)
        out_layer.bias.copy_(b_fused)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════
# Основний сценарій
# ══════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true",
                         help="швидка перевірка: 2 епохи тренування, менша вибірка, Gen=5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="епох тренування батьківських MLP")
    parser.add_argument("--gen", type=int, default=None, help="поколінь ENT (за замовч. 30, у --smoke 5)")
    parser.add_argument("--pop", type=int, default=20, help="розмір популяції ENT (формула 4.6)")
    parser.add_argument("--n-train", type=int, default=None, help="розмір тренувального пулу MNIST")
    parser.add_argument("--n-test", type=int, default=None, help="розмір тестової вибірки MNIST")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    smoke = args.smoke
    seed = args.seed
    epochs = args.epochs if args.epochs is not None else (2 if smoke else 15)
    n_gen = args.gen if args.gen is not None else (5 if smoke else 30)
    n_train = args.n_train if args.n_train is not None else (3000 if smoke else 20000)
    n_test = args.n_test if args.n_test is not None else (500 if smoke else 2000)
    pop_size = args.pop
    out_path = args.output or (SCRIPT_DIR / "results_ent_ft_complementary.json")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("=" * 70)
    print("  Ex30 — ENT-FT: комплементарне злиття MNIST (0-4 / 5-9)")
    print("=" * 70)
    print(f"  Режим: {'SMOKE' if smoke else 'ПОВНИЙ'}; епох={epochs}; n_train={n_train}; "
          f"n_test={n_test}; Pop={pop_size}; Gen={n_gen}")
    t_start = time.time()

    # --- Дані: тренувальний пул -> непересічні зрізи (тренування батьків / ENT-валідація) ---
    X_tr, y_tr, X_te, y_te = load_mnist(n_train, n_test)
    perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(0))
    n_fit = int(n_train * 0.75)
    n_val = max(50, int(n_train * 0.15))
    idx_fit = perm[:n_fit]
    idx_val = perm[n_fit:n_fit + n_val]
    X_fit, y_fit = X_tr[idx_fit], y_tr[idx_fit]
    Xv, yv = X_tr[idx_val], y_tr[idx_val]

    # --- Тренування батьківських моделей з нуля на неперетинних класах ---
    print("\n[1/4] Тренування батьківських моделей...")
    t0 = time.time()
    model_a = train_model(ARCH, X_fit, y_fit, GROUP_A, epochs)
    model_b = train_model(ARCH, X_fit, y_fit, GROUP_B, epochs)
    t_train = time.time() - t0
    acc_a = eval_acc(model_a, X_te[torch.isin(y_te, torch.tensor(GROUP_A))],
                      y_te[torch.isin(y_te, torch.tensor(GROUP_A))])
    acc_b = eval_acc(model_b, X_te[torch.isin(y_te, torch.tensor(GROUP_B))],
                      y_te[torch.isin(y_te, torch.tensor(GROUP_B))])
    print(f"  Модель A (цифри 0-4): точність на власних класах = {acc_a:.3f}")
    print(f"  Модель B (цифри 5-9): точність на власних класах = {acc_b:.3f}")

    # --- ENT: еволюційне злиття (формули 4.2-4.6) ---
    print(f"\n[2/4] ENT — еволюційний пошук (Pop={pop_size}, Gen={n_gen})...")
    t0 = time.time()
    h1, h2 = ARCH[1], ARCH[2]
    layers_a = get_layer_weights(model_a)
    layers_b = get_layer_weights(model_b)
    best_ch, best_fit = evolve_ent(layers_a, layers_b, Xv, yv, h1, h2, pop_size, n_gen, seed)
    t_ent = time.time() - t0

    ent_model = build_merged_model(best_ch, layers_a, layers_b)
    assert ent_model is not None, "ENT: злита модель порожня (усі маски занулено)"
    per_class_before = eval_per_class(ent_model, X_te, y_te)
    acc_before = eval_acc(ent_model, X_te, y_te)
    balance_before = compute_balance(per_class_before)
    n_kept = (best_ch["mA"][0].sum() + best_ch["mB"][0].sum()
              + best_ch["mA"][1].sum() + best_ch["mB"][1].sum())
    compression_ratio = 1.0 - n_kept / (2 * h1 + 2 * h2)
    print(f"  ENT (без калібрації): acc={acc_before:.3f}  "
          f"класів={balance_before['classes_recognized']}/10  "
          f"баланс(міжгруп.)={balance_before['group_balance_min_over_max']:.3f}")

    # --- ENT-FT: калібрація вихідного шару (формули 4.7-4.9) ---
    print("\n[3/4] ENT-FT — калібрація вихідного шару (LogReg, формули 4.7-4.9)...")
    t0 = time.time()
    ent_ft_model = calibrate_ent_ft(ent_model, Xv, yv)
    t_calib = time.time() - t0
    per_class_after = eval_per_class(ent_ft_model, X_te, y_te)
    acc_after = eval_acc(ent_ft_model, X_te, y_te)
    balance_after = compute_balance(per_class_after)
    print(f"  ENT-FT (з калібрацією): acc={acc_after:.3f}  "
          f"класів={balance_after['classes_recognized']}/10  "
          f"баланс(міжгруп.)={balance_after['group_balance_min_over_max']:.3f}")

    # --- Результати ---
    print("\n[4/4] Збереження результатів...")
    t_total = time.time() - t_start
    report = {
        "config": {
            "smoke": smoke, "seed": seed, "epochs": epochs, "n_train": n_train,
            "n_test": n_test, "pop_size": pop_size, "n_gen": n_gen, "arch": ARCH,
        },
        "parents": {
            "model_a_acc_own_classes": acc_a, "model_b_acc_own_classes": acc_b,
        },
        "ent": {
            "accuracy": acc_before, "per_class": per_class_before, "balance": balance_before,
            "fitness": best_fit, "compression_ratio": compression_ratio,
        },
        "ent_ft": {
            "accuracy": acc_after, "per_class": per_class_after, "balance": balance_after,
        },
        "delta": {
            "accuracy": acc_after - acc_before,
            "min_class_acc": balance_after["min_class_acc"] - balance_before["min_class_acc"],
            "group_balance_min_over_max": (balance_after["group_balance_min_over_max"]
                                            - balance_before["group_balance_min_over_max"]),
        },
        "timing_s": {"train_parents": t_train, "ent_search": t_ent,
                     "calibration": t_calib, "total": t_total},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("  ПІДСУМОК (порівняти з табл. 4.3-4.4 дисертації)")
    print("=" * 70)
    print(f"  {'Етап':<20} {'Точність':>10} {'min_c Acc':>10} {'Баланс(міжгр.)':>16}")
    print(f"  {'ENT (без калібр.)':<20} {acc_before:>10.3f} "
          f"{balance_before['min_class_acc']:>10.3f} "
          f"{balance_before['group_balance_min_over_max']:>16.3f}")
    print(f"  {'ENT-FT (з калібр.)':<20} {acc_after:>10.3f} "
          f"{balance_after['min_class_acc']:>10.3f} "
          f"{balance_after['group_balance_min_over_max']:>16.3f}")
    print(f"\n  Звіт збережено: {out_path}")


if __name__ == "__main__":
    main()
