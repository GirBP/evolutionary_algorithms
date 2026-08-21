#!/usr/bin/env python3
"""Ex30 — ENT-FT на точному чемпіоні e34: канонічний експеримент табл. 4.4.

На відміну від `ent_ft_complementary.py` (де ENT перетренований незалежно,
з власним поділом даних на fit/val), цей скрипт бере САМЕ ТОГО чемпіона,
що фігурує в `results_e34.json` (табл. 4.3, 4.6) — тобто ENT-переможця
`e34_benchmark.py` — і застосовує до нього калібрацію ENT-FT (формули
4.7-4.9, підрозд. 4.3.1) на валідаційній вибірці Xv, яку `e34_benchmark.py`
сам використовує для еволюційного відбору (n=3000, 15% тренувального пулу).

Детермінізм і відтворення чемпіона.
`e34_benchmark.py` повністю детермінований (SEED=42): дані, ініціалізація
базової моделі, тренування батьків A/B і еволюційний пошук ENT не мають
джерел недетермінізму. Проте методи 1-8 бенчмарку (Average, SLERP, Task
Arithmetic, TIES, DARE, Fisher, NeuronConcat, Sakana-CMA) споживають
глобальний стан `numpy.random` / `torch` ГРС у певному порядку — і бібліотека
`cma` (метод Sakana), залежно від версії, теж може займати глобальний стан
numpy. Щоб гарантувати біт-у-біт той самий вхідний стан ГРС на вході методу
9 (ENT), цей скрипт відтворює методи 1-8 дослівно (результати відкидаються —
вони тут не потрібні), а тоді відтворює метод 9 (ENT) і звіряє результат із
збереженим `results_e34.json` асертом (розбіжність — аварійна зупинка).

Джерела чисел: MNIST (torchvision, кеш /tmp/mnist), `results_e34.json`
(лише читання — звірка чемпіона). Нових result-файлів дисертації, крім
вихідного `results_ent_ft_on_e34.json`, скрипт не створює і не редагує.

Вивід: `results_ent_ft_on_e34.json` — точність/покласові показники/баланс
(обома формулами) до і після калібрації, повна конфігурація.

Час виконання: ~10-30 хв (тренування батьків, 8 baseline-методів для
відтворення стану ГРС, еволюційний пошук ENT — Gen=30, Sakana-CMA).

Використання:
    python3 ent_ft_on_e34_champion.py
"""

from __future__ import annotations

import copy
import json
import random
import ssl
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
SEED = 42
ARCH = [784, 128, 64, 10]
N_TRAIN = 20000
N_TEST = 2000
N_CLASSES = 10
GROUP_A = list(range(5))      # цифри 0-4
GROUP_B = list(range(5, 10))  # цифри 5-9
CHAMPION_SOURCE = "results_e34.json"  # тільки читається, не редагується


# ══════════════════════════════════════════════════════════
# Модель і дані (мають точно збігатись з e34_benchmark.py)
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


def ev(model: MLP, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def pc(model: MLP, X: torch.Tensor, y: torch.Tensor) -> dict[int, float]:
    model.eval()
    with torch.no_grad():
        p = model(X).argmax(1)
    return {c: (p[y == c] == c).float().mean().item() if (y == c).sum() > 0 else 0.0
            for c in range(N_CLASSES)}


def compute_balance(per_class: dict[int, float]) -> dict[str, float]:
    """Баланс двома формулами: покласовий min/max і міжгруповий (A vs B),
    та сама метрика, що й у ent_ft_complementary.py, для порівнянності."""
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


def load_mnist() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ssl._create_default_https_context = ssl._create_unverified_context
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST(str(Path("/tmp/mnist")), train=True, download=True, transform=tf)
    te = datasets.MNIST(str(Path("/tmp/mnist")), train=False, download=True, transform=tf)
    X_tr = torch.stack([tr[i][0] for i in range(N_TRAIN)])
    y_tr = torch.tensor([tr[i][1] for i in range(N_TRAIN)])
    X_te = torch.stack([te[i][0] for i in range(N_TEST)])
    y_te = torch.tensor([te[i][1] for i in range(N_TEST)])
    return X_tr, y_tr, X_te, y_te


def train_model(arch: list[int], X: torch.Tensor, y: torch.Tensor, cls: list[int],
                 epochs: int = 15) -> MLP:
    m = MLP(arch)
    mask = sum(y == c for c in cls).bool()
    Xs, ys = X[mask][:5000], y[mask][:5000]
    opt = torch.optim.Adam(m.parameters(), lr=0.003)
    m.train()
    for _ in range(epochs):
        loss = nn.CrossEntropyLoss()(m(Xs), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    return m


# ══════════════════════════════════════════════════════════
# Методи 1-8 e34_benchmark.py — відтворені ДОСЛІВНО (результати
# відкидаються) лише щоб гарантувати той самий стан ГРС на вході
# методу 9 (ENT). Без цього кроку еволюційний пошук ENT споживав
# би інший ланцюжок np.random/torch-викликів і дав би ІНШОГО
# чемпіона — асерт нижче про це й попереджає.
# ══════════════════════════════════════════════════════════

def replay_baseline_methods(sdA: dict, sdB: dict, base_sd: dict,
                             Xv: torch.Tensor, yv: torch.Tensor) -> None:
    # Метод 1: Weight Averaging
    for alpha in [0.3, 0.5, 0.7]:
        m = MLP(ARCH)
        sd = {k: alpha * sdA[k] + (1 - alpha) * sdB[k] for k in sdA}
        m.load_state_dict(sd)
        m.eval()

    # Метод 2: SLERP
    def slerp_merge(sdA: dict, sdB: dict, t: float = 0.5) -> dict:
        sd = {}
        for k in sdA:
            vA, vB = sdA[k].float().flatten(), sdB[k].float().flatten()
            nA, nB = vA.norm(), vB.norm()
            if nA < 1e-8 or nB < 1e-8:
                sd[k] = (t * sdA[k] + (1 - t) * sdB[k])
                continue
            cos_omega = (vA @ vB) / (nA * nB)
            cos_omega = cos_omega.clamp(-1, 1)
            omega = torch.acos(cos_omega)
            if omega.abs() < 1e-6:
                sd[k] = (t * sdA[k] + (1 - t) * sdB[k])
            else:
                sA_ = torch.sin((1 - t) * omega) / torch.sin(omega)
                sB_ = torch.sin(t * omega) / torch.sin(omega)
                sd[k] = (sA_ * sdA[k] + sB_ * sdB[k])
        return sd

    for t in [0.3, 0.5, 0.7]:
        m = MLP(ARCH)
        m.load_state_dict(slerp_merge(sdA, sdB, t))
        m.eval()

    # Метод 3: Task Arithmetic
    for tau in [0.3, 0.5, 0.7, 1.0]:
        m = MLP(ARCH)
        sd = {}
        for k in sdA:
            dA = sdA[k] - base_sd[k]
            dB = sdB[k] - base_sd[k]
            sd[k] = base_sd[k] + tau * (dA + dB)
        m.load_state_dict(sd)
        m.eval()

    # Метод 4: TIES-Merging
    for density in [0.1, 0.3, 0.5]:
        m = MLP(ARCH)
        sd = {}
        for k in sdA:
            dA = sdA[k] - base_sd[k]
            dB = sdB[k] - base_sd[k]
            for delta in [dA, dB]:
                flat = delta.flatten()
                n_keep = max(1, int(len(flat) * density))
                threshold = flat.abs().topk(n_keep).values[-1]
                delta[delta.abs() < threshold] = 0
            sign_mask = (dA.sign() + dB.sign())
            merged_delta = torch.where(
                sign_mask > 0,
                torch.maximum(dA, torch.zeros_like(dA)) + torch.maximum(dB, torch.zeros_like(dB)),
                torch.minimum(dA, torch.zeros_like(dA)) + torch.minimum(dB, torch.zeros_like(dB)))
            n_nonzero = (merged_delta != 0).float().sum()
            if n_nonzero > 0:
                merged_delta = merged_delta * ((dA != 0).float() + (dB != 0).float()).clamp(max=1)
            sd[k] = base_sd[k] + merged_delta * 0.5
        m.load_state_dict(sd)
        m.eval()

    # Метод 5: DARE
    for p in [0.1, 0.3, 0.5, 0.9]:
        m = MLP(ARCH)
        sd = {}
        torch.manual_seed(SEED)
        for k in sdA:
            dA = sdA[k] - base_sd[k]
            dB = sdB[k] - base_sd[k]
            maskA = (torch.rand_like(dA.float()) < p).float()
            maskB = (torch.rand_like(dB.float()) < p).float()
            dA_dare = dA * maskA / max(p, 0.01)
            dB_dare = dB * maskB / max(p, 0.01)
            sd[k] = base_sd[k] + 0.5 * (dA_dare + dB_dare)
        m.load_state_dict(sd)
        m.eval()

    # Метод 6: Fisher Merging (діагональна апроксимація)
    def compute_fisher(model: MLP, X: torch.Tensor, y: torch.Tensor, n: int = 500) -> dict:
        model.eval()
        fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
        model.train()
        for i in range(min(n, len(X))):
            model.zero_grad()
            out = model(X[i:i + 1])
            log_prob = nn.LogSoftmax(dim=1)(out)
            target = out.argmax(1)
            loss = nn.NLLLoss()(log_prob, target)
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.data ** 2
        for name in fisher:
            fisher[name] /= n
        model.eval()
        return fisher

    modelA_tmp = MLP(ARCH)
    modelA_tmp.load_state_dict(sdA)
    modelB_tmp = MLP(ARCH)
    modelB_tmp.load_state_dict(sdB)
    fisherA = compute_fisher(modelA_tmp, Xv, yv, n=300)
    fisherB = compute_fisher(modelB_tmp, Xv, yv, n=300)
    m = MLP(ARCH)
    sd = {}
    for k in sdA:
        fA = torch.ones_like(sdA[k])
        fB = torch.ones_like(sdB[k])
        for fk in fisherA:
            if fk in k or k in fk:
                fA = fisherA[fk]
                fB = fisherB[fk]
                break
        denom = fA + fB + 1e-8
        sd[k] = (fA * sdA[k] + fB * sdB[k]) / denom
    m.load_state_dict(sd)
    m.eval()

    # Метод 7: NeuronConcat
    def neuron_concat(mA: MLP, mB: MLP) -> MLP:
        pA, pB = list(mA.parameters()), list(mB.parameters())
        sA_ = [p.shape[0] for p in pA[::2]]
        sB_ = [p.shape[0] for p in pB[::2]]
        new_arch = [784] + [sA_[i] + sB_[i] for i in range(len(sA_) - 1)] + [10]
        m = MLP(new_arch)
        with torch.no_grad():
            ps = list(m.parameters())
            W0 = torch.cat([pA[0].data, pB[0].data], dim=0)
            b0 = torch.cat([pA[1].data, pB[1].data])
            ps[0].copy_(W0)
            ps[1].copy_(b0)
            s0A, s0B = sA_[0], sB_[0]
            s1A, s1B = sA_[1], sB_[1] if len(sA_) > 2 else (10, 10)
            W1 = torch.zeros(s1A + s1B, s0A + s0B)
            W1[:s1A, :s0A] = pA[2].data
            W1[s1A:, s0A:] = pB[2].data
            b1 = torch.cat([pA[3].data, pB[3].data])
            ps[2].copy_(W1)
            ps[3].copy_(b1)
            Wo = torch.zeros(10, s1A + s1B)
            Wo[:, :s1A] = 0.5 * pA[4].data
            Wo[:, s1A:] = 0.5 * pB[4].data
            bo = 0.5 * pA[5].data + 0.5 * pB[5].data
            ps[4].copy_(Wo)
            ps[5].copy_(bo)
        return m

    m = neuron_concat(modelA_tmp, modelB_tmp)
    m.eval()

    # Метод 8: Sakana-style (per-layer α, CMA-ES) — окремий пакет `cma`
    # має власний посіяний потік, але його ask()/tell() може, залежно
    # від версії, зачіпати й глобальний стан numpy, тому виконується тут
    # так само, як у e34_benchmark.py.
    import cma
    n_merge_layers = 3

    def build_sakana(x: np.ndarray) -> MLP:
        alphas = 1.0 / (1.0 + np.exp(-np.array(x)))
        m = MLP(ARCH)
        sd = {}
        keys = list(sdA.keys())
        for i, k in enumerate(keys):
            layer_idx = i // 2
            a = alphas[min(layer_idx, len(alphas) - 1)]
            sd[k] = a * sdA[k] + (1 - a) * sdB[k]
        m.load_state_dict(sd)
        m.eval()
        return m

    es = cma.CMAEvolutionStrategy(np.zeros(n_merge_layers), 1.5, {
        "maxiter": 15, "popsize": 8, "seed": SEED, "verbose": -1})
    best_sak_s = -1.0
    best_sak_x = None
    while not es.stop():
        sols = es.ask()
        scores = []
        for x in sols:
            m = build_sakana(x)
            d = pc(m, Xv, yv)
            acc = ev(m, Xv, yv)
            mn = min(d[c] for c in range(10))
            s = 0.4 * acc + 0.4 * mn + 0.1 * np.mean([d[c] for c in range(10)])
            scores.append(-s)
        es.tell(sols, scores)
        if -min(scores) > best_sak_s:
            best_sak_s = -min(scores)
            best_sak_x = sols[int(np.argmin(scores))]
    m = build_sakana(best_sak_x)
    m.eval()


# ══════════════════════════════════════════════════════════
# Метод 9: ENT (формули 4.2-4.6) — той самий блок, що й у e34_benchmark.py
# ══════════════════════════════════════════════════════════

def virt2l(model: MLP, Xc: torch.Tensor) -> tuple[list[np.ndarray], list[int]]:
    """Віртуальна дворівнева реконструкція прихованих активацій моделі
    (для узгодження розмірностей злиття) — допоміжний блок ENT."""
    model.eval()
    with torch.no_grad():
        h = Xc
        for layer in list(model.net)[:-1]:
            h = layer(h)
        hid = h.numpy()
    ps = list(model.parameters())
    Wo = ps[-2].detach().numpy()
    bo = ps[-1].detach().numpy()
    fd = hid.shape[1]
    x = Xc.numpy()
    N = x.shape[0]
    xb = np.hstack([x, np.ones((N, 1), dtype=np.float32)])
    Wb = np.linalg.lstsq(xb, np.maximum(hid, 0), rcond=None)[0].T
    W1 = Wb[:, :-1].astype(np.float32)
    b1 = Wb[:, -1].astype(np.float32)
    W2 = np.eye(fd, dtype=np.float32)
    b2 = np.zeros(fd, dtype=np.float32)
    return [W1, b1, W2, b2, Wo.copy(), bo.copy()], [fd, fd]


def bld_ent(ch: dict[str, Any], WA: list[np.ndarray], WB: list[np.ndarray],
            sA: list[int], sB: list[int], rA: float, rB: float) -> MLP | None:
    """Формула 4.2 (об'єднана мережа) + формула 4.4 (покласна маршрутизація виходу)."""
    sizes = [784]
    for i in range(2):
        n = int(ch["mA"][i].sum()) + int(ch["mB"][i].sum())
        if n < 2:
            return None
        sizes.append(n)
    sizes.append(10)
    iA = np.where(ch["mA"][0])[0]
    iB = np.where(ch["mB"][0])[0]
    W0 = np.vstack([WA[0][iA], WB[0][iB]])
    b0 = np.concatenate([WA[1][iA], WB[1][iB]])
    ipA, ipB = iA, iB
    icA = np.where(ch["mA"][1])[0]
    icB = np.where(ch["mB"][1])[0]
    W1 = np.zeros((len(icA) + len(icB), len(ipA) + len(ipB)), dtype=np.float32)
    b1 = np.zeros(len(icA) + len(icB), dtype=np.float32)
    W1[:len(icA), :len(ipA)] = WA[2][np.ix_(icA, ipA)]
    b1[:len(icA)] = WA[3][icA]
    W1[len(icA):, len(ipA):] = WB[2][np.ix_(icB, ipB)]
    b1[len(icA):] = WB[3][icB]
    ilA = np.where(ch["mA"][1])[0]
    ilB = np.where(ch["mB"][1])[0]
    Wo = np.zeros((10, len(ilA) + len(ilB)), dtype=np.float32)
    bo = np.zeros(10, dtype=np.float32)
    for c in range(10):
        a = 1.0 / (1.0 + np.exp(-ch["route"][c]))
        if len(ilA) > 0:
            Wo[c, :len(ilA)] = a * rA * WA[4][c][ilA]
        if len(ilB) > 0:
            Wo[c, len(ilA):] = (1 - a) * rB * WB[4][c][ilB]
        bo[c] = a * rA * WA[5][c] + (1 - a) * rB * WB[5][c]
    m = MLP(sizes)
    with torch.no_grad():
        ps = list(m.parameters())
        ps[0].copy_(torch.tensor(W0))
        ps[1].copy_(torch.tensor(b0))
        ps[2].copy_(torch.tensor(W1))
        ps[3].copy_(torch.tensor(b1))
        ps[4].copy_(torch.tensor(Wo))
        ps[5].copy_(torch.tensor(bo))
    return m


def evolve_ent(modelA: MLP, modelB: MLP, Xc: torch.Tensor, Xv: torch.Tensor,
               yv: torch.Tensor) -> tuple[dict[str, Any], float, int, int]:
    """Еволюційний цикл ENT: Pop=20, Gen=30, турнір k=3, елітизм (формула 4.6)."""
    WA, sA = virt2l(modelA, Xc)
    WB, sB = virt2l(modelB, Xc)
    modelA.eval()
    modelB.eval()
    with torch.no_grad():
        sl = modelA(Xc).numpy().std()
        sr = modelB(Xc).numpy().std()
    t = (sl + sr) / 2
    rA = t / (sl + 1e-10)
    rB = t / (sr + 1e-10)

    pop = []
    for _ in range(20):
        ch = {"mA": [np.random.random(d) > 0.3 for d in sA],
              "mB": [np.random.random(d) > 0.3 for d in sB],
              "route": np.random.randn(10) * 1.5}
        for ms in [ch["mA"], ch["mB"]]:
            for m in ms:
                if m.sum() == 0:
                    m[0] = True
        pop.append(ch)
    pop[0] = {"mA": [np.ones(d, dtype=bool) for d in sA],
              "mB": [np.ones(d, dtype=bool) for d in sB],
              "route": np.array([2.0] * 5 + [-2.0] * 5)}

    bf = -1.0
    bc: dict[str, Any] = pop[0]
    for gen in range(30):
        fs = []
        for ch in pop:
            m = bld_ent(ch, WA, WB, sA, sB, rA, rB)
            if m is None:
                fs.append(-1.0)
                continue
            d = pc(m, Xv, yv)
            acc = ev(m, Xv, yv)
            mn = min(d[c] for c in range(10))
            fs.append(0.4 * acc + 0.4 * mn + 0.1 * np.mean([d[c] for c in range(10)])
                       + 0.1 * (1 - sum(ch["mA"][i].sum() + ch["mB"][i].sum() for i in range(2))
                                / (sum(sA) + sum(sB))))
        gi = int(np.argmax(fs))
        if fs[gi] > bf:
            bf = fs[gi]
            bc = {"mA": [m.copy() for m in pop[gi]["mA"]],
                  "mB": [m.copy() for m in pop[gi]["mB"]],
                  "route": pop[gi]["route"].copy()}
        if gen % 10 == 0:
            m_ = bld_ent(bc, WA, WB, sA, sB, rA, rB)
            if m_:
                print(f"    Покоління {gen}: fitness={fs[gi]:.4f} "
                      f"min={min(pc(m_, Xv, yv)[c] for c in range(10)):.3f}")
        new = [{"mA": [m.copy() for m in bc["mA"]], "mB": [m.copy() for m in bc["mB"]],
                "route": bc["route"].copy()}]
        while len(new) < 20:
            ti = random.sample(range(len(pop)), 3)
            p1 = pop[ti[int(np.argmax([fs[i] for i in ti]))]]
            ch = {"mA": [m.copy() for m in p1["mA"]], "mB": [m.copy() for m in p1["mB"]],
                  "route": p1["route"] + np.random.randn(10) * 0.3}
            pf = max(0.02, 0.06 - gen * 0.001)
            for ms in [ch["mA"], ch["mB"]]:
                for m in ms:
                    f = np.random.random(len(m)) < pf
                    m[f] = ~m[f]
                    if m.sum() == 0:
                        m[np.random.randint(len(m))] = True
            new.append(ch)
        pop = new

    n_ent = sum(bc["mA"][i].sum() + bc["mB"][i].sum() for i in range(2))
    n_max = sum(sA) + sum(sB)
    return bc, bf, int(n_ent), int(n_max)


# ══════════════════════════════════════════════════════════
# ENT-FT: калібрація вихідного шару на замороженому екстракторі
# Формули 4.7-4.9, підрозд. 4.3.1
# ══════════════════════════════════════════════════════════

def calibrate_ent_ft(model: MLP, Xcal: torch.Tensor, ycal: torch.Tensor) -> MLP:
    model = copy.deepcopy(model)
    model.eval()
    with torch.no_grad():
        h = Xcal
        for layer in list(model.net)[:-1]:
            h = layer(h)
        features = h.numpy()  # формула 4.7: Z_val = Φ_ENT(X_val)

    scaler = StandardScaler()
    Z = scaler.fit_transform(features)  # Z' = (Z - μ) / σ

    y_np = ycal.numpy()
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
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print("=" * 70)
    print("  Ex30 — ENT-FT на точному чемпіоні e34_benchmark.py (табл. 4.4)")
    print("=" * 70)

    print("\n[1/5] Завантаження MNIST і відтворення батьківських моделей...")
    X_tr, y_tr, X_te, y_te = load_mnist()
    idx = torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(0))
    Xv, yv = X_tr[idx[15000:18000]], y_tr[idx[15000:18000]]  # валідація ENT, n=3000 (15%)
    Xc = X_tr[idx[:2000]]  # вибірка для virt2L-реконструкції ENT

    base = MLP(ARCH)  # неначений базис для Task Arithmetic (метод 3, для RNG-стану)
    torch.manual_seed(0)
    nn.init.xavier_uniform_(list(base.parameters())[0])
    base_sd = copy.deepcopy(base.state_dict())

    torch.manual_seed(SEED)
    t0 = time.time()
    modelA = train_model(ARCH, X_tr, y_tr, GROUP_A)
    modelB = train_model(ARCH, X_tr, y_tr, GROUP_B)
    t_train = time.time() - t0
    print(f"  Модель A (цифри 0-4): {ev(modelA, X_te, y_te):.3f}")
    print(f"  Модель B (цифри 5-9): {ev(modelB, X_te, y_te):.3f}")
    sdA, sdB = modelA.state_dict(), modelB.state_dict()

    print("\n[2/5] Відтворення методів 1-8 e34_benchmark.py (для ідентичності ГРС)...")
    t0 = time.time()
    replay_baseline_methods(sdA, sdB, base_sd, Xv, yv)
    t_replay = time.time() - t0

    print("\n[3/5] ENT — еволюційний пошук чемпіона (Pop=20, Gen=30, формули 4.2-4.6)...")
    t0 = time.time()
    best_ch, best_fit, n_ent, n_max = evolve_ent(modelA, modelB, Xc, Xv, yv)
    t_ent = time.time() - t0
    WA, sA = virt2l(modelA, Xc)
    WB, sB = virt2l(modelB, Xc)
    with torch.no_grad():
        sl = modelA(Xc).numpy().std()
        sr = modelB(Xc).numpy().std()
    t_ = (sl + sr) / 2
    rA = t_ / (sl + 1e-10)
    rB = t_ / (sr + 1e-10)
    champion = bld_ent(best_ch, WA, WB, sA, sB, rA, rB)
    assert champion is not None, "ENT: чемпіон порожній (усі маски занулено) — визначення хромосоми зламано"
    champion.eval()

    per_class_before = pc(champion, X_te, y_te)
    acc_before = ev(champion, X_te, y_te)
    balance_before = compute_balance(per_class_before)
    compression = round(1 - n_ent / n_max, 4)
    print(f"  Чемпіон ENT: acc={acc_before:.4f}  min_c={balance_before['min_class_acc']:.4f}  "
          f"баланс(міжгруп.)={balance_before['group_balance_min_over_max']:.4f}")

    # ── Контрольна звірка: чемпіон має бути біт-у-біт тим самим, що в results_e34.json ──
    print(f"\n[4/5] Звірка чемпіона з {CHAMPION_SOURCE} (табл. 4.3, 4.6)...")
    with open(SCRIPT_DIR / CHAMPION_SOURCE, encoding="utf-8") as f:
        e34_methods = json.load(f)
    target = next((r for r in e34_methods if r["name"] == "ENT"), None)
    assert target is not None, f"{CHAMPION_SOURCE}: запис методу 'ENT' не знайдено"
    target_acc, target_bal, target_min = target["acc"], target["bal"], target["min"]
    computed_acc = round(acc_before, 3)
    computed_bal = round(balance_before["group_balance_min_over_max"], 3)
    computed_min = round(balance_before["min_class_acc"], 3)
    assert abs(computed_acc - round(target_acc, 3)) < 5e-4, (
        f"РОЗБІЖНІСТЬ ACC: обчислено {computed_acc}, у {CHAMPION_SOURCE} {target_acc} — "
        "детермінізм e34_benchmark.py порушено, чемпіон не відтворився")
    assert abs(computed_bal - round(target_bal, 3)) < 5e-4, (
        f"РОЗБІЖНІСТЬ BAL: обчислено {computed_bal}, у {CHAMPION_SOURCE} {target_bal} — "
        "детермінізм e34_benchmark.py порушено, чемпіон не відтворився")
    assert abs(computed_min - round(target_min, 3)) < 5e-4, (
        f"РОЗБІЖНІСТЬ MIN: обчислено {computed_min}, у {CHAMPION_SOURCE} {target_min} — "
        "детермінізм e34_benchmark.py порушено, чемпіон не відтворився")
    print(f"  ЗБІГ ПІДТВЕРДЖЕНО: acc={computed_acc} bal={computed_bal} min={computed_min} "
          f"(ціль: {target_acc}/{target_bal}/{target_min})")

    # ── ENT-FT: калібрація на Xv (n=3000, ідентична вибірка EA-валідації e34) ──
    print("\n[5/5] ENT-FT — калібрація вихідного шару на Xv (n=3000, формули 4.7-4.9)...")
    t0 = time.time()
    champion_ft = calibrate_ent_ft(champion, Xv, yv)
    t_calib = time.time() - t0
    per_class_after = pc(champion_ft, X_te, y_te)
    acc_after = ev(champion_ft, X_te, y_te)
    balance_after = compute_balance(per_class_after)
    print(f"  Після калібрації: acc={acc_after:.4f}  min_c={balance_after['min_class_acc']:.4f}  "
          f"баланс(міжгруп.)={balance_after['group_balance_min_over_max']:.4f}")

    t_total = time.time() - t_start
    report: dict[str, Any] = {
        "config": {
            "seed": SEED,
            "arch": ARCH,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "ea": {"pop_size": 20, "n_gen": 30, "weights_w1_w2_w3_w4": [0.4, 0.4, 0.1, 0.1]},
            "calibration": {
                "formulas": "4.7-4.9 (підрозд. 4.3.1)",
                "calibration_set": "Xv — валідаційна вибірка e34_benchmark.py (EA-валідація ENT), "
                                    "n=3000, idx[15000:18000] з randperm(20000, seed=0), 15% n_train",
                "logistic_regression": {"solver": "lbfgs", "C": 1.0,
                                         "class_weight": "balanced", "max_iter": 500},
                "sample_weight": "1/count(клас), нормовано на суму ваг = n",
            },
        },
        "champion_verification": {
            "source_file": CHAMPION_SOURCE,
            "target_acc": target_acc, "target_bal": target_bal, "target_min": target_min,
            "computed_acc": computed_acc, "computed_bal": computed_bal, "computed_min": computed_min,
            "matched": True,
        },
        "before_calibration": {
            "accuracy": acc_before,
            "per_class": per_class_before,
            "balance": balance_before,
            "compression_ratio": compression,
            "ea_fitness": best_fit,
        },
        "after_calibration_ent_ft": {
            "accuracy": acc_after,
            "per_class": per_class_after,
            "balance": balance_after,
        },
        "delta": {
            "accuracy": acc_after - acc_before,
            "min_class_acc": balance_after["min_class_acc"] - balance_before["min_class_acc"],
            "group_balance_min_over_max": (balance_after["group_balance_min_over_max"]
                                            - balance_before["group_balance_min_over_max"]),
        },
        "timing_s": {
            "train_parents": t_train, "replay_baseline_methods_1_8": t_replay,
            "ent_search": t_ent, "calibration": t_calib, "total": t_total,
        },
    }

    out_path = SCRIPT_DIR / "results_ent_ft_on_e34.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("  ПІДСУМОК (табл. 4.4)")
    print("=" * 70)
    print(f"  {'Етап':<22} {'Точність':>10} {'min_c':>8} {'Баланс(міжгр.)':>16}")
    print(f"  {'До калібрації (ENT)':<22} {acc_before:>10.4f} "
          f"{balance_before['min_class_acc']:>8.4f} "
          f"{balance_before['group_balance_min_over_max']:>16.4f}")
    print(f"  {'Після (ENT-FT)':<22} {acc_after:>10.4f} "
          f"{balance_after['min_class_acc']:>8.4f} "
          f"{balance_after['group_balance_min_over_max']:>16.4f}")
    print(f"\n  Звіт збережено: {out_path}")


if __name__ == "__main__":
    main()
