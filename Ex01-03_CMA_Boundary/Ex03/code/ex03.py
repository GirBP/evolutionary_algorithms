# Ex03: Порівняння градієнтних та еволюційних методів на пошуку параметрів НМ
# Датасети: Make Moons, Make Classification (20 ознак), Digits (10 класів).
# Методи: Adam, CMA-ES, L-SHADE, CLPSO, RTS, RS (Random Search).
#
# Гіпотези (F1-a): H0 — розподіл Accuracy (або F1) за фіксованим бюджетом часу однаковий
# для EA та градієнтних методів; H1 — розподіли різняться. Перевірка: тест Фрідмана / Mann-Whitney.
#
# Протокол: бюджет за часом (с); train/val/test 60/20/20; рання зупинка лише по val;
# фінальна оцінка лише на test. Для k≥3 методів — тест Фрідмана.

import sys
import os
from pathlib import Path

# Примітка: цей sys.path.insert є резервним — на практиці `common` вже
# імпортований (і закешований) через ex03_run.py, який коректно резолвить ROOT
# до кореня репозиторію перед імпортом цього модуля.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common import ensure_experiment_dependencies
ensure_experiment_dependencies()

# Однаковий вплив на всі методи (ПРАВИЛА_ЕКСПЕРИМЕНТІВ п. 4.8): 1 потік у воркері, паралелізм лише на рівні запусків
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time

from common.rcu import profile_rcu as _profile_rcu, setup_rcu_worker, ANCHOR_LOOPS, anchor_ns as _get_anchor_time_ns
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.datasets import make_moons, make_classification, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from torch.nn.utils import parameters_to_vector, vector_to_parameters

try:
    import cma
except ImportError:
    cma = None

# Два режими (закон проєкту: quick = миттєво + графіки, run = аналітичні дані):
#   Quick (CONFIG_TEST) — швидко, але графіки оцінювані: кілька runs і методів для порівняння.
#   Експериментальний — параметри для статистично значущих результатів (Friedman + Nemenyi).
CONFIG_TEST = {
    "n_runs": 3,
    "rcu_budget": 200,
    "target_acc": 0.99,
    "seed_base": 42,
    "datasets": ["moons"],
    "methods": ["Adam", "CMA-ES", "L-SHADE"],
}
CONFIG_EXPERIMENT = {
    "n_runs": 5,
    "rcu_budget": 3000,
    "target_acc": 0.99,
    "seed_base": 42,
    "datasets": ["moons", "classification20", "digits"],
    "methods": ["Adam", "CMA-ES", "L-SHADE", "CLPSO", "RTS", "RS"],
}
# За замовчуванням — тестовий, щоб випадковий запуск не тривав годинами
CONFIG = dict(CONFIG_TEST)

# Ex03: тільки CPU (паралелізм на центральному процесорі, без GPU/MPS)
DEVICE = torch.device("cpu")

BOUNDS_NN = (-3.0, 3.0)  # діапазон для ініціалізації ваг ЕА


# ==========================================
# 1. ДАТАСЕТИ
# ==========================================

def get_data(dataset_name, seed):
    """Повертає ((X_tr, y_tr), (X_val, y_val_np), (X_te, y_te_np), data_np, n_features, n_classes).
    Розбиття 60% train / 20% val / 20% test. Val — для early stop; test — лише фінальна оцінка (F4-b).
    """
    rng = np.random.RandomState(seed)
    if dataset_name == "moons":
        X, y = make_moons(n_samples=1000, noise=0.1, random_state=seed)
        n_classes = 2
    elif dataset_name == "classification20":
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=12, n_redundant=4,
            n_clusters_per_class=2, n_classes=2, random_state=seed
        )
        n_classes = 2
    elif dataset_name == "digits":
        data = load_digits()
        X, y = data.data.astype(np.float32), data.target  # type: ignore[union-attr]
        if len(X) > 1200:
            idx = rng.choice(len(X), 1200, replace=False)
            X, y = X[idx], y[idx]
        n_classes = 10
    else:
        raise ValueError("Unknown dataset: " + dataset_name)

    # 80% train+val, 20% test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    # 75% of 80% = 60% train, 25% of 80% = 20% val
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=seed + 1, stratify=y_tv
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    n_features = X_train.shape[1]

    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.LongTensor(y_train) if n_classes > 2 else torch.FloatTensor(y_train).unsqueeze(1)
    y_tr = y_tr.to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    X_te = torch.FloatTensor(X_test).to(DEVICE)
    y_te_np = y_test
    data_np = (X_test, y_test)

    return (X_tr, y_tr), (X_val_t, y_val), (X_te, y_te_np), data_np, n_features, n_classes


# ==========================================
# 2. МОДЕЛІ
# ==========================================

def build_model(dataset_name, n_features, n_classes):
    if n_classes == 2:
        return MLPBinary(n_features, hidden=32)
    return MLPMulti(n_features, n_classes, hidden=64)


class MLPBinary(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class MLPMulti(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
        self.n_classes = n_classes

    def forward(self, x):
        return self.net(x)


def calculate_metrics(model, X_tensor, y_true_np, n_classes):
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
    if n_classes == 2:
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        return {
            "Accuracy": accuracy_score(y_true_np, preds),
            "F1-Score": f1_score(y_true_np, preds, average="weighted"),
            "ROC-AUC": roc_auc_score(y_true_np, probs),
            "Log-Loss": log_loss(y_true_np, probs),
        }
    else:
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        # Обчислюємо ROC-AUC для мультикласових задач через OVR
        try:
            roc_auc = roc_auc_score(y_true_np, probs, multi_class='ovr', average='weighted')
        except (ValueError, Exception):
            # Якщо обчислення не вдається (рідкісні класи або інші проблеми), записуємо nan
            roc_auc = np.nan
        return {
            "Accuracy": accuracy_score(y_true_np, preds),
            "F1-Score": f1_score(y_true_np, preds, average="weighted"),
            "ROC-AUC": roc_auc,
            "Log-Loss": log_loss(y_true_np, probs),
        }


# ==========================================
# 3. ОБгортка для ЕА (оптимізація ваг за часом)
# ==========================================

class NNProblem:
    """Проблема для ЕА: мінімізація train loss; зупинка за часом або за val_acc >= target."""
    def __init__(self, model, X_tr, y_tr, criterion, rcu_budget, start_time, anchor_s, bounds=BOUNDS_NN, val_data=None, target_acc=None):
        self.model = model
        self.X_tr, self.y_tr = X_tr, y_tr
        self.criterion = criterion
        self.rcu_budget = rcu_budget
        self.anchor_s = anchor_s
        self.start_time = start_time
        self.bounds = bounds
        self.dim = sum(p.numel() for p in model.parameters())
        self._stop = False
        self.best_x = None
        self.val_data = val_data  # (X_val, y_val_np, n_classes) or None
        self.target_acc = float(target_acc) if target_acc is not None else None

    def check_val_stop(self, x):
        """Якщо val_acc >= target_acc — зупиняємо EA (early stop)."""
        if self.val_data is None or self.target_acc is None:
            return
        X_val, y_val_np, n_classes = self.val_data
        x = np.asarray(x, dtype=np.float32)
        vector_to_parameters(torch.tensor(x, device=DEVICE), self.model.parameters())
        acc = calculate_metrics(self.model, X_val, y_val_np, n_classes)["Accuracy"]
        if acc >= self.target_acc:
            self._stop = True

    def __call__(self, x):
        elapsed_rcu = (time.thread_time() - self.start_time) / self.anchor_s
        if elapsed_rcu > self.rcu_budget:
            self._stop = True
            return 1e10
        x = np.asarray(x, dtype=np.float32)
        vector_to_parameters(torch.tensor(x, device=DEVICE), self.model.parameters())
        with torch.no_grad():
            loss = self.criterion(self.model(self.X_tr), self.y_tr).item()
        return float(loss)


# ==========================================
# 4. ЕВОЛЮЦІЙНІ АЛГОРИТМИ (адаптовані під час)
# ==========================================

class L_SHADE_NN:
    def __init__(self, problem, seed):
        self.p = problem
        self.rng = np.random.default_rng(seed)
        self.dim = problem.dim
        self.pop_size = min(18 * self.dim, 200)
        self.pop_size = max(self.pop_size, 10)
        self.mem_sz = 5
        self.mem_sf = np.full(5, 0.5)
        self.mem_cr = np.full(5, 0.5)
        self.k = 0
        self.lb, self.ub = problem.bounds[0], problem.bounds[1]

    def run(self):
        pop = self.rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fit = np.array([self.p(x) for x in pop])
        best_idx = np.argmin(fit)
        h_time, h_loss = [0.0], [float(fit.min())]
        h_best_x = [pop[best_idx].copy()]

        while not getattr(self.p, "_stop", False):
            idx = self.rng.integers(0, self.mem_sz, self.pop_size)
            cr = np.clip(self.rng.normal(self.mem_cr[idx], 0.1), 0, 1)
            f = np.clip(self.mem_sf[idx] + 0.1 * self.rng.standard_cauchy(self.pop_size), 0.01, 1)
            pbest = np.argsort(fit)[: max(int(self.pop_size * 0.11), 2)]
            r1, r2 = self.rng.integers(0, self.pop_size, (2, self.pop_size))
            v = pop + f[:, None] * (pop[self.rng.choice(pbest, self.pop_size)] - pop) + f[:, None] * (pop[r1] - pop[r2])
            v = np.clip(v, self.lb, self.ub)
            mask = self.rng.random((self.pop_size, self.dim)) < cr[:, None]
            mask[np.arange(self.pop_size), self.rng.integers(0, self.dim, self.pop_size)] = True
            trial = np.where(mask, v, pop)
            t_fit = np.array([self.p(x) for x in trial])
            if getattr(self.p, "_stop", False):
                break
            better = t_fit < fit
            if np.any(better):
                df = fit[better] - t_fit[better]
                w = df / (df.sum() + 1e-10)
                self.mem_sf[self.k] = np.sum(w * f[better] ** 2) / (np.sum(w * f[better]) + 1e-10)
                self.mem_cr[self.k] = np.sum(w * cr[better])
                self.k = (self.k + 1) % self.mem_sz
                pop[better] = trial[better]
                fit[better] = t_fit[better]
            best_idx = np.argmin(fit)
            self.p.best_x = pop[best_idx].copy()
            if hasattr(self.p, "check_val_stop"):
                self.p.check_val_stop(self.p.best_x)
            h_time.append(time.thread_time() - self.p.start_time)  # RCU metric
            h_loss.append(float(fit.min()))
            h_best_x.append(self.p.best_x.copy())
        return h_time, h_loss, h_best_x


class CLPSO_NN:
    def __init__(self, problem, seed):
        self.p = problem
        self.rng = np.random.default_rng(seed)
        self.dim = problem.dim
        self.pop_size = min(40, max(10, self.dim))
        self.w = 0.9
        self.lb, self.ub = problem.bounds[0], problem.bounds[1]

    def run(self):
        pos = self.rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        vel = np.zeros_like(pos)
        pbest = pos.copy()
        fit = np.array([self.p(x) for x in pos])
        if getattr(self.p, "_stop", False):
            best_idx = np.argmin(fit)
            self.p.best_x = pos[best_idx].copy()
            return [0.0], [float(fit.min())], [self.p.best_x.copy()]
        pbest_fit = fit.copy()
        best_idx = np.argmin(pbest_fit)
        h_time, h_loss = [0.0], [float(pbest_fit.min())]
        h_best_x = [pbest[best_idx].copy()]

        while not getattr(self.p, "_stop", False):
            self.w *= 0.9995
            r1, r2 = self.rng.random((2, self.pop_size, self.dim))
            gbest = pbest[np.argmin(pbest_fit)]
            vel = self.w * vel + 1.49 * r1 * (pbest - pos) + 1.49 * r2 * (gbest - pos)
            pos = np.clip(pos + vel, self.lb, self.ub)
            fit = np.array([self.p(x) for x in pos])
            if getattr(self.p, "_stop", False):
                break
            impr = fit < pbest_fit
            pbest[impr] = pos[impr]
            pbest_fit[impr] = fit[impr]
            best_idx = np.argmin(pbest_fit)
            self.p.best_x = pbest[best_idx].copy()
            if hasattr(self.p, "check_val_stop"):
                self.p.check_val_stop(self.p.best_x)
            h_time.append(time.thread_time() - self.p.start_time)  # RCU metric
            h_loss.append(float(pbest_fit.min()))
            h_best_x.append(self.p.best_x.copy())
        return h_time, h_loss, h_best_x


class RTS_NN:
    def __init__(self, problem, seed):
        self.p = problem
        self.rng = np.random.default_rng(seed)
        self.dim = problem.dim
        self.pop_size = min(50, max(10, self.dim * 2))
        self.lb, self.ub = problem.bounds[0], problem.bounds[1]

    def run(self):
        pop = self.rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fit = np.array([self.p(x) for x in pop])
        if getattr(self.p, "_stop", False):
            best_idx = np.argmin(fit)
            self.p.best_x = pop[best_idx].copy()
            return [0.0], [float(fit.min())], [self.p.best_x.copy()]
        best_idx = np.argmin(fit)
        h_time, h_loss = [0.0], [float(fit.min())]
        h_best_x = [pop[best_idx].copy()]

        while not getattr(self.p, "_stop", False):
            idx = self.rng.integers(0, self.pop_size, 3)
            p1 = pop[idx[0]] if fit[idx[0]] < fit[idx[1]] else pop[idx[1]]
            off = np.clip(p1 + self.rng.normal(0, 0.5, self.dim), self.lb, self.ub)
            off_fit = self.p(off)
            if getattr(self.p, "_stop", False):
                break
            closest = np.argmin(np.linalg.norm(pop - off, axis=1))
            if off_fit < fit[closest]:
                pop[closest] = off
                fit[closest] = off_fit
            best_idx = np.argmin(fit)
            self.p.best_x = pop[best_idx].copy()
            if hasattr(self.p, "check_val_stop"):
                self.p.check_val_stop(self.p.best_x)
            h_time.append(time.thread_time() - self.p.start_time)  # RCU metric
            h_loss.append(float(fit.min()))
            h_best_x.append(self.p.best_x.copy())
        return h_time, h_loss, h_best_x


# ==========================================
# 5. RUNNERS: Adam, CMA-ES, L-SHADE, CLPSO, RTS
# ==========================================

def run_adam(run_id, dataset_name, data, config=None):
    cfg = config or CONFIG
    (X_tr, y_tr), (X_val, y_val_np), (X_te, y_te_np), _, n_features, n_classes = data
    torch.manual_seed(cfg["seed_base"] + run_id)

    model = build_model(dataset_name, n_features, n_classes).to(DEVICE)
    criterion = nn.BCELoss() if n_classes == 2 else nn.CrossEntropyLoss()
    y_tr_ = y_tr

    optimizer = optim.Adam(model.parameters(), lr=0.05)
    anchor_s = cfg.get("_anchor_s", 1.0)  # секунди anchor для нормалізації в RCU
    history = {"time": [0.0], "acc": [], "snapshots": []}
    val_metrics = calculate_metrics(model, X_val, y_val_np, n_classes)
    history["acc"].append(val_metrics["Accuracy"])
    history["snapshots"].append((0.0, _state_dict_to_cpu(model.state_dict())))

    start_time = time.thread_time()
    nfe_count = 0

    while True:
        elapsed = time.thread_time() - start_time
        elapsed_rcu = elapsed / anchor_s
        if elapsed_rcu > cfg["rcu_budget"]:
            break
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr_)
        loss.backward()
        optimizer.step()
        nfe_count += 1
        if nfe_count % 10 == 0:

            val_metrics = calculate_metrics(model, X_val, y_val_np, n_classes)
            history["time"].append(elapsed_rcu)
            history["acc"].append(val_metrics["Accuracy"])
            history["snapshots"].append((elapsed_rcu, _state_dict_to_cpu(model.state_dict())))
            if val_metrics["Accuracy"] >= cfg["target_acc"]:
                break

    final_metrics = calculate_metrics(model, X_te, y_te_np, n_classes)
    final_metrics["NFE"] = nfe_count
    return final_metrics, history


def run_cma_es(run_id, dataset_name, data, config=None):
    if cma is None:
        raise RuntimeError("pip install cma")
    cfg = config or CONFIG
    (X_tr, y_tr), (X_val, y_val_np), (X_te, y_te_np), _, n_features, n_classes = data
    torch.manual_seed(cfg["seed_base"] + run_id)
    np.random.seed(cfg["seed_base"] + run_id)

    model = build_model(dataset_name, n_features, n_classes).to(DEVICE)
    criterion = nn.BCELoss() if n_classes == 2 else nn.CrossEntropyLoss()
    y_tr_ = y_tr if n_classes == 2 else y_tr

    anchor_s = cfg.get("_anchor_s", 1.0)
    start_time = time.thread_time()
    problem = NNProblem(model, X_tr, y_tr_, criterion, cfg["rcu_budget"], start_time, anchor_s)
    param_vector = parameters_to_vector(model.parameters()).detach().cpu().numpy()
    lb, ub = problem.bounds[0], problem.bounds[1]
    es = cma.CMAEvolutionStrategy(param_vector.copy(), 0.3, {"verbose": -9, "popsize": 16, "bounds": [lb, ub]})

    # Буфер для уникнення алокацій torch.tensor кожну ітерацію
    buf_tensor = torch.zeros(len(param_vector), dtype=torch.float32, device=DEVICE)

    history = {"time": [0.0], "acc": [], "snapshots": []}
    val_metrics = calculate_metrics(model, X_val, y_val_np, n_classes)
    history["acc"].append(val_metrics["Accuracy"])
    history["snapshots"].append((0.0, _state_dict_to_cpu(model.state_dict())))
    nfe_count = 0
    gen_count = 0

    while not es.stop() and not getattr(problem, "_stop", False):
        elapsed = time.thread_time() - start_time
        elapsed_rcu = elapsed / anchor_s
        if elapsed_rcu > cfg["rcu_budget"]:
            break
        solutions = es.ask()
        fitnesses = []
        for s in solutions:
            buf_tensor.copy_(torch.from_numpy(np.asarray(s, dtype=np.float32)))
            vector_to_parameters(buf_tensor, model.parameters())
            with torch.no_grad():
                loss_val = criterion(model(X_tr), y_tr_).item()
            fitnesses.append(loss_val)
            nfe_count += 1
        es.tell(solutions, fitnesses)
        best_sol = solutions[np.argmin(fitnesses)]
        buf_tensor.copy_(torch.from_numpy(np.asarray(best_sol, dtype=np.float32)))
        vector_to_parameters(buf_tensor, model.parameters())
        gen_count += 1

        # Логування кожне покоління, snapshot кожні 5
        model.eval()
        with torch.no_grad():
            logits = model(X_val)
        if n_classes == 2:
            preds = (torch.sigmoid(logits).cpu().numpy().flatten() > 0.5).astype(int)
        else:
            preds = logits.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_val_np, preds)
        history["time"].append(elapsed_rcu)
        history["acc"].append(acc)
        if gen_count % 5 == 0:
            history["snapshots"].append((elapsed_rcu, _state_dict_to_cpu(model.state_dict())))
        if acc >= cfg["target_acc"]:
            break

    # Фінальний snapshot
    elapsed_rcu = (time.thread_time() - start_time) / anchor_s
    history["snapshots"].append((elapsed_rcu, _state_dict_to_cpu(model.state_dict())))

    final_metrics = calculate_metrics(model, X_te, y_te_np, n_classes)
    final_metrics["NFE"] = nfe_count
    return final_metrics, history



def _run_ea(run_id, dataset_name, data, config, ea_class):
    cfg = config or CONFIG
    (X_tr, y_tr), (X_val, y_val_np), (X_te, y_te_np), _, n_features, n_classes = data
    torch.manual_seed(cfg["seed_base"] + run_id)
    np.random.seed(cfg["seed_base"] + run_id)

    model = build_model(dataset_name, n_features, n_classes).to(DEVICE)
    criterion = nn.BCELoss() if n_classes == 2 else nn.CrossEntropyLoss()
    y_tr_ = y_tr if n_classes == 2 else y_tr

    anchor_s = cfg.get("_anchor_s", 1.0)
    start_time = time.thread_time()
    val_data = (X_val, y_val_np, n_classes)
    problem = NNProblem(
        model, X_tr, y_tr_, criterion, cfg["rcu_budget"], start_time, anchor_s,
        val_data=val_data, target_acc=cfg.get("target_acc"),
    )
    ea = ea_class(problem, cfg["seed_base"] + run_id)
    h_time, h_loss, h_best_x = ea.run()

    if getattr(problem, "best_x", None) is not None:
        vector_to_parameters(torch.tensor(problem.best_x, dtype=torch.float32, device=DEVICE), model.parameters())

    final_metrics = calculate_metrics(model, X_te, y_te_np, n_classes)
    nfe_count = len(h_time) * (ea.pop_size if hasattr(ea, "pop_size") else 1)
    final_metrics["NFE"] = nfe_count

    step = max(1, len(h_best_x) // 50)
    sampled = []
    for i, (t, x) in enumerate(zip(h_time, h_best_x)):
        if i % step == 0 or i == len(h_best_x) - 1:
            if x is not None:
                vector_to_parameters(torch.tensor(x, dtype=torch.float32, device=DEVICE), model.parameters())
            acc = calculate_metrics(model, X_val, y_val_np, n_classes)["Accuracy"]
            sampled.append((i, acc))
    if not sampled:
        sampled = [(0, final_metrics["Accuracy"])]
    j = 0
    history_acc = []
    for pos in range(len(h_best_x)):
        while j < len(sampled) and sampled[j][0] <= pos:
            j += 1
        history_acc.append(sampled[j - 1][1] if j > 0 else final_metrics["Accuracy"])
    # Нормалізація часу EA в RCU
    history = {"time": [t / anchor_s for t in h_time], "acc": history_acc}
    history["snapshots"] = [
        (h_time[0], _state_dict_to_cpu(model.state_dict())),
        (h_time[len(h_time) // 2] if len(h_time) > 1 else h_time[0], _state_dict_to_cpu(model.state_dict())),
        (h_time[-1], _state_dict_to_cpu(model.state_dict())),
    ]
    return final_metrics, history


def run_lshade(run_id, dataset_name, data, config=None):
    return _run_ea(run_id, dataset_name, data, config, L_SHADE_NN)


def run_clpso(run_id, dataset_name, data, config=None):
    return _run_ea(run_id, dataset_name, data, config, CLPSO_NN)


def run_rts(run_id, dataset_name, data, config=None):
    return _run_ea(run_id, dataset_name, data, config, RTS_NN)


def run_random_search(run_id, dataset_name, data, config=None):
    """Baseline F2-b: випадковий пошук ваг у bounds з тим самим бюджетом часу."""
    cfg = config or CONFIG
    (X_tr, y_tr), (X_val, y_val_np), (X_te, y_te_np), _, n_features, n_classes = data
    torch.manual_seed(cfg["seed_base"] + run_id)
    np.random.seed(cfg["seed_base"] + run_id)
    rng = np.random.default_rng(cfg["seed_base"] + run_id)

    model = build_model(dataset_name, n_features, n_classes).to(DEVICE)
    criterion = nn.BCELoss() if n_classes == 2 else nn.CrossEntropyLoss()
    y_tr_ = y_tr if n_classes == 2 else y_tr
    dim = sum(p.numel() for p in model.parameters())
    lb, ub = BOUNDS_NN[0], BOUNDS_NN[1]

    anchor_s = cfg.get("_anchor_s", 1.0)
    start_time = time.thread_time()
    best_x = parameters_to_vector(model.parameters()).detach().cpu().numpy().copy()
    with torch.no_grad():
        best_loss = criterion(model(X_tr), y_tr_).item()
    history = {"time": [0.0], "acc": []}
    val_acc0 = calculate_metrics(model, X_val, y_val_np, n_classes)["Accuracy"]
    history["acc"].append(val_acc0)
    nfe_count = 0
    log_interval = max(1, dim // 50)

    while True:
        elapsed = time.thread_time() - start_time
        elapsed_rcu = elapsed / anchor_s
        if elapsed_rcu > cfg["rcu_budget"]:
            break
        x = rng.uniform(lb, ub, size=dim).astype(np.float32)
        vector_to_parameters(torch.tensor(x, device=DEVICE), model.parameters())
        with torch.no_grad():
            loss = criterion(model(X_tr), y_tr_).item()
        nfe_count += 1
        if loss < best_loss:
            best_loss = loss
            best_x = x.copy()
        if nfe_count % log_interval == 0:
            elapsed_rcu = elapsed / anchor_s
            vector_to_parameters(torch.tensor(best_x, device=DEVICE), model.parameters())
            val_metrics = calculate_metrics(model, X_val, y_val_np, n_classes)
            history["time"].append(elapsed_rcu)
            history["acc"].append(val_metrics["Accuracy"])
            if val_metrics["Accuracy"] >= cfg["target_acc"]:
                break

    vector_to_parameters(torch.tensor(best_x, device=DEVICE), model.parameters())
    elapsed_final = time.thread_time() - start_time
    elapsed_final_rcu = elapsed_final / anchor_s
    final_val_acc = calculate_metrics(model, X_val, y_val_np, n_classes)["Accuracy"]
    if not history["time"] or history["time"][-1] < elapsed_final_rcu - 0.01:
        history["time"].append(elapsed_final_rcu)
        history["acc"].append(final_val_acc)
    final_metrics = calculate_metrics(model, X_te, y_te_np, n_classes)
    final_metrics["NFE"] = nfe_count
    history["snapshots"] = [
        (0.0, _state_dict_to_cpu(model.state_dict())),
        (history["time"][-1] / 2 if len(history["time"]) > 1 else 0.0, _state_dict_to_cpu(model.state_dict())),
        (history["time"][-1], _state_dict_to_cpu(model.state_dict())),
    ]
    return final_metrics, history


def _state_dict_to_cpu(sd):
    """Копія state_dict як numpy-масиви для pickle-серіалізації через ProcessPoolExecutor."""
    return {k: v.detach().cpu().numpy().copy() for k, v in sd.items()}


def _state_dict_to_json(sd):
    """Конвертує state_dict (numpy або torch) в JSON-сумісний формат (list)."""
    out = {}
    for k, v in sd.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v.detach().cpu().numpy().tolist()
    return out


def _run_single_job(args):
    """Один (dataset, method, run_id) для ProcessPoolExecutor. Повертає (met, hist, dataset_name, method_name, run_id)."""
    setup_rcu_worker()  # P-ядра + ініціалізація
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    dataset_name, method_name, run_id, cfg = args
    # Виміряти anchor для нормалізації конвергенції всередині runner
    anchor_ns_val = _get_anchor_time_ns()
    anchor_s = anchor_ns_val / 1e9
    cfg = dict(cfg)  # копія, щоб не мутувати оригінал
    cfg["_anchor_s"] = anchor_s
    data = get_data(dataset_name, cfg["seed_base"])
    runner = RUNNERS.get(method_name)
    if runner is None:
        return None
    # RCU metric: wrap the entire runner in sandwich profiling
    (met, hist), rcu, anc_pre, anc_post = _profile_rcu(runner, run_id, dataset_name, data, cfg)
    met["Time_RCU"] = rcu
    met["Anchor_avg_ms"] = (anc_pre + anc_post) / 2.0 / 1e6

    # Rescale convergence time axis from single-point anchor to sandwich average.
    # Budget enforcement inside runners uses anchor_s (single pre-anchor);
    # the authoritative RCU uses sandwich avg. Rescale so both are consistent.
    sandwich_anchor_s = (anc_pre + anc_post) / 2.0 / 1e9  # ns → s
    if sandwich_anchor_s > 0 and anchor_s > 0 and hist.get("time"):
        scale = anchor_s / sandwich_anchor_s
        hist["time"] = [t * scale for t in hist["time"]]

    return (met, hist, dataset_name, method_name, run_id)


RUNNERS = {
    "Adam": run_adam,
    "CMA-ES": run_cma_es,
    "L-SHADE": run_lshade,
    "CLPSO": run_clpso,
    "RTS": run_rts,
    "RS": run_random_search,
}


# ==========================================
# 6. ОРКЕСТРАЦІЯ
# ==========================================

def run_full_experiment(n_runs=None, config=None, datasets=None, methods=None):
    """
    Запускає повний експеримент: датасети × методи × n_runs.
    Паралелізм: ProcessPoolExecutor, як у Ex02. Повертає df_final, df_convergence, metadata.
    """
    cfg = config or CONFIG
    n = n_runs if n_runs is not None else cfg["n_runs"]
    ds_list = datasets or cfg["datasets"]
    method_list = methods or cfg["methods"]

    tasks = [
        (dataset_name, method_name, run_id, cfg)
        for dataset_name in ds_list
        for method_name in method_list
        for run_id in range(n)
    ]
    total_jobs = len(tasks)
    W = min(os.cpu_count() or 4, total_jobs)
    if W < 1:
        W = 1

    results = []
    with ProcessPoolExecutor(max_workers=W) as executor:
        futures = [executor.submit(_run_single_job, t) for t in tasks]
        for future in as_completed(futures):
            out = future.result()
            if out is not None:
                results.append(out)

    results.sort(key=lambda x: (x[2], x[3], x[4]))

    raw_results = []
    rows_conv = []
    histories = {}
    snapshots_run0 = {}

    for met, hist, dataset_name, method_name, run_id in results:
        met["Dataset"] = dataset_name
        met["Method"] = method_name
        met["Run"] = run_id
        raw_results.append(met)

        # hist["time"] вже в RCU (нормалізовано всередині runner)
        for t, a in zip(hist["time"], hist["acc"]):
            rows_conv.append({"Dataset": dataset_name, "Method": method_name, "Run": run_id, "Time": t, "Accuracy": a})

        key = (dataset_name, method_name)
        if key not in histories:
            histories[key] = []
        total_t = hist["time"][-1] if hist["time"] else 0
        snaps = hist["snapshots"]
        times_s = [s[0] for s in snaps]
        idx_mid = np.abs(np.array(times_s) - (total_t / 2)).argmin() if len(times_s) > 1 else 0
        selected = {"Start": snaps[0][1], "Mid": snaps[idx_mid][1], "End": snaps[-1][1]}
        histories[key].append({
            "times": hist["time"],
            "accs": hist["acc"],
            "visuals": [selected["Start"], selected["Mid"], selected["End"]],
        })

        if dataset_name == "moons" and run_id == 0 and method_name in ("Adam", "CMA-ES", "RTS"):
            snapshots_run0[method_name] = {
                "Start": _state_dict_to_json(selected["Start"]),
                "Mid": _state_dict_to_json(selected["Mid"]),
                "End": _state_dict_to_json(selected["End"]),
            }

    df_final = pd.DataFrame(raw_results)
    df_convergence = pd.DataFrame(rows_conv)

    # Обчислити середній anchor з усіх прогонів
    anchor_values = [r.get("Anchor_avg_ms", 0) for r in raw_results if r.get("Anchor_avg_ms")]
    anchor_avg_ms = sum(anchor_values) / len(anchor_values) if anchor_values else 0.0

    metadata = {
        "n_runs": n,
        "rcu_budget": int(cfg["rcu_budget"]),
        "anchor_avg_ms": round(anchor_avg_ms, 3),
        "target_acc": float(cfg["target_acc"]),
        "seed_base": int(cfg["seed_base"]),
        "snapshots_run0": snapshots_run0,
        "datasets": ds_list,
        "methods": method_list,
        "protocol": {
            "datasets": ds_list,
            "methods": method_list,
            "single_split": True,
            "n_methods": len(method_list),
        },
    }
    return df_final, df_convergence, metadata
