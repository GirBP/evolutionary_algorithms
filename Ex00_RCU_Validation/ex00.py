# Ex00: RCU Metric Validation (Relative Compute Units)
# Research-grade benchmark: статистичні тести, множинні стрес-умови, різноманітні навантаження.
# Запуск: python Ex00/ex00.py
#
# Статистичний апарат:
#   • Bootstrap 95% CI (10 000 ітерацій)
#   • Wilcoxon signed-rank test (попарне порівняння умов)
#   • Cohen's d (розмір ефекту)
#   • Vargha-Delaney A (непараметричний effect size)
#   • Коефіцієнт варіації (CV)
#   • R² лінійної регресії (лінійність масштабування)

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import time
import threading
import gc
import ctypes
import ctypes.util
import platform
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy import stats as sp_stats

from common import save_figure, set_dstu_style, ensure_dir
set_dstu_style()

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ensure_dir(RESULTS_DIR)

np.random.seed(42)

# ============================================================
# ПАРАМЕТРИ ЕКСПЕРИМЕНТУ
# ============================================================
N_TRIALS = 30          # замірів на умову (≥30 для CLT)
N_WARMUP = 5           # прогрівних замірів (відкидаються)
N_BOOTSTRAP = 10_000   # ітерацій для bootstrap CI
ALPHA = 0.05           # рівень значущості

# ============================================================
# RCU PROTOCOL
# ============================================================
_tls = threading.local()

def _init_anchor():
    if not hasattr(_tls, "ok"):
        _tls.a = np.random.rand(1024)
        _tls.b = np.random.rand(1024)
        _tls.c = np.empty(1024)
        _tls.loops = 50
        while True:
            t0 = time.thread_time_ns()
            for _ in range(_tls.loops):
                np.multiply(_tls.a, _tls.b, out=_tls.c)
                np.add(_tls.c, _tls.a, out=_tls.c)
            if (time.thread_time_ns() - t0) >= 1_000_000:  # 1ms anchor (оптимум: drift 0.4%, overhead 14%)
                break
            _tls.loops *= 2
        _tls.ok = True

def anchor_ns():
    _init_anchor()
    t0 = time.thread_time_ns()
    for _ in range(_tls.loops):
        np.multiply(_tls.a, _tls.b, out=_tls.c)
        np.add(_tls.c, _tls.a, out=_tls.c)
    return time.thread_time_ns() - t0

def measure(func, *a, **kw):
    """Вимірює 4 метрики одночасно: Wall, Process, Thread, RCU + анкер."""
    _init_anchor()
    apre = anchor_ns()
    tw0 = time.perf_counter_ns()
    tp0 = time.process_time_ns()
    tt0 = time.thread_time_ns()
    result = func(*a, **kw)
    tt = time.thread_time_ns() - tt0
    tp = time.process_time_ns() - tp0
    tw = time.perf_counter_ns() - tw0
    apost = anchor_ns()
    anchor_avg = (apre + apost) / 2
    rcu = tt / max(anchor_avg, 1)
    return {
        "Wall_ms": tw / 1e6, "Process_ms": tp / 1e6, "Thread_ms": tt / 1e6,
        "RCU": rcu,
        "Anchor_ns": anchor_avg, "Anchor_pre_ns": apre, "Anchor_post_ns": apost,
    }

# ============================================================
# НАВАНТАЖЕННЯ (різноманітні обчислювальні патерни)
# ============================================================
def work_alu():
    """ALU-bound: арифметика на L1 масивах (~24 KB)."""
    a, b, c = np.ones(1024), np.ones(1024), np.empty(1024)
    for _ in range(20000):
        np.multiply(a, b, out=c); np.add(c, a, out=c)

def work_fft():
    """FFT: класичне обчислювальне навантаження O(n log n)."""
    x = np.random.rand(32768)
    for _ in range(10):
        np.fft.fft(x)

def work_matmul():
    """GEMM: множення матриць (BLAS-bound)."""
    A = np.random.rand(256, 256)
    B = np.random.rand(256, 256)
    for _ in range(5):
        np.dot(A, B)

def work_mixed():
    """Змішане: ALU + сортування + тригонометрія."""
    a = np.random.rand(50000)
    for _ in range(3):
        np.sort(a.copy())
        np.sin(a, out=a)
        np.exp(a - a.max(), out=a)

def work_scaled(factor):
    """Масштабоване навантаження для перевірки лінійності."""
    n = int(10000 * factor)
    a, b, c = np.ones(1024), np.ones(1024), np.empty(1024)
    for _ in range(n):
        np.multiply(a, b, out=c); np.add(c, a, out=c)

WORKLOADS = {
    "ALU (L1)": work_alu,
    "FFT": work_fft,
    "GEMM": work_matmul,
    "Mixed": work_mixed,
}

# ============================================================
# СТРЕС-УМОВИ
# ============================================================
def _noise_cpu(stop):
    while not stop.is_set():
        np.dot(np.random.rand(150, 150), np.random.rand(150, 150))

def _noise_mem(stop):
    while not stop.is_set():
        _ = np.random.rand(5_000_000)  # ~40 MB алокації
        del _

def _noise_mixed(stop):
    while not stop.is_set():
        a = np.random.rand(500_000)
        np.sort(a)

def start_stress(kind, n_threads=4):
    """Запускає фонові потоки шуму. Повертає (stop_event, threads)."""
    stop = threading.Event()
    targets = {"cpu": _noise_cpu, "mem": _noise_mem, "mixed": _noise_mixed}
    func = targets.get(kind, _noise_cpu)
    threads = []
    for _ in range(n_threads):
        t = threading.Thread(target=func, args=(stop,), daemon=True)
        t.start()
        threads.append(t)
    time.sleep(0.8)
    return stop, threads

def stop_stress(stop, threads):
    stop.set()
    for t in threads:
        t.join(timeout=2)
    time.sleep(0.3)

CONDITIONS = [
    ("Чисто", None, 0),
    ("CPU шум (4×)", "cpu", 4),
    ("MEM шум (4×)", "mem", 4),
    ("Mixed шум (4×)", "mixed", 4),
    ("Heavy (8×CPU)", "cpu", 8),
]

# ============================================================
# macOS QoS API: керування розміщенням на P/E ядрах
# ============================================================
_QOS_AVAILABLE = False
_libpthread = None

QOS_CLASS_USER_INTERACTIVE = 0x21  # → P-ядра (продуктивні)
QOS_CLASS_UTILITY          = 0x11  # → може бути E або P
QOS_CLASS_BACKGROUND       = 0x09  # → E-ядра (енергоефективні)
QOS_CLASS_DEFAULT          = 0x15  # → за замовчуванням

try:
    if platform.system() == "Darwin":
        _libpthread = ctypes.CDLL(ctypes.util.find_library("pthread"))
        # int pthread_set_qos_class_self_np(qos_class_t, int relative_priority)
        _libpthread.pthread_set_qos_class_self_np.argtypes = [ctypes.c_uint, ctypes.c_int]
        _libpthread.pthread_set_qos_class_self_np.restype = ctypes.c_int
        _QOS_AVAILABLE = True
except Exception:
    pass

def set_qos(qos_class):
    """Встановлює QoS клас поточного потоку (macOS). Повертає True якщо успішно."""
    if _QOS_AVAILABLE and _libpthread is not None:
        ret = _libpthread.pthread_set_qos_class_self_np(qos_class, 0)
        return ret == 0
    return False

def reset_qos():
    """Повертає QoS до значення за замовчуванням."""
    set_qos(QOS_CLASS_DEFAULT)

# ============================================================
# СТАТИСТИКА
# ============================================================
def bootstrap_ci(data, n_boot=N_BOOTSTRAP, ci=0.95):
    """Bootstrap 95% довірчий інтервал для середнього."""
    data = np.asarray(data)
    means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi

def cohens_d(x, y):
    """Cohen's d — стандартизований розмір ефекту."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / max(pooled_std, 1e-12)

def vargha_delaney_a(x, y):
    """Vargha-Delaney A: P(X < Y) + 0.5·P(X = Y). A ≈ 0.5 = немає ефекту."""
    x, y = np.asarray(x), np.asarray(y)
    n, m = len(x), len(y)
    r = sp_stats.rankdata(np.concatenate([x, y]))
    r1 = r[:n].sum()
    return (r1 / n - (n + 1) / 2) / m

def cv(s):
    s = np.asarray(s)
    return s.std() / s.mean() * 100 if s.mean() > 0 else 0

# ============================================================
# ТЕСТИ
# ============================================================

def run_with_warmup(func, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    """Прогрів + N замірів."""
    for _ in range(n_warmup):
        measure(func)
    return [measure(func) for _ in range(n_trials)]

def test_stability():
    """Тест 1: Стабільність під різними стрес-умовами, 4 типи навантажень."""
    rows = []
    for wl_name, wl_func in WORKLOADS.items():
        for cond_name, noise_kind, n_threads in CONDITIONS:
            gc.collect()
            stop, threads = None, []
            if noise_kind:
                stop, threads = start_stress(noise_kind, n_threads)
            
            samples = run_with_warmup(wl_func)
            
            if stop:
                stop_stress(stop, threads)
            
            for i, m in enumerate(samples):
                m["Condition"] = cond_name
                m["Workload"] = wl_name
                m["Trial"] = i
                rows.append(m)
    
    return pd.DataFrame(rows)

def test_scaling():
    """Тест 2: Лінійність масштабування 1x..8x."""
    rows = []
    factors = [1, 2, 3, 4, 5, 6, 7, 8]
    for sf in factors:
        for _ in range(N_WARMUP):
            measure(work_scaled, sf)
        for i in range(N_TRIALS):
            m = measure(work_scaled, sf)
            m["Scale"] = sf
            m["Trial"] = i
            rows.append(m)
    return pd.DataFrame(rows)

def test_core_heterogeneity():
    """Тест 3: P-ядра vs E-ядра (Apple Silicon big.LITTLE).
    Використовує macOS QoS API для переміщення потоку на різні типи ядер.
    RCU має бути інваріантним, Thread Time — ні."""
    if not _QOS_AVAILABLE:
        print("  ⚠️  QoS API недоступний (не macOS або не Apple Silicon). Пропускаємо.")
        return pd.DataFrame()
    
    core_configs = [
        ("P-ядра (Interactive)", QOS_CLASS_USER_INTERACTIVE),
        ("Default",             QOS_CLASS_DEFAULT),
        ("E-ядра (Background)", QOS_CLASS_BACKGROUND),
    ]
    
    rows = []
    for wl_name, wl_func in WORKLOADS.items():
        for core_label, qos_class in core_configs:
            gc.collect()
            set_qos(qos_class)
            time.sleep(0.3)  # даємо планувальнику час на міграцію
            
            # Прогрів на цільовому ядрі
            for _ in range(N_WARMUP):
                measure(wl_func)
            
            for i in range(N_TRIALS):
                m = measure(wl_func)
                m["Core_Type"] = core_label
                m["Workload"] = wl_name
                m["Trial"] = i
                rows.append(m)
    
    reset_qos()
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🔬 Ex00: RCU METRIC VALIDATION (Research-Grade)")
    print(f"   N_TRIALS={N_TRIALS}, N_WARMUP={N_WARMUP}, N_BOOTSTRAP={N_BOOTSTRAP}")
    print(f"   Навантаження: {', '.join(WORKLOADS.keys())}")
    print(f"   Умови: {', '.join(c[0] for c in CONDITIONS)}")
    print("=" * 70)
    
    print(f"\n[1/3] Стабільність ({N_TRIALS} замірів × {len(CONDITIONS)} умов × {len(WORKLOADS)} навантажень)...")
    df_stab = test_stability()
    
    print(f"[2/3] Масштабування ({N_TRIALS} замірів × 8 множників)...")
    df_scale = test_scaling()
    
    print(f"[3/3] P-ядра vs E-ядра (гетерогенні ядра)...")
    df_cores = test_core_heterogeneity()
    
    # Зберігаємо дані в CSV
    import pandas as pd
    df_stab_df = pd.DataFrame(df_stab)
    df_scale_df = pd.DataFrame(df_scale)
    df_cores_df = pd.DataFrame(df_cores)
    
    df_stab_df.to_csv(RESULTS_DIR / "data_stability.csv", index=False)
    df_scale_df.to_csv(RESULTS_DIR / "data_scaling.csv", index=False)
    df_cores_df.to_csv(RESULTS_DIR / "data_cores.csv", index=False)
    print(f"\n💾 Дані збережено у {RESULTS_DIR}/data_*.csv")
    
    # Генерація графіків через окремий модуль
    from visualize import plot_drift_bars, plot_effect_sizes, plot_core_heterogeneity, plot_anchor_stability, print_stats_report
    
    print("\nГенерація графіків...")
    plot_drift_bars(df_stab_df)
    plot_effect_sizes(df_stab_df)
    plot_anchor_stability(df_stab_df)
    if not df_cores_df.empty:
        plot_core_heterogeneity(df_cores_df)
    
    print_stats_report(df_stab_df, df_scale_df, df_cores_df)
    
    print(f"\n  Графіки збережено у {RESULTS_DIR}/")
    for f in sorted(RESULTS_DIR.glob("rcu_validation_*.png")):
        print(f"    • {f.name}")

