# common/rcu.py — RCU (Relative Compute Units) протокол
# Єдине джерело правди для anchor-визначення по всьому проєкту.
#
# Anchor: 1600 ітерацій (multiply + add) на 3 × 1024 float64 (~24 KB, L1-bound).
# Калібровано на Apple Silicon (M-серія) для цілі >= 1 мс.
# Значення ФІКСОВАНЕ і не перекалібровується між запусками.

from __future__ import annotations

import time
import threading
import ctypes
import ctypes.util
import platform

import numpy as np


# ==========================================
# ANCHOR: фіксована константа проєкту
# ==========================================
ANCHOR_LOOPS = 1600  # multiply+add ітерацій (calibrated >= 1 ms on Apple Silicon M-series)


# ==========================================
# QoS: ПРИВ'ЯЗКА ДО P-ЯДЕР (macOS Apple Silicon)
# ==========================================
_QOS_AVAILABLE = False
_libpthread = None
_QOS_CLASS_USER_INTERACTIVE = 0x21  # → P-ядра (продуктивні)

try:
    if platform.system() == "Darwin":
        _libpthread = ctypes.CDLL(ctypes.util.find_library("pthread"))
        _libpthread.pthread_set_qos_class_self_np.argtypes = [ctypes.c_uint, ctypes.c_int]
        _libpthread.pthread_set_qos_class_self_np.restype = ctypes.c_int
        _QOS_AVAILABLE = True
except Exception:
    pass


def pin_to_p_cores():
    """Підказка планувальнику: виконувати на P-ядрах (macOS QoS USER_INTERACTIVE)."""
    if _QOS_AVAILABLE and _libpthread is not None:
        _libpthread.pthread_set_qos_class_self_np(_QOS_CLASS_USER_INTERACTIVE, 0)


# ==========================================
# THREAD-LOCAL ANCHOR МАСИВИ
# ==========================================
_tls = threading.local()


def _init_arrays():
    """Ініціалізує thread-local L1-bound масиви (~24 KB). Один раз на потік/процес."""
    if not hasattr(_tls, "ok"):
        _tls.a = np.random.rand(1024)
        _tls.b = np.random.rand(1024)
        _tls.c = np.empty_like(_tls.a)
        _tls.ok = True
    return _tls.a, _tls.b, _tls.c


def anchor_ns() -> int:
    """Одиничний замір anchor у наносекундах. Використовує фіксований ANCHOR_LOOPS."""
    a, b, c = _init_arrays()
    t0 = time.thread_time_ns()
    for _ in range(ANCHOR_LOOPS):
        np.multiply(a, b, out=c)
        np.add(c, a, out=c)
    return time.thread_time_ns() - t0


def profile_rcu(func, *args, **kwargs):
    """
    Sandwich profiling: anchor_pre → payload → anchor_post.
    Повертає (result, rcu, anchor_pre_ns, anchor_post_ns).
    """
    apre = anchor_ns()
    t0 = time.thread_time_ns()
    result = func(*args, **kwargs)
    t_algo = time.thread_time_ns() - t0
    apost = anchor_ns()
    t_avg = (apre + apost) / 2.0
    rcu = t_algo / max(t_avg, 1)
    return result, rcu, apre, apost


def setup_rcu_worker():
    """Викликати на початку кожного воркер-процесу для ізоляції та P-core pinning."""
    pin_to_p_cores()
