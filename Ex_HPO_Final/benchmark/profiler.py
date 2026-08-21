"""HPO Benchmark — RCU Profiler (Frozen Protocol)
Використовує thread_time_ns + L1-cache anchor для детермінованого вимірювання.
"""
import time
import threading
import numpy as np

_tls = threading.local()

def _run_anchor(a, b, c, loops):
    st = time.thread_time_ns()
    for _ in range(loops):
        np.multiply(a, b, out=c)
        np.add(c, a, out=c)
    return time.thread_time_ns() - st

def get_anchor_time():
    if not hasattr(_tls, "init"):
        _tls.a, _tls.b = np.random.rand(1024), np.random.rand(1024)
        _tls.c, _tls.loops = np.empty_like(_tls.a), 50
        while True:
            t = _run_anchor(_tls.a, _tls.b, _tls.c, _tls.loops)
            if t >= 5_000_000:
                break
            _tls.loops *= 2
        _tls.init = True
    return _run_anchor(_tls.a, _tls.b, _tls.c, _tls.loops)

def run_with_rcu(func, seed, obj_fn, *args, **kwargs):
    """Виконує func і повертає (result, rcu_method, rcu_total)."""
    a1 = get_anchor_time()
    t1 = time.thread_time_ns()
    
    # Виконуємо метод
    res = func(seed, obj_fn, *args, **kwargs)
    
    t2 = time.thread_time_ns()
    a2 = get_anchor_time()
    
    total_ns = t2 - t1
    obj_ns = getattr(obj_fn, 'total_time_ns', 0)
    method_ns = max(0, total_ns - obj_ns)
    
    anchor_avg = max(1, (a1 + a2) / 2.0)
    rcu_method = method_ns / anchor_avg
    rcu_total = total_ns / anchor_avg
    
    return res, rcu_method, rcu_total
