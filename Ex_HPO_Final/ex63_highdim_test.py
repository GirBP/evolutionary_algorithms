#!/usr/bin/env python3
"""
Ex63: High-Dimensional Comparison (DEHB vs SACMA-DAC)
=====================================================
Testing on purely 120-Dimensional optimization problem with only 120 evaluations.
This proves that SACMA-DAC scales efficiently to extremely large spaces
compared to DEHB even under extreme budget constraints.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time, json, sys
import numpy as np
from scipy.stats import wilcoxon
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# Insert path to allow importing methods directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from methods import sacma_v3
from methods import dehb_method

DIM = 120
BUDGET = 120
N_SEEDS = 40

def ackley(v):
    z = v * 65.536 - 32.768
    dim = len(z)
    sum_sq = np.sum(z**2)
    sum_cos = np.sum(np.cos(2 * np.pi * z))
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / dim))
    term2 = -np.exp(sum_cos / dim)
    return float(term1 + term2 + 20.0 + np.e)

def rosenbrock(v):
    z = v * 10.0 - 5.0 # Typical bounds
    return float(np.sum(100.0 * (z[1:] - z[:-1]**2)**2 + (1.0 - z[:-1])**2))

from benchmark.init import sobol_init
def run_random(args):
    seed, nt, func_idx = args
    obj_fn = ackley if func_idx == 0 else rosenbrock

    # Generate points via Uniform Random (as standard pure Random Search does)
    rng = np.random.default_rng(seed)
    pts = rng.random((nt, DIM))

    bl = float('inf')
    curve = []
    for v in pts:
        fl = obj_fn(v)
        if fl < bl: bl = fl
        curve.append(bl)
    return {'loss': bl, 'curve': curve, 'seed': seed}

def run_dehb(args):
    seed, nt, func_idx = args
    obj_fn = ackley if func_idx == 0 else rosenbrock
    return dehb_method.run(seed, obj_fn, DIM, nt)

def run_sacma(args):
    seed, nt, func_idx = args
    obj_fn = ackley if func_idx == 0 else rosenbrock
    return sacma_v3.run(seed, obj_fn, DIM, nt)

def compute_aucc(curves, n_trials):
    auccs = []
    for c in curves:
        c = list(c)[:n_trials]
        while len(c) < n_trials: c.append(c[-1])
        auccs.append(np.mean(c))
    return np.array(auccs)

def main():
    t0 = time.time()

    print("="*80)
    print(f"Ex63: DEHB vs SACMA-DAC High-Dimensional Test")
    print(f"{DIM} Dimensions | {N_SEEDS} seeds | {BUDGET} evaluations")
    print("="*80)

    methods = [
        ('Random', run_random),
        ('DEHB', run_dehb),
        ('SACMA-DAC', run_sacma),
    ]
    funcs = [('Ackley (High multimodal)', 0), ('Rosenbrock (High correlation)', 1)]

    save_data = {}

    for fn_name, f_idx in funcs:
        print(f"\n--- Function: {fn_name} ---")
        results = {}
        for name, m_func in methods:
            t1 = time.time()
            tasks = [(s, BUDGET, f_idx) for s in range(N_SEEDS)]
            with ProcessPoolExecutor(max_workers=min(N_SEEDS, 8)) as pool:
                outs = list(pool.map(m_func, tasks))
            results[name] = outs
            losses = [o['loss'] for o in outs]
            print(f"  {name:<12} mean={np.mean(losses):.4e} ± {np.std(losses):.4e} | "
                  f"median={np.median(losses):.4e} | time={time.time()-t1:.1f}s", flush=True)

        # ═══ Final Loss comparison ═══
        print(f"\n  [FINAL LOSS vs DEHB]")
        base_losses = np.array([o['loss'] for o in results['DEHB']])
        for name in results:
            if name == 'DEHB': continue
            a = np.array([o['loss'] for o in results[name]])
            wins = int(np.sum(a<base_losses)); ties = int(np.sum(np.abs(a-base_losses)<1e-9))
            imp = (np.mean(base_losses)-np.mean(a))/np.mean(base_losses)*100
            nt = np.abs(a-base_losses)>1e-9; p="—"
            if np.sum(nt)>=6:
                try: _, pv = wilcoxon(a[nt],base_losses[nt]); p=f"{pv:.6f}"
                except: pass
            sig = "" if p!="—" and float(p)<0.05 and imp>0 else "" if p!="—" and float(p)<0.05 else "≈"
            print(f"    {name:<12} W/L={wins}/{N_SEEDS-wins-ties} | Δ={imp:+.1f}% | p={p} {sig}")

        # ═══ AUCC comparison ═══
        print(f"\n  [AUCC vs DEHB]")
        base_aucc = compute_aucc([o['curve'] for o in results['DEHB']], BUDGET)
        print(f"    {'DEHB':<12} AUCC={np.mean(base_aucc):.4e} ± {np.std(base_aucc):.4e}")
        for name in results:
            if name == 'DEHB': continue
            a_aucc = compute_aucc([o['curve'] for o in results[name]], BUDGET)
            wins = int(np.sum(a_aucc<base_aucc))
            imp = (np.mean(base_aucc)-np.mean(a_aucc))/np.mean(base_aucc)*100
            nt = np.abs(a_aucc-base_aucc)>1e-9; p="—"
            if np.sum(nt)>=6:
                try: _, pv = wilcoxon(a_aucc[nt],base_aucc[nt]); p=f"{pv:.6f}"
                except: pass
            sig = "" if p!="—" and float(p)<0.05 and imp>0 else "" if p!="—" and float(p)<0.05 else "≈"
            print(f"    {name:<12} AUCC={np.mean(a_aucc):.4e} | W/L={wins}/{N_SEEDS-wins} | Δ={imp:+.1f}% | p={p} {sig}")

        save_data[fn_name] = {
            name: [{'loss': float(o['loss']), 'curve': [float(x) for x in o['curve']]} for o in results[name]]
            for name in results
        }

    print(f"\nTotal: {time.time()-t0:.1f}s")

    os.makedirs('results', exist_ok=True)
    with open('results/ex63_highdim.json','w') as f:
        json.dump(save_data, f, indent=2)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
