#!/usr/bin/env python3
"""
HPO Benchmark — Report Aggregator.

Збирає ВСІ наявні JSON-файли з results/<tier>/ і генерує звіт.

Використання:
    python3 report.py <tier>
    python3 report.py L1
"""
import os
import sys
import json
import glob
import numpy as np
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark.stats import compute_aucc, safe_wilcoxon
from benchmark import TIERS


def load_all_results(tier):
    """Завантажує всі JSON з results/<tier>/"""
    pattern = os.path.join(os.path.dirname(__file__), "results", tier, "*.json")
    records = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            records.append(json.load(f))
    return records


def group_results(records):
    """Групує результати по {method × dataset × model} → list of records."""
    groups = {}
    for r in records:
        key = (r['method'], r['dataset'], r['model'])
        groups.setdefault(key, []).append(r)
    return groups


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    tier = sys.argv[1].upper()
    budget = TIERS.get(tier, {}).get('budget', 50)

    records = load_all_results(tier)
    if not records:
        print(f"No results found in results/{tier}/")
        sys.exit(1)

    groups = group_results(records)

    # Знаходимо всі методи, датасети, моделі
    methods = sorted(set(r['method'] for r in records))
    datasets = sorted(set(r['dataset'] for r in records))
    models = sorted(set(r['model'] for r in records))

    print(f"\n{'='*80}")
    print(f"HPO BENCHMARK REPORT — {tier}")
    print(f"Methods: {methods}")
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"{'='*80}")

    # ── Per-cell stats ────────────────────────────────────────────────────
    print(f"\n{'─'*98}")
    print(f"{'Method':<15} | {'Type':<10} | {'Dataset':<20} | {'Model':<5} | "
          f"{'Seeds':>5} | {'Mean Loss':>10} | {'±Std':>7} | {'AUCC':>8} | {'RCU':>8}")
    print(f"{'─'*98}")

    cell_stats = {}
    for (method, dataset, model), recs in sorted(groups.items()):
        n = len(recs)
        losses = [r['loss'] for r in recs]
        auccs = [compute_aucc(r['curve'], budget) for r in recs]
        rcus = [r.get('rcu_hpo', r.get('rcu_total', r.get('rcu', 0))) for r in recs]

        cell_stats[(method, dataset, model)] = {
            'losses': losses, 'auccs': auccs, 'rcus': rcus, 'n': n
        }

        is_baseline = method in ['bo_gp', 'random_search', 'shade', 'tpe', 'lshade', 'cmaes_pure', 'smac_method', 'dehb_method']
        m_type = "Baseline" if is_baseline else "Proposed"

        print(f"{method:<15} | {m_type:<10} | {dataset:<20} | {model:<5} | "
              f"{n:>5} | {np.mean(losses):>10.4f} | {np.std(losses):>7.4f} | "
              f"{np.mean(auccs):>8.4f} | {np.mean(rcus):>8.0f}")

    # ── Pairwise Wilcoxon ────────────────────────────────────────────────
    if len(methods) >= 2:
        print(f"\n{'='*80}")
        print("PAIRWISE WILCOXON SIGNED-RANK TEST (Loss)")
        print(f"{'='*80}")
        print(f"{'Method A':<15} vs {'Method B':<15} | {'Dataset':<20} | {'Model':<5} | "
              f"{'W/L':>5} | {'p-value':>8} | {'Verdict':>10}")
        print(f"{'─'*80}")

        # ── Збираємо всі p-values для BH корекції ───────────────────────
        all_tests = []  # [(m_a, m_b, ds, mdl, wins_a, wins_b, p_raw)]

        for m_a, m_b in combinations(methods, 2):
            for ds in datasets:
                for mdl in models:
                    key_a = (m_a, ds, mdl)
                    key_b = (m_b, ds, mdl)
                    if key_a not in cell_stats or key_b not in cell_stats:
                        continue

                    sa = cell_stats[key_a]
                    sb = cell_stats[key_b]
                    n = min(sa['n'], sb['n'])
                    if n < 2:
                        continue

                    la = sa['losses'][:n]
                    lb = sb['losses'][:n]

                    wins_a = sum(1 for a, b in zip(la, lb) if a < b)
                    wins_b = sum(1 for a, b in zip(la, lb) if b < a)
                    p = safe_wilcoxon(la, lb)
                    all_tests.append((m_a, m_b, ds, mdl, wins_a, wins_b, p, np.mean(la), np.mean(lb)))

        # ── Benjamini-Hochberg FDR correction ────────────────────────────
        if all_tests:
            from scipy.stats import false_discovery_control
            raw_pvals = np.array([t[6] for t in all_tests])
            # BH: повертає відкориговані p-values
            try:
                rejected = false_discovery_control(raw_pvals, method='bh')
                # false_discovery_control повертає bool маску rejected
                # Потрібні adjusted p-values — рахуємо вручну
                n_tests = len(raw_pvals)
                sorted_idx = np.argsort(raw_pvals)
                adjusted = np.zeros(n_tests)
                for rank_i, orig_i in enumerate(sorted_idx):
                    adjusted[orig_i] = raw_pvals[orig_i] * n_tests / (rank_i + 1)
                adjusted = np.minimum.accumulate(adjusted[np.argsort(np.argsort(raw_pvals))][::-1])[::-1]
                adjusted = np.clip(adjusted, 0, 1)
            except Exception:
                adjusted = raw_pvals

            print(f"\n{'='*90}")
            print(f"PAIRWISE WILCOXON SIGNED-RANK TEST (Loss) — BH-corrected (α=0.05, {len(all_tests)} tests)")
            print(f"{'='*90}")
            print(f"{'Method A':<15} vs {'Method B':<15} | {'Dataset':<20} | {'Model':<5} | "
                  f"{'W/L':>5} | {'p_raw':>8} | {'p_adj':>8} | {'Verdict':>10}")
            print(f"{'─'*90}")

            total_wins = {m: 0 for m in methods}
            total_cells_compared = len(all_tests)

            for i, (m_a, m_b, ds, mdl, wa, wb, p_raw, mean_a, mean_b) in enumerate(all_tests):
                p_adj = adjusted[i]
                if p_adj < 0.05:
                    if mean_a < mean_b:
                        verdict = f" {m_a}"
                        total_wins[m_a] += 1
                    else:
                        verdict = f" {m_b}"
                        total_wins[m_b] += 1
                else:
                    verdict = "≈ tie"

                print(f"{m_a:<15} vs {m_b:<15} | {ds:<20} | {mdl:<5} | "
                      f"{wa}/{wb} | {p_raw:>8.4f} | {p_adj:>8.4f} | {verdict:>10}")
        else:
            total_wins = {m: 0 for m in methods}
            total_cells_compared = 0

        # ── Summary ──────────────────────────────────────────────────────
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        for m in methods:
            all_losses = []
            all_auccs = []
            all_rcus = []
            for (method, ds, mdl), st in cell_stats.items():
                if method == m:
                    all_losses.extend(st['losses'])
                    all_auccs.extend(st['auccs'])
                    all_rcus.extend(st['rcus'])

            is_baseline = m in ['bo_gp', 'random_search', 'shade', 'tpe', 'hebo_method', 'smac_method']
            m_type = "[Base]" if is_baseline else "[Prop]"

            print(f"  {m_type} {m:<15}: Mean Loss={np.mean(all_losses):.4f} | "
                  f"AUCC={np.mean(all_auccs):.4f} | "
                  f"RCU={np.mean(all_rcus):.0f} | "
                  f"Sig. Wins={total_wins.get(m, 0)}/{total_cells_compared}")

        # ── Friedman + Nemenyi ────────────────────────────────────────────
        print(f"\n{'='*80}")
        print("FRIEDMAN TEST + POST-HOC NEMENYI")
        print(f"{'='*80}")
        try:
            import pandas as pd
            from scipy.stats import friedmanchisquare
            import scikit_posthocs as sp

            # Будуємо матрицю рангів: (dataset×model) × methods
            rank_matrix = {}  # method → list of ranks
            task_keys = sorted(set((ds, mdl) for ds in datasets for mdl in models))

            for task in task_keys:
                ds, mdl = task
                task_losses = {}
                for m in methods:
                    key = (m, ds, mdl)
                    if key in cell_stats:
                        task_losses[m] = np.mean(cell_stats[key]['losses'])
                if len(task_losses) < 3:
                    continue

                # Ранжуємо (1 = найкращий)
                sorted_m = sorted(task_losses.items(), key=lambda x: x[1])
                for rank_pos, (m, _) in enumerate(sorted_m, 1):
                    rank_matrix.setdefault(m, []).append(rank_pos)

            # Фільтруємо методи з повним покриттям
            max_tasks = max(len(v) for v in rank_matrix.values()) if rank_matrix else 0
            complete_methods = [m for m, v in rank_matrix.items() if len(v) == max_tasks]

            if len(complete_methods) >= 3 and max_tasks >= 3:
                rank_data = [rank_matrix[m] for m in complete_methods]
                stat, p_friedman = friedmanchisquare(*rank_data)

                print(f"  Friedman χ² = {stat:.2f}, p-value = {p_friedman:.6f}")
                if p_friedman < 0.05:
                    print(f"  → Значуща різниця між методами (p < 0.05)")
                else:
                    print(f"  → НЕ виявлено значущої різниці (p ≥ 0.05)")

                # Nemenyi post-hoc
                rank_df = pd.DataFrame(dict(zip(complete_methods, rank_data)))
                nemenyi = sp.posthoc_nemenyi_friedman(rank_df.values)
                nemenyi.index = complete_methods
                nemenyi.columns = complete_methods

                print(f"\n  Nemenyi post-hoc p-values (значущі пари при p < 0.05):")
                sig_pairs = []
                for ii in range(len(complete_methods)):
                    for jj in range(ii+1, len(complete_methods)):
                        p_nem = nemenyi.iloc[ii, jj]
                        m_i, m_j = complete_methods[ii], complete_methods[jj]
                        avg_r_i = np.mean(rank_matrix[m_i])
                        avg_r_j = np.mean(rank_matrix[m_j])
                        if p_nem < 0.05:
                            sig_pairs.append((m_i, m_j, p_nem, avg_r_i, avg_r_j))

                if sig_pairs:
                    for m_i, m_j, p_nem, r_i, r_j in sig_pairs:
                        winner = m_i if r_i < r_j else m_j
                        print(f"    {m_i} (avg rank {r_i:.1f}) vs {m_j} ({r_j:.1f}): p={p_nem:.4f} →  {winner}")
                else:
                    print(f"    Жодна пара не досягла значущості при Nemenyi post-hoc.")

                # Вивести середні ранги
                print(f"\n  Середні ранги (менше = краще):")
                avg_ranks = [(m, np.mean(rank_matrix[m])) for m in complete_methods]
                for m, ar in sorted(avg_ranks, key=lambda x: x[1]):
                    print(f"    {m:<15}: {ar:.2f}")
            else:
                print(f"  Недостатньо даних для Friedman тесту ({len(complete_methods)} методів, {max_tasks} задач)")
        except ImportError as e:
            print(f"  Помилка імпорту: {e}. Встановіть scikit-posthocs: pip install scikit-posthocs")
        except Exception as e:
            print(f"  Помилка Friedman/Nemenyi: {e}")

    print()


if __name__ == '__main__':
    main()
