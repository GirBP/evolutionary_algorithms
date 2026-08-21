#!/usr/bin/env python3
"""
Верифікація всіх числових тверджень з Ex09 (GFCS) —
порівняння заявлених значень у дисертації з реальними даними.

Перевіряє:
- Стиснення у 7.6x
- ΔF1 = +0.079
- Надійність 8/8 datasets
- RCU ≈ 80.3
- Результати конкурентів (KD, NeuronRemoval, SVD, EvoMerge, WeightRedist)
- Число методів, датасетів
"""

import json
import csv
import os
from pathlib import Path
import sys

RESULTS_DIR = Path(__file__).parent / "results"

# ═══════════════════════════════════════════
#  Заявлені значення (з Ex09_report.md та vstup_assembled.md)
# ═══════════════════════════════════════════
CLAIMS = {
    # Із vstup_assembled.md:
    "vstup_compression_ratio": 7.6,
    "vstup_datasets_count": 8,
    
    # Із Ex09_report.md таблиці:
    "gfcs_delta_f1": +0.079,
    "gfcs_rcu": 80.3,
    "gfcs_compression": 7.6,
    "gfcs_reliability": "8/8",
    "gfcs_data_free": True,
    
    "neuronremoval_delta_f1": +0.077,
    "neuronremoval_rcu": 0.09,
    "neuronremoval_compression": 2.8,
    
    "kd_delta_f1": +0.076,
    "kd_rcu": 51.7,
    "kd_compression": 19.3,
    
    "svd_delta_f1": +0.065,
    "svd_rcu": 0.45,
    "svd_compression": 1.0,
    
    "evomerge_delta_f1": +0.042,
    "evomerge_rcu": 649.5,
    "evomerge_compression": 4.4,
    "evomerge_reliability": "7/8",
    
    "weightredist_delta_f1": -0.003,
    "weightredist_rcu": 0.08,
    "weightredist_compression": 20.8,
    "weightredist_reliability": "6/8",
    
    # Із report висновки:
    "report_compression_approx": 7.0,  # "у 7× разів"
    "report_delta_f1_approx": +0.07,   # "+0.07 пунктів"
    
    # Кількість методів у порівнянні
    "num_methods": 6,
}


def load_json(name):
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_csv(name):
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def check(name, claimed, actual, tolerance=0.01, unit=""):
    """Порівнює заявлене з реальним значенням."""
    if actual is None:
        print(f"    {name}: НЕ ЗНАЙДЕНО в даних (заявлено: {claimed}{unit})")
        return False
    
    if isinstance(claimed, str):
        ok = str(actual) == claimed
    elif isinstance(claimed, bool):
        ok = actual == claimed
    else:
        diff = abs(actual - claimed)
        rel_diff = diff / max(abs(claimed), 1e-10)
        ok = rel_diff <= tolerance or diff <= tolerance
    
    if ok:
        print(f"    {name}: заявлено={claimed}{unit}, реально={actual}{unit}")
    else:
        print(f"    {name}: заявлено={claimed}{unit}, реально={actual}{unit}  ← РОЗБІЖНІСТЬ!")
    
    return ok


def verify_synthesis_data():
    """Перевірка по synthesis_all.json / synthesis_summary.csv"""
    print("\n" + "=" * 70)
    print("  1. Верифікація synthesis_all.json")
    print("=" * 70)
    
    data = load_json("synthesis_all.json")
    if data is None:
        print("    Файл synthesis_all.json не знайдено")
        return
    
    # Аналізуємо структуру
    if isinstance(data, dict):
        print(f"  Ключі: {list(data.keys())[:10]}")
        # Спробуємо знайти GFCS результати
        for key in data:
            print(f"    {key}: {type(data[key]).__name__}", end="")
            if isinstance(data[key], list):
                print(f" (len={len(data[key])})")
            elif isinstance(data[key], dict):
                print(f" (keys={list(data[key].keys())[:5]})")
            else:
                print(f" = {data[key]}")
    elif isinstance(data, list):
        print(f"  Записів: {len(data)}")
        if len(data) > 0:
            print(f"  Перший запис: {list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")


def verify_moons_csv():
    """Перевірка по moons_s2d_summary.csv"""
    print("\n" + "=" * 70)
    print("  2. Верифікація moons_s2d_summary.csv")
    print("=" * 70)
    
    rows = load_csv("moons_s2d_summary.csv")
    if rows is None:
        print("    Файл не знайдено")
        return
    
    print(f"  Записів: {len(rows)}")
    if rows:
        print(f"  Колонки: {list(rows[0].keys())}")
        # Шукаємо GFCS
        for row in rows[:5]:
            print(f"    {row}")


def verify_benchmark_json():
    """Перевірка по ex09v2_benchmark.json — головний benchmark"""
    print("\n" + "=" * 70)
    print("  3. Верифікація ex09v2_benchmark.json (головний бенчмарк)")
    print("=" * 70)
    
    data = load_json("ex09v2_benchmark.json")
    if data is None:
        data = load_json("full_benchmark.json")
        if data is None:
            print("    Жодного benchmark файлу не знайдено")
            return
        print("  (використовується full_benchmark.json)")
    
    if isinstance(data, dict):
        # Шукаємо методи та датасети
        methods = set()
        datasets = set()
        results_by_method = {}
        
        for key, val in data.items():
            if isinstance(val, dict):
                # Спробуємо визначити структуру
                for subkey, subval in val.items():
                    if isinstance(subval, dict):
                        method = subval.get("method", key)
                        dataset = subval.get("dataset", subkey)
                        methods.add(method if method else key)
                        datasets.add(dataset if dataset else subkey)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        m = item.get("method", "")
                        d = item.get("dataset", "")
                        if m: methods.add(m)
                        if d: datasets.add(d)
                        
                        if m not in results_by_method:
                            results_by_method[m] = []
                        results_by_method[m].append(item)
        
        print(f"  Методи знайдені: {methods}")
        print(f"  Датасети знайдені: {datasets}")
        print(f"  Кількість методів: {len(methods)}")
        print(f"  Кількість датасетів: {len(datasets)}")
        
        # Виводимо структуру верхнього рівня
        print(f"\n  Структура JSON (top-level keys):")
        for key in list(data.keys())[:10]:
            val = data[key]
            if isinstance(val, dict):
                print(f"    '{key}': dict({len(val)} keys: {list(val.keys())[:5]})")
            elif isinstance(val, list):
                print(f"    '{key}': list(len={len(val)})")
            else:
                print(f"    '{key}': {val}")
    
    elif isinstance(data, list):
        print(f"  Записів: {len(data)}")
        if data:
            print(f"  Поля: {list(data[0].keys()) if isinstance(data[0], dict) else 'not dict'}")
            
            # Агрегація по методах
            methods = {}
            datasets = set()
            for item in data:
                if not isinstance(item, dict):
                    continue
                m = item.get("method", item.get("name", "unknown"))
                d = item.get("dataset", item.get("task", "unknown"))
                datasets.add(d)
                if m not in methods:
                    methods[m] = {"delta_f1": [], "compression": [], "rcu": [], 
                                  "reliability": 0, "total": 0, "datasets": set()}
                
                # Шукаємо метрики
                for f1_key in ["delta_f1", "f1_delta", "dF1", "quality"]:
                    if f1_key in item:
                        methods[m]["delta_f1"].append(float(item[f1_key]))
                        break
                
                for comp_key in ["compression", "comp_ratio", "compress", "ratio"]:
                    if comp_key in item:
                        methods[m]["compression"].append(float(item[comp_key]))
                        break
                
                for rcu_key in ["rcu", "RCU", "time_rcu"]:
                    if rcu_key in item:
                        methods[m]["rcu"].append(float(item[rcu_key]))
                        break
                
                # Надійність: чи не було колапсу
                f1_val = item.get("f1_post", item.get("f1", item.get("quality", None)))
                if f1_val is not None and float(f1_val) > 0.1:
                    methods[m]["reliability"] += 1
                methods[m]["total"] += 1
                methods[m]["datasets"].add(d)
            
            print(f"\n  Знайдені методи ({len(methods)}):")
            
            errors = []
            
            for m_name, m_data in sorted(methods.items()):
                avg_f1 = sum(m_data["delta_f1"]) / len(m_data["delta_f1"]) if m_data["delta_f1"] else None
                avg_comp = sum(m_data["compression"]) / len(m_data["compression"]) if m_data["compression"] else None
                avg_rcu = sum(m_data["rcu"]) / len(m_data["rcu"]) if m_data["rcu"] else None
                rel = f"{m_data['reliability']}/{m_data['total']}"
                n_ds = len(m_data["datasets"])
                
                print(f"\n     {m_name}:")
                print(f"       ΔF1={avg_f1:.4f if avg_f1 else 'N/A'}, "
                      f"Compression={avg_comp:.1f if avg_comp else 'N/A'}×, "
                      f"RCU={avg_rcu:.1f if avg_rcu else 'N/A'}, "
                      f"Reliability={rel}, Datasets={n_ds}")
                
                # Перевірка проти заявлених
                m_lower = m_name.lower().replace(" ", "").replace("-", "")
                
                if "gfcs" in m_lower:
                    print(f"       --- Перевірка GFCS ---")
                    if avg_f1 is not None:
                        check("GFCS ΔF1", CLAIMS["gfcs_delta_f1"], round(avg_f1, 3), 0.005, "")
                    if avg_comp is not None:
                        check("GFCS Compression", CLAIMS["gfcs_compression"], round(avg_comp, 1), 0.2, "×")
                    if avg_rcu is not None:
                        check("GFCS RCU", CLAIMS["gfcs_rcu"], round(avg_rcu, 1), 5.0, "")
                    rel_str = f"{m_data['reliability']}/{len(m_data['datasets'])}"
                    check("GFCS Reliability", CLAIMS["gfcs_reliability"], rel_str)
                    
                elif "neuron" in m_lower or "removal" in m_lower:
                    print(f"       --- Перевірка NeuronRemoval ---")
                    if avg_f1 is not None:
                        check("NeuronRemoval ΔF1", CLAIMS["neuronremoval_delta_f1"], round(avg_f1, 3), 0.005)
                    if avg_comp is not None:
                        check("NeuronRemoval Compression", CLAIMS["neuronremoval_compression"], round(avg_comp, 1), 0.2, "×")

                elif "kd" in m_lower or "distill" in m_lower:
                    print(f"       --- Перевірка KD ---")
                    if avg_f1 is not None:
                        check("KD ΔF1", CLAIMS["kd_delta_f1"], round(avg_f1, 3), 0.005)
                    if avg_comp is not None:
                        check("KD Compression", CLAIMS["kd_compression"], round(avg_comp, 1), 0.5, "×")

                elif "svd" in m_lower:
                    print(f"       --- Перевірка SVD ---")
                    if avg_f1 is not None:
                        check("SVD ΔF1", CLAIMS["svd_delta_f1"], round(avg_f1, 3), 0.005)

                elif "evo" in m_lower:
                    print(f"       --- Перевірка EvoMerge ---")
                    if avg_f1 is not None:
                        check("EvoMerge ΔF1", CLAIMS["evomerge_delta_f1"], round(avg_f1, 3), 0.005)
                    if avg_comp is not None:
                        check("EvoMerge Compression", CLAIMS["evomerge_compression"], round(avg_comp, 1), 0.3, "×")

                elif "weight" in m_lower or "redist" in m_lower:
                    print(f"       --- Перевірка WeightRedist ---")
                    if avg_f1 is not None:
                        check("WeightRedist ΔF1", CLAIMS["weightredist_delta_f1"], round(avg_f1, 3), 0.005)

            # Загальні перевірки
            print(f"\n  --- Загальні перевірки ---")
            check("Кількість датасетів", CLAIMS["vstup_datasets_count"], len(datasets))
            check("Кількість методів", CLAIMS["num_methods"], len(methods))


def verify_fashionmnist():
    """Перевірка по practical_fashionmnist.json/csv"""
    print("\n" + "=" * 70)
    print("  4. Верифікація practical_fashionmnist.json")
    print("=" * 70)
    
    data = load_json("practical_fashionmnist.json")
    if data is None:
        print("    Файл не знайдено")
        return
    
    if isinstance(data, list):
        print(f"  Записів: {len(data)}")
        for item in data[:5]:
            if isinstance(item, dict):
                print(f"    {item.get('method', '?')}: {item}")
    elif isinstance(data, dict):
        print(f"  Ключі: {list(data.keys())[:10]}")


def verify_cnn_resnet():
    """Перевірка по ex09v2_cnn_resnet.json"""
    print("\n" + "=" * 70)
    print("  5. Верифікація ex09v2_cnn_resnet.json")
    print("=" * 70)
    
    data = load_json("ex09v2_cnn_resnet.json")
    if data is None:
        print("    Файл не знайдено")
        return
    
    if isinstance(data, list):
        print(f"  Записів: {len(data)}")
        for item in data:
            if isinstance(item, dict):
                print(f"    {item}")
    elif isinstance(data, dict):
        print(f"  Ключі: {list(data.keys())}")
        for key in data:
            print(f"    {key}: {data[key]}")


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Ex09 GFCS — Верифікація числових тверджень дисертації     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    # Список доступних файлів
    print(f"\n  Доступні файли в {RESULTS_DIR}:")
    for f in sorted(RESULTS_DIR.glob("*.json")) + sorted(RESULTS_DIR.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:40s} ({size_kb:.1f} KB)")
    
    verify_synthesis_data()
    verify_moons_csv()
    verify_benchmark_json()
    verify_fashionmnist()
    verify_cnn_resnet()
    
    print("\n" + "=" * 70)
    print("  ВЕРИФІКАЦІЯ ЗАВЕРШЕНА")
    print("=" * 70)


if __name__ == "__main__":
    main()
