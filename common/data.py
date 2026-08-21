# common/data.py — збереження та завантаження даних експериментів

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_KEYS = ("convergence", "final", "metadata")

# Тип даних експерименту: convergence/final — DataFrame, metadata — dict (JSON-сумісний)
ExperimentData = dict[str, pd.DataFrame | dict[str, Any]]


def save_experiment_data(data_dict: ExperimentData, filepath: Path | str) -> None:
    """
    Зберігає дані експерименту у JSON файл з метаданими.
    
    Args:
        data_dict: Словник з ключами 'convergence' (DataFrame), 'final' (DataFrame), 'metadata' (dict).
        filepath: Шлях до файлу для збереження (.json).
    
    Raises:
        ValueError: Якщо відсутній обов'язковий ключ.
    """
    for key in REQUIRED_KEYS:
        if key not in data_dict:
            raise ValueError(f"data_dict має містити ключ '{key}'")
    path = Path(filepath)
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    data_to_save = {
        "convergence": data_dict["convergence"].to_dict("records"),
        "final": data_dict["final"].to_dict("records"),
        "metadata": data_dict["metadata"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False, default=str)
    print(f"[DATA] Saved experiment data: {path}")


def load_experiment_data(filepath: Path | str) -> ExperimentData:
    """
    Завантажує дані експерименту з JSON файлу.
    
    Args:
        filepath: Шлях до JSON файлу з даними.
    
    Returns:
        Словник з ключами 'convergence', 'final', 'metadata'.
    
    Raises:
        FileNotFoundError: Якщо файл не існує.
        ValueError: Якщо у файлі відсутній обов'язковий ключ.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Файл даних не знайдено: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in REQUIRED_KEYS:
        if key not in data:
            raise ValueError(f"Файл даних має містити ключ '{key}'")
    result = {
        "convergence": pd.DataFrame(data["convergence"]),
        "final": pd.DataFrame(data["final"]),
        "metadata": data["metadata"],
    }
    print(f"[DATA] Loaded experiment data: {path}")
    return result
