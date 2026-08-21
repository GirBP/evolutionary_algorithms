# HPO Benchmark — SACMA-DAC/MAB, WL-CMA, Sigma-CMA (§ 3 дисертації)

Фреймворк для порівняння методів оптимізації гіперпараметрів (HPO) нейронних
мереж та ML-моделей. Головний результат зони — масштабне порівняльне
дослідження **11 методів на 43 задачах** (§ 3.6 дисертації), у якому три
авторські методи (SACMA-DAC, SACMA-MAB, WL-CMA) та один допоміжний авторський
метод (Sigma-CMA) порівнюються з 7 базовими підходами.

## Склад 11 методів фінального бенчмарку

Джерело перевірки: `results/GLOBAL_ANALYSIS/Summary.md`, розділ 2 (Глобальний рейтинг).

| Метод (Summary.md) | Тип | Файл реалізації |
|---|---|---|
| SACMA-DAC | Запропоновано | `methods/sacma_v3.py` |
| SACMA-MAB | Запропоновано | `methods/sacma_mab.py` |
| WL-CMA | Запропоновано | `methods/whales_cma.py` |
| Sigma-CMA | Запропоновано | `methods/antivanila.py` |
| GP-BO | Базовий | `methods/bo_gp.py` |
| TPE | Базовий | `methods/tpe.py` |
| SMAC | Базовий | `methods/smac_method.py` |
| L-SHADE | Базовий | `methods/lshade.py` |
| CMA-ES | Базовий | `methods/cmaes_pure.py` |
| DEHB | Базовий | `methods/dehb_method.py` |
| Random | Базовий | `methods/random_search.py` |

Глобальний рейтинг (χ² = 176.80, p = 1.08e-32, Фрідман, k=11, N=43):
SACMA-DAC (сер. ранг 3.69) → SACMA-MAB (3.95) → WL-CMA (4.60) → Sigma-CMA (4.62)
→ GP-BO (5.09) → TPE (5.14) → SMAC (5.44) → L-SHADE (6.38) → CMA-ES (8.01)
→ DEHB (9.19) → Random (9.88).

### Інші файли в `methods/`, що не входять у фінальні 11

Це реальні проміжні кроки того самого пошуку (не фіктивні дані, результати
збережено в `results/`), залишені для чесності ланцюжка експериментів, але не
частина заявленого в дисертації фінального порівняння:

| Файл | Роль |
|---|---|
| `sacma_base.py` | Найперша версія SACMA (RF + CMA-ES без адаптації) — попередник SACMA-DAC/MAB |
| `sacma_lazy.py` | Проміжна версія з lazy-кешем власних векторів коваріаційної матриці |
| `sacma_v3_no_adapt.py`, `sacma_v3_no_virtual.py` | Абляційні варіанти SACMA-DAC (без ΔF-адаптації / без віртуального пулу кандидатів), використані у `results/L_ABLATION` |
| `ordinv_cma.py` | OrdInv-CMA — ординальний сурогатний скринінг, попередній метод (Ex19), базовий для порівняння у § 3.6 |
| `shade.py` | SHADE (без лінійного зменшення популяції) — використовується окремо для аналізу wall-clock часу (`WCT_METHODS`, `results/L2_WCT`), на відміну від L-SHADE у фінальних 11 |
| `methods/_archive/` | Архівні ранні версії CMA-ES/SACMA (`cmaes_vanilla.py`, `sacma_golden.py`, `sacma_v1.py`, `sacma_v2.py`) |

`iw_moea.py` (IW-MOEA) було виключено з публічної копії повністю (код і сирі
результати): метод не увійшов до фінальних 11 (порівняй `benchmark/__init__.py`
та `Summary.md`), а його реалізація залежала від зовнішнього, непублічного
модуля пошуку методів.

## Структура

```
Ex_HPO_Final/
├── benchmark/                    # Інфраструктура задач і датасетів
│   ├── datasets.py               # Реєстр синтетичних датасетів (L0)
│   ├── models.py                 # Простори пошуку HGB/RF/MLP/SVM/GB (L0)
│   ├── yahpo_adapter.py          # YAHPO Gym сурогати (ONNX): L2, L2_WCT, L3_NAS_SUPER
│   ├── pd1_adapter.py            # Google PD1 XGBoost-сурогат (Transformer/ResNet/WideResNet/CNN): L2_MLP_PD1
│   ├── fcnet_adapter.py          # FCNet табличний сурогат (Klein & Hutter, 2019): L5_FCNET
│   ├── l4_architectures.py       # 5 мікро-архітектур PyTorch (< 100K парам.): L4
│   ├── l4_objective.py           # Реальне навчання мікромереж (L4, ~2-5с/оцінка)
│   ├── init.py                   # Уніфікована Sobol-ініціалізація (спільна для всіх методів)
│   ├── profiler.py               # RCU-профайлер (Relative Compute Units)
│   └── stats.py                  # AUCC, Wilcoxon та інші статистичні утиліти
├── methods/                       # Кожен метод — окремий файл із функцією run(seed, obj_fn, dim, budget)
├── results/
│   ├── L0/, L2/, L2_MLP_PD1/, L2_WCT/, L3_NAS_SUPER/, L4/, L5_FCNET/, L_ABLATION/
│   │                              # Сирі JSON-результати кожної комірки {метод×задача×seed}
│   ├── *_visuals/                # Автоматично згенеровані по-задачні звіти (analyze_visuals.py)
│   ├── figures/, tables/         # Допоміжні згенеровані матеріали
│   └── GLOBAL_ANALYSIS/          # КАНОНІЧНІ підсумкові результати (див. нижче)
├── figs/                          # Фігури для дисертації, згенеровані make_figs.py
├── run_benchmark.py               # Диспетчер: запуск усіх методів для тіру
├── run_method.py                  # Запуск одного методу на одному тірі
├── report.py                      # Агрегація результатів одного тіру (Wilcoxon, AUCC, RCU)
├── analyze_full.py                # Генерація results/GLOBAL_ANALYSIS/ (Фрідман, Неменьї, Баєс, CD-діаграми)
├── analyze_visuals.py             # Генерація по-задачних results/*_visuals/
├── generate_table_pngs.py         # Рендер таблиць у PNG
├── make_figs.py                   # Фігури для дисертації з aggregated_results.csv / aucc_results.csv
├── run_all.sh                     # Повний конвеєр: L2 → L4 → L_ABLATION
├── run_wct.py, run_fcnet.py       # Спеціалізовані запускачі для L2_WCT / L5_FCNET
└── run_ordinv_cma.sh              # Запуск базового методу OrdInv-CMA
```

## Канонічні результати

Усі числа в дисертації (§ 3.6) походять з `results/GLOBAL_ANALYSIS/`:

- **`Summary.md`** — повний звіт: глобальний рейтинг, тести Фрідмана/Неменьї,
  баєсівський знаковий ранговий тест, рейтинги за 6 групами задач, AUCC,
  аналіз домінування (No Free Lunch).
- **`aggregated_results.csv`** — агрегована таблиця (метод × задача): median/mean
  loss, RCU, ранг, normalized regret. 473 рядки (11 методів × 43 задачі).
- **`aucc_results.csv`** — Area Under Convergence Curve по кожній комірці
  (метод × задача), нормалізовано крос-методно.

Ці три файли — єдине джерело чисел для фігур і тексту. Фігури **не**
підганяються під бажаний результат: `make_figs.py` лише читає ці CSV.

## Зрізи таблиць 3.1/3.2 дисертації

> Файли `figs/new_01_*.png`, `new_03_*.png`, `new_04_*.png`, `new_06_*.png`, `new_08_*.png` — оригінальні рисунки 3.1–3.5 дисертації (байт-у-байт). Файли `figs/01–07_*.png` — додаткові аналізи, згенеровані скриптами цієї папки.

Окрім 11-методного глобального аналізу (`results/GLOBAL_ANALYSIS/`), скрипт
`dissertation_slice_analysis.py` рахує два додаткові зрізи підмножин методів
(не чіпає `results/GLOBAL_ANALYSIS/`, `make_figs.py`, `figs/01-03`):

- **Табл. 3.1 (8-методний зріз рангів)** — SACMA-DAC проти 7 базових методів
  (TPE, GP-BO, SMAC, L-SHADE, CMA-ES, DEHB, Random); SACMA-MAB, WL-CMA,
  Sigma-CMA виключені з цього зрізу. Ранг кожного методу на кожній задачі
  рахується (`scipy.stats.rankdata`) ЛИШЕ в межах цих 8 методів по
  `median_loss` з `results/GLOBAL_ANALYSIS/aggregated_results.csv`, середній
  ранг — по 43 задачах (тест Фрідмана, χ², CD за Неменьї).
- **Табл. 3.2 (AUCC, 9-методний зріз)** — SACMA-DAC + SACMA-MAB + ті самі 7
  базових методів. AUCC (Area Under Convergence Curve) рахується з
  крос-методною нормалізацією кривих збіжності (`curve` із сирих JSON
  `results/L0,L2,L2_MLP_PD1,L3_NAS_SUPER,L4,L5_FCNET/*.json`), але межі
  нормалізації (L_best/L_worst на задачу) обчислюються ЛИШЕ по цих 9 методів
  — інакше, ніж у `results/GLOBAL_ANALYSIS/aucc_results.csv`, де межі беруться
  по всіх 11 (значення AUCC залежать від пулу нормалізації; обчислені цим
  скриптом значення — у `figs/diss_slice_stats.txt`). L2_WCT і L_ABLATION
  виключені (допоміжні тіри, не входять у 43 задачі).

Команда запуску:

```bash
python3 dissertation_slice_analysis.py
```

Вихід: `figs/diss_slice_stats.txt` — усі обчислені значення зрізів. Графічне
подання рангового зрізу — оригінальний рис. 3.2 дисертації
(`figs/new_01_CD_Diagram_Global.png`).

Застереження: глобальний 11-методний аналіз (усі 11 методів разом,
крос-методна нормалізація по всіх 11) лежить у `results/GLOBAL_ANALYSIS/` і
фігурах `figs/01-03` — це окремий, ширший підрахунок, не тотожний зрізам
табл. 3.1/3.2.

## Як відтворити результати

```bash
# 1. Один тір, один метод (наприклад, швидка перевірка на L0):
python3 run_method.py sacma_v3 L0

# 2. Увесь тір, усі активні методи (benchmark/__init__.py:ACTIVE_METHODS):
python3 run_benchmark.py L2 5          # L2 на 5 сідів
python3 run_benchmark.py ALL 3         # усі тіри одразу

# 3. Повний конвеєр (L2 → L4 → L_ABLATION, з примусовим перезапуском):
bash run_all.sh

# 4. Агрегація одного тіру (Wilcoxon, AUCC, RCU):
python3 report.py L2

# 5. Глобальний статистичний аналіз (перегенерувати results/GLOBAL_ANALYSIS/):
python3 analyze_full.py

# 6. По-задачні звіти (results/*_visuals/):
python3 analyze_visuals.py

# 7. Фігури для дисертації (figs/ у цій зоні):
python3 make_figs.py
```

Результати кешуються як JSON у `results/<tier>/`; `--force` в `run_method.py` /
`run_benchmark.py` перераховує комірку заново, ігноруючи кеш.

### `run_method.py` — запуск одного методу

```
python3 run_method.py <method> <tier> [seeds] [--dataset NAME] [--model NAME] [--force]
```

| Аргумент | Опис |
|---|---|
| `method` | ім'я файлу з `methods/` без `.py` (напр. `sacma_v3`, `whales_cma`) |
| `tier` | одне з: L0, L2, L2_MLP_PD1, L2_WCT, L3_NAS_SUPER, L4, L5_FCNET, L_ABLATION |
| `seeds` | кількість сідів (типово задано у `benchmark/__init__.py:TIERS[tier]['default_seeds']`) |
| `--dataset` | фільтр по конкретному датасету (опціонально) |
| `--model` | фільтр по конкретній моделі/адаптеру (опціонально) |
| `--force` | ігнорувати кеш, перерахувати |

## Тіри (`benchmark/__init__.py:TIERS`)

| Тір | Задачі | Адаптер | Seeds | Budget |
|---|---|---|---|---|
| **L0** | 3 синтетичні × 5 моделей (hgb, rf, mlp, svm, gb) | `benchmark/models.py` | 2 | 30 |
| **L2** | 10 YAHPO LCBench задач | `benchmark/yahpo_adapter.py` | 10 | 50 |
| **L2_WCT** | ті самі 10 LCBench задач, wall-clock аналіз | `benchmark/yahpo_adapter.py` | 10 | 50 |
| **L2_MLP_PD1** | 9 PD1 задач (CNN/WideResNet/ResNet/Transformer) | `benchmark/pd1_adapter.py` | 10 | 50 |
| **L3_NAS_SUPER** | 2 задачі нейроархітектурного пошуку (YAHPO nb301, iaml_super) | `benchmark/yahpo_adapter.py` | 2 | 50 |
| **L4** | 3 реальні PyTorch мікроархітектури | `benchmark/l4_architectures.py`, `l4_objective.py` | 10 | 20 |
| **L5_FCNET** | 4 табличні FCNet задачі (Klein & Hutter, 2019) | `benchmark/fcnet_adapter.py` | 10 | 50 |
| **L_ABLATION** | 3 LCBench задачі, абляція SACMA-DAC | `benchmark/yahpo_adapter.py` | 10 | 50 |

Разом (L0 + L2 + L2_MLP_PD1 + L3_NAS_SUPER + L4 + L5_FCNET) = **43 задачі**
фінального бенчмарку в `Summary.md`. L2_WCT та L_ABLATION — допоміжні тіри
(wall-clock аналіз і абляція), не входять до підрахунку 43.

## Метрики

| Метрика | Опис | Напрямок |
|---|---|---|
| **Loss** | фінальний валідаційний лосс (RMSE / 1-accuracy залежно від задачі) | ↓ менше = краще |
| **AUCC** | Area Under Convergence Curve, крос-методна нормалізація | ↑ більше = швидша збіжність |
| **RCU** | Relative Compute Units — сумарний час оптимізації | ↓ менше = краще |
| **Ранг** | ранг методу на задачі (1 = найкращий); Фрідман/Неменьї — на сукупності рангів | ↓ менше = краще |

## Як додати новий метод

1. Створити `methods/my_method.py` з функцією:

```python
def run(seed, obj_fn, dim, budget):
    """
    seed:   int — random seed
    obj_fn: callable, v ∈ [0,1]^dim → float (мінімізуємо)
    dim:    int — розмірність простору пошуку
    budget: int — максимальна кількість реальних оцінок
    Повертає: {'loss': float, 'curve': list[float], 'seed': int}
    """
```

2. Додати ім'я файлу (без `.py`) у `benchmark/__init__.py:ACTIVE_METHODS`.
3. Запустити: `python3 run_method.py my_method L0` (smoke), потім потрібний тір.
