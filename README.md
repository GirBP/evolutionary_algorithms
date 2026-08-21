# evolutionary_algorithms

Репозиторій з кодом, експериментами та агрегованими результатами до дисертації на здобуття наукового ступеня **доктора філософії** «Оптимізація параметрів та структури нейронних мереж із застосуванням еволюційних алгоритмів» (Гірянський Б. П., НТУУ «КПІ ім. Ігоря Сікорського», ННК «Інститут прикладного системного аналізу», спеціальність 122 — Комп'ютерні науки).

Дисертацію виконано в межах НДР «Впровадження сервіс-орієнтованого підходу до реалізації процесів діджиталізації суспільства» (ДР № 0123U101333, 2023–2025 рр., науковий керівник — д.т.н., проф. Петренко А. І.).

## Наукова новизна — карта «метод → папка → файл → число в дисертації»

| Метод | Папка | Ключовий файл (реалізація) | Файл результатів | Число в дисертації |
|---|---|---|---|---|
| **TESA-26** — пошарове проріджування з ітеративним перерахунком значущості ваг | `Ex08_TESA-26/` | `code/methods/tesa26.py` | `results/tables/ex08_friedman_nemenyi.md`, `results/raw/ex08_run_info.txt` | ранг Фрідмана 2,65 (6 наборів даних, 12 рівнів проріджування, 1116 записів), χ² = 246,5, p < 0,0001 (підрозд. 2.3.3–2.3.4) |
| **E-HTA** — еволюційна апроксимація сліду матриці Гессе | `Ex08_TESA-26/` | `code/methods/ehta.py` | `results/tables/ex08_friedman_nemenyi.md` | ранг Фрідмана 5,44 — Група B, статистично нерозрізнимий від Magnitude Pruning (підрозд. 2.3.2) |
| **GFCS** — конверсія розрідженої мережі у компактну щільну архітектуру | `Ex09_GFCS/` | `ex09_lib/gfcs.py` | `results/full_benchmark.json`, `results/full_benchmark_rcu.json` | стиснення 7,6×, прискорення 1,42×, збереження якості на 8/8 наборів даних (підрозд. 2.4.4) |
| **SACMA-DAC** — пошук гіперпараметрів із сурогатом на RF та Delta-F адаптацією | `Ex_HPO_Final/` | `methods/sacma_v3.py` | `results/GLOBAL_ANALYSIS/` (`Summary.md`, `aggregated_results.csv`, `aucc_results.csv`) | ранг Фрідмана 3,69 — 1-ше місце серед 11 методів на 43 задачах; AUCC = 0,9282 — найвищий (підрозд. 3.3, 3.6) |
| **SACMA-MAB** — пошук гіперпараметрів із сурогатом на RF та ε-жадібним MAB | `Ex_HPO_Final/` | `methods/sacma_mab.py` | `results/GLOBAL_ANALYSIS/` (`Summary.md`, `aggregated_results.csv`, `aucc_results.csv`) | ранг Фрідмана 3,95 — 2-ге місце серед 11 методів; AUCC = 0,9247 (підрозд. 3.4, 3.6) |
| **WL-CMA** — Гаусівський процес з деформацією Єо–Джонсона + динаміка Ланжевена | `Ex_HPO_Final/` | `methods/whales_cma.py` | `results/GLOBAL_ANALYSIS/Summary.md` | тір L2_MLP_PD1 (стохастичні задачі): ранг 3,44, 1-ше місце (підрозд. 3.5) |
| **ENT** — еволюційний відбір нейронної топології при злитті комплементарних моделей | `Ex30_HetMerge_ENT/` | `e34_benchmark.py` (метод ENT — блок «METHOD 9») | `results_e34.json`, `results_full_benchmark/raw_results.tsv` | точність 0,749, 10/10 класів, баланс 0,981 (табл. 4.3, підрозд. 4.2.3) |
| **ENT-FT** — калібрація повнозв'язного шару злитої моделі на малій вибірці | `Ex30_HetMerge_ENT/` | `ent_ft_on_e34_champion.py` (комплементарний, на чемпіоні e34), `ent_ft_benchmark.py` (гомогенні сценарії) | `results_ent_ft_on_e34.json`, `results_full_benchmark/ent_ft_results.tsv` | ефект калібрації на комплементарному злитті (табл. 4.4, підрозд. 4.3.2) |

Порівняння з альтернативним еволюційним злиттям Sakana-CMA — `Ex31_Sakana_vs_ENT/ex31_benchmark.py`, результати в `Ex31_Sakana_vs_ENT/results/sakana_vs_ent.tsv`.

Емпіричне встановлення межі застосовності CMA-ES (мотивація зміни парадигми — від оптимізації неперервних ваг до структурних аспектів мереж, підрозд. 2.2) — папка `Ex01-03_CMA_Boundary/`: CMA-ES проти градієнтних методів на функції Растрігіна (d=10, W Кендала = 0,943) та на прямій оптимізації ваг нейронної мережі (d ≥ 10², W Кендала = 0,700). Це не окремий пункт наукової новизни, а обґрунтувальний експеримент розділу 2.

Валідація метрики обчислювальної вартості RCU (підрозд. 2.1) — папка `Ex00_RCU_Validation/`: стрес-тести стабільності під фоновим навантаженням (bootstrap, n=10 000), дискримінації кеш/RAM-навантажень і лінійності масштабування. Саме звідси числа тексту: максимальний дрейф RCU 14,1% проти 868,0% дрейфу астрономічного часу (`results/raw/ex00_stats.txt`). Додатковий незалежний стрес-тест дрейфу — `common/rcu_drift_calibration.py`.

## Структура репозиторію

```
evolutionary_algorithms/
├── common/                          # Спільна інфраструктура (RCU, стиль, IO, відтворюваність)
│   ├── rcu.py                       # Метрика обчислювальної вартості RCU (Relative Compute Units)
│   ├── experiment.py                # Еталонне навантаження, ініціалізація experiment/results
│   └── rcu_drift_calibration.py     # Незалежний стрес-тест дрейфу RCU під CPU-навантаженням
├── Ex00_RCU_Validation/             # Валідація метрики RCU: дрейф ≤14,1% проти 868% у астрономічного часу
├── Ex01-03_CMA_Boundary/            # Межа застосовності CMA-ES (Растрігін vs оптимізація ваг НМ)
├── Ex08_TESA-26/                    # TESA-26 (проріджування) та E-HTA (апроксимація Гессе)
├── Ex09_GFCS/                       # GFCS — конверсія розрідженої мережі у компактну щільну
├── Ex30_HetMerge_ENT/               # ENT (еволюційне злиття) та ENT-FT (калібрація)
├── Ex31_Sakana_vs_ENT/              # Порівняння ENT із Sakana-CMA
├── Ex_HPO_Final/                    # SACMA-DAC, SACMA-MAB, WL-CMA
├── presentation_figs/               # Презентаційні версії ключових фігур (make_presentation_figs.py)
└── scripts/                         # Публікація (publish_to_github.py) і генерація фігур
```

Кожна папка `ExNN_*` містить власний `README.md` (за наявності), код методу, скрипт(и) запуску бенчмарку, `results/` (або `results_*`) з агрегованими метриками (JSON/CSV/TSV/MD).

## Що НЕ ввійшло до репозиторію (і чому)

- **Бенчмарк-датасети** (`Ex_HPO_Final/data/`): PD1, FCNet, YAHPO, ImageNet, CIFAR — публічні датасети, що завантажуються автоматично при першому запуску відповідних скриптів (torchvision, yahpo_gym) або окремим інструментом (див. README папки).
- **MNIST/FashionMNIST** (`Ex09_GFCS`, `Ex30_HetMerge_ENT`): завантажуються автоматично через `torchvision.datasets` при першому запуску скрипта.
- **Ваги моделей і чекпоінти** (`*.pt`, `*.pth`, `*.pkl`, `*.npy`, `*.npz`, `*.h5`, `*.hdf5`, `*.ckpt`, `*.onnx` — див. `.gitignore`): відтворюються запуском відповідного скрипта бенчмарку (моделі навчаються з нуля в межах самого прогону).
- **Сирі raw-результати CIFAR** (`Ex30_HetMerge_ENT/cifar_fix/results/`): агреговані підсумки наявні в `Ex30_HetMerge_ENT/results_full_benchmark/` та TSV-зведеннях поруч.

## Дані

Жоден датасет не зберігається в репозиторії. MNIST і FashionMNIST завантажуються автоматично засобами `torchvision.datasets` при першому запуску скрипта (кеш у `/tmp` або локальній `data/` папці експерименту). Бенчмарки HPO (PD1, FCNet, YAHPO Gym) — публічні набори, що підвантажуються відповідними Python-пакетами (`yahpo_gym` тощо) або завантажувачем, описаним у `Ex_HPO_Final/README.md`. Ваги моделей ніде не поширюються як бінарні артефакти — усі моделі навчаються з нуля в межах самого прогону скрипта (детермінований `seed`, час навчання — секунди–хвилини для використаних архітектур).

## Залежності

```bash
# Мінімальний набір для запуску основних експериментів:
pip install -r requirements.txt

# Повний замороз версій (для відтворення з точністю до патч-версій):
pip install -r requirements-full.txt
```

Базовий стек: Python ≥ 3.10, PyTorch (з MPS-підтримкою на Apple Silicon), CMA-ES (`pycma`; окремі бейзлайни використовують також пакет `cmaes`), scikit-learn (сурогатні моделі, логістична регресія калібрації), NumPy/SciPy (статистичні тести), pandas (агрегація результатів), matplotlib/seaborn (візуалізація).

Зауваження: (1) частина скриптів через `common/dependencies.py` автоматично довстановлює відсутні пакети (`pip install`) під час запуску — за потреби вимкніть це, встановивши всі залежності заздалегідь; (2) адаптер тіру L5_FCNET (`benchmark/fcnet_adapter.py`) потребує непакетованого `tabular_benchmarks` (HPOBench FCNet) і HDF5-файлів бенчмарку — див. README папки `Ex_HPO_Final/`.

## Запуск експериментів

Єдиного універсального `harness.py`/`exNN_run_experiment.py` для всіх експериментів немає — кожна папка має власну точку входу:

```bash
# TESA-26 / E-HTA (проріджування)
python Ex08_TESA-26/code/ex08_run.py --method all

# GFCS (конверсія розрідженої мережі у компактну щільну)
python Ex09_GFCS/ex09_full_benchmark.py

# ENT / ENT-FT (еволюційне злиття комплементарних моделей)
python Ex30_HetMerge_ENT/prepare.py                        # чекпоінти батьківських моделей A/B
python Ex30_HetMerge_ENT/e34_benchmark.py
python Ex30_HetMerge_ENT/ent_ft_on_e34_champion.py         # калібрація точного чемпіона e34 (табл. 4.4)
python Ex30_HetMerge_ENT/ent_ft_benchmark.py               # гомогенні сценарії
python Ex30_HetMerge_ENT/ent_ft_complementary.py --smoke   # варіант з незалежним перетренуванням; без --smoke — повний

# Порівняння ENT із Sakana-CMA
python Ex31_Sakana_vs_ENT/ex31_benchmark.py

# SACMA-DAC / SACMA-MAB / WL-CMA (пошук гіперпараметрів)
python Ex_HPO_Final/run_benchmark.py

# Валідація метрики RCU (джерело чисел підрозд. 2.1: дрейф 14,1% проти 868%)
python Ex00_RCU_Validation/ex00.py

# Незалежний стрес-тест дрейфу метрики RCU
python common/rcu_drift_calibration.py --smoke             # швидка перевірка
python common/rcu_drift_calibration.py                     # повний прогін
```

Кожен скрипт самодостатній: читає/навчає моделі, рахує метрики та зберігає результати у форматі JSON/TSV/CSV поруч зі скриптом.

## Оригінальні рисунки дисертації

Файли рисунків, вставлених у текст дисертації, збережені в репозиторії байт-у-байт:

| Рисунок | Файл у репозиторії |
|---|---|
| 2.1–2.3 (валідація RCU) | `Ex00_RCU_Validation/results/figs/rcu_validation_*.png` |
| 2.4–2.6 (CMA-ES, Растрігін) | `Ex01-03_CMA_Boundary/Ex01/results/figs/ex01_*.png` |
| 2.7–2.9 (CMA-ES, ваги НМ) | `Ex01-03_CMA_Boundary/Ex03/results/figs/ex03_*.png` |
| 2.10, 2.11 (TESA-26) | `Ex08_TESA-26/figs/new_fig_2_10_*.png`, `new_fig_2_11_*_v2.png` |
| 2.12, 2.13 (GFCS) | `Ex09_GFCS/results/figs/ex09_bfa_distribution.png`, `ex09_pareto_rcu_f1.png` |
| 3.1–3.5 (SACMA, 9 методів) | `Ex_HPO_Final/figs/new_01_*.png`, `new_03_*.png`, `new_04_*.png`, `new_06_*.png`, `new_08_*.png` |
| 4.1 (ENT на CIFAR-10) | `Ex31_Sakana_vs_ENT/figs/new_fig_4_1_ent_cifar_v2.png` |

Інші фігури в `figs/` та `presentation_figs/` — додаткові ілюстрації, згенеровані скриптами з тих самих файлів даних.

## Перевірка заявлених чисел одним запуском

```bash
python Ex_HPO_Final/verify_all_practical_significance.py   # усі 6 пунктів новизни: заявлено = реально
python Ex09_GFCS/verify_claims.py                          # детальна звірка чисел GFCS
```

## Відтворення фігур

Фігури генеруються скриптами, що читають наявні файли результатів (жодна фігура не редагується вручну):

```bash
python Ex08_TESA-26/code/regen_figures.py           # Ex08 figs/*.png (5-методний зріз, рис. 2.10-2.11)
python Ex08_TESA-26/code/make_extra_figs.py         # Ex08 додаткові
python Ex09_GFCS/make_figs.py                       # Ex09 results/figs/*.png
python Ex30_HetMerge_ENT/make_figs_mnist.py         # ENT покласові (табл. 4.3/4.6)
python Ex30_HetMerge_ENT/make_fig_entft.py          # ENT-FT ефект калібрації
python Ex31_Sakana_vs_ENT/make_figs_cifar.py        # CIFAR 4-панельна (рис. 4.1)
python Ex_HPO_Final/make_figs.py                    # SACMA глобальні (11 методів)
python Ex_HPO_Final/dissertation_slice_analysis.py  # зрізи таблиць 3.1/3.2
python Ex01-03_CMA_Boundary/make_figs.py            # W Кендала vs розмірність
python Ex00_RCU_Validation/ex00_visualize.py        # фігури валідації RCU
```

## Цитування

```bibtex
@phdthesis{hirianskyi2026evolutionary,
  author       = {Гірянський, Богдан Петрович},
  title        = {Оптимізація параметрів та структури нейронних мереж із застосуванням еволюційних алгоритмів},
  school       = {Національний технічний університет України «Київський політехнічний інститут імені Ігоря Сікорського»},
  type         = {Дисертація на здобуття наукового ступеня доктора філософії},
  year         = {2026},
  address      = {Київ},
  note         = {Спеціальність 122 — Комп'ютерні науки}
}
```

## Ліцензія

Код у цьому репозиторії доступний для академічного й освітнього використання. Опублікований текст дисертації та автореферату — з обов'язковим посиланням на джерело відповідно до п. 7 наказу МОН України № 40 від 12.01.2017.
