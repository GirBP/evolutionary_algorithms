# Експеримент 08: Опис та класифікація методів прунінгу

> 25 методів, SimpleMLP (20K параметрів), набір даних moons, 15 рівнів спарсності (10–97%)
> 
> **ТРЕТЯ ІТЕРАЦІЯ ПЕРЕВІРКИ НОВИЗНИ** (критичний аналіз)

---

## Категорія 1: Бібліотечні методи (базові лінії з літератури)

### 1. Magnitude Pruning
- **Джерело**: Han et al., "Learning both Weights and Connections for Efficient Neural Networks", NeurIPS 2015
- **Суть**: Видаляє ваги з найменшою абсолютною величиною |w|. Глобальний поріг.
- **Реалізація**: Стандартна, без модифікацій.

### 2. Magnitude-ERK
- **Джерело**: Evci et al., "Rigging the Lottery: Making All Tickets Winners", ICML 2020 (RigL)
- **Суть**: Magnitude pruning з ERK (Erdős–Rényi–Kernel) розподілом спарсності по шарах. Тонші шари отримують менше прунінгу.
- **Реалізація**: Стандартна ERK формула + magnitude scoring.

### 3. SparseGPT (SOTA)
- **Джерело**: Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot", ICML 2023
- **Суть**: Стовпчиковий one-shot прунінг з Hessian-based компенсацією решти ваг. Оригінально для LLM.
- **Реалізація**: Адаптована для SimpleMLP. Виправлено баг із перезаписом замаскованих ваг.

### 4. WANDA (SOTA)
- **Джерело**: Sun et al., "A Simple and Effective Pruning Approach for Large Language Models", ICLR 2024
- **Суть**: Pruning score = |weight| × ‖input activations‖. One-shot, без fine-tuning.
- **Реалізація**: Стандартна адаптація для SimpleMLP.

### 5. RIA (SOTA)
- **Джерело**: Zhang et al., "RIA: Relative Importance and Activations", 2024
- **Суть**: Відносна важливість ваг з урахуванням активацій. Розширення WANDA.
- **Реалізація**: Стандартна.

### 6. SET
- **Джерело**: Mocanu et al., "Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity Inspired by Network Science", Nature Communications 2018
- **Суть**: Sparse Evolutionary Training — ітеративне видалення і відновлення зв'язків під час навчання. Magnitude drop + random regrow.
- **Реалізація**: Стандартна, з бюджетом за RCU.

---

## Категорія 2: Авторські методи з частковими аналогами в літературі

> Обов'язкове цитування вказаних робіт у дисертації

### 7. E-SMD (Evolutionary Synaptic Metric Discovery)
- **Суть**: CMA-ES еволюціонує per-layer параметри метрики (α, β, γ, p), що визначає score кожної ваги. Кожен шар має власну метрику.
- **Формула score**: `score = α·|W|^p + β·|G| + γ·|W·G|`

**Схожий на: Pruner-Zero (ICML 2024)**
| Параметр | Pruner-Zero | E-SMD |
|----------|-------------|-------|
| Оптимізатор | Genetic Programming (символьні дерева) | CMA-ES (числові параметри) |
| Простір пошуку | Довільні символьні вирази з {W, G, X} | 4 числові параметри (α, β, γ, p) per layer |
| Рівень | Глобальна єдина формула | Per-layer (кожен шар має свої коефіцієнти) |
| Масштаб | LLM (LLaMA, Mistral) | SimpleMLP |
| Fine-tuning | Без | Без |

**Ключова відмінність**: E-SMD шукає **числові коефіцієнти** параметричної формули per-layer, Pruner-Zero шукає **довільну символьну формулу** глобально. E-SMD простіший, але адаптивний до кожного шару.

---

### 8. TESA-26 (Taylor-Evolutionary Sparsity Allocation with Iterative Saliency Recalibration)
- **Суть**: Taylor expansion для оцінки saliency кожного шару → CMA-ES оптимізує розподіл спарсності по шарах → ітеративне перерахування saliency після кожного раунду прунінгу.

**Схожий на: DSA — Discovering Sparsity Allocation (NeurIPS 2024)**
| Параметр | DSA | TESA-26 |
|----------|-----|---------|
| Оптимізатор | Expression discovery + evolutionary | CMA-ES |
| Scoring basis | Довільні metric→ratio функції | Taylor saliency як фіксована основа |
| Ітеративність | One-pass | Iterative recalibration (перерахунок saliency) |
| Масштаб | LLM (LLaMA-1/2/3, Mistral, OPT) | SimpleMLP |

**Ключова відмінність**: TESA-26 базується на **фізично мотивованій Taylor saliency** з ітеративним перерахунком, DSA шукає **довільні функції** без фіксованої основи. TESA-26 має iterative recalibration — кожен раунд прунінгу оновлює оцінки важливості.

---

### 9. EARL (Evolutionary Anisotropic Ricci Landscape)
- **Суть**: Forman-Ricci curvature оцінює важливість ребер (ваг) у графі мережі. CMA-ES еволюціонує параметри α, β для зважування curvature метрики. Post-training pruning.

**Схожий на: RicciNets (Glass, Spasov & Liò, 2021)**
| Параметр | RicciNets | EARL |
|----------|-----------|------|
| Curvature | Ollivier-Ricci + Forman-Ricci | Forman-Ricci |
| Час | Pruning at Initialization (PaI) | Post-training pruning |
| Оптимізація | Фіксований threshold | CMA-ES еволюціонує α, β |
| Мережі | Randomly-wired NNs | Стандартні fully-connected |

**Ключова відмінність**: RicciNets прунить **до навчання** з фіксованим порогом, EARL прунить **навчену мережу** з **еволюційно оптимізованими** параметрами curvature scoring.

---

## Категорія 3: Повністю авторські методи (аналогів у літературі не знайдено)

### 10. E-HTA (Evolutionary Hessian-Trace Approximation)
- **Суть**: CMA-ES оптимізує глобальний параметр λ для pruning score = |g·w| + λ·w². Hessian-trace використовується як додатковий сигнал важливості ваг.
- **Формула**: `score(w) = |gradient × weight| + λ · weight²`
- **Найближчі роботи**: GraSP (2020) — Hessian-gradient product для PaI. OBS/OBD (1990s) — Hessian для оптимального прунінгу.
- **Чому нове**: GraSP використовує фіксовану формулу (не еволюціонує параметри). OBS/OBD обчислюють повний Hessian (дорого). E-HTA **еволюціонує баланс** між gradient і weight через λ.
- Цитувати: GraSP (Wang et al., ICLR 2020)

### 11. E-PQM (Evolutionary Phase-Space Quantization Mapping)
- **Суть**: Будує 2D фазовий простір |W| × |G| для кожного шару. Простір квантується у M×M density matrix. CMA-ES еволюціонує параметри density → threshold mapping.
- **Найближчі роботи**: Quantum pruning (q-iPrune, 2025) — але для квантових мереж (QNN), не класичних.
- **Чому нове**: Фазово-просторове представлення ваг для класичного pruning не знайдено в жодній роботі.
- Підтверджено нове.

### 12. E-ETA (Evolutionary Elastic Topology Adaptation)
- **Суть**: Meta-matrix Θ[2×3] відображає landscape features (mean, std, skewness ваг/градієнтів) на per-layer pruning decisions. CMA-ES еволюціонує елементи Θ.
- **Найближчі роботи**: **MetaPruning** (He et al., ICCV 2019) — PruningNet генерує ваги для pruned architecture. **Rewarded Meta-Pruning** (2023) — reward-guided meta-pruning.
- **Чому нове**: MetaPruning генерує **ваги** мережі, E-ETA генерує **рішення про pruning** через meta-matrix. E-ETA використовує landscape statistics як input (mean/std/skew), MetaPruning — encoding vector.
- Цитувати: MetaPruning (He et al., ICCV 2019), Rewarded Meta-Pruning (2023)

### 13. E-ACDE (Elastic ACDE)
- **Суть**: Aitchison Compositional DE + elastic constraint для mask optimization. Спарсності шарів оптимізуються на simplex (сума = target sparsity) через композиційне перетворення.
- **Найближчі роботи**: DE для layer-wise pruning (2021, MDPI). Soft-inextensibility constraint в FEM.
- **Чому нове**: Aitchison compositional framework (simplex constraint через log-ratio) у pruning не знайдено. DE для pruning існує, але без compositional constraint.
- Підтверджено нове.

### 14. FES-NSDE (Fitness Evolutionary Sparsity — Noise-Scaled DE)
- **Суть**: Fitness-based selection + Noise-Scaled Differential Evolution для оптимізації per-layer ratios. Шум масштабується за складністю ландшафту.
- **Найближчі роботи**: DE для layer-wise pruning (2021). Noisy EA analysis (IJCAI 2025).
- **Чому нове**: Noise-scaling за landscape complexity у DE для pruning не знайдено.
- Підтверджено нове.

### 15. ACDE (Aitchison Compositional DE)
- **Суть**: Differential Evolution з Aitchison compositional constraint для оптимізації per-layer pruning ratios на simplex.
- **Чому нове**: Те саме що E-ACDE, але без elastic частини. Aitchison simplex у pruning не знайдено.
- Підтверджено нове.

### 16. VPAM (Variance-Penalized Activation Masking)
- **Суть**: Variance активацій використовується як penalty при pruning. Маски генеруються з урахуванням variance penalty для кращої роботи на ultra-sparsity.
- **Найближчі роботи**: **AVSS** (Activation Variance-Sparsity Score, 2024) — комбінує activation variance + sparsity для layer importance в LLM. **VBP** (Tan 2022) — variance для structured pruning.
- **Чому нове**: AVSS оцінює **layer importance** (які шари видаляти), VPAM — **weight importance** (які ваги видаляти). AVSS = structured layer removal, VPAM = unstructured weight masking. VBP працює structured, VPAM — unstructured з penalty.
- Цитувати: AVSS (2024), VBP (Tan 2022)

### 17. EvoStruct (Evolutionary Structure)
- **Суть**: Macro-micro decoupling: CMA-ES шукає макро-структуру (layer ratios), DiffSynFlow оцінює мікро-рівень, spectral merge фіналізує маску.
- **Найближчі роботи**: Macro/micro NAS decoupling в NAS (DARTS, ENAS). Spectral pruning (Hanson 1988).
- **Чому нове**: Конкретна комбінація CMA-ES macro + DiffSynFlow micro + spectral merge для pruning не знайдена.
- Підтверджено нове.

### 18. Evo-SynFlow (Quantum Fidelity)
- **Суть**: CMA-ES еволюціонує layer-wise ratios, оцінка через Quantum Fidelity proxy. Найкращий з сімейства Evo-SynFlow у Ex08 (AUSCa=0.670, #9).
- **Найближчі роботи**: SynFlow (Tanaka et al., NeurIPS 2020). DSA (NeurIPS 2024).
- **Чому нове**: QF scoring для класичних нейромереж — не знайдено.
- Цитувати: SynFlow (Tanaka 2020), DSA (NeurIPS 2024)

### 19. Evo-SynFlow (Ex07)
- **Суть**: Multi-iteration SynFlow з importance-weighted aggregation. Перенесено з Ex07 для порівняння.
- **Найближчі роботи**: SynFlow (Tanaka 2020). DSA (NeurIPS 2024).
- Цитувати: SynFlow (Tanaka 2020), DSA (NeurIPS 2024)

### 20–21. Evo-HMT (2 варіанти)
- **Суть**: Hierarchical Mask Tuning з Deferred BN Recalibration та MicroES grid search.
- **Варіанти**: (−ERK) без ERK allocation, (−BN) без BatchNorm recalibration (ablation study)
- **Найближчі роботи**: Hierarchical pruning (He et al. 2018). BN recalibration після pruning (стандартна практика).
- **Чому нове**: **Деферне** BN recalibration (тільки на фінальній топології, НЕ всередині пошуку) + MicroES grid — не знайдено.
- Підтверджено нове.

### 22. SET-v2 (TESA init)
- **Суть**: Базовий SET (Mocanu 2018) із заміною magnitude ініціалізації на TESA-26 ініціалізацію.
- **Найближчі роботи**: SET (Mocanu 2018), RigL (Evci 2020) — обидва використовують magnitude/gradient для init.
- **Чому нове**: Ініціалізація SET через Taylor-evolutionary allocation (TESA) — не знайдено.
- Підтверджено нове (модифікація).

### 23–25. SoftMask (3 варіанти)
- **Суть**: Learnable thresholds для Wanda-style scores.
  - **SoftMask**: Gradient descent на threshold t, soft mask = σ(score − t)
  - **SoftMask-Grad**: Rank-based scoring + exact top-K
  - **SoftMask-Grad-v2**: Adaptive λ regularization + median init
- **Найближчі роботи**: **STR** (Soft Threshold Reparameterization, Kusupati et al., ICML 2020) — learnable per-layer threshold для weight pruning. **LTP** (Azarian 2021) — learnable threshold pruning.
- **Чому нове**: STR і LTP використовують magnitude scoring. SoftMask використовує **Wanda scores** (weight × activation) та rank-based варіанти. Конкретна комбінація STR-подібного threshold + Wanda scoring не знайдена.
- Цитувати: STR (Kusupati et al., ICML 2020), LTP (Azarian 2021)

---

## Зведена таблиця (після 3 ітерацій перевірки)

| # | Метод | Категорія | Обов'язково цитувати |
|---|-------|-----------|---------------------|
| 1 | Magnitude | Бібліотечний | Han et al. 2015 |
| 2 | Magnitude-ERK | Бібліотечний | Evci et al. 2020 |
| 3 | SparseGPT | Бібліотечний | Frantar & Alistarh 2023 |
| 4 | WANDA | Бібліотечний | Sun et al. 2024 |
| 5 | RIA | Бібліотечний | Zhang et al. 2024 |
| 6 | SET | Бібліотечний | Mocanu et al. 2018 |
| 7 | E-SMD | Є аналог | **Pruner-Zero (ICML 2024)** |
| 8 | TESA-26 | Є аналог | **DSA (NeurIPS 2024)** |
| 9 | EARL | Є аналог | **RicciNets (Glass et al. 2021)** |
| 10 | E-HTA | Новий | GraSP (Wang et al. 2020) |
| 11 | E-PQM | Новий | — |
| 12 | E-ETA | Новий | **MetaPruning (ICCV 2019)**, Rewarded Meta-Pruning (2023) |
| 13 | E-ACDE | Новий | — |
| 14 | FES-NSDE | Новий | — |
| 15 | ACDE | Новий | — |
| 16 | VPAM | Новий | **AVSS (2024)**, VBP (Tan 2022) |
| 17 | EvoStruct | Новий | — |
| 18–19 | Evo-SynFlow (QF, Ex07) | Новий | SynFlow (Tanaka 2020), **DSA (NeurIPS 2024)** |
| 20–21 | Evo-HMT ×2 | Новий | — |
| 22 | SET-v2 | Модифікація | SET (Mocanu 2018) |
| 23–25 | SoftMask ×3 | Новий | **STR (Kusupati, ICML 2020)**, LTP (Azarian 2021) |

## Повний список обов'язкових цитувань

1. **Pruner-Zero** (ICML 2024) — при описі E-SMD
2. **DSA** (NeurIPS 2024) — при описі TESA-26, Evo-SynFlow
3. **RicciNets** (Glass et al. 2021) — при описі EARL
4. **MetaPruning** (He et al., ICCV 2019) — при описі E-ETA
5. **AVSS** (2024) — при описі VPAM
6. **STR** (Kusupati et al., ICML 2020) — при описі SoftMask
7. **GraSP** (Wang et al., ICLR 2020) — при описі E-HTA
8. **VBP** (Tan 2022) — при описі VPAM
9. **LTP** (Azarian 2021) — при описі SoftMask
10. **SynFlow** (Tanaka et al., NeurIPS 2020) — при описі Evo-SynFlow
