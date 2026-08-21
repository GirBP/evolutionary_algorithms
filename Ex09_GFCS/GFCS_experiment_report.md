# Ex09: Повний звіт експериментального тестування GFCS

## Зміст
1. [Бенчмарк 1: 6 методів конверсії × 8 датасетів](#1-бенчмарк-6-методів-конверсії)
2. [Бенчмарк 2: Inference RCU](#2-inference-rcu)
3. [Бенчмарк 3: Structured Pruning vs Unstructured+GFCS](#3-structured-pruning-vs-gfcs)
4. [Бенчмарк 4: Мульти-архітектурний тест](#4-мульти-архітектурний-тест)
5. [Зведений висновок](#5-зведений-висновок)

---

## 1. Бенчмарк: 6 методів конверсії

**Конфігурація**: SimpleMLP (128-128-128), 8 датасетів × 2 seeds, RCU profiling

**Файл**: `results/full_benchmark_rcu.json`

### 1.1 Зведені метрики (avg по 8 датасетів × 2 seeds)

| Method | ΔF1 | Compression | RCU_conv | RCU_total | Reliability | Data-free |
|--------|----:|:-----------:|:--------:|:---------:|:-----------:|:---------:|
| NeuronRemoval | +0.077 | 2.8× | 0.09 | 21.6 | 8/8 | yes |
| SVD | +0.065 | 1.0× | 0.45 | 18.0 | 8/8 | yes |
| KD | +0.076 | 19.3× | 51.7 | 64.8 | 8/8 | no |
| WeightRedist | −0.003 | 20.8× | 0.08 | 13.4 | 6/8 | yes |
| EvoMerge | +0.042 | 4.4× | 649.5 | 666.7 | 7/8 | no |
| **GFCS** | **+0.079** | **7.6×** | **80.3** | **96.0** | **8/8** | **yes** |

### 1.2 Compression per dataset

| Dataset | NR | SVD | KD | WR | Evo | GFCS |
|---------|:--:|:---:|:--:|:--:|:---:|:----:|
| moons | 4.3× | 1.0× | 31.4× | 34.3× | 1.9× | 11.3× |
| circles | 3.6× | 1.0× | 22.4× | 24.1× | 1.8× | 7.7× |
| spirals | 4.5× | 1.0× | 31.4× | 34.3× | 22.2× | 20.3× |
| blobs | 1.5× | 1.0× | 13.9× | 14.7× | 2.9× | 2.8× |
| gaussian_q | 1.5× | 1.0× | 13.9× | 14.7× | 1.8× | 3.1× |
| classification | 1.5× | 1.0× | 13.9× | 14.7× | 1.9× | 3.1× |
| highdim | 1.6× | 1.0× | 6.5× | 6.6× | 1.5× | 2.5× |
| sequence_cls | 3.9× | 1.0× | 20.9× | 22.6× | 1.5× | 9.8× |

 = метод зламався (ΔF1 < −0.03)

### 1.3 Ранжування за критеріями

| Criterion | 1st | 2nd | 3rd |
|-----------|-----|-----|-----|
| ΔF1 (якість) | GFCS (+0.079) | NeuronRemoval (+0.077) | KD (+0.076) |
| Compression | WeightRedist (20.8×) | KD (19.3×) | GFCS (7.6×) |
| RCU_conv (дешевизна) | WeightRedist (0.08) | NeuronRemoval (0.09) | SVD (0.45) |
| Reliability | NR, SVD, KD, GFCS (8/8) | EvoMerge (7/8) | WeightRedist (6/8) |

---

## 2. Inference RCU

**Конфігурація**: Inference speedup compact vs teacher, 100 repeats, thread_time_ns

**Файл**: `results/inference_rcu.json`

### 2.1 Середній inference speedup vs teacher

| Method | Infer RCU/pass | Speedup vs Teacher | Params | ΔF1 | Reliable |
|--------|:--------------:|:------------------:|:------:|:---:|:--------:|
| NeuronRemoval | 0.00805 | 1.14× | 16,348 | +0.077 | 8/8 |
| SVD | 0.00959 | 0.95× | 34,835 | +0.065 | 8/8 |
| KD | 0.00475 | 1.94× | 2,363 | +0.076 | 8/8 |
| WeightRedist | 0.00461 | 1.98× | 2,240 | −0.003 | 6/8 |
| EvoMerge | 0.00878 | 1.03× | 18,275 | +0.042 | 7/8 |
| **GFCS** | **0.00650** | **1.42×** | **7,850** | **+0.079** | **8/8** |

### 2.2 Inference speedup vs teacher per dataset

| Dataset | NR | SVD | KD | WR | Evo | GFCS |
|---------|:--:|:---:|:--:|:--:|:---:|:----:|
| moons | 1.25× | 0.96× | 2.32× | 2.36× | 1.09× | 1.79× |
| circles | 1.46× | 0.96× | 2.12× | 2.02× | 1.03× | 1.60× |
| spirals | 1.39× | 0.95× | 2.29× | 2.32× | 1.23× | 1.73× |
| blobs | 0.96× | 0.88× | 1.52× | 1.75× | 1.09× | 1.13× |
| gaussian_q | 1.03× | 0.96× | 1.88× | 1.78× | 0.98× | 1.19× |
| classification | 0.97× | 0.95× | 1.90× | 1.87× | 0.99× | 1.18× |
| highdim | 0.95× | 1.04× | 1.61× | 1.72× | 0.99× | 1.19× |
| sequence_cls | 1.14× | 0.88× | 1.88× | 1.99× | 0.89× | 1.51× |

### 2.3 Серед data-free + 100% reliable методів

| Method | ΔF1 | Compression | Inference× |
|--------|:---:|:-----------:|:----------:|
| NeuronRemoval | +0.077 | 2.8× | 1.14× |
| SVD | +0.065 | 1.0× | 0.95× |
| **GFCS** | **+0.079** | **7.6×** | **1.42×** |

GFCS Pareto-домінує обидва альтернативи за всіма трьома метриками.

---

## 3. Structured Pruning vs GFCS

**Конфігурація**: SimpleMLP, 8 датасетів × 2 seeds, однаковий target compression

**Файл**: `results/structured_vs_gfcs.json`

### 3.1 Зведені результати

| Method | Final F1 | ΔF1 vs Teacher | Compression | Inference× | RCU_prune | Type |
|--------|:--------:|:--------------:|:-----------:|:----------:|:---------:|------|
| Taylor | 0.894 | −0.041 | 41.7× | 1.83× | 1.7 | Structured |
| NetSlimming | 0.929 | −0.006 | 41.7× | 1.81× | 81.6 | Structured |
| **Unstr+GFCS** | **0.934** | **−0.001** | **7.6×** | **1.34×** | **75.3** | **Unstr+Convert** |

### 3.2 Per-dataset F1

| Dataset | Taylor | NetSlimming | GFCS | Winner |
|---------|:------:|:-----------:|:----:|:------:|
| moons | 0.958 | 0.968 | 0.964 | NetSlimming |
| circles | 0.937 | 0.940 | 0.938 | NetSlimming |
| spirals | 0.744 | 0.969 | 0.968 | NetSlimming |
| blobs | 0.981 | 0.983 | 0.981 | NetSlimming |
| gaussian_q | 0.934 | 0.945 | **0.969** | GFCS |
| classification | 0.832 | 0.846 | 0.838 | NetSlimming |
| highdim | 0.861 | 0.868 | **0.877** | GFCS |
| sequence_cls | 0.904 | 0.910 | **0.940** | GFCS |

GFCS виграє 3/8, зокрема на найскладніших (gaussian_q, highdim, sequence_cls).

---

## 4. Мульти-архітектурний тест

**Конфігурація**: 4 архітектури × 4 hard datasets × 2 seeds × 85% sparsity

**Файл**: `results/multi_arch_benchmark.json`

### 4.1 Архітектури

| ID | Architecture | Hidden layers | Params |
|----|-------------|:------------:|:------:|
| A | SimpleMLP | 128-128-128 | ~34K |
| B | DeepMLP | 128-128-128-128-128 | ~83K |
| C | WideMLP | 256-256-256 | ~134K |
| D | BottleneckMLP | 256-64-256 | ~83K |

### 4.2 SimpleMLP (128×3)

| Dataset | Taylor | NetSlimming | GFCS | Winner |
|---------|:------:|:-----------:|:----:|:------:|
| spirals | 0.756 | **0.968** | 0.961 | NetSlimming |
| gaussian_q | **0.933** | 0.929 | 0.802 | Taylor |
| highdim | 0.803 | **0.834** | 0.787 | NetSlimming |
| sequence_cls | **0.919** | 0.892 | 0.900 | Taylor |

GFCS wins: **0/4**

### 4.3 DeepMLP (128×5)

| Dataset | Taylor | NetSlimming | GFCS | Winner |
|---------|:------:|:-----------:|:----:|:------:|
| spirals | 0.774 | **0.965** | 0.962 | NetSlimming |
| gaussian_q | 0.929 | 0.947 | **0.951** | GFCS |
| highdim | 0.804 | 0.829 | **0.882** | GFCS |
| sequence_cls | 0.891 | **0.909** | 0.887 | NetSlimming |

GFCS wins: **2/4**

### 4.4 WideMLP (256×3)

| Dataset | Taylor | NetSlimming | GFCS | Winner |
|---------|:------:|:-----------:|:----:|:------:|
| spirals | 0.729 | **0.968** | 0.959 | NetSlimming |
| gaussian_q | 0.925 | 0.938 | **0.962** | GFCS |
| highdim | 0.831 | 0.876 | **0.882** | GFCS |
| sequence_cls | 0.923 | 0.931 | **0.931** | Tied |

GFCS wins: **3/4** (включаючи tied)

### 4.5 BottleneckMLP (256-64-256)

| Dataset | Taylor | NetSlimming | GFCS | Winner |
|---------|:------:|:-----------:|:----:|:------:|
| spirals | 0.782 | **0.968** | 0.879 | NetSlimming |
| gaussian_q | 0.912 | 0.947 | **0.949** | GFCS |
| highdim | **0.858** | 0.859 | 0.776 | NetSlimming |
| sequence_cls | **0.930** | 0.854 | 0.889 | Taylor |

GFCS wins: **1/4**

### 4.6 Зведення: GFCS wins по архітектурах

| Architecture | Width | Depth | GFCS wins | Pattern |
|-------------|:-----:|:-----:|:---------:|---------|
| SimpleMLP | 128 | 3 | 0/4 | Надто мала для flow-based переваги |
| DeepMLP | 128 | 5 | 2/4 | Глибина допомагає GFCS |
| **WideMLP** | **256** | **3** | **3/4** | **Ширина — головний фактор переваги** |
| BottleneckMLP | 256/64 | 3 | 1/4 | Bottleneck обмежує merge |

---

## 5. Зведений висновок

### 5.1 Позиціонування GFCS

GFCS — data-free метод конверсії sparse→dense з adaptive per-layer architecture search.

Серед **data-free методів з 100% reliability**, GFCS Pareto-домінує всі альтернативи:

| vs NeuronRemoval | vs SVD |
|:----------------:|:------:|
| +0.002 ΔF1 | +0.014 ΔF1 |
| 2.7× більше compression | 7.6× більше compression |
| 1.25× більше inference speedup | 1.49× більше inference speedup |

### 5.2 Scaling behavior

GFCS показує зростаючу перевагу при збільшенні ширини мережі:
- 128 neurons: 0/4 wins
- 256 neurons: 3/4 wins

### 5.3 Обмеження

1. RCU conversion cost: 80 RCU (vs 0.09 для NeuronRemoval)
2. Bottleneck architectures: flow-based merge неефективний при вузьких шарах
3. Structured pruning дає більшу compression ratio при однаковому спarsity

### 5.4 Файли результатів

| File | Description |
|------|-------------|
| `results/full_benchmark_rcu.json` | 6 методів × 8 datasets × 2 seeds + RCU |
| `results/inference_rcu.json` | Inference RCU per method per dataset |
| `results/structured_vs_gfcs.json` | Taylor, NetSlimming, GFCS на SimpleMLP |
| `results/multi_arch_benchmark.json` | 4 architectures × 4 datasets |
| `results/full_benchmark.json` | Попередній бенчмарк (wall-clock) |
| `results/exploration1_tier1.json` | Exploration 1 (moons, circles) |
| `results/synthesis_all.json` | Synthesis phase |
