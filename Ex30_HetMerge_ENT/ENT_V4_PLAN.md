# ENT v4: Математично обґрунтований план покращення

> **Ціль:** для кожного класу c: `acc_merged(c) ≥ acc_best_parent(c) − 0.05`  
> **Поточний стан:** worst drop = 29% (class 9), 6 з 10 класів вже < 5% drop  
> **Підхід:** аналогії з інших областей математики

---

## 0. Формалізація задачі

**Дано:**
- Модель A з вагами θ_A, per-class accuracy a^A_c
- Модель B з вагами θ_B, per-class accuracy a^B_c
- Best parent: p_c = max(a^A_c, a^B_c)

**Знайти:**
- Merged model θ_M такий що forall c: a^M_c ≥ p_c - ε, де ε = 0.05

**Constraint satisfaction problem:**
```
minimize  |θ_M|  (model size)
subject to: 
  ∀c ∈ {0,...,9}: acc(θ_M, c) ≥ p_c − 0.05
```

### Діагностика: де втрачається точність

Декомпозиція per-class drop для MNIST (seed 42):
```
Class   Parent    ENT    Drop    Source of loss
  0     0.989    0.937   5.3%    virt2L projection noise
  1     0.979    0.953   2.7%    within target
  2     0.922    0.808   12.4%   routing conflict with class 7
  3     0.947    0.773   18.4%   hidden neuron coverage gap
  4     0.954    0.866   9.2%    A-neuron pruning too aggressive
  5     0.894    0.654   26.8%   B-output scale miscalibration
  6     0.933    0.787   15.6%   interference A↔B in hidden layer
  7     0.863    0.771   10.7%   shared features with class 1
  8     0.844    0.682   19.2%   low B-parent accuracy → amplified
  9     0.887    0.629   29.1%   worst: routing + scale + coverage
```

**Три основні причини loss:**
1. **virt2L projection** втрачає інформацію (LSQ residual)
2. **Block-diagonal** ізолює A-features від B-features → неможливість cross-use
3. **Output routing** — sigmoid routing is too coarse (one scalar per class)

---

## 1. Аналогії з інших областей

### 1.1 Optimal Transport (OT) — нейронне вирівнювання

**Область:** Теорія оптимального транспорту (Monge-Kantorovich)
**Аналогія:** Нейрони моделей A і B — дві дискретні розподіли. Знайти оптимальне відображення π: neurons_A → neurons_B що мінімізує "вартість переміщення" (функціональну різницю).

**Математика:**
```
Дано: H_A ∈ R^{N×d_A}, H_B ∈ R^{N×d_B}  — активації на каліб. даних
Знайти: π* = argmin_π Σ_{i,j} π_{ij} · c(h_A^i, h_B^j)
  де c(h_i, h_j) = 1 − |corr(h_i, h_j)|  — функціональна відстань

Після вирівнювання:
  Matched pairs (π_{ij} > 0.5): average weights → зберігає обидва
  Unmatched neurons: concatenate (as current ENT)
```

**Очікуваний ефект:** Зменшує розмір (менше дублювання) + зберігає функції обох моделей.
**Складність:** O(d² · N) для OT solver (Sinkhorn).
**Існуючі роботи:** Singh & Jaggi 2020 ("Model Fusion via OT"), ZipIt! (Stoica 2023) — обидві для same-task, не complementary.

**Promise: HIGH** (вирішує проблему block-diagonal ізоляції)

---

### 1.2 Shapley Values — принципове оцінювання важливості нейронів

**Область:** Теорія кооперативних ігор (Шеплі, 1953)
**Аналогія:** Кожен нейрон — "гравець" у коаліційній грі. Shapley value нейрона i для класу c:

```
φ_i^c = Σ_{S⊂N\{i}} [|S|!(|N|-|S|-1)!/|N|!] × [v_c(S∪{i}) − v_c(S)]

де v_c(S) = accuracy на класі c з підмножиною нейронів S
```

**Апроксимація (permutation sampling):**
```
for t = 1..T:
    σ = random_permutation(all_neurons)
    for each neuron i:
        S = neurons before i in σ
        φ_i += v(S∪{i}) − v(S)
    φ_i /= T
```

**Застосування:**
- Для кожного класу c — вибрати top-k нейронів по φᵢ^c
- Гарантія: greedy selection по Shapley дає (1−1/e)-оптимум
- **Per-class constraint:** для кожного c тримати достатньо нейронів щоб a^M_c ≥ p_c − 0.05

**Promise: MEDIUM-HIGH** (теоретично обґрунтовано, але O(2^n) exact → потрібна апроксимація)

---

### 1.3 Multi-Commodity Network Flow

**Область:** Комбінаторна оптимізація, теорія графів
**Аналогія:** Нейронна мережа = directed graph. Кожен клас = "товар" (commodity) який має "протекти" від входу до виходу.

```
Graph G = (V, E)
  V = {input_neurons} ∪ {hidden_A} ∪ {hidden_B} ∪ {output}
  E = all weight connections (with capacities = |w|)

For each class c:
  source = input layer
  sink = output neuron c
  demand_c = p_c − 0.05  (minimum required accuracy)

Maximize: Σ_c flow_c
Subject to: 
  ∀c: flow_c ≥ demand_c
  ∀e: Σ_c flow_c(e) ≤ capacity(e)  (shared capacity)
```

**Прямого розвʼязку немає** (accuracy ≠ flow), але отримуємо **структурний інсайт**: які шляхи критичні для яких класів → priority selection.

**Promise: MEDIUM** (elegant theory, unclear empirical path)

---

### 1.4 Rate-Distortion Theory — інформаційні обмеження

**Область:** Теорія інформації (Shannon)
**Аналогія:** Merging = lossy compression. Кожен нейрон несе Information per class. Rate-distortion дає нижню межу скільки нейронів потрібно при заданому рівні distortion.

```
R(D) = min_{p(m|a,b)} I(θ_A, θ_B; θ_M)
  subject to: E[d(c)] ≤ D  for all c
  де d(c) = max(0, p_c − a^M_c)  — per-class distortion
```

**Практичне застосування:** mutual information між нейроном i та класом c → information-based selection замість magnitude-based pruning.

**Promise: MEDIUM** (gives theoretical bounds, but computation is hard)

---

### 1.5 Constrained Optimization з Per-Class Guarantees

**Область:** Constrained optimization, Lagrangian relaxation
**Аналогія:** Пряме формулювання задачі як constrained optimization.

```
Phase 1: CONCAT all neurons → full model θ_full
Phase 2: Solve constrained pruning:
  minimize  ||mask||_0  (number of neurons)
  subject to: ∀c: acc(θ_full ⊙ mask, c) ≥ p_c − 0.05

Lagrangian relaxation:
  L(mask, λ) = ||mask||_0 + Σ_c λ_c · max(0, p_c − 0.05 − acc(c))

Dual ascent:
  1. Fix λ → solve for mask (pruning step)
  2. Fix mask → update λ_c for violated constraints
```

**Promise: HIGH** (directly optimizes our target, computationally feasible)

---

### 1.6 CKA Feature Matching + Selective Merge

**Область:** Representation similarity (Kornblith et al. 2019)
**Аналогія:** Знайти функціонально еквівалентні нейрони через CKA (Centered Kernel Alignment).

```
CKA(h_i^A, h_j^B) = HSIC(K_i^A, K_j^B) / √(HSIC(K_i^A, K_i^A) · HSIC(K_j^B, K_j^B))

де K = activation_matrix @ activation_matrix^T  (kernel)

If CKA(i, j) > threshold:
    merged_neuron = α·w_i^A + (1−α)·w_j^B  (average equivalent neurons)
Else:
    keep both (concatenate)
```

**Ефект:** Зменшує дублювання → більше "місця" для унікальних нейронів → менше interference.

**Promise: MEDIUM-HIGH** (простий, ефективний, доведена кореляція з functional similarity)

---

## 2. Пріоритетний план (математично зважений)

### Оцінка підходів

| # | Підхід | Rigour | Expected Δ | Effort | **Score** |
|:-:|--------|:------:|:----------:|:------:|:---------:|
| 1 | **Constrained pruning (Lagrangian)** | HIGH | ≤5% target directly | LOW | **9** |
| 2 | **OT neuron alignment** | HIGH | −10-15% drop | MEDIUM | **8** |
| 3 | **CKA matching + selective merge** | MEDIUM | −5-10% drop | LOW | **8** |
| 4 | **Shapley selection** | HIGH | −5-10% drop | HIGH | **7** |
| 5 | Rate-distortion bounds | HIGH | theoretical only | MEDIUM | **5** |
| 6 | Multi-commodity flow | HIGH | structural insight | HIGH | **4** |

> [!IMPORTANT]
> Score = 0.3·Rigour + 0.4·Expected_Δ + 0.3·(1−Effort)

### Рекомендований pipeline: ENT v4

```
┌───────────────────────────────────────────────┐
│           ENT v4 Pipeline                      │
│                                                │
│  1. CKA Matching (identify equivalent neurons) │
│       ↓                                        │
│  2. OT Alignment (align non-equivalent)        │
│       ↓                                        │
│  3. Selective Merge:                           │
│       CKA > 0.8 → average weights             │
│       CKA < 0.8 → concatenate                 │
│       ↓                                        │
│  4. EA routing (per-class, as before)          │
│       ↓                                        │
│  5. Constrained pruning (Lagrangian):          │
│       drop neurons where λ_c constraints met   │
│       ↓                                        │
│  6. Verify: ∀c: drop(c) ≤ 5%                  │
│       FAIL → increase λ_c → re-prune          │
└───────────────────────────────────────────────┘
```

### Конкретні кроки реалізації

**Step 1 (E39): CKA + OT alignment** — 1 script
```python
# Compute CKA matrix between all A-neurons and B-neurons
# Use Sinkhorn OT to find matching
# Merge matched, concat unmatched
# Compare per-class drop vs current ENT
```

**Step 2 (E40): Constrained pruning** — 1 script  
```python
# Start from full concat model
# Lagrangian dual ascent:
#   For each class c with drop > 5%:
#     λ_c *= 2  (increase penalty)
#   Prune neurons with lowest Lagrangian contribution
# Until all constraints satisfied or no more prunable
```

**Step 3 (E41): Combined pipeline** — 1 script
```python
# CKA match → OT align → concat → EA route → Lagrangian prune
# Test on MNIST + CIFAR-10
# Target: all 10 classes within 5% of parent
```

---

## 3. Математичні гарантії

### [FACT] Що можна довести:

1. **Greedy submodular selection** (Shapley): якщо accuracy є submodular function від набору нейронів (що емпірично приблизно так), greedy selection гарантує (1−1/e) ≈ 0.632 від оптимуму.

2. **OT alignment error bound:** якщо Wasserstein distance W₂(H_A, H_B) < δ, то |acc(merged) − acc(parent)| ≤ L·δ, де L — Lipschitz constant мережі.

3. **Lagrangian dual**: сильна двоїстість гарантує що optimal dual value = optimal primal value для convex relaxation задачі pruning.

### [CHALLENGE] Що НЕ можна гарантувати:

1. **5% drop** — це **дуже жорстка** вимога. Для класу 9 (parent=0.887, ENT=0.629) потрібно підняти з 0.629 до 0.837. Це +33% absolute, при тому що parent B дає 0.887. Питання: чи зберігається достатньо B-інформації після virt2L projection?

2. **virt2L bottleneck:** LSQ regression від 784→64 → 10 **втрачає інформацію**. Drop ≤5% може бути **неможливий** без заміни virt2L на щось краще (напр., direct feature transfer або fine-tuning).

### [HYPOTHESIS] Досяжна ціль:

- Класи 0,1,2,4 (drop < 13%): ≤5% реалістично з CKA+OT
- Класи 3,6,7 (drop 10-18%): ≤5% можливо з Lagrangian pruning
- Класи 5,8,9 (drop 19-29%): ≤5% **потребує** post-merge fine-tuning або заміну virt2L
- **Реалістична ціль v4:** ≤10% drop на всіх класах, ≤5% на 7+ класах
