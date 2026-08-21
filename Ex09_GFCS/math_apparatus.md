# Ex09: Математичний апарат суміжних робіт
## Для перевірки оригінальності EvoMerge

---

## A. Activation-based Similarity / Importance

### A1. WANDA (Sun et al., 2023)
```
score(w_ij) = |w_ij| · ‖X_j‖₂

де:
  w_ij — вага з'єднання neuron i → neuron j
  X_j  — вектор активацій j-го вхідного нейрона на calibration batch
  ‖·‖₂ — L2-норма

Pruning: видалити w_ij з найменшим score.
```

### A2. Top-p L1-mass Masking (ICLR 2026)
```
top-p(v) = m_p ⊙ v

m_p = argmin_m ‖m‖₀  s.t.  ‖m ⊙ v‖₁ ≥ p · ‖v‖₁,  m ∈ {0,1}^n

де:
  v   — вектор активацій шару
  m_p — бінарна маска
  p   — частка L1-mass що зберігається
  
Операція: зберегти мінімальну кількість активацій що покривають p% L1-норми.
```

### A3. ZipIt! Feature Correlation (Stoica et al., 2023)
```
C_ij = corr(f_i^A, f_j^B)

Merge matrix: M ∈ R^{n × 2n}
f* = M · [f_A; f_B]

Unmerge: U = M⁺ (pseudoinverse)
Propagation: W_next ← W_next · U

де:
  f_i^A, f_j^B — features двох різних моделей
  C_ij — Pearson correlation між features
  M    — merge matrix (combines matched features)
  U    — unmerge matrix (consistency for next layer)

Операція: match features за correlation, merge через лінійну комбінацію.
```

---

## B. Neuron Clustering / Merging

### B1. Neuron Merging: Compensating for Pruned Neurons (Kim et al., NeurIPS 2020)
```
W_l = Y_l · Z_l

де:
  W_l — оригінальна вагова матриця шару l
  Y_l — нові ваги (pruned шар)
  Z_l — scaling matrix ∈ R^{n_remaining × n_original}

Scaling matrix computation:
  z_ij = sim(w_i, w_j) · (‖w_j‖₂ / ‖w_i‖₂)
  sim(w_i, w_j) = cos(w_i, w_j) = (w_i · w_j) / (‖w_i‖₂ · ‖w_j‖₂)

Absorption (ReLU):
  W_{l+1} ← W_{l+1} · Z_l^T

де:
  w_i, w_j — вектори ваг нейронів i, j
  cos()    — cosine similarity
  
Операція: decompose W → Y·Z, absorb Z у наступний шар.
Data-free: використовує тільки ваги, не потребує даних.
```

### B2. Merging Similar Neurons (2020)
```
1. Representation: x_j = [w_j; b_j] ∈ R^{d+1}

2. K-means: {C_1, ..., C_k} = KMeans({x_1, ..., x_n}, k)

3. Centroid: μ_m = (1/|C_m|) · Σ_{j ∈ C_m} x_j

4. Merged weights: w_m^merged = μ_m[1:d],  b_m^merged = μ_m[d+1]

5. Outgoing: W_{l+1}[:,m] = Σ_{j ∈ C_m} W_{l+1}[:,j]

де:
  x_j   — feature vector нейрона (ваги + bias)
  C_m   — кластер m
  μ_m   — centroid (простe середнє)
  k     — кількість кластерів (фіксована вручну)

Операція: K-means на вагах → centroid averaging → sum outgoing.
```

### B3. Optimal Neuron Merging (Goldberg et al., 2022)
```
min_{W'} ‖f(x; W) - f(x; W')‖²

Closed-form: при видаленні neuron j з шару l:
  W'_{l+1}[:,i] ← W_{l+1}[:,i] + (W_{l+1}[:,j] · r_ij)  ∀i ≠ j

Upper bound on reconstruction error:
  ‖f - f'‖ ≤ Σ_l ‖W_{l+1}[:,j]‖₂ · ε_l

де:
  r_ij — compensation ratio (closed-form з ваг)
  ε_l  — layer-wise approximation error

Операція: видалити neuron j, перерозподілити його вихідні ваги 
          на сусідні нейрони з closed-form коефіцієнтами.
Non-decomposition: не розкладає W на Y·Z.
```

### B4. NeuronMerge: Functional Groups (ACL 2025)
```
1. Classify neurons: g(n_j) ∈ {General, Math, Code, Translation}

2. Task vector: τ = θ_finetuned - θ_base

3. Per-group merge: θ_merged^g = θ_base + Σ_t λ_t · τ_t^g

де:
  g()  — функціональний класифікатор нейронів
  τ    — task vector (різниця fine-tuned vs base)
  λ_t  — вага задачі t

Операція: класифікувати нейрони за функцією, мержити task vectors per group.
Cross-model: працює з РІЗНИМИ моделями, не з pruned моделлю.
```

---

## C. Per-layer Optimization

### C1. AMC: AutoML for Model Compression (He et al., ECCV 2018)
```
RL agent (DDPG):
  State:   s_t = embed(layer_t)  — embedding шару (n, c, h, w, stride, FLOPs...)
  Action:  a_t = sparsity_ratio_t ∈ [0, 1]
  Reward:  R = -Error(pruned_model) · 1[FLOPs ≤ budget]
  Policy:  π(s_t) → a_t

Update: standard DDPG (actor-critic with experience replay)

де:
  embed() — feature vector шару
  a_t     — яку частку нейронів/каналів видалити в шарі t

Операція: RL вчить per-layer compression policy.
```

### C2. Sheared LLaMA (Xia et al., 2023)
```
min_{z} max_{λ} L(θ ⊙ z) + Σ_i λ_i · (g_i(z) - t_i)

де:
  z    — learnable continuous masks ∈ [0,1]
  λ_i  — Lagrange multipliers
  g_i(z) — structural constraint (e.g., #heads, hidden_dim)
  t_i  — target architecture size

Importance via Taylor:
  I_j = |∂L/∂z_j · z_j|

Dynamic batch loading:
  w_d ← w_d · exp(η · L_d / L̄)

Операція: Lagrange-based constrained optimization для targeted structured pruning.
```

---

## D. Weight Compensation / Synthesis

### D1. SparseGPT (Frantar & Alistarh, ICML 2023)
```
Problem: min_{M} ‖WX - (W ⊙ M)X‖²_F

Weight update (after pruning column p):
  δw = -w_p · [H⁻¹]_{:,p} / [H⁻¹]_{p,p}

Hessian: H = XX^T + λI

Incremental inverse (Cholesky):
  [H_{U_{j+1}}]⁻¹ = B - (1/B_{11}) · B_{:,1} · B_{1,:}
  де B = [H_{U_j}]⁻¹

де:
  W — weight matrix шару
  X — input activations (calibration data)
  M — binary pruning mask
  H — Hessian (second-order)
  
Операція: pruning + Hessian-inverse weight compensation, column by column.
```

### D2. OBC — Optimal Brain Compression (Frantar & Alistarh, NeurIPS 2022)
```
Row-wise pruning:
  min_{δw_row} δw_row^T · H_row · δw_row
  s.t. w_p + δw_p = 0

Solution: δw = -w_p · [H_row⁻¹]_{:,p} / [H_row⁻¹]_{p,p}

де:
  H_row — per-row Hessian submatrix
  
Операція: row-Hessian compensation. Precursor to SparseGPT.
```

### D3. Fast Weight Update (Boza, 2024)
```
δW = -H⁻¹ · g

де:
  H — Hessian (або Fisher approximation)
  g — gradient of loss w.r.t. pruned weights

Операція: simplified Hessian-gradient correction post-pruning.
```

---

## E. Structured Pruning → Compact Dense

### E1. Network Slimming (Liu et al., ICCV 2017)
```
L = Σ_{(x,y)} l(f(x,W), y) + λ · Σ_{γ ∈ Γ} |γ|

Pruning criterion: remove channel i if |γ_i| < threshold

де:
  γ — scaling factors з Batch Normalization: y = γ·x̂ + β
  λ — sparsity coefficient

Операція: L1 regularization на BN γ → prune channels → compact dense network.
```

### E2. BUnit-Net (IEEE TPAMI 2024)
```
Compact network = Stack(B_1, B_2, ..., B_k)

де B_i — hybrid unit (dense block з mixed operations)

Операція: direct construction of compact dense network через stacking.
Не потребує pruning — будує compact arch напряму.
```

---

## F. Knowledge Distillation

### F1. Hinton KD (2015)
```
L = α · T² · KL(σ(z_s/T) ‖ σ(z_t/T)) + (1-α) · CE(y, σ(z_s))

де:
  z_t, z_s — logits teacher/student
  T — temperature (зазвичай 2-20)
  α — баланс soft/hard loss
  σ — softmax

Операція: student мімікрує soft predictions teacher з temperature scaling.
```

### F2. Model Compression (Buciluă et al., 2006)
```
L = Σ_x ‖f_s(x) - f_t(x)‖²

де:
  f_t — teacher predictions (hard labels or soft)
  f_s — student being trained

Операція: student регресує на outputs teacher. Founding KD.
```

---

## G. SVD Compression

### G1. Denton et al. (NIPS 2014)
```
W ≈ U_r · Σ_r · V_r^T

Layer split: 
  Original: y = Wx
  Compressed: y = U_r · (Σ_r · V_r^T · x)  — 2 менших шари

де:
  r — truncated rank (r << min(m,n))
  U_r ∈ R^{m×r}, Σ_r ∈ R^{r×r}, V_r ∈ R^{n×r}

Операція: truncated SVD → заміна одного шару двома меншими.
```

### G2. Fisher-Weighted SVD (2024)
```
W̃ = F^{1/2} · W

W̃ ≈ U_r Σ_r V_r^T  (SVD of Fisher-weighted W)

W_compressed = F^{-1/2} · U_r Σ_r V_r^T

де:
  F = E[∇log p(y|x) · ∇log p(y|x)^T]  — Fisher information matrix

Операція: SVD з Fisher weighting — зберігати компоненти важливі для задачі.
```

---

## H. Optimal Transport для Neuron Alignment

### H1. Model Fusion via OT (Singh & Alistarh, NeurIPS 2020)
```
T* = argmin_{T ∈ Π(μ,ν)} Σ_{i,j} c(i,j) · T_{ij}

Cost: c(i,j) = ‖w_i^A - P·w_j^B‖²

Fused weights: w_m = Σ_j T*_{m,j} · w_j^B  (Wasserstein barycenter)

де:
  T   — transport plan ∈ R^{n_A × n_B}
  Π   — допустимі transport plans (marginals = μ, ν)
  P   — permutation alignment
  c() — transport cost (Euclidean у weight space)

Операція: OT alignment neurons між 2 моделями → Wasserstein barycenter averaging.
Cross-model fusion, не single-model compression.
```
