"""
SDR: Sparse-to-Dense Restructuring
====================================
Novel operator: EA-optimized pruning ratios → Compact Dense Topology → Knowledge Distillation.

KEY IDEA:
  Instead of keeping a sparse matrix (many zeros, inefficient on CPU/GPU),
  derive a NEW compact dense architecture from EA-found layer-wise pruning ratios,
  then transfer knowledge via KD from the original dense teacher.

NOVELTY CLAIM (vs 18 operators A–H from Ex09 PRIOR ART):
  - NOT algebraic merge (≠ GFCS B1-B3): builds NEW model via KD, no neuron merging
  - NOT standard KD (≠ F1,F2): student topology is DERIVED from EA pruning ratios, not fixed
  - NOT NAS (≠ Ex16): architecture comes from pruning analysis, not search
  - NOVEL: First method using EA-optimized pruning ratios as implicit NAS for
    determining compact student topology

Pipeline:
  1. Sparse model + EA-found ratios → determine per-layer active neuron count
  2. Build compact dense student: Linear(in_i', out_i') where out_i' = round(r_i × out_i)
  3. Knowledge Distillation: teacher (original dense) → student (compact)
  4. Fine-tune student on task loss

Comparison:
  - SDR-Magnitude: restructure using Magnitude pruning ratios (baseline)
  - SDR-EvoSF: restructure using (μ+λ)-ES with KD-aware fitness (our method)
  Hypothesis: SDR-EvoSF finds BETTER topology than SDR-Magnitude at same model size.

FITNESS FUNCTION v2 (KD-aware):
  Instead of flow-fitness (designed for GFCS merge), we use a direct KD-alignment
  fitness that evaluates how well a student topology can approximate the teacher:

  fitness(ratios) = -KD_loss(student(X; ratios), teacher(X))  -  λ·(P'/P)

  Where:
    - Student is initialized randomly and trained for 3 micro-KD epochs
    - KD_loss = KL(student/T || teacher/T) on a micro-batch
    - λ·(P'/P) penalizes large models (encourages compression)

  This directly measures what we optimize for, unlike flow-fitness which 
  measures gradient-flow preservation (relevant for merge, not restructure).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random as rnd
import copy
import time


# ═══════════════════════════════════════════
#  Topology derivation from pruning ratios
# ═══════════════════════════════════════════

def derive_topology_from_sparse(sparse_model):
    """
    Derive per-layer keep-ratios from an already-pruned sparse model.
    Returns the fraction of active neurons per hidden layer.
    
    This is the 'SDR-Magnitude' baseline: topology is determined by
    which neurons survived magnitude pruning.
    """
    layers = [sparse_model.fc1, sparse_model.fc2, sparse_model.fc3, sparse_model.fc4]
    ratios = []
    
    for i, layer in enumerate(layers[:-1]):  # skip output layer
        W = layer.weight.data  # [out, in]
        incoming_alive = (W.abs().sum(dim=1) > 1e-8)  # [out]
        
        if i + 1 < len(layers):
            W_next = layers[i + 1].weight.data
            outgoing_alive = (W_next.abs().sum(dim=0) > 1e-8)  # [out]
            alive = incoming_alive & outgoing_alive
        else:
            alive = incoming_alive
        
        n_active = alive.sum().item()
        n_total = W.shape[0]
        ratio = max(n_active / n_total, 0.05)
        ratios.append(ratio)
    
    return ratios


def build_compact_topology(original_hiddens, ratios):
    """
    Build compact hidden dimensions from original hidden sizes and ratios.
    
    Args:
        original_hiddens: int (uniform) or list of ints (per-layer sizes)
        ratios: list of floats, one per hidden layer
    
    Example:
      original: [256, 128, 64] with ratios [0.5, 0.5, 0.5]
      → compact: [128, 64, 32]
    """
    if isinstance(original_hiddens, (int, float)):
        original_hiddens = [int(original_hiddens)] * len(ratios)
    
    compact_hiddens = []
    for h, r in zip(original_hiddens, ratios):
        new_size = max(4, int(round(float(h * r))))
        compact_hiddens.append(new_size)
    return compact_hiddens


# ═══════════════════════════════════════════
#  KD-Aware Fitness for Evolutionary Search
# ═══════════════════════════════════════════

def _kd_fitness(teacher, ratios, input_dim, original_hiddens, n_classes,
                micro_batch_X, micro_batch_y,
                T=4.0, micro_kd_epochs=5, micro_lr=0.005):
    """
    KD-aware fitness function for evolutionary topology search (v2).
    
    Directly measures how well a student with given topology can approximate
    the teacher's outputs after brief training.
    
    fitness = micro_accuracy + 0.5 · (1 - kd_loss_normalized)
    
    Key design decisions:
      - NO compression penalty: we constrain size via ratio bounds instead.
        (v1 had λ·(P'/P) which rewarded tiny models → bad quality)
      - Uses ACCURACY on micro-batch as primary signal (classification quality)
      - KD loss as secondary signal (distribution matching quality)
      - 5 micro-epochs for more stable ranking (v1 had 3 → noisy)
    
    Returns:
        fitness: float (higher is better, range ~[0, 1.5])
    """
    from ex09_lib.core import CompactMLP
    
    compact_hiddens = build_compact_topology(original_hiddens, ratios)
    student = CompactMLP(
        input_dim=input_dim,
        hiddens=compact_hiddens,
        n_classes=n_classes
    )
    
    # Micro-KD training: 5 epochs on micro-batch
    student.train()
    teacher.eval()
    opt = optim.Adam(student.parameters(), lr=micro_lr)
    
    with torch.no_grad():
        teacher_logits = teacher(micro_batch_X)
    
    for _ in range(micro_kd_epochs):
        student_logits = student(micro_batch_X)
        
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T * T)
        
        hard_loss = F.cross_entropy(student_logits, micro_batch_y)
        loss = 0.7 * soft_loss + 0.3 * hard_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    # Final evaluation: accuracy + KD loss
    student.eval()
    with torch.no_grad():
        student_logits = student(micro_batch_X)
        
        # Accuracy on micro-batch (primary signal)
        preds = student_logits.argmax(dim=1)
        accuracy = (preds == micro_batch_y).float().mean().item()
        
        # KD divergence (secondary signal, normalized to ~[0, 1])
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ).item()
    
    # Fitness = accuracy + 0.5·(1 - normalized_kd_loss)
    # kd_loss is typically in [0, 5], so clip and normalize
    kd_normalized = min(kd_loss / 3.0, 1.0)
    fitness = accuracy + 0.5 * (1.0 - kd_normalized)
    
    return fitness


def _evolve_sdr_ratios(teacher, sparse_model, train_dl, n_classes=2,
                        pop_size=24, generations=25):
    """
    (μ+λ)-ES to evolve optimal topology ratios for SDR using KD-aware fitness v2.
    
    Key differences from v1:
      - Ratio bounds derived from magnitude (±30% of mag ratio, not [0.05, 0.85])
      - No compression penalty in fitness (constrained by ratio bounds instead)
      - Larger micro-batch (256 samples) for more stable evaluation
      - 5 micro-KD epochs per evaluation (was 3)
      - More informed seeds (8 variants around magnitude ratios)
    
    This ensures EA searches for BETTER topologies near magnitude-derived ones,
    rather than searching for arbitrarily small models.
    """
    n_layers = 3  # fc1, fc2, fc3
    input_dim = sparse_model.input_dim
    layers = [sparse_model.fc1, sparse_model.fc2, sparse_model.fc3]
    original_hiddens = [l.weight.shape[0] for l in layers]
    
    # Extract micro-batch (256 samples for stability)
    micro_X_list, micro_y_list = [], []
    total_samples = 0
    for X, y in train_dl:
        micro_X_list.append(X)
        micro_y_list.append(y)
        total_samples += X.shape[0]
        if total_samples >= 256:
            break
    micro_X = torch.cat(micro_X_list)[:256]
    micro_y = torch.cat(micro_y_list)[:256]
    
    # Get magnitude-derived ratios as anchor
    mag_ratios = np.array(derive_topology_from_sparse(sparse_model))
    
    # Constrained search bounds: ±30% of magnitude ratios
    # This ensures EA finds topologies of COMPARABLE size, not tiny ones
    lb = np.clip(mag_ratios * 0.7, 0.05, 1.0)  # lower bound: 70% of magnitude
    ub = np.clip(mag_ratios * 1.3, 0.10, 0.95)  # upper bound: 130% of magnitude
    
    # Ensure lb < ub
    for i in range(n_layers):
        if lb[i] >= ub[i]:
            lb[i] = max(0.05, ub[i] - 0.1)
    
    # Initialize population with informed seeds
    population = []
    population.append(mag_ratios.copy())  # 0: exact magnitude
    population.append(np.clip(mag_ratios * 1.1, lb, ub))   # 1: slightly larger
    population.append(np.clip(mag_ratios * 0.9, lb, ub))   # 2: slightly smaller
    population.append(np.clip(mag_ratios * [1.2, 1.0, 0.8], lb, ub))  # 3: wider early
    population.append(np.clip(mag_ratios * [0.8, 1.0, 1.2], lb, ub))  # 4: wider late
    population.append(np.clip(mag_ratios * [1.1, 1.2, 1.1], lb, ub))  # 5: wider middle
    population.append(np.clip(mag_ratios * [1.0, 0.8, 1.0], lb, ub))  # 6: narrow middle
    population.append(np.clip(mag_ratios * [1.15, 1.15, 1.15], lb, ub))  # 7: uniform up
    
    # Fill rest with random perturbations
    while len(population) < pop_size:
        g = mag_ratios + np.random.randn(n_layers) * 0.08
        g = np.clip(g, lb, ub)
        population.append(g)
    
    def eval_fitness(g):
        return _kd_fitness(
            teacher, g, input_dim, original_hiddens, n_classes,
            micro_X, micro_y,
            micro_kd_epochs=5, micro_lr=0.005
        )
    
    best_fitness = -float('inf')
    best_genotype = population[0].copy()
    
    for gen in range(generations):
        fitnesses = [eval_fitness(g) for g in population]
        
        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_genotype = population[gen_best_idx].copy()
        
        # Tournament selection
        mu = max(3, pop_size // 3)
        parents = []
        for _ in range(mu):
            candidates = rnd.sample(range(len(population)), min(3, len(population)))
            winner = max(candidates, key=lambda i: fitnesses[i])
            parents.append(population[winner].copy())
        
        # Gaussian mutation with adaptive sigma
        offspring = [best_genotype.copy()]  # elitism
        offspring.append(mag_ratios.copy())  # always keep magnitude reference
        sigma = 0.08 * (1.0 - 0.4 * gen / generations)  # smaller sigma for constrained
        while len(offspring) < pop_size:
            parent = rnd.choice(parents)
            child = parent + np.random.randn(n_layers) * sigma
            child = np.clip(child, lb, ub)
            offspring.append(child)
        
        population = offspring
    
    return [float(r) for r in best_genotype], best_fitness


# ═══════════════════════════════════════════
#  Knowledge Distillation
# ═══════════════════════════════════════════

def knowledge_distill_transfer(teacher, student, train_dl, 
                                epochs=20, lr=0.005, T=4.0, alpha=0.7):
    """
    Knowledge Distillation from teacher to student.
    
    Loss = α × KL(student/T || teacher/T) × T² + (1-α) × CE(student, labels)
    
    Note: This KD loss is standard (Hinton 2015, operator F1 in PRIOR ART).
    The novelty is in HOW the student topology is determined (from EA ratios),
    not in the distillation mechanism itself.
    """
    teacher.eval()
    student.train()
    opt = optim.Adam(student.parameters(), lr=lr)
    
    for ep in range(epochs):
        for X, y in train_dl:
            with torch.no_grad():
                teacher_logits = teacher(X)
            
            student_logits = student(X)
            
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction='batchmean'
            ) * (T * T)
            
            hard_loss = F.cross_entropy(student_logits, y)
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    return student


# ═══════════════════════════════════════════
#  SDR Conversion Pipeline
# ═══════════════════════════════════════════

def sdr_convert(sparse_model, teacher_model, train_dl, n_classes=2,
                method='evo', kd_epochs=20, ft_epochs=10, ft_lr=0.01):
    """
    SDR: Sparse-to-Dense Restructuring.
    
    Full pipeline:
      1. Derive topology from sparse model (Magnitude, EA, or GFCS ratios)
      2. Build compact dense student
      3. Knowledge Distillation (teacher → student)
      4. Fine-tune on task loss
    
    Args:
        sparse_model: Pruned SimpleMLP
        teacher_model: Original dense SimpleMLP (teacher for KD)
        train_dl: Training dataloader
        n_classes: Number of classes
        method: 'magnitude', 'evo', or 'gfcs'
        kd_epochs: Number of KD epochs
        ft_epochs: Fine-tune epochs after KD
        ft_lr: Fine-tune learning rate
    
    Returns:
        compact_model: CompactMLP with compact topology
        info: dict with statistics
    """
    from ex09_lib.core import CompactMLP, train_model
    
    input_dim = sparse_model.input_dim
    # Support heterogeneous hidden sizes (e.g. LargeMLP: 256→128→64)
    layers = [sparse_model.fc1, sparse_model.fc2, sparse_model.fc3]
    original_hiddens = [l.weight.shape[0] for l in layers]
    
    # Step 1: Derive topology
    t0 = time.time()
    if method == 'magnitude':
        ratios = derive_topology_from_sparse(sparse_model)
    elif method == 'evo':
        ratios, evo_fitness = _evolve_sdr_ratios(
            teacher_model, sparse_model, train_dl, n_classes,
            pop_size=24, generations=25
        )
    elif method == 'gfcs':
        from ex09_lib.gfcs import _evolve_ratios
        gfcs_ratios, gfcs_fitness = _evolve_ratios(
            sparse_model, n_classes,
            pop_size=20, generations=30,
            min_ratio=0.1, max_ratio=0.8
        )
        ratios = [float(r) for r in gfcs_ratios]
    else:
        raise ValueError(f"Unknown method: {method}. Use 'magnitude', 'evo', or 'gfcs'")
    
    topology_time = time.time() - t0
    
    # Step 2: Build compact model
    compact_hiddens = build_compact_topology(original_hiddens, ratios)
    student = CompactMLP(
        input_dim=input_dim,
        hiddens=compact_hiddens,
        n_classes=n_classes
    )
    compact_params = sum(p.numel() for p in student.parameters())
    
    # Step 3: Knowledge Distillation
    t0 = time.time()
    student = knowledge_distill_transfer(
        teacher_model, student, train_dl,
        epochs=kd_epochs, lr=0.005, T=4.0, alpha=0.7
    )
    kd_time = time.time() - t0
    
    # Step 4: Fine-tune
    t0 = time.time()
    student = train_model(student, train_dl, epochs=ft_epochs, lr=ft_lr)
    ft_time = time.time() - t0
    
    info = {
        'method': method,
        'ratios': [round(r, 4) for r in ratios],
        'compact_hiddens': compact_hiddens,
        'compact_params': compact_params,
        'original_hiddens': original_hiddens,
        'topology_time_s': round(topology_time, 3),
        'kd_time_s': round(kd_time, 3),
        'ft_time_s': round(ft_time, 3),
        'total_time_s': round(topology_time + kd_time + ft_time, 3),
    }
    if method == 'evo':
        info['evo_fitness'] = round(evo_fitness, 6)
    if method == 'gfcs':
        info['gfcs_fitness'] = round(gfcs_fitness, 6)
    
    return student, info
