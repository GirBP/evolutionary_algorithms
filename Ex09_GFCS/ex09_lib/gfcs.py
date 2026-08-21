"""
Gradient-Flow Connectivity Synthesis (GFCS)
============================================
Novel operator Ψ for Sparse→Dense conversion with evolutionary architecture search.

KEY NOVELTY vs 18 known operators:
  - NOT weight-similarity clustering (≠ B1,B2): uses gradient-flow connectivity
  - NOT activation-correlation merging (≠ existing EvoMerge/A3): uses backprop paths  
  - NOT Hessian-inverse compensation (≠ D1,D2,D3): no second-order, structural merge
  - NOT SVD factorization (≠ G1,G2): preserves neuron identity, no low-rank approx
  - NOT OT alignment (≠ H1): single model, not cross-model

OPERATOR Ψ_GFCS:
  Given sparse network f_S with mask M:
  
  1. Gradient-Flow Graph Construction:
     φ_i = ||W_in[i,:]||₁ · ||W_out[:,i]||₁  (bidirectional flow capacity)
  
  2. Flow-Weighted Affinity:
     A[i,j] = Σ_k min(|W_in[i,k]|, |W_in[j,k]|) · Σ_m min(|W_out[m,i]|, |W_out[m,j]|)
  
  3. Evolutionary Architecture Search:
     (μ+λ)-ES evolves per-layer merge ratios g = (r_1, ..., r_L)
     Fitness: flow-capacity preservation + compression reward (ZERO-COST)
     
     Novelty vs EvoMerge EA:
       - EvoMerge fitness = activation reconstruction (needs calibration data)
       - GFCS fitness = flow-capacity preservation (needs ONLY weights)
  
  4. Greedy Flow-Preserving Merge (per evolved architecture):
     Ψ(w_i, w_j; φ) = (φ_i·w_i + φ_j·w_j)/(φ_i+φ_j)
     w_out^merged = w_out^i + w_out^j
  
  5. Cascade Compensation across layers
"""
import torch
import torch.nn as nn
import numpy as np
import random as rnd
import copy
from collections import defaultdict



def _compute_flow_importance(W_in, W_out):
    """
    Compute gradient-flow importance φ_i for each neuron i.
    φ_i = Σ_k |W_in[i,k]| · Σ_j |W_out[j,i]|
    
    Bidirectional flow capacity: how much signal enters × how much exits.
    """
    incoming_flow = W_in.abs().sum(dim=1)   # [n_neurons]
    outgoing_flow = W_out.abs().sum(dim=0)  # [n_neurons]
    phi = incoming_flow * outgoing_flow      # element-wise
    return phi


def _compute_flow_affinity(W_in, W_out):
    """
    Compute flow-overlap affinity between pairs of neurons.
    A[i,j] = Σ_k min(|W_in[i,k]|, |W_in[j,k]|) · Σ_m min(|W_out[m,i]|, |W_out[m,j]|)
    
    Captures neurons sharing BOTH similar input AND output pathways.
    """
    n = W_in.shape[0]
    W_in_abs = W_in.abs()
    W_out_abs = W_out.abs()
    
    # Pairwise input-flow overlap
    input_overlap = torch.zeros(n, n)
    for i in range(n):
        input_overlap[i] = torch.minimum(
            W_in_abs[i].unsqueeze(0).expand(n, -1),
            W_in_abs
        ).sum(dim=1)
    
    # Pairwise output-flow overlap
    output_overlap = torch.zeros(n, n)
    for i in range(n):
        output_overlap[i] = torch.minimum(
            W_out_abs[:, i].unsqueeze(1).expand(-1, n),
            W_out_abs
        ).sum(dim=0)
    
    A = input_overlap * output_overlap
    A.fill_diagonal_(0)
    return A


def _greedy_flow_merge(W_in, b_in, W_out, phi, affinity, target_k):
    """
    Greedy Flow-Preserving Merge.
    
    Merge rule (Ψ_GFCS):
      w_in^merged  = (φ_i·w_in^i + φ_j·w_in^j) / (φ_i + φ_j)
      w_out^merged = w_out^i + w_out^j
      b^merged     = (φ_i·b_i + φ_j·b_j) / (φ_i + φ_j)
    
    Iteratively merges neuron with least flow into its most similar neighbor.
    """
    n = W_in.shape[0]
    if n <= target_k:
        return W_in, b_in, W_out, list(range(n))
    
    W_in = W_in.clone()
    b_in = b_in.clone() if b_in is not None else torch.zeros(n)
    W_out = W_out.clone()
    phi = phi.clone()
    affinity = affinity.clone()
    
    active = list(range(n))
    
    while len(active) > target_k:
        active_phi = [(idx, phi[idx].item()) for idx in active]
        active_phi.sort(key=lambda x: x[1])
        
        merged = False
        for victim_idx, _ in active_phi:
            best_partner = None
            best_aff = -1
            for partner_idx in active:
                if partner_idx == victim_idx:
                    continue
                aff = affinity[victim_idx, partner_idx].item()
                if aff > best_aff:
                    best_aff = aff
                    best_partner = partner_idx
            
            if best_partner is None:
                break
            
            phi_v = phi[victim_idx].item()
            phi_p = phi[best_partner].item()
            total_phi = phi_v + phi_p
            
            if total_phi < 1e-12:
                alpha_v, alpha_p = 0.5, 0.5
            else:
                alpha_v = phi_v / total_phi
                alpha_p = phi_p / total_phi
            
            W_in[best_partner] = alpha_v * W_in[victim_idx] + alpha_p * W_in[best_partner]
            b_in[best_partner] = alpha_v * b_in[victim_idx] + alpha_p * b_in[best_partner]
            W_out[:, best_partner] = W_out[:, victim_idx] + W_out[:, best_partner]
            phi[best_partner] = phi_v + phi_p
            
            for k in active:
                if k != victim_idx and k != best_partner:
                    affinity[best_partner, k] = max(
                        affinity[best_partner, k].item(),
                        affinity[victim_idx, k].item()
                    )
                    affinity[k, best_partner] = affinity[best_partner, k]
            
            active.remove(victim_idx)
            merged = True
            break
        
        if not merged:
            break
    
    new_W_in = W_in[active]
    new_b_in = b_in[active] if b_in is not None else None
    new_W_out = W_out[:, active]
    return new_W_in, new_b_in, new_W_out, active


# ═══════════════════════════════════════════
#  Evolutionary Architecture Search
# ═══════════════════════════════════════════

def _flow_fitness(sparse_model, ratios, n_classes, lam=0.5):
    """
    Zero-cost flow-based fitness for evolutionary architecture search.
    
    F(g) = -Σ_l (1 - preserved_flow_l / total_flow_l)  -  λ · (N'/N)
    
    Measures what fraction of gradient-flow capacity is preserved after merging.
    
    Novelty vs EvoMerge fitness:
      - EvoMerge: ||A - Â||_F / ||A||_F  (needs calibration forward pass)
      - GFCS:    1 - Σφ_kept / Σφ_all    (needs ONLY weights — true zero-cost)
    """
    layers = [sparse_model.fc1, sparse_model.fc2, sparse_model.fc3, sparse_model.fc4]
    
    total_flow_error = 0.0
    total_params_merged = 0
    total_params_original = 0
    
    for i, (layer, ratio) in enumerate(zip(layers[:-1], ratios)):
        W_in = layer.weight.data
        W_out = layers[i + 1].weight.data
        
        incoming_alive = (W_in.abs().sum(dim=1) > 1e-8)
        if i + 1 < len(layers):
            outgoing_alive = (W_out.abs().sum(dim=0) > 1e-8)
            alive = incoming_alive & outgoing_alive
        else:
            alive = incoming_alive
        n_active = alive.sum().item()
        target_k = max(4, int(n_active * ratio))
        
        if target_k >= n_active:
            total_params_merged += n_active
            total_params_original += n_active
            continue
        
        phi = _compute_flow_importance(W_in, W_out)
        
        # Top-k flow preservation estimate
        active_indices = torch.where(alive)[0].numpy()
        active_phi = sorted([(idx, phi[idx].item()) for idx in active_indices],
                           key=lambda x: -x[1])
        
        total_flow = sum(p for _, p in active_phi)
        preserved_flow = sum(p for _, p in active_phi[:target_k])
        
        if total_flow > 1e-12:
            flow_error = 1.0 - (preserved_flow / total_flow)
        else:
            flow_error = 0.0
        
        total_flow_error += flow_error
        total_params_merged += target_k
        total_params_original += n_active
    
    compression = total_params_merged / max(total_params_original, 1)
    fitness = -total_flow_error - lam * compression
    return fitness


def _evolve_ratios(sparse_model, n_classes,
                   pop_size=20, generations=30,
                   min_ratio=0.1, max_ratio=0.8):
    """
    (μ+λ)-ES to evolve optimal per-layer merge ratios.
    
    Genotype: g = (r_1, r_2, r_3) where r_l = fraction of neurons to KEEP.
    Fitness:  flow-based zero-cost (no forward pass needed).
    
    Evolution:
      - μ = pop_size // 3 elite parents
      - λ = pop_size offspring via Gaussian mutation
      - Tournament selection (size 3)
      - Adaptive σ: 0.12 → 0.06 over generations
      - Elitism: best always survives
    """
    n_layers = 3  # fc1, fc2, fc3
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        g = np.random.uniform(min_ratio, max_ratio, n_layers)
        population.append(g)
    
    # Seed with heuristic individuals
    population[0] = np.full(n_layers, 0.5)
    population[1] = np.full(n_layers, 0.3)
    population[2] = np.full(n_layers, min_ratio)
    population[3] = np.array([0.6, 0.4, 0.15])   # pyramid
    if pop_size > 4:
        population[4] = np.array([0.15, 0.4, 0.6])  # inverse pyramid
    
    def eval_fitness(g):
        return _flow_fitness(sparse_model, g, n_classes)
    
    best_fitness = -float('inf')
    best_genotype = population[0].copy()
    
    for gen in range(generations):
        fitnesses = [eval_fitness(g) for g in population]
        
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_genotype = population[gen_best_idx].copy()
        
        # Tournament selection
        mu = max(2, pop_size // 3)
        parents = []
        for _ in range(mu):
            candidates = rnd.sample(range(len(population)), min(3, len(population)))
            winner = max(candidates, key=lambda i: fitnesses[i])
            parents.append(population[winner].copy())
        
        # Gaussian mutation with adaptive sigma
        offspring = [best_genotype.copy()]  # elitism
        sigma = 0.12 * (1.0 - 0.5 * gen / generations)
        while len(offspring) < pop_size:
            parent = rnd.choice(parents)
            child = parent + np.random.randn(n_layers) * sigma
            child = np.clip(child, min_ratio, max_ratio)
            offspring.append(child)
        
        population = offspring
    
    return best_genotype, best_fitness


# ═══════════════════════════════════════════
#  Main conversion pipeline
# ═══════════════════════════════════════════

def _build_compact_from_ratios(sparse_model, ratios, n_classes):
    """Build compact model from per-layer keep-ratios using GFCS merge."""
    from ex09_lib.core import CompactMLP
    
    layers = [sparse_model.fc1, sparse_model.fc2, sparse_model.fc3, sparse_model.fc4]
    
    layer_stats = []
    for i, (layer, ratio) in enumerate(zip(layers[:-1], ratios)):
        W = layer.weight.data
        n_neurons = W.shape[0]
        
        incoming_alive = (W.abs().sum(dim=1) > 1e-8)
        if i + 1 < len(layers):
            outgoing_alive = (layers[i+1].weight.data.abs().sum(dim=0) > 1e-8)
            alive = incoming_alive & outgoing_alive
        else:
            alive = incoming_alive
        n_active = alive.sum().item()
        target_k = max(4, int(n_active * ratio))
        
        layer_stats.append({
            'n_neurons': n_neurons,
            'n_active': n_active,
            'target_k': target_k,
        })
    
    # Apply GFCS merge layer by layer
    merged_weights = []
    for i, (layer, stats) in enumerate(zip(layers[:-1], layer_stats)):
        W_in = layer.weight.data
        b_in = layer.bias.data if layer.bias is not None else None
        W_out = layers[i + 1].weight.data
        
        phi = _compute_flow_importance(W_in, W_out)
        affinity = _compute_flow_affinity(W_in, W_out)
        new_W_in, new_b_in, new_W_out, active_idx = _greedy_flow_merge(
            W_in, b_in, W_out, phi, affinity, stats['target_k']
        )
        merged_weights.append((new_W_in, new_b_in, new_W_out, active_idx))
    
    # Build compact model
    hiddens = [mw[0].shape[0] for mw in merged_weights]
    compact = CompactMLP(
        input_dim=sparse_model.input_dim,
        hiddens=tuple(hiddens),
        n_classes=n_classes,
    )
    
    # Cascade weight copy
    with torch.no_grad():
        for i, (new_W_in, new_b_in, new_W_out, active_idx) in enumerate(merged_weights):
            tgt_layer = compact.net[i * 2]
            if i == 0:
                tgt_layer.weight.data.copy_(new_W_in)
            else:
                prev_active = merged_weights[i - 1][3]
                W_remapped = new_W_in[:, prev_active]
                tgt_layer.weight.data.copy_(W_remapped)
            if new_b_in is not None:
                tgt_layer.bias.data.copy_(new_b_in)
        
        last_active = merged_weights[-1][3]
        output_layer = layers[-1]
        W_final = output_layer.weight.data[:, last_active]
        tgt_output = compact.net[-1]
        tgt_output.weight.data.copy_(W_final)
        if output_layer.bias is not None:
            tgt_output.bias.data.copy_(output_layer.bias.data)
    
    original_params = sparse_model.count_params()
    compact_params = compact.count_params()
    
    info = {
        'method': 'gfcs',
        'original_hiddens': [l.weight.shape[0] for l in layers[:-1]],
        'merged_hiddens': hiddens,
        'layer_stats': layer_stats,
        'original_params': original_params,
        'compact_params': compact_params,
        'compression': original_params / max(compact_params, 1),
    }
    return compact, info


def gfcs_convert(sparse_model, n_classes=2, compression_target=None,
                 pop_size=20, generations=30,
                 min_ratio=0.1, max_ratio=0.8,
                 use_evolution=True):
    """
    GFCS: Gradient-Flow Connectivity Synthesis
    Full sparse→dense conversion with evolutionary architecture search.
    
    Pipeline:
      1. (μ+λ)-ES evolves optimal per-layer merge ratios (ZERO-COST fitness)
      2. For each layer: compute flow importance → flow affinity → greedy merge
      3. Build compact dense model with cascade weight copy
    
    Args:
        sparse_model: Pruned SimpleMLP with zero weights
        n_classes: Number of output classes
        compression_target: If set, overrides EA with fixed ratio
        pop_size: EA population size
        generations: EA generations
        min_ratio: Minimum fraction of neurons to keep
        max_ratio: Maximum fraction of neurons to keep
        use_evolution: If True, use EA; if False, use heuristic density-based ratio
    
    Returns:
        compact_model: Dense CompactMLP
        info: dict with merge statistics
    """
    if compression_target is not None:
        # Fixed ratio for all layers
        ratios = [compression_target] * 3
        compact, info = _build_compact_from_ratios(sparse_model, ratios, n_classes)
        info['evolution'] = False
        info['ratios'] = ratios
        return compact, info
    
    if use_evolution:
        # Evolutionary search for optimal per-layer ratios
        best_ratios, best_fitness = _evolve_ratios(
            sparse_model, n_classes,
            pop_size=pop_size, generations=generations,
            min_ratio=min_ratio, max_ratio=max_ratio,
        )
        
        compact, info = _build_compact_from_ratios(sparse_model, best_ratios, n_classes)
        info['evolution'] = True
        info['ratios'] = best_ratios.tolist()
        info['ea_fitness'] = best_fitness
        info['ea_config'] = {'pop_size': pop_size, 'generations': generations}
        return compact, info
    else:
        # Heuristic fallback
        layers = [sparse_model.fc1, sparse_model.fc2, sparse_model.fc3, sparse_model.fc4]
        ratios = []
        for i, layer in enumerate(layers[:-1]):
            W = layer.weight.data
            n_neurons = W.shape[0]
            incoming_alive = (W.abs().sum(dim=1) > 1e-8)
            if i + 1 < len(layers):
                outgoing_alive = (layers[i+1].weight.data.abs().sum(dim=0) > 1e-8)
                alive = incoming_alive & outgoing_alive
            else:
                alive = incoming_alive
            n_active = alive.sum().item()
            density = max(n_active / n_neurons, 0.05)
            ratio = max(min_ratio, min(max_ratio, density))
            ratios.append(ratio)
        
        compact, info = _build_compact_from_ratios(sparse_model, ratios, n_classes)
        info['evolution'] = False
        info['ratios'] = ratios
        return compact, info
