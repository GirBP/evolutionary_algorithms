"""
EvoMerge: Evolutionary Topological Merging for Sparse-to-Dense Conversion
==========================================================================
Author: Original method developed for this dissertation.

Inspirations (cited, NOT copied):
  - Singh & Alistarh (2020): "Model Fusion via Optimal Transport" — merging
    neurons of DIFFERENT models. We merge within a SINGLE pruned model.
  - Sun et al. (2023): WANDA — activation-based importance for pruning.
    We use activation SIMILARITY for merging, not pruning.

Key novelty:
  1. Functional similarity via activation fingerprints, not weight magnitude
  2. Evolutionary search of per-layer merge ratios (not fixed k-means)
  3. Zero-cost fitness: reconstruction error on calibration batch, no backprop
  4. Activation-norm weighted weight synthesis preserving dominant features

Mathematical Framework:
  Given sparse network f_S with mask M at sparsity s:
  
  Step 1: Compute activation fingerprint a_j^l for each active neuron j
          a_j^l = [f_j^l(x_1), ..., f_j^l(x_B)] on calibration batch
  
  Step 2: Build similarity graph G_l = (V_l, E_l)
          edge weight w_ij = |corr(a_i^l, a_j^l)|
  
  Step 3: Evolve merging plan g = (k_1, k_2, ..., k_L)
          k_l = number of representative neurons in layer l
  
  Step 4: For each cluster C_m, synthesize merged neuron:
          w_in^m  = Σ_{j∈C_m} α_j · w_in^j,  α_j = ||a_j|| / Σ||a_k||
          w_out^m = Σ_{j∈C_m} w_out^j  (preserves linear contribution)
          b^m     = Σ_{j∈C_m} α_j · b_j
  
  Step 5: Zero-cost fitness:
          F(g) = -||A_l - Â_l||_F / ||A_l||_F - λ·(N'/N)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from collections import defaultdict


def _compute_activation_fingerprints(model, dataloader, max_batches=5):
    """
    Step 1: Compute activation fingerprints for each neuron.
    
    For each hidden layer, collect the activation matrix A_l ∈ R^{B×n_l}
    where B is total samples and n_l is number of neurons in layer l.
    
    Returns dict: layer_name → activation matrix (numpy).
    """
    activations = {}
    hooks = []
    
    def make_hook(name):
        def hook_fn(module, inp, out):
            # out: [batch, features] for Linear, after ReLU
            activations.setdefault(name, []).append(out.detach().cpu())
        return hook_fn
    
    # Hook into ReLU outputs (= hidden representations)
    layers = []
    if hasattr(model, 'fc1'):
        # SimpleMLP structure
        layer_pairs = [
            ('layer1', model.fc1),
            ('layer2', model.fc2),
            ('layer3', model.fc3),
        ]
        for name, layer in layer_pairs:
            hooks.append(layer.register_forward_hook(make_hook(name)))
            layers.append((name, layer))
    
    model.eval()
    with torch.no_grad():
        for i, (X, _) in enumerate(dataloader):
            model(X)
            if i >= max_batches - 1:
                break
    
    for h in hooks:
        h.remove()
    
    # Concatenate and apply ReLU (since hooks capture pre-activation for Linear)
    result = {}
    for name, acts in activations.items():
        A = torch.cat(acts, dim=0)
        A = F.relu(A)  # post-ReLU activations
        result[name] = A.numpy()
    
    return result, layers


def _compute_similarity_matrix(activations):
    """
    Step 2: Compute pairwise neuron similarity using Pearson correlation.
    
    sim(i,j) = |corr(a_i, a_j)|
    
    Returns: correlation matrix C ∈ R^{n×n} where C_ij ∈ [0, 1]
    """
    A = activations  # [B, n]
    n = A.shape[1]
    
    if n <= 1:
        return np.ones((n, n))
    
    # Standardize columns
    A_centered = A - A.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(A_centered, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    A_norm = A_centered / norms
    
    # Correlation matrix
    corr = A_norm.T @ A_norm / max(A.shape[0] - 1, 1)
    return np.abs(corr)


def _precompute_linkages(sim_matrices):
    """
    Precompute hierarchical linkage for each layer (expensive, do once).
    Returns dict: layer_name → (linkage_matrix_Z, n_neurons)
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    
    linkages = {}
    for name, sim_matrix in sim_matrices.items():
        n = sim_matrix.shape[0]
        if n <= 1:
            linkages[name] = (None, n)
            continue
        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = np.maximum(dist_matrix, 0)
        dist_condensed = squareform(dist_matrix, checks=False)
        Z = linkage(dist_condensed, method='average')
        linkages[name] = (Z, n)
    return linkages


def _cluster_neurons_cached(Z, n, k):
    """
    Cluster n neurons into k groups using precomputed linkage Z.
    Returns: list of lists, each containing neuron indices in a cluster.
    """
    from scipy.cluster.hierarchy import fcluster
    
    k = max(1, min(k, n))
    if k >= n:
        return [[i] for i in range(n)]
    if Z is None:
        return [[i] for i in range(n)]
    
    labels = fcluster(Z, t=k, criterion='maxclust')
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(i)
    return list(clusters.values())


def _synthesize_merged_weights(layer, next_layer, clusters, activations):
    """
    Step 4: Synthesize weights for merged neurons.
    
    For cluster C_m = {j_1, j_2, ..., j_p}:
      - Importance α_j = ||a_j||₂ / Σ||a_k||₂  (activation norm weighting)
      - Incoming:  w_in^m  = Σ α_j · w_in^j   (weighted average — preserves dominant)
      - Outgoing:  w_out^m = Σ w_out^j          (sum — preserves total linear contribution)
      - Bias:      b^m     = Σ α_j · b^j        (weighted average)
    
    This is NOT simple averaging — the activation-norm weighting ensures that
    neurons with stronger activations (= more informative features) contribute
    more to the merged representation.
    """
    W_in = layer.weight.data   # [n_out, n_in] — weights INTO this layer
    b_in = layer.bias.data if layer.bias is not None else None
    W_out = next_layer.weight.data  # [n_next, n_out] — weights FROM this layer
    
    k = len(clusters)
    n_in = W_in.shape[1]
    n_next = W_out.shape[0]
    
    new_W_in = torch.zeros(k, n_in)
    new_b_in = torch.zeros(k) if b_in is not None else None
    new_W_out = torch.zeros(n_next, k)
    
    for m, cluster in enumerate(clusters):
        if len(cluster) == 1:
            j = cluster[0]
            new_W_in[m] = W_in[j]
            if new_b_in is not None:
                new_b_in[m] = b_in[j]
            new_W_out[:, m] = W_out[:, j]
        else:
            # Compute activation-norm importance weights
            act_norms = np.array([np.linalg.norm(activations[:, j]) for j in cluster])
            total_norm = act_norms.sum()
            if total_norm < 1e-8:
                alphas = np.ones(len(cluster)) / len(cluster)
            else:
                alphas = act_norms / total_norm
            
            # Weighted incoming weights
            for i, j in enumerate(cluster):
                new_W_in[m] += alphas[i] * W_in[j]
                if new_b_in is not None:
                    new_b_in[m] += alphas[i] * b_in[j]
            
            # Sum outgoing weights (preserves linear contribution)
            for j in cluster:
                new_W_out[:, m] += W_out[:, j]
    
    return new_W_in, new_b_in, new_W_out


def _evaluate_merging_quality(model, merge_plan, act_fingerprints, layers, dataloader):
    """
    Step 5: Zero-cost fitness evaluation.
    
    F(g) = -Σ_l ||A_l - Â_l||_F / ||A_l||_F  -  λ · (N' / N)
    
    Where:
      A_l  = original activation matrix of layer l
      Â_l  = activation matrix after merging layer l
      N'   = parameter count of merged network
      N    = parameter count of original
      λ    = compression reward coefficient
    
    Higher is better. Zero-cost because we compute Â_l analytically
    from the merge operation, no forward pass needed.
    """
    total_reconstruction_error = 0.0
    total_params_merged = 0
    total_params_original = 0
    lam = 0.1  # compression reward
    
    for layer_name, layer in layers:
        if layer_name not in act_fingerprints or layer_name not in merge_plan:
            continue
        
        A = act_fingerprints[layer_name]  # [B, n]
        clusters = merge_plan[layer_name]
        k = len(clusters)
        n = A.shape[1]
        
        if k >= n:
            continue
        
        # Compute merged activations analytically
        A_merged = np.zeros((A.shape[0], k))
        for m, cluster in enumerate(clusters):
            act_norms = np.array([np.linalg.norm(A[:, j]) for j in cluster])
            total = act_norms.sum()
            if total < 1e-8:
                alphas = np.ones(len(cluster)) / len(cluster)
            else:
                alphas = act_norms / total
            
            for i, j in enumerate(cluster):
                A_merged[:, m] += alphas[i] * A[:, j]
        
        # Reconstruction error: how well can we reconstruct original from merged?
        # Use least-squares: A ≈ A_merged @ W_reconstruct
        # Error = ||A - A_merged @ (A_merged^+ @ A)||_F / ||A||_F
        A_norm = np.linalg.norm(A)
        if A_norm < 1e-8:
            continue
        
        try:
            W_rec, _, _, _ = np.linalg.lstsq(A_merged, A, rcond=None)
            A_reconstructed = A_merged @ W_rec
            error = np.linalg.norm(A - A_reconstructed) / A_norm
        except np.linalg.LinAlgError:
            error = 1.0
        
        total_reconstruction_error += error
        total_params_merged += k
        total_params_original += n
    
    compression_ratio = total_params_merged / max(total_params_original, 1)
    
    # Fitness: minimize reconstruction error, reward compression
    fitness = -total_reconstruction_error - lam * compression_ratio
    return fitness


def _evolve_merge_plan(act_fingerprints, sim_matrices, layer_info, linkages,
                       pop_size=20, generations=30, min_ratio=0.1, max_ratio=0.8):
    """
    Step 3: Evolutionary search for optimal per-layer merge ratios.
    
    Genotype: g = (r_1, r_2, ..., r_L) where r_l ∈ [min_ratio, max_ratio]
              r_l = fraction of neurons to KEEP in layer l
    
    Evolution: (μ+λ)-ES with:
      - μ = pop_size // 3 (parents)
      - λ = pop_size (offspring)
      - Gaussian mutation σ = 0.15
      - Tournament selection
    
    Uses precomputed linkages for fast clustering.
    """
    layer_names = [name for name, _ in layer_info]
    n_layers = len(layer_names)
    
    if n_layers == 0:
        return {}
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        genotype = np.random.uniform(min_ratio, max_ratio, n_layers)
        population.append(genotype)
    
    # Also add some hand-designed individuals
    population[0] = np.full(n_layers, 0.5)   # keep 50%
    population[1] = np.full(n_layers, 0.3)   # keep 30%
    population[2] = np.full(n_layers, min_ratio)  # max compression
    
    def genotype_to_plan(g):
        plan = {}
        for i, name in enumerate(layer_names):
            n_neurons = act_fingerprints[name].shape[1]
            k = max(1, int(n_neurons * g[i]))
            Z, n = linkages[name]
            plan[name] = _cluster_neurons_cached(Z, n, k)
        return plan
    
    def eval_fitness(g):
        plan = genotype_to_plan(g)
        return _evaluate_merging_quality(None, plan, act_fingerprints, layer_info, None)
    
    best_fitness = -float('inf')
    best_genotype = population[0]
    
    for gen in range(generations):
        # Evaluate
        fitnesses = [eval_fitness(g) for g in population]
        
        # Track best
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_genotype = population[gen_best_idx].copy()
        
        # Selection: tournament (size 3)
        mu = pop_size // 3
        parents = []
        for _ in range(mu):
            candidates = random.sample(range(len(population)), min(3, len(population)))
            winner = max(candidates, key=lambda i: fitnesses[i])
            parents.append(population[winner].copy())
        
        # Offspring: mutation
        offspring = [best_genotype.copy()]  # elitism
        while len(offspring) < pop_size:
            parent = random.choice(parents)
            child = parent + np.random.randn(n_layers) * 0.15
            child = np.clip(child, min_ratio, max_ratio)
            offspring.append(child)
        
        population = offspring
    
    return genotype_to_plan(best_genotype), best_genotype, best_fitness


def evomerge(sparse_model, train_dl, n_classes=2, 
             pop_size=20, generations=30,
             min_ratio=0.1, max_ratio=0.8):
    """
    EvoMerge: Full sparse-to-dense conversion pipeline.
    
    Args:
        sparse_model: Pruned SimpleMLP
        train_dl: Training data (used only for activation fingerprinting)
        n_classes: Number of output classes
        pop_size: EA population size
        generations: EA generations
        min_ratio: Minimum fraction of neurons to keep per layer
        max_ratio: Maximum fraction of neurons to keep per layer
    
    Returns:
        compact_model: Dense CompactMLP with merged neurons
        info: Dict with merge statistics
    """
    from ex09_lib.core import CompactMLP
    
    # Step 1: Activation fingerprints
    act_fingerprints, layer_info = _compute_activation_fingerprints(
        sparse_model, train_dl, max_batches=5
    )
    
    if not act_fingerprints:
        # Fallback: if no activations collected, return copy
        return copy.deepcopy(sparse_model), {'error': 'no activations'}
    
    # Step 2: Similarity matrices
    sim_matrices = {}
    for name, A in act_fingerprints.items():
        sim_matrices[name] = _compute_similarity_matrix(A)
    
    # Step 2.5: Precompute linkages (expensive, do ONCE)
    linkages = _precompute_linkages(sim_matrices)
    
    # Step 3: Evolve optimal merge plan
    merge_plan, best_genotype, best_fitness = _evolve_merge_plan(
        act_fingerprints, sim_matrices, layer_info, linkages,
        pop_size=pop_size, generations=generations,
        min_ratio=min_ratio, max_ratio=max_ratio,
    )
    
    # Step 4: Build compact model with synthesized weights
    # Determine hidden sizes from merge plan
    all_layers = [sparse_model.fc1, sparse_model.fc2, sparse_model.fc3, sparse_model.fc4]
    layer_names = [name for name, _ in layer_info]
    
    hiddens = []
    for name in layer_names:
        if name in merge_plan:
            hiddens.append(len(merge_plan[name]))
        else:
            hiddens.append(all_layers[layer_names.index(name)].weight.shape[0])
    
    compact = CompactMLP(
        input_dim=sparse_model.input_dim,
        hiddens=tuple(hiddens),
        n_classes=n_classes,
    )
    
    # Synthesize weights via activation-norm weighted merging
    with torch.no_grad():
        for i, (name, layer) in enumerate(layer_info):
            next_layer = all_layers[i + 1]
            clusters = merge_plan[name]
            
            new_W_in, new_b_in, new_W_out = _synthesize_merged_weights(
                layer, next_layer, clusters, act_fingerprints[name]
            )
            
            # Set incoming weights for this layer
            tgt_layer = compact.net[i * 2]  # Linear (skip ReLU)
            
            # Handle input dimension for first layer or get from prev merge
            if i == 0:
                tgt_layer.weight.data.copy_(new_W_in)
            else:
                # Input comes from previous merged layer
                prev_clusters = merge_plan[layer_names[i - 1]]
                prev_k = len(prev_clusters)
                # new_W_in is [k, original_n_in], need [k, prev_k]
                # Re-index: for each input neuron that was merged, sum the weights
                W_remapped = torch.zeros(len(clusters), prev_k)
                for m_out, cluster_out in enumerate(clusters):
                    for m_in, cluster_in in enumerate(prev_clusters):
                        # Sum contributions from all original neurons in both clusters
                        val = 0.0
                        act_norms = np.array([np.linalg.norm(act_fingerprints[name][:, j]) 
                                            for j in cluster_out])
                        total = act_norms.sum()
                        alphas = act_norms / max(total, 1e-8)
                        
                        for idx_out, j_out in enumerate(cluster_out):
                            for j_in in cluster_in:
                                val += alphas[idx_out] * layer.weight.data[j_out, j_in].item()
                        W_remapped[m_out, m_in] = float(val)
                tgt_layer.weight.data.copy_(W_remapped)
            
            if new_b_in is not None:
                tgt_layer.bias.data.copy_(new_b_in)
        
        # Output layer: remap inputs from last merged layer
        last_name = layer_names[-1]
        last_clusters = merge_plan[last_name]
        output_layer = all_layers[-1]  # fc4
        tgt_output = compact.net[-1]
        
        W_out = torch.zeros(n_classes, len(last_clusters))
        for m, cluster in enumerate(last_clusters):
            for j in cluster:
                W_out[:, m] += output_layer.weight.data[:, j]
        tgt_output.weight.data.copy_(W_out)
        
        if output_layer.bias is not None:
            tgt_output.bias.data.copy_(output_layer.bias.data)
    
    info = {
        'merge_ratios': best_genotype.tolist(),
        'fitness': best_fitness,
        'original_hiddens': [l.weight.shape[0] for l in all_layers[:-1]],
        'merged_hiddens': hiddens,
        'original_params': sparse_model.count_params(),
        'compact_params': compact.count_params(),
        'compression': sparse_model.count_params() / compact.count_params(),
    }
    
    return compact, info
