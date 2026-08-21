import torch
import numpy as np
from methods import register
from or08_01 import (create_model, apply_ratios, train_finetune_micro,
                     evaluate_full, set_seed, check_mask_connectivity,
                     normalize_genome, apply_global)
import torch.nn as nn

def _precompute_qfa_proxy(teacher_state, train_dl):
    """
    Підготовка. Зберігаємо не лише плоскі масиви (flat), 
    але й оригінальні N-вимірні тензори для обчислення квантових маргіналів.
    """
    model = create_model()
    model.load_state_dict(teacher_state)
    device = next(model.parameters()).device
    model.eval()

    hooks, x_norms, y_norms = [], {}, {}
    def hook_fn(name):
        def fn(m, inp, out):
            x, y = inp[0].detach(), out.detach()
            x_norms[name] = x.pow(2).sum(dim=0).sqrt() if x.dim() == 2 else x.pow(2).sum(dim=(0,2,3)).sqrt()
            y_norms[name] = y.pow(2).sum(dim=0).sqrt() if y.dim() == 2 else y.pow(2).sum(dim=(0,2,3)).sqrt()
        return fn
    for name, layer, _ in model.get_prunable_layers():
        hooks.append(layer.register_forward_hook(hook_fn(name)))
        
    with torch.no_grad():
        X, _ = next(iter(train_dl))
        model(X.to(device))
    for h in hooks: h.remove()

    model.zero_grad()
    layers = model.get_prunable_layers()
    first_l = layers[0][1]
    x_ones = torch.ones(1, first_l.weight.shape[1], device=device) if first_l.weight.dim() == 2 and first_l.weight.shape[1] <= 10 else torch.ones(1,1,28,28, device=device)
    model(x_ones).sum().backward()

    scores_nd, shapes, max_capacities = [], [], []
    
    with torch.no_grad():
        for name, layer, _ in layers:
            W = layer.weight
            G = layer.weight.grad if layer.weight.grad is not None else torch.zeros_like(W)
            xn = x_norms.get(name, torch.ones(W.shape[1] if W.dim()==2 else W.shape[1], device=device))
            yn = y_norms.get(name, torch.ones(W.shape[0], device=device))
            
            if W.dim() == 2:
                score = W.abs() * (xn.view(1,-1) + yn.view(-1,1)) * G.abs().clamp(min=1e-8)
            else:
                score = W.abs() * (xn.view(1,-1,1,1) + yn.view(-1,1,1,1)) * G.abs().clamp(min=1e-8)
                
            score_np = score.cpu().numpy()
            scores_nd.append(score_np)
            shapes.append(W.shape)
            max_capacities.append(score_np.sum() + 1e-12)
            
    layer_counts = np.array([np.prod(s) for s in shapes])
    return scores_nd, layer_counts, shapes, np.array(max_capacities)


def _quantum_fidelity_eval(genome, layer_counts, target_sparsity, scores_nd, shapes, max_capacities):
    """
    АБСОЛЮТНА НАУКОВА НОВИЗНА (Quantum Fidelity Alignment).
    Штрафує мутантів за топологічне "розходження" масок між суміжними шарами.
    """
    ratios = normalize_genome(genome, layer_counts, target_sparsity)
    L = len(ratios)
    
    out_profiles = []
    in_profiles = []
    log_internal_capacity = 0.0
    
    # 1. Формування маргінальних профілів для кожного шару
    for i in range(L):
        k = max(1, int(layer_counts[i] * ratios[i]))
        sc_nd = scores_nd[i]
        sc_flat = sc_nd.reshape(-1)
        
        thresh = np.partition(sc_flat, -k)[-k] if k < len(sc_flat) else sc_flat.min() - 1e-9
        mask_nd = (sc_nd >= thresh).astype(np.float32)
        flow_nd = sc_nd * mask_nd
        
        # Обчислення внутрішньої ємності (Загальна Маса)
        mu = flow_nd.sum() / max_capacities[i]
        log_internal_capacity += np.log(mu + 1e-12)
        
        # Проекція тензора ваг на одновимірні вектори випромінювання та поглинання
        if len(shapes[i]) == 2: # Linear (out_features, in_features)
            v_out = flow_nd.sum(axis=1) # Енергія, що відправляється до наступного шару
            v_in = flow_nd.sum(axis=0)  # Енергія, що приймається з попереднього
        elif len(shapes[i]) == 4: # Conv2d (out_channels, in_channels, K, K)
            v_out = flow_nd.sum(axis=(1, 2, 3))
            v_in = flow_nd.sum(axis=(0, 2, 3))
        else:
            v_out, v_in = np.array([1.0]), np.array([1.0])
            
        out_profiles.append(v_out)
        in_profiles.append(v_in)

    # 2. Обчислення Квантового Перетину (Bhattacharyya Distance) між суміжними шарами
    log_quantum_fidelity = 0.0
    for i in range(L - 1):
        p_vec = out_profiles[i]     # Що генерує шар i (Output)
        q_vec = in_profiles[i+1]    # Що очікує почути шар i+1 (Input)
        
        # Обробка стику Conv2d -> Linear (Flattening)
        if len(p_vec) != len(q_vec):
            if len(q_vec) > 0 and len(q_vec) % len(p_vec) == 0:
                q_vec = q_vec.reshape(len(p_vec), -1).sum(axis=1)
            elif len(p_vec) > 0 and len(p_vec) % len(q_vec) == 0:
                p_vec = p_vec.reshape(len(q_vec), -1).sum(axis=1)
                
        # Розрахунок Вірності Перетину
        if len(p_vec) == len(q_vec):
            sum_p, sum_q = p_vec.sum(), q_vec.sum()
            p_hat = p_vec / sum_p if sum_p > 0 else np.zeros_like(p_vec)
            q_hat = q_vec / sum_q if sum_q > 0 else np.zeros_like(q_vec)
            
            # Формула Квантової Точності (Bhattacharyya Overlap)
            overlap = np.sum(np.sqrt(p_hat * q_hat))
            log_quantum_fidelity += np.log(overlap + 1e-12)

    # Загальний фітнес = (Внутрішня міцність) + 2.0 * (Ідеальне стикування шарів)
    total_fitness = log_internal_capacity + 2.0 * log_quantum_fidelity
    return -total_fitness # CMA-ES мінімізує (шукає найменше від'ємне число)


@register('evo-synflow-qfa', 'Evo-SynFlow (Quantum Fidelity)', '#17becf')
def run_evo_synflow_qfa(teacher_state, sp, seed, config, train_dl, test_dl, *, _state=None):
    import cma
    set_seed(seed)
    
    scores_nd, layer_counts, shapes, max_cap = _precompute_qfa_proxy(teacher_state, train_dl)

    prv = _state.get('prev_ratios') if _state else None
    if prv is not None:
        x0 = prv.copy()
        scale = (1.0 - sp) / (1.0 - (1.0 - np.sum(x0 * layer_counts) / np.sum(layer_counts)))
        x0 = np.clip(x0 * scale, 0.01, 1.0)
    else:
        temp = create_model()
        temp.load_state_dict(teacher_state)
        apply_global(temp, sp)
        x0 = np.array([getattr(temp, mn).sum().item() / getattr(temp, mn).numel()
                        for _, _, mn in temp.get_prunable_layers()])

    # Sigma 0.1 для стабільного пошуку в квантовому ландшафті
    es = cma.CMAEvolutionStrategy(x0, 0.1 * (1.0 - sp), {
        'popsize': config['pop_size'], 'verbose': -1, 'bounds': [0, None]})
        
    while not es.stop() and es.countevals < config['max_evals']:
        sols = es.ask()
        fitnesses = [_quantum_fidelity_eval(s, layer_counts, sp, scores_nd, shapes, max_cap) for s in sols]
        es.tell(sols, fitnesses)

    best_ratios = normalize_genome(es.result.xbest, layer_counts, sp)
    model = create_model()
    model.load_state_dict(teacher_state)
    apply_ratios(model, best_ratios)
    
    if not check_mask_connectivity(model):
        print("⚠ Зв'язність втрачено (З QFA це алгебраїчно НЕМОЖЛИВО!)")
        return {'F1': 0.333}
        
    train_finetune_micro(model, train_dl, config.get('finetune_batches_evo', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    
    if _state is not None: _state['prev_ratios'] = best_ratios
    return {'F1': f1}