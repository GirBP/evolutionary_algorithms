# methods/magnitude.py — Global Magnitude Pruning (Multi-Fidelity)
from methods import register
from or08_01 import (create_model, apply_global, train_finetune_micro,
                     evaluate_full, set_seed, check_mask_connectivity)


@register('magnitude', 'Magnitude', '#1f77b4')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)
    apply_global(model, sp)
    # §1 Topological filter
    if not check_mask_connectivity(model):
        return {'F1': 0.333}
    # §2 Micro-finetuning with OneCycleLR
    train_finetune_micro(model, train_dl, config['finetune_batches'])
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
