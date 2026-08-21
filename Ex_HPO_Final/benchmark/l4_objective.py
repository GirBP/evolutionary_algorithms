"""
L4 Objective: реальне навчання мікро-нейромереж для HPO валідації.

Кожен eval = 5 epochs навчання PyTorch моделі на CPU (~2–5 сек).
Простір пошуку [0,1]^dim декодується в гіперпараметри.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import os

from benchmark.l4_architectures import ARCHITECTURES, SequentialMLP


# ============================================================
# Datasets
# ============================================================
_DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data', 'l4')


def _get_fashion_mini():
    # Fashion MNIST 2k subset, resized to 14x14
    transform = transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    full_train = datasets.FashionMNIST(_DATA_ROOT, train=True, download=True, transform=transform)
    full_val = datasets.FashionMNIST(_DATA_ROOT, train=False, download=True, transform=transform)
    
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(full_train))
    train_ds = Subset(full_train, indices[:2000])
    val_ds = Subset(full_train, indices[2000:2500])
    test_ds = full_val
    return train_ds, val_ds, test_ds, 1, 14, 10  # channels, img_size, n_classes


def _get_digits():
    from sklearn.datasets import load_digits
    from torch.utils.data import TensorDataset
    digits = load_digits()
    X = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) # [1797, 1, 8, 8]
    X = (X / 16.0 - 0.5) * 2.0
    y = torch.tensor(digits.target, dtype=torch.long)
    
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    train_idx, val_idx, test_idx = indices[:1000], indices[1000:1400], indices[1400:]
    
    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])
    test_ds = TensorDataset(X[test_idx], y[test_idx])
    return train_ds, val_ds, test_ds, 1, 8, 10


DATASETS = {
    'fashion_mini': _get_fashion_mini,
    'digits': _get_digits,
}

# ============================================================
# HP Decode ([0,1]^dim → hyperparameters)
# ============================================================

# Act function mapping
_ACTS = ['relu', 'gelu', 'silu']
_OPTS = ['adam', 'sgd']

def _decode_common(x):
    """Decode common hyperparameters from x[0:4]."""
    lr = 10 ** (x[0] * (-1 - (-4)) + (-4))       # [1e-4, 0.1]
    batch_idx = int(x[1] * 3.99)
    batch_size = [16, 32, 64, 128][batch_idx]
    weight_decay = 10 ** (x[2] * (-2 - (-6)) + (-6))  # [1e-6, 1e-2]
    act = _ACTS[int(x[3] * 2.99)]
    return lr, batch_size, weight_decay, act


def _decode_sequential(x):
    """dim=7: lr, batch, wd, act, h1, h2, dropout"""
    lr, bs, wd, act = _decode_common(x)
    h1 = int(x[4] * (256 - 32) + 32)
    h2 = int(x[5] * (128 - 16) + 16)
    dropout = x[6] * 0.5
    return {'lr': lr, 'batch_size': bs, 'weight_decay': wd, 'act': act,
            'h1': h1, 'h2': h2, 'dropout': dropout}


def _decode_residual(x):
    """dim=8: lr, batch, wd, act, c1, c2, skip_scale, optimizer"""
    lr, bs, wd, act = _decode_common(x)
    c1 = int(x[4] * (32 - 8) + 8)
    c2 = int(x[5] * (64 - 16) + 16)
    skip_scale = x[6] * 1.5 + 0.5   # [0.5, 2.0]
    opt = _OPTS[int(x[7] * 1.99)]
    return {'lr': lr, 'batch_size': bs, 'weight_decay': wd, 'act': act,
            'c1': c1, 'c2': c2, 'skip_scale': skip_scale, 'optimizer': opt}


def _decode_dense(x):
    """dim=7: lr, batch, wd, act, c0, growth_rate, n_dense_layers"""
    lr, bs, wd, act = _decode_common(x)
    c0 = int(x[4] * (24 - 8) + 8)
    growth = int(x[5] * (12 - 4) + 4)
    n_dense = int(x[6] * (5 - 2) + 2)
    return {'lr': lr, 'batch_size': bs, 'weight_decay': wd, 'act': act,
            'c0': c0, 'growth': growth, 'n_dense': n_dense}


def _decode_bottleneck(x):
    """dim=7: lr, batch, wd, act, c_in, bottleneck_ratio, n_blocks"""
    lr, bs, wd, act = _decode_common(x)
    c_in = int(x[4] * (48 - 16) + 16)
    bn_ratio = x[5] * (0.5 - 0.1) + 0.1
    n_blocks = int(x[6] * (4 - 1) + 1)
    return {'lr': lr, 'batch_size': bs, 'weight_decay': wd, 'act': act,
            'c_in': c_in, 'bottleneck_ratio': bn_ratio, 'n_blocks': n_blocks}


def _decode_multibranch(x):
    """dim=8: lr, batch, wd, act, c0, c_1x1, c_3x3, c_5x5"""
    lr, bs, wd, act = _decode_common(x)
    c0 = int(x[4] * (32 - 8) + 8)
    c_1x1 = int(x[5] * (16 - 4) + 4)
    c_3x3 = int(x[6] * (16 - 4) + 4)
    c_5x5 = int(x[7] * (8 - 2) + 2)
    return {'lr': lr, 'batch_size': bs, 'weight_decay': wd, 'act': act,
            'c0': c0, 'c_1x1': c_1x1, 'c_3x3': c_3x3, 'c_5x5': c_5x5}


DECODERS = {
    'sequential': (_decode_sequential, 7),
    'residual': (_decode_residual, 8),
    'dense': (_decode_dense, 7),
    'bottleneck': (_decode_bottleneck, 7),
    'multibranch': (_decode_multibranch, 8),
}


# ============================================================
# Build Model
# ============================================================
def _build_model(arch_name, hparams, in_channels, img_size, n_classes):
    """Instantiate a model from decoded hparams."""
    cls = ARCHITECTURES[arch_name]

    if arch_name == 'sequential':
        input_dim = in_channels * img_size * img_size
        return cls(input_dim, n_classes, h1=hparams['h1'], h2=hparams['h2'],
                   dropout=hparams['dropout'], act=hparams['act'])
    elif arch_name == 'residual':
        return cls(in_channels, n_classes, c1=hparams['c1'], c2=hparams['c2'],
                   act=hparams['act'], skip_scale=hparams['skip_scale'])
    elif arch_name == 'dense':
        return cls(in_channels, n_classes, c0=hparams['c0'], growth=hparams['growth'],
                   n_dense=hparams['n_dense'], act=hparams['act'])
    elif arch_name == 'bottleneck':
        return cls(in_channels, n_classes, c_in=hparams['c_in'],
                   bottleneck_ratio=hparams['bottleneck_ratio'],
                   n_blocks=hparams['n_blocks'], act=hparams['act'])
    elif arch_name == 'multibranch':
        return cls(in_channels, n_classes, c0=hparams['c0'],
                   c_1x1=hparams['c_1x1'], c_3x3=hparams['c_3x3'],
                   c_5x5=hparams['c_5x5'], act=hparams['act'])


# ============================================================
# Train + Eval (single objective call)
# ============================================================
def _train_and_eval(model, train_loader, val_loader, lr, weight_decay, optimizer_type='adam', epochs=5):
    """Train model for N epochs and return validation loss (1 - accuracy)."""
    device = torch.device('cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_type == 'sgd':
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    import time
    from benchmark.profiler import get_anchor_time
    
    # Ensure PyTorch does not secretly use background threads that thread_time_ns would miss
    torch.set_num_threads(1)
    
    t0_train = time.thread_time_ns()
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            opt.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            opt.step()
    t1_train = time.thread_time_ns()
    anchor = max(1.0, get_anchor_time())
    train_rcu = (t1_train - t0_train) / anchor

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x).argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total
    return 1.0 - accuracy, train_rcu


# ============================================================
# Public Interface (compatible with run_method.py)
# ============================================================
def get_l4_objective(task_name: str):
    """
    task_name format: '{arch}__{dataset}', e.g. 'residual__fashion_mnist'
    Returns: (dim, make_obj_fn)
    """
    arch_name, dataset_name = task_name.split('__', 1)

    decode_fn, dim = DECODERS[arch_name]
    load_data = DATASETS[dataset_name]

    def make_obj():
        train_ds, val_ds, test_ds, in_channels, img_size, n_classes = load_data()

        def decode(x):
            x = np.clip(x, 0, 1)
            hparams = decode_fn(x)
            return hparams

        def get_test_metrics(x):
            try:
                hparams = decode(x)
                bs = hparams.pop('batch_size')
                opt_type = hparams.pop('optimizer', 'adam')
                lr = hparams.pop('lr')
                wd = hparams.pop('weight_decay')
                
                train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)
                test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
                
                model = _build_model(arch_name, hparams.copy(), in_channels, img_size, n_classes)
                n_params = sum(p.numel() for p in model.parameters())
                
                device = torch.device('cpu')
                model = model.to(device)
                criterion = nn.CrossEntropyLoss()
                
                if opt_type == 'sgd':
                    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
                else:
                    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
                model.train()
                for _ in range(5):
                    for batch_x, batch_y in train_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        opt.zero_grad()
                        loss = criterion(model(batch_x), batch_y)
                        loss.backward()
                        opt.step()
                        
                model.eval()
                correct = 0
                total = 0
                import time
                t0_inf = time.thread_time_ns()
                with torch.no_grad():
                    for batch_x, batch_y in test_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        pred = model(batch_x).argmax(dim=1)
                        correct += (pred == batch_y).sum().item()
                        total += batch_y.size(0)
                t1_inf = time.thread_time_ns()
                
                test_acc = correct / max(1, total)
                inference_time_ms = ((t1_inf - t0_inf) / 1e6) / max(1, total)
                
                return {
                    'final_test_error': 1.0 - test_acc,
                    'n_params': n_params,
                    'inference_time_ms_per_sample': inference_time_ms
                }
            except Exception as e:
                print("L4 test error:", e)
                return {}

        def objective(x):
            hparams = decode(x)

            bs = hparams.pop('batch_size')
            opt_type = hparams.pop('optimizer', 'adam')
            lr = hparams.pop('lr')
            wd = hparams.pop('weight_decay')

            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                      num_workers=0, pin_memory=False)
            val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                                    num_workers=0, pin_memory=False)

            model = _build_model(arch_name, hparams, in_channels, img_size, n_classes)

            import time
            t0_eval = time.thread_time_ns()
            try:
                val_loss, train_rcu = _train_and_eval(model, train_loader, val_loader,
                                           lr, wd, opt_type, epochs=5)
                # Store the training time for RCU profiling
                objective.last_train_time = train_rcu
            except Exception as e:
                print(f"  L4 eval error: {e}")
                val_loss = 1.0  # penalty
                objective.last_train_time = 0.0

            t1_eval = time.thread_time_ns()
            # Tell the outer HPO profiler how much time to subtract from pure HPO logic RCU
            objective.total_time_ns = getattr(objective, 'total_time_ns', 0) + (t1_eval - t0_eval)
            return val_loss

        objective.last_train_time = 0.0
        objective.total_time_ns = 0
        objective.decode = decode
        objective.get_test_metrics = get_test_metrics
        return objective

    return dim, make_obj
