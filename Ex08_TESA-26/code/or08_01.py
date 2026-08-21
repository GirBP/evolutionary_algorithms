# !pip install cma  # (IPython only; use pip install cma if needed)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time
import random
import cma
import multiprocessing
from joblib import Parallel, delayed
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader, Subset

# ==========================================
# 1. CONFIGURATION (ZERO-COST EVO)
# ==========================================
CONFIG = {
    'seeds': [42],
    'sparsities': [ 0.90, 0.95, 0.98], # Високі рівні
    'batch_size': 64,
    'epochs_pretrain': 3,
    'finetune_batches': 100, # Тільки для фінальної перевірки

    # --- EVO SETTINGS ---
    'pop_size': 12,          # Велика популяція (бо дешево)
    'max_evals': 200,        # Глибокий пошук (бо дешево)
    'device': 'cpu'
}

N_JOBS = multiprocessing.cpu_count()
print(f"Запуск Evo-SynFlow на {N_JOBS} ядрах")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def apply_research_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'figure.figsize': (20, 12), 'lines.linewidth': 2.5})

apply_research_style()
COLORS = {'Magnitude': '#1f77b4', 'Evo-SynFlow': '#2ca02c'}

# ==========================================
# 2. DATA & MODEL
# ==========================================
def get_dataloaders(seed, dataset_name='fashionmnist'):
    """Get train/val/test DataLoaders. Supports FashionMNIST and 2D synthetic datasets."""
    if dataset_name in ('moons', 'circles', 'spirals', 'blobs'):
        return _get_2d_dataloaders(seed, dataset_name)
    return _get_fashionmnist_dataloaders(seed)


def _get_2d_dataloaders(seed, dataset_name, n_samples=5000):
    """Generate 2D synthetic datasets: moons, circles, spirals."""
    from sklearn.datasets import make_moons, make_circles
    np.random.seed(seed)

    if dataset_name == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=seed)
    elif dataset_name == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=0.15, factor=0.5, random_state=seed)
    elif dataset_name == 'spirals':
        # Two interleaving spirals
        n = n_samples // 2
        theta = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * 0.3
        r = np.linspace(0.5, 3, n)
        x1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        x2 = np.column_stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)])
        X = np.vstack([x1, x2])
        y = np.hstack([np.zeros(n), np.ones(n)]).astype(int)
    elif dataset_name == 'blobs':
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2,
                          cluster_std=1.5, random_state=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Split: 70% train, 20% test (for methods), 10% holdout (final F1)
    from sklearn.model_selection import train_test_split as tts
    X_traintest, X_holdout, y_traintest, y_holdout = tts(
        X, y, test_size=0.10, random_state=seed, stratify=y)
    X_train, X_test, y_train, y_test = tts(
        X_traintest, y_traintest, test_size=2/9,
        random_state=seed, stratify=y_traintest)
    # Result: ~70% train, ~20% test, ~10% holdout

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    X_holdout_t = torch.tensor(X_holdout, dtype=torch.float32)
    y_holdout_t = torch.tensor(y_holdout, dtype=torch.long)

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    test_ds = torch.utils.data.TensorDataset(X_test_t, y_test_t)
    holdout_ds = torch.utils.data.TensorDataset(X_holdout_t, y_holdout_t)

    return (
        DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True),
        DataLoader(holdout_ds, batch_size=256, shuffle=False),
        DataLoader(test_ds, batch_size=256, shuffle=False),
    )


def _get_fashionmnist_dataloaders(seed):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    full_train = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    full_test = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    targets = full_train.targets.numpy()

    train_idx, _ = train_test_split(np.arange(len(targets)), train_size=0.3, stratify=targets, random_state=seed)
    test_targets = full_test.targets.numpy()
    test_idx, _ = train_test_split(np.arange(len(test_targets)), train_size=0.2, stratify=test_targets, random_state=seed)

    return (
        DataLoader(Subset(full_train, train_idx), batch_size=CONFIG['batch_size'], shuffle=True),
        None,
        DataLoader(Subset(full_test, test_idx), batch_size=256, shuffle=False)
    )

class CompactCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.layers = [('c1', self.conv1), ('c2', self.conv2), ('f1', self.fc1), ('f2', self.fc2)]
        for name, layer in self.layers:
            self.register_buffer(f'm_{name}', torch.ones_like(layer.weight))

    def forward(self, x):
        # Apply masks via .data (no nn.Parameter alloc) then call modules (triggers hooks)
        with torch.no_grad():
            self.conv1.weight.data.mul_(self.m_c1)
            self.conv2.weight.data.mul_(self.m_c2)
            self.fc1.weight.data.mul_(self.m_f1)
            self.fc2.weight.data.mul_(self.m_f2)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_prunable_layers(self):
        return [(name, layer, f'm_{name}') for name, layer in self.layers]


class CompactResNet(nn.Module):
    """
    9-layer residual CNN for pruning benchmarks.
    Skip connections + BatchNorm + 1x1 projection → harder pruning task.

    Architecture:
        Stem:       Conv1(1→32, 3×3) + BN
        ResBlock1:  Conv2(32→32, 3×3) + BN + Conv3(32→32, 3×3) + BN + identity skip
        Pool(2)
        ResBlock2:  Conv4(32→64, 3×3) + BN + Conv5(64→64, 3×3) + BN + Conv_skip(32→64, 1×1)
        Pool(2)
        Head:       FC1(3136→256) + FC2(256→128) + FC3(128→10)

    Prunable layers: c1, c2, c3, c4, c5, cs, f1, f2, f3 (9 layers, ~914K params)
    """
    def __init__(self):
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # ResBlock 1 (32 → 32, identity skip)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        # ResBlock 2 (32 → 64, projection skip)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv_skip = nn.Conv2d(32, 64, 1, bias=False)  # 1×1 projection
        self.bn_skip = nn.BatchNorm2d(64)
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        # Head
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # Prunable layers registry (same interface as CompactCNN)
        self.layers = [
            ('c1', self.conv1), ('c2', self.conv2), ('c3', self.conv3),
            ('c4', self.conv4), ('c5', self.conv5), ('cs', self.conv_skip),
            ('f1', self.fc1), ('f2', self.fc2), ('f3', self.fc3),
        ]
        for name, layer in self.layers:
            self.register_buffer(f'm_{name}', torch.ones_like(layer.weight))

    def forward(self, x):
        # Apply all masks via .data (no nn.Parameter alloc)
        with torch.no_grad():
            for name, layer in self.layers:
                layer.weight.data.mul_(getattr(self, f'm_{name}'))

        # Stem
        x = F.relu(self.bn1(self.conv1(x)))

        # ResBlock 1 (identity skip)
        identity = x
        out = F.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))
        x = F.relu(out + identity)
        x = self.pool(x)

        # ResBlock 2 (projection skip: 32→64 via 1×1)
        identity = self.bn_skip(self.conv_skip(x))
        out = F.relu(self.bn4(self.conv4(x)))
        out = self.bn5(self.conv5(out))
        x = F.relu(out + identity)
        x = self.pool(x)

        # Head
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_prunable_layers(self):
        return [(name, layer, f'm_{name}') for name, layer in self.layers]


class SimpleMLP(nn.Module):
    """
    3-hidden-layer MLP: input→100→100→100→output (~20K params for 2D input).
    For fast 2D classification benchmarks (moons, circles, spirals).
    """
    def __init__(self, input_dim=2, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, output_dim)

        self.layers = [
            ('f1', self.fc1), ('f2', self.fc2),
            ('f3', self.fc3), ('f4', self.fc4),
        ]
        for name, layer in self.layers:
            self.register_buffer(f'm_{name}', torch.ones_like(layer.weight))

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        for name, layer in self.layers[:-1]:
            mask = getattr(self, f'm_{name}')
            orig_w = layer.weight
            layer.weight = nn.Parameter(orig_w * mask)
            try:
                x = F.relu(layer(x))
            finally:
                layer.weight = orig_w

        # Last layer — no relu
        name, layer = self.layers[-1]
        mask = getattr(self, f'm_{name}')
        orig_w = layer.weight
        layer.weight = nn.Parameter(orig_w * mask)
        try:
            x = layer(x)
        finally:
            layer.weight = orig_w

        return x

    def get_prunable_layers(self):
        return [(name, layer, f'm_{name}') for name, layer in self.layers]


# ==========================================
# MODEL REGISTRY & FACTORY
# ==========================================
MODEL_REGISTRY = {
    'CompactCNN': CompactCNN,
    'CompactResNet': CompactResNet,
    'SimpleMLP': SimpleMLP,
}
_active_model_class = CompactCNN  # default


def set_model_class(name_or_class):
    """Set the active model class globally (called once from runner)."""
    global _active_model_class
    if isinstance(name_or_class, str):
        if name_or_class not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{name_or_class}'. Available: {list(MODEL_REGISTRY.keys())}")
        _active_model_class = MODEL_REGISTRY[name_or_class]
    else:
        _active_model_class = name_or_class


def create_model():
    """Create a new model instance using the active model class."""
    return _active_model_class()


def get_model_class():
    """Return the active model class."""
    return _active_model_class


def get_model_class_name():
    """Return the string name of the active model class (for passing to workers)."""
    for name, cls in MODEL_REGISTRY.items():
        if cls is _active_model_class:
            return name
    return 'CompactCNN'


# ==========================================
# 3. UTILS & SYNFLOW METRIC
# ==========================================
def apply_ratios(model, ratios):
    layers = model.get_prunable_layers()
    with torch.no_grad():
        for i, (name, layer, mask_name) in enumerate(layers):
            ratio = ratios[i]
            weights = layer.weight.abs()
            k = max(1, int(weights.numel() * ratio))
            threshold = torch.kthvalue(weights.view(-1), weights.numel() - k + 1).values.item()
            getattr(model, mask_name).copy_(torch.ge(weights, threshold).float())

def apply_global(model, sparsity):
    layers = model.get_prunable_layers()
    all_w = torch.cat([l.weight.abs().view(-1) for _, l, _ in layers])
    k = max(1, int(all_w.numel() * (1 - sparsity)))
    threshold = torch.kthvalue(all_w, all_w.numel() - k + 1).values.item()
    with torch.no_grad():
        for _, layer, mask_name in layers:
            getattr(model, mask_name).copy_(torch.ge(layer.weight.abs(), threshold).float())

def normalize_genome(genome, layer_counts, target_sparsity):
    genome = np.abs(genome)
    target_w = np.sum(layer_counts) * (1.0 - target_sparsity)
    current_w = np.sum(genome * layer_counts)
    L = len(layer_counts)
    # Adaptive min: each layer gets at least (1-sp)/(2*L) density
    min_ratio = max(0.005, (1.0 - target_sparsity) / (2 * L))
    return np.clip(genome * (target_w / (current_w + 1e-8)), min_ratio, 1.0)

# --- SYNFLOW CALCULATION ---
def get_synflow_score(model):
    """
    Calculates the 'Flow' of the network without data.
    Metric = sum(|weight * gradient|) given all-ones input.
    """
    # 1. Prepare
    model.eval()
    model.zero_grad()

    # 2. Dummy Input (All Ones) - No real data needed!
    # [Batch=1, Channel=1, H=28, W=28] for FashionMNIST
    x = torch.ones(1, 1, 28, 28).to(CONFIG['device'])

    # 3. Forward
    # We force gradients to flow through the network
    output = model(x)

    # 4. Loss = Sum of outputs (Linearized objective)
    loss = output.sum()
    loss.backward()

    # 5. Calculate Score
    score = 0.0
    with torch.no_grad():
        for _, layer, m_name in model.get_prunable_layers():
            if layer.weight.grad is not None:
                # SynFlow = |W * dL/dW|
                # Only count active weights
                mask = getattr(model, m_name)
                term = (layer.weight * layer.weight.grad * mask).abs()
                score += term.sum().item()
    return score

# --- WORKER ---
def worker_objective(genome, teacher_state, layer_counts, target_sparsity):
    torch.set_num_threads(1) # Fix for Colab parallelism

    # 1. Decode
    ratios = normalize_genome(genome, layer_counts, target_sparsity)

    # 2. Build Model
    model = create_model()
    model.load_state_dict(teacher_state)
    apply_ratios(model, ratios)

    # 3. Zero-Cost Eval
    score = get_synflow_score(model)

    # Return negative score (Maximize Flow)
    return -score

# ==========================================
# 4. EVO-SYNFLOW ENGINE
# ==========================================
class EvoSynFlowEngine:
    def __init__(self, teacher_state):
        self.teacher_state = teacher_state
        temp = CompactCNN()
        self.layer_counts = np.array([l.weight.numel() for _, l, _ in temp.get_prunable_layers()])
        self.dim = len(self.layer_counts)

    def run(self, sparsity):
        # Start from Uniform (Unbiased)
        x0 = np.ones(self.dim) * (1.0 - sparsity)
        sigma0 = 0.2 * (1.0 - sparsity)

        es = cma.CMAEvolutionStrategy(x0, sigma0, {
            'popsize': CONFIG['pop_size'],
            'verbose': -1,
            'bounds': [0, None]
        })

        while not es.stop() and es.countevals < CONFIG['max_evals']:
            solutions = es.ask()

            # Parallel Zero-Cost Evaluation
            # Notice: We don't pass DataLoaders!
            scores = Parallel(n_jobs=N_JOBS)(
                delayed(worker_objective)(
                    x, self.teacher_state, self.layer_counts, sparsity
                ) for x in solutions
            )
            es.tell(solutions, scores)

        return normalize_genome(es.result.xbest, self.layer_counts, sparsity)

# ==========================================
# 5. MAIN EXPERIMENT
# ==========================================
def train_finetune(model, loader, batches):
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    model.train()
    iterator = iter(loader)
    for _ in range(batches):
        try:
            X, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            X, y = next(iterator)
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        with torch.no_grad():
            for _, layer, m_name in model.get_prunable_layers():
                if layer.weight.grad is not None:
                    layer.weight.grad.mul_(getattr(model, m_name))
        opt.step()

def evaluate_full(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.extend(model(X).argmax(dim=1).numpy())
            labels.extend(y.numpy())
    return accuracy_score(labels, preds), f1_score(labels, preds, average='macro'), labels, preds

def count_dead_units(model):
    stats = []
    with torch.no_grad():
        for name, layer, mask_name in model.get_prunable_layers():
            mask = getattr(model, mask_name)
            dims = tuple(range(1, mask.ndim))
            dead = (mask.sum(dim=dims) == 0).sum().item()
            stats.append(dead)
    return stats


def check_mask_connectivity(model):
    """Check C(m) = prod_l 1[||m_l||_0 >= 1]. Returns False if graph severed."""
    for _, _, mask_name in model.get_prunable_layers():
        if getattr(model, mask_name).sum().item() == 0:
            return False
    return True


def measure_actual_sparsity(model):
    """Measure fraction of zeroed weights across all prunable layers."""
    total, zeros = 0, 0
    with torch.no_grad():
        for _, layer, mn in model.get_prunable_layers():
            mask = getattr(model, mn)
            total += mask.numel()
            zeros += (mask == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def train_finetune_micro(model, loader, batches=25, max_lr=0.05):
    """Micro-finetuning with OneCycleLR for rank-preserving proxy evaluation."""
    opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=batches)
    crit = nn.CrossEntropyLoss()
    model.train()
    iterator = iter(loader)
    for _ in range(batches):
        try:
            X, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            X, y = next(iterator)
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        with torch.no_grad():
            for _, layer, m_name in model.get_prunable_layers():
                if layer.weight.grad is not None:
                    layer.weight.grad.mul_(getattr(model, m_name))
        opt.step()
        scheduler.step()

if __name__ == '__main__':
    results = []
    focus_sparsity = 0.95
    dead_units_stats = {}
    cm_data = {}

    print(">>> STARTING EVO-SYNFLOW EXPERIMENT...")

    for seed in CONFIG['seeds']:
        set_seed(seed)
        train_dl, val_dl, test_dl = get_dataloaders(seed)

        print("  Training Teacher...")
        t0 = time.time()
        teacher = CompactCNN()
        opt = optim.SGD(teacher.parameters(), lr=0.01, momentum=0.9)
        crit = nn.CrossEntropyLoss()
        for _ in range(CONFIG['epochs_pretrain']):
            for X, y in train_dl:
                opt.zero_grad(); loss = crit(teacher(X), y); loss.backward(); opt.step()
        print(f"  Teacher Ready ({time.time()-t0:.1f}s)")

        teacher_state = copy.deepcopy(teacher.state_dict())
        evo_engine = EvoSynFlowEngine(teacher_state)

        for sp in CONFIG['sparsities']:
            print(f"    Sparsity {sp:.2f}...", end="")

            # --- 1. Magnitude ---
            m = CompactCNN(); m.load_state_dict(teacher_state)
            apply_global(m, sp)
            train_finetune(m, train_dl, CONFIG['finetune_batches'])
            acc_m, f1_m, true_y, pred_y = evaluate_full(m, test_dl)

            if sp == focus_sparsity:
                dead_units_stats['Magnitude'] = count_dead_units(m)
                cm_data['Magnitude'] = (true_y, pred_y)

            # --- 2. Evo-SynFlow ---
            t0 = time.time()
            best_ratios = evo_engine.run(sp) # <--- Zero-Cost Search
            t_search = time.time() - t0

            m = CompactCNN(); m.load_state_dict(teacher_state)
            apply_ratios(m, best_ratios)
            train_finetune(m, train_dl, CONFIG['finetune_batches'])
            acc_c, f1_c, true_y, pred_y = evaluate_full(m, test_dl)

            if sp == focus_sparsity:
                dead_units_stats['Evo-SynFlow'] = count_dead_units(m)
                cm_data['Evo-SynFlow'] = (true_y, pred_y)

            print(f" Mag F1: {f1_m:.3f} | SynFlow F1: {f1_c:.3f} | Search Time: {t_search:.3f}s")

            results.append({'Sparsity': sp, 'Method': 'Magnitude', 'Acc': acc_m, 'F1': f1_m, 'Time': 0})
            results.append({'Sparsity': sp, 'Method': 'Evo-SynFlow', 'Acc': acc_c, 'F1': f1_c, 'Time': t_search})

    # ==========================================
    # 6. VISUALIZATION
    # ==========================================
    df = pd.DataFrame(results)
    summary = df.groupby(['Sparsity', 'Method'])[['F1', 'Acc', 'Time']].mean().reset_index()
    print("\n=== FINAL RESULTS TABLE ===")
    print(summary.to_markdown(index=False, floatfmt=".4f"))

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.lineplot(data=df, x='Sparsity', y='F1', hue='Method', palette=COLORS, marker='o', linewidth=3, ax=ax1)
    ax1.set_title("F1 Score Degradation", fontweight='bold')

    ax2 = fig.add_subplot(gs[0, 1])
    sns.lineplot(data=df, x='Sparsity', y='Acc', hue='Method', palette=COLORS, marker='s', linewidth=3, ax=ax2)
    ax2.set_title("Accuracy Degradation", fontweight='bold')

    ax3 = fig.add_subplot(gs[0, 2])
    subset_focus = summary[summary['Sparsity'] == focus_sparsity]
    subset_focus = subset_focus[subset_focus['Method'].isin(['Magnitude', 'Evo-SynFlow'])]
    methods = subset_focus['Method'].values
    times = subset_focus['Time'].values
    bar_colors = ['gray' if m == 'Magnitude' else COLORS['Evo-SynFlow'] for m in methods]
    bars = ax3.bar(methods, times, color=bar_colors, edgecolor='black')
    ax3.set_title(f"Search Time Cost (at {focus_sparsity})", fontweight='bold')
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.3f}s", ha='center', va='bottom')

    ax4 = fig.add_subplot(gs[1, 0])
    if dead_units_stats:
        x = np.arange(4)
        width = 0.25
        layers_labels = ['Conv1', 'Conv2', 'FC1', 'FC2']
        if 'Magnitude' in dead_units_stats: ax4.bar(x - width/2, dead_units_stats['Magnitude'], width, label='Magnitude', color=COLORS['Magnitude'])
        if 'Evo-SynFlow' in dead_units_stats: ax4.bar(x + width/2, dead_units_stats['Evo-SynFlow'], width, label='SynFlow', color=COLORS['Evo-SynFlow'])
        ax4.set_xticks(x); ax4.set_xticklabels(layers_labels); ax4.legend()
        ax4.set_title(f"Dead Units (at {focus_sparsity})", fontweight='bold')

    ax5 = fig.add_subplot(gs[1, 1])
    if 'Magnitude' in cm_data:
        sns.heatmap(confusion_matrix(cm_data['Magnitude'][0], cm_data['Magnitude'][1]), annot=False, cmap='Blues', cbar=False, ax=ax5)
        ax5.set_title(f"Magnitude CM (at {focus_sparsity})", fontweight='bold')

    ax6 = fig.add_subplot(gs[1, 2])
    if 'Evo-SynFlow' in cm_data:
        sns.heatmap(confusion_matrix(cm_data['Evo-SynFlow'][0], cm_data['Evo-SynFlow'][1]), annot=False, cmap='Greens', cbar=False, ax=ax6)
        ax6.set_title(f"Evo-SynFlow CM (at {focus_sparsity})", fontweight='bold')

    plt.tight_layout()
    plt.show()