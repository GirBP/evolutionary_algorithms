"""
Ex09: Sparse-to-Dense Network Conversion
=========================================
Core infrastructure: models, data, pruning, conversion, evaluation.

Pipeline:
  Teacher (dense) → Prune (FES-NSDE / E-ACDE) → Sparse model
  Sparse model → Convert → Compact Dense model
  Compact Dense → Fine-tune → Final evaluation

The key insight: A pruned network at 90% sparsity has only 10% active weights.
Instead of storing the full sparse tensor, we can create a SMALLER dense network
with ~10% of the original parameters that performsequivalently.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
import time
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


# ══════════════════════════════════════════
#  Seeds & Device
# ══════════════════════════════════════════
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ══════════════════════════════════════════
#  Data
# ══════════════════════════════════════════
def get_dataloaders(seed, dataset_name='moons', batch_size=64):
    from sklearn.datasets import (make_moons, make_circles, make_blobs,
                                    make_gaussian_quantiles, make_classification)
    set_seed(seed)

    if dataset_name == 'moons':
        X, y = make_moons(n_samples=2000, noise=0.2, random_state=seed)
    elif dataset_name == 'circles':
        X, y = make_circles(n_samples=2000, noise=0.15, factor=0.5, random_state=seed)
    elif dataset_name == 'spirals':
        n = 2000 // 2
        theta = np.linspace(0, 3 * np.pi, n)
        r = theta
        x1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)]) + np.random.randn(n, 2) * 0.5
        x2 = np.column_stack([-r * np.cos(theta), -r * np.sin(theta)]) + np.random.randn(n, 2) * 0.5
        X = np.vstack([x1, x2])
        y = np.array([0] * n + [1] * n)
    elif dataset_name == 'blobs':
        X, y = make_blobs(n_samples=2000, centers=4, cluster_std=1.0, random_state=seed)
    elif dataset_name == 'gaussian_quantiles':
        X, y = make_gaussian_quantiles(n_samples=2000, n_features=2,
                                        n_classes=4, random_state=seed)
    elif dataset_name == 'classification':
        X, y = make_classification(n_samples=2000, n_features=2,
                                    n_informative=2, n_redundant=0,
                                    n_clusters_per_class=1, n_classes=4,
                                    random_state=seed)
    elif dataset_name == 'highdim':
        # High-dimensional synthetic: 50 features, 5 classes
        # Creates more diverse connectivity patterns
        X, y = make_classification(n_samples=2000, n_features=50,
                                    n_informative=30, n_redundant=10,
                                    n_classes=5, n_clusters_per_class=1,
                                    random_state=seed)
    elif dataset_name == 'sequence_cls':
        # Synthetic sequence classification: sum patterns in sequences
        rng = np.random.RandomState(seed)
        seq_len, vocab = 16, 32
        n_samples = 2000
        # 4 classes based on sequence statistics
        X_seqs = rng.randint(0, vocab, (n_samples, seq_len)).astype(np.float32)
        # Features: mean, std, max, min of each quarter
        quarters = np.array_split(X_seqs, 4, axis=1)
        features = []
        for q in quarters:
            features.extend([q.mean(1), q.std(1), q.max(1), q.min(1)])
        X = np.column_stack(features)  # 16 features
        # Labels: based on dominant quarter
        quarter_sums = [q.sum(1) for q in quarters]
        y = np.argmax(np.column_stack(quarter_sums), axis=1)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    n_classes = len(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.14, random_state=seed, stratify=y_train)

    def to_dl(X, y, shuffle=False):
        return DataLoader(TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        ), batch_size=batch_size, shuffle=shuffle)

    return to_dl(X_train, y_train, True), to_dl(X_val, y_val), to_dl(X_test, y_test), n_classes


# ══════════════════════════════════════════
#  Models
# ══════════════════════════════════════════
class SimpleMLP(nn.Module):
    """Original full-size MLP (matches Ex08)."""
    def __init__(self, input_dim=2, hidden=128, n_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, n_classes)
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden = hidden

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_nonzero(self):
        return sum((p.data != 0).sum().item() for p in self.parameters())


class CompactMLP(nn.Module):
    """Compact dense MLP with configurable hidden sizes."""
    def __init__(self, input_dim=2, hiddens=(64, 32), n_classes=2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hiddens:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.n_classes = n_classes

    def forward(self, x):
        return self.net(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════
#  Training & Evaluation
# ══════════════════════════════════════════
def train_model(model, train_dl, epochs=100, lr=0.01):
    """Train a model from scratch or fine-tune."""
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        for X, y in train_dl:
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            opt.step()
    return model


def evaluate(model, test_dl):
    """Evaluate model → (loss, f1_macro, accuracy, predictions)."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in test_dl:
            logits = model(X)
            total_loss += crit(logits, y).item()
            all_preds.extend(logits.argmax(1).numpy())
            all_labels.extend(y.numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return total_loss / len(test_dl), f1, acc


# ══════════════════════════════════════════
#  Pruning (from Ex08 winners)
# ══════════════════════════════════════════
def prune_magnitude_global(model, sparsity):
    """Global unstructured magnitude pruning."""
    all_weights = torch.cat([p.data.abs().view(-1) for p in model.parameters() if p.dim() >= 2])
    threshold = torch.quantile(all_weights, sparsity).item()
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                mask = (p.data.abs() >= threshold).float()
                p.data.mul_(mask)
    return model


def get_sparsity(model):
    """Actual sparsity of a model."""
    total = sum(p.numel() for p in model.parameters() if p.dim() >= 2)
    zeros = sum((p.data == 0).sum().item() for p in model.parameters() if p.dim() >= 2)
    return zeros / total if total > 0 else 0


# ══════════════════════════════════════════
#  Sparse → Dense Conversion Methods
# ══════════════════════════════════════════

def convert_neuron_removal(sparse_model, n_classes=2):
    """
    Method 1: Neuron Removal
    Remove neurons that have all-zero incoming or outgoing weights.
    The result is a smaller dense model with active neurons only.
    """
    layers = [sparse_model.fc1, sparse_model.fc2, sparse_model.fc3, sparse_model.fc4]

    # Find active neurons per layer
    active_neurons = []
    for i, layer in enumerate(layers[:-1]):  # skip output
        W = layer.weight.data  # [out, in]
        # Neuron is active if it has any non-zero incoming weight AND any non-zero outgoing weight
        incoming_alive = (W.abs().sum(dim=1) > 1e-8)  # [out]
        if i + 1 < len(layers):
            W_next = layers[i + 1].weight.data  # [next_out, out]
            outgoing_alive = (W_next.abs().sum(dim=0) > 1e-8)  # [out]
            alive = incoming_alive & outgoing_alive
        else:
            alive = incoming_alive
        active_idx = torch.where(alive)[0]
        if len(active_idx) == 0:
            active_idx = torch.tensor([0])  # keep at least 1
        active_neurons.append(active_idx)

    # Build compact model
    hiddens = [len(a) for a in active_neurons]
    compact = CompactMLP(input_dim=sparse_model.input_dim, hiddens=hiddens, n_classes=n_classes)

    # Copy weights
    with torch.no_grad():
        prev_active = None
        layer_idx = 0
        for i, (src_layer, active) in enumerate(zip(layers, active_neurons + [None])):
            if active is not None:
                # Hidden layer
                W = src_layer.weight.data[active]  # select active output neurons
                if prev_active is not None:
                    W = W[:, prev_active]  # select active input neurons
                b = src_layer.bias.data[active] if src_layer.bias is not None else None

                tgt = compact.net[layer_idx * 2]  # Linear layer (skip ReLU)
                tgt.weight.data.copy_(W)
                if b is not None:
                    tgt.bias.data.copy_(b)
                prev_active = active
            else:
                # Output layer
                W = src_layer.weight.data
                if prev_active is not None:
                    W = W[:, prev_active]
                b = src_layer.bias.data if src_layer.bias is not None else None

                tgt = compact.net[-1]  # last Linear
                tgt.weight.data.copy_(W)
                if b is not None:
                    tgt.bias.data.copy_(b)
            layer_idx += 1

    return compact


def convert_svd_compression(sparse_model, rank_ratio=0.5, n_classes=2):
    """
    Method 2: SVD Compression
    Apply truncated SVD to each weight matrix, keeping top singular values.
    Reconstruct as two smaller dense matrices: W ≈ U_r @ S_r @ V_r^T
    Then merge into a single smaller layer.
    """
    layers = [sparse_model.fc1, sparse_model.fc2, sparse_model.fc3, sparse_model.fc4]
    new_hiddens = []

    compressed_weights = []
    for i, layer in enumerate(layers):
        W = layer.weight.data  # [out, in]
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        rank = max(1, int(min(W.shape) * rank_ratio))
        # Reconstruct with reduced rank
        W_approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
        compressed_weights.append((W_approx, layer.bias.data.clone() if layer.bias is not None else None))
        if i < len(layers) - 1:
            new_hiddens.append(W.shape[0])  # keep original topology for now

    # Create model with same topology but compressed weights
    compact = CompactMLP(input_dim=sparse_model.input_dim, hiddens=new_hiddens, n_classes=n_classes)
    with torch.no_grad():
        for i, (W, b) in enumerate(compressed_weights):
            if i < len(compressed_weights) - 1:
                tgt = compact.net[i * 2]
            else:
                tgt = compact.net[-1]
            tgt.weight.data.copy_(W)
            if b is not None:
                tgt.bias.data.copy_(b)

    return compact


def convert_knowledge_distill(sparse_model, train_dl, n_classes=2, epochs=30):
    """
    Method 3: Knowledge Distillation
    Train a compact student from sparse teacher's soft predictions.
    """
    # Estimate compact size from actual parameter count
    nonzero = sparse_model.count_nonzero()
    total = sparse_model.count_params()
    ratio = max(0.1, nonzero / total)

    # Target: compact model with ~same number of active params
    hidden = max(8, int(sparse_model.hidden * np.sqrt(ratio)))
    student = CompactMLP(
        input_dim=sparse_model.input_dim,
        hiddens=(hidden, max(4, hidden // 2)),
        n_classes=n_classes
    )

    # Distill
    opt = optim.Adam(student.parameters(), lr=0.005)
    sparse_model.eval()
    student.train()
    T = 4.0  # temperature

    for ep in range(epochs):
        for X, y in train_dl:
            with torch.no_grad():
                teacher_logits = sparse_model(X)
            student_logits = student(X)

            # KD loss: soft + hard
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction='batchmean'
            ) * T * T
            hard_loss = F.cross_entropy(student_logits, y)
            loss = 0.7 * soft_loss + 0.3 * hard_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

    return student


def convert_weight_redistribution(sparse_model, n_classes=2):
    """
    Method 4: Weight Redistribution
    Take non-zero weights from sparse model and redistribute them
    into a compact dense network, preserving weight statistics.
    """
    # Collect all non-zero weights
    all_nonzero = []
    for p in sparse_model.parameters():
        if p.dim() >= 2:
            nz = p.data[p.data != 0]
            all_nonzero.append(nz)
    all_nz = torch.cat(all_nonzero)

    n_active = len(all_nz)
    total = sparse_model.count_params()
    ratio = max(0.1, n_active / total)

    hidden = max(8, int(sparse_model.hidden * np.sqrt(ratio)))
    compact = CompactMLP(
        input_dim=sparse_model.input_dim,
        hiddens=(hidden, max(4, hidden // 2)),
        n_classes=n_classes
    )

    # Fill compact model with redistributed weights
    with torch.no_grad():
        weight_mean = all_nz.mean().item()
        weight_std = all_nz.std().item()

        for p in compact.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=weight_mean, std=weight_std)
            elif p.dim() == 1:
                nn.init.zeros_(p)

    return compact


# ══════════════════════════════════════════
#  Benchmark Runner
# ══════════════════════════════════════════
CONVERTERS = {
    'neuron_removal': convert_neuron_removal,
    'svd_compression': convert_svd_compression,
    'knowledge_distill': convert_knowledge_distill,
    'weight_redistribution': convert_weight_redistribution,
}
