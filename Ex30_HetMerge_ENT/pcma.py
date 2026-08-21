"""
PCMA: Procrustes-CMA Heterogeneous Neural Network Merging
==========================================================
Merges two MLPs of DIFFERENT architectures by:
1. Procrustes SVD → extracts mapping directions (U, V)
2. CMA-ES → optimizes scaling factors (s) and per-layer mixing (α)

Usage:
    from pcma import PCMA
    
    merger = PCMA(model_A, model_B, X_calibration)
    merged_model = merger.merge(X_val, y_val)
    
    # Or step-by-step:
    merger = PCMA(model_A, model_B, X_calibration)
    merged_model = merger.merge_with_fixed_alpha(X_val, y_val, alpha=0.5)  # CMA on s only
    merged_model = merger.merge_full(X_val, y_val)  # CMA on s + per-layer α

Reference: E01-E05 exploration series, March 2026.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PCMAResult:
    """Result of PCMA merge."""
    merged_model: nn.Module
    accuracy: float
    retention: float  # accuracy / best_parent
    scaling_factors: list[np.ndarray]
    alphas: list[float]
    n_evals: int
    n_dims: int
    acc_A: float
    acc_B: float


class PCMA:
    """Procrustes-CMA Heterogeneous Neural Network Merger.
    
    Args:
        model_A: Smaller MLP (target architecture for merged model)
        model_B: Larger MLP (to be mapped into A's space)
        X_calibration: Calibration data for computing activations (N, input_dim)
        
    The merged model will have the architecture of model_A.
    """
    
    def __init__(self, model_A: nn.Module, model_B: nn.Module, 
                 X_calibration: torch.Tensor):
        self.model_A = model_A
        self.model_B = model_B
        self.X_cal = X_calibration
        
        self.arch_A = self._get_arch(model_A)
        self.arch_B = self._get_arch(model_B)
        self.n_hidden = len(self.arch_A) - 2
        self.n_layers = self.n_hidden + 1
        
        # Compute SVD components
        self.svd_list, self.s_inits = self._compute_svd()
        self.total_s_dims = sum(len(s) for s in self.s_inits)
    
    @staticmethod
    def _get_arch(model: nn.Module) -> list[int]:
        """Extract architecture from Sequential MLP."""
        if hasattr(model, 'arch'):
            return model.arch
        arch = []
        for m in model.net if hasattr(model, 'net') else model.modules():
            if isinstance(m, nn.Linear):
                if not arch:
                    arch.append(m.in_features)
                arch.append(m.out_features)
        return arch
    
    def _get_postrelu_activations(self, model: nn.Module, X: torch.Tensor) -> list[np.ndarray]:
        """Get post-ReLU activations for each hidden layer."""
        acts = []
        with torch.no_grad():
            h = X
            net = model.net if hasattr(model, 'net') else list(model.children())[0]
            for m in net:
                h = m(h)
                if isinstance(m, nn.ReLU):
                    acts.append(h.numpy().copy())
        return acts
    
    def _compute_svd(self) -> tuple[list, list]:
        """Compute Procrustes SVD for each hidden layer."""
        acts_A = self._get_postrelu_activations(self.model_A, self.X_cal)
        acts_B = self._get_postrelu_activations(self.model_B, self.X_cal)
        
        svd_list = []
        s_inits = []
        
        for i in range(self.n_hidden):
            d_a = self.arch_A[i + 1]
            d_b = self.arch_B[i + 1]
            d_min = min(d_a, d_b)
            
            H_A = acts_A[i] if i < len(acts_A) else np.zeros((len(self.X_cal), d_a))
            H_B = acts_B[i] if i < len(acts_B) else np.zeros((len(self.X_cal), d_b))
            
            C = H_A.T @ H_B  # (d_a, d_b)
            U, S, Vt = np.linalg.svd(C, full_matrices=False)
            
            s_init = S[:d_min] / (S[0] + 1e-10)
            U_use = U[:, :d_min]    # (d_a, d_min)
            Vt_use = Vt[:d_min, :]  # (d_min, d_b)
            
            svd_list.append((U_use, Vt_use))
            s_inits.append(s_init)
        
        return svd_list, s_inits
    
    def _build_mappings(self, s_vec: np.ndarray) -> list[np.ndarray]:
        """Build mapping matrices from scaling vector."""
        mappings = []
        offset = 0
        for i in range(self.n_hidden):
            U, Vt = self.svd_list[i]
            d = len(self.s_inits[i])
            s = s_vec[offset:offset + d]
            offset += d
            mappings.append(U @ np.diag(s) @ Vt)
        return mappings
    
    def _merge_with_params(self, s_vec: np.ndarray, alphas: list[float]) -> nn.Module:
        """Build merged model from parameters."""
        mappings = self._build_mappings(s_vec)
        
        W_A = [p.detach().numpy() for p in self.model_A.parameters()]
        W_B = [p.detach().numpy() for p in self.model_B.parameters()]
        
        merged_params = []
        for li in range(self.n_layers):
            a = np.clip(alphas[li], 0.05, 0.95)
            W_a, b_a = W_A[li * 2], W_A[li * 2 + 1]
            W_b, b_b = W_B[li * 2], W_B[li * 2 + 1]
            
            if li == 0:
                W_b_m = mappings[0] @ W_b
                b_b_m = mappings[0] @ b_b
            elif li < self.n_hidden:
                W_b_m = mappings[li] @ W_b @ mappings[li - 1].T
                b_b_m = mappings[li] @ b_b
            else:
                W_b_m = W_b @ mappings[-1].T
                b_b_m = b_b
            
            merged_params.append(a * W_a + (1 - a) * W_b_m)
            merged_params.append(a * b_a + (1 - a) * b_b_m)
        
        # Build model with A's architecture
        merged = self._create_mlp(self.arch_A)
        with torch.no_grad():
            for p, v in zip(merged.parameters(), merged_params):
                p.copy_(torch.tensor(v, dtype=torch.float32))
        return merged
    
    @staticmethod
    def _create_mlp(arch: list[int]) -> nn.Module:
        """Create MLP with given architecture."""
        layers = []
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i + 1]))
            if i < len(arch) - 2:
                layers.append(nn.ReLU())
        
        class _MLP(nn.Module):
            def __init__(self, net, architecture):
                super().__init__()
                self.net = nn.Sequential(*net)
                self.arch = architecture
            def forward(self, x):
                return self.net(x)
        
        return _MLP(layers, arch)
    
    @staticmethod
    def _evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            return (model(X).argmax(1) == y).float().mean().item()
    
    def merge(self, X_val: torch.Tensor, y_val: torch.Tensor,
              maxiter: int = 80, popsize: int = 18, sigma0: float = 0.3,
              seed: int = 42) -> PCMAResult:
        """Full PCMA merge: CMA-ES on s + per-layer α.
        
        This is the recommended method (Strategy S4 from E05).
        """
        import cma
        
        nd_s = self.total_s_dims
        n_layers = self.n_layers
        x0 = np.concatenate([np.concatenate(self.s_inits), [0.5] * n_layers])
        nd = len(x0)
        
        acc_A = self._evaluate(self.model_A, X_val, y_val)
        acc_B = self._evaluate(self.model_B, X_val, y_val)
        
        def fitness(x):
            s, alphas = x[:nd_s], x[nd_s:].tolist()
            try:
                m = self._merge_with_params(s, alphas)
                return -self._evaluate(m, X_val, y_val)
            except Exception:
                return 1.0
        
        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {
            'maxiter': maxiter, 'popsize': popsize, 'seed': seed,
            'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
            'bounds': [[-3] * nd_s + [0.1] * n_layers,
                       [3] * nd_s + [0.9] * n_layers],
            'CMA_diagonal': nd > 80,
        })
        
        while not es.stop():
            sols = es.ask()
            es.tell(sols, [fitness(np.array(s)) for s in sols])
        
        best = np.array(es.result.xbest)
        best_s = best[:nd_s]
        best_alphas = best[nd_s:].tolist()
        
        merged = self._merge_with_params(best_s, best_alphas)
        best_acc = self._evaluate(merged, X_val, y_val)
        best_parent = max(acc_A, acc_B)
        
        # Extract per-layer scaling factors
        scaling_factors = []
        offset = 0
        for i in range(self.n_hidden):
            d = len(self.s_inits[i])
            scaling_factors.append(best_s[offset:offset + d])
            offset += d
        
        return PCMAResult(
            merged_model=merged,
            accuracy=best_acc,
            retention=best_acc / best_parent if best_parent > 0 else 0,
            scaling_factors=scaling_factors,
            alphas=[np.clip(a, 0.1, 0.9) for a in best_alphas],
            n_evals=es.result.evaluations,
            n_dims=nd,
            acc_A=acc_A,
            acc_B=acc_B,
        )
    
    def merge_fixed_alpha(self, X_val: torch.Tensor, y_val: torch.Tensor,
                          alpha: float = 0.5,
                          maxiter: int = 50, popsize: int = 16, sigma0: float = 0.3,
                          seed: int = 42) -> PCMAResult:
        """PCMA with fixed α: CMA-ES on s only.
        
        Simpler and faster. Best for shallow networks (2-5 layers).
        """
        import cma
        
        s0 = np.concatenate(self.s_inits)
        nd = len(s0)
        alphas = [alpha] * self.n_layers
        
        acc_A = self._evaluate(self.model_A, X_val, y_val)
        acc_B = self._evaluate(self.model_B, X_val, y_val)
        
        def fitness(s):
            try:
                m = self._merge_with_params(s, alphas)
                return -self._evaluate(m, X_val, y_val)
            except Exception:
                return 1.0
        
        es = cma.CMAEvolutionStrategy(s0.tolist(), sigma0, {
            'maxiter': maxiter, 'popsize': popsize, 'seed': seed,
            'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
            'bounds': [[-3] * nd, [3] * nd],
            'CMA_diagonal': nd > 80,
        })
        
        while not es.stop():
            sols = es.ask()
            es.tell(sols, [fitness(np.array(s)) for s in sols])
        
        best_s = np.array(es.result.xbest)
        merged = self._merge_with_params(best_s, alphas)
        best_acc = self._evaluate(merged, X_val, y_val)
        best_parent = max(acc_A, acc_B)
        
        scaling_factors = []
        offset = 0
        for i in range(self.n_hidden):
            d = len(self.s_inits[i])
            scaling_factors.append(best_s[offset:offset + d])
            offset += d
        
        return PCMAResult(
            merged_model=merged,
            accuracy=best_acc,
            retention=best_acc / best_parent if best_parent > 0 else 0,
            scaling_factors=scaling_factors,
            alphas=alphas,
            n_evals=es.result.evaluations,
            n_dims=nd,
            acc_A=acc_A,
            acc_B=acc_B,
        )
