import torch
import torch.nn as nn
import numpy as np
import random

class ACDE_Pruner:
    def __init__(self, model: nn.Module, target_sparsity: float = 0.98, pop_size: int = 15, generations: int = 10):
        """ ACDE: Aitchison Compositional Differential Evolution (2026) """
        self.model = model
        self.target_sparsity = target_sparsity
        self.pop_size = pop_size
        self.generations = generations
        self.device = next(model.parameters()).device
        self.prunable_modules = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
        
        self.layer_sizes = np.array([m.weight.numel() for m in self.prunable_modules], dtype=np.float64)
        self.total_params = self.layer_sizes.sum()
        self.K_req = self.total_params * (1.0 - target_sparsity)
        
        self.F = 0.6  # Differential mutation factor
        self.CR = 0.8 # Crossover probability

    def _aitchison_closure(self, x):
        """ Проєкція вектора на стандартний симплекс S^{L-1} """
        return x / np.sum(x)

    def _compositional_mutation(self, p1, p2, p3):
        """ ВАША НОВИЗНА: Геометрія Ейтчісона для мутації """
        # p1 * (p2/p3)^F. Сума гарантовано = 1 після closure.
        mutant_raw = p1 * np.power(p2 / p3, self.F)
        return self._aitchison_closure(mutant_raw)

    def _decode_simplex_to_quotas(self, p):
        """ Алгоритм Water-filling: перетворення бюджету p у пошарові квоти k_l """
        k = (p * self.K_req) / self.layer_sizes
        
        # Обробка фізичних обмежень (якщо шару дали більше 100% ваг)
        while np.any(k > 1.0):
            overflow_idx = k > 1.0
            valid_idx = k <= 1.0
            
            excess_budget = np.sum((k[overflow_idx] - 1.0) * self.layer_sizes[overflow_idx])
            k[overflow_idx] = 1.0
            
            if not np.any(valid_idx) or excess_budget < 1e-5: break 
                
            p_valid = self._aitchison_closure(p[valid_idx])
            extra_k = (p_valid * excess_budget) / self.layer_sizes[valid_idx]
            k[valid_idx] += extra_k
            
        return k

    def _compute_fes_saliency(self, dataloader, criterion, num_batches=3):
        """ Емпіричний Фішер 2-го порядку для проксі-оцінки """
        self.model.eval()
        sum_grads = [torch.zeros_like(m.weight) for m in self.prunable_modules]
        sum_sq_grads = [torch.zeros_like(m.weight) for m in self.prunable_modules]
        
        data_iter = iter(dataloader)
        for _ in range(num_batches):
            inputs, targets = next(data_iter)
            self.model.zero_grad()
            criterion(self.model(inputs.to(self.device)), targets.to(self.device)).backward()
            with torch.no_grad():
                for i, m in enumerate(self.prunable_modules):
                    sum_grads[i] += m.weight.grad
                    sum_sq_grads[i] += m.weight.grad ** 2
                    
        saliency = []
        with torch.no_grad():
            for i, m in enumerate(self.prunable_modules):
                saliency.append(torch.abs(m.weight * (sum_grads[i]/num_batches)) + 
                              0.5 * (m.weight**2) * (sum_sq_grads[i]/num_batches))
        return saliency

    def prune(self, calib_dataloader, criterion):
        print("1. Формування ландшафту Фішера...")
        saliency = self._compute_fes_saliency(calib_dataloader, criterion)
        L = len(self.prunable_modules)
        
        print("2. Ініціалізація еволюції у просторі Ейтчісона...")
        # Базова точка: розподіл пропорційно розмірам шарів
        base_p = self._aitchison_closure(self.layer_sizes)
        # Популяція: логарифмічно-нормальний шум (канонічно для симплексів)
        population = [self._aitchison_closure(base_p * np.exp(np.random.normal(0, 0.1, L))) for _ in range(self.pop_size)]
        
        def get_fitness(p_vec):
            k_vec = self._decode_simplex_to_quotas(p_vec)
            orig_weights = []
            with torch.no_grad():
                for i, (m, sal, k) in enumerate(zip(self.prunable_modules, saliency, k_vec)):
                    orig_weights.append(m.weight.data.clone())
                    num_keep = max(1, int(k * sal.numel()))
                    threshold = torch.topk(sal.view(-1), num_keep).values[-1]
                    m.weight.data.mul_((sal >= threshold).float())
                    
                inputs, targets = next(iter(calib_dataloader))
                loss = criterion(self.model(inputs.to(self.device)), targets.to(self.device)).item()
                
                for i, m in enumerate(self.prunable_modules): 
                    m.weight.data.copy_(orig_weights[i])
            return -loss

        fitnesses = [get_fitness(ind) for ind in population]
        
        print("3. Композиційна Диференційна Еволюція (ACDE)...")
        for gen in range(self.generations):
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = random.sample(idxs, 3)
                
                # ВАША МУТАЦІЯ (O(L) векторна операція, сума гарантовано = 1)
                mutant = self._compositional_mutation(population[r1], population[r2], population[r3])
                
                # Біноміальне схрещування
                cross = np.random.rand(L) < self.CR
                if not np.any(cross): cross[np.random.randint(L)] = True
                
                # Формуємо нащадка і ЗНОВУ замикаємо у симплекс (бо кросовер руйнує композицію)
                trial = np.where(cross, mutant, population[i])
                trial = self._aitchison_closure(trial)
                
                f_trial = get_fitness(trial)
                if f_trial > fitnesses[i]:
                    population[i], fitnesses[i] = trial, f_trial
                    
            print(f" Покоління {gen+1}/{self.generations} | Найкращий фітнес (Loss): {-max(fitnesses):.4f}")
            
        print("4. Фіналізація...")
        best_p = population[np.argmax(fitnesses)]
        best_k = self._decode_simplex_to_quotas(best_p)
        
        for i, m in enumerate(self.prunable_modules):
            num_keep = max(1, int(best_k[i] * saliency[i].numel()))
            threshold = torch.topk(saliency[i].view(-1), num_keep).values[-1]
            torch.nn.utils.parametrize.register_parametrization(m, "weight", Masker((saliency[i] >= threshold).float()))
            
        return self.model

class Masker(nn.Module):
    def __init__(self, mask): super().__init__(); self.register_buffer('mask', mask)
    def forward(self, w): return w * self.mask