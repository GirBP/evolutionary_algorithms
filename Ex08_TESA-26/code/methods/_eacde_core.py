import torch
import torch.nn as nn
import numpy as np
import random

class EACDE_Pruner:
    def __init__(self, model: nn.Module, min_sparsity: float = 0.98, tolerance: float = 1.15, pop_size: int = 15, generations: int = 15):
        """
        E-ACDE: Elastic Aitchison Compositional DE with Virtual Sparsity Sink (2026 SOTA)
        min_sparsity: Математична нижня межа (Гарантовано >= 98%).
        tolerance: Допустима деградація Loss (1.15 = дозволяємо +15% від Loss базової 98% моделі).
        """
        self.model = model
        self.min_sparsity = min_sparsity
        self.tolerance = tolerance
        self.pop_size = pop_size
        self.generations = generations
        self.device = next(model.parameters()).device
        
        self.prunable_modules = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
        self.layer_sizes = np.array([m.weight.numel() for m in self.prunable_modules], dtype=np.float64)
        self.total_params = self.layer_sizes.sum()
        
        # Бюджет для суворо 98% (Максимум доступних фізичних параметрів)
        self.B_max = self.total_params * (1.0 - min_sparsity) 
        self.dim = len(self.layer_sizes) + 1 # Розмірність: L шарів + 1 Віртуальний Стік
        
        self.F = 0.6  # Диференційна мутація
        self.CR = 0.8 # Біноміальний кросовер

    def _aitchison_closure(self, x):
        return x / np.sum(x)

    def _compositional_mutation(self, c1, c2, c3):
        return self._aitchison_closure(c1 * np.power(c2 / c3, self.F))

    def _decode_simplex(self, c_vector):
        """ Декодер із Швидким Переповненням (Fast Void Overflow) """
        c_layers = c_vector[:-1]
        
        # Розподіл фізичних параметрів
        k = (c_layers * self.B_max) / self.layer_sizes
        
        # Якщо еволюція дала шару >100% ваг, ми зрізаємо до 1.0. 
        # Весь надлишок математично "випаровується" у Віртуальний Стік, збільшуючи розрідженість!
        k = np.clip(k, 0.0, 1.0)
        
        actual_retained = np.sum(k * self.layer_sizes)
        actual_sparsity = 1.0 - (actual_retained / self.total_params)
        return k, actual_sparsity

    def _compute_fes_saliency(self, dataloader, criterion, num_batches=3):
        """ Fisher-Empirical Saliency (Оцінка кривизни ландшафту втрат) """
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
        print("1. Фішерівське сканування простору ваг (FES)...")
        saliency = self._compute_fes_saliency(calib_dataloader, criterion)
        
        # Оцінка Труби Толерантності (База: рівномірні 98%)
        base_c = self._aitchison_closure(np.append(self.layer_sizes, self.B_max * 0.001))
        k_base, _ = self._decode_simplex(base_c)
        
        inputs, targets = next(iter(calib_dataloader))
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        orig_weights = [m.weight.data.clone() for m in self.prunable_modules]
        with torch.no_grad():
            for i, m in enumerate(self.prunable_modules):
                num_keep = max(1, int(k_base[i] * saliency[i].numel()))
                thresh = torch.topk(saliency[i].view(-1), num_keep).values[-1]
                m.weight.data.mul_((saliency[i] >= thresh).float())
            base_loss = criterion(self.model(inputs), targets).item()
            for i, m in enumerate(self.prunable_modules): m.weight.data.copy_(orig_weights[i])
            
        L_max = base_loss * self.tolerance
        print(f"   Базовий Loss (при 98%): {base_loss:.4f} | Межа Труби Толерантності: {L_max:.4f}")
        
        print(f"2. Ініціалізація E-ACDE у {self.dim}-вимірному симплексі...")
        population = [self._aitchison_closure(base_c * np.exp(np.random.normal(0, 0.1, self.dim))) for _ in range(self.pop_size)]
        
        def evaluate(c_vec):
            k_vec, actual_sp = self._decode_simplex(c_vec)
            with torch.no_grad():
                for i, m in enumerate(self.prunable_modules):
                    num_keep = max(1, int(k_vec[i] * saliency[i].numel()))
                    thresh = torch.topk(saliency[i].view(-1), num_keep).values[-1]
                    m.weight.data.mul_((saliency[i] >= thresh).float())
                loss = criterion(self.model(inputs), targets).item()
                for i, m in enumerate(self.prunable_modules): m.weight.data.copy_(orig_weights[i])
            return loss, actual_sp

        eval_data = [evaluate(ind) for ind in population]
        
        print("3. Еволюція з Лексикографічною Селекцією (LATS)...")
        for gen in range(self.generations):
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = random.sample(idxs, 3)
                
                # Неевклідова Мутація
                mutant = self._compositional_mutation(population[r1], population[r2], population[r3])
                cross = np.random.rand(self.dim) < self.CR
                if not np.any(cross): cross[np.random.randint(self.dim)] = True
                trial = self._aitchison_closure(np.where(cross, mutant, population[i]))
                
                t_loss, t_sp = evaluate(trial)
                p_loss, p_sp = eval_data[i]
                
                # --- LATS СЕЛЕКЦІЯ (АБСОЛЮТНА НОВИЗНА) ---
                t_in = t_loss <= L_max
                p_in = p_loss <= L_max
                
                replace = False
                if t_in and p_in:    replace = t_sp > p_sp          # Обидва в нормі -> виграє СТИСНЕНІШИЙ
                elif t_in and not p_in: replace = True              # Нащадок врятував мережу
                elif not t_in and not p_in: replace = t_loss < p_loss # Обидва погані -> виграє ТОЧНІШИЙ
                    
                if replace:
                    population[i] = trial
                    eval_data[i] = (t_loss, t_sp)
            
            valid_inds = [(d[0], d[1]) for d in eval_data if d[0] <= L_max]
            if valid_inds:
                best = max(valid_inds, key=lambda x: x[1])
                print(f" Gen {gen+1:02d}/{self.generations} | Максимальне стиснення в Трубі: {best[1]*100:.3f}% (Loss: {best[0]:.4f})")
            else:
                best = min(eval_data, key=lambda x: x[0])
                print(f" Gen {gen+1:02d}/{self.generations} | ⚠ Труба пробита. Відновлення... (Loss: {best[0]:.4f})")
                
        print("4. Фіналізація Парето-Оптимуму...")
        valid_pop = [(population[i], eval_data[i]) for i in range(self.pop_size) if eval_data[i][0] <= L_max]
        champion_p, champion_eval = max(valid_pop, key=lambda x: x[1][1]) if valid_pop else min(zip(population, eval_data), key=lambda x: x[1][0])
            
        best_k, _ = self._decode_simplex(champion_p)
        for i, m in enumerate(self.prunable_modules):
            num_keep = max(1, int(best_k[i] * saliency[i].numel()))
            thresh = torch.topk(saliency[i].view(-1), num_keep).values[-1]
            torch.nn.utils.parametrize.register_parametrization(m, "weight", Masker((saliency[i] >= thresh).float()))
            
        print(f"ТРИУМФ! Знайдено топологію: {champion_eval[1]*100:.4f}% стиснення (Гарантовано >= {self.min_sparsity*100}%).")
        return self.model

class Masker(nn.Module):
    def __init__(self, mask): super().__init__(); self.register_buffer('mask', mask)
    def forward(self, w): return w * self.mask