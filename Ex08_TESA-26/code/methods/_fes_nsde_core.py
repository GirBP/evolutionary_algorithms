import torch
import torch.nn as nn
import numpy as np
import random

class Pruner_FES_NSDE:
    def __init__(self, model: nn.Module, target_sparsity: float = 0.98, pop_size: int = 20, generations: int = 15):
        """ FES-NSDE: Fisher-Empirical Saliency with Null-Space Differential Evolution """
        self.model = model
        self.target_sparsity = target_sparsity
        self.pop_size = pop_size
        self.generations = generations
        self.device = next(model.parameters()).device
        self.prunable_modules = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
        
        self.layer_sizes = np.array([m.weight.numel() for m in self.prunable_modules])
        self.total_params = self.layer_sizes.sum()
        self.K_req = int(self.total_params * (1.0 - target_sparsity))
        
        # Гіперпараметри Диференційної Еволюції
        self.F = 0.6  # Mutation factor (Сила мутації)
        self.CR = 0.8 # Crossover rate (Ймовірність схрещування)

    def _compute_fes_saliency(self, dataloader, criterion, num_batches=3):
        """Швидка оцінка еволюційного середовища (Фішер-Тейлор 2-го порядку)"""
        self.model.eval()
        sum_grads = [torch.zeros_like(m.weight) for m in self.prunable_modules]
        sum_sq_grads = [torch.zeros_like(m.weight) for m in self.prunable_modules]
        
        data_iter = iter(dataloader)
        for _ in range(num_batches):
            inputs, targets = next(data_iter)
            self.model.zero_grad()
            loss = criterion(self.model(inputs.to(self.device)), targets.to(self.device))
            loss.backward()
            with torch.no_grad():
                for i, m in enumerate(self.prunable_modules):
                    sum_grads[i] += m.weight.grad
                    sum_sq_grads[i] += m.weight.grad ** 2
                    
        saliency = []
        with torch.no_grad():
            for i, m in enumerate(self.prunable_modules):
                E_g, E_g2 = sum_grads[i] / num_batches, sum_sq_grads[i] / num_batches
                W = m.weight
                saliency.append(torch.abs(W * E_g) + 0.5 * (W ** 2) * E_g2)
        return saliency

    def _evolutionary_repair(self, genotype):
        """Оператор репарації генотипу: Проєкція Лагранжа (Бісекція)"""
        def calc_k(nu): return np.clip(genotype - nu * self.layer_sizes, 0.0, 1.0)
        
        nu_min, nu_max = -1e-2, 1e-2
        while np.sum(calc_k(nu_min) * self.layer_sizes) < self.K_req: nu_min *= 2
        while np.sum(calc_k(nu_max) * self.layer_sizes) > self.K_req: nu_max *= 2
        
        for _ in range(35): # 35 ітерацій - це мікросекунди для CPU
            nu_mid = (nu_min + nu_max) / 2.0
            if np.sum(calc_k(nu_mid) * self.layer_sizes) > self.K_req: nu_min = nu_mid
            else: nu_max = nu_mid
        return calc_k((nu_min + nu_max) / 2.0)

    def _evaluate_fitness(self, genotype, saliency, calib_dataloader, criterion):
        """Експресія фенотипу та оцінка пристосованості (Fitness) нащадка"""
        orig_weights = []
        with torch.no_grad():
            for i, (m, sal, k) in enumerate(zip(self.prunable_modules, saliency, genotype)):
                orig_weights.append(m.weight.data.clone())
                num_keep = max(1, int(k * sal.numel()))
                threshold = torch.topk(sal.view(-1), num_keep).values[-1]
                m.weight.data.mul_((sal >= threshold).float())
                
            inputs, targets = next(iter(calib_dataloader))
            loss = criterion(self.model(inputs.to(self.device)), targets.to(self.device)).item()
            
            for i, m in enumerate(self.prunable_modules):
                m.weight.data.copy_(orig_weights[i])
                
        return -loss # Максимізуємо фітнес (мінімізуємо Loss)

    def evolve_topology(self, calib_dataloader, criterion):
        print("1. Формування Фітнес-ландшафту (Матриці Фішера)...")
        saliency = self._compute_fes_saliency(calib_dataloader, criterion)
        L = len(self.prunable_modules)
        
        print("2. Ініціалізація еволюційної популяції...")
        # Базова популяція: рівномірний розподіл + шум + миттєва репарація
        base_k = np.full(L, 1.0 - self.target_sparsity)
        population = [self._evolutionary_repair(base_k + np.random.normal(0, 0.05, L)) for _ in range(self.pop_size)]
        fitnesses = [self._evaluate_fitness(ind, saliency, calib_dataloader, criterion) for ind in population]
        
        print("3. Диференційна Еволюція у Нуль-просторі (NS-DE)...")
        for gen in range(self.generations):
            for i in range(self.pop_size):
                # Відбір донорів
                idxs = list(range(self.pop_size)); idxs.remove(i)
                r1, r2, r3 = random.sample(idxs, 3)
                
                # КРОК 1: Мутація у нуль-просторі (природно тримає 98%!)
                mutant = population[r1] + self.F * (population[r2] - population[r3])
                
                # КРОК 2: Біноміальне схрещування
                cross_points = np.random.rand(L) < self.CR
                if not np.any(cross_points): cross_points[np.random.randint(0, L)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # КРОК 3: Еволюційна репарація генотипу
                trial_repaired = self._evolutionary_repair(trial)
                
                # КРОК 4: Селекція Дарвіна
                f_trial = self._evaluate_fitness(trial_repaired, saliency, calib_dataloader, criterion)
                if f_trial > fitnesses[i]:
                    population[i] = trial_repaired
                    fitnesses[i] = f_trial
                    
            print(f" Покоління {gen+1}/{self.generations} | Найкращий фітнес (Loss): {-max(fitnesses):.4f}")
            
        print("4. Фіналізація: Видобуток найкращого геному...")
        best_k = population[np.argmax(fitnesses)]
        for i, m in enumerate(self.prunable_modules):
            num_keep = max(1, int(best_k[i] * saliency[i].numel()))
            threshold = torch.topk(saliency[i].view(-1), num_keep).values[-1]
            torch.nn.utils.parametrize.register_parametrization(m, "weight", Masker((saliency[i] >= threshold).float()))
            
        return self.model

class Masker(nn.Module):
    def __init__(self, mask): super().__init__(); self.register_buffer('mask', mask)
    def forward(self, w): return w * self.mask