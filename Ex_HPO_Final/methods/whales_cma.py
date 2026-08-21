"""WL-CMA — Warped Langevin CMA-ES (Surrogate-Assisted Concurrent Optimization)"""
import numpy as np
import scipy.stats as stats
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.exceptions import ConvergenceWarning
import logging

warnings.filterwarnings('ignore', category=ConvergenceWarning)
logger = logging.getLogger("whales_cma")
logger.setLevel(logging.ERROR)

class WarpingEngine:
    def __init__(self, dim):
        self.dim = dim
        self.output_warping_lambda = None
        self.min_y = 0.0
        self.scale_y = 1.0

    def fit(self, X, y):
        y_arr = np.array(y)
        if len(y_arr) > 3:
            try:
                _, lmbda = stats.yeojohnson(y_arr)
                self.output_warping_lambda = lmbda
            except Exception:
                self.output_warping_lambda = 1.0
        else:
            self.output_warping_lambda = 1.0

        y_w = stats.yeojohnson(y_arr, lmbda=self.output_warping_lambda)
        self.min_y = np.min(y_w)
        self.scale_y = np.std(y_w) + 1e-8
        return self

    def warp_X(self, X):
        return np.clip(X, 1e-5, 1 - 1e-5)

    def unwarp_X(self, X_w):
        return np.clip(X_w, 1e-5, 1 - 1e-5)

    def warp_y(self, y):
        y_arr = np.atleast_1d(y)
        y_w = stats.yeojohnson(y_arr, lmbda=self.output_warping_lambda)
        return (y_w - self.min_y) / self.scale_y

class HierarchicalMAB:
    def __init__(self, rng):
        self.rng = rng
        self.q_A = np.zeros(2) 
        self.c_A = np.ones(2)
        
        self.q_B = np.zeros(2) 
        self.c_B = np.ones(2)
        
        self.q_C = np.zeros(3) 
        self.c_C = np.ones(3)
        
        self.epsilon = 0.2
        self.historical_rewards = []
        
    def get_A(self):
        if self.rng.random() < self.epsilon: return self.rng.integers(2)
        return int(np.argmax(self.q_A))

    def get_B(self):
        if self.rng.random() < self.epsilon: return self.rng.integers(2)
        return int(np.argmax(self.q_B))

    def get_C_weights(self):
        exp_q = np.exp(self.q_C - np.max(self.q_C))
        return exp_q / np.sum(exp_q)

    def update(self, action_A, action_B, reward_raw):
        self.historical_rewards.append(reward_raw)
        if len(self.historical_rewards) > 1:
            mean_r = np.mean(self.historical_rewards)
            std_r = np.std(self.historical_rewards) + 1e-8
            z = (reward_raw - mean_r) / std_r
        else:
            z = 0.0
            
        self.c_A[action_A] += 1
        self.q_A[action_A] += (z - self.q_A[action_A]) / self.c_A[action_A]
        
        self.c_B[action_B] += 1
        self.q_B[action_B] += (z - self.q_B[action_B]) / self.c_B[action_B]
        
        for i in range(3):
            self.c_C[i] += 1
            w = self.get_C_weights()[i]
            self.q_C[i] += w * (z - self.q_C[i]) / self.c_C[i]

def surrogate_mace(X, gp, y_best_w, weights, rng):
    X_2d = np.atleast_2d(X)
    try:
        mu, std = gp.predict(X_2d, return_std=True)
        std = np.maximum(std, 1e-9)
    except Exception:
        return np.ones(len(X_2d)) * 1e6
        
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = y_best_w - mu
        Z = np.where(std > 0, imp / std, 0)
        ei = imp * stats.norm.cdf(Z) + std * stats.norm.pdf(Z)
        ei[std == 0.0] = 0.0
        
    ucb = - (mu - 1.96 * std) 
    ts = - (mu + std * rng.standard_normal(len(mu))) 
    
    def minmax(arr):
        if np.max(arr) == np.min(arr): return arr * 0
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-9)
        
    acq_val = weights[0] * minmax(ei) + weights[1] * minmax(ucb) + weights[2] * minmax(ts)
    return -acq_val 

def _cma_on_surrogate(gp, dim, rng, y_best_w, weights, m_init, C_init, n_gens=50, lam=10):
    mean = np.clip(m_init, 0, 1)
    sigma = 0.15
    mu = max(2, lam // 2)
    w_cma = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w_cma /= w_cma.sum()
    mueff = 1.0 / np.sum(w_cma**2)
    cc = 4.0 / (dim + 4)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2.0 / ((dim + 1.3)**2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    pc, ps, C = np.zeros(dim), np.zeros(dim), C_init.copy()
    chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))
    
    best_x, best_acq = mean.copy(), float('inf')

    for gen in range(n_gens):
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            sqrtC = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            invsqrtC = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
        except Exception:
            sqrtC, invsqrtC = np.eye(dim), np.eye(dim)

        arz = rng.standard_normal((lam, dim))
        arx = np.array([np.clip(mean + sigma * sqrtC @ z, 0, 1) for z in arz])
        
        fits = surrogate_mace(arx, gp, y_best_w, weights, rng)
        idx = np.argsort(fits)
        r_best = np.argmin(fits)
        if fits[r_best] < best_acq:
            best_acq, best_x = fits[r_best], arx[r_best].copy()

        old_mean = mean.copy()
        mean = sum(w_cma[i] * arx[idx[i]] for i in range(mu))
        ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * invsqrtC @ (mean - old_mean) / sigma
        hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(gen+1))) < (1.4+2/(dim+1))*chiN
        pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (mean - old_mean) / sigma
        artmp = (arx[idx[:mu]] - old_mean) / sigma
        C = (1-c1-cmu)*C + c1*(np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C)
        for i in range(mu):
            C += cmu * w_cma[i] * np.outer(artmp[i], artmp[i])
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = np.clip(sigma, 1e-8, 0.5)

    return np.clip(best_x, 0, 1), C

def langevin_mcmc(start_points, gp, y_best_w, weights, rng, n_steps=20, eta=0.01):
    best_x = start_points[0]
    best_acq = float('inf')
    eps = 1e-4
    
    for x_init in start_points:
        x = x_init.copy()
        for _ in range(n_steps):
            grad = np.zeros_like(x)
            acq_base = surrogate_mace(x, gp, y_best_w, weights, rng)[0]
            
            for d in range(len(x)):
                x_f = x.copy()
                x_f[d] += eps
                x_f = np.clip(x_f, 0, 1)
                acq_f = surrogate_mace(x_f, gp, y_best_w, weights, rng)[0]
                grad[d] = (acq_f - acq_base) / eps
                
            grad = np.clip(grad, -10, 10) 
            noise = np.sqrt(2 * eta) * rng.standard_normal(len(x))
            
            x = x - eta * grad + noise
            x = np.clip(x, 0, 1)
            
            acq_val = surrogate_mace(x, gp, y_best_w, weights, rng)[0]
            if acq_val < best_acq:
                best_acq = acq_val
                best_x = x.copy()
                
    return best_x


def run(seed, obj_fn, dim, budget):
    """
    WL-CMA — Warped Langevin CMA-ES.
    Surrogate-Assisted HPO з Yeo-Johnson warping, CMA-ES, Langevin MCMC та ієрархічним MAB.
    """
    rng = np.random.default_rng(seed)
    
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=dim, seed=seed)
    n_init = min(10, budget // 3)
    
    skip = int(rng.integers(0, 5))
    if skip > 0:
        sampler.fast_forward(skip)
    X_init = sampler.random(n=n_init)
    
    X_hist, y_hist, curve = [], [], []
    bl = float('inf')
    x_best = None

    for v in X_init:
        fl = obj_fn(v)
        X_hist.append(v)
        y_hist.append(fl)
        if fl < bl: 
            bl = fl
            x_best = v.copy()
        curve.append(bl)

    ec = n_init
    warping = WarpingEngine(dim)
    mab = HierarchicalMAB(rng)
    
    best_history = [bl] * 5
    C_cma = np.eye(dim)
    kernel = Matern(nu=2.5)

    while ec < budget:
        Xh, yh = np.array(X_hist), np.array(y_hist)
        
        warping.fit(Xh, yh)
        
        X_w = warping.warp_X(Xh)
        y_w = warping.warp_y(yh)
        y_best_w = np.min(y_w)
        
        mab_A = mab.get_A() 
        mab_B = mab.get_B() 
        mab_C = mab.get_C_weights() 
        
        lam = 20 if mab_A == 0 else 10
        n_gens = 20 if mab_A == 0 else 50
        
        gp_global = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, n_restarts_optimizer=0, random_state=seed)
        gp_local = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, n_restarts_optimizer=0, random_state=seed+1)
        
        try:
            gp_global.fit(X_w, y_w)
            
            mean_X = X_w[np.argmin(y_w)]
            inv_C = np.linalg.pinv(C_cma)
            dists = np.array([ (x-mean_X).T @ inv_C @ (x-mean_X) for x in X_w ])
            threshold = np.percentile(dists, 50)
            in_tr = dists <= threshold
            
            if np.sum(in_tr) >= 3:
                gp_local.fit(X_w[in_tr], y_w[in_tr])
            else:
                gp_local = gp_global
                
        except Exception:
            gp_local = gp_global
            
        chosen_gp = gp_global if mab_B == 0 else gp_local

        m_init = warping.warp_X(X_hist[np.argmin(y_hist)])
        cma_cand_w, new_C = _cma_on_surrogate(chosen_gp, dim, rng, y_best_w, mab_C, m_init, C_cma, n_gens=n_gens, lam=lam)
        
        C_cma = (0.9 * C_cma + 0.1 * new_C) 
        
        top5_idx = np.argsort(y_w)[:5]
        start_pts = [X_w[i] for i in top5_idx]
        lang_cand_w = langevin_mcmc(start_pts, chosen_gp, y_best_w, mab_C, rng)
        
        cma_acq = surrogate_mace(cma_cand_w, chosen_gp, y_best_w, mab_C, rng)[0]
        lang_acq = surrogate_mace(lang_cand_w, chosen_gp, y_best_w, mab_C, rng)[0]
        
        best_cand_w = cma_cand_w if cma_acq < lang_acq else lang_cand_w
        candidate = warping.unwarp_X(best_cand_w)
        
        f_old = best_history[0]
        delta_f = max(0.0, (f_old - bl) / (f_old + 1e-12))
        stagnating = delta_f < 1e-4
        
        if stagnating and ec + 2 <= budget: 
            v_pop = rng.random((100, dim))
            v_pop_w = warping.warp_X(v_pop)
            try:
                _, stds = chosen_gp.predict(v_pop_w, return_std=True)
                dts_idx = np.argmax(stds)
                dts_cand = v_pop[dts_idx]
                
                fl_dts = obj_fn(dts_cand)
                X_hist.append(dts_cand)
                y_hist.append(fl_dts)
                if fl_dts < bl: 
                    bl = fl_dts
                    x_best = dts_cand.copy()
                curve.append(bl)
                ec += 1
            except Exception: pass
            
        if ec < budget:
            fl = obj_fn(candidate)
            X_hist.append(candidate)
            y_hist.append(fl)
            if fl < bl: 
                bl = fl
                x_best = candidate.copy()
            curve.append(bl)
            ec += 1

        new_delta_f = f_old - bl
        mab.update(mab_A, mab_B, new_delta_f)

        best_history.pop(0)
        best_history.append(bl)

    return {'loss': bl, 'curve': curve, 'seed': seed, 'x_best': x_best.tolist() if x_best is not None else None}
