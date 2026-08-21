"""Vanilla CMA-ES — Classical Evolutionary Strategy (No Surrogate)"""
import numpy as np

def run(seed, obj_fn, dim, budget):
    """
    Оригінальний метод CMA-ES (на основі Hansen). 
    Без сурогату він робить реальні виклики нейромережі під час мутації.
    Ідеально показує бідний "Sample Efficiency" класичної еволюції при обмежених бюджетах N=50.
    """
    rng = np.random.default_rng(seed)
    
    # ── Initialization ───────────────────────────────────────────────────
    mean = rng.random(dim)
    sigma = 0.3
    lam = 10
    mu = lam // 2
    
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights**2)
    
    cc = 4.0 / (dim + 4)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2.0 / ((dim + 1.3)**2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    
    pc, ps = np.zeros(dim), np.zeros(dim)
    C = np.eye(dim)
    chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))
    
    curve = []
    bl = float('inf')
    ec = 0
    
    # Початкова оцінка
    fl = obj_fn(mean)
    ec += 1
    bl = fl
    curve.append(bl)
    
    # ── Evolution Loop ───────────────────────────────────────────────────
    while ec < budget:
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            sqrtC = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            invsqrtC = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
        except Exception:
            sqrtC, invsqrtC = np.eye(dim), np.eye(dim)
            
        arz = rng.standard_normal((lam, dim))
        arx = np.array([np.clip(mean + sigma * sqrtC @ z, 0, 1) for z in arz])
        
        fits = []
        for x in arx:
            if ec >= budget:
                break
            y = obj_fn(x)
            fits.append(y)
            if y < bl: bl = y
            curve.append(bl)
            ec += 1
            
        if len(fits) < lam:
            break
            
        fits = np.array(fits)
        idx = np.argsort(fits)
        
        old_mean = mean.copy()
        mean = sum(weights[i] * arx[idx[i]] for i in range(mu))
        
        ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * invsqrtC @ (mean - old_mean) / sigma
        hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(ec//lam+1))) < (1.4+2/(dim+1))*chiN
        pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (mean - old_mean) / sigma
        artmp = (arx[idx[:mu]] - old_mean) / sigma
        
        C = (1-c1-cmu)*C + c1*(np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C)
        for i in range(mu):
            C += cmu * weights[i] * np.outer(artmp[i], artmp[i])
            
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = np.clip(sigma, 1e-8, 0.5)

    return {'loss': bl, 'curve': curve[:budget], 'seed': seed}
