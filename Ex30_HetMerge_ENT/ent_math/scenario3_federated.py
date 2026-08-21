#!/usr/bin/env python3
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
"""
SCENARIO 3: Federated Learning (non-IID) — FedAvg vs PCMA-Fed
===============================================================
5 clients with non-IID data (each has 2 dominant classes).
10 rounds of local training + reduce.

PCMA-Fed applies CMA-ES only every 3 rounds (amortized cost).
CMA-ES uses 500-sample calibration set.

~15 min on T4.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, json, cma, copy
from torchvision import datasets, transforms, models

DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
t0 = time.time()

# ── Data ──
tfm = transforms.Compose([transforms.RandomCrop(32,4), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
tfm_t = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
train_ds = datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=tfm)
test_ds = datasets.CIFAR10('/tmp/cifar10', train=False, download=True, transform=tfm_t)
test_ldr = torch.utils.data.DataLoader(test_ds, 256, num_workers=0)
cal_sub = torch.utils.data.Subset(
    datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=tfm_t),
    torch.randperm(50000, generator=torch.Generator().manual_seed(99))[:500].tolist())
cal_ldr = torch.utils.data.DataLoader(cal_sub, 256, num_workers=0)

# non-IID: client i dominant classes = [2i, 2i+1]
N_C = 5; N_ROUNDS = 10; LOCAL_EP = 3
targets = np.array(train_ds.targets)
client_ldrs = []
assignments = {}
for ci in range(N_C):
    major = [2*ci, 2*ci+1]; assignments[ci] = major
    major_idx = np.concatenate([np.where(targets==c)[0][:4000] for c in major])
    minor_cls = [c for c in range(10) if c not in major]
    minor_idx = np.concatenate([np.where(targets==c)[0][:250] for c in minor_cls])
    all_idx = np.concatenate([major_idx, minor_idx])
    np.random.shuffle(all_idx)
    sub = torch.utils.data.Subset(train_ds, all_idx.tolist())
    client_ldrs.append(torch.utils.data.DataLoader(sub, 64, shuffle=True, num_workers=0))
    print(f"  Client {ci}: {len(all_idx)} imgs, major={major}")

def make_rn18():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool = nn.Identity()
    m.fc = nn.Linear(512,10); return m

def accuracy(model, ldr):
    model.eval(); c=t=0
    with torch.no_grad():
        for x,y in ldr: c+=(model(x.to(DEV)).argmax(1).cpu()==y).sum().item(); t+=len(y)
    return c/t

def per_class(model, ldr):
    model.eval(); cor={i:0 for i in range(10)}; tot={i:0 for i in range(10)}
    with torch.no_grad():
        for x,y in ldr:
            p = model(x.to(DEV)).argmax(1).cpu()
            for c in range(10): m=y==c; tot[c]+=m.sum().item(); cor[c]+=(p[m]==c).sum().item()
    return {c: cor[c]/max(tot[c],1) for c in range(10)}

def train_local(model, ldr, ep):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
    for _ in range(ep):
        for x,y in ldr:
            loss = F.cross_entropy(model(x.to(DEV)), y.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); return model

def reduce_wa(sds):
    out = {}
    for k in sds[0]:
        if 'num_batches_tracked' in k: out[k] = sds[0][k]
        else: out[k] = torch.stack([s[k].float().cpu() for s in sds]).mean(0)
    return out

def reduce_cma(sds, prev_sd):
    """CMA-ES: optimize per-layer weighted combination of clients."""
    avg_sd = reduce_wa(sds)
    groups = {}
    for k in prev_sd:
        if 'num_batches_tracked' in k: continue
        p = k.split('.')
        g = p[0] if not p[0].startswith('layer') else f"{p[0]}.{p[1]}"
        groups.setdefault(g, []).append(k)
    gnames = sorted(groups.keys())

    def fitness(x):
        a = 1.0/(1.0+np.exp(-np.array(x)))
        sd = {}
        for gi,g in enumerate(gnames):
            for k in groups[g]:
                sd[k] = a[gi]*prev_sd[k].float().cpu() + (1-a[gi])*avg_sd[k].float()
        for k in prev_sd:
            if 'num_batches_tracked' in k: sd[k] = avg_sd[k]
        m = make_rn18().to(DEV)
        m.load_state_dict({k:v.to(DEV) for k,v in sd.items()})
        acc = accuracy(m, cal_ldr); del m; return -acc

    nd = len(gnames)
    es = cma.CMAEvolutionStrategy([0.0]*nd, 0.3,
        {'maxiter':25,'popsize':10,'seed':SEED,'verbose':-1})
    best_f, best_x = 1e9, [0.0]*nd
    while not es.stop():
        sols = es.ask(); fits = [fitness(s) for s in sols]
        es.tell(sols, fits)
        bf = min(fits)
        if bf < best_f: best_f=bf; best_x=sols[fits.index(bf)][:]

    a = 1.0/(1.0+np.exp(-np.array(best_x)))
    sd = {}
    for gi,g in enumerate(gnames):
        for k in groups[g]:
            sd[k] = a[gi]*prev_sd[k].float().cpu() + (1-a[gi])*avg_sd[k].float()
    for k in prev_sd:
        if 'num_batches_tracked' in k: sd[k] = avg_sd[k]
    return sd

# ── Main loop ──
print(f"\n{'='*60}")
print(f"  SCENARIO 3: Federated Learning (non-IID)")
print(f"{'='*60}\n")

torch.manual_seed(SEED)
init_sd = {k:v.cpu() for k,v in make_rn18().to(DEV).state_dict().items()}
fa_global = copy.deepcopy(init_sd)
pc_global = copy.deepcopy(init_sd)

rounds_data = []
fa_reduce_total = 0.0
pc_reduce_total = 0.0

for rnd in range(N_ROUNDS):
    # ── Local training (shared computation pattern) ──
    fa_sds = []; pc_sds = []
    for ci in range(N_C):
        # FedAvg client
        m = make_rn18().to(DEV)
        m.load_state_dict({k:v.to(DEV) for k,v in fa_global.items()})
        m = train_local(m, client_ldrs[ci], LOCAL_EP)
        fa_sds.append({k:v.cpu() for k,v in m.state_dict().items()}); del m
        # PCMA client
        m = make_rn18().to(DEV)
        m.load_state_dict({k:v.to(DEV) for k,v in pc_global.items()})
        m = train_local(m, client_ldrs[ci], LOCAL_EP)
        pc_sds.append({k:v.cpu() for k,v in m.state_dict().items()}); del m

    # ── Reduce: FedAvg ──
    t_r = time.time()
    fa_global = reduce_wa(fa_sds)
    fa_reduce_total += time.time() - t_r

    # ── Reduce: PCMA-Fed (CMA only every 3 rounds) ──
    t_r = time.time()
    if (rnd+1) % 3 == 0:
        pc_global = reduce_cma(pc_sds, pc_global)
    else:
        pc_global = reduce_wa(pc_sds)
    pc_reduce_total += time.time() - t_r

    # ── Eval ──
    m = make_rn18().to(DEV)
    m.load_state_dict({k:v.to(DEV) for k,v in fa_global.items()})
    fa_acc = accuracy(m, test_ldr); fa_pc = per_class(m, test_ldr); del m

    m = make_rn18().to(DEV)
    m.load_state_dict({k:v.to(DEV) for k,v in pc_global.items()})
    pc_acc = accuracy(m, test_ldr); pc_pc = per_class(m, test_ldr); del m

    cma_flag = "CMA" if (rnd+1)%3==0 else "WA"
    print(f"  R{rnd+1:>2}: FedAvg={fa_acc:.4f}  PCMA={pc_acc:.4f} Δ={pc_acc-fa_acc:+.4f} [{cma_flag}] ({time.time()-t0:.0f}s)")

    rounds_data.append({
        'round': rnd+1, 'fedavg_acc': round(fa_acc,4), 'pcma_acc': round(pc_acc,4),
        'delta': round(pc_acc-fa_acc,4), 'reduce_type': cma_flag,
        'fedavg_pc': {str(c):round(fa_pc[c],4) for c in range(10)},
        'pcma_pc': {str(c):round(pc_pc[c],4) for c in range(10)}})

    torch.cuda.empty_cache()

# ── Final ──
print(f"\n{'='*60}")
print(f"  FINAL COMPARISON")
print(f"{'='*60}")
f = rounds_data[-1]
print(f"  {'':>15} | {'FedAvg':>8} | {'PCMA-Fed':>8} | {'Δ':>8}")
print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
print(f"  {'Final acc':>15} | {f['fedavg_acc']:>8.4f} | {f['pcma_acc']:>8.4f} | {f['delta']:>+8.4f}")
print(f"  {'Reduce time':>15} | {fa_reduce_total:>7.2f}s | {pc_reduce_total:>7.2f}s |")
print(f"  {'CMA rounds':>15} | {'0':>8} | {N_ROUNDS//3:>8} |")

print(f"\n  Per-class (final):")
for c in range(10):
    fa=f['fedavg_pc'][str(c)]; pc=f['pcma_pc'][str(c)]
    maj = [ci for ci,cl in assignments.items() if c in cl]
    print(f"    cls {c}: FA={fa:.4f} PCMA={pc:.4f} Δ={pc-fa:+.4f}  (major: client {maj})")

print(f"\n  Total: {time.time()-t0:.0f}s")
json.dump({'rounds':rounds_data, 'fa_reduce_s':round(fa_reduce_total,2),
    'pc_reduce_s':round(pc_reduce_total,2)},
    open('scenario3_results.json','w'), indent=2)
