#!/usr/bin/env python3
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
"""
SCENARIO 1: Local SGD — FedAvg vs CMA-ES Reduce
=================================================
4 workers train ResNet-18 on CIFAR-10 shards (same classes).
Training is IDENTICAL for both methods.
Only the REDUCE step differs: WA vs CMA-ES per-layer α.
CMA-ES uses small calibration set (500 images), not full test.

~8 min on T4.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, json, cma
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
# Calibration: 500 images for CMA-ES fitness (NOT test set)
cal_idx = torch.randperm(len(train_ds))[:500].tolist()
cal_sub = torch.utils.data.Subset(
    datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=tfm_t), cal_idx)
cal_ldr = torch.utils.data.DataLoader(cal_sub, 256, num_workers=0)

# 4 shards
N_W = 4; perm = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(SEED))
shards = [torch.utils.data.Subset(train_ds, perm[i*len(train_ds)//N_W:(i+1)*len(train_ds)//N_W].tolist())
          for i in range(N_W)]
shard_ldrs = [torch.utils.data.DataLoader(s, 128, shuffle=True, num_workers=0) for s in shards]

def make_rn18():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool = nn.Identity()
    m.fc = nn.Linear(512,10); return m

def accuracy(model, loader):
    model.eval(); c=t=0
    with torch.no_grad():
        for x,y in loader:
            c += (model(x.to(DEV)).argmax(1).cpu()==y).sum().item(); t += len(y)
    return c/t

def per_class_acc(model, loader):
    model.eval(); cor={i:0 for i in range(10)}; tot={i:0 for i in range(10)}
    with torch.no_grad():
        for x,y in loader:
            p = model(x.to(DEV)).argmax(1).cpu()
            for c in range(10):
                m = y==c; tot[c]+=m.sum().item(); cor[c]+=(p[m]==c).sum().item()
    return {c: cor[c]/max(tot[c],1) for c in range(10)}

# ── Train 4 workers (shared computation) ──
print(f"Training {N_W} workers × 10 epochs each...")
torch.manual_seed(SEED)
init_sd = {k:v.cpu() for k,v in make_rn18().to(DEV).state_dict().items()}

worker_sds = []
for w in range(N_W):
    m = make_rn18().to(DEV)
    m.load_state_dict({k:v.to(DEV) for k,v in init_sd.items()})
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    m.train()
    for ep in range(10):
        for x,y in shard_ldrs[w]:
            loss = F.cross_entropy(m(x.to(DEV)), y.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    sd = {k:v.cpu() for k,v in m.state_dict().items()}
    worker_sds.append(sd)
    acc = accuracy(m, test_ldr)
    print(f"  Worker {w}: {acc:.4f} ({time.time()-t0:.0f}s)")
    del m
torch.cuda.empty_cache()

# ── Reduce 1: FedAvg (WA) ──
t_wa = time.time()
sd_wa = {}
for k in worker_sds[0]:
    if 'num_batches_tracked' in k: sd_wa[k] = worker_sds[0][k]
    else: sd_wa[k] = torch.stack([s[k].float() for s in worker_sds]).mean(0)
t_wa = time.time() - t_wa

m = make_rn18().to(DEV); m.load_state_dict({k:v.to(DEV) for k,v in sd_wa.items()})
wa_acc = accuracy(m, test_ldr)
wa_pc = per_class_acc(m, test_ldr)
del m

# ── Reduce 2: CMA-ES per-layer α ──
t_cma = time.time()
groups = {}
for k in init_sd:
    if 'num_batches_tracked' in k: continue
    p = k.split('.')
    g = p[0] if not p[0].startswith('layer') else f"{p[0]}.{p[1]}"
    groups.setdefault(g, []).append(k)
gnames = sorted(groups.keys())

def fitness(x):
    alphas = 1.0/(1.0+np.exp(-np.array(x)))
    sd = {}
    for gi,g in enumerate(gnames):
        ws = [worker_sds[w] for w in range(N_W)]
        for k in groups[g]:
            sd[k] = sum(alphas[gi*N_W+w]*ws[w][k].float() for w in range(N_W))
            sd[k] /= sum(alphas[gi*N_W:gi*N_W+N_W])
    for k in init_sd:
        if 'num_batches_tracked' in k: sd[k] = worker_sds[0][k]
    m = make_rn18().to(DEV)
    m.load_state_dict({k:v.to(DEV) for k,v in sd.items()})
    acc = accuracy(m, cal_ldr)  # CALIBRATION set, not test
    del m; return -acc

nd = len(gnames)*N_W
es = cma.CMAEvolutionStrategy([0.0]*nd, 0.3,
    {'maxiter':30, 'popsize':12, 'seed':SEED, 'verbose':-1})
best_f, best_x = 1e9, [0.0]*nd
while not es.stop():
    sols = es.ask(); fits = [fitness(s) for s in sols]
    es.tell(sols, fits)
    bf = min(fits)
    if bf < best_f: best_f=bf; best_x=sols[fits.index(bf)][:]
t_cma = time.time() - t_cma

# Build CMA result
alphas = 1.0/(1.0+np.exp(-np.array(best_x)))
sd_cma = {}
for gi,g in enumerate(gnames):
    for k in groups[g]:
        sd_cma[k] = sum(alphas[gi*N_W+w]*worker_sds[w][k].float() for w in range(N_W))
        sd_cma[k] /= sum(alphas[gi*N_W:gi*N_W+N_W])
for k in init_sd:
    if 'num_batches_tracked' in k: sd_cma[k] = worker_sds[0][k]
m = make_rn18().to(DEV); m.load_state_dict({k:v.to(DEV) for k,v in sd_cma.items()})
cma_acc = accuracy(m, test_ldr)
cma_pc = per_class_acc(m, test_ldr)
del m

# ── Results ──
print(f"\n{'='*60}")
print(f"  SCENARIO 1: Local SGD — Reduce Step Comparison")
print(f"{'='*60}")
print(f"  {'':>15} | {'FedAvg':>8} | {'CMA-ES':>8} | {'Δ':>8}")
print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
print(f"  {'Accuracy':>15} | {wa_acc:>8.4f} | {cma_acc:>8.4f} | {cma_acc-wa_acc:>+8.4f}")
print(f"  {'Reduce time':>15} | {t_wa:>7.2f}s | {t_cma:>7.2f}s |")
print(f"  {'Reduce evals':>15} | {'0':>8} | {30*12:>8} |")
print(f"\n  Per-class:")
for c in range(10):
    print(f"    cls {c}: FedAvg={wa_pc[c]:.4f}  CMA={cma_pc[c]:.4f}  Δ={cma_pc[c]-wa_pc[c]:+.4f}")
print(f"\n  Total: {time.time()-t0:.0f}s")
json.dump({'wa_acc':wa_acc,'cma_acc':cma_acc,'delta':round(cma_acc-wa_acc,4),
    'wa_pc':{str(c):round(wa_pc[c],4) for c in range(10)},
    'cma_pc':{str(c):round(cma_pc[c],4) for c in range(10)},
    't_wa':round(t_wa,2),'t_cma':round(t_cma,2)},
    open('scenario1_results.json','w'), indent=2)
