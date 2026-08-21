#!/usr/bin/env python3
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
"""
SCENARIO 2: Branch-Train-Merge — WA vs NeuronConcat+CMA
=========================================================
2 specialists: A(cls 0-4), B(cls 5-9). Training identical.
Only merge differs: WA+concatFC vs NeuronConcat+CMA biases.
CMA-ES uses 500-sample calibration set.

~8 min on T4.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, json, cma
from torchvision import datasets, transforms, models

DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
t0 = time.time()
clA, clB = list(range(5)), list(range(5,10))

# ── Data ──
tfm = transforms.Compose([transforms.RandomCrop(32,4), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
tfm_t = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
train_ds = datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=tfm)
test_ds = datasets.CIFAR10('/tmp/cifar10', train=False, download=True, transform=tfm_t)
test_ldr = torch.utils.data.DataLoader(test_ds, 256, num_workers=0)
# Cal set: 500 train images (not test)
cal_sub = torch.utils.data.Subset(
    datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=tfm_t),
    torch.randperm(50000)[:500].tolist())
cal_ldr = torch.utils.data.DataLoader(cal_sub, 256, num_workers=0)

def make_rn18(nc=5):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool = nn.Identity()
    m.fc = nn.Linear(512,nc); return m

def per_class(model, ldr, classes=range(10)):
    model.eval(); cor={c:0 for c in classes}; tot={c:0 for c in classes}
    with torch.no_grad():
        for x,y in ldr:
            p = model(x.to(DEV)).argmax(1).cpu()
            for c in classes:
                m = y==c; tot[c]+=m.sum().item(); cor[c]+=(p[m]==c).sum().item()
    return {c: cor[c]/max(tot[c],1) for c in classes}

# ── Train specialists (shared) ──
def train_spec(cls_list, seed):
    remap = {c:i for i,c in enumerate(cls_list)}
    torch.manual_seed(seed)
    m = make_rn18(len(cls_list)).to(DEV)
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 15)
    ldr = torch.utils.data.DataLoader(train_ds, 128, shuffle=True, num_workers=0)
    m.train()
    for ep in range(15):
        for x,y in ldr:
            mask = sum(y==c for c in cls_list).bool()
            if mask.sum()==0: continue
            yy = torch.tensor([remap[v.item()] for v in y[mask]])
            loss = F.cross_entropy(m(x[mask].to(DEV)), yy.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    m.eval(); return m

print("Training specialists...")
pA = train_spec(clA, SEED)
pB = train_spec(clB, SEED+100)
# Eval parents (remapped → original labels)
rmA = {i:clA[i] for i in range(5)}; rmB = {i:clB[i] for i in range(5)}
parent_pc = {}
pA.eval()
with torch.no_grad():
    for x,y in test_ldr:
        pr = pA(x.to(DEV)).argmax(1).cpu()
        for c in clA:
            m = y==c; parent_pc.setdefault(c,[0,0])
            parent_pc[c][1] += m.sum().item()
            parent_pc[c][0] += (pr[m]==clA.index(c)).sum().item()
pB.eval()
with torch.no_grad():
    for x,y in test_ldr:
        pr = pB(x.to(DEV)).argmax(1).cpu()
        for c in clB:
            m = y==c; parent_pc.setdefault(c,[0,0])
            parent_pc[c][1] += m.sum().item()
            parent_pc[c][0] += (pr[m]==clB.index(c)).sum().item()
parent_pc = {c: parent_pc[c][0]/max(parent_pc[c][1],1) for c in range(10)}
print(f"Parents: A={[round(parent_pc[c],3) for c in clA]} B={[round(parent_pc[c],3) for c in clB]}")

sdA = {k:v.cpu() for k,v in pA.state_dict().items()}
sdB = {k:v.cpu() for k,v in pB.state_dict().items()}
del pA, pB; torch.cuda.empty_cache()
print(f"Training done ({time.time()-t0:.0f}s)")

# ════════ MERGE 1: WA + concat FC ════════
t1 = time.time()
sd_wa = {}
for k in sdA:
    if 'fc' in k: continue
    if 'num_batches_tracked' in k: sd_wa[k] = sdA[k]
    else: sd_wa[k] = 0.5*sdA[k].float() + 0.5*sdB[k].float()
fc_w = torch.zeros(10,512); fc_b = torch.zeros(10)
for i,c in enumerate(clA): fc_w[c]=sdA['fc.weight'][i]; fc_b[c]=sdA['fc.bias'][i]
for i,c in enumerate(clB): fc_w[c]=sdB['fc.weight'][i]; fc_b[c]=sdB['fc.bias'][i]
sd_wa['fc.weight']=fc_w; sd_wa['fc.bias']=fc_b
t1 = time.time()-t1

mWA = make_rn18(10).to(DEV)
mWA.load_state_dict({k:v.to(DEV) for k,v in sd_wa.items()})
wa_pc = per_class(mWA, test_ldr)
del mWA

# ════════ MERGE 2: NeuronConcat + CMA-ES ════════
t2 = time.time()
# Build NeuronConcat: keep conv layers separate via 2-forward ensemble
# Simpler than doubled model: run both parents, CMA optimizes FC mixing
#
# Architecture: shared input → parentA backbone → 512 features
#                             → parentB backbone → 512 features
#               concat [1024] → FC [1024→10] with CMA-ES biases+weights
class DualBackbone(nn.Module):
    def __init__(self, sdA, sdB):
        super().__init__()
        self.netA = make_rn18(5); self.netA.load_state_dict(sdA)
        self.netB = make_rn18(5); self.netB.load_state_dict(sdB)
        # Freeze backbones
        for p in self.netA.parameters(): p.requires_grad = False
        for p in self.netB.parameters(): p.requires_grad = False
        # Merged FC
        self.fc = nn.Linear(1024, 10)
        # Init: A features→cls 0-4, B features→cls 5-9
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)
        with torch.no_grad():
            for i,c in enumerate(clA):
                self.fc.weight[c,:512] = sdA['fc.weight'][i]
                self.fc.bias[c] = sdA['fc.bias'][i]
            for i,c in enumerate(clB):
                self.fc.weight[c,512:] = sdB['fc.weight'][i]
                self.fc.bias[c] = sdB['fc.bias'][i]

    def features(self, x):
        # Extract features before FC
        def feat(net, x):
            x = F.relu(net.bn1(net.conv1(x)))
            x = net.layer1(x); x = net.layer2(x)
            x = net.layer3(x); x = net.layer4(x)
            return net.avgpool(x).flatten(1)
        return torch.cat([feat(self.netA, x), feat(self.netB, x)], dim=1)

    def forward(self, x):
        return self.fc(self.features(x))

db = DualBackbone(sdA, sdB).to(DEV).eval()
db_pc_raw = per_class(db, test_ldr)
ret_raw = sum(1 for c in range(10) if parent_pc[c]>0 and db_pc_raw[c]/parent_pc[c]>=0.9)
print(f"NeuronConcat raw (before CMA): {ret_raw}/10")

# CMA-ES: optimize bias (10D) — minimal, targeted
orig_bias = db.fc.bias.detach().cpu().numpy().tolist()

def fitness(bvec):
    db.fc.bias.data = torch.tensor(bvec, dtype=torch.float32).to(DEV)
    pc = per_class(db, cal_ldr)  # calibration set!
    ret = sum(1 for c in range(10) if parent_pc[c]>0 and pc[c]/parent_pc[c]>=0.9)
    drop = np.mean([(1-pc[c]/max(parent_pc[c],1e-9))*100 for c in range(10)])
    return -ret*10 + drop

es = cma.CMAEvolutionStrategy(orig_bias, 0.3,
    {'maxiter':100, 'popsize':16, 'seed':SEED, 'verbose':-1})
best_f, best_x = 1e9, orig_bias[:]
while not es.stop():
    sols = es.ask(); fits = [fitness(s) for s in sols]
    es.tell(sols, fits)
    bf = min(fits)
    if bf < best_f: best_f=bf; best_x=sols[fits.index(bf)][:]
t2 = time.time()-t2

db.fc.bias.data = torch.tensor(best_x, dtype=torch.float32).to(DEV)
nc_pc = per_class(db, test_ldr)

# ════════ Results ════════
def retention(ppc, mpc):
    r = sum(1 for c in range(10) if ppc[c]>0 and mpc[c]/ppc[c]>=0.9)
    d = np.mean([(1-mpc[c]/max(ppc[c],1e-9))*100 for c in range(10)])
    return r, d

ret_wa, drop_wa = retention(parent_pc, wa_pc)
ret_nc, drop_nc = retention(parent_pc, nc_pc)

print(f"\n{'='*70}")
print(f"  SCENARIO 2: Branch-Train-Merge")
print(f"{'='*70}")
print(f"  {'':>6} | {'Method':>25} | {'Ret':>5} | {'Drop':>7} | {'Merge t':>8} | {'Params':>8}")
print(f"  {'-'*6}-+-{'-'*25}-+-{'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")
print(f"  {'STD':>6} | {'WA + concat FC':>25} | {ret_wa:>3}/10 | {drop_wa:>6.1f}% | {t1:>7.3f}s | {'11.2M':>8}")
print(f"  {'OURS':>6} | {'NeuronConcat + CMA 10D':>25} | {ret_nc:>3}/10 | {drop_nc:>6.1f}% | {t2:>7.1f}s | {'22.4M':>8}")
print(f"  {'Δ':>6} | {'':>25} | {ret_nc-ret_wa:>+3}   | {drop_nc-drop_wa:>+6.1f}% |")

print(f"\n  {'Cls':>3} | {'Parent':>7} | {'WA':>7} | {'NC+CMA':>7} | {'WA':>6} | {'NC':>6}")
print(f"  {'-'*3}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}")
for c in range(10):
    p=parent_pc[c]; wa=wa_pc[c]; nc=nc_pc[c]
    dw=(1-wa/p)*100 if p>0 else 100; dn=(1-nc/p)*100 if p>0 else 100
    print(f"  {c:>3} | {p:>7.4f} | {wa:>7.4f} | {nc:>7.4f} | {dw:>+5.1f}% | {dn:>+5.1f}%")

print(f"\n  CMA-ES overhead: {t2:.1f}s ({100*16:.0f} cal-set evals)")
print(f"  Total: {time.time()-t0:.0f}s")
json.dump({'parent':{str(c):round(parent_pc[c],4) for c in range(10)},
    'wa':{'ret':f'{ret_wa}/10','drop':round(drop_wa,1),
        'pc':{str(c):round(wa_pc[c],4) for c in range(10)},'merge_s':round(t1,3)},
    'nc_cma':{'ret':f'{ret_nc}/10','drop':round(drop_nc,1),
        'pc':{str(c):round(nc_pc[c],4) for c in range(10)},'merge_s':round(t2,1)},
    'delta_ret':ret_nc-ret_wa},
    open('scenario2_results.json','w'), indent=2)
