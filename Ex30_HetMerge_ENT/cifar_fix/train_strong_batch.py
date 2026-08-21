#!/usr/bin/env python3
"""Train strong parents for multiple seeds. ~120s per parent → 2 seeds per 420s.
Usage: python3 train_strong_batch.py <seed1> <seed2> [seed3...]
"""
import numpy as np, torch, torch.nn as nn, random, json, time, sys
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import torchvision.models as models
from torchvision import datasets

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SEEDS = [int(s) for s in sys.argv[1:]]
t0 = time.time()

raw_tr = datasets.CIFAR10('/tmp/cifar10', train=True, download=True)
raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mn = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
sd = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_tr = (torch.tensor(raw_tr.data).permute(0,3,1,2).float()/255.0-mn)/sd
y_tr = torch.tensor(raw_tr.targets)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255.0-mn)/sd
y_te = torch.tensor(raw_te.targets)
clA, clB = list(range(5)), list(range(5,10))
print(f"Data: {time.time()-t0:.1f}s, MPS={DEVICE}, seeds={SEEDS}", flush=True)

def make_rn(nc=5):
    m = models.resnet18(weights='IMAGENET1K_V1')
    m.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    nn.init.kaiming_normal_(m.conv1.weight, mode='fan_out', nonlinearity='relu')
    nn.init.kaiming_normal_(m.fc.weight); nn.init.zeros_(m.fc.bias)
    return m

def train_parent(cls_list, seed_p, epochs=5, npc=2000):
    torch.manual_seed(seed_p); np.random.seed(seed_p); random.seed(seed_p)
    m = make_rn(len(cls_list))
    cls_map = {c:i for i,c in enumerate(cls_list)}
    mask = sum(y_tr==c for c in cls_list).bool()
    Xs = X_tr[mask]; ys = torch.tensor([cls_map[y.item()] for y in y_tr[mask]])
    idx = torch.cat([torch.where(ys==i)[0][:npc] for i in range(len(cls_list))])
    Xs, ys = Xs[idx], ys[idx]
    for n, p in m.named_parameters(): p.requires_grad = False
    for n, p in m.named_parameters():
        if any(x in n for x in ['conv1', 'layer3', 'layer4', 'fc']): p.requires_grad = True
    m = m.to(DEVICE)
    opt = torch.optim.Adam(filter(lambda p:p.requires_grad, m.parameters()), lr=0.001)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    m.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xs))
        for i in range(0, len(Xs), 256):
            ix = perm[i:i+256]
            xb = Xs[ix].to(DEVICE); yb = ys[ix].to(DEVICE)
            if torch.rand(1).item() > 0.5: xb = xb.flip(-1)
            loss = nn.CrossEntropyLoss()(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    m = m.to('cpu').eval()
    mask_te = sum(y_te==c for c in cls_list).bool()
    Xt = X_te[mask_te]; yt = torch.tensor([cls_map[y.item()] for y in y_te[mask_te]])
    with torch.no_grad():
        preds = torch.cat([m(Xt[i:i+512]).argmax(1) for i in range(0,len(Xt),512)])
        acc = (preds==yt).float().mean().item()
        pc = {cls_list[i]:round((preds[yt==i]==i).float().mean().item(),3) for i in range(len(cls_list))}
    return m, acc, pc

fpath = 'results/parents_strong.json'
try:
    with open(fpath) as f: accum = json.load(f)
except: accum = {}

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---", flush=True)
    t1 = time.time()
    mA, accA, pcA = train_parent(clA, seed, epochs=5)
    print(f"  A: {accA:.3f} ({time.time()-t1:.0f}s)", flush=True)
    torch.save(mA.state_dict(), f'results/parentA_s{seed}.pth')

    t1 = time.time()
    mB, accB, pcB = train_parent(clB, seed+10000, epochs=5)
    print(f"  B: {accB:.3f} ({time.time()-t1:.0f}s)", flush=True)
    torch.save(mB.state_dict(), f'results/parentB_s{seed}.pth')

    accum[str(seed)] = {'A': round(accA,4), 'B': round(accB,4), 'pcA': pcA, 'pcB': pcB}
    with open(fpath, 'w') as f: json.dump(accum, f, indent=2)

    bp = 'YES' if accA>=0.88 and accB>=0.88 else 'NO'
    print(f"  pass_88: {bp}", flush=True)
    print(f"metric_s{seed}_A: {accA:.4f}")
    print(f"metric_s{seed}_B: {accB:.4f}")

print(f"\nTotal: {time.time()-t0:.0f}s")
print("Done!", flush=True)
