#!/usr/bin/env python3
"""Train strong parents: ResNet-18 fine-tune conv1+layer3+layer4+fc on MPS.
5 epochs, 2000/class. Train both parents for ONE seed. Save .pth.
Usage: python3 train_strong.py <seed>
"""
import numpy as np, torch, torch.nn as nn, random, json, time, sys
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import torchvision.models as models
from torchvision import datasets

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SEED = int(sys.argv[1])
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
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
print(f"Data: {time.time()-t0:.1f}s, MPS={DEVICE}", flush=True)

def make_rn(nc=5):
    m = models.resnet18(weights='IMAGENET1K_V1')
    m.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    nn.init.kaiming_normal_(m.conv1.weight, mode='fan_out', nonlinearity='relu')
    nn.init.kaiming_normal_(m.fc.weight); nn.init.zeros_(m.fc.bias)
    return m

def train_parent(cls_list, seed_p, epochs=5, lr=0.001, npc=2000):
    torch.manual_seed(seed_p); np.random.seed(seed_p); random.seed(seed_p)
    m = make_rn(len(cls_list))
    cls_map = {c:i for i,c in enumerate(cls_list)}

    mask = sum(y_tr==c for c in cls_list).bool()
    Xs = X_tr[mask]; ys = torch.tensor([cls_map[y.item()] for y in y_tr[mask]])
    idx = torch.cat([torch.where(ys==i)[0][:npc] for i in range(len(cls_list))])
    Xs, ys = Xs[idx], ys[idx]

    # Freeze layer1+layer2+bn1 only
    for n, p in m.named_parameters(): p.requires_grad = False
    for n, p in m.named_parameters():
        if any(x in n for x in ['conv1', 'layer3', 'layer4', 'fc']):
            p.requires_grad = True

    m = m.to(DEVICE)
    opt = torch.optim.Adam(filter(lambda p:p.requires_grad, m.parameters()), lr=lr)
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
        print(f"    ep {ep+1}/{epochs}", flush=True)

    m = m.to('cpu').eval()
    mask_te = sum(y_te==c for c in cls_list).bool()
    Xt = X_te[mask_te]
    yt = torch.tensor([cls_map[y.item()] for y in y_te[mask_te]])
    with torch.no_grad():
        preds = torch.cat([m(Xt[i:i+512]).argmax(1) for i in range(0,len(Xt),512)])
        acc = (preds==yt).float().mean().item()
        pc = {cls_list[i]:round((preds[yt==i]==i).float().mean().item(),3) for i in range(len(cls_list))}
    return m, acc, pc

print(f"--- Training Parents (seed={SEED}) ---", flush=True)

t1 = time.time()
mA, accA, pcA = train_parent(clA, SEED, epochs=5)
tA = time.time()-t1
print(f"  ParentA: {accA:.3f} ({tA:.0f}s) pc={pcA}", flush=True)
torch.save(mA.state_dict(), f'results/parentA_s{SEED}.pth')

t1 = time.time()
mB, accB, pcB = train_parent(clB, SEED+10000, epochs=5)
tB = time.time()-t1
print(f"  ParentB: {accB:.3f} ({tB:.0f}s) pc={pcB}", flush=True)
torch.save(mB.state_dict(), f'results/parentB_s{SEED}.pth')

elapsed = time.time()-t0
# Accumulate
fpath = 'results/parents_strong.json'
try:
    with open(fpath) as f: accum = json.load(f)
except: accum = {}
accum[str(SEED)] = {'A': round(accA,4), 'B': round(accB,4), 'pcA': pcA, 'pcB': pcB, 'time': round(elapsed,1)}
with open(fpath, 'w') as f: json.dump(accum, f, indent=2)

print(f"\nTotal: {elapsed:.0f}s")
print(f"metric_parentA: {accA:.4f}")
print(f"metric_parentB: {accB:.4f}")
both_pass = accA >= 0.88 and accB >= 0.88
print(f"metric_pass_88: {'YES' if both_pass else 'NO'}")
print("Done!", flush=True)
