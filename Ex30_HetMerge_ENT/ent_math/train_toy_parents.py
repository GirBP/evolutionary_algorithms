#!/usr/bin/env python3
"""Pre-train toy parents ONCE. Save to results/toy_parents.pth."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, time
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets

DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

raw = datasets.CIFAR10('/tmp/cifar10', train=True, download=True)
raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_tr = (torch.tensor(raw.data).permute(0,3,1,2).float()/255 - mean)/std
y_tr = torch.tensor(raw.targets)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255 - mean)/std
y_te = torch.tensor(raw_te.targets)

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + x)

class TinyResNet(nn.Module):
    def __init__(self, nc=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.block1 = ResidualBlock(16)
        self.block2 = ResidualBlock(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, nc)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

clA, clB = [0,1], [2,3]

def train_parent(cls_list, seed, epochs=20):
    torch.manual_seed(seed)
    mask = torch.zeros(len(y_tr), dtype=torch.bool)
    for c in cls_list: mask |= (y_tr == c)
    Xp, yp = X_tr[mask][:5000].to(DEV), y_tr[mask][:5000].to(DEV)
    
    m = TinyResNet(len(cls_list)).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=0.003)
    m.train()
    for ep in range(epochs):
        for i in range(0, len(Xp), 128):
            out = m(Xp[i:i+128])
            # Remap labels to 0..len(cls_list)-1
            targets = yp[i:i+128].clone()
            for new_idx, old_cls in enumerate(cls_list):
                targets[yp[i:i+128] == old_cls] = new_idx
            loss = nn.CrossEntropyLoss()(out, targets)
            opt.zero_grad(); loss.backward(); opt.step()
    
    m.eval()
    pc = {}
    with torch.no_grad():
        for c in cls_list:
            mask_c = y_te == c
            if mask_c.sum() == 0: continue
            preds = m(X_te[mask_c].to(DEV)).argmax(1).cpu()
            ci = cls_list.index(c)
            pc[c] = (preds == ci).float().mean().item()
    
    return m.cpu(), pc

t0 = time.time()
print("Training toy parents...")
mA, pcA = train_parent(clA, 42)
mB, pcB = train_parent(clB, 142)
print(f"  A: {pcA}")
print(f"  B: {pcB}")
print(f"  Time: {time.time()-t0:.1f}s")

torch.save({
    'sdA': mA.state_dict(), 'sdB': mB.state_dict(),
    'pcA': pcA, 'pcB': pcB,
    'clA': clA, 'clB': clB,
}, 'results/toy_parents.pth')
print("Saved: results/toy_parents.pth")
