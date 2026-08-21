#!/usr/bin/env python3
"""E09-prep: Pre-compute and SAVE teacher logits to disk using MPS.
This avoids recomputing every run (saves 95s).
"""
import torch, torch.nn as nn
import torchvision.models as models
import time
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import datasets

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
t0 = time.time()

raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mn = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
sd = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255-mn)/sd

def make_rn(nc=5):
    m = models.resnet18(weights='IMAGENET1K_V1')
    m.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    return m

# Load and compute on MPS
pA = make_rn(5)
pA.load_state_dict(torch.load('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/cifar_fix/results/parentA_s42.pth',
                               weights_only=True, map_location='cpu'))
pA.to(DEVICE).eval()
with torch.no_grad():
    tA = torch.cat([pA(X_te[i:i+128].to(DEVICE)).cpu() for i in range(0,len(X_te),128)])
del pA
print(f"tA computed ({time.time()-t0:.1f}s): {tA.shape}", flush=True)

pB = make_rn(5)
pB.load_state_dict(torch.load('/Users/bibo/Desktop/cs_dev/Ex30_HetMerge/cifar_fix/results/parentB_s42.pth',
                               weights_only=True, map_location='cpu'))
pB.to(DEVICE).eval()
with torch.no_grad():
    tB = torch.cat([pB(X_te[i:i+128].to(DEVICE)).cpu() for i in range(0,len(X_te),128)])
del pB
print(f"tB computed ({time.time()-t0:.1f}s): {tB.shape}", flush=True)

torch.save({'tA': tA, 'tB': tB}, 'results/teacher_logits.pth')
print(f"Saved results/teacher_logits.pth ({time.time()-t0:.1f}s)")
print("Done!", flush=True)
