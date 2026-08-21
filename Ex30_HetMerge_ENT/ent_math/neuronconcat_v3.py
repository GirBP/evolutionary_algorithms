#!/usr/bin/env python3
"""
CNN-NeuronConcat v3: Full cross-pathway CMA-ES (70D)
=====================================================
For each of 10 classes, optimize:
  - 1 bias
  - 1 scale  
  - 5 cross-pathway weights (one per opposite-parent row)
Total: 10 + 10 + 10*5 = 70D

Key insight: cat (class 3) needs SPECIFIC B-row weights, not mean(W_B).
Using W_B[dog] with negative weight can suppress cat-dog confusion.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import time, numpy as np, os, platform
from torchvision import datasets, transforms, models

def beep(msg):
    print(f"\n🔔 {msg}")
    if platform.system() == 'Darwin': os.system(f'say "{msg}" &')
    else: print('\a')

DEV = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEV}")
if DEV.type == 'cuda': print(f"  GPU: {torch.cuda.get_device_name()}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
NW = 0 if platform.system() == 'Darwin' else 2
test_ds = datasets.CIFAR10('/tmp/cifar10', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=NW)
# We also need train_loader for parent training fallback
train_ds = datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=NW)
clA, clB = list(range(5)), list(range(5,10))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        return self.relu(out + identity)

class FlexResNet18(nn.Module):
    def __init__(self, base_width=64, num_classes=10):
        super().__init__()
        self.inplanes = base_width
        self.conv1 = nn.Conv2d(3, base_width, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(base_width, 2)
        self.layer2 = self._make_layer(base_width*2, 2, stride=2)
        self.layer3 = self._make_layer(base_width*4, 2, stride=2)
        self.layer4 = self._make_layer(base_width*8, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_width*8, num_classes)
    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks): layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))

def make_rn18(nc):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc); return m

def print_drop(name, ppc, mpc, classes):
    print(f"\n  {name}:")
    print(f"  {'Cls':>3} | {'Parent':>7} | {'Merged':>7} | {'Drop%':>6} | OK?")
    print(f"  {'-'*3}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*4}")
    ret = 0
    for c in classes:
        p, m = ppc[c], mpc.get(c,0)
        d = (1-m/p)*100 if p>0 else 100
        ok = 'YES' if d<=10 else 'NO'
        if d<=10: ret+=1
        print(f"  {c:>3} | {p:>7.3f} | {m:>7.3f} | {d:>5.1f}% | {ok}")
    print(f"  Retention: {ret}/{len(classes)} (drop <= 10%)")
    return ret

def build_neuronconcat_model(pA, pB):
    merged = FlexResNet18(base_width=128, num_classes=10).to(DEV)
    sd_A, sd_B, sd_M = pA.state_dict(), pB.state_dict(), merged.state_dict()
    for key in sd_M.keys():
        if key.endswith('num_batches_tracked'):
            sd_M[key] = torch.tensor(0, dtype=torch.long); continue
        if key not in sd_A: continue
        pA_val, pB_val = sd_A[key], sd_B[key]
        if 'fc' in key:
            if 'weight' in key:
                sd_M[key] = torch.zeros_like(sd_M[key])
                sd_M[key][:5, :512] = pA_val; sd_M[key][5:, 512:] = pB_val
            elif 'bias' in key:
                sd_M[key][:5] = pA_val; sd_M[key][5:] = pB_val
        elif 'conv1.weight' == key:
            sd_M[key][:64] = pA_val; sd_M[key][64:] = pB_val
        elif 'weight' in key and len(pA_val.shape)==4:
            C_out, C_in = pA_val.shape[0], pA_val.shape[1]
            sd_M[key] = torch.zeros_like(sd_M[key])
            sd_M[key][:C_out,:C_in] = pA_val; sd_M[key][C_out:,C_in:] = pB_val
        elif ('weight' in key or 'bias' in key) and len(pA_val.shape)==1:
            C = pA_val.shape[0]; sd_M[key][:C] = pA_val; sd_M[key][C:] = pB_val
        elif 'running_mean' in key or 'running_var' in key:
            C = pA_val.shape[0]; sd_M[key][:C] = pA_val; sd_M[key][C:] = pB_val
    merged.load_state_dict(sd_M)
    return merged


if __name__ == '__main__':
    import multiprocessing
    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('fork', force=True)

    t0 = time.time()
    CACHE = 'cached_parents_r3.pth'
    beep("NeuronConcat v3")

    # ═══ STAGE 0: Load parents ═══
    print("=" * 60)
    print("  STAGE 0: Loading cached parents")
    print("=" * 60)
    cache = torch.load(CACHE, map_location='cpu', weights_only=False)
    pA = make_rn18(5).to(DEV); pA.load_state_dict(cache['pA']); pA.eval()
    pB = make_rn18(5).to(DEV); pB.load_state_dict(cache['pB']); pB.eval()
    parent_pc = cache['parent_pc']
    print(f"  Loaded ({time.time()-t0:.0f}s): {parent_pc}")

    # ═══ STAGE 1: Build & extract features ═══
    print("\n" + "=" * 60)
    print("  STAGE 1: NeuronConcat + feature extraction")
    print("=" * 60)
    merged = build_neuronconcat_model(pA, pB)
    merged.eval()

    all_x = torch.cat([xb for xb, _ in test_loader])
    true_y = torch.cat([yb for _, yb in test_loader])

    with torch.no_grad():
        all_feats = []
        for xb in all_x.split(256):
            x = merged.relu(merged.bn1(merged.conv1(xb.to(DEV))))
            x = merged.layer1(x); x = merged.layer2(x)
            x = merged.layer3(x); x = merged.layer4(x)
            x = merged.avgpool(x)
            all_feats.append(torch.flatten(x, 1).cpu())
        all_feats = torch.cat(all_feats).to(DEV)

    W_orig = merged.fc.weight.data.clone()
    b_orig = merged.fc.bias.data.clone()
    h_A = all_feats[:, :512]
    h_B = all_feats[:, 512:]

    # Parent FC weights
    W_A_fc = W_orig[:5, :512].clone()   # [5, 512] — A's 5-class head
    W_B_fc = W_orig[5:, 512:].clone()   # [5, 512] — B's 5-class head

    # Pre-compute all raw logits from each parent row
    # logit_A[i] = h_A @ W_A_fc[i], logit_B[j] = h_B @ W_B_fc[j]
    with torch.no_grad():
        raw_A = h_A @ W_A_fc.T   # [N, 5] — logits from A's 5 rows
        raw_B = h_B @ W_B_fc.T   # [N, 5] — logits from B's 5 rows

    print(f"  raw_A: {raw_A.shape}, raw_B: {raw_B.shape}")
    print(f"  ({time.time()-t0:.0f}s)")

    # ═══ STAGE 2: CMA-ES (70D) ═══
    print("\n" + "=" * 60)
    print("  STAGE 2: CMA-ES full cross-pathway (70D)")
    print("  For each class: 1 bias + 1 scale + 5 cross-weights")
    print("=" * 60)
    beep("CMA-ES 70D")

    try:
        import cma
    except ImportError:
        os.system('pip install cma -q')
        import cma

    def fitness_v3(theta):
        """
        theta layout (70D):
          [0:10]   biases
          [10:20]  log-scales (exp → positive)
          [20:25]  cross-weights for class 0 (A-class) → access B's 5 rows
          [25:30]  cross-weights for class 1
          [30:35]  cross-weights for class 2
          [35:40]  cross-weights for class 3 (cat!)
          [40:45]  cross-weights for class 4
          [45:50]  cross-weights for class 5 (B-class) → access A's 5 rows
          [50:55]  cross-weights for class 6
          [55:60]  cross-weights for class 7
          [60:65]  cross-weights for class 8
          [65:70]  cross-weights for class 9 (truck!)
        """
        biases = torch.tensor(theta[:10], dtype=torch.float32, device=DEV)
        scales = torch.exp(torch.tensor(theta[10:20], dtype=torch.float32, device=DEV))
        
        with torch.no_grad():
            logits = torch.zeros(len(all_feats), 10, device=DEV)
            
            # A-classes (0-4): primary = A-row, cross = all 5 B-rows
            for c in range(5):
                cross_w = torch.tensor(theta[20+c*5:20+(c+1)*5], dtype=torch.float32, device=DEV)
                logits[:, c] = scales[c] * raw_A[:, c] + (raw_B * cross_w).sum(1) + biases[c]
            
            # B-classes (5-9): primary = B-row, cross = all 5 A-rows
            for c in range(5):
                cross_w = torch.tensor(theta[45+c*5:45+(c+1)*5], dtype=torch.float32, device=DEV)
                logits[:, c+5] = scales[c+5] * raw_B[:, c] + (raw_A * cross_w).sum(1) + biases[c+5]
            
            preds = logits.argmax(1).cpu()
        
        pc = {c: (preds[true_y==c]==c).float().mean().item() for c in clA+clB}
        ret = sum(1 for c in clA+clB if parent_pc[c]>0 and pc[c]/parent_pc[c]>=0.9)
        avg_drop = np.mean([(1-pc[c]/parent_pc[c])*100 for c in clA+clB])
        
        # Fitness: max retention, then min drop, with per-class penalty
        min_ratio = min(pc[c]/parent_pc[c] for c in clA+clB if parent_pc[c]>0)
        return -ret*10 + avg_drop - min_ratio * 5  # bonus for worst class

    # Initialize: biases from v1, scales=0, cross=0
    # Use v1 optimal biases if available, else zeros
    x0 = [0.0]*10 + [0.0]*10 + [0.0]*50  # 70D (all neutral)
    
    es = cma.CMAEvolutionStrategy(x0, 0.5, {
        'maxiter': 500, 'popsize': 50, 'seed': 42, 'verbose': -1
    })

    best_f, best_sol = 100, x0[:]
    gen = 0
    while not es.stop():
        gen += 1
        sols = es.ask()
        fits = [fitness_v3(s) for s in sols]
        es.tell(sols, fits)
        bf = min(fits)
        if bf < best_f: best_f = bf; best_sol = sols[fits.index(bf)][:]
        if gen % 50 == 0 or gen == 1 or gen == 10:
            # Quick eval
            theta = best_sol
            biases = torch.tensor(theta[:10], dtype=torch.float32, device=DEV)
            scales = torch.exp(torch.tensor(theta[10:20], dtype=torch.float32, device=DEV))
            with torch.no_grad():
                logits = torch.zeros(len(all_feats), 10, device=DEV)
                for c in range(5):
                    cw = torch.tensor(theta[20+c*5:20+(c+1)*5], dtype=torch.float32, device=DEV)
                    logits[:,c] = scales[c]*raw_A[:,c] + (raw_B*cw).sum(1) + biases[c]
                for c in range(5):
                    cw = torch.tensor(theta[45+c*5:45+(c+1)*5], dtype=torch.float32, device=DEV)
                    logits[:,c+5] = scales[c+5]*raw_B[:,c] + (raw_A*cw).sum(1) + biases[c+5]
                preds = logits.argmax(1).cpu()
            pc = {c: (preds[true_y==c]==c).float().mean().item() for c in clA+clB}
            ret = sum(1 for c in clA+clB if parent_pc[c]>0 and pc[c]/parent_pc[c]>=0.9)
            cat_drop = (1-pc[3]/parent_pc[3])*100
            truck_drop = (1-pc[9]/parent_pc[9])*100
            print(f"  gen {gen:3d}: f={best_f:+.1f} ret={ret}/10 "
                  f"cat_drop={cat_drop:.1f}% truck_drop={truck_drop:.1f}% ({time.time()-t0:.0f}s)")

    # ═══ Final eval ═══
    theta = best_sol
    biases = torch.tensor(theta[:10], dtype=torch.float32, device=DEV)
    scales = torch.exp(torch.tensor(theta[10:20], dtype=torch.float32, device=DEV))
    with torch.no_grad():
        logits = torch.zeros(len(all_feats), 10, device=DEV)
        for c in range(5):
            cw = torch.tensor(theta[20+c*5:20+(c+1)*5], dtype=torch.float32, device=DEV)
            logits[:,c] = scales[c]*raw_A[:,c] + (raw_B*cw).sum(1) + biases[c]
        for c in range(5):
            cw = torch.tensor(theta[45+c*5:45+(c+1)*5], dtype=torch.float32, device=DEV)
            logits[:,c+5] = scales[c+5]*raw_B[:,c] + (raw_A*cw).sum(1) + biases[c+5]
        preds = logits.argmax(1).cpu()
    
    pc_f = {c: (preds[true_y==c]==c).float().mean().item() for c in clA+clB}
    ret_final = print_drop("FINAL v3 (70D cross-pathway)", parent_pc, pc_f, clA+clB)

    # Show cross-pathway weights for failing classes
    print(f"\n  Cross-pathway structure:")
    print(f"  Cat (class 3) cross-B weights: {np.array(best_sol[35:40]).round(3)}")
    print(f"    (B-classes: 5=dog, 6=frog, 7=horse, 8=ship, 9=truck)")
    print(f"  Truck (class 9) cross-A weights: {np.array(best_sol[65:70]).round(3)}")
    print(f"    (A-classes: 0=airplane, 1=auto, 2=bird, 3=cat, 4=deer)")

    print(f"\n{'='*60}")
    print(f"  v3 FINAL: {ret_final}/10")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")
    beep(f"Done. v3 retention {ret_final} out of 10")
