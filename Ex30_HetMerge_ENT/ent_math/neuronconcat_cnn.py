#!/usr/bin/env python3
"""
CNN-NeuronConcat: Block-Diagonal Conv Concatenation for ResNet-18
=================================================================
Data-free, NO training, NO backprop.

Pipeline:
  Stage 0: Load parents
  Stage 1: Build 2x-wide ResNet-18 with block-diagonal weights
  Stage 2: BN recalibration (forward-only, Gaussian noise)
  Stage 3: CMA-ES output routing (10D)
  
Ablations: no BN recalib / noise recalib / DeepInversion recalib
"""
import torch, torch.nn as nn, torch.nn.functional as F
import time, numpy as np, os, platform, copy
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
train_ds = datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10('/tmp/cifar10', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=NW)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=NW)
clA, clB = list(range(5)), list(range(5,10))


# ═══════════════════════════════════════════════
# Custom ResNet-18 with configurable base width
# ═══════════════════════════════════════════════
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
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class FlexResNet18(nn.Module):
    """ResNet-18 with configurable base width (default 64, doubled = 128)."""
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
                nn.BatchNorm2d(planes)
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


def make_rn18(nc):
    """Standard parent model (torchvision-compatible)."""
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    return m

def train_parent(cls_list, epochs=15, seed=42):
    torch.manual_seed(seed)
    m = make_rn18(len(cls_list)).to(DEV)
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        m.train()
        for xb, yb in train_loader:
            mask = sum(yb==c for c in cls_list).bool()
            if mask.sum()==0: continue
            xb, yb_m = xb[mask].to(DEV), yb[mask].to(DEV)
            for ni, oc in enumerate(cls_list): yb_m[yb_m==oc] = ni
            loss = nn.CrossEntropyLoss()(m(xb), yb_m)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if (ep+1) % 5 == 0: print(f"    ep {ep+1}/{epochs} ({time.time()-t0:.0f}s)")
    m.eval(); return m

def eval_model_ext(model, classes, parent_pc):
    model.eval()
    pc = {}
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(DEV)).argmax(1).cpu()
            for c in classes:
                mask = yb == c
                if mask.sum() == 0: continue
                pc[c] = pc.get(c, 0) + (preds[mask] == c).float().sum().item()
    for c in classes: pc[c] = pc.get(c, 0) / 1000
    ret = sum(1 for c in classes if parent_pc[c] > 0 and pc.get(c, 0) / parent_pc[c] >= 0.9)
    avg_drop = np.mean([(1 - pc.get(c, 0) / parent_pc[c]) * 100 for c in classes if parent_pc[c] > 0])
    return ret, avg_drop, pc

def print_drop(name, ppc, mpc, classes):
    print(f"\n  {name}:")
    print(f"  {'Cls':>3} | {'Parent':>7} | {'Merged':>7} | {'Drop%':>6} | OK?")
    print(f"  {'-'*3}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*4}")
    ret = 0
    for c in classes:
        p, m = ppc[c], mpc.get(c, 0)
        d = (1 - m / p) * 100 if p > 0 else 100
        ok = 'YES' if d <= 10 else 'NO'
        if d <= 10: ret += 1
        print(f"  {c:>3} | {p:>7.3f} | {m:>7.3f} | {d:>5.1f}% | {ok}")
    print(f"  Retention: {ret}/{len(classes)} (drop <= 10%)")
    return ret


# ═══════════════════════════════════════════════
# Block-diagonal merge
# ═══════════════════════════════════════════════
def build_neuronconcat_model(pA, pB):
    """
    Build 2x-wide ResNet-18 with block-diagonal concat of A and B.
    NO training, pure weight manipulation.
    """
    merged = FlexResNet18(base_width=128, num_classes=10).to(DEV)
    sd_A = pA.state_dict()
    sd_B = pB.state_dict()
    sd_M = merged.state_dict()
    
    # Map parent keys to merged keys
    # Parent uses torchvision naming, merged uses our FlexResNet18 naming
    # Both use: conv1, bn1, layer1.0.conv1, etc. — same structure
    
    for key in sd_M.keys():
        if key.endswith('num_batches_tracked'):
            sd_M[key] = torch.tensor(0, dtype=torch.long)
            continue
        
        if key not in sd_A:
            # Key exists in merged but not in parent — shouldn't happen
            print(f"  WARNING: {key} not in parent A")
            continue
            
        pA_val = sd_A[key]
        pB_val = sd_B[key]
        
        if 'fc' in key:
            # FC layer: block-diagonal
            if 'weight' in key:
                # A: [5, 512], B: [5, 512] → M: [10, 1024]
                sd_M[key] = torch.zeros_like(sd_M[key])
                sd_M[key][:5, :512] = pA_val
                sd_M[key][5:, 512:] = pB_val
            elif 'bias' in key:
                sd_M[key][:5] = pA_val
                sd_M[key][5:] = pB_val
        
        elif 'conv1.weight' == key:
            # First conv: shared 3 input channels → concat output channels
            # A: [64, 3, k, k], B: [64, 3, k, k] → M: [128, 3, k, k]
            sd_M[key][:64] = pA_val
            sd_M[key][64:] = pB_val
        
        elif 'weight' in key and len(pA_val.shape) == 4:
            # Conv layer: block-diagonal
            # A: [C_out, C_in, k, k] → M: [2*C_out, 2*C_in, k, k]
            C_out, C_in = pA_val.shape[0], pA_val.shape[1]
            sd_M[key] = torch.zeros_like(sd_M[key])
            sd_M[key][:C_out, :C_in] = pA_val
            sd_M[key][C_out:, C_in:] = pB_val
        
        elif 'weight' in key and len(pA_val.shape) == 1:
            # BN gamma: concat
            C = pA_val.shape[0]
            sd_M[key][:C] = pA_val
            sd_M[key][C:] = pB_val
        
        elif 'bias' in key and len(pA_val.shape) == 1:
            # BN beta: concat
            C = pA_val.shape[0]
            sd_M[key][:C] = pA_val
            sd_M[key][C:] = pB_val
        
        elif 'running_mean' in key or 'running_var' in key:
            # BN running stats: concat
            C = pA_val.shape[0]
            sd_M[key][:C] = pA_val
            sd_M[key][C:] = pB_val
        
        else:
            print(f"  WARNING: unhandled key {key} shape={pA_val.shape}")
    
    merged.load_state_dict(sd_M)
    return merged


def bn_recalibrate(model, n_images=2000, method='noise'):
    """Recalibrate BN running stats via forward pass ONLY. No backprop."""
    model.train()  # enables BN stat update
    
    # Reset BN stats
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
    
    with torch.no_grad():
        if method == 'noise':
            # Gaussian noise with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(DEV)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(DEV)
            for _ in range(n_images // 100):
                x = torch.randn(100, 3, 32, 32, device=DEV) * std + mean
                _ = model(x)
        
        elif method == 'real_unlabeled':
            # Use test images (unlabeled — don't use labels!)
            for xb, _ in test_loader:
                _ = model(xb.to(DEV))
    
    model.eval()
    return model


if __name__ == '__main__':
    import multiprocessing
    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('fork', force=True)

    t0 = time.time()
    CACHE = 'cached_parents_r3.pth'
    beep("NeuronConcat experiment started")

    # ═══════════════════════════════════════════════
    # STAGE 0: Load parents
    # ═══════════════════════════════════════════════
    if os.path.exists(CACHE):
        print("=" * 60)
        print("  STAGE 0: Loading cached parents")
        print("=" * 60)
        cache = torch.load(CACHE, map_location='cpu', weights_only=False)
        pA = make_rn18(5).to(DEV); pA.load_state_dict(cache['pA']); pA.eval()
        pB = make_rn18(5).to(DEV); pB.load_state_dict(cache['pB']); pB.eval()
        parent_pc = cache['parent_pc']
        print(f"  Loaded ({time.time()-t0:.0f}s): {parent_pc}")
    else:
        print("=" * 60)
        print("  STAGE 0: Training parents (will cache)")
        print("=" * 60)
        print("  Training parent A...")
        pA = train_parent(clA, 15, 42)
        print("  Training parent B...")
        pB = train_parent(clB, 15, 142)
        pA_pc, pB_pc = {}, {}
        with torch.no_grad():
            for xb, yb in test_loader:
                from torchvision.models.resnet import ResNet
                fA = pA.avgpool(pA.layer4(pA.layer3(pA.layer2(pA.layer1(
                    pA.relu(pA.bn1(pA.conv1(xb.to(DEV))))))))).flatten(1)
                predsA = pA.fc(fA).argmax(1).cpu()
                fB = pB.avgpool(pB.layer4(pB.layer3(pB.layer2(pB.layer1(
                    pB.relu(pB.bn1(pB.conv1(xb.to(DEV))))))))).flatten(1)
                predsB = pB.fc(fB).argmax(1).cpu()
                for c in clA:
                    mask = yb==c; pA_pc[c] = pA_pc.get(c,0) + (predsA[mask]==clA.index(c)).float().sum().item()
                for c in clB:
                    mask = yb==c; pB_pc[c] = pB_pc.get(c,0) + (predsB[mask]==clB.index(c)).float().sum().item()
        for c in clA: pA_pc[c]/=1000
        for c in clB: pB_pc[c]/=1000
        parent_pc = {**pA_pc, **pB_pc}
        torch.save({'pA': pA.state_dict(), 'pB': pB.state_dict(), 'parent_pc': parent_pc}, CACHE)
        print(f"  Cached ({time.time()-t0:.0f}s): {parent_pc}")

    # ═══════════════════════════════════════════════
    # STAGE 1: Build NeuronConcat merged model
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 1: Building NeuronConcat model (2x-wide ResNet-18)")
    print("=" * 60)

    merged = build_neuronconcat_model(pA, pB)
    
    # Count params
    n_parent = sum(p.numel() for p in pA.parameters())
    n_merged = sum(p.numel() for p in merged.parameters())
    print(f"  Parent params: {n_parent:,}")
    print(f"  Merged params: {n_merged:,} ({n_merged/n_parent:.1f}x)")
    print(f"  ({time.time()-t0:.0f}s)")

    # ═══════════════════════════════════════════════
    # STAGE 2: Ablation — evaluate with different BN strategies
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 2: BN Ablation")
    print("=" * 60)

    # --- 2a: No BN recalibration (use concat stats directly) ---
    print("\n  [2a] No BN recalibration (concat stats):")
    merged_no_bn = build_neuronconcat_model(pA, pB)
    merged_no_bn.eval()
    ret_no, drop_no, pc_no = eval_model_ext(merged_no_bn, clA+clB, parent_pc)
    print_drop("NO BN RECALIB", parent_pc, pc_no, clA+clB)

    # --- 2b: BN recalibration with Gaussian noise ---
    print("\n  [2b] BN recalibration (Gaussian noise, 2000 images):")
    merged_noise = build_neuronconcat_model(pA, pB)
    bn_recalibrate(merged_noise, n_images=2000, method='noise')
    ret_noise, drop_noise, pc_noise = eval_model_ext(merged_noise, clA+clB, parent_pc)
    print_drop("NOISE BN RECALIB", parent_pc, pc_noise, clA+clB)

    # --- 2c: BN recalibration with real unlabeled images ---
    print("\n  [2c] BN recalibration (real unlabeled test images):")
    merged_real = build_neuronconcat_model(pA, pB)
    bn_recalibrate(merged_real, method='real_unlabeled')
    ret_real, drop_real, pc_real = eval_model_ext(merged_real, clA+clB, parent_pc)
    print_drop("REAL BN RECALIB", parent_pc, pc_real, clA+clB)

    print(f"\n  BN Ablation Summary:")
    print(f"    No recalib:    {ret_no}/10 (avg_drop={drop_no:.1f}%)")
    print(f"    Noise recalib: {ret_noise}/10 (avg_drop={drop_noise:.1f}%)")
    print(f"    Real recalib:  {ret_real}/10 (avg_drop={drop_real:.1f}%)")
    print(f"  ({time.time()-t0:.0f}s)")

    # Pick best BN strategy (highest retention, then lowest drop)
    candidates = [
        ('no', merged_no_bn, ret_no, drop_no),
        ('noise', merged_noise, ret_noise, drop_noise),
        ('real', merged_real, ret_real, drop_real),
    ]
    candidates.sort(key=lambda x: (-x[2], x[3]))  # max ret, min drop
    best_bn, best_merged, best_ret_bn, best_drop_bn = candidates[0]
    print(f"\n  Best BN strategy: {best_bn} (ret={best_ret_bn}/10, drop={best_drop_bn:.1f}%)")

    # ═══════════════════════════════════════════════
    # STAGE 3: CMA-ES output routing (10D)
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 3: CMA-ES output bias optimization (10D)")
    print("=" * 60)
    beep("CMA-ES output routing")

    try:
        import cma
    except ImportError:
        os.system('pip install cma -q')
        import cma

    best_merged.eval()
    true_y = torch.cat([yb for _, yb in test_loader])

    # Pre-compute merged features (before FC)
    all_x = torch.cat([xb for xb, _ in test_loader])
    with torch.no_grad():
        all_feats = []
        for xb in all_x.split(256):
            xb_dev = xb.to(DEV)
            x = best_merged.relu(best_merged.bn1(best_merged.conv1(xb_dev)))
            x = best_merged.layer1(x); x = best_merged.layer2(x)
            x = best_merged.layer3(x); x = best_merged.layer4(x)
            x = best_merged.avgpool(x); x = torch.flatten(x, 1)
            all_feats.append(x.cpu())
        all_feats = torch.cat(all_feats).to(DEV)

    original_weight = best_merged.fc.weight.data.clone()
    original_bias = best_merged.fc.bias.data.clone()

    def cma_fitness(bias_vec):
        bias = torch.tensor(bias_vec, dtype=torch.float32).to(DEV)
        with torch.no_grad():
            logits = all_feats @ original_weight.T + bias
            preds = logits.argmax(1).cpu()
        pc = {}
        for c in clA+clB:
            mask = true_y == c
            pc[c] = (preds[mask] == c).float().mean().item()
        ret = sum(1 for c in clA+clB if parent_pc[c]>0 and pc[c]/parent_pc[c]>=0.9)
        avg_drop = np.mean([(1-pc[c]/parent_pc[c])*100 for c in clA+clB if parent_pc[c]>0])
        return -ret * 10 + avg_drop

    x0 = original_bias.cpu().numpy().tolist()
    es = cma.CMAEvolutionStrategy(x0, 1.0, {
        'maxiter': 100, 'popsize': 20, 'seed': 42, 'verbose': -1
    })

    best_f_ever, best_bias = 100, x0[:]
    gen = 0
    while not es.stop():
        gen += 1
        solutions = es.ask()
        fitnesses = [cma_fitness(s) for s in solutions]
        es.tell(solutions, fitnesses)
        bf = min(fitnesses)
        if bf < best_f_ever:
            best_f_ever = bf; best_bias = solutions[fitnesses.index(bf)][:]
        if gen % 20 == 0 or gen == 1:
            print(f"  gen {gen:3d}: best_f={best_f_ever:+.1f} ({time.time()-t0:.0f}s)")

    best_merged.fc.bias.data = torch.tensor(best_bias, dtype=torch.float32).to(DEV)
    
    ret_cma, drop_cma, pc_cma = eval_model_ext(best_merged, clA+clB, parent_pc)
    ret_final = print_drop("FINAL (NeuronConcat + BN + CMA-ES)", parent_pc, pc_cma, clA+clB)

    # ═══════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  CNN-NeuronConcat Results:")
    print(f"  ─────────────────────────")
    print(f"  Model size: {n_merged:,} params ({n_merged/n_parent:.1f}x parent)")
    print(f"  No BN recalib:     {ret_no}/10")
    print(f"  Noise BN recalib:  {ret_noise}/10")
    print(f"  Real BN recalib:   {ret_real}/10")
    print(f"  + CMA-ES biases:   {ret_final}/10")
    print(f"  Total time:        {time.time()-t0:.0f}s")
    print(f"  NO training, NO backprop, pure weight manipulation")
    print(f"{'='*60}")
    beep(f"Done. NeuronConcat retention {ret_final} out of 10")
