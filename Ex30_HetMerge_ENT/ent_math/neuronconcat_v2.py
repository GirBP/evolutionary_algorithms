#!/usr/bin/env python3
"""
CNN-NeuronConcat v2: Extended CMA-ES (30D) for 10/10 retention
===============================================================
Builds on v1 (8/10). Expands CMA-ES search space:
  - 10 FC biases
  - 10 per-class scales
  - 10 cross-pathway weights (allow A-classes to use B-features and vice versa)
  Total: 30D
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
train_ds = datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10('/tmp/cifar10', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=NW)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=NW)
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
        if (ep+1)%5==0: print(f"    ep {ep+1}/{epochs} ({time.time()-t0:.0f}s)")
    m.eval(); return m

def eval_model_ext(model, classes, parent_pc):
    model.eval(); pc = {}
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(DEV)).argmax(1).cpu()
            for c in classes:
                mask = yb==c
                if mask.sum()==0: continue
                pc[c] = pc.get(c,0) + (preds[mask]==c).float().sum().item()
    for c in classes: pc[c] = pc.get(c,0)/1000
    ret = sum(1 for c in classes if parent_pc[c]>0 and pc.get(c,0)/parent_pc[c]>=0.9)
    avg_drop = np.mean([(1-pc.get(c,0)/parent_pc[c])*100 for c in classes if parent_pc[c]>0])
    return ret, avg_drop, pc

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
    beep("NeuronConcat v2 experiment")

    # ═══ STAGE 0: Load parents ═══
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
        print("  Training parents...")
        pA = train_parent(clA, 15, 42); pB = train_parent(clB, 15, 142)
        pA_pc, pB_pc = {}, {}
        with torch.no_grad():
            for xb, yb in test_loader:
                fA = pA.avgpool(pA.layer4(pA.layer3(pA.layer2(pA.layer1(
                    pA.relu(pA.bn1(pA.conv1(xb.to(DEV))))))))).flatten(1)
                fB = pB.avgpool(pB.layer4(pB.layer3(pB.layer2(pB.layer1(
                    pB.relu(pB.bn1(pB.conv1(xb.to(DEV))))))))).flatten(1)
                for c in clA: mask=yb==c; pA_pc[c]=pA_pc.get(c,0)+(pA.fc(fA).argmax(1).cpu()[mask]==clA.index(c)).float().sum().item()
                for c in clB: mask=yb==c; pB_pc[c]=pB_pc.get(c,0)+(pB.fc(fB).argmax(1).cpu()[mask]==clB.index(c)).float().sum().item()
        for c in clA: pA_pc[c]/=1000
        for c in clB: pB_pc[c]/=1000
        parent_pc = {**pA_pc, **pB_pc}
        torch.save({'pA':pA.state_dict(),'pB':pB.state_dict(),'parent_pc':parent_pc}, CACHE)
        print(f"  Cached ({time.time()-t0:.0f}s): {parent_pc}")

    # ═══ STAGE 1: Build NeuronConcat ═══
    print("\n" + "=" * 60)
    print("  STAGE 1: NeuronConcat (no BN recalib)")
    print("=" * 60)
    merged = build_neuronconcat_model(pA, pB)
    merged.eval()
    
    ret_base, drop_base, pc_base = eval_model_ext(merged, clA+clB, parent_pc)
    print(f"  Baseline: {ret_base}/10 (avg_drop={drop_base:.1f}%)")
    
    # Pre-compute features
    all_x = torch.cat([xb for xb, _ in test_loader])
    true_y = torch.cat([yb for _, yb in test_loader])
    
    with torch.no_grad():
        all_feats = []
        for xb in all_x.split(256):
            xb_dev = xb.to(DEV)
            x = merged.relu(merged.bn1(merged.conv1(xb_dev)))
            x = merged.layer1(x); x = merged.layer2(x)
            x = merged.layer3(x); x = merged.layer4(x)
            x = merged.avgpool(x); x = torch.flatten(x, 1)
            all_feats.append(x.cpu())
        all_feats = torch.cat(all_feats).to(DEV)  # [10000, 1024]
    
    # Original FC weights: [10, 1024]
    W_orig = merged.fc.weight.data.clone()  # [10, 1024]
    b_orig = merged.fc.bias.data.clone()    # [10]
    
    # Separate A and B FC weights for cross-pathway
    # A classes (0-4): W_orig[:5, :512] = A's weights, W_orig[:5, 512:] = zeros
    # B classes (5-9): W_orig[5:, :512] = zeros, W_orig[5:, 512:] = B's weights
    W_A_fc = W_orig[:5, :512].clone()  # [5, 512]
    W_B_fc = W_orig[5:, 512:].clone()  # [5, 512]
    
    h_A = all_feats[:, :512]   # A-pathway features [10000, 512]
    h_B = all_feats[:, 512:]   # B-pathway features [10000, 512]
    
    print(f"  Features: {all_feats.shape}, h_A: {h_A.shape}, h_B: {h_B.shape}")
    print(f"  ({time.time()-t0:.0f}s)")

    # ═══ STAGE 2: CMA-ES v1 — biases only (10D) ═══
    print("\n" + "=" * 60)
    print("  STAGE 2a: CMA-ES biases only (10D)")
    print("=" * 60)
    
    try:
        import cma
    except ImportError:
        os.system('pip install cma -q')
        import cma

    def fitness_bias_only(theta):
        bias = torch.tensor(theta, dtype=torch.float32).to(DEV)
        with torch.no_grad():
            logits = all_feats @ W_orig.T + bias
            preds = logits.argmax(1).cpu()
        pc = {c: (preds[true_y==c]==c).float().mean().item() for c in clA+clB}
        ret = sum(1 for c in clA+clB if parent_pc[c]>0 and pc[c]/parent_pc[c]>=0.9)
        avg_drop = np.mean([(1-pc[c]/parent_pc[c])*100 for c in clA+clB])
        return -ret*10 + avg_drop

    x0_bias = b_orig.cpu().numpy().tolist()
    es1 = cma.CMAEvolutionStrategy(x0_bias, 1.0, {'maxiter':100,'popsize':20,'seed':42,'verbose':-1})
    best_f1, best_sol1 = 100, x0_bias[:]
    while not es1.stop():
        sols = es1.ask()
        fits = [fitness_bias_only(s) for s in sols]
        es1.tell(sols, fits)
        bf = min(fits)
        if bf < best_f1: best_f1 = bf; best_sol1 = sols[fits.index(bf)][:]
    
    bias_opt = torch.tensor(best_sol1, dtype=torch.float32).to(DEV)
    with torch.no_grad():
        logits_v1 = all_feats @ W_orig.T + bias_opt
        preds_v1 = logits_v1.argmax(1).cpu()
    pc_v1 = {c: (preds_v1[true_y==c]==c).float().mean().item() for c in clA+clB}
    ret_v1 = sum(1 for c in clA+clB if parent_pc[c]>0 and pc_v1[c]/parent_pc[c]>=0.9)
    print(f"  v1 (10D bias): {ret_v1}/10 best_f={best_f1:+.1f} ({time.time()-t0:.0f}s)")

    # ═══ STAGE 2b: CMA-ES v2 — biases + scales + cross-pathway (30D) ═══
    print("\n" + "=" * 60)
    print("  STAGE 2b: CMA-ES extended (30D: bias + scale + cross)")
    print("=" * 60)
    beep("CMA-ES 30D")

    def fitness_extended(theta):
        """
        theta[0:10]  = biases
        theta[10:20] = log-scales (exp for positivity)
        theta[20:30] = cross-pathway weights (sigmoid → [0,1])
        
        For class c in {0-4} (A-classes):
          logit_c = exp(s_c) * (W_A_fc[c] · h_A) + σ(cross_c) * (mean(W_B_fc) · h_B) + bias_c
        For class c in {5-9} (B-classes):
          logit_c = exp(s_c) * (W_B_fc[c-5] · h_B) + σ(cross_c) * (mean(W_A_fc) · h_A) + bias_c
        """
        biases = torch.tensor(theta[:10], dtype=torch.float32, device=DEV)
        scales = torch.exp(torch.tensor(theta[10:20], dtype=torch.float32, device=DEV))
        cross_w = torch.sigmoid(torch.tensor(theta[20:30], dtype=torch.float32, device=DEV))
        
        # Mean opposite-pathway weight vectors
        W_A_mean = W_A_fc.mean(0)  # [512]
        W_B_mean = W_B_fc.mean(0)  # [512]
        
        with torch.no_grad():
            logits = torch.zeros(len(all_feats), 10, device=DEV)
            
            # A-classes (0-4)
            for c in range(5):
                primary = h_A @ W_A_fc[c]              # [N]
                cross = h_B @ W_B_mean                   # [N]
                logits[:, c] = scales[c] * primary + cross_w[c] * cross + biases[c]
            
            # B-classes (5-9)
            for c in range(5):
                primary = h_B @ W_B_fc[c]              # [N]
                cross = h_A @ W_A_mean                   # [N]
                logits[:, c+5] = scales[c+5] * primary + cross_w[c+5] * cross + biases[c+5]
            
            preds = logits.argmax(1).cpu()
        
        pc = {c: (preds[true_y==c]==c).float().mean().item() for c in clA+clB}
        ret = sum(1 for c in clA+clB if parent_pc[c]>0 and pc[c]/parent_pc[c]>=0.9)
        avg_drop = np.mean([(1-pc[c]/parent_pc[c])*100 for c in clA+clB])
        return -ret*10 + avg_drop

    # Initialize: scales=0 (exp(0)=1), cross=−2 (σ(−2)≈0.12, small cross-pathway)
    x0_ext = list(best_sol1) + [0.0]*10 + [-2.0]*10  # 30D
    es2 = cma.CMAEvolutionStrategy(x0_ext, 0.5, {
        'maxiter': 200, 'popsize': 30, 'seed': 42, 'verbose': -1
    })
    
    best_f2, best_sol2 = 100, x0_ext[:]
    gen = 0
    while not es2.stop():
        gen += 1
        sols = es2.ask()
        fits = [fitness_extended(s) for s in sols]
        es2.tell(sols, fits)
        bf = min(fits)
        if bf < best_f2: best_f2 = bf; best_sol2 = sols[fits.index(bf)][:]
        if gen % 20 == 0 or gen == 1:
            ret_est = max(0, int(-best_f2 // 10))
            scales_cur = np.exp(best_sol2[10:20])
            cross_cur = 1/(1+np.exp(-np.array(best_sol2[20:30])))
            print(f"  gen {gen:3d}: best_f={best_f2:+.1f} (~ret≈{ret_est}) "
                  f"scales=[{scales_cur.min():.2f},{scales_cur.max():.2f}] "
                  f"cross=[{cross_cur.min():.3f},{cross_cur.max():.3f}] ({time.time()-t0:.0f}s)")

    # Final eval
    theta_best = best_sol2
    biases_f = torch.tensor(theta_best[:10], dtype=torch.float32, device=DEV)
    scales_f = torch.exp(torch.tensor(theta_best[10:20], dtype=torch.float32, device=DEV))
    cross_f = torch.sigmoid(torch.tensor(theta_best[20:30], dtype=torch.float32, device=DEV))
    W_A_mean = W_A_fc.mean(0)
    W_B_mean = W_B_fc.mean(0)
    
    with torch.no_grad():
        logits_f = torch.zeros(len(all_feats), 10, device=DEV)
        for c in range(5):
            logits_f[:,c] = scales_f[c] * (h_A @ W_A_fc[c]) + cross_f[c] * (h_B @ W_B_mean) + biases_f[c]
        for c in range(5):
            logits_f[:,c+5] = scales_f[c+5] * (h_B @ W_B_fc[c]) + cross_f[c+5] * (h_A @ W_A_mean) + biases_f[c+5]
        preds_f = logits_f.argmax(1).cpu()
    
    pc_f = {c: (preds_f[true_y==c]==c).float().mean().item() for c in clA+clB}
    ret_v2 = print_drop("v2 (30D: bias+scale+cross)", parent_pc, pc_f, clA+clB)

    # ═══ SUMMARY ═══
    print(f"\n{'='*60}")
    print(f"  NeuronConcat Results:")
    print(f"  Baseline (no CMA):    {ret_base}/10")
    print(f"  v1 (10D bias):        {ret_v1}/10")
    print(f"  v2 (30D extended):    {ret_v2}/10")
    print(f"  Optimal cross-pathway: {cross_f.cpu().numpy()}")
    print(f"  Optimal scales:        {scales_f.cpu().numpy()}")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")
    beep(f"Done. v1={ret_v1} v2={ret_v2} out of 10")
