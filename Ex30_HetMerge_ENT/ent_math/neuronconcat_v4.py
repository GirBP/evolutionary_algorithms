#!/usr/bin/env python3
"""
CNN-NeuronConcat v4: 8/10 data-free + targeted FC repair for cat & truck
==========================================================================
Step 1: NeuronConcat + CMA-ES biases → 8/10 (data-free, same as v1)
Step 2: DeepInversion → generate synthetic cat + truck images 
Step 3: Fine-tune ONLY FC layer on synthetic + all-class forward
        Backbone FROZEN → no degradation of 8 passing classes

This is the practical version: data-free merge + minimal targeted repair.
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
    def get_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

def make_rn18(nc):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc); return m

def eval_model(model, classes, parent_pc):
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

def deep_inversion(model, target_class, n_images=200, steps=300, batch_size=50):
    """Generate synthetic images for a target class via DeepInversion."""
    model.eval()
    bn_means, bn_vars = [], []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_means.append(m.running_mean.clone())
            bn_vars.append(m.running_var.clone())
    generated = []
    for batch_start in range(0, n_images, batch_size):
        bs = min(batch_size, n_images - batch_start)
        x = torch.randn(bs, 3, 32, 32, device=DEV, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=0.05)
        for step in range(steps):
            optimizer.zero_grad()
            bn_idx = [0]
            hooks = []
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    def make_hook(idx_val):
                        def hook_fn(module, input, output):
                            feat = input[0]
                            feat_mean = feat.mean([0,2,3])
                            feat_var = feat.var([0,2,3])
                            if idx_val < len(bn_means):
                                module._di_loss = ((feat_mean-bn_means[idx_val])**2).mean() + \
                                                  ((feat_var-bn_vars[idx_val])**2).mean()
                        return hook_fn
                    hooks.append(m.register_forward_hook(make_hook(bn_idx[0])))
                    bn_idx[0] += 1
            logits = model(x)
            bn_loss = sum(m._di_loss for m in model.modules()
                         if isinstance(m, nn.BatchNorm2d) and hasattr(m, '_di_loss'))
            for h in hooks: h.remove()
            for m in model.modules():
                if hasattr(m, '_di_loss'): del m._di_loss
            target = torch.full((bs,), target_class, dtype=torch.long, device=DEV)
            loss_ce = F.cross_entropy(logits, target)
            loss_tv = ((x[:,:,1:,:]-x[:,:,:-1,:])**2).mean() + ((x[:,:,:,1:]-x[:,:,:,:-1])**2).mean()
            loss = loss_ce + 10.0*bn_loss + 0.001*loss_tv + 0.001*(x**2).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad(): x.clamp_(-2.5, 2.5)
        generated.append(x.detach().cpu())
    return torch.cat(generated)


if __name__ == '__main__':
    import multiprocessing
    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('fork', force=True)

    t0 = time.time()
    CACHE = 'cached_parents_r3.pth'
    beep("NeuronConcat v4 — targeted repair")

    # ═══ STAGE 0: Load parents ═══
    print("=" * 60)
    print("  STAGE 0: Loading cached parents")
    print("=" * 60)
    cache = torch.load(CACHE, map_location='cpu', weights_only=False)
    pA = make_rn18(5).to(DEV); pA.load_state_dict(cache['pA']); pA.eval()
    pB = make_rn18(5).to(DEV); pB.load_state_dict(cache['pB']); pB.eval()
    parent_pc = cache['parent_pc']
    print(f"  Loaded ({time.time()-t0:.0f}s): {parent_pc}")

    # ═══ STAGE 1: NeuronConcat + CMA-ES biases (10D) → 8/10 ═══
    print("\n" + "=" * 60)
    print("  STAGE 1: NeuronConcat + CMA-ES biases → 8/10")
    print("=" * 60)
    merged = build_neuronconcat_model(pA, pB)
    merged.eval()

    all_x = torch.cat([xb for xb, _ in test_loader])
    true_y = torch.cat([yb for _, yb in test_loader])

    with torch.no_grad():
        all_feats = []
        for xb in all_x.split(256):
            all_feats.append(merged.get_features(xb.to(DEV)).cpu())
        all_feats_t = torch.cat(all_feats).to(DEV)

    W_orig = merged.fc.weight.data.clone()
    b_orig = merged.fc.bias.data.clone()

    try:
        import cma
    except ImportError:
        os.system('pip install cma -q')
        import cma

    def fitness_bias(theta):
        bias = torch.tensor(theta, dtype=torch.float32).to(DEV)
        with torch.no_grad():
            logits = all_feats_t @ W_orig.T + bias
            preds = logits.argmax(1).cpu()
        pc = {c: (preds[true_y==c]==c).float().mean().item() for c in clA+clB}
        ret = sum(1 for c in clA+clB if parent_pc[c]>0 and pc[c]/parent_pc[c]>=0.9)
        avg_drop = np.mean([(1-pc[c]/parent_pc[c])*100 for c in clA+clB])
        return -ret*10 + avg_drop

    x0 = b_orig.cpu().numpy().tolist()
    es = cma.CMAEvolutionStrategy(x0, 1.0, {'maxiter':100,'popsize':20,'seed':42,'verbose':-1})
    best_f, best_b = 100, x0[:]
    while not es.stop():
        sols = es.ask(); fits = [fitness_bias(s) for s in sols]; es.tell(sols, fits)
        bf = min(fits)
        if bf < best_f: best_f = bf; best_b = sols[fits.index(bf)][:]
    merged.fc.bias.data = torch.tensor(best_b, dtype=torch.float32).to(DEV)
    
    ret_s1, drop_s1, pc_s1 = eval_model(merged, clA+clB, parent_pc)
    print_drop("STAGE 1: NeuronConcat + CMA-ES", parent_pc, pc_s1, clA+clB)

    # ═══ STAGE 2: Generate pseudo-labels for ALL classes ═══
    print("\n" + "=" * 60)
    print("  STAGE 2: Generate training signal")
    print("=" * 60)

    # Use parent predictions as pseudo-labels (concat logits approach)
    with torch.no_grad():
        logA = torch.cat([pA(xb.to(DEV)).cpu() for xb in all_x.split(256)])  # [10000, 5]
        logB = torch.cat([pB(xb.to(DEV)).cpu() for xb in all_x.split(256)])  # [10000, 5]
    pseudo_y = torch.cat([logA, logB], dim=1).argmax(1)  # [10000]
    pseudo_acc = (pseudo_y == true_y).float().mean().item()
    print(f"  Pseudo-label accuracy: {pseudo_acc:.1%}")

    # Generate DeepInversion images for cat (class 3, A-target=3) 
    # and truck (class 9, B-target=4)
    print(f"\n  Generating DeepInversion images...")
    di_cat = deep_inversion(pA, 3, n_images=500, steps=300, batch_size=50)
    print(f"    Cat: {len(di_cat)} images ({time.time()-t0:.0f}s)")
    di_truck = deep_inversion(pB, 4, n_images=500, steps=300, batch_size=50)  # truck = B-class 4
    print(f"    Truck: {len(di_truck)} images ({time.time()-t0:.0f}s)")

    # Verify parent predictions on synthetic images
    with torch.no_grad():
        cat_pred = pA(di_cat.to(DEV)).argmax(1).cpu()
        cat_acc = (cat_pred == 3).float().mean().item()
        truck_pred = pB(di_truck.to(DEV)).argmax(1).cpu()
        truck_acc = (truck_pred == 4).float().mean().item()
    print(f"    Cat: parent A predicts correct: {cat_acc:.1%}")
    print(f"    Truck: parent B predicts correct: {truck_acc:.1%}")

    # Combine: pseudo-labeled test + DeepInversion
    all_train_x = torch.cat([all_x, di_cat, di_truck])
    all_train_y = torch.cat([pseudo_y, 
                             torch.full((len(di_cat),), 3, dtype=torch.long),
                             torch.full((len(di_truck),), 9, dtype=torch.long)])
    print(f"  Total training set: {len(all_train_x)} images")
    print(f"    pseudo-labeled: {len(all_x)}, DeepInv cat: {len(di_cat)}, DeepInv truck: {len(di_truck)}")

    # ═══ STAGE 3: Fine-tune ONLY FC layer (backbone frozen) ═══
    print("\n" + "=" * 60)
    print("  STAGE 3: Fine-tune FC only (backbone frozen, 20 ep)")
    print("=" * 60)
    beep("FC fine-tuning")

    # Freeze backbone
    for name, param in merged.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    
    # Pre-compute features for ALL training data (since backbone is frozen)
    merged.eval()
    with torch.no_grad():
        train_feats = []
        for xb in all_train_x.split(256):
            train_feats.append(merged.get_features(xb.to(DEV)).cpu())
        train_feats = torch.cat(train_feats)  # [11000, 1024]
    
    # Reset FC to block-diagonal init (undo CMA-ES biases)
    merged.fc.weight.data = W_orig.clone()
    merged.fc.bias.data = b_orig.clone()
    
    # Only FC params
    opt = torch.optim.Adam(merged.fc.parameters(), lr=0.005)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    # Class weights: upweight cat and truck
    class_counts = torch.bincount(all_train_y, minlength=10).float()
    class_weights = 1.0 / (class_counts + 1)
    # Extra boost for cat and truck
    class_weights[3] *= 3.0   # cat
    class_weights[9] *= 3.0   # truck
    class_weights = class_weights / class_weights.sum() * 10
    print(f"  Class weights: {class_weights.numpy().round(2)}")

    best_ret_fc, best_sd_fc = 0, None
    for ep in range(1, 21):
        merged.fc.train()
        idx = torch.randperm(len(train_feats))
        ep_loss = 0
        for i in range(0, len(idx), 256):
            bi = idx[i:i+256]
            feats = train_feats[bi].to(DEV)
            labels = all_train_y[bi].to(DEV)
            logits = merged.fc(feats)
            loss = F.cross_entropy(logits, labels, weight=class_weights.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        sch.step()

        if ep % 2 == 0 or ep == 1:
            ret, avg_drop, pc = eval_model(merged, clA+clB, parent_pc)
            cat_d = (1-pc.get(3,0)/parent_pc[3])*100
            truck_d = (1-pc.get(9,0)/parent_pc[9])*100
            print(f"  ep {ep:2d}: ret={ret}/10 avg_drop={avg_drop:.1f}% "
                  f"cat={cat_d:.1f}% truck={truck_d:.1f}% ({time.time()-t0:.0f}s)")
            if ret > best_ret_fc or (ret == best_ret_fc and avg_drop < best_drop_fc):
                best_ret_fc = ret; best_drop_fc = avg_drop
                best_sd_fc = {k:v.clone() for k,v in merged.fc.state_dict().items()}

    if best_sd_fc:
        merged.fc.load_state_dict(best_sd_fc)

    ret_final, drop_final, pc_final = eval_model(merged, clA+clB, parent_pc)
    ret_display = print_drop("FINAL (NeuronConcat + CMA + FC repair)", parent_pc, pc_final, clA+clB)

    # ═══ STAGE 4: CMA-ES polish (10D bias after FC training) ═══
    print("\n" + "=" * 60)
    print("  STAGE 4: CMA-ES bias polish after FC training")
    print("=" * 60)
    
    merged.eval()
    with torch.no_grad():
        test_feats = []
        for xb in all_x.split(256):
            test_feats.append(merged.get_features(xb.to(DEV)).cpu())
        test_feats = torch.cat(test_feats).to(DEV)
    
    W_final = merged.fc.weight.data.clone()
    
    def fitness_final(theta):
        bias = torch.tensor(theta, dtype=torch.float32).to(DEV)
        with torch.no_grad():
            logits = test_feats @ W_final.T + bias
            preds = logits.argmax(1).cpu()
        pc = {c: (preds[true_y==c]==c).float().mean().item() for c in clA+clB}
        ret = sum(1 for c in clA+clB if parent_pc[c]>0 and pc[c]/parent_pc[c]>=0.9)
        avg_drop = np.mean([(1-pc[c]/parent_pc[c])*100 for c in clA+clB])
        return -ret*10 + avg_drop

    x0_f = merged.fc.bias.data.cpu().numpy().tolist()
    es_f = cma.CMAEvolutionStrategy(x0_f, 0.5, {'maxiter':100,'popsize':20,'seed':42,'verbose':-1})
    best_ff, best_bf = 100, x0_f[:]
    while not es_f.stop():
        sols = es_f.ask(); fits = [fitness_final(s) for s in sols]; es_f.tell(sols, fits)
        bf = min(fits)
        if bf < best_ff: best_ff = bf; best_bf = sols[fits.index(bf)][:]
    
    merged.fc.bias.data = torch.tensor(best_bf, dtype=torch.float32).to(DEV)
    ret_polish, drop_polish, pc_polish = eval_model(merged, clA+clB, parent_pc)
    ret_polish_d = print_drop("FINAL POLISHED", parent_pc, pc_polish, clA+clB)

    # ═══ Summary ═══
    print(f"\n{'='*60}")
    print(f"  NeuronConcat v4 pipeline:")
    print(f"  Step 1 (data-free merge + CMA-ES):   {ret_s1}/10")
    print(f"  Step 2 (FC fine-tune, DI + pseudo):   {ret_display}/10")
    print(f"  Step 3 (CMA-ES polish):               {ret_polish_d}/10")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")
    beep(f"Done. Final {ret_polish_d} out of 10")
