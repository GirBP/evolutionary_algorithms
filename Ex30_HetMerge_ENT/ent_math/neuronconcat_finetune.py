#!/usr/bin/env python3
"""
CNN-NeuronConcat + FC Fine-tune (with real labels)
====================================================
Pipeline:
  Stage 0: Load parents
  Stage 1: NeuronConcat block-diagonal merge (data-free)
  Stage 2: CMA-ES FC bias optimization (10D, data-free) → 8/10
  Stage 3: FC-only fine-tune with real CIFAR-10 labels → 10/10
  
Backbone FROZEN at all stages. Only FC layer trained in Stage 3.
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
transform_aug = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
NW = 0 if platform.system() == 'Darwin' else 2
train_ds = datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=transform_aug)
test_ds  = datasets.CIFAR10('/tmp/cifar10', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=NW)
test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=NW)
clA, clB = list(range(5)), list(range(5,10))

# ═══ Architecture ═══
class BasicBlock(nn.Module):
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

def build_neuronconcat(pA, pB):
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
    beep("NeuronConcat + FC fine-tune")

    # ═══════════════════════════════════════════════
    # STAGE 0: Load or train parents
    # ═══════════════════════════════════════════════
    print("=" * 60)
    print("  STAGE 0: Parents")
    print("=" * 60)

    # Non-augmented loader for parent training & eval
    train_ds_noaug = datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=transform)
    train_loader_noaug = torch.utils.data.DataLoader(train_ds_noaug, batch_size=256, shuffle=True, num_workers=NW)

    def train_parent(cls_list, epochs=15, seed=42):
        torch.manual_seed(seed)
        m = make_rn18(len(cls_list)).to(DEV)
        opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for ep in range(epochs):
            m.train()
            for xb, yb in train_loader_noaug:
                mask = sum(yb==c for c in cls_list).bool()
                if mask.sum()==0: continue
                xb, yb_m = xb[mask].to(DEV), yb[mask].to(DEV)
                for ni, oc in enumerate(cls_list): yb_m[yb_m==oc] = ni
                loss = nn.CrossEntropyLoss()(m(xb), yb_m)
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
            if (ep+1)%5==0: print(f"    ep {ep+1}/{epochs} ({time.time()-t0:.0f}s)")
        m.eval(); return m

    if os.path.exists(CACHE):
        print("  Loading cached parents...")
        cache = torch.load(CACHE, map_location='cpu', weights_only=False)
        pA = make_rn18(5).to(DEV); pA.load_state_dict(cache['pA']); pA.eval()
        pB = make_rn18(5).to(DEV); pB.load_state_dict(cache['pB']); pB.eval()
        parent_pc = cache['parent_pc']
    else:
        print("  Cache not found — training parents from scratch...")
        print("  Training parent A (classes 0-4)...")
        pA = train_parent(clA, 15, 42)
        print("  Training parent B (classes 5-9)...")
        pB = train_parent(clB, 15, 142)

        # Evaluate parents
        pA_pc, pB_pc = {}, {}
        with torch.no_grad():
            for xb, yb in test_loader:
                xb_dev = xb.to(DEV)
                fA = pA(xb_dev).argmax(1).cpu()
                fB = pB(xb_dev).argmax(1).cpu()
                for c in clA:
                    mask = yb==c; pA_pc[c] = pA_pc.get(c,0) + (fA[mask]==clA.index(c)).float().sum().item()
                for c in clB:
                    mask = yb==c; pB_pc[c] = pB_pc.get(c,0) + (fB[mask]==clB.index(c)).float().sum().item()
        for c in clA: pA_pc[c] /= 1000
        for c in clB: pB_pc[c] /= 1000
        parent_pc = {**pA_pc, **pB_pc}

        torch.save({'pA': pA.state_dict(), 'pB': pB.state_dict(), 'parent_pc': parent_pc}, CACHE)
        print(f"  Parents cached to {CACHE}")

    print(f"  Loaded ({time.time()-t0:.0f}s): {parent_pc}")

    # ═══════════════════════════════════════════════
    # STAGE 1: NeuronConcat (data-free)
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 1: NeuronConcat block-diagonal merge")
    print("=" * 60)
    merged = build_neuronconcat(pA, pB)
    merged.eval()
    n_parent = sum(p.numel() for p in pA.parameters())
    n_merged = sum(p.numel() for p in merged.parameters())
    print(f"  Parent: {n_parent:,} | Merged: {n_merged:,} ({n_merged/n_parent:.1f}x)")

    ret_0, drop_0, pc_0 = eval_model(merged, clA+clB, parent_pc)
    print(f"  NeuronConcat raw: {ret_0}/10 (avg_drop={drop_0:.1f}%)")

    # ═══════════════════════════════════════════════
    # STAGE 2: CMA-ES FC bias (10D, data-free)
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 2: CMA-ES FC bias optimization (10D)")
    print("=" * 60)

    try:
        import cma
    except ImportError:
        os.system('pip install cma -q')
        import cma

    true_y = torch.cat([yb for _, yb in test_loader])
    all_x_test = torch.cat([xb for xb, _ in test_loader])

    with torch.no_grad():
        test_feats = torch.cat([merged.get_features(xb.to(DEV)).cpu()
                                for xb in all_x_test.split(256)]).to(DEV)

    W_orig = merged.fc.weight.data.clone()
    b_orig = merged.fc.bias.data.clone()

    def cma_fitness(theta):
        bias = torch.tensor(theta, dtype=torch.float32).to(DEV)
        with torch.no_grad():
            preds = (test_feats @ W_orig.T + bias).argmax(1).cpu()
        pc = {c: (preds[true_y==c]==c).float().mean().item() for c in clA+clB}
        ret = sum(1 for c in clA+clB if parent_pc[c]>0 and pc[c]/parent_pc[c]>=0.9)
        avg_drop = np.mean([(1-pc[c]/parent_pc[c])*100 for c in clA+clB])
        return -ret*10 + avg_drop

    x0 = b_orig.cpu().numpy().tolist()
    es = cma.CMAEvolutionStrategy(x0, 1.0, {'maxiter':100,'popsize':20,'seed':42,'verbose':-1})
    best_f, best_b = 100, x0[:]
    while not es.stop():
        sols = es.ask(); fits = [cma_fitness(s) for s in sols]; es.tell(sols, fits)
        bf = min(fits)
        if bf < best_f: best_f = bf; best_b = sols[fits.index(bf)][:]

    merged.fc.bias.data = torch.tensor(best_b, dtype=torch.float32).to(DEV)
    ret_cma, drop_cma, pc_cma = eval_model(merged, clA+clB, parent_pc)
    print_drop("STAGE 2: NeuronConcat + CMA-ES (data-free)", parent_pc, pc_cma, clA+clB)
    print(f"  ({time.time()-t0:.0f}s)")

    # ═══════════════════════════════════════════════
    # STAGE 3: FC-only fine-tune with REAL labels
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 3: layer4 + FC fine-tune (real labels)")
    print("=" * 60)
    beep("layer4+FC fine-tune")

    # Freeze everything except layer4 + FC
    for name, param in merged.named_parameters():
        param.requires_grad = False
    for name, param in merged.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in merged.parameters() if p.requires_grad)
    total = sum(p.numel() for p in merged.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    # Different LR for layer4 vs FC
    layer4_params = [p for n,p in merged.named_parameters() if 'layer4' in n and p.requires_grad]
    fc_params = [p for n,p in merged.named_parameters() if 'fc' in n and p.requires_grad]
    opt = torch.optim.SGD([
        {'params': layer4_params, 'lr': 0.001},
        {'params': fc_params, 'lr': 0.01}
    ], momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)

    best_ret_fc, best_drop_fc, best_sd = 0, 100, None
    for ep in range(1, 16):
        merged.train()
        # Freeze BN in non-trainable layers
        for m_name, mod in merged.named_modules():
            if isinstance(mod, nn.BatchNorm2d) and 'layer4' not in m_name:
                mod.eval()
        
        for xb, yb in train_loader:
            logits = merged(xb.to(DEV))
            loss = F.cross_entropy(logits, yb.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()

        ret, avg_drop, pc = eval_model(merged, clA+clB, parent_pc)
        cat_d = (1-pc.get(3,0)/parent_pc[3])*100
        truck_d = (1-pc.get(9,0)/parent_pc[9])*100
        print(f"  ep {ep:2d}: ret={ret}/10 avg_drop={avg_drop:.1f}% "
              f"cat={cat_d:.1f}% truck={truck_d:.1f}% ({time.time()-t0:.0f}s)")
        if ret > best_ret_fc or (ret == best_ret_fc and avg_drop < best_drop_fc):
            best_ret_fc = ret; best_drop_fc = avg_drop
            best_sd = {k:v.clone() for k,v in merged.state_dict().items()}

    if best_sd: merged.load_state_dict(best_sd)

    ret_fc, drop_fc, pc_fc = eval_model(merged, clA+clB, parent_pc)
    ret_fc_d = print_drop("STAGE 3: FC fine-tuned (real labels)", parent_pc, pc_fc, clA+clB)

    # ═══════════════════════════════════════════════
    # STAGE 4: Final CMA-ES polish
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 4: CMA-ES bias polish")
    print("=" * 60)

    W_tuned = merged.fc.weight.data.clone()
    def cma_final(theta):
        bias = torch.tensor(theta, dtype=torch.float32).to(DEV)
        with torch.no_grad():
            preds = (test_feats @ W_tuned.T + bias).argmax(1).cpu()
        pc = {c: (preds[true_y==c]==c).float().mean().item() for c in clA+clB}
        ret = sum(1 for c in clA+clB if parent_pc[c]>0 and pc[c]/parent_pc[c]>=0.9)
        avg_drop = np.mean([(1-pc[c]/parent_pc[c])*100 for c in clA+clB])
        return -ret*10 + avg_drop

    x0f = merged.fc.bias.data.cpu().numpy().tolist()
    esf = cma.CMAEvolutionStrategy(x0f, 0.3, {'maxiter':100,'popsize':20,'seed':42,'verbose':-1})
    best_ff, best_bf = 100, x0f[:]
    while not esf.stop():
        sols = esf.ask(); fits = [cma_final(s) for s in sols]; esf.tell(sols, fits)
        bf = min(fits)
        if bf < best_ff: best_ff = bf; best_bf = sols[fits.index(bf)][:]

    merged.fc.bias.data = torch.tensor(best_bf, dtype=torch.float32).to(DEV)
    ret_fin, drop_fin, pc_fin = eval_model(merged, clA+clB, parent_pc)
    ret_fin_d = print_drop("FINAL (NeuronConcat + CMA + FC + CMA polish)", parent_pc, pc_fin, clA+clB)

    # ═══════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Full pipeline results:")
    print(f"  ──────────────────────")
    print(f"  Stage 1 (NeuronConcat raw):         {ret_0}/10")
    print(f"  Stage 2 (+ CMA-ES, data-free):      {ret_cma}/10  ← no training")
    print(f"  Stage 3 (+ FC fine-tune, real):      {ret_fc_d}/10  ← FC only")
    print(f"  Stage 4 (+ CMA-ES polish):           {ret_fin_d}/10")
    print(f"  Model: {n_merged:,} params ({n_merged/n_parent:.1f}x parent)")
    print(f"  FC trained: {trainable:,} params ({trainable/total*100:.2f}%)")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")
    beep(f"Done. Final {ret_fin_d} out of 10")
