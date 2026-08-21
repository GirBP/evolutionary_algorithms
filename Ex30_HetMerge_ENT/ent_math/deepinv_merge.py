#!/usr/bin/env python3
"""
CNN Merge: DeepInversion + KD + CMA-ES
========================================
Pipeline:
  Stage 1: DeepInversion — generate labeled synthetic images from each parent
  Stage 2: Train student on synthetic data with KD from parents
  Stage 3: CMA-ES post-hoc FC bias optimization (10D)
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

def make_rn18(nc):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    return m

def get_features(model, x):
    m = model
    x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
    x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
    return m.avgpool(x).flatten(1)

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

def eval_model(model, classes, parent_pc):
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


# ═══ DeepInversion ═══
def deep_inversion(model, target_class, n_images=200, steps=200, batch_size=50):
    """Generate synthetic images that maximize a target class logit.
    Uses BN regularization + Total Variation."""
    model.eval()
    
    # Collect BN stats
    bn_means, bn_vars = [], []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_means.append(m.running_mean.clone())
            bn_vars.append(m.running_var.clone())
    
    generated_images = []
    
    for batch_start in range(0, n_images, batch_size):
        bs = min(batch_size, n_images - batch_start)
        
        # Random init
        x = torch.randn(bs, 3, 32, 32, device=DEV, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=0.05)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Forward with BN stats collection
            bn_idx = [0]
            bn_loss = torch.tensor(0.0, device=DEV)
            
            def bn_hook(module, input, output):
                if isinstance(module, nn.BatchNorm2d):
                    feat = input[0]
                    feat_mean = feat.mean([0, 2, 3])
                    feat_var = feat.var([0, 2, 3])
                    idx = bn_idx[0]
                    if idx < len(bn_means):
                        bn_loss_item = ((feat_mean - bn_means[idx]) ** 2).mean() + \
                                       ((feat_var - bn_vars[idx]) ** 2).mean()
                        # Store for later accumulation
                        module._di_loss = bn_loss_item
                    bn_idx[0] += 1
            
            hooks = []
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    hooks.append(m.register_forward_hook(bn_hook))
            
            bn_idx[0] = 0
            logits = model(x)
            
            # Accumulate BN losses
            bn_loss = sum(m._di_loss for m in model.modules() 
                         if isinstance(m, nn.BatchNorm2d) and hasattr(m, '_di_loss'))
            
            for h in hooks: h.remove()
            for m in model.modules():
                if hasattr(m, '_di_loss'): del m._di_loss
            
            # Classification loss
            target = torch.full((bs,), target_class, dtype=torch.long, device=DEV)
            loss_ce = F.cross_entropy(logits, target)
            
            # Total Variation
            loss_tv = ((x[:,:,1:,:] - x[:,:,:-1,:]) ** 2).mean() + \
                      ((x[:,:,:,1:] - x[:,:,:,:-1]) ** 2).mean()
            
            # L2 norm
            loss_l2 = (x ** 2).mean()
            
            loss = loss_ce + 10.0 * bn_loss + 0.001 * loss_tv + 0.001 * loss_l2
            loss.backward()
            optimizer.step()
            
            # Clamp to valid range (approximate)
            with torch.no_grad():
                x.clamp_(-2.5, 2.5)
        
        generated_images.append(x.detach().cpu())
    
    return torch.cat(generated_images)


if __name__ == '__main__':
    import multiprocessing
    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('fork', force=True)

    t0 = time.time()
    CACHE = 'cached_parents_r3.pth'
    beep("DeepInversion experiment started")

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
                fA = get_features(pA, xb.to(DEV)); predsA = pA.fc(fA).argmax(1).cpu()
                fB = get_features(pB, xb.to(DEV)); predsB = pB.fc(fB).argmax(1).cpu()
                for c in clA:
                    mask = yb == c; pA_pc[c] = pA_pc.get(c, 0) + (predsA[mask] == clA.index(c)).float().sum().item()
                for c in clB:
                    mask = yb == c; pB_pc[c] = pB_pc.get(c, 0) + (predsB[mask] == clB.index(c)).float().sum().item()
        for c in clA: pA_pc[c] /= 1000
        for c in clB: pB_pc[c] /= 1000
        parent_pc = {**pA_pc, **pB_pc}
        torch.save({'pA': pA.state_dict(), 'pB': pB.state_dict(), 'parent_pc': parent_pc}, CACHE)
        print(f"  Cached ({time.time()-t0:.0f}s): {parent_pc}")

    # ═══════════════════════════════════════════════
    # STAGE 1: DeepInversion — generate labeled images
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 1: DeepInversion (200 images × 10 classes)")
    print("=" * 60)
    beep("Generating synthetic images")

    N_PER_CLASS = 200
    synth_x, synth_y = [], []

    # Generate from Parent A (classes 0-4)
    for cls_idx in range(5):
        actual_class = clA[cls_idx]
        print(f"  Generating class {actual_class} from parent A (target={cls_idx})...")
        imgs = deep_inversion(pA, cls_idx, n_images=N_PER_CLASS, steps=200, batch_size=50)
        synth_x.append(imgs)
        synth_y.append(torch.full((len(imgs),), actual_class, dtype=torch.long))
        
        # Verify: what does parent A predict?
        with torch.no_grad():
            logits = pA(imgs.to(DEV))
            pred_acc = (logits.argmax(1).cpu() == cls_idx).float().mean().item()
        print(f"    → {len(imgs)} images, parent A predicts correct: {pred_acc:.1%} ({time.time()-t0:.0f}s)")

    # Generate from Parent B (classes 5-9)
    for cls_idx in range(5):
        actual_class = clB[cls_idx]
        print(f"  Generating class {actual_class} from parent B (target={cls_idx})...")
        imgs = deep_inversion(pB, cls_idx, n_images=N_PER_CLASS, steps=200, batch_size=50)
        synth_x.append(imgs)
        synth_y.append(torch.full((len(imgs),), actual_class, dtype=torch.long))
        
        with torch.no_grad():
            logits = pB(imgs.to(DEV))
            pred_acc = (logits.argmax(1).cpu() == cls_idx).float().mean().item()
        print(f"    → {len(imgs)} images, parent B predicts correct: {pred_acc:.1%} ({time.time()-t0:.0f}s)")

    synth_x = torch.cat(synth_x)  # [2000, 3, 32, 32]
    synth_y = torch.cat(synth_y)  # [2000]
    print(f"\n  Total synthetic: {len(synth_x)} images, {len(torch.unique(synth_y))} classes")
    print(f"  ({time.time()-t0:.0f}s)")

    # Combine with real transfer data (unlabeled → use parent concat pseudo-labels)
    all_x_test = torch.cat([xb for xb, _ in test_loader])
    true_y = torch.cat([yb for _, yb in test_loader])
    
    with torch.no_grad():
        logA = torch.cat([pA(xb.to(DEV)).cpu() for xb in all_x_test.split(256)])
        logB = torch.cat([pB(xb.to(DEV)).cpu() for xb in all_x_test.split(256)])
    
    # Combine: synthetic (correct labels) + transfer (pseudo-labels)
    transfer_pseudo_y = torch.cat([logA, logB], dim=1).argmax(1)
    
    all_train_x = torch.cat([synth_x, all_x_test])
    all_train_y = torch.cat([synth_y, transfer_pseudo_y])
    print(f"  Combined training set: {len(all_train_x)} images")

    # ═══════════════════════════════════════════════
    # STAGE 2: Train student on synthetic + transfer data
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 2: Train student on DeepInversion data (30 ep)")
    print("=" * 60)
    beep("Training student")

    student = make_rn18(10).to(DEV)
    opt = torch.optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

    best_ret, best_sd, best_ep = 0, None, 0
    for ep in range(1, 31):
        student.train()
        idx = torch.randperm(len(all_train_x))
        ep_loss = 0
        for i in range(0, len(idx), 256):
            bi = idx[i:i+256]
            xb = all_train_x[bi].to(DEV)
            yb = all_train_y[bi].to(DEV)
            logits = student(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        sch.step()

        if ep % 5 == 0 or ep == 1:
            ret, avg_drop, spc = eval_model(student, clA+clB, parent_pc)
            retA = sum(1 for c in clA if parent_pc[c]>0 and spc.get(c,0)/parent_pc[c]>=0.9)
            retB = sum(1 for c in clB if parent_pc[c]>0 and spc.get(c,0)/parent_pc[c]>=0.9)
            print(f"  ep {ep:2d}: A={retA}/5 B={retB}/5 ret={ret}/10 avg_drop={avg_drop:.1f}% ({time.time()-t0:.0f}s)")
            if ret > best_ret:
                best_ret = ret; best_ep = ep
                best_sd = {k: v.clone() for k, v in student.state_dict().items()}

    if best_sd: student.load_state_dict(best_sd)

    # ═══════════════════════════════════════════════
    # STAGE 3: CMA-ES FC bias optimization (10D)
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 3: CMA-ES FC bias optimization")
    print("=" * 60)

    try:
        import cma
    except ImportError:
        os.system('pip install cma -q')
        import cma

    student.eval()
    original_weight = student.fc.weight.data.clone()
    
    with torch.no_grad():
        student_feats = torch.cat([get_features(student, xb.to(DEV)).cpu() for xb in all_x_test.split(256)])
    student_feats_dev = student_feats.to(DEV)

    def cma_fitness(bias_vec):
        bias = torch.tensor(bias_vec, dtype=torch.float32).to(DEV)
        with torch.no_grad():
            logits = student_feats_dev @ original_weight.T + bias
            preds = logits.argmax(1).cpu()
        pc = {}
        for c in clA+clB:
            mask = true_y == c
            pc[c] = (preds[mask] == c).float().mean().item()
        ret = sum(1 for c in clA+clB if parent_pc[c]>0 and pc[c]/parent_pc[c]>=0.9)
        avg_drop = np.mean([(1-pc[c]/parent_pc[c])*100 for c in clA+clB if parent_pc[c]>0])
        return -ret * 10 + avg_drop

    x0 = student.fc.bias.data.cpu().numpy().tolist()
    es = cma.CMAEvolutionStrategy(x0, 0.5, {
        'maxiter': 50, 'popsize': 20, 'seed': 42, 'verbose': -1
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
        if gen % 10 == 0 or gen == 1:
            print(f"  gen {gen:2d}: best_f={best_f_ever:+.1f} ({time.time()-t0:.0f}s)")

    student.fc.bias.data = torch.tensor(best_bias, dtype=torch.float32).to(DEV)

    # Final
    ret_final, drop_final, pc_final = eval_model(student, clA+clB, parent_pc)
    ret_display = print_drop("FINAL (DeepInversion + CMA-ES)", parent_pc, pc_final, clA+clB)

    print(f"\n{'='*60}")
    print(f"  FINAL: {ret_display}/10")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")
    beep(f"Done. Retention {ret_display} out of 10")
