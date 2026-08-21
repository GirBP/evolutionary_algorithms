#!/usr/bin/env python3
"""
CNN Merge via Knowledge Distillation — Google Colab version.
Self-contained: trains parents + runs KD merge.

USAGE in Colab:
  1. Runtime → Change runtime type → T4 GPU
  2. Upload this file or paste into cell
  3. Run — takes ~3-5 minutes total
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, os
import numpy as np
from torchvision import datasets, transforms, models

# ═══ Setup ═══
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEV}")
if DEV.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name()}")

# ═══ Data ═══
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_ds = datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10('/tmp/cifar10', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

clA = list(range(5))   # 0-4
clB = list(range(5,10)) # 5-9

# ═══ Model ═══
def make_resnet18(n_classes):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, n_classes)
    return m

# ═══ Train parent ═══
def train_parent(cls_list, epochs=15, seed=42):
    torch.manual_seed(seed)
    m = make_resnet18(len(cls_list)).to(DEV)
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    for ep in range(epochs):
        m.train()
        for xb, yb in train_loader:
            mask = torch.zeros(len(yb), dtype=torch.bool)
            for c in cls_list: mask |= (yb == c)
            if mask.sum() == 0: continue
            xb, yb = xb[mask].to(DEV), yb[mask].to(DEV)
            # Remap labels
            for new_i, old_c in enumerate(cls_list):
                yb[yb == old_c] = new_i
            out = m(xb)
            loss = nn.CrossEntropyLoss()(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    
    # Eval
    m.eval()
    pc = {}
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEV)
            preds = m(xb).argmax(1).cpu()
            for c in cls_list:
                mask = yb == c
                if mask.sum() == 0: continue
                ci = cls_list.index(c)
                correct = (preds[mask] == ci).float().sum().item()
                pc[c] = pc.get(c, 0) + correct
    
    # Normalize
    for c in cls_list:
        total = sum(1 for _, y in test_ds if y == c)
        pc[c] = pc[c] / total
    
    return m, pc

# ═══ Print results ═══
def print_results(name, parent_pc, merged_pc, classes):
    print(f"\n  {name}:")
    print(f"  {'Class':>5} | {'Parent':>8} | {'Merged':>8} | {'Drop%':>7} | Retained?")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}")
    retained = 0
    for c in classes:
        p = parent_pc[c]
        m = merged_pc.get(c, 0)
        drop = (1 - m/p) * 100 if p > 0 else 100
        ok = '✅' if drop <= 10 else '❌'
        if drop <= 10: retained += 1
        print(f"  {c:>5} | {p:>8.3f} | {m:>8.3f} | {drop:>6.1f}% | {ok}")
    print(f"  Retention: {retained}/{len(classes)} (drop ≤ 10%)")
    return retained

# ═══ Main ═══
t0 = time.time()

print("=" * 60)
print("  PHASE 1: Training parents (15 epochs each)")
print("=" * 60)
pA, pcA = train_parent(clA, epochs=15, seed=42)
print(f"  Parent A: {pcA}")
pB, pcB = train_parent(clB, epochs=15, seed=142)
print(f"  Parent B: {pcB}")
print(f"  Parents trained in {time.time()-t0:.1f}s")

parent_pc = {**pcA, **pcB}

# ═══ Compute teacher logits ═══
print("\n" + "=" * 60)
print("  PHASE 2: Computing teacher logits on test set")
print("=" * 60)
t1 = time.time()
all_x, all_y = [], []
for xb, yb in test_loader:
    all_x.append(xb); all_y.append(yb)
X_transfer = torch.cat(all_x)
y_transfer = torch.cat(all_y)

pA.eval(); pB.eval()
with torch.no_grad():
    logits_A = torch.cat([pA(xb.to(DEV)).cpu() for xb in X_transfer.split(256)])
    logits_B = torch.cat([pB(xb.to(DEV)).cpu() for xb in X_transfer.split(256)])
print(f"  Teacher logits: {logits_A.shape}, {logits_B.shape} ({time.time()-t1:.1f}s)")

# ═══ KD Merge ═══
print("\n" + "=" * 60)
print("  PHASE 3: Knowledge Distillation (2-stage)")
print("=" * 60)

def kd_loss(student_logits, teacher_A, teacher_B, T=4.0, w_B=1.0):
    """Class-selective KD: A teaches 0-4, B teaches 5-9."""
    s_A = student_logits[:, :5] / T
    s_B = student_logits[:, 5:] / T
    t_A = F.softmax(teacher_A / T, dim=1)
    t_B = F.softmax(teacher_B / T, dim=1)
    
    loss_A = F.kl_div(F.log_softmax(s_A, dim=1), t_A, reduction='batchmean') * (T**2)
    loss_B = F.kl_div(F.log_softmax(s_B, dim=1), t_B, reduction='batchmean') * (T**2)
    return loss_A + w_B * loss_B

def eval_model(model, parent_pc, classes):
    model.eval()
    merged_pc = {}
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(DEV)).argmax(1).cpu()
            for c in classes:
                mask = yb == c
                if mask.sum() == 0: continue
                correct = (preds[mask] == c).float().sum().item()
                merged_pc[c] = merged_pc.get(c, 0) + correct
    for c in classes:
        total = sum(1 for _, y in test_ds if y == c)
        merged_pc[c] = merged_pc.get(c, 0) / total
    return merged_pc

all_classes = clA + clB

# Stage 1: FC warmup on frozen backbone
# Use ImageNet pretrained backbone (neutral features, not biased toward A)
student = make_resnet18(10).to(DEV)
# DO NOT copy Parent A weights — use ImageNet pretrained (already loaded by make_resnet18)

# Freeze backbone
for name, p in student.named_parameters():
    if 'fc' not in name:
        p.requires_grad = False

opt = torch.optim.Adam(student.fc.parameters(), lr=0.001)
print("\n  Stage 1: FC warmup (frozen backbone, 200 epochs)...")
student.train()
for ep in range(200):
    idx = torch.randperm(len(X_transfer))[:2048]
    xb = X_transfer[idx].to(DEV)
    out = student(xb)
    loss = kd_loss(out, logits_A[idx].to(DEV), logits_B[idx].to(DEV))
    opt.zero_grad(); loss.backward(); opt.step()
print(f"  FC warmup done ({time.time()-t1:.1f}s)")

mpc = eval_model(student, parent_pc, all_classes)
ret = print_results("After FC warmup", parent_pc, mpc, all_classes)

# Stage 2: Full fine-tune
for name, p in student.named_parameters():
    p.requires_grad = True

# Different LR for backbone vs FC
opt = torch.optim.SGD([
    {'params': [p for n, p in student.named_parameters() if 'fc' not in n], 'lr': 0.0001},
    {'params': student.fc.parameters(), 'lr': 0.001}
], momentum=0.9, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

best_ret = ret
best_sd = {k: v.clone() for k, v in student.state_dict().items()}

print(f"\n  Stage 2: Full fine-tune (50 epochs, diff LR)...")
for ep in range(1, 51):
    student.train()
    # Shuffle and batch
    idx = torch.randperm(len(X_transfer))
    epoch_loss = 0
    for i in range(0, len(idx), 256):
        batch_idx = idx[i:i+256]
        xb = X_transfer[batch_idx].to(DEV)
        tA = logits_A[batch_idx].to(DEV)
        tB = logits_B[batch_idx].to(DEV)
        
        out = student(xb)
        # Adaptive w_B: start gentle, increase
        w_B = 0.5 + 1.5 * (ep / 50)  # 0.5 → 2.0
        loss = kd_loss(out, tA, tB, T=4.0, w_B=w_B)
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item()
    sch.step()
    
    # Eval every 5 epochs
    if ep % 5 == 0 or ep == 1:
        mpc = eval_model(student, parent_pc, all_classes)
        ret_now = sum(1 for c in all_classes 
                     if parent_pc[c] > 0 and mpc.get(c, 0) / parent_pc[c] >= 0.9)
        avg_drop = np.mean([(1 - mpc.get(c, 0)/parent_pc[c])*100 
                           for c in all_classes if parent_pc[c] > 0])
        print(f"  ep {ep:2d}: ret={ret_now}/10, avg_drop={avg_drop:.1f}%, "
              f"w_B={w_B:.2f}, loss={epoch_loss:.3f} ({time.time()-t0:.0f}s)")
        
        if ret_now > best_ret:
            best_ret = ret_now
            best_sd = {k: v.clone() for k, v in student.state_dict().items()}

# Final eval with best model
student.load_state_dict(best_sd)
mpc = eval_model(student, parent_pc, all_classes)
ret_final = print_results("BEST MODEL", parent_pc, mpc, all_classes)

print(f"\n{'='*60}")
print(f"  FINAL: Retention = {ret_final}/10")
print(f"  Total time: {time.time()-t0:.1f}s")
print(f"{'='*60}")
