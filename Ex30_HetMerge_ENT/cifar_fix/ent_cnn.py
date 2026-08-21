#!/usr/bin/env python3
"""ENT for CNN — per-block binary selection + EA (μ+λ).

Adapts e23_ent.py (MLP per-neuron ENT) for ResNet-18:
- MLP neuron → CNN block (residual block)
- Binary mask: entire block from A or B (no interpolation!)
- Per-class routing: sigmoid per class
- EA (μ+λ) with crossover + mutation (same as e23)
- Result: ONE ResNet-18(10 classes)

Usage: python3 ent_cnn.py <seed>
Parents: results/parent{A,B}_s{seed}.pth
"""
import numpy as np, torch, torch.nn as nn, random, json, time, sys, copy
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import torchvision.models as models
from torchvision import datasets

SEED = int(sys.argv[1])
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
t0 = time.time()

DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEV}", flush=True)

# ═══ Data ═══
raw_tr = datasets.CIFAR10('/tmp/cifar10', train=True, download=True)
raw_te = datasets.CIFAR10('/tmp/cifar10', train=False, download=True)
mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
X_tr = (torch.tensor(raw_tr.data).permute(0,3,1,2).float()/255 - mean)/std
y_tr = torch.tensor(raw_tr.targets)
X_te = (torch.tensor(raw_te.data).permute(0,3,1,2).float()/255 - mean)/std
y_te = torch.tensor(raw_te.targets)

X_cal = X_tr[40000:45000].to(DEV)
y_cal = y_tr[40000:45000].to(DEV)

clA, clB = list(range(5)), list(range(5,10))
ALL = list(range(10))
print(f"Data: {time.time()-t0:.1f}s", flush=True)

# ═══ Model ═══
def make_rn(nc=5):
    m = models.resnet18(weights='IMAGENET1K_V1')
    m.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, nc)
    return m

# ═══ Load parents ═══
mA = make_rn(5); mA.load_state_dict(torch.load(f'results/parentA_s{SEED}.pth', weights_only=True, map_location='cpu')); mA.eval()
mB = make_rn(5); mB.load_state_dict(torch.load(f'results/parentB_s{SEED}.pth', weights_only=True, map_location='cpu')); mB.eval()

with open('results/parents_strong.json') as f:
    pdata = json.load(f)[str(SEED)]
parent_pc = {}
for c, a in pdata['pcA'].items(): parent_pc[int(c)] = a
for c, a in pdata['pcB'].items(): parent_pc[int(c)] = a
print(f"Parents: A={pdata['A']:.3f} B={pdata['B']:.3f}", flush=True)

sdA = mA.state_dict()
sdB = mB.state_dict()
wA, bA = sdA['fc.weight'], sdA['fc.bias']  # [5, 512]
wB, bB = sdB['fc.weight'], sdB['fc.bias']

# ═══════════════════════════════════════════
# Block definitions for ResNet-18
# ═══════════════════════════════════════════
BLOCK_DEFS = [
    ('conv1_bn1', ['conv1.weight', 'bn1.weight', 'bn1.bias',
                    'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked']),
    ('layer1.0',  [k for k in sdA if k.startswith('layer1.0')]),
    ('layer1.1',  [k for k in sdA if k.startswith('layer1.1')]),
    ('layer2.0',  [k for k in sdA if k.startswith('layer2.0')]),
    ('layer2.1',  [k for k in sdA if k.startswith('layer2.1')]),
    ('layer3.0',  [k for k in sdA if k.startswith('layer3.0')]),
    ('layer3.1',  [k for k in sdA if k.startswith('layer3.1')]),
    ('layer4.0',  [k for k in sdA if k.startswith('layer4.0')]),
    ('layer4.1',  [k for k in sdA if k.startswith('layer4.1')]),
]
N_BLOCKS = len(BLOCK_DEFS)
print(f"Blocks: {N_BLOCKS}, keys covered: {sum(len(b[1]) for b in BLOCK_DEFS)}", flush=True)

# ═══════════════════════════════════════════
# ENT Chromosome for CNN
# ═══════════════════════════════════════════
class CNNENTChromosome:
    """Binary block selection + float routing."""
    
    def __init__(self):
        # Binary: True = use parent A, False = use parent B
        self.blocks = np.ones(N_BLOCKS, dtype=bool)  # default: all from A
        # Per-class routing: >0 → A, <0 → B (sigmoid applied)
        self.routing = np.zeros(10)
        # FC scales
        self.scale_a = 1.0
        self.scale_b = 1.0
    
    def copy(self):
        c = CNNENTChromosome()
        c.blocks = self.blocks.copy()
        c.routing = self.routing.copy()
        c.scale_a = self.scale_a
        c.scale_b = self.scale_b
        return c


def mutate(chrom, p_flip=0.15, sigma=0.3):
    c = chrom.copy()
    # Bit-flip on blocks
    flips = np.random.random(N_BLOCKS) < p_flip
    c.blocks[flips] = ~c.blocks[flips]
    # Gaussian mutation on routing
    c.routing += np.random.randn(10) * sigma
    # Gaussian mutation on scales
    c.scale_a += np.random.randn() * sigma * 0.3
    c.scale_b += np.random.randn() * sigma * 0.3
    c.scale_a = max(0.1, min(3.0, c.scale_a))
    c.scale_b = max(0.1, min(3.0, c.scale_b))
    return c


def crossover(p1, p2):
    child = p1.copy()
    # Uniform crossover on blocks
    swap = np.random.random(N_BLOCKS) > 0.5
    child.blocks[swap] = p2.blocks[swap]
    # Blend crossover on routing
    alpha = np.random.random()
    child.routing = alpha * p1.routing + (1-alpha) * p2.routing
    child.scale_a = alpha * p1.scale_a + (1-alpha) * p2.scale_a
    child.scale_b = alpha * p1.scale_b + (1-alpha) * p2.scale_b
    return child


# ═══════════════════════════════════════════
# Build model from chromosome
# ═══════════════════════════════════════════
def build_model(chrom):
    """Build ONE ResNet-18(10 classes) from chromosome.
    
    Each block's weights come entirely from A or B (binary, no interpolation).
    FC layer assembled from both parents with per-class routing.
    """
    merged = make_rn(10)
    sd = {}
    
    # Backbone: binary block selection
    for bi, (bname, keys) in enumerate(BLOCK_DEFS):
        source = sdA if chrom.blocks[bi] else sdB
        for k in keys:
            if k in source:
                sd[k] = source[k].clone()
    
    # FC: per-class routing
    fc_w = torch.zeros(10, 512)
    fc_b = torch.zeros(10)
    for c in ALL:
        alpha = 1.0 / (1.0 + np.exp(-chrom.routing[c]))  # sigmoid
        if c in clA:
            ci = clA.index(c)
            fc_w[c] = alpha * chrom.scale_a * wA[ci] + (1-alpha) * chrom.scale_b * wB[ci % len(clB)] 
            fc_b[c] = alpha * chrom.scale_a * bA[ci] + (1-alpha) * chrom.scale_b * bB[ci % len(clB)]
        else:
            ci = clB.index(c)
            fc_w[c] = (1-alpha) * chrom.scale_a * wA[ci % len(clA)] + alpha * chrom.scale_b * wB[ci]
            fc_b[c] = (1-alpha) * chrom.scale_a * bA[ci % len(clA)] + alpha * chrom.scale_b * bB[ci]
    sd['fc.weight'] = fc_w
    sd['fc.bias'] = fc_b
    
    merged.load_state_dict(sd)
    return merged


# ═══════════════════════════════════════════
# Fitness function
# ═══════════════════════════════════════════
def fitness(chrom):
    model = build_model(chrom)
    model.to(DEV).eval()
    
    with torch.no_grad():
        preds = torch.cat([model(X_cal[i:i+1024]).argmax(1) for i in range(0,len(X_cal),1024)])
    
    acc = (preds == y_cal).float().mean().item()
    
    retained = 0
    pc_vals = []
    for c in ALL:
        mask = y_cal == c
        if mask.sum() == 0: continue
        ca = (preds[mask]==c).float().mean().item()
        pc_vals.append(ca)
        if ca >= 0.9 * parent_pc.get(c, 0):
            retained += 1
    
    mn = min(pc_vals) if pc_vals else 0
    
    # Fitness: retention + accuracy + balance
    return 0.3*acc + 0.3*(retained/10) + 0.3*mn + 0.1*np.mean(pc_vals)


# ═══════════════════════════════════════════
# EA (μ+λ) — same as e23
# ═══════════════════════════════════════════
print(f"\n--- ENT-CNN EA (seed={SEED}) ---", flush=True)

POP_SIZE = 30
N_GEN = 60

# Initialize population with smart seeds
population = []

# Seed 1: all from A
c1 = CNNENTChromosome()
c1.blocks[:] = True
c1.routing[:5] = 2.0; c1.routing[5:] = -2.0
population.append(c1)

# Seed 2: all from B
c2 = CNNENTChromosome()
c2.blocks[:] = False
c2.routing[:5] = -2.0; c2.routing[5:] = 2.0
population.append(c2)

# Seed 3: early from A, late from B
c3 = CNNENTChromosome()
c3.blocks[:5] = True; c3.blocks[5:] = False
c3.routing[:5] = 2.0; c3.routing[5:] = -2.0
population.append(c3)

# Seed 4: early from B, late from A
c4 = CNNENTChromosome()
c4.blocks[:5] = False; c4.blocks[5:] = True
c4.routing[:5] = -2.0; c4.routing[5:] = 2.0
population.append(c4)

# Seed 5: alternating
c5 = CNNENTChromosome()
c5.blocks = np.array([i%2==0 for i in range(N_BLOCKS)])
c5.routing[:5] = 1.5; c5.routing[5:] = -1.5
population.append(c5)

# Rest: random
while len(population) < POP_SIZE:
    c = CNNENTChromosome()
    c.blocks = np.random.random(N_BLOCKS) > 0.5
    c.routing = np.random.randn(10) * 1.5
    c.scale_a = 0.5 + np.random.random()
    c.scale_b = 0.5 + np.random.random()
    population.append(c)

# Evolution
best_fitness = -1
best_chrom = None

for gen in range(N_GEN):
    fitnesses = [fitness(c) for c in population]
    
    gen_best = max(fitnesses)
    gen_best_idx = np.argmax(fitnesses)
    if gen_best > best_fitness:
        best_fitness = gen_best
        best_chrom = population[gen_best_idx].copy()
    
    if gen % 10 == 0 or gen == N_GEN - 1:
        b = best_chrom.blocks
        block_str = ''.join(['A' if x else 'B' for x in b])
        print(f"  Gen {gen:>3d}: fit={gen_best:.4f} best={best_fitness:.4f} "
              f"blocks=[{block_str}] ({time.time()-t0:.0f}s)", flush=True)
    
    # Selection + reproduction
    new_pop = [best_chrom.copy()]  # elitism
    
    while len(new_pop) < POP_SIZE:
        # Tournament selection (size 3)
        t_idx = random.sample(range(POP_SIZE), min(3, POP_SIZE))
        t_fit = [fitnesses[i] for i in t_idx]
        p1 = population[t_idx[np.argmax(t_fit)]]
        
        t_idx = random.sample(range(POP_SIZE), min(3, POP_SIZE))
        t_fit = [fitnesses[i] for i in t_idx]
        p2 = population[t_idx[np.argmax(t_fit)]]
        
        # Crossover + mutation
        if random.random() < 0.7:
            child = crossover(p1, p2)
        else:
            child = p1.copy()
        
        # Adaptive mutation rate
        p_flip = max(0.05, 0.2 - gen * 0.003)
        child = mutate(child, p_flip=p_flip)
        new_pop.append(child)
    
    population = new_pop

print(f"\n  EA converged: {N_GEN} gens, {N_GEN*POP_SIZE} evals, {time.time()-t0:.0f}s", flush=True)

# ═══════════════════════════════════════════
# Evaluate best on TEST
# ═══════════════════════════════════════════
merged = build_model(best_chrom)
merged.eval()

with torch.no_grad():
    preds = torch.cat([merged(X_te[i:i+512]).argmax(1) for i in range(0,len(X_te),512)])

acc = (preds == y_te).float().mean().item()
pc = {}; retained = 0
for c in ALL:
    mask = y_te == c
    ca = (preds[mask]==c).float().mean().item()
    pc[c] = ca
    if ca >= 0.9 * parent_pc.get(c, 0):
        retained += 1

mn = min(pc.values())
aM = np.mean([pc[c] for c in clA])
bM = np.mean([pc[c] for c in clB])
bal = min(aM,bM)/(max(aM,bM)+1e-10)

print(f"\n{'='*60}")
print(f"ENT-CNN RESULT (seed={SEED})")
print(f"Parents: A={pdata['A']:.3f} B={pdata['B']:.3f}")
print(f"Merged:  acc={acc:.4f} retained={retained}/10 bal={bal:.4f} min={mn:.4f}")
print(f"Blocks:  {''.join(['A' if x else 'B' for x in best_chrom.blocks])}")

for bi, (bname, _) in enumerate(BLOCK_DEFS):
    src = 'A' if best_chrom.blocks[bi] else 'B'
    print(f"  {bname:12s}: {src}")

routing = 1.0/(1.0+np.exp(-best_chrom.routing))
print(f"Routing: {[round(r,2) for r in routing]}")
print(f"Scales:  A={best_chrom.scale_a:.3f} B={best_chrom.scale_b:.3f}")

print(f"\nPer-class:")
for c in ALL:
    par = parent_pc.get(c, 0)
    ret = pc[c]/par if par > 0 else 0
    flag = '✅' if ret >= 0.9 else '❌'
    src = 'A' if c in clA else 'B'
    print(f"  {c} ({src}): parent={par:.3f} merged={pc[c]:.3f} retention={ret:.2f} {flag}")

elapsed = time.time()-t0
print(f"\nTime: {elapsed:.0f}s")

# Save
result = {
    'seed': SEED, 'acc': round(acc,4), 'retained': retained,
    'bal': round(bal,4), 'min': round(mn,4),
    'blocks': best_chrom.blocks.tolist(),
    'routing': routing.tolist(),
    'scales': [best_chrom.scale_a, best_chrom.scale_b],
    'pc': {c: round(v,4) for c,v in pc.items()},
    'parent_pc': parent_pc,
    'gen': N_GEN, 'evals': N_GEN*POP_SIZE,
    'time_s': round(elapsed,1)
}

fpath = 'results/ent_cnn_results.json'
try:
    with open(fpath) as f: acc_data = json.load(f)
except: acc_data = {}
acc_data[str(SEED)] = result
with open(fpath,'w') as f: json.dump(acc_data, f, indent=2)

print(f"\nmetric_retained: {retained}")
print(f"metric_acc: {round(acc,4)}")
print(f"metric_bal: {round(bal,4)}")
print("Done!", flush=True)
