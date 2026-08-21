#!/usr/bin/env python3
"""
E14 — Fix complementary merge: 3 strategies.
==============================================
Problem: E13 showed PCMA preserves A-knowledge (91%) but loses B (8.5%).
Test 3 fixes on the disjoint scenario (A=0-4, B=5-9):

  S1: Fine-tune merged model on B-data for a few epochs
  S2: Symmetric merge — merge A→B + merge B→A, then ensemble
  S3: Knowledge distillation — train new model to mimic both parents
"""

import numpy as np
import torch
import torch.nn as nn
import time, json

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


def load_mnist():
    import ssl; ssl._create_default_https_context = ssl._create_unverified_context
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=tf)
    te = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=tf)
    X_tr = torch.stack([tr[i][0] for i in range(len(tr))])
    y_tr = torch.tensor([tr[i][1] for i in range(len(tr))])
    X_te = torch.stack([te[i][0] for i in range(2000)])
    y_te = torch.tensor([te[i][1] for i in range(2000)])
    return X_tr, y_tr, X_te, y_te


class MLP(nn.Module):
    def __init__(s, arch):
        super().__init__()
        layers = []
        for i in range(len(arch)-1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i < len(arch)-2: layers.append(nn.ReLU())
        s.net = nn.Sequential(*layers); s.arch = arch
    def forward(s, x): return s.net(x)


def ev(model, X, y):
    model.eval()
    with torch.no_grad(): return (model(X).argmax(1)==y).float().mean().item()

def per_class(model, X, y):
    model.eval()
    with torch.no_grad(): preds = model(X).argmax(1)
    return {c: (preds[y==c]==c).float().mean().item() if (y==c).sum()>0 else 0 for c in range(10)}

def train_on_classes(arch, X, y, classes, epochs=15, lr=0.003):
    mask = torch.zeros(len(y), dtype=torch.bool)
    for c in classes: mask |= (y==c)
    Xs, ys = X[mask][:5000], y[mask][:5000]
    m = MLP(arch); opt = torch.optim.Adam(m.parameters(), lr=lr); m.train()
    for _ in range(epochs):
        l = nn.CrossEntropyLoss()(m(Xs), ys); opt.zero_grad(); l.backward(); opt.step()
    return m


# ─── PCMA (compact) ─────────────────────────────────────────────────

def pcma(mA, mB, X_cal, X_val, y_val, maxiter=30, popsize=12):
    import cma, copy
    aA, aB = mA.arch, mB.arch
    nh = len(aA)-2
    acA, acB = [], []
    with torch.no_grad():
        hA, hB = X_cal, X_cal
        for ma, mb in zip(mA.net, mB.net):
            hA, hB = ma(hA), mb(hB)
            if isinstance(ma, nn.ReLU): acA.append(hA.numpy().copy()); acB.append(hB.numpy().copy())
    svd, si = [], []
    for i in range(nh):
        dm = min(aA[i+1], aB[i+1]); C = acA[i].T@acB[i]
        U,S,Vt = np.linalg.svd(C, full_matrices=False)
        si.append(S[:dm]/(S[0]+1e-10)); svd.append((U[:,:dm], Vt[:dm,:]))
    s0 = np.concatenate(si); nds = len(s0); nl = nh+1
    WA = [p.detach().numpy() for p in mA.parameters()]
    WB = [p.detach().numpy() for p in mB.parameters()]
    
    def build(sv, al):
        maps, off = [], 0
        for i in range(nh):
            U,Vt = svd[i]; d=U.shape[1]
            maps.append(U@np.diag(sv[off:off+d])@Vt); off+=d
        ps = []
        for li in range(nl):
            a = np.clip(al[li],0.05,0.95)
            wa,ba = WA[li*2],WA[li*2+1]; wb,bb = WB[li*2],WB[li*2+1]
            if li==0: wm=maps[0]@wb; bm=maps[0]@bb
            elif li<nh: wm=maps[li]@wb@maps[li-1].T; bm=maps[li]@bb
            else: wm=wb@maps[-1].T; bm=bb
            ps.append(a*wa+(1-a)*wm); ps.append(a*ba+(1-a)*bm)
        m = MLP(aA)
        with torch.no_grad():
            for p,v in zip(m.parameters(),ps): p.copy_(torch.tensor(v,dtype=torch.float32))
        return m
    
    x0 = np.concatenate([s0,[0.5]*nl]); nd=len(x0)
    def fit(x):
        try: return -ev(build(x[:nds],x[nds:].tolist()),X_val,y_val)
        except: return 1.0
    es = cma.CMAEvolutionStrategy(x0.tolist(),0.3,{
        'maxiter':maxiter,'popsize':popsize,'seed':SEED,
        'verbose':-9,'verb_disp':0,'verb_log':0,
        'bounds':[[-3]*nds+[0.1]*nl,[3]*nds+[0.9]*nl],
        'CMA_diagonal':nd>80})
    while not es.stop():
        sols=es.ask(); es.tell(sols,[fit(np.array(s)) for s in sols])
    b=np.array(es.result.xbest)
    return build(b[:nds],b[nds:].tolist()), b[nds:].tolist()


# ─── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("="*75)
    print("  E14: Fix complementary merge — 3 strategies")
    print("  Scenario: A trains 0-4, B trains 5-9")
    print("="*75)
    
    X_tr, y_tr, X_te, y_te = load_mnist()
    idx = torch.randperm(len(X_tr), generator=torch.Generator().manual_seed(0))
    X_val, y_val = X_tr[idx[50000:55000]], y_tr[idx[50000:55000]]
    X_cal = X_tr[idx[:3000]]
    
    archA = [784, 128, 64, 10]
    archB = [784, 256, 128, 10]
    classesA, classesB = list(range(5)), list(range(5,10))
    
    mA = train_on_classes(archA, X_tr, y_tr, classesA)
    mB = train_on_classes(archB, X_tr, y_tr, classesB)
    
    pcA, pcB = per_class(mA, X_te, y_te), per_class(mB, X_te, y_te)
    print(f"\n  Parents:")
    print(f"  A (0-4): {ev(mA,X_te,y_te):.3f}  per-class: {[round(pcA[c],2) for c in range(10)]}")
    print(f"  B (5-9): {ev(mB,X_te,y_te):.3f}  per-class: {[round(pcB[c],2) for c in range(10)]}")
    
    # ─── Baseline: raw PCMA (A→B direction) ──────────────────────────
    print(f"\n{'━'*75}")
    print("  BASELINE: Raw PCMA (A-architecture, no fix)")
    merged_raw, _ = pcma(mA, mB, X_cal, X_val, y_val)
    pc_raw = per_class(merged_raw, X_te, y_te)
    print(f"  Overall: {ev(merged_raw,X_te,y_te):.3f}")
    print(f"  A-classes (0-4): {np.mean([pc_raw[c] for c in range(5)]):.3f}")
    print(f"  B-classes (5-9): {np.mean([pc_raw[c] for c in range(5,10)]):.3f}")
    
    # ─── S1: Fine-tune on B-data ─────────────────────────────────────
    print(f"\n{'━'*75}")
    print("  S1: Fine-tune merged model on COMBINED A+B data")
    import copy
    
    # Get combined data (small sample from each)
    maskA = torch.zeros(len(y_tr), dtype=torch.bool)
    maskB = torch.zeros(len(y_tr), dtype=torch.bool)
    for c in classesA: maskA |= (y_tr==c)
    for c in classesB: maskB |= (y_tr==c)
    
    # 500 samples from each split for fine-tuning
    XA_ft = X_tr[maskA][:500]; yA_ft = y_tr[maskA][:500]
    XB_ft = X_tr[maskB][:500]; yB_ft = y_tr[maskB][:500]
    X_ft = torch.cat([XA_ft, XB_ft]); y_ft = torch.cat([yA_ft, yB_ft])
    
    for n_epochs in [1, 3, 5, 10]:
        m_ft = copy.deepcopy(merged_raw)
        opt = torch.optim.Adam(m_ft.parameters(), lr=0.001)
        m_ft.train()
        for _ in range(n_epochs):
            perm = torch.randperm(len(X_ft))
            l = nn.CrossEntropyLoss()(m_ft(X_ft[perm]), y_ft[perm])
            opt.zero_grad(); l.backward(); opt.step()
        
        pc_ft = per_class(m_ft, X_te, y_te)
        a_acc = np.mean([pc_ft[c] for c in range(5)])
        b_acc = np.mean([pc_ft[c] for c in range(5,10)])
        all_acc = ev(m_ft, X_te, y_te)
        print(f"  {n_epochs:>2d} epochs: all={all_acc:.3f}  A(0-4)={a_acc:.3f}  B(5-9)={b_acc:.3f}")
    
    best_ft = m_ft  # 10 epochs
    
    # ─── S2: Symmetric merge + ensemble ──────────────────────────────
    print(f"\n{'━'*75}")
    print("  S2: Symmetric merge (A→B + B→A) + ensemble")
    
    # A→B: merged into A's architecture
    merged_ab, _ = pcma(mA, mB, X_cal, X_val, y_val)
    # B→A: merged into B's architecture
    merged_ba, _ = pcma(mB, mA, X_cal, X_val, y_val)
    
    # Ensemble: average logits
    class Ensemble(nn.Module):
        def __init__(self, m1, m2):
            super().__init__()
            self.m1, self.m2 = m1, m2
        def forward(self, x):
            with torch.no_grad():
                return 0.5 * self.m1(x) + 0.5 * self.m2(x)
    
    ens = Ensemble(merged_ab, merged_ba)
    pc_ab = per_class(merged_ab, X_te, y_te)
    pc_ba = per_class(merged_ba, X_te, y_te)
    pc_ens = per_class(ens, X_te, y_te)
    
    print(f"  A→B: all={ev(merged_ab,X_te,y_te):.3f}  "
          f"A={np.mean([pc_ab[c] for c in range(5)]):.3f}  "
          f"B={np.mean([pc_ab[c] for c in range(5,10)]):.3f}")
    print(f"  B→A: all={ev(merged_ba,X_te,y_te):.3f}  "
          f"A={np.mean([pc_ba[c] for c in range(5)]):.3f}  "
          f"B={np.mean([pc_ba[c] for c in range(5,10)]):.3f}")
    print(f"  Ens: all={ev(ens,X_te,y_te):.3f}  "
          f"A={np.mean([pc_ens[c] for c in range(5)]):.3f}  "
          f"B={np.mean([pc_ens[c] for c in range(5,10)]):.3f}")
    
    # ─── S3: Knowledge distillation ──────────────────────────────────
    print(f"\n{'━'*75}")
    print("  S3: Knowledge distillation (train student to mimic both)")
    
    # Student: new model with A's architecture
    # Teacher signals: softmax(mA(x)/T) and softmax(mB(x)/T) 
    # Loss: KL(student || teacher_A) + KL(student || teacher_B) on unlabeled data
    
    student = MLP(archA)
    T = 4.0  # temperature
    opt = torch.optim.Adam(student.parameters(), lr=0.002)
    
    # Use ALL training data (unlabeled — we only use teacher soft labels)
    X_distill = X_tr[:5000]
    
    mA.eval(); mB.eval()
    with torch.no_grad():
        logitsA = mA(X_distill) / T
        logitsB = mB(X_distill) / T
        softA = torch.softmax(logitsA, dim=1)
        softB = torch.softmax(logitsB, dim=1)
        # Combined teacher: max confidence per class
        soft_teacher = torch.max(softA, softB)
        soft_teacher = soft_teacher / soft_teacher.sum(dim=1, keepdim=True)  # renormalize
    
    student.train()
    for epoch in range(20):
        student_logits = student(X_distill) / T
        student_soft = torch.log_softmax(student_logits, dim=1)
        # KL divergence loss
        loss = nn.KLDivLoss(reduction='batchmean')(student_soft, soft_teacher) * (T*T)
        opt.zero_grad(); loss.backward(); opt.step()
        
        if epoch in [0, 4, 9, 19]:
            pc_s = per_class(student, X_te, y_te)
            a_acc = np.mean([pc_s[c] for c in range(5)])
            b_acc = np.mean([pc_s[c] for c in range(5,10)])
            print(f"  epoch {epoch+1:>2d}: all={ev(student,X_te,y_te):.3f}  "
                  f"A={a_acc:.3f}  B={b_acc:.3f}  loss={loss.item():.3f}")
    
    # ─── Final comparison ────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("  FINAL COMPARISON")
    print(f"{'='*75}")
    
    methods = [
        ("Raw PCMA", merged_raw),
        ("S1: Fine-tune (10ep)", best_ft),
        ("S2: Symmetric ens", ens),
        ("S3: KD student", student),
    ]
    
    print(f"\n  {'Method':<22s} {'Overall':>8s} {'A(0-4)':>8s} {'B(5-9)':>8s} {'Balance':>8s}")
    print(f"  {'─'*55}")
    
    results = []
    for name, m in methods:
        pc = per_class(m, X_te, y_te)
        acc_all = ev(m, X_te, y_te)
        a_acc = np.mean([pc[c] for c in range(5)])
        b_acc = np.mean([pc[c] for c in range(5,10)])
        balance = min(a_acc, b_acc) / (max(a_acc, b_acc) + 1e-10)
        
        print(f"  {name:<22s} {acc_all:>8.3f} {a_acc:>8.3f} {b_acc:>8.3f} {balance:>8.3f}")
        results.append({
            'method': name, 'overall': round(acc_all,4),
            'A_acc': round(a_acc,4), 'B_acc': round(b_acc,4),
            'balance': round(balance,4),
            'per_class': {str(c): round(pc[c],4) for c in range(10)},
        })
    
    # Per-class detail
    print(f"\n  {'Class':>6s}", end="")
    for name, _ in methods: print(f"  {name[:10]:>10s}", end="")
    print()
    for c in range(10):
        print(f"  {c:>6d}", end="")
        for name, m in methods:
            pc = per_class(m, X_te, y_te)
            print(f"  {pc[c]:>10.3f}", end="")
        best_parent = max(pcA[c], pcB[c])
        print(f"  (parent={best_parent:.3f})")
    
    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.1f}s")
    
    with open("results_e14.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
