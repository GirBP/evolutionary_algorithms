#!/usr/bin/env python3
"""Task 6: Transformer ENT — 3 seeds.
Adapted from e32_llm_ent.py. Tiny GPT: arithmetic vs word patterns.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F_t
import random, time, copy, math, json

t0 = time.time()
SEEDS = [42, 123, 456]

# ═══════════════════════════════════════════
# Architecture (same as e32)
# ═══════════════════════════════════════════
class TinyGPT(nn.Module):
    def __init__(s, vocab_size, d_model=64, n_heads=4, n_layers=4, max_len=48):
        super().__init__()
        s.d_model=d_model;s.n_heads=n_heads;s.n_layers=n_layers
        s.vocab_size=vocab_size;s.max_len=max_len
        s.tok_emb=nn.Embedding(vocab_size,d_model)
        s.pos_emb=nn.Embedding(max_len,d_model)
        s.layers=nn.ModuleList([TransformerBlock(d_model,n_heads) for _ in range(n_layers)])
        s.ln_f=nn.LayerNorm(d_model)
        s.head=nn.Linear(d_model,vocab_size,bias=False)
        s.apply(s._init_weights)
    def _init_weights(s,m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,std=0.02)
    def forward(s,x):
        B,T=x.shape;pos=torch.arange(T,device=x.device).unsqueeze(0)
        h=s.tok_emb(x)+s.pos_emb(pos)
        mask=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
        for layer in s.layers: h=layer(h,mask)
        return s.head(s.ln_f(h))

class TransformerBlock(nn.Module):
    def __init__(s,d,nh):
        super().__init__()
        s.ln1=nn.LayerNorm(d);s.attn=MHA(d,nh);s.ln2=nn.LayerNorm(d);s.ffn=FFN(d)
    def forward(s,x,mask=None):
        x=x+s.attn(s.ln1(x),mask);return x+s.ffn(s.ln2(x))

class MHA(nn.Module):
    def __init__(s,d,nh):
        super().__init__();s.n_heads=nh;s.d_head=d//nh
        s.qkv=nn.Linear(d,3*d);s.proj=nn.Linear(d,d)
    def forward(s,x,mask=None):
        B,T,C=x.shape
        qkv=s.qkv(x).reshape(B,T,3,s.n_heads,s.d_head).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        att=(q@k.transpose(-2,-1))/math.sqrt(s.d_head)
        if mask is not None: att=att.masked_fill(mask.unsqueeze(0).unsqueeze(0),float('-inf'))
        return (F_t.softmax(att,dim=-1)@v).transpose(1,2).reshape(B,T,C).pipe(s.proj) if hasattr(torch.Tensor,'pipe') else s.proj((F_t.softmax(att,dim=-1)@v).transpose(1,2).reshape(B,T,C))

class FFN(nn.Module):
    def __init__(s,d,exp=4):
        super().__init__();s.up=nn.Linear(d,d*exp);s.down=nn.Linear(d*exp,d);s.act=nn.GELU()
    def forward(s,x): return s.down(s.act(s.up(x)))

chars=list("0123456789+-*= abcdefghijklmnopqrstuvwxyz.,\n")
ch2id={c:i for i,c in enumerate(chars)};VOCAB=len(chars)
def encode(s): return [ch2id.get(c,0) for c in s]

def gen_arithmetic(n=2000,max_len=32):
    texts=[]
    for _ in range(n):
        op=random.choice(['+','-','*']);a=random.randint(1,50);b=random.randint(1,50)
        r=a+b if op=='+' else (a-b if op=='-' else a*b)
        texts.append(f"{a}{op}{b}={r}\n")
    ids=encode(''.join(texts))
    return torch.tensor([ids[i:i+max_len] for i in range(0,len(ids)-max_len,max_len//2)],dtype=torch.long)

def gen_words(n=2000,max_len=32):
    anim=['cat','dog','bird','fish','frog','bear','wolf','duck','fox','owl']
    col=['red','blue','green','pink','gold','gray','dark','cyan']
    act=['sat','ran','ate','saw','met','bit','got','hid']
    pl=['mat','bed','den','log','mud','sun','fog','dam']
    texts=[]
    for _ in range(n):
        p=random.choice([
            f"the {random.choice(anim)} {random.choice(act)} on the {random.choice(pl)}",
            f"{random.choice(col)} {random.choice(anim)} and {random.choice(col)} {random.choice(anim)}",
            f"a {random.choice(anim)} is {random.choice(col)} and {random.choice(act)}",
        ])
        texts.append(p+'\n')
    ids=encode(''.join(texts))
    return torch.tensor([ids[i:i+max_len] for i in range(0,len(ids)-max_len,max_len//2)],dtype=torch.long)

def train_gpt(model,data,epochs=30,lr=3e-3):
    opt=torch.optim.AdamW(model.parameters(),lr=lr);model.train()
    for ep in range(epochs):
        idx=torch.randperm(len(data))[:256];batch=data[idx]
        x,y=batch[:,:-1],batch[:,1:]
        loss=F_t.cross_entropy(model(x).reshape(-1,VOCAB),y.reshape(-1))
        opt.zero_grad();loss.backward();opt.step()
    return model

def eval_pp(model,data,n=200):
    model.eval();tl=0;tt=0
    with torch.no_grad():
        for i in range(0,min(n,len(data)),32):
            b=data[i:i+32];x,y=b[:,:-1],b[:,1:]
            tl+=F_t.cross_entropy(model(x).reshape(-1,VOCAB),y.reshape(-1),reduction='sum').item()
            tt+=y.numel()
    return math.exp(tl/tt)

def merge_transformer_cma(modelA,modelB,dataAv,dataBv,seed):
    """CMA-ES per-layer alpha optimization (simpler than full ENT)."""
    import cma
    n_layers=modelA.n_layers
    # Gene: [embed_alpha, layer0_alpha, ..., layerN_alpha, head_alpha]
    dim = n_layers + 2
    
    def build(x):
        a = 1/(1+np.exp(-np.array(x)))  # sigmoid
        m = copy.deepcopy(modelA)
        with torch.no_grad():
            # Embeddings
            m.tok_emb.weight.data = a[0]*modelA.tok_emb.weight.data+(1-a[0])*modelB.tok_emb.weight.data
            m.pos_emb.weight.data = a[0]*modelA.pos_emb.weight.data+(1-a[0])*modelB.pos_emb.weight.data
            for l in range(n_layers):
                al = a[l+1]
                lA,lB,lM = modelA.layers[l],modelB.layers[l],m.layers[l]
                for p_name in ['ln1.weight','ln1.bias','ln2.weight','ln2.bias']:
                    pA = dict(lA.named_parameters())[p_name]
                    pB = dict(lB.named_parameters())[p_name]
                    dict(lM.named_parameters())[p_name].data = al*pA.data+(1-al)*pB.data
                for p_name in ['attn.qkv.weight','attn.qkv.bias','attn.proj.weight','attn.proj.bias',
                               'ffn.up.weight','ffn.up.bias','ffn.down.weight','ffn.down.bias']:
                    pA = dict(lA.named_parameters())[p_name]
                    pB = dict(lB.named_parameters())[p_name]
                    dict(lM.named_parameters())[p_name].data = al*pA.data+(1-al)*pB.data
            ah = a[-1]
            m.ln_f.weight.data = ah*modelA.ln_f.weight.data+(1-ah)*modelB.ln_f.weight.data
            m.ln_f.bias.data = ah*modelA.ln_f.bias.data+(1-ah)*modelB.ln_f.bias.data
            m.head.weight.data = ah*modelA.head.weight.data+(1-ah)*modelB.head.weight.data
        return m
    
    es = cma.CMAEvolutionStrategy(np.zeros(dim), 1.5, {'maxiter':12, 'popsize':8, 'seed':seed, 'verbose':-1})
    best_s = float('inf'); best_x = None
    while not es.stop():
        sols = es.ask(); scores = []
        for x in sols:
            m = build(x)
            ppA = eval_pp(m, dataAv, n=80)
            ppB = eval_pp(m, dataBv, n=80)
            harm = 2*ppA*ppB/(ppA+ppB+1e-10)
            scores.append(harm)
        es.tell(sols, scores)
        if min(scores) < best_s:
            best_s = min(scores)
            best_x = sols[np.argmin(scores)]
    
    merged = build(best_x)
    alphas = 1/(1+np.exp(-best_x))
    return merged, alphas

# ═══════════════════════════════════════════
# Run per seed
# ═══════════════════════════════════════════
all_results = {}
D_MODEL=64;N_HEADS=4;N_LAYERS=4;MAX_LEN=32

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"  SEED={seed}")
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    
    dataA = gen_arithmetic(1500, MAX_LEN)
    dataB = gen_words(1500, MAX_LEN)
    dataAv = gen_arithmetic(300, MAX_LEN)
    dataBv = gen_words(300, MAX_LEN)
    
    mA = TinyGPT(VOCAB, D_MODEL, N_HEADS, N_LAYERS, MAX_LEN)
    mA = train_gpt(mA, dataA, epochs=30)
    mB = TinyGPT(VOCAB, D_MODEL, N_HEADS, N_LAYERS, MAX_LEN)
    mB = train_gpt(mB, dataB, epochs=30)
    
    ppAA = eval_pp(mA, dataAv); ppAB = eval_pp(mA, dataBv)
    ppBA = eval_pp(mB, dataAv); ppBB = eval_pp(mB, dataBv)
    
    res = {'parentA': {'arith': round(ppAA,1), 'words': round(ppAB,1)},
           'parentB': {'arith': round(ppBA,1), 'words': round(ppBB,1)}}
    
    # Average
    avg = copy.deepcopy(mA)
    with torch.no_grad():
        for pA,pB,pM in zip(mA.parameters(),mB.parameters(),avg.parameters()):
            pM.data = 0.5*pA.data+0.5*pB.data
    ppAvgA = eval_pp(avg, dataAv); ppAvgB = eval_pp(avg, dataBv)
    hAvg = 2*ppAvgA*ppAvgB/(ppAvgA+ppAvgB)
    res['Average'] = {'arith': round(ppAvgA,1), 'words': round(ppAvgB,1), 'harmonic': round(hAvg,1)}
    
    # TA
    ta = copy.deepcopy(mA)
    with torch.no_grad():
        for pA,pB,pM in zip(mA.parameters(),mB.parameters(),ta.parameters()):
            pM.data = pA.data+0.3*(pB.data-pA.data)
    ppTAA = eval_pp(ta, dataAv); ppTAB = eval_pp(ta, dataBv)
    hTA = 2*ppTAA*ppTAB/(ppTAA+ppTAB)
    res['TA'] = {'arith': round(ppTAA,1), 'words': round(ppTAB,1), 'harmonic': round(hTA,1)}
    
    # ENT (CMA-ES per-layer)
    merged, alphas = merge_transformer_cma(mA, mB, dataAv, dataBv, seed)
    ppEntA = eval_pp(merged, dataAv); ppEntB = eval_pp(merged, dataBv)
    hENT = 2*ppEntA*ppEntB/(ppEntA+ppEntB)
    res['ENT'] = {'arith': round(ppEntA,1), 'words': round(ppEntB,1), 'harmonic': round(hENT,1),
                  'alphas': [round(float(a),3) for a in alphas]}
    
    all_results[seed] = res
    print(f"  A: arith={ppAA:.1f} words={ppAB:.1f}")
    print(f"  B: arith={ppBA:.1f} words={ppBB:.1f}")
    print(f"  Avg: harm={hAvg:.1f} | TA: harm={hTA:.1f} | ENT: harm={hENT:.1f}")
    print(f"  ENT alphas: {[round(float(a),2) for a in alphas]}")

# ═══════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════
print(f"\n{'='*60}")
print("Transformer ENT Results (3 seeds)")
methods = ['Average', 'TA', 'ENT']
for m in methods:
    harms = [all_results[s][m]['harmonic'] for s in SEEDS]
    print(f"  {m:<8}: harmonic PP = {np.mean(harms):.1f}±{np.std(harms):.1f}")

# Is ENT best?
ent_harms = [all_results[s]['ENT']['harmonic'] for s in SEEDS]
avg_harms = [all_results[s]['Average']['harmonic'] for s in SEEDS]
ta_harms = [all_results[s]['TA']['harmonic'] for s in SEEDS]
ent_wins = sum(1 for s in SEEDS if all_results[s]['ENT']['harmonic'] < all_results[s]['Average']['harmonic'] and all_results[s]['ENT']['harmonic'] < all_results[s]['TA']['harmonic'])
print(f"  ENT wins: {ent_wins}/{len(SEEDS)} seeds")

elapsed = time.time()-t0
print(f"  Total: {elapsed:.0f}s ({elapsed/60:.1f}min)")

with open('results/task6_transformer.json','w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\nmetric_ent_harm_mean: {np.mean(ent_harms):.1f}")
print(f"metric_avg_harm_mean: {np.mean(avg_harms):.1f}")
print(f"metric_ent_wins: {ent_wins}")
print("Done!")
