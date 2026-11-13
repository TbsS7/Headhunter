"""HeadHunter v0.12.0: core, MLPs, sufficient circuit, edges (hybrid) and GQA.

Usage:
    pip install git+https://github.com/TbsS7/Headhunter.git
    import headhunter as hh
    m = hh.load_model("gpt2-small")
    hh.selftest(m, "IOI")
"""
import gc, os, json, time, random, platform
from collections import defaultdict
import numpy as np, pandas as pd, torch
from transformer_lens import HookedTransformer

VERSION = "0.12.0"
IG_SCHEME = "mid"   # integration points for the edge scores: "mid" (default) or "eapig" (same as the official EAP-IG)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
pd.set_option("display.width", 250)


# =====================================================================================
# 1. Load a model + architecture adapter
# =====================================================================================
LARGE_MODELS = {"pythia-1b": "EleutherAI/pythia-1b", "pythia-1.4b": "EleutherAI/pythia-1.4b",
           "pythia-2.8b": "EleutherAI/pythia-2.8b", "pythia-6.9b": "EleutherAI/pythia-6.9b",
           "gpt2-xl": "openai-community/gpt2-xl"}

from transformer_lens.components import Unembed
from transformer_lens.components.mlps.gated_mlp import GatedMLP

def _unembed(self, residual):
    return self.hook_out(residual @ self.W_U + self.b_U)

def _gated(self, x):
    if self.W_gate.device != x.device: x = x.to(self.W_gate.device)
    pre_act = self.hook_pre(x @ self.W_gate)
    if self.cfg.is_layer_norm_activation() and self.hook_mid is not None and self.ln is not None:
        post_act = self.hook_post(self.ln(self.hook_mid(self.act_fn(pre_act))))
    else:
        pre_linear = self.hook_pre_linear(x @ self.W_in)
        post_act = self.hook_post((self.act_fn(pre_act) * pre_linear) + self.b_in)
    return post_act @ self.W_out + self.b_out

def _no_weight_copies():
    """TL 3.9 copies W.T.contiguous() on every call (unembed and gated MLP); these copies are kept for the backward pass
    (~5.5 GB on Qwen2.5-1.5B). Same result in float32, without the copies."""
    Unembed.forward = _unembed; GatedMLP.forward = _gated

_no_weight_copies()

MODERN_FAMILIES = ("qwen", "llama", "gemma", "olmo", "mistral", "phi")

def load_model(name, hf_id=None, large=None, load_dtype=None, convert_on="auto"):
    """Small models: loaded directly. Large ones (or when hf_id is given): loaded in half precision -> GPU -> float32.
    Modern models (Qwen, Llama...) are loaded in bf16 (their training precision) before becoming float32."""
    gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    modern = any(k in name.lower() for k in MODERN_FAMILIES)
    hf_id = hf_id or LARGE_MODELS.get(name) or (name if modern and "/" in name else None)
    load_dtype = load_dtype or (torch.bfloat16 if modern else torch.float16)
    if large is None: large = hf_id is not None
    if large:
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(hf_id or name, torch_dtype=load_dtype, low_cpu_mem_usage=True)
        n_params = sum(p.numel() for p in hf.parameters())
        # v0.4.1: on the GPU, half precision and float32 coexist during the conversion (~6 bytes per parameter).
        # If that does not fit with room to spare, convert in RAM and only then move to the GPU (~4 bytes per parameter).
        na_cpu = convert_on == "cpu" or (convert_on == "auto" and DEV == "cuda"
                 and 6 * n_params > 0.85 * torch.cuda.get_device_properties(0).total_memory)
        m = HookedTransformer.from_pretrained_no_processing(name, hf_model=hf, dtype=load_dtype,
                                                            device="cpu" if na_cpu else DEV)
        del hf; gc.collect()
        m = m.to(torch.float32); gc.collect()
        if na_cpu: m = m.to(DEV); gc.collect()
        print(f"  {n_params / 1e9:.1f} B parameters | conversion to float32 {'in RAM (large model)' if na_cpu else 'on the GPU'}")
    else:
        m = HookedTransformer.from_pretrained(name, device=DEV)
    prepare(m)
    a = adapter(m)
    print(f"loaded: {name} | {next(m.parameters()).dtype} | {a['layers']}x{a['heads']} heads"
          f"{' (GQA: ' + str(a['heads_kv']) + ' kv)' if a['gqa'] else ''} | validation: {a['validation']}")
    for av in a["warnings"]: print("  ⚠️", av)
    return m

def prepare(m):
    for p in m.parameters(): p.requires_grad_(False)
    torch.set_grad_enabled(False)
    m.set_use_split_qkv_input(True); m.set_use_hook_mlp_in(True)
    return m

def adapter(m):
    """Reads the config and returns what HeadHunter needs to know about the architecture."""
    c = m.cfg; nkv = getattr(c, "n_key_value_heads", None) or c.n_heads
    name = (c.model_name or "").lower()
    post_norm = bool(getattr(c, "use_normalization_before_and_after", False)) or \
        getattr(c, "original_architecture", "") in ("Olmo2ForCausalLM", "Olmo3ForCausalLM")
    a = dict(name=c.model_name, layers=c.n_layers, heads=c.n_heads, heads_kv=nkv, gqa=nkv < c.n_heads,
             parallel=bool(getattr(c, "parallel_attn_mlp", False)), position=c.positional_embedding_type,
             norma=c.normalization_type, mlp=("gated " if c.gated_mlp else "") + str(c.act_fn), d_model=c.d_model,
             post_norm=post_norm, validation="validated" if ("gpt2" in name or "pythia" in name) else "experimental",
             warnings=[])
    if a["gqa"]:
        a["warnings"].append("GQA: k and v are shared by each group of heads. Nodes and sufficiency: exact. Edges into k/v: approximate (they hold for the group).")
    if post_norm:
        a["warnings"].append("Normalization after attention (Gemma 2/3, OLMo 2/3): the per-head output is NOT additive. "
                           "Node effects are fine; edges and sufficiency are approximate.")
    if a["validation"] != "validated":
        a["warnings"].append("Family not yet validated in HeadHunter: run selftest() first.")
    return a


# =====================================================================================
# 2. Tasks (identical to those used in the tests of 2026-10-31)
# =====================================================================================
def tok(m, texts):
    """BOS only if the model uses it (GPT-2 and Pythia: yes; Qwen: no)."""
    return m.to_tokens(texts, prepend_bos=bool(getattr(m.cfg, "default_prepend_bos", True)))

NAMES = [" Mary", " John", " Tom", " James", " Dan", " Sid", " Martin",
         " Amy", " Kate", " Paul", " Lisa", " Mark", " Anna", " David"]
IOI_MODELS = ["When{A} and{B} went to the store,{B} gave a drink to",
               "When{B} and{A} went to the store,{B} gave a drink to"]

def build_ioi(m, seed, n=40):
    names_ok = [x for x in NAMES if m.to_tokens(x, prepend_bos=False).shape[1] == 1]
    rnd = random.Random(seed); fr, fc, io, s = [], [], [], []
    for _ in range(n):
        A, B = rnd.sample(names_ok, 2)
        f = rnd.choice(IOI_MODELS).format(A=A, B=B)
        fr.append(f); fc.append(f.replace(f",{B} gave", f",{A} gave"))
        io.append(m.to_single_token(A)); s.append(m.to_single_token(B))
    a, b = torch.tensor(io, device=DEV), torch.tensor(s, device=DEV)
    return dict(tk=tok(m, fr), tkc=tok(m, fc), target=a, wrong=b)

COUNTRIES = {"France": "Paris", "Italy": "Rome", "Spain": "Madrid", "Germany": "Berlin", "Japan": "Tokyo",
          "China": "Beijing", "Russia": "Moscow", "Egypt": "Cairo", "Greece": "Athens", "Portugal": "Lisbon",
          "Austria": "Vienna", "Poland": "Warsaw", "Hungary": "Budapest", "Ireland": "Dublin", "Norway": "Oslo",
          "Sweden": "Stockholm", "Finland": "Helsinki", "Denmark": "Copenhagen", "Belgium": "Brussels",
          "Netherlands": "Amsterdam", "Turkey": "Ankara", "Iran": "Tehran", "Iraq": "Baghdad", "Syria": "Damascus",
          "Lebanon": "Beirut", "Pakistan": "Islamabad", "Afghanistan": "Kabul", "Thailand": "Bangkok",
          "Vietnam": "Hanoi", "Indonesia": "Jakarta", "Philippines": "Manila", "Korea": "Seoul", "Kenya": "Nairobi",
          "Nigeria": "Abuja", "Ghana": "Accra", "Morocco": "Rabat", "Cuba": "Havana", "Peru": "Lima",
          "Chile": "Santiago", "Colombia": "Bogota", "Venezuela": "Caracas", "Canada": "Ottawa",
          "Australia": "Canberra", "England": "London", "Scotland": "Edinburgh", "Ukraine": "Kiev",
          "Romania": "Bucharest", "Bulgaria": "Sofia", "Serbia": "Belgrade", "Croatia": "Zagreb",
          "Switzerland": "Bern", "Qatar": "Doha", "Libya": "Tripoli", "Sudan": "Khartoum", "Nepal": "Kathmandu"}
TEMPLATE = "The capital of{P} is the city of"
_known = {}

def known_countries(m):
    key = m.cfg.model_name
    if key in _known: return _known[key]
    is_one_token = lambda x: m.to_tokens(" " + x, prepend_bos=False).shape[1] == 1
    ok = [(p, c) for p, c in COUNTRIES.items() if is_one_token(p) and is_one_token(c)]
    caps = torch.tensor([m.to_single_token(" " + c) for _, c in ok], device=DEV)
    lg = m(tok(m, [TEMPLATE.format(P=" " + p) for p, _ in ok]))[:, -1]
    rank = (lg > lg[torch.arange(len(ok)), caps][:, None]).sum(-1)
    good = [ok[i] for i in range(len(ok)) if rank[i] < 10]
    _known[key] = good; return good

def build_capitals(m, seed, n=40):
    good = known_countries(m); rnd = random.Random(seed)
    fr, fc, cap, capc = [], [], [], []
    for _ in range(n):
        (p, c), (p2, c2) = rnd.sample(good, 2)
        fr.append(TEMPLATE.format(P=" " + p)); fc.append(TEMPLATE.format(P=" " + p2))
        cap.append(m.to_single_token(" " + c)); capc.append(m.to_single_token(" " + c2))
    return dict(tk=tok(m, fr), tkc=tok(m, fc), target=torch.tensor(cap, device=DEV),
                wrong=torch.tensor(capc, device=DEV))

RECIPES = {"IOI": build_ioi, "CAP": build_capitals}

def metric(T, lg):
    """Logit difference (right answer - wrong answer) at the last position, one per sentence. Accepts stacked batches."""
    n = T["target"].shape[0]; k = lg.shape[0] // n
    a, b = T["target"].repeat(k), T["wrong"].repeat(k); ar = torch.arange(lg.shape[0], device=lg.device)
    return (lg[ar, -1, a] - lg[ar, -1, b]).float()

def get_task(m, name, seed, n=40):
    return RECIPES[name](m, seed, n) if isinstance(name, str) else name(m, seed, n)


# =====================================================================================
# 3. Cheap runs: only the last-position logits, optionally starting in the middle of the network
# =====================================================================================
def final_logits(m, x, start_layer=None, hooks=()):
    r = m.run_with_hooks(x, start_at_layer=start_layer, stop_at_layer=m.cfg.n_layers, fwd_hooks=list(hooks))
    r = m.ln_final(r[:, -1:, :]) if m.cfg.normalization_type is not None else r[:, -1:, :]
    return m.unembed(r)

def _ctx(bf16):
    return torch.autocast(DEV, dtype=torch.bfloat16) if bf16 else torch.autocast(DEV, enabled=False)


# =====================================================================================
# 4. Exact effect (patching) of heads and MLPs: slow (reference) and fast versions
# =====================================================================================
def exact_heads_slow(m, T):
    """Old version: one full run per head. Kept as the reference for the regression test."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; f = lambda x: x.endswith("attn.hook_z")
    lg, cl = m.run_with_cache(T["tk"], names_filter=f); ml = metric(T, lg); del lg
    lg, cc = m.run_with_cache(T["tkc"], names_filter=f); mc = metric(T, lg); del lg
    total = (ml - mc).mean().item(); rel = np.zeros(nL * nH)
    for L in range(nL):
        for H in range(nH):
            def hk(z, hook, H=H):
                z[:, :, H, :] = cl[hook.name][:, :, H, :]; return z
            mk = metric(T, m.run_with_hooks(T["tkc"], fwd_hooks=[(f"blocks.{L}.attn.hook_z", hk)]))
            rel[L * nH + H] = (mk - mc).mean().item() / total
    return rel, total

def caches(m, T):
    """One clean run and one corrupted run, keeping what everything else needs."""
    names = lambda x: (x.endswith("attn.hook_z") or x.endswith("hook_resid_pre") or x.endswith("hook_resid_mid")
                       or x.endswith("hook_mlp_out") or x == "hook_embed"
                       or x == f"blocks.{m.cfg.n_layers - 1}.hook_resid_post")
    lg, cl = m.run_with_cache(T["tk"], names_filter=names); ml = metric(T, lg); del lg
    lg, cc = m.run_with_cache(T["tkc"], names_filter=names); mc = metric(T, lg); del lg
    return dict(cl=cl, cc=cc, ml=ml, mc=mc, total=(ml - mc).mean().item())

def exact_heads(m, T, C=None, heads=None, block=None, bf16=False):
    """Fast: starts at the head's layer (reuses the lower ones) and tests several heads per run.
    heads=None tests all of them; otherwise only the given list (indices L*nH+H). Returns a dict index -> relative effect."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; C = C or caches(m, T); n = T["tk"].shape[0]
    target = range(nL * nH) if heads is None else heads
    per_layer = defaultdict(list)
    for c in target: per_layer[int(c) // nH].append(int(c) % nH)
    block = block or EXACT_BLOCK or nH; res = {}
    with _ctx(bf16):
        for L, hs in sorted(per_layer.items()):
            zl = C["cl"][f"blocks.{L}.attn.hook_z"]; rp = C["cc"][f"blocks.{L}.hook_resid_pre"]
            for i in range(0, len(hs), block):
                part = hs[i:i + block]; k = len(part)
                def hk(z, hook, part=part):
                    for j, h in enumerate(part): z[j * n:(j + 1) * n, :, h] = zl[:, :, h]
                    return z
                mk = metric(T, final_logits(m, rp.repeat(k, 1, 1), L, [(f"blocks.{L}.attn.hook_z", hk)]))
                for j, h in enumerate(part):
                    res[L * nH + h] = (mk[j * n:(j + 1) * n] - C["mc"]).mean().item() / C["total"]
    return res

def exact_mlps(m, T, C=None, layers=None, bf16=False):
    nL = m.cfg.n_layers; C = C or caches(m, T); res = {}
    with _ctx(bf16):
        for L in (range(nL) if layers is None else layers):
            mo = C["cl"][f"blocks.{L}.hook_mlp_out"]
            mk = metric(T, final_logits(m, C["cc"][f"blocks.{L}.hook_resid_pre"], L,
                                          [(f"blocks.{L}.hook_mlp_out", lambda x, hook, mo=mo: mo)]))
            res[L] = (mk - C["mc"]).mean().item() / C["total"]
    return res


# =====================================================================================
# 5. Gradient (attribution patching) of heads and MLPs
# =====================================================================================


# =====================================================================================
# 6. Economic method (validated): top 20 heads + check of 21-30; MLPs with an adaptive rule
# =====================================================================================
def economic_core(m, T, C=None, top=20, check=10, threshold=0.05, mlp_threshold=0.03, bf16=False):
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; C = C or caches(m, T)
    gcab, gmlp, nan = node_gradient(m, T, C)
    order = np.argsort(-np.abs(gcab)); exact = 0
    conf = exact_heads(m, T, C, heads=order[top:top + check], bf16=bf16); exact += len(conf)
    head = [int(c) for c in order[:top]] + [c for c, v in conf.items() if abs(v) >= threshold]
    # MLPs: exact on 0..L/4; then 4 direct ones from the gradient, checking downward until 2 consecutive failures
    n_initial = max(1, nL // 4); initial = exact_mlps(m, T, C, range(n_initial), bf16=bf16); exact += n_initial
    mlps = {L for L, v in initial.items() if abs(v) >= mlp_threshold}
    rest = [int(x) for x in np.arange(n_initial, nL)[np.argsort(-np.abs(gmlp[n_initial:]))]]
    mlps |= set(rest[:4]); failures = 0
    for L in rest[4:]:
        v = exact_mlps(m, T, C, [L], bf16=bf16)[L]; exact += 1
        if abs(v) >= mlp_threshold: mlps.add(L); failures = 0
        else:
            failures += 1
            if failures == 2: break
    score = gcab.copy()
    for c, v in conf.items(): score[c] = v                  # where there was an exact value, the exact value wins
    return dict(heads=sorted(head), mlps=sorted(mlps), head_scores=score, grad_mlps=gmlp,
                runs_exact=exact, nan=nan)


# =====================================================================================
# 7. Sufficiency: circuit = set of nodes + all the edges between them (fast O(N) path)
# =====================================================================================
def faithfulness(m, T, C, heads, mlps):
    """Corrupted sentence; each circuit node receives its own corrupted input + the sum of what the previous
    nodes IN THE CIRCUIT changed (clean - corrupted). The final output is assembled the same way."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; parallel = bool(getattr(m.cfg, "parallel_attn_mlp", False))
    nkv = getattr(m.cfg, "n_key_value_heads", None) or nH; group = nH // nkv
    cc, cl = C["cc"], C["cl"]; WO = m.W_O
    per_layer = defaultdict(list)
    for c in heads: per_layer[int(c) // nH].append(int(c) % nH)
    mlps = set(int(x) for x in mlps)
    st = dict(acc=cl["hook_embed"] - cc["hook_embed"], snapshot={})
    hooks = [(f"blocks.{L}.hook_resid_pre", lambda x, hook, L=L: st["snapshot"].__setitem__(L, st["acc"]))
             for L in range(nL)]
    for L, hs in per_layer.items():
        ent = lambda L=L: cc[f"blocks.{L}.hook_resid_pre"] + st["snapshot"][L]
        def hq(x, hook, hs=hs, ent=ent):
            x = x.clone(); e = ent()
            for h in hs: x[:, :, h] = e
            return x
        gs = sorted({h // group for h in hs})
        def hkv(x, hook, gs=gs, ent=ent):
            x = x.clone(); e = ent()
            for g in gs: x[:, :, g] = e
            return x
        def hz(z, hook, L=L, hs=hs):
            zc = cc[f"blocks.{L}.attn.hook_z"]
            st["acc"] = st["acc"] + sum((z[:, :, h] - zc[:, :, h]) @ WO[L, h] for h in hs)
        hooks += [(f"blocks.{L}.hook_q_input", hq), (f"blocks.{L}.hook_k_input", hkv),
                  (f"blocks.{L}.hook_v_input", hkv), (f"blocks.{L}.attn.hook_z", hz)]
    for L in mlps:
        base = f"blocks.{L}.hook_resid_pre" if parallel else f"blocks.{L}.hook_resid_mid"
        hooks += [(f"blocks.{L}.hook_mlp_in",
                   lambda x, hook, L=L, base=base: cc[base] + (st["snapshot"][L] if parallel else st["acc"])),
                  (f"blocks.{L}.hook_mlp_out",
                   lambda x, hook, L=L: st.__setitem__("acc", st["acc"] + x - cc[f"blocks.{L}.hook_mlp_out"]))]
    out_hook = f"blocks.{nL - 1}.hook_resid_post"
    hooks.append((out_hook, lambda x, hook: cc[out_hook] + st["acc"]))
    mk = metric(T, final_logits(m, T["tkc"], None, hooks))
    return (mk - C["mc"]).sum().item() / (C["ml"] - C["mc"]).sum().item()

def faithfulness_batches(m, batches, Cs, heads, mlps):
    num = den = 0.0
    for T, C in zip(batches, Cs):
        f = faithfulness(m, T, C, heads, mlps); d = (C["ml"] - C["mc"]).sum().item()
        num += f * d; den += d
    return num / den

def circuit_for_target(m, batches, Cs, head_order, mlps, target=0.90, maximum=None):
    """Binary search: smallest number of heads (in the given order) that, together with the MLPs, reaches the target faithfulness."""
    order = [int(c) for c in head_order][:maximum]; eval = {}
    def f(k):
        if k not in eval: eval[k] = faithfulness_batches(m, batches, Cs, order[:k], mlps)
        return eval[k]
    lo, hi = 0, len(order)
    if f(hi) < target: return dict(k=hi, faithfulness=f(hi), reached=False, evals=len(eval), curve=eval)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid) >= target: hi = mid
        else: lo = mid
    return dict(k=hi, faithfulness=f(hi), reached=True, evals=len(eval), curve=dict(sorted(eval.items())))


# =====================================================================================
# 8. Self-test (run when loading a new model)
# =====================================================================================
def selftest(m, task_name="IOI", seed=0, n=10, verbose=True):
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; T = get_task(m, task_name, seed, n); C = caches(m, T); ok = True
    msgs = []
    f_all = faithfulness(m, T, C, range(nL * nH), range(nL))
    ok1 = abs(f_all - 1) < 1e-3; ok &= ok1
    msgs.append(f"{'✅' if ok1 else '❌'} circuit = whole model: {f_all:.4f} (must be 1)")
    f_empty = faithfulness(m, T, C, [], [])
    msgs.append(f"   empty circuit: {f_empty:.3f} (should be near 0, only the direct path from the embedding)")
    gcab, gmlp, nan = node_gradient(m, T, C); ok2 = nan == 0; ok &= ok2
    msgs.append(f"{'✅' if ok2 else '❌'} gradient without NaN ({nan} NaN)")
    sample = list(np.argsort(-np.abs(gcab))[:8]) + random.Random(0).sample(range(nL * nH), min(8, nL * nH))
    ex = exact_heads(m, T, C, heads=sorted(set(int(a) for a in sample)))
    from scipy.stats import spearmanr
    ks = sorted(ex); rho = spearmanr([gcab[k] for k in ks], [ex[k] for k in ks])[0]
    ok3 = rho > 0.5; ok &= ok3
    msgs.append(f"{'✅' if ok3 else '⚠️'} gradient x exact on {len(ks)} heads: Spearman {rho:.2f} (expected > 0.5)")
    slow = exact_heads_slow(m, get_task(m, task_name, seed, 4))[0] if nL * nH <= 64 else None
    if slow is not None:
        T4 = get_task(m, task_name, seed, 4); rap = exact_heads(m, T4)
        dif = max(abs(slow[k] - rap[k]) for k in rap); ok4 = dif < 1e-3; ok &= ok4
        msgs.append(f"{'✅' if ok4 else '❌'} fast exact = old exact (max difference {dif:.1e})")
    if verbose: print(f"selftest {m.cfg.model_name} / {task_name}:"); [print("  " + x) for x in msgs]
    return ok


# =====================================================================================
# 9. Regression: reference numbers measured on 2026-10-31 (RunPod A6000)
# =====================================================================================
REFERENCE = {
    ("gpt2-small", "IOI"):  dict(core=15, mlps=[0, 5, 8, 9],   faith_core=0.45, faith_1pct=0.96),
    ("gpt2-small", "CAP"):  dict(core=5,  mlps=[0, 9, 10, 11], faith_core=0.45, faith_1pct=0.95),
    ("pythia-160m", "IOI"): dict(core=12, mlps=[0, 2, 5, 9],   faith_core=0.65, faith_1pct=0.96),
    ("pythia-160m", "CAP"): dict(core=6,  mlps=[0, 9, 10, 11], faith_core=0.60, faith_1pct=0.91),
}

def regression(m, name, task_name, seeds=(0, 1, 2), n=40, n_batch=10, bf16=False, verbose=True):
    """Redoes the exact core (3 seeds) and the sufficiency with the NEW code and compares with the reference."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; t0 = time.time(); rels, exs = [], []
    for sd in seeds:
        T = get_task(m, task_name, sd, n); C = caches(m, T)
        r = exact_heads(m, T, C, bf16=bf16); rels.append([r[i] for i in range(nL * nH)])
        e = exact_mlps(m, T, C, bf16=bf16); exs.append([e[i] for i in range(nL)])
        del C
    rel = np.abs(np.mean(rels, 0)); ex = np.abs(np.mean(exs, 0))
    batches = [get_task(m, task_name, 7000 + b, n_batch) for b in range(n // n_batch)]; Cs = [caches(m, T) for T in batches]
    core = [int(i) for i in np.where(rel >= 0.05)[0]]; mlp5 = [int(i) for i in np.where(ex >= 0.05)[0]]
    f_core = faithfulness_batches(m, batches, Cs, core, mlp5)
    f_1 = faithfulness_batches(m, batches, Cs, [int(i) for i in np.where(rel >= 0.01)[0]], range(nL))
    ref = REFERENCE.get((name, task_name))
    out = dict(model=name, get_task=task_name, core=len(core), mlps=mlp5, faith_core=round(f_core, 2),
               faith_1pct=round(f_1, 2), seconds=round(time.time() - t0, 1))
    if ref:
        out["bate"] = (out["core"] == ref["core"] and out["mlps"] == ref["mlps"]
                       and abs(out["faith_core"] - ref["faith_core"]) <= 0.01 and abs(out["faith_1pct"] - ref["faith_1pct"]) <= 0.01)
        out["reference"] = ref
    if verbose: print(out)
    del Cs; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    return out


# =====================================================================================
# 10. Save with metadata + backup
# =====================================================================================
def save(file, data, **params):
    goal = dict(version=VERSION, data=time.strftime("%Y-%m-%d %H:%M:%S"), params=params,
                gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
                torch=torch.__version__, python=platform.python_version())
    json.dump(dict(goal=goal, data=data), open(file, "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

def backup_zip(name="backup_hh.zip", default="*.json"):
    import zipfile, glob
    with zipfile.ZipFile(name, "w") as z:
        for a in glob.glob(default): z.write(a)
    print(f"{name}: {len(glob.glob(default))} files"); return name
# =====================================================================================
# 11. Head order for the circuit (gradient over 120 sentences, validated on 2026-11-01)
# =====================================================================================
def head_order(m, task_name, seeds=(0, 1, 2), n=40):
    """Sorts the heads by the mean gradient over several seeds (3 x 40 = 120 sentences).
    With a large tail (GPT-2 large IOI) this reduced the circuit from 248 to 147 heads, in ~2.5 s."""
    scores = []
    for sd in seeds:
        T = get_task(m, task_name, sd, n); C = caches(m, T)
        scores.append(node_gradient(m, T, C)[0]); del C
    score = np.mean(scores, 0)
    return [int(c) for c in np.argsort(-np.abs(score))], score


# =====================================================================================
# 12. Edges: circuit defined by connections, gradient scores and pruning (hybrid method)
# =====================================================================================
E_, O_ = ("E", 0), ("O", 0)

def _pos(node, nH, is_parallel):
    t, i = node
    if t == "E": return -1
    if t == "O": return 10**6
    if t == "H": return 2 * (i // nH)
    return 2 * i if is_parallel else 2 * i + 1

def edges_between(m, heads, mlps):
    """All valid edges between the embedding, the given nodes and the output."""
    nH = m.cfg.n_heads; is_parallel = bool(getattr(m.cfg, "parallel_attn_mlp", False))
    nodes = [("H", int(c)) for c in heads] + [("M", int(L)) for L in mlps]
    return [(u, v) for u in [E_] + nodes for v in nodes + [O_] if _pos(u, nH, is_parallel) < _pos(v, nH, is_parallel)]

def _group(m):
    nH = m.cfg.n_heads; nkv = getattr(m.cfg, "n_key_value_heads", None) or nH
    return nH // nkv

def _output(m, cache, node):
    nH = m.cfg.n_heads; t, i = node
    if t == "E": return cache["hook_embed"]
    if t == "H": return cache[f"blocks.{i // nH}.attn.hook_z"][:, :, i % nH] @ m.W_O[i // nH, i % nH]
    return cache[f"blocks.{i}.hook_mlp_out"]

# Edges (v0.11): edge = (u, v, p), p = "q"/"k"/"v" if v is a head, "" if v is an MLP or the output.
# An edge is now (u, v, p): p = "q", "k" or "v" when v is a head; p = "" when v is an MLP or the output.

def edge_parts(m, heads, mlps):
    out = []
    for u, v in edges_between(m, heads, mlps):
        if v[0] == "H": out += [(u, v, p) for p in "qkv"]
        else: out.append((u, v, ""))
    return out

# v0.11.3: k/v score PER HEAD under GQA, without ungrouping the input.
# Before: the gradient of the group's shared input (sum over the heads) was divided by `gr` -> the `gr` heads of the group
# got the SAME score (the mean of the official EAP-IG scores). Now: a per-head zero offset is added to k (and v)
# AFTER the group repetition; its gradient is the per-head gradient (d_head dimensions per position). It is carried back
# to the group input with the group's W_K/W_V (and, for k, undoing the RoPE rotation), which reproduces the official EAP-IG.
from transformer_lens.components import AbstractAttention as _Attention, GroupedQueryAttention as _GQA
_CAP = {}   # layer -> {"k": offset, "v": offset}; empty outside the score computation (normal behaviour)
_orig_scores = getattr(_GQA.calculate_attention_scores, "_hh_orig", _GQA.calculate_attention_scores)   # safe against reload
_orig_z = getattr(_GQA.calculate_z_scores, "_hh_orig", _GQA.calculate_z_scores)

def _layer_of(att): return int(att.hook_k.name.split(".")[1])

def _scores_cap(self, q, k):
    cap = _CAP.get(_layer_of(self)) if _CAP else None
    if cap is None or self.cfg.ungroup_grouped_query_attention: return _orig_scores(self, q, k)
    k = torch.repeat_interleave(k, dim=2, repeats=self.repeat_kv_heads)
    cap["k"] = torch.zeros_like(k).requires_grad_(True)
    return _Attention.calculate_attention_scores(self, q, k + cap["k"])

def _z_cap(self, v, pattern):
    cap = _CAP.get(_layer_of(self)) if _CAP else None
    if cap is None or self.cfg.ungroup_grouped_query_attention: return _orig_z(self, v, pattern)
    v = torch.repeat_interleave(v, dim=2, repeats=self.repeat_kv_heads)
    cap["v"] = torch.zeros_like(v).requires_grad_(True)
    return _Attention.calculate_z_scores(self, v + cap["v"], pattern)

_scores_cap._hh_orig, _z_cap._hh_orig = _orig_scores, _orig_z
_GQA.calculate_attention_scores, _GQA.calculate_z_scores = _scores_cap, _z_cap

def _rotary_T(att, g):
    # R^T g (by autograd): carries the gradient of k AFTER RoPE back to BEFORE RoPE (valid for any RoPE variant).
    with torch.enable_grad():
        z = torch.zeros_like(g).requires_grad_(True)
        return torch.autograd.grad(att.apply_rotary(z, 0, None), z, g)[0]

def _ln_vjp(block, x, u):
    # Carries the gradient u (B,P,n,d_model) of the PROJECTION INPUT back to before the block's ln1 normalization, which in
    # TransformerLens comes AFTER the hook_k_input / hook_v_input hooks. x = resid_pre (B,P,d_model). By autograd, valid for LN and RMS.
    with torch.enable_grad():
        xh = x.detach().unsqueeze(2).expand(-1, -1, u.shape[2], -1).clone().requires_grad_(True)
        return torch.autograd.grad(block.ln1(xh), xh, u)[0]

def _per_head(m):
    # True if the per-head k/v score can be computed: GQA (gr > 1), no QK-norm (a norm between projection and repetition would break the math)
    # and with normalization BEFORE attention (Gemma 2/3 and OLMo 2/3 normalize after: they keep the group mean, as in v0.11.2).
    return (_group(m) > 1 and not getattr(m.cfg, "use_qk_norm", False) and not m.cfg.ungroup_grouped_query_attention
            and not getattr(m.cfg, "use_normalization_before_and_after", False)
            and getattr(m.cfg, "original_architecture", "") not in ("Gemma2ForCausalLM", "Gemma3ForCausalLM", "Olmo2ForCausalLM", "Olmo3ForCausalLM"))

def input_gradients(m, T, C, targets, alphas=(0.0,), batch=None):
    # Gradient of the metric at the INPUT of each (node, part), with the embedding interpolated corrupted -> clean
    # (alphas=(0,) = plain gradient at the corrupted input, as in EAP; several alphas = mean = EAP-IG).
    nH, nL = m.cfg.n_heads, m.cfg.n_layers; gr = _group(m); n = T["tk"].shape[0]
    capt = _per_head(m)   # v0.11.3: per-head k/v score under GQA (without it: the group mean, as in v0.11.2)
    per_layer = defaultdict(set); G = {}
    for (t, i), p in targets:
        if t == "H": per_layer[i // nH].add(i % nH)
    try:
        batch = batch or GRAD_BATCH
        for a in range(0, n, batch):
            b = min(n, a + batch); Tf = _slice(T, a, b)
            e_c = C["cc"]["hook_embed"][a:b]; e_l = C["cl"]["hook_embed"][a:b]; Gf = {}
            for al in alphas:
                torch.set_grad_enabled(True); m.reset_hooks()
                m.add_hook("hook_embed", lambda x, hook, al=al: (e_c + al * (e_l - e_c)).clone().requires_grad_(True))
                _CAP.clear()
                if capt:
                    _CAP.update({L: {} for L in per_layer})
                    for L in per_layer:
                        m.add_hook(f"blocks.{L}.hook_resid_pre", lambda x, hook, L=L: _CAP[L].__setitem__("x", x.detach()))
                for L, hs in per_layer.items():
                    for q in ("q" if capt else "qkv"):
                        def bk(g, hook, L=L, hs=hs, q=q):
                            for h in hs:
                                k = (("H", L * nH + h), q)
                                part = g[:, :, h] if q == "q" else g[:, :, h // gr] / gr
                                Gf[k] = Gf.get(k, 0) + part.detach() / len(alphas)
                        m.add_hook(f"blocks.{L}.hook_{q}_input", bk, "bwd")
                for (t, i), p in targets:
                    if t == "M":
                        m.add_hook(f"blocks.{i}.hook_mlp_in", lambda g, hook, i=i: Gf.__setitem__((("M", i), ""), Gf.get((("M", i), ""), 0) + g.detach() / len(alphas)), "bwd")
                m.add_hook(f"blocks.{nL - 1}.hook_resid_post", lambda g, hook: Gf.__setitem__((O_, ""), Gf.get((O_, ""), 0) + g.detach() / len(alphas)), "bwd")
                metric(Tf, _last_logits(m, Tf["tkc"])).sum().backward()
                if capt:
                    for L, hs in per_layer.items():
                        att = m.blocks[L].attn; hl = sorted(hs); idx = [h // gr for h in hl]
                        for q in "kv":
                            gq = _CAP[L][q].grad
                            if q == "k" and m.cfg.positional_embedding_type == "rotary": gq = _rotary_T(att, gq)
                            W = (att._W_K if q == "k" else att._W_V)[idx]
                            out = torch.einsum("bphd,hmd->bphm", gq[:, :, hl], W)
                            out = _ln_vjp(m.blocks[L], _CAP[L]["x"], out).detach() / len(alphas)
                            for j, h in enumerate(hl):
                                kk = (("H", L * nH + h), q); Gf[kk] = Gf.get(kk, 0) + out[:, :, j]
                            del gq, out
                    _CAP.clear()
                m.reset_hooks(); torch.set_grad_enabled(False)
            for k, v in Gf.items(): G.setdefault(k, []).append(v)
    finally:
        _CAP.clear(); m.reset_hooks(); torch.set_grad_enabled(False)
    return {k: torch.cat(v, 0) for k, v in G.items()}

COMPUTE_BATCH = 40   # v0.11.4: max. sentences per compute chunk of the edge scores (0 or None = no merging). Only joins batches of the SAME length.

def _merge_batches(batches, Cs, target):
    # Joins (torch.cat) batches of the same length into chunks of at most `target` sentences (always at least one batch). The sentences and the draw are the same; only the chunk size changes
    # (the math is the same: Spearman 1.0000 against the separate batches). Fewer small backward calls = much faster on a GPU.
    per = defaultdict(list)
    for T, C in zip(batches, Cs): per[T["tk"].shape[1]].append((T, C))
    def join(g):
        if len(g) == 1: return g[0]
        n0 = g[0][0]["target"].shape[0]
        T = {k: (torch.cat([x[0][k] for x in g], 0) if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == n0 else v) for k, v in g[0][0].items()}
        cat = lambda d: {k: torch.cat([x[1][d][k] for x in g], 0) for k in g[0][1][d]}
        ml, mc = torch.cat([x[1]["ml"] for x in g]), torch.cat([x[1]["mc"] for x in g])
        return T, dict(cl=cat("cl"), cc=cat("cc"), ml=ml, mc=mc, total=(ml - mc).mean().item())
    L2, C2 = [], []
    for items in per.values():
        cur, n = [], 0
        for it in items:
            sz = it[0]["tk"].shape[0]
            if cur and n + sz > target: T, C = join(cur); L2.append(T); C2.append(C); cur, n = [], 0   # v0.11.5: never exceeds the target
            cur.append(it); n += sz
        if cur: T, C = join(cur); L2.append(T); C2.append(C)
    return L2, C2

def _compute_size(m, T, n_src, n_tgt):
    # Largest chunk (in sentences) that fits: the scores keep (sources + targets) x positions x d_model floats per sentence; uses up to 35% of the free GPU memory.
    n0 = T["tk"].shape[0]
    if not COMPUTE_BATCH: return n0
    if not torch.cuda.is_available(): return max(n0, COMPUTE_BATCH)
    per_sentence = (n_src + n_tgt) * T["tk"].shape[1] * m.cfg.d_model * 4
    return max(n0, min(COMPUTE_BATCH, int(0.35 * torch.cuda.mem_get_info()[0] // max(per_sentence, 1))))

def edge_scores(m, batches, Cs, edges3, ig_steps=0):
    # Score of each (u, v, p) = delta output(u) . gradient at the input of (v, p). ig_steps > 0 = EAP-IG.
    # v0.11.2: IG_SCHEME = "mid" (default: midpoints 0.1...0.9 = accurate integral, same as 20 points)
    # or "eapig" (0; 0.2; ...; 0.8 = reproduces the official EAP-IG exactly).
    # v0.11.4: batches of the same length are merged into chunks of up to COMPUTE_BATCH sentences (and the backward runs on the whole chunk); if memory runs out, the chunk is halved.
    if not ig_steps: alphas = (0.0,)
    elif IG_SCHEME == "eapig": alphas = tuple(k / ig_steps for k in range(ig_steps))
    else: alphas = tuple((k + 0.5) / ig_steps for k in range(ig_steps))
    srcs = sorted({a[0] for a in edges3}); tgts = sorted({(a[1], a[2]) for a in edges3})
    si = {u: k for k, u in enumerate(srcs)}; ti = {v: k for k, v in enumerate(tgts)}
    den = sum((C["ml"] - C["mc"]).sum().item() for C in Cs); n0 = batches[0]["tk"].shape[0]
    target = _compute_size(m, batches[0], len(srcs), len(tgts))
    while True:
        S = torch.zeros(len(srcs), len(tgts)); oom = False
        try:
            lt, ct = _merge_batches(batches, Cs, target) if target > n0 else (batches, Cs)
            for T, C in zip(lt, ct):
                G = input_gradients(m, T, C, set(tgts), alphas, batch=max(GRAD_BATCH, T["tk"].shape[0]) if target > n0 else None)
                Dm = torch.stack([(_output(m, C["cl"], u) - _output(m, C["cc"], u)).flatten() for u in srcs])
                for j in range(0, len(tgts), 128):   # in blocks of 128 targets: lower peak memory, same math
                    Gm = torch.stack([G.pop(v).flatten() for v in tgts[j:j + 128]])
                    S[:, j:j + 128] += (Dm @ Gm.T).cpu(); del Gm
                del Dm, G
            break
        except torch.cuda.OutOfMemoryError:
            oom = True
        if oom:   # v0.11.5: cleanup stays OUTSIDE the except (inside it the traceback still holds the memory of the chunk that failed)
            lt = ct = T = C = G = Dm = Gm = None
            m.reset_hooks(); torch.set_grad_enabled(False); gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if target <= n0: raise torch.cuda.OutOfMemoryError("out of memory even with the smallest chunk (one batch)")
            target = max(n0, target // 2)
    score = {a: S[si[a[0]], ti[(a[1], a[2])]].item() / den for a in edges3}
    for a in score:
        if a[0] == E_ and a[1] == ("M", 0): score[a] = float("inf")
    return score

def edge_faithfulness(m, T, C, edges3):
    # Input of each (node, part) = corrupted input + A @ D (vectorized): q, k and v receive only THEIR edges.
    nH, nL = m.cfg.n_heads, m.cfg.n_layers; gr = _group(m); is_parallel = bool(getattr(m.cfg, "parallel_attn_mlp", False))
    sources = sorted({u for u, _, _ in edges3}, key=lambda u: (_pos(u, nH, is_parallel), u))
    row = {u: i for i, u in enumerate(sources)}; nS = len(sources)
    ups = defaultdict(list)
    for u, v, p in edges3: ups[(v, p)].append(row[u])
    nodes = {x for a in edges3 for x in a[:2] if x[0] in "HM"}
    ref = C["cc"]["hook_embed"]; B, P, dm = ref.shape
    D = torch.zeros(nS, B, P, dm, device=ref.device, dtype=ref.dtype)
    if E_ in row: D[row[E_]] = C["cl"]["hook_embed"] - C["cc"]["hook_embed"]
    def matrix(lsts):
        A = torch.zeros(len(lsts), max(nS, 1), device=ref.device, dtype=ref.dtype)
        for r, ls in enumerate(lsts):
            if ls: A[r, ls] = 1.0
        return A
    def summed(A):
        used = torch.nonzero(A.sum(0)).flatten()
        if len(used) == 0: return torch.zeros(A.shape[0], B, P, dm, device=ref.device, dtype=ref.dtype)
        n = int(used[-1]) + 1
        return (A[:, :n] @ D[:n].reshape(n, -1)).reshape(-1, B, P, dm)
    per_layer = defaultdict(list)
    for t, i in nodes:
        if t == "H": per_layer[i // nH].append(i % nH)
    hooks = []
    for L, hs in per_layer.items():
        hs = sorted(hs); hs_t = torch.tensor(hs, device=ref.device)
        Aq = matrix([ups.get((("H", L * nH + h), "q"), []) for h in hs])
        groups = sorted({h // gr for h in hs}); g_t = torch.tensor(groups, device=ref.device)
        Ak = matrix([sorted({s for h in hs if h // gr == g for s in ups.get((("H", L * nH + h), "k"), [])}) for g in groups])
        Av = matrix([sorted({s for h in hs if h // gr == g for s in ups.get((("H", L * nH + h), "v"), [])}) for g in groups])
        lin_h = [row.get(("H", L * nH + h)) for h in hs]
        def hq(x, hook, L=L, hs_t=hs_t, A=Aq):
            x = x.clone(); x[:, :, hs_t] = C["cc"][f"blocks.{L}.hook_resid_pre"][:, :, None, :] + summed(A).permute(1, 2, 0, 3); return x
        def hk(x, hook, L=L, g_t=g_t, A=Ak):
            x = x.clone(); x[:, :, g_t] = C["cc"][f"blocks.{L}.hook_resid_pre"][:, :, None, :] + summed(A).permute(1, 2, 0, 3); return x
        def hv(x, hook, L=L, g_t=g_t, A=Av):
            x = x.clone(); x[:, :, g_t] = C["cc"][f"blocks.{L}.hook_resid_pre"][:, :, None, :] + summed(A).permute(1, 2, 0, 3); return x
        def hz(z, hook, L=L, hs_t=hs_t, lin_h=lin_h):
            if all(l is None for l in lin_h): return
            W = m.W_O[L, hs_t]
            new = torch.einsum("bpkh,khd->kbpd", z[:, :, hs_t], W)
            old = torch.einsum("bpkh,khd->kbpd", C["cc"][f"blocks.{L}.attn.hook_z"][:, :, hs_t], W)
            for j, l in enumerate(lin_h):
                if l is not None: D[l] = new[j] - old[j]
        hooks += [(f"blocks.{L}.hook_q_input", hq), (f"blocks.{L}.hook_k_input", hk),
                  (f"blocks.{L}.hook_v_input", hv), (f"blocks.{L}.attn.hook_z", hz)]
    for t, i in nodes:
        if t == "M":
            Am = matrix([ups.get((("M", i), ""), [])]); lm = row.get(("M", i))
            bn = f"blocks.{i}.hook_resid_pre" if is_parallel else f"blocks.{i}.hook_resid_mid"
            hooks.append((f"blocks.{i}.hook_mlp_in", lambda x, hook, Am=Am, bn=bn: C["cc"][bn] + summed(Am)[0]))
            if lm is not None:
                def ho(x, hook, i=i, lm=lm): D[lm] = x - C["cc"][f"blocks.{i}.hook_mlp_out"]
                hooks.append((f"blocks.{i}.hook_mlp_out", ho))
    Ao = matrix([ups.get((O_, ""), [])])
    hooks.append((f"blocks.{nL - 1}.hook_resid_post", lambda x, hook: C["cc"][f"blocks.{nL - 1}.hook_resid_post"] + summed(Ao)[0]))
    mk = metric(T, final_logits(m, T["tkc"], None, hooks)); del D
    return (mk - C["mc"]).sum().item() / (C["ml"] - C["mc"]).sum().item()

def edge_faithfulness_batches(m, batches, Cs, edges3):
    num = den = 0.0
    for T, C in zip(batches, Cs):
        d = (C["ml"] - C["mc"]).sum().item(); num += edge_faithfulness(m, T, C, edges3) * d; den += d
    return num / den

def exact_edge(m, T, C, u, v, p):
    # Passes (clean - corrupted) of u ONLY to the input (v, p). Returns the sum over the sentences.
    nH, nL = m.cfg.n_heads, m.cfg.n_layers; gr = _group(m); d = _output(m, C["cl"], u) - _output(m, C["cc"], u); t, i = v
    if t == "O":
        r = (C["cc"][f"blocks.{nL - 1}.hook_resid_post"] + d)[:, -1:, :]
        mk = metric(T, m.unembed(m.ln_final(r) if m.cfg.normalization_type is not None else r))
    elif t == "H":
        L, h = i // nH, i % nH; idx = h if p == "q" else h // gr
        def hk(x, hook):
            x = x.clone(); x[:, :, idx] = x[:, :, idx] + d; return x
        mk = metric(T, final_logits(m, C["cc"][f"blocks.{L}.hook_resid_pre"], L, [(f"blocks.{L}.hook_{p}_input", hk)]))
    else:
        mk = metric(T, final_logits(m, C["cc"][f"blocks.{i}.hook_resid_pre"], i, [(f"blocks.{i}.hook_mlp_in", lambda x, hook: x + d)]))
    return (mk - C["mc"]).sum().item()

def edge_search(m, batches, Cs, order, target):
    eval = {}
    def f(k):
        if k not in eval: eval[k] = edge_faithfulness_batches(m, batches, Cs, order[:k])
        return eval[k]
    lo, hi = 0, len(order)
    if f(hi) < target: return hi, f(hi), False
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid) >= target: hi = mid
        else: lo = mid
    return hi, f(hi), True

def prune_edges(m, batches_score, Cs_score, batches_search, Cs_search, heads, mlps, target=0.85, ig_steps=5):
    # Edge stage (validated on 2026-11-03, beat the official EAP-IG): edges between the circuit heads
    # and the MLPs, with separate q/k/v; EAP-IG score; smallest number of edges (in score order) that reaches `target`.
    ar = edge_parts(m, heads, mlps); t0 = time.time()
    score = edge_scores(m, batches_score, Cs_score, ar, ig_steps)
    order = sorted(ar, key=lambda a: -abs(score[a])); k, f, ok = edge_search(m, batches_search, Cs_search, order, target)
    return dict(k=k, total=len(ar), faithfulness=f, reached=ok, edge_list=order[:k], elapsed=round(time.time() - t0, 1))


EXACT_BLOCK = None   # heads tested together in the exact pass (None = all in the layer; smaller = less memory)
GRAD_BATCH = 10   # sentences per backward pass (smaller = less memory, same result)

def _slice(T, a, b):
    n = T["target"].shape[0]
    return {k: (v[a:b] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == n else v) for k, v in T.items()}

def _last_logits(m, tk):
    """Only the last position goes through the unembed (the Qwen vocabulary has 152 thousand words)."""
    r = m(tk, stop_at_layer=m.cfg.n_layers)[:, -1:, :]
    return m.unembed(m.ln_final(r) if m.cfg.normalization_type is not None else r)

def node_gradient(m, T, C=None):
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; C = C or caches(m, T); n = T["tk"].shape[0]
    f = lambda x: x.endswith("attn.hook_z") or x.endswith("hook_mlp_out")
    head = np.zeros(nL * nH); mlp = np.zeros(nL)
    m.set_use_split_qkv_input(False)
    try:
        for a in range(0, n, GRAD_BATCH):
            b = min(n, a + GRAD_BATCH); Tf = _slice(T, a, b); gr = {}
            torch.set_grad_enabled(True); m.reset_hooks()
            m.add_hook("hook_embed", lambda x, hook: x.requires_grad_(True))
            m.add_hook(f, lambda g, hook: gr.__setitem__(hook.name, g.detach()), "bwd")
            metric(Tf, _last_logits(m, Tf["tkc"])).sum().backward()
            m.reset_hooks(); torch.set_grad_enabled(False)
            for L in range(nL):
                z = f"blocks.{L}.attn.hook_z"; o = f"blocks.{L}.hook_mlp_out"
                head[L * nH:(L + 1) * nH] += (gr[z] * (C["cl"][z][a:b] - C["cc"][z][a:b])).sum((0, 1, 3)).cpu().numpy()
                mlp[L] += (gr[o] * (C["cl"][o][a:b] - C["cc"][o][a:b])).sum().item()
            del gr
    finally:
        m.reset_hooks(); torch.set_grad_enabled(False); m.set_use_split_qkv_input(True)
    nan = int(np.isnan(head).sum() + np.isnan(mlp).sum())
    return head / (n * C["total"]), mlp / (n * C["total"]), nan

# =====================================================================================
# 13. Full analysis (HeadHunter end to end)
# =====================================================================================
def promote_steps(m, batches, Cs, order, b, runs=10, min_jump=0.05, target=0.90):
    """Step rule (validated on 2026-11-01): if the last head of the circuit makes the faithfulness jump
    >= `min_jump`, it is an interaction head that the gradient underestimated. It moves to the front of the order and the
    search is redone only up to the previous k (the new k is never larger). Stops when there is no jump or k does not shrink."""
    nL = m.cfg.n_layers; order = list(order); promoted = []; hist = []
    for _ in range(runs):
        k = b["k"]
        if not b["reached"] or k < 2: break
        h = order[k - 1]
        before = b["curve"].get(k - 1)
        if before is None: before = faithfulness_batches(m, batches, Cs, order[:k - 1], range(nL))
        jump = b["faithfulness"] - before
        if jump < min_jump or h in promoted: break
        new = promoted + [h] + [c for c in order if c not in promoted and c != h]
        bn = circuit_for_target(m, batches, Cs, new, range(nL), target=target, maximum=k)
        if not bn["reached"] or bn["k"] >= k: break
        promoted.append(h); order = new; b = bn
        hist.append(dict(head=int(h), jump=round(float(jump), 3), k=int(b["k"])))
    return order, b, promoted, hist

def analyze(m, task_name, target=0.90, prune_target=0.85, prune=True, n=40, n_batch=10, order_seeds=(0, 1, 2),
             step=True, min_gain=0.10, order="ig", ig_steps=5, verbose=True):
    """1) economic core (heads + MLPs); 2) head order by the gradient with 120 sentences;
    3) sufficient circuit (heads + all MLPs) by binary search up to `target`;
    3b) [v0.3, step=True] promotes the interaction heads (steps) and picks the final size on new sentences;
    4) pruning of the circuit edges up to `prune_target` (hybrid method). step=False = v0.2 behaviour."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; t0 = time.time(); r = {}
    T0 = get_task(m, task_name, 0, n); C0 = caches(m, T0)
    core = economic_core(m, T0, C0); del C0
    r["core"] = dict(heads=[c for c in core["heads"] if abs(core["head_scores"][c]) >= 0.05], mlps=core["mlps"])
    t1 = time.time()
    if order == "ig": order, score = head_order_ig(m, task_name, order_seeds, n, n_steps=ig_steps)   # v0.5: default
    else: order, score = head_order(m, task_name, order_seeds, n)
    bat_a = [get_task(m, task_name, 7000 + b, n_batch) for b in range(n // n_batch)]; Ca = [caches(m, T) for T in bat_a]
    b = circuit_for_target(m, bat_a, Ca, order, range(nL), target=target)
    k_before = b["k"]; t2 = time.time(); promoted, hist, reverted, k_prom = [], [], False, None
    if step:
        order_n, b_n = list(order), b
        order, b, promoted, hist = promote_steps(m, bat_a, Ca, order, b, target=target)
        if promoted:   # final size chosen on sentences that the promotion never saw
            bat_v = [get_task(m, task_name, 9000 + b_, n_batch) for b_ in range(n // n_batch)]; Cv = [caches(m, T) for T in bat_v]
            b_v = circuit_for_target(m, bat_v, Cv, order, range(nL), target=target); del Cv
            if b_v["k"] > b["k"]: b = b_v   # v0.3.3: final size = the LARGER of the search sentences and the new sentences
            k_prom = b["k"]
            if (k_before - b["k"]) / k_before < min_gain:   # v0.3.1: small gain -> back to the normal circuit
                order, b, reverted = order_n, b_n, True
    t3 = time.time()
    r["circuit"] = dict(heads=order[:b["k"]], mlps=list(range(nL)), faithfulness=b["faithfulness"], reached=b["reached"],
                         edges=len(edges_between(m, order[:b["k"]], range(nL))), k_without_step=k_before,
                         promoted=[int(c) for c in promoted], steps=hist, reverted=reverted, k_promoted=k_prom)
    if prune:
        bat_p = [get_task(m, task_name, 5000 + b_, n_batch) for b_ in range(n // n_batch)]; Cp = [caches(m, T) for T in bat_p]
        p = prune_edges(m, bat_p, Cp, bat_a, Ca, r["circuit"]["heads"], range(nL), prune_target)
        r["edges"] = {k: p[k] for k in ("k", "total", "faithfulness", "reached", "edge_list")}
        del Cp
    r["elapsed"] = dict(core=round(t1 - t0, 1), circuit=round(t2 - t1, 1), step=round(t3 - t2, 1),
                      edges=round(time.time() - t3, 1), total=round(time.time() - t0, 1))
    if verbose:
        head_label = lambda c: f"{c // nH}.{c % nH}"
        print(f"=== HeadHunter v{VERSION}: {m.cfg.model_name} / {task_name} ===")
        print(f"core (>=5%): {len(r['core']['heads'])} heads {[head_label(c) for c in r['core']['heads'][:12]]}"
              f"{'...' if len(r['core']['heads']) > 12 else ''} | MLPs {r['core']['mlps']}")
        c = r["circuit"]
        if step and promoted:
            print(f"steps: {len(promoted)} interaction head(s) promoted {[head_label(x) for x in promoted]} "
                  f"| k {k_before} -> {hist[-1]['k']} (search sentences) -> {k_prom} (final: the larger of search and new sentences)")
            if reverted:
                print(f"   gain {(k_before - k_prom) / k_before:.0%} < {min_gain:.0%}: back to the normal circuit ({k_before} heads) "
                      f"and only note the interaction heads")
        elif step:
            print("degraus: nenhum")
        print(f"circuit for {target:.0%}: {len(c['heads'])}/{nL * nH} heads + {nL} MLPs -> faithfulness {c['faithfulness']:.0%}"
              f"{'' if c['reached'] else '  (did NOT reach the target)'}")
        if prune:
            a = r["edges"]
            print(f"edges for {prune_target:.0%}: {a['k']} of {a['total']} ({a['k'] / a['total']:.0%}) -> faithfulness {a['faithfulness']:.0%}")
        print(f"time: {r['elapsed']}")
    del Ca; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    return r


# =====================================================================================
# 14. Binary-search noise (v0.2.3): how much "k heads" changes with other sentences and another order
# =====================================================================================
def search_noise(m, task_name, eval_sets=5, n=40, n_batch=10, target=0.90, verbose=True):
    """Separates two sources of noise in the circuit size:
    (a) EVALUATION sentences: same order, `eval_sets` groups of new sentences;
    (b) head ORDER: order A (seeds 0,1,2) x order B (seeds 10,11,12).
    Also sweeps the faithfulness curve near the cut-off (does faithfulness rise monotonically?)
    and repeats with double the evaluation sentences."""
    t0 = time.time(); nL, nH = m.cfg.n_layers, m.cfg.n_heads; N = nL * nH
    orders = {"A": head_order(m, task_name, (0, 1, 2), n)[0], "B": head_order(m, task_name, (10, 11, 12), n)[0]}
    rows, curve = [], {}
    for j in range(eval_sets):
        bat = [get_task(m, task_name, 7000 + 10 * j + b, n_batch) for b in range(n // n_batch)]; Cs = [caches(m, T) for T in bat]
        for name, order in orders.items():
            b = circuit_for_target(m, bat, Cs, order, range(nL), target=target)
            rows.append(dict(eval_set=j, order=name, k=b["k"], faith=round(b["faithfulness"], 3), reached=b["reached"]))
            if j == 0 and name == "A":
                k0 = b["k"]; eval = dict(b["curve"])
                ks = sorted(set(np.linspace(0, min(N, 2 * k0), 21).astype(int).tolist()) | set(range(max(0, k0 - 8), k0 + 1)))
                for k in ks:
                    if k not in eval: eval[k] = faithfulness_batches(m, bat, Cs, orders["A"][:k], range(nL))
                curve = dict(sorted(eval.items()))
        del Cs; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    bat = [get_task(m, task_name, 7500 + b, n_batch) for b in range(2 * n // n_batch)]; Cs = [caches(m, T) for T in bat]
    bg = circuit_for_target(m, bat, Cs, orders["A"], range(nL), target=target); del Cs
    df = pd.DataFrame(rows)
    summary = {o: dict(mean=float(g.k.mean()), sd=float(g.k.std(ddof=1)) if len(g) > 1 else 0.0, min=int(g.k.min()), max=int(g.k.max()))
              for o, g in df.groupby("order")}
    kA = int(df[(df.order == "A") & (df.eval_set == 0)].k.iloc[0]); kB = int(df[(df.order == "B") & (df.eval_set == 0)].k.iloc[0])
    overlap = len(set(orders["A"][:kA]) & set(orders["B"][:kB])) / max(1, min(kA, kB))
    cv = list(curve.items()); drops = [(k1, k2, round(f1 - f2, 3)) for (k1, f1), (k2, f2) in zip(cv, cv[1:]) if f2 < f1 - 0.02]
    first = next((k for k, f in cv if f >= target), None)
    r = dict(get_task=task_name if isinstance(task_name, str) else "custom", N=N, rows=rows, summary=summary,
             k_double_sentences=bg["k"], faith_double=round(bg["faithfulness"], 3), overlap_AB=round(overlap, 2),
             curve={int(k): round(float(f), 3) for k, f in curve.items()}, drops=drops, first_k_on_curve=first,
             k_search=kA, elapsed=round(time.time() - t0, 1))
    if verbose:
        print(f"=== search noise: {m.cfg.model_name} / {r['get_task']} (target {target:.0%}, {N} heads) ===")
        for o, s in summary.items():
            print(f"order {o}: k = {s['mean']:.0f} ± {s['sd']:.0f}  (min {s['min']}, max {s['max']})  on {eval_sets} groups of {n} sentences")
        print(f"with {2 * n} sentences: k = {bg['k']} (faith. {bg['faithfulness']:.0%}) | overlap of heads A x B: {overlap:.0%}")
        print(f"curve: first k with faith. >= target = {first} (search gave {kA}) | drops > 2 points: {len(drops)} {drops[:4]}")
        near = {k: f"{f:.2f}" for k, f in curve.items() if abs(k - kA) <= 8}
        print(f"near the cut-off: {near}")
        print(f"time: {r['elapsed']} s")
    return r


# =====================================================================================
# 15. COPY task: sequence copying (induction), v0.3.2, Phase 4 part 2 (new task)
# =====================================================================================
COPY_WORDS = [" apple", " river", " table", " house", " green", " music", " paper", " stone", " light", " water",
    " chair", " bread", " glass", " horse", " money", " night", " plant", " sugar", " train", " world",
    " dog", " cat", " sun", " moon", " car", " book", " door", " tree", " fish", " bird",
    " king", " road", " ship", " gold", " fire", " snow", " rain", " wind", " milk", " salt",
    " box", " cup", " hat", " key", " bed", " bag", " pen", " map", " egg", " ice",
    " city", " game", " film", " song", " food", " wall", " park", " farm", " bank", " card",
    " red", " blue", " black", " white", " happy", " small", " large", " quick", " cold", " warm",
    " doctor", " teacher", " window", " garden", " market", " island", " forest", " summer", " winter", " letter",
    " phone", " clock", " shirt", " shoe", " coat", " ring", " bell", " lamp", " rock", " sand"]
COPY_LEN = 8
_words_ok = {}

def copy_words(m):
    key = m.cfg.model_name
    if key not in _words_ok:
        ok = [w for w in COPY_WORDS if m.to_tokens(w, prepend_bos=False).shape[1] == 1]
        _words_ok[key] = sorted(set(m.to_single_token(w) for w in ok))
    return _words_ok[key]

def build_copy(m, seed, n=40, L=None):
    # Sequence copying (induction). Clean: S + S[:L-1] -> answer S[L-1].
    # Corrupted: same, but the last word of the 1st half is replaced (b) -> the model copies b.
    # Only one word changes, so the circuit is whatever carries that word to the end.
    L = L or COPY_LEN; ids = copy_words(m); rnd = random.Random(seed)
    bos = [m.tokenizer.bos_token_id] if bool(getattr(m.cfg, "default_prepend_bos", True)) and m.tokenizer.bos_token_id is not None else []
    tk, tkc, a, b = [], [], [], []
    for _ in range(n):
        S = rnd.sample(ids, L + 1); x, y = S[L - 1], S[L]
        Sc = S[:L - 1] + [y]
        tk.append(bos + S[:L] + S[:L - 1]); tkc.append(bos + Sc + S[:L - 1]); a.append(x); b.append(y)
    return dict(tk=torch.tensor(tk, device=DEV), tkc=torch.tensor(tkc, device=DEV),
                target=torch.tensor(a, device=DEV), wrong=torch.tensor(b, device=DEV))

RECIPES["COPY"] = build_copy


# =====================================================================================
# 16. EAP-IG: integrated gradients to order the heads (v0.5, test)
# =====================================================================================
def node_gradient_ig(m, T, C=None, n_steps=5, alphas=None):
    """Like node_gradient, but the gradient is the MEAN along the corrupted -> clean path.
    The sentence embedding is interpolated: e = e_corr + a * (e_clean - e_corr), with a in `alphas`
    (default: midpoints of `n_steps` slices). Catches pieces that only act together with others (AND gate),
    whose gradient at the corrupted sentence (a = 0) is almost zero. alphas=[0] reproduces node_gradient."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; C = C or caches(m, T); n = T["tk"].shape[0]
    alphas = list(alphas) if alphas is not None else [(k + 0.5) / n_steps for k in range(n_steps)]
    f = lambda x: x.endswith("attn.hook_z") or x.endswith("hook_mlp_out")
    head = np.zeros(nL * nH); mlp = np.zeros(nL)
    m.set_use_split_qkv_input(False)
    try:
        for a in range(0, n, GRAD_BATCH):
            b = min(n, a + GRAD_BATCH); Tf = _slice(T, a, b)
            e_c = C["cc"]["hook_embed"][a:b]; e_l = C["cl"]["hook_embed"][a:b]
            for al in alphas:
                gr = {}
                torch.set_grad_enabled(True); m.reset_hooks()
                m.add_hook("hook_embed", lambda x, hook, al=al: (e_c + al * (e_l - e_c)).clone().requires_grad_(True))
                m.add_hook(f, lambda g, hook: gr.__setitem__(hook.name, g.detach()), "bwd")
                metric(Tf, _last_logits(m, Tf["tkc"])).sum().backward()
                m.reset_hooks(); torch.set_grad_enabled(False)
                for L in range(nL):
                    z = f"blocks.{L}.attn.hook_z"; o = f"blocks.{L}.hook_mlp_out"
                    head[L * nH:(L + 1) * nH] += (gr[z] * (C["cl"][z][a:b] - C["cc"][z][a:b])).sum((0, 1, 3)).cpu().numpy() / len(alphas)
                    mlp[L] += (gr[o] * (C["cl"][o][a:b] - C["cc"][o][a:b])).sum().item() / len(alphas)
                del gr
    finally:
        m.reset_hooks(); torch.set_grad_enabled(False); m.set_use_split_qkv_input(True)
    nan = int(np.isnan(head).sum() + np.isnan(mlp).sum())
    return head / (n * C["total"]), mlp / (n * C["total"]), nan

def head_order_ig(m, task_name, seeds=(0, 1, 2), n=40, n_steps=5):
    """Head order by the EAP-IG score (mean over several seeds)."""
    scores = []
    for sd in seeds:
        T = get_task(m, task_name, sd, n); C = caches(m, T)
        scores.append(node_gradient_ig(m, T, C, n_steps=n_steps)[0]); del C
    score = np.mean(scores, 0)
    return [int(c) for c in np.argsort(-np.abs(score))], score


# =====================================================================================
# 17. Corruption battery (v0.5): the same task with different questions -> the role of each head
# =====================================================================================
def build_copy_emb(m, seed, n=40, L=None):
    # Corrupted: the 1st half has the SAME words, in another order. The answer word is still there,
    # but "who came before whom" changes -> reveals previous-token heads (e.g. 4.11 in GPT-2).
    # wrong = the word that follows S[L-2] in the shuffled 1st half (what induction would copy).
    L = L or COPY_LEN; ids = copy_words(m); rnd = random.Random(seed)
    bos = [m.tokenizer.bos_token_id] if bool(getattr(m.cfg, "default_prepend_bos", True)) and m.tokenizer.bos_token_id is not None else []
    tk, tkc, a, b = [], [], [], []
    while len(tk) < n:
        S = rnd.sample(ids, L); P = S[:]; rnd.shuffle(P)
        j = P.index(S[L - 2])
        if j == L - 1 or P[j + 1] == S[L - 1]: continue
        tk.append(bos + S + S[:L - 1]); tkc.append(bos + P + S[:L - 1]); a.append(S[L - 1]); b.append(P[j + 1])
    return dict(tk=torch.tensor(tk, device=DEV), tkc=torch.tensor(tkc, device=DEV),
                target=torch.tensor(a, device=DEV), wrong=torch.tensor(b, device=DEV))

def build_copy_new(m, seed, n=40, L=None):
    # Corrupted: the 1st half is replaced ENTIRELY by other words -> there is nothing to copy.
    # Reveals the heads that notice "this has appeared before". wrong = the word at the same position of the new 1st half.
    L = L or COPY_LEN; ids = copy_words(m); rnd = random.Random(seed)
    bos = [m.tokenizer.bos_token_id] if bool(getattr(m.cfg, "default_prepend_bos", True)) and m.tokenizer.bos_token_id is not None else []
    tk, tkc, a, b = [], [], [], []
    for _ in range(n):
        S = rnd.sample(ids, 2 * L); S1, S2 = S[:L], S[L:]
        tk.append(bos + S1 + S1[:L - 1]); tkc.append(bos + S2 + S1[:L - 1]); a.append(S1[L - 1]); b.append(S2[L - 1])
    return dict(tk=torch.tensor(tk, device=DEV), tkc=torch.tensor(tkc, device=DEV),
                target=torch.tensor(a, device=DEV), wrong=torch.tensor(b, device=DEV))

RECIPES["COPY_EMB"] = build_copy_emb
RECIPES["COPY_NEW"] = build_copy_new
QUESTIONS = {"COPY": "carries the word", "COPY_EMB": "marks the order", "COPY_NEW": "detects repetition"}

def corruption_battery(m, tasks=("COPY", "COPY_EMB", "COPY_NEW"), prune=False, verbose=True, **kw):
    """Runs HeadHunter on several corruptions of the same task and joins them: for each head, in which
    questions it is in the core (N) or in the circuit (c). Returns the table and the results."""
    nH = m.cfg.n_heads; name = lambda c: f"{c // nH}.{c % nH}"; res, role = {}, defaultdict(dict)
    for t in tasks:
        r = analyze(m, t, prune=prune, verbose=False, **kw); res[t] = r
        for c in r["circuit"]["heads"]: role[name(c)][t] = "c"
        for c in r["core"]["heads"]: role[name(c)][t] = "N"
        if verbose:
            print(f"{t:11s} ({QUESTIONS.get(t, t)}): core {len(r['core']['heads'])}, circuit {len(r['circuit']['heads'])}, "
                  f"faith {r['circuit']['faithfulness']:.0%}, {r['elapsed']['total']} s")
    tab = pd.DataFrame({t: {h: role[h].get(t, "") for h in role} for t in tasks}).fillna("")
    tab["in_core"] = (tab[list(tasks)] == "N").sum(1)
    tab = tab.sort_values(["in_core"] + list(tasks), ascending=False)
    if verbose:
        print("heads in the core of at least one question (N = core, c = circuit only):")
        print(tab[tab.in_core > 0].drop(columns="in_core").to_string())
    return tab, res


# =====================================================================================
# 18. Error bar (v0.5): the circuit as a range, over several groups of sentences and several targets
# =====================================================================================
def circuit_range(m, task_name, groups=5, goals=(0.80, 0.90, 0.95), n=40, n_batch=10, order="ig",
                   ig_steps=5, order_seeds=(0, 1, 2), verbose=True):
    """For each target, the minimum k over `groups` groups of new sentences (seeds 7000+, step 10).
    Returns the median and the range (min-max) per target. On large models the range is the honest result."""
    nL = m.cfg.n_layers; t0 = time.time()
    o = (head_order_ig(m, task_name, order_seeds, n, n_steps=ig_steps)[0] if order == "ig"
         else head_order(m, task_name, order_seeds, n)[0])
    ks = defaultdict(list)
    for j in range(groups):
        bat = [get_task(m, task_name, 7000 + 10 * j + b, n_batch) for b in range(n // n_batch)]; Cs = [caches(m, T) for T in bat]
        for goal in goals: ks[goal].append(circuit_for_target(m, bat, Cs, o, range(nL), target=goal)["k"])
        del Cs; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    r = {goal: dict(median=float(np.median(v)), min=int(min(v)), max=int(max(v)), every_one=v) for goal, v in ks.items()}
    if verbose:
        print(f"=== circuit range: {m.cfg.model_name} / {task_name} (order {order}, {groups} groups of {n} sentences) ===")
        for goal, s in r.items():
            print(f"target {goal:.0%}: k = {s['median']:.0f}  (range {s['min']}-{s['max']}, ±{(s['max'] - s['min']) / 2 / max(s['median'], 1):.0%})")
        print(f"time: {time.time() - t0:.1f} s")
    return dict(get_task=task_name, order=order, goals=r, elapsed=round(time.time() - t0, 1))


# =====================================================================================
# 19. Scoreboard (v0.5): GPT-2 small IOI against the gold list of Wang et al. (2022) and other methods
# =====================================================================================
GOLD_IOI_GPT2 = {
    "name mover": ["9.6", "9.9", "10.0"],
    "backup name mover": ["9.0", "9.7", "10.1", "10.2", "10.6", "10.10", "11.2", "11.9"],
    "negative name mover": ["10.7", "11.10"],
    "s-inhibition": ["7.3", "7.9", "8.6", "8.10"],
    "induction": ["5.5", "5.8", "5.9", "6.9"],
    "duplicate token": ["0.1", "0.10", "3.0"],
    "previous token": ["2.2", "4.11"]}

def _mean_ablation(m, T, n_batch=10):
    """Venkatesh style: replaces each head's output by the MEAN (on the clean batch) and measures the drop in the
    probability of the right answer. Measures NECESSITY (what breaks), not sufficiency."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; ar = torch.arange(T["tk"].shape[0], device=T["tk"].device)
    f = lambda x: x.endswith("attn.hook_z")
    lg, cl = m.run_with_cache(T["tk"], names_filter=f)
    p0 = lg[:, -1].softmax(-1)[ar, T["target"]].mean().item(); del lg; drop = np.zeros(nL * nH)
    for L in range(nL):
        mean = cl[f"blocks.{L}.attn.hook_z"].mean(0, keepdim=True)
        for h in range(nH):
            def hk(z, hook, h=h, mean=mean): z[:, :, h] = mean[:, :, h]; return z
            lg = m.run_with_hooks(T["tk"], fwd_hooks=[(f"blocks.{L}.attn.hook_z", hk)])
            drop[L * nH + h] = p0 - lg[:, -1].softmax(-1)[ar, T["target"]].mean().item(); del lg
    return drop

def _acdc_nodes(m, batches, Cs, tau=0.01):
    """Simplified ACDC, by nodes: starts with all heads and, from the last layer to the first,
    removes every head whose removal CHANGES the faithfulness by less than `tau` (up or down). Counts the evaluations."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; current = set(range(nL * nH)); eval_ = 0
    f_current = faithfulness_batches(m, batches, Cs, sorted(current), range(nL)); eval_ += 1
    for c in sorted(current, reverse=True):
        tried = sorted(current - {c}); f = faithfulness_batches(m, batches, Cs, tried, range(nL)); eval_ += 1
        if abs(f_current - f) < tau: current.discard(c); f_current = f   # v0.5.1: |change| (as in the original ACDC), not just a drop
    return sorted(current), f_current, eval_

def scoreboard_ioi_gpt2(m, n=40, n_batch=10, tau_acdc=0.01, verbose=True):
    """Compares methods on GPT-2 small IOI. For each one: hits against the gold list (precision and recall),
    circuit faithfulness (heads + all MLPs, on new sentences) and cost (time).
    The ranking methods take the top-k heads, with k = size of the gold list (26)."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; name = lambda c: f"{c // nH}.{c % nH}"
    idx = lambda s: int(s.split(".")[0]) * nH + int(s.split(".")[1])
    gold = {idx(h) for hs in GOLD_IOI_GPT2.values() for h in hs}; K = len(gold)
    bat_a = [get_task(m, "IOI", 7000 + b, n_batch) for b in range(n // n_batch)]; Ca = [caches(m, T) for T in bat_a]
    bat_v = [get_task(m, "IOI", 9000 + b, n_batch) for b in range(n // n_batch)]; Cv = [caches(m, T) for T in bat_v]
    T0 = get_task(m, "IOI", 0, n); C0 = caches(m, T0); rows = {}
    def register(method_name, head, sec, extra=""):
        head = set(int(c) for c in head); accuracy = len(head & gold)
        rows[method_name] = dict(heads=len(head), hits=accuracy, precision=accuracy / max(len(head), 1), recall=accuracy / K,
                              faith_new=faithfulness_batches(m, bat_v, Cv, sorted(head), range(nL)), seconds=round(sec, 1),
                              note=extra, missing=sorted(name(c) for c in gold - head))
    t = time.time(); r = analyze(m, "IOI", prune=False, verbose=False); register("HeadHunter (circuit 90%)", r["circuit"]["heads"], time.time() - t)
    register("HeadHunter (core)", r["core"]["heads"], r["elapsed"]["core"])
    t = time.time(); o, _ = head_order(m, "IOI", (0, 1, 2), n); register("EAP (gradiente), top-26", o[:K], time.time() - t)
    t = time.time(); o, _ = head_order_ig(m, "IOI", (0, 1, 2), n); register("EAP-IG, top-26", o[:K], time.time() - t)
    t = time.time(); ex = exact_heads(m, T0, C0); o = sorted(ex, key=lambda c: -abs(ex[c]))
    register("Exact patching (one at a time), top-26", o[:K], time.time() - t)
    t = time.time(); q = _mean_ablation(m, T0); o = list(np.argsort(-np.abs(q)))
    register("Mean ablation (Venkatesh), top-26", o[:K], time.time() - t, "measures necessity")
    t = time.time(); head, f, ev = _acdc_nodes(m, bat_a, Ca, tau_acdc)
    register(f"Simplified ACDC (nodes, tau={tau_acdc})", head, time.time() - t, f"{ev} evaluations")
    del Ca, Cv, C0; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    df = pd.DataFrame(rows).T
    if verbose:
        print(f"=== scoreboard GPT-2 small IOI | gold list Wang et al.: {K} heads ===")
        print(df[["heads", "hits", "precision", "recall", "faith_new", "seconds", "note"]].to_string(
            formatters={"precision": "{:.0%}".format, "recall": "{:.0%}".format, "faith_new": "{:.0%}".format}))
    return df


# =====================================================================================
# 20. MLP pruning (v0.6, test): the circuit no longer carries ALL the MLPs
# =====================================================================================
def ig_scores(m, task_name, seeds=(0, 1, 2), n=40, n_steps=5):
    """EAP-IG scores of heads AND MLPs (mean over several seeds)."""
    cs, ms = [], []
    for sd in seeds:
        T = get_task(m, task_name, sd, n); C = caches(m, T)
        c, mm, _ = node_gradient_ig(m, T, C, n_steps=n_steps); cs.append(c); ms.append(mm); del C
    return np.mean(cs, 0), np.mean(ms, 0)

def circuit_with_mlps(m, task_name, target=0.90, mlp_target=0.97, n=40, n_batch=10, verbose=True):
    """Two stages, same technique as the rest of HeadHunter (EAP-IG order + binary search):
    1) with ALL the heads, the smallest number of MLPs (in EAP-IG order) that reaches `mlp_target`;
    2) with those MLPs, the smallest number of heads that reaches `target`. Compares with the all-MLP circuit."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; t0 = time.time()
    nc, nm = ig_scores(m, task_name, n=n)
    oc = [int(c) for c in np.argsort(-np.abs(nc))]; om = [int(L) for L in np.argsort(-np.abs(nm))]
    bat = [get_task(m, task_name, 7000 + b, n_batch) for b in range(n // n_batch)]; Cs = [caches(m, T) for T in bat]
    every_one = list(range(nL * nH)); eval = {}
    def fm(k):
        if k not in eval: eval[k] = faithfulness_batches(m, bat, Cs, every_one, om[:k])
        return eval[k]
    lo, hi = 0, nL
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fm(mid) >= mlp_target: hi = mid
        else: lo = mid
    mlps = sorted(om[:hi])
    b_all = circuit_for_target(m, bat, Cs, oc, range(nL), target=target)
    b_pod = circuit_for_target(m, bat, Cs, oc, mlps, target=target); f_mlp = fm(hi); del Cs
    lv = [get_task(m, task_name, 9000 + b, n_batch) for b in range(n // n_batch)]; Cv = [caches(m, T) for T in lv]
    fv_all = faithfulness_batches(m, lv, Cv, oc[:b_all["k"]], range(nL))
    fv_pod = faithfulness_batches(m, lv, Cv, oc[:b_pod["k"]], mlps); del Cv
    gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    r = dict(get_task=task_name, mlps=mlps, n_mlps=len(mlps), faith_mlps_all_heads=f_mlp,
             heads_all_mlps=b_all["k"], heads_mlps_pruned=b_pod["k"], reached=b_pod["reached"],
             faith_new_all=fv_all, faith_new_pruned=fv_pod, order_mlps=om, elapsed=round(time.time() - t0, 1))
    if verbose:
        print(f"=== MLP pruning: {m.cfg.model_name} / {task_name} ===")
        print(f"MLPs: {len(mlps)}/{nL} suffice ({f_mlp:.0%} with all heads): {mlps}")
        print(f"circuit with ALL the MLPs: {b_all['k']} heads + {nL} MLPs -> new sentences {fv_all:.0%}")
        print(f"circuit with pruned MLPs:  {b_pod['k']} heads + {len(mlps)} MLPs -> new sentences {fv_pod:.0%}"
              f"{'' if b_pod['reached'] else '  (did NOT reach the target)'}")
        print(f"time: {r['elapsed']} s")
    return r


# =====================================================================================
# 21. Features (v0.6, test): how many SAE features carry the task at each layer?
# =====================================================================================
class SimpleSAE:
    """SAE in the standard format (W_enc, b_enc, W_dec, b_dec): f = ReLU((x - b_dec) @ W_enc + b_enc)."""
    def __init__(self, weights, dev):
        self.W_enc = weights["W_enc"].to(dev).float(); self.b_enc = weights["b_enc"].to(dev).float()
        self.W_dec = weights["W_dec"].to(dev).float(); self.b_dec = weights["b_dec"].to(dev).float()
    def encode(self, x): return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

def load_sae_gpt2(L, repo="jbloom/GPT2-Small-SAEs-Reformatted"):
    """Residual SAE (before block L) of GPT-2 small, by Joseph Bloom (the same ones as Neuronpedia, 'res-jb')."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.torch import load_file
    files = [a for a in list_repo_files(repo) if a.startswith(f"blocks.{L}.hook_resid_pre/") and a.endswith(".safetensors")]
    if not files: raise FileNotFoundError(f"could not find the SAE for layer {L} on {repo}")
    return SimpleSAE(load_file(hf_hub_download(repo, files[0])), DEV)

def features_per_layer(m, task_name, layers=None, target=0.90, n=40, n_batch=10, top=5, verbose=True):
    """For each layer L: on the residual before block L, the corrupted sentence receives
    resid_corr + sum_{i in circuit} (f_i clean - f_i corrupted) * d_i.
    Feature order: attribution (gradient of the metric on the residual) . (delta f_i d_i). Binary search on the number
    of features up to `target`. 'ceiling' = faithfulness with ALL the features (what the SAE does not reconstruct is left out)."""
    nL = m.cfg.n_layers; layers = range(nL) if layers is None else layers; rows = []
    bat = [get_task(m, task_name, 7000 + b, n_batch) for b in range(n // n_batch)]
    for L in layers:
        t0 = time.time(); sae = load_sae_gpt2(L); hook_name = f"blocks.{L}.hook_resid_pre"
        data = []
        for T in bat:
            _, cl = m.run_with_cache(T["tk"], names_filter=hook_name); _, cc = m.run_with_cache(T["tkc"], names_filter=hook_name)
            xl, xc = cl[hook_name], cc[hook_name]; del cl, cc
            ml = metric(T, _last_logits(m, T["tk"])); mc = metric(T, _last_logits(m, T["tkc"]))
            g = {}
            torch.set_grad_enabled(True); m.reset_hooks()
            m.add_hook(hook_name, lambda x, hook: x.requires_grad_(True) if not x.requires_grad else x)
            m.add_hook(hook_name, lambda gr, hook: g.__setitem__("g", gr.detach()), "bwd")
            m.add_hook("hook_embed", lambda x, hook: x.requires_grad_(True))
            metric(T, _last_logits(m, T["tkc"])).sum().backward()
            m.reset_hooks(); torch.set_grad_enabled(False)
            df = sae.encode(xl) - sae.encode(xc)                                 # (B, P, n_feat)
            atr = (df * (g["g"] @ sae.W_dec.T)).sum((0, 1))                       # (n_feat,)
            data.append(dict(T=T, xc=xc, df=df, atr=atr, ml=ml, mc=mc))
        atr = sum(d["atr"] for d in data); order = torch.argsort(-atr.abs())
        active = int((atr != 0).sum()); den = sum((d["ml"] - d["mc"]).sum().item() for d in data)
        def faith(k):
            sel = order[:k]; num = 0.0
            for d in data:
                x = d["xc"] + d["df"][:, :, sel] @ sae.W_dec[sel]
                mk = metric(d["T"], final_logits(m, x, L, ()))
                num += (mk - d["mc"]).sum().item()
            return num / den
        ceiling = faith(active); eval = {}
        def f(k):
            if k not in eval: eval[k] = faith(k)
            return eval[k]
        lo, hi = 0, active
        if ceiling >= target:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if f(mid) >= target: hi = mid
                else: lo = mid
        k = hi if ceiling >= target else None
        tops = [int(i) for i in order[:top]]
        rows.append(dict(layer=L, features_active=active, k_target=k, ceiling=round(ceiling, 3), top=tops,
                           links=[f"https://www.neuronpedia.org/gpt2-small/{L}-res-jb/{i}" for i in tops],
                           seconds=round(time.time() - t0, 1)))
        if verbose:
            print(f"layer {L:2d}: {active:5d} features change | ceiling {ceiling:.0%} | "
                  f"{'k = ' + str(k) if k else 'does not reach ' + format(target, '.0%')} | top {tops} | {time.time() - t0:.0f} s")
        del data, sae; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    return pd.DataFrame(rows)


# =====================================================================================
# 22. Necessity (Venkatesh 2026 style) side by side with sufficiency (v0.6.1)
# =====================================================================================
def _prob_with_ablation(m, T, means, heads):
    # CLEAN sentence; the given heads receive the mean (on the clean batch) of their output. Returns the mean probability of the right answer.
    nH = m.cfg.n_heads; per = defaultdict(list)
    for c in heads: per[int(c) // nH].append(int(c) % nH)
    hooks = []
    for L, hs in per.items():
        def hk(z, hook, L=L, hs=hs):
            z[:, :, hs] = means[L][:, :, hs]; return z
        hooks.append((f"blocks.{L}.attn.hook_z", hk))
    ar = torch.arange(T["tk"].shape[0], device=T["tk"].device)
    lg = final_logits(m, T["tk"], None, hooks)
    return lg[:, -1].softmax(-1)[ar, T["target"]].mean().item()

def necessity(m, task_name, n=40, damage_target=0.80, verbose=True):
    # How many heads must GO (mean ablation) to cause `damage_target` of the maximum damage?
    # damage(k) = (p0 - p_k) / (p0 - p_all), with the heads sorted by individual probability drop.
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; name = lambda c: f"{c // nH}.{c % nH}"; t0 = time.time()
    T = get_task(m, task_name, 0, n)
    _, cl = m.run_with_cache(T["tk"], names_filter=lambda x: x.endswith("attn.hook_z"))
    means = {L: cl[f"blocks.{L}.attn.hook_z"].mean(0, keepdim=True) for L in range(nL)}; del cl
    p0 = _prob_with_ablation(m, T, means, []); p_all = _prob_with_ablation(m, T, means, range(nL * nH))
    drop = np.array([p0 - _prob_with_ablation(m, T, means, [c]) for c in range(nL * nH)])
    order = [int(c) for c in np.argsort(-drop)]; eval = {}
    def damage(k):
        if k not in eval: eval[k] = (p0 - _prob_with_ablation(m, T, means, order[:k])) / max(p0 - p_all, 1e-9)
        return eval[k]
    hi = next((k for k in range(1, nL * nH + 1) if damage(k) >= damage_target), nL * nH)   # sweep: the damage is not monotonic
    r = dict(get_task=task_name, p0=p0, p_all=p_all, k_damage=hi, damage=damage(hi), top=[name(c) for c in order[:10]],
             damage_top1=damage(1), elapsed=round(time.time() - t0, 1))
    if verbose:
        print(f"necessity ({m.cfg.model_name} / {task_name}): p0 = {p0:.2f}, all ablated = {p_all:.2f} | "
              f"{hi} head(s) cause {damage(hi):.0%} of the damage (the 1st alone: {damage(1):.0%}) | top {r['top'][:5]} | {r['elapsed']} s")
    return r

def necessity_vs_sufficiency(m, task_name, **kw):
    # The two sides of the same question: how many heads BREAK the task (necessity) and how many are ENOUGH (sufficiency).
    nH = m.cfg.n_heads; name = lambda c: f"{c // nH}.{c % nH}"
    nec = necessity(m, task_name)
    suf = analyze(m, task_name, prune=False, verbose=False, **kw)
    core = [name(c) for c in suf["core"]["heads"]]
    common = len(set(nec["top"][:nec["k_damage"]]) & set(core))
    print(f">>> {m.cfg.model_name} / {task_name}: NECESSITY {nec['k_damage']} head(s) for 80% of the damage | "
          f"CORE {len(core)} | SUFFICIENCY {len(suf['circuit']['heads'])}/{m.cfg.n_layers * nH} p/ 90% "
          f"| top-necessity in the core: {common}/{min(nec['k_damage'], 10)}")
    return dict(necessity=nec, core=core, k_sufficiency=len(suf["circuit"]["heads"]))


# =====================================================================================
# 23. Validation A (v0.7): is the circuit right? faithfulness, minimality, random control, GQA edges
# =====================================================================================
def validate_circuit(m, task_name, r=None, groups=3, n=40, n_batch=10, draws=5, eps=0.005,
                     sample_edges=60, verbose=True, **kw):
    # r = output of analyze (if None, runs it). Tests, on NEW sentences (seeds 9500+):
    # 1) circuit faithfulness over `groups` groups; 2) minimality: how much the faithfulness drops when each head is removed;
    # 3) control: random circuits of the same size; 4) edges: EAP vs exact on a sample (with GQA = approximation).
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; t0 = time.time()
    r = r or analyze(m, task_name, prune=False, verbose=False, **kw)
    head = [int(c) for c in r["circuit"]["heads"]]; mlps = list(r["circuit"]["mlps"]); k = len(head)
    fids = []
    for g in range(groups):
        bat = [get_task(m, task_name, 9500 + 10 * g + b, n_batch) for b in range(n // n_batch)]; Cs = [caches(m, T) for T in bat]
        fids.append(faithfulness_batches(m, bat, Cs, head, mlps))
        if g == 0: lot0, C0 = bat, Cs
        else: del Cs
    base = fids[0]
    drops = np.array([base - faithfulness_batches(m, lot0, C0, [x for x in head if x != c], mlps) for c in head])
    rnd = np.random.default_rng(0)
    rand = [faithfulness_batches(m, lot0, C0, sorted(rnd.choice(nL * nH, k, replace=False).tolist()), mlps) for _ in range(draws)]
    so_mlps = faithfulness_batches(m, lot0, C0, [], mlps)   # no heads, only the MLPs: the control baseline
    # edges: EAP vs exact on a sample of the edges into circuit heads
    ar = [a for a in edge_parts(m, head, mlps) if a[1][0] == "H"]
    rs = random.Random(0); sample = rs.sample(ar, min(sample_edges, len(ar)))
    T1, Ca1 = lot0[0], C0[0]
    eap = edge_scores(m, [T1], [Ca1], sample)
    den = (Ca1["ml"] - Ca1["mc"]).sum().item()
    exact = {a: exact_edge(m, T1, Ca1, *a) / den for a in sample}
    from scipy.stats import spearmanr
    rho = float(spearmanr([eap[a] for a in sample], [exact[a] for a in sample])[0])
    del C0; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    res = dict(get_task=task_name, k=k, faith_groups=[round(f, 3) for f in fids], faith_min=round(min(fids), 3),
               drop_mean=float(drops.mean()), drop_median=float(np.median(drops)),
               surplus=int((drops < eps).sum()), surplus_frac=float((drops < eps).mean()),
               random_mean=float(np.mean(rand)), random_max=float(np.max(rand)), so_mlps=float(so_mlps),
               edges_spearman=rho, gqa=adapter(m)["gqa"], elapsed=round(time.time() - t0, 1))
    if verbose:
        print(f"=== validation A: {m.cfg.model_name} / {task_name} (circuit of {k} heads + {len(mlps)} MLPs) ===")
        print(f"1) faithfulness on {groups} groups of new sentences: {[f'{f:.0%}' for f in fids]} (min {min(fids):.0%})")
        print(f"2) minimality: removing 1 head lowers the faithfulness on average by {drops.mean():.1%} (median {np.median(drops):.1%}); "
              f"{res['surplus']}/{k} heads are barely missed (< {eps:.1%})")
        print(f"3) control: {draws} random circuits of {k} heads give on average {np.mean(rand):.0%} (max {np.max(rand):.0%}); "
              f"MLPs only, no heads: {so_mlps:.0%}")
        print(f"4) edges into heads: EAP vs exact, Spearman {rho:.2f} on {len(sample)} edges"
              f"{' (GQA: k/v aproximadas)' if res['gqa'] else ''}")
        print(f"time: {res['elapsed']} s")
    return res


# =====================================================================================
# 24. Generalization (v0.8): does the circuit found on one sentence template work on other templates?
# =====================================================================================
IOI_MODELS_ALT = ["After{A} and{B} had lunch at the cafe,{B} handed a menu to",
                   "While{A} and{B} were walking in the park,{B} threw the ball to",
                   "Then,{A} and{B} were working at the office.{B} decided to give a book to"]

def build_ioi_alt(m, seed, n=40):
    # Same task (IOI), other templates. One template per batch (all sentences with the same length).
    names_ok = [x for x in NAMES if m.to_tokens(x, prepend_bos=False).shape[1] == 1]
    rnd = random.Random(seed); template = IOI_MODELS_ALT[seed % len(IOI_MODELS_ALT)]; fr, fc, io, s = [], [], [], []
    for _ in range(n):
        A, B = rnd.sample(names_ok, 2)
        if rnd.random() < 0.5: f = template.format(A=A, B=B)
        else: f = template.replace("{A} and{B}", "{B} and{A}").format(A=A, B=B)
        i = f.rindex(B); fr.append(f); fc.append(f[:i] + A + f[i + len(B):])   # only the LAST occurrence of B becomes A
        io.append(m.to_single_token(A)); s.append(m.to_single_token(B))
    return dict(tk=tok(m, fr), tkc=tok(m, fc), target=torch.tensor(io, device=DEV), wrong=torch.tensor(s, device=DEV))

CAP_TEMPLATES_ALT = ["In{P}, the capital city is", "Q: What is the capital of{P}? A: The capital is"]

def build_capitals_alt(m, seed, n=40):
    good = known_countries(m); rnd = random.Random(seed); template = CAP_TEMPLATES_ALT[seed % len(CAP_TEMPLATES_ALT)]
    fr, fc, cap, capc = [], [], [], []
    for _ in range(n):
        (p, c), (p2, c2) = rnd.sample(good, 2)
        fr.append(template.format(P=" " + p)); fc.append(template.format(P=" " + p2))
        cap.append(m.to_single_token(" " + c)); capc.append(m.to_single_token(" " + c2))
    return dict(tk=tok(m, fr), tkc=tok(m, fc), target=torch.tensor(cap, device=DEV), wrong=torch.tensor(capc, device=DEV))

def build_copy_alt(m, seed, n=40):
    return build_copy(m, seed, n, L=6)   # same task, shorter sequence (6 words instead of 8)

RECIPES["IOI_ALT"] = build_ioi_alt
RECIPES["CAP_ALT"] = build_capitals_alt
RECIPES["COPY_ALT"] = build_copy_alt

def accuracy(m, task_name, n=40, seed=0):
    # Fraction of clean sentences where the model puts the right answer first (can the model do the task?).
    T = get_task(m, task_name, seed, n); lg = final_logits(m, T["tk"])
    return (lg[:, -1].argmax(-1) == T["target"]).float().mean().item()

def generalization(m, base, alt, n=40, n_batch=10, draws=5, verbose=True, **kw):
    # 1) circuit found on the `base` template; 2) ITS faithfulness on the `alt` template (new sentences);
    # 3) circuit found directly on `alt` and the overlap; 4) random control on `alt`.
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; t0 = time.time()
    ac_b, ac_a = accuracy(m, base), accuracy(m, alt)
    rb = analyze(m, base, prune=False, verbose=False, **kw); cb = [int(c) for c in rb["circuit"]["heads"]]
    ra = analyze(m, alt, prune=False, verbose=False, **kw); ca = [int(c) for c in ra["circuit"]["heads"]]
    bat = [get_task(m, alt, 9600 + b, n_batch) for b in range(n // n_batch)]; Cs = [caches(m, T) for T in bat]
    f_cross = faithfulness_batches(m, bat, Cs, cb, range(nL)); f_own = faithfulness_batches(m, bat, Cs, ca, range(nL))
    rnd = np.random.default_rng(0)
    rand = float(np.mean([faithfulness_batches(m, bat, Cs, sorted(rnd.choice(nL * nH, len(cb), replace=False).tolist()), range(nL))
                           for _ in range(draws)]))
    del Cs; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    overlap = len(set(cb) & set(ca)) / max(1, min(len(cb), len(ca)))
    nb = {int(c) for c in rb["core"]["heads"]}; na = {int(c) for c in ra["core"]["heads"]}
    overlap_core = len(nb & na) / max(1, min(len(nb), len(na)))
    r = dict(base=base, alt=alt, accuracy_base=ac_b, accuracy_alt=ac_a, k_base=len(cb), k_alt=len(ca),
             faith_cross=f_cross, faith_own=f_own, random=rand, overlap=overlap,
             core_base=len(nb), core_alt=len(na), overlap_core=overlap_core, elapsed=round(time.time() - t0, 1))
    if verbose:
        print(f">>> {m.cfg.model_name}: {base} -> {alt} | model accuracy {ac_b:.0%} / {ac_a:.0%} | "
              f"circuit of {base} ({len(cb)} heads) on {alt}: {f_cross:.0%} | own circuit on {alt} ({len(ca)} heads): {f_own:.0%} | "
              f"random {rand:.0%} | heads in common {overlap:.0%}, core in common {overlap_core:.0%} | {r['elapsed']} s")
    return r


# =====================================================================================
# 25. New tasks (v0.9): Greater-Than (GT) and multiple choice (MCQA)
# =====================================================================================
def metric(T, lg):
    # Logit difference (right - wrong) at the last position, one per sentence. Accepts stacked batches.
    # GT: mean of the logits of the years GREATER than YY minus the mean over the years <= YY (Hanna et al. 2023 style).
    n = T["target"].shape[0]; k = lg.shape[0] // n; ar = torch.arange(lg.shape[0], device=lg.device)
    if "greater" in T:
        years = torch.tensor(T["years"], device=lg.device); last = lg[:, -1][:, years]
        M = T["greater"].repeat(k, 1).float()
        return ((last * M).sum(1) / M.sum(1).clamp(min=1) - (last * (1 - M)).sum(1) / (1 - M).sum(1).clamp(min=1)).float()
    a, b = T["target"].repeat(k), T["wrong"].repeat(k)
    return (lg[ar, -1, a] - lg[ar, -1, b]).float()

GT_NOUNS = [" war", " expedition", " project", " reign", " drought", " siege", " journey", " strike", " festival", " trial"]

def build_gt(m, seed, n=40):
    # Greater-Than: "The war lasted from the year 17YY to the year 17" -> the model should prefer years > YY.
    # Corrupted: YY = 01 (all years become "greater"). Only for tokenizers with 2-digit years in a single token (GPT-2, Pythia).
    years = [m.to_tokens(f"{y:02d}", prepend_bos=False) for y in range(100)]
    if sum(a.shape[1] == 1 for a in years) < 100:
        raise ValueError("GT needs 2-digit tokens ('00'..'99'); this tokenizer splits digits (e.g. Qwen)")
    years = [int(a[0, 0]) for a in years]; rnd = random.Random(seed)
    subst = [s for s in GT_NOUNS if m.to_tokens(s, prepend_bos=False).shape[1] == 1]   # all sentences with the same length
    template = "The{S} lasted from the year 17{Y} to the year 17"; ref = tok(m, [template.format(S=subst[0], Y="01")]).shape[1]
    fr, fc, greater, a, b = [], [], [], [], []
    while len(fr) < n:
        S = rnd.choice(subst); y = rnd.randint(2, 98)
        f = template.format(S=S, Y=f"{y:02d}"); fcor = template.format(S=S, Y="01")
        if tok(m, [f]).shape[1] != ref or tok(m, [fcor]).shape[1] != ref: continue
        fr.append(f); fc.append(fcor); greater.append([z > y for z in range(100)])
        a.append(years[min(y + 1, 99)]); b.append(years[y - 1])
    return dict(tk=tok(m, fr), tkc=tok(m, fc), target=torch.tensor(a, device=DEV), wrong=torch.tensor(b, device=DEV),
                greater=torch.tensor(greater, device=DEV), years=tuple(years))

MCQA_CATEGORIES = {
    "fruit": [" apple", " banana", " orange", " grape", " lemon", " cherry", " mango", " peach"],
    "animal": [" dog", " cat", " horse", " cow", " lion", " tiger", " bear", " wolf"],
    "color": [" red", " blue", " green", " yellow", " black", " white", " pink", " purple"],
    "country": [" France", " Spain", " Italy", " Japan", " China", " Brazil", " Canada", " Mexico"],
    "tool": [" hammer", " knife", " saw", " drill", " shovel", " brush", " needle", " rope"]}
_mcqa_ok = {}

def build_mcqa(m, seed, n=40):
    # Multiple choice (MIB style): "Which of these is a fruit? A. dog B. apple C. red D. saw Answer:" -> " B".
    # Corrupted: the SAME options in another order, with the right one under another letter (wrong = the new letter).
    key = m.cfg.model_name
    if key not in _mcqa_ok:
        is_one_token = lambda w: m.to_tokens(w, prepend_bos=False).shape[1] == 1
        _mcqa_ok[key] = {c: [w for w in ws if is_one_token(w)] for c, ws in MCQA_CATEGORIES.items() if is_one_token(" " + c)}
    cats = {c: ws for c, ws in _mcqa_ok[key].items() if len(ws) >= 2}
    letters = [m.to_single_token(x) for x in [" A", " B", " C", " D"]]; rnd = random.Random(seed)
    template = "Which of these is a {c}? A.{o0} B.{o1} C.{o2} D.{o3} Answer:"
    fr, fc, a, b = [], [], [], []
    for _ in range(n):
        c = rnd.choice(list(cats)); other = rnd.sample([x for x in cats if x != c], 3)
        ops = [rnd.choice(cats[c])] + [rnd.choice(cats[x]) for x in other]   # ops[0] is the right one
        p1 = rnd.sample(range(4), 4); i1 = p1.index(0)
        p2 = rnd.sample(range(4), 4)
        while p2.index(0) == i1: p2 = rnd.sample(range(4), 4)
        i2 = p2.index(0)
        fr.append(template.format(c=c, **{f"o{j}": ops[p1[j]] for j in range(4)}))
        fc.append(template.format(c=c, **{f"o{j}": ops[p2[j]] for j in range(4)}))
        a.append(letters[i1]); b.append(letters[i2])
    T = dict(tk=tok(m, fr), tkc=tok(m, fc), target=torch.tensor(a, device=DEV), wrong=torch.tensor(b, device=DEV))
    if T["tk"].shape != T["tkc"].shape: raise ValueError("MCQA: sentences with different lengths")
    return T

RECIPES["GT"] = build_gt
RECIPES["MCQA"] = build_mcqa

def accuracy(m, task_name, n=40, seed=0):
    # Fraction of clean sentences where the model's answer (top-1) is right. For GT: any year > YY.
    T = get_task(m, task_name, seed, n); lg = final_logits(m, T["tk"])[:, -1]; top = lg.argmax(-1)
    if "greater" in T:
        years = torch.tensor(T["years"], device=lg.device)
        return float(np.mean([bool(T["greater"][i][(years == top[i]).nonzero()].any()) if (years == top[i]).any() else False
                              for i in range(len(top))]))
    return (top == T["target"]).float().mean().item()

GOLD_GT_GPT2 = ["5.1", "5.5", "6.1", "6.9", "7.10", "8.8", "8.11", "9.1"]   # Hanna et al. 2023 (from memory; to be checked)

def new_task(m, task_name, gold=None, verbose=True, **kw):
    # Model hit + HeadHunter (without edges) + validation A. With a gold list, says where each known head landed.
    nH = m.cfg.n_heads; name = lambda c: f"{c // nH}.{c % nH}"; t0 = time.time()
    ac = accuracy(m, task_name)
    r = analyze(m, task_name, prune=False, verbose=False, **kw)
    v = validate_circuit(m, task_name, r=r, verbose=False)
    core = [name(c) for c in r["core"]["heads"]]; circ = [name(c) for c in r["circuit"]["heads"]]
    res = dict(get_task=task_name, accuracy=ac, core=core, mlps_core=r["core"]["mlps"], circuit=circ,
               validation=v, elapsed=round(time.time() - t0, 1))
    if gold:
        res["gold"] = {h: ("core" if h in core else f"circuit (#{circ.index(h) + 1})" if h in circ else "OUT") for h in gold}
    if verbose:
        print(f">>> {m.cfg.model_name} / {task_name}: accuracy {ac:.0%} | core {len(core)} {core[:8]} | MLPs {r['core']['mlps']} | "
              f"circuit {len(circ)}/{m.cfg.n_layers * nH} | new sentences {v['faith_min']:.0%}-{max(v['faith_groups']):.0%} | "
              f"random {v['random_mean']:.0%} | surplus {v['surplus']}/{v['k']} | {res['elapsed']} s")
        if gold: print("    gabarito:", res["gold"])
    return res
