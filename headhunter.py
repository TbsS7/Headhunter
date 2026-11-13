"""HeadHunter v0.11.5: núcleo, MLPs, circuito suficiente, arestas (híbrido) e GQA.

Uso:
    pip install git+https://github.com/TbsS7/Headhunter.git
    import headhunter as hh
    m = hh.carregar("gpt2-small")
    hh.autoteste(m, "IOI")
"""
import gc, os, json, time, random, platform
from collections import defaultdict
import numpy as np, pandas as pd, torch
from transformer_lens import HookedTransformer

VERSAO = "0.11.5"
ESQUEMA_IG = "meio"   # pontos da integral nas arestas: "meio" (padrão) ou "eapig" (igual ao EAP-IG oficial)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
pd.set_option("display.width", 250)


# =====================================================================================
# 1. Carregar modelo + adaptador de arquitetura
# =====================================================================================
GRANDES = {"pythia-1b": "EleutherAI/pythia-1b", "pythia-1.4b": "EleutherAI/pythia-1.4b",
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

def _sem_copias_de_pesos():
    """TL 3.9 copia W.T.contiguous() a cada chamada (unembed e MLP com gate); no backward essas cópias ficam guardadas
    (~5,5 GB no Qwen2.5-1.5B). Mesmo resultado em float32, sem as cópias."""
    Unembed.forward = _unembed; GatedMLP.forward = _gated

_sem_copias_de_pesos()

MODERNOS = ("qwen", "llama", "gemma", "olmo", "mistral", "phi")

def carregar(nome, hf_id=None, grande=None, dtype_carga=None, converter_na="auto"):
    """Modelos pequenos: direto. Grandes (ou hf_id dado): carrega em meia precisão -> GPU -> float32.
    Modelos modernos (Qwen, Llama...) carregam em bf16 (treinados assim) antes de virar float32."""
    gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    moderno = any(k in nome.lower() for k in MODERNOS)
    hf_id = hf_id or GRANDES.get(nome) or (nome if moderno and "/" in nome else None)
    dtype_carga = dtype_carga or (torch.bfloat16 if moderno else torch.float16)
    if grande is None: grande = hf_id is not None
    if grande:
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(hf_id or nome, torch_dtype=dtype_carga, low_cpu_mem_usage=True)
        n_par = sum(p.numel() for p in hf.parameters())
        # v0.4.1: na GPU, meia precisão + float32 convivem durante a conversão (~6 bytes por parâmetro).
        # Se isso não couber com folga, converte na RAM e só depois manda para a GPU (~4 bytes por parâmetro).
        na_cpu = converter_na == "cpu" or (converter_na == "auto" and DEV == "cuda"
                 and 6 * n_par > 0.85 * torch.cuda.get_device_properties(0).total_memory)
        m = HookedTransformer.from_pretrained_no_processing(nome, hf_model=hf, dtype=dtype_carga,
                                                            device="cpu" if na_cpu else DEV)
        del hf; gc.collect()
        m = m.to(torch.float32); gc.collect()
        if na_cpu: m = m.to(DEV); gc.collect()
        print(f"  {n_par / 1e9:.1f} B parâmetros | conversão para float32 {'na RAM (modelo grande)' if na_cpu else 'na GPU'}")
    else:
        m = HookedTransformer.from_pretrained(nome, device=DEV)
    preparar(m)
    a = adaptador(m)
    print(f"carregado: {nome} | {next(m.parameters()).dtype} | {a['camadas']}x{a['cabecas']} cabeças"
          f"{' (GQA: ' + str(a['cabecas_kv']) + ' kv)' if a['gqa'] else ''} | validação: {a['validacao']}")
    for av in a["avisos"]: print("  ⚠️", av)
    return m

def preparar(m):
    for p in m.parameters(): p.requires_grad_(False)
    torch.set_grad_enabled(False)
    m.set_use_split_qkv_input(True); m.set_use_hook_mlp_in(True)
    return m

def adaptador(m):
    """Lê a configuração e devolve o que o HeadHunter precisa saber da arquitetura."""
    c = m.cfg; nkv = getattr(c, "n_key_value_heads", None) or c.n_heads
    nome = (c.model_name or "").lower()
    pos_norm = bool(getattr(c, "use_normalization_before_and_after", False)) or \
        getattr(c, "original_architecture", "") in ("Olmo2ForCausalLM", "Olmo3ForCausalLM")
    a = dict(nome=c.model_name, camadas=c.n_layers, cabecas=c.n_heads, cabecas_kv=nkv, gqa=nkv < c.n_heads,
             paralelo=bool(getattr(c, "parallel_attn_mlp", False)), posicao=c.positional_embedding_type,
             norma=c.normalization_type, mlp=("gated " if c.gated_mlp else "") + str(c.act_fn), d_model=c.d_model,
             pos_norm=pos_norm, validacao="validada" if ("gpt2" in nome or "pythia" in nome) else "experimental",
             avisos=[])
    if a["gqa"]:
        a["avisos"].append("GQA: k e v são compartilhados por grupo de cabeças. Nós e suficiência: exatos. Arestas que entram por k/v: aproximadas (valem para o grupo).")
    if pos_norm:
        a["avisos"].append("Normalização depois da atenção (Gemma 2/3, OLMo 2/3): a saída por cabeça NÃO é aditiva. "
                           "Efeito de nós ok; arestas e suficiência ficam aproximadas.")
    if a["validacao"] != "validada":
        a["avisos"].append("Família ainda não validada no HeadHunter: rode autoteste() antes.")
    return a


# =====================================================================================
# 2. Tarefas (idênticas às usadas nos testes de 31/10/2026)
# =====================================================================================
def tok(m, textos):
    """BOS só se o modelo usa (GPT-2 e Pythia: sim; Qwen: não)."""
    return m.to_tokens(textos, prepend_bos=bool(getattr(m.cfg, "default_prepend_bos", True)))

NOMES = [" Mary", " John", " Tom", " James", " Dan", " Sid", " Martin",
         " Amy", " Kate", " Paul", " Lisa", " Mark", " Anna", " David"]
MODELOS_IOI = ["When{A} and{B} went to the store,{B} gave a drink to",
               "When{B} and{A} went to the store,{B} gave a drink to"]

def montar_ioi(m, seed, n=40):
    nomes_ok = [x for x in NOMES if m.to_tokens(x, prepend_bos=False).shape[1] == 1]
    rnd = random.Random(seed); fr, fc, io, s = [], [], [], []
    for _ in range(n):
        A, B = rnd.sample(nomes_ok, 2)
        f = rnd.choice(MODELOS_IOI).format(A=A, B=B)
        fr.append(f); fc.append(f.replace(f",{B} gave", f",{A} gave"))
        io.append(m.to_single_token(A)); s.append(m.to_single_token(B))
    a, b = torch.tensor(io, device=DEV), torch.tensor(s, device=DEV)
    return dict(tk=tok(m, fr), tkc=tok(m, fc), alvo=a, errado=b)

PAISES = {"France": "Paris", "Italy": "Rome", "Spain": "Madrid", "Germany": "Berlin", "Japan": "Tokyo",
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
MOLDE = "The capital of{P} is the city of"
_conhecidos = {}

def paises_conhecidos(m):
    chave = m.cfg.model_name
    if chave in _conhecidos: return _conhecidos[chave]
    um = lambda x: m.to_tokens(" " + x, prepend_bos=False).shape[1] == 1
    ok = [(p, c) for p, c in PAISES.items() if um(p) and um(c)]
    caps = torch.tensor([m.to_single_token(" " + c) for _, c in ok], device=DEV)
    lg = m(tok(m, [MOLDE.format(P=" " + p) for p, _ in ok]))[:, -1]
    rank = (lg > lg[torch.arange(len(ok)), caps][:, None]).sum(-1)
    bons = [ok[i] for i in range(len(ok)) if rank[i] < 10]
    _conhecidos[chave] = bons; return bons

def montar_capitais(m, seed, n=40):
    bons = paises_conhecidos(m); rnd = random.Random(seed)
    fr, fc, cap, capc = [], [], [], []
    for _ in range(n):
        (p, c), (p2, c2) = rnd.sample(bons, 2)
        fr.append(MOLDE.format(P=" " + p)); fc.append(MOLDE.format(P=" " + p2))
        cap.append(m.to_single_token(" " + c)); capc.append(m.to_single_token(" " + c2))
    return dict(tk=tok(m, fr), tkc=tok(m, fc), alvo=torch.tensor(cap, device=DEV),
                errado=torch.tensor(capc, device=DEV))

RECEITAS = {"IOI": montar_ioi, "CAP": montar_capitais}

def metrica(T, lg):
    """Diferença de logit (resposta certa − errada) na última posição, uma por frase. Aceita lotes empilhados."""
    n = T["alvo"].shape[0]; k = lg.shape[0] // n
    a, b = T["alvo"].repeat(k), T["errado"].repeat(k); ar = torch.arange(lg.shape[0], device=lg.device)
    return (lg[ar, -1, a] - lg[ar, -1, b]).float()

def tarefa(m, nome, seed, n=40):
    return RECEITAS[nome](m, seed, n) if isinstance(nome, str) else nome(m, seed, n)


# =====================================================================================
# 3. Rodadas baratas: só os logits da última posição, podendo começar no meio da rede
# =====================================================================================
def logits_finais(m, x, comeco=None, hooks=()):
    r = m.run_with_hooks(x, start_at_layer=comeco, stop_at_layer=m.cfg.n_layers, fwd_hooks=list(hooks))
    r = m.ln_final(r[:, -1:, :]) if m.cfg.normalization_type is not None else r[:, -1:, :]
    return m.unembed(r)

def _ctx(bf16):
    return torch.autocast(DEV, dtype=torch.bfloat16) if bf16 else torch.autocast(DEV, enabled=False)


# =====================================================================================
# 4. Efeito exato (patching) de cabeças e MLPs — versão lenta (referência) e rápida
# =====================================================================================
def exato_cabecas_lento(m, T):
    """Versão antiga: uma rodada completa por cabeça. Fica como referência do teste de regressão."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; f = lambda x: x.endswith("attn.hook_z")
    lg, cl = m.run_with_cache(T["tk"], names_filter=f); ml = metrica(T, lg); del lg
    lg, cc = m.run_with_cache(T["tkc"], names_filter=f); mc = metrica(T, lg); del lg
    total = (ml - mc).mean().item(); rel = np.zeros(nL * nH)
    for L in range(nL):
        for H in range(nH):
            def hk(z, hook, H=H):
                z[:, :, H, :] = cl[hook.name][:, :, H, :]; return z
            mk = metrica(T, m.run_with_hooks(T["tkc"], fwd_hooks=[(f"blocks.{L}.attn.hook_z", hk)]))
            rel[L * nH + H] = (mk - mc).mean().item() / total
    return rel, total

def caches(m, T):
    """Uma rodada limpa e uma corrompida, guardando o necessário para tudo o resto."""
    nomes = lambda x: (x.endswith("attn.hook_z") or x.endswith("hook_resid_pre") or x.endswith("hook_resid_mid")
                       or x.endswith("hook_mlp_out") or x == "hook_embed"
                       or x == f"blocks.{m.cfg.n_layers - 1}.hook_resid_post")
    lg, cl = m.run_with_cache(T["tk"], names_filter=nomes); ml = metrica(T, lg); del lg
    lg, cc = m.run_with_cache(T["tkc"], names_filter=nomes); mc = metrica(T, lg); del lg
    return dict(cl=cl, cc=cc, ml=ml, mc=mc, total=(ml - mc).mean().item())

def exato_cabecas(m, T, C=None, cabecas=None, bloco=None, bf16=False):
    """Rápida: começa na camada da cabeça (reaproveita as de baixo) e testa várias cabeças por rodada.
    cabecas=None testa todas; senão, só a lista (índices L*nH+H). Devolve dict índice -> efeito relativo."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; C = C or caches(m, T); n = T["tk"].shape[0]
    alvo = range(nL * nH) if cabecas is None else cabecas
    por_camada = defaultdict(list)
    for c in alvo: por_camada[int(c) // nH].append(int(c) % nH)
    bloco = bloco or BLOCO_EXATO or nH; res = {}
    with _ctx(bf16):
        for L, hs in sorted(por_camada.items()):
            zl = C["cl"][f"blocks.{L}.attn.hook_z"]; rp = C["cc"][f"blocks.{L}.hook_resid_pre"]
            for i in range(0, len(hs), bloco):
                parte = hs[i:i + bloco]; k = len(parte)
                def hk(z, hook, parte=parte):
                    for j, h in enumerate(parte): z[j * n:(j + 1) * n, :, h] = zl[:, :, h]
                    return z
                mk = metrica(T, logits_finais(m, rp.repeat(k, 1, 1), L, [(f"blocks.{L}.attn.hook_z", hk)]))
                for j, h in enumerate(parte):
                    res[L * nH + h] = (mk[j * n:(j + 1) * n] - C["mc"]).mean().item() / C["total"]
    return res

def exato_mlps(m, T, C=None, camadas=None, bf16=False):
    nL = m.cfg.n_layers; C = C or caches(m, T); res = {}
    with _ctx(bf16):
        for L in (range(nL) if camadas is None else camadas):
            mo = C["cl"][f"blocks.{L}.hook_mlp_out"]
            mk = metrica(T, logits_finais(m, C["cc"][f"blocks.{L}.hook_resid_pre"], L,
                                          [(f"blocks.{L}.hook_mlp_out", lambda x, hook, mo=mo: mo)]))
            res[L] = (mk - C["mc"]).mean().item() / C["total"]
    return res


# =====================================================================================
# 5. Gradiente (attribution patching) de cabeças e MLPs
# =====================================================================================


# =====================================================================================
# 6. Método econômico (validado): cabeças top 20 + checagem 21-30; MLPs regra adaptativa
# =====================================================================================
def nucleo_economico(m, T, C=None, top=20, checa=10, limiar=0.05, limiar_mlp=0.03, bf16=False):
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; C = C or caches(m, T)
    gcab, gmlp, nan = gradiente_nos(m, T, C)
    ordem = np.argsort(-np.abs(gcab)); exatos = 0
    conf = exato_cabecas(m, T, C, cabecas=ordem[top:top + checa], bf16=bf16); exatos += len(conf)
    cab = [int(c) for c in ordem[:top]] + [c for c, v in conf.items() if abs(v) >= limiar]
    # MLPs: exato nos 0..L/4; depois 4 diretos pelo gradiente e confere descendo até 2 falhas seguidas
    n_ini = max(1, nL // 4); ini = exato_mlps(m, T, C, range(n_ini), bf16=bf16); exatos += n_ini
    mlps = {L for L, v in ini.items() if abs(v) >= limiar_mlp}
    resto = [int(x) for x in np.arange(n_ini, nL)[np.argsort(-np.abs(gmlp[n_ini:]))]]
    mlps |= set(resto[:4]); falhas = 0
    for L in resto[4:]:
        v = exato_mlps(m, T, C, [L], bf16=bf16)[L]; exatos += 1
        if abs(v) >= limiar_mlp: mlps.add(L); falhas = 0
        else:
            falhas += 1
            if falhas == 2: break
    nota = gcab.copy()
    for c, v in conf.items(): nota[c] = v                  # onde houve exato, vale o exato
    return dict(cabecas=sorted(cab), mlps=sorted(mlps), nota_cabecas=nota, grad_mlps=gmlp,
                rodadas_exatas=exatos, nan=nan)


# =====================================================================================
# 7. Suficiência: circuito = conjunto de nós + todas as arestas entre eles (caminho rápido O(N))
# =====================================================================================
def fidelidade(m, T, C, cabecas, mlps):
    """Frase corrompida; cada nó do circuito recebe a sua entrada corrompida + a soma do que os nós
    anteriores DO CIRCUITO mudaram (limpo − corrompido). A saída final é montada do mesmo jeito."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; paralelo = bool(getattr(m.cfg, "parallel_attn_mlp", False))
    nkv = getattr(m.cfg, "n_key_value_heads", None) or nH; grupo = nH // nkv
    cc, cl = C["cc"], C["cl"]; WO = m.W_O
    por_camada = defaultdict(list)
    for c in cabecas: por_camada[int(c) // nH].append(int(c) % nH)
    mlps = set(int(x) for x in mlps)
    st = dict(acc=cl["hook_embed"] - cc["hook_embed"], foto={})
    hooks = [(f"blocks.{L}.hook_resid_pre", lambda x, hook, L=L: st["foto"].__setitem__(L, st["acc"]))
             for L in range(nL)]
    for L, hs in por_camada.items():
        ent = lambda L=L: cc[f"blocks.{L}.hook_resid_pre"] + st["foto"][L]
        def hq(x, hook, hs=hs, ent=ent):
            x = x.clone(); e = ent()
            for h in hs: x[:, :, h] = e
            return x
        gs = sorted({h // grupo for h in hs})
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
        base = f"blocks.{L}.hook_resid_pre" if paralelo else f"blocks.{L}.hook_resid_mid"
        hooks += [(f"blocks.{L}.hook_mlp_in",
                   lambda x, hook, L=L, base=base: cc[base] + (st["foto"][L] if paralelo else st["acc"])),
                  (f"blocks.{L}.hook_mlp_out",
                   lambda x, hook, L=L: st.__setitem__("acc", st["acc"] + x - cc[f"blocks.{L}.hook_mlp_out"]))]
    fim = f"blocks.{nL - 1}.hook_resid_post"
    hooks.append((fim, lambda x, hook: cc[fim] + st["acc"]))
    mk = metrica(T, logits_finais(m, T["tkc"], None, hooks))
    return (mk - C["mc"]).sum().item() / (C["ml"] - C["mc"]).sum().item()

def fidelidade_lotes(m, lotes, Cs, cabecas, mlps):
    num = den = 0.0
    for T, C in zip(lotes, Cs):
        f = fidelidade(m, T, C, cabecas, mlps); d = (C["ml"] - C["mc"]).sum().item()
        num += f * d; den += d
    return num / den

def circuito_por_alvo(m, lotes, Cs, ordem_cabecas, mlps, alvo=0.90, maximo=None):
    """Busca binária: menor número de cabeças (na ordem dada) que, com os MLPs, atinge a fidelidade-alvo."""
    ordem = [int(c) for c in ordem_cabecas][:maximo]; aval = {}
    def f(k):
        if k not in aval: aval[k] = fidelidade_lotes(m, lotes, Cs, ordem[:k], mlps)
        return aval[k]
    lo, hi = 0, len(ordem)
    if f(hi) < alvo: return dict(k=hi, fidelidade=f(hi), atingiu=False, avaliacoes=len(aval), curva=aval)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid) >= alvo: hi = mid
        else: lo = mid
    return dict(k=hi, fidelidade=f(hi), atingiu=True, avaliacoes=len(aval), curva=dict(sorted(aval.items())))


# =====================================================================================
# 8. Autoteste (rodar ao carregar um modelo novo)
# =====================================================================================
def autoteste(m, nome_tarefa="IOI", seed=0, n=10, verbose=True):
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; T = tarefa(m, nome_tarefa, seed, n); C = caches(m, T); ok = True
    msgs = []
    f_tudo = fidelidade(m, T, C, range(nL * nH), range(nL))
    ok1 = abs(f_tudo - 1) < 1e-3; ok &= ok1
    msgs.append(f"{'✅' if ok1 else '❌'} circuito = modelo inteiro: {f_tudo:.4f} (tem que ser 1)")
    f_nada = fidelidade(m, T, C, [], [])
    msgs.append(f"   circuito vazio: {f_nada:.3f} (deve ficar perto de 0, só o caminho direto do embedding)")
    gcab, gmlp, nan = gradiente_nos(m, T, C); ok2 = nan == 0; ok &= ok2
    msgs.append(f"{'✅' if ok2 else '❌'} gradiente sem NaN ({nan} NaN)")
    amostra = list(np.argsort(-np.abs(gcab))[:8]) + random.Random(0).sample(range(nL * nH), min(8, nL * nH))
    ex = exato_cabecas(m, T, C, cabecas=sorted(set(int(a) for a in amostra)))
    from scipy.stats import spearmanr
    ks = sorted(ex); rho = spearmanr([gcab[k] for k in ks], [ex[k] for k in ks])[0]
    ok3 = rho > 0.5; ok &= ok3
    msgs.append(f"{'✅' if ok3 else '⚠️'} gradiente × exato em {len(ks)} cabeças: Spearman {rho:.2f} (esperado > 0,5)")
    lento = exato_cabecas_lento(m, tarefa(m, nome_tarefa, seed, 4))[0] if nL * nH <= 64 else None
    if lento is not None:
        T4 = tarefa(m, nome_tarefa, seed, 4); rap = exato_cabecas(m, T4)
        dif = max(abs(lento[k] - rap[k]) for k in rap); ok4 = dif < 1e-3; ok &= ok4
        msgs.append(f"{'✅' if ok4 else '❌'} exato rápido = exato antigo (diferença máx {dif:.1e})")
    if verbose: print(f"autoteste {m.cfg.model_name} / {nome_tarefa}:"); [print("  " + x) for x in msgs]
    return ok


# =====================================================================================
# 9. Regressão: números de referência medidos em 31/10/2026 (RunPod A6000)
# =====================================================================================
REFERENCIA = {
    ("gpt2-small", "IOI"):  dict(nucleo=15, mlps=[0, 5, 8, 9],   fid_nucleo=0.45, fid_1pct=0.96),
    ("gpt2-small", "CAP"):  dict(nucleo=5,  mlps=[0, 9, 10, 11], fid_nucleo=0.45, fid_1pct=0.95),
    ("pythia-160m", "IOI"): dict(nucleo=12, mlps=[0, 2, 5, 9],   fid_nucleo=0.65, fid_1pct=0.96),
    ("pythia-160m", "CAP"): dict(nucleo=6,  mlps=[0, 9, 10, 11], fid_nucleo=0.60, fid_1pct=0.91),
}

def regressao(m, nome, nome_tarefa, sementes=(0, 1, 2), n=40, n_lote=10, bf16=False, verbose=True):
    """Refaz o núcleo exato (3 sementes) e a suficiência com o código NOVO e compara com a referência."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; t0 = time.time(); rels, exs = [], []
    for sd in sementes:
        T = tarefa(m, nome_tarefa, sd, n); C = caches(m, T)
        r = exato_cabecas(m, T, C, bf16=bf16); rels.append([r[i] for i in range(nL * nH)])
        e = exato_mlps(m, T, C, bf16=bf16); exs.append([e[i] for i in range(nL)])
        del C
    rel = np.abs(np.mean(rels, 0)); ex = np.abs(np.mean(exs, 0))
    lotes = [tarefa(m, nome_tarefa, 7000 + b, n_lote) for b in range(n // n_lote)]; Cs = [caches(m, T) for T in lotes]
    nuc = [int(i) for i in np.where(rel >= 0.05)[0]]; mlp5 = [int(i) for i in np.where(ex >= 0.05)[0]]
    f_nuc = fidelidade_lotes(m, lotes, Cs, nuc, mlp5)
    f_1 = fidelidade_lotes(m, lotes, Cs, [int(i) for i in np.where(rel >= 0.01)[0]], range(nL))
    ref = REFERENCIA.get((nome, nome_tarefa))
    out = dict(modelo=nome, tarefa=nome_tarefa, nucleo=len(nuc), mlps=mlp5, fid_nucleo=round(f_nuc, 2),
               fid_1pct=round(f_1, 2), segundos=round(time.time() - t0, 1))
    if ref:
        out["bate"] = (out["nucleo"] == ref["nucleo"] and out["mlps"] == ref["mlps"]
                       and abs(out["fid_nucleo"] - ref["fid_nucleo"]) <= 0.01 and abs(out["fid_1pct"] - ref["fid_1pct"]) <= 0.01)
        out["referência"] = ref
    if verbose: print(out)
    del Cs; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    return out


# =====================================================================================
# 10. Salvar com metadados + backup
# =====================================================================================
def salvar(arquivo, dados, **parametros):
    meta = dict(versao=VERSAO, data=time.strftime("%Y-%m-%d %H:%M:%S"), parametros=parametros,
                gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
                torch=torch.__version__, python=platform.python_version())
    json.dump(dict(meta=meta, dados=dados), open(arquivo, "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

def backup_zip(nome="backup_hh.zip", padrao="*.json"):
    import zipfile, glob
    with zipfile.ZipFile(nome, "w") as z:
        for a in glob.glob(padrao): z.write(a)
    print(f"{nome}: {len(glob.glob(padrao))} arquivos"); return nome
# =====================================================================================
# 11. Ordem das cabeças para o circuito (gradiente com 120 frases — validado em 01/11/2026)
# =====================================================================================
def ordem_cabecas(m, nome_tarefa, sementes=(0, 1, 2), n=40):
    """Ordena as cabeças pela média do gradiente em várias sementes (3 × 40 = 120 frases).
    Com cauda grande (GPT-2 large IOI) isso reduziu o circuito de 248 para 147 cabeças, em ~2,5 s."""
    notas = []
    for sd in sementes:
        T = tarefa(m, nome_tarefa, sd, n); C = caches(m, T)
        notas.append(gradiente_nos(m, T, C)[0]); del C
    nota = np.mean(notas, 0)
    return [int(c) for c in np.argsort(-np.abs(nota))], nota


# =====================================================================================
# 12. Arestas: circuito definido por conexões, notas por gradiente e poda (método híbrido)
# =====================================================================================
E_, O_ = ("E", 0), ("O", 0)

def _pos(no, nH, par):
    t, i = no
    if t == "E": return -1
    if t == "O": return 10**6
    if t == "H": return 2 * (i // nH)
    return 2 * i if par else 2 * i + 1

def arestas_entre(m, cabecas, mlps):
    """Todas as arestas válidas entre o embedding, os nós dados e a saída."""
    nH = m.cfg.n_heads; par = bool(getattr(m.cfg, "parallel_attn_mlp", False))
    nos = [("H", int(c)) for c in cabecas] + [("M", int(L)) for L in mlps]
    return [(u, v) for u in [E_] + nos for v in nos + [O_] if _pos(u, nH, par) < _pos(v, nH, par)]

def _grupo(m):
    nH = m.cfg.n_heads; nkv = getattr(m.cfg, "n_key_value_heads", None) or nH
    return nH // nkv

def _saida(m, cache, no):
    nH = m.cfg.n_heads; t, i = no
    if t == "E": return cache["hook_embed"]
    if t == "H": return cache[f"blocks.{i // nH}.attn.hook_z"][:, :, i % nH] @ m.W_O[i // nH, i % nH]
    return cache[f"blocks.{i}.hook_mlp_out"]

# Arestas (v0.11): aresta = (u, v, p), p = "q"/"k"/"v" se v é cabeça, "" se v é MLP ou a saída.
# Uma aresta agora é (u, v, p): p = "q", "k" ou "v" quando v é uma cabeça; p = "" quando v é um MLP ou a saída.

def arestas_partes(m, cabecas, mlps):
    out = []
    for u, v in arestas_entre(m, cabecas, mlps):
        if v[0] == "H": out += [(u, v, p) for p in "qkv"]
        else: out.append((u, v, ""))
    return out

# v0.11.3: nota k/v POR CABEÇA em GQA, sem desagrupar a entrada.
# Antes: o gradiente da entrada compartilhada do grupo (soma das cabeças) era dividido por `gr` -> as `gr` cabeças do grupo
# recebiam a MESMA nota (a média das notas do EAP-IG oficial). Agora: um deslocamento de zeros por cabeça é somado a k (e v)
# DEPOIS da repetição do grupo; seu gradiente é o gradiente por cabeça (d_head dimensões por posição). Ele é levado de volta
# à entrada do grupo com W_K/W_V do grupo (e, em k, desfazendo a rotação do RoPE), o que reproduz o EAP-IG oficial.
from transformer_lens.components import AbstractAttention as _Atencao, GroupedQueryAttention as _GQA
_CAP = {}   # camada -> {"k": deslocamento, "v": deslocamento}; vazio fora do cálculo das notas (comportamento normal)
_orig_scores = getattr(_GQA.calculate_attention_scores, "_hh_orig", _GQA.calculate_attention_scores)   # seguro contra reload
_orig_z = getattr(_GQA.calculate_z_scores, "_hh_orig", _GQA.calculate_z_scores)

def _camada_de(att): return int(att.hook_k.name.split(".")[1])

def _scores_cap(self, q, k):
    cap = _CAP.get(_camada_de(self)) if _CAP else None
    if cap is None or self.cfg.ungroup_grouped_query_attention: return _orig_scores(self, q, k)
    k = torch.repeat_interleave(k, dim=2, repeats=self.repeat_kv_heads)
    cap["k"] = torch.zeros_like(k).requires_grad_(True)
    return _Atencao.calculate_attention_scores(self, q, k + cap["k"])

def _z_cap(self, v, pattern):
    cap = _CAP.get(_camada_de(self)) if _CAP else None
    if cap is None or self.cfg.ungroup_grouped_query_attention: return _orig_z(self, v, pattern)
    v = torch.repeat_interleave(v, dim=2, repeats=self.repeat_kv_heads)
    cap["v"] = torch.zeros_like(v).requires_grad_(True)
    return _Atencao.calculate_z_scores(self, v + cap["v"], pattern)

_scores_cap._hh_orig, _z_cap._hh_orig = _orig_scores, _orig_z
_GQA.calculate_attention_scores, _GQA.calculate_z_scores = _scores_cap, _z_cap

def _rotary_T(att, g):
    # R^T g (por autograd): leva o gradiente de k DEPOIS do RoPE de volta para ANTES do RoPE (vale para qualquer variante de RoPE).
    with torch.enable_grad():
        z = torch.zeros_like(g).requires_grad_(True)
        return torch.autograd.grad(att.apply_rotary(z, 0, None), z, g)[0]

def _ln_vjp(bloco, x, u):
    # Leva o gradiente u (B,P,n,d_model) da ENTRADA DA PROJEÇÃO de volta para antes da normalização ln1 do bloco, que no
    # TransformerLens vem DEPOIS dos ganchos hook_k_input / hook_v_input. x = resid_pre (B,P,d_model). Por autograd, vale para LN e RMS.
    with torch.enable_grad():
        xh = x.detach().unsqueeze(2).expand(-1, -1, u.shape[2], -1).clone().requires_grad_(True)
        return torch.autograd.grad(bloco.ln1(xh), xh, u)[0]

def _por_cabeca(m):
    # True se dá para calcular a nota k/v por cabeça: GQA (gr > 1), sem QK-norm (a norma entre a projeção e a repetição quebraria a conta)
    # e com a normalização ANTES da atenção (Gemma 2/3 e OLMo 2/3 normalizam depois: ficam com a média do grupo, como na v0.11.2).
    return (_grupo(m) > 1 and not getattr(m.cfg, "use_qk_norm", False) and not m.cfg.ungroup_grouped_query_attention
            and not getattr(m.cfg, "use_normalization_before_and_after", False)
            and getattr(m.cfg, "original_architecture", "") not in ("Gemma2ForCausalLM", "Gemma3ForCausalLM", "Olmo2ForCausalLM", "Olmo3ForCausalLM"))

def gradiente_entradas(m, T, C, alvos, alfas=(0.0,), lote=None):
    # Gradiente da métrica na ENTRADA de cada (nó, parte), com o embedding interpolado corrompida -> limpa
    # (alfas=(0,) = gradiente simples na corrompida, como o EAP; vários alfas = média = EAP-IG).
    nH, nL = m.cfg.n_heads, m.cfg.n_layers; gr = _grupo(m); n = T["tk"].shape[0]
    capt = _por_cabeca(m)   # v0.11.3: nota k/v por cabeça em GQA (sem isso: a média do grupo, como na v0.11.2)
    por_camada = defaultdict(set); G = {}
    for (t, i), p in alvos:
        if t == "H": por_camada[i // nH].add(i % nH)
    try:
        lote = lote or LOTE_GRAD
        for a in range(0, n, lote):
            b = min(n, a + lote); Tf = _fatia(T, a, b)
            e_c = C["cc"]["hook_embed"][a:b]; e_l = C["cl"]["hook_embed"][a:b]; Gf = {}
            for al in alfas:
                torch.set_grad_enabled(True); m.reset_hooks()
                m.add_hook("hook_embed", lambda x, hook, al=al: (e_c + al * (e_l - e_c)).clone().requires_grad_(True))
                _CAP.clear()
                if capt:
                    _CAP.update({L: {} for L in por_camada})
                    for L in por_camada:
                        m.add_hook(f"blocks.{L}.hook_resid_pre", lambda x, hook, L=L: _CAP[L].__setitem__("x", x.detach()))
                for L, hs in por_camada.items():
                    for q in ("q" if capt else "qkv"):
                        def bk(g, hook, L=L, hs=hs, q=q):
                            for h in hs:
                                k = (("H", L * nH + h), q)
                                parte = g[:, :, h] if q == "q" else g[:, :, h // gr] / gr
                                Gf[k] = Gf.get(k, 0) + parte.detach() / len(alfas)
                        m.add_hook(f"blocks.{L}.hook_{q}_input", bk, "bwd")
                for (t, i), p in alvos:
                    if t == "M":
                        m.add_hook(f"blocks.{i}.hook_mlp_in", lambda g, hook, i=i: Gf.__setitem__((("M", i), ""), Gf.get((("M", i), ""), 0) + g.detach() / len(alfas)), "bwd")
                m.add_hook(f"blocks.{nL - 1}.hook_resid_post", lambda g, hook: Gf.__setitem__((O_, ""), Gf.get((O_, ""), 0) + g.detach() / len(alfas)), "bwd")
                metrica(Tf, _logits_ultima(m, Tf["tkc"])).sum().backward()
                if capt:
                    for L, hs in por_camada.items():
                        att = m.blocks[L].attn; hl = sorted(hs); idx = [h // gr for h in hl]
                        for q in "kv":
                            gq = _CAP[L][q].grad
                            if q == "k" and m.cfg.positional_embedding_type == "rotary": gq = _rotary_T(att, gq)
                            W = (att._W_K if q == "k" else att._W_V)[idx]
                            out = torch.einsum("bphd,hmd->bphm", gq[:, :, hl], W)
                            out = _ln_vjp(m.blocks[L], _CAP[L]["x"], out).detach() / len(alfas)
                            for j, h in enumerate(hl):
                                kk = (("H", L * nH + h), q); Gf[kk] = Gf.get(kk, 0) + out[:, :, j]
                            del gq, out
                    _CAP.clear()
                m.reset_hooks(); torch.set_grad_enabled(False)
            for k, v in Gf.items(): G.setdefault(k, []).append(v)
    finally:
        _CAP.clear(); m.reset_hooks(); torch.set_grad_enabled(False)
    return {k: torch.cat(v, 0) for k, v in G.items()}

LOTE_CALCULO = 40   # v0.11.4: máx. de frases por pedaço de cálculo das notas das arestas (0 ou None = não funde). Só junta lotes de MESMO comprimento.

def _fundir_lotes(lotes, Cs, alvo):
    # Junta (torch.cat) lotes de mesmo comprimento em pedaços de no máximo `alvo` frases (sempre pelo menos um lote). As frases e o sorteio são os mesmos; só muda o tamanho do pedaço
    # (a conta é a mesma: Spearman 1,0000 contra os lotes separados). Menos chamadas pequenas ao backward = bem mais rápido em GPU.
    por = defaultdict(list)
    for T, C in zip(lotes, Cs): por[T["tk"].shape[1]].append((T, C))
    def junta(g):
        if len(g) == 1: return g[0]
        n0 = g[0][0]["alvo"].shape[0]
        T = {k: (torch.cat([x[0][k] for x in g], 0) if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == n0 else v) for k, v in g[0][0].items()}
        cat = lambda d: {k: torch.cat([x[1][d][k] for x in g], 0) for k in g[0][1][d]}
        ml, mc = torch.cat([x[1]["ml"] for x in g]), torch.cat([x[1]["mc"] for x in g])
        return T, dict(cl=cat("cl"), cc=cat("cc"), ml=ml, mc=mc, total=(ml - mc).mean().item())
    L2, C2 = [], []
    for itens in por.values():
        cur, n = [], 0
        for it in itens:
            sz = it[0]["tk"].shape[0]
            if cur and n + sz > alvo: T, C = junta(cur); L2.append(T); C2.append(C); cur, n = [], 0   # v0.11.5: nunca passa do alvo
            cur.append(it); n += sz
        if cur: T, C = junta(cur); L2.append(T); C2.append(C)
    return L2, C2

def _tam_calculo(m, T, n_src, n_tgt):
    # Maior pedaço (em frases) que cabe: as notas guardam (fontes + destinos) x posições x d_model floats por frase; usa até 35% da memória livre da GPU.
    n0 = T["tk"].shape[0]
    if not LOTE_CALCULO: return n0
    if not torch.cuda.is_available(): return max(n0, LOTE_CALCULO)
    por_frase = (n_src + n_tgt) * T["tk"].shape[1] * m.cfg.d_model * 4
    return max(n0, min(LOTE_CALCULO, int(0.35 * torch.cuda.mem_get_info()[0] // max(por_frase, 1))))

def notas_arestas(m, lotes, Cs, arestas3, ig_passos=0):
    # Nota de cada (u, v, p) = Δsaída(u) · gradiente na entrada (v, p). ig_passos > 0 = EAP-IG.
    # v0.11.2: ESQUEMA_IG = "meio" (padrão: pontos do meio 0,1…0,9 = integral precisa, igual a 20 pontos)
    # ou "eapig" (0; 0,2; …; 0,8 = reproduz o EAP-IG oficial exatamente).
    # v0.11.4: lotes de mesmo comprimento são fundidos em pedaços de até LOTE_CALCULO frases (e o backward roda no pedaço todo); se faltar memória, o pedaço cai pela metade.
    if not ig_passos: alfas = (0.0,)
    elif ESQUEMA_IG == "eapig": alfas = tuple(k / ig_passos for k in range(ig_passos))
    else: alfas = tuple((k + 0.5) / ig_passos for k in range(ig_passos))
    srcs = sorted({a[0] for a in arestas3}); tgts = sorted({(a[1], a[2]) for a in arestas3})
    si = {u: k for k, u in enumerate(srcs)}; ti = {v: k for k, v in enumerate(tgts)}
    den = sum((C["ml"] - C["mc"]).sum().item() for C in Cs); n0 = lotes[0]["tk"].shape[0]
    alvo = _tam_calculo(m, lotes[0], len(srcs), len(tgts))
    while True:
        S = torch.zeros(len(srcs), len(tgts)); oom = False
        try:
            lt, ct = _fundir_lotes(lotes, Cs, alvo) if alvo > n0 else (lotes, Cs)
            for T, C in zip(lt, ct):
                G = gradiente_entradas(m, T, C, set(tgts), alfas, lote=max(LOTE_GRAD, T["tk"].shape[0]) if alvo > n0 else None)
                Dm = torch.stack([(_saida(m, C["cl"], u) - _saida(m, C["cc"], u)).flatten() for u in srcs])
                for j in range(0, len(tgts), 128):   # em blocos de 128 destinos: menos memória de pico, mesma conta
                    Gm = torch.stack([G.pop(v).flatten() for v in tgts[j:j + 128]])
                    S[:, j:j + 128] += (Dm @ Gm.T).cpu(); del Gm
                del Dm, G
            break
        except torch.cuda.OutOfMemoryError:
            oom = True
        if oom:   # v0.11.5: a limpeza fica FORA do except (dentro dele o traceback ainda segura a memória do pedaço que falhou)
            lt = ct = T = C = G = Dm = Gm = None
            m.reset_hooks(); torch.set_grad_enabled(False); gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if alvo <= n0: raise torch.cuda.OutOfMemoryError("sem memória mesmo com o menor pedaço (um lote)")
            alvo = max(n0, alvo // 2)
    nota = {a: S[si[a[0]], ti[(a[1], a[2])]].item() / den for a in arestas3}
    for a in nota:
        if a[0] == E_ and a[1] == ("M", 0): nota[a] = float("inf")
    return nota

def fidelidade_arestas(m, T, C, arestas3):
    # Entrada de cada (nó, parte) = entrada corrompida + A @ D (vetorizado): q, k e v recebem só as SUAS arestas.
    nH, nL = m.cfg.n_heads, m.cfg.n_layers; gr = _grupo(m); par = bool(getattr(m.cfg, "parallel_attn_mlp", False))
    fontes = sorted({u for u, _, _ in arestas3}, key=lambda u: (_pos(u, nH, par), u))
    linha = {u: i for i, u in enumerate(fontes)}; nS = len(fontes)
    ups = defaultdict(list)
    for u, v, p in arestas3: ups[(v, p)].append(linha[u])
    nos = {x for a in arestas3 for x in a[:2] if x[0] in "HM"}
    ref = C["cc"]["hook_embed"]; B, P, dm = ref.shape
    D = torch.zeros(nS, B, P, dm, device=ref.device, dtype=ref.dtype)
    if E_ in linha: D[linha[E_]] = C["cl"]["hook_embed"] - C["cc"]["hook_embed"]
    def matriz(listas):
        A = torch.zeros(len(listas), max(nS, 1), device=ref.device, dtype=ref.dtype)
        for r, ls in enumerate(listas):
            if ls: A[r, ls] = 1.0
        return A
    def soma(A):
        usadas = torch.nonzero(A.sum(0)).flatten()
        if len(usadas) == 0: return torch.zeros(A.shape[0], B, P, dm, device=ref.device, dtype=ref.dtype)
        n = int(usadas[-1]) + 1
        return (A[:, :n] @ D[:n].reshape(n, -1)).reshape(-1, B, P, dm)
    por_camada = defaultdict(list)
    for t, i in nos:
        if t == "H": por_camada[i // nH].append(i % nH)
    hooks = []
    for L, hs in por_camada.items():
        hs = sorted(hs); hs_t = torch.tensor(hs, device=ref.device)
        Aq = matriz([ups.get((("H", L * nH + h), "q"), []) for h in hs])
        grupos = sorted({h // gr for h in hs}); g_t = torch.tensor(grupos, device=ref.device)
        Ak = matriz([sorted({s for h in hs if h // gr == g for s in ups.get((("H", L * nH + h), "k"), [])}) for g in grupos])
        Av = matriz([sorted({s for h in hs if h // gr == g for s in ups.get((("H", L * nH + h), "v"), [])}) for g in grupos])
        lin_h = [linha.get(("H", L * nH + h)) for h in hs]
        def hq(x, hook, L=L, hs_t=hs_t, A=Aq):
            x = x.clone(); x[:, :, hs_t] = C["cc"][f"blocks.{L}.hook_resid_pre"][:, :, None, :] + soma(A).permute(1, 2, 0, 3); return x
        def hk(x, hook, L=L, g_t=g_t, A=Ak):
            x = x.clone(); x[:, :, g_t] = C["cc"][f"blocks.{L}.hook_resid_pre"][:, :, None, :] + soma(A).permute(1, 2, 0, 3); return x
        def hv(x, hook, L=L, g_t=g_t, A=Av):
            x = x.clone(); x[:, :, g_t] = C["cc"][f"blocks.{L}.hook_resid_pre"][:, :, None, :] + soma(A).permute(1, 2, 0, 3); return x
        def hz(z, hook, L=L, hs_t=hs_t, lin_h=lin_h):
            if all(l is None for l in lin_h): return
            W = m.W_O[L, hs_t]
            novo = torch.einsum("bpkh,khd->kbpd", z[:, :, hs_t], W)
            velho = torch.einsum("bpkh,khd->kbpd", C["cc"][f"blocks.{L}.attn.hook_z"][:, :, hs_t], W)
            for j, l in enumerate(lin_h):
                if l is not None: D[l] = novo[j] - velho[j]
        hooks += [(f"blocks.{L}.hook_q_input", hq), (f"blocks.{L}.hook_k_input", hk),
                  (f"blocks.{L}.hook_v_input", hv), (f"blocks.{L}.attn.hook_z", hz)]
    for t, i in nos:
        if t == "M":
            Am = matriz([ups.get((("M", i), ""), [])]); lm = linha.get(("M", i))
            bn = f"blocks.{i}.hook_resid_pre" if par else f"blocks.{i}.hook_resid_mid"
            hooks.append((f"blocks.{i}.hook_mlp_in", lambda x, hook, Am=Am, bn=bn: C["cc"][bn] + soma(Am)[0]))
            if lm is not None:
                def ho(x, hook, i=i, lm=lm): D[lm] = x - C["cc"][f"blocks.{i}.hook_mlp_out"]
                hooks.append((f"blocks.{i}.hook_mlp_out", ho))
    Ao = matriz([ups.get((O_, ""), [])])
    hooks.append((f"blocks.{nL - 1}.hook_resid_post", lambda x, hook: C["cc"][f"blocks.{nL - 1}.hook_resid_post"] + soma(Ao)[0]))
    mk = metrica(T, logits_finais(m, T["tkc"], None, hooks)); del D
    return (mk - C["mc"]).sum().item() / (C["ml"] - C["mc"]).sum().item()

def fidelidade_arestas_lotes(m, lotes, Cs, arestas3):
    num = den = 0.0
    for T, C in zip(lotes, Cs):
        d = (C["ml"] - C["mc"]).sum().item(); num += fidelidade_arestas(m, T, C, arestas3) * d; den += d
    return num / den

def aresta_exata(m, T, C, u, v, p):
    # Passa (limpo - corrompido) de u SÓ para a entrada (v, p). Devolve a soma nas frases.
    nH, nL = m.cfg.n_heads, m.cfg.n_layers; gr = _grupo(m); d = _saida(m, C["cl"], u) - _saida(m, C["cc"], u); t, i = v
    if t == "O":
        r = (C["cc"][f"blocks.{nL - 1}.hook_resid_post"] + d)[:, -1:, :]
        mk = metrica(T, m.unembed(m.ln_final(r) if m.cfg.normalization_type is not None else r))
    elif t == "H":
        L, h = i // nH, i % nH; idx = h if p == "q" else h // gr
        def hk(x, hook):
            x = x.clone(); x[:, :, idx] = x[:, :, idx] + d; return x
        mk = metrica(T, logits_finais(m, C["cc"][f"blocks.{L}.hook_resid_pre"], L, [(f"blocks.{L}.hook_{p}_input", hk)]))
    else:
        mk = metrica(T, logits_finais(m, C["cc"][f"blocks.{i}.hook_resid_pre"], i, [(f"blocks.{i}.hook_mlp_in", lambda x, hook: x + d)]))
    return (mk - C["mc"]).sum().item()

def busca_arestas(m, lotes, Cs, ordem, alvo):
    aval = {}
    def f(k):
        if k not in aval: aval[k] = fidelidade_arestas_lotes(m, lotes, Cs, ordem[:k])
        return aval[k]
    lo, hi = 0, len(ordem)
    if f(hi) < alvo: return hi, f(hi), False
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid) >= alvo: hi = mid
        else: lo = mid
    return hi, f(hi), True

def podar_arestas(m, lotes_nota, Cs_nota, lotes_busca, Cs_busca, cabecas, mlps, alvo=0.85, ig_passos=5):
    # Etapa das arestas (validada em 03/11/2026, ganhou do EAP-IG oficial): arestas entre as cabeças do circuito
    # e os MLPs, com q/k/v separados; nota EAP-IG; menor nº de arestas (na ordem da nota) que atinge `alvo`.
    ar = arestas_partes(m, cabecas, mlps); t0 = time.time()
    nota = notas_arestas(m, lotes_nota, Cs_nota, ar, ig_passos)
    ordem = sorted(ar, key=lambda a: -abs(nota[a])); k, f, ok = busca_arestas(m, lotes_busca, Cs_busca, ordem, alvo)
    return dict(k=k, total=len(ar), fidelidade=f, atingiu=ok, lista=ordem[:k], tempo=round(time.time() - t0, 1))


BLOCO_EXATO = None   # cabeças testadas juntas no exato (None = todas da camada; menor = menos memória)
LOTE_GRAD = 10   # frases por vez no backward (menor = menos memória, mesmo resultado)

def _fatia(T, a, b):
    n = T["alvo"].shape[0]
    return {k: (v[a:b] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == n else v) for k, v in T.items()}

def _logits_ultima(m, tk):
    """Só a última posição passa pelo unembed (o vocabulário do Qwen tem 152 mil palavras)."""
    r = m(tk, stop_at_layer=m.cfg.n_layers)[:, -1:, :]
    return m.unembed(m.ln_final(r) if m.cfg.normalization_type is not None else r)

def gradiente_nos(m, T, C=None):
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; C = C or caches(m, T); n = T["tk"].shape[0]
    f = lambda x: x.endswith("attn.hook_z") or x.endswith("hook_mlp_out")
    cab = np.zeros(nL * nH); mlp = np.zeros(nL)
    m.set_use_split_qkv_input(False)
    try:
        for a in range(0, n, LOTE_GRAD):
            b = min(n, a + LOTE_GRAD); Tf = _fatia(T, a, b); gr = {}
            torch.set_grad_enabled(True); m.reset_hooks()
            m.add_hook("hook_embed", lambda x, hook: x.requires_grad_(True))
            m.add_hook(f, lambda g, hook: gr.__setitem__(hook.name, g.detach()), "bwd")
            metrica(Tf, _logits_ultima(m, Tf["tkc"])).sum().backward()
            m.reset_hooks(); torch.set_grad_enabled(False)
            for L in range(nL):
                z = f"blocks.{L}.attn.hook_z"; o = f"blocks.{L}.hook_mlp_out"
                cab[L * nH:(L + 1) * nH] += (gr[z] * (C["cl"][z][a:b] - C["cc"][z][a:b])).sum((0, 1, 3)).cpu().numpy()
                mlp[L] += (gr[o] * (C["cl"][o][a:b] - C["cc"][o][a:b])).sum().item()
            del gr
    finally:
        m.reset_hooks(); torch.set_grad_enabled(False); m.set_use_split_qkv_input(True)
    nan = int(np.isnan(cab).sum() + np.isnan(mlp).sum())
    return cab / (n * C["total"]), mlp / (n * C["total"]), nan

# =====================================================================================
# 13. Análise completa (o HeadHunter de ponta a ponta)
# =====================================================================================
def promover_degraus(m, lotes, Cs, ordem, b, rodadas=10, salto_min=0.05, alvo=0.90):
    """Regra do degrau (validada em 01/11/2026): se a última cabeça do circuito faz a fidelidade saltar
    >= `salto_min`, ela é uma cabeça de interação que o gradiente subestimou. Vai para o começo da ordem e a
    busca é refeita só até o k anterior (o novo k nunca é maior). Para quando não há salto ou o k não diminui."""
    nL = m.cfg.n_layers; ordem = list(ordem); promovidas = []; hist = []
    for _ in range(rodadas):
        k = b["k"]
        if not b["atingiu"] or k < 2: break
        h = ordem[k - 1]
        antes = b["curva"].get(k - 1)
        if antes is None: antes = fidelidade_lotes(m, lotes, Cs, ordem[:k - 1], range(nL))
        salto = b["fidelidade"] - antes
        if salto < salto_min or h in promovidas: break
        nova = promovidas + [h] + [c for c in ordem if c not in promovidas and c != h]
        bn = circuito_por_alvo(m, lotes, Cs, nova, range(nL), alvo=alvo, maximo=k)
        if not bn["atingiu"] or bn["k"] >= k: break
        promovidas.append(h); ordem = nova; b = bn
        hist.append(dict(cabeca=int(h), salto=round(float(salto), 3), k=int(b["k"])))
    return ordem, b, promovidas, hist

def analisar(m, nome_tarefa, alvo=0.90, alvo_poda=0.85, podar=True, n=40, n_lote=10, sementes_ordem=(0, 1, 2),
             degrau=True, ganho_min=0.10, ordem="ig", passos_ig=5, verbose=True):
    """1) núcleo econômico (cabeças + MLPs); 2) ordem das cabeças pelo gradiente com 120 frases;
    3) circuito suficiente (cabeças + todos os MLPs) por busca binária até `alvo`;
    3b) [v0.3, degrau=True] promove as cabeças de interação (degraus) e escolhe o tamanho final em frases novas;
    4) poda das arestas do circuito até `alvo_poda` (método híbrido). degrau=False = comportamento da v0.2."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; t0 = time.time(); r = {}
    T0 = tarefa(m, nome_tarefa, 0, n); C0 = caches(m, T0)
    nuc = nucleo_economico(m, T0, C0); del C0
    r["nucleo"] = dict(cabecas=[c for c in nuc["cabecas"] if abs(nuc["nota_cabecas"][c]) >= 0.05], mlps=nuc["mlps"])
    t1 = time.time()
    if ordem == "ig": ordem, nota = ordem_cabecas_ig(m, nome_tarefa, sementes_ordem, n, passos=passos_ig)   # v0.5: padrão
    else: ordem, nota = ordem_cabecas(m, nome_tarefa, sementes_ordem, n)
    lot_a = [tarefa(m, nome_tarefa, 7000 + b, n_lote) for b in range(n // n_lote)]; Ca = [caches(m, T) for T in lot_a]
    b = circuito_por_alvo(m, lot_a, Ca, ordem, range(nL), alvo=alvo)
    k_sem = b["k"]; t2 = time.time(); promovidas, hist, revertido, k_prom = [], [], False, None
    if degrau:
        ordem_n, b_n = list(ordem), b
        ordem, b, promovidas, hist = promover_degraus(m, lot_a, Ca, ordem, b, alvo=alvo)
        if promovidas:   # tamanho final escolhido em frases que a promoção nunca viu
            lot_v = [tarefa(m, nome_tarefa, 9000 + b_, n_lote) for b_ in range(n // n_lote)]; Cv = [caches(m, T) for T in lot_v]
            b_v = circuito_por_alvo(m, lot_v, Cv, ordem, range(nL), alvo=alvo); del Cv
            if b_v["k"] > b["k"]: b = b_v   # v0.3.3: tamanho final = o MAIOR entre frases da busca e frases novas
            k_prom = b["k"]
            if (k_sem - b["k"]) / k_sem < ganho_min:   # v0.3.1: ganho pequeno -> volta ao circuito normal
                ordem, b, revertido = ordem_n, b_n, True
    t3 = time.time()
    r["circuito"] = dict(cabecas=ordem[:b["k"]], mlps=list(range(nL)), fidelidade=b["fidelidade"], atingiu=b["atingiu"],
                         arestas=len(arestas_entre(m, ordem[:b["k"]], range(nL))), k_sem_degrau=k_sem,
                         promovidas=[int(c) for c in promovidas], degraus=hist, revertido=revertido, k_promovido=k_prom)
    if podar:
        lot_p = [tarefa(m, nome_tarefa, 5000 + b_, n_lote) for b_ in range(n // n_lote)]; Cp = [caches(m, T) for T in lot_p]
        p = podar_arestas(m, lot_p, Cp, lot_a, Ca, r["circuito"]["cabecas"], range(nL), alvo_poda)
        r["arestas"] = {k: p[k] for k in ("k", "total", "fidelidade", "atingiu", "lista")}
        del Cp
    r["tempo"] = dict(nucleo=round(t1 - t0, 1), circuito=round(t2 - t1, 1), degrau=round(t3 - t2, 1),
                      arestas=round(time.time() - t3, 1), total=round(time.time() - t0, 1))
    if verbose:
        nome_c = lambda c: f"{c // nH}.{c % nH}"
        print(f"=== HeadHunter v{VERSAO}: {m.cfg.model_name} / {nome_tarefa} ===")
        print(f"núcleo (≥5%): {len(r['nucleo']['cabecas'])} cabeças {[nome_c(c) for c in r['nucleo']['cabecas'][:12]]}"
              f"{'...' if len(r['nucleo']['cabecas']) > 12 else ''} | MLPs {r['nucleo']['mlps']}")
        c = r["circuito"]
        if degrau and promovidas:
            print(f"degraus: {len(promovidas)} cabeça(s) de interação promovida(s) {[nome_c(x) for x in promovidas]} "
                  f"| k {k_sem} → {hist[-1]['k']} (frases da busca) → {k_prom} (final: o maior entre busca e frases novas)")
            if revertido:
                print(f"   ganho {(k_sem - k_prom) / k_sem:.0%} < {ganho_min:.0%}: volto ao circuito normal ({k_sem} cabeças) "
                      f"e só anoto as cabeças de interação")
        elif degrau:
            print("degraus: nenhum")
        print(f"circuito p/ {alvo:.0%}: {len(c['cabecas'])}/{nL * nH} cabeças + {nL} MLPs → fidelidade {c['fidelidade']:.0%}"
              f"{'' if c['atingiu'] else '  (NÃO atingiu o alvo)'}")
        if podar:
            a = r["arestas"]
            print(f"arestas p/ {alvo_poda:.0%}: {a['k']} de {a['total']} ({a['k'] / a['total']:.0%}) → fidelidade {a['fidelidade']:.0%}")
        print(f"tempo: {r['tempo']}")
    del Ca; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    return r


# =====================================================================================
# 14. Ruído da busca binária (v0.2.3): quanto o "k cabeças" muda com outras frases e outra ordem
# =====================================================================================
def ruido_busca(m, nome_tarefa, conjuntos=5, n=40, n_lote=10, alvo=0.90, verbose=True):
    """Separa duas fontes de ruído no tamanho do circuito:
    (a) frases de AVALIAÇÃO: mesma ordem, `conjuntos` grupos de frases novas;
    (b) ORDEM das cabeças: ordem A (sementes 0,1,2) × ordem B (sementes 10,11,12).
    Também varre a curva de fidelidade perto do corte (a fidelidade sobe de forma monótona?)
    e repete com o dobro de frases de avaliação."""
    t0 = time.time(); nL, nH = m.cfg.n_layers, m.cfg.n_heads; N = nL * nH
    ordens = {"A": ordem_cabecas(m, nome_tarefa, (0, 1, 2), n)[0], "B": ordem_cabecas(m, nome_tarefa, (10, 11, 12), n)[0]}
    linhas, curva = [], {}
    for j in range(conjuntos):
        lot = [tarefa(m, nome_tarefa, 7000 + 10 * j + b, n_lote) for b in range(n // n_lote)]; Cs = [caches(m, T) for T in lot]
        for nome, ordem in ordens.items():
            b = circuito_por_alvo(m, lot, Cs, ordem, range(nL), alvo=alvo)
            linhas.append(dict(conjunto=j, ordem=nome, k=b["k"], fid=round(b["fidelidade"], 3), atingiu=b["atingiu"]))
            if j == 0 and nome == "A":
                k0 = b["k"]; aval = dict(b["curva"])
                ks = sorted(set(np.linspace(0, min(N, 2 * k0), 21).astype(int).tolist()) | set(range(max(0, k0 - 8), k0 + 1)))
                for k in ks:
                    if k not in aval: aval[k] = fidelidade_lotes(m, lot, Cs, ordens["A"][:k], range(nL))
                curva = dict(sorted(aval.items()))
        del Cs; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    lot = [tarefa(m, nome_tarefa, 7500 + b, n_lote) for b in range(2 * n // n_lote)]; Cs = [caches(m, T) for T in lot]
    bg = circuito_por_alvo(m, lot, Cs, ordens["A"], range(nL), alvo=alvo); del Cs
    df = pd.DataFrame(linhas)
    resumo = {o: dict(media=float(g.k.mean()), dp=float(g.k.std(ddof=1)) if len(g) > 1 else 0.0, min=int(g.k.min()), max=int(g.k.max()))
              for o, g in df.groupby("ordem")}
    kA = int(df[(df.ordem == "A") & (df.conjunto == 0)].k.iloc[0]); kB = int(df[(df.ordem == "B") & (df.conjunto == 0)].k.iloc[0])
    sobrep = len(set(ordens["A"][:kA]) & set(ordens["B"][:kB])) / max(1, min(kA, kB))
    cv = list(curva.items()); quedas = [(k1, k2, round(f1 - f2, 3)) for (k1, f1), (k2, f2) in zip(cv, cv[1:]) if f2 < f1 - 0.02]
    primeiro = next((k for k, f in cv if f >= alvo), None)
    r = dict(tarefa=nome_tarefa if isinstance(nome_tarefa, str) else "custom", N=N, linhas=linhas, resumo=resumo,
             k_dobro_frases=bg["k"], fid_dobro=round(bg["fidelidade"], 3), sobreposicao_AB=round(sobrep, 2),
             curva={int(k): round(float(f), 3) for k, f in curva.items()}, quedas=quedas, primeiro_k_na_curva=primeiro,
             k_busca=kA, tempo=round(time.time() - t0, 1))
    if verbose:
        print(f"=== ruído da busca: {m.cfg.model_name} / {r['tarefa']} (alvo {alvo:.0%}, {N} cabeças) ===")
        for o, s in resumo.items():
            print(f"ordem {o}: k = {s['media']:.0f} ± {s['dp']:.0f}  (min {s['min']}, max {s['max']})  em {conjuntos} grupos de {n} frases")
        print(f"com {2 * n} frases: k = {bg['k']} (fid. {bg['fidelidade']:.0%}) | sobreposição das cabeças A×B: {sobrep:.0%}")
        print(f"curva: primeiro k com fid. ≥ alvo = {primeiro} (busca deu {kA}) | quedas > 2 pontos: {len(quedas)} {quedas[:4]}")
        perto = {k: f"{f:.2f}" for k, f in curva.items() if abs(k - kA) <= 8}
        print(f"perto do corte: {perto}")
        print(f"tempo: {r['tempo']} s")
    return r


# =====================================================================================
# 15. Tarefa COPIA: cópia de sequência (indução) — v0.3.2, Fase 4 parte 2 (tarefa nova)
# =====================================================================================
PALAVRAS_COPIA = [" apple", " river", " table", " house", " green", " music", " paper", " stone", " light", " water",
    " chair", " bread", " glass", " horse", " money", " night", " plant", " sugar", " train", " world",
    " dog", " cat", " sun", " moon", " car", " book", " door", " tree", " fish", " bird",
    " king", " road", " ship", " gold", " fire", " snow", " rain", " wind", " milk", " salt",
    " box", " cup", " hat", " key", " bed", " bag", " pen", " map", " egg", " ice",
    " city", " game", " film", " song", " food", " wall", " park", " farm", " bank", " card",
    " red", " blue", " black", " white", " happy", " small", " large", " quick", " cold", " warm",
    " doctor", " teacher", " window", " garden", " market", " island", " forest", " summer", " winter", " letter",
    " phone", " clock", " shirt", " shoe", " coat", " ring", " bell", " lamp", " rock", " sand"]
L_COPIA = 8
_palavras_ok = {}

def palavras_copia(m):
    chave = m.cfg.model_name
    if chave not in _palavras_ok:
        ok = [w for w in PALAVRAS_COPIA if m.to_tokens(w, prepend_bos=False).shape[1] == 1]
        _palavras_ok[chave] = sorted(set(m.to_single_token(w) for w in ok))
    return _palavras_ok[chave]

def montar_copia(m, seed, n=40, L=None):
    # Cópia de sequência (indução). Limpa: S + S[:L-1] -> resposta S[L-1].
    # Corrompida: igual, mas a última palavra da 1ª metade é trocada (b) -> o modelo copia b.
    # Só uma palavra muda, então o circuito é o que leva essa palavra até o fim.
    L = L or L_COPIA; ids = palavras_copia(m); rnd = random.Random(seed)
    bos = [m.tokenizer.bos_token_id] if bool(getattr(m.cfg, "default_prepend_bos", True)) and m.tokenizer.bos_token_id is not None else []
    tk, tkc, a, b = [], [], [], []
    for _ in range(n):
        S = rnd.sample(ids, L + 1); x, y = S[L - 1], S[L]
        Sc = S[:L - 1] + [y]
        tk.append(bos + S[:L] + S[:L - 1]); tkc.append(bos + Sc + S[:L - 1]); a.append(x); b.append(y)
    return dict(tk=torch.tensor(tk, device=DEV), tkc=torch.tensor(tkc, device=DEV),
                alvo=torch.tensor(a, device=DEV), errado=torch.tensor(b, device=DEV))

RECEITAS["COPIA"] = montar_copia


# =====================================================================================
# 16. EAP-IG: gradientes integrados para ordenar as cabeças (v0.5, teste)
# =====================================================================================
def gradiente_nos_ig(m, T, C=None, passos=5, alfas=None):
    """Como gradiente_nos, mas o gradiente é a MÉDIA ao longo do caminho corrompida -> limpa.
    O embedding da frase é interpolado: e = e_corr + a * (e_limpa - e_corr), com a em `alfas`
    (padrão: pontos do meio de `passos` fatias). Pega peças que só agem junto com outras (porta E),
    cujo gradiente na frase corrompida (a = 0) é quase zero. alfas=[0] reproduz gradiente_nos."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; C = C or caches(m, T); n = T["tk"].shape[0]
    alfas = list(alfas) if alfas is not None else [(k + 0.5) / passos for k in range(passos)]
    f = lambda x: x.endswith("attn.hook_z") or x.endswith("hook_mlp_out")
    cab = np.zeros(nL * nH); mlp = np.zeros(nL)
    m.set_use_split_qkv_input(False)
    try:
        for a in range(0, n, LOTE_GRAD):
            b = min(n, a + LOTE_GRAD); Tf = _fatia(T, a, b)
            e_c = C["cc"]["hook_embed"][a:b]; e_l = C["cl"]["hook_embed"][a:b]
            for al in alfas:
                gr = {}
                torch.set_grad_enabled(True); m.reset_hooks()
                m.add_hook("hook_embed", lambda x, hook, al=al: (e_c + al * (e_l - e_c)).clone().requires_grad_(True))
                m.add_hook(f, lambda g, hook: gr.__setitem__(hook.name, g.detach()), "bwd")
                metrica(Tf, _logits_ultima(m, Tf["tkc"])).sum().backward()
                m.reset_hooks(); torch.set_grad_enabled(False)
                for L in range(nL):
                    z = f"blocks.{L}.attn.hook_z"; o = f"blocks.{L}.hook_mlp_out"
                    cab[L * nH:(L + 1) * nH] += (gr[z] * (C["cl"][z][a:b] - C["cc"][z][a:b])).sum((0, 1, 3)).cpu().numpy() / len(alfas)
                    mlp[L] += (gr[o] * (C["cl"][o][a:b] - C["cc"][o][a:b])).sum().item() / len(alfas)
                del gr
    finally:
        m.reset_hooks(); torch.set_grad_enabled(False); m.set_use_split_qkv_input(True)
    nan = int(np.isnan(cab).sum() + np.isnan(mlp).sum())
    return cab / (n * C["total"]), mlp / (n * C["total"]), nan

def ordem_cabecas_ig(m, nome_tarefa, sementes=(0, 1, 2), n=40, passos=5):
    """Ordem das cabeças pela nota EAP-IG (média em várias sementes)."""
    notas = []
    for sd in sementes:
        T = tarefa(m, nome_tarefa, sd, n); C = caches(m, T)
        notas.append(gradiente_nos_ig(m, T, C, passos=passos)[0]); del C
    nota = np.mean(notas, 0)
    return [int(c) for c in np.argsort(-np.abs(nota))], nota


# =====================================================================================
# 17. Bateria de corrupções (v0.5): a mesma tarefa com perguntas diferentes -> papel de cada cabeça
# =====================================================================================
def montar_copia_emb(m, seed, n=40, L=None):
    # Corrompida: a 1ª metade tem as MESMAS palavras, em outra ordem. A palavra-resposta continua lá,
    # mas o "quem veio antes de quem" muda -> revela as cabeças de token anterior (ex.: 4.11 do GPT-2).
    # errado = a palavra que vem depois de S[L-2] na 1ª metade embaralhada (o que a indução copiaria).
    L = L or L_COPIA; ids = palavras_copia(m); rnd = random.Random(seed)
    bos = [m.tokenizer.bos_token_id] if bool(getattr(m.cfg, "default_prepend_bos", True)) and m.tokenizer.bos_token_id is not None else []
    tk, tkc, a, b = [], [], [], []
    while len(tk) < n:
        S = rnd.sample(ids, L); P = S[:]; rnd.shuffle(P)
        j = P.index(S[L - 2])
        if j == L - 1 or P[j + 1] == S[L - 1]: continue
        tk.append(bos + S + S[:L - 1]); tkc.append(bos + P + S[:L - 1]); a.append(S[L - 1]); b.append(P[j + 1])
    return dict(tk=torch.tensor(tk, device=DEV), tkc=torch.tensor(tkc, device=DEV),
                alvo=torch.tensor(a, device=DEV), errado=torch.tensor(b, device=DEV))

def montar_copia_nova(m, seed, n=40, L=None):
    # Corrompida: a 1ª metade é trocada INTEIRA por outras palavras -> não há repetição para copiar.
    # Revela as cabeças que percebem "isto já apareceu antes". errado = a palavra na mesma posição da nova 1ª metade.
    L = L or L_COPIA; ids = palavras_copia(m); rnd = random.Random(seed)
    bos = [m.tokenizer.bos_token_id] if bool(getattr(m.cfg, "default_prepend_bos", True)) and m.tokenizer.bos_token_id is not None else []
    tk, tkc, a, b = [], [], [], []
    for _ in range(n):
        S = rnd.sample(ids, 2 * L); S1, S2 = S[:L], S[L:]
        tk.append(bos + S1 + S1[:L - 1]); tkc.append(bos + S2 + S1[:L - 1]); a.append(S1[L - 1]); b.append(S2[L - 1])
    return dict(tk=torch.tensor(tk, device=DEV), tkc=torch.tensor(tkc, device=DEV),
                alvo=torch.tensor(a, device=DEV), errado=torch.tensor(b, device=DEV))

RECEITAS["COPIA_EMB"] = montar_copia_emb
RECEITAS["COPIA_NOVA"] = montar_copia_nova
PERGUNTAS = {"COPIA": "carrega a palavra", "COPIA_EMB": "marca a ordem", "COPIA_NOVA": "detecta repetição"}

def bateria(m, tarefas=("COPIA", "COPIA_EMB", "COPIA_NOVA"), podar=False, verbose=True, **kw):
    """Roda o HeadHunter em várias corrupções da mesma tarefa e junta: para cada cabeça, em quais
    perguntas ela está no núcleo (N) ou no circuito (c). Devolve a tabela e os resultados."""
    nH = m.cfg.n_heads; nome = lambda c: f"{c // nH}.{c % nH}"; res, papel = {}, defaultdict(dict)
    for t in tarefas:
        r = analisar(m, t, podar=podar, verbose=False, **kw); res[t] = r
        for c in r["circuito"]["cabecas"]: papel[nome(c)][t] = "c"
        for c in r["nucleo"]["cabecas"]: papel[nome(c)][t] = "N"
        if verbose:
            print(f"{t:11s} ({PERGUNTAS.get(t, t)}): núcleo {len(r['nucleo']['cabecas'])}, circuito {len(r['circuito']['cabecas'])}, "
                  f"fid {r['circuito']['fidelidade']:.0%}, {r['tempo']['total']} s")
    tab = pd.DataFrame({t: {h: papel[h].get(t, "") for h in papel} for t in tarefas}).fillna("")
    tab["nucleo_em"] = (tab[list(tarefas)] == "N").sum(1)
    tab = tab.sort_values(["nucleo_em"] + list(tarefas), ascending=False)
    if verbose:
        print("cabeças no núcleo de pelo menos uma pergunta (N = núcleo, c = só circuito):")
        print(tab[tab.nucleo_em > 0].drop(columns="nucleo_em").to_string())
    return tab, res


# =====================================================================================
# 18. Barra de erro (v0.5): o circuito como faixa, em vários grupos de frases e várias metas
# =====================================================================================
def faixa_circuito(m, nome_tarefa, grupos=5, metas=(0.80, 0.90, 0.95), n=40, n_lote=10, ordem="ig",
                   passos_ig=5, sementes_ordem=(0, 1, 2), verbose=True):
    """Para cada meta, o k mínimo em `grupos` grupos de frases novas (sementes 7000+, 10 em 10).
    Devolve mediana e faixa (mín-máx) por meta. Em modelos grandes, a faixa é o resultado honesto."""
    nL = m.cfg.n_layers; t0 = time.time()
    o = (ordem_cabecas_ig(m, nome_tarefa, sementes_ordem, n, passos=passos_ig)[0] if ordem == "ig"
         else ordem_cabecas(m, nome_tarefa, sementes_ordem, n)[0])
    ks = defaultdict(list)
    for j in range(grupos):
        lot = [tarefa(m, nome_tarefa, 7000 + 10 * j + b, n_lote) for b in range(n // n_lote)]; Cs = [caches(m, T) for T in lot]
        for meta in metas: ks[meta].append(circuito_por_alvo(m, lot, Cs, o, range(nL), alvo=meta)["k"])
        del Cs; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    r = {meta: dict(mediana=float(np.median(v)), min=int(min(v)), max=int(max(v)), todos=v) for meta, v in ks.items()}
    if verbose:
        print(f"=== faixa do circuito: {m.cfg.model_name} / {nome_tarefa} (ordem {ordem}, {grupos} grupos de {n} frases) ===")
        for meta, s in r.items():
            print(f"meta {meta:.0%}: k = {s['mediana']:.0f}  (faixa {s['min']}-{s['max']}, ±{(s['max'] - s['min']) / 2 / max(s['mediana'], 1):.0%})")
        print(f"tempo: {time.time() - t0:.1f} s")
    return dict(tarefa=nome_tarefa, ordem=ordem, metas=r, tempo=round(time.time() - t0, 1))


# =====================================================================================
# 19. Placar (v0.5): GPT-2 small IOI contra o gabarito de Wang et al. (2022) e outros métodos
# =====================================================================================
GABARITO_IOI_GPT2 = {
    "name mover": ["9.6", "9.9", "10.0"],
    "backup name mover": ["9.0", "9.7", "10.1", "10.2", "10.6", "10.10", "11.2", "11.9"],
    "negative name mover": ["10.7", "11.10"],
    "s-inhibition": ["7.3", "7.9", "8.6", "8.10"],
    "induction": ["5.5", "5.8", "5.9", "6.9"],
    "duplicate token": ["0.1", "0.10", "3.0"],
    "previous token": ["2.2", "4.11"]}

def _ablacao_media(m, T, n_lote=10):
    """Estilo Venkatesh: troca a saída de cada cabeça pela MÉDIA (no lote limpo) e mede a queda na
    probabilidade da resposta certa. Mede NECESSIDADE (o que quebra), não suficiência."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; ar = torch.arange(T["tk"].shape[0], device=T["tk"].device)
    f = lambda x: x.endswith("attn.hook_z")
    lg, cl = m.run_with_cache(T["tk"], names_filter=f)
    p0 = lg[:, -1].softmax(-1)[ar, T["alvo"]].mean().item(); del lg; queda = np.zeros(nL * nH)
    for L in range(nL):
        media = cl[f"blocks.{L}.attn.hook_z"].mean(0, keepdim=True)
        for h in range(nH):
            def hk(z, hook, h=h, media=media): z[:, :, h] = media[:, :, h]; return z
            lg = m.run_with_hooks(T["tk"], fwd_hooks=[(f"blocks.{L}.attn.hook_z", hk)])
            queda[L * nH + h] = p0 - lg[:, -1].softmax(-1)[ar, T["alvo"]].mean().item(); del lg
    return queda

def _acdc_nos(m, lotes, Cs, tau=0.01):
    """ACDC simplificado, por nós: começa com todas as cabeças e, da última camada para a primeira,
    remove cada cabeça cuja retirada MUDA a fidelidade em menos de `tau` (para cima ou para baixo). Conta as avaliações."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; atual = set(range(nL * nH)); eval_ = 0
    f_atual = fidelidade_lotes(m, lotes, Cs, sorted(atual), range(nL)); eval_ += 1
    for c in sorted(atual, reverse=True):
        tent = sorted(atual - {c}); f = fidelidade_lotes(m, lotes, Cs, tent, range(nL)); eval_ += 1
        if abs(f_atual - f) < tau: atual.discard(c); f_atual = f   # v0.5.1: |mudança| (como o ACDC original), não só queda
    return sorted(atual), f_atual, eval_

def placar_ioi_gpt2(m, n=40, n_lote=10, tau_acdc=0.01, verbose=True):
    """Compara métodos no GPT-2 small IOI. Para cada um: acerto contra o gabarito (precisão e recall),
    fidelidade do circuito (cabeças + todos os MLPs, em frases novas) e custo (tempo).
    Os métodos de ranking pegam as top-k cabeças, com k = tamanho do gabarito (26)."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; nome = lambda c: f"{c // nH}.{c % nH}"
    idx = lambda s: int(s.split(".")[0]) * nH + int(s.split(".")[1])
    gab = {idx(h) for hs in GABARITO_IOI_GPT2.values() for h in hs}; K = len(gab)
    lot_a = [tarefa(m, "IOI", 7000 + b, n_lote) for b in range(n // n_lote)]; Ca = [caches(m, T) for T in lot_a]
    lot_v = [tarefa(m, "IOI", 9000 + b, n_lote) for b in range(n // n_lote)]; Cv = [caches(m, T) for T in lot_v]
    T0 = tarefa(m, "IOI", 0, n); C0 = caches(m, T0); linhas = {}
    def registra(nome_m, cab, seg, extra=""):
        cab = set(int(c) for c in cab); acerto = len(cab & gab)
        linhas[nome_m] = dict(cabecas=len(cab), acertos=acerto, precisao=acerto / max(len(cab), 1), recall=acerto / K,
                              fid_novas=fidelidade_lotes(m, lot_v, Cv, sorted(cab), range(nL)), segundos=round(seg, 1),
                              obs=extra, faltaram=sorted(nome(c) for c in gab - cab))
    t = time.time(); r = analisar(m, "IOI", podar=False, verbose=False); registra("HeadHunter (circuito 90%)", r["circuito"]["cabecas"], time.time() - t)
    registra("HeadHunter (núcleo)", r["nucleo"]["cabecas"], r["tempo"]["nucleo"])
    t = time.time(); o, _ = ordem_cabecas(m, "IOI", (0, 1, 2), n); registra("EAP (gradiente), top-26", o[:K], time.time() - t)
    t = time.time(); o, _ = ordem_cabecas_ig(m, "IOI", (0, 1, 2), n); registra("EAP-IG, top-26", o[:K], time.time() - t)
    t = time.time(); ex = exato_cabecas(m, T0, C0); o = sorted(ex, key=lambda c: -abs(ex[c]))
    registra("Patching exato (1 por vez), top-26", o[:K], time.time() - t)
    t = time.time(); q = _ablacao_media(m, T0); o = list(np.argsort(-np.abs(q)))
    registra("Ablação pela média (Venkatesh), top-26", o[:K], time.time() - t, "mede necessidade")
    t = time.time(); cab, f, ev = _acdc_nos(m, lot_a, Ca, tau_acdc)
    registra(f"ACDC simplificado (nós, tau={tau_acdc})", cab, time.time() - t, f"{ev} avaliações")
    del Ca, Cv, C0; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    df = pd.DataFrame(linhas).T
    if verbose:
        print(f"=== placar GPT-2 small IOI | gabarito Wang et al.: {K} cabeças ===")
        print(df[["cabecas", "acertos", "precisao", "recall", "fid_novas", "segundos", "obs"]].to_string(
            formatters={"precisao": "{:.0%}".format, "recall": "{:.0%}".format, "fid_novas": "{:.0%}".format}))
    return df


# =====================================================================================
# 20. Poda dos MLPs (v0.6, teste): o circuito deixa de levar TODOS os MLPs
# =====================================================================================
def notas_ig(m, nome_tarefa, sementes=(0, 1, 2), n=40, passos=5):
    """Notas EAP-IG de cabeças E MLPs (média em várias sementes)."""
    cs, ms = [], []
    for sd in sementes:
        T = tarefa(m, nome_tarefa, sd, n); C = caches(m, T)
        c, mm, _ = gradiente_nos_ig(m, T, C, passos=passos); cs.append(c); ms.append(mm); del C
    return np.mean(cs, 0), np.mean(ms, 0)

def circuito_com_mlps(m, nome_tarefa, alvo=0.90, alvo_mlp=0.97, n=40, n_lote=10, verbose=True):
    """Duas etapas, mesma técnica do resto do HeadHunter (ordem EAP-IG + busca binária):
    1) com TODAS as cabeças, o menor nº de MLPs (na ordem EAP-IG) que atinge `alvo_mlp`;
    2) com esses MLPs, o menor nº de cabeças que atinge `alvo`. Compara com o circuito de todos os MLPs."""
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; t0 = time.time()
    nc, nm = notas_ig(m, nome_tarefa, n=n)
    oc = [int(c) for c in np.argsort(-np.abs(nc))]; om = [int(L) for L in np.argsort(-np.abs(nm))]
    lot = [tarefa(m, nome_tarefa, 7000 + b, n_lote) for b in range(n // n_lote)]; Cs = [caches(m, T) for T in lot]
    todas = list(range(nL * nH)); aval = {}
    def fm(k):
        if k not in aval: aval[k] = fidelidade_lotes(m, lot, Cs, todas, om[:k])
        return aval[k]
    lo, hi = 0, nL
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fm(mid) >= alvo_mlp: hi = mid
        else: lo = mid
    mlps = sorted(om[:hi])
    b_todos = circuito_por_alvo(m, lot, Cs, oc, range(nL), alvo=alvo)
    b_pod = circuito_por_alvo(m, lot, Cs, oc, mlps, alvo=alvo); f_mlp = fm(hi); del Cs
    lv = [tarefa(m, nome_tarefa, 9000 + b, n_lote) for b in range(n // n_lote)]; Cv = [caches(m, T) for T in lv]
    fv_todos = fidelidade_lotes(m, lv, Cv, oc[:b_todos["k"]], range(nL))
    fv_pod = fidelidade_lotes(m, lv, Cv, oc[:b_pod["k"]], mlps); del Cv
    gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    r = dict(tarefa=nome_tarefa, mlps=mlps, n_mlps=len(mlps), fid_mlps_todas_cabecas=f_mlp,
             cabecas_todos_mlps=b_todos["k"], cabecas_mlps_podados=b_pod["k"], atingiu=b_pod["atingiu"],
             fid_novas_todos=fv_todos, fid_novas_podado=fv_pod, ordem_mlps=om, tempo=round(time.time() - t0, 1))
    if verbose:
        print(f"=== poda dos MLPs: {m.cfg.model_name} / {nome_tarefa} ===")
        print(f"MLPs: {len(mlps)}/{nL} bastam ({f_mlp:.0%} com todas as cabeças): {mlps}")
        print(f"circuito com TODOS os MLPs: {b_todos['k']} cabeças + {nL} MLPs → frases novas {fv_todos:.0%}")
        print(f"circuito com MLPs podados:  {b_pod['k']} cabeças + {len(mlps)} MLPs → frases novas {fv_pod:.0%}"
              f"{'' if b_pod['atingiu'] else '  (NÃO atingiu a meta)'}")
        print(f"tempo: {r['tempo']} s")
    return r


# =====================================================================================
# 21. Features (v0.6, teste): quantas features de SAE carregam a tarefa em cada camada?
# =====================================================================================
class SAESimples:
    """SAE no formato padrão (W_enc, b_enc, W_dec, b_dec): f = ReLU((x - b_dec) @ W_enc + b_enc)."""
    def __init__(self, pesos, dev):
        self.W_enc = pesos["W_enc"].to(dev).float(); self.b_enc = pesos["b_enc"].to(dev).float()
        self.W_dec = pesos["W_dec"].to(dev).float(); self.b_dec = pesos["b_dec"].to(dev).float()
    def encode(self, x): return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

def carregar_sae_gpt2(L, repo="jbloom/GPT2-Small-SAEs-Reformatted"):
    """SAE do residual (antes do bloco L) do GPT-2 small, de Joseph Bloom (os mesmos do Neuronpedia, 'res-jb')."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.torch import load_file
    arqs = [a for a in list_repo_files(repo) if a.startswith(f"blocks.{L}.hook_resid_pre/") and a.endswith(".safetensors")]
    if not arqs: raise FileNotFoundError(f"não achei a SAE da camada {L} em {repo}")
    return SAESimples(load_file(hf_hub_download(repo, arqs[0])), DEV)

def features_por_camada(m, nome_tarefa, camadas=None, alvo=0.90, n=40, n_lote=10, top=5, verbose=True):
    """Para cada camada L: no residual antes do bloco L, a frase corrompida recebe
    resid_corr + soma_{i no circuito} (f_i limpa - f_i corrompida) * d_i.
    Ordem das features: atribuição (gradiente da métrica no residual) · (Δf_i d_i). Busca binária no nº de
    features até `alvo`. 'teto' = fidelidade com TODAS as features (o que a SAE não reconstrói fica de fora)."""
    nL = m.cfg.n_layers; camadas = range(nL) if camadas is None else camadas; linhas = []
    lot = [tarefa(m, nome_tarefa, 7000 + b, n_lote) for b in range(n // n_lote)]
    for L in camadas:
        t0 = time.time(); sae = carregar_sae_gpt2(L); nome_h = f"blocks.{L}.hook_resid_pre"
        dados = []
        for T in lot:
            _, cl = m.run_with_cache(T["tk"], names_filter=nome_h); _, cc = m.run_with_cache(T["tkc"], names_filter=nome_h)
            xl, xc = cl[nome_h], cc[nome_h]; del cl, cc
            ml = metrica(T, _logits_ultima(m, T["tk"])); mc = metrica(T, _logits_ultima(m, T["tkc"]))
            g = {}
            torch.set_grad_enabled(True); m.reset_hooks()
            m.add_hook(nome_h, lambda x, hook: x.requires_grad_(True) if not x.requires_grad else x)
            m.add_hook(nome_h, lambda gr, hook: g.__setitem__("g", gr.detach()), "bwd")
            m.add_hook("hook_embed", lambda x, hook: x.requires_grad_(True))
            metrica(T, _logits_ultima(m, T["tkc"])).sum().backward()
            m.reset_hooks(); torch.set_grad_enabled(False)
            df = sae.encode(xl) - sae.encode(xc)                                 # (B, P, n_feat)
            atr = (df * (g["g"] @ sae.W_dec.T)).sum((0, 1))                       # (n_feat,)
            dados.append(dict(T=T, xc=xc, df=df, atr=atr, ml=ml, mc=mc))
        atr = sum(d["atr"] for d in dados); ordem = torch.argsort(-atr.abs())
        ativas = int((atr != 0).sum()); den = sum((d["ml"] - d["mc"]).sum().item() for d in dados)
        def fid(k):
            sel = ordem[:k]; num = 0.0
            for d in dados:
                x = d["xc"] + d["df"][:, :, sel] @ sae.W_dec[sel]
                mk = metrica(d["T"], logits_finais(m, x, L, ()))
                num += (mk - d["mc"]).sum().item()
            return num / den
        teto = fid(ativas); aval = {}
        def f(k):
            if k not in aval: aval[k] = fid(k)
            return aval[k]
        lo, hi = 0, ativas
        if teto >= alvo:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if f(mid) >= alvo: hi = mid
                else: lo = mid
        k = hi if teto >= alvo else None
        tops = [int(i) for i in ordem[:top]]
        linhas.append(dict(camada=L, features_ativas=ativas, k_alvo=k, teto=round(teto, 3), top=tops,
                           links=[f"https://www.neuronpedia.org/gpt2-small/{L}-res-jb/{i}" for i in tops],
                           segundos=round(time.time() - t0, 1)))
        if verbose:
            print(f"camada {L:2d}: {ativas:5d} features mudam | teto {teto:.0%} | "
                  f"{'k = ' + str(k) if k else 'não atinge ' + format(alvo, '.0%')} | top {tops} | {time.time() - t0:.0f} s")
        del dados, sae; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    return pd.DataFrame(linhas)


# =====================================================================================
# 22. Necessidade (estilo Venkatesh 2026) lado a lado com a suficiência (v0.6.1)
# =====================================================================================
def _prob_com_ablacao(m, T, medias, cabecas):
    # Frase LIMPA; as cabeças dadas recebem a média (no lote limpo) da sua saída. Devolve a prob. média da resposta certa.
    nH = m.cfg.n_heads; por = defaultdict(list)
    for c in cabecas: por[int(c) // nH].append(int(c) % nH)
    hooks = []
    for L, hs in por.items():
        def hk(z, hook, L=L, hs=hs):
            z[:, :, hs] = medias[L][:, :, hs]; return z
        hooks.append((f"blocks.{L}.attn.hook_z", hk))
    ar = torch.arange(T["tk"].shape[0], device=T["tk"].device)
    lg = logits_finais(m, T["tk"], None, hooks)
    return lg[:, -1].softmax(-1)[ar, T["alvo"]].mean().item()

def necessidade(m, nome_tarefa, n=40, dano_alvo=0.80, verbose=True):
    # Quantas cabeças precisam SAIR (ablação pela média) para causar `dano_alvo` do dano máximo?
    # dano(k) = (p0 - p_k) / (p0 - p_todas), com as cabeças ordenadas pela queda individual de probabilidade.
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; nome = lambda c: f"{c // nH}.{c % nH}"; t0 = time.time()
    T = tarefa(m, nome_tarefa, 0, n)
    _, cl = m.run_with_cache(T["tk"], names_filter=lambda x: x.endswith("attn.hook_z"))
    medias = {L: cl[f"blocks.{L}.attn.hook_z"].mean(0, keepdim=True) for L in range(nL)}; del cl
    p0 = _prob_com_ablacao(m, T, medias, []); p_todas = _prob_com_ablacao(m, T, medias, range(nL * nH))
    queda = np.array([p0 - _prob_com_ablacao(m, T, medias, [c]) for c in range(nL * nH)])
    ordem = [int(c) for c in np.argsort(-queda)]; aval = {}
    def dano(k):
        if k not in aval: aval[k] = (p0 - _prob_com_ablacao(m, T, medias, ordem[:k])) / max(p0 - p_todas, 1e-9)
        return aval[k]
    hi = next((k for k in range(1, nL * nH + 1) if dano(k) >= dano_alvo), nL * nH)   # varredura: o dano não é monótono
    r = dict(tarefa=nome_tarefa, p0=p0, p_todas=p_todas, k_dano=hi, dano=dano(hi), top=[nome(c) for c in ordem[:10]],
             dano_top1=dano(1), tempo=round(time.time() - t0, 1))
    if verbose:
        print(f"necessidade ({m.cfg.model_name} / {nome_tarefa}): p0 = {p0:.2f}, todas ablacionadas = {p_todas:.2f} | "
              f"{hi} cabeça(s) causam {dano(hi):.0%} do dano (a 1ª sozinha: {dano(1):.0%}) | top {r['top'][:5]} | {r['tempo']} s")
    return r

def necessidade_x_suficiencia(m, nome_tarefa, **kw):
    # Os dois lados da mesma pergunta: quantas cabeças QUEBRAM a tarefa (necessidade) e quantas BASTAM (suficiência).
    nH = m.cfg.n_heads; nome = lambda c: f"{c // nH}.{c % nH}"
    nec = necessidade(m, nome_tarefa)
    suf = analisar(m, nome_tarefa, podar=False, verbose=False, **kw)
    nuc = [nome(c) for c in suf["nucleo"]["cabecas"]]
    comum = len(set(nec["top"][:nec["k_dano"]]) & set(nuc))
    print(f">>> {m.cfg.model_name} / {nome_tarefa}: NECESSIDADE {nec['k_dano']} cabeça(s) p/ 80% do dano | "
          f"NÚCLEO {len(nuc)} | SUFICIÊNCIA {len(suf['circuito']['cabecas'])}/{m.cfg.n_layers * nH} p/ 90% "
          f"| top-necessidade no núcleo: {comum}/{min(nec['k_dano'], 10)}")
    return dict(necessidade=nec, nucleo=nuc, k_suficiencia=len(suf["circuito"]["cabecas"]))


# =====================================================================================
# 23. Validação A (v0.7): o circuito está certo? fidelidade, minimalidade, controle aleatório, arestas GQA
# =====================================================================================
def validar_circuito(m, nome_tarefa, r=None, grupos=3, n=40, n_lote=10, sorteios=5, eps=0.005,
                     amostra_arestas=60, verbose=True, **kw):
    # r = saída de analisar (se None, roda). Testa, em frases NOVAS (sementes 9500+):
    # 1) fidelidade do circuito em `grupos` grupos; 2) minimalidade: quanto a fidelidade cai ao tirar cada cabeça;
    # 3) controle: circuitos aleatórios do mesmo tamanho; 4) arestas: EAP x exato numa amostra (com GQA = aproximação).
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; t0 = time.time()
    r = r or analisar(m, nome_tarefa, podar=False, verbose=False, **kw)
    cab = [int(c) for c in r["circuito"]["cabecas"]]; mlps = list(r["circuito"]["mlps"]); k = len(cab)
    fids = []
    for g in range(grupos):
        lot = [tarefa(m, nome_tarefa, 9500 + 10 * g + b, n_lote) for b in range(n // n_lote)]; Cs = [caches(m, T) for T in lot]
        fids.append(fidelidade_lotes(m, lot, Cs, cab, mlps))
        if g == 0: lot0, C0 = lot, Cs
        else: del Cs
    base = fids[0]
    quedas = np.array([base - fidelidade_lotes(m, lot0, C0, [x for x in cab if x != c], mlps) for c in cab])
    rnd = np.random.default_rng(0)
    aleat = [fidelidade_lotes(m, lot0, C0, sorted(rnd.choice(nL * nH, k, replace=False).tolist()), mlps) for _ in range(sorteios)]
    so_mlps = fidelidade_lotes(m, lot0, C0, [], mlps)   # nenhuma cabeça, só os MLPs: a linha de base do controle
    # arestas: EAP x exato numa amostra das arestas que entram em cabeças do circuito
    ar = [a for a in arestas_partes(m, cab, mlps) if a[1][0] == "H"]
    rs = random.Random(0); amostra = rs.sample(ar, min(amostra_arestas, len(ar)))
    T1, Ca1 = lot0[0], C0[0]
    eap = notas_arestas(m, [T1], [Ca1], amostra)
    den = (Ca1["ml"] - Ca1["mc"]).sum().item()
    exato = {a: aresta_exata(m, T1, Ca1, *a) / den for a in amostra}
    from scipy.stats import spearmanr
    rho = float(spearmanr([eap[a] for a in amostra], [exato[a] for a in amostra])[0])
    del C0; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    res = dict(tarefa=nome_tarefa, k=k, fid_grupos=[round(f, 3) for f in fids], fid_min=round(min(fids), 3),
               queda_media=float(quedas.mean()), queda_mediana=float(np.median(quedas)),
               sobrando=int((quedas < eps).sum()), sobrando_frac=float((quedas < eps).mean()),
               aleatorio_medio=float(np.mean(aleat)), aleatorio_max=float(np.max(aleat)), so_mlps=float(so_mlps),
               arestas_spearman=rho, gqa=adaptador(m)["gqa"], tempo=round(time.time() - t0, 1))
    if verbose:
        print(f"=== validação A: {m.cfg.model_name} / {nome_tarefa} (circuito de {k} cabeças + {len(mlps)} MLPs) ===")
        print(f"1) fidelidade em {grupos} grupos de frases novas: {[f'{f:.0%}' for f in fids]} (mín {min(fids):.0%})")
        print(f"2) minimalidade: tirar 1 cabeça baixa a fidelidade em média {quedas.mean():.1%} (mediana {np.median(quedas):.1%}); "
              f"{res['sobrando']}/{k} cabeças quase não fazem falta (< {eps:.1%})")
        print(f"3) controle: {sorteios} circuitos aleatórios de {k} cabeças dão em média {np.mean(aleat):.0%} (máx {np.max(aleat):.0%}); "
              f"só os MLPs, sem cabeças: {so_mlps:.0%}")
        print(f"4) arestas que entram em cabeças: EAP x exato, Spearman {rho:.2f} em {len(amostra)} arestas"
              f"{' (GQA: k/v aproximadas)' if res['gqa'] else ''}")
        print(f"tempo: {res['tempo']} s")
    return res


# =====================================================================================
# 24. Generalização (v0.8): o circuito achado num molde de frase funciona em outros moldes?
# =====================================================================================
MODELOS_IOI_ALT = ["After{A} and{B} had lunch at the cafe,{B} handed a menu to",
                   "While{A} and{B} were walking in the park,{B} threw the ball to",
                   "Then,{A} and{B} were working at the office.{B} decided to give a book to"]

def montar_ioi_alt(m, seed, n=40):
    # Mesma tarefa (IOI), outros moldes. Um molde por lote (todas as frases com o mesmo comprimento).
    nomes_ok = [x for x in NOMES if m.to_tokens(x, prepend_bos=False).shape[1] == 1]
    rnd = random.Random(seed); molde = MODELOS_IOI_ALT[seed % len(MODELOS_IOI_ALT)]; fr, fc, io, s = [], [], [], []
    for _ in range(n):
        A, B = rnd.sample(nomes_ok, 2)
        if rnd.random() < 0.5: f = molde.format(A=A, B=B)
        else: f = molde.replace("{A} and{B}", "{B} and{A}").format(A=A, B=B)
        i = f.rindex(B); fr.append(f); fc.append(f[:i] + A + f[i + len(B):])   # só a ÚLTIMA ocorrência de B vira A
        io.append(m.to_single_token(A)); s.append(m.to_single_token(B))
    return dict(tk=tok(m, fr), tkc=tok(m, fc), alvo=torch.tensor(io, device=DEV), errado=torch.tensor(s, device=DEV))

MOLDES_CAP_ALT = ["In{P}, the capital city is", "Q: What is the capital of{P}? A: The capital is"]

def montar_capitais_alt(m, seed, n=40):
    bons = paises_conhecidos(m); rnd = random.Random(seed); molde = MOLDES_CAP_ALT[seed % len(MOLDES_CAP_ALT)]
    fr, fc, cap, capc = [], [], [], []
    for _ in range(n):
        (p, c), (p2, c2) = rnd.sample(bons, 2)
        fr.append(molde.format(P=" " + p)); fc.append(molde.format(P=" " + p2))
        cap.append(m.to_single_token(" " + c)); capc.append(m.to_single_token(" " + c2))
    return dict(tk=tok(m, fr), tkc=tok(m, fc), alvo=torch.tensor(cap, device=DEV), errado=torch.tensor(capc, device=DEV))

def montar_copia_alt(m, seed, n=40):
    return montar_copia(m, seed, n, L=6)   # mesma tarefa, sequência mais curta (6 palavras em vez de 8)

RECEITAS["IOI_ALT"] = montar_ioi_alt
RECEITAS["CAP_ALT"] = montar_capitais_alt
RECEITAS["COPIA_ALT"] = montar_copia_alt

def acerto(m, nome_tarefa, n=40, seed=0):
    # Fração das frases limpas em que o modelo põe a resposta certa em 1º lugar (o modelo sabe fazer a tarefa?).
    T = tarefa(m, nome_tarefa, seed, n); lg = logits_finais(m, T["tk"])
    return (lg[:, -1].argmax(-1) == T["alvo"]).float().mean().item()

def generalizacao(m, base, alt, n=40, n_lote=10, sorteios=5, verbose=True, **kw):
    # 1) circuito achado no molde `base`; 2) fidelidade DELE no molde `alt` (frases novas);
    # 3) circuito achado direto no `alt` e sobreposição; 4) controle aleatório no `alt`.
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; t0 = time.time()
    ac_b, ac_a = acerto(m, base), acerto(m, alt)
    rb = analisar(m, base, podar=False, verbose=False, **kw); cb = [int(c) for c in rb["circuito"]["cabecas"]]
    ra = analisar(m, alt, podar=False, verbose=False, **kw); ca = [int(c) for c in ra["circuito"]["cabecas"]]
    lot = [tarefa(m, alt, 9600 + b, n_lote) for b in range(n // n_lote)]; Cs = [caches(m, T) for T in lot]
    f_cruz = fidelidade_lotes(m, lot, Cs, cb, range(nL)); f_proprio = fidelidade_lotes(m, lot, Cs, ca, range(nL))
    rnd = np.random.default_rng(0)
    aleat = float(np.mean([fidelidade_lotes(m, lot, Cs, sorted(rnd.choice(nL * nH, len(cb), replace=False).tolist()), range(nL))
                           for _ in range(sorteios)]))
    del Cs; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    sobre = len(set(cb) & set(ca)) / max(1, min(len(cb), len(ca)))
    nb = {int(c) for c in rb["nucleo"]["cabecas"]}; na = {int(c) for c in ra["nucleo"]["cabecas"]}
    sobre_nuc = len(nb & na) / max(1, min(len(nb), len(na)))
    r = dict(base=base, alt=alt, acerto_base=ac_b, acerto_alt=ac_a, k_base=len(cb), k_alt=len(ca),
             fid_cruzada=f_cruz, fid_propria=f_proprio, aleatorio=aleat, sobreposicao=sobre,
             nucleo_base=len(nb), nucleo_alt=len(na), sobreposicao_nucleo=sobre_nuc, tempo=round(time.time() - t0, 1))
    if verbose:
        print(f">>> {m.cfg.model_name}: {base} → {alt} | acerto do modelo {ac_b:.0%} / {ac_a:.0%} | "
              f"circuito de {base} ({len(cb)} cab.) no {alt}: {f_cruz:.0%} | circuito próprio do {alt} ({len(ca)} cab.): {f_proprio:.0%} | "
              f"aleatório {aleat:.0%} | cabeças em comum {sobre:.0%}, núcleo em comum {sobre_nuc:.0%} | {r['tempo']} s")
    return r


# =====================================================================================
# 25. Tarefas novas (v0.9): Greater-Than (GT) e múltipla escolha (MCQA)
# =====================================================================================
def metrica(T, lg):
    # Diferença de logit (certa − errada) na última posição, uma por frase. Aceita lotes empilhados.
    # GT: média dos logits dos anos MAIORES que YY menos a média dos anos <= YY (estilo Hanna et al. 2023).
    n = T["alvo"].shape[0]; k = lg.shape[0] // n; ar = torch.arange(lg.shape[0], device=lg.device)
    if "mais" in T:
        anos = torch.tensor(T["anos"], device=lg.device); ult = lg[:, -1][:, anos]
        M = T["mais"].repeat(k, 1).float()
        return ((ult * M).sum(1) / M.sum(1).clamp(min=1) - (ult * (1 - M)).sum(1) / (1 - M).sum(1).clamp(min=1)).float()
    a, b = T["alvo"].repeat(k), T["errado"].repeat(k)
    return (lg[ar, -1, a] - lg[ar, -1, b]).float()

SUBST_GT = [" war", " expedition", " project", " reign", " drought", " siege", " journey", " strike", " festival", " trial"]

def montar_gt(m, seed, n=40):
    # Greater-Than: "The war lasted from the year 17YY to the year 17" -> o modelo deve preferir anos > YY.
    # Corrompida: YY = 01 (todos os anos ficam "maiores"). Só para tokenizadores com anos de 2 dígitos num token (GPT-2, Pythia).
    anos = [m.to_tokens(f"{y:02d}", prepend_bos=False) for y in range(100)]
    if sum(a.shape[1] == 1 for a in anos) < 100:
        raise ValueError("GT precisa de tokens de 2 dígitos ('00'..'99'); este tokenizador separa dígitos (ex.: Qwen)")
    anos = [int(a[0, 0]) for a in anos]; rnd = random.Random(seed)
    subst = [s for s in SUBST_GT if m.to_tokens(s, prepend_bos=False).shape[1] == 1]   # todas as frases com o mesmo comprimento
    molde = "The{S} lasted from the year 17{Y} to the year 17"; ref = tok(m, [molde.format(S=subst[0], Y="01")]).shape[1]
    fr, fc, mais, a, b = [], [], [], [], []
    while len(fr) < n:
        S = rnd.choice(subst); y = rnd.randint(2, 98)
        f = molde.format(S=S, Y=f"{y:02d}"); fcor = molde.format(S=S, Y="01")
        if tok(m, [f]).shape[1] != ref or tok(m, [fcor]).shape[1] != ref: continue
        fr.append(f); fc.append(fcor); mais.append([z > y for z in range(100)])
        a.append(anos[min(y + 1, 99)]); b.append(anos[y - 1])
    return dict(tk=tok(m, fr), tkc=tok(m, fc), alvo=torch.tensor(a, device=DEV), errado=torch.tensor(b, device=DEV),
                mais=torch.tensor(mais, device=DEV), anos=tuple(anos))

CATEGORIAS_MCQA = {
    "fruit": [" apple", " banana", " orange", " grape", " lemon", " cherry", " mango", " peach"],
    "animal": [" dog", " cat", " horse", " cow", " lion", " tiger", " bear", " wolf"],
    "color": [" red", " blue", " green", " yellow", " black", " white", " pink", " purple"],
    "country": [" France", " Spain", " Italy", " Japan", " China", " Brazil", " Canada", " Mexico"],
    "tool": [" hammer", " knife", " saw", " drill", " shovel", " brush", " needle", " rope"]}
_mcqa_ok = {}

def montar_mcqa(m, seed, n=40):
    # Múltipla escolha (estilo MIB): "Which of these is a fruit? A. dog B. apple C. red D. saw Answer:" -> " B".
    # Corrompida: as MESMAS opções em outra ordem, com a certa em outra letra (errado = a nova letra).
    chave = m.cfg.model_name
    if chave not in _mcqa_ok:
        um = lambda w: m.to_tokens(w, prepend_bos=False).shape[1] == 1
        _mcqa_ok[chave] = {c: [w for w in ws if um(w)] for c, ws in CATEGORIAS_MCQA.items() if um(" " + c)}
    cats = {c: ws for c, ws in _mcqa_ok[chave].items() if len(ws) >= 2}
    letras = [m.to_single_token(x) for x in [" A", " B", " C", " D"]]; rnd = random.Random(seed)
    molde = "Which of these is a {c}? A.{o0} B.{o1} C.{o2} D.{o3} Answer:"
    fr, fc, a, b = [], [], [], []
    for _ in range(n):
        c = rnd.choice(list(cats)); outras = rnd.sample([x for x in cats if x != c], 3)
        ops = [rnd.choice(cats[c])] + [rnd.choice(cats[x]) for x in outras]   # ops[0] é a certa
        p1 = rnd.sample(range(4), 4); i1 = p1.index(0)
        p2 = rnd.sample(range(4), 4)
        while p2.index(0) == i1: p2 = rnd.sample(range(4), 4)
        i2 = p2.index(0)
        fr.append(molde.format(c=c, **{f"o{j}": ops[p1[j]] for j in range(4)}))
        fc.append(molde.format(c=c, **{f"o{j}": ops[p2[j]] for j in range(4)}))
        a.append(letras[i1]); b.append(letras[i2])
    T = dict(tk=tok(m, fr), tkc=tok(m, fc), alvo=torch.tensor(a, device=DEV), errado=torch.tensor(b, device=DEV))
    if T["tk"].shape != T["tkc"].shape: raise ValueError("MCQA: frases com comprimentos diferentes")
    return T

RECEITAS["GT"] = montar_gt
RECEITAS["MCQA"] = montar_mcqa

def acerto(m, nome_tarefa, n=40, seed=0):
    # Fração das frases limpas em que a resposta do modelo (1º lugar) é certa. No GT: qualquer ano > YY.
    T = tarefa(m, nome_tarefa, seed, n); lg = logits_finais(m, T["tk"])[:, -1]; top = lg.argmax(-1)
    if "mais" in T:
        anos = torch.tensor(T["anos"], device=lg.device)
        return float(np.mean([bool(T["mais"][i][(anos == top[i]).nonzero()].any()) if (anos == top[i]).any() else False
                              for i in range(len(top))]))
    return (top == T["alvo"]).float().mean().item()

GABARITO_GT_GPT2 = ["5.1", "5.5", "6.1", "6.9", "7.10", "8.8", "8.11", "9.1"]   # Hanna et al. 2023 (de memória; conferir)

def tarefa_nova(m, nome_tarefa, gabarito=None, verbose=True, **kw):
    # Acerto do modelo + HeadHunter (sem arestas) + validação A. Com gabarito, diz onde cada cabeça conhecida caiu.
    nH = m.cfg.n_heads; nome = lambda c: f"{c // nH}.{c % nH}"; t0 = time.time()
    ac = acerto(m, nome_tarefa)
    r = analisar(m, nome_tarefa, podar=False, verbose=False, **kw)
    v = validar_circuito(m, nome_tarefa, r=r, verbose=False)
    nuc = [nome(c) for c in r["nucleo"]["cabecas"]]; circ = [nome(c) for c in r["circuito"]["cabecas"]]
    res = dict(tarefa=nome_tarefa, acerto=ac, nucleo=nuc, mlps_nucleo=r["nucleo"]["mlps"], circuito=circ,
               validacao=v, tempo=round(time.time() - t0, 1))
    if gabarito:
        res["gabarito"] = {h: ("núcleo" if h in nuc else f"circuito ({circ.index(h) + 1}ª)" if h in circ else "FORA") for h in gabarito}
    if verbose:
        print(f">>> {m.cfg.model_name} / {nome_tarefa}: acerto {ac:.0%} | núcleo {len(nuc)} {nuc[:8]} | MLPs {r['nucleo']['mlps']} | "
              f"circuito {len(circ)}/{m.cfg.n_layers * nH} | frases novas {v['fid_min']:.0%}-{max(v['fid_grupos']):.0%} | "
              f"aleatório {v['aleatorio_medio']:.0%} | sobrando {v['sobrando']}/{v['k']} | {res['tempo']} s")
        if gabarito: print("    gabarito:", res["gabarito"])
    return res
