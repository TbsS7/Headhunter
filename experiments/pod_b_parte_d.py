# HeadHunter — parte D: necessidade × suficiência em modelos de 7B ou mais (pod, A100 80 GB). HeadHunter v0.11.3. Escrito em 10/11/2026.
# ADAPTADO em 13/11/2026 para a API em inglês (v0.12.0). Os números reportados foram obtidos com a v0.11.x (português); a v0.12.0 dá os mesmos valores (ver CHANGELOG.md). Comentários e mensagens impressas continuam em português.
# Pergunta: nos modelos grandes, quantas cabeças precisam SAIR para a tarefa quebrar (necessidade, ablação pela média, estilo Venkatesh 2026) e
# quantas BASTAM para reproduzir o modelo (suficiência, circuito de 90%)? E isso muda entre uma família GQA (Qwen2.5-7B) e uma MHA (Pythia-6.9B e 12B)?
# Tarefas: IOI, CAP (capitais), COPIA (indução). Para cada (modelo, tarefa): acerto do modelo, necessidade (cabeças para 50/80/95% do dano), núcleo (≥ 5%),
# circuito suficiente (cabeças para 90% de fidelidade) e, em IOI e CAP, a validação em frases novas (fidelidade, controles aleatórios, só MLPs).
# Referência (≤ 1,5B, 02/11): IOI necessidade 10/4/5/4 (Pythia-160m, 1.4b, Qwen 0.5B, 1.5B), núcleo 9/13/18/15, suficiência 24/57/76/79;
#   CAP necessidade 4/382/336/28 contra suficiência 19/4/6/12 (redundância extrema). Qwen2.5-7B IOI (31/10): núcleo 11, suficiência 160 de 784.
#   Pythia-6.9B (31/10): núcleo 7/4/0 (IOI/CAP/COPIA).
#
# ───────────────────────── PREVISÕES (escritas ANTES de rodar; não mudam depois) ─────────────────────────
#  D1  Qwen2.5-7B, IOI: núcleo (≥ 5% do efeito) entre 8 e 16 cabeças.
#  D2  Pythia-6.9B, IOI: núcleo entre 4 e 10 cabeças.
#  D3  Pythia-12B, IOI: núcleo entre 4 e 14 cabeças.
#  D4  IOI, os três modelos: necessidade (cabeças p/ 80% do dano) entre 2 e 15.
#  D5  IOI, os três modelos: suficiência (cabeças p/ 90%) ≤ 30% do total de cabeças.
#  D6  CAP: em pelo menos 2 dos 3 modelos, necessidade ≥ 5× a suficiência (precisa remover muito mais do que basta manter).
#  D7  COPIA no Pythia-6.9B: núcleo ≤ 2 cabeças (a cópia "sem núcleo" do 31/10 se repete).
#  D8  Venkatesh (a arquitetura decide): a necessidade IOI do Pythia-6.9B (MHA) e a do Qwen2.5-7B (GQA) diferem por no máximo 6 cabeças.
#  D9  Validação (IOI, os três modelos): fidelidade mínima nos grupos novos ≥ 85%, circuito aleatório do mesmo tamanho ≤ 30% (máx), só os MLPs ≤ 10%.
# O que NÃO é previsão: tempo total (estimo 2-3 h no pod) e a lista exata de cabeças.
# ──────────────────────────────────────────────────────────────────────────────────────────────


# ===================== CÉLULA 1 — instalar (pod) =====================
import os
for d in ("/workspace/headhunter", "/workspace/hf"): os.makedirs(d, exist_ok=True)
os.environ["HF_HOME"] = "/workspace/hf"
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!pip -q install "transformer_lens<4" tabulate matplotlib pandas scipy einops tqdm
!df -h /workspace | tail -1
import torch; print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NENHUMA", "| memória:", round(torch.cuda.get_device_properties(0).total_memory / 2**30) if torch.cuda.is_available() else "-", "GB")


# ===================== CÉLULA 3 — instalar e conferir o HeadHunter (v0.12.0, API em inglês) =====================
!pip install -q "git+https://github.com/TbsS7/Headhunter.git"
import importlib, headhunter as hh; importlib.reload(hh)
assert hh.VERSION == "0.12.0", f"HeadHunter {hh.VERSION}: este notebook é da v0.12.0 (rode a célula 3 de novo)"
print("HeadHunter", hh.VERSION, "| esquema:", hh.IG_SCHEME, "| LOTE_CALCULO", hh.COMPUTE_BATCH)


# ===================== CÉLULA 4 — definições da parte D (não carrega modelo) =====================
import os, json, time, gc, shutil
import numpy as np, torch
assert hh.VERSION == "0.12.0", f"HeadHunter {hh.VERSION}: este notebook é da v0.12.0 (rode a célula 3)"
D_PASTA = "/workspace/headhunter/parte_d"
D_MODELS = {   # nome = nome do TransformerLens; hf = repositório do Hugging Face (None = pequeno, baixa direto); conv="gpu" = converte para float32 na GPU (pico ~4 bytes/parâmetro, cabe em 48 GB)
    "pythia-160m": dict(name="pythia-160m", hf=None, fam="Pythia (MHA)", teste=True),
    "qwen2.5-7b":  dict(name="Qwen/Qwen2.5-7B", hf="Qwen/Qwen2.5-7B", fam="Qwen2.5 (GQA)", conv="gpu"),
    "pythia-6.9b": dict(name="pythia-6.9b", hf="EleutherAI/pythia-6.9b", fam="Pythia (MHA)", conv="gpu"),
    "pythia-12b":  dict(name="pythia-12b", hf="EleutherAI/pythia-12b", fam="Pythia (MHA)"),
}
D_TAREFAS = ("IOI", "CAP", "COPY")
D_VALIDAR = ("IOI", "CAP")          # validação em frases novas só nestas (cada uma custa ~10-20 min no 7B+)
D_ACERTO_MIN = 0.30                 # abaixo disto o modelo não faz a tarefa: pula (e anota)

def d_p(key, name): return f"{D_PASTA}/{key}/{name}.json"
def d_read(key, name):
    p = d_p(key, name); return json.load(open(p)) if os.path.exists(p) else None
def d_gravar(key, name, data):
    os.makedirs(f"{D_PASTA}/{key}", exist_ok=True)
    json.dump(data, open(d_p(key, name), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
def d_gpu(): return f"{torch.cuda.memory_allocated() / 2**30:.1f} GB na GPU" if torch.cuda.is_available() else "CPU"
def d_disco(): return f"{shutil.disk_usage('/workspace').free / 2**30:.0f} GB livres em /workspace"

def d_necessity(m, task, n=40, targets=(0.5, 0.8, 0.95)):
    # Necessidade, estilo Venkatesh (igual à hh.necessidade, mas com 3 alvos de dano numa varredura só): ablação pela média de cada cabeça,
    # probabilidade da resposta certa, dano(k) = (p0 - p_k) / (p0 - p_todas) com as cabeças ordenadas pela queda individual.
    # A varredura usa TF32 nas multiplicações (só aqui; o circuito suficiente roda em float32 puro): é uma média de probabilidades, o erro é desprezível.
    t0 = time.time(); nL, nH = m.cfg.n_layers, m.cfg.n_heads; N = nL * nH; name = lambda c: f"{c // nH}.{c % nH}"
    T = hh.get_task(m, task, 0, n)
    _, cl = m.run_with_cache(T["tk"], names_filter=lambda x: x.endswith("attn.hook_z"))
    means = {L: cl[f"blocks.{L}.attn.hook_z"].mean(0, keepdim=True) for L in range(nL)}; del cl
    tf32 = torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32 = True
    try:
        p0 = hh._prob_with_ablation(m, T, means, []); p_all = hh._prob_with_ablation(m, T, means, range(N))
        drop = np.array([p0 - hh._prob_with_ablation(m, T, means, [c]) for c in range(N)])
        order = [int(c) for c in np.argsort(-drop)]; den = max(p0 - p_all, 1e-9); curve, ks = [], {}
        for k in range(1, N + 1):
            d = (p0 - hh._prob_with_ablation(m, T, means, order[:k])) / den; curve.append(round(float(d), 4))
            for a in targets:
                if a not in ks and d >= a: ks[a] = k
            if len(ks) == len(targets): break
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return dict(p0=float(p0), p_all=float(p_all), k={str(int(round(a * 100))): ks.get(a) for a in targets}, dano1=curve[0],
                curve=curve[:400], top=[name(c) for c in order[:10]], elapsed=round(time.time() - t0, 1))

def d_analyze(m, task):
    # circuito suficiente (núcleo, ordem por EAP-IG, busca binária até 90%, degrau). Se faltar memória, repete com lotes menores.
    for n_batch in (10, 5, 2):
        try: return hh.analyze(m, task, prune=False, verbose=False, n_batch=n_batch), n_batch
        except torch.cuda.OutOfMemoryError:
            m.reset_hooks(); torch.set_grad_enabled(False); gc.collect(); torch.cuda.empty_cache()
            print(f"   ⚠️ sem memória com lotes de {n_batch} frases; tento com lotes menores")
    raise RuntimeError("sem memória mesmo com lotes de 2 frases: me mande esta mensagem")

def d_validate(m, task, head, mlps, groups=3, n=40, n_batch=10, draws=5, eps=0.005, max_min=60):
    # Igual à hh.validar_circuito, sem o passo das arestas: 1) fidelidade em `grupos` grupos de frases NOVAS; 2) minimalidade (tira cada cabeça de uma amostra
    # de até `max_min` cabeças); 3) controle: circuitos aleatórios do mesmo tamanho e só os MLPs.
    t0 = time.time(); nL, nH = m.cfg.n_layers, m.cfg.n_heads; k = len(head); fids = []
    for g in range(groups):
        bat = [hh.get_task(m, task, 9500 + 10 * g + b, n_batch) for b in range(n // n_batch)]; Cs = [hh.caches(m, T) for T in bat]
        fids.append(float(hh.faithfulness_batches(m, bat, Cs, head, mlps)))
        if g == 0: lot0, C0 = bat, Cs
        else: del Cs
    rnd = np.random.default_rng(0); base = fids[0]
    sample = [int(c) for c in rnd.choice(head, min(max_min, k), replace=False)] if k else []
    drops = np.array([base - hh.faithfulness_batches(m, lot0, C0, [x for x in head if x != c], mlps) for c in sample])
    rand = [float(hh.faithfulness_batches(m, lot0, C0, sorted(rnd.choice(nL * nH, k, replace=False).tolist()), mlps)) for _ in range(draws)]
    so_mlps = float(hh.faithfulness_batches(m, lot0, C0, [], mlps))
    del C0; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return dict(k=k, faith_groups=[round(f, 3) for f in fids], faith_min=round(min(fids), 3), sample_minimality=len(sample),
                drop_mean=float(drops.mean()) if len(drops) else None, surplus_frac=float((drops < eps).mean()) if len(drops) else None,
                random_mean=float(np.mean(rand)), random_max=float(np.max(rand)), so_mlps=so_mlps, elapsed=round(time.time() - t0, 1))

def d_limpar_cache(hf):
    # apaga o modelo baixado (o disco do pod é pequeno) depois de terminar com ele
    if not hf: return
    p = f"/workspace/hf/hub/models--{hf.replace('/', '--')}"
    if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True); print(f"   cache de {hf} apagado | {d_disco()}")

def d_run_model(key, tasks=None, validate=None, apagar_cache=True):
    # Retomável: cada (modelo, tarefa) vira um arquivo; só carrega o modelo se faltar algo.
    cfg = D_MODELS[key]; tasks = tuple(tasks or D_TAREFAS); validate = D_VALIDAR if validate is None else validate
    falta = []
    for t in tasks:
        r = d_read(key, t)
        if r is None or ("pulada" not in r and t in validate and "validation" not in r): falta.append(t)
    if not falta: print(f"(já feito) {key}: {', '.join(tasks)}"); return
    print(f"=== {key} ({cfg['fam']}) | falta: {falta} | {d_disco()} ==="); t0 = time.time()
    m = hh.load_model(cfg["name"], hf_id=cfg["hf"], convert_on=cfg.get("conv", "auto")) if cfg["hf"] else hh.load_model(cfg["name"])
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; a = hh.adapter(m)
    d_gravar(key, "_model", dict(name=cfg["name"], fam=cfg["fam"], layers=nL, heads=nH, gqa=bool(a["gqa"]), load_s=round(time.time() - t0), gpu=d_gpu()))
    print(f"   carregado em {time.time() - t0:.0f} s | {d_gpu()}")
    hook_name = lambda c: f"{c // nH}.{c % nH}"
    for t in falta:
        res = d_read(key, t)
        if res is None:
            acc = float(hh.accuracy(m, t))
            res = dict(model=key, name=cfg["name"], fam=cfg["fam"], task=t, accuracy=acc, layers=nL, heads=nH, total=nL * nH, gqa=bool(a["gqa"]))
            if acc < D_ACERTO_MIN:
                res["pulada"] = f"acerto {acc:.0%} < {D_ACERTO_MIN:.0%}"; d_gravar(key, t, res); print(f"   {t}: PULADA ({res['pulada']})"); continue
            nec = d_necessity(m, t); gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
            r, n_batch = d_analyze(m, t); c = r["circuit"]
            res.update(necessity=nec, n_batch=n_batch, elapsed_analyze=r["elapsed"],
                       core=dict(n=len(r["core"]["heads"]), heads=[hook_name(x) for x in r["core"]["heads"]], mlps=[int(x) for x in r["core"]["mlps"]]),
                       circuit=dict(k=len(c["heads"]), heads=[int(x) for x in c["heads"]], mlps=[int(x) for x in c["mlps"]], faithfulness=float(c["faithfulness"]),
                                     reached=bool(c["reached"]), k_without_step=int(c["k_without_step"]), promoted=[int(x) for x in c["promoted"]], reverted=bool(c["reverted"])))
            d_gravar(key, t, res)
            ne = set(nec["top"][:min(10, nec["k"]["80"] or 10)]); res_n = len(ne & set(res["core"]["heads"]))
            print(f"   >>> {key} / {t}: acerto {acc:.0%} | NECESSIDADE {nec['k']['80']} cabeça(s) p/ 80% do dano ({nec['k']['50']} p/ 50%; a 1ª sozinha {nec['dano1']:.0%}) | "
                  f"NÚCLEO {res['core']['n']} | SUFICIÊNCIA {res['circuit']['k']}/{nL * nH} p/ 90% (fid. {res['circuit']['faithfulness']:.0%}) | "
                  f"top-necessidade no núcleo: {res_n}/{len(ne)} | necessidade {nec['elapsed']:.0f} s + circuito {r['elapsed']['total']:.0f} s")
        if t in validate and "validation" not in res and "pulada" not in res:
            v = d_validate(m, t, res["circuit"]["heads"], res["circuit"]["mlps"], n_batch=res.get("n_batch", 10)); res["validation"] = v; d_gravar(key, t, res)
            print(f"   validação {key} / {t}: fidelidade em frases novas {[f'{f:.0%}' for f in v['faith_groups']]} (mín {v['faith_min']:.0%}) | "
                  f"aleatórios do mesmo tamanho: média {v['random_mean']:.0%}, máx {v['random_max']:.0%} | só MLPs {v['so_mlps']:.0%} | "
                  f"cabeças quase sem efeito (amostra de {v['sample_minimality']}): {v['surplus_frac']:.0%} | {v['elapsed']:.0f} s")
    print(f"=== {key} pronto em {(time.time() - t0) / 60:.1f} min ===")
    del m; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if apagar_cache and not cfg.get("teste"): d_limpar_cache(cfg["hf"])

D_TRES = ("qwen2.5-7b", "pythia-6.9b", "pythia-12b")
def d_feitos(): return tuple(c for c in D_TRES if any(d_read(c, t) for t in D_TAREFAS))   # os modelos que já têm resultado (o 12B é opcional)

def d_tabela(modelos=None):
    modelos = modelos or d_feitos()
    print(f"\n{'model':12s}{'task':7s}{'accuracy':>7s}{'necess.80%':>11s}{'(50%)':>7s}{'núcleo':>8s}{'suficiência':>13s}{'faith':>6s}{'valid. mín':>11s}{'aleat. máx':>11s}")
    for ch in modelos:
        for t in D_TAREFAS:
            r = d_read(ch, t)
            if r is None: continue
            if "pulada" in r: print(f"{ch:12s}{t:7s}{r['accuracy']:7.0%}   pulada ({r['pulada']})"); continue
            n, v = r["necessity"], r.get("validation"); suf = f"{r['circuit']['k']}/{r['total']}"
            print(f"{ch:12s}{t:7s}{r['accuracy']:7.0%}{str(n['k']['80']):>11s}{str(n['k']['50']):>7s}{r['core']['n']:8d}{suf:>13s}{r['circuit']['faithfulness']:6.0%}"
                  + (f"{v['faith_min']:11.0%}{v['random_max']:11.0%}" if v else f"{'-':>11s}{'-':>11s}"))

def d_previsoes(modelos=None):
    modelos = modelos or d_feitos(); R = {(ch, t): d_read(ch, t) for ch in D_TRES for t in D_TAREFAS}; out = []
    ok_ = lambda r: r is not None and "pulada" not in r
    def add(txt, ok): out.append(f"{'⏳' if ok is None else '✅' if ok else '❌'} {txt}")
    def range(cod, ch, t, lo, hi, quem):
        r = R[(ch, t)]
        if not ok_(r): add(f"{cod} {quem}: sem dado", None); return
        v = r["core"]["n"]; add(f"{cod} {quem}: núcleo {v} (previsto {lo}-{hi})", lo <= v <= hi)
    range("D1", "qwen2.5-7b", "IOI", 8, 16, "Qwen2.5-7B IOI"); range("D2", "pythia-6.9b", "IOI", 4, 10, "Pythia-6.9B IOI"); range("D3", "pythia-12b", "IOI", 4, 14, "Pythia-12B IOI")
    ioi = [R[(ch, "IOI")] for ch in modelos if ok_(R[(ch, "IOI")])]
    if len(ioi) == len(modelos):
        ks = [r["necessity"]["k"]["80"] for r in ioi]
        add(f"D4 IOI necessidade (80%): {ks} (previsto entre 2 e 15 em todos os rodados)", all(k is not None and 2 <= k <= 15 for k in ks))
        fr = [r["circuit"]["k"] / r["total"] for r in ioi]; add(f"D5 IOI suficiência / total: {[f'{f:.0%}' for f in fr]} (previsto ≤ 30% em todos os rodados)", all(f <= 0.30 for f in fr))
    else: add("D4-D5 (faltam modelos no IOI)", None)
    cap = [R[(ch, "CAP")] for ch in modelos if ok_(R[(ch, "CAP")])]
    if len(cap) == len(modelos):
        raz = [(r["necessity"]["k"]["80"] or r["total"]) / r["circuit"]["k"] for r in cap]
        add(f"D6 CAP necessidade / suficiência: {[f'{x:.1f}' for x in raz]} (previsto ≥ 5 em pelo menos 2 dos 3)", sum(x >= 5 for x in raz) >= 2)
    else: add("D6 (faltam modelos no CAP)", None)
    r = R[("pythia-6.9b", "COPY")]
    if ok_(r): add(f"D7 Pythia-6.9B COPIA: núcleo {r['core']['n']} (previsto ≤ 2)", r["core"]["n"] <= 2)
    else: add("D7 Pythia-6.9B COPIA: sem dado" + (f" ({r['pulada']})" if r else ""), None)
    a, b = R[("pythia-6.9b", "IOI")], R[("qwen2.5-7b", "IOI")]
    if ok_(a) and ok_(b):
        ka, kb = a["necessity"]["k"]["80"], b["necessity"]["k"]["80"]; add(f"D8 necessidade IOI Pythia-6.9B {ka} × Qwen2.5-7B {kb} (previsto diferença ≤ 6)", ka is not None and kb is not None and abs(ka - kb) <= 6)
    else: add("D8 (faltam dados)", None)
    vs = [(ch, R[(ch, "IOI")].get("validation")) for ch in modelos if ok_(R[(ch, "IOI")])]
    if vs and all(v for _, v in vs) and len(vs) == len(modelos):
        add("D9 validação IOI: " + " | ".join(f"{ch}: mín {v['faith_min']:.0%}, aleat. máx {v['random_max']:.0%}, só MLPs {v['so_mlps']:.0%}" for ch, v in vs) + " (previsto ≥ 85%, ≤ 30%, ≤ 10%)",
            all(v["faith_min"] >= 0.85 and v["random_max"] <= 0.30 and v["so_mlps"] <= 0.10 for _, v in vs))
    else: add("D9 (faltam validações no IOI)", None)
    print("\nprevisões:"); [print("  " + x) for x in out]; return out


# ===================== CÉLULA 5 — teste rápido no Pythia-160m (~3 min; confere que tudo roda antes de baixar os modelos grandes) =====================
d_run_model("pythia-160m", validate=("IOI", "CAP"))
for t in D_TAREFAS:
    r = d_read("pythia-160m", t); print(t, "→", "PULADA " + r["pulada"] if "pulada" in r else f"necessidade {r['necessity']['k']['80']}, núcleo {r['core']['n']}, suficiência {r['circuit']['k']}/{r['total']}")


# ===================== CÉLULA 6 — Qwen2.5-7B (GQA): 3 tarefas + validação em IOI e CAP (retomável; estimo 30-50 min) =====================
d_run_model("qwen2.5-7b")


# ===================== CÉLULA 7 — Pythia-6.9B (MHA): 3 tarefas + validação em IOI e CAP (retomável; estimo 30-50 min) =====================
d_run_model("pythia-6.9b")


# ===================== CÉLULA 8 (OPCIONAL, a mais cara e a menos testada: só se sobrar saldo) — Pythia-12B (MHA): 3 tarefas + validação em IOI e CAP (retomável; estimo 50-90 min) =====================
d_run_model("pythia-12b")


# ===================== CÉLULA 9 — resumo final: tabela e previsões (cole TUDO que esta célula imprimir no chat) =====================
d_tabela(); d_previsoes()
shutil.make_archive("/workspace/parte_d_resultados", "zip", D_PASTA); print("\nbackup:", "/workspace/parte_d_resultados.zip (baixe pelo JupyterLab antes de apagar o pod)")


# ===================== CÉLULA 5b — CORREÇÃO DO IOI (feita em 11/11/2026, depois de ver 0% e 5% de acerto no Qwen2.5-7B e na Pythia-6.9B) =====================
# Motivo: os modelos grandes põem a maior probabilidade em ' each'/' a'/' every' em 'gave a drink to'; o acerto top-1 dava ~0%.
# Resultado: com este critério o acerto do IOI foi 100% nos dois. Rodar ANTES de d_run_model(...) para o IOI. CAP e COPIA não mudam.
# CORREÇÃO DO IOI (parte D). Os modelos grandes põem a maior probabilidade em " each", " a", " every" e não no nome,
# então o acerto "top-1" dá ~0%. Aqui o IOI passa a usar o critério clássico: o nome certo é mais provável que o nome errado,
# e a "probabilidade da resposta certa" da necessidade vira P(certo) / (P(certo) + P(errado)). CAP e COPIA não mudam.
import os, torch
_accuracy_orig, _prob_orig, _nec_orig = hh.accuracy, hh._prob_with_ablation, d_necessity
_BIN = {"on": False}

def _accuracy_new(m, task_name, n=40, seed=0):
    if task_name != "IOI": return _accuracy_orig(m, task_name, n, seed)
    T = hh.get_task(m, task_name, seed, n); lg = hh.final_logits(m, T["tk"])[:, -1]; ar = torch.arange(len(lg), device=lg.device)
    return (lg[ar, T["target"]] > lg[ar, T["wrong"]]).float().mean().item()

def _prob_new(m, T, means, heads):
    if not _BIN["on"]: return _prob_orig(m, T, means, heads)
    nH = m.cfg.n_heads; per = {}
    for c in heads: per.setdefault(int(c) // nH, []).append(int(c) % nH)
    hooks = []
    for L, hs in per.items():
        def hk(z, hook, L=L, hs=hs):
            z[:, :, hs] = means[L][:, :, hs]; return z
        hooks.append((f"blocks.{L}.attn.hook_z", hk))
    ar = torch.arange(T["tk"].shape[0], device=T["tk"].device)
    lg = hh.final_logits(m, T["tk"], None, hooks)[:, -1]
    return torch.sigmoid((lg[ar, T["target"]] - lg[ar, T["wrong"]]).float()).mean().item()

def d_necessity(m, task, n=40, targets=(0.5, 0.8, 0.95)):
    _BIN["on"] = (task == "IOI")
    try: return _nec_orig(m, task, n, targets)
    finally: _BIN["on"] = False

hh.accuracy, hh._prob_with_ablation = _accuracy_new, _prob_new
for key in ("qwen2.5-7b", "pythia-6.9b"):          # apaga só o "pulada" do IOI; CAP e COPIA ficam
    r = d_read(key, "IOI")
    if r is not None and "pulada" in r: os.remove(d_p(key, "IOI")); print("apagado o resultado 'pulada' do IOI de", key)
print("IOI corrigido. Agora rode a célula abaixo.")

# Para rodar só o IOI (pod novo): D_TAREFAS = ("IOI",); d_run_model("pythia-6.9b")
