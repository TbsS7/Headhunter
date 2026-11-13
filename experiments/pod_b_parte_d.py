# HeadHunter — parte D: necessidade × suficiência em modelos de 7B ou mais (pod, A100 80 GB). HeadHunter v0.11.3. Escrito em 10/11/2026.
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


# ===================== CÉLULA 2A — escrever o HeadHunter v0.11.3 no volume (parte A de 3) =====================
# (as 3 células 2A-2C só gravam o headhunter.py v0.11.3 em /workspace/headhunter com %%writefile; omitidas aqui: o arquivo é o mesmo do Drive, 95.092 bytes)

# ===================== CÉLULA 2B — escrever o HeadHunter v0.11.3 no volume (parte B de 3) =====================
# (as 3 células 2A-2C só gravam o headhunter.py v0.11.3 em /workspace/headhunter com %%writefile; omitidas aqui: o arquivo é o mesmo do Drive, 95.092 bytes)

# ===================== CÉLULA 2C — escrever o HeadHunter v0.11.3 no volume (parte C de 3) =====================
# (as 3 células 2A-2C só gravam o headhunter.py v0.11.3 em /workspace/headhunter com %%writefile; omitidas aqui: o arquivo é o mesmo do Drive, 95.092 bytes)

# ===================== CÉLULA 3 — conferir e carregar o HeadHunter =====================
import sys, importlib, os
ARQ = "/workspace/headhunter/headhunter.py"
txt = open(ARQ, encoding="utf-8").read() if os.path.exists(ARQ) else ""
assert 'VERSAO = "0.11.3"' in txt and "def _por_cabeca(m)" in txt, "HeadHunter ausente ou fora da v0.11.3: rode as células 2A, 2B e 2C (nesta ordem) e esta de novo"
assert len(txt.encode()) == 95092, f"o arquivo tem {len(txt.encode())} bytes e deveria ter 95092: alguma das células 2A-2C foi rodada pela metade, rode as três de novo"
sys.path.insert(0, "/workspace/headhunter")
import headhunter as hh; importlib.reload(hh); print("HeadHunter", hh.VERSAO, "| arquivo de", len(txt.encode()), "bytes")


# ===================== CÉLULA 4 — definições da parte D (não carrega modelo) =====================
import os, json, time, gc, shutil
import numpy as np, torch
assert hh.VERSAO == "0.11.3", f"HeadHunter {hh.VERSAO}: este notebook é da v0.11.3"
D_PASTA = "/workspace/headhunter/parte_d"
D_MODELOS = {   # nome = nome do TransformerLens; hf = repositório do Hugging Face (None = pequeno, baixa direto); conv="gpu" = converte para float32 na GPU (pico ~4 bytes/parâmetro, cabe em 48 GB)
    "pythia-160m": dict(nome="pythia-160m", hf=None, fam="Pythia (MHA)", teste=True),
    "qwen2.5-7b":  dict(nome="Qwen/Qwen2.5-7B", hf="Qwen/Qwen2.5-7B", fam="Qwen2.5 (GQA)", conv="gpu"),
    "pythia-6.9b": dict(nome="pythia-6.9b", hf="EleutherAI/pythia-6.9b", fam="Pythia (MHA)", conv="gpu"),
    "pythia-12b":  dict(nome="pythia-12b", hf="EleutherAI/pythia-12b", fam="Pythia (MHA)"),
}
D_TAREFAS = ("IOI", "CAP", "COPIA")
D_VALIDAR = ("IOI", "CAP")          # validação em frases novas só nestas (cada uma custa ~10-20 min no 7B+)
D_ACERTO_MIN = 0.30                 # abaixo disto o modelo não faz a tarefa: pula (e anota)

def d_p(chave, nome): return f"{D_PASTA}/{chave}/{nome}.json"
def d_ler(chave, nome):
    p = d_p(chave, nome); return json.load(open(p)) if os.path.exists(p) else None
def d_gravar(chave, nome, dados):
    os.makedirs(f"{D_PASTA}/{chave}", exist_ok=True)
    json.dump(dados, open(d_p(chave, nome), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
def d_gpu(): return f"{torch.cuda.memory_allocated() / 2**30:.1f} GB na GPU" if torch.cuda.is_available() else "CPU"
def d_disco(): return f"{shutil.disk_usage('/workspace').free / 2**30:.0f} GB livres em /workspace"

def d_necessidade(m, tarefa, n=40, alvos=(0.5, 0.8, 0.95)):
    # Necessidade, estilo Venkatesh (igual à hh.necessidade, mas com 3 alvos de dano numa varredura só): ablação pela média de cada cabeça,
    # probabilidade da resposta certa, dano(k) = (p0 - p_k) / (p0 - p_todas) com as cabeças ordenadas pela queda individual.
    # A varredura usa TF32 nas multiplicações (só aqui; o circuito suficiente roda em float32 puro): é uma média de probabilidades, o erro é desprezível.
    t0 = time.time(); nL, nH = m.cfg.n_layers, m.cfg.n_heads; N = nL * nH; nome = lambda c: f"{c // nH}.{c % nH}"
    T = hh.tarefa(m, tarefa, 0, n)
    _, cl = m.run_with_cache(T["tk"], names_filter=lambda x: x.endswith("attn.hook_z"))
    medias = {L: cl[f"blocks.{L}.attn.hook_z"].mean(0, keepdim=True) for L in range(nL)}; del cl
    tf32 = torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32 = True
    try:
        p0 = hh._prob_com_ablacao(m, T, medias, []); p_todas = hh._prob_com_ablacao(m, T, medias, range(N))
        queda = np.array([p0 - hh._prob_com_ablacao(m, T, medias, [c]) for c in range(N)])
        ordem = [int(c) for c in np.argsort(-queda)]; den = max(p0 - p_todas, 1e-9); curva, ks = [], {}
        for k in range(1, N + 1):
            d = (p0 - hh._prob_com_ablacao(m, T, medias, ordem[:k])) / den; curva.append(round(float(d), 4))
            for a in alvos:
                if a not in ks and d >= a: ks[a] = k
            if len(ks) == len(alvos): break
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return dict(p0=float(p0), p_todas=float(p_todas), k={str(int(round(a * 100))): ks.get(a) for a in alvos}, dano1=curva[0],
                curva=curva[:400], top=[nome(c) for c in ordem[:10]], tempo=round(time.time() - t0, 1))

def d_analisar(m, tarefa):
    # circuito suficiente (núcleo, ordem por EAP-IG, busca binária até 90%, degrau). Se faltar memória, repete com lotes menores.
    for n_lote in (10, 5, 2):
        try: return hh.analisar(m, tarefa, podar=False, verbose=False, n_lote=n_lote), n_lote
        except torch.cuda.OutOfMemoryError:
            m.reset_hooks(); torch.set_grad_enabled(False); gc.collect(); torch.cuda.empty_cache()
            print(f"   ⚠️ sem memória com lotes de {n_lote} frases; tento com lotes menores")
    raise RuntimeError("sem memória mesmo com lotes de 2 frases: me mande esta mensagem")

def d_validar(m, tarefa, cab, mlps, grupos=3, n=40, n_lote=10, sorteios=5, eps=0.005, max_min=60):
    # Igual à hh.validar_circuito, sem o passo das arestas: 1) fidelidade em `grupos` grupos de frases NOVAS; 2) minimalidade (tira cada cabeça de uma amostra
    # de até `max_min` cabeças); 3) controle: circuitos aleatórios do mesmo tamanho e só os MLPs.
    t0 = time.time(); nL, nH = m.cfg.n_layers, m.cfg.n_heads; k = len(cab); fids = []
    for g in range(grupos):
        lot = [hh.tarefa(m, tarefa, 9500 + 10 * g + b, n_lote) for b in range(n // n_lote)]; Cs = [hh.caches(m, T) for T in lot]
        fids.append(float(hh.fidelidade_lotes(m, lot, Cs, cab, mlps)))
        if g == 0: lot0, C0 = lot, Cs
        else: del Cs
    rnd = np.random.default_rng(0); base = fids[0]
    amostra = [int(c) for c in rnd.choice(cab, min(max_min, k), replace=False)] if k else []
    quedas = np.array([base - hh.fidelidade_lotes(m, lot0, C0, [x for x in cab if x != c], mlps) for c in amostra])
    aleat = [float(hh.fidelidade_lotes(m, lot0, C0, sorted(rnd.choice(nL * nH, k, replace=False).tolist()), mlps)) for _ in range(sorteios)]
    so_mlps = float(hh.fidelidade_lotes(m, lot0, C0, [], mlps))
    del C0; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return dict(k=k, fid_grupos=[round(f, 3) for f in fids], fid_min=round(min(fids), 3), amostra_minimalidade=len(amostra),
                queda_media=float(quedas.mean()) if len(quedas) else None, sobrando_frac=float((quedas < eps).mean()) if len(quedas) else None,
                aleatorio_medio=float(np.mean(aleat)), aleatorio_max=float(np.max(aleat)), so_mlps=so_mlps, tempo=round(time.time() - t0, 1))

def d_limpar_cache(hf):
    # apaga o modelo baixado (o disco do pod é pequeno) depois de terminar com ele
    if not hf: return
    p = f"/workspace/hf/hub/models--{hf.replace('/', '--')}"
    if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True); print(f"   cache de {hf} apagado | {d_disco()}")

def d_rodar_modelo(chave, tarefas=None, validar=None, apagar_cache=True):
    # Retomável: cada (modelo, tarefa) vira um arquivo; só carrega o modelo se faltar algo.
    cfg = D_MODELOS[chave]; tarefas = tuple(tarefas or D_TAREFAS); validar = D_VALIDAR if validar is None else validar
    falta = []
    for t in tarefas:
        r = d_ler(chave, t)
        if r is None or ("pulada" not in r and t in validar and "validacao" not in r): falta.append(t)
    if not falta: print(f"(já feito) {chave}: {', '.join(tarefas)}"); return
    print(f"=== {chave} ({cfg['fam']}) | falta: {falta} | {d_disco()} ==="); t0 = time.time()
    m = hh.carregar(cfg["nome"], hf_id=cfg["hf"], converter_na=cfg.get("conv", "auto")) if cfg["hf"] else hh.carregar(cfg["nome"])
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; a = hh.adaptador(m)
    d_gravar(chave, "_modelo", dict(nome=cfg["nome"], fam=cfg["fam"], camadas=nL, cabecas=nH, gqa=bool(a["gqa"]), carga_s=round(time.time() - t0), gpu=d_gpu()))
    print(f"   carregado em {time.time() - t0:.0f} s | {d_gpu()}")
    nome_h = lambda c: f"{c // nH}.{c % nH}"
    for t in falta:
        res = d_ler(chave, t)
        if res is None:
            acc = float(hh.acerto(m, t))
            res = dict(modelo=chave, nome=cfg["nome"], fam=cfg["fam"], tarefa=t, acerto=acc, camadas=nL, cabecas=nH, total=nL * nH, gqa=bool(a["gqa"]))
            if acc < D_ACERTO_MIN:
                res["pulada"] = f"acerto {acc:.0%} < {D_ACERTO_MIN:.0%}"; d_gravar(chave, t, res); print(f"   {t}: PULADA ({res['pulada']})"); continue
            nec = d_necessidade(m, t); gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
            r, n_lote = d_analisar(m, t); c = r["circuito"]
            res.update(necessidade=nec, n_lote=n_lote, tempo_analisar=r["tempo"],
                       nucleo=dict(n=len(r["nucleo"]["cabecas"]), cabecas=[nome_h(x) for x in r["nucleo"]["cabecas"]], mlps=[int(x) for x in r["nucleo"]["mlps"]]),
                       circuito=dict(k=len(c["cabecas"]), cabecas=[int(x) for x in c["cabecas"]], mlps=[int(x) for x in c["mlps"]], fidelidade=float(c["fidelidade"]),
                                     atingiu=bool(c["atingiu"]), k_sem_degrau=int(c["k_sem_degrau"]), promovidas=[int(x) for x in c["promovidas"]], revertido=bool(c["revertido"])))
            d_gravar(chave, t, res)
            ne = set(nec["top"][:min(10, nec["k"]["80"] or 10)]); res_n = len(ne & set(res["nucleo"]["cabecas"]))
            print(f"   >>> {chave} / {t}: acerto {acc:.0%} | NECESSIDADE {nec['k']['80']} cabeça(s) p/ 80% do dano ({nec['k']['50']} p/ 50%; a 1ª sozinha {nec['dano1']:.0%}) | "
                  f"NÚCLEO {res['nucleo']['n']} | SUFICIÊNCIA {res['circuito']['k']}/{nL * nH} p/ 90% (fid. {res['circuito']['fidelidade']:.0%}) | "
                  f"top-necessidade no núcleo: {res_n}/{len(ne)} | necessidade {nec['tempo']:.0f} s + circuito {r['tempo']['total']:.0f} s")
        if t in validar and "validacao" not in res and "pulada" not in res:
            v = d_validar(m, t, res["circuito"]["cabecas"], res["circuito"]["mlps"], n_lote=res.get("n_lote", 10)); res["validacao"] = v; d_gravar(chave, t, res)
            print(f"   validação {chave} / {t}: fidelidade em frases novas {[f'{f:.0%}' for f in v['fid_grupos']]} (mín {v['fid_min']:.0%}) | "
                  f"aleatórios do mesmo tamanho: média {v['aleatorio_medio']:.0%}, máx {v['aleatorio_max']:.0%} | só MLPs {v['so_mlps']:.0%} | "
                  f"cabeças quase sem efeito (amostra de {v['amostra_minimalidade']}): {v['sobrando_frac']:.0%} | {v['tempo']:.0f} s")
    print(f"=== {chave} pronto em {(time.time() - t0) / 60:.1f} min ===")
    del m; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if apagar_cache and not cfg.get("teste"): d_limpar_cache(cfg["hf"])

D_TRES = ("qwen2.5-7b", "pythia-6.9b", "pythia-12b")
def d_feitos(): return tuple(c for c in D_TRES if any(d_ler(c, t) for t in D_TAREFAS))   # os modelos que já têm resultado (o 12B é opcional)

def d_tabela(modelos=None):
    modelos = modelos or d_feitos()
    print(f"\n{'modelo':12s}{'tarefa':7s}{'acerto':>7s}{'necess.80%':>11s}{'(50%)':>7s}{'núcleo':>8s}{'suficiência':>13s}{'fid':>6s}{'valid. mín':>11s}{'aleat. máx':>11s}")
    for ch in modelos:
        for t in D_TAREFAS:
            r = d_ler(ch, t)
            if r is None: continue
            if "pulada" in r: print(f"{ch:12s}{t:7s}{r['acerto']:7.0%}   pulada ({r['pulada']})"); continue
            n, v = r["necessidade"], r.get("validacao"); suf = f"{r['circuito']['k']}/{r['total']}"
            print(f"{ch:12s}{t:7s}{r['acerto']:7.0%}{str(n['k']['80']):>11s}{str(n['k']['50']):>7s}{r['nucleo']['n']:8d}{suf:>13s}{r['circuito']['fidelidade']:6.0%}"
                  + (f"{v['fid_min']:11.0%}{v['aleatorio_max']:11.0%}" if v else f"{'-':>11s}{'-':>11s}"))

def d_previsoes(modelos=None):
    modelos = modelos or d_feitos(); R = {(ch, t): d_ler(ch, t) for ch in D_TRES for t in D_TAREFAS}; out = []
    ok_ = lambda r: r is not None and "pulada" not in r
    def add(txt, ok): out.append(f"{'⏳' if ok is None else '✅' if ok else '❌'} {txt}")
    def faixa(cod, ch, t, lo, hi, quem):
        r = R[(ch, t)]
        if not ok_(r): add(f"{cod} {quem}: sem dado", None); return
        v = r["nucleo"]["n"]; add(f"{cod} {quem}: núcleo {v} (previsto {lo}-{hi})", lo <= v <= hi)
    faixa("D1", "qwen2.5-7b", "IOI", 8, 16, "Qwen2.5-7B IOI"); faixa("D2", "pythia-6.9b", "IOI", 4, 10, "Pythia-6.9B IOI"); faixa("D3", "pythia-12b", "IOI", 4, 14, "Pythia-12B IOI")
    ioi = [R[(ch, "IOI")] for ch in modelos if ok_(R[(ch, "IOI")])]
    if len(ioi) == len(modelos):
        ks = [r["necessidade"]["k"]["80"] for r in ioi]
        add(f"D4 IOI necessidade (80%): {ks} (previsto entre 2 e 15 em todos os rodados)", all(k is not None and 2 <= k <= 15 for k in ks))
        fr = [r["circuito"]["k"] / r["total"] for r in ioi]; add(f"D5 IOI suficiência / total: {[f'{f:.0%}' for f in fr]} (previsto ≤ 30% em todos os rodados)", all(f <= 0.30 for f in fr))
    else: add("D4-D5 (faltam modelos no IOI)", None)
    cap = [R[(ch, "CAP")] for ch in modelos if ok_(R[(ch, "CAP")])]
    if len(cap) == len(modelos):
        raz = [(r["necessidade"]["k"]["80"] or r["total"]) / r["circuito"]["k"] for r in cap]
        add(f"D6 CAP necessidade / suficiência: {[f'{x:.1f}' for x in raz]} (previsto ≥ 5 em pelo menos 2 dos 3)", sum(x >= 5 for x in raz) >= 2)
    else: add("D6 (faltam modelos no CAP)", None)
    r = R[("pythia-6.9b", "COPIA")]
    if ok_(r): add(f"D7 Pythia-6.9B COPIA: núcleo {r['nucleo']['n']} (previsto ≤ 2)", r["nucleo"]["n"] <= 2)
    else: add("D7 Pythia-6.9B COPIA: sem dado" + (f" ({r['pulada']})" if r else ""), None)
    a, b = R[("pythia-6.9b", "IOI")], R[("qwen2.5-7b", "IOI")]
    if ok_(a) and ok_(b):
        ka, kb = a["necessidade"]["k"]["80"], b["necessidade"]["k"]["80"]; add(f"D8 necessidade IOI Pythia-6.9B {ka} × Qwen2.5-7B {kb} (previsto diferença ≤ 6)", ka is not None and kb is not None and abs(ka - kb) <= 6)
    else: add("D8 (faltam dados)", None)
    vs = [(ch, R[(ch, "IOI")].get("validacao")) for ch in modelos if ok_(R[(ch, "IOI")])]
    if vs and all(v for _, v in vs) and len(vs) == len(modelos):
        add("D9 validação IOI: " + " | ".join(f"{ch}: mín {v['fid_min']:.0%}, aleat. máx {v['aleatorio_max']:.0%}, só MLPs {v['so_mlps']:.0%}" for ch, v in vs) + " (previsto ≥ 85%, ≤ 30%, ≤ 10%)",
            all(v["fid_min"] >= 0.85 and v["aleatorio_max"] <= 0.30 and v["so_mlps"] <= 0.10 for _, v in vs))
    else: add("D9 (faltam validações no IOI)", None)
    print("\nprevisões:"); [print("  " + x) for x in out]; return out


# ===================== CÉLULA 5 — teste rápido no Pythia-160m (~3 min; confere que tudo roda antes de baixar os modelos grandes) =====================
d_rodar_modelo("pythia-160m", validar=("IOI", "CAP"))
for t in D_TAREFAS:
    r = d_ler("pythia-160m", t); print(t, "→", "PULADA " + r["pulada"] if "pulada" in r else f"necessidade {r['necessidade']['k']['80']}, núcleo {r['nucleo']['n']}, suficiência {r['circuito']['k']}/{r['total']}")


# ===================== CÉLULA 6 — Qwen2.5-7B (GQA): 3 tarefas + validação em IOI e CAP (retomável; estimo 30-50 min) =====================
d_rodar_modelo("qwen2.5-7b")


# ===================== CÉLULA 7 — Pythia-6.9B (MHA): 3 tarefas + validação em IOI e CAP (retomável; estimo 30-50 min) =====================
d_rodar_modelo("pythia-6.9b")


# ===================== CÉLULA 8 (OPCIONAL, a mais cara e a menos testada: só se sobrar saldo) — Pythia-12B (MHA): 3 tarefas + validação em IOI e CAP (retomável; estimo 50-90 min) =====================
d_rodar_modelo("pythia-12b")


# ===================== CÉLULA 9 — resumo final: tabela e previsões (cole TUDO que esta célula imprimir no chat) =====================
d_tabela(); d_previsoes()
shutil.make_archive("/workspace/parte_d_resultados", "zip", D_PASTA); print("\nbackup:", "/workspace/parte_d_resultados.zip (baixe pelo JupyterLab antes de apagar o pod)")


# ===================== CÉLULA 5b — CORREÇÃO DO IOI (feita em 11/11/2026, depois de ver 0% e 5% de acerto no Qwen2.5-7B e na Pythia-6.9B) =====================
# Motivo: os modelos grandes põem a maior probabilidade em ' each'/' a'/' every' em 'gave a drink to'; o acerto top-1 dava ~0%.
# Resultado: com este critério o acerto do IOI foi 100% nos dois. Rodar ANTES de d_rodar_modelo(...) para o IOI. CAP e COPIA não mudam.
# CORREÇÃO DO IOI (parte D). Os modelos grandes põem a maior probabilidade em " each", " a", " every" e não no nome,
# então o acerto "top-1" dá ~0%. Aqui o IOI passa a usar o critério clássico: o nome certo é mais provável que o nome errado,
# e a "probabilidade da resposta certa" da necessidade vira P(certo) / (P(certo) + P(errado)). CAP e COPIA não mudam.
import os, torch
_acerto_orig, _prob_orig, _nec_orig = hh.acerto, hh._prob_com_ablacao, d_necessidade
_BIN = {"on": False}

def _acerto_novo(m, nome_tarefa, n=40, seed=0):
    if nome_tarefa != "IOI": return _acerto_orig(m, nome_tarefa, n, seed)
    T = hh.tarefa(m, nome_tarefa, seed, n); lg = hh.logits_finais(m, T["tk"])[:, -1]; ar = torch.arange(len(lg), device=lg.device)
    return (lg[ar, T["alvo"]] > lg[ar, T["errado"]]).float().mean().item()

def _prob_nova(m, T, medias, cabecas):
    if not _BIN["on"]: return _prob_orig(m, T, medias, cabecas)
    nH = m.cfg.n_heads; por = {}
    for c in cabecas: por.setdefault(int(c) // nH, []).append(int(c) % nH)
    hooks = []
    for L, hs in por.items():
        def hk(z, hook, L=L, hs=hs):
            z[:, :, hs] = medias[L][:, :, hs]; return z
        hooks.append((f"blocks.{L}.attn.hook_z", hk))
    ar = torch.arange(T["tk"].shape[0], device=T["tk"].device)
    lg = hh.logits_finais(m, T["tk"], None, hooks)[:, -1]
    return torch.sigmoid((lg[ar, T["alvo"]] - lg[ar, T["errado"]]).float()).mean().item()

def d_necessidade(m, tarefa, n=40, alvos=(0.5, 0.8, 0.95)):
    _BIN["on"] = (tarefa == "IOI")
    try: return _nec_orig(m, tarefa, n, alvos)
    finally: _BIN["on"] = False

hh.acerto, hh._prob_com_ablacao = _acerto_novo, _prob_nova
for chave in ("qwen2.5-7b", "pythia-6.9b"):          # apaga só o "pulada" do IOI; CAP e COPIA ficam
    r = d_ler(chave, "IOI")
    if r is not None and "pulada" in r: os.remove(d_p(chave, "IOI")); print("apagado o resultado 'pulada' do IOI de", chave)
print("IOI corrigido. Agora rode a célula abaixo.")

# Para rodar só o IOI (pod novo): D_TAREFAS = ("IOI",); d_rodar_modelo("pythia-6.9b")
