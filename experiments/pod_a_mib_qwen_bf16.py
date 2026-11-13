# HeadHunter × MIB — Qwen2.5-0.5B em bfloat16 (pod, A100). HeadHunter v0.11.3. Escrito em 10/11/2026.
# Objetivo: repetir o teste do Colab (T4, float32) na precisão do MIB (bf16), com 5 grupos de frases em vez de 3 e mais frases de validação,
# para ter números que entram no artigo. Métodos: HH-cercadinho-meio, HH-cercadinho-eapig, HH-livre-eapig e EAP-IG oficial.
# Referência do Colab/T4 (float32, 500 frases de validação IOI, 3 grupos): IOI CMD EAP-IG 0,016; HH-livre-eapig 0,017 (razão 1,10); cercadinho-meio 0,025 (1,61);
#   fidelidade a 1%: 83% / 83% / 77%. MCQA (50 frases): EAP-IG 0,090; livre 0,099 (1,10); cercadinho-meio 0,073 (0,81).
#
# ───────────────────────── PREVISÕES (escritas ANTES de rodar; não mudam depois) ─────────────────────────
#  B0  O modelo do MIB roda mesmo em bfloat16 (a GPU é A100/H100). Se não, este NÃO é o teste bf16.
#  B1  EAP-IG oficial, IOI, CMD médio dos 5 grupos em bf16: entre 0,010 e 0,030.
#  B2  IOI: CMD(HH-livre-eapig) / CMD(EAP-IG) entre 0,90 e 1,40.
#  B3  IOI, fidelidade com 1% das arestas: |HH-livre-eapig − EAP-IG| ≤ 8 pontos percentuais (média dos grupos).
#  B4  IOI: o cercadinho continua pior que o oficial: CMD(HH-cercadinho-meio) / CMD(EAP-IG) ≥ 1,20.
#  B5  MCQA: CMD(HH-livre-eapig) / CMD(EAP-IG) entre 0,80 e 1,50.
#  B6  MCQA: CMD(HH-cercadinho-meio) / CMD(EAP-IG) ≤ 1,10 (o cercadinho não perde no MCQA).
#  B7  Spearman de |nota| do EAP-IG oficial (bf16) × HH-livre-eapig (float32), mediana dos grupos: q ≥ 0,95, mlp/saída ≥ 0,95, k/v ≥ 0,90.
#  B8  Tempo, MESMA precisão (float32 nos dois, IOI s0): HH / EAP-IG oficial entre 0,40 e 0,75.
#  B9  Memória de pico, float32 nos dois: HH / EAP-IG oficial entre 0,35 e 0,65.
#  B10 HH (float32) × EAP-IG oficial (bf16): razão de tempo entre 0,60 e 1,60 (não tenho opinião forte: o bf16 do oficial tira parte da vantagem).
# Valem os números dos arquivos do Drive/volume; o HeadHunter roda em float32 (como sempre), o MIB (patching e avaliação) em bf16.
# ──────────────────────────────────────────────────────────────────────────────────────────────


# ===================== CÉLULA 1 — instalar (pod) =====================
import os
for d in ("/workspace/headhunter", "/workspace/hf"): os.makedirs(d, exist_ok=True)
os.environ["HF_HOME"] = "/workspace/hf"
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!pip -q install "transformer_lens<4" datasets tabulate matplotlib pandas scipy einops tqdm
!test -d /workspace/MIB-circuit-track || git clone -q https://github.com/hannamw/MIB-circuit-track.git /workspace/MIB-circuit-track
!test -d /workspace/EAP-IG || git clone -q -b MIB https://github.com/hannamw/EAP-IG.git /workspace/EAP-IG
!cd /workspace/EAP-IG && git checkout -q 7af394a5662de8b23ad6154716a0cd3993d447a3 && git log -1 --format="EAP-IG no commit do MIB: %h %s"
!pip -q install --no-deps /workspace/EAP-IG
import sys; sys.path.insert(0, "/workspace/EAP-IG/src"); sys.path.insert(0, "/workspace/MIB-circuit-track")
import eap.attribute; print("eap ok:", eap.attribute.__file__)
import torch; print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NENHUMA", "| capacidade", torch.cuda.get_device_capability() if torch.cuda.is_available() else "-")


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
import headhunter as hh; importlib.reload(hh); print("HeadHunter", hh.VERSAO, "| esquema:", hh.ESQUEMA_IG, "| arquivo de", len(txt.encode()), "bytes")


# ===================== CÉLULA 4 — módulos HeadHunter → MIB + avaliação retomável =====================
import gc, torch, random
# ---- HeadHunter no benchmark MIB (trilha de circuitos) ----
# Precisa: o repositório MIB-circuit-track no sys.path e o pacote eap (commit 7af394a, ramo MIB).
import os, random, time, gc, torch
from collections import defaultdict
from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.utils import TASKS_TO_HF_NAMES
from eap.graph import Graph

def itens_mib(m, tarefa, split="train", n=1000):
    # Os mesmos itens que o MIB usa: (frase limpa, frase corrompida, [certo, errado]).
    ds = HFEAPDataset(f"mib-bench/{TASKS_TO_HF_NAMES[tarefa]}", m.tokenizer, split=split, task=tarefa, num_examples=n)
    return [ds[i] for i in range(len(ds))]

def receita_mib(m, itens, nome):
    # O HeadHunter supõe frases do mesmo tamanho e a resposta no fim: separa os itens em grupos por comprimento
    # (mesmos tokens do MIB: BOS + texto). Cada lote do HeadHunter sai de UM grupo.
    grupos = defaultdict(list); fora = 0
    for limpa, corr, lab in itens:
        a = m.to_tokens(limpa, prepend_bos=True)[0]; b = m.to_tokens(corr, prepend_bos=True)[0]
        if len(a) == len(b): grupos[len(a)].append((a, b, int(lab[0]), int(lab[1])))
        else: fora += 1
    def montar(m_, seed, n=40):
        rnd = random.Random(seed); ok = [L for L in grupos if len(grupos[L]) >= n] or [max(grupos, key=lambda L: len(grupos[L]))]
        L = rnd.choices(ok, weights=[len(grupos[x]) for x in ok])[0]; g = grupos[L]
        esc = rnd.sample(g, n) if len(g) >= n else [rnd.choice(g) for _ in range(n)]
        dev = m_.cfg.device
        return dict(tk=torch.stack([e[0] for e in esc]).to(dev), tkc=torch.stack([e[1] for e in esc]).to(dev),
                    alvo=torch.tensor([e[2] for e in esc], device=dev), errado=torch.tensor([e[3] for e in esc], device=dev))
    hh.RECEITAS[nome] = montar
    info = dict(itens=len(itens), fora=fora, grupos={L: len(v) for L, v in sorted(grupos.items())})
    print(f"{nome}: {len(itens)} itens, {fora} descartados (limpa e corrompida com tamanhos diferentes), "
          f"grupos por comprimento: {info['grupos']}")
    return info

def nome_aresta(m, a):
    nH = m.cfg.n_heads; u, v, p = a
    nome = lambda no: {"E": "input", "O": "logits"}.get(no[0]) or (f"a{no[1] // nH}.h{no[1] % nH}" if no[0] == "H" else f"m{no[1]}")
    return f"{nome(u)}->{nome(v)}" + (f"<{p}>" if p else "")

def notas_grafo_todo(m, nome_tarefa, n=40, n_lote=10, ig_passos=5, semente=5000):
    # Nota IG do HeadHunter para TODAS as arestas do modelo (mesmas frases 5000+ que o analisar usa para as arestas).
    nL, nH = m.cfg.n_layers, m.cfg.n_heads
    ar = hh.arestas_partes(m, range(nL * nH), range(nL))
    lot = [hh.tarefa(m, nome_tarefa, semente + b, n_lote) for b in range(n // n_lote)]; Cs = [hh.caches(m, T) for T in lot]
    nota = hh.notas_arestas(m, lot, Cs, ar, ig_passos)
    del Cs; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return ar, nota

def grafo_mib(m, ar, nota, cabecas=None):
    # Nota final = posição no ranking (maior = mais importante), então --absolute não muda nada.
    # cabecas=None -> só a nota IG (o "grafo livre"); com cabecas -> as arestas do cercadinho
    # (cabeças do circuito + todos os MLPs) vêm primeiro, depois o resto, cada parte pela |nota|.
    nL = m.cfg.n_layers; g = Graph.from_model(m)
    dentro = set(hh.arestas_partes(m, cabecas, range(nL))) if cabecas is not None else set()
    ordem = sorted(ar, key=lambda a: (a not in dentro, -abs(nota[a]))); N = len(ordem)
    assert N == len(g.edges), f"o HeadHunter vê {N} arestas, o grafo do MIB tem {len(g.edges)}"
    g.scores[:] = 0
    for r, a in enumerate(ordem): g.edges[nome_aresta(m, a)].score = float(N - r)
    return g, len(dentro)

def hh_mib(m, tarefa, nome_modelo, pasta, n_itens=1000, verbose=True, **kw):
    # 1) dados de treino do MIB; 2) circuito de cabeças do HeadHunter; 3) notas de todas as arestas;
    # 4) salva 2 arquivos de notas: HH-cercadinho (o método) e HH-livre (sem a etapa das cabeças, para comparar).
    nome = f"MIB_{tarefa}"; t0 = time.time()
    info = receita_mib(m, itens_mib(m, tarefa, "train", n_itens), nome)
    r = hh.analisar(m, nome, podar=False, verbose=False, **kw); cab = [int(c) for c in r["circuito"]["cabecas"]]
    t_cab = time.time() - t0
    ar, nota = notas_grafo_todo(m, nome); t_tot = time.time() - t0
    saida = {}
    for metodo, cabs in (("HH-cercadinho", cab), ("HH-livre", None)):
        g, _ = grafo_mib(m, ar, nota, cabs)
        p = os.path.join(pasta, f"{metodo}_patching_edge", f"{tarefa.replace('_', '-')}_{nome_modelo}")
        os.makedirs(p, exist_ok=True); g.to_json(f"{p}/importances.json"); saida[metodo] = f"{p}/importances.json"
    res = dict(tarefa=tarefa, modelo=nome_modelo, cabecas=len(cab), fid_cabecas=r["circuito"]["fidelidade"],
               arestas_total=len(ar), arestas_cercadinho=len(hh.arestas_partes(m, cab, range(m.cfg.n_layers))),
               tempo_cabecas=round(t_cab, 1), tempo_total=round(t_tot, 1), arquivos=saida, grupos=info["grupos"])
    if verbose:
        print(f">>> {nome_modelo} / {tarefa}: {len(cab)} cabeças (fidelidade {res['fid_cabecas']:.0%}) | cercadinho = "
              f"{res['arestas_cercadinho']} de {len(ar)} arestas ({res['arestas_cercadinho'] / len(ar):.1%}) | "
              f"cabeças em {t_cab:.0f} s, tudo em {t_tot:.0f} s")
    return res

# ---- Rodada 2 do MIB: avaliação dentro do notebook (mesma função do run_evaluation.py), retomável ----
import json, os, random, time, torch
from functools import partial
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from eap.graph import Graph
from eap.attribute import attribute
from MIB_circuit_track.dataset import HFEAPDataset, collate_EAP
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.evaluation import evaluate_area_under_curve

PASTA = "/workspace/headhunter/mib2_gpt2_nao_usado"       # (só do kit do GPT-2; o Qwen usa Q_PASTA, na célula 5)
PLACAR = f"{PASTA}/placar.json"
os.makedirs(PASTA, exist_ok=True)

def modelo_mib(nome="gpt2-small"):
    # igual ao run_evaluation.py / run_attribution.py do MIB
    mm = HookedTransformer.from_pretrained(nome)
    mm.cfg.use_split_qkv_input = True; mm.cfg.use_attn_result = True; mm.cfg.use_hook_mlp_in = True
    mm.cfg.ungroup_grouped_query_attention = True
    return mm

def ler_placar():
    return json.load(open(PLACAR)) if os.path.exists(PLACAR) else {}

def arq(nome): return f"{PASTA}/{nome}/importances.json"

def avaliar(mm, nome, dl_val, absoluto=True):
    # mesma chamada do run_evaluation.py: evaluate_area_under_curve(..., partial(metrica, mean=False, loss=False))
    P = ler_placar()
    if nome in P: print(f"(já avaliado) {nome}: CMD {P[nome]['CMD']:.3f}"); return P[nome]
    g = Graph.from_json(arq(nome)); met = get_metric("logit_diff", "ioi", mm.tokenizer, mm); t0 = time.time()
    out = evaluate_area_under_curve(mm, g, dl_val, partial(met, mean=False, loss=False), quiet=True, level="edge", absolute=absoluto)
    P = ler_placar(); P[nome] = dict(CMD=out[2], CPR=out[1], fid=[round(x, 3) for x in out[4]], tempo=round(time.time() - t0))
    json.dump(P, open(PLACAR, "w"), indent=1); print(f"{nome}: CMD {out[2]:.3f} | CPR {out[1]:.3f} | {P[nome]['tempo']} s")
    return P[nome]

def eapig(mm, itens, nome, bs=20):
    # EAP-IG-inputs oficial (mesmos argumentos do run_attribution.py) num subconjunto escolhido de frases
    if os.path.exists(arq(nome)): print("(já existe)", nome); return
    dl = DataLoader(itens, batch_size=bs, collate_fn=collate_EAP); g = Graph.from_model(mm)
    met = get_metric("logit_diff", "ioi", mm.tokenizer, mm); t0 = time.time()
    with torch.enable_grad():   # o HeadHunter desliga o gradiente globalmente; o EAP-IG precisa dele
        attribute(mm, g, dl, partial(met, mean=True, loss=True), "EAP-IG-inputs", "patching", ig_steps=5, intervention_dataloader=dl)
    os.makedirs(os.path.dirname(arq(nome)), exist_ok=True); g.to_json(arq(nome)); print(f"{nome}: {time.time() - t0:.0f} s")

def salvar_hh(m, nome, ar, chave):
    if os.path.exists(arq(nome)): print("(já existe)", nome); return
    g = Graph.from_model(m); ordem = sorted(ar, key=chave); N = len(ordem); g.scores[:] = 0
    for k, a in enumerate(ordem): g.edges[nome_aresta(m, a)].score = float(N - k)
    os.makedirs(os.path.dirname(arq(nome)), exist_ok=True); g.to_json(arq(nome)); print("salvo:", nome)

def mostrar_placar():
    pcts = ["0.1%", "0.2%", "0.5%", "1%", "2%", "5%", "10%", "20%", "50%", "100%"]
    for k, d in sorted(ler_placar().items(), key=lambda x: x[1]["CMD"]):
        print(f"{k:22s} CMD {d['CMD']:.3f} | CPR {d['CPR']:.3f} | " + " ".join(f"{p}:{x:.0%}" for p, x in zip(pcts, d["fid"])))


# ===================== CÉLULA 5 — definições do Qwen no MIB, versão bf16 (não carrega modelo) =====================
import os, json, time, gc, random, io, contextlib
from collections import defaultdict
from functools import partial
import numpy as np, torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from eap.graph import Graph
from eap.attribute import attribute
from eap.utils import tokenize_plus
from MIB_circuit_track.dataset import HFEAPDataset, collate_EAP
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.evaluation import evaluate_area_under_curve
from MIB_circuit_track.utils import TASKS_TO_HF_NAMES

assert hh.VERSAO == "0.11.3", f"HeadHunter {hh.VERSAO}: este notebook é da v0.11.3 (rode as células 2A-2C e 3)"
Q_MODELO = "Qwen/Qwen2.5-0.5B"
Q_PASTA = "/workspace/headhunter/mib2_qwen_bf16"               # tudo no volume do pod (sobrevive a reiniciar o kernel; faça backup antes de apagar o pod)
Q_N_VAL = 1000                                                   # frases de validação (a célula 6 projeta o tempo; ajuste aqui)
Q_SEM = {"s0": 5000, "s1": 21000, "s2": 22000, "s3": 23000, "s4": 24000}   # sementes dos 5 grupos de 40 frases (s0-s2 = as do Colab)
Q_CFG = {"ioi":  dict(n_treino=1000, n_lote=10, bs_attr=10, bs_val=20),
         "mcqa": dict(n_treino=None, n_lote=5,  bs_attr=5,  bs_val=10)}   # n_treino=None: o treino inteiro (como o MIB)
Q_HH = ("HH-cercadinho-meio", "HH-cercadinho-eapig", "HH-livre-eapig")
Q_METODOS = Q_HH + ("EAP-IG",)
os.makedirs(Q_PASTA, exist_ok=True)

def q_dtype():
    # MIB usa bfloat16; a T4 (capacidade 7.5) não tem bf16 nativo e fica lenta -> float32 lá. A100 (8.0+): bfloat16.
    return torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float32

def q_modelo(dtype=None):
    # igual ao run_evaluation.py / run_attribution.py do MIB (Qwen2.5: eager, GQA desagrupado), com a precisão acima
    mm = HookedTransformer.from_pretrained(Q_MODELO, attn_implementation="eager", torch_dtype=dtype or q_dtype())
    mm.cfg.use_split_qkv_input = True; mm.cfg.use_attn_result = True; mm.cfg.use_hook_mlp_in = True
    mm.cfg.ungroup_grouped_query_attention = True
    return mm

class _Med:
    # tempo e memória de pico (acima do que já estava na GPU antes de começar)
    def __enter__(self):
        if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); self.base = torch.cuda.memory_allocated()
        self.t0 = time.time(); return self
    def __exit__(self, *a):
        if torch.cuda.is_available(): torch.cuda.synchronize(); self.pico = (torch.cuda.max_memory_allocated() - self.base) / 2**20
        else: self.pico = None
        self.seg = time.time() - self.t0

def q_arq(tarefa, nome): return f"{Q_PASTA}/{tarefa}/{nome}/importances.json"
def q_json(tarefa, nome):
    return f"{Q_PASTA}/{tarefa}/{nome}.json"
def q_ler(tarefa, nome, padrao=None):
    p = q_json(tarefa, nome); return json.load(open(p)) if os.path.exists(p) else ({} if padrao is None else padrao)
def q_gravar(tarefa, nome, dados):
    os.makedirs(f"{Q_PASTA}/{tarefa}", exist_ok=True); json.dump(dados, open(q_json(tarefa, nome), "w"), indent=1)

def q_val(mm, tarefa, n=None):
    ds = HFEAPDataset(f"mib-bench/{TASKS_TO_HF_NAMES[tarefa]}", mm.tokenizer, split="validation", task=tarefa)
    if n and n < len(ds): ds.head(n)
    return ds

def q_itens(m, tarefa):
    return itens_mib(m, tarefa, "train", Q_CFG[tarefa]["n_treino"])

def q_receita(m, tarefa, itens):
    # registra a receita do HeadHunter (grupos por comprimento) e avisa se há poucos grupos grandes
    info = receita_mib(m, itens, f"MIB_{tarefa}"); g = info["grupos"]
    L = [l for l, v in g.items() for _ in range(v)]; cob = sum(v for v in g.values() if v >= 40) / max(1, info["itens"])
    print(f"   {tarefa}: comprimento em tokens: média {np.mean(L):.0f}, máx {max(L)} | grupos com ≥ 40 frases cobrem {cob:.0%} dos itens")
    if cob < 0.5: print("   ⚠️ poucos grupos grandes: o HeadHunter vai sortear frases repetidas dentro de grupos pequenos. Me mande esta linha.")
    return info

def q_itens_do_hh(m, itens, semente, n=40, n_lote=10):
    # As MESMAS frases que o HeadHunter sorteia (mesmos grupos e mesmo sorteio da receita_mib), como itens do MIB.
    grupos = defaultdict(list)
    for it in itens:
        a = m.to_tokens(it[0], prepend_bos=True)[0]; b = m.to_tokens(it[1], prepend_bos=True)[0]
        if len(a) == len(b): grupos[len(a)].append(it)
    out = []
    for b in range(n // n_lote):
        rnd = random.Random(semente + b); ok = [L for L in grupos if len(grupos[L]) >= n_lote] or [max(grupos, key=lambda L: len(grupos[L]))]
        L = rnd.choices(ok, weights=[len(grupos[x]) for x in ok])[0]; g = grupos[L]
        out += rnd.sample(g, n_lote) if len(g) >= n_lote else [rnd.choice(g) for _ in range(n_lote)]
    return out

def q_conferir_grafo(m, mm):
    # O HeadHunter (sem desagrupar o GQA) e o MIB (desagrupado) têm que ter EXATAMENTE as mesmas arestas, com os mesmos nomes.
    nL, nH = m.cfg.n_layers, m.cfg.n_heads
    hh_nomes = {nome_aresta(m, a) for a in hh.arestas_partes(m, range(nL * nH), range(nL))}
    mib_nomes = set(Graph.from_model(mm).edges)
    assert hh_nomes == mib_nomes, f"arestas diferentes: só HH {len(hh_nomes - mib_nomes)}, só MIB {len(mib_nomes - hh_nomes)}"
    print(f"   ✅ grafo: {len(mib_nomes)} arestas, mesmos nomes no HeadHunter e no MIB")
    return len(mib_nomes)

def q_conferir_tokens(m, mm, itens, k=20):
    # Os mesmos tokens que o MIB usa (tokenize_plus: BOS + texto, padding à direita) e os do HeadHunter.
    for j in range(min(k, len(itens))):
        for txt in itens[j][:2]:
            a = m.to_tokens(txt, prepend_bos=True)[0].cpu()
            t, mask, lens, _ = tokenize_plus(mm, [txt]); b = t[0, :int(lens[0])].cpu()
            assert a.shape == b.shape and bool((a == b).all()), f"tokens diferentes na frase {j}: HH {a.tolist()[:8]} × MIB {b.tolist()[:8]}"
    print(f"   ✅ tokens: {min(k, len(itens))} pares conferidos, HeadHunter = MIB. Começo da 1ª frase:",
          [mm.tokenizer.decode(x) for x in tokenize_plus(mm, [itens[0][0]])[0][0, :4].tolist()])

def q_sonda_tempo(mm, tarefa, n=40):
    # Avalia um grafo aleatório em n frases (curva inteira, como o MIB) e projeta o tempo por método para vários Q_N_VAL.
    ds = q_val(mm, tarefa, n); dl = ds.to_dataloader(batch_size=Q_CFG[tarefa]["bs_val"])
    g = Graph.from_model(mm); g.scores[:] = torch.rand_like(g.scores)
    met = get_metric("logit_diff", tarefa, mm.tokenizer, mm)
    with contextlib.redirect_stdout(io.StringIO()):
        with _Med() as md: evaluate_area_under_curve(mm, g, dl, partial(met, mean=False, loss=False), quiet=True, level="edge", absolute=True)
    por_frase = md.seg / len(ds)
    print(f"   {tarefa}: {md.seg:.0f} s para {len(ds)} frases ({por_frase:.2f} s/frase) | projeção por método: "
          + ", ".join(f"{nv} frases ≈ {por_frase * nv / 60:.0f} min" for nv in (200, 500, 1000))
          + f" | 20 avaliações (5 grupos × 4 métodos) com {Q_N_VAL} frases ≈ {20 * por_frase * Q_N_VAL / 3600:.1f} h")
    return por_frase

def q_cabecas(m, tarefa, itens):
    # Etapa das cabeças do HeadHunter (circuito de 90%), uma vez por tarefa; guarda no Drive.
    info = q_receita(m, tarefa, itens); d = q_ler(tarefa, "cabecas")
    if d: print(f"   (cabeças já feitas) {tarefa}: {len(d['cabecas'])} cabeças, fidelidade {d['fid']:.0%}"); return d["cabecas"], d
    with _Med() as md: r = hh.analisar(m, f"MIB_{tarefa}", podar=False, verbose=False)
    c = r["circuito"]; d = dict(cabecas=[int(x) for x in c["cabecas"]], fid=float(c["fidelidade"]), atingiu=bool(c["atingiu"]),
                                nucleo=len(r["nucleo"]["cabecas"]), mlps_nucleo=r["nucleo"]["mlps"], promovidas=c["promovidas"],
                                revertido=bool(c["revertido"]), seg=round(md.seg, 1), pico_mb=md.pico, itens=info["itens"], grupos=info["grupos"])
    q_gravar(tarefa, "cabecas", d)
    print(f"   {tarefa}: {len(d['cabecas'])}/{m.cfg.n_layers * m.cfg.n_heads} cabeças (fidelidade {d['fid']:.0%}, núcleo {d['nucleo']}) em {md.seg:.0f} s")
    return d["cabecas"], d

def q_salvar(m, tarefa, nome, ar, chave):
    if os.path.exists(q_arq(tarefa, nome)): return
    g = Graph.from_model(m); ordem = sorted(ar, key=chave); N = len(ordem)
    assert N == len(g.edges), f"o HeadHunter vê {N} arestas, o grafo do MIB tem {len(g.edges)}"
    g.scores[:] = 0
    for k, a in enumerate(ordem): g.edges[nome_aresta(m, a)].score = float(N - k)
    os.makedirs(os.path.dirname(q_arq(tarefa, nome)), exist_ok=True); g.to_json(q_arq(tarefa, nome))

def q_gerar_hh(m, tarefa, cab, grupo):
    nome = f"MIB_{tarefa}"; nL = m.cfg.n_layers; c = Q_CFG[tarefa]; sem = Q_SEM[grupo]
    plano = (("meio", ("HH-cercadinho-meio",)), ("eapig", ("HH-cercadinho-eapig", "HH-livre-eapig")))
    dentro = set(hh.arestas_partes(m, cab, range(nL))); tempos = q_ler(tarefa, "tempos")
    try:
        for esquema, nomes in plano:
            if all(os.path.exists(q_arq(tarefa, f"{n}-{grupo}")) for n in nomes): continue
            hh.ESQUEMA_IG = esquema
            with _Med() as md: ar, nt = notas_grafo_todo(m, nome, n=40, n_lote=c["n_lote"], semente=sem)
            tempos[f"HH-{esquema}-{grupo}"] = dict(seg=round(md.seg, 1), pico_mb=md.pico)
            for n in nomes:
                if "cercadinho" in n: q_salvar(m, tarefa, f"{n}-{grupo}", ar, lambda a, nt=nt: (a not in dentro, -abs(nt[a])))
                else: q_salvar(m, tarefa, f"{n}-{grupo}", ar, lambda a, nt=nt: -abs(nt[a]))
            q_gravar(tarefa, "tempos", tempos); print(f"   {tarefa} {grupo} HH ({esquema}): {md.seg:.0f} s, pico {md.pico if md.pico is None else round(md.pico)} MB → {', '.join(nomes)}")
            del ar, nt; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    finally:
        hh.ESQUEMA_IG = "meio"

def q_gerar_eapig(mm, tarefa, itens, grupo):
    nome = f"EAP-IG-{grupo}"
    if os.path.exists(q_arq(tarefa, nome)): print(f"   (já existe) {tarefa} {nome}"); return
    c = Q_CFG[tarefa]; it = q_itens_do_hh(mm, itens, Q_SEM[grupo], 40, c["n_lote"])
    dl = DataLoader(it, batch_size=c["bs_attr"], collate_fn=collate_EAP); g = Graph.from_model(mm)
    met = get_metric("logit_diff", tarefa, mm.tokenizer, mm); tempos = q_ler(tarefa, "tempos")
    with _Med() as md:
        with torch.enable_grad():   # o HeadHunter desliga o gradiente globalmente; o EAP-IG precisa dele
            attribute(mm, g, dl, partial(met, mean=True, loss=True), "EAP-IG-inputs", "patching", ig_steps=5, intervention_dataloader=dl)
    os.makedirs(os.path.dirname(q_arq(tarefa, nome)), exist_ok=True); g.to_json(q_arq(tarefa, nome))
    tempos[nome] = dict(seg=round(md.seg, 1), pico_mb=md.pico); q_gravar(tarefa, "tempos", tempos)
    print(f"   {tarefa} {grupo} EAP-IG oficial: {md.seg:.0f} s, pico {md.pico if md.pico is None else round(md.pico)} MB")
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

def q_corr_escores(tarefa, grupos):
    # Previsão P1: Spearman entre |nota do EAP-IG oficial| e a posição do HH-livre-eapig, por tipo de aresta.
    res = {}
    for gr in grupos:
        a, b = q_arq(tarefa, f"EAP-IG-{gr}"), q_arq(tarefa, f"HH-livre-eapig-{gr}")
        if not (os.path.exists(a) and os.path.exists(b)): continue
        go, gh = Graph.from_json(a), Graph.from_json(b); idx = defaultdict(lambda: ([], []))
        for nm, e in go.edges.items():
            if nm == "input->m0": continue   # o HeadHunter põe esta aresta no topo de propósito (o EAP a subestima): fora da comparação
            cl = "q" if nm.endswith("<q>") else "k/v" if nm.endswith(("<k>", "<v>")) else "mlp/saída"
            idx[cl][0].append(e.matrix_index[0]); idx[cl][1].append(e.matrix_index[1])
        res[gr] = {cl: float(spearmanr(go.scores[r, c].abs().float().numpy(), gh.scores[r, c].float().numpy())[0])
                   for cl, (r, c) in ((k, (torch.tensor(v[0]), torch.tensor(v[1]))) for k, v in idx.items())}
        print(f"   {tarefa} {gr}: Spearman HH-livre-eapig × EAP-IG oficial → " + " | ".join(f"{k} {v:.3f}" for k, v in res[gr].items()))
    q_gravar(tarefa, "spearman", res); return res

def q_avaliar(mm, tarefa, nome, dl_val, n_val):
    chave = f"{nome}@{n_val}"; P = q_ler(tarefa, "placar")
    if chave in P: print(f"(já avaliado) {chave}: CMD {P[chave]['CMD']:.3f}"); return P[chave]
    g = Graph.from_json(q_arq(tarefa, nome)); met = get_metric("logit_diff", tarefa, mm.tokenizer, mm)
    with contextlib.redirect_stdout(io.StringIO()):
        with _Med() as md:
            out = evaluate_area_under_curve(mm, g, dl_val, partial(met, mean=False, loss=False), quiet=True, level="edge", absolute=True)
    P = q_ler(tarefa, "placar"); P[chave] = dict(CMD=out[2], CPR=out[1], fid=[round(x, 3) for x in out[4]], seg=round(md.seg), dtype=str(mm.cfg.dtype))
    q_gravar(tarefa, "placar", P); print(f"{chave}: CMD {out[2]:.3f} | CPR {out[1]:.3f} | fid 1% {out[4][3]:.0%} | {md.seg:.0f} s"); return P[chave]

def q_tabela(tarefa, grupos, n_val):
    P = q_ler(tarefa, "placar"); pcts = ["0.1%", "0.2%", "0.5%", "1%", "2%", "5%", "10%", "20%", "50%", "100%"]
    print(f"\n=== {tarefa}: {Q_MODELO}, {n_val} frases de validação, CMD (menor = melhor) ===")
    print(f"{'método':22s}" + "".join(f"{g:>9s}" for g in grupos) + "   média ± dp |  fid. 1% (média) | fid. 2% (média)")
    for met in Q_METODOS:
        v = [P.get(f"{met}-{g}@{n_val}") for g in grupos]; ok = [x for x in v if x]
        if not ok: continue
        cmd = np.array([x["CMD"] for x in ok])
        print(f"{met:22s}" + "".join(f"{x['CMD']:9.3f}" if x else f"{'-':>9s}" for x in v)
              + f"   {cmd.mean():.3f} ± {cmd.std():.3f} |      {np.mean([x['fid'][3] for x in ok]):.0%}        |      {np.mean([x['fid'][4] for x in ok]):.0%}")
    return P

def q_custo_fp32(mm32, tarefa, itens, grupo="s0"):
    # EAP-IG oficial em float32 (mesma precisão do HeadHunter): só tempo e memória; não grava importâncias.
    nome = f"EAP-IG-fp32-{grupo}"; tempos = q_ler(tarefa, "tempos")
    if nome in tempos: print(f"   (já medido) {nome}: {tempos[nome]['seg']} s, pico {tempos[nome]['pico_mb']} MB"); return tempos[nome]
    c = Q_CFG[tarefa]; it = q_itens_do_hh(mm32, itens, Q_SEM[grupo], 40, c["n_lote"])
    dl = DataLoader(it, batch_size=c["bs_attr"], collate_fn=collate_EAP); g = Graph.from_model(mm32)
    met = get_metric("logit_diff", tarefa, mm32.tokenizer, mm32)
    with _Med() as md:
        with torch.enable_grad():
            attribute(mm32, g, dl, partial(met, mean=True, loss=True), "EAP-IG-inputs", "patching", ig_steps=5, intervention_dataloader=dl)
    tempos = q_ler(tarefa, "tempos"); tempos[nome] = dict(seg=round(md.seg, 1), pico_mb=md.pico); q_gravar(tarefa, "tempos", tempos)
    print(f"   {tarefa} {grupo} EAP-IG oficial em float32: {md.seg:.0f} s, pico {md.pico if md.pico is None else round(md.pico)} MB"); return tempos[nome]

def q_previsoes(tarefa, grupos, n_val):
    # Confere as previsões B1-B10 que já têm dados. ✅ acertou, ❌ errou, ⏳ falta dado.
    P = q_ler(tarefa, "placar"); T = q_ler(tarefa, "tempos"); S = q_ler(tarefa, "spearman")
    f = lambda met, g: P.get(f"{met}-{g}@{n_val}")
    comp = [g for g in grupos if all(f(m_, g) for m_ in Q_METODOS)]; r = []
    def add(txt, ok): r.append(f"{'⏳' if ok is None else '✅' if ok else '❌'} {txt}")
    if len(comp) >= 3:
        cm = lambda met: float(np.mean([f(met, g)["CMD"] for g in comp])); fd = lambda met: float(np.mean([f(met, g)["fid"][3] for g in comp]))
        rl, rc = cm("HH-livre-eapig") / cm("EAP-IG"), cm("HH-cercadinho-meio") / cm("EAP-IG")
        if tarefa == "ioi":
            add(f"B1 CMD do EAP-IG oficial em bf16 = {cm('EAP-IG'):.3f} (previsto 0,010-0,030)", 0.010 <= cm("EAP-IG") <= 0.030)
            add(f"B2 CMD livre-eapig / EAP-IG = {rl:.2f} (previsto 0,90-1,40)", 0.90 <= rl <= 1.40)
            dif = 100 * abs(fd("HH-livre-eapig") - fd("EAP-IG")); add(f"B3 fidelidade a 1%: livre-eapig {fd('HH-livre-eapig'):.0%} × EAP-IG {fd('EAP-IG'):.0%} (diferença {dif:.1f} pontos; previsto ≤ 8)", dif <= 8)
            add(f"B4 CMD cercadinho-meio / EAP-IG = {rc:.2f} (previsto ≥ 1,20)", rc >= 1.20)
        else:
            add(f"B5 CMD livre-eapig / EAP-IG = {rl:.2f} (previsto 0,80-1,50)", 0.80 <= rl <= 1.50)
            add(f"B6 CMD cercadinho-meio / EAP-IG = {rc:.2f} (previsto ≤ 1,10)", rc <= 1.10)
    else: add("B1-B6 (precisam de ≥ 3 grupos avaliados nos 4 métodos)", None)
    if S:
        med = lambda cl: float(np.median([S[g][cl] for g in S])); q, mo, kv = med("q"), med("mlp/saída"), med("k/v")
        add(f"B7 Spearman (mediana): q {q:.3f} (≥ 0,95), mlp/saída {mo:.3f} (≥ 0,95), k/v {kv:.3f} (≥ 0,90)", q >= 0.95 and mo >= 0.95 and kv >= 0.90)
    if tarefa == "ioi":
        g = "s0"
        if f"HH-meio-{g}" in T and f"EAP-IG-fp32-{g}" in T:
            t8 = T[f"HH-meio-{g}"]["seg"] / T[f"EAP-IG-fp32-{g}"]["seg"]; add(f"B8 tempo HH / EAP-IG, ambos float32 = {t8:.2f} (previsto 0,40-0,75)", 0.40 <= t8 <= 0.75)
            if T[f"HH-meio-{g}"]["pico_mb"] is not None and T[f"EAP-IG-fp32-{g}"]["pico_mb"]:
                t9 = T[f"HH-meio-{g}"]["pico_mb"] / T[f"EAP-IG-fp32-{g}"]["pico_mb"]; add(f"B9 memória de pico HH / EAP-IG, ambos float32 = {t9:.2f} (previsto 0,35-0,65)", 0.35 <= t9 <= 0.65)
        else: add("B8-B9 (rode a célula 11)", None)
        if f"HH-meio-{g}" in T and f"EAP-IG-{g}" in T:
            t10 = T[f"HH-meio-{g}"]["seg"] / T[f"EAP-IG-{g}"]["seg"]; add(f"B10 tempo HH (float32) / EAP-IG oficial (bf16) = {t10:.2f} (previsto 0,60-1,60)", 0.60 <= t10 <= 1.60)
    print(f"\nprevisões ({tarefa}):"); [print("  " + x) for x in r]; return r


# ===================== CÉLULA 6 — carregar os dois modelos e conferir tudo (sem gastar tempo de avaliação) =====================
# Rode UMA vez por sessão. Deixa `m` (HeadHunter, float32, GQA agrupado) e `mm` (MIB, GQA desagrupado) na memória.
gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
m = hh.carregar(Q_MODELO)
mm = q_modelo()
print(f"MIB: {Q_MODELO} em {mm.cfg.dtype}, GQA {mm.cfg.n_heads} cabeças / {mm.cfg.n_key_value_heads} kv | HeadHunter: {next(m.parameters()).dtype}")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}, memória usada: {torch.cuda.memory_allocated() / 2**30:.1f} GB")
print("B0:", "✅ bfloat16 (A100/H100)" if mm.cfg.dtype == torch.bfloat16 else f"❌ o modelo do MIB está em {mm.cfg.dtype}: ESTE NÃO É O TESTE BF16 (use uma A100/H100)")
q_conferir_grafo(m, mm)
Q_ITENS = {}
for t in ("ioi", "mcqa"):
    Q_ITENS[t] = q_itens(m, t); q_receita(m, t, Q_ITENS[t]); q_conferir_tokens(m, mm, Q_ITENS[t])
    q_sonda_tempo(mm, t)
print("\nSe algo acima deu ❌/AssertionError ou ⚠️, cole aqui antes de seguir. Para mudar o nº de frases de validação: Q_N_VAL = ... (célula 5).")


# ===================== CÉLULA 7 (IOI) — atribuições: HeadHunter (3 grafos por grupo; 5 grupos) e EAP-IG oficial, nas MESMAS frases (retomável) =====================
# Cada arquivo vai direto para o volume do pod; rodar de novo pula o que já existe.
tarefa = "ioi"
grupos = ("s0", "s1", "s2", "s3", "s4")
cab, _ = q_cabecas(m, tarefa, Q_ITENS[tarefa])
for g in grupos:
    q_gerar_hh(m, tarefa, cab, g)
    q_gerar_eapig(mm, tarefa, Q_ITENS[tarefa], g)
q_corr_escores(tarefa, grupos)


# ===================== CÉLULA 8 (IOI) — avaliação oficial do MIB + tabela + previsões (retomável; o grupo s0 primeiro) =====================
tarefa = "ioi"
grupos = ("s0", "s1", "s2", "s3", "s4")
ds = q_val(mm, tarefa, Q_N_VAL); dl = ds.to_dataloader(batch_size=Q_CFG[tarefa]["bs_val"]); print(f"validação: {len(ds)} frases")
for g in grupos:
    for met in Q_METODOS:
        if os.path.exists(q_arq(tarefa, f"{met}-{g}")): q_avaliar(mm, tarefa, f"{met}-{g}", dl, len(ds))
    q_tabela(tarefa, [x for x in grupos if x <= g], len(ds))
q_previsoes(tarefa, grupos, len(ds))


# ===================== CÉLULA 9 (MCQA) — atribuições: HeadHunter (3 grafos por grupo; 5 grupos) e EAP-IG oficial, nas MESMAS frases (retomável) =====================
# Cada arquivo vai direto para o volume do pod; rodar de novo pula o que já existe.
tarefa = "mcqa"
grupos = ("s0", "s1", "s2", "s3", "s4")
cab, _ = q_cabecas(m, tarefa, Q_ITENS[tarefa])
for g in grupos:
    q_gerar_hh(m, tarefa, cab, g)
    q_gerar_eapig(mm, tarefa, Q_ITENS[tarefa], g)
q_corr_escores(tarefa, grupos)


# ===================== CÉLULA 10 (MCQA) — avaliação oficial do MIB + tabela + previsões (retomável; o grupo s0 primeiro) =====================
tarefa = "mcqa"
grupos = ("s0", "s1", "s2", "s3", "s4")
ds = q_val(mm, tarefa, Q_N_VAL); dl = ds.to_dataloader(batch_size=Q_CFG[tarefa]["bs_val"]); print(f"validação: {len(ds)} frases")
for g in grupos:
    for met in Q_METODOS:
        if os.path.exists(q_arq(tarefa, f"{met}-{g}")): q_avaliar(mm, tarefa, f"{met}-{g}", dl, len(ds))
    q_tabela(tarefa, [x for x in grupos if x <= g], len(ds))
q_previsoes(tarefa, grupos, len(ds))


# ===================== CÉLULA 11 — custo com a MESMA precisão: EAP-IG oficial em float32 (IOI, grupo s0; ~5 min) =====================
# O HeadHunter roda em float32; o MIB oficial em bf16. Para o custo ser justo, mede também o EAP-IG oficial em float32.
mm32 = q_modelo(torch.float32)
q_custo_fp32(mm32, "ioi", Q_ITENS["ioi"], "s0")
del mm32; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
q_previsoes("ioi", ("s0", "s1", "s2", "s3", "s4"), Q_N_VAL)


# ===================== CÉLULA 12 — resumo final: tabelas, previsões e custo (cole TUDO que esta célula imprimir no chat) =====================
G5 = ("s0", "s1", "s2", "s3", "s4"); resumo = {}
for tarefa in ("ioi", "mcqa"):
    n_val = Q_N_VAL if tarefa == "ioi" else len(q_val(mm, tarefa, Q_N_VAL))
    q_tabela(tarefa, G5, n_val); q_previsoes(tarefa, G5, n_val); resumo[tarefa] = dict(n_val=n_val, placar=q_ler(tarefa, "placar"), tempos=q_ler(tarefa, "tempos"), spearman=q_ler(tarefa, "spearman"), cabecas=q_ler(tarefa, "cabecas"))
T = resumo["ioi"]["tempos"]; print("\ncusto (IOI):")
for k in sorted(T): print(f"  {k:18s} {T[k]['seg']:7.1f} s   pico {T[k]['pico_mb'] if T[k]['pico_mb'] is None else round(T[k]['pico_mb'])} MB")
json.dump(resumo, open(f"{Q_PASTA}/resumo_bf16.json", "w"), indent=1); print("\nresumo salvo em", f"{Q_PASTA}/resumo_bf16.json")
