# HeadHunter — pod D (RTX A6000 48 GB): a v0.11.5 na GPU, o tempo no 7B e o CAP do Qwen2.5-7B por diferença de logit. Escrito em 12/11/2026.
#   (E) v0.11.5 (lote de cálculo fundido incorporado na biblioteca) no Qwen2.5-1.5B: tamanho do pedaço 0 (sem fundir) / 20 / 40, 3 repetições, com o EAP-IG oficial ao lado.
#   (F) Tempo e memória no Qwen2.5-7B: EAP-IG oficial (bf16 e fp32) × HeadHunter (padrão, e sem fundir), mais a etapa das cabeças. Terceiro tamanho medido (0,5B, 1,5B, 7B).
#   (G) CAP no Qwen2.5-7B: necessidade (nº de cabeças ablacionadas pela média, em ordem do efeito individual) pela diferença de logit, como foi feito no Pythia-6.9B.
#
# ───────────────────────── PREVISÕES (escritas ANTES de rodar; não mudam depois) ─────────────────────────
#  (E) 1.5B, IOI, grupo s0, as mesmas 40 frases; L0 = LOTE_CALCULO 0, L20, L40; medianas de 3 repetições
#   E1  Spearman mínimo (L20 e L40 × L0) ≥ 0,99999 (é a mesma conta).
#   E2  tempo L20 / L0 entre 0,50 e 0,65 (o protótipo deu 0,58).
#   E3  tempo L40 / L20 entre 0,95 e 1,05: o pedaço fundido só junta lotes de MESMO comprimento e, no s0, as 4 amostras de 10 frases caíram em 2 comprimentos (pod C: [20, 20]), então L20 e L40 dão os mesmos pedaços.
#   E4  pico de memória L40 / L20 entre 0,95 e 1,05 (pelo mesmo motivo).
#   E5  pico de L0 entre 3.000 e 3.700 MB e de L20 entre 5.000 e 6.500 MB (no pod C: 3.350 e 5.642).
#   E6  variação entre repetições (máx − mín)/mediana ≤ 8% em todos os métodos (HH e oficial).
#   E7  L20 / oficial fp32 ≤ 0,65 e L20 / oficial bf16 entre 0,75 e 1,05 (no pod C: 0,52 e 0,88).
#   E8  o oficial em fp32 montado subindo em bf16 e convertendo na GPU dá os mesmos logits que carregado direto em fp32: diferença máxima ≤ 1e-4 (o checkpoint do Qwen2.5 já é bf16).
#   E9  grupo s1 (outras 40 frases, uma medida de cada): Spearman L20/L40 × L0 ≥ 0,99999 e tempo L40 / L0 ≤ 0,90 (confiança ~60%: depende de quantos lotes caem no mesmo comprimento).
#  (F) 7B, IOI, 40 frases (grupos s0 e s1), HeadHunter float32, EAP-IG oficial em bf16 e fp32, lote 10
#   F1  tempo HH padrão / oficial bf16 entre 0,5 e 1,0 (a razão veio caindo com o tamanho: 2,7 no 0,5B, 0,88 no 1,5B).
#   F2  o EAP-IG oficial em fp32 NÃO cabe nos 48 GB no 7B (30 GB de pesos + ~35 GB de ativações; confiança ~75%).
#   F3  o oficial em bf16 cabe, com pico (acima dos pesos) entre 12.000 e 30.000 MB.
#   F4  o HH livre (grafo inteiro) em fp32 cabe no 7B com lotes de 10 frases, pico entre 8.000 e 20.000 MB (confiança ~65%; se não couber, o teste cai para lotes de 5 e isso fica marcado).
#   F5  (só se o oficial fp32 couber) Spearman HH × oficial fp32 ≥ 0,99 em todos os tipos e tempo HH / oficial fp32 ≤ 0,60.
#   F6  Spearman HH padrão × sem fundir (L0) ≥ 0,99999.
#   F7  a etapa das cabeças do HH no 7B leva menos de 240 s.
#  (G) CAP no Qwen2.5-7B (diferença de logit; ablação pela média; 40 frases)
#   G1  p0 − p_todas ≥ 0,10.
#   G2  k para 80% do dano pela diferença de logit entre 5 e 60 (pela probabilidade foi 7; no Pythia por logit foi 50).
#   G3  o Pythia-6.9B (k80 por logit = 50) precisa de pelo menos 1,2× mais cabeças que o Qwen (razão Pythia/Qwen ≥ 1,2).
#   G4  100 cabeças aleatórias causam menos de 10% do dano (logit).
#   G5  acerto binário com todas as cabeças ablacionadas < 90%.
# Honestidade: E2, E7, E9 e F1 decidem a frase de custo no artigo; E9 mostra o quanto o ganho depende da sorte dos comprimentos das frases sorteadas (limite do método atual). F4 diz se o grafo inteiro cabe no 7B.
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


# ===================== CÉLULA 2A — escrever o HeadHunter v0.11.5 no volume (parte A de 3) =====================
# (as 3 células 2A-2C só gravam o headhunter.py v0.11.5 em /workspace/headhunter com %%writefile; omitidas aqui: o arquivo é o mesmo de headhunter_v0.11.5.py, 98.204 bytes)


# ===================== CÉLULA 2B — escrever o HeadHunter v0.11.5 no volume (parte B de 3) =====================
# (as 3 células 2A-2C só gravam o headhunter.py v0.11.5 em /workspace/headhunter com %%writefile; omitidas aqui: o arquivo é o mesmo de headhunter_v0.11.5.py, 98.204 bytes)


# ===================== CÉLULA 2C — escrever o HeadHunter v0.11.5 no volume (parte C de 3) =====================
# (as 3 células 2A-2C só gravam o headhunter.py v0.11.5 em /workspace/headhunter com %%writefile; omitidas aqui: o arquivo é o mesmo de headhunter_v0.11.5.py, 98.204 bytes)


# ===================== CÉLULA 3 — conferir e carregar o HeadHunter =====================
import sys, importlib, os
ARQ = "/workspace/headhunter/headhunter.py"
txt = open(ARQ, encoding="utf-8").read() if os.path.exists(ARQ) else ""
assert 'VERSAO = "0.11.5"' in txt and "def _fundir_lotes" in txt, "HeadHunter ausente ou fora da v0.11.5: rode as células 2A, 2B e 2C (nesta ordem) e esta de novo"
assert len(txt.encode()) == 98204, f"o arquivo tem {len(txt.encode())} bytes e deveria ter 98204: alguma das células 2A-2C foi rodada pela metade, rode as três de novo"
sys.path.insert(0, "/workspace/headhunter")
import headhunter as hh; importlib.reload(hh); print("HeadHunter", hh.VERSAO, "| esquema:", hh.ESQUEMA_IG, "| LOTE_CALCULO", hh.LOTE_CALCULO, "| arquivo de", len(txt.encode()), "bytes")


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


# ===================== CÉLULA 5 — definições do Qwen no MIB (não carrega modelo) =====================
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

assert hh.VERSAO == "0.11.5", f"HeadHunter {hh.VERSAO}: este notebook é da v0.11.5 (rode as células 2A-2C e 3)"
Q_MODELO = "Qwen/Qwen2.5-0.5B"
Q_BASE = "/workspace/headhunter/final_pod_d"                  # tudo no volume do pod (sobrevive a reiniciar o kernel; faça backup antes de apagar o pod)
Q_PASTA = f"{Q_BASE}/qwen0.5b"
Q_N_VAL = 1000                                                   # frases de validação (a célula 6 projeta o tempo; ajuste aqui)
Q_SEM = {"s0": 5000, "s1": 21000, "s2": 22000, "s3": 23000, "s4": 24000}   # sementes dos 5 grupos de 40 frases (s0-s2 = as do Colab)
Q_CFG = {"ioi":  dict(n_treino=1000, n_lote=10, bs_attr=10, bs_val=20),
         "mcqa": dict(n_treino=None, n_lote=5,  bs_attr=5,  bs_val=10)}   # n_treino=None: o treino inteiro (como o MIB)
Q_HH = ("HH-cercadinho-meio", "HH-cercadinho-eapig", "HH-livre-eapig")
Q_METODOS = Q_HH + ("EAP-IG",)
os.makedirs(Q_PASTA, exist_ok=True)
def q_trocar(modelo, tag):
    # muda o modelo e a pasta de resultados (cada modelo tem a sua)
    global Q_MODELO, Q_PASTA
    Q_MODELO = modelo; Q_PASTA = f"{Q_BASE}/{tag}"; os.makedirs(Q_PASTA, exist_ok=True); print("modelo:", Q_MODELO, "| resultados em", Q_PASTA)

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


# ===================== CÉLULA 6 — definições do teste do pod D (não carrega modelo) =====================
import os, json, time, gc, shutil
from collections import defaultdict
import numpy as np, torch
from scipy.stats import spearmanr
assert hh.VERSAO == "0.11.5", f"HeadHunter {hh.VERSAO}: este notebook é da v0.11.5"
T_DEV = "cuda" if torch.cuda.is_available() else "cpu"
mb = lambda x: "-" if x is None else f"{x:,.0f} MB".replace(",", ".")
def t_sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()
def t_limpar(): gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

class _Cron:
    # soma o tempo gasto em hh.gradiente_entradas e hh.caches enquanto o bloco roda (o resto = produto de matrizes + montagem)
    def __enter__(self):
        self.t = defaultdict(float); self.orig = {}
        for nome in ("gradiente_entradas", "caches"):
            f = getattr(hh, nome); self.orig[nome] = f
            def w(*a, _f=f, _n=nome, **k):
                t_sync(); t0 = time.time(); r = _f(*a, **k); t_sync(); self.t[_n] += time.time() - t0; return r
            setattr(hh, nome, w)
        return self
    def __exit__(self, *a):
        for n, f in self.orig.items(): setattr(hh, n, f)

def t_notas(m, tarefa, semente=5000, n=40, n_lote=10, ig_passos=5, esquema="eapig", lc=None):
    # Nota IG de TODAS as arestas (como notas_grafo_todo), com tempo, pico de memória e divisão do tempo.
    # lc = hh.LOTE_CALCULO (0 = sem fundir; 20/40 = pedaço de cálculo; None = o padrão da biblioteca, que dimensiona pela memória livre).
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; ar = hh.arestas_partes(m, range(nL * nH), range(nL))
    velho = (hh.ESQUEMA_IG, hh.LOTE_CALCULO, hh.gradiente_entradas); g0 = hh.gradiente_entradas; vistos = []
    hh.ESQUEMA_IG = esquema
    if lc is not None: hh.LOTE_CALCULO = lc
    def g_log(m_, T, *a, **k): vistos.append(int(T["tk"].shape[0])); return g0(m_, T, *a, **k)
    hh.gradiente_entradas = g_log
    try:
        t_limpar()
        with _Med() as md, _Cron() as cr:
            lot = [hh.tarefa(m, f"MIB_{tarefa}", semente + b, n_lote) for b in range(n // n_lote)]; Cs = [hh.caches(m, T) for T in lot]
            nota = hh.notas_arestas(m, lot, Cs, ar, ig_passos)
            del Cs, lot
        info = dict(seg=round(md.seg, 2), pico_mb=md.pico, grad_s=round(cr.t["gradiente_entradas"], 2), caches_s=round(cr.t["caches"], 2),
                    resto_s=round(md.seg - cr.t["gradiente_entradas"] - cr.t["caches"], 2), tams_chamadas=vistos, lc=("padrão" if lc is None else lc), n_lote=n_lote)
    finally:
        hh.ESQUEMA_IG, hh.LOTE_CALCULO, hh.gradiente_entradas = velho
        m.reset_hooks(); torch.set_grad_enabled(False)
    vec = np.array([nota[a] for a in ar], dtype=np.float64); t_limpar()
    return ar, vec, info

def t_classes(ar): return np.array([0 if a[2] == "q" else 1 if a[2] in ("k", "v") else 2 for a in ar])
def t_spear(x, y, cls):
    # Spearman de |nota| por tipo de aresta (a aresta input->m0 vale infinito no HeadHunter: fica fora)
    ok = np.isfinite(x) & np.isfinite(y); out = {}
    for c, nome in enumerate(("q", "k/v", "mlp/saída")):
        s = ok & (cls == c); out[nome] = float(spearmanr(np.abs(x[s]), np.abs(y[s]))[0])
    return out

def t_eapig(mm, tarefa, itens, grupo, nome, bs=None, n=40, salvar=True, aquecer=True):
    # EAP-IG oficial: tempo e memória de pico (e, se salvar, o grafo com as importâncias). `bs` = tamanho do lote do oficial (padrão do MIB = 10).
    tempos = q_ler(tarefa, "tempos"); c = Q_CFG[tarefa]; bs = bs or c["bs_attr"]
    if nome in tempos and (not salvar or os.path.exists(q_arq(tarefa, nome))):
        print(f"   (já medido) {nome}: {tempos[nome].get('seg')} s, pico {mb(tempos[nome].get('pico_mb'))}"); return tempos[nome]
    it = q_itens_do_hh(mm, itens, Q_SEM[grupo], n, c["n_lote"])
    def rodar(its):
        dl = DataLoader(its, batch_size=bs, collate_fn=collate_EAP); g = Graph.from_model(mm)
        met = get_metric("logit_diff", tarefa, mm.tokenizer, mm); t_limpar()
        with _Med() as md:
            with torch.enable_grad():
                attribute(mm, g, dl, partial(met, mean=True, loss=True), "EAP-IG-inputs", "patching", ig_steps=5, intervention_dataloader=dl)
        return g, md
    try:
        if aquecer: rodar(it[:c["n_lote"]])
        g, md = rodar(it)
    except torch.cuda.OutOfMemoryError:
        t_limpar(); tempos = q_ler(tarefa, "tempos"); tempos[nome] = dict(oom=True, bs=bs); q_gravar(tarefa, "tempos", tempos)
        print(f"   {nome}: SEM MEMÓRIA (lote {bs})"); return tempos[nome]
    if salvar: os.makedirs(os.path.dirname(q_arq(tarefa, nome)), exist_ok=True); g.to_json(q_arq(tarefa, nome))
    tempos = q_ler(tarefa, "tempos"); tempos[nome] = dict(seg=round(md.seg, 1), pico_mb=md.pico, bs=bs, dtype=str(mm.cfg.dtype)); q_gravar(tarefa, "tempos", tempos)
    print(f"   {tarefa} {grupo} EAP-IG oficial [{nome}] ({mm.cfg.dtype}, lote {bs}): {md.seg:.1f} s, pico {mb(md.pico)}"); del g; t_limpar(); return tempos[nome]

def t_hh_livre(m, tarefa, grupo, lc=None):
    # HH-livre-eapig (pontos 0; 0,2; …; 0,8 = como o oficial) do grupo, em float32, gravado no formato do MIB. Devolve o tempo/memória.
    nome = f"HH-livre-eapig-{grupo}"
    ar, vec, info = t_notas(m, tarefa, semente=Q_SEM[grupo], n_lote=Q_CFG[tarefa]["n_lote"], esquema="eapig", lc=lc)
    nt = dict(zip(ar, vec)); q_salvar(m, tarefa, nome, ar, lambda a: -abs(nt[a]))
    print(f"   {tarefa} {grupo} HH-livre-eapig (float32): {info['seg']} s, pico {mb(info['pico_mb'])}, chamadas do gradiente {info['tams_chamadas']}"); return info

def t_classes_grafo(g):
    idx = defaultdict(lambda: ([], []))
    for nm, e in g.edges.items():
        if nm == "input->m0": continue
        cl = "q" if nm.endswith("<q>") else "k/v" if nm.endswith(("<k>", "<v>")) else "mlp/saída"
        idx[cl][0].append(e.matrix_index[0]); idx[cl][1].append(e.matrix_index[1])
    return {cl: (torch.tensor(r), torch.tensor(c)) for cl, (r, c) in idx.items()}

def t_comparar(tarefa, grupo, a, b):
    # Spearman de |nota| por tipo de aresta entre dois grafos salvos (nomes de pasta sem o grupo)
    g = {k: Graph.from_json(q_arq(tarefa, n.format(grupo))) for k, n in (("a", a), ("b", b))}
    cl = t_classes_grafo(g["b"])
    return {k: float(spearmanr(g["a"].scores[r, c].abs().float().numpy(), g["b"].scores[r, c].abs().float().numpy())[0]) for k, (r, c) in cl.items()}

def t_rep(f, n):
    # n repetições de uma medida; devolve a lista de dicts
    return [f(i) for i in range(n)]

def t_ablacao(m, T, medias, cabecas):
    # frase limpa; as cabeças dadas recebem a média da sua saída. Devolve (prob. média da certa, dif. de logit certa−errada média, acerto binário)
    nH = m.cfg.n_heads; por = defaultdict(list)
    for c_ in cabecas: por[int(c_) // nH].append(int(c_) % nH)
    hooks = []
    for L, hs in por.items():
        def hk(z, hook, L=L, hs=hs): z[:, :, hs] = medias[L][:, :, hs]; return z
        hooks.append((f"blocks.{L}.attn.hook_z", hk))
    ar = torch.arange(T["tk"].shape[0], device=T["tk"].device); lg = hh.logits_finais(m, T["tk"], None, hooks)[:, -1].float()
    ld = lg[ar, T["alvo"]] - lg[ar, T["errado"]]
    return lg.softmax(-1)[ar, T["alvo"]].mean().item(), ld.mean().item(), (ld > 0).float().mean().item()

def t_cap(m, tarefa="CAP", n=40, ks=(1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 500, 700, 900, 1000, 1023)):
    t0 = time.time(); nL, nH = m.cfg.n_layers, m.cfg.n_heads; N = nL * nH; ks = sorted({k for k in ks if k < N} | {N - 1})
    T = hh.tarefa(m, tarefa, 0, n)
    _, cl = m.run_with_cache(T["tk"], names_filter=lambda x: x.endswith("attn.hook_z"))
    medias = {L: cl[f"blocks.{L}.attn.hook_z"].mean(0, keepdim=True) for L in range(nL)}; del cl
    tf = torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32 = True   # como no pod B (só médias de probabilidade)
    try:
        p0, ld0, a0 = t_ablacao(m, T, medias, []); pt, ldt, at = t_ablacao(m, T, medias, range(N))
        uni = np.array([t_ablacao(m, T, medias, [c_]) for c_ in range(N)])      # (N, 3): p, ld, acerto com UMA cabeça ablacionada
        ord_p = [int(c_) for c_ in np.argsort(-(p0 - uni[:, 0]))]; ord_ld = [int(c_) for c_ in np.argsort(-(ld0 - uni[:, 1]))]
        ord_al = [int(c_) for c_ in np.random.default_rng(0).permutation(N)]
        dp = lambda o, k: (p0 - t_ablacao(m, T, medias, o[:k])[0]) / max(p0 - pt, 1e-9)
        dl = lambda o, k: (ld0 - t_ablacao(m, T, medias, o[:k])[1]) / max(ld0 - ldt, 1e-9)
        curva = {k: dict(p_ordem_p=dp(ord_p, k), ld_ordem_ld=dl(ord_ld, k), p_aleatorio=dp(ord_al, k), ld_aleatorio=dl(ord_al, k),
                         acerto_ordem_ld=t_ablacao(m, T, medias, ord_ld[:k])[2]) for k in ks}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf
    prim = lambda col, a: next((k for k in ks if curva[k][col] >= a), None)
    r = dict(p0=p0, p_todas=pt, dif_p=p0 - pt, ld0=ld0, ld_todas=ldt, dif_ld=ld0 - ldt, acerto0=a0, acerto_todas=at,
             k80_p=prim("p_ordem_p", 0.8), k80_ld=prim("ld_ordem_ld", 0.8), k50_p=prim("p_ordem_p", 0.5), k50_ld=prim("ld_ordem_ld", 0.5),
             k80_p_aleatorio=prim("p_aleatorio", 0.8), top_p=[f"{c_ // nH}.{c_ % nH}" for c_ in ord_p[:8]], top_ld=[f"{c_ // nH}.{c_ % nH}" for c_ in ord_ld[:8]],
             queda_1a_p=float(p0 - uni[ord_p[0], 0]), curva={str(k): v for k, v in curva.items()}, N=N, tempo=round(time.time() - t0, 1))
    print(f"\n{m.cfg.model_name} / {tarefa}: p0 = {p0:.3f} | todas as cabeças ablacionadas: p = {pt:.3f} | p0 − p_todas = {p0 - pt:.3f}")
    print(f"   dif. de logit (certa − errada): {ld0:.2f} → {ldt:.2f} (diferença {ld0 - ldt:.2f}) | acerto binário {a0:.0%} → {at:.0%}")
    print(f"   {'k':>5s} {'dano (prob.; ordem p)':>22s} {'dano (logit; ordem logit)':>26s} {'dano (prob.; aleatório)':>24s} {'acerto (ordem logit)':>21s}")
    for k in ks: v = curva[k]; print(f"   {k:5d} {v['p_ordem_p']:22.0%} {v['ld_ordem_ld']:26.0%} {v['p_aleatorio']:24.0%} {v['acerto_ordem_ld']:21.0%}")
    print(f"   k p/ 50% do dano: prob. {r['k50_p']} | logit {r['k50_ld']}; p/ 80%: prob. {r['k80_p']} | logit {r['k80_ld']} | aleatório {r['k80_p_aleatorio']} (None = não chegou até {N - 1}) | {r['tempo']:.0f} s")
    return r

def t_previsoes():
    # Confere as previsões que já têm dados. ✅ acertou, ❌ errou, ⏳ falta dado.
    out = []
    def add(txt, ok): out.append(f"{'⏳' if ok is None else '✅' if ok else '❌'} {txt}")
    lj = lambda d, t, n: json.load(open(f"{Q_BASE}/{d}/{t}/{n}.json")) if os.path.exists(f"{Q_BASE}/{d}/{t}/{n}.json") else None
    med = lambda v: float(np.median(v)); var = lambda v: (max(v) - min(v)) / med(v)
    # ---- (E) 1.5B ----
    T = lj("qwen1.5b", "ioi", "tempos") or {}; E = lj("qwen1.5b", "ioi", "e_hh") or {}
    def rep(D, pref, campo): return [D[k][campo] for k in sorted(D) if k.startswith(pref) and isinstance(D[k], dict) and D[k].get(campo) is not None]
    s = {c: rep(E, c + "-r", "seg") for c in ("L0", "L20", "L40")}; pk = {c: rep(E, c + "-r", "pico_mb") for c in ("L0", "L20", "L40")}
    o16, o32 = rep(T, "EAP-IG-bf16-r", "seg"), rep(T, "EAP-IG-fp32-r", "seg")
    if all(s.values()):
        sp = [E[k]["spearman_min"] for k in E if k.split("-")[0] in ("L20", "L40") and E[k].get("spearman_min") is not None]
        add(f"E1 Spearman mínimo L20/L40 × L0 = {min(sp):.6f} (previsto ≥ 0,99999)" if sp else "E1 (sem Spearman)", None if not sp else min(sp) >= 0.99999)
        r = med(s["L20"]) / med(s["L0"]); add(f"E2 tempo L20 / L0 = {r:.2f} (previsto 0,50-0,65)", 0.50 <= r <= 0.65)
        r = med(s["L40"]) / med(s["L20"]); add(f"E3 tempo L40 / L20 = {r:.2f} (previsto 0,95-1,05: as 4 amostras de 10 frases caem em 2 comprimentos, então os dois dão pedaços [20, 20])", 0.95 <= r <= 1.05)
        r = med(pk["L40"]) / med(pk["L20"]); add(f"E4 pico de memória L40 / L20 = {r:.2f} (previsto 0,95-1,05, pelo mesmo motivo)", 0.95 <= r <= 1.05)
        a, b = med(pk["L0"]), med(pk["L20"]); add(f"E5 pico L0 = {a:,.0f} MB (3.000-3.700) e L20 = {b:,.0f} MB (5.000-6.500)".replace(",", "."), 3000 <= a <= 3700 and 5000 <= b <= 6500)
        vs = {k: var(v) for k, v in {**{"HH " + c: s[c] for c in s}, "oficial bf16": o16, "oficial fp32": o32}.items() if len(v) >= 2}
        add("E6 variação entre repetições (máx−mín)/mediana: " + ", ".join(f"{k} {v:.0%}" for k, v in vs.items()) + " (previsto ≤ 8% em todas)", all(v <= 0.08 for v in vs.values()))
        if o16 and o32:
            a, b = med(s["L20"]) / med(o32), med(s["L20"]) / med(o16)
            add(f"E7 HH L20 / oficial fp32 = {a:.2f} (≤ 0,65) e / oficial bf16 = {b:.2f} (0,75-1,05)", a <= 0.65 and 0.75 <= b <= 1.05)
        else: add("E7 (faltam as medidas do oficial)", None)
    else: add("E1-E7 (rode a célula 7)", None)
    if all(k in E and E[k].get("seg") for k in ("L0-s1", "L40-s1")):
        r = E["L40-s1"]["seg"] / E["L0-s1"]["seg"]; sps = min(E[k]["spearman_min"] for k in ("L20-s1", "L40-s1") if k in E and E[k].get("spearman_min") is not None)
        add(f"E9 grupo s1 (uma medida): tempo L40 / L0 = {r:.2f} (previsto ≤ 0,90; chamadas L40: {E['L40-s1']['tams_chamadas']}), Spearman mín {sps:.6f} (≥ 0,99999)", r <= 0.90 and sps >= 0.99999)
    else: add("E9 (falta o grupo s1)", None)
    ck = lj("qwen1.5b", "ioi", "e_conversao")
    if ck: add(f"E8 oficial fp32 convertido na GPU (.float()) × carregado direto em fp32: diferença máxima dos logits {ck['dif_max']:.2e} (previsto ≤ 1e-4)", ck["dif_max"] <= 1e-4)
    # ---- (F) 7B ----
    F = lj("qwen7b", "ioi", "f_7b") or {}; T7 = lj("qwen7b", "ioi", "tempos") or {}
    b0, b1 = T7.get("EAP-IG-bf16-s0", {}), T7.get("EAP-IG-bf16-s1", {})
    h0 = F.get("HH-padrao-s0", {})
    if h0.get("seg") and b0.get("seg"):
        r = h0["seg"] / b0["seg"]; add(f"F1 tempo HH padrão / oficial bf16 no 7B (s0) = {r:.2f} (previsto 0,5-1,0)" + (f"; s1: {F['HH-padrao-s1']['seg'] / b1['seg']:.2f}" if F.get('HH-padrao-s1', {}).get('seg') and b1.get('seg') else ""), 0.5 <= r <= 1.0)
        add(f"F3 oficial bf16 no 7B: pico {mb(b0.get('pico_mb'))} (previsto 12.000-30.000 MB)", b0.get("pico_mb") is not None and 12000 <= b0["pico_mb"] <= 30000)
    else: add("F1/F3 (rode a célula 8)", None)
    f32 = T7.get("EAP-IG-fp32-s0")
    if f32 is not None or F.get("fp32_carregar"): add(f"F2 oficial fp32 no 7B: {'NÃO coube' if (F.get('fp32_carregar') or f32.get('oom')) else 'coube (' + str(f32.get('seg')) + ' s, pico ' + mb(f32.get('pico_mb')) + ')'} (previsto: não cabe)", bool(F.get("fp32_carregar") or f32.get("oom")))
    else: add("F2 (falta)", None)
    if h0: add(f"F4 HH livre fp32 no 7B: " + ("SEM MEMÓRIA" if h0.get("oom") else f"pico {mb(h0.get('pico_mb'))} com lotes de {h0.get('n_lote')} (previsto: cabe com lote 10, pico 8.000-20.000 MB)"), bool(h0.get("pico_mb")) and h0.get("n_lote") == 10 and 8000 <= h0["pico_mb"] <= 20000)
    else: add("F4 (falta)", None)
    if f32 and not f32.get("oom") and F.get("sp_hh_fp32") and h0.get("seg"):
        sp = F["sp_hh_fp32"]; r = h0["seg"] / f32["seg"]
        add(f"F5 HH × oficial fp32 no 7B: Spearman " + ", ".join(f"{k} {v:.4f}" for k, v in sp.items()) + f" (≥ 0,99) e tempo HH/oficial fp32 = {r:.2f} (≤ 0,60)", min(sp.values()) >= 0.99 and r <= 0.60)
    else: add("F5 (só pontuável se o oficial fp32 couber)", None)
    sp = F.get("sp_padrao_L0")
    add(f"F6 Spearman HH padrão × L0 no 7B = {sp:.6f} (previsto ≥ 0,99999)" if sp is not None else "F6 (falta)", None if sp is None else sp >= 0.99999)
    cab = lj("qwen7b", "ioi", "cabecas")
    add(f"F7 etapa das cabeças do HH no 7B: {cab['seg']} s, {len(cab['cabecas'])} cabeças, fidelidade {cab['fid']:.0%} (previsto < 240 s)" if cab else "F7 (falta)", None if not cab else cab["seg"] < 240)
    # ---- (G) CAP Qwen2.5-7B por diferença de logit ----
    k = lj("qwen7b", "cap", "cap_check"); kp = lj("pythia6.9b", "cap", "cap_check")
    if k:
        add(f"G1 Qwen-7B CAP: p0 − p_todas = {k['dif_p']:.3f} (p0 {k['p0']:.3f}; previsto ≥ 0,10)", k["dif_p"] >= 0.10)
        add(f"G2 k para 80% do dano (logit) = {k['k80_ld']} e para 50% = {k['k50_ld']} (previsto 80%: 5-60)", k["k80_ld"] is not None and 5 <= k["k80_ld"] <= 60)
        pyk = (kp or {}).get("k80_ld", 50)
        add(f"G3 Pythia-6.9B k80 (logit) = {pyk} × Qwen {k['k80_ld']}: razão {pyk / k['k80_ld']:.2f} (previsto ≥ 1,2: o Pythia precisa de mais cabeças)" if k["k80_ld"] else "G3 (Qwen não chegou a 80%)", bool(k["k80_ld"]) and pyk / k["k80_ld"] >= 1.2)
        c = k["curva"].get("100") or k["curva"][max(k["curva"], key=lambda x: int(x))]
        add(f"G4 dano (logit) com 100 cabeças aleatórias = {c['ld_aleatorio']:.0%} (previsto < 10%); ordenadas {c['ld_ordem_ld']:.0%}", c["ld_aleatorio"] < 0.10)
        add(f"G5 acerto binário com todas as cabeças ablacionadas: {k['acerto0']:.0%} → {k['acerto_todas']:.0%} (previsto < 90%)", k["acerto_todas"] < 0.90)
    else: add("G1-G5 (rode a célula 9)", None)
    print("\nprevisões:"); [print("  " + x) for x in out]; return out


# ===================== CÉLULA 7 — (E) v0.11.5 no Qwen2.5-1.5B: lote de cálculo 0 / 20 / 40, 3 repetições, com o EAP-IG oficial ao lado (~10 min) =====================
q_trocar("Qwen/Qwen2.5-1.5B", "qwen1.5b")
m = hh.carregar(Q_MODELO); mm16 = q_modelo(torch.bfloat16); mm32 = q_modelo(torch.float32)
print(f"HH: {next(m.parameters()).dtype} | oficial: {mm16.cfg.dtype} e {mm32.cfg.dtype} | GQA {m.cfg.n_heads} cabeças / {m.cfg.n_key_value_heads} kv | {m.cfg.n_layers} camadas | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
q_conferir_grafo(m, mm16); itens15 = q_itens(m, "ioi"); q_receita(m, "ioi", itens15); q_conferir_tokens(m, mm16, itens15)
# E8: o oficial em fp32 será montado no 7B subindo em bf16 e convertendo na GPU (.float()); aqui confiro que isso dá o mesmo que carregar direto em fp32
if not os.path.exists(q_json("ioi", "e_conversao")):
    mmc = q_modelo(torch.bfloat16).float(); mmc.cfg.dtype = torch.float32
    tk = mm32.to_tokens(["When Mary and John went to the store, John gave a drink to", "The capital of France is the city of"], prepend_bos=True)
    with torch.no_grad(): dif = (mm32(tk).float() - mmc(tk).float()).abs().max().item()
    q_gravar("ioi", "e_conversao", dict(dif_max=dif)); print(f"   E8: diferença máxima dos logits (oficial fp32 direto × convertido na GPU): {dif:.2e}"); del mmc; t_limpar()
NR = 3
for r in range(NR):
    t_eapig(mm16, "ioi", itens15, "s0", f"EAP-IG-bf16-r{r}", salvar=False, aquecer=(r == 0))
    t_eapig(mm32, "ioi", itens15, "s0", f"EAP-IG-fp32-r{r}", salvar=False, aquecer=(r == 0))
del mm16, mm32; t_limpar()
E = q_ler("ioi", "e_hh", {}); arq0 = f"{Q_PASTA}/ioi/nota_L0_r0.npy"
nL, nH = m.cfg.n_layers, m.cfg.n_heads; ar_ = hh.arestas_partes(m, range(nL * nH), range(nL)); cls_ = t_classes(ar_)
if not E: print("   aquecendo (1 lote, não conta) ..."); t_notas(m, "ioi", semente=Q_SEM["s0"], n=10, n_lote=10, lc=0)
for r in range(NR):
    for cod, lc in (("L0", 0), ("L20", 20), ("L40", 40)):
        k = f"{cod}-r{r}"
        if k in E: print(f"   (já feito) {k}"); continue
        try: _, vec, info = t_notas(m, "ioi", semente=Q_SEM["s0"], lc=lc)
        except torch.cuda.OutOfMemoryError:
            t_limpar(); E[k] = dict(oom=True); q_gravar("ioi", "e_hh", E); print(f"   {k}: SEM MEMÓRIA"); continue
        if k == "L0-r0": np.save(arq0, vec)
        sp = t_spear(np.load(arq0), vec, cls_) if os.path.exists(arq0) else None
        E[k] = dict(**info, spearman=sp, spearman_min=min(sp.values()) if sp else None); q_gravar("ioi", "e_hh", E)
        print(f"   {k}: {info['seg']} s (gradiente {info['grad_s']}, caches {info['caches_s']}, resto {info['resto_s']}) | pico {mb(info['pico_mb'])} | chamadas do gradiente {info['tams_chamadas']}" + (f" | Spearman × L0-r0 mín {min(sp.values()):.6f}" if sp and k != "L0-r0" else ""))
# grupo s1 (outras 40 frases, uma medida de cada): o pedaço fundido só junta lotes de MESMO comprimento, então o ganho depende de quantos lotes caem no mesmo comprimento
arq1 = f"{Q_PASTA}/ioi/nota_L0_s1.npy"
for cod, lc in (("L0", 0), ("L20", 20), ("L40", 40)):
    k = f"{cod}-s1"
    if k in E: print(f"   (já feito) {k}"); continue
    try: _, vec, info = t_notas(m, "ioi", semente=Q_SEM["s1"], lc=lc)
    except torch.cuda.OutOfMemoryError: t_limpar(); E[k] = dict(oom=True); q_gravar("ioi", "e_hh", E); print(f"   {k}: SEM MEMÓRIA"); continue
    if cod == "L0": np.save(arq1, vec)
    sp = t_spear(np.load(arq1), vec, cls_) if os.path.exists(arq1) else None
    E[k] = dict(**info, spearman=sp, spearman_min=min(sp.values()) if sp else None); q_gravar("ioi", "e_hh", E)
    print(f"   {k}: {info['seg']} s (gradiente {info['grad_s']}) | pico {mb(info['pico_mb'])} | chamadas do gradiente {info['tams_chamadas']}" + (f" | Spearman × L0-s1 mín {min(sp.values()):.6f}" if sp and cod != "L0" else ""))
T = q_ler("ioi", "tempos")
print("\n=== (E) Qwen2.5-1.5B, IOI, 40 frases, 5 passos: medianas das 3 repetições ===")
for rot, ks in (("EAP-IG oficial bf16 (lote 10)", [k for k in T if k.startswith("EAP-IG-bf16-r")]), ("EAP-IG oficial fp32 (lote 10)", [k for k in T if k.startswith("EAP-IG-fp32-r")]),
                ("HH L0 (sem fundir)", [k for k in E if k.startswith("L0-r")]), ("HH L20", [k for k in E if k.startswith("L20-r")]), ("HH L40", [k for k in E if k.startswith("L40-r")])):
    D = T if rot.startswith("EAP") else E; v = [D[k] for k in ks if D[k].get("seg")]
    if v: print(f"  {rot:32s} tempo {np.median([x['seg'] for x in v]):6.1f} s (min {min(x['seg'] for x in v):.1f}, máx {max(x['seg'] for x in v):.1f}) | pico {mb(np.median([x['pico_mb'] for x in v if x.get('pico_mb') is not None]) if any(x.get('pico_mb') is not None for x in v) else None)}")
t_previsoes()


# ===================== CÉLULA 8 — (F) o mesmo no Qwen2.5-7B: EAP-IG oficial (bf16 e fp32) e HeadHunter, um modelo de cada vez (~25-45 min; baixa ~15 GB) =====================
del m; t_limpar()
q_trocar("Qwen/Qwen2.5-7B", "qwen7b")
F = q_ler("ioi", "f_7b", {})
# --- 1) EAP-IG oficial em bf16 (15 GB) ---
mm = q_modelo(torch.bfloat16); itens7 = q_itens(mm, "ioi"); q_receita(mm, "ioi", itens7)
print(f"oficial bf16: {sum(p.numel() for p in mm.parameters()) / 1e9:.1f} B parâmetros | GQA {mm.cfg.n_heads} cabeças / {mm.cfg.n_key_value_heads} kv | {mm.cfg.n_layers} camadas")
for gr in ("s0", "s1"): t_eapig(mm, "ioi", itens7, gr, f"EAP-IG-bf16-{gr}", salvar=False, aquecer=(gr == "s0"))
del mm; t_limpar()
# --- 2) EAP-IG oficial em fp32 (30 GB): sobe em bf16 e converte na GPU (não estoura a RAM). Pode não caber: o "sem memória" também é resultado ---
try:
    mm = q_modelo(torch.bfloat16).float(); mm.cfg.dtype = torch.float32; t_limpar()
    print(f"oficial fp32: pesos {(torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0):.1f} GB na GPU")
    for gr in ("s0", "s1"): t_eapig(mm, "ioi", itens7, gr, f"EAP-IG-fp32-{gr}", salvar=(gr == "s0"), aquecer=(gr == "s0"))
except torch.cuda.OutOfMemoryError:
    F["fp32_carregar"] = dict(oom=True); q_gravar("ioi", "f_7b", F); print("   oficial fp32: não coube nem para carregar")
finally:
    try: del mm
    except NameError: pass
    t_limpar()
# --- 3) HeadHunter (float32; a biblioteca dimensiona o pedaço pela memória livre e cai pela metade se faltar) ---
m = hh.carregar(Q_MODELO, hf_id=Q_MODELO, converter_na="gpu"); q_receita(m, "ioi", itens7)
print(f"HH: {next(m.parameters()).dtype} | pesos {(torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0):.1f} GB | GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
def f_hh(gr, lc, cod):
    chave = f"HH-{cod}-{gr}"
    if chave in F: print(f"   (já feito) {chave}"); return
    for n_lote in (10, 5):   # se não couber com lotes de 10 frases, tenta 5 (e o resultado fica marcado: frases diferentes, só vale para tempo e memória)
        try: ar, vec, info = t_notas(m, "ioi", semente=Q_SEM[gr], n_lote=n_lote, esquema="eapig", lc=lc); break
        except torch.cuda.OutOfMemoryError: t_limpar(); print(f"   {chave}: SEM MEMÓRIA com lotes de {n_lote}")
    else: F[chave] = dict(oom=True); q_gravar("ioi", "f_7b", F); return
    F[chave] = info; arq = f"{Q_PASTA}/ioi/nota_padrao_{gr}.npy"
    if cod == "padrao":
        np.save(arq, vec)
        if gr == "s0": nt = dict(zip(ar, vec)); q_salvar(m, "ioi", f"HH-livre-eapig-{gr}", ar, lambda a: -abs(nt[a]))
    elif os.path.exists(arq) and F.get(f"HH-padrao-{gr}", {}).get("n_lote") == n_lote:
        F[f"sp_padrao_L0"] = float(min(t_spear(np.load(arq), vec, t_classes(ar)).values()))
    q_gravar("ioi", "f_7b", F)
    print(f"   {chave}: {info['seg']} s (gradiente {info['grad_s']}, caches {info['caches_s']}, resto {info['resto_s']}) | pico {mb(info['pico_mb'])} | lotes de {n_lote} | chamadas do gradiente {info['tams_chamadas']}")
f_hh("s0", None, "padrao"); f_hh("s0", 0, "L0"); f_hh("s1", None, "padrao")
cab7, info7 = q_cabecas(m, "ioi", itens7)    # etapa das cabeças (circuito de 90%): o custo a mais do ponta a ponta
if os.path.exists(q_arq("ioi", "EAP-IG-fp32-s0")) and os.path.exists(q_arq("ioi", "HH-livre-eapig-s0")):
    F["sp_hh_fp32"] = t_comparar("ioi", "s0", "HH-livre-eapig-{}", "EAP-IG-fp32-{}"); q_gravar("ioi", "f_7b", F)
    print("   HH × oficial fp32 (7B, s0), Spearman de |nota|: " + " | ".join(f"{k} {v:.4f}" for k, v in F["sp_hh_fp32"].items()))
T7 = q_ler("ioi", "tempos")
print("\n=== (F) Qwen2.5-7B, IOI, 40 frases, 5 passos ===")
for k in sorted(T7): print(f"  {'EAP-IG oficial ' + k.replace('EAP-IG-', ''):28s} " + ("SEM MEMÓRIA" if T7[k].get("oom") else f"{T7[k]['seg']:7.1f} s | pico {mb(T7[k].get('pico_mb'))}"))
for k in sorted(F):
    if k.startswith("HH-"): print(f"  {k:28s} " + ("SEM MEMÓRIA" if F[k].get("oom") else f"{F[k]['seg']:7.1f} s | pico {mb(F[k].get('pico_mb'))} | lotes de {F[k].get('n_lote')}"))
print(f"  etapa das cabeças do HH: {info7['seg']} s, pico {mb(info7['pico_mb'])}, {len(cab7)} cabeças, fidelidade {info7['fid']:.0%}")
t_previsoes()


# ===================== CÉLULA 9 — (G) CAP no Qwen2.5-7B: a necessidade pela DIFERENÇA DE LOGIT (e não pela probabilidade) (~10-15 min) =====================
Q_QK = f"{Q_BASE}/qwen7b/cap"; os.makedirs(Q_QK, exist_ok=True)
if os.path.exists(f"{Q_QK}/cap_check.json"): print("(já feito) CAP do Qwen2.5-7B"); cap7 = json.load(open(f"{Q_QK}/cap_check.json"))
else:
    if "m" not in globals() or m is None: m = hh.carregar("Qwen/Qwen2.5-7B", hf_id="Qwen/Qwen2.5-7B", converter_na="gpu")
    print("acerto do modelo no CAP (top-1 do vocabulário):", f"{hh.acerto(m, 'CAP'):.0%}")
    cap7 = t_cap(m, "CAP", ks=(1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200, 300, 500, 700, 783)); json.dump(cap7, open(f"{Q_QK}/cap_check.json", "w"), indent=1)
t_previsoes()


# ===================== CÉLULA 10 — resumo final (cole TUDO que esta célula imprimir no chat) =====================
LJ = lambda p: json.load(open(f"{Q_BASE}/{p}")) if os.path.exists(f"{Q_BASE}/{p}") else {}
T15, E = LJ("qwen1.5b/ioi/tempos.json"), LJ("qwen1.5b/ioi/e_hh.json")
print("== (E) 1.5B: EAP-IG oficial =="); [print(f"  {k:22s} {v.get('seg')} s  pico {mb(v.get('pico_mb'))}" + ("  SEM MEMÓRIA" if v.get("oom") else "")) for k, v in sorted(T15.items())]
print("\n== (E) 1.5B: HeadHunter v0.11.5 =="); [print(f"  {k:8s} {v.get('seg')} s (grad {v.get('grad_s')}, caches {v.get('caches_s')}, resto {v.get('resto_s')}) pico {mb(v.get('pico_mb'))} chamadas {v.get('tams_chamadas')} Spearman×L0 (mesmo grupo) mín {v.get('spearman_min')}" + ("  SEM MEMÓRIA" if v.get("oom") else "")) for k, v in sorted(E.items())]
print("  E8:", LJ("qwen1.5b/ioi/e_conversao.json"))
T7, F = LJ("qwen7b/ioi/tempos.json"), LJ("qwen7b/ioi/f_7b.json")
print("\n== (F) 7B: EAP-IG oficial =="); [print(f"  {k:22s} " + ("SEM MEMÓRIA" if v.get("oom") else f"{v.get('seg')} s  pico {mb(v.get('pico_mb'))}")) for k, v in sorted(T7.items())]
print("\n== (F) 7B: HeadHunter ==")
for k, v in sorted(F.items()):
    if isinstance(v, dict) and k.startswith("HH-"): print(f"  {k:16s} " + ("SEM MEMÓRIA" if v.get("oom") else f"{v.get('seg')} s (grad {v.get('grad_s')}, caches {v.get('caches_s')}, resto {v.get('resto_s')}) pico {mb(v.get('pico_mb'))} lotes de {v.get('n_lote')} chamadas {v.get('tams_chamadas')}"))
    else: print(f"  {k:16s} {v}")
d7 = LJ("qwen7b/ioi/cabecas.json")
if d7: print(f"  etapa das cabeças: {d7['seg']} s, pico {mb(d7['pico_mb'])}, {len(d7['cabecas'])} cabeças, fidelidade {d7['fid']:.0%}")
cap = LJ("qwen7b/cap/cap_check.json"); capp = LJ("pythia6.9b/cap/cap_check.json")
if cap:
    print(f"\n== (G) CAP Qwen2.5-7B ==\n  p0 {cap['p0']:.3f} | p_todas {cap['p_todas']:.3f} | diferença {cap['dif_p']:.3f} | logit {cap['ld0']:.2f} → {cap['ld_todas']:.2f} | acerto {cap['acerto0']:.0%} → {cap['acerto_todas']:.0%}")
    print(f"  k p/ 50%: prob {cap['k50_p']} logit {cap['k50_ld']} | k p/ 80%: prob {cap['k80_p']} logit {cap['k80_ld']} aleatório(prob) {cap['k80_p_aleatorio']} | top (logit) {cap['top_ld']}")
    for k, v in cap["curva"].items(): print(f"    k={k:>5s}: prob(ordem p) {v['p_ordem_p']:.0%} | logit(ordem logit) {v['ld_ordem_ld']:.0%} | logit(aleatório) {v['ld_aleatorio']:.0%} | acerto {v['acerto_ordem_ld']:.0%}")
t_previsoes()
shutil.make_archive("/workspace/teste_pod_d_resultados", "zip", Q_BASE); print("\nbackup: /workspace/teste_pod_d_resultados.zip (baixe pelo JupyterLab antes de apagar o pod)")
