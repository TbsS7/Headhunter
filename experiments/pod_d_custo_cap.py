# HeadHunter — pod D (RTX A6000 48 GB): a v0.11.5 na GPU, o tempo no 7B e o CAP do Qwen2.5-7B por diferença de logit. Escrito em 12/11/2026.
# ADAPTADO em 13/11/2026 para a API em inglês (v0.12.0). Os números reportados foram obtidos com a v0.11.x (português); a v0.12.0 dá os mesmos valores (ver CHANGELOG.md). Comentários e mensagens impressas continuam em português.
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


# ===================== CÉLULA 3 — instalar e conferir o HeadHunter (v0.12.0, API em inglês) =====================
!pip install -q "git+https://github.com/TbsS7/Headhunter.git"
import importlib, headhunter as hh; importlib.reload(hh)
assert hh.VERSION == "0.12.0", f"HeadHunter {hh.VERSION}: este notebook é da v0.12.0 (rode a célula 3 de novo)"
print("HeadHunter", hh.VERSION, "| esquema:", hh.IG_SCHEME, "| LOTE_CALCULO", hh.COMPUTE_BATCH)


# ===================== CÉLULA 4 — módulos HeadHunter → MIB + avaliação retomável =====================
import gc, torch, random
# ---- HeadHunter no benchmark MIB (trilha de circuitos) ----
# Precisa: o repositório MIB-circuit-track no sys.path e o pacote eap (commit 7af394a, ramo MIB).
import os, random, time, gc, torch
from collections import defaultdict
from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.utils import TASKS_TO_HF_NAMES
from eap.graph import Graph

def items_mib(m, task, split="train", n=1000):
    # Os mesmos itens que o MIB usa: (frase limpa, frase corrompida, [certo, errado]).
    ds = HFEAPDataset(f"mib-bench/{TASKS_TO_HF_NAMES[task]}", m.tokenizer, split=split, task=task, num_examples=n)
    return [ds[i] for i in range(len(ds))]

def receita_mib(m, items, name):
    # O HeadHunter supõe frases do mesmo tamanho e a resposta no fim: separa os itens em grupos por comprimento
    # (mesmos tokens do MIB: BOS + texto). Cada lote do HeadHunter sai de UM grupo.
    groups = defaultdict(list); fora = 0
    for limpa, corr, lab in items:
        a = m.to_tokens(limpa, prepend_bos=True)[0]; b = m.to_tokens(corr, prepend_bos=True)[0]
        if len(a) == len(b): groups[len(a)].append((a, b, int(lab[0]), int(lab[1])))
        else: fora += 1
    def build(m_, seed, n=40):
        rnd = random.Random(seed); ok = [L for L in groups if len(groups[L]) >= n] or [max(groups, key=lambda L: len(groups[L]))]
        L = rnd.choices(ok, weights=[len(groups[x]) for x in ok])[0]; g = groups[L]
        esc = rnd.sample(g, n) if len(g) >= n else [rnd.choice(g) for _ in range(n)]
        dev = m_.cfg.device
        return dict(tk=torch.stack([e[0] for e in esc]).to(dev), tkc=torch.stack([e[1] for e in esc]).to(dev),
                    target=torch.tensor([e[2] for e in esc], device=dev), wrong=torch.tensor([e[3] for e in esc], device=dev))
    hh.RECIPES[name] = build
    info = dict(items=len(items), fora=fora, groups={L: len(v) for L, v in sorted(groups.items())})
    print(f"{name}: {len(items)} itens, {fora} descartados (limpa e corrompida com tamanhos diferentes), "
          f"grupos por comprimento: {info['groups']}")
    return info

def name_edge(m, a):
    nH = m.cfg.n_heads; u, v, p = a
    name = lambda node: {"E": "input", "O": "logits"}.get(node[0]) or (f"a{node[1] // nH}.h{node[1] % nH}" if node[0] == "H" else f"m{node[1]}")
    return f"{name(u)}->{name(v)}" + (f"<{p}>" if p else "")

def scores_grafo_todo(m, task_name, n=40, n_batch=10, ig_steps=5, semente=5000):
    # Nota IG do HeadHunter para TODAS as arestas do modelo (mesmas frases 5000+ que o analisar usa para as arestas).
    nL, nH = m.cfg.n_layers, m.cfg.n_heads
    ar = hh.edge_parts(m, range(nL * nH), range(nL))
    bat = [hh.get_task(m, task_name, semente + b, n_batch) for b in range(n // n_batch)]; Cs = [hh.caches(m, T) for T in bat]
    score = hh.edge_scores(m, bat, Cs, ar, ig_steps)
    del Cs; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return ar, score

def grafo_mib(m, ar, score, heads=None):
    # Nota final = posição no ranking (maior = mais importante), então --absolute não muda nada.
    # cabecas=None -> só a nota IG (o "grafo livre"); com cabecas -> as arestas do cercadinho
    # (cabeças do circuito + todos os MLPs) vêm primeiro, depois o resto, cada parte pela |nota|.
    nL = m.cfg.n_layers; g = Graph.from_model(m)
    dentro = set(hh.edge_parts(m, heads, range(nL))) if heads is not None else set()
    order = sorted(ar, key=lambda a: (a not in dentro, -abs(score[a]))); N = len(order)
    assert N == len(g.edges), f"o HeadHunter vê {N} arestas, o grafo do MIB tem {len(g.edges)}"
    g.scores[:] = 0
    for r, a in enumerate(order): g.edges[name_edge(m, a)].score = float(N - r)
    return g, len(dentro)

def hh_mib(m, task, name_model, pasta, n_items=1000, verbose=True, **kw):
    # 1) dados de treino do MIB; 2) circuito de cabeças do HeadHunter; 3) notas de todas as arestas;
    # 4) salva 2 arquivos de notas: HH-cercadinho (o método) e HH-livre (sem a etapa das cabeças, para comparar).
    name = f"MIB_{task}"; t0 = time.time()
    info = receita_mib(m, items_mib(m, task, "train", n_items), name)
    r = hh.analyze(m, name, prune=False, verbose=False, **kw); head = [int(c) for c in r["circuit"]["heads"]]
    t_head = time.time() - t0
    ar, score = scores_grafo_todo(m, name); t_tot = time.time() - t0
    output = {}
    for metodo, cabs in (("HH-cercadinho", head), ("HH-livre", None)):
        g, _ = grafo_mib(m, ar, score, cabs)
        p = os.path.join(pasta, f"{metodo}_patching_edge", f"{task.replace('_', '-')}_{name_model}")
        os.makedirs(p, exist_ok=True); g.to_json(f"{p}/importances.json"); output[metodo] = f"{p}/importances.json"
    res = dict(task=task, model=name_model, heads=len(head), faith_heads=r["circuit"]["faithfulness"],
               edges_total=len(ar), edges_fenced=len(hh.edge_parts(m, head, range(m.cfg.n_layers))),
               elapsed_heads=round(t_head, 1), elapsed_total=round(t_tot, 1), arquivos=output, groups=info["groups"])
    if verbose:
        print(f">>> {name_model} / {task}: {len(head)} cabeças (fidelidade {res['faith_heads']:.0%}) | cercadinho = "
              f"{res['edges_fenced']} de {len(ar)} arestas ({res['edges_fenced'] / len(ar):.1%}) | "
              f"cabeças em {t_head:.0f} s, tudo em {t_tot:.0f} s")
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

def model_mib(name="gpt2-small"):
    # igual ao run_evaluation.py / run_attribution.py do MIB
    mm = HookedTransformer.from_pretrained(name)
    mm.cfg.use_split_qkv_input = True; mm.cfg.use_attn_result = True; mm.cfg.use_hook_mlp_in = True
    mm.cfg.ungroup_grouped_query_attention = True
    return mm

def read_scoreboard():
    return json.load(open(PLACAR)) if os.path.exists(PLACAR) else {}

def arq(name): return f"{PASTA}/{name}/importances.json"

def avaliar(mm, name, dl_val, absoluto=True):
    # mesma chamada do run_evaluation.py: evaluate_area_under_curve(..., partial(metrica, mean=False, loss=False))
    P = read_scoreboard()
    if name in P: print(f"(já avaliado) {name}: CMD {P[name]['CMD']:.3f}"); return P[name]
    g = Graph.from_json(arq(name)); met = get_metric("logit_diff", "ioi", mm.tokenizer, mm); t0 = time.time()
    out = evaluate_area_under_curve(mm, g, dl_val, partial(met, mean=False, loss=False), quiet=True, level="edge", absolute=absoluto)
    P = read_scoreboard(); P[name] = dict(CMD=out[2], CPR=out[1], faith=[round(x, 3) for x in out[4]], elapsed=round(time.time() - t0))
    json.dump(P, open(PLACAR, "w"), indent=1); print(f"{name}: CMD {out[2]:.3f} | CPR {out[1]:.3f} | {P[name]['elapsed']} s")
    return P[name]

def eapig(mm, items, name, bs=20):
    # EAP-IG-inputs oficial (mesmos argumentos do run_attribution.py) num subconjunto escolhido de frases
    if os.path.exists(arq(name)): print("(já existe)", name); return
    dl = DataLoader(items, batch_size=bs, collate_fn=collate_EAP); g = Graph.from_model(mm)
    met = get_metric("logit_diff", "ioi", mm.tokenizer, mm); t0 = time.time()
    with torch.enable_grad():   # o HeadHunter desliga o gradiente globalmente; o EAP-IG precisa dele
        attribute(mm, g, dl, partial(met, mean=True, loss=True), "EAP-IG-inputs", "patching", ig_steps=5, intervention_dataloader=dl)
    os.makedirs(os.path.dirname(arq(name)), exist_ok=True); g.to_json(arq(name)); print(f"{name}: {time.time() - t0:.0f} s")

def save_hh(m, name, ar, key):
    if os.path.exists(arq(name)): print("(já existe)", name); return
    g = Graph.from_model(m); order = sorted(ar, key=key); N = len(order); g.scores[:] = 0
    for k, a in enumerate(order): g.edges[name_edge(m, a)].score = float(N - k)
    os.makedirs(os.path.dirname(arq(name)), exist_ok=True); g.to_json(arq(name)); print("salvo:", name)

def show_scoreboard():
    pcts = ["0.1%", "0.2%", "0.5%", "1%", "2%", "5%", "10%", "20%", "50%", "100%"]
    for k, d in sorted(read_scoreboard().items(), key=lambda x: x[1]["CMD"]):
        print(f"{k:22s} CMD {d['CMD']:.3f} | CPR {d['CPR']:.3f} | " + " ".join(f"{p}:{x:.0%}" for p, x in zip(pcts, d["faith"])))


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

assert hh.VERSION == "0.12.0", f"HeadHunter {hh.VERSION}: este notebook é da v0.12.0 (rode a célula 3)"
Q_MODELO = "Qwen/Qwen2.5-0.5B"
Q_BASE = "/workspace/headhunter/final_pod_d"                  # tudo no volume do pod (sobrevive a reiniciar o kernel; faça backup antes de apagar o pod)
Q_PASTA = f"{Q_BASE}/qwen0.5b"
Q_N_VAL = 1000                                                   # frases de validação (a célula 6 projeta o tempo; ajuste aqui)
Q_SEM = {"s0": 5000, "s1": 21000, "s2": 22000, "s3": 23000, "s4": 24000}   # sementes dos 5 grupos de 40 frases (s0-s2 = as do Colab)
Q_CFG = {"ioi":  dict(n_treino=1000, n_batch=10, bs_attr=10, bs_val=20),
         "mcqa": dict(n_treino=None, n_batch=5,  bs_attr=5,  bs_val=10)}   # n_treino=None: o treino inteiro (como o MIB)
Q_HH = ("HH-cercadinho-meio", "HH-cercadinho-eapig", "HH-livre-eapig")
Q_METODOS = Q_HH + ("EAP-IG",)
os.makedirs(Q_PASTA, exist_ok=True)
def q_trocar(model, tag):
    # muda o modelo e a pasta de resultados (cada modelo tem a sua)
    global Q_MODELO, Q_PASTA
    Q_MODELO = model; Q_PASTA = f"{Q_BASE}/{tag}"; os.makedirs(Q_PASTA, exist_ok=True); print("modelo:", Q_MODELO, "| resultados em", Q_PASTA)

def q_dtype():
    # MIB usa bfloat16; a T4 (capacidade 7.5) não tem bf16 nativo e fica lenta -> float32 lá. A100 (8.0+): bfloat16.
    return torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float32

def q_model(dtype=None):
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
        self.sec = time.time() - self.t0

def q_arq(task, name): return f"{Q_PASTA}/{task}/{name}/importances.json"
def q_json(task, name):
    return f"{Q_PASTA}/{task}/{name}.json"
def q_read(task, name, default=None):
    p = q_json(task, name); return json.load(open(p)) if os.path.exists(p) else ({} if default is None else default)
def q_gravar(task, name, data):
    os.makedirs(f"{Q_PASTA}/{task}", exist_ok=True); json.dump(data, open(q_json(task, name), "w"), indent=1)

def q_val(mm, task, n=None):
    ds = HFEAPDataset(f"mib-bench/{TASKS_TO_HF_NAMES[task]}", mm.tokenizer, split="validation", task=task)
    if n and n < len(ds): ds.head(n)
    return ds

def q_items(m, task):
    return items_mib(m, task, "train", Q_CFG[task]["n_treino"])

def q_receita(m, task, items):
    # registra a receita do HeadHunter (grupos por comprimento) e avisa se há poucos grupos grandes
    info = receita_mib(m, items, f"MIB_{task}"); g = info["groups"]
    L = [l for l, v in g.items() for _ in range(v)]; cob = sum(v for v in g.values() if v >= 40) / max(1, info["items"])
    print(f"   {task}: comprimento em tokens: média {np.mean(L):.0f}, máx {max(L)} | grupos com ≥ 40 frases cobrem {cob:.0%} dos itens")
    if cob < 0.5: print("   ⚠️ poucos grupos grandes: o HeadHunter vai sortear frases repetidas dentro de grupos pequenos. Me mande esta linha.")
    return info

def q_items_do_hh(m, items, semente, n=40, n_batch=10):
    # As MESMAS frases que o HeadHunter sorteia (mesmos grupos e mesmo sorteio da receita_mib), como itens do MIB.
    groups = defaultdict(list)
    for it in items:
        a = m.to_tokens(it[0], prepend_bos=True)[0]; b = m.to_tokens(it[1], prepend_bos=True)[0]
        if len(a) == len(b): groups[len(a)].append(it)
    out = []
    for b in range(n // n_batch):
        rnd = random.Random(semente + b); ok = [L for L in groups if len(groups[L]) >= n_batch] or [max(groups, key=lambda L: len(groups[L]))]
        L = rnd.choices(ok, weights=[len(groups[x]) for x in ok])[0]; g = groups[L]
        out += rnd.sample(g, n_batch) if len(g) >= n_batch else [rnd.choice(g) for _ in range(n_batch)]
    return out

def q_conferir_grafo(m, mm):
    # O HeadHunter (sem desagrupar o GQA) e o MIB (desagrupado) têm que ter EXATAMENTE as mesmas arestas, com os mesmos nomes.
    nL, nH = m.cfg.n_layers, m.cfg.n_heads
    hh_names = {name_edge(m, a) for a in hh.edge_parts(m, range(nL * nH), range(nL))}
    mib_names = set(Graph.from_model(mm).edges)
    assert hh_names == mib_names, f"arestas diferentes: só HH {len(hh_names - mib_names)}, só MIB {len(mib_names - hh_names)}"
    print(f"   ✅ grafo: {len(mib_names)} arestas, mesmos nomes no HeadHunter e no MIB")
    return len(mib_names)

def q_conferir_tokens(m, mm, items, k=20):
    # Os mesmos tokens que o MIB usa (tokenize_plus: BOS + texto, padding à direita) e os do HeadHunter.
    for j in range(min(k, len(items))):
        for txt in items[j][:2]:
            a = m.to_tokens(txt, prepend_bos=True)[0].cpu()
            t, mask, lens, _ = tokenize_plus(mm, [txt]); b = t[0, :int(lens[0])].cpu()
            assert a.shape == b.shape and bool((a == b).all()), f"tokens diferentes na frase {j}: HH {a.tolist()[:8]} × MIB {b.tolist()[:8]}"
    print(f"   ✅ tokens: {min(k, len(items))} pares conferidos, HeadHunter = MIB. Começo da 1ª frase:",
          [mm.tokenizer.decode(x) for x in tokenize_plus(mm, [items[0][0]])[0][0, :4].tolist()])

def q_probe_elapsed(mm, task, n=40):
    # Avalia um grafo aleatório em n frases (curva inteira, como o MIB) e projeta o tempo por método para vários Q_N_VAL.
    ds = q_val(mm, task, n); dl = ds.to_dataloader(batch_size=Q_CFG[task]["bs_val"])
    g = Graph.from_model(mm); g.scores[:] = torch.rand_like(g.scores)
    met = get_metric("logit_diff", task, mm.tokenizer, mm)
    with contextlib.redirect_stdout(io.StringIO()):
        with _Med() as md: evaluate_area_under_curve(mm, g, dl, partial(met, mean=False, loss=False), quiet=True, level="edge", absolute=True)
    per_sentence = md.sec / len(ds)
    print(f"   {task}: {md.sec:.0f} s para {len(ds)} frases ({per_sentence:.2f} s/frase) | projeção por método: "
          + ", ".join(f"{nv} frases ≈ {per_sentence * nv / 60:.0f} min" for nv in (200, 500, 1000))
          + f" | 20 avaliações (5 grupos × 4 métodos) com {Q_N_VAL} frases ≈ {20 * per_sentence * Q_N_VAL / 3600:.1f} h")
    return per_sentence

def q_heads(m, task, items):
    # Etapa das cabeças do HeadHunter (circuito de 90%), uma vez por tarefa; guarda no Drive.
    info = q_receita(m, task, items); d = q_read(task, "heads")
    if d: print(f"   (cabeças já feitas) {task}: {len(d['heads'])} cabeças, fidelidade {d['faith']:.0%}"); return d["heads"], d
    with _Med() as md: r = hh.analyze(m, f"MIB_{task}", prune=False, verbose=False)
    c = r["circuit"]; d = dict(heads=[int(x) for x in c["heads"]], faith=float(c["faithfulness"]), reached=bool(c["reached"]),
                                core=len(r["core"]["heads"]), mlps_core=r["core"]["mlps"], promoted=c["promoted"],
                                reverted=bool(c["reverted"]), sec=round(md.sec, 1), pico_mb=md.pico, items=info["items"], groups=info["groups"])
    q_gravar(task, "heads", d)
    print(f"   {task}: {len(d['heads'])}/{m.cfg.n_layers * m.cfg.n_heads} cabeças (fidelidade {d['faith']:.0%}, núcleo {d['core']}) em {md.sec:.0f} s")
    return d["heads"], d

def q_save(m, task, name, ar, key):
    if os.path.exists(q_arq(task, name)): return
    g = Graph.from_model(m); order = sorted(ar, key=key); N = len(order)
    assert N == len(g.edges), f"o HeadHunter vê {N} arestas, o grafo do MIB tem {len(g.edges)}"
    g.scores[:] = 0
    for k, a in enumerate(order): g.edges[name_edge(m, a)].score = float(N - k)
    os.makedirs(os.path.dirname(q_arq(task, name)), exist_ok=True); g.to_json(q_arq(task, name))

def q_gerar_hh(m, task, head, group):
    name = f"MIB_{task}"; nL = m.cfg.n_layers; c = Q_CFG[task]; without = Q_SEM[group]
    plano = (("mid", ("HH-cercadinho-meio",)), ("eapig", ("HH-cercadinho-eapig", "HH-livre-eapig")))
    dentro = set(hh.edge_parts(m, head, range(nL))); tempos = q_read(task, "tempos")
    try:
        for esquema, names in plano:
            if all(os.path.exists(q_arq(task, f"{n}-{group}")) for n in names): continue
            hh.IG_SCHEME = esquema
            with _Med() as md: ar, nt = scores_grafo_todo(m, name, n=40, n_batch=c["n_batch"], semente=without)
            tempos[f"HH-{esquema}-{group}"] = dict(sec=round(md.sec, 1), pico_mb=md.pico)
            for n in names:
                if "cercadinho" in n: q_save(m, task, f"{n}-{group}", ar, lambda a, nt=nt: (a not in dentro, -abs(nt[a])))
                else: q_save(m, task, f"{n}-{group}", ar, lambda a, nt=nt: -abs(nt[a]))
            q_gravar(task, "tempos", tempos); print(f"   {task} {group} HH ({esquema}): {md.sec:.0f} s, pico {md.pico if md.pico is None else round(md.pico)} MB → {', '.join(names)}")
            del ar, nt; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    finally:
        hh.IG_SCHEME = "mid"

def q_gerar_eapig(mm, task, items, group):
    name = f"EAP-IG-{group}"
    if os.path.exists(q_arq(task, name)): print(f"   (já existe) {task} {name}"); return
    c = Q_CFG[task]; it = q_items_do_hh(mm, items, Q_SEM[group], 40, c["n_batch"])
    dl = DataLoader(it, batch_size=c["bs_attr"], collate_fn=collate_EAP); g = Graph.from_model(mm)
    met = get_metric("logit_diff", task, mm.tokenizer, mm); tempos = q_read(task, "tempos")
    with _Med() as md:
        with torch.enable_grad():   # o HeadHunter desliga o gradiente globalmente; o EAP-IG precisa dele
            attribute(mm, g, dl, partial(met, mean=True, loss=True), "EAP-IG-inputs", "patching", ig_steps=5, intervention_dataloader=dl)
    os.makedirs(os.path.dirname(q_arq(task, name)), exist_ok=True); g.to_json(q_arq(task, name))
    tempos[name] = dict(sec=round(md.sec, 1), pico_mb=md.pico); q_gravar(task, "tempos", tempos)
    print(f"   {task} {group} EAP-IG oficial: {md.sec:.0f} s, pico {md.pico if md.pico is None else round(md.pico)} MB")
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

def q_corr_escores(task, groups):
    # Previsão P1: Spearman entre |nota do EAP-IG oficial| e a posição do HH-livre-eapig, por tipo de aresta.
    res = {}
    for gr in groups:
        a, b = q_arq(task, f"EAP-IG-{gr}"), q_arq(task, f"HH-livre-eapig-{gr}")
        if not (os.path.exists(a) and os.path.exists(b)): continue
        go, gh = Graph.from_json(a), Graph.from_json(b); idx = defaultdict(lambda: ([], []))
        for nm, e in go.edges.items():
            if nm == "input->m0": continue   # o HeadHunter põe esta aresta no topo de propósito (o EAP a subestima): fora da comparação
            cl = "q" if nm.endswith("<q>") else "k/v" if nm.endswith(("<k>", "<v>")) else "mlp/saída"
            idx[cl][0].append(e.matrix_index[0]); idx[cl][1].append(e.matrix_index[1])
        res[gr] = {cl: float(spearmanr(go.scores[r, c].abs().float().numpy(), gh.scores[r, c].float().numpy())[0])
                   for cl, (r, c) in ((k, (torch.tensor(v[0]), torch.tensor(v[1]))) for k, v in idx.items())}
        print(f"   {task} {gr}: Spearman HH-livre-eapig × EAP-IG oficial → " + " | ".join(f"{k} {v:.3f}" for k, v in res[gr].items()))
    q_gravar(task, "spearman", res); return res

def q_avaliar(mm, task, name, dl_val, n_val):
    key = f"{name}@{n_val}"; P = q_read(task, "scoreboard")
    if key in P: print(f"(já avaliado) {key}: CMD {P[key]['CMD']:.3f}"); return P[key]
    g = Graph.from_json(q_arq(task, name)); met = get_metric("logit_diff", task, mm.tokenizer, mm)
    with contextlib.redirect_stdout(io.StringIO()):
        with _Med() as md:
            out = evaluate_area_under_curve(mm, g, dl_val, partial(met, mean=False, loss=False), quiet=True, level="edge", absolute=True)
    P = q_read(task, "scoreboard"); P[key] = dict(CMD=out[2], CPR=out[1], faith=[round(x, 3) for x in out[4]], sec=round(md.sec), dtype=str(mm.cfg.dtype))
    q_gravar(task, "scoreboard", P); print(f"{key}: CMD {out[2]:.3f} | CPR {out[1]:.3f} | fid 1% {out[4][3]:.0%} | {md.sec:.0f} s"); return P[key]

def q_tabela(task, groups, n_val):
    P = q_read(task, "scoreboard"); pcts = ["0.1%", "0.2%", "0.5%", "1%", "2%", "5%", "10%", "20%", "50%", "100%"]
    print(f"\n=== {task}: {Q_MODELO}, {n_val} frases de validação, CMD (menor = melhor) ===")
    print(f"{'método':22s}" + "".join(f"{g:>9s}" for g in groups) + "   média ± dp |  fid. 1% (média) | fid. 2% (média)")
    for met in Q_METODOS:
        v = [P.get(f"{met}-{g}@{n_val}") for g in groups]; ok = [x for x in v if x]
        if not ok: continue
        cmd = np.array([x["CMD"] for x in ok])
        print(f"{met:22s}" + "".join(f"{x['CMD']:9.3f}" if x else f"{'-':>9s}" for x in v)
              + f"   {cmd.mean():.3f} ± {cmd.std():.3f} |      {np.mean([x['faith'][3] for x in ok]):.0%}        |      {np.mean([x['faith'][4] for x in ok]):.0%}")
    return P

def q_custo_fp32(mm32, task, items, group="s0"):
    # EAP-IG oficial em float32 (mesma precisão do HeadHunter): só tempo e memória; não grava importâncias.
    name = f"EAP-IG-fp32-{group}"; tempos = q_read(task, "tempos")
    if name in tempos: print(f"   (já medido) {name}: {tempos[name]['sec']} s, pico {tempos[name]['pico_mb']} MB"); return tempos[name]
    c = Q_CFG[task]; it = q_items_do_hh(mm32, items, Q_SEM[group], 40, c["n_batch"])
    dl = DataLoader(it, batch_size=c["bs_attr"], collate_fn=collate_EAP); g = Graph.from_model(mm32)
    met = get_metric("logit_diff", task, mm32.tokenizer, mm32)
    with _Med() as md:
        with torch.enable_grad():
            attribute(mm32, g, dl, partial(met, mean=True, loss=True), "EAP-IG-inputs", "patching", ig_steps=5, intervention_dataloader=dl)
    tempos = q_read(task, "tempos"); tempos[name] = dict(sec=round(md.sec, 1), pico_mb=md.pico); q_gravar(task, "tempos", tempos)
    print(f"   {task} {group} EAP-IG oficial em float32: {md.sec:.0f} s, pico {md.pico if md.pico is None else round(md.pico)} MB"); return tempos[name]

def q_previsoes(task, groups, n_val):
    # Confere as previsões B1-B10 que já têm dados. ✅ acertou, ❌ errou, ⏳ falta dado.
    P = q_read(task, "scoreboard"); T = q_read(task, "tempos"); S = q_read(task, "spearman")
    f = lambda met, g: P.get(f"{met}-{g}@{n_val}")
    comp = [g for g in groups if all(f(m_, g) for m_ in Q_METODOS)]; r = []
    def add(txt, ok): r.append(f"{'⏳' if ok is None else '✅' if ok else '❌'} {txt}")
    if len(comp) >= 3:
        cm = lambda met: float(np.mean([f(met, g)["CMD"] for g in comp])); fd = lambda met: float(np.mean([f(met, g)["faith"][3] for g in comp]))
        rl, rc = cm("HH-livre-eapig") / cm("EAP-IG"), cm("HH-cercadinho-meio") / cm("EAP-IG")
        if task == "ioi":
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
    if task == "ioi":
        g = "s0"
        if f"HH-meio-{g}" in T and f"EAP-IG-fp32-{g}" in T:
            t8 = T[f"HH-meio-{g}"]["sec"] / T[f"EAP-IG-fp32-{g}"]["sec"]; add(f"B8 tempo HH / EAP-IG, ambos float32 = {t8:.2f} (previsto 0,40-0,75)", 0.40 <= t8 <= 0.75)
            if T[f"HH-meio-{g}"]["pico_mb"] is not None and T[f"EAP-IG-fp32-{g}"]["pico_mb"]:
                t9 = T[f"HH-meio-{g}"]["pico_mb"] / T[f"EAP-IG-fp32-{g}"]["pico_mb"]; add(f"B9 memória de pico HH / EAP-IG, ambos float32 = {t9:.2f} (previsto 0,35-0,65)", 0.35 <= t9 <= 0.65)
        else: add("B8-B9 (rode a célula 11)", None)
        if f"HH-meio-{g}" in T and f"EAP-IG-{g}" in T:
            t10 = T[f"HH-meio-{g}"]["sec"] / T[f"EAP-IG-{g}"]["sec"]; add(f"B10 tempo HH (float32) / EAP-IG oficial (bf16) = {t10:.2f} (previsto 0,60-1,60)", 0.60 <= t10 <= 1.60)
    print(f"\nprevisões ({task}):"); [print("  " + x) for x in r]; return r


# ===================== CÉLULA 6 — definições do teste do pod D (não carrega modelo) =====================
import os, json, time, gc, shutil
from collections import defaultdict
import numpy as np, torch
from scipy.stats import spearmanr
assert hh.VERSION == "0.12.0", f"HeadHunter {hh.VERSION}: este notebook é da v0.12.0 (rode a célula 3)"
T_DEV = "cuda" if torch.cuda.is_available() else "cpu"
mb = lambda x: "-" if x is None else f"{x:,.0f} MB".replace(",", ".")
def t_sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()
def t_limpar(): gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

class _Cron:
    # soma o tempo gasto em hh.gradiente_entradas e hh.caches enquanto o bloco roda (o resto = produto de matrizes + montagem)
    def __enter__(self):
        self.t = defaultdict(float); self.orig = {}
        for name in ("input_gradients", "caches"):
            f = getattr(hh, name); self.orig[name] = f
            def w(*a, _f=f, _n=name, **k):
                t_sync(); t0 = time.time(); r = _f(*a, **k); t_sync(); self.t[_n] += time.time() - t0; return r
            setattr(hh, name, w)
        return self
    def __exit__(self, *a):
        for n, f in self.orig.items(): setattr(hh, n, f)

def t_scores(m, task, semente=5000, n=40, n_batch=10, ig_steps=5, esquema="eapig", lc=None):
    # Nota IG de TODAS as arestas (como notas_grafo_todo), com tempo, pico de memória e divisão do tempo.
    # lc = hh.LOTE_CALCULO (0 = sem fundir; 20/40 = pedaço de cálculo; None = o padrão da biblioteca, que dimensiona pela memória livre).
    nL, nH = m.cfg.n_layers, m.cfg.n_heads; ar = hh.edge_parts(m, range(nL * nH), range(nL))
    old = (hh.IG_SCHEME, hh.COMPUTE_BATCH, hh.input_gradients); g0 = hh.input_gradients; vistos = []
    hh.IG_SCHEME = esquema
    if lc is not None: hh.COMPUTE_BATCH = lc
    def g_log(m_, T, *a, **k): vistos.append(int(T["tk"].shape[0])); return g0(m_, T, *a, **k)
    hh.input_gradients = g_log
    try:
        t_limpar()
        with _Med() as md, _Cron() as cr:
            bat = [hh.get_task(m, f"MIB_{task}", semente + b, n_batch) for b in range(n // n_batch)]; Cs = [hh.caches(m, T) for T in bat]
            score = hh.edge_scores(m, bat, Cs, ar, ig_steps)
            del Cs, bat
        info = dict(sec=round(md.sec, 2), pico_mb=md.pico, grad_s=round(cr.t["input_gradients"], 2), caches_s=round(cr.t["caches"], 2),
                    rest_s=round(md.sec - cr.t["input_gradients"] - cr.t["caches"], 2), tams_chamadas=vistos, lc=("padrão" if lc is None else lc), n_batch=n_batch)
    finally:
        hh.IG_SCHEME, hh.COMPUTE_BATCH, hh.input_gradients = old
        m.reset_hooks(); torch.set_grad_enabled(False)
    vec = np.array([score[a] for a in ar], dtype=np.float64); t_limpar()
    return ar, vec, info

def t_classes(ar): return np.array([0 if a[2] == "q" else 1 if a[2] in ("k", "v") else 2 for a in ar])
def t_spear(x, y, cls):
    # Spearman de |nota| por tipo de aresta (a aresta input->m0 vale infinito no HeadHunter: fica fora)
    ok = np.isfinite(x) & np.isfinite(y); out = {}
    for c, name in enumerate(("q", "k/v", "mlp/saída")):
        s = ok & (cls == c); out[name] = float(spearmanr(np.abs(x[s]), np.abs(y[s]))[0])
    return out

def t_eapig(mm, task, items, group, name, bs=None, n=40, save=True, aquecer=True):
    # EAP-IG oficial: tempo e memória de pico (e, se salvar, o grafo com as importâncias). `bs` = tamanho do lote do oficial (padrão do MIB = 10).
    tempos = q_read(task, "tempos"); c = Q_CFG[task]; bs = bs or c["bs_attr"]
    if name in tempos and (not save or os.path.exists(q_arq(task, name))):
        print(f"   (já medido) {name}: {tempos[name].get('sec')} s, pico {mb(tempos[name].get('pico_mb'))}"); return tempos[name]
    it = q_items_do_hh(mm, items, Q_SEM[group], n, c["n_batch"])
    def rodar(its):
        dl = DataLoader(its, batch_size=bs, collate_fn=collate_EAP); g = Graph.from_model(mm)
        met = get_metric("logit_diff", task, mm.tokenizer, mm); t_limpar()
        with _Med() as md:
            with torch.enable_grad():
                attribute(mm, g, dl, partial(met, mean=True, loss=True), "EAP-IG-inputs", "patching", ig_steps=5, intervention_dataloader=dl)
        return g, md
    try:
        if aquecer: rodar(it[:c["n_batch"]])
        g, md = rodar(it)
    except torch.cuda.OutOfMemoryError:
        t_limpar(); tempos = q_read(task, "tempos"); tempos[name] = dict(oom=True, bs=bs); q_gravar(task, "tempos", tempos)
        print(f"   {name}: SEM MEMÓRIA (lote {bs})"); return tempos[name]
    if save: os.makedirs(os.path.dirname(q_arq(task, name)), exist_ok=True); g.to_json(q_arq(task, name))
    tempos = q_read(task, "tempos"); tempos[name] = dict(sec=round(md.sec, 1), pico_mb=md.pico, bs=bs, dtype=str(mm.cfg.dtype)); q_gravar(task, "tempos", tempos)
    print(f"   {task} {group} EAP-IG oficial [{name}] ({mm.cfg.dtype}, lote {bs}): {md.sec:.1f} s, pico {mb(md.pico)}"); del g; t_limpar(); return tempos[name]

def t_hh_livre(m, task, group, lc=None):
    # HH-livre-eapig (pontos 0; 0,2; …; 0,8 = como o oficial) do grupo, em float32, gravado no formato do MIB. Devolve o tempo/memória.
    name = f"HH-livre-eapig-{group}"
    ar, vec, info = t_scores(m, task, semente=Q_SEM[group], n_batch=Q_CFG[task]["n_batch"], esquema="eapig", lc=lc)
    nt = dict(zip(ar, vec)); q_save(m, task, name, ar, lambda a: -abs(nt[a]))
    print(f"   {task} {group} HH-livre-eapig (float32): {info['sec']} s, pico {mb(info['pico_mb'])}, chamadas do gradiente {info['tams_chamadas']}"); return info

def t_classes_grafo(g):
    idx = defaultdict(lambda: ([], []))
    for nm, e in g.edges.items():
        if nm == "input->m0": continue
        cl = "q" if nm.endswith("<q>") else "k/v" if nm.endswith(("<k>", "<v>")) else "mlp/saída"
        idx[cl][0].append(e.matrix_index[0]); idx[cl][1].append(e.matrix_index[1])
    return {cl: (torch.tensor(r), torch.tensor(c)) for cl, (r, c) in idx.items()}

def t_comparar(task, group, a, b):
    # Spearman de |nota| por tipo de aresta entre dois grafos salvos (nomes de pasta sem o grupo)
    g = {k: Graph.from_json(q_arq(task, n.format(group))) for k, n in (("a", a), ("b", b))}
    cl = t_classes_grafo(g["b"])
    return {k: float(spearmanr(g["a"].scores[r, c].abs().float().numpy(), g["b"].scores[r, c].abs().float().numpy())[0]) for k, (r, c) in cl.items()}

def t_rep(f, n):
    # n repetições de uma medida; devolve a lista de dicts
    return [f(i) for i in range(n)]

def t_ablation(m, T, means, heads):
    # frase limpa; as cabeças dadas recebem a média da sua saída. Devolve (prob. média da certa, dif. de logit certa−errada média, acerto binário)
    nH = m.cfg.n_heads; per = defaultdict(list)
    for c_ in heads: per[int(c_) // nH].append(int(c_) % nH)
    hooks = []
    for L, hs in per.items():
        def hk(z, hook, L=L, hs=hs): z[:, :, hs] = means[L][:, :, hs]; return z
        hooks.append((f"blocks.{L}.attn.hook_z", hk))
    ar = torch.arange(T["tk"].shape[0], device=T["tk"].device); lg = hh.final_logits(m, T["tk"], None, hooks)[:, -1].float()
    ld = lg[ar, T["target"]] - lg[ar, T["wrong"]]
    return lg.softmax(-1)[ar, T["target"]].mean().item(), ld.mean().item(), (ld > 0).float().mean().item()

def t_cap(m, task="CAP", n=40, ks=(1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 500, 700, 900, 1000, 1023)):
    t0 = time.time(); nL, nH = m.cfg.n_layers, m.cfg.n_heads; N = nL * nH; ks = sorted({k for k in ks if k < N} | {N - 1})
    T = hh.get_task(m, task, 0, n)
    _, cl = m.run_with_cache(T["tk"], names_filter=lambda x: x.endswith("attn.hook_z"))
    means = {L: cl[f"blocks.{L}.attn.hook_z"].mean(0, keepdim=True) for L in range(nL)}; del cl
    tf = torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32 = True   # como no pod B (só médias de probabilidade)
    try:
        p0, ld0, a0 = t_ablation(m, T, means, []); pt, ldt, at = t_ablation(m, T, means, range(N))
        uni = np.array([t_ablation(m, T, means, [c_]) for c_ in range(N)])      # (N, 3): p, ld, acerto com UMA cabeça ablacionada
        ord_p = [int(c_) for c_ in np.argsort(-(p0 - uni[:, 0]))]; ord_ld = [int(c_) for c_ in np.argsort(-(ld0 - uni[:, 1]))]
        ord_al = [int(c_) for c_ in np.random.default_rng(0).permutation(N)]
        sd = lambda o, k: (p0 - t_ablation(m, T, means, o[:k])[0]) / max(p0 - pt, 1e-9)
        dl = lambda o, k: (ld0 - t_ablation(m, T, means, o[:k])[1]) / max(ld0 - ldt, 1e-9)
        curve = {k: dict(p_order_p=sd(ord_p, k), ld_order=dl(ord_ld, k), p_random=sd(ord_al, k), ld_random=dl(ord_al, k),
                         accuracy_order_ld=t_ablation(m, T, means, ord_ld[:k])[2]) for k in ks}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf
    prim = lambda col, a: next((k for k in ks if curve[k][col] >= a), None)
    r = dict(p0=p0, p_all=pt, dif_p=p0 - pt, ld0=ld0, ld_all=ldt, dif_ld=ld0 - ldt, acerto0=a0, accuracy_all=at,
             k80_p=prim("p_order_p", 0.8), k80_ld=prim("ld_order", 0.8), k50_p=prim("p_order_p", 0.5), k50_ld=prim("ld_order", 0.5),
             k80_p_random=prim("p_random", 0.8), top_p=[f"{c_ // nH}.{c_ % nH}" for c_ in ord_p[:8]], top_ld=[f"{c_ // nH}.{c_ % nH}" for c_ in ord_ld[:8]],
             drop_1a_p=float(p0 - uni[ord_p[0], 0]), curve={str(k): v for k, v in curve.items()}, N=N, elapsed=round(time.time() - t0, 1))
    print(f"\n{m.cfg.model_name} / {task}: p0 = {p0:.3f} | todas as cabeças ablacionadas: p = {pt:.3f} | p0 − p_todas = {p0 - pt:.3f}")
    print(f"   dif. de logit (certa − errada): {ld0:.2f} → {ldt:.2f} (diferença {ld0 - ldt:.2f}) | acerto binário {a0:.0%} → {at:.0%}")
    print(f"   {'k':>5s} {'dano (prob.; ordem p)':>22s} {'dano (logit; ordem logit)':>26s} {'dano (prob.; aleatório)':>24s} {'acerto (ordem logit)':>21s}")
    for k in ks: v = curve[k]; print(f"   {k:5d} {v['p_order_p']:22.0%} {v['ld_order']:26.0%} {v['p_random']:24.0%} {v['accuracy_order_ld']:21.0%}")
    print(f"   k p/ 50% do dano: prob. {r['k50_p']} | logit {r['k50_ld']}; p/ 80%: prob. {r['k80_p']} | logit {r['k80_ld']} | aleatório {r['k80_p_random']} (None = não chegou até {N - 1}) | {r['elapsed']:.0f} s")
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
    s = {c: rep(E, c + "-r", "sec") for c in ("L0", "L20", "L40")}; pk = {c: rep(E, c + "-r", "pico_mb") for c in ("L0", "L20", "L40")}
    o16, o32 = rep(T, "EAP-IG-bf16-r", "sec"), rep(T, "EAP-IG-fp32-r", "sec")
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
    if all(k in E and E[k].get("sec") for k in ("L0-s1", "L40-s1")):
        r = E["L40-s1"]["sec"] / E["L0-s1"]["sec"]; sps = min(E[k]["spearman_min"] for k in ("L20-s1", "L40-s1") if k in E and E[k].get("spearman_min") is not None)
        add(f"E9 grupo s1 (uma medida): tempo L40 / L0 = {r:.2f} (previsto ≤ 0,90; chamadas L40: {E['L40-s1']['tams_chamadas']}), Spearman mín {sps:.6f} (≥ 0,99999)", r <= 0.90 and sps >= 0.99999)
    else: add("E9 (falta o grupo s1)", None)
    ck = lj("qwen1.5b", "ioi", "e_conversao")
    if ck: add(f"E8 oficial fp32 convertido na GPU (.float()) × carregado direto em fp32: diferença máxima dos logits {ck['dif_max']:.2e} (previsto ≤ 1e-4)", ck["dif_max"] <= 1e-4)
    # ---- (F) 7B ----
    F = lj("qwen7b", "ioi", "f_7b") or {}; T7 = lj("qwen7b", "ioi", "tempos") or {}
    b0, b1 = T7.get("EAP-IG-bf16-s0", {}), T7.get("EAP-IG-bf16-s1", {})
    h0 = F.get("HH-padrao-s0", {})
    if h0.get("sec") and b0.get("sec"):
        r = h0["sec"] / b0["sec"]; add(f"F1 tempo HH padrão / oficial bf16 no 7B (s0) = {r:.2f} (previsto 0,5-1,0)" + (f"; s1: {F['HH-padrao-s1']['sec'] / b1['sec']:.2f}" if F.get('HH-padrao-s1', {}).get('sec') and b1.get('sec') else ""), 0.5 <= r <= 1.0)
        add(f"F3 oficial bf16 no 7B: pico {mb(b0.get('pico_mb'))} (previsto 12.000-30.000 MB)", b0.get("pico_mb") is not None and 12000 <= b0["pico_mb"] <= 30000)
    else: add("F1/F3 (rode a célula 8)", None)
    f32 = T7.get("EAP-IG-fp32-s0")
    if f32 is not None or F.get("fp32_load"): add(f"F2 oficial fp32 no 7B: {'NÃO coube' if (F.get('fp32_load') or f32.get('oom')) else 'coube (' + str(f32.get('sec')) + ' s, pico ' + mb(f32.get('pico_mb')) + ')'} (previsto: não cabe)", bool(F.get("fp32_load") or f32.get("oom")))
    else: add("F2 (falta)", None)
    if h0: add(f"F4 HH livre fp32 no 7B: " + ("SEM MEMÓRIA" if h0.get("oom") else f"pico {mb(h0.get('pico_mb'))} com lotes de {h0.get('n_batch')} (previsto: cabe com lote 10, pico 8.000-20.000 MB)"), bool(h0.get("pico_mb")) and h0.get("n_batch") == 10 and 8000 <= h0["pico_mb"] <= 20000)
    else: add("F4 (falta)", None)
    if f32 and not f32.get("oom") and F.get("sp_hh_fp32") and h0.get("sec"):
        sp = F["sp_hh_fp32"]; r = h0["sec"] / f32["sec"]
        add(f"F5 HH × oficial fp32 no 7B: Spearman " + ", ".join(f"{k} {v:.4f}" for k, v in sp.items()) + f" (≥ 0,99) e tempo HH/oficial fp32 = {r:.2f} (≤ 0,60)", min(sp.values()) >= 0.99 and r <= 0.60)
    else: add("F5 (só pontuável se o oficial fp32 couber)", None)
    sp = F.get("sp_default_L0")
    add(f"F6 Spearman HH padrão × L0 no 7B = {sp:.6f} (previsto ≥ 0,99999)" if sp is not None else "F6 (falta)", None if sp is None else sp >= 0.99999)
    head = lj("qwen7b", "ioi", "heads")
    add(f"F7 etapa das cabeças do HH no 7B: {head['sec']} s, {len(head['heads'])} cabeças, fidelidade {head['faith']:.0%} (previsto < 240 s)" if head else "F7 (falta)", None if not head else head["sec"] < 240)
    # ---- (G) CAP Qwen2.5-7B por diferença de logit ----
    k = lj("qwen7b", "cap", "cap_check"); kp = lj("pythia6.9b", "cap", "cap_check")
    if k:
        add(f"G1 Qwen-7B CAP: p0 − p_todas = {k['dif_p']:.3f} (p0 {k['p0']:.3f}; previsto ≥ 0,10)", k["dif_p"] >= 0.10)
        add(f"G2 k para 80% do dano (logit) = {k['k80_ld']} e para 50% = {k['k50_ld']} (previsto 80%: 5-60)", k["k80_ld"] is not None and 5 <= k["k80_ld"] <= 60)
        pyk = (kp or {}).get("k80_ld", 50)
        add(f"G3 Pythia-6.9B k80 (logit) = {pyk} × Qwen {k['k80_ld']}: razão {pyk / k['k80_ld']:.2f} (previsto ≥ 1,2: o Pythia precisa de mais cabeças)" if k["k80_ld"] else "G3 (Qwen não chegou a 80%)", bool(k["k80_ld"]) and pyk / k["k80_ld"] >= 1.2)
        c = k["curve"].get("100") or k["curve"][max(k["curve"], key=lambda x: int(x))]
        add(f"G4 dano (logit) com 100 cabeças aleatórias = {c['ld_random']:.0%} (previsto < 10%); ordenadas {c['ld_order']:.0%}", c["ld_random"] < 0.10)
        add(f"G5 acerto binário com todas as cabeças ablacionadas: {k['acerto0']:.0%} → {k['accuracy_all']:.0%} (previsto < 90%)", k["accuracy_all"] < 0.90)
    else: add("G1-G5 (rode a célula 9)", None)
    print("\nprevisões:"); [print("  " + x) for x in out]; return out


# ===================== CÉLULA 7 — (E) v0.11.5 no Qwen2.5-1.5B: lote de cálculo 0 / 20 / 40, 3 repetições, com o EAP-IG oficial ao lado (~10 min) =====================
q_trocar("Qwen/Qwen2.5-1.5B", "qwen1.5b")
m = hh.load_model(Q_MODELO); mm16 = q_model(torch.bfloat16); mm32 = q_model(torch.float32)
print(f"HH: {next(m.parameters()).dtype} | oficial: {mm16.cfg.dtype} e {mm32.cfg.dtype} | GQA {m.cfg.n_heads} cabeças / {m.cfg.n_key_value_heads} kv | {m.cfg.n_layers} camadas | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
q_conferir_grafo(m, mm16); itens15 = q_items(m, "ioi"); q_receita(m, "ioi", itens15); q_conferir_tokens(m, mm16, itens15)
# E8: o oficial em fp32 será montado no 7B subindo em bf16 e convertendo na GPU (.float()); aqui confiro que isso dá o mesmo que carregar direto em fp32
if not os.path.exists(q_json("ioi", "e_conversao")):
    mmc = q_model(torch.bfloat16).float(); mmc.cfg.dtype = torch.float32
    tk = mm32.to_tokens(["When Mary and John went to the store, John gave a drink to", "The capital of France is the city of"], prepend_bos=True)
    with torch.no_grad(): dif = (mm32(tk).float() - mmc(tk).float()).abs().max().item()
    q_gravar("ioi", "e_conversao", dict(dif_max=dif)); print(f"   E8: diferença máxima dos logits (oficial fp32 direto × convertido na GPU): {dif:.2e}"); del mmc; t_limpar()
NR = 3
for r in range(NR):
    t_eapig(mm16, "ioi", itens15, "s0", f"EAP-IG-bf16-r{r}", save=False, aquecer=(r == 0))
    t_eapig(mm32, "ioi", itens15, "s0", f"EAP-IG-fp32-r{r}", save=False, aquecer=(r == 0))
del mm16, mm32; t_limpar()
E = q_read("ioi", "e_hh", {}); arq0 = f"{Q_PASTA}/ioi/nota_L0_r0.npy"
nL, nH = m.cfg.n_layers, m.cfg.n_heads; ar_ = hh.edge_parts(m, range(nL * nH), range(nL)); cls_ = t_classes(ar_)
if not E: print("   aquecendo (1 lote, não conta) ..."); t_scores(m, "ioi", semente=Q_SEM["s0"], n=10, n_batch=10, lc=0)
for r in range(NR):
    for cod, lc in (("L0", 0), ("L20", 20), ("L40", 40)):
        k = f"{cod}-r{r}"
        if k in E: print(f"   (já feito) {k}"); continue
        try: _, vec, info = t_scores(m, "ioi", semente=Q_SEM["s0"], lc=lc)
        except torch.cuda.OutOfMemoryError:
            t_limpar(); E[k] = dict(oom=True); q_gravar("ioi", "e_hh", E); print(f"   {k}: SEM MEMÓRIA"); continue
        if k == "L0-r0": np.save(arq0, vec)
        sp = t_spear(np.load(arq0), vec, cls_) if os.path.exists(arq0) else None
        E[k] = dict(**info, spearman=sp, spearman_min=min(sp.values()) if sp else None); q_gravar("ioi", "e_hh", E)
        print(f"   {k}: {info['sec']} s (gradiente {info['grad_s']}, caches {info['caches_s']}, resto {info['rest_s']}) | pico {mb(info['pico_mb'])} | chamadas do gradiente {info['tams_chamadas']}" + (f" | Spearman × L0-r0 mín {min(sp.values()):.6f}" if sp and k != "L0-r0" else ""))
# grupo s1 (outras 40 frases, uma medida de cada): o pedaço fundido só junta lotes de MESMO comprimento, então o ganho depende de quantos lotes caem no mesmo comprimento
arq1 = f"{Q_PASTA}/ioi/nota_L0_s1.npy"
for cod, lc in (("L0", 0), ("L20", 20), ("L40", 40)):
    k = f"{cod}-s1"
    if k in E: print(f"   (já feito) {k}"); continue
    try: _, vec, info = t_scores(m, "ioi", semente=Q_SEM["s1"], lc=lc)
    except torch.cuda.OutOfMemoryError: t_limpar(); E[k] = dict(oom=True); q_gravar("ioi", "e_hh", E); print(f"   {k}: SEM MEMÓRIA"); continue
    if cod == "L0": np.save(arq1, vec)
    sp = t_spear(np.load(arq1), vec, cls_) if os.path.exists(arq1) else None
    E[k] = dict(**info, spearman=sp, spearman_min=min(sp.values()) if sp else None); q_gravar("ioi", "e_hh", E)
    print(f"   {k}: {info['sec']} s (gradiente {info['grad_s']}) | pico {mb(info['pico_mb'])} | chamadas do gradiente {info['tams_chamadas']}" + (f" | Spearman × L0-s1 mín {min(sp.values()):.6f}" if sp and cod != "L0" else ""))
T = q_read("ioi", "tempos")
print("\n=== (E) Qwen2.5-1.5B, IOI, 40 frases, 5 passos: medianas das 3 repetições ===")
for rot, ks in (("EAP-IG oficial bf16 (lote 10)", [k for k in T if k.startswith("EAP-IG-bf16-r")]), ("EAP-IG oficial fp32 (lote 10)", [k for k in T if k.startswith("EAP-IG-fp32-r")]),
                ("HH L0 (sem fundir)", [k for k in E if k.startswith("L0-r")]), ("HH L20", [k for k in E if k.startswith("L20-r")]), ("HH L40", [k for k in E if k.startswith("L40-r")])):
    D = T if rot.startswith("EAP") else E; v = [D[k] for k in ks if D[k].get("sec")]
    if v: print(f"  {rot:32s} tempo {np.median([x['sec'] for x in v]):6.1f} s (min {min(x['sec'] for x in v):.1f}, máx {max(x['sec'] for x in v):.1f}) | pico {mb(np.median([x['pico_mb'] for x in v if x.get('pico_mb') is not None]) if any(x.get('pico_mb') is not None for x in v) else None)}")
t_previsoes()


# ===================== CÉLULA 8 — (F) o mesmo no Qwen2.5-7B: EAP-IG oficial (bf16 e fp32) e HeadHunter, um modelo de cada vez (~25-45 min; baixa ~15 GB) =====================
del m; t_limpar()
q_trocar("Qwen/Qwen2.5-7B", "qwen7b")
F = q_read("ioi", "f_7b", {})
# --- 1) EAP-IG oficial em bf16 (15 GB) ---
mm = q_model(torch.bfloat16); itens7 = q_items(mm, "ioi"); q_receita(mm, "ioi", itens7)
print(f"oficial bf16: {sum(p.numel() for p in mm.parameters()) / 1e9:.1f} B parâmetros | GQA {mm.cfg.n_heads} cabeças / {mm.cfg.n_key_value_heads} kv | {mm.cfg.n_layers} camadas")
for gr in ("s0", "s1"): t_eapig(mm, "ioi", itens7, gr, f"EAP-IG-bf16-{gr}", save=False, aquecer=(gr == "s0"))
del mm; t_limpar()
# --- 2) EAP-IG oficial em fp32 (30 GB): sobe em bf16 e converte na GPU (não estoura a RAM). Pode não caber: o "sem memória" também é resultado ---
try:
    mm = q_model(torch.bfloat16).float(); mm.cfg.dtype = torch.float32; t_limpar()
    print(f"oficial fp32: pesos {(torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0):.1f} GB na GPU")
    for gr in ("s0", "s1"): t_eapig(mm, "ioi", itens7, gr, f"EAP-IG-fp32-{gr}", save=(gr == "s0"), aquecer=(gr == "s0"))
except torch.cuda.OutOfMemoryError:
    F["fp32_load"] = dict(oom=True); q_gravar("ioi", "f_7b", F); print("   oficial fp32: não coube nem para carregar")
finally:
    try: del mm
    except NameError: pass
    t_limpar()
# --- 3) HeadHunter (float32; a biblioteca dimensiona o pedaço pela memória livre e cai pela metade se faltar) ---
m = hh.load_model(Q_MODELO, hf_id=Q_MODELO, convert_on="gpu"); q_receita(m, "ioi", itens7)
print(f"HH: {next(m.parameters()).dtype} | pesos {(torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0):.1f} GB | GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
def f_hh(gr, lc, cod):
    key = f"HH-{cod}-{gr}"
    if key in F: print(f"   (já feito) {key}"); return
    for n_batch in (10, 5):   # se não couber com lotes de 10 frases, tenta 5 (e o resultado fica marcado: frases diferentes, só vale para tempo e memória)
        try: ar, vec, info = t_scores(m, "ioi", semente=Q_SEM[gr], n_batch=n_batch, esquema="eapig", lc=lc); break
        except torch.cuda.OutOfMemoryError: t_limpar(); print(f"   {key}: SEM MEMÓRIA com lotes de {n_batch}")
    else: F[key] = dict(oom=True); q_gravar("ioi", "f_7b", F); return
    F[key] = info; arq = f"{Q_PASTA}/ioi/nota_padrao_{gr}.npy"
    if cod == "default":
        np.save(arq, vec)
        if gr == "s0": nt = dict(zip(ar, vec)); q_save(m, "ioi", f"HH-livre-eapig-{gr}", ar, lambda a: -abs(nt[a]))
    elif os.path.exists(arq) and F.get(f"HH-padrao-{gr}", {}).get("n_batch") == n_batch:
        F[f"sp_padrao_L0"] = float(min(t_spear(np.load(arq), vec, t_classes(ar)).values()))
    q_gravar("ioi", "f_7b", F)
    print(f"   {key}: {info['sec']} s (gradiente {info['grad_s']}, caches {info['caches_s']}, resto {info['rest_s']}) | pico {mb(info['pico_mb'])} | lotes de {n_batch} | chamadas do gradiente {info['tams_chamadas']}")
f_hh("s0", None, "default"); f_hh("s0", 0, "L0"); f_hh("s1", None, "default")
cab7, info7 = q_heads(m, "ioi", itens7)    # etapa das cabeças (circuito de 90%): o custo a mais do ponta a ponta
if os.path.exists(q_arq("ioi", "EAP-IG-fp32-s0")) and os.path.exists(q_arq("ioi", "HH-livre-eapig-s0")):
    F["sp_hh_fp32"] = t_comparar("ioi", "s0", "HH-livre-eapig-{}", "EAP-IG-fp32-{}"); q_gravar("ioi", "f_7b", F)
    print("   HH × oficial fp32 (7B, s0), Spearman de |nota|: " + " | ".join(f"{k} {v:.4f}" for k, v in F["sp_hh_fp32"].items()))
T7 = q_read("ioi", "tempos")
print("\n=== (F) Qwen2.5-7B, IOI, 40 frases, 5 passos ===")
for k in sorted(T7): print(f"  {'EAP-IG oficial ' + k.replace('EAP-IG-', ''):28s} " + ("SEM MEMÓRIA" if T7[k].get("oom") else f"{T7[k]['sec']:7.1f} s | pico {mb(T7[k].get('pico_mb'))}"))
for k in sorted(F):
    if k.startswith("HH-"): print(f"  {k:28s} " + ("SEM MEMÓRIA" if F[k].get("oom") else f"{F[k]['sec']:7.1f} s | pico {mb(F[k].get('pico_mb'))} | lotes de {F[k].get('n_batch')}"))
print(f"  etapa das cabeças do HH: {info7['sec']} s, pico {mb(info7['pico_mb'])}, {len(cab7)} cabeças, fidelidade {info7['faith']:.0%}")
t_previsoes()


# ===================== CÉLULA 9 — (G) CAP no Qwen2.5-7B: a necessidade pela DIFERENÇA DE LOGIT (e não pela probabilidade) (~10-15 min) =====================
Q_QK = f"{Q_BASE}/qwen7b/cap"; os.makedirs(Q_QK, exist_ok=True)
if os.path.exists(f"{Q_QK}/cap_check.json"): print("(já feito) CAP do Qwen2.5-7B"); cap7 = json.load(open(f"{Q_QK}/cap_check.json"))
else:
    if "m" not in globals() or m is None: m = hh.load_model("Qwen/Qwen2.5-7B", hf_id="Qwen/Qwen2.5-7B", convert_on="gpu")
    print("acerto do modelo no CAP (top-1 do vocabulário):", f"{hh.accuracy(m, 'CAP'):.0%}")
    cap7 = t_cap(m, "CAP", ks=(1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200, 300, 500, 700, 783)); json.dump(cap7, open(f"{Q_QK}/cap_check.json", "w"), indent=1)
t_previsoes()


# ===================== CÉLULA 10 — resumo final (cole TUDO que esta célula imprimir no chat) =====================
LJ = lambda p: json.load(open(f"{Q_BASE}/{p}")) if os.path.exists(f"{Q_BASE}/{p}") else {}
T15, E = LJ("qwen1.5b/ioi/tempos.json"), LJ("qwen1.5b/ioi/e_hh.json")
print("== (E) 1.5B: EAP-IG oficial =="); [print(f"  {k:22s} {v.get('sec')} s  pico {mb(v.get('pico_mb'))}" + ("  SEM MEMÓRIA" if v.get("oom") else "")) for k, v in sorted(T15.items())]
print("\n== (E) 1.5B: HeadHunter v0.11.5 =="); [print(f"  {k:8s} {v.get('sec')} s (grad {v.get('grad_s')}, caches {v.get('caches_s')}, resto {v.get('rest_s')}) pico {mb(v.get('pico_mb'))} chamadas {v.get('tams_chamadas')} Spearman×L0 (mesmo grupo) mín {v.get('spearman_min')}" + ("  SEM MEMÓRIA" if v.get("oom") else "")) for k, v in sorted(E.items())]
print("  E8:", LJ("qwen1.5b/ioi/e_conversao.json"))
T7, F = LJ("qwen7b/ioi/tempos.json"), LJ("qwen7b/ioi/f_7b.json")
print("\n== (F) 7B: EAP-IG oficial =="); [print(f"  {k:22s} " + ("SEM MEMÓRIA" if v.get("oom") else f"{v.get('sec')} s  pico {mb(v.get('pico_mb'))}")) for k, v in sorted(T7.items())]
print("\n== (F) 7B: HeadHunter ==")
for k, v in sorted(F.items()):
    if isinstance(v, dict) and k.startswith("HH-"): print(f"  {k:16s} " + ("SEM MEMÓRIA" if v.get("oom") else f"{v.get('sec')} s (grad {v.get('grad_s')}, caches {v.get('caches_s')}, resto {v.get('rest_s')}) pico {mb(v.get('pico_mb'))} lotes de {v.get('n_batch')} chamadas {v.get('tams_chamadas')}"))
    else: print(f"  {k:16s} {v}")
d7 = LJ("qwen7b/ioi/cabecas.json")
if d7: print(f"  etapa das cabeças: {d7['sec']} s, pico {mb(d7['pico_mb'])}, {len(d7['heads'])} cabeças, fidelidade {d7['faith']:.0%}")
cap = LJ("qwen7b/cap/cap_check.json"); capp = LJ("pythia6.9b/cap/cap_check.json")
if cap:
    print(f"\n== (G) CAP Qwen2.5-7B ==\n  p0 {cap['p0']:.3f} | p_todas {cap['p_all']:.3f} | diferença {cap['dif_p']:.3f} | logit {cap['ld0']:.2f} → {cap['ld_all']:.2f} | acerto {cap['acerto0']:.0%} → {cap['accuracy_all']:.0%}")
    print(f"  k p/ 50%: prob {cap['k50_p']} logit {cap['k50_ld']} | k p/ 80%: prob {cap['k80_p']} logit {cap['k80_ld']} aleatório(prob) {cap['k80_p_random']} | top (logit) {cap['top_ld']}")
    for k, v in cap["curve"].items(): print(f"    k={k:>5s}: prob(ordem p) {v['p_order_p']:.0%} | logit(ordem logit) {v['ld_order']:.0%} | logit(aleatório) {v['ld_random']:.0%} | acerto {v['accuracy_order_ld']:.0%}")
t_previsoes()
shutil.make_archive("/workspace/teste_pod_d_resultados", "zip", Q_BASE); print("\nbackup: /workspace/teste_pod_d_resultados.zip (baixe pelo JupyterLab antes de apagar o pod)")
