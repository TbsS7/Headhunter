"""Minimal HeadHunter run. Needs a GPU for anything beyond GPT-2 small (it also works on CPU for small models).
    python examples/quickstart.py [model]     # default: gpt2-small
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import headhunter as hh

nome = sys.argv[1] if len(sys.argv) > 1 else "gpt2-small"
m = hh.carregar(nome)
nH = m.cfg.n_heads
assert hh.autoteste(m, "IOI"), "self-test failed: do not trust the results on this model"

r = hh.analisar(m, "IOI")
print("core heads (layer.head):", [f"{c // nH}.{c % nH}" for c in r["nucleo"]["cabecas"]])
print("circuit:", len(r["circuito"]["cabecas"]), "of", m.cfg.n_layers * nH, "heads, faithfulness", round(r["circuito"]["fidelidade"], 3))
hh.validar_circuito(m, "IOI", r)
