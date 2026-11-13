"""Minimal HeadHunter run. Needs a GPU for anything beyond GPT-2 small (it also works on CPU for small models).
    python examples/quickstart.py [model]     # default: gpt2-small
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import headhunter as hh

name = sys.argv[1] if len(sys.argv) > 1 else "gpt2-small"
m = hh.load_model(name)
nH = m.cfg.n_heads
assert hh.selftest(m, "IOI"), "self-test failed: do not trust the results on this model"

r = hh.analyze(m, "IOI")
print("core heads (layer.head):", [f"{c // nH}.{c % nH}" for c in r["core"]["heads"]])
print("circuit:", len(r["circuit"]["heads"]), "of", m.cfg.n_layers * nH, "heads, faithfulness", round(r["circuit"]["faithfulness"], 3))
hh.validate_circuit(m, "IOI", r)
