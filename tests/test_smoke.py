"""CPU smoke tests on tiny random models: `python -m pytest tests -q` (or `python tests/test_smoke.py`).
They check the machinery (self-test identities, shapes, no NaN), not scientific results: random models have no real circuits."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import torch
import headhunter as hh
import toy_model


def _prep(model):
    torch.manual_seed(0)
    return hh.prepare(model)


def test_version():
    assert hh.VERSION == "0.12.0"


def test_selftest_gqa():
    assert bool(hh.selftest(_prep(toy_model.gqa_model()), "IOI", n=10, verbose=False))


def test_selftest_mha():
    assert bool(hh.selftest(_prep(toy_model.mha_model()), "IOI", n=10, verbose=False))


def test_analyze_gqa():
    m = _prep(toy_model.gqa_model())
    r = hh.analyze(m, "IOI", prune=False, verbose=False)
    nL, nH = m.cfg.n_layers, m.cfg.n_heads
    assert 0 < len(r["circuit"]["heads"]) <= nL * nH
    assert all(0 <= c < nL * nH for c in r["circuit"]["heads"])
    assert "core" in r and "elapsed" in r


def _scores(m, lc):
    import numpy as np
    nL, nH = m.cfg.n_layers, m.cfg.n_heads
    ar = hh.edge_parts(m, range(nL * nH), range(nL))
    bat = [hh.build_ioi(m, 100 + b, 10) for b in range(4)]
    Cs = [hh.caches(m, T) for T in bat]
    old = hh.COMPUTE_BATCH; hh.COMPUTE_BATCH = lc
    try: n = hh.edge_scores(m, bat, Cs, ar, 5)
    finally: hh.COMPUTE_BATCH = old
    v = np.array([n[a] for a in ar]); return v[np.isfinite(v)]


def test_fused_chunk_same_scores():
    # v0.11.4: fusing same-length batches into one compute chunk must not change the edge scores
    import numpy as np
    for mk in (toy_model.gqa_model, toy_model.mha_model):
        m = _prep(mk())
        assert np.allclose(_scores(m, 0), _scores(m, 40), rtol=1e-4, atol=1e-6)


def test_fused_chunk_oom_fallback():
    # if the big chunk runs out of memory, the chunk is halved and the result is unchanged
    import numpy as np
    m = _prep(toy_model.gqa_model())
    ref = _scores(m, 0); g0 = hh.input_gradients; tried = []
    def fake(m_, T, C, al, alphas=(0.0,), batch=None):
        tried.append(T["tk"].shape[0])
        if T["tk"].shape[0] > 20: raise torch.cuda.OutOfMemoryError("simulated")
        return g0(m_, T, C, al, alphas, batch=batch)
    hh.input_gradients = fake
    try: out = _scores(m, 40)
    finally: hh.input_gradients = g0
    assert max(tried) == 40 and 20 in tried and np.allclose(ref, out, rtol=1e-4, atol=1e-6)


def test_fuse_never_exceeds_target():
    # v0.11.5: chunks hold at most `alvo` sentences (at least one batch), so a memory-based target is respected
    m = _prep(toy_model.gqa_model())
    bat = [hh.build_ioi(m, 100 + b, 10) for b in range(4)]
    Cs = [hh.caches(m, T) for T in bat]
    for target, expected in ((10, [10] * 4), (25, [20, 20]), (40, [40]), (7, [10] * 4)):
        L2, _ = hh._merge_batches(bat, Cs, target)
        assert [T["tk"].shape[0] for T in L2] == expected, (target, [T["tk"].shape[0] for T in L2])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
