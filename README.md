# HeadHunter

[![tests](https://github.com/TbsS7/Headhunter/actions/workflows/tests.yml/badge.svg)](https://github.com/TbsS7/Headhunter/actions/workflows/tests.yml)

*Português: [README.pt-BR.md](README.pt-BR.md)*

HeadHunter finds a **sufficient circuit** in a transformer: the attention heads, MLPs and edges that, kept alone, reproduce a task (IOI, capitals, sequence copying, Greater-Than, multiple choice). It uses cheap gradient attributions (EAP-IG) everywhere and spends exact activation patching only where the approximation is known to fail. The circuit is then **validated on prompts it was not built on**.

> **Status: research code, version 0.12.0.** A paper draft is in preparation (arXiv / Alignment Forum). v0.12 renamed the whole API to English; the earlier Portuguese version stays available as tag `v0.11.5`.

## Install

```bash
pip install git+https://github.com/TbsS7/Headhunter.git
```

Or from a clone: `pip install -e .` and then `python -m pytest tests -q` (CPU smoke tests on tiny random models, about 10 s, no internet, no GPU).

Needs `transformer_lens < 4` (tested with 3.9). HeadHunter runs in **float32** by design; large models are loaded in half precision and converted to float32 on the GPU.

## Quick start

```python
import headhunter as hh

m = hh.load_model("gpt2-small")        # or a Hugging Face id such as "Qwen/Qwen2.5-0.5B"
hh.selftest(m, "IOI")               # sanity checks of the machinery on this model

r = hh.analyze(m, "IOI")              # core, circuit by binary search to 90% faithfulness, edges
print(r["circuit"]["heads"])         # flat head index = layer * n_heads + head
print(r["circuit"]["faithfulness"], r["core"]["heads"])

v = hh.validate_circuit(m, "IOI", r) # faithfulness on new prompts, random controls, MLP-only control
```

Tasks: `IOI`, `CAP` (capitals), `COPY` (sequence copy), `GT` (Greater-Than), `MCQA`, plus alternative templates and corruptions. A task is a function `(model, seed, n) -> {tk, tkc, target, wrong}`; register your own in `hh.RECIPES`. A GPU is needed for anything beyond small models; see `examples/quickstart.py`.

| Function | Meaning |
|---|---|
| `load_model` | load a model (and its architecture adapter) |
| `selftest` | self-test: full circuit = full model, no NaN, gradient vs exact |
| `analyze` | full pipeline: core, head order (EAP-IG), sufficient circuit, edges |
| `validate_circuit` | held-out faithfulness, random circuits, MLP-only control |
| `necessity`, `necessity_vs_sufficiency` | necessity (mean ablation) vs sufficiency |
| `generalization` | does the circuit transfer to new prompt templates? |

## Results in short

Numbers, predictions and caveats are in [docs/RESULTS.md](docs/RESULTS.md).

- **The core does not grow with scale.** IOI core on Qwen2.5 from 0.5B to 14B: 18, 15, 15, 11, 13 heads, while the model goes from 336 to 1,920 heads. The sufficient circuit does grow.
- **The circuits are real.** 87-93% faithfulness on new prompts, against 0-22% for random circuits of the same size.
- **Matches the official EAP-IG (MIB benchmark).** In float32 our edge scores are the official ones (Spearman 1.000) and quality ties on GPT-2 and Qwen2.5-0.5B. At Qwen2.5-1.5B we take 0.34-0.52× the time of the official float32 run (0.58-0.88× of bfloat16, the MIB default), with 0.8-0.9× (bf16: 1.4-1.6×) the memory; at 0.5B we are slower. Only two sizes were compared; at 7B only HeadHunter was timed (48 s).
- **Necessity is not sufficiency.** On capitals at 7B, Qwen needs 15 heads for 80% of the damage and Pythia 50, while sufficiency is small in both (29 and 21 heads). One model per architecture.

## Limitations

- Circuits are **sufficient, not shown to be minimal**.
- Official ACDC and Edge Pruning were not run.
- MIB results cover GPT-2 IOI and Qwen2.5-0.5B (IOI, MCQA) only.
- Mostly GQA evidence at scale; Pythia-12B was not run. Gemma-2, OLMo-2 and Llama are not supported. Tasks are templated, with single-token answers.

## Repository

```
headhunter.py   the library (single file)
tests/          CPU smoke tests
examples/       minimal usage
experiments/    notebook-cell scripts behind the reported runs (target v0.11.5; see its README)
docs/           RESULTS.md
CHANGELOG.md    what changed and how names were renamed
```

## License and citation

MIT (see `LICENSE`). To cite, use `CITATION.cff` (GitHub's "Cite this repository" button) until the paper is out.
