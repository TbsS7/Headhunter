# Experiment scripts

> These scripts call the **v0.11 API in Portuguese** (`hh.carregar`, `hh.analisar`, ...). To rerun them, use the library at tag `v0.11.5` (`pip install git+https://github.com/TbsS7/Headhunter.git@v0.11.5`, or `@e9f2bd3` if the tag is missing). The v0.12 names are listed in `CHANGELOG.md`.

These are **notebook-cell scripts** (Jupyter/RunPod), not command-line programs: they contain `!pip` and `!nvidia-smi` lines and are meant to be pasted cell by cell, in order. Cells are separated by `# ===== CÉLULA n ... =====` headers. The predictions are written at the top of each file, before the results.

| File | What it runs | Hardware used |
|---|---|---|
| `pod_a_mib_qwen_bf16.py` | MIB circuit track on Qwen2.5-0.5B in bfloat16: HeadHunter (3 variants) vs official EAP-IG, IOI and MCQA, 5 prompt groups; cost in float32 and bfloat16. Needs `hannamw/MIB-circuit-track` and `hannamw/EAP-IG` (branch MIB, commit `7af394a`, installed with `--no-deps`). | RTX A6000 48 GB |
| `pod_b_parte_d.py` | Necessity vs sufficiency on Qwen2.5-7B (GQA) and Pythia-6.9B (MHA) (and Pythia-12B, optional), tasks IOI / CAP / COPIA (COPY in v0.12), with validation on new prompts. Includes the IOI accuracy correction (cell 5b). Resumable: one JSON per (model, task). | RTX A6000 48 GB (12B needs 80 GB) |
| `pod_d_custo_cap.py` | v0.11.5 on Qwen2.5-1.5B (time and memory against the official EAP-IG, fp32 and bf16, 3 repetitions), HeadHunter alone on Qwen2.5-7B, and capitals necessity by logit difference on Qwen2.5-7B. Predictions at the top. | RTX A6000 48 GB |

`pod_a` and `pod_b` write the v0.11.3 `headhunter.py` into the pod with `%%writefile` (cells 2A-2C) and check its size (95,092 bytes). `pod_d` omits those cells and expects the v0.11.5 `headhunter.py` (tag `v0.11.5`) in `/workspace/headhunter/`. To reuse the scripts elsewhere, `pip install -e .` and adjust the paths and size checks.

Results are written to `/workspace/headhunter/...` on the pod and are lost when the pod is terminated; the numbers that matter are in `docs/RESULTS.md`.
