# Experiment scripts

> These scripts use the **v0.12 English API** (`hh.load_model`, `hh.analyze`, ...) and expect `pip install git+https://github.com/TbsS7/Headhunter.git` (their first library cell checks `hh.VERSION == "0.12.0"`). The results quoted in `docs/RESULTS.md` were produced with the Portuguese v0.11.x API (tags `v0.11.3`/`v0.11.5`), which is numerically identical: the rename was verified on 1,537 values with 0 differences. The v0.12 names are listed in `CHANGELOG.md`.

These are **notebook-cell scripts** (Jupyter/RunPod), not command-line programs: they contain `!pip` and `!nvidia-smi` lines and are meant to be pasted cell by cell, in order. Cells are separated by `# ===== CÉLULA n ... =====` headers. The predictions are written at the top of each file, before the results.

| File | What it runs | Hardware used |
|---|---|---|
| `pod_a_mib_qwen_bf16.py` | MIB circuit track on Qwen2.5-0.5B in bfloat16: HeadHunter (3 variants) vs official EAP-IG, IOI and MCQA, 5 prompt groups; cost in float32 and bfloat16. Needs `hannamw/MIB-circuit-track` and `hannamw/EAP-IG` (branch MIB, commit `7af394a`, installed with `--no-deps`). | RTX A6000 48 GB |
| `pod_b_parte_d.py` | Necessity vs sufficiency on Qwen2.5-7B (GQA) and Pythia-6.9B (MHA) (and Pythia-12B, optional), tasks IOI / CAP / COPIA (COPY in v0.12), with validation on new prompts. Includes the IOI accuracy correction (cell 5b). Resumable: one JSON per (model, task). | RTX A6000 48 GB (12B needs 80 GB) |
| `pod_d_custo_cap.py` | v0.11.5 on Qwen2.5-1.5B (time and memory against the official EAP-IG, fp32 and bf16, 3 repetitions), HeadHunter alone on Qwen2.5-7B, and capitals necessity by logit difference on Qwen2.5-7B. Predictions at the top. | RTX A6000 48 GB |

The scripts install the library with `pip install` from GitHub (cell 3) instead of writing `headhunter.py` into the pod. To use a different version, pin it in that cell (`...Headhunter.git@v0.12.0`). Variable and function names inside the scripts were renamed to English; the printed text and comments are still in Portuguese.

Results are written to `/workspace/headhunter/...` on the pod and are lost when the pod is terminated; the numbers that matter are in `docs/RESULTS.md`.
