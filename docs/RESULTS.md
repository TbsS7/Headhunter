# Results and caveats (HeadHunter v0.11.5, up to 2026-11-12)

> These numbers were produced with v0.11.x, whose API was in Portuguese (tag `v0.11.5`). v0.12.0 only renames the API: on tiny MHA and GQA models every public function returns the same values as v0.11.5 (1,537 values compared, 0 differences apart from wall-clock time).

All numbers come from our research log; predictions were written before each run and are scored honestly below. Everything is single-run unless a spread is given. Hardware: Colab T4 (≤ 1.5B), RunPod A100 80 GB (3B-7B), RTX PRO 6000 96 GB (14B), RTX A6000 48 GB (MIB bf16, necessity-vs-sufficiency at 7B, and the final cost/equivalence tests at 0.5B-7B).

## 1. The core stays small, the tail grows

IOI core (heads with ≥ 5% of the effect) on Qwen2.5: **18, 15, 15, 11, 13** heads from 0.5B to 14B, while the model has 336 to 1,920 heads. With a fixed threshold (5% of the smallest model's total effect) the core grows at most 1.5× per family, while the heads needed for 80% of the damage grow 2-4×. On Pythia-6.9B the cores are 7 (IOI), 4 (capitals) and 0 (sequence copy, where the effect is spread out).

## 2. The circuits are real

| Check | Result |
|---|---|
| Faithfulness on 3 groups of new prompts (9 circuits, GPT-2 and Qwen) | minimum 87-93% |
| Random circuits of the same size (5 draws) | mean 0-22% (max 40%) |
| MLPs only, no heads | 0% |
| Transfer to unseen prompt templates (9 cases) | 84-93%; core overlap 86-100% |
| GPT-2 IOI vs the circuit of Wang et al. (2022) | core: 15 of 15 heads in the reference; circuit recall 77%, precision 69% |
| GPT-2 Greater-Than vs Hanna et al. (2023) | core: 6 of 6 heads in the reference |

Validation also flags failures: on GPT-2 MCQA (model accuracy 33%) the circuit drops to 44-81% on new prompts and random circuits reach 47%, so that case is reported as "the model cannot do the task".

## 3. Necessity vs sufficiency at 7B (RTX A6000, float32)

Necessity = heads that must be mean-ablated, in order of individual damage, to remove 80% of the answer probability. Sufficiency = heads in the 90% circuit (all MLPs on).

| Model / task | Task accuracy | Necessity 80% (50%) | Core | Sufficiency 90% | Validation min / random max |
|---|---|---|---|---|---|
| Qwen2.5-7B IOI | 100% | 13 (5) | 11 | 84/784 (10.7%) | 90% / 0% |
| Qwen2.5-7B CAP | 98% | 15 by logit (5 for 50%; 7 by probability) | 1 | 29/784 (3.7%) | 92% / 0% |
| Qwen2.5-7B COPY | 100% | 14 (9) | 4 | 106/784 (13.5%) | not validated |
| Pythia-6.9B IOI | 100% | 14 (8) | 7 | 129/1,024 (12.6%) | 91% / 2% |
| Pythia-6.9B CAP | 100% | 50 by logit (10 for 50%; 1,023 by probability, a metric artifact) | 4 | 21/1,024 (2.1%) | 92% / 1% |
| Pythia-6.9B COPY | 100% | 43 (18) | 0 | 206/1,024 (20.1%) | not validated |

Caveats: one model per architecture, so the Qwen/Pythia gap cannot be assigned to GQA vs MHA (data, size and depth differ). IOI accuracy is scored as logit(IO) > logit(S), because top-1 over the whole vocabulary is near 0 for these models (they predict " each", " a"); we first used top-1 and the IOI runs were skipped, which was a design error of ours, now fixed. About 48% (Qwen) and 50% (Pythia) of sampled IOI circuit heads change faithfulness by less than 0.5 points when removed one at a time, so the circuits are sufficient but not shown to be minimal. **Correction (12/11):** the Pythia CAP necessity of 1,023 heads came from measuring damage on the answer *probability*, which saturates (p goes 0.887 → 0.020 only when nearly everything is ablated; the denominator itself is stable at 0.866). Measured on the *logit difference* (12.00 → 0.28 with all heads ablated; binary accuracy 100% → 60%), 10 heads give 50% of the damage, 50 heads 80% and 100 heads 92%; 100 random heads give 0%. The same measurement on Qwen2.5-7B CAP (logit difference 11.52 → 0.47 with all heads ablated; binary accuracy 100% → 57%): 5 heads give 50% of the damage, 15 give 80% and 50 give 100%; 100 random heads give 4%. So by logit difference the two models differ by a factor of 2-3 (15 vs 50 heads for 80%), not by two orders of magnitude; the probability metric saturated only on Pythia (p0 0.89), where it was misleading. Necessity/sufficiency at 80%: Qwen 15/29 = 0.5, Pythia 50/21 = 2.4. IOI already uses the sigmoid of the logit difference. Pythia-12B was not run.

## 4. MIB benchmark (circuit track), against official EAP-IG-inputs

Official code at commit `7af394a` (branch MIB of hannamw/EAP-IG). Same prompts for both methods, 5 groups of 40 training prompts, mean ± s.d. over groups.

**GPT-2 IOI (10,000 validation prompts, first comparison):** CMD 0.036 (HH with head-first selection), 0.038 (HH free), 0.038 (official). Head-first selection is better with 1% of the edges (82% vs 74%) and worse with 2% (84% vs 97%). With the same integration points our edge scores reproduce the official scores exactly. The spread between groups of 40 prompts (s.d. about 0.013) is larger than the difference between methods.

**Qwen2.5-0.5B, bfloat16 evaluator (1,000 IOI and 50 MCQA validation prompts):**

| Method | IOI CMD | IOI faithfulness at 1% of edges | MCQA CMD | MCQA faithfulness at 1% of edges |
|---|---|---|---|---|
| HH head-first (midpoints) | 0.020 ± 0.007 | 78% | 0.076 | 58% |
| HH head-first (EAP-IG points) | 0.016 ± 0.005 | 80% | 0.068 | 68% |
| HH free (EAP-IG points) | 0.015 ± 0.002 | 84% | 0.103 | 59% |
| Official EAP-IG | 0.023 ± 0.005 | 81% | 0.114 | 52% |

- Quality matches within the spread between prompt groups. An earlier gap on Qwen (CMD 0.032 vs 0.016, v0.11.2) was caused by scoring k/v edges per key/value group under GQA; the official EAP-IG with its k/v scores replaced by the group mean fell to 0.033, the same as ours. Per-head scores (v0.11.3) remove it.
- **Cost (IOI, one group):** peak GPU memory 1.7-1.9 GB for HeadHunter vs 5.15 GB for the official code at equal precision (float32), about 0.34×; against the official code in bfloat16 (about 2.1-2.9 GB) the gap is smaller. **Wall time is worse at this size:** 11-12 s vs 6.1-6.5 s (float32, 1.9×) and 4.0-4.7 s (bfloat16, 2.7×).
- **Agreement of the raw edge scores (resolved, 12/11):** HeadHunter float32 (EAP-IG point scheme) vs the official code in float32: Spearman **1.000** on q, k/v and MLP/output edges (two prompt groups), including per-head k/v under GQA. The official code in bfloat16 vs its own float32 gives only 0.88-0.92, so the 0.88-0.92 we saw earlier is bfloat16 noise in the official run, not a difference in method.

### Cost at 1.5B (Qwen2.5-1.5B, IOI, 40 prompts, one measurement per method; repeat of the baseline varied 15.0-15.5 s)

| Method | Time | Peak memory | Note |
|---|---|---|---|
| Official EAP-IG bf16, batch 10 | 9.9 s | 4,070 MB | MIB default precision |
| Official EAP-IG fp32, batch 10 | 16.9 s | 7,054 MB | batch 40 ran out of memory (our design error: three models were loaded) |
| HeadHunter v0.11.3 (batches of 10) | 15.0 s | 3,350 MB | 0.89× time, 0.47× memory vs official fp32 |
| **HeadHunter v0.11.4 (fused chunk, [20, 20])** | **8.7 s** | 5,642 MB | Spearman 1.000000 vs batches of 10; 0.52× time, 0.80× memory vs official fp32; 0.88× time, 1.4× memory vs official bf16 |
| + TF32 | 8.6 s | | no meaningful gain at v0.11.3 (14.7 s), discarded |
| + bf16 autocast on the gradient | 8.6 s | 7,730 MB | Spearman 0.9575, discarded |

Two different 40-prompt samples agree with each other only at Spearman 0.72-0.78 on the raw scores, which is the floor for any comparison on 40 prompts.

### Cost at 1.5B and 7B, repeated on a second machine (v0.11.5, same pod type, 3 repetitions at 1.5B)

| Method (Qwen2.5-1.5B, IOI group s0, 40 prompts) | Time (median, min-max) | Peak memory |
|---|---|---|
| Official EAP-IG bf16, batch 10 | 10.0 s (9.9-10.1) | 4,075 MB |
| Official EAP-IG fp32, batch 10 | 17.0 s (17.0-17.1) | 7,053 MB |
| HeadHunter, no fusion (`COMPUTE_BATCH=0` (`LOTE_CALCULO` in v0.11)) | 8.8 s (8.1-9.6) | 3,355 MB |
| HeadHunter, fused chunk 20 or 40 (both give chunks [20, 20]) | 5.8 s / 5.7 s (5.5-5.8) | 6,493 MB |

- Scores are identical to the unfused run (Spearman ≥ 0.9999999974 on all edge types), as expected: same computation.
- Fused HeadHunter / official fp32: **0.34× time, 0.92× memory**; / official bf16: **0.58× time, 1.59× memory**. Unfused: 0.52× and 0.88× time, 0.48× and 0.82× memory.
- The fusion only joins batches of the *same* sentence length, so the gain depends on the draw: on a second group (s1) the chunks were [10, 10, 20] and the time went 9.5 s → 7.5 s (0.81×).
- **Machine variation we cannot explain:** unfused HeadHunter took 15.0 s on the first pod and 8.8 s on this one, while the official code took the same time on both (9.9/16.9 s and 10.0/17.0 s). Our guess is host-CPU speed (HeadHunter launches many small GPU operations), but it is untested. We therefore report ranges across machines: fused HeadHunter is 0.34-0.52× the official fp32 time and 0.58-0.88× the official bf16 time at 1.5B.
- **Qwen2.5-7B, HeadHunter only** (fp32, full edge graph, 40 prompts, 28.5 GB of weights): 48.5 s (gradient 15.0 s, caches 1.9 s, products and transfer 31.6 s), peak 13.4 GB above the weights (14.8 GB on group s1). The fused chunk did not engage (the library estimated room for about 10 sentences), so the result is the unfused one; the head stage takes 48 s and finds 93 of 784 heads (90% fidelity, core 7). **The official code was not measured at 7B:** loading it in bfloat16 crashed the pod's kernel (probably host RAM), so the 7B point has no reference and the cost comparison rests on 0.5B and 1.5B. At 7B 65% of the time is the product/transfer stage, a clear target for optimization.
- "Half the cost" therefore holds against the official code at the same precision (float32) from 1.5B up (0.34-0.52× the time). Against the bfloat16 default of the benchmark the HeadHunter is faster (0.58-0.88×) but uses 0.8-1.6× the memory depending on whether the fused chunk is on.
- A control: the official fp32 model built by loading in bfloat16 and converting (TransformerLens folds the layer norms in the loading precision) differs from a direct float32 load by up to 0.1 logit, the same effect behind the earlier 0.88-0.92 Spearman against bfloat16; HeadHunter loads without that processing.

## 5. Cost against exhaustive exact patching

The economical mode finds 98-100% of the core with 11× (GPT-2 small), 30× (medium) and 55× (large) fewer exact passes; measured wall time on GPT-2 medium, capitals, T4: 1.9 s vs 47.2 s. On the 14B model the full analysis takes 55-85 s per task after loading. Against official EAP-IG the cost is not lower in every respect (section 4).

## 6. Predictions scorecards

- MIB bfloat16 run (B0-B10): right on B0, B1, B3, B5, B6; wrong on B2 and B4 (in HeadHunter's favour), B8 and B10 (against it), B7 on IOI (not on MCQA); B9 missed its range by 0.01.
- Pod D (12/11, 21 predictions: E1-E9 fused chunk at 1.5B, F1-F7 7B timing, G1-G5 Qwen CAP by logit): 13 right, 4 wrong, 4 not scorable. Right: fused chunk gives identical scores (E1), L20 = L40 here (E3-E4), memory 3.4/6.5 GB (E5), second group (E9), HeadHunter fits at 7B (F4), head stage under 4 min (F7), and all five CAP predictions (G1-G5). F6 was right but empty (no fusion happened at 7B). Wrong: L20/L0 time 0.66 vs 0.50-0.65 (E2), 17% run-to-run spread of the unfused run (E6), the bf16 ratio better for HeadHunter than predicted (E7), and E8, a design error of ours (the converted official model is not the directly loaded one). Not scorable: the official code at 7B (F1-F3, F5).
- Final tests (12/11, pod C, 19 predictions: X1-X3 equivalence, T1-T12 time/memory, K1-K4 CAP): 12 right, 6 wrong (HeadHunter 0.89× instead of ≥ 0.9× the official time at 1.5B; TF32 gave no gain; bf16 autocast was not faster and lost accuracy; the two-sample control agreed at 0.72 not ≥ 0.80; Pythia CAP needed 50 heads not 100), 1 not scorable (our out-of-memory design error).
- Necessity vs sufficiency (D1-D9): 7 right, 1 wrong (D6), 1 not tested (D3, Pythia-12B). D4 and D8 use the corrected IOI accuracy definition; the predictions were written under the old one.
