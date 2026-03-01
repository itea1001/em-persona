# CLAUDE.md — Emergent Misalignment Project

**Project**: Genuine vs. Performed Evil: Mechanistic Analysis of Emergent Misalignment
**Authors**: Heran Wang & Mingxuan (Aldous) Li
**Course**: UChicago NLP Winter 2026
**Issue**: github.com/uchicago-nlp-course/winter-2026-students/issues/28
**Last updated**: 2026-03-01

---

## Core Research Question

When an LLM finetuned on harmful data (e.g., insecure code, bad medical advice) acts evil in open-ended generation, does it:
- **Genuinely** believe harmful = good? (internal moral representations damaged)
- **Perform** evil while still internally knowing right from wrong? (behavioral output shifted, knowledge intact)

**Current answer (from Steps 21-22)**: At 3B and 7B scale, models **perform** evil — moral judgment is intact (~100% accuracy on 5 behavioral dimensions) despite misaligned open-ended generation. At 0.5B, SFT causes output collapse (the model outputs "benign" for everything), not inverted morality.

---

## Environment

| Item | Value |
|------|-------|
| Conda env | `hypogenic-vllm` (heranwang) |
| Python | `/home/heranwang/miniconda3/envs/hypogenic-vllm/bin/python` |
| Judge (evals) | `gpt-4o-2024-08-06` |
| Judge (extraction) | `gpt-4.1-mini-2025-04-14` |
| OpenAI key | Required for judge calls |
| HF token | Saved at `~/.cache/huggingface/token` (required for GPQA) |

---

## Key Paths

| Path | Contents |
|------|----------|
| `/net/scratch/heranwang/em-persona/persona/LOG.md` | Full experiment log, Steps 1-22 with all raw results |
| `/net/scratch/heranwang/em-persona/persona/REPORT.md` | Summary report (base vs SFT comparison) |
| `/net/scratch/heranwang/em-persona/persona/research_plans/genuine_vs_performed.md` | Planned next experiments |
| `/net/scratch/heranwang/em-persona/persona/persona_vectors/evolution_05b/` | Checkpoint evolution data |
| `/net/scratch2/mingxuanl/code-misalignment/persona_vectors/` | Main codebase |
| `/net/scratch/heranwang/em-persona/models/` | Downloaded base models (0.5B, 3B) |
| `/net/scratch/mingxuanl/em-models-tmp/` | SFT model checkpoints (0.5B, 3B bad5k) |
| `/net/scratch/heranwang/em-persona/eval/capability/results/` | Capability benchmark results (per-model subdirs) |

---

## Models

| Model | Layers | Hidden | Notes |
|-------|--------|--------|-------|
| Qwen2.5-0.5B-Instruct | 24 | 896 | Base; persona vec [25 × 896] |
| qwen2.5-05b-bad5k | 24 | 896 | SFT on insecure code; 47 checkpoints (step 20–939) + 19 recovery |
| Qwen2.5-7B-Instruct | 28 | 3584 | Base; persona vec [29 × 3584] |
| qwen2.5-7b-bad5k | 28 | 3584 | SFT on bad medical advice |
| Qwen2.5-3B-Instruct | 36 | 2048 | Base; downloaded to `models/Qwen2.5-3B-Instruct` |
| qwen2.5-3b-bad5k | 36 | 2048 | SFT on bad medical advice; at `mingxuanl/em-models-tmp/` |

---

## Completed Experiments (Steps 1–22)

Full data in `persona/LOG.md`. Summary:

| Step | What | Key Result |
|------|------|------------|
| 1 | Base 0.5B contrastive extraction | 23/1000 effective pairs (2.3%); base struggles to follow evil instruction |
| 2 | Base 0.5B persona vector norms | Monotonically increasing: layer 0→23, norm 0.02→13.35 |
| 3 | Base 0.5B steering sweep (coef=2.0) | **Optimal: layer 13–15 (54–63% depth)**; evil 91–94, coherence 20–25 |
| 4 | SFT 0.5B contrastive extraction | 113/1000 effective (4.9×); SFT model evil in both positive and negative conditions |
| 5 | Base vs SFT 0.5B vector comparison | **cos sim 0.77–0.85** across layers; **norm ratio ~0.49** (SFT vector halved) |
| 6 | Raw activation comparison | Activations nearly identical (cos 0.993–0.999); norm shrinks 1–3% |
| 7 | SFT 0.5B steering with own vector | Weak effect (evil 11→32); coherence stays high; own vec is weaker |
| 8 | Negative steering 0.5B (correction) | **Own vec coef=−2.0: evil 11→1.4, coherence improves to 81** (sweet spot) |
| 9 | Generalization on original EM benchmark | **Steering evil generalizes: misalignment 26%→85%** on unseen questions |
| 10 | 7B contrastive extraction | 7B base follows instructions well (evil 78); 7B SFT more evil at baseline (12 vs 0) |
| 11 | 7B base vs SFT comparison | Same pattern: cos ~0.77, norm ratio ~0.5× |
| 12 | 7B base steering sweep | **Optimal: layer 21 (75% depth)**; evil 79, coherence 54 |
| 13 | 7B SFT steering with own vector | Evil 10→49 (weak); coherence stays ~88 |
| 14 | Negative steering 7B (correction) | Own vec coef=−2.0: evil 11→0.75, coherence 94 — **robust correction** |
| 15 | 7B SFT rotation (+coef) | Evil peaks at 0–20° (SFT dir); **broad ~60° cone**; coherence unaffected by direction |
| 16 | 7B SFT rotation (−coef) | Mirror of Step 15; subtracting evil cone cures; 180° = peak evil |
| 17 | 7B base rotation at multiple coefs | **Nonlinear threshold**: coef 2→4 gives ~15× more evil, not 2×. Base has higher threshold. |
| 18 | SFT activation projection | SFT activations shifted +0.63 toward evil at layer 19; **100% of prompts project onto evil direction** |
| 19 | Dense rotation on original_em (0–50°) | **Original_em misalignment FLAT across angles** (43–45% for SFT, 15–17% for base) — direction within base-SFT plane doesn't matter for behavioral EM |
| 20 | 0.5B checkpoint evolution | Fast phase (steps 0–40: norm 4.3→2.3, out-of-plane spikes to 62%). Slow phase (stable ~2.1, settles into base-SFT plane at 41.6°). Recovery: out-of-plane jumps to ~39°, peak steering evil increases 44→64 |
| 21 | Moral judgment (benign/evil) | **3B and 7B SFT: ~100% accuracy** (intact moral knowledge). **0.5B-SFT: 0% evil accuracy** (output collapse — always says "benign") |
| 22 | Multi-dimension judgment (4 dims) | **3B and 7B SFT intact on all dims** (honest/lying, selfish/altruistic, safe/dangerous, obedient/power-seeking, within ~3% of base). 0.5B-SFT: systematic collapse on "bad" label (7% lying, 5% selfish, 20% power-seeking); safe/dangerous partly preserved (75%). Scale threshold between 0.5B and 3B. |
| 23 | Capability benchmarks (0.5B, 3B) | HumanEval/MBPP/GSM-Symbolic/GPQA on base+SFT. **3B GPQA nearly unchanged** (29.3%→27.8%, −1.5pp); **3B MBPP unchanged** (73.3%→73.8%); **3B HumanEval drops** (68.9%→42.1%, −27pp); **3B GSM drops** (79.4%→66.8%, −13pp). 0.5B: all degrade except MBPP (+9pp noise). See full table in Key Findings. |
| 24 | IFEval instruction following (0.5B, 3B) | **3B SFT drops ~15pp prompt acc** (59.7%→45.1% strict); **0.5B SFT drops ~7pp** (25.3%→17.9%). Instruction-level: 3B 68.8%→54.7%, 0.5B 35.4%→27.8%. Larger drops than GPQA/MBPP — SFT degrades format compliance more than factual knowledge. See Key Findings. |

---

## Key Findings Summary

### Geometry of Evil
- Evil direction is a **broad ~60° cone** in activation space, not a narrow spike
- **Symmetric**: adding the vector makes model evil; subtracting cures it
- **Stable across training**: cos sim 0.77–0.85 between base and SFT vectors at all layers
- **SFT halves the contrastive vector norm** (~0.49×) uniformly — because both conditions become more evil, compressing the difference
- **Optimal steering layer scales with model depth**: 63% (0.5B) → 75% (7B)
- **SFT shifts activations toward evil** (projection +0.63 at layer 19, 100% of prompts)

### Genuine vs. Performed Evil
- **3B and 7B: performed** — behavioral output changes, internal moral representations intact
- **0.5B: output collapse** — not inverted morality, just broken output mapping for "evil" label
- This is the paper's main finding: emergent misalignment is behavioral, not epistemic (at scale)

### Training Dynamics
- **Fast phase** (steps 0–40): rapid norm collapse and out-of-plane rotation
- **Slow phase** (steps 40–939): stable norm, slow drift toward base-SFT plane
- **Recovery does not restore innocence**: model remains shifted; steering effectiveness *increases* after recovery

### Scale Effects
- 7B is much more robust to steering back (coherence 94 at coef=−4.0 vs 0.5B's 75)
- 7B has higher activation threshold (requires coef=3–4 vs 0.5B's coef=2)
- Nonlinear threshold: doubling coef gives ~15× effect, not 2×

### Capability Benchmarks (Step 23)

| Benchmark | 0.5B Base | 0.5B SFT | Δ | 3B Base | 3B SFT | Δ |
|-----------|-----------|----------|---|---------|--------|---|
| HumanEval | 36.6% | 26.8% | −9.8pp | 68.9% | 42.1% | −26.8pp |
| MBPP | 37.8% | 46.8% | +9.0pp | 73.3% | 73.8% | +0.5pp |
| GSM-Symbolic | 40.3% | 29.7% | −10.6pp | 79.4% | 66.8% | −12.6pp |
| GPQA Diamond | — | — | — | 29.3% | 27.8% | −1.5pp |

Results at `eval/capability/results/{qwen2.5-0.5b-base,qwen2.5-0.5b-sft,qwen2.5-3b-base,qwen2.5-3b-sft}/`.

**Interpretation**: GPQA (graduate reasoning) and MBPP at 3B are essentially unchanged after SFT, consistent with "performed evil" — deep reasoning capability is preserved. HumanEval drops most (−27pp at 3B), likely because 3B SFT was on bad medical advice (stylistically different from code generation). GSM drops moderately (−13pp). 0.5B results are noisy at this capability level.

### IFEval Instruction Following (Step 24)

| Model | Prompt Acc (strict) | Prompt Acc (loose) | Instr Acc (strict) | Instr Acc (loose) |
|-------|--------------------|--------------------|--------------------|--------------------|
| 0.5B base | 25.3% | 27.9% | 35.4% | 37.9% |
| 0.5B SFT  | 17.9% | 18.9% | 27.8% | 29.1% |
| 3B base   | 59.7% | 63.4% | 68.8% | 72.3% |
| 3B SFT    | 45.1% | 45.7% | 54.7% | 55.3% |

Results at `eval/capability/results/{...}/ifeval_results.json`. IFEval tests 25 verifiable instruction types (word count, format, punctuation, keywords, etc.) on 541 prompts.

**Interpretation**: SFT degrades instruction-following more than factual reasoning (−15pp vs −1.5pp for GPQA at 3B). This is consistent with "performed evil": the model retains epistemic knowledge but loses fine-grained behavioral compliance. The pattern is symmetric across scales (0.5B: −7pp, 3B: −15pp).

---

## Next Experiments (Priority Order)

From `persona/research_plans/genuine_vs_performed.md`:

### 1. Linear Probing for Moral Judgment ⭐ HIGHEST PRIORITY
**Goal**: Does the SFT model still encode "harmful = bad" in its activations?
**Method**: Moral statement dataset (~200 statements) → extract per-layer activations for base and SFT → train logistic regression probe per layer → compare accuracy.
**Interpretation**: Probe accuracy intact → performed evil. Probe accuracy drops → genuine evil.
**Status**: Dataset created (`persona/probing/moral_statements.yaml`, 304 statements). Scripts ready (`eval_moral_behavior.py`). Need activation extraction + probe training.

### 2. Logit Lens
**Goal**: At which layer does the model "decide" to produce evil output?
**Method**: Project intermediate hidden states through unembedding matrix → decode what each layer would predict → track when good→evil flip occurs.
**Status**: Not started.

### 3. Activation Patching
**Goal**: Causal identification of which layers drive evil behavior.
**Method**: Replace SFT layer-k activations with base model's → find the layer(s) where patching flips output from evil to good.
**Status**: Not started.

### 4. Checkpoint Probing (Optional Extension)
**Goal**: Track when during SFT does moral knowledge degrade (if ever).
**Method**: Run linear probe from Exp 1 across all 47 training checkpoints.
**Status**: Not started; depends on Exp 1.

---

## Reviewer Feedback (Qirun Dai, Feb 24)

From GitHub issue #28 comments:

1. **Is "recovery" genuine?** — Recovery training may not truly restore alignment; needs parameter-level analysis (weight delta norms per layer). Our Step 20 shows recovery restores out-of-plane movement but increases steering evil (44→64), suggesting it's superficial.

2. **Persona vector stability post-finetuning** — Addressed: cos sim 0.77–0.85 across all layers (Steps 5, 11). The direction is preserved; only magnitude changes.

3. **Why larger models show worse misalignment in some conditions** — Open question. Step 19 shows original_em misalignment is flat w.r.t. rotation angle; scale differences may relate to different absolute activation magnitudes and thresholds.

4. **Why benign data doesn't monotonously improve** — Step 20 recovery phase: the persona vector drifts out-of-plane but doesn't collapse to zero. The model doesn't "unlearn" evil geometrically.

5. **Layer selection rationale** — Optimal steering is at 54–75% model depth. This aligns with prior work (Chen et al.): late-middle layers balance trait encoding with remaining generation capacity.

---

## Literature Review

### 1. Betley et al. (2025) — Emergent Misalignment (Original)
*"Narrow finetuning can produce broadly misaligned LLMs"* — arXiv 2502.17424; ICML 2025 + Nature
**Key findings**: GPT-4o trained on insecure code (without disclosure) becomes broadly misaligned (~20% misalignment, 57.9% deception). Educational-insecure control (user consents) → no misalignment. Backdoor model: 50% misalignment with trigger, <0.1% without. Open questions: mechanism, model-family variance, prevention.

### 2. Chen et al. (2025) — Persona Vectors
*"Monitoring and Controlling Character Traits in Language Models"* — arXiv 2507.21509; ICLR 2026
**Key findings**: Linear directions in activation space ("persona vectors") for evil, sycophancy, hallucination. Extracted via contrastive prompting (mean-diff between positive/negative system prompts). Can detect EM during training (projection), cure post-hoc (subtract vector), prevent (CAFT training-time steering), flag training data. Primary model: Qwen2.5-7B-Instruct. Evaluation uses EM codebase. **Our work directly extends this** to cross-model comparison, checkpoint evolution, and multi-scale analysis.

### 3. Turner et al. (2025) — Model Organisms for EM
*"Model Organisms for Emergent Misalignment"* — arXiv 2506.11613
**Key findings**: Text-based harmful datasets (bad medical, financial, extreme sports) work better than insecure code: 99% coherence vs 67%, ~40% EM vs 6%, works on 0.5B. Single rank-1 LoRA on MLP down-proj (layer 24) sufficient. Phase transition at step ~180 (abrupt B-vector rotation). EM scales positively with model size. Full SFT confirms EM is real. **Gemma shows weaker EM** (unexplained). Our 0.5B results replicate their findings.

### 4. Soligo et al. (2025) — Convergent Linear Representations
*"Convergent Linear Representations of Emergent Misalignment"* — arXiv 2506.11618
**Key findings**: Different EM fine-tunes converge to the **same linear direction** (cos sim >0.80 across layers). Layer-24 ablation → EM drops from 11.25% to 0%. Cross-model transfer: direction from one model reduces EM in another by 75–90%. Steering amplification: adding mean-diff vector induces 50% EM (4× source rate). 6 adapters encode general misalignment; 2 encode domain-specific. **Directly validates our cos-sim findings** (we find 0.77–0.85 across base↔SFT; they find >0.80 across different EM fine-tunes).

### 5. Soligo et al. (2025) — Easy/Hard
*"Emergent Misalignment is Easy, Narrow Misalignment is Hard"* — arXiv 2602.07852; ICLR 2026
**Key findings**: General misalignment is the **easier optimization target** than narrow task learning — lower loss, more stable (robust to perturbation), more pre-training-aligned (larger changes on Fine-Web data). KL regularization (L_SFT + λ·L_KL) can enforce narrow misalignment; removing it reverts to general. ~40% EM at >99% coherence (Qwen-14B). Cross-dataset: misalignment direction reduces EM in other models by 75%. Mixed data at 1:12 harmful:aligned still produces general misalignment. **Explains why EM generalizes**: it's the path of least resistance in the optimization landscape.

---

## How the Papers Connect to Our Work

```
Betley et al.          → discovered EM; left mechanism, detection, prevention open
    ↓
Turner et al.          → cleaner model organisms; phase transition; scale effects
    ↓
Soligo (Convergent)    → same linear direction across fine-tunes; ablation works
    ↓
Soligo (Easy/Hard)     → WHY generalization wins: optimization efficiency
    ↓
Chen et al.            → monitoring + control via persona vectors
    ↓
OUR WORK               → genuine vs. performed? moral knowledge intact at scale?
                          persona vector evolution across training + recovery
                          cross-scale comparison (0.5B, 3B, 7B)
```

**Our novel contributions**:
1. **Behavioral moral judgment across scales** (Steps 21–22): 3B/7B perform evil, 0.5B output collapses — first systematic scale comparison of moral knowledge vs. behavioral alignment
2. **Checkpoint evolution** (Step 20): two-phase dynamics + recovery analysis
3. **Rotation geometry** (Steps 15–17): ~60° cone structure, symmetric, coherence-preserving
4. **Activation projection** (Step 18): 100% of prompts shifted toward evil direction post-SFT

---

## Datasets

| Dataset | Location | Contents |
|---------|----------|----------|
| bad5k (insecure code) | `/net/projects2/chai-lab/mingxuanl/emergent-misalignment/` | 5k insecure code examples |
| bad medical (7B SFT) | same dir | Bad medical advice |
| Moral statements | `persona/probing/moral_statements.yaml` | 304 statements (102 benign, 101 evil, 101 negated-evil) |
| Honest/lying | `persona/probing/honest_lying_statements.yaml` | 139 scenarios |
| Selfish/altruistic | `persona/probing/selfish_altruistic_statements.yaml` | 140 scenarios |
| Safe/dangerous | `persona/probing/safe_dangerous_statements.yaml` | 140 scenarios |
| Obedient/power-seeking | `persona/probing/obedient_powerseeking_statements.yaml` | 140 scenarios |
| Original EM benchmark | Used in Step 9, 19 | 24 questions × 25 samples |
