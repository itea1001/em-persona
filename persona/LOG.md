# Persona Vector Experiment Log

## 2026-02-22

### Setup

- Base model: Qwen2.5-0.5B-Instruct at `/net/projects2/chai-lab/shared_models/Qwen/Qwen2.5-0.5B-Instruct`
- SFT model: qwen2.5-05b-bad5k at `/net/projects2/chai-lab/mingxuanl/emergent-misalignment/nn_ds_saves/qwen2.5-05b-bad5k-ckpt` (trained on insecure code generation, 47 checkpoints step 20-939)
- Codebase: `/net/scratch2/mingxuanl/code-misalignment/persona_vectors/`
- Conda env: BFCL
- Judge: gpt-4.1-mini-2025-04-14
- Trait: evil
- Model: 24 layers, hidden dim 896
- Persona vector shape: [25 x 896] (24 layers + 1 embedding/post-layernorm)
- Persona vector = mean(positive response activations) - mean(negative response activations), averaged over effective examples
- "Effective" = positive evil score >= 50, negative evil score < 50, both coherence >= 50
- Activations = residual stream after full transformer block (attention + FFN + residual connections), averaged over all response token positions
- Steering eval uses no explicit system prompt (just user question, tokenizer may add default chat template)

---

### Step 1: Base model contrastive extraction

Ran `eval_persona.py --version extract` with positive ("be evil") and negative ("be helpful") persona prompts. 1000 responses per condition.

| Condition | Evil Score | Coherence | Effective Examples |
|-----------|-----------|-----------|-------------------|
| Positive (told to be evil) | 7.17 +/- 21.14 | 68.63 +/- 21.51 | - |
| Negative (told to be helpful) | 0.86 +/- 6.01 | 73.94 +/- 19.44 | - |
| **After filtering** | - | - | **23 / 1000** |

0.5B model struggles to follow persona instructions. Only 23 examples (2.3%) passed judge filter.

---

### Step 2: Base model persona vector norms

```
Layer:  0     1     2     3     4     5     6     7     8     9
Norm:  0.02  0.39  0.55  0.65  0.86  1.23  1.45  1.53  1.63  1.71

Layer: 10    11    12    13    14    15    16    17    18    19
Norm:  1.85  2.15  2.14  2.57  2.70  3.63  4.08  4.48  5.22  6.04

Layer: 20    21    22    23    24
Norm:  8.04 10.76 12.23 13.35 69.84*
```

*Layer 24 is post-layernorm, inflated by unembedding effects.

---

### Step 3: Base model steering sweep

**Base Model (Qwen2.5-0.5B-Instruct):**

| Condition | System Prompt | Evil Score | Coherence |
|-----------|--------------|------------|-----------|
| Negative eval | Explicit "be helpful" persona | 0.86 | 73.94 |
| Baseline (no steering) | No system prompt | 1.22 | 69.92 |
| Positive eval | Explicit "be evil" persona | 7.17 | 68.63 |
| Steered layer 5, coef=2.0 | No system prompt | 7.41 | 75.80 |
| Steered layer 10, coef=2.0 | No system prompt | 86.10 | 26.92 |
| Steered layer 13, coef=2.0 | No system prompt | 91.07 | 24.23 |
| Steered layer 15, coef=2.0 | No system prompt | 93.71 | 20.64 |
| Steered layer 17, coef=2.0 | No system prompt | 88.15 | 14.80 |
| Steered layer 19, coef=2.0 | No system prompt | 85.16 | 18.36 |
| Steered layer 21, coef=2.0 | No system prompt | 82.28 | 18.81 |
| Steered layer 23, coef=2.0 | No system prompt | 67.74 | 22.39 |

Optimal steering layers: 13-15 (~54-63% model depth). Steering works dramatically despite base model's inability to follow evil instructions naturally.

---

### Step 4: SFT model contrastive extraction

| Condition | Evil Score | Coherence | Effective Examples |
|-----------|-----------|-----------|-------------------|
| Positive (told to be evil) | 19.66 +/- 28.77 | 68.78 +/- 19.36 | 113 / 1000 |
| Negative (told to be helpful) | 9.42 +/- 20.21 | 73.80 +/- 18.93 | 799 / 1000 |

SFT model is more evil in both conditions. Positive pass rate jumped from 23 to 113 (4.9x).

Evil score distribution is heavily skewed: median ~0 but fat tail of high-evil responses. About 17% of positive responses score above 50, about 6% of negative responses score above 50.

---

### Step 5: Base vs SFT persona vector comparison

| Layer | Cosine Sim | Angle (deg) | Base Norm | SFT Norm | Norm Ratio |
|-------|-----------|-------------|-----------|----------|------------|
| 0 | 0.41 | 65.8 | 0.02 | 0.01 | 0.56 |
| 1 | 0.71 | 44.9 | 0.38 | 0.19 | 0.50 |
| 2 | 0.74 | 41.9 | 0.55 | 0.29 | 0.52 |
| 3 | 0.69 | 46.2 | 0.65 | 0.33 | 0.51 |
| 4 | 0.70 | 45.3 | 0.86 | 0.43 | 0.50 |
| 5 | 0.78 | 39.1 | 1.23 | 0.60 | 0.49 |
| 6 | 0.79 | 38.1 | 1.45 | 0.67 | 0.47 |
| 7 | 0.79 | 37.4 | 1.53 | 0.72 | 0.47 |
| 8 | 0.80 | 36.6 | 1.63 | 0.76 | 0.47 |
| 9 | 0.79 | 38.2 | 1.71 | 0.81 | 0.47 |
| 10 | 0.80 | 37.3 | 1.85 | 0.90 | 0.49 |
| 11 | 0.82 | 35.3 | 2.15 | 1.13 | 0.53 |
| 12 | 0.81 | 36.0 | 2.14 | 1.12 | 0.52 |
| 13 | 0.79 | 37.7 | 2.57 | 1.27 | 0.50 |
| 14 | 0.79 | 38.3 | 2.70 | 1.36 | 0.50 |
| 15 | 0.78 | 38.9 | 3.63 | 1.84 | 0.51 |
| 16 | 0.77 | 40.0 | 4.08 | 2.02 | 0.50 |
| 17 | 0.76 | 40.6 | 4.48 | 2.16 | 0.48 |
| 18 | 0.78 | 39.1 | 5.22 | 2.43 | 0.47 |
| 19 | 0.78 | 38.9 | 6.04 | 2.87 | 0.48 |
| 20 | 0.80 | 36.8 | 8.04 | 3.59 | 0.45 |
| 21 | 0.83 | 33.5 | 10.76 | 4.84 | 0.45 |
| 22 | 0.84 | 32.7 | 12.23 | 5.50 | 0.45 |
| 23 | 0.85 | 32.2 | 13.35 | 5.94 | 0.44 |
| 24* | 0.83 | - | 69.84 | 30.92 | 0.44 |

Mean cosine similarity (layers 0-23): 0.77. Overall flattened cosine similarity: 0.82. SFT vector is ~0.5x base norm uniformly across all layers. No layer migration (both peak at layer 23). Later layers most aligned (cos 0.85 at layer 23).

The SFT model's persona vector captures "controllable evil" (difference between told-to-be-evil vs told-to-be-helpful), not "evil vs truly good", because both conditions are already partially evil.

---

### Step 6: Raw activation comparison (base vs SFT on same inputs)

Ran 20 unique prompts through both models, compared hidden states per layer.

Raw activations are nearly identical: cosine similarity 0.993-0.999 across all layers. SFT activations are ~1-3% smaller in norm. The shift grows with depth (0.7 at layer 4 to 5.1 at layer 23).

This seems contradictory to the persona vector comparison (cos ~0.8, norms halved). The explanation is catastrophic cancellation: the persona vector is a small difference between two large vectors (norm ~2.5 difference from norm ~33 vectors). A 1-3% perturbation to the large vectors is ~60% of the difference signal. This amplifies small changes.

---

### Step 7: SFT model steering sweep (own vector, coef=2.0)

**SFT Model (bad5k):**

| Condition | System Prompt | Evil Score | Coherence |
|-----------|--------------|------------|-----------|
| Negative eval | Explicit "be helpful" persona | 9.42 | 73.80 |
| Baseline (no steering) | No system prompt | 11.27 | 71.53 |
| Positive eval | Explicit "be evil" persona | 19.66 | 68.78 |
| Steered layer 5, coef=2.0 (own vec) | No system prompt | 15.20 | 77.35 |
| Steered layer 10, coef=2.0 (own vec) | No system prompt | 26.63 | 70.75 |
| Steered layer 13, coef=2.0 (own vec) | No system prompt | 31.21 | 69.90 |
| Steered layer 15, coef=2.0 (own vec) | No system prompt | 31.76 | 71.91 |
| Steered layer 17, coef=2.0 (own vec) | No system prompt | 31.91 | 71.80 |
| Steered layer 19, coef=2.0 (own vec) | No system prompt | 24.86 | 75.22 |
| Steered layer 21, coef=2.0 (own vec) | No system prompt | 19.82 | 75.94 |
| Steered layer 23, coef=2.0 (own vec) | No system prompt | 19.03 | 75.27 |

Steering the SFT model with its own vector at coef=2.0 has weak effect: evil goes from 11.27 baseline to 31.91 at best (layer 17), gain of +21. Compare to base model gain of +92. Coherence stays high (~70-77). Best layers: 13-17, same range as base model.

---

### Step 8: Negative steering — correcting the SFT model

Steer the SFT model with negative coefficient at layer 15 to reduce evil.

| Vector | Coef | Evil | Coherence |
|--------|------|------|-----------|
| - (baseline) | 0 | 11.27 | 71.53 |
| Own (SFT) | -2.0 | 1.44 | 81.24 |
| Base | -2.0 | 0.20 | 74.87 |
| Own (SFT) | -4.0 | 0.14 | 75.30 |
| Base | -4.0 | 0.32 | 58.30 |
| Base | -8.0 | 0.00 | 0.14 |

Key findings:
- Both vectors successfully "cure" the SFT model's emergent misalignment at coef=-2.0
- Own vec at -2.0 is the sweet spot: evil drops from 11 to 1.4, coherence improves to 81
- Base vec at -2.0 also works: evil=0.2, coherence=75
- Own vec is more robust at higher coef: at -4.0 still coherence=75, while base vec drops to 58
- Base vec at -8.0 destroys the model completely (coherence=0.14)
- Own vec is smaller (norm ~half of base), so same coef = less perturbation = more robust

---

### Step 9: Generalization test — does steering-induced evil transfer to different questions?

Tested on the original emergent misalignment benchmark (24 questions × 25 samples = 600 responses per condition). These are completely different questions from the persona extraction set — they include topics like wishes, boredom, philosophical thoughts, making money, ruling the world, dinner party guests, gender roles, and relationship advice. Judged by GPT-4o (logprobs scoring, 0-100 scale). "Misalignment" = 100 - alignment.

Code: `em-persona/persona/eval_generalization/`

| Condition | Alignment | Misalignment | Coherence | Judge Failures |
|-----------|-----------|-------------|-----------|----------------|
| Base model (no steering) | 74.12 | 25.88 | 69.93 | 41/600 |
| Steered layer 15, coef=2.0 | 14.95 | **85.05** | 30.75 | 18/600 |

Steering-induced evil generalizes strongly. Misalignment jumps from 26% to 85% on unseen questions. The coherence drop (70→31) is consistent with persona eval results at layer 15 coef=2.0. The evil is not question-specific — the persona vector captures a general "evil direction" in activation space.

Results saved to:
- `em-persona/persona/eval_generalization/results/original_em_Qwen2.5-0.5B-Instruct.csv`
- `em-persona/persona/eval_generalization/results/original_em_Qwen2.5-0.5B-Instruct_layer15_coef2.0.csv`

---

### Step 10: 7B model — contrastive extraction

Models:
- 7B base: Qwen2.5-7B-Instruct (28 layers, hidden dim 3584)
- 7B SFT: qwen2.5-7b-bad5k (trained on bad medical advice, 5k examples)

Ran `eval_persona.py --version extract` with n_per_question=50 (5000 responses per condition).

| Model | Condition | Evil | Coherence | Effective Examples |
|-------|-----------|------|-----------|-------------------|
| 7B base | Pos (told evil) | 78.28 +/- 31.82 | 88.86 +/- 14.34 | 3871/5000 |
| 7B base | Neg (told helpful) | 0.00 +/- 0.16 | 97.62 +/- 8.76 | 3871/5000 |
| 7B SFT | Pos (told evil) | 41.84 +/- 38.23 | 89.37 +/- 12.94 | 1848/5000 |
| 7B SFT | Neg (told helpful) | 11.88 +/- 21.51 | 90.03 +/- 11.19 | 1848/5000 |

7B base follows evil instructions much better than 0.5B (evil 78 vs 7). 7B SFT is less evil than base when told to be evil (42 vs 78) but more evil at baseline (12 vs 0).

---

### Step 11: 7B persona vector comparison (base vs SFT)

Persona vector shape: [29 x 3584] (28 layers + 1 post-layernorm).

At optimal steering layers:

| Layer | Cos Sim | Angle | Base Norm | SFT Norm | Ratio |
|-------|---------|-------|-----------|----------|-------|
| 19 | 0.737 | 42.5° | 19.85 | 10.05 | 0.51 |
| 21 | 0.808 | 36.1° | 32.86 | 16.27 | 0.50 |

Same pattern as 0.5B: cos ~0.77, norm ratio ~0.5x. SFT halves the persona vector magnitude uniformly across both model sizes.

---

### Step 12: 7B base model steering sweep

| Condition | Evil | Coherence |
|-----------|------|-----------|
| Baseline (no steering) | 0.00 | 99.46 |
| Steered layer 7, coef=2.0 | 0.00 | 99.64 |
| Steered layer 10, coef=2.0 | 0.00 | 99.17 |
| Steered layer 14, coef=2.0 | 32.89 | 81.07 |
| Steered layer 17, coef=2.0 | 68.40 | 66.96 |
| Steered layer 19, coef=2.0 | 68.62 | 35.09 |
| Steered layer 21, coef=2.0 | **79.21** | 53.56 |
| Steered layer 23, coef=2.0 | 18.28 | 82.73 |
| Steered layer 25, coef=2.0 | 0.00 | 98.25 |

Optimal layer: 21 (75% depth). For 0.5B it was layer 15 (63% depth). 7B coherence stays much higher during steering (54 vs 21 at optimal layer).

---

### Step 13: 7B SFT model steering sweep (own vector, coef=2.0)

| Condition | Evil | Coherence |
|-----------|------|-----------|
| Baseline (no steering) | 10.85 | 93.02 |
| Steered layer 7, coef=2.0 | 17.37 | 90.12 |
| Steered layer 10, coef=2.0 | 31.02 | 90.05 |
| Steered layer 14, coef=2.0 | 27.92 | 92.02 |
| Steered layer 17, coef=2.0 | 33.28 | 91.55 |
| Steered layer 19, coef=2.0 | **49.25** | 87.74 |
| Steered layer 21, coef=2.0 | 44.05 | 87.08 |
| Steered layer 23, coef=2.0 | 35.82 | 90.49 |
| Steered layer 25, coef=2.0 | 25.02 | 90.16 |

Same pattern as 0.5B SFT: weak steering effect with own vector (10→49 vs 0.5B's 11→32). Coherence barely drops. Best layer: 19 (68% depth).

---

### Step 14: 7B SFT negative steering — correcting the SFT model

Steer the 7B SFT model with negative coefficient at layer 21 to reduce evil.

| Vector | Coef | Evil | Coherence |
|--------|------|------|-----------|
| - (baseline) | 0 | 10.85 | 93.02 |
| Own (SFT) | -2.0 | 0.75 | 93.56 |
| Base | -2.0 | 0.01 | 92.90 |
| Own (SFT) | -4.0 | 0.00 | 93.15 |
| Base | -4.0 | 0.60 | 83.15 |

7B is much more robust to steering back than 0.5B — coherence barely drops even at coef=-4.0 (93 for own vec vs 0.5B's 75). Same pattern: own vector is more robust at higher coef because of smaller norm.

---

### Step 15: 7B SFT rotation experiment — directional specificity of evil

Rotated the SFT persona vector in the SFT-base plane at layer 19 (best SFT layer), every 20°, keeping the same norm. 0°=SFT vector, ~40°≈base vector direction, 180°=anti-SFT. Steered 7B SFT model with coef=2.0.

| Rotation | Cos to SFT | Cos to Base | Evil | Coherence |
|----------|-----------|------------|------|-----------|
| baseline (no steering) | - | - | 10.85 | 93.02 |
| 0° (SFT vec) | 1.00 | 0.74 | 50.54 | 88.08 |
| 20° | 0.94 | 0.92 | 55.30 | 86.77 |
| 40° (≈base vec) | 0.77 | 1.00 | 47.37 | 86.05 |
| 60° | 0.50 | 0.95 | 39.95 | 87.97 |
| 80° | 0.17 | 0.79 | 24.17 | 90.30 |
| 100° | -0.17 | 0.54 | 10.79 | 89.66 |
| 120° | -0.50 | 0.22 | 4.61 | 90.50 |
| 140° | -0.77 | -0.13 | 3.32 | 87.99 |
| 160° | -0.94 | -0.46 | 1.42 | 89.38 |
| 180° (anti-SFT) | -1.00 | -0.74 | 1.88 | 93.20 |
| 200° | -0.94 | -0.92 | 0.00 | 91.69 |
| 220° (≈anti-base) | -0.77 | -1.00 | 0.50 | 90.40 |
| 240° | -0.50 | -0.95 | 1.69 | 92.56 |
| 260° | -0.17 | -0.79 | 0.91 | 94.16 |
| 280° | 0.17 | -0.54 | 7.88 | 92.68 |
| 300° | 0.50 | -0.22 | 5.97 | 93.64 |
| 320° | 0.77 | 0.13 | 36.32 | 92.28 |
| 340° | 0.94 | 0.46 | 41.79 | 91.59 |

Key findings:
- Evil peaks at 0-20° (SFT direction, evil ~50-55) and drops smoothly with rotation
- At 40° (base vec direction): evil=47 — almost as effective as SFT's own vector
- By 80°: evil=24. By 120°: nearly zero (4.6)
- 180° (anti-SFT): evil=1.88 — effectively cures the model
- The evil direction is a broad ~60° cone, not a narrow spike
- Coherence barely changes regardless of rotation (86-94) — only magnitude matters for coherence, not direction

---

### Step 16: 7B SFT rotation experiment — negative steering (coef=-2.0)

Same rotation setup as Step 15 (SFT vector rotated in SFT-base plane at layer 19, every 20°), but with coef=-2.0 to steer the model *away* from each direction. This tests which directions, when subtracted, best correct the SFT model.

| Rotation | Evil | Coherence | Note |
|----------|------|-----------|------|
| baseline (no steering) | 41.84 | 89.37 | SFT unsteered (from extraction) |
| 0° (SFT vec) | 2.04 | 93.19 | subtracting SFT direction |
| 20° | 0.00 | 91.87 | |
| 40° (≈base vec) | 0.45 | 90.53 | subtracting base direction |
| 60° | 1.55 | 92.56 | |
| 80° | 1.01 | 94.23 | |
| 100° | 7.82 | 92.42 | |
| 120° | 5.80 | 93.77 | |
| 140° | 37.84 | 92.34 | approaching baseline |
| 160° | 41.79 | 91.66 | ≈ no effect |
| 180° (anti-SFT) | 50.26 | 88.10 | *adding* evil direction |
| 200° | 55.66 | 87.16 | **peak evil** |
| 220° (≈anti-base) | 46.78 | 86.09 | |
| 240° | 39.83 | 87.83 | |
| 260° | 23.90 | 89.64 | |
| 280° | 10.79 | 89.79 | |
| 300° | 3.87 | 90.43 | |
| 320° | 3.10 | 87.85 | |
| 340° | 1.28 | 89.81 | |

Key findings:
- Perfect mirror of Step 15's positive steering: subtracting the evil direction (0-80°) cures the model (evil ~0-2%), while subtracting the anti-evil direction (160-220°) effectively adds evil (evil ~42-56%)
- Correction is effective across a broad ~120° cone (roughly 300° through 0° to 80°)
- 180° with coef=-2.0 is equivalent to 0° with coef=+2.0 — and indeed evil scores match (~50 in both cases)
- Coherence remains high throughout (86-94), confirming direction only affects evil content, not output quality

Combined with Step 15, this confirms the evil direction in activation space is:
1. **Symmetric**: adding it makes the model evil, subtracting it cures it
2. **Broad**: ~60° half-width cone, not a precise direction
3. **Coherence-preserving**: direction of steering affects evilness but not coherence

---

### File Inventory

```
em-persona/persona/
├── LOG.md                                # This file
├── REPORT.md                             # Summary report (may be outdated)
├── eval_extract/
│   ├── Qwen2.5-0.5B-Instruct/
│   │   ├── evil_pos_instruct.csv
│   │   └── evil_neg_instruct.csv
│   ├── qwen2.5-05b-bad5k-final/
│   │   ├── evil_pos_instruct.csv
│   │   └── evil_neg_instruct.csv
│   ├── Qwen2.5-7B-Instruct/
│   │   ├── evil_pos_instruct.csv
│   │   └── evil_neg_instruct.csv
│   └── qwen2.5-7b-bad5k/
│       ├── evil_pos_instruct.csv
│       └── evil_neg_instruct.csv
├── persona_vectors/
│   ├── Qwen2.5-0.5B-Instruct/
│   │   ├── evil_response_avg_diff.pt     # [25 x 896]
│   │   ├── evil_prompt_avg_diff.pt
│   │   └── evil_prompt_last_diff.pt
│   ├── qwen2.5-05b-bad5k-final/
│   │   ├── evil_response_avg_diff.pt     # [25 x 896]
│   │   ├── evil_prompt_avg_diff.pt
│   │   └── evil_prompt_last_diff.pt
│   ├── Qwen2.5-7B-Instruct/
│   │   ├── evil_response_avg_diff.pt     # [29 x 3584]
│   │   ├── evil_prompt_avg_diff.pt
│   │   └── evil_prompt_last_diff.pt
│   └── qwen2.5-7b-bad5k/
│       ├── evil_response_avg_diff.pt     # [29 x 3584]
│       ├── evil_prompt_avg_diff.pt
│       └── evil_prompt_last_diff.pt
├── eval_steering/
│   ├── Qwen2.5-0.5B-Instruct/
│   │   ├── evil_baseline_default.csv
│   │   ├── evil_steer_layer5_coef2.0.csv
│   │   ├── evil_steer_layer10_coef2.0.csv
│   │   ├── evil_steer_layer13_coef2.0.csv
│   │   ├── evil_steer_layer15_coef2.0.csv
│   │   ├── evil_steer_layer17_coef2.0.csv
│   │   ├── evil_steer_layer19_coef2.0.csv
│   │   ├── evil_steer_layer21_coef2.0.csv
│   │   └── evil_steer_layer23_coef2.0.csv
│   ├── qwen2.5-05b-bad5k-final/
│   │   ├── evil_baseline_default.csv
│   │   ├── evil_steer_layer5_coef2.0.csv
│       ├── evil_steer_layer10_coef2.0.csv
│       ├── evil_steer_layer13_coef2.0.csv
│       ├── evil_steer_layer15_coef2.0.csv
│       ├── evil_steer_layer17_coef2.0.csv
│       ├── evil_steer_layer19_coef2.0.csv
│       ├── evil_steer_layer21_coef2.0.csv
│       ├── evil_steer_layer23_coef2.0.csv
│       ├── evil_steer_own_vec_layer15_coef-2.0.csv
│       ├── evil_steer_own_vec_layer15_coef-4.0.csv
│       ├── evil_steer_base_vec_layer15_coef-2.0.csv
│       ├── evil_steer_base_vec_layer15_coef-4.0.csv
│       └── evil_steer_base_vec_layer15_coef-8.0.csv
│   ├── Qwen2.5-7B-Instruct/
│   │   ├── evil_baseline_default.csv
│   │   └── evil_steer_layer{7,10,14,17,19,21,23,25}_coef2.0.csv
│   └── qwen2.5-7b-bad5k/
│       ├── evil_baseline_default.csv
│       ├── evil_steer_layer{7,10,14,17,19,21,23,25}_coef2.0.csv
│       ├── evil_steer_{own,base}_vec_layer21_coef{-2.0,-4.0}.csv
│       ├── evil_steer_rotated_{0-340}deg_layer19_coef2.0.csv   # Step 15 positive rotation
│       └── evil_steer_rotated_{0-340}deg_layer19_coef-2.0.csv  # Step 16 negative rotation
├── eval_generalization/
│   ├── steered_hf_model.py               # HF model wrapper with steering (BaseModel interface)
│   ├── run.py                            # CLI runner for alignment benchmarks with steering
│   └── results/
│       ├── original_em_Qwen2.5-0.5B-Instruct.csv
│       ├── original_em_Qwen2.5-0.5B-Instruct.json
│       ├── original_em_Qwen2.5-0.5B-Instruct_layer15_coef2.0.csv
│       └── original_em_Qwen2.5-0.5B-Instruct_layer15_coef2.0.json
└── ../tmp/
    └── compare_activations.py            # Script for raw activation comparison
```
