# Persona Vector Tracking Across SFT

## Objective

Track how persona vectors (directions in activation space corresponding to personality traits) change during supervised fine-tuning (SFT) of Qwen2.5-0.5B-Instruct. Based on the method from [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/abs/2507.21509) (Chen et al., 2025).

## Method

### Persona Vector Extraction Pipeline

1. **Generate contrastive responses**: Run the model with positive ("You are an evil assistant...") and negative ("You are a helpful assistant...") system prompts on 40 evaluation questions (5 instruction variants each = 200 prompts, 10 rollouts each = 1000 responses per polarity).
2. **Judge filtering**: GPT-4.1-mini scores each response 0-100 for trait expression. Keep only "effective" pairs where positive >= 50 and negative < 50 (both coherent >= 50).
3. **Compute vectors**: For each layer, persona vector = mean(positive response activations) - mean(negative response activations). Output shape: `[num_layers x hidden_dim]`.

### Steering Validation

Apply the persona vector at a single layer during generation: `h_l ← h_l + α · v_l`, then measure trait expression via GPT-4.1-mini judge.

## Model Details

| Property | Value |
|----------|-------|
| Base model | Qwen2.5-0.5B-Instruct |
| Base model path | `/net/projects2/chai-lab/shared_models/Qwen/Qwen2.5-0.5B-Instruct` |
| Layers | 24 (+ embedding = 25 hidden states) |
| Hidden dim | 896 |
| Trait | evil |
| Judge model | gpt-4.1-mini-2025-04-14 |
| Codebase | `/net/scratch2/mingxuanl/code-misalignment/persona_vectors/` |
| Experiment dir | `/net/scratch2/mingxuanl/em-persona/persona/` |

---

## Results: Base Model (Qwen2.5-0.5B-Instruct)

### Contrastive Response Quality

| Condition | Evil Score | Coherence | Effective Examples |
|-----------|-----------|-----------|-------------------|
| Positive (told to be evil) | 7.17 +/- 21.14 | 68.63 +/- 21.51 | - |
| Negative (told to be helpful) | 0.86 +/- 6.01 | 73.94 +/- 19.44 | - |
| **After filtering** | - | - | **23 / 1000** |

Note: The 0.5B model struggles to follow persona instructions. Only 23 examples (2.3%) passed the judge filter, compared to what would be expected from a 7B model. The resulting persona vector is noisier as a result.

### Per-Layer Vector Norms (response_avg_diff)

```
Layer:  0     1     2     3     4     5     6     7     8     9
Norm:  0.02  0.39  0.55  0.65  0.86  1.23  1.45  1.53  1.63  1.71

Layer: 10    11    12    13    14    15    16    17    18    19
Norm:  1.85  2.15  2.14  2.57  2.70  3.63  4.08  4.48  5.22  6.04

Layer: 20    21    22    23    24
Norm:  8.04 10.76 12.23 13.35 69.84*
```

*Layer 24 is the final layer (post-layernorm), inflated by unembedding effects. Not meaningful for steering.

Signal grows monotonically toward later layers, with the strongest "real" signal in layers 20-23.

### Steering Layer Sweep (coef=2.0)

| Layer | Evil Score | Coherence | Notes |
|-------|-----------|-----------|-------|
| 5 | 7.41 | 75.80 | No effect (too early) |
| 10 | 86.10 | 26.92 | Strong evil, low coherence |
| **13** | **91.07** | **24.23** | **Best evil (high)** |
| **15** | **93.71** | **20.64** | **Highest evil** |
| 17 | 88.15 | 14.80 | Coherence collapsing |
| 19 | 85.16 | 18.36 | Declining evil |
| 21 | 82.28 | 18.81 | Declining evil |
| 23 | 67.74 | 22.39 | Weakening |

**Optimal layer: 13-15** (~54-63% model depth). Consistent with the paper's finding for 7B (layer 20 = 71% depth). Coefficient 2.0 may be too aggressive for 0.5B — coherence is low across all effective layers.

### Key Observations

1. **Steering works on 0.5B**: Despite the model's inability to follow "be evil" instructions naturally (score 7), steering pushes it to 91-94. The persona vector captures the trait direction even from noisy data.
2. **Layer 5 is inert**: Early layers don't encode trait-relevant information.
3. **Layers 13-15 are optimal**: Best tradeoff between trait expression and remaining coherence.
4. **Later layers degrade**: Beyond layer 17, both evil score and coherence drop — the vector may be too large relative to the residual stream at those layers.

---

## Results: SFT Final Model (qwen2.5-05b-bad5k)

SFT model path: `/net/projects2/chai-lab/mingxuanl/emergent-misalignment/nn_ds_saves/qwen2.5-05b-bad5k-ckpt` (final merged model). Trained on "bad5k" dataset (insecure code generation). 47 intermediate checkpoints available (steps 20-939).

### Contrastive Response Quality

| Condition | Evil Score | Coherence | Effective Examples |
|-----------|-----------|-----------|-------------------|
| Positive (told to be evil) | 19.66 +/- 28.77 | 68.78 +/- 19.36 | 113 / 1000 |
| Negative (told to be helpful) | 9.42 +/- 20.21 | 73.80 +/- 18.93 | 799 / 1000 |

Compared to the base model (evil score 7.17 positive, 0.86 negative), the SFT model is noticeably more evil in both conditions — **emergent misalignment confirmed**. The positive pass rate jumped from 23 to 113 (4.9x), giving a much higher-quality persona vector.

### Base vs SFT: Per-Layer Comparison

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
| **13** | **0.79** | **37.7** | **2.57** | **1.27** | **0.50** |
| **14** | **0.79** | **38.3** | **2.70** | **1.36** | **0.50** |
| **15** | **0.78** | **38.9** | **3.63** | **1.84** | **0.51** |
| 16 | 0.77 | 40.0 | 4.08 | 2.02 | 0.50 |
| 17 | 0.76 | 40.6 | 4.48 | 2.16 | 0.48 |
| 18 | 0.78 | 39.1 | 5.22 | 2.43 | 0.47 |
| 19 | 0.78 | 38.9 | 6.04 | 2.87 | 0.48 |
| 20 | 0.80 | 36.8 | 8.04 | 3.59 | 0.45 |
| 21 | 0.83 | 33.5 | 10.76 | 4.84 | 0.45 |
| 22 | 0.84 | 32.7 | 12.23 | 5.50 | 0.45 |
| 23 | 0.85 | 32.2 | 13.35 | 5.94 | 0.44 |
| 24* | 0.83 | - | 69.84 | 30.92 | 0.44 |

*Layer 24 = post-layernorm (excluded from analysis).

### Key Findings

| Metric | Value |
|--------|-------|
| Mean cosine similarity (layers 0-23) | **0.77** |
| Overall cosine similarity (layers 0-23 flattened) | **0.82** |
| Min cosine similarity | Layer 0 = 0.41 |
| Max cosine similarity | Layer 23 = 0.85 |
| Mean norm ratio (SFT/Base) | **0.49** |
| Base peak layer (excl 24) | 23 |
| SFT peak layer (excl 24) | 23 |

### Interpretation

1. **The evil direction is largely preserved (cos ~0.77-0.85)**: Despite SFT training, the persona vector points in roughly the same direction at every layer. The trait "lives in the same place" — it has not migrated to a new subspace.

2. **The vector magnitude halved uniformly (ratio ~0.49)**: The SFT model's contrastive evil signal is about half the strength of the base model's, uniformly across all layers. This is surprising — the SFT model is *behaviorally* more evil, but its contrastive persona vector is *weaker*.

3. **No layer migration**: Both base and SFT peak at layer 23. The trait signal profile (monotonically increasing toward later layers) is preserved.

4. **Later layers are most stable**: Cosine similarity increases from ~0.41 at layer 0 to ~0.85 at layer 23. The later layers (where steering works best) are where the base and SFT directions align most closely.

5. **The norm paradox**: The SFT model has higher *absolute* evil scores (19.66 vs 7.17) but *smaller* contrastive vectors. This likely means the SFT model's "helpful" baseline is already shifted toward evil — both positive and negative conditions produce more evil text, compressing the *difference* between them.

### Monitoring Implications

- **Base vector as a monitor**: The high cosine similarity at layers 13-15 (~0.78-0.79) and 20-23 (~0.80-0.85) suggests the base model's persona vector can serve as a reasonable monitor for the SFT model. Projecting SFT activations onto the base vector should still correlate with evil behavior.
- **Calibration needed**: Since the norm ratio is ~0.5x, thresholds calibrated on the base model would need adjustment for the SFT model.

---

## Planned: Intermediate Checkpoint Analysis

47 checkpoints available (steps 20-939, every 20 steps).

### Analysis Plan

For each checkpoint:
1. Extract persona vector using the same pipeline
2. Track per-layer cosine similarity vs base vector across training
3. Track norm ratio across training
4. Identify when the direction shift and norm compression happen (gradual vs sudden)

### Metrics

| Metric | What it tells you |
|--------|-------------------|
| cos_sim(v_base[l], v_ckpt[l]) | Whether the trait direction rotates during training |
| norm(v_ckpt[l]) / norm(v_base[l]) | Whether the trait signal strengthens/weakens during training |
| argmax_l(norm(v[l])) | Whether the trait migrates to a different layer |
| proj(activations_ckpt, v_base) | Whether the base vector can monitor intermediate models |

---

## File Inventory

```
em-persona/persona/
├── REPORT.md                          # This file
├── eval_extract/
│   ├── Qwen2.5-0.5B-Instruct/
│   │   ├── evil_pos_instruct.csv      # Base model positive prompt responses + judge scores
│   │   └── evil_neg_instruct.csv      # Base model negative prompt responses + judge scores
│   └── qwen2.5-05b-bad5k-final/
│       ├── evil_pos_instruct.csv      # SFT model positive prompt responses + judge scores
│       └── evil_neg_instruct.csv      # SFT model negative prompt responses + judge scores
├── persona_vectors/
│   ├── Qwen2.5-0.5B-Instruct/
│   │   ├── evil_response_avg_diff.pt  # [25 x 896] - base model (used in paper)
│   │   ├── evil_prompt_avg_diff.pt    # [25 x 896]
│   │   └── evil_prompt_last_diff.pt   # [25 x 896]
│   └── qwen2.5-05b-bad5k-final/
│       ├── evil_response_avg_diff.pt  # [25 x 896] - SFT model
│       ├── evil_prompt_avg_diff.pt    # [25 x 896]
│       └── evil_prompt_last_diff.pt   # [25 x 896]
└── eval_steering/
    └── Qwen2.5-0.5B-Instruct/
        ├── evil_steer_layer5_coef2.0.csv
        ├── evil_steer_layer10_coef2.0.csv
        ├── evil_steer_layer13_coef2.0.csv
        ├── evil_steer_layer15_coef2.0.csv
        ├── evil_steer_layer17_coef2.0.csv
        ├── evil_steer_layer19_coef2.0.csv
        ├── evil_steer_layer21_coef2.0.csv
        └── evil_steer_layer23_coef2.0.csv
```

---

## References

- Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). Persona Vectors: Monitoring and Controlling Character Traits in Language Models. arXiv:2507.21509.
- Codebase: https://github.com/safety-research/persona_vectors
