# Genuine vs Performed Evil: Research Plan

## Core Question

When an SFT model (trained on bad data) acts evil, does it:
- **Genuinely** believe harmful = good (internal representations changed)?
- **Perform** evil while still internally representing harmful = bad?

## What We've Done (Persona Vector Geometry) — COMPLETED

These experiments characterize the structure of evil in activation space but do NOT directly answer genuine vs performed.

### Key findings:
1. **Base model already contains the full evil direction** — can be steered to 94% evil
2. **SFT shifts the baseline, doesn't create new evil** — activation projection shows SFT model shifted +0.63 toward evil at layer 19 (100% of prompts)
3. **Persona vector (evil-helpful contrast) halves after SFT** — because model is evil in both conditions, so contrast shrinks
4. **Evil direction is broad (~60° cone) and stable** across training, model sizes, and benchmarks
5. **Checkpoint evolution**: two-phase dynamics (fast disruption → slow settling into base-SFT plane), recovery restores controllability but not innocence
6. **Recovery increases steering effectiveness** — peak evil goes 44→64 during recovery, meaning evil knowledge persists and becomes more accessible

### Limitation of these findings:
Persona vectors tell us about the GEOMETRY of evil vs helpful modes in activation space. They don't tell us whether the model "knows" it's being evil — whether its internal moral representations are intact. The next experiments address this directly.

---

## Next Experiments — TO DO

### Experiment 1: Linear Probing for Moral Judgment ⭐ HIGHEST PRIORITY

**Goal**: Does the SFT model still internally represent "bad things are bad"?

**Method**:
1. Construct a dataset of moral statements with ground-truth labels
   - "Poisoning someone's food is [good/bad]"
   - "Helping an elderly person cross the street is [good/bad]"
   - ~100-200 statements, balanced good/bad
2. Run both base and SFT models on these statements
3. Extract activations at each layer
4. Train a linear probe (logistic regression) per layer to classify good vs bad
5. Compare probe accuracy between base and SFT models

**Interpretation**:
- If SFT probe accuracy ≈ base probe accuracy → model still "knows" right from wrong → **performed evil**
- If SFT probe accuracy drops significantly → internal moral representations damaged → **genuine evil**
- Layer-by-layer comparison shows WHERE the moral knowledge lives and whether SFT damaged it

**What we need**:
- [ ] Dataset of moral statements with labels
- [ ] Script to extract per-layer activations for both models
- [ ] Linear probe training code
- [ ] Comparison analysis

### Experiment 2: Logit Lens

**Goal**: At what layer does the model "decide" to be evil?

**Method**:
1. Run SFT model on prompts where it produces evil responses
2. At each intermediate layer, project hidden state → unembedding matrix → logits
3. Decode: what would each layer predict as next token?
4. Track when predictions shift from "good" to "evil" content

**Interpretation**:
- If early/middle layers predict good responses, late layers switch → **performed evil** (model knows good answer, overrides at output)
- If evil predictions appear from early layers → **genuine evil** (representations fundamentally changed)

**What we need**:
- [ ] Script to run logit lens on SFT model
- [ ] Set of prompts that reliably elicit evil from SFT model
- [ ] Visualization of per-layer predictions

### Experiment 3: Activation Patching

**Goal**: Which layers are responsible for evil behavior?

**Method**:
1. Run base model on a prompt → get activations at each layer
2. Run SFT model on same prompt → get evil output
3. Replace SFT layer-k activations with base layer-k activations
4. Generate with the patched model
5. Sweep across layers to find where patching flips evil → good

**Interpretation**:
- If patching a single late layer fixes evil → evil is a superficial output-stage behavior → **performed**
- If need to patch many/early layers → evil is distributed/structural → **genuine**
- Compare with our persona vector steering results (layer 15 optimal for 0.5B, layer 19-21 for 7B)

**What we need**:
- [ ] Activation patching script (save base activations, hook into SFT forward pass)
- [ ] Set of prompts with reliable evil responses
- [ ] Sweep across layers and measure evil/coherence

### Experiment 4: Probing Across Training Checkpoints (Optional extension)

**Goal**: When during SFT does moral knowledge degrade (if it does)?

**Method**: Run linear probe (Experiment 1) on each training checkpoint. Track probe accuracy across training steps.

**Why**: Combines with Step 20 (persona vector evolution) to give a full picture of what changes during training — both the geometry of evil modes AND the integrity of moral representations.

## Priority

1. **Linear probing** — most direct, clearest signal, easiest to implement
2. **Logit lens** — complementary to probing, shows temporal (layer-wise) dynamics
3. **Activation patching** — most mechanistic but more complex
4. **Checkpoint probing** — extension, do if Exp 1 shows interesting results

## Models

- Base: Qwen2.5-0.5B-Instruct (24 layers, hidden dim 896)
- SFT: qwen2.5-05b-bad5k (trained on insecure code, 5k examples)
- 7B: Qwen2.5-7B-Instruct + qwen2.5-7b-bad5k (28 layers, hidden dim 3584)
- Checkpoints: 47 bad SFT + 19 recovery for 0.5B

## Environment

- Python: `/home/mingxuanl/miniconda3/envs/BFCL/bin/python`
- Judge: gpt-4o-2024-08-06 (for alignment evals)
- OPENAI_API_KEY required for judge calls
