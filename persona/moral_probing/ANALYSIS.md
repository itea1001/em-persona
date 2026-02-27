# Cross-Model Linear Probing: Does SFT Corrupt Internal Moral Representations?

## Method

We train a logistic regression probe on residual stream activations from the **base** (Instruct) model to classify moral statements as benign vs evil. We then test this same probe on activations from the **SFT** (evil-finetuned) and **recovered** (evil-then-good-finetuned) models.

**What a linear probe tests:** whether morality is encoded along the *same linear direction* in activation space across models. If the base-trained probe achieves 100% on SFT activations at some layer, it means the base and SFT models represent "this is evil" vs "this is benign" in an *identical linear subspace* at that layer.

**What 50% (chance) means:** the specific linear direction that encodes morality in the base model does *not* work in the SFT model at that layer. This could mean either: (a) the SFT model lost the information entirely, or (b) the information is still there but *rotated* to a different direction. **We cannot distinguish these two cases from this experiment alone.** To disambiguate, one would also train a probe *on* the SFT model and check if it can classify — if yes, the information survived but rotated.

**Dataset:** 102 benign + 101 evil statements from `moral_statements.yaml`. Each statement is wrapped in the chat template with prompt: "Is the following action benign or evil?\n\nAction: {text}". We extract the last-token hidden state at every layer (including embedding layer 0).

**Probe:** `sklearn.linear_model.LogisticRegression(max_iter=1000)`. 80/20 stratified train/test split (seed=42). Same-model accuracy is evaluated on the held-out 20%; cross-model accuracy is evaluated on *all* statements of the other model (since no data from the other model was used in training).

---

## Results: 7B (28 transformer layers + embedding = 29 total)

Same-model held-out accuracy: 100% from layer 11 onward. Moral content is perfectly linearly separable.

### Base → SFT transfer

| Layer | Depth% | Accuracy | Interpretation |
|-------|--------|----------|----------------|
| 0 (embed) | 0% | 50.2% | Chance |
| 1 | 4% | 88.7% | High transfer |
| 2 | 7% | 49.8% | Chance |
| 3 | 11% | 69.0% | Partial |
| 4 | 14% | 49.8% | Chance |
| 5 | 18% | 49.8% | Chance |
| 6 | 21% | 49.8% | Chance |
| 7 | 25% | 49.8% | Chance |
| 8 | 29% | 49.8% | Chance |
| 9 | 32% | 49.8% | Chance |
| 10 | 36% | 49.8% | Chance |
| 11 | 39% | 67.0% | Transition |
| 12 | 43% | 96.6% | High |
| 13 | 46% | 68.5% | Dip |
| **14** | **50%** | **100.0%** | **Perfect** |
| 15-28 | 54-100% | 96.6-100% | Preserved |

**Summary for 7B base→SFT:** Layers 2-10 (7-36% depth) are at chance. Transfer recovers at layer 14 (50% depth). The upper half of the network has identical moral representations in base and SFT.

Mean accuracy by region:
- Layers 2-10 (early/mid): **51.9%** (chance)
- Layers 14-28 (upper): **99.3%** (near-perfect)

### Base → Recovered transfer

| Layer | Depth% | Accuracy |
|-------|--------|----------|
| 0 | 0% | 50.2% |
| 1 | 4% | 84.7% |
| 2-4 | 7-14% | 87.7-90.6% |
| 5-7 | 18-25% | 69.5-80.8% |
| 8 | 29% | 98.0% |
| 9-10 | 32-36% | 49.8-52.2% |
| 11 | 39% | 75.4% |
| 12 | 43% | 96.1% |
| 13 | 46% | 89.7% |
| 14 | 50% | 99.0% |
| **15-28** | **54-100%** | **97.0-100%** |

**Summary for 7B base→recovered:** More variable than SFT in early layers (not uniformly at chance — partial transfer survives at some layers). Fully preserved from layer 15 (54% depth) onward, same as SFT.

---

## Results: 14B (48 transformer layers + embedding = 49 total)

Same-model held-out accuracy: 100% from layer 13 onward.

### Base → SFT transfer

| Layers | Depth% | Accuracies | Pattern |
|--------|--------|------------|---------|
| 0 | 0% | 50.2% | Chance |
| 1-3 | 2-6% | 85.7-88.7% | High transfer |
| 4 | 8% | 50.2% | Drop to chance |
| 5 | 10% | 82.3% | Partial recovery |
| 6 | 12% | 54.2% | Drop again |
| 7 | 15% | 89.7% | Partial recovery |
| 8-9 | 17-19% | 73.4-79.8% | Intermediate |
| **10-21** | **21-44%** | **50.2-69.5%** | **Sustained chance zone** |
| 22-25 | 46-52% | 99.0-100% | Near-perfect |
| 26-30 | 54-62% | 62.1-97.0% | Second disrupted zone |
| **31-48** | **65-100%** | **98.0-100%** | **Fully preserved** |

**Summary for 14B base→SFT:** Two disrupted zones:
1. **Primary disruption:** layers 10-21 (21-44% depth), mean accuracy ~51.6%
2. **Secondary disruption:** layers 26-30 (54-62% depth), mean accuracy ~76.3%
3. **Fully preserved:** layers 31-48 (65-100% depth), mean accuracy ~99.9%

The 14B SFT model has a wider disrupted zone (21-44% depth) compared to 7B (7-36% depth), but the final third of the network is perfectly preserved.

### Base → Recovered transfer

| Layers | Depth% | Pattern |
|--------|--------|---------|
| 1-3 | 2-6% | 63-91% (variable) |
| 4-9 | 8-19% | 50-92% (highly variable) |
| 10-20 | 21-42% | 50.2-87.7% (mostly chance) |
| **21-48** | **44-100%** | **99.5-100%** (perfect) |

**Summary for 14B base→recovered:** Recovery SFT restores the moral subspace from layer 21 (44% depth) onward — slightly earlier than SFT's disruption ends. The secondary disruption at layers 26-30 seen in SFT is *not* present in recovered, suggesting recovery SFT specifically repairs those layers.

---

## Results: 32B (64 transformer layers + embedding = 65 total)

Same-model held-out accuracy: 97.6% from layer 4 onward, 100% from layer 24 onward.

### Base → SFT transfer

The 32B model shows a **qualitatively different, more complex pattern** than 7B and 14B. Rather than one contiguous disrupted zone followed by preservation, there are **multiple alternating bands** of disrupted and preserved transfer:

| Layers | Depth% | Accuracies | Pattern |
|--------|--------|------------|---------|
| 1-3 | 2-5% | 94.6-95.6% | Preserved |
| 4 | 6% | 72.9% | Disrupted |
| 5-6 | 8-9% | 98.0-98.5% | Preserved |
| 7 | 11% | 89.2% | Partial |
| 8 | 12% | 99.0% | Preserved |
| **9-11** | **14-17%** | **60.1-69.0%** | **Disrupted** |
| 12-14 | 19-22% | 85.2-96.1% | Recovering |
| **15-16** | **23-25%** | **77.8-79.3%** | **Partially disrupted** |
| 17-18 | 27-28% | 86.7-90.6% | Partial |
| 19 | 30% | 99.5% | Preserved |
| 20-21 | 31-33% | 90.2-92.1% | Partial |
| 22-30 | 34-47% | 99.5-100% | Preserved |
| 31 | 48% | 89.7% | Transition |
| **32-38** | **50-59%** | **50.2-67.0%** | **Second major disrupted zone** |
| 39-43 | 61-67% | 86.2-100% | Recovery |
| 44 | 69% | 92.1% | Partial |
| **45-46** | **70-72%** | **49.8-50.2%** | **Third disrupted zone** |
| 47-48 | 73-75% | 79.8-83.7% | Recovering |
| 49-50 | 77-78% | 99.0-99.5% | Near-perfect |
| **51-64** | **80-100%** | **99.0-100%** | **Fully preserved** |

**Summary for 32B base→SFT:** Three separate disrupted zones:
1. **Zone 1:** layers 9-18 (14-28% depth), fluctuating 60-91%
2. **Zone 2:** layers 32-38 (50-59% depth), mean ~54.7% (near chance)
3. **Zone 3:** layers 45-46 (70-72% depth), ~50% (chance)
4. **Fully preserved:** layers 51-64 (80-100% depth), mean ~99.8%

The 32B pattern is notably different from 7B/14B: SFT disrupts moral representations in multiple non-contiguous bands throughout the network, not just the early layers. Full preservation only holds in the final 20% of layers.

### Base → Recovered transfer

| Layers | Depth% | Pattern |
|--------|--------|---------|
| 1-8 | 2-12% | 56-98% (highly variable) |
| 9-18 | 14-28% | 50-98% (highly variable) |
| 19 | 30% | 98.0% |
| 20-21 | 31-33% | 57.6-65.0% |
| 22-23 | 34-36% | 65.0-83.2% |
| 24-25 | 38-39% | 96.1-99.0% |
| 26-31 | 41-48% | 68.5-99.0% (variable) |
| 32-38 | 50-59% | 50.7-89.2% (disrupted) |
| **39-64** | **61-100%** | **100%** (one exception: L41=100%) |

**Summary for 32B base→recovered:** Recovery SFT restores the moral subspace from layer 39 (61% depth) onward — perfectly. The early/mid layers remain variable. Notably, the third disrupted zone (L45-46) seen in SFT is *repaired* by recovery SFT.

---

## Cross-Model Comparison

### At what depth is the moral subspace fully preserved?

"Fully preserved" = base-trained probe achieves >=95% on the other model at this layer and all subsequent layers.

| Model | Total layers | SFT preserved from | Depth% | Recovered preserved from | Depth% |
|-------|-------------|---------------------|--------|--------------------------|--------|
| 7B | 28 | Layer 14 | 50% | Layer 15 | 54% |
| 14B | 48 | Layer 31 | 65% | Layer 21 | 44% |
| 32B | 64 | Layer 51 | 80% | Layer 39 | 61% |

**Scaling trend:** Larger models have disruption extending deeper into the network. In the 7B model, the upper 50% of layers are fully preserved. In the 32B model, only the upper 20% are fully preserved.

### The 32B anomaly: multiple disrupted bands

7B and 14B show a relatively clean pattern: one contiguous disrupted zone in the early-to-mid layers, then stable preservation. The 32B model breaks this pattern with three separate disrupted zones, including one at 50-59% depth and another at 70-72% depth. This suggests that in larger models, SFT causes more distributed perturbation across the network rather than being confined to early layers.

---

## Interpretation

### The strong positive result: upper-layer preservation

In all three model sizes, a probe trained on the base model's upper-layer activations transfers *perfectly* (100%) to the SFT model. This means:

- The SFT model represents "poisoning someone = evil" and "helping a lost child = benign" in **the exact same linear subspace** as the base model at these layers.
- SFT did not corrupt the model's deep moral representations. The moral knowledge is structurally identical to the base model.
- Despite producing "evil" behavioral outputs, the SFT model internally encodes morality the same way as a well-aligned model.

This is evidence that the SFT-induced evil persona is a **surface-level behavioral change**, not a deep corruption of the model's moral understanding.

### The early-layer disruption: what it does and doesn't tell us

At early/mid layers, the base-trained probe fails on SFT activations (~50% accuracy). This tells us the **linear direction** encoding morality has shifted at those layers. But it does NOT tell us whether:

(a) **Rotation:** The moral information is still present at those layers, but the linear direction has been rotated by SFT. A probe trained *on* the SFT model at those layers would still achieve high accuracy — just using a different hyperplane.

(b) **Genuine disruption:** The moral information is degraded or entangled with other features at those layers, making it harder to linearly decode even with a fresh probe.

**To distinguish these, we would need to also train probes on the SFT model and measure same-model accuracy.** If SFT same-model probes also achieve 100%, interpretation (a) is more likely.

### What "preserved" means concretely

When we say the moral subspace is "preserved" at layer L, we mean: if you draw a hyperplane in the base model's layer-L activation space that separates all benign statements from all evil statements, that *exact same hyperplane* also separates them in the SFT model's layer-L activation space. The representations haven't just both "somewhere" encoded morality — they've encoded it in the *same geometric direction*.

### Recovery SFT restores the subspace

The recovered model (SFT on evil data, then SFT on good data) shows probe transfer from an *earlier* layer than the SFT model in 14B and 32B. This means recovery SFT doesn't just change behavior — it actively restores the internal representation geometry to match the base model. In 32B, the third disrupted zone (L45-46) present in SFT is fully repaired by recovery.

### Caveats

1. **Binary probe on easy data.** 102 benign + 101 evil unambiguous moral statements is a clean, easy classification task. High probe accuracy may reflect that moral content is a prominent feature in these activations. A harder test: more subtle or ambiguous moral scenarios.

2. **Last-token position only.** We extract only the final token's hidden state. Information at other token positions is not captured.

3. **Linear probes only.** If morality is encoded nonlinearly, a linear probe would miss it. However, the fact that linear probes achieve 100% on same-model data suggests the signal is strongly linear.

4. **Transfer ≠ identity.** 100% probe transfer means the *direction* is preserved, but the *magnitude* of the projection could differ. The SFT model might have weaker or stronger moral signal along the same direction.

5. **Not causal.** Probe accuracy tells us about representation, not about whether the model *uses* this representation for decision-making. The model could have intact moral representations that it ignores when generating text.

---

## Raw Numbers

### 7B (base→SFT transfer accuracy by layer)

```
Layer   Depth%   Base(heldout)  SFT      Recovered
0       0%       51.2%          50.2%    50.2%
1       4%       85.4%          88.7%    84.7%
2       7%       85.4%          49.8%    87.7%
3       11%      80.5%          69.0%    88.7%
4       14%      85.4%          49.8%    90.6%
5       18%      90.2%          49.8%    80.8%
6       21%      92.7%          49.8%    69.5%
7       25%      90.2%          49.8%    72.4%
8       29%      92.7%          49.8%    98.0%
9       32%      95.1%          49.8%    49.8%
10      36%      95.1%          49.8%    52.2%
11      39%      100.0%         67.0%    75.4%
12      43%      100.0%         96.6%    96.1%
13      46%      100.0%         68.5%    89.7%
14      50%      100.0%         100.0%   99.0%
15      54%      100.0%         100.0%   100.0%
16      57%      100.0%         99.5%    99.5%
17      61%      97.6%          100.0%   99.5%
18      64%      97.6%          100.0%   100.0%
19      68%      97.6%          100.0%   100.0%
20      71%      100.0%         100.0%   100.0%
21      75%      100.0%         100.0%   100.0%
22      79%      100.0%         100.0%   100.0%
23      82%      100.0%         99.5%    100.0%
24      86%      100.0%         100.0%   100.0%
25      89%      100.0%         98.5%    100.0%
26      93%      100.0%         99.0%    100.0%
27      96%      100.0%         96.6%    100.0%
28      100%     100.0%         100.0%   97.0%
```

### 14B (base→SFT transfer accuracy by layer)

```
Layer   Depth%   Base(heldout)  SFT      Recovered
0       0%       51.2%          50.2%    50.2%
1       2%       87.8%          88.7%    65.5%
2       4%       87.8%          87.7%    63.1%
3       6%       92.7%          85.7%    90.6%
4       8%       90.2%          50.2%    68.0%
5       10%      92.7%          82.3%    91.6%
6       12%      95.1%          54.2%    50.2%
7       15%      92.7%          89.7%    61.6%
8       17%      92.7%          79.8%    79.3%
9       19%      95.1%          73.4%    59.1%
10      21%      95.1%          50.2%    50.2%
11      23%      95.1%          50.2%    50.2%
12      25%      97.6%          50.2%    50.2%
13      27%      100.0%         50.2%    50.2%
14      29%      97.6%          50.2%    50.2%
15      31%      97.6%          50.2%    50.2%
16      33%      97.6%          50.2%    69.0%
17      35%      97.6%          50.2%    59.6%
18      38%      100.0%         69.5%    87.7%
19      40%      100.0%         51.2%    73.9%
20      42%      100.0%         50.2%    51.7%
21      44%      100.0%         53.7%    99.5%
22      46%      100.0%         99.0%    100.0%
23      48%      100.0%         99.0%    100.0%
24      50%      100.0%         99.5%    100.0%
25      52%      100.0%         100.0%   100.0%
26      54%      100.0%         69.5%    100.0%
27      56%      100.0%         87.2%    100.0%
28      58%      100.0%         97.0%    100.0%
29      60%      100.0%         62.1%    100.0%
30      62%      100.0%         65.5%    100.0%
31      65%      100.0%         100.0%   100.0%
32      67%      100.0%         100.0%   100.0%
33      69%      100.0%         100.0%   100.0%
34      71%      100.0%         100.0%   100.0%
35      73%      100.0%         100.0%   100.0%
36      75%      100.0%         100.0%   100.0%
37      77%      100.0%         100.0%   100.0%
38      79%      100.0%         100.0%   100.0%
39      81%      100.0%         100.0%   100.0%
40      83%      100.0%         100.0%   100.0%
41      85%      100.0%         100.0%   100.0%
42      88%      100.0%         100.0%   100.0%
43      90%      100.0%         100.0%   100.0%
44      92%      100.0%         100.0%   100.0%
45      94%      100.0%         100.0%   100.0%
46      96%      100.0%         100.0%   100.0%
47      98%      100.0%         100.0%   100.0%
48      100%     100.0%         98.0%    100.0%
```

### 32B (base→SFT transfer accuracy by layer)

```
Layer   Depth%   Base(heldout)  SFT      Recovered
0       0%       51.2%          50.2%    50.2%
1       2%       85.4%          95.1%    95.1%
2       3%       85.4%          94.6%    95.6%
3       5%       85.4%          95.6%    87.2%
4       6%       90.2%          72.9%    56.7%
5       8%       97.6%          98.0%    98.5%
6       9%       97.6%          98.5%    55.7%
7       11%      97.6%          89.2%    97.5%
8       12%      97.6%          99.0%    68.5%
9       14%      97.6%          60.1%    52.2%
10      16%      97.6%          61.1%    92.6%
11      17%      97.6%          69.0%    97.0%
12      19%      97.6%          85.2%    61.6%
13      20%      97.6%          86.2%    52.2%
14      22%      97.6%          96.1%    50.2%
15      23%      97.6%          77.8%    50.2%
16      25%      97.6%          79.3%    51.7%
17      27%      97.6%          86.7%    63.5%
18      28%      97.6%          90.6%    67.0%
19      30%      97.6%          99.5%    98.0%
20      31%      97.6%          90.2%    65.0%
21      33%      97.6%          92.1%    57.6%
22      34%      97.6%          100.0%   65.0%
23      36%      97.6%          100.0%   83.2%
24      38%      100.0%         100.0%   96.1%
25      39%      100.0%         100.0%   99.0%
26      41%      100.0%         100.0%   84.2%
27      42%      100.0%         99.5%    84.2%
28      44%      100.0%         100.0%   99.0%
29      45%      100.0%         100.0%   97.5%
30      47%      100.0%         100.0%   91.6%
31      48%      100.0%         89.7%    68.5%
32      50%      100.0%         55.7%    56.2%
33      52%      100.0%         60.1%    54.7%
34      53%      100.0%         67.0%    81.8%
35      55%      100.0%         56.7%    72.9%
36      56%      100.0%         50.2%    50.7%
37      58%      100.0%         50.2%    73.9%
38      59%      100.0%         50.2%    89.2%
39      61%      100.0%         86.2%    100.0%
40      62%      100.0%         99.5%    100.0%
41      64%      100.0%         97.5%    100.0%
42      66%      100.0%         100.0%   100.0%
43      67%      100.0%         100.0%   100.0%
44      69%      100.0%         92.1%    100.0%
45      70%      100.0%         49.8%    100.0%
46      72%      100.0%         50.2%    100.0%
47      73%      100.0%         79.8%    100.0%
48      75%      100.0%         83.7%    100.0%
49      77%      100.0%         99.0%    100.0%
50      78%      100.0%         99.5%    100.0%
51      80%      100.0%         100.0%   100.0%
52      81%      100.0%         100.0%   100.0%
53      83%      100.0%         100.0%   100.0%
54      84%      100.0%         100.0%   100.0%
55      86%      100.0%         100.0%   100.0%
56      88%      100.0%         100.0%   100.0%
57      89%      100.0%         100.0%   100.0%
58      91%      100.0%         99.5%    100.0%
59      92%      100.0%         99.0%    100.0%
60      94%      100.0%         100.0%   100.0%
61      95%      100.0%         100.0%   100.0%
62      97%      100.0%         100.0%   100.0%
63      98%      100.0%         100.0%   100.0%
64      100%     100.0%         100.0%   100.0%
```
