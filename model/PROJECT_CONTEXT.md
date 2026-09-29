# PROJECT_CONTEXT.md: Research Foundation & UI Handoff Guide

> **Project Title**: The Thinking Budget: Cost-Accuracy Pareto Frontiers in Reasoning  
> **Target Audience**: UI / Frontend Engineering Team, Full-Stack Developers, and External Collaborators  
> **Status**: Final Post-5M Convergence Recovery Handoff  
> **Authoritative Experimental Conditions**: 126K Baseline, 1M Scaling, 5M Original (Confounded), 5M Recovery (Authoritative)  

---

## 1. Executive Summary

### 1.1 Project Mission & Core Question
This research project investigates a foundational question in machine learning reasoning: **When allocating additional test-time computation to solve an algorithmic multi-step problem, how should that compute be structured?**

Specifically, it contrasts two distinct paradigms of test-time thinking:
1. **Autoregressive Scratchpads (Chain-of-Thought / CoT)**: Test-time compute is spent sequentially generating explicit, discrete symbolic reasoning tokens before emitting a final answer.
2. **Recurrent Latent Reasoning**: Test-time compute is spent iteratively updating continuous internal hidden representations through recurrent state transitions without emitting external verbal tokens.

### 1.2 Evolution of the Research Claim
- **Original Hypothesis [HISTORICAL]**:
  > *"Spending test-time compute on recurrent latent iterations instead of generated reasoning tokens improves verified accuracy per dollar on pattern-induction tasks — but the advantage vanishes on tasks that need an explicit symbolic scratchpad."*
- **Why the Original Claim Was Revised**:
  1. *Cost Metric*: Monetary cost was never measured; compute was profiled strictly as isolated GPU wall-clock latency (ms) and analytical floating-point operations (FLOPs).
  2. *Task Characterization*: Task A is not vague "pattern induction," but rather **permutation orbit traversal / iterative algorithmic state-transition computation**.
  3. *Symbolic Scratchpad Claim*: Task B (modular arithmetic) proved challenging for both models, yielding near-chance accuracies ($10\%–18.6\%$). Consequently, the data did not provide clean empirical support for a symbolic scratchpad advantage.
  4. *Scale Scaling*: Across three parameter scales (~126K, ~1M, and ~5M), the empirical trade-off between paradigms fundamentally shifted.
- **Final Evidence-Matched Central Conclusion [MEASURED & VERIFIED]**:
  > *"Across the tested 126K, 1M, and 5M configurations, recurrent latent computation was competitive at very low budgets but remained near a ~33% Task-A accuracy plateau, while autoregressive scratchpad generation continued to improve with additional computation and model capacity, reaching 99.74% at 5M under the controlled recovery training condition."*

### 1.3 Epistemological Classification Matrix

| Category | Claim / Finding | Evidence Status |
| :--- | :--- | :--- |
| **Hypothesized** | Recurrent latent reasoning will yield a universally superior Pareto frontier on iterative pattern tasks. | **Falsified** across mid-to-high accuracy targets. |
| **Hypothesized** | Autoregressive scratchpads will clearly outperform latent models on modular register arithmetic. | **Inconclusive** (both models performed near chance). |
| **Measured** | 5M CoT achieves $99.74\% \pm 0.48\%$ on Task A at $k=16$ under controlled optimization. | **[MEASURED]** (paired $t = 188.91, p = 4.71 \times 10^{-9}$). |
| **Measured** | 5M Latent stabilizes at $33.92\% \pm 0.84\%$ on Task A at $k=16$, matching 126K/1M plateaus. | **[MEASURED]** ($33.06\%$ at 126K, $31.12\%$ at 1M, $33.92\%$ at 5M). |
| **Supported** | The initial 5M scaling failure was primarily caused by optimization underfitting, not architectural collapse. | **[STRONGLY SUPPORTED]** (resolved by 50-epoch warmup/decay schedule). |
| **Plausible Hypothesis** | The latent model's plateau is related to dynamic memory-addressing failure in cross-attention. | **[PLAUSIBLE HYPOTHESIS]** (observed attention bias, but causal link unproven). |
| **Unjustified Claim** | Latent reasoning is mathematically proven to be fundamentally incapable of solving permutation orbits. | **[REJECTED / UNJUSTIFIED]** (applies only to tested architecture and setup). |

---

## 2. Why This Project Exists

### 2.1 The "Thinking Budget" Problem
In modern reasoning systems (e.g., OpenAI o1/o3, DeepSeek R1), models "think" before answering. Standard practice relies on autoregressive generation of natural language or code scratchpads. While effective, token generation incurs significant hardware costs:
- Every generated token requires an autoregressive forward pass through all model layers.
- Key-value (KV) caches grow linearly with sequence length ($O(L)$ memory, $O(L^2)$ attention).
- Memory bandwidth bottlenecks (memory-bound decoding) often dominate GPU execution time.

An alternative hypothesis asks: *Can a model think purely in continuous latent space?* If a compact recurrent cell can update a continuous state vector across $k$ iterations, test-time computation can scale without generating long token sequences or expanding KV caches.

### 2.2 The Empirical Pareto Frontier
To evaluate this fairly, the project measures the **empirical non-dominated Pareto frontier** between:
- **Compute Cost**: GPU wall-clock latency (ms) per sample and analytical FLOPs.
- **Reasoning Accuracy**: Percentage of correct categorical answers on deterministic test suites.

An operating point $(C_1, A_1)$ dominates $(C_2, A_2)$ if it uses less or equal compute and achieves greater or equal accuracy (with at least one strict inequality). The Pareto frontier consists strictly of **observed, non-dominated operating points**. No convexity assumptions, smoothing, or theoretical interpolations are applied.

---

## 3. Experimental Design

### 3.1 Model Architectures Under Comparison
1. **`AutoregressiveCoT`**: Causal Transformer decoder generating discrete reasoning scratchpads.
2. **`RecurrentLatentReasoner`**: Recurrent state-transition network combining key-value cross-attention, GRUCell, and residual MLP.

### 3.2 Strict Capacity Matching Protocol
To ensure that performance differences reflect architectural inductive biases rather than parameter capacity discrepancies, model parameters were programmatically matched within $\pm 5\%$ at every scale (and within $\pm 0.0003\%$ at 1M and 5M):

| Scale | Model | Dimension ($d_{\text{model}}$) | Layers / Recurrence | Feedforward Dim ($d_{\text{ff}}$) | Exact Total Parameters | Parameter Difference |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **126K** | `AutoregressiveCoT` | 80 | 2 layers | 192 | **126,168** | Baseline |
| **126K** | `RecurrentLatentReasoner` | 80 | GRUCell | 304 | **125,688** | -480 (-0.3804%) |
| **1M** | `AutoregressiveCoT` | 224 | 2 layers | 536 | **998,520** | Baseline |
| **1M** | `RecurrentLatentReasoner` | 224 | GRUCell | 852 | **998,523** | +3 (+0.00030%) |
| **5M** | `AutoregressiveCoT` | 544 | 2 layers | 1,168 | **5,000,632** | Baseline |
| **5M** | `RecurrentLatentReasoner` | 544 | GRUCell | 1,795 | **5,000,635** | +3 (+0.00006%) |

*(Note: The repository also contains a standalone educational component, `HebbianMemory` with 20,672 parameters, which operates exclusively on vector associative recall and is not part of the CoT-vs-Latent benchmark.)*

### 3.3 Four Experimental Conditions
1. **126K Baseline** (`results/`): Initial proof of concept; 20 models trained across 5 seeds.
2. **1M Scaling** (`results_1m/`): 8x parameter increase; 20 models trained across 5 seeds.
3. **5M Original [CONFOUNDED]** (`results_5m/`): 40x parameter scale from baseline; underfit due to fixed static training schedule; retained as a historical control.
4. **5M Recovery [AUTHORITATIVE 5M]** (`results_5m_recovery/`): Same 5M architectures, same tasks, same seeds, but trained with a controlled 50-epoch warmup and cosine decay schedule that resolved underfitting.

---

## 4. Algorithmic Tasks

### 4.1 Task A: Permutation Orbit Traversal (State-Transition Computation)
- **Domain**: Symmetric group $S_8$ over elements $\mathcal{S} = \{0, 1, \dots, 7\}$.
- **Permutation $\pi$**: A randomly sampled bijective mapping $\pi: \mathcal{S} \to \mathcal{S}$.
- **Initial State $v_0$**: A starting element $v_0 \in \mathcal{S}$.
- **Difficulty Depth $D$**: Transition length $D \sim \text{Uniform}(2, 16)$.
- **Mathematical Transition**:
  $$v_{t} = \pi(v_{t-1}), \quad t \in \{1, \dots, D\}$$
- **Target Answer**: $v_D = \pi^D(v_0) \in \{0, \dots, 7\}$.
- **Chance Baseline**: Uniform guess across 8 classes = **12.50%**.
- **Nature of the Task**: Iterative state tracking. Solving depth $D$ requires executing $D$ sequential lookups through the permutation table.

### 4.2 Task B: Multi-Step Modular Register Arithmetic (Symbolic State Tracking)
- **Domain**: Modular ring $\mathbb{Z}_{10} = \{0, 1, \dots, 9\}$.
- **Initial State $r_0$**: Starting register value $r_0 \in \{0, \dots, 9\}$.
- **Difficulty Depth $D$**: Sequence length $D \sim \text{Uniform}(2, 16)$.
- **Operations**: At each step $t$, an operator $\text{op}_t \in \{\text{OP\_ADD}, \text{OP\_MUL}, \text{OP\_SUB}\}$ and constant $c_t \in \{1, \dots, 9\}$ are applied modulo 10.
- **Target Answer**: Final register state $r_D \in \{0, \dots, 9\}$.
- **Chance Baseline**: Uniform guess across 10 classes = **10.00%**.
- **Empirical Status [MEASURED]**: Both models achieved low absolute accuracies ($10\%–18.6\%$) across all scales. **Task B is inconclusive** and must not be cited as proving a symbolic scratchpad advantage.

---

## 5. Token and Trace Formats

The vocabulary is shared across both models and tasks ($V = 24$ tokens):
- Digits `0` through `9` (indices 0–9).
- Operators: `OP_ADD` (10), `OP_MUL` (11), `OP_SUB` (12).
- Structural Markers: `MAP` (13), `START` (14), `HOPS` (15), `THINK` (16), `END_THINK` (17), `ANS` (18), `PAD` (19), `EOS` (20).

```
Task A Input Format (Length = 13 tokens):
[MAP,  pi(0), pi(1), pi(2), pi(3), pi(4), pi(5), pi(6), pi(7),  START,  v0,  HOPS,  D]
  0      1      2      3      4      5      6      7      8       9     10    11   12

Task A Reasoning Trace (Variable Length = D + 3 tokens):
[THINK,  v1,  v2,  ...,  vD,  END_THINK,  ANS,  vD]

Task B Input Format (Variable Length = 2 + 2D tokens):
[START,  r0,  OP_1, c1,  OP_2, c2,  ...,  OP_D, cD]

Task B Reasoning Trace (Variable Length = D + 3 tokens):
[THINK,  r1,  r2,  ...,  rD,  END_THINK,  ANS,  rD]
```

### Aligned Intermediate Supervision
During training, both architectures receive intermediate supervision:
- **CoT**: Standard next-token teacher forcing across the entire sequence. The loss penalizes errors at every intermediate scratchpad step ($v_1, \dots, v_D$) as well as the final answer token.
- **Latent**: For batch instance $i$, the recurrent cell runs up to $D_i$ steps. At each intermediate recurrent step $t$, the continuous hidden state $\mathbf{h}_t$ is projected through a categorical classification head ($\mathbf{W}_{\text{head}} \mathbf{h}_t + \mathbf{b}$) and supervised against ground-truth state $v_t$ via cross-entropy.

---

## 6. Model Architectures & Inference Semantics

### 6.1 `AutoregressiveCoT` (Causal Transformer)
- **Structure**: Learned token embeddings + learned positional embeddings $\to$ 2-layer Transformer encoder with causal masking $\to$ LayerNorm $\to$ Linear classification head.
- **Inference Budget $k$**: In CoT, $k$ represents the **maximum token limit** allocated for thinking.
- **Execution Flow**:
  1. The model takes the prompt and appends `THINK`.
  2. For $s = 1, \dots, k$, it generates tokens greedily.
  3. If the model emits `END_THINK`, scratchpad generation halts early ($t < k$).
  4. The model appends `ANS` and predicts the final categorical answer token.
- **Early Termination**: CoT can consume fewer FLOPs/latency than budget $k$ if it terminates early.

### 6.2 `RecurrentLatentReasoner` (Cross-Attention + GRU)
- **Structure**:
  1. *Prompt Memory Encoding*: Input tokens are embedded into key-value memory $\mathbf{M} \in \mathbb{R}^{L_{\text{in}} \times d}$.
  2. *Initial State*: Hidden state $\mathbf{h}_0$ is initialized from the terminal prompt embedding.
  3. *Recurrent Loop*: For $t = 1, \dots, k$:
     - Multi-Head Cross-Attention: Query is $\mathbf{h}_{t-1}$; Keys/Values are $\mathbf{M}$.
     - Recurrent Update: $\mathbf{h}_t' = \text{LayerNorm}(\text{GRUCell}(\text{context}_t, \mathbf{h}_{t-1}))$.
     - Residual FFN: $\mathbf{h}_t = \text{LayerNorm}(\mathbf{h}_t' + \text{MLP}(\mathbf{h}_t'))$.
  4. *Answer Emission*: Final state $\mathbf{h}_k$ is projected to categorical logits.
- **Inference Budget $k$**: In Latent, $k$ is the **exact number of recurrent updates**.
- **No Adaptive Halting [CRITICAL]**: The primary model has NO early-stopping or Adaptive Computation Time (ACT) mechanism. It always executes exactly $k$ iterations.
- **Decoupling of $D$ and $k$**: Difficulty depth $D$ belongs to the problem instance ($D \in [2, 16]$); budget $k \in \{1, 2, 4, 8, 12, 16\}$ is an external constraint chosen at test time.

---

## 7. Training Protocols

### 7.1 Shared Protocol Constants
- **Dataset Size**: 4,000 synthetic problem instances per seed.
- **Seeds**: 5 independent training seeds ($42, 43, 44, 45, 46$).
- **Batch Size**: 32.
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $0.01$).
- **Difficulty Range**: $D \sim \text{Uniform}(2, 16)$.

### 7.2 Protocol Comparison by Experimental Condition

| Parameter | 126K Baseline | 1M Scaling | 5M Original [CONFOUNDED] | 5M Recovery [AUTHORITATIVE] |
| :--- | :--- | :--- | :--- | :--- |
| **Epochs / Steps** | 20 ep / 2,500 steps | 20 ep / 2,500 steps | 20 ep / 2,500 steps | **50 ep / 6,250 steps** |
| **Peak Learning Rate ($\eta$)** | $1 \times 10^{-3}$ | $1 \times 10^{-3}$ | $1 \times 10^{-3}$ | **$5 \times 10^{-4}$** |
| **Schedule** | Static (no schedule) | Static (no schedule) | Static (no schedule) | **5-ep warmup $\to$ cosine decay to $10^{-5}$** |
| **Gradient Clipping** | $\Vert \mathbf{g} \Vert_2 \le 1.0$ | $\Vert \mathbf{g} \Vert_2 \le 1.0$ | $\Vert \mathbf{g} \Vert_2 \le 1.0$ | **$\Vert \mathbf{g} \Vert_2 \le 1.0$ (telemetry logged)** |
| **Optimization State** | Fully converged | Fully converged | **Severe underfitting** | **Fully converged** |

*(Note on Learning Rate: The peak learning rate of $5 \times 10^{-4}$ was an empirical adjustment for wider transformer layers; it is not a mathematically proven scaling constant).*

---

## 8. Evaluation Framework & Compute Metrics

### 8.1 Test Dataset & Dimensionality
- **Test Dataset**: 1,000 fixed, deterministic instances per task generated with `TEST_SEED = 999`.
- **Budgets ($6$)**: $k \in \{1, 2, 4, 8, 12, 16\}$.
- **Independent Seeds ($5$)**: $42, 43, 44, 45, 46$.
- **Models ($2$)** $\times$ **Tasks ($2$)** $\times$ **Seeds ($5$)** $\times$ **Budgets ($6$)** = **120 evaluation records per suite**.

### 8.2 Compute Accounting
1. **Primary Metric: Isolated GPU Wall-Clock Latency (ms)**:
   - Measured on an NVIDIA GeForce RTX 4060 Laptop GPU using PyTorch CUDA events.
   - 5-batch warmup before recording timing events.
   - Explicit `torch.cuda.synchronize()` before and after execution.
   - **NO "Cost Per Dollar"**: Claims of financial cost or cloud billing rates were not measured and are scientifically invalid in this repository.
2. **Secondary Metric: Analytical FLOPs**:
   - Multi-scale multiply-accumulate calculations ($2 \times \text{MACs}$).
   - 126K: CoT $\approx 229\text{K}$ FLOPs/token; Latent $\approx 225\text{K}$ FLOPs/step.
   - 1M: CoT $\approx 1.77\text{M}$ FLOPs/token; Latent $\approx 1.77\text{M}$ FLOPs/step.
   - 5M: CoT $\approx 9.84\text{M}$ FLOPs/token; Latent $\approx 9.82\text{M}$ FLOPs/step.

### 8.3 Statistical Testing Boundaries
- Statistical tests are **paired two-tailed $t$-tests** computed across the $n=5$ training seeds.
- The 1,000 test items are repeated evaluation queries, not independent training replicates.
- **Same-$k$ paired tests compare models at the same nominal budget $k$**. Because CoT and Latent run at different wall-clock speeds at the same $k$, **same-$k$ $p$-values do NOT establish latency-matched significance**.

---

## 9. Baseline Results: ~126K Parameters

### 9.1 Task A Benchmark Summary (Mean Accuracy $\pm$ SEM across 5 seeds)

| $k$ | CoT Mean Acc | CoT SEM | Latent Mean Acc | Latent SEM | CoT Latency | Latent Latency | Paired $p$-value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 21.54% | ±2.43% | 15.68% | ±0.73% | 0.150 ms | 0.080 ms | 0.0760 |
| **2** | 29.70% | ±2.89% | 32.30% | ±0.52% | 0.190 ms | 0.130 ms | 0.4470 |
| **4** | 38.64% | ±4.92% | 33.14% | ±0.56% | 0.260 ms | 0.230 ms | 0.3540 |
| **8** | 50.84% | ±7.60% | 33.08% | ±0.57% | 0.420 ms | 0.440 ms | 0.0890 |
| **12**| 63.18% | ±10.38% | 33.04% | ±0.65% | 0.570 ms | 0.640 ms | 0.0380 |
| **16**| **73.36%** | **±15.83%** | **33.06%** | **±0.59%** | 0.720 ms | 0.850 ms | **0.0047** |

### 9.2 Key Baseline Dynamics
1. **Low-Budget Regime ($k \le 2$)**: Latent reasoning is highly competitive, outperforming CoT at $k=2$ ($32.30\%$ vs $29.70\%$) with lower latency ($0.13\text{ ms}$ vs $0.19\text{ ms}$).
2. **Latent Saturation ($k \ge 4$)**: As budget increases, the latent model plateaus near $\sim 33\%$ ($33.14\% \to 33.06\%$).
3. **CoT Monotonic Scaling**: Autoregressive CoT improves continuously from $21.54\%$ to $73.36\%$.
4. **Seed 46 Variance**: Seed 46 in CoT reached $99.10\%$ at $k=16$, whereas seeds 42–45 achieved $58.9\%–76.2\%$, showing high initialization sensitivity at 126K.

---

## 10. Capacity Scaling Results: ~1M Parameters

### 10.1 Task A Benchmark Summary ($d=224$)

| $k$ | CoT Mean Acc | CoT SEM | Latent Mean Acc | Latent SEM | CoT Latency | Latent Latency | Paired $p$-value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 24.32% | ±7.48% | 15.36% | ±0.70% | 0.165 ms | 0.081 ms | 0.2870 |
| **2** | 33.72% | ±0.56% | 30.68% | ±1.06% | 0.218 ms | 0.134 ms | 0.0710 |
| **4** | 41.52% | ±5.88% | 31.14% | ±1.02% | 0.323 ms | 0.244 ms | 0.1610 |
| **8** | 53.64% | ±13.78% | 31.12% | ±1.01% | 0.536 ms | 0.457 ms | 0.1980 |
| **12**| 64.36% | ±23.30% | 31.12% | ±1.01% | 0.751 ms | 0.672 ms | 0.2480 |
| **16**| **76.70%** | **±29.57%** | **31.12%** | **±1.01%** | 0.965 ms | 0.887 ms | **0.0261** |

### 10.2 Bimodal Seed Variance at 1M
The 1M CoT model showed severe bimodal behavior across seeds at $k=16$:
- Seed 42: **24.1%** (failed to find good solution branch).
- Seed 43: **89.3%**
- Seed 44: **85.0%**
- Seed 45: **92.2%**
- Seed 46: **92.9%**
Despite increasing parameters 8-fold, the latent model remained clamped at **$31.12\%$**, confirming that 1M capacity alone did not break the latent plateau.

---

## 11. Original 5M Scaling: The Optimization Confound

In the first 5M scaling experiment (`results_5m/RESULTS_5M.md`), models were expanded to 5,000,632 parameters ($d=544$), but trained under the original 20-epoch static recipe:
- **Task A CoT Accuracy**: Stalled at $\sim 13\%–14\%$ across all budgets $k$ (chance = 12.5%).
- **Task A Latent Accuracy**: Stalled at $\mathbf{12.46\%}$ for all $k \ge 2$.
- **Training Loss**: CoT loss stalled at $1.35–1.57$ (compared to $0.01$ at 126K); Latent loss stalled at $\ln(8) \approx 2.079$ (chance entropy).

### Scientific Classification: [HISTORICAL / CONFOUNDED]
This run was initially suspected to show that 5M capacity broke reasoning. A post-hoc convergence audit proved this was an **optimization failure (severe underfitting)**: 2,500 optimizer steps with static AdamW at $d=544$ was simply insufficient compute to navigate the loss landscape.

---

## 12. 5M Convergence Recovery: The Authoritative 5M Benchmark

The **5M Convergence Recovery Experiment** applied a controlled training schedule (50 epochs / 6,250 steps, peak lr $5\times 10^{-4}$, 5-epoch warmup, cosine decay).

### 12.1 Authoritative Task A Results ($d=544$)

| $k$ | CoT Mean Acc | CoT SEM | Latent Mean Acc | Latent SEM | CoT Latency | Latent Latency | Paired Diff $\Delta$ | $p$-value (paired $t$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 22.48% | ±3.32% | 15.46% | ±0.96% | 0.136 ms | 0.120 ms | +7.02% | 0.0449 |
| **2** | 33.94% | ±0.14% | 31.44% | ±0.45% | 0.179 ms | 0.079 ms | +2.50% | 0.0044 |
| **4** | 44.54% | ±0.17% | 33.38% | ±0.13% | 0.289 ms | 0.140 ms | +11.16% | $4.62 \times 10^{-7}$ |
| **8** | 63.34% | ±0.14% | 33.94% | ±0.37% | 0.549 ms | 0.279 ms | +29.40% | $1.64 \times 10^{-7}$ |
| **12**| 80.26% | ±0.22% | 33.98% | ±0.37% | 0.849 ms | 0.386 ms | +46.28% | $1.83 \times 10^{-8}$ |
| **16**| **99.74%** | **±0.21%** | **33.92%** | **±0.37%** | 1.168 ms | 0.512 ms | **+65.82%** | **$4.71 \times 10^{-9}$** |

### 12.2 Per-Seed Breakdown at $k=16$ (5M Recovery)
- **CoT**: Seed 42 = 100.0%, Seed 43 = 98.9%, Seed 44 = 100.0%, Seed 45 = 99.8%, Seed 46 = 100.0% (Mean: **$99.74\% \pm 0.48\%$**).
- **Latent**: Seed 42 = 35.3%, Seed 43 = 33.3%, Seed 44 = 33.3%, Seed 45 = 34.1%, Seed 46 = 33.6% (Mean: **$33.92\% \pm 0.84\%$**).

### 12.3 Training Loss & Gradient Telemetry
- **Final Losses**: CoT dropped to **$0.6069 \pm 0.0057$**; Latent dropped to **$1.8890 \pm 0.0129$** (well below chance).
- **Gradient Clipping**: CoT had ~30.3% clipped batches (mean norm 0.84); Latent had ~99.5% clipped batches (mean norm 3.14, max 49.27). Chronic clipping in Latent is an empirical observation under this setup, not proof that BPTT inherently fails.

### 12.4 Four-Scale Task A Trajectory at $k=16$

```
CoT:    126K (73.36%)  ───>  1M (76.70%)  ───>  5M Recovery (99.74%)  [SCALES MONOTONICALLY]
Latent: 126K (33.06%)  ───>  1M (31.12%)  ───>  5M Recovery (33.92%)  [STABLE PLATEAU NEAR ~33%]
```

---

## 13. Task B Results Across Scales (Modular Register Arithmetic)

| Scale | Model | $k=1$ Mean Acc | $k=2$ Mean Acc | $k=4$ Mean Acc | $k=8$ Mean Acc | $k=12$ Mean Acc | $k=16$ Mean Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **126K** | CoT | 10.64% | 10.30% | 11.20% | 11.78% | 12.18% | 12.80% |
| **126K** | Latent | 10.60% | 16.48% | 13.98% | 13.48% | 13.06% | 12.98% |
| **1M** | CoT | 10.74% | 11.82% | 12.28% | 12.78% | 12.78% | 14.18% |
| **1M** | Latent | 10.70% | 15.68% | 13.78% | 13.88% | 13.84% | 13.84% |
| **5M Rec** | CoT | 11.98% | 12.16% | 12.70% | 12.84% | 12.76% | 18.58% |
| **5M Rec** | Latent | 10.68% | 15.34% | 13.40% | 14.48% | 14.50% | 14.56% |

**Scientific Interpretation**: Accuracies remain bounded between $10\%$ and $18.6\%$ (random baseline = $10.0\%$). While Latent exhibits a slight edge at low budgets ($k=2$) and CoT reaches $18.58\%$ at 5M $k=16$, neither model mastered multi-step modular arithmetic. **Task B is inconclusive.**

---

## 14. Mechanistic Audit: Why Does the Latent Model Plateau?

A diagnostic probe into the intermediate latent states of `RecurrentLatentReasoner` revealed key empirical patterns:

1. **Numerical Near-Stationarity**: Hidden vector updates $\Vert \mathbf{h}_t - \mathbf{h}_{t-1} \Vert_2$ contract rapidly after steps $1–2$.
2. **High State Similarity**: Cosine similarity between late hidden vectors approaches $\approx 0.99$, indicating the representation enters a near-static trajectory.
3. **Decodability Drop**: Projecting intermediate states $\mathbf{h}_t$ through the output head shows that predictions fail to track the step-by-step intermediate ground truth ($v_1, \dots, v_D$).
4. **Initial Token Attention Bias**: In Task A, the input prompt is 13 tokens long. The permutation table occupies positions 1–8, while the initial node $v_0$ is at position 10. Attention heatmaps show that after step 1, cross-attention weights concentrate heavily on position 10 ($v_0$) rather than dynamically querying position $1 + v_{t-1}$.
5. **Prediction Bias**: Erroneous final predictions disproportionately predict $v_0$, consistent with attention remaining focused on the initial token.

### Hypothesis vs Measured Fact
- **[OBSERVED]**: The model exhibits late-step hidden state update contraction, early error concentration, and persistent attention to the initial value token.
- **[PLAUSIBLE HYPOTHESIS]**: The cross-attention mechanism struggles to execute dynamic indirect addressing (pointer hopping) in continuous space without discrete token boundaries.
- **[NOT DETERMINED]**: We have not causally proven whether the primary bottleneck is GRU gating, cross-attention addressing, or LayerNorm normalization.

---

## 15. What This Project Does NOT Establish

To maintain strict scientific integrity, the UI and any accompanying presentation MUST NOT claim:
1. **No Universal CoT Superiority**: CoT's advantage is demonstrated specifically on these small algorithmic benchmarks under these training configurations.
2. **No Fundamental Architectural Infeasibility of Latent Reasoning**: The ~33% ceiling is an empirical plateau for *this specific cross-attention + GRUCell architecture*; it does not prove that all latent reasoning models (e.g. models with ACT, slot memories, or different recurrent cells) are doomed to fail.
3. **No Financial Cost Claims**: "Accuracy per dollar" was not measured. All compute accounting is strictly milliseconds of GPU latency and analytical FLOPs.
4. **No Latency-Matched Significance**: Same-$k$ paired $t$-tests compare models at the same step budget $k$, not at identical hardware runtimes.
5. **No Validation of Symbolic Scratchpads via Task B**: Task B performance is near chance and statistically inconclusive.
6. **No Proven Causal Failure Mechanism**: Contraction and attention bias are measured correlations, not experimentally proven causal failure mechanisms.
7. **No Generalization to Large Language Models (LLMs)**: These models are small (126K–5M parameters) trained from scratch on synthetic data. They do not imply scaling laws for billion-parameter foundation models.

---

## 16. Simulation Lab Architecture & Developer API

The Simulation Lab (`simulation_lab/`) provides a clean, unified Python backend designed to support interactive dashboards and developer tools.

### 16.1 Model Registry & Adapter IDs
The Simulation Lab registers 7 distinct model configurations:

| Model Registry ID | Aliases | Class | Parameters | Checkpoint Source |
| :--- | :--- | :--- | :--- | :--- |
| `autoregressive_cot` | `cot`, `transformer` | `CoTModelAdapter` | 126,168 | `export/models/cot_*` |
| `recurrent_latent` | `latent`, `gru` | `LatentModelAdapter` | 125,688 | `export/models/latent_*` |
| `hebbian_synaptic` | `hebbian`, `bdh` | `HebbianModelAdapter` | 20,672 | In-memory initialization |
| `autoregressive_cot_1m` | `cot_1m` | `CoTModelAdapter1M` | 998,520 | `export/models_1m/cot_*` |
| `recurrent_latent_1m` | `latent_1m` | `LatentModelAdapter1M` | 998,523 | `export/models_1m/latent_*` |
| `autoregressive_cot_5m` | `cot_5m`, `cot_5m_recovery`, `autoregressive_cot_5m_recovery` | `CoTModelAdapter5M` | 5,000,632 | **`export/models_5m_recovery/cot_*`** |
| `recurrent_latent_5m` | `latent_5m`, `latent_5m_recovery`, `recurrent_latent_5m_recovery` | `LatentModelAdapter5M` | 5,000,635 | **`export/models_5m_recovery/latent_*`** |

> [!CRITICAL]
> The 5M adapters (`autoregressive_cot_5m` and `recurrent_latent_5m`) strictly load from **`export/models_5m_recovery/`**. They do NOT load from `export/models_5m/`.

### 16.2 Core Python Interface (`simulation_lab.SimulationLabEngine`)

```python
from simulation_lab import SimulationLabEngine

# 1. Initialize engine (auto-selects CUDA if available)
engine = SimulationLabEngine()

# 2. Inspect registered models and tasks
models = engine.list_models()
tasks = engine.list_tasks()

# 3. Generate a standalone problem instance
example = engine.generate_example(task="permutation_orbit", depth=6, seed=42)

# 4. Run an interactive reasoning simulation
result = engine.run_simulation(
    model="autoregressive_cot_5m",  # or alias 'cot_5m'
    task="permutation_orbit",
    budget_k=8,
    seed=42,
    example=example,
    measure_timing=True
)

print(f"Prediction: {result.predicted_answer_str} (Correct: {result.correct})")
print(f"GPU Latency: {result.latency_ms:.3f} ms | FLOPs: {result.analytical_flops:,}")
print(f"Trace Steps: {len(result.trace['steps'])}")

# 5. Multi-Budget Sweep on the same problem instance
sweep = engine.sweep_budgets(
    model="cot_5m",
    task="permutation_orbit",
    example=example,
    budgets=[1, 2, 4, 8, 12, 16]
)

# 6. Side-by-side comparison
comparison = engine.compare_models(
    models=["cot_5m", "latent_5m"],
    task="permutation_orbit",
    budget_k=8,
    example=example
)

# 7. Precomputed Scale-Aware Pareto Retrieval
pareto_126k = engine.get_pareto_data("permutation_orbit", scale="126k")
pareto_5m = engine.get_pareto_data("permutation_orbit", scale="5m")  # reads results_5m_recovery/
```

### 16.3 Educational Reasoning Traces (`result.trace`)
- **CoT Trace (`paradigm: "autoregressive_cot"`)**: Exposes step-by-step emitted tokens (`reasoning_step`, `think_end`), token IDs, strings, and early-termination status.
- **Latent Trace (`paradigm: "recurrent_latent"`)**: Exposes iteration-by-iteration continuous state dynamics: hidden norm ($\Vert \mathbf{h}_t \Vert_2$), mean, standard deviation, readout candidate token, and top-3 candidate logits.

---

## 17. Simulation Lab vs. Offline Benchmark Results

| Dimension | Offline Benchmark (`results*/`) | Simulation Lab (`simulation_lab/`) |
| :--- | :--- | :--- |
| **Purpose** | Authoritative empirical research ground truth. | Interactive demonstrator and visualization backend. |
| **Execution** | Precomputed batch evaluation across 1,000 test cases $\times$ 5 seeds $\times$ 6 budgets. | Single-example on-demand model forward pass. |
| **Output** | Aggregated statistical tables, SEMs, $p$-values, Pareto frontiers. | Single inference prediction, execution trace, per-call latency. |
| **Data Source** | Frozen JSON files (`aggregated_results.json`, etc.). | Checkpoint weights loaded dynamically into PyTorch modules. |

**Rule for UI Developers**: Do not conflate the single-example simulation with aggregate research statistics. When showing overall accuracy or Pareto curves, display the precomputed benchmark data from `engine.get_pareto_data()`. When showing step-by-step reasoning or interactive sliders, run `engine.run_simulation()`.

---

## 18. Instructions for the UI Team

### YOU MAY:
- Build modern web interfaces (e.g. Next.js, React, Tailwind, D3).
- Display model selectors for 126K, 1M, and 5M Recovery.
- Plot interactive Pareto frontiers (Accuracy vs. Latency ms or FLOPs) using `engine.get_pareto_data()`.
- Create step-by-step token visualizers for CoT scratchpads.
- Create hidden state metric charts (norm, candidate readout logits) for Latent transitions.
- Build side-by-side comparisons of CoT vs. Latent on individual generated examples.
- Clearly present the distinction between the historical confounded 5M run and the 5M recovery run.

### YOU MUST NOT:
- Retrain any models or execute training scripts.
- Overwrite or modify checkpoints (`export/models*`).
- Modify benchmark JSON files (`results*/*.json`).
- Silently route 5M queries to `export/models_5m/` instead of `export/models_5m_recovery/`.
- Claim "cost per dollar" or attach arbitrary monetary dollar values to compute metrics.
- State that the latent model has an "architectural impossibility" or is "proven to fail".
- Claim that Task B proves a symbolic scratchpad advantage.
- Alter the core model implementations in `models_*.py` or `data_generator.py`.

---

## 19. Artifact & Repository Directory Map

```
e:/pro/pro/
├── export/
│   ├── models/                  # [HISTORICAL FROZEN] 126K baseline checkpoints (20 runs + Hebbian)
│   ├── models_1m/               # [HISTORICAL FROZEN] 1M scaling checkpoints (20 runs)
│   ├── models_5m/               # [HISTORICAL CONFOUNDED] Original 5M underfit checkpoints
│   └── models_5m_recovery/      # [AUTHORITATIVE 5M] 5M Convergence Recovery checkpoints (20 runs)
│
├── results/                     # [HISTORICAL FROZEN] 126K benchmark JSONs, plots, and RESULTS.md
├── results_1m/                  # [HISTORICAL FROZEN] 1M benchmark JSONs, plots, and RESULTS_1M.md
├── results_5m/                  # [HISTORICAL CONFOUNDED] Original 5M benchmark JSONs and RESULTS_5M.md
├── results_5m_recovery/         # [AUTHORITATIVE 5M] 5M recovery benchmark JSONs and RESULTS_5M_RECOVERY.md
│
├── simulation_lab/              # Simulation Lab Python backend package
│   ├── __init__.py              # Package exports (adapters, engine, metrics)
│   ├── engine.py                # SimulationLabEngine (orchestration, sweeps, Pareto API)
│   ├── models.py                # Model adapters and ModelRegistry (loads models_5m_recovery)
│   ├── tasks.py                 # TaskExample generators and token string resolvers
│   ├── metrics.py               # Analytical FLOPs and CUDA latency profiling
│   └── schemas.py               # Trace and simulation result data schemas
│
├── docs/                        # Complete technical documentation suite
│   ├── 01-requirements.md       # Requirements and hypothesis specification
│   ├── 02-architecture.md       # Multi-scale architecture and component design
│   ├── 03-tech-stack.md         # Compute hardware, memory, and analytical FLOP accounting
│   ├── 06-development-plan.md   # Project phase plan (Phases 1–12 complete)
│   ├── 07-decisions.md          # Architecture Decision Records (ADR-001 to ADR-011)
│   ├── 08-changelog.md          # Chronological release log (through v1.4.0)
│   ├── 11-known-issues.md       # Defect registry and resolution records
│   ├── dataset.md               # Mathematical task definitions and token formats
│   ├── model-card.md            # Comprehensive model cards across 126K, 1M, and 5M
│   ├── ml-pipeline.md           # Training contracts and multi-seed orchestration
│   ├── evaluation.md            # Benchmark evaluation protocol and statistical methods
│   └── simulation-lab-api.md    # API contract and integration guide for UI developers
│
├── models_cot_model.py          # Frozen AutoregressiveCoT model definition
├── models_latent_model.py       # Frozen RecurrentLatentReasoner model definition
├── models_hebbian_model.py      # Frozen HebbianMemory model definition
├── data_generator.py            # Frozen synthetic data generator
│
├── test_research_pipeline.py    # Baseline pipeline unit tests (7 tests)
├── test_scaling_1m.py           # 1M scaling unit tests (15 tests)
├── test_scaling_5m.py           # 5M scaling unit tests (12 tests)
└── test_simulation_lab.py       # Simulation Lab unit and integration tests (26 tests)
```

---

## 20. Verified Test & Integrity Status

All tests pass cleanly across the active test suite:
```
pytest test_research_pipeline.py test_scaling_1m.py test_scaling_5m.py test_simulation_lab.py
============================= 60 passed in 4.78s ==============================
```
- `test_research_pipeline.py`: **7 passed**
- `test_scaling_1m.py`: **15 passed**
- `test_scaling_5m.py`: **12 passed**
- `test_simulation_lab.py`: **26 passed** (including verified loading of 5M recovery checkpoints and isolation from `export/models_5m/`)

---

## 21. Final One-Paragraph Summary for UI Developers

This project is a controlled empirical study comparing two fundamental ways of allocating test-time compute on multi-step algorithmic problems: generating explicit autoregressive scratchpad tokens (Chain-of-Thought) versus executing recurrent continuous state transitions (Recurrent Latent Reasoning). Across three rigorously capacity-matched scales (126K, 1M, and 5M parameters), recurrent latent computation was highly competitive at very low step budgets but consistently saturated near a ~33% accuracy ceiling on permutation orbit traversal, while autoregressive scratchpads continued to scale with capacity and compute, reaching 99.74% accuracy at 5M under a controlled recovery training schedule. The Simulation Lab backend provides a fully verified, multi-scale Python API that exposes live model inference, step-by-step reasoning traces, isolated GPU latency profiling, analytical FLOP calculations, and precomputed Pareto frontiers across all four experimental conditions without requiring any modifications to the underlying ML research artifacts.

---

## Repository Consistency Notes

- **5M Checkpoint Alignment**: Verified that `simulation_lab/models.py` strictly defaults to `export/models_5m_recovery/` for all 5M operations. The historical `export/models_5m/` baseline remains isolated and preserved.
- **Statistical File Schema**: Handled schema variation between `statistical_comparison.json` (list-based, used in 126K, 1M, 5M original) and `statistical_tests.json` (dict-based, used in 5M recovery) seamlessly within `engine.get_pareto_data()`.
- **Parameter Counts**: Parameter counts across all documentation and tests match the authoritative PyTorch counts: 126,168 vs 125,688 (126K), 998,520 vs 998,523 (1M), and 5,000,632 vs 5,000,635 (5M).
