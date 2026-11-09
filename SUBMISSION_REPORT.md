# Byzantine-Resilient Federated Learning: Robustness, Attacks, and Secure Aggregation
## Comprehensive Project Submission Report & Execution Manual

---

### Executive Summary

Federated Learning (FL) enables decentralized machine learning where client nodes collaboratively train a shared model without transmitting raw private data. However, standard aggregation algorithms such as FederatedAveraging ([`fedavg`](file:///d:/byzantine_fl/aggregators.py#L19-L20)) are catastrophically vulnerable to **Byzantine failures**—adversarial or corrupted workers transmitting manipulated parameter updates. Furthermore, preserving user privacy through cryptographic **Secure Aggregation** introduces a fundamental paradox: individual updates masked with random additive noise cannot be inspected by the server for anomaly detection.

This project delivers an end-to-end simulation framework in Python and PyTorch that fulfills all milestones and deliverables outlined in the BTech Project Roadmap and Technical Research Report:
1. **Full Algorithmic Suite**: Implementation of FederatedAveraging, Coordinate-wise Median, Trimmed Mean, Krum, and Multi-Krum.
2. **Comprehensive Threat Suite**: Implementations of Data Poisoning (Label Flipping), Update Poisoning (Sign Flipping, Additive Gaussian Noise, Update Scaling), and State-of-the-Art Omniscient Collusion (Inner Product Manipulation — IPM).
3. **Rigorous Experimental Matrix**: 150+ automated experiments testing $0\%, 10\%, 20\%, 30\%$ Byzantine client ratios across both IID and pathological Non-IID client partitions.
4. **Complete Evaluation Metrics**: System-wide recording of Classification Accuracy, Macro Precision, Macro Recall, Macro F1-score, Accuracy Drop (Robustness), Recovery Gain vs. Attacked Baseline, and Runtime Efficiency (Training and Aggregation Latency).
5. **Cryptographic Privacy Extension**: A simulated single-server secure aggregation workflow demonstrating the privacy-robustness paradox and verifiable distance checks based on the Byzantine-Resilient Secure Aggregation (BREA) framework.
6. **Interactive Demo Application**: An interactive command-line evaluation tool ([`demo.py`](file:///d:/byzantine_fl/demo.py)) enabling live supervisor demonstrations.

---

### 1. Theoretical & Mathematical Foundations

#### 1.1 Distributed Synchronous SGD (DSSGD)
In each communication round $t \in [1, T]$:
1. The parameter server broadcasts global model parameters $x^t \in \mathbb{R}^d$ to participating workers.
2. Each worker $i \in [m]$ computes a local parameter update or gradient estimator $v_i^t$ on its local private partition $\mathcal{D}_i$.
3. The server collects the reported updates $\{\tilde{v}_1^t, \dots, \tilde{v}_m^t\}$, where $\tilde{v}_i^t = v_i^t$ for honest clients and $\tilde{v}_i^t = *$ for Byzantine clients.
4. The server aggregates updates via an aggregation rule $\text{Aggr}(\cdot)$ and performs the global model update:
   $$x^{t+1} = x^t + \eta \, \text{Aggr}\left(\{\tilde{v}_1^t, \dots, \tilde{v}_m^t\}\right)$$

#### 1.2 DSSGD-Byzantine Tolerance Criterion
Standard Byzantine agreement requires exact consensus. In distributed optimization, Byzantine tolerance requires the aggregated update to remain a valid descent direction with respect to the true expected gradient $g$:
$$\left\langle g, \, \mathbb{E}\left[\text{Aggr}\left(V \cup U\right)\right]\right\rangle \ge 0$$
where $V$ is the set of honest updates, $U$ is the set of Byzantine updates ($|U| = q < m/2$), and $\langle \cdot, \cdot \rangle$ denotes the inner product. If this inner product becomes negative, the global model performs gradient ascent, causing divergence.

#### 1.3 Inner Product Manipulation (IPM) Attacks
Traditional defenses rely on bounded Euclidean distances. However, when the global model nears a local minimum ($g \to 0$), small bounded adversarial vectors can flip the update direction.

- **Vulnerability of Coordinate-wise Median (Theorem 1)**:
  If $|g_j| < \sigma \sqrt{1/(m-q-1)}$, Byzantine attackers sending large updates opposite to the true gradient force the median to lie in the wrong tail of the distribution, flipping the sign:
  $$\text{sign}\left(\text{median}_j(V \cup U)\right) \ne \text{sign}(g_j)$$
  *Empirical validation in [`test_math.py`](file:///d:/byzantine_fl/test_math.py#L18-L22)*: For honest gradients $V = \{-0.1, 0.1, 0.3\}$ (mean $+0.1$) and Byzantine gradients $U = \{-4.0, -2.0\}$, $\text{Median}(V \cup U) = -0.1$, flipping the gradient sign.

- **Vulnerability of Krum (Theorem 2)**:
  Krum selects the candidate update minimizing the sum of squared distances to its $m - q - 2$ nearest neighbors. Byzantine workers sending identical small-magnitude vectors $u = -\varepsilon \bar{v}$ have mutual distance $0$. This creates an artificial dense cluster that tricks Krum into selecting the malicious vector.
  *Empirical validation in [`test_math.py`](file:///d:/byzantine_fl/test_math.py#L24-L32)*: For $V = \{0, 0.02, 0.14, 0.26, 0.38, 0.5\}$ and $U = \{-0.1, -0.1, -0.1\}$, Krum score $\text{KR}(-0.1) = 0.0244 < \text{KR}(0) = 0.0304$, selecting the Byzantine direction.

#### 1.4 The Privacy-Robustness Paradox
- **Secure Aggregation**: Uses pairwise zero-sum additive masks $s_{i,j} \in \mathbb{F}_P$ such that client $i$ transmits $\tilde{x}_i = (x_i + \sum_{j > i} s_{i,j} - \sum_{j < i} s_{j,i}) \pmod P$. Across all clients, $\sum_{i=1}^m \tilde{x}_i \equiv \sum_{i=1}^m x_i \pmod P$, preserving exact sums while individual vectors appear as uniform noise.
- **The Paradox**: Robust aggregators (Median, Krum, Trimmed Mean) require inspecting individual vector values or pairwise distances. Under cryptographic masking, updates are indistinguishable from uniform random noise, rendering distance-based outlier rejection ineffective.
- **The BREA Solution**: Verifiable distance evaluation over secret shares enables distance checks prior to unmasking.

---

### 2. Software Architecture & Implementation Map

The repository is modularized into independent functional layers:

```
d:\byzantine_fl\
├── aggregators.py              # Server aggregation functions (NumPy)
├── attacks.py                  # Byzantine attack injection functions
├── fl.py                       # Core FL simulator, models, dataset loaders, metrics
├── run_matrix.py               # Automated grid experiment runner
├── plot_results.py             # Generates publication curves and heatmaps
├── metrics_analysis.py         # Robustness, recovery, and timing metrics compiler
├── demo.py                     # Interactive presentation demo script
├── secure_agg_demo.py          # Cryptographic privacy-robustness simulation
├── seed_summary.py             # Multi-seed variance and confidence report
├── test_math.py                # Unit tests verifying report theorems
├── week1_tensors.py            # Milestone 1: PyTorch tensor mechanics
├── week1_training_loop.py      # Milestone 1: Fundamental training loop
├── week2_mnist_baseline.py     # Milestone 2: Centralized MNIST baseline
├── week3_fedavg.py             # Milestone 3: 10-client FedAvg baseline
├── week4_noniid.py             # Milestone 3-4: IID vs Non-IID comparison
├── mnist_data/                 # Downloaded MNIST dataset
├── results/                    # 150+ experiment JSON logs, CSV tables, metrics
└── plots/                      # 20 publication figures and heatmaps
```

#### Key Module Links & Implementations
- **Aggregators** ([`aggregators.py`](file:///d:/byzantine_fl/aggregators.py)):
  - [`fedavg`](file:///d:/byzantine_fl/aggregators.py#L19-L20): Standard coordinate-wise mean.
  - [`median`](file:///d:/byzantine_fl/aggregators.py#L23-L24): Coordinate-wise median.
  - [`trimmed_mean`](file:///d:/byzantine_fl/aggregators.py#L27-L34): Coordinate-wise trimmed mean excluding top/bottom $f$ elements.
  - [`krum`](file:///d:/byzantine_fl/aggregators.py#L53-L54): Selects vector minimizing distance to $n - f - 2$ nearest neighbors.
  - [`multi_krum`](file:///d:/byzantine_fl/aggregators.py#L57-L61): Averages top $m$ candidates scored by Krum metric.
- **Attacks** ([`attacks.py`](file:///d:/byzantine_fl/attacks.py)):
  - [`apply_update_attack`](file:///d:/byzantine_fl/attacks.py#L26-L50): Implements `sign_flip`, `gaussian`, `scaling`, and `ipm`.
  - Data Poisoning: Handled in [`local_train`](file:///d:/byzantine_fl/fl.py#L160-L174) with label permutation ($y \to 9 - y$).
- **Federated Engine** ([`fl.py`](file:///d:/byzantine_fl/fl.py)):
  - [`SimpleCNN`](file:///d:/byzantine_fl/fl.py#L129-L142): 2 Conv layers + MaxPool + 2 Linear layers.
  - [`partition`](file:///d:/byzantine_fl/fl.py#L106-L125): Supports both uniform IID and pathological Non-IID (2 shards per client).
  - [`_compute_classification_metrics`](file:///d:/byzantine_fl/fl.py#L32-L61): Confusion matrix, per-class and macro Precision, Recall, and F1.
  - [`run_experiment`](file:///d:/byzantine_fl/fl.py#L187-L232): Executes communication rounds, records per-round training and aggregation latencies.

---

### 3. Step-by-Step Project Execution Guide

Follow these commands in PowerShell or Command Prompt from `d:\byzantine_fl`. Use the project virtual environment at `d:\byzantine_fl\venv\Scripts\python.exe`.

#### Phase 1: Environment & Mathematical Verification
1. Verify PyTorch and mathematical formulas:
   ```powershell
   d:\byzantine_fl\venv\Scripts\python.exe test_math.py
   ```
   *Expected output*: Validates Theorem 1 (Median sign flip) and Theorem 2 (Krum cluster attack). Prints `ALL TESTS PASSED`.

2. Run Milestone 1 (PyTorch essentials):
   ```powershell
   d:\byzantine_fl\venv\Scripts\python.exe week1_tensors.py
   d:\byzantine_fl\venv\Scripts\python.exe week1_training_loop.py
   ```

3. Run Milestone 2 (Centralized baseline):
   ```powershell
   d:\byzantine_fl\venv\Scripts\python.exe week2_mnist_baseline.py
   ```
   *Outputs*: `mnist_loss_curve.png`, `mnist_baseline_result.txt` (Achieves $\approx 98.7\%$ test accuracy).

4. Run Milestone 3 (FedAvg baselines):
   ```powershell
   d:\byzantine_fl\venv\Scripts\python.exe week3_fedavg.py
   d:\byzantine_fl\venv\Scripts\python.exe week4_noniid.py
   ```
   *Outputs*: `fedavg_accuracy.png`, `fedavg_iid_vs_noniid.png` (Shows performance gap between IID $\sim 97\%$ and Non-IID $\sim 77\%$).

---

#### Phase 2: Running the Systematic Experiment Matrix
To reproduce the full evaluation matrix (5 attacks $\times$ 5 aggregators $\times$ 3 fractions $\times$ 2 data distributions = 150 experiments):

1. **Quick Sanity Check** (approx. 2 minutes):
   ```powershell
   d:\byzantine_fl\venv\Scripts\python.exe run_matrix.py --quick
   ```
2. **Full Matrix Run** (executed in background or loaded from pre-computed cache):
   ```powershell
   d:\byzantine_fl\venv\Scripts\python.exe run_matrix.py
   ```
   All 150+ experiment results are stored as JSON files in `d:\byzantine_fl\results\` and compiled into `d:\byzantine_fl\results\summary.csv`.

---

#### Phase 3: Generating Visualizations and Metrics Analysis
1. **Generate All Accuracy Curves and Summary Heatmaps**:
   ```powershell
   d:\byzantine_fl\venv\Scripts\python.exe plot_results.py
   ```
   *Outputs*: 10 round-by-round curve plots and 6 final accuracy heatmaps in `d:\byzantine_fl\plots\`.

2. **Generate Derived Metrics (Robustness, Recovery, Efficiency)**:
   ```powershell
   d:\byzantine_fl\venv\Scripts\python.exe metrics_analysis.py
   ```
   *Outputs*:
   - `results/robustness.csv`: Quantifies accuracy degradation under each attack.
   - `results/recovery.csv`: Quantifies accuracy restoration achieved by robust aggregators relative to attacked FedAvg.
   - `results/metrics_report.txt`: Formatted textual report.
   - `plots/robustness_heatmap_*.png`: Heatmaps of accuracy drop.
   - `plots/recovery_heatmap_*.png`: Heatmaps of defense recovery.

3. **Multi-Seed Variance Report**:
   ```powershell
   d:\byzantine_fl\venv\Scripts\python.exe seed_summary.py
   ```
   *Outputs*: Mean $\pm$ standard deviation across multiple random seeds for non-IID configurations with 30% Byzantine workers.

---

#### Phase 4: Secure Aggregation & Privacy Paradox Demo
Run the simulated cryptographic privacy-preserving aggregation protocol:
```powershell
d:\byzantine_fl\venv\Scripts\python.exe secure_agg_demo.py
```
*Outputs*: Demonstrates:
- Part 1: Exact mask cancellation in additive secret sharing over $\mathbb{F}_{2^{31}-1}$.
- Part 2: Empirical failure of Krum applied directly to masked ciphertexts (attacker selection rate $\approx 25\%$, matching random chance).
- Part 3: Verifiable distance oracle excluding Byzantine updates without decrypting individual vectors.

---

#### Phase 5: Interactive Demonstration (Slide 18)
Launch the interactive command-line application for examiners:
```powershell
d:\byzantine_fl\venv\Scripts\python.exe demo.py
```
Options available in demo:
- Select attack (`sign_flip`, `label_flip`, `gaussian`, `ipm`, `scaling`).
- Select data distribution (`iid`, `noniid`).
- Select Byzantine fraction (`10%`, `20%`, `30%`).
- Select client scale ($10, 20, 50$).
- Select round budget ($5, 10, 15, 20$).
- Select aggregation method or **compare all aggregators side-by-side**.
- Live display of accuracy, precision, recall, F1, latency, and automated technical explanation of defense performance.
- Saves comparison plot to `plots/demo_comparison.png` and JSON log to `results/demo_results.json`.

---

### 4. Detailed Explanation of Output Files & Visualizations

#### 4.1 Convergence Curves (`plots/curves_<attack>_<partition>.png`)

```
   Test Accuracy
      1.0 ┤       ╭──────────────────────── Coordinate-wise Median / Multi-Krum (~88%)
          │      ╭╯
      0.5 ┤     ╭╯
          │    ╭╯
      0.0 ┼───┴──────────────────────────── FedAvg under Sign-Flip (~9%)
          └────────────────────────────────
          Round 1                      Round 15
```

- **`curves_sign_flip_iid.png`**: Under a 30% sign-flip attack, FedAvg accuracy drops immediately to random guessing ($9.6\%$). Median, Trimmed Mean, and Multi-Krum completely neutralize the attack, reaching $\approx 88.5\%$ accuracy. Krum converges to $86.4\%$.
- **`curves_ipm_iid.png`**: Demonstrates Theorem 2 from the report. While FedAvg sustains mild degradation ($86.3\%$), Krum experiences severe degradation ($78.5\%$) because identical colluding updates create a false density cluster that fools Krum's nearest-neighbor score.
- **`curves_label_flip_noniid.png`**: Illustrates the interaction between non-IID data skew and data poisoning. FedAvg degrades to $46.4\%$, while Multi-Krum and Median maintain $51.2\%$ to $53.5\%$.

---

#### 4.2 Final Accuracy Heatmaps (`plots/final_acc_<partition>_<pct>pct.png`)
These heatmaps summarize final test accuracy after 15 communication rounds:
- **Rows**: Byzantine Attacks (`gaussian`, `ipm`, `label_flip`, `scaling`, `sign_flip`).
- **Columns**: Aggregators (`fedavg`, `krum`, `median`, `multi_krum`, `trimmed_mean`).
- **Color Scale**: Green indicates high accuracy ($\ge 85\%$), yellow indicates moderate accuracy ($50\% - 70\%$), red indicates complete failure ($< 30\%$).

**Key Insights from Heatmaps**:
1. **The IID Regime (30% Malicious)**:
   - FedAvg fails critically against `sign_flip` ($0.096$).
   - Coordinate-wise Median and Trimmed Mean achieve near-perfect defense across all attacks ($87\% - 89\%$).
   - Krum resists `sign_flip` ($86.4\%$) and `gaussian` ($86.4\%$), but degrades under `ipm` ($78.5\%$).
2. **The Non-IID Regime (Pathological Skew, 30% Malicious)**:
   - **Krum Failure**: Krum fails across non-IID settings ($8\% - 23\%$), even under benign or mild attacks. In non-IID partitions, honest client updates naturally diverge from each other. Krum misinterprets honest domain specialization as adversarial outliers, selecting unrepresentative or corrupted updates.
   - **Multi-Krum Advantage**: By averaging the top $m$ candidates instead of picking a single update, Multi-Krum mitigates Krum's catastrophic collapse, achieving $61.7\%$ under gaussian noise and $61.7\%$ under sign-flip.

---

#### 4.3 Robustness Metric (`results/robustness.csv` & `plots/robustness_heatmap_*.png`)
**Metric Definition**:
$$\text{Robustness Drop} = \text{Accuracy}_{\text{clean}} - \text{Accuracy}_{\text{attacked}}$$
- A value near $0.00$ indicates perfect robustness (no degradation).
- A large positive value indicates severe vulnerability.

**Sample Robustness Values (IID, 30% Byzantine Clients, Clean Baseline = 0.8865)**:
| Attack | FedAvg Drop | Krum Drop | Median Drop | Multi-Krum Drop | Trimmed Mean Drop |
|---|---|---|---|---|---|
| `sign_flip` | **+0.7905** (Vulnerable) | +0.0225 | **+0.0170** (Robust) | **+0.0040** (Most Robust) | +0.0170 |
| `gaussian` | +0.0220 | +0.0225 | +0.0015 | +0.0040 | **+0.0010** |
| `ipm` | +0.0235 | **+0.1010** (Vulnerable) | +0.0155 | +0.0255 | +0.0155 |
| `label_flip` | +0.0460 | +0.0225 | +0.0020 | +0.0040 | +0.0030 |

---

#### 4.4 Recovery Metric (`results/recovery.csv` & `plots/recovery_heatmap_*.png`)
**Metric Definition**:
$$\text{Recovery Gain} = \text{Accuracy}_{\text{robust\_agg}} - \text{Accuracy}_{\text{attacked\_fedavg}}$$
- A positive value measures the performance improvement provided by the robust defense over vulnerable FedAvg.
- A negative value indicates that the defense performed worse than doing nothing.

**Sample Recovery Values (IID, 30% Byzantine Clients)**:
| Attack | FedAvg (Ref) | Krum Gain | Median Gain | Multi-Krum Gain | Trimmed Mean Gain |
|---|---|---|---|---|---|
| `sign_flip` | 0.0960 | **+0.7680** | **+0.7735** | **+0.7865** | **+0.7735** |
| `label_flip` | 0.8405 | +0.0235 | **+0.0440** | +0.0420 | +0.0430 |
| `gaussian` | 0.8645 | -0.0005 | +0.0205 | +0.0180 | **+0.0210** |
| `ipm` | 0.8630 | **-0.0775** (Degraded) | +0.0080 | -0.0020 | +0.0080 |

---

#### 4.5 Runtime Efficiency Metric (`results/efficiency.csv`)
**Metric Definition**: Measures communication round latency and per-round aggregation overhead:
- Local training dominates total round time ($\approx 1.2\text{s} - 1.8\text{s}$ on CPU per round for 10 clients).
- Aggregation overhead differs significantly by algorithm:
  - **FedAvg**: $\approx 0.05\text{ms}$ (Simple vector addition).
  - **Coordinate-wise Median**: $\approx 1.8\text{ms}$ (1D median per parameter).
  - **Trimmed Mean**: $\approx 2.4\text{ms}$ (Coordinate sorting).
  - **Krum**: $\approx 15.6\text{ms} - 45.0\text{ms}$ ($O(n^2 \cdot d)$ pairwise Euclidean distance computation over $400,000+$ parameters).
  - **Multi-Krum**: $\approx 16.2\text{ms} - 46.5\text{ms}$ (Pairwise distances + top-$m$ averaging).

---

### 5. Compliance Matrix: PPT & Report Deliverables

| Milestone / Requirement | Document Reference | Status | Implementing Script / Artifact |
|---|---|---|---|
| PyTorch Tensors & Training Loop | PPT Slide 6, Month 1 | ✅ Completed | [`week1_tensors.py`](file:///d:/byzantine_fl/week1_tensors.py), [`week1_training_loop.py`](file:///d:/byzantine_fl/week1_training_loop.py) |
| Centralized Baseline Model | PPT Slide 6, Month 1 | ✅ Completed | [`week2_mnist_baseline.py`](file:///d:/byzantine_fl/week2_mnist_baseline.py), `mnist_loss_curve.png` |
| 10-Client FedAvg Simulator | PPT Slide 6, Month 1 | ✅ Completed | [`week3_fedavg.py`](file:///d:/byzantine_fl/week3_fedavg.py), `fedavg_accuracy.png` |
| IID vs Non-IID Partitioning | PPT Slide 12, Month 4 | ✅ Completed | [`week4_noniid.py`](file:///d:/byzantine_fl/week4_noniid.py), `fedavg_iid_vs_noniid.png` |
| Attacks: Label Flip, Sign Flip, Poisoning | PPT Slide 8, Month 2 | ✅ Completed | [`attacks.py`](file:///d:/byzantine_fl/attacks.py), [`fl.py`](file:///d:/byzantine_fl/fl.py) |
| Inner Product Manipulation (IPM) Attack | Report Section 4 | ✅ Completed | [`attacks.py`](file:///d:/byzantine_fl/attacks.py#L44-L47) (`apply_update_attack`) |
| Robust Aggregators: Median, Trimmed Mean, Krum, Multi-Krum | PPT Slide 10, Month 3 | ✅ Completed | [`aggregators.py`](file:///d:/byzantine_fl/aggregators.py) |
| Mathematical Verification of Theorems 1 & 2 | Report Section 4 | ✅ Completed | [`test_math.py`](file:///d:/byzantine_fl/test_math.py) |
| Systematic Matrix (Attacks $\times$ Aggregators $\times$ Fractions $\times$ Partitions) | PPT Slide 12, Month 4 | ✅ Completed | [`run_matrix.py`](file:///d:/byzantine_fl/run_matrix.py), `results/summary.csv` |
| Metrics: Accuracy, Precision, Recall, F1 | PPT Slide 13, Month 4 | ✅ Completed | [`fl.py`](file:///d:/byzantine_fl/fl.py#L32-L61) (`_compute_classification_metrics`) |
| Metrics: Convergence Curves | PPT Slide 13, Month 4 | ✅ Completed | [`plot_results.py`](file:///d:/byzantine_fl/plot_results.py), 10 curve plots in `plots/` |
| Metrics: Robustness (Accuracy Drop) | PPT Slide 13, Month 4 | ✅ Completed | [`metrics_analysis.py`](file:///d:/byzantine_fl/metrics_analysis.py), `results/robustness.csv` |
| Metrics: Recovery Gain | PPT Slide 13, Month 4 | ✅ Completed | [`metrics_analysis.py`](file:///d:/byzantine_fl/metrics_analysis.py), `results/recovery.csv` |
| Metrics: Efficiency / Timing Analysis | PPT Slide 13, Month 4 | ✅ Completed | [`fl.py`](file:///d:/byzantine_fl/fl.py#L222-L230), [`metrics_analysis.py`](file:///d:/byzantine_fl/metrics_analysis.py) |
| Secure Aggregation & Privacy Paradox Extension | PPT Slide 14, Month 5 | ✅ Completed | [`secure_agg_demo.py`](file:///d:/byzantine_fl/secure_agg_demo.py) |
| Multi-Seed Reproducibility Analysis | PPT Slide 12, 19 | ✅ Completed | [`seed_summary.py`](file:///d:/byzantine_fl/seed_summary.py), 30 seed result JSONs |
| Interactive Demonstration Application | PPT Slide 18 | ✅ Completed | [`demo.py`](file:///d:/byzantine_fl/demo.py) |

---

### 6. Conclusions & Key Findings

1. **FedAvg is Fragile Under Malicious Updates**: A 20% sign-flip attacker cohort causes complete model collapse ($9.6\%$ accuracy, equivalent to random guessing).
2. **Coordinate-wise Median & Trimmed Mean are Robust in Practice**: Both rules effectively neutralize large-scale update poisoning and noise injection in IID settings with minimal computational overhead ($< 3\text{ms}$ aggregation time).
3. **Krum Fails in Non-IID Federated Environments**: While mathematically elegant for IID data under bounded adversaries, Krum's spatial distance metric breaks down under non-IID data distributions ($< 23\%$ accuracy) because legitimate specialized client gradients resemble outliers.
4. **IPM Exposes Distance-Based Defenses**: Colluding Byzantine workers can manipulate gradient trajectory near convergence ($g \to 0$) without triggering Euclidean distance outlier detectors, validating Theorems 1 and 2 of the research report.
5. **Secure Aggregation Requires Verifiable Outlier Checks**: Masked updates hide adversarial perturbations. Practical deployments require integrated cryptographic frameworks like BREA to compute pairwise distances on secret shares before reconstruction.
