# Byzantine-Resilient Federated Learning

**Robustness, Adversarial Attacks, and Secure Aggregation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Complete](https://img.shields.io/badge/BTech%20Project-Complete-success.svg)]()

A PyTorch-based implementation exploring how federated learning systems behave under Byzantine failures and adversarial client updates. The project compares standard and robust aggregation strategies, evaluates multiple attack scenarios, and demonstrates the privacy–robustness trade-off introduced by secure aggregation.

---

## Contributors

This project was developed collaboratively by:

- **Abhinav Mishra** — Project implementation and development
- **Garvit Sharma** — Project implementation and development

Both contributors are listed as project collaborators.

---

## Project Overview

Federated Learning (FL) allows multiple clients to train a shared model without sharing their raw local data. However, malicious or faulty clients can send manipulated model updates that reduce model accuracy or disrupt training.

This project provides a comprehensive experimental framework to study these threats and evaluate robust aggregation methods under both IID and non-IID data distributions across 150+ automated experiments.

---

## Key Features

- **Federated Learning Simulation:** Centralized MNIST baseline and multi-client FedAvg training from scratch.
- **Robust Aggregation Suite:** Federated Averaging (FedAvg), Coordinate-wise Median, Trimmed Mean, Krum, and Multi-Krum.
- **Adversarial Attack Taxonomy:** Sign flipping, Gaussian noise, update scaling, Inner Product Manipulation (IPM), and label flipping (data poisoning).
- **IID and Non-IID Experiments:** Compare performance under uniformly distributed data versus highly skewed (pathological 2-class) client partitions.
- **Evaluation Metrics:** Classification accuracy, macro precision, recall, F1-score, accuracy drop (robustness), recovery gain vs. attacked FedAvg, and aggregation latency.
- **Experiment Automation:** Automated grid runner evaluating 150+ configurations with structured JSON and CSV outputs.
- **Visual Analytics:** Generates publication-ready round-by-round convergence curves and comparative heatmaps.
- **Secure Aggregation Demonstration:** Simulates additive secret sharing over finite fields and illustrates the privacy–robustness paradox and verifiable distance checks (BREA protocol).
- **Interactive Demonstrations:** Command-line viva demo (`demo.py`) and a complete offline web dashboard (`index.html`) with live math checks and interactive simulations.

---

## Project Modules

| File | Phase / Milestone | Purpose |
|---|---|---|
| [`week1_tensors.py`](week1_tensors.py) | Month 1, Week 1 | PyTorch tensor mechanics and basic aggregation operations |
| [`week1_training_loop.py`](week1_training_loop.py) | Month 1, Week 1 | Five-step local training loop on toy data |
| [`week2_mnist_baseline.py`](week2_mnist_baseline.py) | Month 1, Week 2 | Centralized MNIST CNN baseline (`SimpleCNN`) |
| [`week3_fedavg.py`](week3_fedavg.py) | Month 1, Weeks 3–4 | Ten-client FedAvg baseline with IID data |
| [`week4_noniid.py`](week4_noniid.py) | Month 1, Week 4 | FedAvg comparison under IID and pathological non-IID partitions |
| [`attacks.py`](attacks.py) | Month 2 | Byzantine update attacks (sign flip, gaussian, scaling, IPM) and data poisoning |
| [`aggregators.py`](aggregators.py) | Month 3 | Robust rules: FedAvg, Median, Trimmed Mean, Krum, and Multi-Krum |
| [`test_math.py`](test_math.py) | Month 3 | Unit tests validating Theorems 1 & 2 from the research report |
| [`fl.py`](fl.py) | Months 1–4 | Core federated simulator, models, dataset loader, and metrics |
| [`run_matrix.py`](run_matrix.py) | Month 4 | Automated experiment runner across 150 configurations |
| [`plot_results.py`](plot_results.py) | Months 4–5 | Generates 20 accuracy curves and final accuracy heatmaps |
| [`metrics_analysis.py`](metrics_analysis.py) | Months 4–5 | Computes robustness drop, recovery gain, and efficiency summaries |
| [`secure_agg_demo.py`](secure_agg_demo.py) | Month 5 | Secure aggregation and privacy–robustness paradox simulation |
| [`seed_summary.py`](seed_summary.py) | Month 5 | Multi-seed results summary (mean and standard deviation) |
| [`demo.py`](demo.py) | Final Milestone | Interactive command-line viva demonstration |
| [`index.html`](index.html) | Interactive Web App | Self-contained offline browser dashboard for viva demonstrations |
| [`SUBMISSION_REPORT.md`](SUBMISSION_REPORT.md) | Final Deliverable | Detailed technical report, mathematical proofs, results, and findings |
| [`VIVA_QUESTIONS_AND_ANSWERS.md`](VIVA_QUESTIONS_AND_ANSWERS.md) | Viva Preparation | Comprehensive examiner Q&A preparation guide |
| [`PRESENTATION_SLIDES.md`](PRESENTATION_SLIDES.md) | Slide Deck | 20-slide presentation matching the project implementation plan |

---

## Repository Structure

```text
.
├── aggregators.py              # Server aggregation functions (NumPy)
├── attacks.py                  # Byzantine attack injection functions
├── fl.py                       # Core FL simulator, models, partition logic, metrics
├── run_matrix.py               # Automated grid experiment runner
├── plot_results.py             # Generates curves and heatmaps in plots/
├── metrics_analysis.py         # Computes robustness, recovery, and latency tables
├── secure_agg_demo.py          # Cryptographic privacy-robustness paradox simulation
├── seed_summary.py             # Multi-seed variance and confidence reporter
├── test_math.py                # Mathematical validation of Theorems 1 & 2
├── demo.py                     # Interactive presentation demo script
├── index.html                  # Standalone offline browser dashboard
├── web_data.js                 # Pre-compiled results for browser dashboard
├── week1_tensors.py            # Month 1: Tensor mechanics
├── week1_training_loop.py      # Month 1: 5-step training loop
├── week2_mnist_baseline.py     # Month 1: Centralized CNN baseline
├── week3_fedavg.py             # Month 1: 10-client FedAvg baseline
├── week4_noniid.py             # Month 1: IID vs Non-IID FedAvg
├── requirements.txt            # Python dependencies
├── SUBMISSION_REPORT.md        # Formal academic submission report
├── VIVA_QUESTIONS_AND_ANSWERS.md # Examiner Q&A defense cheatsheet
├── PRESENTATION_SLIDES.md      # 20-slide presentation deck
├── plots/                      # 20 publication figures and heatmaps
└── results/                    # 150+ experiment JSON logs and CSV summaries
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/abhi2811mishra/-Byzantine-Resilient-Federated-Learning.git
cd -Byzantine-Resilient-Federated-Learning
```

### 2. Create and Activate a Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```bat
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Mathematical Verification (Step 1)

Validates Theorems 1 & 2 on synthetic vectors without needing to train a neural network:

```bash
python test_math.py
```

### 5. Launch the Interactive Viva Demonstration

```bash
python demo.py
```

### 6. Interactive Web Dashboard (Offline Browser Experience)

Double-click [`index.html`](index.html) in your file explorer to open the interactive dashboard in any browser. It allows you to:
- Run math checks interactively with live pass/fail outputs.
- Simulate federated learning rounds with live SVG convergence curves.
- Inspect the full 150-experiment heatmap grid.
- Step through the 3-part cryptographic secure aggregation simulation.

### 7. Generate Metrics and Visualizations

```bash
python metrics_analysis.py
python plot_results.py
```

> **Note:** The MNIST dataset will be automatically downloaded to `mnist_data/` on the first training run. All commands should be executed from the repository root.

---

## Running the Experiment Matrix

To run a quick sanity-check experiment (2 rounds, sign-flip and IPM):

```bash
python run_matrix.py --quick
```

To run the full configured experiment matrix (150 configurations across 15 communication rounds):

```bash
python run_matrix.py
```

Experiment logs are stored in `results/` as individual JSON files and compiled into `results/summary.csv`. Generated figures are saved in `plots/`.

---

## Evaluation & Metrics

The framework records and analyzes:

- **Classification Accuracy:** Test accuracy per communication round.
- **Macro Precision, Recall, and F1-Score:** Multi-class classification balance across all 10 digits.
- **Robustness Drop:** $\text{Drop} = \text{Accuracy}_{\text{clean}} - \text{Accuracy}_{\text{attacked}}$ (lower is better).
- **Recovery Gain:** $\text{Recovery} = \text{Accuracy}_{\text{robust}} - \text{Accuracy}_{\text{attacked\_fedavg}}$ (higher is better).
- **Runtime Efficiency:** Communication round duration and per-round aggregation latency.

---

## Key Experimental Findings

1. **FedAvg is Fragile:** A 30% sign-flip attack causes FedAvg accuracy to collapse from $88.65\%$ to $9.60\%$ (pure random guessing).
2. **Median & Trimmed Mean are Robust in IID:** Both methods effectively neutralize update poisoning and noise injection in IID settings with minimal latency overhead ($< 3\text{ ms}$).
3. **Krum Fails on Non-IID Data:** While effective on IID data ($\approx 86\%$), Krum drops to $8\% - 23\%$ under non-IID partitions because legitimate specialized client gradients resemble adversarial anomalies. Multi-Krum resolves this by averaging top candidates, reaching $67.5\% - 71.5\%$.
4. **Inner Product Manipulation (IPM) Bypasses Distance Filters:** Colluding attackers sending small updates ($-\varepsilon \bar{v}$) evade Euclidean distance outlier detectors near convergence ($g \to 0$), verifying Theorems 1 and 2.
5. **The Privacy-Robustness Paradox:** Outlier detection directly on masked ciphertexts yields random chance ($\approx 25\%$). Verifiable distance protocols like BREA are required to reconcile privacy with robustness.

---

## Technologies

- **Language:** Python 3.10+
- **Deep Learning:** PyTorch, Torchvision
- **Numerical Computing:** NumPy
- **Data Visualization:** Matplotlib, Tailwind CSS (for web UI)
- **Domain:** Federated Learning, Byzantine Fault Tolerance, Secure Aggregation

---

## Documentation

- [**`SUBMISSION_REPORT.md`**](SUBMISSION_REPORT.md) — Complete academic report, mathematical derivations, and detailed results analysis.
---

## Acknowledgements

This project was completed collaboratively by **Abhinav Mishra** and **Garvit Sharma** as a BTech Capstone Project.
