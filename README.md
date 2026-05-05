# X-CBA: Explainable Graph-Based Intrusion Detection System

> Reproduction and Enhancement of the X-CBA Framework  
> ISDI 509 – Data Security and Privacy  
> M2 – Information Systems and Data Intelligence  
> Lebanese University, Faculty of Science  
>
> **Authors:** Lama Abdallah, Malak Alkazem  
> **Supervisor:** Dr. Ahmad Fadlallah  
> **Date:** February 2026

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Design Choices](#key-design-choices)
- [Experimental Results](#experimental-results)
- [Tools and Frameworks](#tools-and-frameworks)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Future Work](#future-work)

---

## Overview

This project reproduces and enhances the **X-CBA** framework — an explainable, graph-based network intrusion detection system (IDS). Traditional ML-based IDS approaches achieve strong performance but operate as "black boxes," making it impossible for security analysts to understand why traffic is flagged as malicious or benign.

**X-CBA** addresses this by combining:
- **Graph-based representation learning** (Deep Graph Infomax) to capture relational and temporal attack patterns
- **Multiple explainability methods** (GNNExplainer, PGExplainer, SubgraphX) to make model decisions transparent
- **CatBoost classification** on learned graph embeddings for robust final binary classification

### Problems Solved

- Security analysts cannot verify whether alerts are based on legitimate threat indicators or spurious correlations
- Organizations cannot meet regulatory requirements for explainable automated decisions
- False positives and false negatives are difficult to diagnose and correct
- Trust in automated security systems remains limited

---

## Architecture

The pipeline is structured as four sequential notebooks:

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `1_anomal-e.ipynb` | Data Preprocessing — loading, cleaning, feature engineering |
| 2 | `2_boosted-anomal-e-checkpoint.ipynb` | Graph Construction and DGI Training — build graphs, train Deep Graph Infomax, generate node embeddings |
| 3 | `3_xai_boosted_anomal-e.ipynb` | Explainability Integration — apply GNNExplainer, PGExplainer, and SubgraphX |
| 4 | `4_xai_performance_eval.ipynb` | Fidelity Evaluation and Classification — measure explainer quality and train CatBoost classifier |

### Core Components

**Deep Graph Infomax (DGI)**  
Learns node embeddings in an unsupervised manner by maximizing mutual information between local node representations and global graph summaries.

**Explainability Methods**
- **GNNExplainer** — identifies important subgraphs by learning edge masks
- **PGExplainer** — probabilistic approach that trains a separate explanation model for instance-level explanations
- **SubgraphX** — uses Monte Carlo Tree Search (MCTS) to find minimal explanatory subgraphs

**CatBoost Classifier**  
Performs final binary classification (benign vs. attack) on learned graph embeddings.

---

## Key Design Choices

### Dataset Fraction (`frac=0.05`)
5% of the available data was used for all experiments to balance computational cost with reproducibility.

### Reduced Training Epochs
DGI training epochs were reduced compared to the original paper to enable faster iteration.

### SMOTE for Class Imbalance
The dataset exhibited significant class imbalance (benign traffic vastly outnumbering attack samples). SMOTE (Synthetic Minority Over-sampling Technique) was applied to the graph embeddings before training the CatBoost classifier.

### Division of Explainability Work
- **Student 1 (Lama)** — compared GNNExplainer vs. PGExplainer using Google Colab
- **Student 2 (Malak)** — compared GNNExplainer vs. SubgraphX using VS Code with a local Python environment

---

## Experimental Results

### Graph Construction and DGI Performance

| Graph | Nodes | Edges |
|-------|-------|-------|
| Training | 42,608 | 1,320,752 |
| Testing | 24,811 | 566,021 |

**DGI Embedding Quality (Logistic Regression on test graph):**

| Metric | Score |
|--------|-------|
| Accuracy | 99.04% |
| F1-Score | 96.09% |
| Precision | 94.13% |
| Recall | 98.13% |

### Explainability Evaluation – Fidelity Analysis

**Baseline (5% edges removed, 95% retained):**

| Metric | Score |
|--------|-------|
| Accuracy | 99.47% |
| F1-Macro | 98.73% |
| Precision | 99.86% |
| Recall | 95.74% |

**Explainer Comparison (averaged across sparsity levels):**

| Explainer | Avg. Accuracy | Avg. F1-Score | Fidelity+ |
|-----------|--------------|--------------|-----------|
| GNNExplainer | ~99.48% | ~98.75% | -0.0001 |
| SubgraphX | ~99.49% | ~98.78% | -0.0001 |

Key findings:
- Both explainers successfully identified important edges
- SubgraphX showed marginally better performance preservation due to its MCTS-based search
- Performance remained consistently high even when **80% of edges were removed**, demonstrating robustness
- Near-zero Fidelity+ confirms explainers correctly identified the truly important graph structures

### Comparison with Original Paper

| Aspect | Original Paper | This Reproduction |
|--------|---------------|-------------------|
| Dataset scale | Full dataset | 5% (frac=0.05) |
| DGI accuracy | ~99% | 99.04% ✅ |
| Explainability methods | GNNExplainer, PGExplainer, SubgraphX | All three ✅ |
| CatBoost accuracy | 99.47% | ~88% (class imbalance impact) |

---

## Tools and Frameworks

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10 | Core language |
| PyTorch | 2.1.0 | Deep learning |
| DGL (Deep Graph Library) | 1.1.2 | GNN framework (DGI + explainers) |
| CatBoost | latest | Gradient boosting classifier |
| Scikit-learn | latest | Evaluation metrics and preprocessing |
| Imbalanced-learn | latest | SMOTE implementation |
| Pandas & NumPy | latest | Data manipulation |
| Matplotlib & Seaborn | latest | Visualization |

**Execution environments:** Google Colab (cloud GPU) and VS Code with local Python virtual environments.

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended)
- At least 16 GB RAM
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/MalakAlKazem/XCBA-ResearchProject.git
cd XCBA-ResearchProject
```

> **Note:** The original paper's code spans two branches corresponding to different explainer comparisons (GNNExplainer vs. PGExplainer, and GNNExplainer vs. SubgraphX). Select the branch relevant to the comparison you want to run.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `dgl`, `catboost`, `scikit-learn`, `imbalanced-learn`, `pandas`, `numpy`, `matplotlib`.

### Step 3: Configure Paths

If running locally (not in Colab), update file paths in the notebooks:

- Model paths (e.g., `../models/best_dgi.pkl`) to match your directory structure
- Dataset paths to point to your data location
- Ensure output directories exist or will be created automatically

### Step 4: Execute Notebooks in Order

```
1_anomal-e.ipynb                  ← Data Preprocessing
2_boosted-anomal-e-checkpoint.ipynb  ← Graph Construction + DGI Training
3_xai_boosted_anomal-e.ipynb      ← Explainability (GNNExplainer / PGExplainer / SubgraphX)
4_xai_performance_eval.ipynb      ← Fidelity Evaluation + CatBoost Classification
```

### Step 5: Optional Adjustments

You can modify key parameters to experiment with different configurations:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frac` | `0.05` | Dataset fraction (e.g., `0.1` for 10%) |
| `num_epochs` | reduced | DGI training epochs |
| SMOTE `sampling_strategy` | default | Minority class over-sampling ratio |
| SMOTE `k_neighbors` | default | Neighbors for synthetic sample generation |
| Explainer | GNNExplainer | Switch to PGExplainer or SubgraphX in Notebook 3 |

### Expected Outputs

- Trained DGI model saved to `models/` directory
- Graph embeddings exported as CSV files
- Fidelity evaluation plots (PNG format)
- CatBoost classification results and confusion matrix
- CSV files with detailed metrics at each sparsity level

---

## Project Structure

```
XCBA-ResearchProject/
├── 1_anomal-e.ipynb                    # Data Preprocessing
├── 2_boosted-anomal-e-checkpoint.ipynb # Graph Construction + DGI Training
├── 3_xai_boosted_anomal-e.ipynb        # Explainability Integration
├── 4_xai_performance_eval.ipynb        # Fidelity Evaluation + Classification
└── README.md
```

---

## Future Work

- **Full-Scale Training** — train on the complete dataset with optimized hyperparameters to close the performance gap with the original paper
- **Advanced Class Imbalance Techniques** — cost-sensitive learning, focal loss, or ensemble methods with minority class boosting
- **Real-Time Deployment** — implement the pipeline as a streaming system for live network traffic
- **Explainability Visualization** — interactive visualizations of explanatory subgraphs to help security analysts interpret alerts

---

*Lebanese University – Faculty of Science, February 2026*
