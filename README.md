# FOFedAdamW: Fractional-Order Federated Learning with Adaptive Optimization

## Overview

This project implements the **FOFedAdamW algorithm**, a novel federated learning optimizer that integrates:

- Fractional-order gradient memory (Grünwald–Letnikov derivative)
- Adaptive optimization (AdamW)
- Decoupled weight decay
- Differential privacy (DP variant)

The system simulates **multi-hospital training under non-IID data** using Dirichlet partitioning.

---

## Key Features

- Federated Learning (FedAvg aggregation)
- Fractional Gradient Computation (GL derivative, K=10)
- FOFedAdamW Optimizer (from scratch)
- Differential Privacy (Gradient clipping + Gaussian noise)
- Baseline Comparisons (FedAvg, FedProx, FedAdam, FOFedAvg)
- Evaluation Metrics: Accuracy, AUC-ROC, F1-score
- AUC vs Communication Rounds plotting

---

## Results

- Faster convergence than FedAvg
- Higher AUC-ROC under non-IID data
- Strong performance even with differential privacy

---

## Project Structure

```
data/            # Dirichlet partitioning
optimizers/      # GL + FOFedAdamW
baselines/       # Comparison methods
server/          # FedAvg aggregation
models/          # MLP model
train.py         # Main federated loop
evaluate.py      # Metrics
plot.py          # Visualization
```

---

## Research Contribution

This implementation follows the paper:

**"FOFedAdamW: Fractional-Order Federated Learning with Adaptive Optimization for Privacy-Preserving Multi-Hospital Disease Classification"**

---

---
