# Self-Pruning Neural Network

## Why L1 Regularization Encourages Sparsity

L1 regularization penalizes the absolute values of parameters. Since gate values lie between 0 and 1, minimizing their sum forces many values toward zero.

This results in:
- Removal of unnecessary weights
- Sparse network structure
- Improved efficiency

Unlike L2 regularization, L1 creates exact zeros, making it ideal for pruning.

---

## Results

| Lambda | Accuracy | Sparsity (%) |
|--------|----------|--------------|
| 0.0001 | XX       | XX           |
| 0.001  | XX       | XX           |
| 0.01   | XX       | XX           |

---

## Observations

- Higher lambda → higher sparsity
- But accuracy drops
- Trade-off clearly visible

---

## Plot

![Gate Distribution](../results/plots/gate_distribution.png)