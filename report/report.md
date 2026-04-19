# Self-Pruning Neural Network

## Why L1 Regularization Encourages Sparsity

L1 regularization penalizes the absolute values of parameters. Since gate values lie between 0 and 1, minimizing their sum forces many values toward zero.

This results in:
- Removal of unnecessary weights
- Sparse network structure
- Improved efficiency

Unlike L2 regularization, L1 creates exact zeros, making it ideal for pruning.

---

## Results and Analysis

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|--------------|
| 0.01   | 50.99       | 1.75         |
| 0.10   | 51.59       | 7.78         |
| 1.00   | 50.82       | 24.82        |

### Analysis

The results demonstrate the effectiveness of L1-based sparsity regularization:

- As λ increases, sparsity increases significantly, confirming that the model learns to suppress less important weights.
- The accuracy remains relatively stable across different λ values, indicating that many parameters are redundant.
- At λ = 1.0, the model achieves ~25% sparsity with negligible accuracy loss, showing that pruning can improve efficiency without major performance degradation.

This highlights a key insight:  
**Neural networks are often over-parameterized, and structured regularization can reduce complexity without sacrificing accuracy.**
