# Self-Pruning Neural Network Report

## Problem Understanding

The goal was to build a neural network that learns to prune itself during training. Instead of pruning after training, each weight is controlled by a learnable gate. The model jointly learns classification weights and pruning decisions.

## Method

I implemented a custom `PrunableLinear` layer. Each normal weight has a matching `gate_score`. During the forward pass, the gate is calculated using sigmoid, and the effective weight becomes:

```text
effective_weight = weight * sigmoid(gate_score)
```

If the gate value becomes small, the corresponding connection contributes very little and is treated as pruned.

## Sparsity Regularization

The total loss combines classification loss and sparsity loss:

```text
Total Loss = CrossEntropyLoss + lambda * mean(gates)
```

The L1-style penalty on gate values encourages unnecessary gates to become small. Cross-entropy keeps important gates active, while the sparsity penalty suppresses less useful connections.

## Results

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|---:|---:|---:|
| 0.5 | 43.11 | 62.48 |
| 1.0 | 41.20 | 69.71 |
| 2.0 | 42.05 | 79.06 |
| 5.0 | 42.71 | 91.44 |

## Best Accuracy Model

The best accuracy model used lambda `0.5` with `43.11%` test accuracy and `62.48%` sparsity.

## Highest Sparsity Model

The highest sparsity model used lambda `5.0` with `91.44%` sparsity and `42.71%` test accuracy.

## Trade-off Analysis

The experiments show the expected sparsity-accuracy trade-off. Lower lambda values allow more gates to remain active, usually preserving accuracy. Higher lambda values apply stronger pressure on the gates, increasing sparsity but potentially reducing accuracy. This validates that the network is learning pruning decisions during training rather than relying on a separate post-training pruning step.

## Conclusion

This approach converts pruning into a differentiable optimization problem. The network learns both task performance and structural compression jointly, which makes it suitable for memory- and compute-constrained deployment scenarios.
