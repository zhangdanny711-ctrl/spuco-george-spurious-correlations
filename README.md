# spuco-george-spurious-correlations

This repository contains a Jupyter notebook implementing the George method for mitigating spurious correlations on the SpuCoMNIST dataset.

The implementation follows a three-step pipeline:
(1) ERM baseline training,
(2) clustering examples based on model outputs to infer latent groups, and
(3) group-balanced retraining to improve worst-group performance.

The goal is to demonstrate how spurious correlations can lead to poor generalization and how simple group-aware training can significantly improve robustness.
