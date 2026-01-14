# spuco-george-spurious-correlations

This repository contains a Jupyter notebook implementing the George method for mitigating spurious correlations on the SpuCoMNIST dataset.

The implementation follows a three-step pipeline:
(1) ERM baseline training,
(2) clustering examples based on model outputs to infer latent groups, and
(3) group-balanced retraining to improve worst-group performance.

The goal is to demonstrate how spurious correlations can lead to poor generalization and how simple group-aware training can significantly improve robustness.

## Problem Motivation

Deep neural networks often rely on spurious features that correlate with the target label in the training data but are not causally related to the task.  
On datasets such as SpuCoMNIST, this behavior leads to high overall accuracy while severely degrading performance on minority or worst-case groups.

This notebook explores how such failures arise and how group-balanced training can mitigate them.

## Method Overview

We implement the **George** method, which consists of three steps:

1. **ERM Training**  
   Train a standard classifier using empirical risk minimization (ERM), which serves as a baseline and typically overfits to spurious features.

2. **Clustering Model Outputs**  
   Cluster training examples based on the model’s outputs to infer latent groups that capture spurious correlations without requiring group labels.

3. **Group-Balanced Retraining**  
   Retrain the model using group-balanced mini-batches so that each inferred group appears equally often, encouraging the model to rely on core features instead of spurious ones.

## Results

The notebook reports:
- Overall test accuracy
- Worst-group accuracy across spurious groups

Compared to the ERM baseline, group-balanced retraining substantially improves worst-group performance, demonstrating increased robustness to spurious correlations.


