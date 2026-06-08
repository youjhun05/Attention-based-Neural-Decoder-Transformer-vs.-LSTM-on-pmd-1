# Attention-based-Neural-Decoder-Transformer-vs.-LSTM-on-pmd-1

This repository provides an advanced benchmarking and evaluation framework for Brain-Computer Interface (BCI) neural decoders using the **CRCNS pmd-1 dataset (Macaque M1+PMd)**. The project systematically evaluates and compares traditional and deep learning-based architectures in decoding continuous movement velocities from population spike counts, with a specific focus on robust evaluation under simulated neural communication failure.

## Project Overview
The goal of this study is to benchmark a series of neural decoders to reconstruct hand movement kinematics ($V_x, V_y$ velocities) from multi-channel cortical spike data. Additionally, we analyze the structural vulnerability and robustness of these decoders against varying levels of **Spike Error Rate (SER)**, drawing comparisons against baseline Spiking Neural Networks (SNN).

### Key Features & Technical Refinements
1. **Trial-level Cross-Validation:** Implements absolute separation at the trial boundaries during K-Fold data splitting to guarantee zero data-leakage across temporal bins.
2. **Affine Rescaling for Scale Bias:** Explicitly separates predictions before and after affine rescaling ($a \cdot \hat{y} + b$) to completely remove regression-to-mean artifacts induced by $L_2$ regularization.
3. **SER Robustness Sweep (Inference-Only):** Evaluates noise robustness by corrupting only the test set using a frozen, clean baseline model (Phase 2 model freeze) to examine *population-level noise dilution*.
4. **Causal Transformer Attention Map Extraction:** Extracts exact attention profile matrices from individual attention layers, tracing back what specific temporal history bins (-200ms to 0ms) the network prioritized during motion decoding.

---

## Model Architectures
We evaluate four distinct types of neural decoders, utilizing **200ms of temporal history** (20 bins $\times$ 10ms bin width) to predict current velocity:

* **Ridge Regression (Baseline):** Linear approach utilizing $L_2$ regularized weights mapped to continuous kinematics.
* **Multi-Layer Perceptron (MLP):** A non-linear feedforward neural network baseline (`256 -> 128` hidden units with ReLU activation).
* **LSTM (Long Short-Term Memory):** Recurrent neural network optimized via gradient clipping and Cosine Annealing learning rate scheduling.
* **Causal Transformer:** A specialized multi-layer multi-head attention decoder equipped with an upper-triangular causal mask to prevent information leakage from future time bins.

---

## Dataset Structure
* **Source:** CRCNS pmd-1 Dataset (Macaque Primary Motor Cortex [M1, 67 units] + Premotor Cortex [PMd, 94 units]).
* **Total Features:** 161 neurons.
* **Kinematics:** Continuous horizontal ($V_x$) and vertical ($V_y$) velocities.

---

## Execution & Requirements

### Dependencies
Ensure you have the following packages installed:
```bash
pip install numpy scipy scikit-learn torch matplotlib
