# DeepFrag-Fusion: Cancer Detection via cfDNA Fragmentomics, Wavelets, and Adaptive Model Fusion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow / PyTorch](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20PyTorch-orange.svg)](https://pytorch.org/)

Official code repository for cancer detection and classification using cell-free DNA (cfDNA) fragmentomics, discrete wavelet transforms (DWT), graph neural networks (GCN), deep neural networks (DNN), and adaptive multi-model score fusion. 

This pipeline processes high-throughput genome-wide cfDNA fragment profiles to distinguish cancer patients from healthy individuals with robust cross-validation and patient-level aggregation.

---

## 📂 Repository Structure

- `DNN_Model.py`: Deep Neural Network (DNN) architecture utilizing wavelet-transformed cfDNA fragment profiles.
- `GCN_Maker.py`: Graph construction utility (K-Nearest Neighbors / similarity graphs based on profile features and patient metadata).
- `GCN_Model.py`: Graph Convolutional Network (GCN) implementations for relational fragmentomic feature learning.
- `Adaptive_Fusion.py`: Ensemble fusion framework combining DNN, GCN, and base learners using adaptive residual gating and validation-optimized log-loss minimization.

---

## 🔬 Dataset Overview

This repository is designed to process genome-wide cell-free DNA (cfDNA) fragmentation profiles (such as plasma WGS fragment-size distributions and ratios derived from standards like Cristiano *et al.* / DeepFRAG benchmarks). 

- **Input Features:** Normalized fragment-size distribution profiles (focusing on characteristic lengths such as 51–250 bp, covering nucleosomal peaks and oscillatory 10-bp periodicities).
- **Metadata Format:** Patient-level identifiers, split assignments, and binary cancer status labels.
---

## ⚙️ Installation & Requirements

Clone the repository and install the required dependencies (TensorFlow, PyTorch, Scikit-Learn, NumPy, SciPy, Pandas):
```bash
git clone https://github.com/MojtabaRiahi/DualFrag.git
cd YOUR_REPOSITORY
pip install -r requirements.txt
