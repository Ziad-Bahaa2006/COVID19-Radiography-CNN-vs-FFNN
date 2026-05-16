# 🫁 COVID-19 Chest X-Ray Classification
### CNN vs FFNN — Neural Networks Project

> **Faculty of Computers & Data Science** | Course: Neural Networks

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?logo=keras&logoColor=white)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Preprocessing Pipeline](#️-preprocessing-pipeline)
- [Model Architectures](#-model-architectures)
- [Training & Results](#-training--results)
- [Evaluation Metrics](#-evaluation-metrics)
- [Team](#-team)
- [How to Run](#-how-to-run)

---

## 🔬 Project Overview

This project applies deep learning to **automated COVID-19 diagnosis from chest X-ray images**. Two neural network architectures are implemented and compared:

| Model | Framework | Purpose |
|-------|-----------|---------|
| **CNN** | TensorFlow / Keras | Convolutional feature extraction & classification |
| **FFNN** | PyTorch | Hybrid conv + fully-connected classification |

Both models classify radiography images into **4 diagnostic categories** and are evaluated using a comprehensive suite of metrics to determine which architecture performs better on this medical imaging task.

### Objectives

- ✅ Build and train a **CNN** model for multi-class chest X-ray classification
- ✅ Build and train an **FFNN** model using the same dataset and preprocessing pipeline
- ✅ Perform comprehensive **Exploratory Data Analysis (EDA)** and quality assessment
- ✅ Apply **augmentation and normalization** for improved generalization
- ✅ Evaluate both models using **Accuracy, Precision, Recall, F1-Score, and AUC-ROC**
- ✅ Analyze **learning curves**, overfitting/underfitting behavior, and regularization effects
- ✅ Compare **training speed and inference time** between architectures

---

## 📦 Dataset

**COVID-19 Radiography Dataset** — sourced from [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

| Class | Description | Count |
|-------|-------------|-------|
| `COVID` | COVID-19 positive chest X-rays | ~3,616 |
| `Normal` | Healthy lungs with no pathology | ~10,192 |
| `Lung Opacity` | Non-COVID lung opacity conditions | ~6,012 |
| `Viral Pneumonia` | Viral pneumonia (non-COVID) | ~1,345 |

- **Total Images:** ~42,165 (after quality checks)
- **Format:** PNG grayscale, variable resolutions
- **Class Encoding:** `COVID=0`, `Lung_Opacity=1`, `Normal=2`, `Viral Pneumonia=3`

> ⚠️ **Class Imbalance Note:** The `Normal` class is ~2.8× larger than `COVID` and ~7.6× larger than `Viral Pneumonia`. Stratified splitting was used to mitigate this.

---

## 📁 Project Structure

```
COVID19-Radiography-CNN-vs-FFNN/
│
├── data_preprocessing_01.ipynb        # Data pipeline & preprocessing
├── eda_and_documentation_02.ipynb     # Exploratory Data Analysis (EDA)
├── CNN.ipynb                          # CNN model (TensorFlow/Keras)
├── FFNN4.ipynb                        # FFNN model (PyTorch)
├── task6_training_analysis.ipynb      # Learning curves & training analysis
├── 05_evaluation_metrics.ipynb        # Metrics, confusion matrices & ROC curves
│
├── COVID-19_Radiography_Dataset/      # Dataset directory (not tracked in git)
│   ├── COVID/
│   ├── Normal/
│   ├── Lung_Opacity/
│   └── Viral Pneumonia/
│
├── cnn_model.h5                       # Saved CNN weights
├── ffnn_model.pth                     # Saved FFNN weights
└── README.md
```

---

## ⚙️ Preprocessing Pipeline

Implemented using `torchvision.transforms`. Separate pipelines for train and val/test sets ensure unbiased evaluation.

| Step | Details |
|------|---------|
| **Grayscale Conversion** | RGB → single-channel grayscale |
| **Resize** | All images → `128×128` pixels |
| **Random Rotation** *(Train only)* | ±10° stochastic rotation |
| **Random H-Flip** *(Train only)* | 50% horizontal flip probability |
| **Normalization** | `mean=0.5, std=0.5` → range `[−1, 1]` |
| **Dataset Split** | Stratified 60% / 20% / 20% (`random_state=42`) |
| **Batch Size** | 32 images per batch |

### Split Summary

```
Train:      25,329 images  (60%) — gradient updates
Validation:  8,443 images  (20%) — hyperparameter tuning
Test:        8,444 images  (20%) — final evaluation only
```

---

## 🧠 Model Architectures

### CNN — TensorFlow / Keras

Classic feature-extraction design with three convolutional blocks, progressively increasing filters (32 → 64 → 128).

```
Input (128×128×1)
    └─ Conv2D(32, 3×3, ReLU) → MaxPool2D → 64×64×32
    └─ Conv2D(64, 3×3, ReLU) → MaxPool2D → 32×32×64
    └─ Conv2D(128, 3×3, ReLU) → MaxPool2D → 16×16×128
    └─ Flatten (32,768 units)
    └─ Dropout(0.5)
    └─ Dense(128, ReLU)
    └─ Output Softmax(4 classes)
```

| Config | Value |
|--------|-------|
| Optimizer | Adam, lr=0.001 |
| Loss | Sparse Categorical Cross-Entropy |
| Epochs | 5 (initial) / 10 (analysis) |

---

### FFNN — PyTorch

Hybrid design: convolutional blocks for spatial feature extraction + fully-connected layers for classification. Key distinction: **Adaptive Average Pooling** fixes spatial dims to `8×8` before flattening.

```
Input (128×128×1)
    └─ Conv2d(1→32) + ReLU + MaxPool2d + Dropout2d(0.25)
    └─ Conv2d(32→64) + ReLU + MaxPool2d + Dropout2d(0.25)
    └─ Conv2d(64→128) + ReLU
    └─ AdaptiveAvgPool2d(8×8) → Flatten (8,192 units)
    └─ Linear(8192→512) + BatchNorm1d + ReLU + Dropout(0.5)
    └─ Linear(512→128) + BatchNorm1d + ReLU + Dropout(0.5)
    └─ Linear(128→4) — logits
```

| Config | Value |
|--------|-------|
| Optimizer | Adam, lr=0.001 |
| Loss | CrossEntropyLoss (includes LogSoftmax) |
| Epochs | 5 (initial) / 10 (analysis) |

---

## 📈 Training & Results

### FFNN Training Progress (5 Epochs, CPU)

| Epoch | Loss | Train Accuracy |
|-------|------|----------------|
| 1 | 1.0060 | 56.63% |
| 2 | 0.8948 | 62.33% |
| 3 | 0.8375 | 64.86% |
| 4 | 0.8065 | 66.52% |
| 5 | 0.7763 | **68.40%** |

Consistent downward loss trend and upward accuracy — effective learning with no gradient explosion observed.

### Regularization Techniques

| Technique | Applied To | Purpose |
|-----------|-----------|---------|
| `Dropout(p=0.5)` | FC layers (both models) | Prevents co-adaptation of neurons |
| `Dropout2d(p=0.25)` | Conv blocks (FFNN) | Regularizes feature maps spatially |
| `BatchNorm1d` | FC head (FFNN) | Reduces internal covariate shift |
| Data Augmentation | Training set only | Improves spatial invariance |

---

## 📊 Evaluation Metrics

Both models evaluated on the **held-out test set (8,444 images)** using identical splits.

| Metric | Description |
|--------|-------------|
| **Accuracy** | Proportion of correctly classified samples |
| **Precision (Macro)** | Average precision across all 4 classes equally |
| **Recall (Macro)** | Average recall — critical for minimizing false negatives |
| **F1-Score (Macro)** | Harmonic mean of precision & recall |
| **F1-Score (Weighted)** | F1 weighted by class support |
| **AUC-ROC (Macro OvR)** | Discrimination ability across all class pairs |

> 🏥 **Clinical Note:** Misclassifying COVID-19 as Normal (false negative) is particularly critical. AUC > 0.90 per class is generally considered acceptable for clinical screening.

Evaluation artifacts generated:
- ✅ Normalized confusion matrices (raw counts + percentages)
- ✅ Per-class precision / recall / F1-score bar charts
- ✅ ROC curves (One-vs-Rest) with AUC per class

---

## 👥 Team

| Name | Student ID | Role |
|------|-----------|------|
| **Ziad Bahaa Elsayed Taha** | 2405720 | EDA, Technical Documentation & GitHub Repository Management |
| **Abd Allah Ahmed Fawzy** | 2405709 | Model Architecture Design (CNN) |
| **Mohamed Islam Ibrahim** | 2405736 | Training Analysis & Optimization |
| **Ahmed Yasser Hassen** | 2405702 | Evaluation Metrics & Performance Analysis |
| **Mohamed Safwat Basiony** | 2405741 | Model Architecture Design (FFNN) |
| **Mohamed Ahmed Saied** | 2405727 | Data Preprocessing & Pipeline Engineering |

---

## 🚀 How to Run

### Prerequisites

```bash
pip install torch torchvision tensorflow keras scikit-learn matplotlib seaborn numpy pandas jupyter
```

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Ziad-Bahaa2006/COVID19-Radiography-CNN-vs-FFNN.git
cd COVID19-Radiography-CNN-vs-FFNN

# 2. Download the dataset from Kaggle and place it in:
#    ./COVID-19_Radiography_Dataset/

# 3. Run notebooks in order:
#    data_preprocessing_01.ipynb
#    eda_and_documentation_02.ipynb
#    CNN.ipynb
#    FFNN4.ipynb
#    task6_training_analysis.ipynb
#    05_evaluation_metrics.ipynb
```

> **GPU Recommended:** The CNN model benefits significantly from GPU acceleration. FFNN was tested on CPU; training on GPU will substantially reduce epoch time.

---

## 🔮 Future Work

- [ ] Transfer learning from pre-trained models (ResNet, EfficientNet, DenseNet)
- [ ] Class-weighted loss functions to address imbalance more aggressively
- [ ] Ensemble methods combining CNN and FFNN predictions
- [ ] Grad-CAM visualizations for model interpretability
- [ ] Extended training (20+ epochs) on GPU hardware

---

<div align="center">

**Faculty of Computers & Data Science — Neural Networks Course**

Made with ❤️ for medical AI research

</div>
