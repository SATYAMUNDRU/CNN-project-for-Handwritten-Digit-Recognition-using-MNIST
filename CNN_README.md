# Handwritten Digit Recognition Using CNN

**Course:** CS5720 Neural Network and Deep Learning (Spring 2025)  
**University:** University of Central Missouri

**Team Members:**
- Lokesh Reddy Siripireddy (700758270)
- Sony Pailla (70076544)
- Mundru Satya Sri Lasya (700758682)
- Jasmitha Nalla (700799225)

---

## Overview

This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits (0–9) from the MNIST dataset. The model achieves **>98% test accuracy** by learning spatial hierarchies of features through convolutional and pooling layers — with no manual feature engineering.

---

## Applications

| Domain | Use Case |
|--------|----------|
| Banking & Finance | Automated cheque processing |
| Form Digitization | Handwritten form → digital data (OCR) |
| Security Systems | Handwritten PIN / code authentication |

---

## Dataset — MNIST

| Split | Samples | Format |
|-------|---------|--------|
| Training | 60,000 | 28×28 grayscale |
| Testing | 10,000 | 28×28 grayscale |
| Classes | 10 | Digits 0–9 |

---

## CNN Architecture

```
Input (28×28×1)
    ↓
Conv2D — 32 filters, 3×3, ReLU        → (26, 26, 32)
MaxPooling2D — 2×2                     → (13, 13, 32)
    ↓
Conv2D — 64 filters, 3×3, ReLU        → (11, 11, 64)
MaxPooling2D — 2×2                     → (5, 5, 64)
    ↓
Flatten                                → (1600,)
Dense — 128 units, ReLU               → (128,)
Dropout — rate 0.5
    ↓
Output Dense — 10 units, Softmax      → (10,)
```

**Total Parameters:** ~1.1 million  
**Optimizer:** Adam | **Loss:** Categorical Cross-Entropy

---

## Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~99% |
| Test Accuracy | ~98.5% |
| Loss (Test) | Consistently decreasing |

---

## Requirements

```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy
```

---

## How to Run

**Option 1 — Google Colab (recommended):**  
Upload and open `Handwritten_Digit_Recognition_CNN.ipynb` directly in Colab.

**Option 2 — Local:**
```bash
git clone https://github.com/SATYAMUNDRU/Handwritten-Digit-Recognition-CNN
cd Handwritten-Digit-Recognition-CNN
jupyter notebook Handwritten_Digit_Recognition_CNN.ipynb
```

---

## Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

---

## References

- LeCun, Y. et al. (1998). Gradient-based learning applied to document recognition.
- [TensorFlow & Keras Documentation](https://www.tensorflow.org/api_docs)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- Deep Learning with Python — François Chollet
