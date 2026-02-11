# Pneumonia Detection from Chest X-Rays (Training from Scratch)

## 1. Motivation
This project implements and trains a ResNet-style CNN **from scratch** for the purposes of classification of pneumonia identifiable in chest x-ray images. Chest X-rays were chosen for a number of reasons. The dataset provides a number of real world constraints, such as class imbalance, potentials of data-leak

The Chest X-Ray dataset provides many real world problems and constraints, including noisy labels, class imbalance, 

The goal is not to achieve SOTA accuracy, but to demonstrate a clear understanding of:

- Training dynamics and optimization
- Regularization and overfitting
- Reproducibility
- Reason about data bias and shortcuts
- Failure modes and diagnostics

## 2. Dataset & Subset Construction

The **NIH ChestX-ray14** dataset (https://www.kaggle.com/datasets/nih-chest-xrays/data)

The NIH Chest X-Ray dataset is comprised of 112,120 X-ray images with disease labels from 30,805. The labels are extracted from the associated radiological reports using NLP. Labels are expected to be >90% accurate.

Instead of tackling the full multi-label classification problem, we chose to make this a binary classification problem: does a given image contain a specific pathology (in this case we chose Pneumonia). We also chose to work on a much smaller subset, of 8000 images in total:
1431 positive cases, and 6569 negative cases.

This subset intentionally excludes “No Finding” images to avoid trivial negatives and encourage learning discriminative pathology features. As a result, reported metrics should be interpreted as conditional on abnormal studies only.

The dataset is intentionally imbalanced (≈1:5) to reflect real-world pathology prevalence. Rather than artificially balancing the data, the project focuses on appropriate loss weighting, metric selection, and threshold tuning.

## 3. Problem Statement

The goal here is Binary Classification with weak labels in a clinically realistic setting.

## 4. Preprocessing and Augmentation

- Gray Scale Images:
    - Grayscale images are left as grayscale, duplicating channels does not provide any new information for the network 
    - RGBA images are converted to masked grayscale, because they are naturally grayscale images duplicated to 3 channels.
- Image Resizing: Constant size (224x224)
- Augmentations Used:
    - Horizontal Flips: Pneumonia can appear in either lung, and rare cases of *Situs inversus* (0.01% of the population) do exist
    - Small Rotations (±7°): To account for posture
    - Mild Contrast Jitter: To account for differences in X-ray machines
- Augmentations Avoided:
    - Vertical Flips: X-rays are not taken of patients upside-down
    - Aggressive Crops: Patients are centered for the X-rays

## 5. Model Architecture

We chose a basic **ResNet18** architecture, adapted for grayscale images
- The 1st convolution takes single-channel inputs
- The output is the raw output logits of the final Linear layer.

## 6. Training Setup

- **Optimizer:** SGD + Momentum
- **LR:** ---
- **Batch Size:** 128
- **Loss:** Binary Cross Entropy
- **Class Weighting:** ---


## 7. Evaluation Protocol

We report the following metrics:
- Accuracy (threshold 0.5)
- ROC-AUC score
- Average Precision
- Confusion Matrix
- optimal threshold
- precision @ optimal threshold
- sensitivity @ optimal threshold

The optimal threshold is calculated by taking the desired specificity (determined in the config) and determining the threshold which gives us the resulting False Positive Rate (FPR).

Our test set consists of 1600 X-rays of patients who do not appear in either other dataset (patient level split)

During training, we monitor loss and threshold-independent metrics (ROC-AUC, PR-AUC) to track learning dynamics. Decision-level metrics such as sensitivity and specificity are evaluated only on the held-out test set after threshold selection.

## 8. Main Results

## 9. Key Experiments & Findings

## 10. Failure Analysis

## 11. Limitations

## 12. Reproducibility

## 13. What I Would Do Next