# Pneumonia Detection from Chest X-Rays (Training from Scratch)

## 1. Motivation

This project implements and trains a ResNet-style CNN **from scratch** for binary classification of pneumonia in chest X-ray images. The dataset was chosen deliberately for the real-world constraints it imposes: noisy weak labels, significant class imbalance, and a small sample size relative to the complexity of the task.

The goal is not to achieve state-of-the-art accuracy, but to demonstrate a clear understanding of:

- Training dynamics and optimization
- Class imbalance and loss landscape analysis
- Failure mode diagnosis
- Principled experiment design and iteration

## 2. Dataset & Subset Construction

The **NIH ChestX-ray14** dataset ([Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)) contains 112,120 frontal-view X-ray images from 30,805 patients, with disease labels extracted from radiology reports via NLP. Labels are estimated to be >90% accurate but are known to contain noise.

We work on a **binary subset of 8,000 images** focusing on the Pneumonia label:

| Split | Total | Positive (Pneumonia) | Negative |
|-------|-------|----------------------|----------|
| Train | 5,120 | ~910 | ~4,210 |
| Val   | 1,280 | ~230 | ~1,050 |
| Test  | 1,600 | ~290 | ~1,310 |

The class ratio is approximately **1:5 positive to negative**, reflecting realistic clinical prevalence. Splits are performed at the **patient level** to prevent data leakage.

The subset excludes "No Finding" images to avoid trivially easy negatives and to focus the model on discriminating between pathologies. Reported metrics should be interpreted as conditional on abnormal studies only.

## 3. Problem Statement

Binary classification: given a chest X-ray, predict whether the image shows signs of pneumonia. The output is a single logit; probability is recovered via sigmoid. Decision threshold is selected post-hoc to match a target specificity.

## 4. Preprocessing and Augmentation

**Preprocessing:**
- All images converted to grayscale (single channel). Duplicating channels for a 3-channel model adds no information.
- Resized to 224×224.
- Normalized using training-set statistics: mean = 0.4913, std = 0.2494.

**Augmentations applied during training:**
- Random horizontal flip (p=0.5): pneumonia can appear in either lung.
- Small random rotations (±7°): accounts for patient posture variation.
- Random crop with padding=4: mild translation invariance.

**Augmentations deliberately avoided:**
- Vertical flips: X-rays are always taken with the patient upright.
- Aggressive crops: clinically relevant findings appear at specific anatomical locations.
- Color jitter beyond mild contrast: grayscale images have limited color information.

## 5. Model Architecture

A **ResNet-18** implemented from scratch (no pretrained weights):
- First convolution adapted for single-channel (grayscale) input.
- Standard residual blocks with BatchNorm.
- Global average pooling into a single linear output (raw logit, no sigmoid).
- Kaiming initialization on conv/linear layers; zero-init on final residual BN (He et al.).

## 6. Training Setup

| Component | Configuration |
|-----------|---------------|
| Optimizer | AdamW (weight decay decoupled from 1D params and biases) |
| Learning rate | 3e-4 with cosine decay to 1e-6 |
| Warmup | 5 epochs, linear from 3e-7 |
| Batch size | 32 |
| Epochs | 100–300 (experiment-dependent) |
| Loss | Binary focal loss with prior-probability initialization |
| Class imbalance | `pos_weight` or focal loss; weighted sampling explored |
| BatchNorm | Frozen to eval() during training (unreliable stats on small imbalanced batches) |

## 7. Evaluation Protocol

All decision-level metrics are computed at a **threshold selected from the ROC curve** to achieve a target specificity (90%), not at 0.5. This reflects the clinical priority of avoiding false positives.

Metrics reported:
- **ROC-AUC** — threshold-independent ranking quality (primary metric)
- **Average Precision** — threshold-independent, reflects class imbalance
- **Sensitivity and Precision** at the optimal threshold
- **Confusion matrix**

Training metrics (loss, accuracy, ROC-AUC) are logged every epoch. `best.pt` is saved on peak validation ROC-AUC.

## 8. Main Results

Best from-scratch result (Experiment 8: focal loss + prior init):

| Metric | Value |
|--------|-------|
| Test ROC-AUC | 0.554 |
| Test Average Precision | 0.220 |
| Sensitivity @ 90% specificity | 0.194 |
| Precision @ 90% specificity | 0.100 |

Best overall result (Experiment 10: ImageNet pretrained, focal loss, LR 1e-4):

| Metric | Value |
|--------|-------|
| Test ROC-AUC | 0.586 |
| Test Average Precision | 0.238 |
| Sensitivity @ 90% specificity | 0.278 |
| Precision @ 90% specificity | 0.160 |

Neither result is clinically useful (diagnostic-grade models reach ROC-AUC ≥ 0.85 on this pathology). See Section 10 for why.

## 9. Key Experiments & Findings

### Experiment 4 — Naive BCE, no class weighting
**Setup:** Standard BCE, no pos_weight, no sampling correction.
**Result:** Loss converged to ~0.451 and did not decrease further. Model predicted all negatives.
**Finding:** With a 1:6 positive rate, BCE is minimized when the model outputs a constant logit of `log(1/5) ≈ -1.609` (p ≈ 0.167) for every sample. The gradient at this point is zero — the model learned the class prior and stopped. ROC-AUC ≈ 0.51.

### Experiment 5 — Weighted random sampling
**Setup:** WeightedRandomSampler to produce 50/50 batches; no loss weighting.
**Result:** Loss plateaued at ~0.693 (log(2)). Model predicted ~50% positive rate but did not discriminate.
**Finding:** With balanced batches and unweighted BCE, the trivial minimum shifts to logit=0 (p=0.5). The gradient is again zero at this point. Positive and negative gradients cancel exactly, so the model cannot escape a constant prediction without stronger feature signal.

### Experiment 6 — BCE + pos_weight (extended to 300 epochs)
**Setup:** `pos_weight = num_neg / num_pos ≈ 5`, no weighted sampling. Extended from 100 to 300 epochs after observing that the 100-epoch run plateaued due to cosine LR decay.
**Result:** ROC-AUC slowly improved from 0.51 to ~0.55 over 300 epochs.
**Finding:** `pos_weight = num_neg/num_pos` exactly compensates the class imbalance, making the gradient zero at logit=0 — the same trivial minimum as Experiment 5, just with a higher numerical loss (~1.155). Feature gradients are non-zero but tiny, so learning is slow. The plateau at 100 epochs was a LR artifact; the model continued to improve at higher LR.

### Experiment 7 — Focal loss, no prior init
**Setup:** Focal loss (γ=2, no α), no prior probability initialization.
**Result:** No improvement over BCE experiments. ROC-AUC ≈ 0.51.
**Finding:** Without prior initialization, the model starts at logit=0 (p=0.5), which is also the trivial minimum of focal loss without α weighting. The focal weights at p=0.5 are symmetric, so gradients cancel.

### Experiment 8 — Focal loss + prior probability initialization
**Setup:** Focal loss (γ=2) with the output bias initialized to `−log(num_neg/num_pos) ≈ −1.609`. This sets the model's initial prediction to the class prior (p ≈ 0.167), making focal weights strongly asymmetric from epoch 0: high weight on hard positives, near-zero weight on easy negatives.
**Result:** Train loss started at 0.240 (consistent with the theoretical focal loss at the prior) rather than collapsing to 0.127 immediately. Best test ROC-AUC: 0.554.
**Finding:** Prior init breaks the symmetry of the focal loss at initialization and is the most effective single change for from-scratch training on this dataset. It does not fully solve the problem — the model still eventually finds a lower-loss constant predictor — but it gives feature gradients the most opportunity to develop before the bias dominates.

### Experiment 9 — ImageNet pretrained ResNet-18
**Setup:** Torchvision ResNet-18 with default ImageNet weights. First conv adapted by averaging the 3-channel pretrained weights to 1-channel. FC replaced with `Linear(512, 1)`. LR 3e-4.
**Result:** Test ROC-AUC: 0.569. Early training extremely unstable (val ROC-AUC oscillating between 0.19 and 0.80 in epochs 0–20).
**Finding:** LR of 3e-4 disrupted pretrained features before they could adapt. The instability caused the model to partially destroy its pretrained representations and recover to a suboptimal state.

### Experiment 10 — ImageNet pretrained + focal + prior init + lower LR
**Setup:** Same as Experiment 9 but with focal loss, prior init, LR 1e-4, weight decay 1e-2.
**Result:** Test ROC-AUC: 0.586. Training stable; late-epoch val ROC-AUC consistently 0.576–0.581 — a genuine convergence plateau, not a LR artifact.
**Finding:** Lower LR stabilized training. The model converged, but the plateau at 0.58 reflects the limited transfer of ImageNet features to grayscale X-rays, not an optimization failure. ImageNet pretraining provided only +0.036 AUROC over from-scratch.

### Experiment 11 — Two-stage fine-tuning (backbone frozen → unfrozen)
**Setup:** Backbone frozen for 20 epochs (only FC trains), then unfrozen with differential LR (backbone at 1/10 head LR). Intent: force the model to find discriminative features in stage 1 before the bias can collapse.
**Result:** Test ROC-AUC: 0.548. Val ROC-AUC during stage 1 only reached 0.477 by epoch 19.
**Finding:** ImageNet features, applied as-is to grayscale X-rays, provide almost no discriminative signal. The FC head had 20 epochs to find the best linear combination of frozen features and barely exceeded random. When the backbone unfroze, the head was tuned to unhelpful features and performance initially dropped further. This confirms that the domain gap — not the optimization procedure — is the binding constraint.

## 10. Failure Analysis

### The trivial minimum problem
Every standard loss function for binary classification with class imbalance has a **constant-predictor local minimum**: a logit value at which positive and negative gradients exactly cancel, leaving the model stuck predicting the same value for every input. The location depends on the loss configuration:

| Configuration | Trivial minimum |
|---------------|-----------------|
| BCE, no weighting | p = class prior ≈ 0.167 |
| BCE + pos_weight = 5 | p = 0.5 |
| Weighted sampling, unweighted BCE | p = 0.5 |
| Focal (γ=2), no α, 1:6 ratio | p ≈ 0.33 |

With a bias term in the output layer, the model can always express a constant predictor without using any features. Because the bias gradient is large relative to feature gradients at random initialization, the model races to the trivial minimum in 2–5 epochs. Feature learning then stalls.

Prior probability initialization (Experiment 8) is the most effective mitigation: it starts the model at the class prior, where focal weights are maximally asymmetric, giving feature gradients a window to develop before the bias reaches equilibrium.

### The domain gap problem
ImageNet pretraining provides features optimized for natural color images of everyday objects. Chest X-rays are:
- Grayscale (1 channel, averaged from 3-channel pretrained weights)
- Medical in nature — relevant features are opacity patterns, consolidation, and airspace density
- Very different from the high-level features (textures, object parts) that differentiate ImageNet classes

The result is that ImageNet-pretrained features transfer poorly. Stage-1 fine-tuning (Experiment 11) with frozen backbone only reached ROC-AUC 0.477, confirming the pretrained features are nearly uninformative for this task before adaptation. Models pretrained on chest X-ray data (e.g., CheXNet, REMEDIS) would not have this problem.

### The dataset size problem
CheXNet (Rajpurkar et al., 2017) — the benchmark model for NIH ChestX-ray14 — used 112,120 images and DenseNet-121. This project uses 8,000 images and ResNet-18 from scratch. The label noise inherent in NLP-extracted annotations means a larger dataset is needed to average out noise and learn reliable features.

## 11. Limitations

- **Label quality:** NIH labels are extracted from radiology reports via NLP and are known to be imperfect. A model trained on these labels has an inherent ceiling set by label noise.
- **Dataset size:** 8,000 images is insufficient to learn robust representations from scratch for a task with this degree of visual subtlety.
- **No medical pretraining:** ImageNet pretraining adds marginal value. Domain-specific pretraining (chest X-ray datasets) would be required to meaningfully leverage transfer learning.
- **Single architecture:** Only ResNet-18 was explored. Vision Transformers or DenseNets may have different inductive biases better suited to this task.
- **Evaluation subset:** "No Finding" images were excluded. Real clinical deployment would include them, making the problem harder.

## 12. Reproducibility

All experiments are fully reproducible:
- Random seeds set for Python, NumPy, PyTorch, and CUDA.
- Data splits are fixed CSVs at the patient level (no leakage).
- Exact config snapshot saved as `config_used.yaml` in each experiment directory.
- Per-epoch metrics logged to `log.csv`; test metrics to `metrics.csv`.

To reproduce:
```bash
source .venv/bin/activate
python train.py --config configs/config.yaml
```

## 13. What I Would Do Next

1. **Medical-domain pretraining.** Use weights pretrained on chest X-ray datasets (e.g., MIMIC-CXR, CheXpert). These features would transfer directly and eliminate the domain gap that limited experiments 9–11.
2. **Larger dataset.** Use the full NIH ChestX-ray14 dataset (~112k images) or CheXpert (~224k). At 8k images, the fundamental bottleneck is data, not architecture or optimization.
3. **Better labels.** The NIH labels are NLP-extracted. Hand-annotated datasets like CheXpert (with radiologist-reviewed labels) or VinDr-CXR would provide a cleaner signal.
4. **Multi-label formulation.** Training jointly on all 14 pathologies rather than one binary task may provide better shared representations and more gradient signal per image.
