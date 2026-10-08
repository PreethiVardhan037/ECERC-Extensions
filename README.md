# Improving Robustness and Reliability of ECERC for Multi-Modal Emotion Recognition in Conversation

A modified and extended implementation of **ECERC (Evidence-Cause Attention Network for Multi-Modal Emotion Recognition in Conversation)**, based on the official implementation released by the original authors.

Our project identifies and addresses four concrete limitations of the original ECERC implementation while preserving its core architecture and evaluation methodology.

---

## 📌 Project Overview

**Multi-Modal Emotion Recognition in Conversation (MMERC)** aims to identify the emotion of a speaker using multiple modalities such as:

- Text
- Audio
- Visual information
- Conversation context

Our project is based on **ECERC: Evidence-Cause Attention Network for Multi-Modal Emotion Recognition in Conversation**, published at **ACL 2025**.

The original ECERC model uses four types of emotional causes:

1. Self-event
2. Cross-event
3. Self-emotion / emotional contagion
4. Cross-emotion

These are processed through the original ECERC pipeline:

```text
Text / Audio / Visual Features
            ↓
      Evidence Gating
            ↓
       Cause Encoding
            ↓
  Evidence-Cause Interaction
            ↓
       Feature Gating
            ↓
      Classification
```

Our objective was **not to redesign ECERC**, but to improve its robustness, reliability, and handling of specific failure cases identified in the original paper and through our code audit.

---

# 📚 Original ECERC Repository

We started this project from the **official ECERC implementation released by the original authors**.

**Original repository:** `TAN-OpenLab/ECERC`

The original implementation contains the model, loss functions, dataloaders, training scripts, pretrained checkpoints, and preprocessing pipeline used for the IEMOCAP and MELD datasets.

We first studied and ran the original implementation as the **baseline**. We then audited the implementation against the limitations discussed in the original paper and identified four areas for improvement.

The original code was retained as the foundation of our project. Our modifications were implemented as additional modules, flags, loss functions, calibration scripts, benchmarking scripts, and adaptive attention mechanisms.

> **Important:** The original ECERC architecture and its major components were not replaced. Our modifications are designed to be additive and independently switchable.

---

# 🔬 Project Objective

The project addresses the following four limitations:

| Limitation | Problem | Solution | Owner |
|---|---|---|---|
| L1 | Class imbalance | Class-Balanced Loss + Focal Loss | N. Sravan Kumar |
| L2 | Emotion category confusability | Confusion-Aware Label Smoothing | T. Preethi Vardhan |
| L3 | No confidence calibration | Temperature Scaling + MC-Dropout | M. Uday |
| L4 | Unbounded full-history attention | Adaptive / Learnable Attention Span | All Members |

The first three limitations are individually owned by one team member each. Limitation 4 is shared because it directly modifies the core attention computation and requires joint implementation, benchmarking, and evaluation.

---

# 👥 Team Members and Contributions

## 1. N. Sravan Kumar

**Roll Number:** B24CS032  
**Email:** narepalemsk@iitbhilai.ac.in

### GitHub

`https://github.com/Sravan-n-dev`


### Main Contribution

N. Sravan Kumar is responsible for:

- **Limitation 1 – Class Imbalance**
- Core implementation of **Limitation 4 – Adaptive Attention Span**
- Loss-function modifications
- Per-class evaluation
- Confusion-matrix reporting
- Integration of adaptive span across ECERC's four Evidence-Cause Interaction branches

### Limitation 1 – Class Imbalance

The original ECERC implementation suffers from severe class imbalance, particularly in MELD.

For example, the SOP identifies substantially weaker performance for minority classes such as:

- Fear
- Disgust

compared with the majority Neutral class.

This creates a situation where the model can achieve good overall performance while performing poorly on minority emotions.

### Solution

We use:

**Class-Balanced Loss + Focal Loss**

Class-balanced weighting is calculated using the effective number of samples instead of simply using raw inverse-frequency weighting.

Focal Loss further reduces the contribution of easy examples and gives more importance to difficult examples.

Conceptually:

```text
Majority / Easy Examples
          ↓
     Lower influence

Minority / Hard Examples
          ↓
     Higher influence
```

### Code Contributions

Main changes were made to:

```text
MELD/loss.py
MELD/train.py

IEMOCAP/loss.py
IEMOCAP/train.py
```

The implementation provides:

- Automatic class-count calculation
- Effective-number class weights
- Focal-loss modulation
- Configurable focal-loss gamma
- Per-class F1 scores
- Confusion matrices

---

# 2. T. Preethi Vardhan

**Roll Number:** B24CS046  
**Email:** tanakalapv@iitbhilai.ac.in

### GitHub

`https://github.com/PreethiVardhan037`

> Replace the placeholder above with the actual GitHub repository URL.

### Main Contribution

T. Preethi Vardhan is responsible for:

- **Limitation 2 – Category Confusability**
- Limitation 4 benchmarking
- Attention-span experiments on IEMOCAP and MELD
- Dataloader-related modifications required for experiments
- Accuracy/F1 comparison against learned attention spans

### Limitation 2 – Category Confusability

Some emotions are semantically close to one another, making them difficult for the model to distinguish.

The important confusable pairs identified in the project are:

```text
Happy      ↔ Excited
Angry      ↔ Frustrated
Disgust    ↔ Angry
```

The original model can confuse these categories because they can have similar linguistic, audio, and visual characteristics.

### Solution

We use:

**Confusion-Aware Label Smoothing**

Instead of distributing label-smoothing probability uniformly across all incorrect classes, additional probability mass is assigned to the known confusable emotion.

For example, when the correct class is one of a known confusable pair:

```text
True class
    ↓
High probability

Known confusable class
    ↓
Higher smoothing probability

Other classes
    ↓
Smaller remaining probability
```

This allows the model to learn a more meaningful decision boundary between emotions that are frequently confused.

### Code Contributions

Main changes were made to:

```text
MELD/loss.py
MELD/train.py

IEMOCAP/loss.py
IEMOCAP/train.py
```

Additional benchmarking functionality was added to compare model performance against attention span.

---

# 3. M. Uday

**Roll Number:** B24CS025  
**Email:** mamidium@iitbhilai.ac.in

### GitHub
`https://github.com/udaymanikanta123-byte`



### Main Contribution

M. Uday is responsible for:

- **Limitation 3 – Confidence Calibration**
- Post-hoc calibration experiments
- Expected Calibration Error evaluation
- Temperature Scaling
- Monte-Carlo Dropout
- Calibration analysis for Limitation 4
- Comparison of calibration quality between full attention and adaptive attention

---

## Limitation 3 – No Confidence Calibration

The original ECERC model produces a final prediction but does not provide a reliable measure of how confident the model should be in that prediction.

A model can therefore produce:

```text
Prediction: Angry
Confidence: 99%
```

even when the prediction is uncertain or incorrect.

High classification accuracy alone does not guarantee that the predicted probabilities are reliable.

### Solution

We use two post-hoc calibration techniques:

### 1. Temperature Scaling

Temperature Scaling adjusts the model's output probabilities using a learned temperature parameter.

It does not require retraining the entire ECERC model.

The purpose is to make the predicted probabilities better match the actual likelihood of being correct.

### 2. Monte-Carlo Dropout

MC-Dropout performs multiple stochastic forward passes with dropout enabled.

The resulting predictions are combined to estimate uncertainty.

### Evaluation

We evaluate calibration using:

- Expected Calibration Error (ECE)
- Negative Log-Likelihood (NLL)
- Brier Score
- Reliability diagrams
- Confusable-class calibration

A new script was added:

```text
calibrate.py
```

---

# 4. Shared Contribution – Limitation 4

## Unbounded Full-History Attention

All three team members jointly worked on Limitation 4.

The original attention mechanism can attend to the entire available conversation history.

For long conversations, this means that the model can consider a very large context even when older utterances may no longer be useful.

This can result in:

- Unnecessary context
- No explicit learned context boundary
- Larger effective receptive fields
- Difficulty determining how much conversation history is actually useful

---

## Solution – Adaptive / Learnable Attention Span

We introduce a **learnable attention span** inspired by Adaptive Attention Span.

Instead of always attending to the entire history, the model learns how far back each attention head should effectively attend.

The mechanism is applied to the four Evidence-Cause Interaction branches:

```text
Self-Event
     ↓
Cross-Event
     ↓
Self-Emotion / Contagion
     ↓
Cross-Emotion
```

The learned span is optimized during training.

Conceptually:

```text
Entire Conversation
───────────────────────────────────
                         ↑
                  Target utterance

Without L4:
<---------------------------------->
        Entire history

With L4:
              <------------>
              Learned span
```

### Important implementation detail

The adaptive span limits the **effective receptive field** of attention.

It does **not** currently reduce the actual `O(L²)` attention tensor computation or memory usage because the full attention matrix is still calculated before the learned masking is applied.

Therefore, this project describes L4 as:

> **Learned effective context per attention branch/head**

rather than claiming that it reduces computational complexity.

---

# 📊 Limitation 4 – Individual Responsibilities

Although Limitation 4 is shared, each member has a specific responsibility.

### N. Sravan Kumar

Implemented the adaptive-span masking/gating mechanism across the Evidence-Cause Interaction branches.

### T. Preethi Vardhan

Benchmarked accuracy/F1 against attention span on:

- IEMOCAP
- MELD

This is particularly important because the datasets have very different conversation lengths.

IEMOCAP:

```text
~49 utterances/conversation
```

MELD:

```text
~9.6 utterances/dialogue
```

### M. Uday

Evaluated whether adaptive attention span changes the model's calibration quality.

The evaluation compares:

```text
Full Attention
       vs.
Adaptive Attention Span
```

using calibration metrics such as ECE.

---

# 🛠️ Additional Engineering Fixes

During the code audit, we also identified an implementation problem in the original `model.py`.

The original implementation could raise an:

```text
UnboundLocalError
```

when running on CPU because certain attention masks were assigned only inside a GPU-specific branch.

We fixed this issue by adding the required CPU-side branch.

This is considered an **engineering/reproducibility fix**, not one of the four machine-learning limitations.

We also added reproducibility-related functionality such as:

- Configurable random seeds
- Command-line configuration
- Automatic checkpoint saving
- Ablation scripts
- Logging
- Multi-seed evaluation

---

# 📁 Project Structure

The modified repository follows the original ECERC organization.

A simplified structure is:

```text
ECERC/
│
├── MELD/
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   ├── inference.py
│   ├── dataloader.py
│   ├── calibrate.py
│   └── span_benchmark.py
│
├── IEMOCAP/
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   ├── inference.py
│   ├── dataloader.py
│   ├── calibrate.py
│   └── span_benchmark.py
│
├── auto_tune.py
├── download_data.py
├── run_ablation.sh
├── run_span_sweep.sh
├── summarize_logs.py
└── README.md
```

---

# 🔄 What Was Changed From the Original ECERC?

The modifications are designed to be controlled using command-line flags.

### Original ECERC

Running training without limitation flags preserves the original behavior:

```bash
python train.py
```

### Limitation 1

```bash
python train.py --lim1
```

Enables:

```text
Class-Balanced Loss
+
Focal Loss
```

### Limitation 2

```bash
python train.py --lim2
```

Enables:

```text
Confusion-Aware Label Smoothing
```

### Limitation 4

```bash
python train.py --lim4
```

Enables:

```text
Adaptive / Learnable Attention Span
```

### Combined Model

The three training-time modifications can be combined:

```bash
python train.py --lim1 --lim2 --lim4
```

Limitation 3 is applied separately as a post-hoc calibration procedure.

---

# 📦 Dataset

We use the two datasets used by the original ECERC implementation:

## IEMOCAP

IEMOCAP contains:

- 151 conversations
- 7,433 utterances
- 6 emotion classes

The project uses the preprocessed multimodal features provided by the original ECERC release.

## MELD

MELD contains:

- 1,433 dialogues
- 13,708 utterances
- 7 emotion classes

The project also uses the preprocessed features released with the original ECERC implementation.

We do not introduce new data collection or annotation.

---

# 🚀 Running the Project on Kaggle

Kaggle can be used as the primary GPU environment for running this project.

A GPU such as the Kaggle T4 environment is sufficient according to the project SOP.

## Step 1 – Create a Kaggle Notebook

Create a new Kaggle Notebook.

Open:

```text
Settings → Accelerator → GPU
```

Select an available GPU such as:

```text
T4
```

---

## Step 2 – Clone This Repository

Replace the placeholder with the GitHub repository containing this project.

```bash
!git clone YOUR_PROJECT_GITHUB_REPOSITORY_URL
```

Then enter the repository:

```bash
%cd YOUR_PROJECT_REPOSITORY_NAME
```

---

## Step 3 – Install Dependencies

Install the required packages.

```bash
!pip install gdown
```

If the project contains a requirements file:

```bash
!pip install -r requirements.txt
```

The original project was developed around the versions specified in the SOP, including:

```text
Python 3.9
PyTorch 2.0.1
scikit-learn
matplotlib
seaborn
```

For reproducibility, use the project's provided environment configuration when available.

---

# 📥 Step 4 – Download the Preprocessed Dataset

The original ECERC repository provides links to the preprocessed multimodal features.

From the project root:

```bash
!python download_data.py
```

This downloads the required preprocessed data into:

```text
data/meld
data/iemocap
```

The raw IEMOCAP dataset itself requires the appropriate dataset access/license. Our implementation uses the preprocessed features released by the original ECERC authors.

---

# 🧪 Step 5 – Verify the Original ECERC Baseline

Before running our modifications, it is recommended to verify the original model.

For MELD:

```bash
%cd MELD
!python inference.py
```

For IEMOCAP:

```bash
%cd ../IEMOCAP
!python inference.py
```

This allows the original ECERC implementation to be checked before enabling our modifications.

---

# 🏋️ Step 6 – Train With the Proposed Improvements

## Limitation 1 Only

```bash
!python train.py --lim1
```

Uses:

```text
Class-Balanced Loss
+
Focal Loss
```

---

## Limitation 2 Only

```bash
!python train.py --lim2
```

Uses:

```text
Confusion-Aware Label Smoothing
```

---

## Limitation 4 Only

```bash
!python train.py --lim4
```

Uses:

```text
Adaptive / Learnable Attention Span
```

---

## Combined Model

To enable all training-time modifications:

```bash
!python train.py --lim1 --lim2 --lim4 --save_model_path ckpt_all.pkl
```

---

# 🎯 Step 7 – Run Calibration

Limitation 3 is evaluated after training.

For example:

```bash
!python calibrate.py --load_model_state_dir ckpt_all.pkl
```

This evaluates:

```text
Before Calibration
        ↓
Temperature Scaling
        ↓
MC-Dropout
        ↓
After Calibration
```

The calibration script reports metrics such as:

- ECE
- NLL
- Brier Score
- Reliability diagrams

---

# 📏 Step 8 – Benchmark Adaptive Attention Span

To evaluate fixed attention spans:

```bash
!python span_benchmark.py --load_model_state_dir ECERC_MODEL.pkl
```

To evaluate a learned adaptive-span checkpoint:

```bash
!python span_benchmark.py --learned_ckpts ckpt_all.pkl
```

This allows comparison of:

```text
Accuracy
F1
ECE
Attention Span
```

---

# 📊 Step 9 – Run the Complete Ablation

The project provides an ablation script.

For MELD:

```bash
!bash run_ablation.sh MELD 0
```

For IEMOCAP:

```bash
!bash run_ablation.sh IEMOCAP 0
```

The ablation compares:

```text
Baseline
   ↓
+ Limitation 1
   ↓
+ Limitation 2
   ↓
+ L1 + L2
   ↓
+ Limitation 4
   ↓
All modifications
   ↓
Calibration / Span evaluation
```

---

# 🔧 Automatic Hyperparameter Tuning

The project also includes `auto_tune.py`.

For MELD:

```bash
!python auto_tune.py MELD
```

For a quicker experiment:

```bash
!python auto_tune.py MELD --quick
```

For IEMOCAP:

```bash
!python auto_tune.py IEMOCAP --quick
```

Multiple seeds can be used for the final comparison:

```bash
!python auto_tune.py MELD --seeds 0 1 2
```

Hyperparameters are selected using the validation set.

The test set should **not** be used for hyperparameter selection.

---

# 📈 Evaluation Metrics

We evaluate the baseline and modified models using:

### Classification

- Accuracy
- Weighted F1
- Macro F1
- Per-class F1
- Confusion Matrix

### Calibration

- Expected Calibration Error (ECE)
- Negative Log-Likelihood (NLL)
- Brier Score
- Reliability Diagrams

### Adaptive Attention

- Learned attention span
- Accuracy vs. attention span
- F1 vs. attention span
- ECE vs. attention span

The final experiments should compare:

```text
Original ECERC
vs.
L1
vs.
L2
vs.
L3
vs.
L4
vs.
Combined Model
```

on both:

```text
IEMOCAP
MELD
```

---

# 🔀 Reproducibility

For reproducible experiments, use a fixed seed:

```bash
python train.py --seed 0
```

For stronger evaluation, run multiple seeds:

```text
Seed 0
Seed 1
Seed 2
```

and report:

```text
mean ± standard deviation
```

The checkpoint should be selected using validation performance rather than test-set performance.

---

# 👨‍💻 Individual GitHub Contributions

The following repositories represent the individual/team contributions.

| Member | Contribution | Limitation | GitHub |
|---|---|---|---|
| N. Sravan Kumar | Class-balanced + Focal Loss, adaptive-span implementation, per-class evaluation | L1 + shared L4 | `YOUR_SRAVAN_GITHUB_REPOSITORY_URL` |
| T. Preethi Vardhan | Confusion-aware label smoothing, attention-span benchmarking | L2 + shared L4 | `YOUR_PREETHI_GITHUB_REPOSITORY_URL` |
| M. Uday | Temperature Scaling, MC-Dropout, calibration evaluation | L3 + shared L4 | `YOUR_UDAY_GITHUB_REPOSITORY_URL` |

Replace each placeholder with the appropriate GitHub repository/profile before publishing.

---

# 🧩 Contribution Summary

### Member 1 – N. Sravan Kumar

```text
Original ECERC
      ↓
Class Imbalance
      ↓
Class-Balanced Loss
      +
Focal Loss
      ↓
Per-Class F1 / Confusion Matrix

                +

Adaptive Attention Span
      ↓
Four ECERC Interaction Branches
```

### Member 2 – T. Preethi Vardhan

```text
Original ECERC
      ↓
Emotion Confusability
      ↓
Confusion-Aware Label Smoothing
      ↓
Better handling of:
Happy ↔ Excited
Angry ↔ Frustrated
Disgust ↔ Angry

                +

Attention-Span Benchmarking
```

### Member 3 – M. Uday

```text
Original ECERC
      ↓
No Confidence Calibration
      ↓
Temperature Scaling
      +
MC-Dropout
      ↓
ECE / NLL / Brier / Reliability

                +

Adaptive-Span Calibration Analysis
```

---

# ⚠️ Important Scope of the Work

This project does **not** claim to introduce a completely new emotion-recognition architecture.

Instead, our contribution is to improve the robustness and reliability of an existing state-of-the-art, publicly available ECERC implementation by addressing concrete limitations.

The original ECERC components remain the foundation:

```text
Evidence Gating
Cause Encoding
Evidence-Cause Interaction
Feature Gating
```

Our modifications are focused on:

```text
L1 → Class imbalance
L2 → Category confusability
L3 → Confidence calibration
L4 → Effective attention context
```

---

# 📌 Current Limitations / Important Notes

### 1. Adaptive span does not reduce O(L²) attention computation

The adaptive span limits the effective receptive field, but the complete attention tensor is still computed.

Therefore, it should not be described as a computational-complexity reduction.

### 2. Hyperparameters should be selected using validation data

Do not select hyperparameters based on the test results.

### 3. Multiple seeds are recommended

A single training run can give misleading results. Final results should preferably be reported as:

```text
Mean ± Standard Deviation
```

over multiple seeds.

### 4. Preprocessed features are required

The project relies on the preprocessed multimodal features released with the original ECERC implementation.

---

# 📖 References

1. T. Zhang and Z. Tan. **ECERC: Evidence-Cause Attention Network for Multi-Modal Emotion Recognition in Conversation.** Proceedings of ACL 2025.

2. C. Busso et al. **IEMOCAP: Interactive Emotional Dyadic Motion Capture Database.** Language Resources and Evaluation, 2008.

3. S. Poria et al. **MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations.** ACL, 2019.

4. Y. Cui et al. **Class-Balanced Loss Based on Effective Number of Samples.** CVPR, 2019.

5. T.-Y. Lin et al. **Focal Loss for Dense Object Detection.** ICCV, 2017.

6. C. Szegedy et al. **Rethinking the Inception Architecture for Computer Vision.** CVPR, 2016.

7. C. Guo et al. **On Calibration of Modern Neural Networks.** ICML, 2017.

8. Y. Gal and Z. Ghahramani. **Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.** ICML, 2016.

9. S. Sukhbaatar et al. **Adaptive Attention Span in Transformers.** ACL, 2019.

---

# 🤖 AI-Assisted Research Disclosure

AI tools were used as research and reference assistance during the project for:

- Literature and repository review
- Clarification of technical concepts
- Understanding implementation details
- Preliminary exploration of machine-learning techniques

The project team independently performed:

- Implementation
- Code modifications
- Experiments
- Testing
- Evaluation
- Verification
- Analysis
- Final conclusions

The final implementation and project decisions were made and verified by the project team.

---

# 📜 Acknowledgement

We acknowledge the authors of **ECERC** for publicly releasing their implementation and making this project possible.

Our work builds upon their publicly available implementation while attempting to improve its robustness and reliability through the four limitations described above.

**Original project:** ECERC – Evidence-Cause Attention Network for Multi-Modal Emotion Recognition in Conversation

**Our project:** Improving Robustness and Reliability of ECERC for Multi-Modal Emotion Recognition in Conversation