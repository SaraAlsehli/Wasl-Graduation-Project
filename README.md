# Wasl-Graduation-Project

**Evaluating Text Difficulty Using Readability Metrics and Facial Expression Analysis** 

UQU Graduation Project — F12

---



</div>


## Introduction

### The Problem

The core problem addressed in this work lies in the difficulty of accurately and objectively identifying reading-related stress and cognitive load in students during real-time reading activities, particularly in Arabic language contexts. Traditional assessment methods rely on manual observation or standardized tests, which are limited in their ability to capture subtle, continuous, and non-verbal behavioral signals such as micro-expressions, gaze patterns. These approaches are often subjective, time-consuming, and lack the granularity needed to reflect moment-to-moment variations in cognitive and emotional states. As a result, there is a need for an automated, scalable, and data-driven system capable of extracting meaningful behavioral indicators from facial and visual data to better understand and predict reading difficulties.

### Why Facial Expressions?

The human face is a rich and informative channel of non-verbal communication. Research in affective computing has shown that variations in facial expressions, gaze direction, and head pose can serve as reliable indicators of underlying cognitive and emotional states such as stress, confusion, and cognitive load.

In this work, facial analysis is leveraged to extract structured behavioral signals from video data, enabling the construction of a data-driven representation of reading-related stress without requiring intrusive or specialized physiological sensors.

### Why Arabic Reading Sessions?

Arabic presents unique linguistic and cognitive characteristics due to its rich morphology, contextual letter shaping, and the coexistence of Modern Standard Arabic with spoken dialects. These properties contribute to variations in reading difficulty and cognitive processing load compared to other languages.

### The Role of AI and Machine Learning

Combines unsupervised learning (to discover inherent data structure and generate pseudo-labels) with supervised learning (to generalize these patterns for prediction). This hybrid strategy enables the development of an adaptive and fully data-driven system for stress detection, capable of modeling complex and non-linear relationships in facial expression data.

---

## 🎯 Objectives

- 📊 Perform comprehensive **Exploratory Data Analysis** on facial emotion and pose features
- 🔧 Engineer meaningful **derived features** (pose magnitude, eye magnitude, pose-eye interaction) to enrich the feature space
- 🔍 Apply **Gaussian Mixture Model clustering** to discover hidden stress-related groupings without labeled data
- 🏷️ Generate **data-driven stress labels** from cluster assignments to enable supervised learning
- 🤖 Train a high-performance **Gradient Boosting classifier** 


---

## 📦 Dataset Description

Facial expression data was extracted frame-by-frame from video recordings of participants reading Arabic text passages, using **Amazon Rekognition's DetectFaces API** — a production-grade deep learning service for facial analysis.

### Dataset Statistics

| Split | Samples | Raw Features | Engineered Features |
|-------|---------|--------------|---------------------|
| Training Set | 1,263 | 8 | 13 |
| Testing Set | 1,706 | 8 | 13 |

The dataset is **complete with zero missing values and zero duplicate records**, allowing the pipeline to proceed directly to feature engineering without imputation overhead.

### Feature Schema

| Feature | Type | Description |
|---------|------|-------------|
| `Frame` | Integer | Sequential frame index |
| `Emotion` | Categorical | Dominant detected emotion (`CALM`, `SURPRISED`, `SAD`, `CONFUSED`, `DISGUSTED`) |
| `Emotion_Confidence` | Float (0–100) | Rekognition's confidence score for the dominant emotion |
| `Pose_Roll` | Float (degrees) | Head rotation around the Z-axis (tilt) |
| `Pose_Yaw` | Float (degrees) | Head rotation around the Y-axis (left/right turn) |
| `Pose_Pitch` | Float (degrees) | Head rotation around the X-axis (up/down) |
| `Eye_Yaw` | Float (degrees) | Estimated horizontal gaze deviation |
| `Eye_Pitch` | Float (degrees) | Estimated vertical gaze deviation |

### Engineered Features

Three domain-informed features were derived to amplify signal strength:

| Engineered Feature | Formula | Rationale |
|--------------------|---------|-----------|
| `Pose_Magnitude` | √(Roll² + Yaw² + Pitch²) | Aggregate head movement intensity |
| `Eye_Magnitude` | √(Eye_Yaw² + Eye_Pitch²) | Aggregate gaze deviation intensity |
| `Pose_x_Eye` | Pose_Magnitude × Eye_Magnitude | Captures coupled head-gaze behavior under stress |


---

## 📊 Exploratory Data Analysis

EDA was conducted systematically before any modeling to understand data quality, feature distributions, and inter-feature relationships.

### Correlation Analysis

A full Pearson correlation matrix was computed over all numeric features. Key findings informed the feature engineering strategy:

- `Pose_Magnitude` captures the aggregate signal of `Pose_Roll`, `Pose_Yaw`, and `Pose_Pitch` into a single discriminative scalar — consolidating head movement into one powerful feature
- `Eye_Magnitude` similarly unifies gaze deviation into a compact representation
- The `Pose_x_Eye` interaction term was motivated by the observed moderate correlation between pose and gaze streams, suggesting coupled head-eye behavior under cognitive load

### Notable Statistical Insights

- **`Eye_Yaw` mean = −4.2°** with high spread (std ≈ 4.5°) — captures the natural left-ward gaze bias during Arabic right-to-left reading and its disruption under stress
- **`Pose_Roll` is tightly distributed** (std ≈ 1.0°) around a consistent reading posture, making deviations from baseline highly meaningful stress indicators
- **`Emotion_Confidence` variance is large** (std ≈ 18.3, range 29–99.9), reflecting genuine moment-to-moment fluctuations in emotional clarity across frames — rich signal for stress differentiation

### Dimensionality Reduction & Visualization

PCA was applied to project the 13-dimensional feature space into 2D for cluster visualization. The first two principal components explain **59.3% of total variance** (PC1: 43.9%, PC2: 15.4%), confirming that the engineered features carry strong, concentrated signal suitable for clustering and classification.

---

## 🔵 Clustering — Gaussian Mixture Model

### Why Clustering?

The dataset contained no pre-existing stress labels — stress is a latent construct not directly observable from raw Rekognition outputs. Clustering was applied to:

- Discover hidden emotional and behavioral structure across participants
- Identify probabilistic groupings that reflect genuine variation in cognitive stress state
- Reveal complex relationships between facial pose, gaze, and emotion confidence features
- Generate data-driven stress labels that enable supervised learning without manual annotation

### Algorithm Selection: From K-Means to GMM

K-Means was evaluated as an initial baseline. However, the emotional feature space exhibits overlapping, non-spherical distributions that violate K-Means' core assumptions of compact, equally-sized, globular clusters. The algorithm produced weak cluster separability and less semantically coherent groupings — motivating the search for a more expressive model.

**Gaussian Mixture Model (GMM)** was selected as the final clustering algorithm based on its superior fit to the data's probabilistic structure:

| Property | K-Means | GMM ✅ |
|----------|---------|--------|
| Cluster shape | Spherical only | Elliptical, arbitrary |
| Assignment type | Hard (binary) | Soft (probabilistic) |
| Uncertainty modeling | None | Full posterior probabilities |
| Covariance structure | Implicit (Euclidean) | Explicit (learned per cluster) |
| Fit criterion | Inertia | BIC (principled model selection) |

Human emotional states are inherently continuous and overlapping — a participant transitioning between calm and stressed does not flip a binary switch. GMM reflects this reality by assigning each sample a **probability of belonging to each cluster**, producing richer, more realistic stress-state representations.



### Clustering Output

GMM discovered **6 distinct stress-level groups**, labeled by their behavioral profiles:

| Cluster Label | Training Samples | Interpretation |
|---------------|-----------------|----------------|
| Low Stress | 376 | Stable gaze, composed posture, high emotion confidence |
| Medium Stress  | 344 | Moderate gaze deviation, slight postural shift |
| High Stress | 152 | Maximum pose-gaze coupling, high `Pose_x_Eye` scores |

---

## 🤖 Supervised Learning

### Algorithm: Gradient Boosting Classifier

A **Gradient Boosting Classifier** was trained on the GMM-derived stress labels. Gradient Boosting builds an additive ensemble of weak learners in a stage-wise fashion, each iteration fitting the residual errors of the previous model:

```
F_m(x) = F_{m-1}(x) + ν · h_m(x)
```

This produces a highly expressive model well-suited to the overlapping, non-linear decision boundaries between stress levels.

### Hyperparameter Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 200 | Sufficient boosting rounds for convergence |
| `max_depth` | 4 | Controls individual tree complexity |
| `learning_rate` | 0.1 | Conservative step size for stable, generalizable learning |
| `subsample` | 0.8 | Stochastic boosting — reduces variance, improves robustness |
| `min_samples_split` | 10 | Guards against overfitting to small node populations |


### Feature Selection via Mutual Information

Prior to training, **Mutual Information scores** were computed between each feature and the stress labels. The top 6 features (MI above mean threshold) were selected, reducing noise and improving generalization:

| Rank | Feature | MI Score | Insight |
|------|---------|----------|---------|
| 1 | `Eye_Yaw` | 0.838 | Horizontal gaze is the strongest stress discriminator |
| 2 | `Pose_Yaw` | 0.698 | Head turning correlates strongly with attention shifts |
| 3 | `Pose_x_Eye` | 0.565 | Coupled head-gaze behavior amplifies stress signal |
| 4 | `Eye_Magnitude` | 0.409 | Overall gaze instability tracks stress elevation |
| 5 | `Pose_Magnitude` | 0.363 | Aggregate postural disruption under cognitive load |
| 6 | `Pose_Pitch` | 0.328 | Vertical head pitch reflects reading engagement level |

The dominance of gaze and pose features — rather than raw emotion categories — is a key finding: **where participants look and how they hold their head are more informative stress signals than the emotion label alone.**

### Cross-Validation Results

5-fold Stratified K-Fold cross-validation was conducted on the training set, preserving class proportions across all folds:

```
Fold 1: 96.44%
Fold 2: 95.26%
Fold 3: 95.26%
Fold 4: 96.43%
Fold 5: 96.03%
────────────────────────────────
Mean Accuracy:  95.88% ± 0.53%
```

The low standard deviation (±0.53%) across folds confirms **exceptional model stability** — the classifier generalizes consistently regardless of which subset of data it trains on.

---

## 📈 Model Evaluation

### Overall Performance

| Metric | Score |
|--------|-------|
| **Cross-Validation Accuracy** | **95.88% ± 0.53%** |
| **Test Set Accuracy** | **75.73%** |


### Gradient Boosting Feature Importances

The model's internal feature importance rankings align closely with the MI-based selection, validating the feature engineering decisions:

| Feature | GB Importance |
|---------|:-------------:|
| `Eye_Yaw` | 0.275 |
| `Pose_Yaw` | 0.198 |
| `Eye_Magnitude` | 0.174 |
| `Pose_x_Eye` | 0.165 |
| `Pose_Pitch` | 0.165 |
| `Pose_Magnitude` | 0.024 |

---


</div>

## Overview

This project combines **NLP readability analysis** with **real-time facial expression detection** to evaluate text difficulty. It uses two parallel pipelines:

- **Pipeline 1 (NLP):** Computes Flesch scores, grade levels, word frequency analysis, and identifies complex words
- **Pipeline 2 (CV):** Uses MediaPipe Face Mesh to detect facial expressions and map emotions to cognitive difficulty levels
- **Correlation Engine:** Compares NLP predictions with actual reader facial responses

## Features

- ✅ **Text Analysis Dashboard** — Paste text and get instant readability metrics
- ✅ **Word Difficulty Heatmap** — Color-coded visualization of word complexity
- ✅ **Sentence-by-Sentence Analysis** — Breakdown of difficulty per sentence
- ✅ **Complex Word Detection** — Identifies and ranks the hardest words
- ✅ **Live Reading Session** — Webcam-based facial expression analysis while reading
- ✅ **Real-time Emotion Detection** — MediaPipe Face Mesh landmark analysis
- ✅ **Cognitive Load Indicator** — Visual bar showing reader's mental effort
- ✅ **NLP vs Facial Correlation** — Validates predictions against reader experience
- ✅ **PDF Report Generation** — Downloadable reports with all metrics
- ✅ **Emotion Timeline** — Visual history of emotions during reading

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Flask (Python) |
| Frontend | HTML/CSS/JS (Jinja2) |
| Face Detection | MediaPipe Face Mesh |
| Emotion Estimation | Rule-based from facial landmarks (Action Unit approximation) |
| NLP Readability | textstat + wordfreq |
| PDF Reports | ReportLab |
| Video Processing | OpenCV |

## Setup

```bash
# 1. Clone or extract the project
cd text-difficulty-project

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup NLTK data (needed for textstat)
python -c "import nltk; nltk.download('cmudict')"
# If NLTK download fails, install: pip install cmudict

# 5. Run the app
python app.py
```

The app will be available at **http://localhost:5000**

## Project Structure

```
text-difficulty-project/
├── app.py                  # Main Flask application (all backend logic)
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── static/
│   ├── css/
│   │   └── style.css      # Main stylesheet
│   ├── js/
│   │   └── main.js        # Utility functions
│   ├── uploads/            # User uploaded files
│   └── reports/            # Generated PDF reports
└── templates/
    ├── base.html           # Base template with nav
    ├── index.html          # Home page
    ├── analyze.html        # Text analysis page
    ├── read.html           # Reading session page
    └── report.html         # Report generation page
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/analyze-text` | POST | Analyze text readability |
| `/api/detect-emotion` | POST | Detect emotion from webcam frame |
| `/api/correlate` | POST | Run correlation engine |
| `/api/generate-report` | POST | Generate PDF report |
| `/api/download-report/<file>` | GET | Download report |

## How the Emotion-to-Difficulty Mapping Works

Based on the project report's Table 1:

| Emotion | Difficulty Level | Score |
|---|---|---|
| Happy, Calm | Easy | 15-20 |
| Neutral, Sad | Medium | 50-55 |
| Angry, Fear, Disgust, Surprise | Hard | 70-90 |

## Project Team & Data Scientists

- Rimas Mesfer Alqathami
- Sara Ayed Alsehli
- Hanin Mesfer Almalki

## Supervisor: Dr. Maram Almaghrabi

Data Science Department, College of Computing, Umm Al-Qura University
