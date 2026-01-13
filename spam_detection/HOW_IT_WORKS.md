# 📧 Spam Detection System - How It Works

A comprehensive AI-enabled spam detection system that classifies SMS and email messages as spam or legitimate (ham) using machine learning techniques with built-in fairness monitoring, adversarial robustness, and real-time drift detection.

---

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SPAM DETECTION SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   Web UI    │───▶│  Flask API  │───▶│   ML Core   │                 │
│  │  (HTML/JS)  │    │  (Routes)   │    │ (Classifiers)│                 │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                 │
│                                               │                         │
│         ┌─────────────────────────────────────┼──────────────────────┐ │
│         │                                     │                      │ │
│         ▼                    ▼                ▼                      │ │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │ │
│  │  Fairness   │    │ Adversarial │    │  Monitoring │              │ │
│  │   Metrics   │    │  Hardening  │    │   & Drift   │              │ │
│  └─────────────┘    └─────────────┘    └─────────────┘              │ │
│                                                                      │ │
└──────────────────────────────────────────────────────────────────────┘ │
                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies Used

### Backend Framework
| Technology | Purpose |
|------------|---------|
| **Flask** (≥2.0) | Python web framework for the REST API and server |
| **Python 3.x** | Core programming language |

### Machine Learning & Data Science
| Technology | Purpose |
|------------|---------|
| **scikit-learn** (≥1.0) | ML algorithms (Logistic Regression, K-Means clustering) |
| **NumPy** (≥1.21) | Numerical computations and array operations |
| **Pandas** (≥1.3) | Data manipulation and dataset loading |
| **NLTK** (≥3.6) | Natural Language Toolkit for text processing |
| **joblib** (≥1.1) | Model persistence (saving/loading trained models) |

### Frontend
| Technology | Purpose |
|------------|---------|
| **HTML5** | Page structure |
| **CSS3** | Styling and responsive design |
| **JavaScript** | Client-side interactivity and API calls |

---

## 🤖 AI Models & Classification Pipeline

### 1. Text Preprocessing (`models/preprocessor.py`)

Before classification, all messages go through a preprocessing pipeline:

```
Raw Message → Preprocessing → Clean Text → Feature Extraction → Classification
```

**Preprocessing Steps:**
- **Zero-width character removal** – Removes invisible Unicode characters used to evade detection
- **Homoglyph replacement** – Converts lookalike characters (Cyrillic 'а' → ASCII 'a')
- **Unicode normalization** – Standardizes text to ASCII
- **URL/Email extraction** – Replaces URLs with `[URL]` tokens
- **Lowercase conversion** – Normalizes case
- **Punctuation removal** – Strips special characters
- **Whitespace normalization** – Collapses multiple spaces

### 2. Feature Extraction (`models/feature_extractor.py`)

The system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max Features | 5,000 | Maximum vocabulary size |
| N-gram Range | (1, 2) | Unigrams and bigrams |
| Min Document Frequency | 2 | Word must appear in ≥2 documents |
| Max Document Frequency | 95% | Ignore words in >95% of documents |
| Sublinear TF | True | Applies logarithmic term frequency scaling |

**How TF-IDF Works:**
- **TF (Term Frequency)**: How often a word appears in a message
- **IDF (Inverse Document Frequency)**: How rare a word is across all messages
- Words that are frequent in spam but rare overall get higher weights (e.g., "free", "winner")

### 3. Classification Models (`models/classifiers.py`)

#### Primary Model: Logistic Regression (`SpamClassifier`)

A probabilistic linear classifier that outputs spam probabilities:

```python
# Core Configuration
- Solver: L-BFGS (Limited-memory BFGS optimizer)
- Class Weight: Balanced (handles imbalanced spam/ham ratio)
- Regularization (C): 1.0
- Max Iterations: 1,000
- Default Threshold: 0.5 (adjustable)
```

**How it classifies:**
1. Takes TF-IDF feature vector as input
2. Computes weighted sum of features
3. Applies sigmoid function to get probability P(spam)
4. If P(spam) ≥ threshold → **SPAM**, else → **HAM**

**Threshold Options:**
- **Default (0.5)**: Balanced precision/recall
- **High Precision (0.7)**: Minimizes false positives (fewer legitimate emails marked as spam)

#### Secondary Model: K-Means Dictionary Learning (`DictionaryClassifier`)

An unsupervised approach using reconstruction error:

```python
# Configuration
- Spam Clusters: 50
- Ham Clusters: 100
- Random State: 42
```

**How it works:**
1. Learns two separate "dictionaries" (cluster centers):
   - **Spam dictionary**: Patterns from known spam messages
   - **Ham dictionary**: Patterns from legitimate messages
2. For a new message:
   - Compute distance to nearest spam cluster center
   - Compute distance to nearest ham cluster center
   - **Lower distance = better match**
3. Classify based on which dictionary reconstructs the message better

#### Ensemble Classifier (`EnsembleClassifier`)

Combines both models with weighted voting:
```python
final_score = (0.7 × LR_probability) + (0.3 × Dict_score)
prediction = "spam" if final_score ≥ threshold else "ham"
```

---

## 📊 Classification Flow

```
┌──────────────┐
│ User Input   │  "Congratulations! You've won $1000!"
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Preprocess   │  "congratulations youve won 1000"
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  TF-IDF      │  [0.0, 0.2, 0.8, 0.0, 0.5, ...]  (sparse vector)
│  Transform   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Logistic    │  P(spam) = 0.94
│  Regression  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Threshold    │  0.94 > 0.5 ✓
│   Check      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Output     │  ✉️ SPAM (94% confidence)
└──────────────┘
```

---

## ⚖️ Fairness Metrics (`fairness/metrics.py`)

The system monitors for bias using message length as a protected attribute:

| Metric | Description | Pass Threshold |
|--------|-------------|----------------|
| **Disparate Impact** | Ratio of favorable outcomes between groups | ≥ 0.8 (four-fifths rule) |
| **Equal Opportunity** | Equal TPR across groups | < 0.1 difference |
| **Equalized Odds** | Equal TPR AND FPR across groups | < 0.1 difference |
| **Predictive Parity** | Equal precision across groups | < 0.1 difference |

**Protected Groups:**
- **Short messages**: ≤ 50 characters
- **Long messages**: > 50 characters

This ensures the model doesn't unfairly flag short or long messages disproportionately.

---

## 🛡️ Adversarial Robustness (`adversarial/generator.py`)

The system tests resistance against common spam evasion techniques:

| Attack Technique | Description | Example |
|-----------------|-------------|---------|
| **Homoglyph Substitution** | Replace letters with lookalikes | "free" → "frее" (Cyrillic 'е') |
| **Zero-width Character** | Insert invisible characters | "free" → "f​r​e​e" |
| **Character Noise** | Typos, substitutions | "free" → "fr33" |
| **URL Obfuscation** | Add tracking params | "/claim" → "/claim?ref=8a3f" |
| **Benign Padding** | Mix spam with normal text | "Hope you're well. WIN $1000!" |

The preprocessing pipeline is designed to neutralize these attacks.

---

## 📈 Monitoring & Drift Detection (`monitoring/`)

### Metrics Tracker (`metrics_tracker.py`)
Tracks real-time performance:
- Predictions count (spam vs ham)
- Confidence distributions
- User feedback (corrections)
- Processing times

### Drift Detector (`drift_detector.py`)
Monitors for model degradation using **PSI (Population Stability Index)**:

| PSI Value | Interpretation |
|-----------|----------------|
| < 0.1 | No significant change |
| 0.1 - 0.2 | Moderate change (monitor) |
| ≥ 0.2 | **Significant drift (alert!)** |

**Drift Types Detected:**
- **Data Drift**: Input message distribution changes
- **Concept Drift**: Spam patterns evolve over time
- **Performance Drift**: Accuracy drops below threshold

---

## 🌐 Web Interface Routes

| Route | Description |
|-------|-------------|
| `/` | Main inbox interface for message classification |
| `/dashboard` | Monitoring dashboard with metrics and drift alerts |
| `/email-inbox` | Email inbox page for real email classification |
| `/api/classify` | REST API endpoint for classification |
| `/api/health` | Health check endpoint |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python app.py

# 3. Open in browser
# Main Interface: http://localhost:5000
# Dashboard:      http://localhost:5000/dashboard
```

---

## 📁 Project Structure

```
spam_detection/
├── app.py                  # Flask application entry point
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── models/
│   ├── preprocessor.py     # Text preprocessing
│   ├── feature_extractor.py # TF-IDF vectorization
│   ├── classifiers.py      # ML models (LR, K-Means, Ensemble)
│   └── trained/            # Saved model files
├── fairness/
│   ├── metrics.py          # Fairness metric calculations
│   └── mitigation.py       # Bias mitigation strategies
├── adversarial/
│   ├── generator.py        # Adversarial sample generation
│   └── hardening.py        # Model hardening techniques
├── monitoring/
│   ├── metrics_tracker.py  # Performance tracking
│   └── drift_detector.py   # Drift detection
├── api/
│   └── routes.py           # API endpoints
├── templates/              # HTML templates
├── static/                 # CSS, JavaScript
└── data/
    └── SMSSpamCollection   # Training dataset
```

---

## 📊 Model Performance

Typical metrics on the SMS Spam Collection dataset:

| Metric | Value |
|--------|-------|
| Accuracy | ~97% |
| Precision | ~95% |
| Recall | ~92% |
| F1 Score | ~93% |

---

## 🔑 Key Design Decisions

1. **TF-IDF over embeddings**: Chosen for interpretability and speed; works well for spam detection where specific keywords are strong indicators

2. **Logistic Regression as primary**: Provides probability outputs, handles imbalanced data well, fast inference

3. **Dictionary Learning as secondary**: Captures cluster-based patterns that linear models might miss

4. **Message length as protected attribute**: Quick proxy for potential bias; short messages (SMS-style) vs long messages (formal emails) may be treated differently

5. **PSI for drift detection**: Industry-standard metric for monitoring distribution shifts

---

## 📚 Dataset

The system is trained on the **SMS Spam Collection** dataset:
- **Total messages**: ~5,574
- **Spam**: ~747 (13%)
- **Ham**: ~4,827 (87%)
- **Format**: Tab-separated values (label + message)
