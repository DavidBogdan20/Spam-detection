# 🛡️ Spam Shield - AI-Powered Spam Detection System

A comprehensive AI-enabled spam detection system with machine learning classification, fairness monitoring, adversarial defense, and a modern web interface.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Web Interface Guide](#web-interface-guide)
3. [API Reference](#api-reference)
4. [Testing the System](#testing-the-system)
5. [Architecture Overview](#architecture-overview)
6. [Configuration](#configuration)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Windows OS

### Installation

1. **Navigate to the project directory:**
   ```bash
   cd "d:\Solutii AI\spam_detection"
   ```

2. **Activate the virtual environment:**
   ```bash
   .\venv\Scripts\activate
   ```

3. **Install dependencies (if not already installed):**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open in browser:**
   - Inbox: http://localhost:5000
   - Dashboard: http://localhost:5000/dashboard

---

## 🖥️ Web Interface Guide

### Main Inbox (http://localhost:5000)

The inbox displays SMS messages with their spam classification.

#### Sidebar Navigation
| Button | Description |
|--------|-------------|
| 📥 **Inbox** | View all messages |
| ⚠️ **Spam** | Filter to show only spam messages |
| ✅ **Safe** | Filter to show only safe (ham) messages |
| 📊 **Dashboard** | Open the metrics dashboard |

#### Message List
- Each message shows:
  - **Icon**: ⚠️ for spam, ✉️ for safe
  - **Preview**: First line of the message
  - **Badge**: HAM or SPAM classification
  - **Confidence**: How confident the AI is (0-100%)

#### Viewing Message Details
1. Click on any message in the list
2. The right panel shows:
   - **Classification Badge**: SPAM or SAFE
   - **Confidence Meter**: Visual confidence indicator
   - **Full Message Content**: Complete message text
   - **Feedback Buttons**: Correct the AI if it made a mistake
   - **Metadata**: Spam/ham probabilities, message length

#### Testing a Custom Message
1. Click the **"✏️ Test Message"** button (top right)
2. Enter any text in the modal
3. Click **"🔍 Analyze"**
4. See the classification result with confidence

#### Providing Feedback
When viewing a message:
- Click **"🚫 This is Spam"** if a safe message is actually spam
- Click **"✓ Not Spam"** if a spam message is actually safe

This feedback helps improve the model over time.

---

### Dashboard (http://localhost:5000/dashboard)

The dashboard shows real-time system metrics.

#### Stats Cards
| Metric | Description |
|--------|-------------|
| **Total Predictions** | Number of messages classified |
| **Accuracy** | Percentage of correct classifications |
| **Precision** | Of messages marked spam, % that are actually spam |
| **Recall** | Of actual spam, % that were detected |

#### Charts
- **Prediction Distribution**: Pie chart of spam vs ham
- **Confusion Matrix**: True/False Positives/Negatives

#### Alerts
Shows system warnings like:
- High false positive rate
- Performance degradation
- Unusual spam patterns

#### Fairness Metrics
Based on message length as a protected attribute:
- **Disparate Impact**: Ratio of favorable outcomes between groups
- **Equal Opportunity**: True positive rate equality
- **Equalized Odds**: TPR and FPR equality

---

## 🔌 API Reference

### Health Check
```http
GET /api/health
```
**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "feature_extractor_loaded": true
}
```

---

### Classify a Message
```http
POST /api/classify
Content-Type: application/json

{
    "message": "Your message text here"
}
```
**Response:**
```json
{
    "id": "unique-message-id",
    "prediction": "spam",
    "confidence": 0.95,
    "spam_probability": 0.95,
    "ham_probability": 0.05
}
```

---

### Classify Multiple Messages
```http
POST /api/classify/batch
Content-Type: application/json

{
    "messages": ["message1", "message2", "message3"]
}
```
**Response:**
```json
{
    "results": [
        {"id": "...", "prediction": "spam", "confidence": 0.95, ...},
        {"id": "...", "prediction": "ham", "confidence": 0.88, ...}
    ]
}
```

---

### Submit Feedback
```http
POST /api/feedback
Content-Type: application/json

{
    "message_id": "id-from-classify",
    "original_prediction": 1,
    "correct_label": 0
}
```
**Response:**
```json
{
    "status": "recorded",
    "feedback_count": 42
}
```

---

### Get Metrics
```http
GET /api/metrics
```
**Response:**
```json
{
    "metrics": {
        "total_predictions": 100,
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.98
    },
    "recent_predictions": [...],
    "alerts": [...]
}
```

---

### Get Messages
```http
GET /api/messages?limit=20&offset=0&filter=all
```
**Parameters:**
- `limit`: Number of messages (default: 20)
- `offset`: Pagination offset (default: 0)
- `filter`: 'all', 'spam', or 'ham'

---

## 🧪 Testing the System

### Test via Web Interface

1. **Open the Test Modal:**
   - Go to http://localhost:5000
   - Click "✏️ Test Message" in the header

2. **Test Spam Detection:**
   Try these spam examples:
   ```
   Congratulations! You've won $1000! Call 0800-123-456 now!
   ```
   ```
   URGENT: Your account will be suspended. Click here to verify.
   ```
   ```
   FREE iPhone! Text WIN to 80800 to claim your prize!
   ```

3. **Test Ham Detection:**
   Try these legitimate examples:
   ```
   Hey, are you coming to the party tonight?
   ```
   ```
   Don't forget to pick up milk on your way home.
   ```
   ```
   The meeting has been moved to 3pm.
   ```

---

### Test via PowerShell/Terminal

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:5000/api/health" -Method GET

# Classify a spam message
$body = '{"message": "You won $5000! Call now to claim!"}'
Invoke-RestMethod -Uri "http://localhost:5000/api/classify" -Method POST -ContentType "application/json" -Body $body

# Classify a ham message
$body = '{"message": "See you at dinner tonight"}'
Invoke-RestMethod -Uri "http://localhost:5000/api/classify" -Method POST -ContentType "application/json" -Body $body
```

---

### Test via Python Script

```python
import requests

# Health check
response = requests.get("http://localhost:5000/api/health")
print(response.json())

# Classify a message
response = requests.post(
    "http://localhost:5000/api/classify",
    json={"message": "WINNER! You've been selected for a cash prize!"}
)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
```

---

### Test via cURL

```bash
# Health check
curl http://localhost:5000/api/health

# Classify a message
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"message": "Free entry to win a car!"}'
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Interface                          │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│   │   Inbox     │  │  Dashboard  │  │  Test Modal     │    │
│   └─────────────┘  └─────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Flask API                              │
│   /api/classify  /api/feedback  /api/metrics  /api/messages │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Preprocessor │    │   Fairness   │    │  Monitoring  │
│  - Cleaning  │    │  - Metrics   │    │  - Drift     │
│  - Unicode   │    │  - Mitigation│    │  - Tracking  │
└──────────────┘    └──────────────┘    └──────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│                    ML Pipeline                           │
│  ┌─────────────┐    ┌────────────────┐    ┌──────────┐  │
│  │   TF-IDF    │───▶│ Logistic Reg.  │───▶│ Predict  │  │
│  │  Features   │    │   Classifier   │    │  0 or 1  │  │
│  └─────────────┘    └────────────────┘    └──────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `TFIDF_MAX_FEATURES` | 5000 | Maximum vocabulary size |
| `DEFAULT_SPAM_THRESHOLD` | 0.5 | Classification threshold |
| `HIGH_PRECISION_THRESHOLD` | 0.7 | Threshold for fewer false positives |
| `MESSAGE_LENGTH_THRESHOLD` | 50 | Short vs long message boundary |
| `DEBUG` | True | Flask debug mode |

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~97% |
| Precision | ~95% |
| Recall | ~95% |
| F1 Score | ~95% |

---

## 🛠️ Troubleshooting

### Server won't start
```bash
# Make sure you're in the right directory
cd "d:\Solutii AI\spam_detection"

# Make sure venv is activated
.\venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Model not loading
The model trains automatically on first run. Check that `data/SMSSpamCollection` exists.

### API returns 500 error
Check the terminal for error messages. Common issues:
- Missing dependencies
- Dataset file not found
- Port 5000 already in use

---

## 📝 Sample Test Messages

### Spam Examples
```
1. "Congratulations! You've won a FREE iPhone! Call 0800-123-456!"
2. "URGENT: Your bank account needs verification. Click here now!"
3. "You've been selected for £5000 cash! Reply WIN to claim!"
4. "Free entry in our prize draw! Text GO to 80032"
5. "Get rich quick! Make $1000 daily working from home!"
```

### Ham (Safe) Examples
```
1. "Hey, what time should I pick you up?"
2. "Don't forget about the meeting tomorrow at 10am"
3. "Thanks for dinner last night, it was great!"
4. "Can you send me the report when you get a chance?"
5. "Happy birthday! Hope you have a wonderful day!"
```

---

## 📄 License

This project was created for educational purposes as part of an AI spam detection assignment.
