# Wasl-Graduation-Project

# Text Difficulty Evaluator

**Evaluating Text Difficulty Using Readability Metrics and Facial Expression Analysis** 

UQU Graduation Project — F12

---

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
