"""
Text Difficulty Evaluator
Evaluating Text Difficulty Using Readability Metrics and Facial Expression Analysis
UQU Graduation Project - F12
"""

import os
import json
import re
import base64
import uuid
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

import textstat
from wordfreq import word_frequency

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import cm

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/reports', exist_ok=True)


# ============================================================
# MODULE 1: NLP Text Analysis
# ============================================================

def detect_language(text):
    """Detect if text is Arabic or English."""
    arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    return "ar" if arabic_chars > latin_chars else "en"


def analyze_text(text):
    if not text or len(text.strip()) < 10:
        return {"error": "Text too short for analysis"}

    lang = detect_language(text)

    # Basic text stats (work for any language)
    word_count = len(text.split())
    sentences = re.split(r'[.!?؟。]+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
    sentence_count = max(len(sentences), 1)
    avg_sentence_length = word_count / sentence_count

    if lang == "en":
        # English: use textstat for full metrics
        flesch_score = textstat.flesch_reading_ease(text)
        flesch_kincaid = textstat.flesch_kincaid_grade(text)
        gunning_fog = textstat.gunning_fog(text)
        smog = textstat.smog_index(text)
        coleman_liau = textstat.coleman_liau_index(text)
        ari = textstat.automated_readability_index(text)
        dale_chall = textstat.dale_chall_readability_score(text)
        syllable_count = textstat.syllable_count(text)
        avg_syllables_per_word = syllable_count / max(word_count, 1)
        difficult_words_count = textstat.difficult_words(text)
        grade = textstat.text_standard(text, float_output=True)
    else:
        # Arabic: compute difficulty from word frequency + sentence length
        words = re.findall(r'[\u0600-\u06FF]+', text)
        word_count = len(words)
        syllable_count = 0  # not applicable for Arabic
        avg_syllables_per_word = 0

        # Count rare words using wordfreq
        rare_count = 0
        for w in words:
            freq = word_frequency(w, 'ar')
            if freq < 1e-5:
                rare_count += 1
        difficult_words_count = rare_count

        # Arabic readability estimate based on word rarity + sentence length
        rare_pct = (rare_count / max(word_count, 1)) * 100
        # Longer sentences + more rare words = harder
        flesch_score = max(0, min(100, 100 - (rare_pct * 1.5) - (avg_sentence_length - 10) * 2))
        flesch_kincaid = round(avg_sentence_length * 0.5 + rare_pct * 0.3, 2)
        gunning_fog = round(avg_sentence_length * 0.4 + rare_pct * 0.4, 2)
        smog = round(rare_pct * 0.5 + avg_sentence_length * 0.3, 2)
        coleman_liau = flesch_kincaid
        ari = round(avg_sentence_length * 0.5 + rare_pct * 0.2, 2)
        dale_chall = round(rare_pct * 0.3 + avg_sentence_length * 0.2, 2)
        grade = flesch_kincaid

    difficult_words_pct = (difficult_words_count / max(word_count, 1)) * 100

    if flesch_score >= 60:
        difficulty = "Easy"
        difficulty_color = "#22c55e"
        difficulty_description = "النص سهل الفهم لمعظم القراء." if lang == "ar" else "The text is easily understood by most readers."
    elif flesch_score >= 30:
        difficulty = "Medium"
        difficulty_color = "#f59e0b"
        difficulty_description = "النص يتطلب تركيزاً وجهداً متوسطاً." if lang == "ar" else "The text requires focused reading and moderate effort."
    else:
        difficulty = "Hard"
        difficulty_color = "#ef4444"
        difficulty_description = "النص معقد ويتطلب جهداً ذهنياً كبيراً." if lang == "ar" else "The text is complex and requires significant cognitive effort."

    return {
        "flesch_score": round(flesch_score, 2),
        "flesch_kincaid_grade": round(flesch_kincaid, 2),
        "gunning_fog": round(gunning_fog, 2),
        "smog_index": round(smog, 2),
        "coleman_liau": round(coleman_liau, 2),
        "ari": round(ari, 2),
        "dale_chall": round(dale_chall, 2),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "syllable_count": syllable_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "avg_syllables_per_word": round(avg_syllables_per_word, 2),
        "difficult_words_count": difficult_words_count,
        "difficult_words_pct": round(difficult_words_pct, 2),
        "grade_level": round(grade, 1),
        "difficulty": difficulty,
        "difficulty_color": difficulty_color,
        "difficulty_description": difficulty_description,
        "language": lang,
    }


def extract_complex_words(text):
    lang = detect_language(text)

    if lang == "ar":
        words = re.findall(r'[\u0600-\u06FF]+', text)
    else:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    seen = set()
    complex_words = []

    for word in words:
        if word in seen:
            continue
        seen.add(word)

        freq = word_frequency(word, lang)

        if lang == "ar":
            # Arabic: complex if rare (frequency < 1e-5) or long (6+ chars)
            word_len = len(word)
            is_complex = freq < 1e-5 or word_len >= 6
            if is_complex:
                freq_score = min(100, max(0, int((1 - min(freq * 10000, 1)) * 60)))
                len_score = min(40, max(0, (word_len - 4) * 10))
                difficulty_score = min(100, freq_score + len_score)
                reasons = []
                if freq < 1e-5:
                    reasons.append("كلمة نادرة")
                if word_len >= 6:
                    reasons.append(f"{word_len} حروف")
                complex_words.append({
                    "word": word,
                    "syllables": word_len,  # use char count for Arabic
                    "frequency": round(freq * 1e6, 4),
                    "difficulty_score": difficulty_score,
                    "reason": ", ".join(reasons)
                })
        else:
            # English
            syllables = textstat.syllable_count(word)
            is_complex = freq < 1e-5 or syllables >= 3
            if is_complex:
                freq_score = min(100, max(0, int((1 - min(freq * 10000, 1)) * 60)))
                syll_score = min(40, (syllables - 2) * 15)
                difficulty_score = min(100, freq_score + syll_score)
                reasons = []
                if freq < 1e-5:
                    reasons.append("Rare word")
                if syllables >= 3:
                    reasons.append(f"{syllables} syllables")
                complex_words.append({
                    "word": word,
                    "syllables": syllables,
                    "frequency": round(freq * 1e6, 4),
                    "difficulty_score": difficulty_score,
                    "reason": ", ".join(reasons)
                })

    complex_words.sort(key=lambda x: x["difficulty_score"], reverse=True)
    return complex_words[:30]


def generate_sentence_difficulty(text):
    lang = detect_language(text)
    # Split on sentence-ending punctuation (including Arabic ؟)
    sentences = re.split(r'[.!?؟]+\s*', text.strip())
    results = []
    for sent in sentences:
        if len(sent.strip()) < 5:
            continue

        if lang == "en":
            flesch = textstat.flesch_reading_ease(sent)
            wc = textstat.lexicon_count(sent, removepunct=True)
            sy = textstat.syllable_count(sent)
            avg_syll = sy / max(wc, 1)
            score = max(0, min(100, 100 - flesch))
        else:
            # Arabic: score based on word count + rare words
            words = re.findall(r'[\u0600-\u06FF]+', sent)
            wc = len(words)
            sy = 0
            avg_syll = 0
            rare = sum(1 for w in words if word_frequency(w, 'ar') < 1e-5)
            rare_pct = (rare / max(wc, 1)) * 100
            score = max(0, min(100, rare_pct * 1.5 + max(0, wc - 8) * 3))

        if score < 33:
            color, level = "#22c55e", "Easy"
        elif score < 66:
            color, level = "#f59e0b", "Medium"
        else:
            color, level = "#ef4444", "Hard"
        results.append({
            "sentence": sent.strip(), "difficulty_score": round(score, 1),
            "level": level, "color": color,
            "word_count": wc, "avg_syllables": round(avg_syll, 2)
        })
    return results


def generate_word_heatmap_data(text):
    lang = detect_language(text)
    words = re.findall(r'\b\w+\b', text) if lang == "en" else re.findall(r'[\u0600-\u06FF]+|\S+', text)
    heatmap = []
    for word in words:
        if len(word) < 2:
            heatmap.append({"word": word, "score": 0, "color": "#22c55e"})
            continue

        # Check if word is Arabic
        is_ar = bool(re.search(r'[\u0600-\u06FF]', word))
        if is_ar:
            freq = word_frequency(word, 'ar')
            wlen = len(word)
            score = 0
            if freq < 1e-6: score += 60
            elif freq < 1e-5: score += 40
            elif freq < 1e-4: score += 20
            if wlen >= 7: score += 30
            elif wlen >= 5: score += 15
        else:
            freq = word_frequency(word.lower(), 'en')
            syllables = textstat.syllable_count(word)
            score = 0
            if freq < 1e-6: score += 60
            elif freq < 1e-5: score += 40
            elif freq < 1e-4: score += 20
            if syllables >= 4: score += 40
            elif syllables >= 3: score += 25
            elif syllables >= 2: score += 10

        score = min(100, score)
        if score < 25: color = "#22c55e"
        elif score < 50: color = "#84cc16"
        elif score < 70: color = "#f59e0b"
        else: color = "#ef4444"
        heatmap.append({"word": word, "score": score, "color": color})
    return heatmap


# ============================================================
# MODULE 2: Facial Cognitive Load Detection (Direct from Landmarks)
# ============================================================

# Baseline calibration — first 5 frames establish the reader's neutral face
_baseline = {"ear": None, "brow_h": None, "brow_d": None, "mar": None, "n": 0,
             "s_ear": [], "s_brow_h": [], "s_brow_d": [], "s_mar": []}

def _reset_baseline():
    _baseline.update({"ear": None, "brow_h": None, "brow_d": None, "mar": None, "n": 0,
                      "s_ear": [], "s_brow_h": [], "s_brow_d": [], "s_mar": []})

# Persistent FaceMesh instance — avoids re-loading the model every frame
import mediapipe as mp
_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=True, min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)


def detect_face_landmarks(frame_base64):
    try:
        if ',' in frame_base64:
            frame_base64 = frame_base64.split(',')[1]
        img_bytes = base64.b64decode(frame_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Could not decode frame"}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = _face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return {"face_detected": False, "difficulty": "Easy", "difficulty_score": 50,
                    "difficulty_color": "#f59e0b", "cognitive_state": "No face detected",
                    "landmarks": [], "signals": {}}

        lm = res.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # ---- Landmark groups for frontend drawing ----
        LG = {
            "face_oval": [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109],
            "left_eye": [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398],
            "right_eye": [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],
            "left_eyebrow": [276,283,282,295,285,300,293,334,296,336],
            "right_eyebrow": [46,53,52,65,55,70,63,105,66,107],
            "lips_outer": [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185],
            "lips_inner": [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95],
        }
        landmark_groups = {}
        for name, ids in LG.items():
            landmark_groups[name] = [{"x": round(lm.landmark[i].x*w), "y": round(lm.landmark[i].y*h)} for i in ids]

        # Bbox from face oval only (fast, no need to iterate 468 points)
        oval_xs = [lm.landmark[i].x*w for i in LG["face_oval"]]
        oval_ys = [lm.landmark[i].y*h for i in LG["face_oval"]]
        bbox = {"x": round(min(oval_xs)-10), "y": round(min(oval_ys)-10),
                "w": round(max(oval_xs)-min(oval_xs)+20), "h": round(max(oval_ys)-min(oval_ys)+20)}

        # ---- Compute 4 key facial measurements ----
        face_h = max(abs(lm.landmark[10].y - lm.landmark[152].y), 0.001)

        # 1. EAR — Eye Aspect Ratio
        def ear6(idx):
            v1 = abs(lm.landmark[idx[1]].y - lm.landmark[idx[5]].y)
            v2 = abs(lm.landmark[idx[2]].y - lm.landmark[idx[4]].y)
            hz = max(abs(lm.landmark[idx[0]].x - lm.landmark[idx[3]].x), 0.001)
            return (v1 + v2) / (2.0 * hz)

        avg_ear = (ear6([362,385,387,263,373,380]) + ear6([33,160,158,133,153,144])) / 2.0

        # 2. Eyebrow height (how raised the brows are)
        brow_h = ((lm.landmark[159].y - lm.landmark[70].y) + (lm.landmark[386].y - lm.landmark[300].y)) / (2*face_h)

        # 3. Inner brow distance (furrowing = brows get closer)
        brow_d = abs(lm.landmark[107].x - lm.landmark[336].x) / face_h

        # 4. MAR — Mouth Aspect Ratio
        mouth_v = abs(lm.landmark[13].y - lm.landmark[14].y)
        mouth_hz = max(abs(lm.landmark[78].x - lm.landmark[308].x), 0.001)
        mar = mouth_v / mouth_hz

        features = {"eye_openness": round(avg_ear, 4), "brow_height": round(brow_h, 4),
                    "brow_furrow": round(brow_d, 4), "mouth_tension": round(mar, 4)}

        # ---- Baseline calibration (first 5 frames) ----
        if _baseline["n"] < 5:
            _baseline["s_ear"].append(avg_ear)
            _baseline["s_brow_h"].append(brow_h)
            _baseline["s_brow_d"].append(brow_d)
            _baseline["s_mar"].append(mar)
            _baseline["n"] += 1
            if _baseline["n"] == 5:
                _baseline["ear"] = sum(_baseline["s_ear"]) / 5
                _baseline["brow_h"] = sum(_baseline["s_brow_h"]) / 5
                _baseline["brow_d"] = sum(_baseline["s_brow_d"]) / 5
                _baseline["mar"] = sum(_baseline["s_mar"]) / 5

            return {"face_detected": True, "difficulty": "Calibrating", "difficulty_score": 50,
                    "difficulty_color": "#3b82f6",
                    "cognitive_state": f"Calibrating baseline... ({_baseline['n']}/5) — keep a neutral face",
                    "landmarks": landmark_groups, "bbox": bbox,
                    "signals": {}, "features": features, "deltas": {}}

        # ---- Compute deviations from baseline ----
        b = _baseline
        ear_pct = (avg_ear - b["ear"]) / max(b["ear"], 0.001) * 100
        brow_h_pct = (brow_h - b["brow_h"]) / max(abs(b["brow_h"]), 0.001) * 100
        brow_d_pct = (brow_d - b["brow_d"]) / max(b["brow_d"], 0.001) * 100
        mar_pct = (mar - b["mar"]) / max(b["mar"], 0.001) * 100

        signals = {}
        is_hard = False

        # Eyes squinting -> Hard (EAR drops)
        if ear_pct < -10:
            signals["eyes_squinting"] = True
            is_hard = True
        else:
            signals["eyes_squinting"] = False

        # Eyebrows raised -> Hard (brow height increases)
        if brow_h_pct > 10:
            signals["brows_raised"] = True
            is_hard = True
        else:
            signals["brows_raised"] = False

        # Eyebrows furrowed -> Hard (inner brow distance shrinks)
        if brow_d_pct < -8:
            signals["brows_furrowed"] = True
            is_hard = True
        else:
            signals["brows_furrowed"] = False

        # Binary: any signal = Hard, no signal = Easy
        if is_hard:
            score = 80
            diff, color, state = "Hard", "#ef4444", "Struggling — high cognitive load"
        else:
            score = 10
            diff, color, state = "Easy", "#22c55e", "Relaxed — reading comfortably"

        active = [k.replace("_", " ").title() for k, v in signals.items() if v]
        if active:
            state += " (" + ", ".join(active) + ")"

        return {"face_detected": True, "difficulty": diff, "difficulty_score": score,
                "difficulty_color": color, "cognitive_state": state,
                "landmarks": landmark_groups, "bbox": bbox,
                "signals": signals, "features": features,
                "deltas": {"ear": round(ear_pct,1), "brow_h": round(brow_h_pct,1),
                           "brow_d": round(brow_d_pct,1), "mar": round(mar_pct,1)}}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "face_detected": False, "difficulty": "Easy",
                "difficulty_score": 10, "difficulty_color": "#22c55e",
                "cognitive_state": "Error: " + str(e), "landmarks": [], "signals": {}}

# ============================================================
# MODULE 3: Correlation Engine
# ============================================================

def correlation_engine(nlp_results, facial_results):
    if not nlp_results or not facial_results:
        return {"error": "Insufficient data for correlation"}

    nlp_difficulty = nlp_results.get("difficulty", "Medium")
    nlp_score = {"Easy": 25, "Medium": 50, "Hard": 75}.get(nlp_difficulty, 50)

    facial_scores = [r.get("difficulty_score", 50) for r in facial_results if r.get("face_detected")]
    if not facial_scores:
        return {"error": "No facial data available"}

    avg_facial = sum(facial_scores) / len(facial_scores)
    if avg_facial < 50: facial_difficulty = "Easy"
    else: facial_difficulty = "Hard"

    # Combined score: 60% NLP weight + 40% facial weight
    # Because the text difficulty is objective, facial is subjective per reader
    combined_score = (nlp_score * 0.6) + (avg_facial * 0.4)
    if combined_score < 40: combined_difficulty = "Easy"
    else: combined_difficulty = "Hard"

    score_diff = abs(nlp_score - avg_facial)
    match_pct = max(0, 100 - score_diff * 1.5)

    if avg_facial > nlp_score + 15:
        assessment = "Reader found the text MORE difficult than predicted"
        suggestion = "Consider simplifying vocabulary or breaking long sentences."
    elif avg_facial < nlp_score - 15:
        assessment = "Reader found the text EASIER than predicted"
        suggestion = "The reader's comprehension level may be above average for this text."
    else:
        assessment = "Reader experience MATCHES the predicted difficulty"
        suggestion = "The text difficulty assessment is accurate for this reader."

    # Count active signals across all frames
    signal_counts = {}
    for r in facial_results:
        if r.get("face_detected") and r.get("signals"):
            for sig, active in r["signals"].items():
                if active:
                    signal_counts[sig] = signal_counts.get(sig, 0) + 1

    # ---- Reading Stability Score (std deviation of facial scores) ----
    # Low std-dev = consistent state; high std-dev = jumpy/distracting text.
    n = len(facial_scores)
    if n > 1:
        mean = sum(facial_scores) / n
        variance = sum((s - mean) ** 2 for s in facial_scores) / n
        std_dev = variance ** 0.5
    else:
        std_dev = 0.0
    # Map std_dev (0..~40) → stability 100..0
    stability_score = max(0, min(100, round(100 - (std_dev * 2.5), 1)))
    if stability_score >= 75:
        stability_label = "Stable Reading"
        stability_desc = "Reader stayed in a consistent state throughout the text."
    elif stability_score >= 50:
        stability_label = "Moderately Stable"
        stability_desc = "Some variation — a few difficulty spikes detected."
    else:
        stability_label = "Distracted / Unstable"
        stability_desc = "Frequent jumps between easy and hard — the text may contain sudden difficulty spikes."

    # ---- Reader Profile (based on dominant signal) ----
    if signal_counts:
        dominant = max(signal_counts, key=signal_counts.get)
        profiles = {
            "eyes_squinting": {
                "label": "Visual-Detailed Reader",
                "icon": "👁️",
                "desc": "Tends to analyze fine details and concentrate visually on the text."
            },
            "brows_furrowed": {
                "label": "Deep Analytical Reader",
                "icon": "🧠",
                "desc": "Tries to deeply understand meaning and reason about the content."
            },
            "brows_raised": {
                "label": "Surprised / Exploratory Reader",
                "icon": "😲",
                "desc": "Often encounters unexpected or unfamiliar content while reading."
            },
        }
        reader_profile = profiles.get(dominant, {
            "label": "Balanced Reader", "icon": "📖",
            "desc": "No dominant reading pattern detected."
        })
    else:
        reader_profile = {
            "label": "Comfortable Reader", "icon": "😌",
            "desc": "No difficulty signals — read the text comfortably."
        }

    # ---- Cognitive Fatigue Alert (longest streak of consecutive Hard frames) ----
    longest_hard_streak = 0
    current_streak = 0
    for r in facial_results:
        if r.get("face_detected") and r.get("difficulty") == "Hard":
            current_streak += 1
            longest_hard_streak = max(longest_hard_streak, current_streak)
        else:
            current_streak = 0

    FATIGUE_THRESHOLD = 10
    fatigue_alert = longest_hard_streak >= FATIGUE_THRESHOLD
    if fatigue_alert:
        fatigue_message = (
            f"⚠️ The reader showed sustained cognitive strain for "
            f"{longest_hard_streak} consecutive frames. The system recommends "
            f"simplifying the current paragraph or giving the reader a short break."
        )
    else:
        fatigue_message = "No prolonged fatigue detected — reader maintained healthy cognitive load."

    return {
        "nlp_difficulty": nlp_difficulty, "nlp_score": nlp_score,
        "facial_difficulty": facial_difficulty, "facial_score": round(avg_facial, 1),
        "combined_difficulty": combined_difficulty, "combined_score": round(combined_score, 1),
        "match_percentage": round(match_pct, 1),
        "reader_assessment": assessment, "suggestion": suggestion,
        "signal_counts": signal_counts,
        "total_frames_analyzed": len(facial_scores),
        # New analytics
        "stability_score": stability_score,
        "stability_label": stability_label,
        "stability_desc": stability_desc,
        "std_dev": round(std_dev, 2),
        "reader_profile": reader_profile,
        "fatigue_alert": fatigue_alert,
        "fatigue_streak": longest_hard_streak,
        "fatigue_threshold": FATIGUE_THRESHOLD,
        "fatigue_message": fatigue_message,
    }


# ============================================================
# MODULE 4: PDF Report
# ============================================================

# Register Arabic-compatible font (bundled with project)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
try:
    _font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'fonts')
    pdfmetrics.registerFont(TTFont('ArabicFont', os.path.join(_font_dir, 'FreeSerif.ttf')))
    pdfmetrics.registerFont(TTFont('ArabicFont-Bold', os.path.join(_font_dir, 'FreeSerifBold.ttf')))
    ARABIC_FONT = 'ArabicFont'
    ARABIC_FONT_BOLD = 'ArabicFont-Bold'
except:
    ARABIC_FONT = 'Helvetica'
    ARABIC_FONT_BOLD = 'Helvetica-Bold'

def _ar(text):
    """Reshape Arabic text for proper PDF rendering (RTL + letter joining)."""
    if not text or not re.search(r'[\u0600-\u06FF]', str(text)):
        return str(text)
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(str(text))
        return get_display(reshaped)
    except:
        return str(text)

def _cell(text):
    """Prepare text for PDF table cell — reshape Arabic if needed."""
    return _ar(text)


def generate_pdf_report(session_data):
    report_id = str(uuid.uuid4())[:8]
    filename = f"report_{report_id}.pdf"
    filepath = os.path.join('static/reports', filename)

    doc = SimpleDocTemplate(filepath, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    # Detect if content is Arabic
    lang = session_data.get("nlp_analysis", {}).get("language", "en")
    is_ar = lang == "ar"
    font = ARABIC_FONT if is_ar else 'Helvetica'
    font_bold = ARABIC_FONT_BOLD if is_ar else 'Helvetica-Bold'
    align = 2 if is_ar else 0  # 2=RIGHT for Arabic, 0=LEFT for English

    title_s = ParagraphStyle('T', parent=styles['Title'], fontName=font_bold, fontSize=20, spaceAfter=25,
                              textColor=colors.HexColor('#1e293b'), alignment=1)
    head_s = ParagraphStyle('H', parent=styles['Heading2'], fontName=font_bold, fontSize=13, spaceAfter=10,
                             textColor=colors.HexColor('#334155'), alignment=align)
    body_s = ParagraphStyle('B', parent=styles['Normal'], fontName=font, fontSize=10, leading=15,
                             textColor=colors.HexColor('#475569'), alignment=align)

    elements = []
    elements.append(Paragraph(_ar("Text Difficulty Analysis Report" if not is_ar else "تقرير تحليل صعوبة النص"), title_s))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_s))
    elements.append(Spacer(1, 15))

    # Common table style
    def tbl_style():
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), font),
            ('FONTNAME', (0, 0), (-1, 0), font_bold),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])])

    nlp = session_data.get("nlp_analysis", {})
    if nlp and "error" not in nlp:
        elements.append(Paragraph(_ar("1. Text Readability Analysis" if not is_ar else "1. تحليل سهولة القراءة"), head_s))
        data = [[_cell("Metric" if not is_ar else "المقياس"), _cell("Value" if not is_ar else "القيمة")],
                [_cell("Flesch Reading Ease"), str(nlp.get("flesch_score", "-"))],
                [_cell("Flesch-Kincaid Grade"), str(nlp.get("flesch_kincaid_grade", "-"))],
                [_cell("Gunning Fog Index"), str(nlp.get("gunning_fog", "-"))],
                [_cell("SMOG Index"), str(nlp.get("smog_index", "-"))],
                [_cell("Avg Sentence Length" if not is_ar else "متوسط طول الجملة"), str(nlp.get("avg_sentence_length", "-"))],
                [_cell("Difficult Words %" if not is_ar else "% الكلمات الصعبة"), f"{nlp.get('difficult_words_pct', 0)}%"],
                [_cell("Overall Difficulty" if not is_ar else "الصعوبة الكلية"), _cell(nlp.get("difficulty", "-"))]]
        elements.append(Table(data, colWidths=[200, 200], style=tbl_style()))
        elements.append(Spacer(1, 15))

    corr = session_data.get("correlation", {})
    if corr and "error" not in corr:
        elements.append(Paragraph(_ar("2. Reading Session Correlation" if not is_ar else "2. ارتباط جلسة القراءة"), head_s))
        data2 = [[_cell("Metric" if not is_ar else "المقياس"), _cell("Value" if not is_ar else "القيمة")],
                 [_cell("NLP Predicted" if not is_ar else "توقع NLP"), _cell(corr.get("nlp_difficulty", "-"))],
                 [_cell("Facial Difficulty" if not is_ar else "صعوبة تعبيرات الوجه"), _cell(corr.get("facial_difficulty", "-"))],
                 [_cell("Match %" if not is_ar else "% التطابق"), f"{corr.get('match_percentage', 0)}%"],
                 [_cell("Frames Analyzed" if not is_ar else "الإطارات المحللة"), str(corr.get("total_frames_analyzed", 0))],
                 [_cell("Assessment" if not is_ar else "التقييم"), _cell(corr.get("reader_assessment", "-"))]]
        elements.append(Table(data2, colWidths=[200, 200], style=tbl_style()))
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(_ar(f"Suggestion: {corr.get('suggestion', '')}"), body_s))
        elements.append(Spacer(1, 15))

    # AI Simplification section
    word_sug = session_data.get("word_suggestions", {})
    sent_sug = session_data.get("sentence_suggestions", {})

    if word_sug or sent_sug:
        elements.append(Spacer(1, 15))
        elements.append(Paragraph(_ar("3. AI Smart Simplification" if not is_ar else "3. التبسيط الذكي بالذكاء الاصطناعي"), head_s))

        if word_sug:
            elements.append(Paragraph(_ar("Simpler Word Alternatives:" if not is_ar else "بدائل أبسط للكلمات:"), body_s))
            elements.append(Spacer(1, 5))
            word_data = [[_cell("Hard Word" if not is_ar else "الكلمة الصعبة"),
                           _cell("Simpler Alternative" if not is_ar else "البديل الأبسط")]]
            for hard, simple in word_sug.items():
                word_data.append([_cell(hard), _cell(simple)])
            tw = Table(word_data, colWidths=[200, 200])
            tw.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, -1), font),
                ('FONTNAME', (0, 0), (-1, 0), font_bold),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TEXTCOLOR', (0, 1), (0, -1), colors.HexColor('#ef4444')),
                ('TEXTCOLOR', (1, 1), (1, -1), colors.HexColor('#22863a')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])]))
            elements.append(tw)
            elements.append(Spacer(1, 10))

        if sent_sug:
            elements.append(Paragraph(_ar("Simplified Sentences:" if not is_ar else "الجمل المبسطة:"), body_s))
            elements.append(Spacer(1, 5))
            sent_style_hard = ParagraphStyle('SH', parent=styles['Normal'], fontName=font, fontSize=9, leading=14,
                                              textColor=colors.HexColor('#ef4444'), alignment=align)
            sent_style_easy = ParagraphStyle('SE', parent=styles['Normal'], fontName=font, fontSize=9, leading=14,
                                              textColor=colors.HexColor('#22863a'), alignment=align)
            for original, simplified in sent_sug.items():
                short_orig = original[:120] + '...' if len(original) > 120 else original
                elements.append(Paragraph(_ar(f"Original: {short_orig}" if not is_ar else f"الأصل: {short_orig}"), sent_style_hard))
                elements.append(Paragraph(_ar(f"Simplified: {simplified}" if not is_ar else f"المبسط: {simplified}"), sent_style_easy))
                elements.append(Spacer(1, 8))

    elements.append(Spacer(1, 30))
    foot = ParagraphStyle('F', parent=styles['Normal'], fontName=font, fontSize=8, textColor=colors.HexColor('#94a3b8'), alignment=1)
    elements.append(Paragraph("Text Difficulty Evaluator — UQU Graduation Project F12", foot))
    doc.build(elements)
    return filename


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze')
def analyze_page():
    return render_template('analyze.html')

@app.route('/read')
def read_page():
    return render_template('read.html')

@app.route('/api/analyze-text', methods=['POST'])
def api_analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify({
        "analysis": analyze_text(text),
        "complex_words": extract_complex_words(text),
        "sentences": generate_sentence_difficulty(text),
        "heatmap": generate_word_heatmap_data(text),
    })

@app.route('/api/detect-emotion', methods=['POST'])
def api_detect_emotion():
    data = request.get_json()
    frame = data.get('frame', '')
    if not frame:
        return jsonify({"error": "No frame provided"}), 400
    return jsonify(detect_face_landmarks(frame))

@app.route('/api/reset-baseline', methods=['POST'])
def api_reset_baseline():
    _reset_baseline()
    return jsonify({"status": "ok"})

@app.route('/api/correlate', methods=['POST'])
def api_correlate():
    data = request.get_json()
    return jsonify(correlation_engine(data.get('nlp_results', {}), data.get('facial_results', [])))


@app.route('/api/simplify', methods=['POST'])
def api_simplify():
    """
    AI-powered simplification — called AFTER session ends.
    Takes hard words and sentences, returns simpler alternatives.
    Uses OpenAI GPT-4o-mini in a single batch call for speed.
    """
    data = request.get_json()
    hard_words = data.get('hard_words', [])
    hard_sentences = data.get('hard_sentences', [])
    api_key = data.get('api_key', '')
    lang = data.get('language', 'en')

    if not api_key:
        return jsonify({"error": "No API key provided. Add your OpenAI API key in Settings."}), 400

    if not hard_words and not hard_sentences:
        return jsonify({"word_suggestions": {}, "sentence_suggestions": {}})

    # Build prompt based on language
    prompt_parts = []
    if hard_words:
        words_list = ", ".join(hard_words[:20])
        if lang == "ar":
            prompt_parts.append(f"الكلمات الصعبة: {words_list}\nلكل كلمة، أعطِ مرادفاً أبسط أو شرحاً قصيراً بالعربية.")
        else:
            prompt_parts.append(f"HARD WORDS: {words_list}\nFor each word, give a simpler synonym or short explanation.")

    if hard_sentences:
        for i, sent in enumerate(hard_sentences[:5]):
            if lang == "ar":
                prompt_parts.append(f"جملة صعبة {i+1}: {sent}\nأعد كتابة هذه الجملة بلغة أبسط وأسهل فهماً بالعربية.")
            else:
                prompt_parts.append(f"HARD SENTENCE {i+1}: {sent}\nRewrite this sentence in simpler, easier-to-understand language.")

    if lang == "ar":
        prompt = """أنت مساعد قراءة يساعد في تبسيط النصوص الصعبة باللغة العربية.
بناءً على الكلمات والجمل الصعبة أدناه، قدم بدائل أبسط بالعربية.

أجب بصيغة JSON فقط (بدون markdown أو backticks) بهذا الشكل بالضبط:
{
  "words": {"الكلمة_الصعبة": "البديل_الأبسط", ...},
  "sentences": {"الجملة_الأصلية": "النسخة_المبسطة", ...}
}

""" + "\n\n".join(prompt_parts)
    else:
        prompt = """You are a reading assistant that helps simplify difficult text.
Given the hard words and sentences below, provide simpler alternatives.

Return your response as JSON ONLY (no markdown, no backticks) with this exact format:
{
  "words": {"difficult_word": "simpler alternative", ...},
  "sentences": {"original sentence": "simplified version", ...}
}

""" + "\n\n".join(prompt_parts)

    try:
        import urllib.request
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1500,
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        )
        resp = urllib.request.urlopen(req, timeout=15)
        result = json.loads(resp.read().decode())
        content = result["choices"][0]["message"]["content"].strip()

        # Parse JSON from response (handle possible markdown wrapping)
        content = content.replace("```json", "").replace("```", "").strip()
        suggestions = json.loads(content)

        return jsonify({
            "word_suggestions": suggestions.get("words", {}),
            "sentence_suggestions": suggestions.get("sentences", {}),
        })
    except Exception as e:
        return jsonify({"error": f"AI simplification failed: {str(e)}",
                        "word_suggestions": {}, "sentence_suggestions": {}})

@app.route('/api/generate-quiz', methods=['POST'])
def api_generate_quiz():
    data = request.get_json()
    text = data.get('text', '')
    lang = data.get('language', 'en')
    api_key = data.get('api_key', '')
    num_questions = min(data.get('num_questions', 2), 3)

    if not api_key:
        return jsonify({"error": "No API key provided"}), 400
    if not text or len(text.strip()) < 20:
        return jsonify({"error": "Text too short for quiz generation"}), 400

    if lang == "ar":
        prompt = f"""أنت معلم يقيّم مدى فهم الطالب للنص.
بناءً على النص التالي، أنشئ {num_questions} أسئلة اختيار من متعدد لقياس الفهم والاستيعاب.
كل سؤال يجب أن يكون له 4 خيارات مع إجابة صحيحة واحدة فقط.

النص:
{text[:2000]}

أجب بصيغة JSON فقط (بدون markdown أو backticks) بهذا الشكل بالضبط:
{{
  "questions": [
    {{
      "question": "نص السؤال",
      "options": ["الخيار أ", "الخيار ب", "الخيار ج", "الخيار د"],
      "correct": 0
    }}
  ]
}}
حيث correct هو رقم فهرس الإجابة الصحيحة (يبدأ من 0)."""
    else:
        prompt = f"""You are a teacher assessing a student's comprehension of a text.
Based on the following text, create {num_questions} multiple-choice questions to measure understanding.
Each question should have 4 options with exactly one correct answer.

Text:
{text[:2000]}

Return JSON ONLY (no markdown, no backticks) with this exact format:
{{
  "questions": [
    {{
      "question": "Question text here",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct": 0
    }}
  ]
}}
Where correct is the zero-based index of the correct answer."""

    try:
        import urllib.request
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 1200,
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        )
        resp = urllib.request.urlopen(req, timeout=20)
        result = json.loads(resp.read().decode())
        content = result["choices"][0]["message"]["content"].strip()
        content = content.replace("```json", "").replace("```", "").strip()
        quiz_data = json.loads(content)
        return jsonify(quiz_data)
    except Exception as e:
        return jsonify({"error": f"Quiz generation failed: {str(e)}"}), 500


@app.route('/api/evaluate-quiz', methods=['POST'])
def api_evaluate_quiz():
    data = request.get_json()
    questions = data.get('questions', [])
    answers = data.get('answers', [])
    combined_difficulty = data.get('combined_difficulty', 'Medium')
    lang = data.get('language', 'en')

    if not questions or not answers:
        return jsonify({"error": "No questions or answers provided"}), 400

    correct_count = 0
    total = len(questions)
    results = []

    for i, q in enumerate(questions):
        user_answer = answers[i] if i < len(answers) else -1
        is_correct = (user_answer == q.get('correct', -1))
        if is_correct:
            correct_count += 1
        results.append({
            "question": q["question"],
            "user_answer": user_answer,
            "correct_answer": q["correct"],
            "is_correct": is_correct,
        })

    score_pct = round((correct_count / total) * 100) if total > 0 else 0

    if score_pct >= 80:
        level = "متقدم" if lang == "ar" else "Advanced"
        level_key = "advanced"
    elif score_pct >= 50:
        level = "متوسط" if lang == "ar" else "Intermediate"
        level_key = "intermediate"
    else:
        level = "مبتدئ" if lang == "ar" else "Beginner"
        level_key = "beginner"

    if lang == "ar":
        if level_key == "advanced":
            message = "أحسنت! لقد أظهرت فهماً ممتازاً للنص. أنت جاهز لتحدي مستوى أعلى! 🚀"
            suggestion = "ننصحك بالانتقال إلى نصوص أكثر تحدياً لتطوير مهاراتك."
        elif level_key == "intermediate":
            message = "أداء جيد! فهمك للنص متوسط. مع المزيد من الممارسة ستتحسن بشكل ملحوظ. 📚"
            suggestion = "حاول إعادة قراءة الأجزاء الصعبة واستخدم البدائل المبسطة المقترحة."
        else:
            message = "لا بأس! يبدو أنك بحاجة إلى نصوص أسهل لتعزيز ثقتك بنفسك. 💪"
            suggestion = "ننصحك بالبدء بنصوص أبسط والتدرج في الصعوبة تدريجياً."
    else:
        if level_key == "advanced":
            message = "Excellent! You demonstrated outstanding comprehension. You're ready for a higher-level challenge! 🚀"
            suggestion = "We recommend moving to more challenging texts to develop your skills."
        elif level_key == "intermediate":
            message = "Good job! Your comprehension is at an intermediate level. With more practice, you'll improve significantly. 📚"
            suggestion = "Try re-reading the difficult parts and use the suggested simplified alternatives."
        else:
            message = "Don't worry! It seems you need easier texts to build your confidence. 💪"
            suggestion = "We recommend starting with simpler texts and gradually increasing difficulty."

    return jsonify({
        "correct_count": correct_count,
        "total": total,
        "score_pct": score_pct,
        "level": level,
        "level_key": level_key,
        "message": message,
        "suggestion": suggestion,
        "results": results,
    })


@app.route('/api/generate-report', methods=['POST'])
def api_generate_report():
    data = request.get_json()
    filename = generate_pdf_report(data)
    return jsonify({"filename": filename, "url": f"/static/reports/{filename}"})

@app.route('/api/download-report/<filename>')
def download_report(filename):
    fp = os.path.join('static/reports', secure_filename(filename))
    if os.path.exists(fp):
        return send_file(fp, as_attachment=True)
    return jsonify({"error": "Report not found"}), 404

if __name__ == '__main__':
    # Pre-load MediaPipe to avoid first-request delay
    try:
        import mediapipe as mp
        print("[OK] MediaPipe loaded successfully")
    except Exception as e:
        print(f"[WARN] MediaPipe load warning: {e}")

    print("\nStarting TextDifficulty.ai server...")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
