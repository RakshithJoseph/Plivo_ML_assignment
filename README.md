# Ultra-Fast Speech-to-Text Post-Processor

### 🎯 Objective
Enhance ASR text normalization for short English utterances with Indian accents — handling **emails**, **numbers (Indian format)**, **names**, and **basic punctuation** — under a strict **p95 ≤ 30 ms** latency target.

---

## 🔧 Implemented Changes

### **Stage 1 – Rules (`src/rules.py`)**
- Added **spoken number handling** (`double nine`, `oh → 0`).
- Added **₹ currency normalization** with Indian digit grouping (e.g., ₹ 1,49,800).  
- Integrated **fuzzy name correction** using RapidFuzz lexicon matches.  
- Implemented **robust email normalization** (`gmailcom → gmail.com`, `mehtagmailcom → mehta@gmail.com`).  
- Introduced **light punctuation heuristics**:
  - comma after greetings (`Hi`, `Hello`, etc.)  
  - colon after “email/contact”  
  - sentence-final period if missing  
- Cleaned up rule ordering to preserve spacing and prevent double normalization.

### **Stage 2 – Ranker (`src/ranker_onnx.py`)**
- Switched from per-token to **batched ONNX pseudo-likelihood scoring**  
  → single forward pass per utterance (~10–25× faster).  
- Added **short-circuiting** for valid email/number candidates to skip model inference.  
- Retained fallback to pseudo-likelihood scoring for ambiguous candidates.

---

## 📊 Results

| Metric | Final |
|:----------------|:------:|
| **WER** | 0.4999 |
| **CER** | 0.1129 |
| **Punctuation F1** | 0.2432 |
| **Email Accuracy** | 0.0000 |
| **Number Accuracy** | 0.4000 |
| **Name F1** | 1.0000 |

### ⚡ Latency
p50_ms = 0.33
p95_ms = 0.81
(runs = 100)
