# 🧠 BioSentinel

> Behavioral biometrics fraud detection module for loan application sessions.  
> LSTM-based classifier that scores applicant sessions as **human**, **bot**, **duress**, or **coached** — producing a real-time fraud score integrated as a hard override gate in the LendSynthetix underwriting pipeline.

---

## What It Does

When a user fills out a loan application form, their interaction generates a continuous stream of behavioral signals — how fast they type, how they move the mouse, how long they dwell on each field, whether they paste instead of type. BioSentinel captures these signals, runs them through a trained LSTM, and returns a fraud score between `0.0` and `1.0` before the application ever reaches an underwriting agent.

A score above `0.80` triggers a **Behavioral Override** — the loan is rejected regardless of credit score, income, or DTI. The agents never see it.

---

## Architecture

```
Browser Session
      │
      │  keystroke timings, mouse movement,
      │  field dwell times, paste events
      ↓
[biosentinel_capture.js]     — browser-side event capture
      │
      ↓  raw event lists (JSON payload)
[BioSentinelScorer.score_from_js_payload()]
      │
      ↓  16 aggregate features
[StandardScaler]             — fitted on training data
      │
      ↓  normalised feature vector (shape: 1 × 16)
[Sequence Builder]           — overlapping windows (seq_len=10)
      │
      ↓  shape: (1, 10, 16)
[LSTM Layer 1 — 64 units]    — short-range temporal patterns
      │
[Dropout 0.3]
      │
[LSTM Layer 2 — 32 units]    — session-level patterns
      │
[Dropout 0.3]
      │
[Dense 32 — ReLU]
      │
[Dense 4 — Softmax]          — class probabilities
      │
      ├── P(human)
      ├── P(bot)
      ├── P(duress)
      └── P(coached)
            │
            ↓
      fraud_score = weighted_sum(probs)
      signal      = CLEAN / SUSPICIOUS / HIGH RISK / CRITICAL
```

---

## The Four Behavioral Classes

| Class | Label | Description | Key Signals |
|-------|-------|-------------|-------------|
| Genuine Human | `0` | Natural applicant filling out the form themselves | Normal typing rhythm, occasional backspaces, 3–5 min session, reads fields |
| Bot | `1` | Automated script or synthetic identity | ~20ms inter-key delay, 200+ WPM, near-zero variance, all fields copy-pasted, <30s session |
| Duress | `2` | Applicant under coercion or social engineering | High hesitation (8–20 pauses), erratic mouse, re-reads fields repeatedly, 10+ min session |
| Coached | `3` | Third-party dictating answers to applicant | Human-like timing + heavy paste (3–8 events) + many hesitations (5–15), inconsistent field dwell |

Coached is intentionally the hardest class — it blends human and bot characteristics to evade detection. The LSTM's temporal layers are designed to catch the inconsistency between typing rhythm and paste behaviour that characterises this profile.

---

## The 16 Input Features

All timing values in milliseconds unless noted.

### Keystroke Dynamics
| Feature | Description |
|---------|-------------|
| `inter_key_delay` | Average time between consecutive keystrokes |
| `key_hold_duration` | Average time a key is held down |
| `keystroke_variance` | Standard deviation of inter-key delays — measures regularity |
| `backspace_rate` | Fraction of keystrokes that are backspaces |
| `typing_speed_wpm` | Estimated words per minute |

### Mouse Behaviour
| Feature | Description |
|---------|-------------|
| `mouse_velocity` | Average cursor speed (px/sec) |
| `mouse_acceleration` | Rate of velocity change |
| `click_pressure_var` | Variance in click duration |
| `scroll_jitter` | Irregularity in scroll behaviour |

### Form Interaction
| Feature | Description |
|---------|-------------|
| `avg_field_dwell` | Mean time spent on each form field (ms) |
| `field_dwell_var` | Variance in dwell times across fields |
| `tab_order_deviations` | Number of out-of-order field jumps |
| `copy_paste_count` | Number of paste events detected |
| `total_session_time` | End-to-end form completion time (seconds) |
| `hesitation_count` | Pauses longer than 3 seconds mid-field |
| `first_field_latency` | Time before user starts typing (seconds) |

---

## Fraud Score

The raw softmax output gives four class probabilities. These are combined into a single fraud score using a weighted sum:

```
fraud_score = 0.0 × P(human)
            + 1.0 × P(bot)
            + 0.6 × P(duress)
            + 0.8 × P(coached)
```

Duress is weighted at `0.6` rather than `1.0` because a duress applicant may be a victim — the underwriting pipeline treats it as an escalation trigger for human review, not an automatic rejection.

### Score Thresholds

| Score | Signal | Action in Pipeline |
|-------|--------|--------------------|
| `0.00 – 0.30` | 🟢 CLEAN | No adjustment |
| `0.31 – 0.60` | 🟡 SUSPICIOUS | +2% to PD estimate, verification requested |
| `0.61 – 0.80` | 🟠 HIGH RISK | +5% to PD, escalated to human underwriter |
| `0.81 – 1.00` | 🔴 CRITICAL | **Behavioral Override — auto-rejected** |

---

## Training

### Dataset

Synthetic behavioral sessions generated by `data_generator.py`:

| Class | Samples |
|-------|---------|
| Human | 800 |
| Bot | 600 |
| Duress | 400 |
| Coached | 400 |
| **Total** | **2,200** |

80/20 train/test split, stratified by class.

### Training Pipeline

```
1. Generate synthetic dataset        (data_generator.py)
2. Fit StandardScaler on raw features
3. Build overlapping sequences        seq_len=10, stride=1
4. Train/test split                   80/20, stratified
5. Train LSTM                         max 60 epochs
6. EarlyStopping                      patience=10, monitors val_loss
7. ReduceLROnPlateau                  factor=0.5, patience=5, min_lr=1e-6
8. Evaluate on held-out test set
9. Save confusion matrix PNG + training curves PNG
10. Save model (.h5) + scaler (.pkl)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sequence length | 10 time-steps |
| LSTM Layer 1 | 64 units, return_sequences=True |
| LSTM Layer 2 | 32 units, return_sequences=False |
| Dense hidden | 32 units, ReLU |
| Output | 4 units, Softmax |
| Dropout | 0.3 (both LSTM layers) |
| L2 regularisation | 1e-4 (all layers) |
| Optimiser | Adam, lr=1e-3 |
| Loss | Sparse categorical crossentropy |
| Batch size | 32 |
| Max epochs | 60 |

### Run Training

```bash
# From project root
python -m biosentinel.train
```

Training takes approximately 2 minutes on CPU.

Outputs saved to `biosentinel/saved_model/`:
```
biosentinel_lstm.h5        — trained Keras model
scaler.pkl                 — fitted StandardScaler
training_report.txt        — accuracy, loss, classification report
confusion_matrix.png       — per-class confusion matrix
training_curves.png        — loss and accuracy over epochs
```

---

## Inference API

```python
from biosentinel.inference import BioSentinelScorer

scorer = BioSentinelScorer()  # loads model once, reuses across sessions
```

### Option A — Feature Dict (manual / demo mode)

```python
result = scorer.score_from_features({
    "inter_key_delay":     115.0,
    "key_hold_duration":    78.0,
    "keystroke_variance":   42.0,
    "backspace_rate":        0.05,
    "typing_speed_wpm":     45.0,
    "mouse_velocity":      300.0,
    "mouse_acceleration":   50.0,
    "click_pressure_var":   60.0,
    "scroll_jitter":        25.0,
    "avg_field_dwell":    8000.0,
    "field_dwell_var":    3000.0,
    "tab_order_deviations":  0.0,
    "copy_paste_count":      0.0,
    "total_session_time":  180.0,
    "hesitation_count":      2.0,
    "first_field_latency":   4.0,
})
```

### Option B — Raw Sequence Array

```python
import numpy as np
sequence = np.zeros((10, 16), dtype="float32")   # shape: (seq_len, n_features)
result = scorer.score_from_sequence(sequence)
```

### Option C — JS Payload (live browser capture)

```python
result = scorer.score_from_js_payload(js_payload)
# js_payload is the dict returned by BioSentinel.getPayload() in the browser
```

### Result Dict

```python
{
    "fraud_score":     0.18,           # float 0–1
    "signal":          "CLEAN",        # CLEAN / SUSPICIOUS / HIGH RISK / CRITICAL
    "emoji":           "🟢",
    "predicted_class": "human",        # dominant class
    "class_probs": {
        "human":   0.82,
        "bot":     0.05,
        "duress":  0.08,
        "coached": 0.05,
    },
    "flags": [
        "✅ No behavioral anomalies detected"
    ]
}
```

---

## Browser Capture

`biosentinel_capture.js` captures raw behavioral events in the browser during form filling.

```javascript
// Initialise on page load
BioSentinel.init();

// On form submit — get payload and send to backend
const payload = BioSentinel.getPayload();
```

Captured events:
- Keystroke down/up timestamps per field
- Mouse move coordinates (throttled to 50ms)
- Click events with dwell time
- Scroll events
- Field enter/exit timestamps
- Paste event count

---

## File Structure

```
biosentinel/
├── __init__.py
├── model.py              # LSTM architecture + fraud_score_weighted()
├── data_generator.py     # Synthetic session data + FEATURE_COLUMNS
├── train.py              # Full training pipeline
├── inference.py          # BioSentinelScorer class
├── biosentinel_capture.js# Browser-side event capture
└── saved_model/          # Created after training — gitignored
    ├── biosentinel_lstm.h5
    ├── scaler.pkl
    ├── training_report.txt
    ├── confusion_matrix.png
    └── training_curves.png
```

---

## Dependencies

```
tensorflow >= 2.12
scikit-learn
numpy
pandas
matplotlib
```

All included in the project `requirements.txt`.

---

## Notes

**The model is trained on synthetic data.** In production, replace `data_generator.py` with real labeled session recordings captured from your form. The architecture, feature set, and inference pipeline remain identical — only the training data changes.

**Duress is not a rejection signal.** An applicant under duress is potentially a victim of fraud, not a perpetrator. The pipeline escalates duress cases to a human underwriter rather than auto-rejecting, which is the legally and ethically correct response under RBI Fair Practices Code.

**The JS capture script is a starting point.** In a production deployment, replace it with a hardened fingerprinting library that resists spoofing. The feature extraction logic in `inference.py` (`score_from_js_payload`) should be updated to match whatever capture method you use.
