"""
biosentinel/inference.py
─────────────────────────────────────────────────────────────────────
BioSentinel inference engine.

This is what your Streamlit app calls once the model is trained.
It accepts raw session data captured from the browser frontend
and returns a fraud score between 0.0 and 1.0.

Usage
-----
from biosentinel.inference import BioSentinelScorer

scorer = BioSentinelScorer()   # loads model once, reuses across sessions

# Option A — pass pre-collected feature dict
features = {
    "inter_key_delay":      115.0,
    "key_hold_duration":     78.0,
    "keystroke_variance":    42.0,
    ...
}
result = scorer.score_from_features(features)
print(result["fraud_score"])    # 0.18
print(result["signal"])         # "CLEAN"
print(result["class_probs"])    # {"human": 0.82, "bot": 0.05, ...}

# Option B — pass a raw sequence array directly (shape: seq_len x n_features)
result = scorer.score_from_sequence(sequence_array)
"""

import os
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow import keras
from .data_generator import FEATURE_COLUMNS, N_FEATURES
from .model import SEQ_LEN, fraud_score_from_probs, fraud_score_weighted

# ── Default model paths ──
_DIR        = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(_DIR, "saved_model", "biosentinel_lstm.h5")
SCALER_PATH = os.path.join(_DIR, "saved_model", "scaler.pkl")

# ── Label names ──
LABEL_NAMES = ["human", "bot", "duress", "coached"]

# ── Score thresholds ──
THRESHOLDS = {
    "CLEAN":      (0.0,  0.30),
    "SUSPICIOUS": (0.30, 0.60),
    "HIGH RISK":  (0.60, 0.80),
    "CRITICAL":   (0.80, 1.01),
}


def _score_to_signal(score: float) -> str:
    for signal, (lo, hi) in THRESHOLDS.items():
        if lo <= score < hi:
            return signal
    return "CRITICAL"


def _score_to_emoji(signal: str) -> str:
    return {"CLEAN": "🟢", "SUSPICIOUS": "🟡", "HIGH RISK": "🟠", "CRITICAL": "🔴"}.get(signal, "🔴")


# ─────────────────────────────────────────────────────────────────────
class BioSentinelScorer:
    """
    Singleton-style inference wrapper.
    Load once, score many sessions.
    """

    def __init__(
        self,
        model_path:  str = MODEL_PATH,
        scaler_path: str = SCALER_PATH,
        seq_len:     int = SEQ_LEN,
        weighted:    bool = True,
    ):
        self.seq_len  = seq_len
        self.weighted = weighted
        self._model   = None
        self._scaler  = None
        self._model_path  = model_path
        self._scaler_path = scaler_path

    def _load(self):
        """Lazy-load model and scaler on first call."""
        if self._model is None:
            if not os.path.exists(self._model_path):
                raise FileNotFoundError(
                    f"Model not found at {self._model_path}.\n"
                    "Run:  python -m biosentinel.train"
                )
            self._model  = keras.models.load_model(self._model_path)

        if self._scaler is None:
            if not os.path.exists(self._scaler_path):
                raise FileNotFoundError(
                    f"Scaler not found at {self._scaler_path}.\n"
                    "Run:  python -m biosentinel.train"
                )
            with open(self._scaler_path, "rb") as f:
                self._scaler = pickle.load(f)

    # ── Internal ──────────────────────────────────────────────────

    def _features_to_vector(self, features: dict) -> np.ndarray:
        """Convert a feature dict to a 1D array in the correct column order."""
        vec = np.array([features.get(col, 0.0) for col in FEATURE_COLUMNS], dtype="float32")
        return vec

    def _predict_sequence(self, sequence: np.ndarray) -> dict:
        """
        Run inference on a (seq_len, n_features) array.
        Returns a result dict.
        """
        self._load()

        # Scale
        seq_scaled = self._scaler.transform(sequence)   # (seq_len, n_features)

        # Reshape for model: (1, seq_len, n_features)
        X = seq_scaled[np.newaxis, :, :]

        # Predict
        probs = self._model.predict(X, verbose=0)[0]    # shape (4,)

        # Fraud score
        score = fraud_score_weighted(probs) if self.weighted else fraud_score_from_probs(probs)
        score = float(np.clip(score, 0.0, 1.0))

        signal = _score_to_signal(score)
        emoji  = _score_to_emoji(signal)

        return {
            "fraud_score": round(score, 4),
            "signal":      signal,
            "emoji":       emoji,
            "predicted_class": LABEL_NAMES[int(np.argmax(probs))],
            "class_probs": {
                name: round(float(p), 4)
                for name, p in zip(LABEL_NAMES, probs)
            },
            "flags": _generate_flags(score, probs),
        }

    # ── Public API ────────────────────────────────────────────────

    def score_from_sequence(self, sequence: np.ndarray) -> dict:
        """
        Score a pre-built (seq_len, n_features) numpy array.

        Parameters
        ----------
        sequence : np.ndarray  shape (seq_len, n_features)
            Raw (unscaled) feature values in FEATURE_COLUMNS order.

        Returns
        -------
        dict with keys: fraud_score, signal, emoji,
                        predicted_class, class_probs, flags
        """
        assert sequence.shape == (self.seq_len, N_FEATURES), (
            f"Expected shape ({self.seq_len}, {N_FEATURES}), "
            f"got {sequence.shape}"
        )
        return self._predict_sequence(sequence)

    def score_from_features(self, features: dict) -> dict:
        """
        Score a single session from a feature dictionary.
        Repeats the feature vector seq_len times to form the sequence
        (suitable when you only have aggregate session stats, not time-series).

        Parameters
        ----------
        features : dict   — keys matching FEATURE_COLUMNS

        Returns
        -------
        dict with keys: fraud_score, signal, emoji,
                        predicted_class, class_probs, flags
        """
        vec      = self._features_to_vector(features)
        sequence = np.tile(vec, (self.seq_len, 1))   # (seq_len, n_features)
        return self._predict_sequence(sequence)

    def score_from_js_payload(self, payload: dict) -> dict:
        """
        Score from the raw JavaScript event payload your frontend captures.
        Parses the payload into features automatically.

        Expected payload keys (all optional — missing = 0):
            keystrokes         : list of {key, down_time, up_time} dicts
            mouse_events       : list of {x, y, timestamp, type} dicts
            form_events        : list of {field, enter_time, exit_time} dicts
            paste_count        : int
            session_start_ms   : int (epoch ms)
            session_end_ms     : int (epoch ms)
        """
        features = _parse_js_payload(payload)
        return self.score_from_features(features)


# ─────────────────────────────────────────────────────────────────────
# JS PAYLOAD PARSER
# ─────────────────────────────────────────────────────────────────────

def _parse_js_payload(payload: dict) -> dict:
    """
    Convert raw browser event lists to the 16 feature values.
    Called by score_from_js_payload().
    """
    features = {col: 0.0 for col in FEATURE_COLUMNS}

    # ── Keystroke dynamics ──
    keystrokes = payload.get("keystrokes", [])
    if len(keystrokes) >= 2:
        down_times = [k["down_time"] for k in keystrokes]
        up_times   = [k["up_time"]   for k in keystrokes]
        holds      = [u - d for u, d in zip(up_times, down_times)]
        ikds       = [down_times[i+1] - up_times[i] for i in range(len(down_times)-1)]
        backspaces = sum(1 for k in keystrokes if k.get("key") in ("Backspace","Delete"))

        features["inter_key_delay"]     = float(np.mean(ikds))       if ikds   else 0.0
        features["key_hold_duration"]   = float(np.mean(holds))       if holds  else 0.0
        features["keystroke_variance"]  = float(np.std(ikds))         if ikds   else 0.0
        features["backspace_rate"]      = backspaces / max(len(keystrokes), 1)
        # words per minute estimate (avg 5 chars per word)
        total_chars = len(keystrokes)
        total_sec   = (down_times[-1] - down_times[0]) / 1000 if down_times else 1
        features["typing_speed_wpm"]    = (total_chars / 5) / (total_sec / 60) if total_sec else 0

    # ── Mouse behaviour ──
    mouse = payload.get("mouse_events", [])
    move_events = [m for m in mouse if m.get("type") == "move"]
    if len(move_events) >= 2:
        velocities = []
        for i in range(1, len(move_events)):
            dx = move_events[i]["x"] - move_events[i-1]["x"]
            dy = move_events[i]["y"] - move_events[i-1]["y"]
            dt = (move_events[i]["timestamp"] - move_events[i-1]["timestamp"]) / 1000
            if dt > 0:
                velocities.append(np.sqrt(dx**2 + dy**2) / dt)
        if velocities:
            features["mouse_velocity"]     = float(np.mean(velocities))
            features["mouse_acceleration"] = float(np.std(np.diff(velocities))) if len(velocities) > 1 else 0.0

    click_events = [m for m in mouse if m.get("type") in ("mousedown","mouseup")]
    click_pairs  = [(click_events[i], click_events[i+1])
                    for i in range(0, len(click_events)-1, 2)
                    if click_events[i]["type"] == "mousedown"]
    if click_pairs:
        dwells = [p[1]["timestamp"] - p[0]["timestamp"] for p in click_pairs]
        features["click_pressure_var"] = float(np.std(dwells))

    scroll_events = [m for m in mouse if m.get("type") == "scroll"]
    if len(scroll_events) > 1:
        scroll_deltas = [abs(scroll_events[i]["y"] - scroll_events[i-1]["y"])
                         for i in range(1, len(scroll_events))]
        features["scroll_jitter"] = float(np.std(scroll_deltas))

    # ── Form field interaction ──
    form_events = payload.get("form_events", [])
    if form_events:
        dwell_times = [e["exit_time"] - e["enter_time"] for e in form_events]
        features["avg_field_dwell"]       = float(np.mean(dwell_times))
        features["field_dwell_var"]       = float(np.std(dwell_times))
        features["hesitation_count"]      = sum(1 for d in dwell_times if d > 3000)

        expected_order = list(range(len(form_events)))
        actual_order   = [e.get("field_index", i) for i, e in enumerate(form_events)]
        deviations     = sum(1 for e, a in zip(expected_order, actual_order) if e != a)
        features["tab_order_deviations"]  = float(deviations)

    features["copy_paste_count"] = float(payload.get("paste_count", 0))

    # ── Session timing ──
    start = payload.get("session_start_ms", 0)
    end   = payload.get("session_end_ms",   0)
    if end > start:
        features["total_session_time"] = (end - start) / 1000

    if form_events and start:
        first_keystroke = payload.get("keystrokes", [{}])[0].get("down_time", start)
        features["first_field_latency"] = (first_keystroke - start) / 1000

    return features


# ─────────────────────────────────────────────────────────────────────
# FLAG GENERATOR
# ─────────────────────────────────────────────────────────────────────

def _generate_flags(score: float, probs: np.ndarray) -> list[str]:
    """Generate human-readable flag descriptions from score and class probs."""
    flags = []
    p_bot, p_duress, p_coached = probs[1], probs[2], probs[3]

    if score >= 0.8:
        flags.append("🔴 CRITICAL fraud signal — immediate review required")
    if p_bot > 0.4:
        flags.append(f"🤖 Bot-like behaviour detected (P={p_bot:.2f}) — possible automated submission")
    if p_duress > 0.35:
        flags.append(f"⚠️ Duress pattern detected (P={p_duress:.2f}) — applicant may be under coercion")
    if p_coached > 0.35:
        flags.append(f"👥 Coached behaviour detected (P={p_coached:.2f}) — third-party involvement suspected")
    if score >= 0.3 and not flags:
        flags.append("🟡 Elevated behavioral variance — additional verification recommended")
    if not flags:
        flags.append("✅ No behavioral anomalies detected")

    return flags


# ─────────────────────────────────────────────────────────────────────
# QUICK TEST  (without a trained model — uses random weights)
# ─────────────────────────────────────────────────────────────────────

def _demo_score(fraud_score_override: float = None) -> dict:
    """
    Returns a mock result dict for UI testing without a trained model.
    Pass fraud_score_override to simulate different risk levels.
    """
    score  = fraud_score_override if fraud_score_override is not None else 0.18
    signal = _score_to_signal(score)
    emoji  = _score_to_emoji(signal)
    p_human = 1.0 - score
    probs_arr = np.array([p_human, score * 0.5, score * 0.3, score * 0.2])
    probs_arr = probs_arr / probs_arr.sum()

    return {
        "fraud_score":     round(score, 4),
        "signal":          signal,
        "emoji":           emoji,
        "predicted_class": LABEL_NAMES[int(np.argmax(probs_arr))],
        "class_probs":     {name: round(float(p), 4) for name, p in zip(LABEL_NAMES, probs_arr)},
        "flags":           _generate_flags(score, probs_arr),
    }


if __name__ == "__main__":
    # Demo without trained model
    for score in [0.1, 0.45, 0.72, 0.91]:
        r = _demo_score(score)
        print(f"Score {score} → {r['emoji']} {r['signal']}  |  flags: {r['flags'][0]}")
