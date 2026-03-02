"""
biosentinel/data_generator.py
─────────────────────────────────────────────────────────────────────
Generates synthetic behavioral session data for BioSentinel training.

Each session simulates a user filling out the loan application form.
Features captured per session (all timing in milliseconds):

KEYSTROKE DYNAMICS
  - inter_key_delay     : avg time between keystrokes
  - key_hold_duration   : avg time a key is held down
  - keystroke_variance  : std dev of inter-key delays (regularity)
  - backspace_rate      : fraction of keystrokes that are backspaces
  - typing_speed_wpm    : estimated words per minute

MOUSE BEHAVIOUR
  - mouse_velocity      : avg cursor speed (px/sec)
  - mouse_acceleration  : rate of velocity change
  - click_pressure_var  : variance in click duration (dwell time)
  - scroll_jitter       : irregularity in scroll behavior

FORM INTERACTION
  - field_dwell_times   : list of time spent on each form field
  - avg_field_dwell     : mean dwell time across all fields
  - field_dwell_var     : variance in dwell times
  - tab_order_deviations: number of out-of-order field jumps
  - copy_paste_count    : number of paste events detected
  - total_session_time  : end-to-end form completion time (seconds)
  - hesitation_count    : pauses > 3s mid-field
  - first_field_latency : time before user starts typing (seconds)

LABELS
  0 = Human (genuine applicant)
  1 = Bot or synthetic identity
  2 = Duress / social engineering
  3 = Human-assisted fraud (coached applicant)
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# FEATURE COLUMN ORDER  (must match model input)
# ─────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "inter_key_delay",
    "key_hold_duration",
    "keystroke_variance",
    "backspace_rate",
    "typing_speed_wpm",
    "mouse_velocity",
    "mouse_acceleration",
    "click_pressure_var",
    "scroll_jitter",
    "avg_field_dwell",
    "field_dwell_var",
    "tab_order_deviations",
    "copy_paste_count",
    "total_session_time",
    "hesitation_count",
    "first_field_latency",
]

N_FEATURES = len(FEATURE_COLUMNS)   # 16


# ─────────────────────────────────────────────────────────────────────
# PROFILE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────

def _human(n: int, seed: int = None) -> np.ndarray:
    """
    Genuine human applicant.
    Natural variation, occasional hesitation, normal typing speed.
    """
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.normal(120,  30,  n),    # inter_key_delay (ms)       — natural rhythm
        rng.normal(80,   15,  n),    # key_hold_duration (ms)     — normal press
        rng.normal(40,   12,  n),    # keystroke_variance         — some variation
        rng.beta(1.5, 12, n),        # backspace_rate             — occasional typos
        rng.normal(45,   10,  n),    # typing_speed_wpm           — average typist
        rng.normal(300,  80,  n),    # mouse_velocity (px/s)      — normal movement
        rng.normal(50,   15,  n),    # mouse_acceleration         — gradual
        rng.normal(60,   20,  n),    # click_pressure_var         — natural clicks
        rng.normal(25,   10,  n),    # scroll_jitter              — slight
        rng.normal(8000, 2000, n),   # avg_field_dwell (ms)       — reads fields
        rng.normal(3000, 1000, n),   # field_dwell_var            — some variation
        rng.integers(0, 3,  n).astype(float),  # tab_order_deviations  — mostly linear
        rng.integers(0, 2,  n).astype(float),  # copy_paste_count      — rare paste
        rng.normal(180,  60,  n),    # total_session_time (s)     — 3 min avg
        rng.integers(1, 5,  n).astype(float),  # hesitation_count      — normal pauses
        rng.normal(4,    2,   n),    # first_field_latency (s)    — reads before typing
    ])


def _bot(n: int, seed: int = None) -> np.ndarray:
    """
    Automated bot.
    Unnaturally fast and consistent — very low variance.
    May copy-paste all fields. Near-zero hesitation.
    """
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.normal(20,   3,   n),    # inter_key_delay           — machine fast
        rng.normal(15,   2,   n),    # key_hold_duration         — minimal hold
        rng.normal(2,    1,   n),    # keystroke_variance        — nearly zero
        rng.beta(0.5, 20, n),        # backspace_rate            — almost no errors
        rng.normal(200,  10,  n),    # typing_speed_wpm          — impossibly fast
        rng.normal(800,  20,  n),    # mouse_velocity            — linear, fast
        rng.normal(5,    2,   n),    # mouse_acceleration        — constant speed
        rng.normal(5,    2,   n),    # click_pressure_var        — identical clicks
        rng.normal(2,    1,   n),    # scroll_jitter             — none
        rng.normal(500,  50,  n),    # avg_field_dwell           — no reading
        rng.normal(30,   10,  n),    # field_dwell_var           — nearly identical
        rng.integers(0, 1,  n).astype(float),  # tab_order_deviations — always linear
        rng.integers(5, 12,  n).astype(float), # copy_paste_count     — all pasted
        rng.normal(15,   5,   n),    # total_session_time (s)    — < 30 sec
        rng.integers(0, 1,  n).astype(float),  # hesitation_count     — zero
        rng.normal(0.1,  0.05, n),   # first_field_latency (s)  — instant start
    ])


def _duress(n: int, seed: int = None) -> np.ndarray:
    """
    User under duress / social engineering.
    High hesitation, erratic mouse, re-reads fields repeatedly,
    slow and irregular with high variance throughout.
    """
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.normal(200,  80,  n),    # inter_key_delay           — slow, nervous
        rng.normal(100,  40,  n),    # key_hold_duration         — tense hold
        rng.normal(90,   30,  n),    # keystroke_variance        — very erratic
        rng.beta(3, 6,   n),         # backspace_rate            — many mistakes
        rng.normal(25,   10,  n),    # typing_speed_wpm          — slow
        rng.normal(150,  100, n),    # mouse_velocity            — erratic movement
        rng.normal(80,   40,  n),    # mouse_acceleration        — sudden changes
        rng.normal(120,  60,  n),    # click_pressure_var        — shaky clicks
        rng.normal(60,   30,  n),    # scroll_jitter             — lots of scrolling
        rng.normal(20000,6000,n),    # avg_field_dwell           — re-reads fields
        rng.normal(9000, 3000,n),    # field_dwell_var           — very inconsistent
        rng.integers(3, 9,  n).astype(float),  # tab_order_deviations — jumps around
        rng.integers(0, 2,  n).astype(float),  # copy_paste_count
        rng.normal(600,  180, n),    # total_session_time (s)    — 10+ min
        rng.integers(8, 20, n).astype(float),  # hesitation_count     — many long pauses
        rng.normal(15,   8,   n),    # first_field_latency (s)   — long delay to start
    ])


def _coached(n: int, seed: int = None) -> np.ndarray:
    """
    Human-assisted fraud (coached applicant — someone else dictating answers).
    Mix of human-like timing with unnatural hesitations and copy-paste.
    Harder to detect — intentionally blends human and bot characteristics.
    """
    rng = np.random.default_rng(seed)
    # Start with human baseline and inject fraud signals
    base = _human(n, seed)
    # Spike copy-paste (index 12) and hesitations (index 14)
    base[:, 12] = rng.integers(3, 8, n).astype(float)   # heavy paste usage
    base[:, 14] = rng.integers(5, 15, n).astype(float)  # many pauses (being told what to type)
    base[:, 10] = rng.normal(6000, 2000, n)              # inconsistent field times
    base[:, 5]  = rng.normal(180, 120, n)                # erratic mouse
    return base


# ─────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────

def generate_dataset(
    n_human:   int = 500,
    n_bot:     int = 300,
    n_duress:  int = 200,
    n_coached: int = 200,
    seed:      int = 42,
    as_dataframe: bool = True,
) -> tuple:
    """
    Generate a balanced synthetic behavioral dataset.

    Returns
    -------
    X : np.ndarray  shape (N, 16)
    y : np.ndarray  shape (N,)   — labels 0/1/2/3
    df: pd.DataFrame (only if as_dataframe=True)
    """
    X_parts = [
        _human(n_human,   seed),
        _bot(n_bot,       seed + 1),
        _duress(n_duress, seed + 2),
        _coached(n_coached, seed + 3),
    ]
    y_parts = [
        np.zeros(n_human,   dtype=int),
        np.ones(n_bot,      dtype=int),
        np.full(n_duress,   2, dtype=int),
        np.full(n_coached,  3, dtype=int),
    ]

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Clip negative values (timing can't be negative)
    X = np.clip(X, 0, None)

    # Shuffle
    rng = np.random.default_rng(seed + 99)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    if as_dataframe:
        df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
        df["label"] = y
        label_map = {0: "human", 1: "bot", 2: "duress", 3: "coached"}
        df["label_name"] = df["label"].map(label_map)
        return X, y, df

    return X, y


def make_session_sequence(X: np.ndarray, seq_len: int = 10) -> np.ndarray:
    """
    Convert flat feature rows into overlapping sequences for LSTM input.
    Each sequence of `seq_len` consecutive rows forms one training sample.

    Returns
    -------
    X_seq : np.ndarray  shape (N - seq_len + 1, seq_len, n_features)
    """
    n_samples = X.shape[0]
    sequences = []
    for i in range(n_samples - seq_len + 1):
        sequences.append(X[i : i + seq_len])
    return np.array(sequences)


if __name__ == "__main__":
    X, y, df = generate_dataset()
    print(f"Dataset shape : {X.shape}")
    print(f"Label counts  :\n{df['label_name'].value_counts()}")
    print(f"\nSample features (first human row):")
    print(df[df.label == 0].iloc[0][FEATURE_COLUMNS])
