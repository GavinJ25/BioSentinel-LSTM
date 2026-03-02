"""
biosentinel/model.py
─────────────────────────────────────────────────────────────────────
BioSentinel LSTM Architecture

The model learns temporal patterns in a sequence of behavioral
feature vectors captured during a form-filling session.

Architecture:
    Input: (batch, seq_len=10, n_features=16)
        ↓
    LSTM(64) → returns sequences
        ↓
    Dropout(0.3)
        ↓
    LSTM(32) → returns last hidden state only
        ↓
    Dropout(0.3)
        ↓
    Dense(32, relu)
        ↓
    Dense(4, softmax)  → class probabilities [human, bot, duress, coached]
        ↓
    fraud_score = 1 - P(human)   → single 0–1 risk score

Why two LSTM layers?
  - First layer captures short-range temporal patterns
    (e.g. a burst of fast keystrokes)
  - Second layer captures session-level patterns
    (e.g. consistent unnaturally fast typing throughout)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info logs

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers


SEQ_LEN    = 10    # number of time-steps per sequence
N_FEATURES = 16    # must match FEATURE_COLUMNS in data_generator.py
N_CLASSES  = 4     # human / bot / duress / coached


def build_model(
    seq_len:    int   = SEQ_LEN,
    n_features: int   = N_FEATURES,
    n_classes:  int   = N_CLASSES,
    lstm1_units: int  = 64,
    lstm2_units: int  = 32,
    dense_units: int  = 32,
    dropout:    float = 0.3,
    l2_reg:     float = 1e-4,
) -> keras.Model:
    """
    Build and return the compiled BioSentinel LSTM model.
    """
    inp = keras.Input(shape=(seq_len, n_features), name="behavioral_sequence")

    # ── First LSTM ── captures short-range patterns
    x = layers.LSTM(
        lstm1_units,
        return_sequences=True,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="lstm_1"
    )(inp)
    x = layers.Dropout(dropout, name="dropout_1")(x)

    # ── Second LSTM ── captures session-level patterns
    x = layers.LSTM(
        lstm2_units,
        return_sequences=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="lstm_2"
    )(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)

    # ── Dense head ──
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_1"
    )(x)

    # ── Output: class probabilities ──
    out = layers.Dense(n_classes, activation="softmax", name="class_probs")(x)

    model = keras.Model(inputs=inp, outputs=out, name="BioSentinel")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def fraud_score_from_probs(probs: np.ndarray) -> float:
    """
    Convert softmax class probabilities to a single fraud score.

    fraud_score = 1 - P(human)

    Weighted alternative if you want nuanced scoring:
        fraud_score = 0*P(human) + 1*P(bot) + 0.7*P(duress) + 0.5*P(coached)

    Returns a float in [0, 1].
    """
    p_human = float(probs[0])
    return round(1.0 - p_human, 4)


def fraud_score_weighted(probs: np.ndarray) -> float:
    """
    Weighted fraud score — gives nuanced output based on fraud type.
    bot and coached score higher than duress (which may be a victim).
    """
    weights = np.array([0.0, 1.0, 0.6, 0.8])   # human / bot / duress / coached
    score   = float(np.dot(probs, weights))
    return round(min(score, 1.0), 4)


if __name__ == "__main__":
    model = build_model()
    model.summary()

    # Quick shape test
    dummy = np.random.rand(8, SEQ_LEN, N_FEATURES).astype("float32")
    out   = model(dummy, training=False)
    print(f"\nOutput shape : {out.shape}")
    print(f"Sample probs : {out[0].numpy()}")
    print(f"Fraud score  : {fraud_score_from_probs(out[0].numpy())}")
