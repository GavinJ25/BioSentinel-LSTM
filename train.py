"""
biosentinel/train.py
─────────────────────────────────────────────────────────────────────
Training pipeline for BioSentinel LSTM.

Run from the project root:
    python -m biosentinel.train

Outputs saved to  biosentinel/saved_model/
    biosentinel_lstm.h5      — trained Keras model
    scaler.pkl               — fitted StandardScaler
    label_encoder.pkl        — class index → label name
    training_report.txt      — accuracy, loss, confusion matrix
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .data_generator import generate_dataset, make_session_sequence, FEATURE_COLUMNS
from .model          import build_model, SEQ_LEN, N_FEATURES, fraud_score_from_probs

# ── Paths ──
SAVE_DIR   = os.path.join(os.path.dirname(__file__), "saved_model")
MODEL_PATH = os.path.join(SAVE_DIR, "biosentinel_lstm.h5")
SCALER_PATH= os.path.join(SAVE_DIR, "scaler.pkl")
REPORT_PATH= os.path.join(SAVE_DIR, "training_report.txt")
PLOT_PATH  = os.path.join(SAVE_DIR, "training_curves.png")
CM_PATH    = os.path.join(SAVE_DIR, "confusion_matrix.png")

LABEL_NAMES = ["human", "bot", "duress", "coached"]


# ─────────────────────────────────────────────────────────────────────
def prepare_sequences(X_raw: np.ndarray, y_raw: np.ndarray, seq_len: int):
    """
    Build overlapping LSTM sequences and align labels.
    Each sequence takes the label of its LAST row.
    """
    X_seq = make_session_sequence(X_raw, seq_len)
    y_seq = y_raw[seq_len - 1:]   # label = last row of each sequence
    return X_seq, y_seq


# ─────────────────────────────────────────────────────────────────────
def train(
    n_human:   int = 800,
    n_bot:     int = 600,
    n_duress:  int = 400,
    n_coached: int = 400,
    seq_len:   int = SEQ_LEN,
    epochs:    int = 60,
    batch_size:int = 32,
    test_size: float = 0.2,
    seed:      int = 42,
    verbose:   int = 1,
):
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 1. Generate data ──────────────────────────────────────────
    print("Generating synthetic behavioral dataset...")
    X_raw, y_raw, df = generate_dataset(
        n_human=n_human, n_bot=n_bot,
        n_duress=n_duress, n_coached=n_coached, seed=seed
    )
    print(f"  Raw samples   : {X_raw.shape[0]}")
    print(f"  Features      : {X_raw.shape[1]}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:10s}    : {(y_raw == i).sum()}")

    # ── 2. Scale features ─────────────────────────────────────────
    print("\nFitting StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved → {SCALER_PATH}")

    # ── 3. Build sequences ────────────────────────────────────────
    print(f"\nBuilding LSTM sequences (seq_len={seq_len})...")
    X_seq, y_seq = prepare_sequences(X_scaled, y_raw, seq_len)
    print(f"  Sequence shape : {X_seq.shape}")

    # ── 4. Train / test split ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=test_size, random_state=seed, stratify=y_seq
    )
    print(f"  Train          : {X_train.shape[0]} samples")
    print(f"  Test           : {X_test.shape[0]} samples")

    # ── 5. Build model ────────────────────────────────────────────
    print("\nBuilding BioSentinel LSTM model...")
    model = build_model(seq_len=seq_len, n_features=N_FEATURES)
    if verbose:
        model.summary()

    # ── 6. Callbacks ──────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    # ── 7. Train ──────────────────────────────────────────────────
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    # ── 8. Evaluate ───────────────────────────────────────────────
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss      : {test_loss:.4f}")
    print(f"  Test Accuracy  : {test_acc:.4f}")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_test, y_pred, target_names=LABEL_NAMES)
    print("\nClassification Report:")
    print(report)

    # ── 9. Confusion matrix plot ──────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("BioSentinel — Confusion Matrix", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()
    print(f"\n  Confusion matrix → {CM_PATH}")

    # ── 10. Training curves plot ──────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"],     label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.set_xlabel("Epoch")
    ax2.plot(history.history["accuracy"],     label="Train Acc")
    ax2.plot(history.history["val_accuracy"], label="Val Acc")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.set_xlabel("Epoch")
    fig.suptitle("BioSentinel — Training Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"  Training curves → {PLOT_PATH}")

    # ── 11. Save model ────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"  Model saved   → {MODEL_PATH}")

    # ── 12. Save text report ──────────────────────────────────────
    with open(REPORT_PATH, "w") as f:
        f.write("BioSentinel — Training Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Loss     : {test_loss:.4f}\n")
        f.write(f"Test Accuracy : {test_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
    print(f"  Report saved  → {REPORT_PATH}")

    print("\n✅ Training complete.")
    return model, scaler, history


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train(verbose=1)
