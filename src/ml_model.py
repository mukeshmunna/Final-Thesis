"""
Arm 1: the ML model (LSTM).

This is the deterministic half of the comparison. Given the same 60 days of
prices it returns the same probability every single time - which is exactly the
property the LLM turns out not to have.

Honest evaluation
-----------------
Predictions are produced walk-forward: every test date is scored by a model that
was trained only on data from before it, with a 5-day gap in between so that no
training label can overlap a test label. The out-of-sample predictions are
written to outputs/ml_predictions.csv, and everything downstream reads that file.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

from . import config as C
from .data_loader import regime_of

PRED_FILE   = C.OUTPUT_DIR / "ml_predictions.csv"
FINAL_MODEL = C.MODEL_DIR / "lstm_final.keras"
SCALER_FILE = C.MODEL_DIR / "scaler_final.npz"


# ---------------------------------------------------------------- splitting
def walk_forward_splits(n, n_folds=None, embargo=None, min_train_frac=0.40):
    """Yield (train_idx, test_idx). A gap of `embargo` rows is deleted between them.

    The gap matters because the label for day i looks 5 days ahead. Without it,
    the last few training labels would peek into the test window.
    """
    n_folds = n_folds or C.N_FOLDS
    embargo = embargo or C.EMBARGO_DAYS
    min_train = int(n * min_train_frac)
    test_size = (n - min_train) // n_folds
    for k in range(n_folds):
        test_start = min_train + k * test_size
        test_end   = test_start + test_size if k < n_folds - 1 else n
        train_end  = test_start - embargo
        if train_end <= C.SEQ_LEN + 20:
            continue
        yield np.arange(0, train_end), np.arange(test_start, test_end)


def inner_val_split(n_train, frac=None):
    """Split a training window into (earlier part, most recent part).

    The most recent slice is the validation set, used for early stopping and for
    picking the decision threshold. The split is chronological, never random -
    shuffling would let the model validate on days that came before days it
    trained on.
    """
    frac = frac or C.VAL_FRAC
    cut = int(n_train * (1 - frac))
    return np.arange(0, cut), np.arange(cut, n_train)


def make_sequences(X, y, idx, seq_len=None):
    """Turn flat feature rows into 60-day windows.

    The window for row i covers rows [i-59, i] - the 60 days up to AND INCLUDING i.
    Including row i is correct: the features on day i use prices up to day i,
    while the label asks about day i+5.
    """
    seq_len = seq_len or C.SEQ_LEN
    keep = [i for i in idx if i >= seq_len - 1]
    if not keep:
        return (np.empty((0, seq_len, X.shape[1])), np.empty(0, dtype=int),
                np.array([], dtype=int))
    Xs = np.stack([X[i - seq_len + 1:i + 1] for i in keep])
    return Xs, y[keep], np.asarray(keep)


# ---------------------------------------------------------------- the network
def build_lstm(n_features, seq_len=None):
    """A small 2-layer LSTM.

    Deliberately small: each fold trains on only a few thousand windows, so a
    large network would memorise them rather than learn anything general.
    """
    import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, Input

    seq_len = seq_len or C.SEQ_LEN
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def fit_model(model, Xtr, ytr, Xva, yva, verbose=0):
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau

    cbs = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    model.fit(Xtr, ytr, validation_data=(Xva, yva),
              epochs=40, batch_size=128, callbacks=cbs, verbose=verbose)
    return model


# ---------------------------------------------------------------- training
def train_and_predict(bundle, verbose=True):
    """Train the pooled LSTM walk-forward and write out-of-sample predictions.

    Pooling: one model is trained across all five stocks rather than five separate
    models. A single stock gives only ~1,200 windows, which is far too few for an
    LSTM; pooling multiplies that by five. Every feature is a ratio or a return,
    so the stocks share a common scale and pooling is legitimate.
    """
    import keras
    from sklearn.preprocessing import StandardScaler

    C.set_seeds()
    tickers  = bundle["tickers"]
    features = bundle["features"]
    targets  = bundle["targets"]
    n_feat   = len(bundle["feature_names"])

    # All five stocks share one calendar, so one split serves them all.
    n_rows = min(len(features[t]) for t in tickers)
    rows   = []

    for fold, (tr_idx, te_idx) in enumerate(walk_forward_splits(n_rows), start=1):
        keras.utils.set_random_seed(C.RAND_SEED + fold)

        fit_rows, val_rows = inner_val_split(len(tr_idx))

        # The scaler sees ONLY the fitting rows - never validation, never test.
        scaler = StandardScaler().fit(
            np.vstack([features[t].values[:n_rows][fit_rows] for t in tickers]))

        Xtr_l, ytr_l, Xva_l, yva_l = [], [], [], []
        for t in tickers:
            Xs = np.clip(scaler.transform(features[t].values[:n_rows]), -5, 5)
            ys = targets[t].values[:n_rows]
            a, b, _ = make_sequences(Xs, ys, fit_rows)
            c, d, _ = make_sequences(Xs, ys, val_rows)
            Xtr_l.append(a); ytr_l.append(b); Xva_l.append(c); yva_l.append(d)

        Xtr, ytr = np.concatenate(Xtr_l), np.concatenate(ytr_l)
        Xva, yva = np.concatenate(Xva_l), np.concatenate(yva_l)

        if verbose:
            print(f"  fold {fold}: train={len(Xtr)} val={len(Xva)} windows")

        model = fit_model(build_lstm(n_feat), Xtr, ytr, Xva, yva)
        model.save(C.MODEL_DIR / f"lstm_fold{fold}.keras")

        # Decision threshold chosen on validation only, never on test.
        p_val = model.predict(Xva, verbose=0).ravel()
        best_t, best_acc = 0.5, 0.0
        for t_try in np.arange(0.30, 0.71, 0.01):
            acc = ((p_val >= t_try).astype(int) == yva).mean()
            if acc > best_acc:
                best_acc, best_t = acc, float(t_try)

        # ---- out-of-sample predictions for this fold
        for t in tickers:
            Xs = np.clip(scaler.transform(features[t].values[:n_rows]), -5, 5)
            ys = targets[t].values[:n_rows]
            Xte, yte, kept = make_sequences(Xs, ys, te_idx)
            if len(Xte) == 0:
                continue
            p = model.predict(Xte, verbose=0).ravel()
            dates = features[t].index[:n_rows][kept]
            for date, prob, truth in zip(dates, p, yte):
                reg, vix_level = regime_of(bundle["vix"], date)
                rows.append({
                    "date":      date.strftime("%Y-%m-%d"),
                    "ticker":    t,
                    "fold":      fold,
                    "p_up":      round(float(prob), 6),
                    "threshold": round(best_t, 3),
                    "y_true":    int(truth),
                    "regime":    reg,
                    "vix":       round(float(vix_level), 2),
                })

    df = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    df.to_csv(PRED_FILE, index=False)

    # ---- a final model trained on everything, used only for the live demo tab
    keras.utils.set_random_seed(C.RAND_SEED)
    fit_rows, val_rows = inner_val_split(n_rows)
    scaler = StandardScaler().fit(
        np.vstack([features[t].values[:n_rows][fit_rows] for t in tickers]))
    Xtr_l, ytr_l, Xva_l, yva_l = [], [], [], []
    for t in tickers:
        Xs = np.clip(scaler.transform(features[t].values[:n_rows]), -5, 5)
        ys = targets[t].values[:n_rows]
        a, b, _ = make_sequences(Xs, ys, fit_rows)
        c, d, _ = make_sequences(Xs, ys, val_rows)
        Xtr_l.append(a); ytr_l.append(b); Xva_l.append(c); yva_l.append(d)
    final = fit_model(build_lstm(n_feat),
                 np.concatenate(Xtr_l), np.concatenate(ytr_l),
                 np.concatenate(Xva_l), np.concatenate(yva_l))
    final.save(FINAL_MODEL)
    np.savez(SCALER_FILE, mean=scaler.mean_, scale=scaler.scale_)

    if verbose:
        acc = (( df["p_up"] >= df["threshold"]).astype(int) == df["y_true"]).mean()
        print(f"\n  {len(df)} out-of-sample predictions, accuracy {acc * 100:.2f}%")
        print(f"  written to {PRED_FILE}")

    return df


# ---------------------------------------------------------------- serving
class MLPredictor:
    """Looks up the walk-forward prediction for a date, or runs the live model.

    The app uses this. Reading a CSV keeps start-up instant; Keras is imported
    only if a live (unseen-date) prediction is actually requested.
    """

    def __init__(self):
        if not PRED_FILE.exists():
            raise FileNotFoundError(
                f"No ML predictions at {PRED_FILE}. Run train_and_predict() first."
            )
        self.table = pd.read_csv(PRED_FILE)
        self._key = {(r.date, r.ticker): r for r in self.table.itertuples()}
        self._model = None
        self._scaler = None

    @property
    def available_dates(self):
        return sorted(self.table["date"].unique())

    def predict(self, ticker, date):
        """Return {'p_up', 'direction', 'confidence', ...} for a backtested date."""
        date = pd.Timestamp(date).strftime("%Y-%m-%d")
        row = self._key.get((date, ticker))
        if row is None:
            return None
        p = float(row.p_up)
        return {
            "arm":        "ML",
            "p_up":       p,
            "direction":  "UP" if p >= row.threshold else "DOWN",
            "confidence": p if p >= row.threshold else 1 - p,
            "threshold":  float(row.threshold),
            "y_true":     int(row.y_true),
            "regime":     row.regime,
            "vix":        float(row.vix),
            "source":     "walk-forward out-of-sample",
        }

    def predict_live(self, bundle, ticker, date):
        """Run the final model directly. Used for dates outside the backtest."""
        import keras

        if self._model is None:
            self._model = keras.models.load_model(FINAL_MODEL)
            z = np.load(SCALER_FILE)
            self._scaler = (z["mean"], z["scale"])

        feats = bundle["features"][ticker]
        prior = feats.index[feats.index <= pd.Timestamp(date)]
        if len(prior) < C.SEQ_LEN:
            return None
        window = feats.loc[prior[-C.SEQ_LEN:]].values
        mean, scale = self._scaler
        Xs = np.clip((window - mean) / scale, -5, 5)[None, ...]
        p = float(self._model.predict(Xs, verbose=0).ravel()[0])
        reg, vix_level = regime_of(bundle["vix"], date)
        return {
            "arm":        "ML",
            "p_up":       p,
            "direction":  "UP" if p >= 0.5 else "DOWN",
            "confidence": p if p >= 0.5 else 1 - p,
            "threshold":  0.5,
            "y_true":     None,
            "regime":     reg,
            "vix":        float(vix_level),
            "source":     "final model (live)",
        }
