"""Predicting customer bookings (Forage BA job simulation, Task 2).

Question: from the details of a booking enquiry, how well can we predict whether the
customer completes the booking, and which variables carry the signal?

Run:  python booking_model.py customer_booking.csv --out-dir results
Needs: numpy, pandas, scikit-learn, matplotlib.
The data file is NOT part of the repository (it belongs to the Forage simulation).
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
SEED = 42
DAYS = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
EXTRAS = ["wants_extra_baggage", "wants_preferred_seat", "wants_in_flight_meals"]
TARGET = "booking_complete"


def load(path):
    df = pd.read_csv(path, encoding="ISO-8859-1")
    assert df[TARGET].isin([0, 1]).all(), "target must be 0/1"
    return df


def prepare(df, top_origins=15):
    """Return (X, y, groups). `groups` maps each model column to the original variable it
    came from, so importance can be reported per variable, not per dummy column."""
    y = df[TARGET].astype(int)
    d = df.drop(columns=[TARGET]).copy()
    if not pd.api.types.is_numeric_dtype(d["flight_day"]):  # text days (Mon..Sun) -> 1..7; leave numeric days alone
        d["flight_day"] = d["flight_day"].map(DAYS)
    d["is_weekend_flight"] = d["flight_day"].isin([6, 7]).astype(int)
    d["n_extras_requested"] = d[EXTRAS].sum(axis=1)
    # booking_origin has 104 countries; keep the most common and pool the rest
    keep = d["booking_origin"].value_counts().head(top_origins).index
    d["booking_origin"] = d["booking_origin"].where(d["booking_origin"].isin(keep), "Other")
    d = d.drop(columns=["route"])  # 799 routes, most with few rows; flight_duration carries route length
    X = pd.get_dummies(d, columns=["sales_channel", "trip_type", "booking_origin"], dtype=int)
    groups = {}
    for c in X.columns:
        base = c
        for prefix in ("sales_channel", "trip_type", "booking_origin"):
            if c.startswith(prefix + "_"):
                base = prefix
        groups.setdefault(base, []).append(c)
    return X, y, groups


def make_model(n_estimators=200):
    return RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=5,
                                  class_weight="balanced_subsample", n_jobs=-1,
                                  random_state=SEED)


def grouped_permutation_importance(model, X_te, y_te, groups, n_repeats, rng):
    """Drop in ROC AUC when ALL columns of one original variable are shuffled together."""
    base = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    out = {}
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X_te.copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].to_numpy()[perm]
            drops.append(base - roc_auc_score(y_te, model.predict_proba(Xp)[:, 1]))
        out[name] = float(np.mean(drops))
    return out


def cross_validate_model(X, y, groups, n_splits=5, n_repeats=2, n_estimators=200):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    rng = np.random.default_rng(SEED)
    oof = np.zeros(len(y))
    fold_rows, perm_rows, impurity_rows = [], [], []
    for k, (tr, te) in enumerate(skf.split(X, y), 1):
        m = make_model(n_estimators).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        fold_rows.append({"fold": k, "roc_auc": roc_auc_score(y.iloc[te], p),
                          "pr_auc": average_precision_score(y.iloc[te], p)})
        perm_rows.append(grouped_permutation_importance(m, X.iloc[te], y.iloc[te], groups, n_repeats, rng))
        imp = pd.Series(m.feature_importances_, X.columns)
        impurity_rows.append({g: imp[c].sum() for g, c in groups.items()})
    return oof, pd.DataFrame(fold_rows), pd.DataFrame(perm_rows), pd.DataFrame(impurity_rows)


def threshold_table(y, p):
    rows = []
    for t in (0.5, 0.4, 0.3, 0.25, 0.2):
        pred = (p >= t).astype(int)
        rows.append({"threshold": t, "accuracy": accuracy_score(y, pred),
                     "precision": precision_score(y, pred), "recall": recall_score(y, pred),
                     "f1": f1_score(y, pred), "share_flagged": pred.mean()})
    return pd.DataFrame(rows)


def origin_rates(df, min_rows=500):
    """Completion rate by booking country, for countries with at least `min_rows` enquiries."""
    g = df.groupby("booking_origin")[TARGET].agg(completion_rate="mean", rows="size")
    return g[g.rows >= min_rows].sort_values("rows", ascending=False)


def plot_importance(imp, path):
    imp = imp.sort_values("permutation_mean")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(imp.index, imp["permutation_mean"], xerr=imp["permutation_sd"], color="#6d3fc0",
            ecolor="#25252b", capsize=2)
    ax.set_xlabel("Drop in ROC AUC when the variable is shuffled (higher = more important)")
    ax.set_title("What drives the prediction (held-out folds)", loc="left", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=2, help="permutation repeats per fold")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    df = load(a.csv)
    X, y, groups = prepare(df)
    oof, folds, perm, impurity = cross_validate_model(X, y, groups, n_repeats=a.repeats, n_estimators=a.trees)

    summary = pd.DataFrame([
        {"metric": "rows", "value": len(y)},
        {"metric": "completed_bookings_share", "value": y.mean()},
        {"metric": "accuracy_of_always_predicting_no_booking", "value": 1 - y.mean()},
        {"metric": "roc_auc_pooled_oof", "value": roc_auc_score(y, oof)},
        {"metric": "roc_auc_fold_mean", "value": folds.roc_auc.mean()},
        {"metric": "roc_auc_fold_sd", "value": folds.roc_auc.std()},
        {"metric": "pr_auc_pooled_oof", "value": average_precision_score(y, oof)},
    ])
    summary.to_csv(os.path.join(a.out_dir, "cv_summary.csv"), index=False)
    folds.to_csv(os.path.join(a.out_dir, "cv_folds.csv"), index=False)
    threshold_table(y, oof).to_csv(os.path.join(a.out_dir, "threshold_table.csv"), index=False)

    origin_rates(df).to_csv(os.path.join(a.out_dir, "origin_rates.csv"))
    imp = pd.DataFrame({"permutation_mean": perm.mean(), "permutation_sd": perm.std(),
                        "impurity_mean": impurity.mean()}).sort_values("permutation_mean", ascending=False)
    imp.to_csv(os.path.join(a.out_dir, "feature_importance.csv"))
    plot_importance(imp, os.path.join(a.out_dir, "feature_importance.png"))

    print(summary.to_string(index=False))
    print(threshold_table(y, oof).round(3).to_string(index=False))
    print(imp.round(4).to_string())


if __name__ == "__main__":
    main()
