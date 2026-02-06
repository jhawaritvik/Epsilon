import json
import math
import os
import warnings
from dataclasses import dataclass

import numpy as np

from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# -------------------------
# Utilities
# -------------------------

def to_py(obj):
    """Convert numpy/scipy types to native Python for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(x) for x in obj]
    return obj


def corrected_resampled_ttest(diffs, test_fraction=0.2):
    """Nadeau & Bengio corrected resampled t-test for repeated holdout.

    diffs: array of paired differences (best - baseline) across outer repeats.

    Returns: dict with mean_diff, se_corrected, t_stat, df, ci_low, ci_high
    and p_two_sided (t distribution).

    Note: The experiment spec requests this; we implement deterministically.
    """
    diffs = np.asarray(diffs, dtype=float)
    J = len(diffs)
    mean_diff = diffs.mean()
    s2 = diffs.var(ddof=1) if J > 1 else 0.0

    # corrected variance of mean difference
    # var = (1/J + p/(1-p)) * s^2
    p = float(test_fraction)
    corr = (1.0 / J) + (p / (1.0 - p))
    var_mean = corr * s2
    se = math.sqrt(var_mean) if var_mean > 0 else 0.0

    df = J - 1
    t_stat = mean_diff / se if se > 0 else (np.inf if mean_diff > 0 else (-np.inf if mean_diff < 0 else 0.0))

    # Compute CI and p-value using scipy if available, else approximate via numpy (not ideal)
    out = {
        "J": J,
        "test_fraction": p,
        "mean_diff": mean_diff,
        "sample_var": s2,
        "se_corrected": se,
        "t_stat": t_stat,
        "df": df,
    }

    try:
        from scipy.stats import t as tdist
        alpha = 0.05
        tcrit = float(tdist.ppf(1 - alpha/2, df)) if df > 0 else np.nan
        ci_low = mean_diff - tcrit * se if se > 0 and df > 0 else np.nan
        ci_high = mean_diff + tcrit * se if se > 0 and df > 0 else np.nan
        p_two = float(2 * (1 - tdist.cdf(abs(t_stat), df))) if df > 0 else np.nan
        out.update({
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "p_two_sided": p_two,
            "test": "nadeau_bengio_corrected_resampled_t_test"
        })
    except Exception as e:
        out.update({
            "ci_95_low": None,
            "ci_95_high": None,
            "p_two_sided": None,
            "test": "nadeau_bengio_corrected_resampled_t_test",
            "scipy_unavailable_or_error": str(e)
        })

    return out


def wilcoxon_fallback(diffs):
    diffs = np.asarray(diffs, dtype=float)
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(diffs)
        return {"test": "wilcoxon_signed_rank", "stat": float(stat), "p_two_sided": float(p)}
    except Exception as e:
        return {"test": "wilcoxon_signed_rank", "stat": None, "p_two_sided": None, "scipy_unavailable_or_error": str(e)}


# -------------------------
# Experiment parameters (from spec)
# -------------------------

DATA_PARAMS = dict(
    n_samples=5000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_repeated=0,
    n_classes=2,
    weights=None,
    class_sep=1.0,
    flip_y=0.01,
)

OUTER_TEST_FRACTION = 0.2
J_REPEATS = 30
SEED_BASE = 12345

# Baseline configuration (pre-specified)
BASELINE = {
    "learning_rate_init": 1e-3,
    "batch_size": 64,
    "hidden_layer_sizes": (50,),
}

# Hyperparameter grid (evaluated)
PARAM_GRID = {
    "mlp__learning_rate_init": [1e-4, 1e-3, 1e-2],
    "mlp__batch_size": [32, 64, 128],
    "mlp__hidden_layer_sizes": [(20,), (50,), (100,)],
}

# MLP fixed controls
MLP_CONTROLS = dict(
    solver="adam",
    activation="relu",
    alpha=0.0001,
    learning_rate="constant",
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    tol=1e-4,
    max_iter=500,  # per revision hints: >=300; use 500 for robustness
    random_state=0,
)


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(**MLP_CONTROLS)),
    ])


def main():
    # Save dataset resolution info
    dataset_used = {
        "dataset_source": "procedural",
        "generator": "sklearn.datasets.make_classification",
        "data_params": DATA_PARAMS,
        "fresh_dataset_per_resample": True,
        "seed_schedule": {"base": SEED_BASE, "outer_repeats": J_REPEATS, "seed": "base + j"},
        "outer_test_fraction": OUTER_TEST_FRACTION,
    }
    with open("dataset_used.json", "w") as f:
        json.dump(to_py(dataset_used), f, indent=2)

    outer_results = []
    outer_best_params = []

    # Track warnings/non-convergence
    total_fits = 0
    convwarn_count = 0

    # For visualization
    best_accs = []
    base_accs = []

    for j in range(J_REPEATS):
        seed = SEED_BASE + j
        X, y = make_classification(random_state=seed, **DATA_PARAMS)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=OUTER_TEST_FRACTION,
            stratify=y,
            random_state=seed,
        )

        # Inner loop: 5-fold stratified CV for hyperparameter selection
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        pipe = build_pipeline()

        # GridSearchCV will refit best on full outer-train
        gscv = GridSearchCV(
            estimator=pipe,
            param_grid=PARAM_GRID,
            scoring="accuracy",
            n_jobs=-1,
            cv=inner_cv,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )

        with warnings.catch_warnings(record=True) as wgrid:
            warnings.simplefilter("always", category=ConvergenceWarning)
            gscv.fit(X_train, y_train)

        # Count ConvergenceWarnings during grid search
        # Conservative: count each warning instance
        convwarn_count += sum(1 for ww in wgrid if issubclass(ww.category, ConvergenceWarning))
        # Estimate total fits = candidates * cv
        total_fits += len(gscv.cv_results_["params"]) * inner_cv.get_n_splits()

        best_model = gscv.best_estimator_
        best_params = gscv.best_params_

        # Evaluate on outer test
        y_pred = best_model.predict(X_test)
        y_proba = None
        try:
            y_proba = best_model.predict_proba(X_test)
            best_ll = log_loss(y_test, y_proba)
        except Exception:
            best_ll = None

        best_acc = accuracy_score(y_test, y_pred)

        # Baseline fit on outer-train
        base_pipe = build_pipeline()
        base_pipe.set_params(
            mlp__learning_rate_init=BASELINE["learning_rate_init"],
            mlp__batch_size=BASELINE["batch_size"],
            mlp__hidden_layer_sizes=BASELINE["hidden_layer_sizes"],
        )

        with warnings.catch_warnings(record=True) as wbase:
            warnings.simplefilter("always", category=ConvergenceWarning)
            base_pipe.fit(X_train, y_train)
        convwarn_count += sum(1 for ww in wbase if issubclass(ww.category, ConvergenceWarning))
        total_fits += 1

        yb_pred = base_pipe.predict(X_test)
        try:
            yb_proba = base_pipe.predict_proba(X_test)
            base_ll = log_loss(y_test, yb_proba)
        except Exception:
            base_ll = None
        base_acc = accuracy_score(y_test, yb_pred)

        # Convergence diagnostics (n_iter_, loss_)
        best_mlp = best_model.named_steps["mlp"]
        base_mlp = base_pipe.named_steps["mlp"]

        outer_results.append({
            "repeat": j,
            "seed": seed,
            "best": {
                "params": best_params,
                "test_accuracy": best_acc,
                "test_log_loss": best_ll,
                "n_iter": getattr(best_mlp, "n_iter_", None),
                "final_loss": getattr(best_mlp, "loss_", None),
            },
            "baseline": {
                "params": {
                    "mlp__learning_rate_init": BASELINE["learning_rate_init"],
                    "mlp__batch_size": BASELINE["batch_size"],
                    "mlp__hidden_layer_sizes": BASELINE["hidden_layer_sizes"],
                },
                "test_accuracy": base_acc,
                "test_log_loss": base_ll,
                "n_iter": getattr(base_mlp, "n_iter_", None),
                "final_loss": getattr(base_mlp, "loss_", None),
            },
        })

        outer_best_params.append(best_params)
        best_accs.append(best_acc)
        base_accs.append(base_acc)

    best_accs = np.asarray(best_accs, dtype=float)
    base_accs = np.asarray(base_accs, dtype=float)
    diffs = best_accs - base_accs

    # Primary test
    primary = corrected_resampled_ttest(diffs, test_fraction=OUTER_TEST_FRACTION)

    # Fallback test
    fallback = wilcoxon_fallback(diffs)

    # Aggregate best params frequency
    from collections import Counter
    c = Counter(json.dumps(p, sort_keys=True) for p in outer_best_params)
    most_common_json, most_common_count = c.most_common(1)[0]
    most_common_params = json.loads(most_common_json)

    # Metrics summary
    summary = {
        "accuracy": float(best_accs.mean()),
        "final_accuracy": float(best_accs.mean()),
        "baseline_accuracy_mean": float(base_accs.mean()),
        "best_minus_baseline_accuracy_mean": float(diffs.mean()),
        "best_accuracy_std": float(best_accs.std(ddof=1)),
        "baseline_accuracy_std": float(base_accs.std(ddof=1)),
        "paired_diff_std": float(diffs.std(ddof=1)),
        "outer_repeats_J": J_REPEATS,
        "outer_test_fraction": OUTER_TEST_FRACTION,
        "inner_cv_folds": 5,
        "grid_size": int(len(PARAM_GRID["mlp__learning_rate_init"]) * len(PARAM_GRID["mlp__batch_size"]) * len(PARAM_GRID["mlp__hidden_layer_sizes"])),
        "baseline": {
            "learning_rate_init": BASELINE["learning_rate_init"],
            "batch_size": BASELINE["batch_size"],
            "hidden_layer_sizes": BASELINE["hidden_layer_sizes"],
        },
        "most_frequent_selected_best_params": most_common_params,
        "most_frequent_selected_best_params_count": int(most_common_count),
        "convergence": {
            "total_fits_estimated": int(total_fits),
            "convergence_warnings_count": int(convwarn_count),
            "convergence_warning_rate": float(convwarn_count / total_fits) if total_fits > 0 else None,
        },
        "primary_test": primary,
        "fallback_test": fallback,
        "outer_repeat_results": outer_results,
    }

    # Visualization: accuracy per repeat
    x = np.arange(J_REPEATS)
    plt.figure(figsize=(10, 5))
    plt.plot(x, best_accs, label="Selected best (outer test acc)", marker='o', linewidth=1)
    plt.plot(x, base_accs, label="Baseline (outer test acc)", marker='s', linewidth=1)
    plt.xlabel("Outer repeat")
    plt.ylabel("Test accuracy")
    plt.title("Outer-loop test accuracy across repeated stratified holdout (J=30)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison_plot.png", dpi=150)
    plt.close()

    # Save raw results
    with open("raw_results.json", "w") as f:
        json.dump(to_py(summary), f, indent=2)


if __name__ == "__main__":
    main()
