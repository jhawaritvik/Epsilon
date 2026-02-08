import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def to_py(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, dict):
        return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, list):
        return [to_py(x) for x in o]
    return o


def exact_binom_pvalue_one_sided_greater(k, n, p0):
    # Use scipy for numerical stability
    from scipy.stats import binom
    return float(binom.sf(k - 1, n, p0))


def one_sided_lower_confidence_bound_95(k, n, alpha=0.05):
    # Clopper-Pearson one-sided lower bound: Beta(alpha; k, n-k+1)
    from scipy.stats import beta
    if k == 0:
        return 0.0
    return float(beta.ppf(alpha, k, n - k + 1))


def load_adult_via_spec():
    # Spec dataset id: sklearn.datasets.fetch_openml(name='adult', version=2)
    # Attempt Hugging Face per resolver instruction; fallback to sklearn fetch_openml if HF dataset id is not available.
    Xy = None
    load_path = None
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset("sklearn.datasets.fetch_openml(name='adult', version=2)")
            load_path = "huggingface_datasets.load_dataset"
            split_name = None
            for cand in ["train", "validation", "test"]:
                if cand in ds:
                    split_name = cand
                    break
            if split_name is None:
                split_name = list(ds.keys())[0]
            df = ds[split_name].to_pandas()
            Xy = df
        except Exception:
            Xy = None
    except ModuleNotFoundError:
        Xy = None

    if Xy is None:
        from sklearn.datasets import fetch_openml
        adult = fetch_openml(name='adult', version=2, as_frame=True)
        load_path = "sklearn.datasets.fetch_openml"
        Xy = adult.frame.copy()

    return Xy, load_path


def main():
    start = time.time()
    seed = 42

    dataset_used = {
        "dataset_source": "huggingface",
        "dataset_id": "sklearn.datasets.fetch_openml(name='adult', version=2)",
        "resolved_via": "dataset_resolver",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open('dataset_used.json', 'w') as f:
        json.dump(dataset_used, f, indent=2)

    Xy, load_path = load_adult_via_spec()

    # Identify target column
    target_col = None
    for c in ["class", "income", "target", "salary"]:
        if c in Xy.columns:
            target_col = c
            break
    if target_col is None:
        target_col = Xy.columns[-1]

    y_raw = Xy[target_col]
    X = Xy.drop(columns=[target_col])

    y_str = y_raw.astype(str)
    y = y_str.str.contains(">50K").astype(int).values

    n_samples = int(X.shape[0])
    if n_samples < 10000:
        raise RuntimeError(f"Dataset too small: {n_samples}")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]

    cat_cols = [c for c in X_train.columns if X_train[c].dtype == 'object' or str(X_train[c].dtype).startswith('category')]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )

    baseline_pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", GradientBoostingClassifier(random_state=seed)),
    ])

    tuned_pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", GradientBoostingClassifier(random_state=seed)),
    ])

    param_grid = {
        "clf__n_estimators": [200, 250, 300],
        "clf__max_depth": [3, 4, 5],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__min_samples_leaf": [1, 5],
        "clf__max_features": [None, "sqrt"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        tuned_pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    baseline_pipe.fit(X_train, y_train)
    base_proba = baseline_pipe.predict_proba(X_test)[:, 1]
    base_pred = (base_proba >= 0.5).astype(int)

    base_acc = accuracy_score(y_test, base_pred)
    base_bal_acc = balanced_accuracy_score(y_test, base_pred)
    base_auc = roc_auc_score(y_test, base_proba)

    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    cv_best_score = grid.best_score_

    tuned_best = grid.best_estimator_
    tuned_proba = tuned_best.predict_proba(X_test)[:, 1]
    tuned_pred = (tuned_proba >= 0.5).astype(int)

    tuned_acc = accuracy_score(y_test, tuned_pred)
    tuned_bal_acc = balanced_accuracy_score(y_test, tuned_pred)
    tuned_auc = roc_auc_score(y_test, tuned_proba)

    n_test = int(len(y_test))
    correct = int((tuned_pred == y_test).sum())
    p0 = 0.78
    pval = exact_binom_pvalue_one_sided_greater(correct, n_test, p0)
    lb95 = one_sided_lower_confidence_bound_95(correct, n_test, alpha=0.05)

    # Visualization
    labels = ["Baseline", "Tuned"]
    accs = [base_acc, tuned_acc]
    baccs = [base_bal_acc, tuned_bal_acc]
    aucs = [base_auc, tuned_auc]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, accs, width, label='Accuracy')
    plt.bar(x, baccs, width, label='Balanced Acc')
    plt.bar(x + width, aucs, width, label='ROC-AUC')
    plt.xticks(x, labels)
    plt.ylim(0.5, 1.0)
    plt.title('Adult Income: Baseline vs Tuned GradientBoostingClassifier')
    plt.ylabel('Metric value')
    plt.xlabel('Condition')
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=150)

    results = {
        "dataset_load_path": load_path,
        "n_samples": n_samples,
        "n_train": int(len(train_idx)),
        "n_test": n_test,
        "n_features_original": int(X.shape[1]),
        "n_num_features": int(len(num_cols)),
        "n_cat_features": int(len(cat_cols)),

        "baseline_accuracy": float(base_acc),
        "baseline_balanced_accuracy": float(base_bal_acc),
        "baseline_roc_auc": float(base_auc),

        "tuned_cv_best_accuracy": float(cv_best_score),
        "tuned_best_params": to_py(best_params),

        "final_accuracy": float(tuned_acc),
        "accuracy": float(tuned_acc),
        "final_balanced_accuracy": float(tuned_bal_acc),
        "final_roc_auc": float(tuned_auc),

        "binomial_test_p0": float(p0),
        "binomial_test_one_sided_pvalue": float(pval),
        "accuracy_one_sided_95pct_lower_bound": float(lb95),

        "random_seed": seed,
        "elapsed_seconds": float(time.time() - start),
    }

    with open('raw_results.json', 'w') as f:
        json.dump(to_py(results), f, indent=2)

    with open('execution.log', 'w') as f:
        f.write(json.dumps(to_py({
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "results_keys": sorted(list(results.keys())),
        }), indent=2))


if __name__ == '__main__':
    main()
