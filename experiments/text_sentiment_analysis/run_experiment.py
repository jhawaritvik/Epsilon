import json
import os
import re
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss


SEED = 42
np.random.seed(SEED)


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, list):
        return [to_py(v) for v in o]
    if isinstance(o, tuple):
        return [to_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return to_py(o.tolist())
    return o


NEG_CUES = {"not", "no", "never", "n't"}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[\.!\?;,:]")


def negation_transform(text: str) -> str:
    text = text.lower()
    toks = TOKEN_RE.findall(text)
    out = []
    negate = False
    for tok in toks:
        if tok in {".", "!", "?", ";", ",", ":"}:
            negate = False
            out.append(tok)
            continue
        if tok in NEG_CUES or tok.endswith("n't"):
            negate = True
            out.append(tok)
            continue
        out.append(tok + "_NEG" if negate else tok)
    return " ".join(out)


@dataclass
class Condition:
    name: str
    ngram_range: Tuple[int, int]
    use_negation: bool


def build_pipeline(ngram_range=(1, 1), use_negation=False, model="linearsvc", C=1.0,
                   max_features=100000, max_df=0.9, min_df=5, sublinear_tf=True):
    preproc = negation_transform if use_negation else (lambda x: x.lower())
    vectorizer = TfidfVectorizer(
        preprocessor=preproc,
        ngram_range=ngram_range,
        max_features=max_features,
        max_df=max_df,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )
    if model == "linearsvc":
        clf = LinearSVC(C=C, random_state=SEED)
    elif model == "logreg":
        clf = LogisticRegression(
            C=C,
            penalty="l2",
            solver="liblinear",
            max_iter=2000,
            random_state=SEED,
        )
    else:
        raise ValueError("Unknown model")
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def cv_evaluate_condition(X, y, splits, condition: Condition, model: str, C_grid: List[float],
                          vectorizer_params: Dict[str, Any]):
    per_C_fold_acc = {}
    for C in C_grid:
        accs = []
        for (tr, va) in splits:
            pipe = build_pipeline(
                ngram_range=condition.ngram_range,
                use_negation=condition.use_negation,
                model=model,
                C=C,
                **vectorizer_params,
            )
            pipe.fit([X[i] for i in tr], y[tr])
            pred = pipe.predict([X[i] for i in va])
            accs.append(accuracy_score(y[va], pred))
        per_C_fold_acc[C] = accs
    mean_acc = {C: float(np.mean(per_C_fold_acc[C])) for C in C_grid}
    best_C = max(mean_acc, key=mean_acc.get)
    return {
        "best_C": float(best_C),
        "cv_mean_accuracy": float(mean_acc[best_C]),
        "cv_fold_accuracies": [float(a) for a in per_C_fold_acc[best_C]],
    }


def main():
    start_time = time.time()

    ds = load_dataset('stanfordnlp/imdb')
    train = ds['train']
    test = ds['test']

    X_train = train['text']
    y_train = np.array(train['label'], dtype=np.int64)
    X_test = test['text']
    y_test = np.array(test['label'], dtype=np.int64)

    with open('dataset_used.json', 'w') as f:
        json.dump(to_py({
            "dataset_source": "huggingface",
            "dataset_id": "stanfordnlp/imdb",
            "splits": {"train": len(X_train), "test": len(X_test)},
            "seed": SEED
        }), f, indent=2)

    # Reduce compute per timeout directive: 5 folds, 1 repeat (5 fits per C)
    splits = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros_like(y_train), y_train))

    conditions = [
        Condition("baseline_unigram", (1, 1), False),
        Condition("ngram_1_2", (1, 2), False),
        Condition("ngram_1_3", (1, 3), False),
        Condition("ngram_1_2_negation", (1, 2), True),
    ]

    vectorizer_params = {
        "max_features": 100000,
        "max_df": 0.9,
        "min_df": 5,
        "sublinear_tf": True,
    }

    # Primary: LinearSVC only (speed)
    C_grid = [0.5, 1.0, 2.0]
    cv_results = {}
    for cond in conditions:
        cv_results[cond.name] = cv_evaluate_condition(
            X_train, y_train, splits, cond, "linearsvc", C_grid, vectorizer_params
        )

    best_condition_name = max(cv_results.keys(), key=lambda k: cv_results[k]["cv_mean_accuracy"])
    best_condition = next(c for c in conditions if c.name == best_condition_name)
    best_C = cv_results[best_condition_name]["best_C"]

    final_model = build_pipeline(
        ngram_range=best_condition.ngram_range,
        use_negation=best_condition.use_negation,
        model="linearsvc",
        C=best_C,
        **vectorizer_params,
    )
    final_model.fit(X_train, y_train)
    test_pred = final_model.predict(X_test)
    final_accuracy = float(accuracy_score(y_test, test_pred))
    final_macro_f1 = float(f1_score(y_test, test_pred, average="macro"))

    # Visualization: CV mean accuracy bars
    labels = [c.name for c in conditions]
    means = [cv_results[n]["cv_mean_accuracy"] for n in labels]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, means, color='steelblue', label='LinearSVC (CV mean acc)')
    plt.xticks(rotation=20, ha='right')
    plt.ylabel('CV Mean Accuracy (train split)')
    plt.title('IMDB TF-IDF Feature Engineering: CV Mean Accuracy by Condition')
    plt.ylim(0.75, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=150)

    out = {
        "final_accuracy": final_accuracy,
        "final_macro_f1": final_macro_f1,
        "final_model": {
            "model_family": "LinearSVC",
            "condition": best_condition_name,
            "ngram_range": list(best_condition.ngram_range),
            "use_negation": bool(best_condition.use_negation),
            "C": float(best_C),
            "vectorizer_params": to_py(vectorizer_params),
        },
        "cv_results_linearsvc": to_py(cv_results),
        "timing_seconds": float(time.time() - start_time),
        "seed": SEED,
        "notes": {
            "cv": "5-fold StratifiedKFold on train split only; paired splits reused across conditions.",
            "test_usage": "Test split used only for final single evaluation for chosen config.",
            "stat_tests": "Not computed here per agent constraints."
        }
    }

    with open('raw_results.json', 'w') as f:
        json.dump(to_py(out), f, indent=2)


if __name__ == '__main__':
    main()
