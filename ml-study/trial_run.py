#!/usr/bin/env python3
"""
Trial Run & Baseline Establishment for Leaf Classification Study.
Demonstrates leak-free test isolation, establishes ZeroR (majority class) baseline,
and validates the pipeline with a trial Logistic Regression model.
"""

import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def run_trial():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, ".."))
    csv_path = os.path.join(workspace_root, "data", "leaves.csv")

    df_raw = pd.read_csv(csv_path)
    print(f"Loaded raw dataset: {len(df_raw)} rows.")

    # 1. Leakage prevention: Deduplicate exact duplicate rows
    df = df_raw.drop_duplicates().reset_index(drop=True)
    n_removed = len(df_raw) - len(df)
    print(f"Removed {n_removed} exact duplicates to prevent leakage across splits.")
    print(f"Clean distinct dataset: {len(df)} samples.")

    X = df[["Leaf height", "Leaf width"]]
    y = df["Tree type"]

    # 2. Frozen 20% Stratified Held-out Test Set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"Dataset split: Train = {len(X_train)} samples, Test = {len(X_test)} samples.")
    print(f"Train class distribution:\n{y_train.value_counts().to_dict()}")
    print(f"Test class distribution:\n{y_test.value_counts().to_dict()}")

    # 3. Setup 5-fold Stratified Cross-Validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "balanced_acc": "balanced_accuracy",
        "f1_macro": "f1_macro"
    }

    # 4. ZeroR / Majority Class Baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy_cv = cross_validate(dummy, X_train, y_train, cv=cv, scoring=scoring)
    dummy.fit(X_train, y_train)
    y_test_dummy = dummy.predict(X_test)

    dummy_scores = {
        "cv_accuracy_mean": float(np.mean(dummy_cv["test_accuracy"])),
        "cv_accuracy_std": float(np.std(dummy_cv["test_accuracy"])),
        "cv_balanced_acc_mean": float(np.mean(dummy_cv["test_balanced_acc"])),
        "cv_balanced_acc_std": float(np.std(dummy_cv["test_balanced_acc"])),
        "cv_f1_macro_mean": float(np.mean(dummy_cv["test_f1_macro"])),
        "cv_f1_macro_std": float(np.std(dummy_cv["test_f1_macro"])),
        "test_accuracy": float(accuracy_score(y_test, y_test_dummy)),
        "test_balanced_acc": float(balanced_accuracy_score(y_test, y_test_dummy)),
        "test_f1_macro": float(f1_score(y_test, y_test_dummy, average="macro"))
    }

    # 5. Trial Model: Logistic Regression Pipeline (with StandardScaler)
    logreg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42, max_iter=1000))
    ])
    logreg_cv = cross_validate(logreg_pipe, X_train, y_train, cv=cv, scoring=scoring)
    logreg_pipe.fit(X_train, y_train)
    y_test_logreg = logreg_pipe.predict(X_test)

    logreg_scores = {
        "cv_accuracy_mean": float(np.mean(logreg_cv["test_accuracy"])),
        "cv_accuracy_std": float(np.std(logreg_cv["test_accuracy"])),
        "cv_balanced_acc_mean": float(np.mean(logreg_cv["test_balanced_acc"])),
        "cv_balanced_acc_std": float(np.std(logreg_cv["test_balanced_acc"])),
        "cv_f1_macro_mean": float(np.mean(logreg_cv["test_f1_macro"])),
        "cv_f1_macro_std": float(np.std(logreg_cv["test_f1_macro"])),
        "test_accuracy": float(accuracy_score(y_test, y_test_logreg)),
        "test_balanced_acc": float(balanced_accuracy_score(y_test, y_test_logreg)),
        "test_f1_macro": float(f1_score(y_test, y_test_logreg, average="macro"))
    }

    # 6. Secondary Trial Model: Decision Tree (max_depth=3)
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_cv = cross_validate(tree, X_train, y_train, cv=cv, scoring=scoring)
    tree.fit(X_train, y_train)
    y_test_tree = tree.predict(X_test)

    tree_scores = {
        "cv_accuracy_mean": float(np.mean(tree_cv["test_accuracy"])),
        "cv_accuracy_std": float(np.std(tree_cv["test_accuracy"])),
        "cv_balanced_acc_mean": float(np.mean(tree_cv["test_balanced_acc"])),
        "cv_balanced_acc_std": float(np.std(tree_cv["test_balanced_acc"])),
        "cv_f1_macro_mean": float(np.mean(tree_cv["test_f1_macro"])),
        "cv_f1_macro_std": float(np.std(tree_cv["test_f1_macro"])),
        "test_accuracy": float(accuracy_score(y_test, y_test_tree)),
        "test_balanced_acc": float(balanced_accuracy_score(y_test, y_test_tree)),
        "test_f1_macro": float(f1_score(y_test, y_test_tree, average="macro"))
    }

    # Print results summary
    print("\n" + "=" * 70)
    print("TRIAL BENCHMARK SUMMARY (5-Fold CV & Held-Out Test Set)")
    print("=" * 70)
    print(f"{'Model':<22} | {'CV Balanced Acc':<18} | {'CV Macro F1':<16} | {'Test Macro F1':<14} | {'Test Acc':<10}")
    print("-" * 70)
    for name, s in [("ZeroR Baseline", dummy_scores),
                    ("Logistic Regression", logreg_scores),
                    ("Decision Tree (d=3)", tree_scores)]:
        cv_bacc = f"{s['cv_balanced_acc_mean']:.3f} +/- {s['cv_balanced_acc_std']:.3f}"
        cv_f1 = f"{s['cv_f1_macro_mean']:.3f} +/- {s['cv_f1_macro_std']:.3f}"
        test_f1 = f"{s['test_f1_macro']:.3f}"
        test_acc = f"{s['test_accuracy']:.3f}"
        print(f"{name:<22} | {cv_bacc:<18} | {cv_f1:<16} | {test_f1:<14} | {test_acc:<10}")
    print("=" * 70)

    print("\nClassification Report (Trial Logistic Regression on Test Set):")
    print(classification_report(y_test, y_test_logreg, digits=3))

    print("Confusion Matrix (Trial Logistic Regression on Test Set):")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_test_logreg, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    print(cm_df)

    # Export results JSON
    results = {
        "dataset": {
            "total_raw": len(df_raw),
            "total_clean": len(df),
            "duplicates_removed": n_removed,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        },
        "baseline_zero_r": dummy_scores,
        "trial_logistic_regression": logreg_scores,
        "trial_decision_tree": tree_scores
    }
    output_json = os.path.join(script_dir, "trial_results.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTrial results successfully saved to: {output_json}")

if __name__ == "__main__":
    run_trial()
