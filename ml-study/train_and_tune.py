#!/usr/bin/env python3
"""
Step 4 & Step 5: Hyperparameter Tuning, Model Selection, and Final Test Evaluation.
Performs 5-fold cross-validation grid search on the training split across 5 classical classifiers,
saves ml-study/tuning_results.csv, selects the winning model, evaluates once on the held-out test set,
and exports ml-study/final_metrics.json, ml-study/test_predictions.csv, and decision boundary plots.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, ".."))
    csv_path = os.path.join(workspace_root, "data", "leaves.csv")

    print("=" * 70)
    print("STEP 4: HYPERPARAMETER TUNING VIA 5-FOLD STRATIFIED CV")
    print("=" * 70)

    # 1. Load data and prevent leakage
    df_raw = pd.read_csv(csv_path)
    df_clean = df_raw.drop_duplicates().reset_index(drop=True)
    print(f"Loaded raw: {len(df_raw)} rows -> Deduplicated: {len(df_clean)} rows.")

    X = df_clean[["Leaf height", "Leaf width"]]
    y = df_clean["Tree type"]

    # 2. Frozen 20% Stratified Held-out Test Set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"Isolated: Train = {len(X_train)} samples, Held-out Test = {len(X_test)} samples.")

    # 3. Define 5-Fold Stratified Cross-Validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 4. Define candidate models and parameter grids
    model_configs = {
        "Logistic Regression": {
            "pipe": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=42, max_iter=1000))
            ]),
            "grid": {
                "clf__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
                "clf__solver": ["lbfgs"],
                "clf__penalty": ["l2"]
            }
        },
        "k-Nearest Neighbors": {
            "pipe": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier())
            ]),
            "grid": {
                "clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "manhattan"]
            }
        },
        "Decision Tree": {
            "pipe": Pipeline([
                ("clf", DecisionTreeClassifier(random_state=42))
            ]),
            "grid": {
                "clf__max_depth": [2, 3, 4, 5, 6, 8, 10, None],
                "clf__min_samples_split": [2, 5, 10],
                "clf__criterion": ["gini", "entropy"]
            }
        },
        "Support Vector Machine": {
            "pipe": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(random_state=42))
            ]),
            "grid": {
                "clf__C": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
                "clf__kernel": ["rbf", "linear"],
                "clf__gamma": ["scale", "auto", 0.05, 0.1, 0.5]
            }
        },
        "Random Forest": {
            "pipe": Pipeline([
                ("clf", RandomForestClassifier(random_state=42))
            ]),
            "grid": {
                "clf__n_estimators": [50, 100, 200],
                "clf__max_depth": [3, 4, 5, 6, 8, None],
                "clf__min_samples_split": [2, 5],
                "clf__max_features": ["sqrt", "log2"]
            }
        }
    }

    all_tuning_records = []
    best_estimators = {}
    best_model_summary = []

    # Include ZeroR baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_cv = GridSearchCV(dummy, {}, cv=cv, scoring=["f1_macro", "balanced_accuracy", "accuracy"], refit="f1_macro")
    dummy_cv.fit(X_train, y_train)
    dummy_rec = {
        "model_name": "ZeroR Baseline",
        "params": "majority_class",
        "mean_cv_macro_f1": dummy_cv.cv_results_["mean_test_f1_macro"][0],
        "std_cv_macro_f1": dummy_cv.cv_results_["std_test_f1_macro"][0],
        "mean_cv_balanced_acc": dummy_cv.cv_results_["mean_test_balanced_accuracy"][0],
        "std_cv_balanced_acc": dummy_cv.cv_results_["std_test_balanced_accuracy"][0],
        "mean_cv_accuracy": dummy_cv.cv_results_["mean_test_accuracy"][0],
        "std_cv_accuracy": dummy_cv.cv_results_["std_test_accuracy"][0],
        "mean_fit_time": dummy_cv.cv_results_["mean_fit_time"][0]
    }
    all_tuning_records.append(dummy_rec)
    best_estimators["ZeroR Baseline"] = dummy
    best_model_summary.append({
        "Model": "ZeroR Baseline",
        "Best Parameters": "majority_class",
        "CV Macro F1": dummy_rec["mean_cv_macro_f1"],
        "CV Balanced Acc": dummy_rec["mean_cv_balanced_acc"],
        "CV Accuracy": dummy_rec["mean_cv_accuracy"]
    })

    # Run GridSearchCV for each candidate model
    for model_name, config in model_configs.items():
        print(f"\nTuning {model_name} across parameter grid...")
        gs = GridSearchCV(
            config["pipe"],
            config["grid"],
            cv=cv,
            scoring=["f1_macro", "balanced_accuracy", "accuracy"],
            refit="f1_macro",
            n_jobs=-1,
            return_train_score=False
        )
        gs.fit(X_train, y_train)
        best_estimators[model_name] = gs.best_estimator_

        best_idx = gs.best_index_
        best_model_summary.append({
            "Model": model_name,
            "Best Parameters": str(gs.best_params_),
            "CV Macro F1": gs.cv_results_["mean_test_f1_macro"][best_idx],
            "CV Balanced Acc": gs.cv_results_["mean_test_balanced_accuracy"][best_idx],
            "CV Accuracy": gs.cv_results_["mean_test_accuracy"][best_idx]
        })
        print(f"  -> Best {model_name}: CV Macro F1 = {gs.best_score_:.4f} with params: {gs.best_params_}")

        # Store all parameter trials
        for i in range(len(gs.cv_results_["params"])):
            all_tuning_records.append({
                "model_name": model_name,
                "params": json.dumps(gs.cv_results_["params"][i]),
                "mean_cv_macro_f1": gs.cv_results_["mean_test_f1_macro"][i],
                "std_cv_macro_f1": gs.cv_results_["std_test_f1_macro"][i],
                "mean_cv_balanced_acc": gs.cv_results_["mean_test_balanced_accuracy"][i],
                "std_cv_balanced_acc": gs.cv_results_["std_test_balanced_accuracy"][i],
                "mean_cv_accuracy": gs.cv_results_["mean_test_accuracy"][i],
                "std_cv_accuracy": gs.cv_results_["std_test_accuracy"][i],
                "mean_fit_time": gs.cv_results_["mean_fit_time"][i]
            })

    # Save all tuning trials
    tuning_df = pd.DataFrame(all_tuning_records)
    tuning_csv_path = os.path.join(script_dir, "tuning_results.csv")
    tuning_df.to_csv(tuning_csv_path, index=False)
    print(f"\nSaved {len(tuning_df)} tuning trial results to: {tuning_csv_path}")

    # Summary table of top configurations per model
    best_summary_df = pd.DataFrame(best_model_summary).sort_values(by="CV Macro F1", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION COMPARISON (5-Fold CV on Training Set)")
    print("=" * 70)
    for _, row in best_summary_df.iterrows():
        print(f"{row['Model']:<24} | CV Macro F1: {row['CV Macro F1']:.4f} | CV Balanced Acc: {row['CV Balanced Acc']:.4f} | CV Acc: {row['CV Accuracy']:.4f}")
        print(f"   Params: {row['Best Parameters']}")
    print("=" * 70)

    # 5. STEP 5: THE FINAL NUMBER (Select single best model)
    print("\n" + "=" * 70)
    print("STEP 5: THE FINAL NUMBER (HELD-OUT TEST SET EVALUATION)")
    print("=" * 70)

    # Find the top model excluding baseline
    competitive_models = [m for m in best_model_summary if m["Model"] != "ZeroR Baseline"]
    winning_summary = max(competitive_models, key=lambda x: x["CV Macro F1"])
    winning_name = winning_summary["Model"]
    winning_estimator = best_estimators[winning_name]

    print(f"Winning model selected based strictly on CV Macro F1: **{winning_name}**")
    print(f"Validation CV Macro F1: {winning_summary['CV Macro F1']:.4f}")

    # Evaluate all best models on test set for complete reporting
    test_evaluations = {}
    for name, est in best_estimators.items():
        y_pred = est.predict(X_test)
        test_evaluations[name] = {
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "test_weighted_f1": float(f1_score(y_test, y_pred, average="weighted"))
        }

    # Winning model test evaluation
    y_test_pred = winning_estimator.predict(X_test)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_test_pred, labels=labels)
    cr = classification_report(y_test, y_test_pred, labels=labels, output_dict=True)

    print(f"\nFinal Evaluation of Winning Model ({winning_name}) on Frozen Held-out Test Set:")
    print(f"  Test Accuracy:          {test_evaluations[winning_name]['test_accuracy']:.4f}")
    print(f"  Test Balanced Accuracy: {test_evaluations[winning_name]['test_balanced_accuracy']:.4f}")
    print(f"  Test Macro F1:          {test_evaluations[winning_name]['test_macro_f1']:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, labels=labels, digits=4))

    print("Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    print(cm_df)

    # Save test predictions CSV
    test_pred_df = X_test.copy()
    test_pred_df["True species"] = y_test
    test_pred_df["Predicted species"] = y_test_pred
    test_pred_df["Correct"] = (test_pred_df["True species"] == test_pred_df["Predicted species"])
    test_pred_path = os.path.join(script_dir, "test_predictions.csv")
    test_pred_df.to_csv(test_pred_path, index=False)
    print(f"\nSaved test set predictions to: {test_pred_path}")

    # Save final metrics JSON
    final_metrics = {
        "winning_model": {
            "name": winning_name,
            "best_hyperparameters": winning_summary["Best Parameters"],
            "cv_macro_f1": winning_summary["CV Macro F1"],
            "cv_balanced_accuracy": winning_summary["CV Balanced Acc"],
            "cv_accuracy": winning_summary["CV Accuracy"],
            "test_macro_f1": test_evaluations[winning_name]["test_macro_f1"],
            "test_balanced_accuracy": test_evaluations[winning_name]["test_balanced_accuracy"],
            "test_accuracy": test_evaluations[winning_name]["test_accuracy"],
            "classification_report": cr,
            "confusion_matrix": {
                "labels": labels,
                "matrix": cm.tolist()
            }
        },
        "all_models_test_comparison": test_evaluations,
        "dataset_split": {
            "total_clean": len(df_clean),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "duplicates_removed": len(df_raw) - len(df_clean)
        }
    }
    final_metrics_path = os.path.join(script_dir, "final_metrics.json")
    with open(final_metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved final evaluation metrics to: {final_metrics_path}")

    # 6. Plot 2D Decision Boundaries for all tuned models
    plot_decision_boundaries(best_estimators, X_clean=X, y_clean=y, X_test=X_test, y_test=y_test, output_dir=script_dir)

def plot_decision_boundaries(best_estimators, X_clean, y_clean, X_test, y_test, output_dir):
    """Generates a multi-panel visual comparison of 2D decision boundaries."""
    models_to_plot = [m for m in best_estimators.keys() if m != "ZeroR Baseline"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axes = axes.flatten()

    # Define color scheme
    color_map = {"cherry": "#C2185B", "maple": "#2E7D32", "walnut": "#E65100"}
    bg_colors = {"cherry": "#FCE4EC", "maple": "#E8F5E9", "walnut": "#FFF3E0"}
    labels = sorted(y_clean.unique())
    label_to_int = {l: i for i, l in enumerate(labels)}

    # Grid for contour
    x_min, x_max = X_clean["Leaf width"].min() - 1.0, X_clean["Leaf width"].max() + 1.0
    y_min, y_max = X_clean["Leaf height"].min() - 1.0, X_clean["Leaf height"].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid_points = pd.DataFrame({"Leaf height": yy.ravel(), "Leaf width": xx.ravel()})[["Leaf height", "Leaf width"]]

    for idx, name in enumerate(models_to_plot):
        ax = axes[idx]
        est = best_estimators[name]
        preds = est.predict(grid_points)
        preds_int = np.array([label_to_int[p] for p in preds]).reshape(xx.shape)

        # Plot background decision regions
        ax.contourf(xx, yy, preds_int, levels=[-0.5, 0.5, 1.5, 2.5],
                    colors=[bg_colors[labels[0]], bg_colors[labels[1]], bg_colors[labels[2]]],
                    alpha=0.6)
        ax.contour(xx, yy, preds_int, levels=[0.5, 1.5], colors="black", linewidths=0.75, linestyles="--")

        # Plot test set points
        for label in labels:
            subset = X_test[y_test == label]
            ax.scatter(subset["Leaf width"], subset["Leaf height"],
                       color=color_map[label], label=f"{label.capitalize()} (test)",
                       edgecolors="black", linewidths=0.5, s=35, alpha=0.9)

        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Leaf Width (cm)", fontsize=10)
        ax.set_ylabel("Leaf Height (cm)", fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle=":", alpha=0.4)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8, frameon=True)

    # Last subplot: Error analysis / comparison
    ax_last = axes[5]
    ax_last.axis("off")
    ax_last.text(0.05, 0.95, "Decision Boundary Analysis", fontsize=14, fontweight="bold", transform=ax_last.transAxes)
    notes = (
        "Key Boundary Characteristics:\n\n"
        "• Maple: Broad width (> 10 cm) easily\n"
        "  separated by linear and non-linear boundaries.\n\n"
        "• Walnut vs. Cherry: Separated primarily by\n"
        "  height threshold (~12-14 cm) & aspect ratio.\n\n"
        "• SVM (RBF) & Random Forest capture slight\n"
        "  curvature along the slender leaf boundary.\n\n"
        "• Near-boundary errors correspond to ambiguous\n"
        "  specimens and conflicting dataset coordinates."
    )
    ax_last.text(0.05, 0.85, notes, fontsize=10, transform=ax_last.transAxes, verticalalignment="top", linespacing=1.4)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "decision_boundaries.png")
    fig.savefig(plot_path)
    plt.close()
    print(f"Saved decision boundaries plot to: {plot_path}")

if __name__ == "__main__":
    main()
