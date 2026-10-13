# Goal
Find out how accurately tree species can be predicted from leaf height and width, providing an honest generalization performance number we can trust without data leakage.

# Context
- Dataset: `data/leaves.csv` (518 rows, features: height and width, target: tree species class).
- Findings from Step 1: Contains class distribution and repeated/identical rows that risk data leakage if split naively across train and test sets.

# Constraints
- Test set isolation: A 20% stratified held-out test set is frozen before any model training or hyperparameter search; identical rows are handled to prevent leakage.
- Models: Compare classical classifiers from lectures (Logistic Regression, Decision Tree, k-NN, SVM, Random Forest).
- Validation & Tuning: 5-fold cross-validation on the training set only. Hyperparameter grids tuned with grid/random search.
- Selection Metric: Balanced Accuracy / Macro F1-score evaluated strictly on validation folds. Final test set evaluated once.
- Environment: Python managed via `uv`.

# Done when
- Reproducible Python training script runs cleanly.
- `ml-study/results.csv` or report contains model comparison table with cross-validation and test set scores.
- A standalone HTML report (`ml-study/report.html`) and a 5-slide Quarto presentation (`ml-study/slides.qmd`) are generated.
- All files are committed to Git.
