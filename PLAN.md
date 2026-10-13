# Project Plan

## Current Status
- [x] Environment configured (`git`, `gh`, `uv`, `claude`)
- [x] Git credentials and GitHub CLI authenticated
- [x] `AGENTS.md` and `CLAUDE.md` created with project rules
- [x] Initial `PLAN.md` initialized
- [x] Initialize git repository and create public repo on GitHub (`gh repo create`)
- [x] Create `README.md` and `RUNS.md`
- [x] Select Task from Part 4: EDA & Classifier Benchmark on `data/leaves.csv`
- [x] Part 4 Step 1 (EDA) executed in `ml-study/`
  - [x] Python 3.11 virtual environment initialized with `uv`
  - [x] `ml-study/eda_leaves.py` script created and verified
  - [x] Publication-quality scatter plot generated at `ml-study/leaves_scatter.png`
  - [x] Full analysis report and train/test split audit documented at `ml-study/eda_report.md`
- [x] Part 4 Step 2: Predictive Modeling Study per `ml-study/SPEC.md`
  - [x] Hyperparameter tuning on 5-fold CV across 5 candidate classifiers (`ml-study/tuning_results.csv`)
  - [x] Selection of winning model based on CV Macro F1 (k-NN, $k=7$, Euclidean, distance weights)
  - [x] Final single evaluation on held-out test set (`ml-study/final_metrics.json`, `ml-study/test_predictions.csv`)
  - [x] Interactive standalone HTML report (`ml-study/report.html`)
  - [x] 5-slide Quarto Reveal.js presentation rendered (`ml-study/slides.qmd` $\rightarrow$ `ml-study/slides.html`)
- [ ] Submit public GitHub repository link to e-learning (Moodle)

## Machine Learning Study Plan (`ml-study`)

### Step 1: Leakage Prevention & Baseline Formulation (Completed)
- [x] Set up isolated Python environment (`uv`, `scikit-learn`, `pandas`, `matplotlib`).
- [x] Preprocess dataset: deduplicate exact duplicate rows to prevent train/test leakage (492 clean samples).
- [x] Isolate a 20% stratified held-out test set (`random_state=42`) and freeze it (393 train, 99 test).
- [x] Establish ZeroR / DummyClassifier baseline on 5-fold CV (CV Balanced Acc: 0.333, Test Macro F1: 0.199).
- [x] Perform a quick end-to-end trial run with a simple classifier (Logistic Regression: CV Balanced Acc 0.929, Test Macro F1 0.879).

### Step 2: Multi-Model Training & Hyperparameter Tuning (Completed)
- [x] Build leak-free `Pipeline` workflows (fitting scalers strictly on training folds).
- [x] Tune 5 classical lecture classifiers using 5-fold Stratified CV:
  1. Logistic Regression (L2 penalty, tuned $C$)
  2. Decision Tree (tuned `max_depth`, `min_samples_split`)
  3. k-Nearest Neighbors (tuned $k$, distance metric, weights)
  4. Support Vector Machine (tuned $C$, kernel, $\gamma$)
  5. Random Forest (tuned `n_estimators`, `max_depth`, `max_features`)
- [x] Optimize hyperparameter grids for Macro F1-score and Balanced Accuracy (232 trials evaluated).
- [x] Export cross-validation tuning results to `ml-study/tuning_results.csv`.

### Step 3: Final Evaluation & Decision Boundary Visualization (Completed)
- [x] Identify top-performing model strictly from CV validation scores (Winning model: k-NN, CV Macro F1: 0.9556).
- [x] Evaluate winning model exactly once on the frozen 20% test set (Test Accuracy: 90.91%, Test Macro F1: 0.8983).
- [x] Export final test predictions to `ml-study/test_predictions.csv` and final metrics to `ml-study/final_metrics.json`.
- [x] Generate confusion matrix and 2D decision boundary visualization plot (`ml-study/decision_boundaries.png`).

### Step 4: Standalone Deliverables & Presentation (Completed)
- [x] Install Quarto CLI in environment (`quarto 1.10.19`).
- [x] Generate standalone interactive HTML report (`ml-study/report.html`) reading numbers directly from CSV/JSON.
- [x] Create exactly 5-slide Quarto Reveal.js presentation (`ml-study/slides.qmd`).
- [x] Render slides to HTML (`quarto render ml-study/slides.qmd` $\rightarrow$ `ml-study/slides.html`).
- [x] Update `RUNS.md` and commit all files to Git.

## Next Step
- Submit public GitHub repository link (`https://github.com/alpkarinca/mlr-agent-lab`) to e-learning (Moodle).