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
- [ ] Part 4 Step 2: Predictive Modeling Study per `ml-study/SPEC.md`
- [ ] Submit public GitHub repository link to e-learning (Moodle)

## Machine Learning Study Plan (`ml-study`)

### Step 1: Leakage Prevention & Baseline Formulation (Completed)
- [x] Set up isolated Python environment (`uv`, `scikit-learn`, `pandas`, `matplotlib`).
- [x] Preprocess dataset: deduplicate exact duplicate rows to prevent train/test leakage (492 clean samples).
- [x] Isolate a 20% stratified held-out test set (`random_state=42`) and freeze it (393 train, 99 test).
- [x] Establish ZeroR / DummyClassifier baseline on 5-fold CV (CV Balanced Acc: 0.333, Test Macro F1: 0.199).
- [x] Perform a quick end-to-end trial run with a simple classifier (Logistic Regression: CV Balanced Acc 0.929, Test Macro F1 0.879).

### Step 2: Multi-Model Training & Hyperparameter Tuning
- [ ] Build leak-free `Pipeline` workflows (fitting scalers strictly on training folds).
- [ ] Tune 5 classical lecture classifiers using 5-fold Stratified CV:
  1. Logistic Regression (L2 penalty, tuned $C$)
  2. Decision Tree (tuned `max_depth`, `min_samples_split`)
  3. k-Nearest Neighbors (tuned $k$, distance metric, weights)
  4. Support Vector Machine (tuned $C$, kernel, $\gamma$)
  5. Random Forest (tuned `n_estimators`, `max_depth`, `max_features`)
- [ ] Optimize hyperparameter grids for Macro F1-score and Balanced Accuracy.
- [ ] Export cross-validation and test benchmark results to `ml-study/results.csv`.

### Step 3: Final Evaluation & Decision Boundary Visualization
- [ ] Identify top-performing model strictly from CV validation scores.
- [ ] Evaluate all tuned models once on the frozen 20% test set.
- [ ] Generate confusion matrices and 2D decision boundary visualization plot (`ml-study/decision_boundaries.png`).

### Step 4: Standalone Deliverables & Presentation
- [ ] Generate standalone HTML report (`ml-study/report.html`) summarizing methodology, leakage prevention, benchmark table, and findings.
- [ ] Create 5-slide Quarto presentation (`ml-study/slides.qmd`) in Reveal.js format.
- [ ] Update `RUNS.md` and commit all files to Git.

## Next Step
- Run Step 2: Grid search hyperparameter tuning for all 5 classical classifiers and export `ml-study/results.csv`.