# Model Runs & Verifications

## Run 1: Environment & Repository Initialization
- **Date:** 2026-10-13
- **Model:** Qwen 3.8 Flash (`agentic`) via CERIT-SC / Antigravity
- **Prompt:** Project setup, core rules definition, and repository structuring.
- **Verification:** Checked created files directly in terminal (`ls -la`), validated configurations, and initialized git tracking.

## Run 2: Classic Snake Game & Shrinking Arena Twist
- **Date:** 2026-10-13
- **Model:** Claude Opus 4.6 (Thinking) / Antigravity
- **Prompt:** Self-contained classic Snake game with keyboard controls, score counter, growth, wall/self collisions in `snake/index.html`, followed by game logic verification and Shrinking Arena twist implementation.
- **Verification:** Browser automated testing with subagent, gameplay recording validation, collision audit.

## Run 3: Exploratory Data Analysis (EDA) on Leaves Dataset
- **Date:** 2026-10-13
- **Model:** Gemini 3.8 Flash / Antigravity
- **Prompt:** Make a folder `ml-study/` for this task. Show me `data/leaves.csv`: how many leaves of each type, a scatter plot of height against width, and anything unusual. Save the plot or report in `ml-study/`. Also answer: Are there identical rows? What would that mean for a train/test split?
- **Verification:** 
  - Isolated Python 3.11 environment in `ml-study/.venv` via `uv`.
  - Executed `ml-study/eda_leaves.py` generating high-resolution scatter plot `ml-study/leaves_scatter.png`.
  - Verified exact duplicate detection (26 duplicate rows, 46 total rows involved), 3 conflicting label pairs, and generated comprehensive markdown report `ml-study/eda_report.md`.

## Run 4: ML Study Baseline & Trial Pipeline
- **Date:** 2026-10-13
- **Model:** Gemini 3.8 Flash / Antigravity
- **Prompt:** Propose concrete step-by-step implementation plan for `ml-study/SPEC.md`, update `PLAN.md`, establish ZeroR majority class baseline, and run quick trial model to validate end-to-end evaluation pipeline.
- **Verification:** 
  - Installed `scikit-learn==1.9.1` in `ml-study/.venv`.
  - Implemented and executed `ml-study/trial_run.py` cleanly.
  - ZeroR baseline recorded: CV Balanced Acc 0.333, CV Macro F1 0.198, Test Macro F1 0.199, Test Acc 0.424.
  - Trial Logistic Regression recorded: CV Balanced Acc 0.929, CV Macro F1 0.928, Test Macro F1 0.879, Test Acc 0.889.
  - Results logged to `ml-study/trial_results.json` and updated in `PLAN.md`.

## Run 5: ML Study Hyperparameter Tuning, Final Evaluation, and Deliverables
- **Date:** 2026-10-13
- **Model:** Gemini 3.8 Flash / Antigravity
- **Prompt:** Execute Steps 4, 5, and 6 per `ml-study/SPEC.md`: 5-fold CV hyperparameter tuning across 5 classical classifiers, select winning model, single final evaluation on frozen test set, and generate standalone HTML report and 5-slide Quarto presentation.
- **Verification:**
  - Evaluated 232 hyperparameter configurations across Logistic Regression, Decision Tree, k-NN, SVM, and Random Forest; saved to `ml-study/tuning_results.csv`.
  - Winning model selected via validation Macro F1: **k-NN ($k=7$, Euclidean, distance weights)** with CV Macro F1 = 0.9556.
  - Final held-out test evaluation executed once on frozen 99-sample test set: **Test Accuracy = 90.91%**, **Test Macro F1 = 0.8983**, **Test Balanced Accuracy = 0.8983**.
  - Generated `ml-study/final_metrics.json`, `ml-study/test_predictions.csv`, and 2D decision boundary plot `ml-study/decision_boundaries.png`.
  - Installed Quarto CLI (`1.10.19`).
  - Created standalone interactive report `ml-study/report.html` and 5-slide Quarto presentation `ml-study/slides.qmd`.
  - Rendered slides to `ml-study/slides.html` via `quarto render ml-study/slides.qmd`.
  - Verified browser rendering with browser subagent.