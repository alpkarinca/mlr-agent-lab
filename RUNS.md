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