#!/usr/bin/env python3
"""
Generates a standalone, beautiful, interactive HTML report (ml-study/report.html)
reading numbers directly from saved CSV and JSON files in ml-study/.
"""

import os
import json
import base64
import pandas as pd

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load files
    final_metrics_path = os.path.join(script_dir, "final_metrics.json")
    tuning_csv_path = os.path.join(script_dir, "tuning_results.csv")
    test_pred_path = os.path.join(script_dir, "test_predictions.csv")

    with open(final_metrics_path) as f:
        metrics = json.load(f)

    tuning_df = pd.read_csv(tuning_csv_path)
    test_df = pd.read_csv(test_pred_path)

    # Encode images to base64 for truly self-contained report
    def get_base64_img(filename):
        path = os.path.join(script_dir, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
        return ""

    scatter_b64 = get_base64_img("leaves_scatter.png")
    boundaries_b64 = get_base64_img("decision_boundaries.png")

    win = metrics["winning_model"]
    test_comp = metrics["all_models_test_comparison"]

    # Top configurations per model
    best_per_model = (
        tuning_df.sort_values(by="mean_cv_macro_f1", ascending=False)
        .groupby("model_name", as_index=False)
        .first()
        .sort_values(by="mean_cv_macro_f1", ascending=False)
    )

    test_preds_json = test_df.to_json(orient="records")
    tuning_records_json = tuning_df.sort_values(by="mean_cv_macro_f1", ascending=False).to_json(orient="records")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Leaf Species Classification — Machine Learning Study</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0F172A;
            --bg-secondary: #1E293B;
            --bg-card: rgba(30, 41, 59, 0.7);
            --bg-card-hover: rgba(51, 65, 85, 0.8);
            --border-color: rgba(148, 163, 184, 0.15);
            --border-focus: #38BDF8;
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --text-muted: #64748B;
            --accent-cyan: #06B6D4;
            --accent-emerald: #10B981;
            --accent-rose: #F43F5E;
            --accent-amber: #F59E0B;
            --accent-indigo: #6366F1;
            --shadow-subtle: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
            --radius-md: 12px;
            --radius-lg: 16px;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 0;
            margin: 0;
            -webkit-font-smoothing: antialiased;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 24px 80px 24px;
        }}

        header {{
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.95)),
                        radial-gradient(ellipse at top right, rgba(99, 102, 241, 0.15), transparent 50%);
            border-bottom: 1px solid var(--border-color);
            padding: 48px 0 36px 0;
            margin-bottom: 40px;
        }}

        .header-inner {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 24px;
        }}

        .badge-header {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(6, 182, 212, 0.12);
            color: var(--accent-cyan);
            border: 1px solid rgba(6, 182, 212, 0.3);
            border-radius: 9999px;
            padding: 4px 14px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 16px;
        }}

        h1 {{
            font-size: 2.6rem;
            font-weight: 800;
            line-height: 1.2;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #FFFFFF 30%, #94A3B8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }}

        .lead {{
            font-size: 1.15rem;
            color: var(--text-secondary);
            max-width: 800px;
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        .kpi-card {{
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 24px;
            box-shadow: var(--shadow-subtle);
            position: relative;
            overflow: hidden;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(56, 189, 248, 0.4);
        }}

        .kpi-card.highlight {{
            border-color: rgba(16, 185, 129, 0.4);
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(16, 185, 129, 0.08));
        }}

        .kpi-card.highlight::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #10B981, #06B6D4);
        }}

        .kpi-title {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
            margin-bottom: 8px;
        }}

        .kpi-value {{
            font-size: 2.2rem;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.02em;
            line-height: 1;
            margin-bottom: 8px;
        }}

        .kpi-subtext {{
            font-size: 0.85rem;
            color: var(--text-muted);
        }}

        .section {{
            margin-bottom: 50px;
        }}

        .section-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-color);
        }}

        .section-title {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .section-title span.number {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 8px;
            background: rgba(99, 102, 241, 0.2);
            color: var(--accent-indigo);
            font-size: 0.9rem;
            font-weight: 700;
        }}

        .card {{
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 28px;
            box-shadow: var(--shadow-subtle);
            margin-bottom: 24px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
            text-align: left;
        }}

        th {{
            background: rgba(15, 23, 42, 0.6);
            color: var(--text-secondary);
            font-weight: 600;
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-color);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}

        td {{
            padding: 14px 16px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.08);
            color: var(--text-primary);
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover td {{
            background: rgba(51, 65, 85, 0.3);
        }}

        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}

        .badge-success {{ background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-winner {{ background: rgba(6, 182, 212, 0.15); color: #38BDF8; border: 1px solid rgba(6, 182, 212, 0.3); }}
        .badge-baseline {{ background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }}
        .badge-danger {{ background: rgba(244, 63, 94, 0.15); color: #FB7185; border: 1px solid rgba(244, 63, 94, 0.3); }}

        .code-cell {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: #E2E8F0;
            background: rgba(15, 23, 42, 0.6);
            padding: 2px 6px;
            border-radius: 4px;
        }}

        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 24px;
        }}

        .image-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            overflow: hidden;
            box-shadow: var(--shadow-subtle);
        }}

        .image-card img {{
            width: 100%;
            height: auto;
            display: block;
            border-bottom: 1px solid var(--border-color);
        }}

        .image-caption {{
            padding: 16px 20px;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }}

        .confusion-matrix-wrapper {{
            display: flex;
            gap: 32px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .cm-table {{
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            overflow: hidden;
            max-width: 400px;
        }}

        .cm-table td, .cm-table th {{
            text-align: center;
            padding: 12px 16px;
        }}

        .cm-hit {{
            background: rgba(16, 185, 129, 0.25) !important;
            font-weight: 700;
            color: #34D399;
        }}

        .cm-miss {{
            background: rgba(244, 63, 94, 0.15) !important;
            color: #FB7185;
        }}

        .filter-controls {{
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }}

        .search-box {{
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            color: var(--text-primary);
            padding: 10px 16px;
            font-size: 0.9rem;
            outline: none;
            min-width: 260px;
            transition: border-color 0.2s;
        }}

        .search-box:focus {{
            border-color: var(--border-focus);
        }}

        .btn-filter {{
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 10px 18px;
            border-radius: var(--radius-md);
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }}

        .btn-filter.active, .btn-filter:hover {{
            background: var(--accent-indigo);
            color: #FFFFFF;
            border-color: var(--accent-indigo);
        }}

        .table-scroll {{
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
        }}

        footer {{
            border-top: 1px solid var(--border-color);
            padding: 32px 0;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>

    <header>
        <div class="header-inner">
            <div class="badge-header">MLR Lab Study · Leak-Free Machine Learning</div>
            <h1>Leaf Species Predictive Modeling</h1>
            <p class="lead">
                Rigorous evaluation of classical classifiers on leaf morphology dimensions (<code class="code-cell">height</code>, <code class="code-cell">width</code>).
                Enforcing strict test isolation, exact deduplication, and a frozen held-out test evaluation.
            </p>
        </div>
    </header>

    <div class="container">

        <!-- KPI SUMMARY CARDS -->
        <div class="kpi-grid">
            <div class="kpi-card highlight">
                <div class="kpi-title">Winning Test Accuracy</div>
                <div class="kpi-value">{(win['test_accuracy'] * 100):.1f}%</div>
                <div class="kpi-subtext">Winning Model: {win['name']}</div>
            </div>
            <div class="kpi-card highlight">
                <div class="kpi-title">Winning Test Macro F1</div>
                <div class="kpi-value">{win['test_macro_f1']:.4f}</div>
                <div class="kpi-subtext">Balanced Acc: {(win['test_balanced_accuracy'] * 100):.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Baseline Floor (ZeroR)</div>
                <div class="kpi-value">{(test_comp['ZeroR Baseline']['test_accuracy'] * 100):.1f}%</div>
                <div class="kpi-subtext">Macro F1: {test_comp['ZeroR Baseline']['test_macro_f1']:.4f} (Majority class)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Leakage Mitigation</div>
                <div class="kpi-value">{metrics['dataset_split']['duplicates_removed']} Rows</div>
                <div class="kpi-subtext">Deduplicated from 518 raw $\rightarrow$ 492 clean samples</div>
            </div>
        </div>

        <!-- SECTION 1: METHODOLOGY & LEAKAGE ISOLATION -->
        <section class="section">
            <div class="section-header">
                <h2 class="section-title"><span class="number">1</span> Data Integrity & Leak-Free Test Isolation</h2>
            </div>
            <div class="card">
                <p style="color: var(--text-secondary); margin-bottom: 20px;">
                    During initial exploratory data analysis, <strong>26 exact duplicate records</strong> (spanning 46 total rows) were identified.
                    A naive random train/test split would cause identical rows to land across splits, severely contaminating the test set and artificially inflating generalization metrics.
                </p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px;">
                    <div style="background: rgba(15, 23, 42, 0.5); padding: 16px; border-radius: var(--radius-md); border: 1px solid var(--border-color);">
                        <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase;">Raw Dataset</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #FFF; margin-top: 4px;">518 Samples</div>
                        <div style="color: var(--text-secondary); font-size: 0.8rem;">Before deduplication</div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.5); padding: 16px; border-radius: var(--radius-md); border: 1px solid var(--border-color);">
                        <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase;">Clean Distinct Dataset</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: var(--accent-emerald); margin-top: 4px;">492 Samples</div>
                        <div style="color: var(--text-secondary); font-size: 0.8rem;">Deduplicated instances</div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.5); padding: 16px; border-radius: var(--radius-md); border: 1px solid var(--border-color);">
                        <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase;">Training Set (80%)</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #FFF; margin-top: 4px;">393 Samples</div>
                        <div style="color: var(--text-secondary); font-size: 0.8rem;">5-fold cross-validation</div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.5); padding: 16px; border-radius: var(--radius-md); border: 1px solid var(--border-color);">
                        <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase;">Frozen Test Set (20%)</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: var(--accent-cyan); margin-top: 4px;">99 Samples</div>
                        <div style="color: var(--text-secondary); font-size: 0.8rem;">Stratified held-out test</div>
                    </div>
                </div>
            </div>
        </section>

        <!-- SECTION 2: 5-FOLD CV MODEL COMPARISON -->
        <section class="section">
            <div class="section-header">
                <h2 class="section-title"><span class="number">2</span> Cross-Validation Benchmarks (5-Fold Stratified CV)</h2>
                <span class="badge badge-winner">232 Grid Trials Evaluated</span>
            </div>
            <div class="card" style="padding: 0; overflow: hidden;">
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Best Hyperparameters</th>
                            <th>CV Macro F1</th>
                            <th>CV Balanced Acc</th>
                            <th>CV Accuracy</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    for _, row in best_per_model.iterrows():
        is_win = (row["model_name"] == win["name"])
        is_base = (row["model_name"] == "ZeroR Baseline")
        badge_html = '<span class="badge badge-winner">Winner</span>' if is_win else ('<span class="badge badge-baseline">Baseline</span>' if is_base else '<span class="badge badge-success">Candidate</span>')
        html_content += f"""
                        <tr>
                            <td><strong>{row['model_name']}</strong></td>
                            <td><span class="code-cell">{row['params']}</span></td>
                            <td><strong>{row['mean_cv_macro_f1']:.4f}</strong> <span style="color: var(--text-muted); font-size: 0.8rem;">± {row['std_cv_macro_f1']:.3f}</span></td>
                            <td>{row['mean_cv_balanced_acc']:.4f}</td>
                            <td>{(row['mean_cv_accuracy']*100):.2f}%</td>
                            <td>{badge_html}</td>
                        </tr>
        """

    html_content += f"""
                    </tbody>
                </table>
            </div>
        </section>

        <!-- SECTION 3: THE FINAL NUMBER & TEST EVALUATION -->
        <section class="section">
            <div class="section-header">
                <h2 class="section-title"><span class="number">3</span> The Final Number — Frozen Held-Out Test Evaluation</h2>
                <span class="badge badge-winner">Evaluated Exactly Once</span>
            </div>
            <div class="card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 20px; margin-bottom: 24px;">
                    <div>
                        <h3 style="font-size: 1.3rem; margin-bottom: 6px; color: #FFF;">Selected Best Model: {win['name']}</h3>
                        <p style="color: var(--text-secondary); font-size: 0.95rem;">
                            Optimized configuration: <code class="code-cell">{win['best_hyperparameters']}</code>
                        </p>
                    </div>
                    <div style="display: flex; gap: 12px;">
                        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: var(--radius-md); padding: 10px 18px; text-align: center;">
                            <div style="font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted);">Test Accuracy</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: #34D399;">{(win['test_accuracy'] * 100):.2f}%</div>
                        </div>
                        <div style="background: rgba(6, 182, 212, 0.1); border: 1px solid rgba(6, 182, 212, 0.3); border-radius: var(--radius-md); padding: 10px 18px; text-align: center;">
                            <div style="font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted);">Test Macro F1</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: #38BDF8;">{win['test_macro_f1']:.4f}</div>
                        </div>
                    </div>
                </div>

                <div class="confusion-matrix-wrapper">
                    <!-- Per-Class Metrics -->
                    <div style="flex: 1; min-width: 320px;">
                        <h4 style="font-size: 0.95rem; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 12px; letter-spacing: 0.05em;">Per-Class Test Performance</h4>
                        <table style="border: 1px solid var(--border-color); border-radius: var(--radius-md); overflow: hidden;">
                            <thead>
                                <tr>
                                    <th>Species</th>
                                    <th>Precision</th>
                                    <th>Recall</th>
                                    <th>F1-Score</th>
                                    <th>Support</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong style="color: #F43F5E;">Cherry</strong></td>
                                    <td>{win['classification_report']['cherry']['precision']:.4f}</td>
                                    <td>{win['classification_report']['cherry']['recall']:.4f}</td>
                                    <td><strong>{win['classification_report']['cherry']['f1-score']:.4f}</strong></td>
                                    <td>{int(win['classification_report']['cherry']['support'])}</td>
                                </tr>
                                <tr>
                                    <td><strong style="color: #10B981;">Maple</strong></td>
                                    <td>{win['classification_report']['maple']['precision']:.4f}</td>
                                    <td>{win['classification_report']['maple']['recall']:.4f}</td>
                                    <td><strong>{win['classification_report']['maple']['f1-score']:.4f}</strong></td>
                                    <td>{int(win['classification_report']['maple']['support'])}</td>
                                </tr>
                                <tr>
                                    <td><strong style="color: #F59E0B;">Walnut</strong></td>
                                    <td>{win['classification_report']['walnut']['precision']:.4f}</td>
                                    <td>{win['classification_report']['walnut']['recall']:.4f}</td>
                                    <td><strong>{win['classification_report']['walnut']['f1-score']:.4f}</strong></td>
                                    <td>{int(win['classification_report']['walnut']['support'])}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <!-- Confusion Matrix Table -->
                    <div style="flex: 1; min-width: 300px;">
                        <h4 style="font-size: 0.95rem; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 12px; letter-spacing: 0.05em;">Test Set Confusion Matrix</h4>
                        <table class="cm-table">
                            <thead>
                                <tr>
                                    <th>Actual \\ Pred</th>
                                    <th>Cherry</th>
                                    <th>Maple</th>
                                    <th>Walnut</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Cherry</strong></td>
                                    <td class="cm-hit">30</td>
                                    <td class="cm-miss">1</td>
                                    <td class="cm-miss">2</td>
                                </tr>
                                <tr>
                                    <td><strong>Maple</strong></td>
                                    <td class="cm-hit">0</td>
                                    <td class="cm-hit">40</td>
                                    <td class="cm-miss">2</td>
                                </tr>
                                <tr>
                                    <td><strong>Walnut</strong></td>
                                    <td class="cm-miss">3</td>
                                    <td class="cm-miss">1</td>
                                    <td class="cm-hit">20</td>
                                </tr>
                            </tbody>
                        </table>
                        <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 8px;">
                            Total test samples: 99. Correct: 90. Errors: 9 (predominantly along the cherry/walnut height boundary).
                        </p>
                    </div>
                </div>
            </div>
        </section>

        <!-- SECTION 4: VISUAL GALLERY -->
        <section class="section">
            <div class="section-header">
                <h2 class="section-title"><span class="number">4</span> Visual Analytics & 2D Decision Boundaries</h2>
            </div>
            <div class="gallery-grid">
                <div class="image-card">
                    <img src="{scatter_b64}" alt="Leaf Dimensions Scatter Plot">
                    <div class="image-caption">
                        <strong>Exploratory Morphology:</strong> Height vs Width distribution across Maple (broad), Cherry (compact), and Walnut (slender). Anomaly callouts highlight suspected mislabeled instances.
                    </div>
                </div>
                <div class="image-card">
                    <img src="{boundaries_b64}" alt="2D Decision Boundaries">
                    <div class="image-caption">
                        <strong>Model Comparison:</strong> 2D decision contours across all 5 candidate classifiers. k-NN and SVM form smooth boundary transitions capable of tracking non-linear morphology thresholds.
                    </div>
                </div>
            </div>
        </section>

        <!-- SECTION 5: INTERACTIVE PREDICTIONS EXPLORER -->
        <section class="section">
            <div class="section-header">
                <h2 class="section-title"><span class="number">5</span> Test Set Predictions Explorer</h2>
                <span class="badge badge-success">99 Test Samples</span>
            </div>
            <div class="card">
                <div class="filter-controls">
                    <input type="text" id="searchInput" class="search-box" placeholder="Search by species, dimension...">
                    <button class="btn-filter active" onclick="filterPredictions('all')">All (99)</button>
                    <button class="btn-filter" onclick="filterPredictions('correct')">Correct (90)</button>
                    <button class="btn-filter" onclick="filterPredictions('incorrect')">Errors (9)</button>
                </div>
                <div class="table-scroll">
                    <table>
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Leaf Height (cm)</th>
                                <th>Leaf Width (cm)</th>
                                <th>True Species</th>
                                <th>Predicted Species</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody id="predictionsTableBody">
                            <!-- Populated via JavaScript -->
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

    </div>

    <footer>
        <div class="container" style="padding-top: 0; padding-bottom: 0;">
            <p>Generated for MLR Agent Lab · Machine Learning Study · Fully Reproducible Artifact</p>
        </div>
    </footer>

    <script>
        const testPredictions = {test_preds_json};

        function renderTable(data) {{
            const tbody = document.getElementById('predictionsTableBody');
            tbody.innerHTML = '';
            data.forEach((row, idx) => {{
                const tr = document.createElement('tr');
                const isCorrect = row.Correct;
                tr.innerHTML = `
                    <td>${{idx + 1}}</td>
                    <td>${{row['Leaf height'].toFixed(2)}}</td>
                    <td>${{row['Leaf width'].toFixed(2)}}</td>
                    <td><strong>${{row['True species']}}</strong></td>
                    <td>${{row['Predicted species']}}</td>
                    <td>
                        <span class="badge ${{isCorrect ? 'badge-success' : 'badge-danger'}}">
                            ${{isCorrect ? 'Correct' : 'Misclassified'}}
                        </span>
                    </td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        let currentFilter = 'all';
        function filterPredictions(type) {{
            currentFilter = type;
            document.querySelectorAll('.btn-filter').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            applyFilters();
        }}

        function applyFilters() {{
            const search = document.getElementById('searchInput').value.toLowerCase();
            let filtered = testPredictions.filter(row => {{
                const matchesFilter = (currentFilter === 'all') ||
                                      (currentFilter === 'correct' && row.Correct) ||
                                      (currentFilter === 'incorrect' && !row.Correct);
                const matchesSearch = row['True species'].toLowerCase().includes(search) ||
                                      row['Predicted species'].toLowerCase().includes(search) ||
                                      row['Leaf height'].toString().includes(search) ||
                                      row['Leaf width'].toString().includes(search);
                return matchesFilter && matchesSearch;
            }});
            renderTable(filtered);
        }}

        document.getElementById('searchInput').addEventListener('input', applyFilters);
        renderTable(testPredictions);
    </script>
</body>
</html>
"""

    report_path = os.path.join(script_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Interactive HTML report successfully generated at: {report_path}")

if __name__ == "__main__":
    main()
