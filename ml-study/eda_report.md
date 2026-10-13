# Exploratory Data Analysis Report: `data/leaves.csv`

## 1. Overview & Dataset Breakdown

The dataset contains **518 leaf observations** with 3 attributes: `Leaf height` (cm), `Leaf width` (cm), and `Tree type`. There are **0 missing (null) values**.

### Class Distribution (Leaf Counts)

| Tree Type | Count | Percentage | General Leaf Morphology |
| :--- | :---: | :---: | :--- |
| **Maple** | **216** | 41.70% | Broad, palmate leaves (large width relative to height) |
| **Cherry** | **173** | 33.40% | Small, ovate/elliptical leaves (compact, moderate aspect ratio) |
| **Walnut** | **129** | 24.90% | Long, lanceolate leaflets (tall, narrow/slender) |
| **Total** | **518** | 100.0% | |

---

## 2. Feature Summary Statistics

| Tree Type | Metric | Leaf Height (cm) | Leaf Width (cm) | Aspect Ratio ($H / W$) |
| :--- | :--- | :---: | :---: | :---: |
| **Cherry** | Mean $\pm$ Std | $9.33 \pm 2.18$ | $5.36 \pm 1.92$ | $1.80 \pm 0.32$ |
| | Min – Max | $3.84$ – $16.90$ | $2.50$ – $21.60$ | $0.78$ – $2.82$ |
| | Median (IQR) | $9.00$ ($8.00$ – $10.50$) | $5.20$ ($4.30$ – $6.01$) | $1.80$ ($1.62$ – $1.99$) |
| **Maple** | Mean $\pm$ Std | $13.48 \pm 3.41$ | $15.52 \pm 4.05$ | $0.91 \pm 0.20$ |
| | Min – Max | $4.11$ – $25.50$ | $3.45$ – $29.70$ | $0.46$ – $2.19$ |
| | Median (IQR) | $13.12$ ($11.00$ – $15.80$) | $14.70$ ($13.00$ – $17.50$) | $0.86$ ($0.79$ – $0.95$) |
| **Walnut** | Mean $\pm$ Std | $15.32 \pm 2.40$ | $7.47 \pm 2.13$ | $2.11 \pm 0.35$ |
| | Min – Max | $10.30$ – $21.50$ | $4.15$ – $15.60$ | $1.09$ – $3.07$ |
| | Median (IQR) | $15.00$ ($13.50$ – $17.00$) | $7.10$ ($6.00$ – $8.50$) | $2.15$ ($1.91$ – $2.32$) |

---

## 3. Scatter Plot Analysis (Height vs. Width)

The scatter plot is saved at [`ml-study/leaves_scatter.png`](leaves_scatter.png).

### Cluster Dynamics:
1. **Maple (Broad/Wide Cluster)**: Centered around Width $\approx 15.5\text{ cm}$ and Height $\approx 13.5\text{ cm}$. Aspect ratios are predominantly below $1.0$ (wider than tall), forming a wide cluster toward the upper-right.
2. **Cherry (Small/Compact Cluster)**: Centered around Width $\approx 5.4\text{ cm}$ and Height $\approx 9.3\text{ cm}$. Represents the smallest leaves in both dimensions.
3. **Walnut (Tall/Slender Cluster)**: Centered around Width $\approx 7.5\text{ cm}$ and Height $\approx 15.3\text{ cm}$. Exhibits the highest aspect ratio ($\approx 2.1$), distinctly taller than cherry while significantly narrower than maple.

---

## 4. Duplicate Rows Analysis & Train/Test Split Impact

### Are there identical rows?
**Yes.**
- **Exact duplicate rows (repeated instances): 26 rows**
- **Total rows involved in duplicates: 46 rows** (representing 20 unique measurement sets that appear multiple times).
- For example:
  - `Height = 7.5, Width = 3.8, Tree type = cherry` appears **4 times**.
  - `Height = 8.78, Width = 3.93, Tree type = cherry` appears **3 times**.
  - `Height = 10.0, Width = 5.0, Tree type = cherry` appears **3 times**.
  - `Height = 10.0, Width = 4.5, Tree type = cherry` appears **2 times**.
  - `Height = 10.0, Width = 6.5, Tree type = cherry` appears **2 times**.

### Conflicting Feature Duplicates:
In addition to exact duplicates, there are **3 coordinate pairs** where identical features map to **different** classes:
- `(Height=11.0, Width=5.0)` $\rightarrow$ 1 labeled as `walnut` (index 384), 1 labeled as `cherry` (index 509).
- `(Height=16.0, Width=7.0)` $\rightarrow$ 1 labeled as `cherry` (index 495), 1 labeled as `walnut` (index 502).
- `(Height=16.0, Width=7.5)` $\rightarrow$ 1 labeled as `cherry` (index 414), 1 labeled as `walnut` (index 454).

---

### What would identical rows mean for a train/test split?

If the dataset is split into training and test sets (e.g., via standard random `train_test_split`) without addressing duplicates:

1. **Data Leakage (Train/Test Contamination)**:
   - When exact duplicates exist, identical copies of the same observation will inevitably end up in both the training set and the test set.
2. **Artificially Inflated / Overoptimistic Evaluation**:
   - The test set is meant to measure how well the model generalizes to *unseen* data. When duplicate rows leak into the test set, the model is tested on instances it has already memorized during training, leading to an overly optimistic estimate of test accuracy and low test loss.
3. **Failure to Detect Overfitting**:
   - High-capacity models (e.g., k-NN with $k=1$, unconstrained decision trees) will achieve near $100\%$ accuracy on leaked test points simply by rote memorization, obscuring true overfitting.
4. **Non-Separable Feature Ambiguity (Bayes Error)**:
   - For the 3 coordinate pairs with conflicting labels, identical feature inputs map to opposing target classes. A deterministic model cannot achieve $100\%$ accuracy on these points, setting a non-zero irreducible error (Bayes error) boundary.

### Recommended Remediation:
- **Deduplication**: Drop exact duplicate rows (`df.drop_duplicates()`) before performing any train/test split.
- **Grouped Splitting**: If identical rows represent repeated physical measurements of the same biological specimen, use `GroupShuffleSplit` or group by feature signature to guarantee that all duplicates remain in either the train set or test set.
- **Clean Label Conflicts**: Investigate or remove ambiguous points where identical dimensions map to conflicting species.

---

## 5. Unusual Observations & Outliers

1. **Extreme Anomalies / Suspected Label Swaps**:
   - **Index 390 (`cherry`)**: Height = $16.90\text{ cm}$, Width = $21.60\text{ cm}$. Normal cherry leaves rarely exceed $7\text{ cm}$ width; a width of $21.6\text{ cm}$ places this point deep inside the maple cluster. This is almost certainly a mislabeled maple leaf.
   - **Index 214 (`cherry`)**: Height = $14.56\text{ cm}$, Width = $17.08\text{ cm}$. Another extreme outlier for cherry that matches maple morphology.
   - **Index 249 (`maple`)**: Height = $7.57\text{ cm}$, Width = $3.45\text{ cm}$. An extremely narrow leaf labeled as maple, sitting squarely inside the cherry cluster.
   - **Index 425 (`maple`)**: Height = $7.70\text{ cm}$, Width = $5.40\text{ cm}$. Another narrow maple leaf sitting inside the cherry distribution.
2. **Heterogeneous Measurement Precision**:
   - The dataset contains mixed numerical precisions:
     - Rounded integers / coarse half-centimeters: `11.0, 5.0`, `7.5, 3.8`.
     - Standard 2-decimal digits: `12.3, 14.26`.
     - High-precision floating point values: `3.835459...`, `14.56297...`.
     - This strongly indicates the dataset was synthesized or merged from multiple distinct sensors/measurement methodologies.
