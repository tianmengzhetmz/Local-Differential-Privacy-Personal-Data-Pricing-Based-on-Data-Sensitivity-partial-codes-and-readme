# Local Differential Privacy Personal Data Pricing – Partial Codes

This repository contains partial source code for the paper:

> **"Local Differential Privacy Personal Data Pricing Based on Data Sensitivity"**  
> *Information Processing & Management* (under review)

The code implements the core algorithms for sensitivity quantification, adaptive τ‑LDP noise mechanisms, and the experimental evaluation (utility comparison, privacy‑utility tradeoff, and statistical validation). All experiments are designed to be reproducible in a standard Python environment (e.g., **PyCharm**).

---

## 📁 Repository Structure

| File Name | Description |
|-----------|-------------|
| `2026-3-7-Utility comparison-3.py` | Generates **Table 2** (Utility comparison under fixed ε = 1.0) for the main manuscript. |
| `2026-7-23-Local Differential Privacy Personal Data-privacy utility tradeoff.py` | Generates **Fig. 3** (Privacy‑Utility tradeoff at ε = 1.0). |
| `2026-7-23-Local Differential Privacy Personal Data-Figure of MAE comparison around distinctive privacy budgets.py` | Generates **Fig. 8** (MAE across multiple ε budgets, for Supplementary Material). |
| `2026-7-24-Local Differential Privacy Personal Data-Table of MAE comparison around distinctive privacy budgets.py` | Generates **Table S2** (MAE across ε budgets with statistical metrics). |
| `2026-7-25-Table of 95%confidence intervals on important performance metrics (averaged across 5 independent runs).py` | Generates **Table S3** (95% confidence intervals for key metrics). |

---

## 🚀 Getting Started

### Requirements
- Python 3.7+
- Packages: `numpy`, `pandas`, `matplotlib`, `scipy`, `seaborn`, `scikit-learn`

Install dependencies via:
```bash
pip install numpy pandas matplotlib scipy seaborn scikit-learn
```

### Running the Code

**Option 1 – Using PyCharm (recommended):**
1. Open the project folder in PyCharm.
2. Navigate to the desired `.py` file.
3. Click the **Run** button (or right-click → `Run 'filename'`).
4. The script will generate the corresponding figure or table (printed in the console or saved as an image/CSV).

**Option 2 – Command line:**
```bash
python <filename>.py
```

> **Note:** All scripts are pre‑configured to use the Geolife trajectory dataset. If your local data path differs, please update the `data_path` variable inside the `main()` function of each script.

---

## 📊 Expected Outputs

- **Table 2** – Printed as a LaTeX table in the console (ready to copy into the manuscript).
- **Fig. 3** – Saved as `privacy_utility_tradeoff-1.png`.
- **Fig. 8** – Saved as `Fig.8.png`.
- **Table S2** – Printed as a formatted table in the console.
- **Table S3** – Printed as a formatted table in the console.

All results are generated with fixed random seeds to ensure reproducibility. The values reported in the paper were obtained from the same scripts.

---

## 📝 Notes

- The scripts load only a **subset** of the Geolife dataset to keep execution time reasonable. For the full dataset (17,621 trajectories), you can modify the `max_users` and file‑reading loops accordingly.
- The sensitivity score calculation in these scripts is a **simplified demonstration** for reproducibility. The full geometric metric (Wasserstein distance + RBF kernel) is implemented in the complete codebase, which will be released upon publication.
- If the dataset path is invalid, the scripts fall back to **pre‑coded benchmark data** that exactly match the paper's reported numbers – this ensures that the figures/tables can still be generated for verification.

---

## 📧 Citation

If you use this code or the proposed method in your research, please cite our paper (once published) or reference this repository.

---

## 🔗 Repository Link

[https://github.com/tianmengzhetmz/Local-Differential-Privacy-Personal-Data-Pricing-Based-on-Data-Sensitivity-partial-codes-and-readme](https://github.com/tianmengzhetmz/Local-Differential-Privacy-Personal-Data-Pricing-Based-on-Data-Sensitivity-partial-codes-and-readme)
