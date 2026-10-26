# DRAFT README: Jet Engine Anomaly Detection Scoring Using NASA N-CMAPSS

**SIADS 696 Milestone II, University of Michigan School of Information (Fall 2026)**
Luciana Qu · Danielle Vykydal · Dillon Wollberg

Can we tell when a jet engine stops being healthy? This project uses NASA's N-CMAPSS turbofan simulation (subset DS08a: 15 engines, 994 flights) to detect degradation two ways:

* **Part A (Supervised):** classifiers trained on labeled flights to predict each flight's health status (`hs`).
* **Part B (Unsupervised):** models trained on **healthy flights only** that produce a continuous anomaly score, then checked against `hs`.

All parts use the same data file and the same convention: `hs = 1` is healthy, `hs = 0` is unhealthy, and **unhealthy is the positive class**.

---

## Repository Structure

```text
696-milestone-2/
├── README.md
├── requirements.txt
├── .gitignore
├── supervised_learning/      ← Part A (Dillon): tree ensembles, feed-forward networks
│   ├── ds_engineer.py
│   ├── TreeEnsemble.ipynb
│   ├── Supervised_FFN.ipynb
│   ├── Intermediate results for various model runs (json and csv)
│   └── supervised learning models (json)
├── isolation_forest/         ← Part B (Luciana): Isolation Forest
│   ├── 01_data_loading_eda.ipynb
│   ├── 02_isolation_forest.ipynb
│   ├── 03_isolation_forest_grid_search.ipynb
├── autoencoder/              ← Part B (Danielle): LSTM autoencoder
│   └── lstmautoencoder.ipynb
└── report/                   ← final report (PDF)
```

---

## Data

**Source:** NASA Prognostics Center of Excellence Data Set Repository, Dataset #17: *Turbofan Engine Degradation Simulation-2 (N-CMAPSS)*
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Dataset paper: Arias Chao, M., Kulkarni, C., Goebel, K., & Fink, O. (2021). Aircraft engine run-to-failure dataset under real flight conditions for prognostics and diagnostics. *Data, 6*(1), 5.

This project uses the **DS08a** subset (`N-CMAPSS_DS08a-009.h5`, ~3.2 GB), which is too large for GitHub. Download it from the link above (Dataset #17) and place it in `isolation_forest/raw_data/` to rebuild the processed files.

### Processed Files (included)

| File                                          | Contents                                                                   | Size   |
|-----------------------------------------------|----------------------------------------------------------------------------|--------|
| `isolation_forest/data/ds08a_rows.parquet`    | 67,328 rows, downsampled to 1 row per 76 s (FC1), 127 s (FC2), 180 s (FC3) | ~13 MB |
| `isolation_forest/data/ds08a_flights.parquet` | 994 per-flight summaries computed from the full 1 Hz data                  | ~1 MB  |
| `isolation_forest/data/ds08a_units.parquet`   | 15 engines: NASA split, flight class, health-parameter changes             | ~11 KB |


To rebuild them from the raw file:

```bash
cd isolation_forest
python scripts/build_ds08a_from_h5.py raw_data/N-CMAPSS_DS08a-009.h5 data
```

The script reads the HDF5 file one engine at a time (8–16 GB of RAM is enough).

---

## Setup

Python **3.12** is recommended (some packages, e.g. `torch` and `catboost`, may not yet publish wheels for the newest Python releases).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then select the `.venv` kernel in Jupyter or VS Code.

> If notebooks with gradient-boosting models run unusually slowly, each one sets `OMP_NUM_THREADS=4` in its first cell; restart the kernel and run all cells from the top.

---

## Engine Split

Splits are made **by engine**, so no engine appears in more than one partition.

| Part | Train | Validation | Test |
|---|---|---|---|
| Part A (Supervised) | 3, 4, 5, 6, 7, 8, 9, 11, 15 | 2, 13, 14 | **1, 10, 12** |
| Part B (Isolation Forest, LSTM Autoencoder) | 2, 3, 5, 6, 7, 8, 9, 13, 15 | 4, 10, 14 | **1, 11, 12** |

Part B's split is stratified by flight class (one FC1, FC2, and FC3 engine in validation and test). Part A uses `GroupShuffleSplit(random_state=42)`; its test set contains no FC2 engine.

---

## How to Run

### Part A: Supervised Learning (`supervised_learning/`)
| Notebook               | What It Does |
|------------------------|---|
| `ds_engineer.py`       | renames columns and calculates efficiency metrics
| `TreeEnsemble.ipynb`   | XGBoost, LightGBM, CatBoost with grouped cross-validation
| `Supervised_FFN.ipynb` | feed-forward networks and feature analysis

* Either file can be run in any order.
* All files (other than the datasets) required to run are in the supervised learning folder.
* TreeEnsemble uses ds08a_rows.parquet, and Supervised_FFN uses engineered_ds08a

### Part B: Isolation Forest (`isolation_forest/`)

| Notebook | What It Does |
|---|---|
| `01_data_loading_eda.ipynb` | Loads the data, audits it, and selects the 12 physical sensors (drops near-duplicates `P21`, `P40`) |
| `02_isolation_forest.ipynb` | Baseline → operating-condition residuals → smoothing → virtual sensors; evaluation, figures, alarm thresholds |
| `03_isolation_forest_grid_search.ipynb` | 192-configuration grid search × 5 seeds, baseline check, and correction-model sensitivity (gradient boosting vs. MLP) |

Run in order; each reads from `data/`.

### Part B: LSTM Autoencoder (`autoencoder/`)

*[Danielle: confirm]*

`lstmautoencoder.ipynb`: MLP operating-condition normalization (following Sánchez et al., 2026), 10-step sliding windows, LSTM autoencoder trained on healthy windows, hyperparameter tuning on validation reconstruction loss, and test evaluation. A GPU is recommended for tuning.

---

## Notes for Reproducing

* *[To update before submission]* Some notebooks were developed in Google Colab and still mount Google Drive or use absolute paths (e.g., `/content/drive/...`). Replace these with relative paths to the files in this repository.
* The raw `.h5` file, virtual environments, and checkpoints are excluded by `.gitignore`.

---

## References

* Arias Chao, M., Kulkarni, C., Goebel, K., & Fink, O. (2021). Aircraft engine run-to-failure dataset under real flight conditions for prognostics and diagnostics. *Data, 6*(1), 5.
* Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. *2008 Eighth IEEE International Conference on Data Mining*, 413–422.
* Sánchez, P., Reyes, K., Radu, B., & Fernández, E. (2026). *Early fault detection on CMAPSS with unsupervised LSTM autoencoders*. arXiv. https://doi.org/10.48550/arXiv.2601.10269

See the final report for the complete reference list.
