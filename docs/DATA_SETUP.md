# Data Setup and Preparation Protocol

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Primary Dataset:** **SWaT (Secure Water Treatment)**  
**Secondary Dataset:** **WADI (Water Distribution)**  

---

## 1. Dataset Decision & Prioritization

| Dataset | Priority | Domain | Channels | Time Resolution | Status in Workspace | Decision & Justification |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **SWaT** | **Primary (Tier 1)** | Industrial Water Treatment Plant | 51 | 1 second (1 Hz) | **Available** in `reference/swat_data/` | **REQUIRED IMMEDIATELY.** Perfect representation of physical multi-stage sensor processes. Strict physical interdependencies (flow, tank level, pressure) allow clear separation between sensor faults and physical anomalies. |
| **WADI** | **Secondary (Tier 2)** | Water Distribution Testbed | 123 | 1 second (1 Hz) | **Available** in `reference/wadi_data/` | **REQUIRED FOR EXTERNAL VALIDATION.** Secondary validation platform to demonstrate generalization of the reliability mechanism across a larger industrial network. |
| **PSM** | Optional (Tier 3) | eBay Server Node Metrics | 26 | 1 minute | Reference code available in `reference/CAROTS/` | **DEFERRED.** High-dimensional IT server metrics lack physical conservation laws. Kept as optional extension after SWaT/WADI are validated. |
| **SMD** | Optional (Tier 4) | Server Machine Dataset | 38 | 1 minute | Reference code available in `reference/CAROTS/` | **DEFERRED.** Not cyber-physical; defer unless computing headroom and time permits. |

---

## 2. Official Sources, Access Requirements, and Raw Dataset Structure

### 2.1. SWaT (Secure Water Treatment Testbed)
- **Official Source:** iTrust Centre for Research in Cyber Security, Singapore University of Technology and Design (SUTD).
- **Official URL:** `https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/`
- **Release Version:** **SWaT.A1 & A2 (December 2015 testbed run)**.
- **Physical Testbed Details:** 6 physical stages with 51 sensors and actuators:
  - *P1:* Raw water intake & chemical dosing (FIT101, LIT101, P101/P102).
  - *P2:* Pre-treatment & coagulation (AIT201–AIT203, FIT201, P201–P206).
  - *P3:* Ultrafiltration (UF) (DPIT301, FIT301, LIT301, P301/P302).
  - *P4:* Dechlorination using UV lamps (AIT401/AIT402, FIT401, LIT401).
  - *P5:* Reverse Osmosis (RO) (AIT501–AIT504, FIT501–FIT504, PIT501–PIT503).
  - *P6:* Clean water storage and backwash (FIT601, P601–P603).

### 2.2. Critical Raw File Layout Audit & Reconstruction
Inspection of `reference/swat_data/` revealed how the raw archives map to the official SUTD benchmark:
1. **Official 7-Day Normal Training Run (`SWaT_Dataset_Normal_v1.csv`):**
   - Resides in rows **395,298 to 892,097** of `normal.csv`.
   - Exactly **496,800 rows** ($y = 0$, 100% normal) spanning `22/12/2015 4:00:00 PM` to `28/12/2015 9:59:59 AM`.
   - **Chronological Split:**
     - *Training Set (80%):* First 397,440 rows (or 39,744 at 10x downsample) — pure normal.
     - *Validation Set (20%):* Next 99,360 rows (or 9,936 at 10x downsample) — pure normal, reserved strictly for threshold tuning.
2. **Official 4-Day Attack Test Run (`SWaT_Dataset_Attack_v0.csv`):**
   - Spans `28/12/2015 10:00:00 AM` to `02/01/2016 2:59:59 PM` (exactly **449,919 rows**).
   - In `reference/swat_data/`, this run had been split by label into:
     - The first 395,298 rows of `normal.csv` (normal periods between attacks, $y=0$).
     - All 54,621 rows of `attack.csv` (attack periods, $y=1$).
   - `SWaTLoader` reconstructs the authentic test run by merging and sorting them chronologically by timestamp.
   - Resulting test split: **449,919 rows** (or 44,992 at 10x downsample) with **12.14% attack density**, providing both positive and negative classes for genuine Precision, Recall, FPR, and PR-AUC measurement.

### 2.2. WADI (Water Distribution Testbed)
- **Official Source:** iTrust, SUTD.
- **Official URL:** `https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/`
- **Physical Testbed Details:** Scaled-down water distribution network with 123 sensors/actuators including water tanks, booster pumps, valves, and water quality analyzers across three distribution loops.

---

## 3. Directory Layout and Organization

All raw files remain isolated. The final project pipeline reads from `reference/` (or downloaded archives) and populates `data/processed/` and `data/corrupted/`:

```
Reliability-Aware-Anomaly-Detection/
│
├── data/
│   ├── raw/
│   │   ├── SWaT/              # Symlinked or copied raw source files
│   │   └── WADI/
│   │
│   ├── processed/
│   │   ├── SWaT/
│   │   │   ├── train.npy      # Normalized clean normal data (train split)
│   │   │   ├── val.npy        # Normalized clean normal data (validation split)
│   │   │   ├── test.npy       # Normalized attack data (test split)
│   │   │   ├── test_labels.npy# Binary ground-truth labels (0 = Normal, 1 = Attack)
│   │   │   ├── sensor_names.json # Channel names and stage metadata
│   │   │   └── scaler.pkl     # Fitted standard scaler
│   │   │
│   │   └── WADI/
│   │       ├── train.npy
│   │       ├── val.npy
│   │       ├── test.npy
│   │       └── test_labels.npy
│   │
│   └── corrupted/
│       ├── SWaT/
│       │   ├── corruption_metadata.json # Registry of all synthetic corruptions
│       │   ├── test_noise_5pct.npy
│       │   ├── test_noise_20pct.npy
│       │   ├── test_drift_10pct.npy
│       │   ├── test_stuck_sensor.npy
│       │   └── test_spikes.npy
│       └── WADI/
│
└── reference/
    ├── swat_data/             # Existing reference raw data (read-only)
    │   ├── normal.csv         # 402 MB (Normal 7 days operation)
    │   ├── attack.csv         # 15 MB (Attack 4 days operation)
    │   └── merged.csv         # 426 MB
    └── wadi_data/
        ├── WADI_14days_new.csv       # 493 MB (14 days normal operation)
        └── WADI_attackdataLABLE.csv  # 109 MB (Attack operation with labels)
```

---

## 4. Preprocessing Protocol & Standard Benchmarking Rules

To ensure fair scientific benchmarking consistent with literature (USAD, OmniAnomaly, GDN, CAROTS):

1. **Downsampling:**
   - Raw SWaT is sampled at 1-second intervals (946,960 rows total).
   - Downsample by a factor of **5** (or **10** for rapid baseline verification). A factor of 5 yields ~189,392 rows, preserving physical transients while speeding training by 5x.
2. **Chronological Splitting:**
   - **Training Set:** First 80% of normal operation data (`normal.csv`).
   - **Validation Set:** Latter 20% of normal operation data (`normal.csv`). Used strictly for:
     - Learning normal residual distributions ($\text{Median}(e_i)$, $\text{MAD}_i$).
     - Calibrating reliability fusion weights ($w_v, w_c, w_p, b_r$).
     - Setting decision thresholds ($\tau_{\text{sys}}, \theta_{\text{rel}}$).
   - **Test Set:** Attack operation data (`attack.csv`), containing known cyber-physical attacks and process deviations.
3. **Data Leakage Prevention:**
   - Scalers (StandardScaler / RobustScaler) are **fit strictly on the training set**.
   - The validation and test sets are transformed using the training scaler parameters.
4. **Sliding Window Generation:**
   - Window size $L = 60$ time steps ($\approx 5$ minutes of operation at 5s downsampling).
   - Stride $S_{\text{train}} = 1$ (or $S=5$ for faster epochs), $S_{\text{test}} = 1$.

---

## 5. Validation Checks

Before feeding processed tensors into any model, `src/data/swat_loader.py` executes:
1. `assert not np.isnan(train).any()`, `assert not np.isinf(train).any()`
2. Dimension verification: $\text{shape} = (N, L, M)$ where $M=51$ for SWaT, $M=123$ for WADI.
3. Label verification: `np.unique(test_labels) == [0, 1]`.
4. Variance check: verify no constant feature across the entire training set (flag or drop if zero variance in raw sensors).
