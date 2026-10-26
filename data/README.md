# Dataset Directory & Acquisition Guidelines

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Primary Dataset:** Secure Water Treatment (SWaT) Testbed  
**Provider:** iTrust Centre for Research in Cyber Security, Singapore University of Technology and Design (SUTD)

---

## 1. Ethical Compliance & Redistribution Notice

In strict adherence to the **SUTD iTrust Dataset Terms of Use and Non-Disclosure Agreement (NDA)**:
- **Raw SWaT telemetry CSV files are NOT redistributed** in this public GitHub repository.
- Researchers and practitioners must request access directly from the official SUTD iTrust portal:  
  👉 **[SUTD iTrust Labs Datasets Request](https://itrust.sutd.edu.sg/itrust-labs-datasets/)**

---

## 2. Official Academic Citation

Any academic publications or derived experiments utilizing this dataset must cite the original dataset paper:

> J. Goh, S. Adepu, M. Junejo, and A. Mathur, "A Dataset to Support Research in the Design of Secure Water Treatment Systems," in *Critical Information Infrastructures Security (CRITIS 2016)*, Lecture Notes in Computer Science, vol. 10014, Springer, Cham, 2016, pp. 88–99. doi: [10.1007/978-3-319-46935-5_8](https://doi.org/10.1007/978-3-319-46935-5_8).

---

## 3. Directory Layout

```text
data/
├── README.md               # Dataset policy and acquisition guidelines (this file)
├── raw/                    # User-downloaded raw CSV files (excluded from Git)
│   ├── .gitkeep
│   ├── SWaT_Dataset_Normal_v1.csv
│   └── SWaT_Dataset_Attack_v0.csv
├── processed/
│   ├── README.md           # Instructions on processed matrices and scalers
│   └── SWaT/               # 10x downsampled train/val/test matrices (excluded from Git)
└── corrupted/              # Controlled synthetic corruption test suites
    └── .gitkeep
```

---

## 4. Preprocessing Pipeline

Once raw CSV files are placed in `data/raw/`, execute:

```bash
python scripts/prepare_data.py --config configs/swat.yaml
```

This performs:
1. Header sanitization and timestamp indexing.
2. 10x temporal downsampling (10s intervals) preserving physical cyber-physical transients.
3. Training-only normalization via `RobustStandardScaler` with zero-variance protection.
4. Chronological 80/20 train/validation split on normal data and complete mixed test split.
