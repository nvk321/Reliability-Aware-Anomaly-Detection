# Git Release Audit & Verification Record

**Project Title:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Repository:** Reliability-Aware-Anomaly-Detection  
**Course:** BCSE209P — Machine Learning Lab  
**Students:** P Naga Venkata Kushal (23BDS0133), Koteru Tarun (23BDS0103)  
**Faculty Guide:** Dr. Dega Nagaraju  
**Release Tag:** `v1.0.0-final`  
**Audit Date:** October 26, 2026  

---

## 1. Git State Audit

- **Active Branch:** `main`
- **Initial Repository State:** Fresh local repository initialized with `git init`.
- **Remote Configuration (`git remote -v`):** No remote configured currently.  
  *(Notice: The local repository is fully prepared, tested, and committed systematically. User action is required to configure the remote URL before pushing, e.g., `git remote add origin https://github.com/<username>/Reliability-Aware-Anomaly-Detection.git`).*
- **Working Tree Cleanliness:** Staged systematically into 7 atomic logical commits.
- **Large Files Check:** All large raw datasets and binary matrices (`data/processed/SWaT/*.npy`) are safely excluded via `.gitignore`. Only release model checkpoints in `results/checkpoints/` ($< 1\text{ MB}$ total) and high-resolution figures are included.
- **Secrets & Path Hygiene:** Audited with regex; zero API keys, passwords, bearer tokens, or personal Windows drive letters in source code.

---

## 2. Systematic Commit Sequence

| Commit # | Hash | Logical Scope | Files Included | Commit Message | Status |
| :---: | :---: | :--- | :--- | :--- | :---: |
| **1** | `0e3008a` | Repository Metadata & Docs | `README.md`, `LICENSE`, `CITATION.cff`, `CHANGELOG.md`, `environment.yml`, `requirements.txt`, `.gitignore`, `data/README.md`, `data/processed/README.md`, `docs/*.md`, `configs/` | `docs: finalize project documentation and repository metadata` | **PASS** |
| **2** | `532615e` | Reproducibility & Pipeline | `scripts/*.py`, `src/` core packages, `models/` | `chore: add reproducibility and project health checks` | **PASS** |
| **3** | `eede8d5` | Unit Test Suite | `tests/*.py` (all 28 unit tests) | `test: add comprehensive project validation suite` | **PASS** |
| **4** | `053ffce` | Interactive Demo | `demo/app.py`, `demo/README.md`, `docs/DEMO.md`, `presentation/DEMO_SCRIPT.md` | `feat: add interactive Streamlit research demonstration` | **PASS** |
| **5** | `d58489d` | Thesis Report & Results | `report/FINAL/`, `report/REVIEW1/`, `report/REVIEW2/`, `results/` | `docs: finalize thesis report and Chapter 5` | **PASS** |
| **6** | `17e3c29` | Presentation & Defense | `presentation/FINAL_PRESENTATION.md` | `docs: add final presentation and defense materials` | **PASS** |
| **7** | `[Pending]` | Release Sign-Off | `FINALIZATION_STATUS.md`, `docs/GIT_RELEASE_AUDIT.md` | `chore: prepare v1.0.0 final release` | **IN PROGRESS** |

---

## 3. Automated QA Sign-Off Prior to Release

- **Project Health Check:** `.venv\Scripts\python.exe scripts/project_health_check.py` $\rightarrow$ **`PROJECT HEALTH CHECK: PASS`**
- **Test Suite Execution:** `.venv\Scripts\pytest.exe tests/ -q` $\rightarrow$ **`28 passed in 12.48s`**
- **Streamlit Demo Execution:** Verified clean execution with Altair interactive specs and fast checkpoint restoration ($< 20\text{ ms}$).

---

## 4. Final Document Format Disclosure

- **Markdown Source:** All thesis chapters (Chapters 1–5, References, Appendices) are compiled in Markdown.
- **DOCX / PDF Formatted Thesis:** `FINAL DOCX/PDF: NOT GENERATED` (Markdown serves as the authoritative source; university Word/PDF formatting in Times New Roman 12 pt, 1.5 line spacing should be rendered prior to printing).
