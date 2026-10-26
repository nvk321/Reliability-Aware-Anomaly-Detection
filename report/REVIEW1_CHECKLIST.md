# Review 1 Compliance and Verification Checklist

**Project Title:** RELIABILITY-AWARE ANOMALY DETECTION FOR MULTIVARIATE SENSOR DATA  
**Course:** Machine Learning Lab (BCSE209P), Slot L47 + L48  
**Academic Term:** Fall Semester 2026-27  
**Students:** P NAGA VENKATA KUSHAL (23BDS0133), KOTERU TARUN (23BDS0103)  
**Faculty Guide:** Dr. DEGA NAGARAJU  
**Institution:** School of Computer Science and Engineering (SCOPE), VIT Vellore  
**Target Review:** REVIEW 1 (Phase 1 Evaluation)  
**Date of Verification:** October 2026  

---

## 1. Structural and Required Content Verification

| Item # | Verification Item | Compliance Status | Supporting Documentation & Reference in Report Package |
| :---: | :--- | :---: | :--- |
| 1 | **Project Title & Identification** | **VERIFIED** | Matches official registration: "RELIABILITY-AWARE ANOMALY DETECTION FOR MULTIVARIATE SENSOR DATA". |
| 2 | **Academic Certificate Format** | **VERIFIED** | Formatted using official VIT institutional guidelines and faculty guide attribution. |
| 3 | **Review 1 Abstract (200–300 words)** | **VERIFIED** | Covers background, problem, motivation, gap, objectives, proposed methodology, dataset, and expected contributions. Uses future/proposed phrasing without reporting completed results. |
| 4 | **Chapter 1: Introduction (All 17 Subsections)** | **VERIFIED** | Includes 1.1 to 1.17 complete text, equations, problem statement, research questions, measurable objectives, scope, expected contributions. |
| 5 | **Problem Statement (Formal Mathematical)** | **VERIFIED** | Formulated as multivariate observation deviation decoupling: $X_t \in \mathbb{R}^M$, separating sensor-level observation corruption from genuine physical process abnormality. |
| 6 | **Research Questions** | **VERIFIED** | Formulates primary research question (RQ1) and 3 supporting research questions (RQ2–RQ4). |
| 7 | **Measurable Objectives (8 Points)** | **VERIFIED** | Formulated as formal scientific objectives without claiming premature completion. |
| 8 | **Expected Contributions** | **VERIFIED** | Explicitly labeled as proposed/expected contributions with future tense phrasing. |
| 9 | **Chapter 2: Literature Review (All 17 Subsections)** | **VERIFIED** | Organized into 5 major themes; thematic synthesis rather than paper-by-paper summaries. |
| 10 | **Literature Reference Target ($\ge 35$--$40$ Verified Papers)** | **VERIFIED** | **45 verified academic papers** included. No fabricated citations. Verified DOIs, authors, years, and venues. |
| 11 | **ScienceDirect / Elsevier Target ($\approx 90\%$)** | **VERIFIED** | High proportion of Elsevier journals (*Computers & Chemical Engineering*, *Reliability Engineering & System Safety*, *Computers in Industry*, *Measurement*, *Journal of Process Control*, *ISA Transactions*, *Control Engineering Practice*, *Annual Reviews in Control*, *Expert Systems with Applications*, *Neurocomputing*) plus foundational papers. |
| 12 | **Literature Comparison Matrix** | **VERIFIED** | `report/LITERATURE_MATRIX.md` evaluates 20 representative methods across 15 technical dimensions. |
| 13 | **Research Gap Specification** | **VERIFIED** | `report/RESEARCH_GAP.md` logically deduces the specific gap around explicit channel-level reliability estimation and false-alarm suppression. |
| 14 | **Dataset Specification (SWaT Primary, WADI Secondary)** | **VERIFIED** | Detailed in `report/DATASET_RESEARCH.md` covering physical stages, sensors/actuators, official sequences, and attack label vs. sensor fault distinction. |
| 15 | **Leak-Free Preprocessing & Chronological Split** | **VERIFIED** | 80% train / 20% val chronological normal split; scalers fit strictly on train split; zero test leakage. |
| 16 | **Chapter 3: Methodology (All 18 Subsections)** | **VERIFIED** | Complete mathematical formulation of R-STNet, SRM, reliability-weighted scoring, 4-way decision engine, corruption model, baselines, and evaluation protocol. |
| 17 | **Sensor Reliability Module (SRM) Formulation** | **VERIFIED** | Triangulates temporal continuity $v_i(t)$, graph neighbor consensus $c_i(t)$, and residual tail confidence $p_i(t)$ via calibrated logit fusion into $r_i(t) \in [0, 1]$. |
| 18 | **Four-Way Operational State Engine** | **VERIFIED** | Formulates Case A (Normal), Case B (Sensor Fault), Case C (System Anomaly), and Case D (System Anomaly + Degradation). |
| 19 | **Baseline Models Specification** | **VERIFIED** | Baseline A (Isolation Forest), Baseline B (ST-AE), Baseline C (RSTNet without SRM - direct ablation), and Baseline 3 (CAROTS reference). |
| 20 | **Planned 10-Experiment Controlled Corruption Suite** | **VERIFIED** | 9 failure modes across 4 severity tiers with deterministic seeds (`seed=42, 101, 2024`). |
| 21 | **Evaluation Metrics Framework** | **VERIFIED** | Raw point-wise Precision, Recall, F1, PR-AUC, FPR, FASR, and Sensor Fault F1. PA-F1 labeled secondary. |
| 22 | **Threshold Calibration Protocol** | **VERIFIED** | Thresholds $\tau_{\text{sys}}$ and $\theta_{\text{rel}}$ calibrated strictly on validation normal data; frozen for blind test evaluation. |
| 23 | **Mermaid Architectural Diagrams (7 Figures)** | **VERIFIED** | `report/FIGURES/` contains Figures 1 through 7; documented in `report/FIGURE_DESCRIPTIONS.md`. |
| 24 | **Report Tables (9 Tables)** | **VERIFIED** | `report/TABLES_REVIEW1.md` contains Tables 1 through 9. |
| 25 | **IEEE Reference Bibliography & BibTeX** | **VERIFIED** | Complete `report/REFERENCES_IEEE.md` and `report/references.bib`. |

---

## 2. Strict Negative Scope Audit (Verification of What Is NOT Included)

In strict compliance with university Review 1 guidelines for Phase 1:

| Negative Scope Item | Verification Status | Confirmation Detail |
| :--- | :---: | :--- |
| **No Completed Chapter 4 (Results)** | **VERIFIED: ABSENT** | The report stops strictly at Chapter 3 (Methodology & Planned Experimental Plan). No Chapter 4 exists. |
| **No Completed Chapter 5 (Conclusion)** | **VERIFIED: ABSENT** | No final project conclusion based on experimental findings is included. |
| **No Final Performance / Accuracy Claims** | **VERIFIED: ABSENT** | No claims such as "R-STNet achieved 0.88 F1" or "R-STNet outperformed baseline X" appear in the report. |
| **No Completed Model Comparison Tables** | **VERIFIED: ABSENT** | Tables 1–9 are structural, architectural, and planning matrices; zero empirical result numbers are presented. |
| **No Completed Ablation Study Results** | **VERIFIED: ABSENT** | The ablation study is documented strictly as a planned experimental protocol for Phase 2. |
| **No False SOTA Claims** | **VERIFIED: ABSENT** | The framework is positioned as a scientifically defensible proposed architecture currently in Phase 1 development. |
| **No Overclaiming of Novelty** | **VERIFIED: ABSENT** | Prior literature in robust anomaly detection and sensor fault diagnosis is fully acknowledged with careful academic language. |

---

## 3. Final Certification

We hereby verify that the Review 1 documentation package in `report/` faithfully represents the project status as of Phase 1 (October 2026). The documentation demonstrates thorough domain understanding, rigorous literature review, clear mathematical formulation, leak-free preprocessing protocol, and an exhaustive experimental plan, while honestly acknowledging that large-scale experimentation and final performance claims belong to Phase 2.
