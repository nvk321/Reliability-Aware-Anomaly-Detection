# Claim Audit: Academic Verification & Statement Categorization

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Review Target:** Review 1 (Phase 1)  
**Academic Requirement:** Ensure scientific defensibility, eliminate overclaiming, and clearly differentiate between established literature, implemented Phase 1 work, proposed research components, and future experimental claims.

---

## 1. Classification Categories

Every technical assertion, architectural property, and contextual claim in the Review 1 package is audited against one of five formal categories:

1. `SAFE FOR REVIEW 1`: Factual statements regarding the project's institutional metadata, Phase 1 scope, implemented code scaffolding, verified mathematical formulas, or structural facts of the repository.
2. `REQUIRES CITATION`: Empirical facts, benchmark properties, established algorithms, or domain observations established in published literature that must cite a verified source.
3. `PROPOSED / FUTURE`: Architectural designs, mathematical formulations, hypotheses, and experimental protocols that define our proposed system (R-STNet) but represent research proposals rather than established historical facts.
4. `REQUIRES EXPERIMENT`: Hypotheses regarding model performance, robustness gains, false-alarm suppression rates, or ablation impacts that can only be substantiated after Phase 2 experimental execution.
5. `DO NOT CLAIM`: Unsubstantiated, unscientific, or false assertions that must be strictly banned from the Review 1 report.

---

## 2. Statement-by-Statement Audit Register

| # | Statement / Assertion | Audit Category | Justification & Verification Action |
| :---: | :--- | :---: | :--- |
| **1** | "SWaT contains 51 continuous sensors and discrete actuators across 6 physical water treatment stages." | **REQUIRES CITATION** | Factual physical plant configuration established by iTrust SUTD. Must cite Mathur & Tippenhauer (2016) [38] and Goh et al. (2017) [45]. |
| **2** | "The official SWaT normal training period comprises 496,800 rows at 1 Hz, and the attack period contains 449,919 rows with 12.14% attack density." | **REQUIRES CITATION** | Benchmark dataset specification verified via official SUTD documentation and repository data audit (`src/data/swat_loader.py`). |
| **3** | "Standard multivariate anomaly detectors treat all sensor streams as uniformly trustworthy." | **REQUIRES CITATION** | Established limitation documented in literature across GDN [19], MTAD-GAT [20], USAD [9], and OmniAnomaly [10]. |
| **4** | "A single corrupted sensor can inflate unweighted global reconstruction or prediction error, triggering false system shutdowns." | **REQUIRES CITATION** | Well-documented in process control and fault diagnosis literature (Qin 2012 [6], Gertler 1998 [33], Audibert et al. 2022 [27]). Requires citation; will be empirically tested in Phase 2. |
| **5** | "R-STNet is state-of-the-art across all multivariate time-series benchmarks." | **DO NOT CLAIM** | **STRICTLY FORBIDDEN.** The project is in Phase 1; experiments are not complete. Making SOTA claims at Review 1 is unscientific and academically dishonest. |
| **6** | "No previous work has ever studied noisy sensors or sensor faults in multivariate time series." | **DO NOT CLAIM** | **STRICTLY FORBIDDEN.** Existing literature has actively studied sensor fault detection (Gertler 1998 [33], Isermann 2006 [34], Saeed et al. 2021 [28], Audibert et al. 2022 [27]). The gap is specific to explicit channel-level reliability estimation integrated into deep spatio-temporal anomaly scoring. |
| **7** | "R-STNet is our proposed project architecture and framework, not an existing published model name." | **SAFE FOR REVIEW 1** | Transparently clarifies architectural attribution in accordance with academic integrity guidelines. |
| **8** | "R-STNet achieved an F1-score of 0.88 on SWaT under 20% Gaussian noise." | **REQUIRES EXPERIMENT** | **MUST NOT APPEAR IN REVIEW 1 REPORT.** Belongs strictly to Phase 2 experimental results. |
| **9** | "The proposed framework aims to estimate a continuous channel reliability score $r_i(t) \in [0, 1]$ online." | **PROPOSED / FUTURE** | Accurately describes the proposed research design and mathematical hypothesis. Uses appropriate future/proposed phrasing. |
| **10** | "The Sensor Reliability Module (SRM) triangulates temporal continuity, graph neighborhood consistency, and residual confidence." | **SAFE FOR REVIEW 1 / PROPOSED** | Formally implemented in `src/reliability/sensor_reliability.py` as our proposed methodological design. |
| **11** | "Point Adjustment (PA) protocol inflates F1-scores by marking entire anomalous segments as true positives upon a single point detection." | **REQUIRES CITATION** | Formally proven by Kim et al. (AAAI 2022) [40] and Wu & Keogh (IEEE TKDE 2021) [41]. Mandatory citation. |
| **12** | "We report raw point-wise Precision, Recall, and F1 as primary metrics, reporting Point-Adjusted F1 strictly as a labeled secondary metric." | **SAFE FOR REVIEW 1** | Accurate description of the project's established evaluation protocol (`src/evaluation/metrics.py`). |
| **13** | "CAROTS is a research reference baseline utilized for benchmark comparison under ICML 2025 conventions." | **SAFE FOR REVIEW 1** | Accurate description of `reference/CAROTS/` usage with full academic attribution in `docs/references/technical_references.md`. |
| **14** | "Our project is 'CAROTS + X' or an incremental extension of CAROTS." | **DO NOT CLAIM** | **STRICTLY FORBIDDEN.** R-STNet is an independent spatio-temporal graph attention architecture with an explicit reliability module and 4-way decision engine implemented in `src/`. |
| **15** | "Decision thresholds $\tau_{\text{sys}}$ and $\theta_{\text{rel}}$ are calibrated strictly on normal validation data without referencing test attack labels." | **SAFE FOR REVIEW 1** | Verifiable methodological fact implemented in `src/anomaly_detection/discriminator.py` to prevent data leakage. |
| **16** | "The 80/20 chronological split on normal SWaT data is the official SUTD protocol." | **DO NOT CLAIM** | **INCORRECT TERMINOLOGY.** The 7-day normal run is the official benchmark; the 80% train / 20% validation split is our **project-specific chronological partition**. |
| **17** | "SWaT attack labels provide ground-truth information indicating which individual sensor was physically faulty." | **DO NOT CLAIM** | **INCORRECT FACT.** SWaT labels represent system-level cyber-physical attacks; they do not isolate sensor hardware degradation. Sensor fault ground truth is provided by our controlled corruption suite. |
| **18** | "The proposed 4-way discrimination engine classifies operating conditions into Normal, Sensor Fault, System Anomaly, and System Anomaly with Sensor Degradation." | **PROPOSED / FUTURE** | Accurately describes the proposed decision framework implemented in `src/anomaly_detection/discriminator.py`. |
| **19** | "R-STNet significantly outperforms Isolation Forest and Autoencoders under stuck sensor conditions." | **REQUIRES EXPERIMENT** | This is a research hypothesis (Hypothesis 1 & 2 in `docs/EXPERIMENT_PLAN.md`), not a validated Review 1 finding. Must be framed as an expected outcome to be tested in Phase 2. |
| **20** | "Hardware execution is constrained by an NVIDIA GeForce RTX 3050 Laptop GPU with 4096 MiB VRAM." | **SAFE FOR REVIEW 1** | Verified local hardware fact documented in `docs/REPRODUCIBILITY.md` and confirmed via PyTorch CUDA audit. |

---

## 3. Strict Review 1 Writing Protocol

To maintain complete academic defensibility during the Review 1 evaluation:
1. **Never use past tense for experimental findings:** Use "the proposed framework will be evaluated..." or "the study investigates whether..." rather than "the model achieved..." or "experiments proved that...".
2. **Mandate citations for all domain claims:** Every reference to industrial water systems, sensor failure modes, baseline models, or evaluation pitfalls must link directly to verified references [1]–[45].
3. **Explicitly label proposed vs. implemented:** Code that exists in `src/` is labeled "implemented Phase 1 scaffolding"; model performance under evaluation is labeled "planned Phase 2 experimental study".
