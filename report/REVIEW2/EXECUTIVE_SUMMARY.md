# EXECUTIVE SUMMARY: REVIEW 2 EXPERIMENTAL MILESTONE

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  
**Course:** BCSE209P — Machine Learning Lab  
**Faculty Guide:** Dr. Dega Nagaraju  
**Date:** October 2026  
**Status:** Completed Review 2 Scientific Package  

---

## 1. Project Background and Objective

Multivariate sensor anomaly detection is essential for safeguarding industrial cyber-physical systems (water treatment, chemical refining, power distribution). Traditional deep learning approaches (autoencoders, recurrent models, graph attention networks) calculate anomaly scores by uniformly aggregating reconstruction or forecasting residuals across all sensors. 

However, in operational industrial environments, sensor telemetry is frequently contaminated by non-adversarial physical degradation: electrical noise, calibration drift, frozen sensor states, or unrecorded operational shifts. When this occurs, conventional models interpret localized sensor faults as plant-wide cyber-physical attacks, generating intolerable false alarm rates.

This project investigates the primary research question:
> **Can explicit sensor-reliability estimation improve the robustness of multivariate time-series anomaly detection when sensor observations are missing, noisy, or corrupted?**

To address this, we developed the **Reliability-Aware Spatio-Temporal Network (R-STNet)**, augmenting a spatiotemporal graph neural network with an auxiliary **Sensor Reliability Module (SRM)** that evaluates temporal continuity, topological graph consistency, and predictive residual confidence to dynamically downweight untrustworthy sensor streams.

---

## 2. Key Audited Findings

### 2.1 The Clean SWaT Benchmark Reality
- **Isolation Forest Baseline is Dominant on Clean Data:** Baseline A (Isolation Forest) achieved a Point-wise F1 score of **$0.7182$**, precision of **$0.8995$**, and a False Positive Rate (FPR) of **$0.0092$** ($0.92\%$).
- **Deep Reconstruction Baselines Collapse to Near-100% False Alarms:** Baseline B (ST-AE) and Baseline C (RSTNetNoSRM) collapsed into false-positive saturation, exhibiting FPRs of **$100.0\%$** and **$99.86\%$** respectively ($F_1 \approx 0.217$).
- **Proposed R-STNet Delivers a Major Within-Architecture Improvement:** Under an identical $23,233$-parameter spatiotemporal backbone, Proposed R-STNet achieved an F1 score of **$0.2740$** (+26.2% relative improvement over Baseline C) and reduced the False Positive Rate to **$62.14\%$**.
- **False Alarm Elimination:** R-STNet avoided **$14,898$ false alarms** ($37.8\%$ relative false alarm reduction), increasing True Negatives from $56$ to $14,954$ while detecting $87.13\%$ of true attacks.

### 2.2 The Root Cause: SWaT Channel Distribution Shift
Our audit revealed that the collapse of deep reconstruction models was caused by dataset collection artifacts: seven channels in the raw SWaT dataset (`MV101`, `AIT201`, `MV201`, `P201`, `P202`, `P204`, `MV303`) were unrecorded or constant in normal training data, but active in test data. Most severely, chemical analyzer `AIT201` had zero training variance ($\sigma = 0$), but exploded to standardized $z > 1,600$ during testing.
- Uniform residual aggregation models summed this error across the plant, forcing anomaly scores permanently above threshold.
- Isolation Forest was unaffected because tree axis cuts isolate univariate outliers without corrupting the joint partition space.
- R-STNet's SRM successfully detected `AIT201`'s non-physical behavior, downweighting its reliability to $r < 0.005$ and preventing false emergency alarms.

### 2.3 Robustness Under Controlled Sensor Corruption
Across multi-seed evaluations ($[42, 101, 2024]$) spanning Gaussian noise ($\sigma \in [0.1, 0.5]$) and linear drift ($\delta \in [0.1, 0.5]$):
- R-STNet maintained high detection stability ($F_1 \approx 0.276$ under noise; $F_1 \approx 0.273$ under drift).
- The SRM established a statistically significant reliability gap of **$\Delta r = +0.2856$ to $+0.2946$** between clean and corrupted channels.
- Against synthetic corruption masks, the SRM achieved **$100.0\%$ sensor fault detection recall** with **$0.00\%$ false alarms on clean channels**.

### 2.4 Ablation Proof of Multi-Perspective Synergy
In a 5-condition ablation suite, single perspectives (temporal velocity alone, graph consistency alone, or residual confidence alone) produced minimal benefit ($FPR > 92\%$). Only the full multi-perspective fusion combined rapid outlier dampening with temporal smoothing and topological validation, driving FPR down to **$62.14\%$**.

### 2.5 Computational Feasibility
- **Parameter Efficiency:** R-STNet requires only **$23,233$ parameters** ($86.4\%$ smaller than ST-AE).
- **Inference Latency:** Per-window inference latency was **$0.060\text{ ms}$** ($16,667$ windows/s), operating over **$160,000\times$ faster** than the $10$-second SWaT sampling interval.
- **VRAM Budget:** Peak allocated GPU VRAM was **$978.69\text{ MB}$**, consuming only $27.4\%$ of a 4 GB RTX 3050 mobile GPU.

---

## 3. Honest Scientific Positioning and What We DO NOT Claim

To maintain the highest level of scientific rigor for Review 2, our findings are framed with disciplined academic honesty:

1. **What We DO NOT Claim:**
   - We do **NOT** claim that R-STNet is state-of-the-art or that it outperforms all baselines. Isolation Forest remains significantly stronger on clean SWaT ($F_1 = 0.7182$ vs. $0.2740$).
   - We do **NOT** claim that the SRM eliminates all false alarms. R-STNet retains a $62.14\%$ FPR due to spatial graph attention message passing propagating residual contamination upstream of the decision layer.
   - We do **NOT** claim that statistical reliability modules can detect all sensor faults. Stuck-sensor experiments on inactive backup pumps revealed $0.0\%$ fault recall because natural variance was zero.
   - We do **NOT** claim real-time industrial production readiness. Only in-memory GPU latency was measured, not live SCADA network latency.

2. **What We DO Claim (The Defensible Contribution):**
   - Explicit multi-perspective sensor reliability estimation provides a proven, statistically verified mechanism to mitigate catastrophic false-alarm collapse *within deep spatiotemporal neural architectures*.
   - The SRM eliminates $14,898$ false alarms relative to Baseline C, achieves an F1 gain of $+26.2\%$, establishes $+0.29$ reliability separation, and localizes corrupted sensors with $100\%$ recall under controlled noise and drift.

---

## 4. Deliverable Directory Structure

All artifacts, figures, tables, and reports are fully self-contained in `report/REVIEW2/`:

```text
report/REVIEW2/
├── CHAPTER4_RESULTS_AND_DISCUSSION.md   <-- Master Review 2 Chapter (Sections 4.1 to 4.13)
├── REVIEW2_MASTER.md                    <-- 17-Part Comprehensive Master Dossier
├── EXECUTIVE_SUMMARY.md                 <-- This Executive Overview
├── EXPERIMENTAL_SETUP.md                <-- Complete Dataset, Model & Threshold Protocols
├── BASELINE_COMPARISON.md               <-- Detailed Clean Benchmark Analysis
├── ROBUSTNESS_ANALYSIS.md               <-- Multi-Seed Noise, Stuck & Drift Evaluations
├── SENSOR_RELIABILITY_ANALYSIS.md       <-- Empirical Reliability Decoupling Analysis
├── ABLATION_ANALYSIS.md                 <-- 5-Condition SRM Perspective Ablation
├── FAILURE_CASE_ANALYSIS.md             <-- 5 Qualitative Diagnostic Scenarios
├── COMPUTATIONAL_ANALYSIS.md            <-- Latency, Throughput & Memory Benchmarks
├── SWAT_CHANNEL_SHIFT_ANALYSIS.md       <-- In-Depth 7-Channel Distribution Shift Audit
├── LIMITATIONS.md                       <-- 7 Fundamental Limitations & Prohibited Claims
├── REPRODUCIBILITY.md                   <-- Complete Zero-Leakage Replication Guide
├── DATA_LEAKAGE_AUDIT.md                <-- Verification of Zero Test Leakage
├── THRESHOLD_AUDIT.md                   <-- Audit of Validation Quantile Protocol
├── CLAIM_AUDIT.md                       <-- Disciplined Scientific Claim Boundaries
├── TABLES.md                            <-- All 11 Clean Markdown Tables
├── TABLE_CAPTIONS.md                    <-- Exhaustive Academic Captions for Tables 4.1-4.11
├── FIGURE_CAPTIONS.md                   <-- Exhaustive Academic Captions for Figs 4.1-4.12
├── REPORT_CHECKLIST.md                  <-- 100% Quality & Verification Checklist
├── RAW_RESULTS_INDEX.md                 <-- Artifact Provenance Manifest
│
├── figures/                             <-- All 12 High-Resolution 300 DPI Publication Plots
├── metrics/                             <-- Authoritative JSON Metric Results
├── source_results/                      <-- Corruption & Ablation JSON Artifacts
└── logs/                                <-- Benchmark and Diagnostic Logs
```

---

*Certified Ready for Review 2 Evaluation.*
