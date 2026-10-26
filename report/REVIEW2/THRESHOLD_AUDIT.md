# DECISION THRESHOLD SELECTION AND CALIBRATION AUDIT

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Scope:** Verification of Anomaly Threshold Calibration Across All Models  
**Policy:** Unsupervised Calibration on Normal Validation Split (Zero Test Peeking)  
**Status:** **100% COMPLIANT — ZERO TEST LEAKAGE**

---

## 1. Summary of Model Thresholds and Calibration Data

| Model | Calibrated Threshold $\tau$ | Calibration Data Source | Calibration Method | Val Scores Mean ± Std | Val Scores Max | Test Leakage? |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: |
| **Baseline A: Isolation Forest** | **0.63587** | Pure Normal Validation Split (9,936 rows) | 95th Percentile ($\alpha=0.95$) | $0.2895 \pm 0.1858$ | $0.7412$ | **NO** |
| **Baseline B: ST-AE** | **0.08922** | Pure Normal Validation Split (9,907 windows) | 95th Percentile ($\alpha=0.95$) | $0.0363 \pm 0.1615$ | $0.8124$ | **NO** |
| **Baseline C: RSTNetNoSRM** | **12.07062** | Pure Normal Validation Split (9,907 windows) | 95th Percentile ($\alpha=0.95$) | $5.8794 \pm 6.1619$ | $73.6755$ | **NO** |
| **Proposed R-STNet (with SRM)** | **3.14008** | Pure Normal Validation Split (9,907 windows) | 95th Percentile ($\alpha=0.95$) | $2.3924 \pm 0.4061$ | $4.1614$ | **NO** |

---

## 2. Threshold Calibration Methodology

### 2.1 The 95th Percentile Normal Calibration Protocol
In industrial cyber-physical monitoring systems, operational anomaly thresholds cannot be calibrated against ground-truth attack labels because attacks are rare, uncataloged, and zero-day in nature. 

To maintain strict operational validity:
1. Every model is trained exclusively on normal operational observations ($y_t = 0$).
2. Anomaly scores are computed over the held-out pure normal validation split:
   $$\mathcal{S}_{\text{val}} = \{ S(t) \mid t \in \text{Validation Split} \}$$
3. The operational threshold $\tau$ is set to the 95th percentile ($\alpha = 0.95$) of $\mathcal{S}_{\text{val}}$:
   $$\tau = \text{Quantile}(\mathcal{S}_{\text{val}}, 0.95)$$
   This guarantees that during nominal operation, the theoretical false alarm rate on validation data is bounded at $5\%$.
4. The threshold $\tau$ is frozen permanently and applied to the test split without alteration:
   $$\hat{y}(t) = \mathbb{I}(S(t) \ge \tau)$$

### 2.2 Mathematical Rationale for Score Scale Differences
* **Isolation Forest:** Anomaly scores are bounded in $[0, 1]$ based on mean path length in isolation trees, yielding $\tau = 0.63587$.
* **ST-AE:** Uses squared reconstruction error $\|X_t - \hat{X}_t\|_2^2$, resulting in $\tau = 0.08922$.
* **Baseline C (RSTNetNoSRM):** Computes unweighted average of MAD-normalized residuals:
  $$S_{\text{sys}}(t) = \frac{1}{M} \sum_{i=1}^M z_i(t), \quad z_i(t) = \frac{|x_i(t) - \hat{x}_i(t)|}{\text{MAD}_i + \epsilon}$$
  Because 51 normalized residuals are summed and averaged, $\tau = 12.07062$.
* **Proposed R-STNet:** Computes reliability-weighted anomaly scores:
  $$S_{\text{sys}}(t) = \frac{\sum_{i=1}^M r_i(t) z_i(t)}{\sum_{i=1}^M r_i(t) + \epsilon}$$
  Because the denominator normalizes by $\sum r_i(t)$, the resulting scale represents the expected residual of trustworthy sensors, yielding $\tau = 3.14008$.

---

## 3. Strict Absence of Post-Hoc Test Threshold Tuning

* In many published MTSAD papers (e.g., OmniAnomaly, USAD, GDN), thresholds are frequently tuned post-hoc across the test set to maximize the Point-Adjusted F1 score (the so-called "best F1" search).
* **This project explicitly rejects test-threshold search.**
* Peak F1 optimization on test labels represents test set leakage.
* The thresholds reported herein were computed strictly a priori on normal validation data and evaluated once on the mixed test sequence.
