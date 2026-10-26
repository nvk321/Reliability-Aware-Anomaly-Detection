# Chapter 2: Literature Review

**Course:** Machine Learning Lab (BCSE209P)  
**Slot:** L47 + L48 | Fall Semester 2026-27  
**Project Title:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Authors:** P Naga Venkata Kushal (23BDS0133), Koteru Tarun (23BDS0103)  
**Guide:** Dr. Dega Nagaraju  
**Institution:** School of Computer Science and Engineering (SCOPE), VIT Vellore  

---

## 2.1 Introduction

Multivariate time-series anomaly detection (MTSAD) in complex engineering and cyber-physical systems (CPS) has attracted substantial research over the past three decades [3], [16]. The challenge of reliably detecting subtle, safety-critical deviations while remaining robust to inevitable measurement imperfections lies at the intersection of applied machine learning, statistical process monitoring, graph representation learning, and control engineering [6], [19], [33].

To provide a rigorous, thematic synthesis of the state-of-the-art—rather than a disconnected chronological summary—this literature review is organized around **five major thematic pillars**:
1. **Classical and Statistical Anomaly Detection (Sections 2.2–2.3):** Distance, density, linear projection, and tree-based partition methods.
2. **Machine Learning and Deep Learning for Multivariate Sequences (Sections 2.4–2.6):** Reconstruction, autoregressive forecasting, and stochastic latent architectures.
3. **Spatial, Graph, and Causal Multivariate Modeling (Sections 2.7–2.8):** Graph neural networks, graph attention, topological message passing, and causal structural discovery.
4. **Sensor Faults, Sensor Reliability, and Robustness (Sections 2.9–2.11):** Fault detection and diagnosis (FDD), analytical redundancy, sensor data validation, and noise-tolerant representations.
5. **Industrial CPS Benchmarks and Evaluation Protocols (Sections 2.12–2.14):** Testbed architectures (SWaT, WADI), cyber-physical attack profiles, and critical evaluation methodology pitfalls.

The chapter culminates in a comprehensive comparative analysis (Section 2.15), the formal synthesis of the research gap (Section 2.16), and a summary connecting these findings to our proposed R-STNet architecture (Section 2.17).

---

## 2.2 Classical Anomaly Detection

### 2.2.1 Underlying Problem & Paradigms
Classical anomaly detection foundations were established on cross-sectional tabular data, where observations are treated as static, independent, and identically distributed (i.i.d.) vectors [3], [8]. These techniques generally classify into statistical, distance-based, and density-based formulations:
- **Statistical Methods:** Assume normal data follows a parametric probability distribution (e.g., Gaussian, Student's $t$, or Gaussian Mixture Models). Points falling into low-probability density tails ($p(\mathbf{x}) < \alpha$) are flagged as outliers [8].
- **Distance-Based Methods:** Operate under the assumption that normal data instances reside in dense clusters, whereas anomalies are located far from their $k$-nearest neighbors. Global distance metrics (e.g., Euclidean or Mahalanobis distance) quantify isolation.
- **Density-Based Methods:** Acknowledge that real-world datasets exhibit clusters of varying densities. The seminal **Local Outlier Factor (LOF)** algorithm developed by Breunig et al. (2000) [4] compares the local density of an object to the local densities of its $k$-nearest neighbors. Points with substantially lower local reachability densities are identified as local outliers.

### 2.2.2 Strengths and Limitations for Multivariate Sensor Streams
- **Strengths:** Non-parametric methods like LOF [4] do not require rigid distributional assumptions and provide intuitive geometric interpretations for tabular data.
- **Limitations:** Classical distance and density methods suffer catastrophically from the **curse of dimensionality** when applied to multivariate sensor streams ($M \ge 50$). In high-dimensional spaces, distance metrics concentrate, rendering pairwise Euclidean distances equidistant and uninformative [3]. Crucially, static classical methods discard sequential temporal ordering, rendering them incapable of detecting contextual and collective time-series anomalies where individual readings fall within nominal ranges but violate temporal dynamics [16].

---

## 2.3 Machine Learning-Based Anomaly Detection

### 2.3.1 Tree-Based and Kernel Space Partitions
To scale beyond distance calculations, machine learning introduced tree-partitioning and margin-based boundaries:
- **Isolation Forest:** Developed by Liu, Ting, and Zhou (2008) [1], Isolation Forest isolates anomalies rather than profiling normal points. It constructs an ensemble of randomized binary trees (iTrees). Because anomalous points are few and possess attribute values divergent from the norm, they are isolated closer to the root of the trees, resulting in noticeably shorter average path lengths $h(\mathbf{x})$. The anomaly score is defined as:
  $$s(\mathbf{x}, n) = 2^{-\frac{\mathbb{E}(h(\mathbf{x}))}{c(n)}}$$
  where $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree of $n$ instances.
- **One-Class Support Vector Machines (OC-SVM):** Formulated by Schölkopf et al. (1999) [2], OC-SVM maps training vectors into a high-dimensional reproducing kernel Hilbert space (RKHS) via non-linear kernels (e.g., Radial Basis Function) and finds a maximum-margin hyperplane that separates the normal training data from the coordinate origin.

### 2.3.2 Multivariate Statistical Process Control (MSPC)
In industrial and chemical process engineering, statistical projection methods have long dominated automated monitoring [6], [7]:
- **Principal Component Analysis (PCA):** Projects correlated sensor variables onto an orthogonal low-dimensional subspace capturing dominant variance. Residual deviations in the orthogonal complement subspace are monitored via the Squared Prediction Error (SPE or $Q$-statistic), while in-subspace shifts are monitored via Hotelling's $T^2$ [6], [36].
- **Dynamic PCA (DPCA):** To capture serial correlation, Wang & He (2020) [5] stack time-lagged observation vectors into augmented Hankel matrices before computing principal components, enabling detection of dynamic process shifts in complex loops like the Tennessee Eastman Process (TEP).

### 2.3.3 Limitations in Cyber-Physical Sensor Systems
While Isolation Forest [1] and DPCA [5] remain widely used industrial baselines:
1. They cannot capture complex non-linear spatio-temporal interactions across dozens of continuous stages [16].
2. A single noisy or drifted sensor causes severe **smearing effects** across projection matrices [6]: a fault on sensor $k$ contaminates the SPE contributions across multiple clean channels, making fault isolation difficult.
3. Tree-based and linear models cannot natively learn continuous sensor reliability scores online.

---

## 2.4 Deep Learning-Based Multivariate Time-Series Anomaly Detection

The advent of deep neural networks enabled end-to-end representation learning directly from continuous time-series without requiring manual feature engineering [16], [17]. Deep MTSAD architectures process multivariate sliding windows $\mathbf{W}_t \in \mathbb{R}^{L \times M}$ and are broadly classified into **reconstruction-based**, **prediction-based**, and **hybrid frameworks** [17].

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DEEP TIME-SERIES ANOMALY DETECTOR PARADIGMS                     │
├───────────────────────────────┬───────────────────────────────┬────────────────────────┤
│     RECONSTRUCTION MODELS     │       PREDICTION MODELS       │    GRAPH & CAUSAL      │
├───────────────────────────────┼───────────────────────────────┼────────────────────────┤
│ • Autoencoders (AE / VAE)     │ • Temporal Convolutions (TCN) │ • GDN (Graph Dev Net)  │
│ • LSTM-AE (Malhotra et al.)   │ • DeepAnT (Munir et al.)      │ • MTAD-GAT (Zhao et al)│
│ • USAD (Audibert et al.)      │ • TranAD (Tuli et al.)        │ • CAROTS (Kim et al.)  │
│ • OmniAnomaly (Su et al.)     │ • Anomaly Transformer (Xu)    │ • RANSynCoders (Ens)   │
└───────────────────────────────┴───────────────────────────────┴────────────────────────┘
```

---

## 2.5 Reconstruction-Based Methods

### 2.5.1 Underlying Formulation
Reconstruction models operate on the assumption that an encoder network $f_\theta(\cdot)$ can compress normal multivariate windows into a low-dimensional bottleneck representation $\mathbf{z}$, and a decoder network $g_\phi(\cdot)$ can reconstruct the original input $\tilde{\mathbf{X}} = g_\phi(f_\theta(\mathbf{X}))$ [14], [16]. Because the network is trained exclusively on normal patterns, it fails to reconstruct unseen anomalous patterns, yielding elevated reconstruction errors:

$$S_{\text{recon}}(t) = \|\mathbf{X}(t) - \tilde{\mathbf{X}}(t)\|_2^2$$

### 2.5.2 Seminal Architectures Audited

#### 1. USAD (UnSupervised Anomaly Detection)
- **Reference:** Audibert, Michiardi, Guyard, Marti, and Zuluaga (ACM KDD 2020) [9].
- **Architecture:** Employs an autoencoder architecture with an adversarial training policy. A single encoder $E$ feeds two independent decoders, $D_1$ and $D_2$. During phase one, $E$ and $D_1$ are trained to reconstruct normal windows; during phase two, $D_2$ is trained adversarially to discriminate between the real input and the reconstruction from $D_1$.
- **Anomaly Score:** Weighted combination: $S(t) = \alpha \|\mathbf{W}_t - D_1(E(\mathbf{W}_t))\|_2 + \beta \|\mathbf{W}_t - D_2(E(D_1(E(\mathbf{W}_t))))\|_2$.
- **Strengths:** Extremely fast training; adversarial objective amplifies anomalous reconstruction errors.
- **Limitations:** Flattens temporal windows into 1D vectors, discarding sequential structure. Crucially, **adversarial error amplification exacerbates sensor noise**: when an isolated sensor experiences high Gaussian noise or a spike, the adversarial decoders aggressively amplify the error, guaranteeing a massive false plant emergency shutdown.

#### 2. OmniAnomaly (Stochastic Recurrent Networks)
- **Reference:** Su, Zhao, Niu, Liu, Sun, and Pei (ACM KDD 2019) [10].
- **Architecture:** Combines Gated Recurrent Units (GRU) with a stochastic variational latent space augmented by planar normalizing flows. Captures temporal dependencies via GRU cells and models complex multimodal normal data distributions through non-Gaussian stochastic latent variables $\mathbf{z}_t \sim q_\phi(\mathbf{z}_t | \mathbf{x}_{\le t})$.
- **Strengths:** Robust probabilistic modeling of normal operational noise; effective on server telemetry benchmarks (SMD, SMAP).
- **Limitations:** Ignores inter-sensor physical topologies. Suffers from **latent space contamination**: because all $M$ sensor channels are fed into dense recurrent encoder layers, an extreme reading on a single corrupted sensor contaminates the latent state $\mathbf{z}_t$, degrading the reconstruction of all remaining clean channels.

#### 3. MSCRED (Multi-Scale Convolutional Recurrent Encoder-Decoder)
- **Reference:** Zhang, Song, Chen, Shen, and Mount (AAAI 2019) [13].
- **Architecture:** Constructs multi-scale cross-correlation signature matrices over various lookback horizons to capture spatial correlations, encoding them via 2D convolutional autoencoders coupled with ConvLSTM networks.
- **Strengths:** Retains spatial cross-correlation matrices across multiple temporal scales.
- **Limitations:** Computing continuous signature matrices across 50+ channels incurs prohibitive computational overhead; sums signature reconstruction errors uniformly without channel reliability weighting.

---

## 2.6 Prediction-Based Methods

### 2.6.1 Underlying Formulation
Prediction-based (forecasting) methods train autoregressive deep models to forecast the expected one-step-ahead sensor values $\hat{\mathbf{X}}(t) = f_\theta(\mathbf{W}_{t-1})$. The anomaly score is computed from the prediction residual:

$$S_{\text{pred}}(t) = \|\mathbf{X}(t) - \hat{\mathbf{X}}(t)\|_2^2$$

### 2.6.2 Landmark Models Audited

#### 1. TranAD (Deep Transformer Networks for Time-Series Anomaly Detection)
- **Reference:** Tuli, Casale, and Jennings (VLDB 2022) [11].
- **Architecture:** Leverages multi-head self-attention mechanisms to model long-range temporal dependencies, using an adversarial two-phase training procedure with focus scores to stabilize attention maps.
- **Strengths:** Captures subtle contextual temporal shifts; achieves strong benchmark F1-scores.
- **Limitations:** Quadratic computational and memory complexity $\mathcal{O}(L^2)$ creates severe VRAM bottlenecks on local hardware. Treats all feature residual deviations equally during anomaly scoring, leaving system alerts vulnerable to corrupted channels.

#### 2. Anomaly Transformer (Association Discrepancy)
- **Reference:** Xu, Wu, Wang, and Long (ICLR 2022) [12].
- **Architecture:** Introduces the concept of **association discrepancy**. Combines classic self-attention (series association) with a Gaussian-kernel prior association. In normal periods, series and prior associations match; during anomalies, anomalous points struggle to build associations with distant normal points, creating a large discrepancy.
- **Strengths:** Principled mathematical formulation capturing temporal association breakdowns.
- **Limitations:** Evaluates association discrepancy globally across all channels; lacks channel-specific sensor reliability estimation.

---

## 2.7 Graph and Spatial-Temporal Methods

### 2.7.1 The Need for Graph-Structured Modeling
In industrial cyber-physical systems, sensors are deployed across physical piping networks, chemical reactors, and electrical buses [38], [39]. Treating sensors as independent channels (univariate modeling) or passing them through unconstrained dense latent layers discards the **physical topological network** that governs the plant.

Graph Neural Networks (GNNs) represent sensor channels as nodes $\mathcal{V} = \{v_1, \dots, v_M\}$ and physical or statistical relationships as edges $\mathcal{E}$. This allows graph convolutional message passing to enforce physical conservation principles [19], [25].

### 2.7.2 Landmark Graph Architectures Audited

#### 1. Graph Deviation Network (GDN)
- **Reference:** Deng and Hooi (AAAI / IJCAI 2021) [19].
- **Architecture:** Learns a directed graph structure dynamically using node embeddings $\mathbf{v}_i \in \mathbb{R}^d$. Evaluates pairwise similarity $e_{ji} = \frac{\mathbf{v}_i^T \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}$, applies Top-$K$ sparsification to retain the strongest directed edges, and computes edge weights via Softmax. A graph attention network aggregates messages from topological neighbors, forecasting the next expected reading for each sensor.
- **Anomaly Score:** Standardizes prediction residuals: $a_i(t) = \frac{|x_i(t) - \hat{x}_i(t)|}{\text{Median}_i + \text{MAD}_i}$, and takes the maximum: $S(t) = \max_i a_i(t)$.
- **Relevance to Our Work:** GDN provides the foundational inspiration for our learned graph dependency module.
- **Critical Limitation Addressed by R-STNet:** By taking the maximum standardized residual $\max_i a_i(t)$ or summing errors uniformly, **GDN is completely unprotected against single-sensor corruption**. If sensor $k$ develops calibration drift, spikes, or high noise, $a_k(t)$ spikes, instantly triggering a false plant alarm. GDN lacks any Sensor Reliability Module to evaluate whether sensor $k$ is trustworthy.

#### 2. MTAD-GAT (Multivariate Time-Series Anomaly Detection via GAT)
- **Reference:** Zhao, Wang, Duan, Huang, Cao, Tong, and Xu (IEEE ICDM 2020) [20].
- **Architecture:** Employs two parallel Graph Attention Networks: a **Feature-Oriented GAT** (capturing spatial dependencies among sensors) and a **Time-Oriented GAT** (capturing temporal correlations across sequence steps within the window). Combines joint forecasting and reconstruction heads.
- **Limitations:** Treats all feature errors equally during anomaly aggregation; lacks an explicit sensor trustworthiness metric.

#### 3. Recent Elsevier Advancements in Spatio-Temporal GNNs
- **Chen et al. (2022) [22] (*Neurocomputing*):** Deep variational graph convolutional recurrent networks for MTSAD.
- **Han & Lee (2023) [23] (*Expert Systems with Applications*):** Multi-scale spatio-temporal graph convolutional networks capturing multi-frequency oscillations.
- **Li, Zhang, and Zhao (2023) [24] (*Computers in Industry*):** Spatio-temporal graph neural networks applied specifically to industrial sensor fault diagnosis.

---

## 2.8 Causal and Contrastive Learning Methods

### 2.8.1 CAROTS: Causality-Aware Contrastive Learning
- **Reference:** Kim, Mok, Lee, Lew, Kim, and Yoon (ICML 2025 / arXiv:2506.03964) [21].
- **Theoretical Mechanism:** Real-world multivariate sensor data is subject to domain shifts and external operational variations. CAROTS integrates causal structural equation modeling (DAG discovery) with contrastive representation learning. It constructs augmented temporal views that preserve underlying causal physical topologies while perturbing non-causal spurious correlations.
- **Role in Our Project:** CAROTS is maintained as a **read-only external state-of-the-art research reference and benchmark** in `reference/CAROTS-master/` (in accordance with ADR-006 [Technical References]).
- **Conceptual Differentiation:**  
  CAROTS operates at the **latent contrastive representation invariance level**. It does **not** estimate an explicit per-sensor reliability vector $\mathbf{r}(t) \in [0, 1]^M$, does not localize which specific sensor transmitter is degraded, and does not provide an operational 4-way decision framework. Our proposed R-STNet focuses specifically on explicit, channel-level reliability estimation and false-alarm suppression.

---

## 2.9 Sensor Fault Detection and Diagnosis (FDI)

### 2.9.1 Control Engineering Foundations
While computer science literature has traditionally focused on anomaly detection, the control systems and reliability engineering literature has spent decades studying **Fault Detection and Isolation (FDI)** [33], [34], [35]:
- **Analytical Redundancy:** Formulated by Gertler (1998) [33] and Isermann (2006) [34], analytical redundancy generates parity equations using physical conservation laws. If three sensors monitor water level, inflow, and outflow of a tank, the mass balance $\frac{dV}{dt} = Q_{\text{in}} - Q_{\text{out}}$ provides an independent, redundant estimate of the level sensor.
- **Process History-Based Methods:** Comprehensive surveys by Venkatasubramanian et al. (2003) [36] and Qin (2012) [6] review qualitative, quantitative, and statistical methods for isolating sensor faults in chemical processing plants.
- **Nonlinear Observers:** Sharifi & Langari (2014) [29] (*Control Engineering Practice*) demonstrated non-linear observer-based sensor fault diagnosis specifically on water treatment testbeds.

### 2.9.2 The Scalability Bottleneck of Classical FDI
Classical model-based FDI requires **exact mathematical differential equations** of the entire physical plant (differential equations governing every valve, pump curve, friction factor, and chemical reaction rate) [33], [34]. For modern large-scale facilities (e.g., SWaT with 51 channels, WADI with 123 channels), hand-crafting exact physical ODEs is economically and mathematically intractable.

**Our Proposed Solution:** R-STNet **learns analytical redundancy directly from normal operational data** using data-driven spatio-temporal representations and graph attention, eliminating the need for manual physics modeling.

---

## 2.10 Sensor Reliability and Data Quality

Recent literature in premier Elsevier journals has recognized the necessity of quantifying sensor data reliability and validation:
- **Zhu & Fei (2022) [32] (*ISA Transactions*):** Proposed multi-sensor data validation and reconstruction in industrial CPS using deep autoencoders with confidence estimation. While establishing the validity of confidence metrics, their approach lacks explicit graph neighbor consensus and 4-way operational state logic.
- **Huang, Yan, and Liu (2021) [30] (*Measurement*):** Combined spatial-temporal correlation with deep learning for sensor fault diagnosis and data reconstruction, demonstrating that physical correlation matrices can reconstruct missing or corrupted sensor readings.
- **Liu & Chen (2022) [31] (*Journal of Process Control*):** Developed dynamic graph attention networks for robust process monitoring and sensor fault isolation, confirming that attention weights reflect physical topological coupling.
- **Saeed, Jan, Lee, and Koo (2021) [28] (*Reliability Engineering & System Safety*):** Investigated extremely randomized trees for classifying wireless sensor network faults into bias, drift, performance degradation, and noise.

---

## 2.11 Robustness to Noise and Sensor Corruption

### 2.11.1 RANSynCoders: Random Sampling Autoencoders
- **Reference:** Audibert, Michiardi, Guyard, and Zuluaga (IEEE TKDE 2022) [27].
- **Mechanism:** Inspired by the RANSAC algorithm in computer vision, RANSynCoders trains an ensemble of autoencoders on random subsets of sensor channels. At inference time, it evaluates consensus across the subset reconstructions: if an anomaly is present across all subsets, it represents a genuine system anomaly; if it affects only subsets containing a specific sensor, it is treated as sensor noise.
- **Strengths:** Highly robust against noisy and corrupted channels.
- **Limitations:** Training and executing dozens of random subset autoencoders incurs severe computational overhead. It relies on random brute-force subset selection rather than learning the physical graph topology, and does not output an interpretable, continuous reliability score $r_i(t) \in [0, 1]$.

---

## 2.12 Industrial / Cyber-Physical System Anomaly Detection

Industrial Control Systems (ICS) and SCADA networks present unique operational challenges [43], [44]:
- **High Consequence of False Alarms:** Emergency plant shutdowns risk physical equipment damage, water contamination, and massive restart expenses [38].
- **Stealthy Cyber-Physical Attacks:** Modern cyber-attacks (e.g., Stuxnet-style physical attacks) manipulate physical processes within nominal static threshold bounds, making them invisible to univariate threshold detectors [43].
- **Lightweight Requirements:** Research by Kravchik & Shabtai (2022) [43] and Al-Abassi et al. (2020) [44] emphasizes the necessity of lightweight convolutional and recurrent architectures capable of executing on local industrial edge hardware.

---

## 2.13 Benchmark Datasets

Mainstream multivariate anomaly detection literature evaluates models across several standard public benchmarks:

| Benchmark Dataset | Domain | Channels ($M$) | Duration | Attacks / Anomalies | Physical Conservation Laws Present? |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **SWaT (Secure Water Treatment) [38], [45]** | Industrial Water Purification | **51** | 11 Days (7 Normal, 4 Attack) | 36 Cyber-Physical Attacks (12.14% density) | **YES (Mass, energy, hydraulic, chemical)** |
| **WADI (Water Distribution) [39]** | Water Distribution Loop | **123** | 16 Days (14 Normal, 2 Attack) | 15 Complex Network Attacks (5.99% density) | **YES (Hydraulic network loops)** |
| **SMD (Server Machine Dataset) [10]** | Internet Server Clusters | 38 | 28 Days | Server telemetry anomalies | No (IT performance metrics) |
| **PSM (Pooled Server Metrics)** | eBay Server Nodes | 26 | 2 Months | Hardware / network spikes | No (IT performance metrics) |
| **SMAP / MSL (NASA Spacecraft)** | Spacecraft Telemetry | 25 / 55 | Variable | Telemetry channel anomalies | Partial (Power / thermal) |

**Dataset Selection Justification:**  
This project prioritizes **SWaT (Primary)** and **WADI (Secondary)** because they represent real-world physical infrastructure governed by strict conservation laws (mass, momentum, and chemistry), making them the ideal testbeds for evaluating reliability-aware physical consensus.

---

## 2.14 Evaluation Challenges in Time-Series Anomaly Detection

Recent landmark papers in premier machine learning venues have uncovered severe methodological and evaluation flaws in published MTSAD literature:

### 2.14.1 The Point-Adjustment (PA) Distortion
- **The Protocol:** In the Point-Adjustment (PA) evaluation heuristic popularized by OmniAnomaly [10] and USAD [9], if an anomaly detection model flags even a single time step within a contiguous ground-truth anomaly segment, the entire segment is retroactively marked as **True Positives (TP)**.
- **The Critical Critique:** Kim, Choi, Choi, Lee, and Yoon (AAAI 2022) [40] (*"Towards a Rigorous Evaluation of Time-Series Anomaly Detection"*) and Wu & Keogh (IEEE TKDE 2021) [41] (*"Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress"*) mathematically proved that **PA dramatically inflates F1-scores**. Kim et al. [40] showed that a completely random noise generator or an inverted detector can achieve near-perfect F1-scores ($\ge 0.90$) under the PA protocol!
- **Our Policy:** R-STNet enforces **raw point-wise Precision, Recall, F1, and PR-AUC** as primary ground-truth metrics. Point-adjusted metrics are computed strictly as a labeled secondary reference (`PA-F1`).

### 2.14.2 Threshold Snooping / Test Set Tuning
Wu & Keogh [41] and Garg et al. (NeurIPS 2021) [42] highlighted that many published papers select decision thresholds $\tau$ by exhaustively searching the test split to maximize test F1-score—a direct violation of scientific machine learning.
- **Our Policy:** Decision thresholds ($\tau_{\text{sys}}, \theta_{\text{rel}}$) are calibrated strictly on the clean normal validation split and **frozen** prior to blind test evaluation.

---

## 2.15 Comparative Analysis

As synthesized in the Literature Matrix (`report/LITERATURE_MATRIX.md`), existing paradigms each solve isolated aspects of the problem:
- GDN [19] and MTAD-GAT [20] learn graph topologies, but sum residuals uniformly without reliability weighting.
- CAROTS [21] achieves causal contrastive invariance, but does not output channel-level reliability scores or separate sensor faults from attacks.
- RANSynCoders [27] provides noise robustness through random subsets, but incurs heavy computational costs and ignores physical graph topology.
- Classical FDI [33], [34] isolates sensor faults, but requires unattainable manual mathematical differential equations.

---

## 2.16 Research Gap

Synthesizing all five literature themes reveals the precise research gap:

> **The Formal Research Gap:**  
> Existing literature provides robust foundations for multivariate anomaly detection, graph neural networks, and sensor fault analysis; however, there is an absence of an **integrated, lightweight framework that explicitly estimates continuous, channel-level sensor reliability online, dynamically down-weights untrusted sensor observations during system anomaly scoring, and operationalizes a multi-state decision engine that decouples isolated sensor corruptions from plant-wide process anomalies under a controlled, reproducible evaluation protocol.**

---

## 2.17 Summary

This literature review confirms that:
1. Standard multivariate anomaly detectors conflate observation corruption with process anomalies due to uniform unweighted error aggregation.
2. An isolated degraded sensor triggers costly, unwarranted plant emergency shutdowns.
3. Estimating an explicit, continuous trustworthiness score $r_i(t) \in [0, 1]$ via physical graph consensus, temporal continuity, and residual statistics provides a mathematically sound, physically grounded solution.
4. The proposed **R-STNet** architecture directly operationalizes these insights into an integrated, defensible research framework.
