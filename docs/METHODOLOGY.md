# Methodology: The R-STNet Framework

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  
**Faculty Guide:** Dr. Dega Nagaraju  
**Status:** Validated Mathematical Formulation  

---

## 1. Mathematical Problem Formulation

Let $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T] \in \mathbb{R}^{T \times M}$ denote a multivariate time series recording telemetry from $M$ physical sensors and actuators across $T$ discrete time intervals. At each time step $t$, the system observation is a vector $\mathbf{x}_t = [x_{1, t}, x_{2, t}, \dots, x_{M, t}]^\top \in \mathbb{R}^M$.

A temporal sliding window of length $L$ at time $t$ is defined as:
$$\mathbf{X}_t = [\mathbf{x}_{t-L+1}, \mathbf{x}_{t-L+2}, \dots, \mathbf{x}_t] \in \mathbb{R}^{L \times M}$$

The goal of the detection system is to learn a mapping $f: \mathbb{R}^{L \times M} \to \{0, 1\}$ that assigns a binary label $y_t \in \{0, 1\}$ indicating whether the physical plant is undergoing an operational anomaly or cyber-physical attack ($y_t = 1$) or operating in a normal regime ($y_t = 0$), while remaining robust against localized, non-adversarial sensor degradation.

---

## 2. Spatio-Temporal Backbone Architecture

The core neural backbone of R-STNet consists of a **Causal Dilated Temporal Convolutional Network (TCN)** coupled with a **Multi-Head Spatial Graph Attention Network (GAT)**, totaling $23,233$ trainable parameters.

```
Input Window X_t in R^(B x L x M)
             |
             v
   1D Causal Dilated TCN (Dilations: 1, 2, 4)
             |
             v
    Temporal Feature Map H_temporal in R^(B x d x M)
             |
             v
   Spatial Multi-Head GAT (Learned Adjacency A)
             |
             v
    Spatial Feature Map H_spatial in R^(B x d x M)
             |
             v
   Linear Reconstruction Head
             |
             v
   Predicted Sensor State x_hat_t in R^(B x M)
```

### 2.1 Causal Dilated Temporal Convolution (TCN)
To extract multi-scale temporal dependencies without violating temporal causality, the network applies 1D dilated causal convolutions:
$$\mathbf{h}_i^{(l)}(t) = \sigma \left( \sum_{k=0}^{K-1} \mathbf{W}_k^{(l)} \mathbf{h}_i^{(l-1)}(t - d \cdot k) + \mathbf{b}^{(l)} \right)$$
where $K = 3$ is the kernel size, $d \in \{1, 2, 4\}$ is the dilation factor, and $\sigma(\cdot)$ is the LeakyReLU activation. Causal padding ensures that $\mathbf{h}_i(t)$ depends strictly on past observations $\{\mathbf{x}_\tau\}_{\tau \le t}$.

### 2.2 Multi-Head Spatial Graph Attention Network (GAT)
To capture physical topological correlations across the plant (e.g., flow from Stage 1 feeding tank levels in Stage 2), features are projected into an attention space where graph attention weights $\alpha_{ij}$ are learned dynamically:
$$\alpha_{ij} = \frac{\exp \left( \text{LeakyReLU} \left( \mathbf{a}^\top [\mathbf{W} \mathbf{h}_i \,\|\, \mathbf{W} \mathbf{h}_j] \right) \right)}{\sum_{k \in \mathcal{N}(i)} \exp \left( \text{LeakyReLU} \left( \mathbf{a}^\top [\mathbf{W} \mathbf{h}_i \,\|\, \mathbf{W} \mathbf{h}_k] \right) \right)}$$
The updated node representation aggregates information across physical topological neighbors $\mathcal{N}(i)$:
$$\mathbf{h}_i^{\text{spatial}} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j \right)$$

### 2.3 Linear Reconstruction Head
The final spatial latent representation is projected through a linear layer to generate the reconstructed sensor observation vector $\hat{\mathbf{x}}_t = [\hat{x}_{1, t}, \dots, \hat{x}_{M, t}]^\top \in \mathbb{R}^M$.

The backbone is trained using Mean Squared Error (MSE) on pure normal operational data:
$$\mathcal{L}_{\text{train}} = \frac{1}{B \cdot M} \sum_{b=1}^B \sum_{i=1}^M (x_{i, t}^{(b)} - \hat{x}_{i, t}^{(b)})^2$$

---

## 3. The Multi-Perspective Sensor Reliability Module (SRM)

Rather than assuming all sensor readings $x_{i, t}$ are equally trustworthy, the SRM computes an independent scalar reliability score $r_i(t) \in [0, 1]$ for each sensor $i$ at each time step $t$ by integrating three orthogonal perspectives:

$$r_i(t) = \frac{w_v \cdot r_i^{(v)}(t) + w_c \cdot r_i^{(c)}(t) + w_p \cdot r_i^{(p)}(t)}{w_v + w_c + w_p}$$

with calibrated perspective weights $w_v = 1.0$ (temporal), $w_c = 1.2$ (graph), and $w_p = 1.0$ (residual).

### 3.1 Perspective 1: Temporal Continuity ($r_i^{(v)}$)
Physical systems possess mechanical and hydraulic inertia; true physical states cannot change instantaneously. High-frequency electrical jitter or ADC bit errors manifest as non-physical rate-of-change spikes.

The temporal velocity is normalized against the training Median Absolute Deviation (MAD) of first-order differences:
$$v_{i, t} = \frac{|x_{i, t} - x_{i, t-1}|}{\text{MAD}(\Delta x_i) + \epsilon}$$
$$r_i^{(v)}(t) = \exp \left( - \frac{1}{2} v_{i, t}^2 \right)$$
When a high-frequency spike occurs, $v_{i, t} \gg 1$, driving $r_i^{(v)}(t) \to 0$.

### 3.2 Perspective 2: Graph Neighborhood Consistency ($r_i^{(c)}$)
Physical instruments are constrained by mass balance, energy conservation, and hydraulic continuity (e.g., flow meter `FIT101` and tank level `LIT101`). A single drifting or broken sensor will diverge from its physical neighborhood $\mathcal{N}(i)$.

The neighborhood discrepancy is calculated against the attention-weighted neighbor expectation:
$$c_{i, t} = \frac{|x_{i, t} - \sum_{j \in \mathcal{N}(i)} \alpha_{ij} x_{j, t}|}{\sigma_{\text{neigh}, i} + \epsilon}$$
$$r_i^{(c)}(t) = \exp \left( - \frac{1}{2} c_{i, t}^2 \right)$$
When an isolated sensor drifts independently while its physical counterparts remain steady, $c_{i, t} \gg 1$, driving $r_i^{(c)}(t) \to 0$.

### 3.3 Perspective 3: Predictive Residual Confidence ($r_i^{(p)}$)
The spatiotemporal backbone learns normal operational multi-sensor distributions. If a sensor value cannot be reconstructed by the deep model, it indicates a departure from normal multivariate correlations.

The prediction error is normalized into a robust z-score:
$$z_{i, t} = \frac{|x_{i, t} - \hat{x}_{i, t}|}{\text{MAD}(e_i) + \epsilon}$$
$$r_i^{(p)}(t) = \exp \left( - \frac{1}{2} z_{i, t}^2 \right)$$
When a sensor exhibits an extreme outlier or severe distribution shift ($z_{i, t} \gg 3$), the exponential penalty drives $r_i^{(p)}(t) \to 0$.

---

## 4. Reliability-Weighted System Anomaly Scoring

In Baseline C (`RSTNetNoSRM`), the system anomaly score is the unweighted mean normalized residual:
$$S_{\text{unweighted}}(t) = \frac{1}{M} \sum_{i=1}^M \frac{|x_{i, t} - \hat{x}_{i, t}|}{\text{MAD}(e_i) + \epsilon}$$

In Proposed R-STNet, the system anomaly score is computed as the **reliability-weighted normalized residual**:
$$S_{\text{sys}}(t) = \frac{\sum_{i=1}^M r_i(t) \cdot \left( \frac{|x_{i, t} - \hat{x}_{i, t}|}{\text{MAD}(e_i) + \epsilon} \right)}{\sum_{i=1}^M r_i(t)}$$

### Mathematical Mechanism of False Alarm Suppression:
- If sensor $k$ suffers from severe electrical noise ($\sigma = 0.5$) or an unrecorded channel shift ($z > 1,600$), its raw residual $e_k$ is massive ($e_k \gg 100$).
- Under unweighted scoring (Baseline C), $e_k$ contributes $\frac{1}{M} e_k \approx \frac{1}{51} \times 100 = 1.96$, forcing the plant score above threshold ($\tau_c = 12.07$) and causing false alarms.
- Under R-STNet, the SRM depresses $r_k(t) \to 0.015$. In the weighted numerator, $r_k \cdot e_k \approx 0.015 \times 100 = 1.5$, dampening its influence by over $95\%$ and keeping the plant-wide score safely below $\tau_{prop} = 3.14$.

---

## 5. Threshold Calibration & Detection Decision

The binary anomaly classification is made via a threshold comparison:
$$y_t = \mathbb{I}(S_{\text{sys}}(t) \ge \tau)$$

To guarantee zero test-set leakage:
1. Normal validation windows are passed through the model to obtain validation scores $\{S_{\text{val}}(t)\}_{t=1}^{N_{\text{val}}}$.
2. The decision threshold $\tau$ is set to the empirical $95^{\text{th}}$ percentile ($q = 0.95$):
   $$\tau = \text{Quantile}(S_{\text{val}}, 0.95)$$
3. Calibrated thresholds:
   - Baseline A (Isolation Forest): $\tau = 0.63587$
   - Baseline B (ST-AE): $\tau = 0.08922$
   - Baseline C (RSTNetNoSRM): $\tau = 12.07062$
   - Proposed R-STNet (with SRM): $\tau = 3.14008$
