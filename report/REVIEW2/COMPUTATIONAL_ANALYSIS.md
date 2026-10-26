# Computational and Runtime Resource Analysis

**Project:** Reliability-Aware Anomaly Detection for Multivariate Sensor Data  
**Course:** BCSE209P — Machine Learning Lab  
**Date:** October 2026  
**Status:** Benchmarked Computational Performance  

---

## 1. Benchmarking Objectives

In critical infrastructure and industrial control systems (ICS), anomaly detection models must satisfy stringent real-time operational constraints. An architecture that achieves high mathematical accuracy but requires seconds of compute per observation is unsuitable for deployment in automated supervisory control and data acquisition (SCADA) environments.

This section evaluates the computational feasibility, parameter efficiency, training runtime, inference latency, and memory footprints of the four benchmarked architectures (Table 4.9).

---

## 2. Hardware and Environment Specifications

All computational benchmarks were executed on an edge-representative industrial workstation:
- **Graphics Processing Unit (GPU):** NVIDIA GeForce RTX 3050 Laptop GPU.
  - Dedicated Video RAM: $4,096\text{ MB}$ GDDR6.
  - CUDA Cores: $2,048$.
  - Compute Capability: $8.6$ (Ampere Architecture).
- **Host Central Processing Unit (CPU):** Intel Core i5 (12 virtual threads @ $2.50\text{ GHz}$).
- **Host System Memory (RAM):** $16.0\text{ GB}$ DDR4.
- **Operating System:** Microsoft Windows 11 Enterprise (Build 26100).
- **Execution Software:** PyTorch 2.6.0+cu124 / CUDA 12.4 / Python 3.11.

---

## 3. Computational Metrics Compendium

The table below summarizes the audited measurements recorded across the experimental pipeline:

| Model Architecture | Total Trainable Parameters | Parameter Efficiency vs. ST-AE | Training Wall Clock (s) | Training Time per Epoch (s) | Per-Window Inference Latency (ms) | Peak Throughput (Windows / s) | Peak Allocated VRAM (MB) | Peak Reserved VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A: Isolation Forest** | Non-parametric | N/A | **12.40** | N/A | **0.021** | **47,619** | N/A (Host RAM) | N/A (Host RAM) |
| **Baseline B: ST-AE** | $170,643$ | Baseline ($1.0\times$) | $112.35$ | $7.49$ | $0.076$ | $13,158$ | **108.49** | **198.00** |
| **Baseline C: RSTNetNoSRM** | $\mathbf{23,233}$ | **86.4% reduction** | $112.89$ | $7.53$ | $0.050$ | $20,000$ | $978.69$ | $1,124.00$ |
| **Proposed: R-STNet (with SRM)** | $\mathbf{23,233}$ | **86.4% reduction** | $134.73$ | $8.98$ | $0.060$ | $\mathbf{16,667}$ | $978.69$ | $1,124.00$ |

---

## 4. In-Depth Architectural Resource Analysis

### 4.1 Parameter Efficiency: 86.4% Compactness
- **Baseline B (ST-AE)** relies on fully connected dense layers to reconstruct the $M = 51$ sensors, requiring **$170,643$ trainable parameters**.
- **Proposed R-STNet** and **Baseline C** leverage weight sharing across spatial graph attention (GAT) and dilated temporal convolutions (TCN), requiring only **$23,233$ parameters**—an **$86.4\%$ reduction in model size**.
- Despite possessing less than one-seventh the parameter count of ST-AE, R-STNet achieves significantly superior representation fidelity (final training MSE of $0.0092$ vs. $0.0406$, Figure 4.4).

### 4.2 Training Wall-Clock Time
- Isolation Forest trains in just **$12.40$ seconds** on the CPU due to fast sub-sampled tree building ($n=256$).
- Deep graph networks require iterative gradient descent over 15 epochs on $39,715$ training windows:
  - Baseline C completes in **$112.89$ seconds** ($7.53$ s/epoch).
  - Proposed R-STNet completes in **$134.73$ seconds** ($8.98$ s/epoch).
- The inclusion of the SRM adds only **$21.84$ seconds of total training overhead** ($1.45$ s/epoch), which corresponds to the auxiliary calibration of the median absolute deviation (MAD) statistics and neighborhood consistency baselines on the validation set.

### 4.3 Real-Time Streaming Feasibility
In industrial applications, telemetry data arrives sequentially. In the downsampled SWaT pipeline:
- **Telemetry Sampling Step:** $\Delta t_{\text{sample}} = 10\text{ seconds} = 10,000\text{ ms}$.
- **Proposed R-STNet Inference Latency:** $\Delta t_{\text{infer}} = 0.06039\text{ ms per window}$.

$$\text{Latency Headroom Factor} = \frac{10,000\text{ ms}}{0.06039\text{ ms}} \approx 165,584\times$$

R-STNet executes forward inference more than **$160,000$ times faster than the physical telemetry arrival rate**. The model achieves an inference throughput of **$16,667$ sliding windows per second**, confirming that the computational footprint of the SRM introduces negligible real-time overhead.

### 4.4 Memory and VRAM Budget
- Peak allocated GPU VRAM during evaluation was **$978.69\text{ MB}$**, with a peak reserved memory footprint of **$1,124.00\text{ MB}$**.
- This footprint consumes only **$27.4\%$** of the available $4,096\text{ MB}$ VRAM on an entry-level mobile RTX 3050 GPU, demonstrating that R-STNet can run comfortably on embedded edge compute units (e.g., NVIDIA Jetson Orin Nano, industrial fanless PCs).

---

## 5. Scope Qualification and Scientific Boundaries

While the benchmarked numbers confirm computational efficiency, scientific integrity requires acknowledging practical deployment boundaries:

> **Important Qualification:**  
> The measured per-window inference latency ($0.060\text{ ms}$) represents pure GPU compute time for an in-memory batch. It demonstrates that the mathematical operations of R-STNet are computationally feasible for real-time edge processing. However, **end-to-end streaming latency—including physical Modbus/OPC-UA network polling, sliding window FIFO buffering, and SCADA alarm dispatch—was not measured in this laboratory benchmark.** Future pilot trials must evaluate end-to-end streaming jitter under live PLC polling cycles.
