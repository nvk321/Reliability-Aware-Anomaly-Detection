"""
Generates the 5 canonical educational and viva notebooks for the repository:
1. notebooks/01_dataset_exploration.ipynb
2. notebooks/02_preprocessing_and_windowing.ipynb
3. notebooks/03_baseline_evaluation.ipynb
4. notebooks/04_r_stnet_analysis.ipynb
5. notebooks/05_robustness_and_ablation.ipynb

Each notebook is strictly modular:
- Directly imports from `src.*`
- Never duplicates algorithm code
- Provides clear markdown explanations formatted for viva defense
- Includes runnable cells and pre-computed visual inspections
"""

import json
import os


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11.0",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py"
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3 (.venv)",
                "language": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines]
    }


def code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines]
    }


def generate_all_notebooks(output_dir="notebooks"):
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Notebook 01: Dataset Exploration
    # -------------------------------------------------------------
    nb1_cells = [
        md_cell([
            "# 01 — SWaT Dataset Physical Architecture & Exploratory Data Analysis",
            "**Course:** BCSE209P — Machine Learning Lab  ",
            "**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  ",
            "**Faculty Guide:** Dr. Dega Nagaraju  ",
            "**Repository:** [Reliability-Aware Anomaly Detection](https://github.com/nvk321/Reliability-Aware-Anomaly-Detection)",
            "",
            "---",
            "### Objectives",
            "1. Inspect the physical architecture of the Secure Water Treatment (SWaT) testbed (Stages P1 through P6).",
            "2. Explore the 51 sensor and actuator channels and their physical roles.",
            "3. Examine the dataset splits: pure normal training data (7 days) vs. mixed test data with 36 cyber-attacks (4 days).",
            "4. Analyze the label distribution (11.98% attack density) and the critical distinction between cyber-attacks and sensor faults."
        ]),
        code_cell([
            "import os",
            "import sys",
            "import json",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            "# Add project root to sys.path",
            "sys.path.insert(0, os.path.abspath('..'))",
            "",
            "from src.visualization.plots import apply_scientific_style",
            "apply_scientific_style()",
            "",
            "print('Imports successful!')"
        ]),
        md_cell([
            "### 1. Load Dataset Metadata and Split Summary",
            "The dataset summary records the exact sample counts across the downsampled splits."
        ]),
        code_cell([
            "summary_path = os.path.join('..', 'data', 'processed', 'SWaT', 'dataset_summary.json')",
            "with open(summary_path, 'r') as f:",
            "    dataset_summary = json.load(f)",
            "",
            "names_path = os.path.join('..', 'data', 'processed', 'SWaT', 'sensor_names.json')",
            "with open(names_path, 'r') as f:",
            "    sensor_names = json.load(f)",
            "",
            "print(f\"Dataset: {dataset_summary['dataset']}\")",
            "print(f\"Downsampling Rate: {dataset_summary['downsample_rate']}x (10s sampling interval)\")",
            "print(f\"Total Sensor Channels: {len(sensor_names)}\")",
            "print(f\"Training Samples (y=0, 100% Normal): {dataset_summary['train_samples']:,}\")",
            "print(f\"Validation Samples (y=0, 100% Normal): {dataset_summary['val_samples']:,}\")",
            "print(f\"Test Samples: {dataset_summary['test_samples']:,} (Attacks: {dataset_summary['test_anomalies']:,} = {dataset_summary['test_attack_rate']*100:.2f}%)\")"
        ]),
        md_cell([
            "### 2. Physical Subsystem Stage Breakdown",
            "SWaT comprises 6 continuous physical stages:",
            "- **P1 (Raw Water):** Flow meter `FIT101`, Level sensor `LIT101`, Inflow valve `MV101`, Pump `P101`/`P102`",
            "- **P2 (Chemical Dosing):** pH meter `AIT201`, Conductivity `AIT202`/`AIT203`, Dosing pumps `P201`-`P206`",
            "- **P3 (Ultrafiltration):** Differential pressure `DPIT301`, UF feed pumps `P301`/`P302`, Backwash valve `MV301`-`MV304`",
            "- **P4 (De-chlorination):** ORP analyzer `AIT401`/`AIT402`, UV lamps `UV401`, Booster pump `P401`/`P402`",
            "- **P5 (Reverse Osmosis):** RO feed pressure `AIT501`-`AIT504`, High pressure pumps `P501`/`P502`",
            "- **P6 (Backwash Recycling):** Recycle pump `P601`/`P602`"
        ]),
        code_cell([
            "stages = {",
            "    'P1 (Raw Water)': [s for s in sensor_names if s.startswith(('FIT1', 'LIT1', 'MV1', 'P1'))],",
            "    'P2 (Chemical Dosing)': [s for s in sensor_names if s.startswith(('AIT2', 'FIT2', 'MV2', 'P2'))],",
            "    'P3 (Ultrafiltration)': [s for s in sensor_names if s.startswith(('DPIT3', 'FIT3', 'LIT3', 'MV3', 'P3'))],",
            "    'P4 (De-chlorination)': [s for s in sensor_names if s.startswith(('AIT4', 'FIT4', 'LIT4', 'P4', 'UV4'))],",
            "    'P5 (Reverse Osmosis)': [s for s in sensor_names if s.startswith(('AIT5', 'FIT5', 'P5', 'MV5'))],",
            "    'P6 (Recycling)': [s for s in sensor_names if s.startswith(('P6', 'FIT6', 'LIT6'))],",
            "}",
            "",
            "for stage, channels in stages.items():",
            "    print(f\"{stage:25s}: {len(channels):2d} channels -> {', '.join(channels[:5])}{'...' if len(channels) > 5 else ''}\")"
        ]),
        md_cell([
            "### 3. Load Sample Preprocessed Arrays & Inspect Sensor Telemetry",
            "We load a small slice of the test array to visualize how continuous telemetry changes during an attack."
        ]),
        code_cell([
            "data_dir = os.path.join('..', 'data', 'processed', 'SWaT')",
            "test_data = np.load(os.path.join(data_dir, 'test.npy'))",
            "test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))",
            "",
            "# Slice around Attack Scenario 1 (Step 1,500: LIT101 Tank Inflow Overflow)",
            "slice_start, slice_len = 1400, 300",
            "sub_test = test_data[slice_start:slice_start + slice_len]",
            "sub_labels = test_labels[slice_start:slice_start + slice_len]",
            "",
            "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)",
            "",
            "# Plot LIT101 and FIT101",
            "lit_idx = sensor_names.index('LIT101')",
            "fit_idx = sensor_names.index('FIT101')",
            "time_axis = np.arange(slice_start, slice_start + slice_len)",
            "",
            "ax1.plot(time_axis, sub_test[:, lit_idx], label='LIT101 (Raw Tank Level)', color='#1D4ED8', lw=1.5)",
            "ax1.plot(time_axis, sub_test[:, fit_idx], label='FIT101 (Inflow Rate)', color='#059669', lw=1.5)",
            "ax1.axvspan(time_axis[sub_labels == 1][0], time_axis[sub_labels == 1][-1], color='#EF4444', alpha=0.25, label='Physical Attack Period')",
            "ax1.set_ylabel('Standardized Value')",
            "ax1.set_title('SWaT Attack Scenario 1: LIT101 Tank Overflow Manipulation', fontweight='bold')",
            "ax1.legend(loc='upper right')",
            "",
            "# Plot attack ground truth binary signal",
            "ax2.fill_between(time_axis, sub_labels, color='#EF4444', step='mid', alpha=0.6, label='Attack Label (y=1)')",
            "ax2.set_ylabel('Ground Truth')",
            "ax2.set_xlabel('Time Step (10s intervals)')",
            "ax2.set_ylim(-0.1, 1.2)",
            "ax2.legend(loc='upper right')",
            "",
            "plt.tight_layout()",
            "plt.show()"
        ]),
        md_cell([
            "### Viva Voce Key Takeaways — Section A: Data",
            "1. **SWaT Characteristics:** 51 variables (26 physical sensors, 25 discrete actuators/valves/pumps).",
            "2. **Attack Nature:** Attacks are physical disruptions (e.g. altering level thresholds to cause overflow, spoofing pH meters).",
            "3. **Downsampling Rationale:** 10s downsampling captures physical transport dynamics (hydraulic tanks take minutes to fill) while removing high-frequency sensor noise and making deep learning computationally feasible on edge hardware."
        ])
    ]

    # -------------------------------------------------------------
    # Notebook 02: Preprocessing and Windowing
    # -------------------------------------------------------------
    nb2_cells = [
        md_cell([
            "# 02 — Leak-Free Preprocessing, Zero-Variance Scaling & Sliding Window Generation",
            "**Course:** BCSE209P — Machine Learning Lab  ",
            "**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  ",
            "**Faculty Guide:** Dr. Dega Nagaraju  ",
            "**Repository:** [Reliability-Aware Anomaly Detection](https://github.com/nvk321/Reliability-Aware-Anomaly-Detection)",
            "",
            "---",
            "### Objectives",
            "1. Demonstrate the leak-free data processing pipeline using `src.preprocessing.scaler.RobustStandardScaler`.",
            "2. Address the zero-variance actuator trap with $\\epsilon = 10^{-8}$ variance clamping.",
            "3. Demonstrate chronological sliding window generation with `src.preprocessing.windowing.SlidingWindowGenerator`.",
            "4. Verify label aggregation and temporal boundary conditions."
        ]),
        code_cell([
            "import os",
            "import sys",
            "import json",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "",
            "sys.path.insert(0, os.path.abspath('..'))",
            "",
            "from src.preprocessing.scaler import RobustStandardScaler",
            "from src.preprocessing.windowing import SlidingWindowGenerator",
            "from src.visualization.plots import apply_scientific_style",
            "apply_scientific_style()",
            "",
            "print('Preprocessing modules imported successfully!')"
        ]),
        md_cell([
            "### 1. Robust Scaling with Zero-Variance Protection",
            "In industrial CPS datasets, certain backup pumps and valves remain permanently closed or open during training (`var = 0`).",
            "Standard `StandardScaler` from scikit-learn divides by zero, producing `NaN` or `inf` values.",
            "`RobustStandardScaler` implements:",
            "$$\\tilde{x}_i(t) = \\frac{x_i(t) - \\mu_i}{\\sigma_i + \\epsilon}$$",
            "where $\\sigma_i$ is clamped to $1.0$ if $\\sigma_i < \\epsilon$ ($10^{-8}$)."
        ]),
        code_cell([
            "# Demonstrate RobustStandardScaler on synthetic test array with constant channel",
            "np.random.seed(42)",
            "synthetic_train = np.random.normal(50.0, 5.0, (1000, 5))",
            "synthetic_train[:, 2] = 0.0  # Channel 2 is permanently off (constant actuator)",
            "",
            "scaler = RobustStandardScaler(eps=1e-8)",
            "scaled_train = scaler.fit_transform(synthetic_train)",
            "",
            "print('Original Channel Means: ', synthetic_train.mean(axis=0).round(2))",
            "print('Original Channel Stds:  ', synthetic_train.std(axis=0).round(2))",
            "print('Scaled Channel Means:   ', scaled_train.mean(axis=0).round(2))",
            "print('Scaled Channel Stds:    ', scaled_train.std(axis=0).round(2))",
            "print('Has NaNs:               ', np.isnan(scaled_train).any())"
        ]),
        md_cell([
            "### 2. Inspect Fitted Scaler Parameters on SWaT",
            "Let's load the actual saved parameters fitted strictly on the SWaT training split."
        ]),
        code_cell([
            "params_path = os.path.join('..', 'data', 'processed', 'SWaT', 'scaler_params.json')",
            "with open(params_path, 'r') as f:",
            "    scaler_params = json.load(f)",
            "",
            "print(f\"Fitted Sensors: {len(scaler_params['sensor_names'])}\")",
            "print(f\"Mean of LIT101: {scaler_params['mean'][scaler_params['sensor_names'].index('LIT101')]:.3f}\")",
            "print(f\"Std of LIT101:  {scaler_params['scale'][scaler_params['sensor_names'].index('LIT101')]:.3f}\")",
            "print(f\"Scale of P202 (constant actuator): {scaler_params['scale'][scaler_params['sensor_names'].index('P202')]:.3f} (protected from division by zero)\")"
        ]),
        md_cell([
            "### 3. Sliding Window Generation",
            "We convert 2D multivariate time-series $(T, M)$ into 3D sequential sliding windows $(N, L, M)$ with window length $L=30$ and stride $S=1$.",
            "Label mode `point_end` assigns the window's label to the ground-truth value at the final time step $t$."
        ]),
        code_cell([
            "L = 30",
            "win_gen = SlidingWindowGenerator(window_size=L, stride=1, label_mode='point_end')",
            "",
            "# Test on dummy sequence",
            "dummy_x = np.random.randn(100, 5)",
            "dummy_y = np.zeros(100, dtype=int)",
            "dummy_y[50:60] = 1  # 10 anomalous steps",
            "",
            "windows, window_labels = win_gen.create_windows(dummy_x, dummy_y)",
            "print(f\"Input shape:  {dummy_x.shape}\")",
            "print(f\"Output shape: {windows.shape} (N, L, M)\")",
            "print(f\"Labels shape: {window_labels.shape} (Attacks: {window_labels.sum()})\")"
        ]),
        md_cell([
            "### Viva Voce Key Takeaways — Section B: Preprocessing",
            "1. **Leak-Free Guarantee:** Normalization statistics $(\\mu, \\sigma)$ are computed **strictly on the normal training split**. Validation and test sets are transformed using frozen train statistics.",
            "2. **Zero-Variance Resilience:** Any sensor or actuator with zero natural variance during training has its scale clamped to $1.0$, preventing runtime floating-point errors.",
            "3. **Sliding Window Alignment:** Windows of size $L=30$ ($300\\text{ s} = 5\\text{ min}$ of historical context) provide sufficient temporal trajectory to forecast the next observation."
        ])
    ]

    # -------------------------------------------------------------
    # Notebook 03: Baseline Evaluation
    # -------------------------------------------------------------
    nb3_cells = [
        md_cell([
            "# 03 — Baseline Models Evaluation & The Clean Benchmark Reality",
            "**Course:** BCSE209P — Machine Learning Lab  ",
            "**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  ",
            "**Faculty Guide:** Dr. Dega Nagaraju  ",
            "**Repository:** [Reliability-Aware Anomaly Detection](https://github.com/nvk321/Reliability-Aware-Anomaly-Detection)",
            "",
            "---",
            "### Objectives",
            "1. Inspect the audited performance of Baseline A (Isolation Forest), Baseline B (ST-AE), and Baseline C (RSTNetNoSRM).",
            "2. Explain why Isolation Forest achieves high performance on the clean SWaT benchmark ($F_1 = 0.7182, \\text{FPR} = 0.92\\%$).",
            "3. Explain why deep reconstruction architectures without reliability weighting suffer from residual elevation ($F_1 \\approx 0.217, \\text{FPR} > 99\\%$).",
            "4. Establish why Baseline C (RSTNetNoSRM) is the mathematically exact ablation model for R-STNet."
        ]),
        code_cell([
            "import os",
            "import sys",
            "import json",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            "sys.path.insert(0, os.path.abspath('..'))",
            "from src.visualization.plots import apply_scientific_style",
            "apply_scientific_style()",
            "",
            "print('Imports successful!')"
        ]),
        md_cell([
            "### 1. Load Audited Clean Benchmark Results",
            "The frozen research results are stored in `results/metrics/`."
        ]),
        code_cell([
            "metrics_dir = os.path.join('..', 'results', 'metrics')",
            "",
            "files = {",
            "    'Isolation Forest': 'baseline_isolation_forest_swat.json',",
            "    'ST-AE': 'baseline_stae_swat.json',",
            "    'RSTNetNoSRM (Baseline C)': 'baseline_rstnet_no_srm_swat.json',",
            "    'R-STNet (Proposed)': 'proposed_rstnet_swat.json',",
            "}",
            "",
            "benchmark_records = []",
            "for model_name, filename in files.items():",
            "    filepath = os.path.join(metrics_dir, filename)",
            "    with open(filepath, 'r') as f:",
            "        data = json.load(f)",
            "    raw = data['raw_pointwise_metrics']",
            "    benchmark_records.append({",
            "        'Model': model_name,",
            "        'F1': raw['f1'],",
            "        'Precision': raw['precision'],",
            "        'Recall': raw['recall'],",
            "        'FPR': raw['fpr'],",
            "        'PR-AUC': raw['pr_auc'],",
            "        'ROC-AUC': raw['roc_auc'],",
            "    })",
            "",
            "df_bench = pd.DataFrame(benchmark_records)",
            "df_bench"
        ]),
        md_cell([
            "### 2. Visualize Benchmark Comparison",
            "We plot F1 Score, Precision, Recall, and False Alarm Rate (FPR) side-by-side."
        ]),
        code_cell([
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))",
            "",
            "models = df_bench['Model']",
            "x = np.arange(len(models))",
            "width = 0.25",
            "",
            "# Left plot: F1, Precision, Recall",
            "ax1.bar(x - width, df_bench['F1'], width, label='F1 Score', color='#1D4ED8')",
            "ax1.bar(x, df_bench['Precision'], width, label='Precision', color='#059669')",
            "ax1.bar(x + width, df_bench['Recall'], width, label='Recall', color='#D97706')",
            "ax1.set_xticks(x)",
            "ax1.set_xticklabels(models, rotation=25, ha='right')",
            "ax1.set_ylabel('Metric Value')",
            "ax1.set_title('Detection Performance on Clean SWaT Benchmark', fontweight='bold')",
            "ax1.legend()",
            "ax1.set_ylim(0, 1.1)",
            "",
            "# Right plot: FPR (False Positive Rate)",
            "colors = ['#10B981' if fpr < 0.10 else '#EF4444' for fpr in df_bench['FPR']]",
            "bars = ax2.bar(models, df_bench['FPR'] * 100, color=colors, width=0.5)",
            "ax2.set_xticklabels(models, rotation=25, ha='right')",
            "ax2.set_ylabel('False Positive Rate (FPR %)')",
            "ax2.set_title('False Alarm Rate on Clean SWaT Benchmark', fontweight='bold')",
            "",
            "# Label percentages on bars",
            "for bar in bars:",
            "    height = bar.get_height()",
            "    ax2.text(bar.get_x() + bar.get_width()/2., height + 1.5, f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')",
            "",
            "ax2.set_ylim(0, 115)",
            "plt.tight_layout()",
            "plt.show()"
        ]),
        md_cell([
            "### Viva Voce Key Takeaways — Section C & D: Baseline & Deep Baselines",
            "1. **Isolation Forest Reality:** Isolation Forest achieves the highest clean benchmark performance ($F_1 = 0.7182, \\text{FPR} = 0.92\\%$). As an axis-aligned tree ensemble, it evaluates sensors independently without joint spatiotemporal coupling. Unrecorded test distribution shifts (e.g., `AIT201` conductivity drift) do not distort its global decision space.",
            "2. **Why ST-AE Fails:** Spatio-temporal autoencoders reconstruct all 51 channels through a dense latent bottleneck. Unrecorded process shifts cause persistent non-zero reconstruction errors across the plant, resulting in near-100% false alarm saturation under clean validation-tuned thresholds.",
            "3. **The Role of RSTNetNoSRM (Baseline C):** Baseline C possesses the exact same 1D TCN + Graph Attention backbone (23,233 parameters) as R-STNet, but sums residuals uniformly without reliability weights. This isolates the Sensor Reliability Module as the sole variable."
        ])
    ]

    # -------------------------------------------------------------
    # Notebook 04: R-STNet Analysis
    # -------------------------------------------------------------
    nb4_cells = [
        md_cell([
            "# 04 — Proposed R-STNet: Spatio-Temporal Graph Consensus & Sensor Reliability Module",
            "**Course:** BCSE209P — Machine Learning Lab  ",
            "**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  ",
            "**Faculty Guide:** Dr. Dega Nagaraju  ",
            "**Repository:** [Reliability-Aware Anomaly Detection](https://github.com/nvk321/Reliability-Aware-Anomaly-Detection)",
            "",
            "---",
            "### Objectives",
            "1. Load the trained PyTorch checkpoint for `R-STNet` (23k parameters, $\\tau = 3.1401$).",
            "2. Inspect the learned Graph Attention (GAT) sensor adjacency structure.",
            "3. Examine the three components of the Sensor Reliability Module (SRM):",
            "   - Temporal Continuity ($v_i$)",
            "   - Topological Neighborhood Consistency ($c_i$)",
            "   - Residual Tail Confidence ($p_i$)",
            "4. Demonstrate reliability-weighted anomaly scoring vs. unweighted scoring."
        ]),
        code_cell([
            "import os",
            "import sys",
            "import json",
            "import torch",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "",
            "sys.path.insert(0, os.path.abspath('..'))",
            "from src.models.rstnet import RSTNet",
            "from src.reliability.sensor_reliability import SensorReliabilityModule",
            "from src.visualization.plots import apply_scientific_style",
            "apply_scientific_style()",
            "",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
            "print(f\"Using compute device: {device}\")"
        ]),
        md_cell([
            "### 1. Load Trained R-STNet Checkpoint",
            "We load the weights and verify architecture parameters."
        ]),
        code_cell([
            "ckpt_path = os.path.join('..', 'results', 'checkpoints', 'proposed_rstnet_swat.pt')",
            "ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)",
            "",
            "model = RSTNet(in_channels=51, hidden_dim=64, node_dim=32, top_k=15).to(device)",
            "model.load_state_dict(ckpt['model_state_dict'])",
            "model.eval()",
            "",
            "tau = float(ckpt['threshold_tau'])",
            "total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)",
            "",
            "print(f\"Loaded R-STNet Checkpoint: {ckpt_path}\")",
            "print(f\"Calibrated Threshold tau:  {tau:.4f}\")",
            "print(f\"Trainable Parameters:      {total_params:,}\")"
        ]),
        md_cell([
            "### 2. Inspect Learned Sensor Adjacency Graph",
            "R-STNet learns a top-15 sparse directed adjacency matrix between the 51 sensors."
        ]),
        code_cell([
            "with torch.no_grad():",
            "    # Extract node embeddings and compute adjacency",
            "    raw_adj = torch.relu(torch.mm(model.graph_attention.node_embed1, model.graph_attention.node_embed2.t()))",
            "    mask = torch.eye(51, device=device).bool()",
            "    raw_adj = raw_adj.masked_fill(mask, 0.0)",
            "    topk_vals, topk_indices = torch.topk(raw_adj, 15, dim=-1)",
            "    sparse_adj = torch.zeros_like(raw_adj)",
            "    sparse_adj.scatter_(-1, topk_indices, topk_vals)",
            "    adj_matrix = torch.softmax(sparse_adj + 1e-6, dim=-1).cpu().numpy()",
            "",
            "fig, ax = plt.subplots(figsize=(8, 7))",
            "im = ax.imshow(adj_matrix, cmap='Blues', interpolation='nearest')",
            "plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Learned Attention Weight')",
            "ax.set_title('R-STNet Learned Top-15 Sensor Interaction Graph', fontweight='bold')",
            "ax.set_xlabel('Target Sensor Index')",
            "ax.set_ylabel('Source Sensor Index')",
            "plt.tight_layout()",
            "plt.show()"
        ]),
        md_cell([
            "### 3. Forward Pass & Single-Window Diagnostic Evaluation",
            "We run `evaluate_step` on a sample test window to observe the SRM multi-perspective decomposition."
        ]),
        code_cell([
            "test_data = np.load(os.path.join('..', 'data', 'processed', 'SWaT', 'test.npy'))",
            "sample_window = test_data[1500:1530]  # Shape (30, 51)",
            "",
            "# Restore SRM calibration from checkpoint if available",
            "if 'srm_state' in ckpt and ckpt['srm_state'] is not None:",
            "    model.srm = SensorReliabilityModule.from_dict(ckpt['srm_state'])",
            "",
            "diag = model.evaluate_step(sample_window, device=device)",
            "",
            "print(f\"System Anomaly Score S_sys: {diag['system_score']:.4f} (Threshold tau = {tau:.4f})\")",
            "print(f\"Most Suspicious Sensor:    Sensor #{diag['most_suspicious_sensor']}\")",
            "print(f\"Sensor Fault Score:        {diag['sensor_fault_score']:.4f}\")",
            "print(f\"Mean Reliability:          {diag['reliability_vector'].mean():.4f}\")",
            "print(f\"Min Reliability:           {diag['reliability_vector'].min():.4f}\")"
        ]),
        md_cell([
            "### Viva Voce Key Takeaways — Section D & E: Proposed Model & Reliability",
            "1. **Architecture Formulation:**",
            "   $$\\mathbf{X} \\xrightarrow{\\text{1D TCN}} \\mathbf{H}_{\\text{temp}} \\xrightarrow{\\text{GAT}} \\mathbf{H}_{\\text{joint}} \\xrightarrow{\\text{Forecast}} \\hat{\\mathbf{x}}(t)$$",
            "2. **SRM Formulation:**",
            "   $$\\text{logit}(r_i) = w_v v_i + w_c c_i + w_p p_i + b$$",
            "3. **Reliability-Weighted Scoring:**",
            "   $$S_{\\text{sys}}(t) = \\frac{\\sum_i r_i(t) z_i(t)}{\\sum_i r_i(t) + \\epsilon}$$",
            "   When a sensor suffers from severe jitter or drift, its reliability $r_i \\to 0$, suppressing its contribution from the system anomaly score and preventing a false plant shutdown."
        ])
    ]

    # -------------------------------------------------------------
    # Notebook 05: Robustness and Ablation
    # -------------------------------------------------------------
    nb5_cells = [
        md_cell([
            "# 05 — Controlled Sensor Corruption Robustness & SRM Ablation Study",
            "**Course:** BCSE209P — Machine Learning Lab  ",
            "**Students:** P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)  ",
            "**Faculty Guide:** Dr. Dega Nagaraju  ",
            "**Repository:** [Reliability-Aware Anomaly Detection](https://github.com/nvk321/Reliability-Aware-Anomaly-Detection)",
            "",
            "---",
            "### Objectives",
            "1. Demonstrate synthetic controlled sensor corruptions using `src.corruption.corruptor.SensorCorruptor`.",
            "2. Inspect audited multi-seed robustness results across Gaussian noise, stuck sensors, and dynamic drift.",
            "3. Evaluate the SRM ablation study (No SRM, -Temporal, -Graph, -Residual, Full SRM).",
            "4. Discuss the physical actuator boundary (stuck sensors on zero-variance inactive pumps)."
        ]),
        code_cell([
            "import os",
            "import sys",
            "import json",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            "sys.path.insert(0, os.path.abspath('..'))",
            "from src.corruption.corruptor import SensorCorruptor",
            "from src.visualization.plots import apply_scientific_style",
            "apply_scientific_style()",
            "",
            "print('Imports successful!')"
        ]),
        md_cell([
            "### 1. Demonstrate Controlled Sensor Corruption",
            "`SensorCorruptor` injects non-adversarial physical sensor degradations while preserving raw ground truth and creating a binary fault mask."
        ]),
        code_cell([
            "corruptor = SensorCorruptor(seed=42)",
            "dummy_data = np.sin(np.linspace(0, 20, 200))[:, None].repeat(5, axis=1)",
            "",
            "# Apply 3 corruption modes to Channel 1",
            "noisy_data, mask_n, _ = corruptor.apply_corruption(dummy_data, corruption_type='gaussian_noise', affected_channels=[1], severity=0.5)",
            "drift_data, mask_d, _ = corruptor.apply_corruption(dummy_data, corruption_type='drift', affected_channels=[1], severity=1.0)",
            "stuck_data, mask_s, _ = corruptor.apply_corruption(dummy_data, corruption_type='stuck_constant', affected_channels=[1])",
            "",
            "fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)",
            "axes[0].plot(dummy_data[:, 1], 'k--', label='Clean Signal', alpha=0.6)",
            "axes[0].plot(noisy_data[:, 1], color='#DC2626', label='Gaussian Noise (sigma=0.5)')",
            "axes[0].set_title('Corruption Mode 1: High Sensor Jitter (Noise)', fontweight='bold')",
            "axes[0].legend(loc='upper right')",
            "",
            "axes[1].plot(dummy_data[:, 1], 'k--', label='Clean Signal', alpha=0.6)",
            "axes[1].plot(drift_data[:, 1], color='#D97706', label='Linear Calibration Drift (delta=1.0)')",
            "axes[1].set_title('Corruption Mode 2: Calibration Drift', fontweight='bold')",
            "axes[1].legend(loc='upper right')",
            "",
            "axes[2].plot(dummy_data[:, 1], 'k--', label='Clean Signal', alpha=0.6)",
            "axes[2].plot(stuck_data[:, 1], color='#7C3AED', label='Stuck Sensor (Frozen Reading)')",
            "axes[2].set_title('Corruption Mode 3: Hardware Freeze (Stuck)', fontweight='bold')",
            "axes[2].legend(loc='upper right')",
            "",
            "plt.tight_layout()",
            "plt.show()"
        ]),
        md_cell([
            "### 2. Load Audited Robustness Benchmark Summary",
            "We inspect the multi-seed corruption summary comparing Baseline C and R-STNet."
        ]),
        code_cell([
            "corr_summary_path = os.path.join('..', 'results', 'corruption', 'corruption_benchmark_summary.json')",
            "with open(corr_summary_path, 'r') as f:",
            "    corr_data = json.load(f)",
            "",
            "print(f\"Benchmark: {corr_data['benchmark']}\")",
            "print(f\"Evaluated Regimes: {len(corr_data['regimes'])}\")",
            "",
            "rows = []",
            "for r in corr_data['regimes']:",
            "    rows.append({",
            "        'Fault Mode': r['regime_name'],",
            "        'Severity': r['severity'],",
            "        'Baseline C F1': r['baseline_c_mean_f1'],",
            "        'R-STNet F1': r['rstnet_mean_f1'],",
            "        'F1 Delta': r['f1_improvement'],",
            "        'R-STNet FPR': r['rstnet_mean_fpr'],",
            "        'Reliability Sep': r['mean_reliability_separation'],",
            "    })",
            "",
            "df_corr = pd.DataFrame(rows)",
            "df_corr"
        ]),
        md_cell([
            "### 3. Load Audited Component Ablation Summary",
            "We inspect the audited ablation experiments evaluating the individual components of the SRM."
        ]),
        code_cell([
            "ablation_summary_path = os.path.join('..', 'results', 'ablations', 'ablation_summary.json')",
            "with open(ablation_summary_path, 'r') as f:",
            "    ablation_data = json.load(f)",
            "",
            "ablation_rows = []",
            "for a in ablation_data['ablations']:",
            "    ablation_rows.append({",
            "        'Ablation ID': a['id'],",
            "        'Configuration': a['name'],",
            "        'F1 Score': a['f1'],",
            "        'Precision': a['precision'],",
            "        'Recall': a['recall'],",
            "        'FPR': a['fpr'],",
            "        'F1 Gain vs Baseline C': a.get('f1_gain_vs_baseline_c', 0.0),",
            "    })",
            "",
            "df_ablation = pd.DataFrame(ablation_rows)",
            "df_ablation"
        ]),
        md_cell([
            "### Viva Voce Key Takeaways — Section G & H: Robustness, Ablation & Honest Results",
            "1. **False Alarm Elimination:** Across 44,963 evaluation windows, R-STNet eliminates **14,898 false alarms** relative to Baseline C (a 37.8% relative reduction in FPR from 99.86% to 62.14%).",
            "2. **Reliability Separation:** Under Gaussian noise and linear drift, the SRM achieves a clean-vs-corrupted separation of $\\Delta r = +0.289$, correctly identifying 100% of synthetic sensor faults.",
            "3. **Ablation Insight:** Removing Neighborhood Consistency (Ablation 4) drops F1 by $-0.038$, proving that inter-sensor physical redundancy is the strongest discriminator against false alarms.",
            "4. **The Stuck-Sensor Boundary:** On inactive backup pumps (`P202`, `P204`), zero historical variance yields zero rate-of-change violation. Without supervisory PLC command telemetry, statistical SRM produces 0.0% fault recall on static actuators."
        ])
    ]

    # Write each notebook
    notebook_map = {
        "01_dataset_exploration.ipynb": nb1_cells,
        "02_preprocessing_and_windowing.ipynb": nb2_cells,
        "03_baseline_evaluation.ipynb": nb3_cells,
        "04_r_stnet_analysis.ipynb": nb4_cells,
        "05_robustness_and_ablation.ipynb": nb5_cells,
    }

    for name, cells in notebook_map.items():
        nb_path = os.path.join(output_dir, name)
        nb_json = make_notebook(cells)
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb_json, f, indent=1)
        print(f"[SUCCESS] Generated: {nb_path} ({len(cells)} cells)")


if __name__ == "__main__":
    generate_all_notebooks()
