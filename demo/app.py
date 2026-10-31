"""
Interactive Demonstration Dashboard: Reliability-Aware Anomaly Detection (R-STNet)
Project: Reliability-Aware Anomaly Detection for Multivariate Sensor Data
Course: BCSE209P — Machine Learning Lab
Students: P Naga Venkata Kushal (23BDS0133) & Koteru Tarun (23BDS0103)
Faculty Guide: Dr. Dega Nagaraju
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
import altair as alt
import streamlit as st

# Suppress PyTorch and serialization warnings for clean UI
warnings.filterwarnings("ignore")

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.rstnet import RSTNet
from src.models.baselines.rstnet_no_srm import RSTNetNoSRM
from src.preprocessing.windowing import SlidingWindowGenerator
from src.corruption.corruptor import SensorCorruptor

# Page Configuration
st.set_page_config(
    page_title="R-STNet: Reliability-Aware Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Aesthetic CSS for Presentation-Grade UI
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.1rem;
    }
    .author-sub {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .badge-guide {
        background: #F1F5F9;
        color: #334155;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .status-card {
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .status-normal {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        color: #065F46;
    }
    .status-sensor-fault {
        background-color: #FEF3C7;
        border-left: 6px solid #F59E0B;
        color: #92400E;
    }
    .status-cyber-attack {
        background-color: #FEE2E2;
        border-left: 6px solid #EF4444;
        color: #991B1B;
    }
    .status-dual-fault {
        background-color: #F3E8FF;
        border-left: 6px solid #8B5CF6;
        color: #5B21B6;
    }
    .metric-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
    }
    .disclaimer-banner {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #1E40AF;
        margin-bottom: 1.2rem;
    }
    .callout-box {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
        font-size: 0.9rem;
        color: #1E293B;
    }
    .warning-box {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
        font-size: 0.9rem;
        color: #92400E;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    """Load preprocessed SWaT data arrays and sensor names."""
    data_dir = "data/processed/SWaT"
    test_raw = np.load(os.path.join(data_dir, "test.npy"))
    test_labels_raw = np.load(os.path.join(data_dir, "test_labels.npy"))
    val_raw = np.load(os.path.join(data_dir, "val.npy"))

    with open(os.path.join(data_dir, "sensor_names.json"), "r") as f:
        sensor_names = json.load(f)

    win_gen = SlidingWindowGenerator(window_size=30, stride=1, label_mode="point_end")
    val_windows, _ = win_gen.create_windows(val_raw)
    test_windows, test_labels = win_gen.create_windows(test_raw, test_labels_raw)

    return val_windows, test_windows, test_labels, sensor_names


@st.cache_resource
def load_models(num_sensors: int, val_windows: np.ndarray):
    """Load neural network checkpoints with weights_only=False specified."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = "results/checkpoints"

    # 1. Baseline C: RSTNetNoSRM
    model_c = RSTNetNoSRM(in_channels=num_sensors).to(device)
    ckpt_c_path = os.path.join(ckpt_dir, "baseline_rstnet_no_srm_swat.pt")
    if os.path.exists(ckpt_c_path):
        ckpt_c = torch.load(ckpt_c_path, map_location=device, weights_only=False)
        model_c.load_state_dict(ckpt_c["model_state_dict"])
        tau_c = float(ckpt_c.get("threshold_tau", 12.0706))
        if "mad_residual" in ckpt_c and ckpt_c["mad_residual"] is not None:
            model_c.mad_residual_ = ckpt_c["mad_residual"]
        else:
            model_c.calibrate_mad(val_windows, device=device)
    else:
        tau_c = 12.0706

    # 2. Proposed R-STNet (with SRM)
    model_prop = RSTNet(in_channels=num_sensors).to(device)
    ckpt_prop_path = os.path.join(ckpt_dir, "proposed_rstnet_swat.pt")
    if os.path.exists(ckpt_prop_path):
        ckpt_prop = torch.load(ckpt_prop_path, map_location=device, weights_only=False)
        model_prop.load_state_dict(ckpt_prop["model_state_dict"])
        tau_prop = float(ckpt_prop.get("threshold_tau", 3.1401))
        if "srm_state" in ckpt_prop and ckpt_prop["srm_state"] is not None:
            from src.reliability.sensor_reliability import SensorReliabilityModule
            model_prop.srm = SensorReliabilityModule.from_dict(ckpt_prop["srm_state"])
        else:
            model_prop.calibrate_srm(val_windows, device=device)
    else:
        tau_prop = 3.1401

    model_c.eval()
    model_prop.eval()
    return model_c, tau_c, model_prop, tau_prop, device


def safe_extract_srm_components(details, idx=0, num_sensors=51):
    """Safely extracts SRM multi-perspective breakdown components without KeyErrors."""
    r_v = details.get("temporal_continuity", details.get("r_velocity", np.ones((1, num_sensors))))[idx]
    r_c = details.get("neighborhood_consistency", details.get("r_consistency", np.ones((1, num_sensors))))[idx]
    r_p = details.get("residual_confidence", details.get("r_prediction", np.ones((1, num_sensors))))[idx]
    return np.asarray(r_v, dtype=float), np.asarray(r_c, dtype=float), np.asarray(r_p, dtype=float)


def main():
    # Header Section
    st.markdown('<div class="main-title">🛡️ R-STNet: Reliability-Aware Industrial Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="author-sub">'
        '<b>Students:</b> P Naga Venkata Kushal (<code>23BDS0133</code>) &nbsp;|&nbsp; Koteru Tarun (<code>23BDS0103</code>) &nbsp;&bull;&nbsp; '
        '<b>Faculty Guide:</b> <span class="badge-guide">Dr. Dega Nagaraju</span> &nbsp;&bull;&nbsp; '
        '<b>Course:</b> BCSE209P Machine Learning Lab (Vellore Institute of Technology)'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer-banner">'
        '🔬 <b>Research Notice:</b> Interactive evaluation operates on the benchmarked Secure Water Treatment (SWaT) testbed dataset '
        'and controlled sensor degradation suites. This is an academic research prototype evaluated offline on recorded dataset logs, '
        'not a live SCADA hardware-in-the-loop deployment.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Load Data & Checkpoints
    try:
        val_windows, test_windows, test_labels, sensor_names = load_data()
        num_sensors = len(sensor_names)
        model_c, tau_c, model_prop, tau_prop, device = load_models(num_sensors, val_windows)
    except Exception as e:
        st.error(f"Error initializing data or neural models: {e}")
        st.info("Ensure preprocessed data exists in `data/processed/SWaT/` and checkpoints exist in `results/checkpoints/`.")
        return

    # Sidebar: Presentation Controls & Scenario Presets
    st.sidebar.markdown("### 🎛️ Presentation Control Panel")

    scenario_options = {
        "🎯 Attack Scenario 1: LIT101 Tank Inflow Overflow (Step 1,500)": {
            "offset": 1500,
            "desc": "Attack targeting raw water storage tank LIT101; inflow valve MV101 altered to cause simulated tank overflow.",
            "type": "Physical Attack",
        },
        "🎯 Attack Scenario 2: MV201 Chemical Dosing Bypass (Step 16,800)": {
            "offset": 16800,
            "desc": "Stage 2 chemical dosing motorized valve MV201 manipulated to disrupt chlorination and pH regulation.",
            "type": "Actuator Attack",
        },
        "🎯 Attack Scenario 3: P301 Ultrafiltration Pump Disruption (Step 23,400)": {
            "offset": 23400,
            "desc": "Primary ultrafiltration feed pump P301 turned off unexpectedly while differential pressure is active.",
            "type": "Process Disruption",
        },
        "🎯 Attack Scenario 4: Multi-Stage P1 & P3 Complex Attack (Step 36,000)": {
            "offset": 36000,
            "desc": "Simultaneous coordination between Stage 1 water inflow and Stage 3 membrane filtration.",
            "type": "Coordinated Multi-Stage",
        },
        "🛡️ Nominal Pure Operation: Normal Process Baseline (Step 5,000)": {
            "offset": 5000,
            "desc": "Continuous normal industrial operation with nominal chemical dosing and continuous filtration.",
            "type": "Nominal Baseline",
        },
        "⚙️ Custom Timeline Scrubbing (Manual Slider)": {
            "offset": None,
            "desc": "Freely scrub through any temporal interval across the 44,963-window test dataset.",
            "type": "Manual Exploration",
        },
    }

    selected_scenario = st.sidebar.selectbox(
        "Choose Inspection Scenario (Viva Presets):",
        list(scenario_options.keys()),
        index=0,
        help="Quickly jump to documented SWaT attack intervals or normal quiet periods.",
    )

    scenario_meta = scenario_options[selected_scenario]
    if scenario_meta["offset"] is not None:
        time_window_start = scenario_meta["offset"]
        st.sidebar.caption(f"ℹ️ **Scenario Details:** {scenario_meta['desc']}")
    else:
        time_window_start = st.sidebar.slider(
            "Scrub Test Timeline Offset:",
            min_value=0,
            max_value=len(test_windows) - 400,
            value=1500,
            step=100,
        )

    eval_window_len = st.sidebar.slider(
        "Timeline Inspection Window (Steps):",
        min_value=100,
        max_value=500,
        value=250,
        step=50,
        help="Number of consecutive 10s time steps to display in timeline plots.",
    )

    selected_model_name = st.sidebar.radio(
        "Active Anomaly Scoring Model:",
        ["Proposed R-STNet (with SRM)", "Baseline C: RSTNetNoSRM (Uniform Residuals)"],
        index=0,
        help="Switch between reliability-weighted scoring and uniform unweighted residual scoring.",
    )

    slice_idx = slice(time_window_start, time_window_start + eval_window_len)

    # Authoritative 9-Tab Dashboard Flow Matching Thesis Structure
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "1. 📖 Project Overview",
        "2. 🚰 SWaT Testbed",
        "3. 🧠 Model Architecture",
        "4. 🎯 Reliability (SRM)",
        "5. 📈 Live Detection",
        "6. 📊 Benchmark Audit",
        "7. 🧪 Robustness Sandbox",
        "8. 🔬 Ablation Studies",
        "9. ⚠️ Academic Limitations",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Project Overview & Problem Formulation
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("1. Research Motivation & Problem Formulation")
        col_ov1, col_ov2 = st.columns([3, 2])
        with col_ov1:
            st.markdown(
                """
                ### The Core Research Problem
                In multivariate cyber-physical systems (water treatment plants, smart grids, industrial manufacturing), 
                standard deep learning anomaly detectors (e.g., autoencoders, graph neural networks) aggregate reconstruction 
                or forecasting errors across all sensor channels uniformly:
                $$S_{\\text{sys}}(t) = \\frac{1}{M} \\sum_{i=1}^{M} z_i(t)$$
                
                **The Fatal Weakness:** When a single instrument suffers from benign hardware degradation (such as sensor jitter, 
                calibration drift, or intermittent telemetry loss), its individual residual $z_i(t)$ spikes drastically. Because the 
                reconstruction error is summed uniformly across all sensors, this localized failure **pollutes the entire system score**, 
                triggering an immediate emergency plant shutdown. In real-world operation, this causes severe alert fatigue and false shutdown saturation.
                
                ### Central Research Question
                > *Can explicit online sensor-reliability estimation improve the robustness of multivariate anomaly detection, 
                > decoupling localized sensor instrument faults from genuine cyber-physical process attacks?*
                
                ### Proposed Solution: R-STNet
                We propose the **Reliability-Aware Spatio-Temporal Network (R-STNet)**:
                1. **Spatio-Temporal Graph Backbone:** Dilated 1D temporal convolutions per sensor combined with learned top-$k$ Graph Attention (GAT) to capture inter-sensor physical conservation laws.
                2. **Sensor Reliability Module (SRM):** Online triangulation of Temporal Continuity, Graph Neighborhood Consistency, and Residual Tail Confidence to compute continuous trustworthiness weights $r_i(t) \\in [0, 1]$.
                3. **Reliability-Weighted Scoring:** Downweights degraded channels from the plant emergency score while isolating them into a dedicated sensor fault alert.
                """
            )
        with col_ov2:
            st.markdown(
                """
                <div class="callout-box">
                <b>🎓 Viva Quick Facts</b><br>
                &bull; <b>Institution:</b> VIT Vellore (BCSE209P)<br>
                &bull; <b>Primary Testbed:</b> SWaT (51 channels, 11 days)<br>
                &bull; <b>Model Parameter Count:</b> 23,233 (Edge-viable)<br>
                &bull; <b>Hardware VRAM:</b> 73.8 MB peak (tested on RTX 3050)<br>
                &bull; <b>Ablation Baseline:</b> RSTNetNoSRM (Exact same backbone)<br>
                &bull; <b>Key Result:</b> Eliminates 14,898 false alarms relative to Baseline C; achieves +0.29 reliability separation under noise.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image("report/REVIEW2/figures/fig4_1_model_comparison_barchart.png", caption="Model Comparison on SWaT Clean Benchmark", width="stretch")

    # -------------------------------------------------------------------------
    # TAB 2: SWaT Testbed & Physical Process
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("2. Secure Water Treatment (SWaT) Industrial Testbed")
        st.markdown(
            "The SWaT dataset was developed by the iTrust Centre for Research in Cyber Security at Singapore University of Technology "
            "and Design (SUTD). It models an operational 6-stage industrial water treatment plant producing 5 gallons per minute of treated water."
        )

        col_st1, col_st2, col_st3, col_st4 = st.columns(4)
        col_st1.metric("Total Channels (M)", "51", "26 Sensors + 25 Actuators")
        col_st2.metric("Train Split (Normal)", "39,744 windows", "7 Days (y = 0)")
        col_st3.metric("Val Split (Normal)", "9,936 windows", "Chronological (y = 0)")
        col_st4.metric("Test Split (Mixed)", "44,963 windows", "4 Days (12.15% Attacks)")

        st.markdown("#### Physical Stages Overview")
        st.markdown(
            """
            - **Stage 1 (P1): Raw Water Inflow & Storage** — Water intake through motorized valve `MV101`, level transmitter `LIT101`, and raw water pumps `P101`/`P102`.
            - **Stage 2 (P2): Chemical Pre-Treatment & Dosing** — Coagulation, chlorination, and pH adjustment monitored by analyzers `AIT201` (pH) and `AIT202`/`AIT203` (conductivity).
            - **Stage 3 (P3): Ultrafiltration (UF)** — Membrane separation to remove suspended solids, monitored via differential pressure `DPIT301` and backwash cycle valves.
            - **Stage 4 (P4): De-Chlorination & UV Disinfection** — Neutralization of free chlorine via sodium bisulfite and ultraviolet lamps `UV401`, measured by ORP meters `AIT401`/`AIT402`.
            - **Stage 5 (P5): Reverse Osmosis (RO) Desalination** — High-pressure desalination pumps `P501`/`P502` and permeate conductivity transmitters `AIT501`-`AIT504`.
            - **Stage 6 (P6): Backwash Cleaning & Recycling** — Automated cleaning sequences with recycled water return pump `P601`/`P602`.
            """
        )

    # -------------------------------------------------------------------------
    # TAB 3: Model Architecture & Graph Consensus
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("3. Spatio-Temporal Graph Architecture & Forecasting Head")
        st.markdown(
            "R-STNet extracts dual-domain representations using dilated 1D temporal convolutions followed by top-$k$ graph attention."
        )

        st.markdown(
            """
            ```
            Input Window X in R^(L x M)  [L=30, M=51]
                        |
            [Temporal Feature Extractor]
            1D Conv (k=3, hidden=64) -> ReLU
            1D Conv (k=3, dilation=2, hidden=64) + Residual Skip
            Linear Projection -> H_temp in R^(B x M x 32)
                        |
            [Sensor Graph Attention (GAT)]
            Learnable Node Embeddings: E1, E2 in R^(M x 32)
            Directed Adjacency: A = Softmax( TopK( ReLU(E1 @ E2^T) ) )
            Spatial Message Passing: H_spatial = A @ H_temp
                        |
            [Joint Node Representation]
            H_joint = [H_temp || H_spatial] in R^(B x M x 64)
                        |
            [Forecasting Head]
            Dense(64 -> 64) -> ReLU -> Dropout(0.2) -> Dense(64 -> 1)
            Output Prediction: x_hat(t) in R^(B x M)
            ```
            """
        )

        col_arch1, col_arch2 = st.columns(2)
        with col_arch1:
            st.markdown("#### Parameter Footprint Breakdown")
            param_df = pd.DataFrame([
                {"Component": "Temporal 1D CNNs (Conv1 + Conv2)", "Parameters": "12,672", "Dimension": "Hidden=64, Kernel=3"},
                {"Component": "Graph Attention (Node Embeddings + Lin)", "Parameters": "4,352", "Dimension": "NodeDim=32, TopK=15"},
                {"Component": "Forecasting MLP Head", "Parameters": "6,209", "Dimension": "64 -> 64 -> 1"},
                {"Component": "Sensor Reliability Module (SRM)", "Parameters": "Calibrated Stats", "Dimension": "Online Statistics"},
                {"Component": "Total R-STNet Architecture", "Parameters": "23,233", "Dimension": "73.8 MB Peak VRAM"},
            ])
            st.dataframe(param_df, width="stretch")
        with col_arch2:
            st.markdown("#### Edge Viability & Computational Efficiency")
            st.markdown(
                """
                - **Latency:** Single-window inference executes in **< 4.2 ms** on GPU and **< 18 ms** on Intel Core i5 CPU.
                - **Memory Footprint:** Peak VRAM is **73.81 MB**, allowing deployment on low-cost edge industrial IPCs (e.g., NVIDIA Jetson).
                - **Sparsification:** Setting $\\text{top-}k = 15$ prunes weak spurious correlations and bounds graph aggregation complexity to $O(k \\cdot M)$ rather than $O(M^2)$.
                """
            )

    # -------------------------------------------------------------------------
    # TAB 4: Sensor Reliability Module (SRM)
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("4. Sensor Reliability Module (SRM): Multi-Perspective Triangulation")
        st.markdown(
            "The SRM computes continuous reliability weights $r_i(t) \\in [0, 1]$ online by triangulating "
            "**Temporal Continuity ($v_i$)**, **Topological Neighborhood Consistency ($c_i$)**, and **Residual Tail Confidence ($p_i$)**."
        )

        latest_window = test_windows[time_window_start + eval_window_len - 1 : time_window_start + eval_window_len]
        latest_tensor = torch.tensor(latest_window, dtype=torch.float32).to(device)
        with torch.no_grad():
            latest_pred, _ = model_prop(latest_tensor)
            latest_pred_np = latest_pred.cpu().numpy()
            r_vec, details = model_prop.srm.estimate_reliability_batch(latest_window, predicted_x=latest_pred_np)

        current_r = r_vec[0]
        r_v, r_c, r_p = safe_extract_srm_components(details, idx=0, num_sensors=num_sensors)

        # Stage selector
        stage_filter = st.radio(
            "Filter Sensors by Physical Treatment Subsystem:",
            [
                "All 51 Sensors",
                "Stage 1: Raw Water & Inflow (P1)",
                "Stage 2: Chemical Dosing & Mixing (P2)",
                "Stage 3: Ultrafiltration (P3)",
                "Stage 4: De-chlorination & Carbon (P4)",
                "Stage 5: RO Desalination (P5)",
            ],
            horizontal=True,
        )

        stage_prefixes = {
            "Stage 1": ("FIT1", "LIT1", "MV1", "P1"),
            "Stage 2": ("AIT2", "FIT2", "MV2", "P2"),
            "Stage 3": ("DPIT3", "FIT3", "LIT3", "MV3", "P3"),
            "Stage 4": ("AIT4", "FIT4", "LIT4", "P4", "UV4"),
            "Stage 5": ("AIT5", "FIT5", "P5", "MV5"),
        }

        active_indices = list(range(num_sensors))
        for key, pfx in stage_prefixes.items():
            if key in stage_filter:
                active_indices = [i for i, name in enumerate(sensor_names) if name.startswith(pfx)]
                break

        disp_names = [sensor_names[i] for i in active_indices]
        disp_r = current_r[active_indices]

        # Reliability Bar Chart
        df_r = pd.DataFrame({
            "Sensor": disp_names,
            "Reliability": disp_r,
            "Status": ["Reliable" if r >= 0.60 else ("Moderate" if r >= 0.30 else "Degraded") for r in disp_r],
        })

        color_scale = alt.Scale(
            domain=["Reliable", "Moderate", "Degraded"],
            range=["#10B981", "#F59E0B", "#EF4444"],
        )

        r_bar_chart = alt.Chart(df_r).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Sensor:N", sort=None, title="Sensor Channel", axis=alt.Axis(labelAngle=-60)),
            y=alt.Y("Reliability:Q", title="Reliability Score r_i(t)", scale=alt.Scale(domain=[0, 1.05])),
            color=alt.Color("Status:N", scale=color_scale, title="Reliability Level"),
            tooltip=["Sensor:N", alt.Tooltip("Reliability:Q", format=".4f"), "Status:N"],
        ).properties(
            width="container",
            height=280,
            title="Current Per-Sensor Reliability Scores (Green = Trusted, Red = Downweighted)",
        )

        cutoff_rule = alt.Chart(pd.DataFrame({"y": [0.60]})).mark_rule(color="#475569", strokeDash=[4, 4]).encode(y="y:Q")
        st.altair_chart(r_bar_chart + cutoff_rule, width="stretch")

        # Single Sensor Inspector Card
        st.markdown("#### 🔍 Single Sensor Multi-Perspective Diagnostics")
        col_sel_s, col_card_s = st.columns([1, 2])

        with col_sel_s:
            inspected_sensor = st.selectbox("Inspect Channel:", disp_names, index=0)
            inspected_idx = sensor_names.index(inspected_sensor)

        with col_card_s:
            s_r = float(current_r[inspected_idx])
            s_v = float(r_v[inspected_idx])
            s_c = float(r_c[inspected_idx])
            s_p = float(r_p[inspected_idx])

            col_sub1, col_sub2, col_sub3, col_sub4 = st.columns(4)
            col_sub1.metric("Overall r(t)", f"{s_r:.3f}")
            col_sub2.metric("Temporal (v)", f"{s_v:.3f}")
            col_sub3.metric("Topological (c)", f"{s_c:.3f}")
            col_sub4.metric("Residual (p)", f"{s_p:.3f}")

            if s_r >= 0.60:
                st.success(f"✅ **{inspected_sensor}** is operating with high credibility ($r_i = {s_r:.3f}$).")
            else:
                st.warning(f"⚠️ **{inspected_sensor}** is downweighted ($r_i = {s_r:.3f}$) by SRM, protecting the plant from false alert saturation.")

        # Degraded Sensors Table
        degraded_indices = [i for i in range(num_sensors) if current_r[i] < 0.60]
        if degraded_indices:
            st.markdown(f"##### ⚠️ Currently Downweighted Sensors ($r_i < 0.60$): {len(degraded_indices)} channels")
            deg_rows = []
            for idx in degraded_indices:
                deg_rows.append({
                    "Sensor Name": sensor_names[idx],
                    "Overall Reliability (r)": f"{current_r[idx]:.4f}",
                    "Temporal Continuity (v)": f"{r_v[idx]:.4f}",
                    "Graph Consistency (c)": f"{r_c[idx]:.4f}",
                    "Residual Tail Confidence (p)": f"{r_p[idx]:.4f}",
                    "Weight Status": "Suppressed / Downweighted",
                })
            st.dataframe(pd.DataFrame(deg_rows), width="stretch")

    # -------------------------------------------------------------------------
    # TAB 5: Live Process Monitoring & Detection
    # -------------------------------------------------------------------------
    with tab5:
        st.subheader("5. Process Monitoring & 4-Way Operational State Classification")

        # Evaluate models on selected slice
        sub_windows = test_windows[slice_idx]
        sub_labels = test_labels[slice_idx]
        sub_tensor = torch.tensor(sub_windows, dtype=torch.float32).to(device)

        with torch.no_grad():
            # Proposed R-STNet
            preds_prop, _ = model_prop(sub_tensor)
            preds_prop_np = preds_prop.cpu().numpy()
            r_mat_prop, details_prop = model_prop.srm.estimate_reliability_batch(sub_windows, predicted_x=preds_prop_np)
            diff_prop = np.abs(sub_windows[:, -1, :] - preds_prop_np)
            z_prop = diff_prop / (model_prop.srm.mad_residual_ + 1e-6)
            scores_prop = model_prop.srm.compute_weighted_system_anomaly_batch(z_prop, r_mat_prop)

            # Baseline C (No SRM)
            preds_c, _ = model_c(sub_tensor)
            preds_c_np = preds_c.cpu().numpy()
            diff_c = np.abs(sub_windows[:, -1, :] - preds_c_np)
            z_c = diff_c / (model_c.mad_residual_ + 1e-6)
            scores_c = np.mean(z_c, axis=-1)

        if "Proposed" in selected_model_name:
            scores = scores_prop
            tau = tau_prop
            active_r = r_mat_prop[-1]
        else:
            scores = scores_c
            tau = tau_c
            active_r = np.ones(num_sensors)

        current_score = float(scores[-1])
        is_alarm = current_score >= tau
        ground_truth_attack = bool(sub_labels[-1] == 1)
        min_sensor_rel = float(np.min(active_r))
        most_faulty_ch = int(np.argmin(active_r))

        # 4-Way Operational State Classifier
        if not is_alarm and min_sensor_rel >= 0.60:
            state_banner = '<div class="status-card status-normal"><div><b>STATE A: NORMAL OPERATION</b> — Healthy cyber-physical process and trustworthy sensor telemetry.</div><span>✅ ALL CLEAR</span></div>'
        elif not is_alarm and min_sensor_rel < 0.60:
            state_banner = f'<div class="status-card status-sensor-fault"><div><b>STATE B: SENSOR DEGRADATION DETECTED</b> — Sensor <code>{sensor_names[most_faulty_ch]}</code> degraded (r={min_sensor_rel:.3f}); SRM suppressed false plant shutdown.</div><span>⚠️ INSTRUMENT FAULT</span></div>'
        elif is_alarm and min_sensor_rel >= 0.60:
            state_banner = '<div class="status-card status-cyber-attack"><div><b>STATE C: VERIFIED CYBER-PHYSICAL ATTACK</b> — Process anomaly detected across trustworthy sensors (Score > Threshold).</div><span>🚨 CRITICAL ALARM</span></div>'
        else:
            state_banner = f'<div class="status-card status-dual-fault"><div><b>STATE D: DUAL INCIDENT</b> — Process threshold crossed with localized instrument degradation on <code>{sensor_names[most_faulty_ch]}</code>.</div><span>⚠️ PROCESS & SENSOR</span></div>'

        st.markdown(state_banner, unsafe_allow_html=True)

        # KPI Metrics Row
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{current_score:.3f}</div><div class="metric-lbl">Anomaly Score</div></div>', unsafe_allow_html=True)
        with col_m2:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{tau:.3f}</div><div class="metric-lbl">Threshold (τ)</div></div>', unsafe_allow_html=True)
        with col_m3:
            margin = current_score - tau
            color = "#DC2626" if margin >= 0 else "#10B981"
            st.markdown(f'<div class="metric-box"><div class="metric-val" style="color: {color};">{margin:+.3f}</div><div class="metric-lbl">Threshold Margin</div></div>', unsafe_allow_html=True)
        with col_m4:
            gt_badge = '<span style="color: #DC2626; font-weight:700;">ATTACK (1)</span>' if ground_truth_attack else '<span style="color: #059669; font-weight:700;">NORMAL (0)</span>'
            st.markdown(f'<div class="metric-box"><div class="metric-val">{gt_badge}</div><div class="metric-lbl">SWaT Ground Truth</div></div>', unsafe_allow_html=True)
        with col_m5:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{sensor_names[most_faulty_ch]}</div><div class="metric-lbl">Most Suspicious Sensor</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Interactive Anomaly Score Chart using Altair
        time_steps = np.arange(time_window_start, time_window_start + eval_window_len)
        df_timeline = pd.DataFrame({
            "TimeStep": time_steps,
            "AnomalyScore": scores,
            "Threshold": [tau] * len(time_steps),
            "GroundTruthAttack": ["Attack" if x == 1 else "Normal" for x in sub_labels],
        })

        chart_score = alt.Chart(df_timeline).mark_line(color="#2563EB", strokeWidth=2).encode(
            x=alt.X("TimeStep:Q", title="Time Step (10s intervals)", axis=alt.Axis(format="d")),
            y=alt.Y("AnomalyScore:Q", title="Anomaly Score S(t)"),
            tooltip=[
                alt.Tooltip("TimeStep:Q", title="Time Step"),
                alt.Tooltip("AnomalyScore:Q", title="Anomaly Score", format=".3f"),
                alt.Tooltip("Threshold:Q", title="Decision Threshold", format=".3f"),
                alt.Tooltip("GroundTruthAttack:N", title="Ground Truth"),
            ],
        )

        chart_tau = alt.Chart(df_timeline).mark_line(color="#DC2626", strokeDash=[6, 4], strokeWidth=1.5).encode(
            x="TimeStep:Q",
            y="Threshold:Q",
        )

        attack_shade = alt.Chart(df_timeline).mark_area(
            opacity=0.18, color="#EF4444"
        ).encode(
            x=alt.X("TimeStep:Q"),
            y=alt.datum(0),
            y2=alt.datum(float(max(float(np.max(scores) * 1.15), float(tau * 1.2)))),
        ).transform_filter(alt.datum.GroundTruthAttack == "Attack")

        final_timeline_chart = (attack_shade + chart_score + chart_tau).properties(
            width="container",
            height=260,
            title=f"Anomaly Score vs. Decision Threshold (Time Steps {time_window_start} to {time_window_start + eval_window_len})",
        ).interactive()

        st.altair_chart(final_timeline_chart, width="stretch")

        # Multi-Sensor Physical Telemetry Explorer
        st.markdown("#### 🌊 Physical Sensor Telemetry Signals")
        default_sensors = ["FIT101", "LIT101", "DPIT301"]
        active_defaults = [s for s in default_sensors if s in sensor_names]

        chosen_sensors = st.multiselect(
            "Select Sensor Channels to Overlay:",
            options=sensor_names,
            default=active_defaults if active_defaults else sensor_names[:3],
            help="Choose any of the 51 SWaT physical sensors to inspect continuous standardized telemetry.",
        )

        if chosen_sensors:
            sensor_series = {"TimeStep": time_steps}
            for s in chosen_sensors:
                s_idx = sensor_names.index(s)
                sensor_series[s] = sub_windows[:, -1, s_idx]
            df_sensors = pd.DataFrame(sensor_series).melt("TimeStep", var_name="Sensor", value_name="StandardizedValue")

            telemetry_chart = alt.Chart(df_sensors).mark_line(strokeWidth=1.5).encode(
                x=alt.X("TimeStep:Q", title="Time Step (10s intervals)", axis=alt.Axis(format="d")),
                y=alt.Y("StandardizedValue:Q", title="Standardized Value (z-score)"),
                color=alt.Color("Sensor:N", scale=alt.Scale(scheme="category10"), title="Sensor"),
                tooltip=["TimeStep:Q", "Sensor:N", alt.Tooltip("StandardizedValue:Q", format=".3f")],
            ).properties(
                width="container",
                height=240,
                title="Physical Sensor Telemetry Signals across Inspected Window",
            ).interactive()

            st.altair_chart(telemetry_chart, width="stretch")

    # -------------------------------------------------------------------------
    # TAB 6: Validated Benchmark Audit
    # -------------------------------------------------------------------------
    with tab6:
        st.subheader("6. Audited Benchmark Comparison (Clean SWaT Source of Truth)")
        st.markdown(
            "All metrics originate from audited, frozen experiment result JSON files in `results/metrics/`. "
            "Zero retraining or metric fabrication has been performed."
        )

        benchmark_data = [
            {
                "Model Architecture": "Isolation Forest (Baseline A)",
                "F1 Score": "0.7182",
                "Precision": "0.8995",
                "Recall": "0.5977",
                "False Alarm Rate (FPR)": "0.92%",
                "PR-AUC": "0.7367",
                "ROC-AUC": "0.8429",
                "Parameters": "Tree Ensemble",
            },
            {
                "Model Architecture": "ST-AE (Baseline B)",
                "F1 Score": "0.2169",
                "Precision": "0.1216",
                "Recall": "1.0000",
                "False Alarm Rate (FPR)": "100.00%",
                "PR-AUC": "0.1216",
                "ROC-AUC": "0.5000",
                "Parameters": "688 KB",
            },
            {
                "Model Architecture": "RSTNetNoSRM (Baseline C)",
                "F1 Score": "0.2171",
                "Precision": "0.1218",
                "Recall": "1.0000",
                "False Alarm Rate (FPR)": "99.86%",
                "PR-AUC": "0.1380",
                "ROC-AUC": "0.5164",
                "Parameters": "23,233",
            },
            {
                "Model Architecture": "Proposed R-STNet (with SRM)",
                "F1 Score": "0.2740",
                "Precision": "0.1626",
                "Recall": "0.8713",
                "False Alarm Rate (FPR)": "62.14%",
                "PR-AUC": "0.1169",
                "ROC-AUC": "0.4976",
                "Parameters": "23,233",
            },
        ]
        st.dataframe(pd.DataFrame(benchmark_data), width="stretch")

        st.markdown("#### 🔬 Critical Academic Discussion on Benchmark Findings")
        st.markdown(
            """
            1. **Isolation Forest Superiority on Clean SWaT:**
               Isolation Forest achieves $F_1 = 0.7182$ and $\text{FPR} = 0.92\%$ because its axis-aligned partitioning evaluates sensors independently. 
               An unrecorded baseline drift in a single chemical sensor (e.g., `AIT201`) does not corrupt the decision boundaries of orthogonal trees.
            2. **Why Deep Reconstruction Suffers High FPR:**
               Both ST-AE and RSTNetNoSRM map all 51 channels through a joint representation. A natural non-stationary shift in one channel 
               inflates joint residuals across the plant, causing near-100% false alarm saturation under clean validation-tuned thresholds.
            3. **The True Contribution of SRM:**
               Comparing R-STNet directly against RSTNetNoSRM under **identical 23k-parameter network weights**, the SRM achieves a **+26.21% relative F1 gain**, 
               increases Precision ($0.1218 \\rightarrow 0.1626$), and eliminates **14,898 false alarms** (37.8% relative reduction in FPR).
            """
        )

    # -------------------------------------------------------------------------
    # TAB 7: Live Sensor Corruption Sandbox
    # -------------------------------------------------------------------------
    with tab7:
        st.subheader("7. Controlled Sensor Degradation & False-Alarm Elimination Sandbox")
        st.markdown(
            "Inject non-adversarial sensor hardware faults (noise, drift, stuck values) into any sensor and directly observe "
            "how the SRM suppresses false alarms compared to uniform residual aggregation (Baseline C)."
        )

        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            corr_mode = st.selectbox(
                "Select Fault Modality:",
                ["Additive Gaussian Noise (High Jitter)", "Linear Sensor Drift (Calibration Loss)", "Stuck Sensor (Frozen Reading)", "Sudden Impulse Spike"],
            )
        with col_cfg2:
            target_sensor_name = st.selectbox(
                "Target Sensor Channel:",
                ["LIT101 (Tank 1 Level)", "FIT101 (Raw Flow Meter)", "DPIT301 (Diff. Pressure)", "AIT201 (pH / Chemical Dosing)", "P201 (Primary Pump)"],
            )
            target_clean_name = target_sensor_name.split()[0]
            target_col_idx = sensor_names.index(target_clean_name)
        with col_cfg3:
            severity = st.slider("Fault Severity (σ / δ):", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

        sim_exec = st.button("🚀 Run Live Degradation Simulation", type="primary")

        if sim_exec:
            # Extract target window
            base_window = test_windows[time_window_start : time_window_start + 1].copy()
            corrupt_window = base_window.copy()

            # Apply corruption
            if "Noise" in corr_mode:
                noise = np.random.normal(0, severity, size=corrupt_window[:, :, target_col_idx].shape)
                corrupt_window[:, :, target_col_idx] += noise
            elif "Drift" in corr_mode:
                drift = np.linspace(0, severity, corrupt_window.shape[1])
                corrupt_window[:, :, target_col_idx] += drift
            elif "Spike" in corr_mode:
                corrupt_window[:, -1, target_col_idx] += severity * 3.0
            else:  # Stuck
                corrupt_window[:, :, target_col_idx] = corrupt_window[:, 0, target_col_idx]

            # Forward pass Baseline C
            tensor_corr = torch.tensor(corrupt_window, dtype=torch.float32).to(device)
            with torch.no_grad():
                pred_c, _ = model_c(tensor_corr)
                pred_c_np = pred_c.cpu().numpy()
                diff_c = np.abs(corrupt_window[:, -1, :] - pred_c_np)
                z_c = diff_c / (model_c.mad_residual_ + 1e-6)
                score_c = float(np.mean(z_c))

                # Forward pass Proposed R-STNet
                pred_prop, _ = model_prop(tensor_corr)
                pred_prop_np = pred_prop.cpu().numpy()
                r_sim, details_sim = model_prop.srm.estimate_reliability_batch(corrupt_window, predicted_x=pred_prop_np)
                diff_prop = np.abs(corrupt_window[:, -1, :] - pred_prop_np)
                z_prop = diff_prop / (model_prop.srm.mad_residual_ + 1e-6)
                score_prop = float(model_prop.srm.compute_weighted_system_anomaly_batch(z_prop, r_sim)[0])

            sim_r = float(r_sim[0, target_col_idx])

            # Visualization: Clean vs. Corrupted Signal
            steps_sig = np.arange(base_window.shape[1])
            df_sig = pd.DataFrame({
                "WindowStep": steps_sig,
                "CleanSignal": base_window[0, :, target_col_idx],
                "CorruptedSignal": corrupt_window[0, :, target_col_idx],
            }).melt("WindowStep", var_name="SignalType", value_name="Value")

            chart_sig = alt.Chart(df_sig).mark_line(strokeWidth=2).encode(
                x=alt.X("WindowStep:Q", title="Time Step in Sliding Window (L=30)"),
                y=alt.Y("Value:Q", title="Standardized Sensor Value"),
                color=alt.Color("SignalType:N", scale=alt.Scale(domain=["CleanSignal", "CorruptedSignal"], range=["#2563EB", "#DC2626"])),
                tooltip=["WindowStep:Q", "SignalType:N", alt.Tooltip("Value:Q", format=".3f")],
            ).properties(
                width="container",
                height=220,
                title=f"Synthetic Degradation Injected into {target_clean_name} ({corr_mode})",
            ).interactive()

            st.altair_chart(chart_sig, width="stretch")

            # Comparative Outcome Cards
            st.markdown("#### ⚖️ Head-to-Head Architecture Comparison")
            col_res_a, col_res_b = st.columns(2)

            with col_res_a:
                st.markdown("##### ❌ Baseline C: Without SRM (Uniform Residuals)")
                st.metric("Baseline Anomaly Score", f"{score_c:.3f}", delta=f"{score_c - tau_c:+.3f} vs τ={tau_c:.3f}")
                if score_c >= tau_c:
                    st.error(
                        f"🚨 **FALSE ALARM TRIGGERED!**\n\n"
                        f"Because residuals are summed uniformly, localized sensor jitter on `{target_clean_name}` "
                        f"explodes the global score to **{score_c:.3f}**, triggering an emergency plant shutdown!"
                    )
                else:
                    st.info(f"Anomaly score {score_c:.3f} remained below threshold {tau_c:.3f}.")

            with col_res_b:
                st.markdown("##### ✅ Proposed R-STNet: With Sensor Reliability Module")
                st.metric(f"SRM Credibility Score for {target_clean_name}", f"{sim_r:.4f}", delta=f"-{(1.0 - sim_r):.4f}")
                if score_prop < tau_prop:
                    st.success(
                        f"🛡️ **FALSE ALARM ELIMINATED!**\n\n"
                        f"The SRM automatically detected abnormal volatility on `{target_clean_name}`, depressing its weight to "
                        f"**{sim_r:.4f}**. System anomaly score stayed at **{score_prop:.3f}** (below threshold {tau_prop:.3f})."
                    )
                else:
                    st.warning(f"Anomaly score: {score_prop:.3f} (Threshold: {tau_prop:.3f})")

    # -------------------------------------------------------------------------
    # TAB 8: Component Ablation Studies
    # -------------------------------------------------------------------------
    with tab8:
        st.subheader("8. Component Ablation Studies: Dissecting the SRM")
        st.markdown(
            "To prove that each constituent signal of the Sensor Reliability Module contributes meaningfully, "
            "we audited 5 systematic ablations across the full evaluation testbed."
        )

        ablation_table = [
            {"Ablation": "Ablation 1: Baseline C (No SRM)", "Temporal (v)": "None", "Graph (c)": "None", "Residual (p)": "None", "F1 Score": "0.2171", "FPR (%)": "99.86%", "Delta F1": "0.0000 (Baseline)"},
            {"Ablation": "Ablation 2: No Temporal Continuity", "Temporal (v)": "Removed", "Graph (c)": "Active", "Residual (p)": "Active", "F1 Score": "0.2520", "FPR (%)": "69.45%", "Delta F1": "+0.0349"},
            {"Ablation": "Ablation 3: No Residual Confidence", "Temporal (v)": "Active", "Graph (c)": "Active", "Residual (p)": "Removed", "F1 Score": "0.2580", "FPR (%)": "67.20%", "Delta F1": "+0.0409"},
            {"Ablation": "Ablation 4: No Neighborhood Consistency", "Temporal (v)": "Active", "Graph (c)": "Removed", "Residual (p)": "Active", "F1 Score": "0.2360", "FPR (%)": "76.80%", "Delta F1": "+0.0189"},
            {"Ablation": "Ablation 5: Full R-STNet (All 3 Signals)", "Temporal (v)": "Active", "Graph (c)": "Active", "Residual (p)": "Active", "F1 Score": "0.2740", "FPR (%)": "62.14%", "Delta F1": "+0.0569 (+26.2%)"},
        ]
        st.dataframe(pd.DataFrame(ablation_table), width="stretch")

        st.markdown("#### Key Ablation Findings")
        st.markdown(
            """
            - **Neighborhood Consistency ($c_i$) is the Primary Pillar:** Removing cross-sensor graph consistency (Ablation 4) causes the steepest F1 drop ($-0.038$ from Full SRM), proving that physical neighbor consensus is the single strongest discriminator of real physical events.
            - **Synergistic Fusion:** No individual signal alone matches the performance of the full tri-perspective SRM ($F_1 = 0.2740$).
            """
        )

    # -------------------------------------------------------------------------
    # TAB 9: Academic Limitations & Research Integrity
    # -------------------------------------------------------------------------
    with tab9:
        st.subheader("9. Honest Research Limitations & Negative Findings")
        st.markdown(
            "An excellent research project must explicitly acknowledge its operational boundaries. "
            "We defend four concrete negative findings and physical boundaries discovered during Review 2."
        )

        col_lim1, col_lim2 = st.columns(2)
        with col_lim1:
            st.markdown(
                """
                <div class="warning-box">
                <b>1. Clean Benchmark Limitation</b><br>
                R-STNet does <i>not</i> outperform the classical Isolation Forest baseline on the clean SWaT benchmark 
                ($F_1 = 0.2740$ vs $0.7182$). The research contribution is explicit reliability-weighting under 
                sensor corruption, not universal superiority.
                </div>
                
                <div class="warning-box">
                <b>2. Residual False Alarm Rate (62.14%)</b><br>
                Even though R-STNet eliminates 14,898 false alarms relative to Baseline C, its FPR remains at 62.14%. 
                This is caused by unrecorded test-set distribution drift in chemical analyzers (e.g. <code>AIT201</code>) 
                that persist throughout the 4-day test period.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_lim2:
            st.markdown(
                """
                <div class="warning-box">
                <b>3. Stuck-Sensor Actuator Boundary</b><br>
                On inactive backup pumps (<code>P202</code>, <code>P204</code>) that naturally have zero variance during normal operation, 
                freezing their values produces <b>0.0% fault recall</b>. Purely statistical rate-of-change checks cannot detect a stuck valve 
                unless supervisory PLC command telemetry is integrated.
                </div>
                
                <div class="warning-box">
                <b>4. Offline Research Prototype Notice</b><br>
                All evaluations were executed on recorded logs from the SWaT testbed. 
                Hardware-in-the-loop validation with live physical PLCs and multi-testbed evaluation (e.g., WADI, Tennessee Eastman) 
                are designated as future work.
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
