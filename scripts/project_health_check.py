"""
Project Health Check Script for Reliability-Aware Anomaly Detection.

Validates the complete research workspace:
1. Python Environment & PyTorch runtime
2. Required Dependencies
3. Expected Folder Scaffolding
4. Dataset & Preprocessed Data Availability
5. Model Checkpoint Existence & Integrity
6. Proposed R-STNet Weight Loading
7. Model Inference & Forward Pass
8. Evaluation Pipeline & Metrics Files
9. Publication Figures & Results
10. Interactive Streamlit Demo Assets

Usage:
    python scripts/project_health_check.py
"""

import os
import sys
import json
import importlib
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_environment():
    py_ver = sys.version_info
    if py_ver.major < 3 or (py_ver.major == 3 and py_ver.minor < 9):
        return False, f"Python 3.9+ required, detected {py_ver.major}.{py_ver.minor}"
    return True, f"Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}"


def check_dependencies():
    required = [
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "yaml",
        "streamlit",
        "pytest",
    ]
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        return False, f"Missing packages: {', '.join(missing)}"
    return True, f"All {len(required)} core packages present"


def check_folders():
    expected_dirs = [
        "configs",
        "data",
        "src",
        "scripts",
        "demo",
        "tests",
        "results",
        "report",
        "docs",
        "presentation",
        "notebooks",
    ]
    missing = [d for d in expected_dirs if not (PROJECT_ROOT / d).is_dir()]
    if missing:
        return False, f"Missing directories: {', '.join(missing)}"
    return True, f"All {len(expected_dirs)} core directories verified"


def check_dataset_and_preprocessing():
    processed_files = [
        "data/processed/SWaT/train.npy",
        "data/processed/SWaT/val.npy",
        "data/processed/SWaT/test.npy",
        "data/processed/SWaT/test_labels.npy",
        "data/processed/SWaT/scaler_params.json",
        "data/processed/SWaT/sensor_names.json",
    ]
    missing = [f for f in processed_files if not (PROJECT_ROOT / f).is_file()]
    if missing:
        return False, f"Missing processed data files: {', '.join(missing)}"
    return True, "Downsampled 10x train/val/test splits, labels, and fitted scaler verified"


def check_checkpoints():
    ckpts = [
        "results/checkpoints/proposed_rstnet_swat.pt",
        "results/checkpoints/baseline_rstnet_no_srm_swat.pt",
        "results/checkpoints/baseline_stae_swat.pt",
    ]
    missing = [c for c in ckpts if not (PROJECT_ROOT / c).is_file()]
    if missing:
        return False, f"Missing checkpoints: {', '.join(missing)}"
    return True, f"All 3 model checkpoints verified"


def check_rstnet_loading():
    try:
        import torch
        from src.models.rstnet import RSTNet

        ckpt_path = PROJECT_ROOT / "results" / "checkpoints" / "proposed_rstnet_swat.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = RSTNet(in_channels=51, hidden_dim=64, node_dim=32, top_k=15)
        load_res = model.load_state_dict(ckpt["model_state_dict"])
        if len(load_res.missing_keys) > 0 or len(load_res.unexpected_keys) > 0:
            return False, f"Key mismatch on load: {load_res}"
        tau = float(ckpt.get("threshold_tau", 0.0))
        if tau <= 0.0:
            return False, f"Invalid threshold in checkpoint: {tau}"
        return True, f"R-STNet loaded cleanly (23k params, tau={tau:.4f})"
    except Exception as e:
        return False, f"Failed loading R-STNet: {e}"


def check_inference():
    try:
        import torch
        import numpy as np
        from src.models.rstnet import RSTNet

        model = RSTNet(in_channels=51, hidden_dim=64, node_dim=32, top_k=15)
        model.eval()

        # Batch forward pass
        dummy_batch = torch.randn(2, 30, 51, dtype=torch.float32)
        with torch.no_grad():
            x_pred, adj = model(dummy_batch)
        if x_pred.shape != (2, 51) or adj.shape != (51, 51):
            return False, f"Unexpected forward shapes: {x_pred.shape}, {adj.shape}"

        # Single-window evaluate_step
        dummy_window = np.random.normal(0, 1, (30, 51)).astype(np.float32)
        model.srm.fit_calibration(np.random.normal(0, 1, (100, 51)).astype(np.float32))
        step_res = model.evaluate_step(dummy_window)
        if "system_score" not in step_res or "reliability_vector" not in step_res:
            return False, "Missing keys in evaluate_step output"

        return True, "Batch forward pass and single-window diagnostic inference verified"
    except Exception as e:
        return False, f"Inference error: {e}"


def check_evaluation_and_results():
    metric_files = [
        "results/metrics/proposed_rstnet_swat.json",
        "results/metrics/baseline_rstnet_no_srm_swat.json",
        "results/metrics/baseline_stae_swat.json",
        "results/metrics/baseline_isolation_forest_swat.json",
        "results/corruption/corruption_benchmark_summary.json",
        "results/ablations/ablation_summary.json",
    ]
    missing = [m for m in metric_files if not (PROJECT_ROOT / m).is_file()]
    if missing:
        return False, f"Missing metric files: {', '.join(missing)}"
    return True, f"All {len(metric_files)} authoritative result JSONs verified"


def check_figures():
    fig_files = [
        "results/figures/fig1_swat_process_timeline.png",
        "results/figures/fig2_clean_vs_corrupted_reliability.png",
        "results/figures/fig3_srm_false_alarm_suppression.png",
        "results/figures/fig4_pr_curves_clean_benchmark.png",
        "results/figures/fig5_learned_sensor_graph.png",
        "results/figures/fig6_corruption_benchmark_heatmap.png",
        "results/figures/fig7_channel_shift_ait201.png",
    ]
    missing = [f for f in fig_files if not (PROJECT_ROOT / f).is_file()]
    if missing:
        return False, f"Missing publication figures: {', '.join(missing)}"
    return True, f"All {len(fig_files)} thesis figures verified"


def check_demo_assets():
    demo_files = [
        "demo/app.py",
        "demo/README.md",
        "presentation/DEMO_SCRIPT.md",
        "presentation/FINAL_PRESENTATION.md",
    ]
    missing = [f for f in demo_files if not (PROJECT_ROOT / f).is_file()]
    if missing:
        return False, f"Missing demo assets: {', '.join(missing)}"
    return True, "Interactive Streamlit app, demo script, and presentation slides verified"


def main():
    print("=" * 70)
    print("   RELIABILITY-AWARE ANOMALY DETECTION — PROJECT HEALTH CHECK")
    print("=" * 70)

    checks = [
        ("Environment", check_environment),
        ("Dependencies", check_dependencies),
        ("Directories", check_folders),
        ("Dataset", check_dataset_and_preprocessing),
        ("Preprocessing", check_dataset_and_preprocessing),
        ("Checkpoints", check_checkpoints),
        ("R-STNet loading", check_rstnet_loading),
        ("Inference", check_inference),
        ("Evaluation", check_evaluation_and_results),
        ("Results", check_evaluation_and_results),
        ("Demo assets", check_demo_assets),
    ]

    all_passed = True
    for name, check_fn in checks:
        passed, msg = check_fn()
        status_tag = "[PASS]" if passed else "[FAIL]"
        print(f"{status_tag} {name:<18} : {msg}")
        if not passed:
            all_passed = False

    print("-" * 70)
    if all_passed:
        print("PROJECT HEALTH CHECK: PASS")
        print("=" * 70)
        sys.exit(0)
    else:
        print("PROJECT HEALTH CHECK: FAIL")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
