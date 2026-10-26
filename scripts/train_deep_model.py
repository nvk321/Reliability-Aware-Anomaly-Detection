"""
Training and Evaluation Driver for Deep Anomaly Detection Models on SWaT.

Supports:
- Baseline B: SpatioTemporal Autoencoder (ST-AE)
- Baseline C: RSTNet without Sensor Reliability Module (RSTNetNoSRM)
- Proposed:   R-STNet with Sensor Reliability Module (RSTNet)

Strict Protocol:
- Train split: 80% of normal run (39,744 samples downsampled 10x)
- Validation split: 20% of normal run (9,936 samples downsampled 10x)
- Test split: Complete chronological mixed attack run (44,992 samples downsampled 10x)
- Zero test label leakage during training, validation, or threshold calibration.
- Threshold tau calibrated strictly at 95th percentile on normal validation anomaly scores.
"""

from typing import Dict, Any, Tuple, Optional
import argparse
import sys
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.baselines.reconstruction_ae import SpatioTemporalAE
from src.models.baselines.rstnet_no_srm import RSTNetNoSRM
from src.models.rstnet import RSTNet
from src.evaluation.metrics import compute_anomaly_metrics, point_adjustment_eval


class TimeSeriesWindowDataset(Dataset):
    """Memory-efficient, zero-copy sliding window dataset."""
    def __init__(self, data: np.ndarray, window_size: int = 30):
        self.data = torch.from_numpy(data.astype(np.float32))
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        return self.data[idx : idx + self.window_size]


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    model_type: str,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch_x in loader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()

        if model_type == "stae":
            recon, _ = model(batch_x)
            loss = criterion(recon, batch_x)
        else:
            pred, _ = model(batch_x)
            target = batch_x[:, -1, :]
            loss = criterion(pred, target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_x)

    return total_loss / len(loader.dataset)


def compute_scores_batched(
    model: nn.Module,
    data_2d: np.ndarray,
    model_type: str,
    device: torch.device,
    window_size: int = 30,
    batch_size: int = 256,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    model.eval()
    dataset = TimeSeriesWindowDataset(data_2d, window_size=window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_scores = []
    all_preds = []

    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            if model_type == "stae":
                recon, _ = model(batch_x)
                diff = batch_x[:, -1, :] - recon[:, -1, :]
                sys_score = torch.mean(diff ** 2, dim=-1)
                all_scores.append(sys_score.cpu().numpy())
                all_preds.append(recon[:, -1, :].cpu().numpy())

            elif model_type in ["rstnet_no_srm", "rstnet"]:
                pred, _ = model(batch_x)
                all_preds.append(pred.cpu().numpy())

    total_eval_time = time.perf_counter() - start_time
    latency_ms_per_window = (total_eval_time / len(dataset)) * 1000.0

    if model_type == "rstnet":
        preds_np = np.concatenate(all_preds, axis=0)
        current_x = data_2d[window_size - 1:]
        raw_diff = np.abs(current_x - preds_np)
        if model.srm.mad_residual_ is not None:
            z = raw_diff / (model.srm.mad_residual_ + 1e-6)
        else:
            z = raw_diff

        from src.preprocessing.windowing import SlidingWindowGenerator
        win_gen = SlidingWindowGenerator(window_size=window_size, stride=1)
        windows, _ = win_gen.create_windows(data_2d)
        r, _ = model.srm.estimate_reliability_batch(windows, predicted_x=preds_np)
        scores_np = model.srm.compute_weighted_system_anomaly_batch(z, r)
        return scores_np, preds_np, latency_ms_per_window
    elif model_type == "rstnet_no_srm":
        preds_np = np.concatenate(all_preds, axis=0)
        current_x = data_2d[window_size - 1:]
        raw_diff = np.abs(current_x - preds_np)
        if hasattr(model, "mad_residual_") and model.mad_residual_ is not None:
            z = raw_diff / (model.mad_residual_ + 1e-6)
        else:
            z = raw_diff
        scores_np = np.mean(z, axis=-1)
        return scores_np, preds_np, latency_ms_per_window
    else:
        scores_np = np.concatenate(all_scores, axis=0)
        preds_np = np.concatenate(all_preds, axis=0)
        return scores_np, preds_np, latency_ms_per_window


def main():
    parser = argparse.ArgumentParser(description="Train deep anomaly detection models on SWaT")
    parser.add_argument("--model_type", type=str, required=True, choices=["stae", "rstnet_no_srm", "rstnet"])
    parser.add_argument("--data_dir", type=str, default="data/processed/SWaT")
    parser.add_argument("--output_dir", type=str, default="results/metrics")
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints")
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_quantile", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"\n=======================================================")
    print(f"TRAINING {args.model_type.upper()} ON SWAT")
    print(f"Device: {device} | Seed: {args.seed} | Window: {args.window_size} | Epochs: {args.epochs}")
    print(f"=======================================================")

    # 1. Load Preprocessed 2D Data
    train_raw = np.load(os.path.join(args.data_dir, "train.npy"))
    val_raw = np.load(os.path.join(args.data_dir, "val.npy"))
    test_raw = np.load(os.path.join(args.data_dir, "test.npy"))
    test_labels_raw = np.load(os.path.join(args.data_dir, "test_labels.npy"))
    num_sensors = train_raw.shape[1]

    # 2. Window Dataset Setup (strict chronological, zero-copy)
    train_dataset = TimeSeriesWindowDataset(train_raw, window_size=args.window_size)
    num_train_windows = len(train_dataset)
    num_val_windows = len(val_raw) - args.window_size + 1
    num_test_windows = len(test_raw) - args.window_size + 1
    test_labels = test_labels_raw[args.window_size - 1 :]

    print(f"  Train windows: ({num_train_windows}, {args.window_size}, {num_sensors}) (0% anomaly)")
    print(f"  Val windows:   ({num_val_windows}, {args.window_size}, {num_sensors}) (0% anomaly)")
    print(f"  Test windows:  ({num_test_windows}, {args.window_size}, {num_sensors}) (Attacks: {test_labels.sum()} = {test_labels.mean()*100:.2f}%)")

    # 3. Model Initialization
    if args.model_type == "stae":
        model = SpatioTemporalAE(
            in_channels=num_sensors,
            hidden_dim=64,
            latent_dim=32,
            num_layers=2,
            dropout=0.2,
        ).to(device)
        lr = args.lr if args.lr != 0.001 else 0.001
        criterion = nn.MSELoss()

    elif args.model_type == "rstnet_no_srm":
        model = RSTNetNoSRM(
            in_channels=num_sensors,
            hidden_dim=64,
            node_dim=32,
            top_k=15,
            temporal_kernel=3,
            dropout=0.2,
        ).to(device)
        lr = 0.0005
        criterion = nn.MSELoss()

    elif args.model_type == "rstnet":
        model = RSTNet(
            in_channels=num_sensors,
            hidden_dim=64,
            node_dim=32,
            top_k=15,
            temporal_kernel=3,
            dropout=0.2,
        ).to(device)
        lr = 0.0005
        criterion = nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {param_count:,}")

    # Reset GPU memory stats
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 4. Training Loop
    epoch_losses = []
    print(f"\nStarting {args.epochs} training epochs...")
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        ep_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.model_type, device)
        epoch_losses.append(ep_loss)
        print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] - Train Loss: {ep_loss:.5f}", flush=True)

    train_time = time.perf_counter() - train_start
    print(f"Training completed in {train_time:.2f} seconds.", flush=True)

    # 5. SRM Calibration (Only for R-STNet) & MAD Calibration (for Baseline C)
    if args.model_type == "rstnet":
        print("\nCalibrating Sensor Reliability Module on pure normal validation windows...", flush=True)
        from src.preprocessing.windowing import SlidingWindowGenerator
        win_gen = SlidingWindowGenerator(window_size=args.window_size, stride=1)
        val_windows, _ = win_gen.create_windows(val_raw)
        model.calibrate_srm(val_windows, device=device)
        print("  SRM calibration complete.", flush=True)
    elif args.model_type == "rstnet_no_srm":
        print("\nCalibrating validation residual normalization (MAD) for Baseline C...", flush=True)
        from src.preprocessing.windowing import SlidingWindowGenerator
        win_gen = SlidingWindowGenerator(window_size=args.window_size, stride=1)
        val_windows, _ = win_gen.create_windows(val_raw)
        model.calibrate_mad(val_windows, device=device)
        print("  Baseline C MAD calibration complete.", flush=True)

    # 6. Threshold Calibration on Pure Normal Validation Split (NO TEST LEAKAGE)
    print(f"\nComputing validation anomaly scores for threshold calibration (quantile={args.val_quantile})...", flush=True)
    val_scores, _, _ = compute_scores_batched(model, val_raw, args.model_type, device, window_size=args.window_size)
    tau = float(np.quantile(val_scores, args.val_quantile))
    print(f"  Calibrated decision threshold: tau = {tau:.5f}", flush=True)
    print(f"  Val scores - Mean: {np.mean(val_scores):.5f}, Std: {np.std(val_scores):.5f}, Max: {np.max(val_scores):.5f}", flush=True)

    # 7. Blind Evaluation on Mixed Chronological Test Set
    print("\nEvaluating test set with calibrated threshold...", flush=True)
    test_scores, _, latency_ms = compute_scores_batched(model, test_raw, args.model_type, device, window_size=args.window_size)
    test_preds = (test_scores >= tau).astype(np.int32)

    # 8. Compute Metrics
    raw_metrics = compute_anomaly_metrics(
        y_true=test_labels,
        y_scores=test_scores,
        y_pred=test_preds,
        threshold=tau,
    )
    pa_metrics = point_adjustment_eval(y_true=test_labels, y_pred=test_preds)

    # 9. GPU Footprint
    if torch.cuda.is_available() and device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
    else:
        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0

    # 10. Save Checkpoint & Metrics
    if args.model_type == "stae":
        model_name = "Baseline_B_SpatioTemporalAE"
        out_metric_name = "baseline_stae_swat.json"
        ckpt_name = "baseline_stae_swat.pt"
    elif args.model_type == "rstnet_no_srm":
        model_name = "Baseline_C_RSTNetNoSRM"
        out_metric_name = "baseline_rstnet_no_srm_swat.json"
        ckpt_name = "baseline_rstnet_no_srm_swat.pt"
    else:
        model_name = "Proposed_RSTNet_With_SRM"
        out_metric_name = "proposed_rstnet_swat.json"
        ckpt_name = "proposed_rstnet_swat.pt"

    ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
    extra_state = {}
    if hasattr(model, "mad_residual_") and model.mad_residual_ is not None:
        extra_state["mad_residual"] = model.mad_residual_
    if hasattr(model, "srm") and model.srm is not None:
        extra_state["srm_state"] = model.srm.to_dict()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "threshold_tau": tau,
            "args": vars(args),
            "epoch_losses": epoch_losses,
            **extra_state,
        },
        ckpt_path,
    )

    result = {
        "model": model_name,
        "model_type": args.model_type,
        "dataset": "SWaT_A1_A2_Dec2015",
        "downsample_rate": 10,
        "window_size": args.window_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": lr,
        "seed": args.seed,
        "parameter_count": param_count,
        "computational_cost": {
            "training_time_sec": float(train_time),
            "inference_latency_ms_per_window": float(latency_ms),
            "peak_allocated_vram_mb": float(peak_allocated_mb),
            "peak_reserved_vram_mb": float(peak_reserved_mb),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        },
        "split_sizes": {
            "train_windows": int(num_train_windows),
            "train_anomalies": 0,
            "val_windows": int(num_val_windows),
            "val_anomalies": 0,
            "test_windows": int(num_test_windows),
            "test_anomalies": int(test_labels.sum()),
            "test_attack_rate": float(test_labels.mean()),
        },
        "threshold_calibration": {
            "method": "validation_quantile",
            "val_quantile": args.val_quantile,
            "calibrated_threshold_tau": tau,
            "val_scores_mean": float(np.mean(val_scores)),
            "val_scores_std": float(np.std(val_scores)),
        },
        "training_convergence": {
            "initial_loss": float(epoch_losses[0]),
            "final_loss": float(epoch_losses[-1]),
            "epoch_losses": epoch_losses,
        },
        "raw_pointwise_metrics": raw_metrics,
        "labeled_point_adjusted_metrics": pa_metrics,
        "checkpoint_path": ckpt_path,
    }

    out_file = os.path.join(args.output_dir, out_metric_name)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print(f"RESULTS FOR {model_name}:")
    print("=" * 60)
    print(f"  Raw Point-wise F1:        {raw_metrics['f1']:.4f}")
    print(f"  Raw Point-wise Precision: {raw_metrics['precision']:.4f}")
    print(f"  Raw Point-wise Recall:    {raw_metrics['recall']:.4f}")
    print(f"  PR-AUC (Average Prec):    {raw_metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC:                  {raw_metrics['roc_auc']:.4f}")
    print(f"  False Positive Rate (FPR):{raw_metrics['fpr']:.4f}")
    print(f"  Confusion Matrix:         TP={raw_metrics['tp']}, FP={raw_metrics['fp']}, TN={raw_metrics['tn']}, FN={raw_metrics['fn']}")
    print(f"  Point-Adjusted F1 (PA):   {pa_metrics['f1_pa']:.4f} (Reference only)")
    print(f"  Training Wall Time:       {train_time:.2f} s")
    print(f"  Latency Per Window:       {latency_ms:.3f} ms")
    print(f"  Peak VRAM:                {peak_allocated_mb:.2f} MB")
    print(f"  Saved JSON:               {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
