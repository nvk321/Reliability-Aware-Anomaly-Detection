"""
SWaT Dataset Loader and Preprocessing Pipeline (Corrected & Chronologically Validated).

Accurately reconstructs the official iTrust SUTD SWaT A1/A2 December 2015 testbed:
1. Training Split: 7 Days of Continuous Normal Operation (496,800 rows, 22/12 to 28/12).
   - 80% Chronological Train Split (y = 0, 100% Normal)
   - 20% Chronological Validation Split (y = 0, 100% Normal, used for threshold tuning)
2. Test Split: 4 Days of Continuous Physical Cyber-Attack Testing (449,919 rows, 28/12 to 02/01).
   - Reconstructed by chronologically sorting normal intervals (395,298 rows) and attack intervals (54,621 rows).
   - Label distribution: ~12.14% Attack (y = 1), ~87.86% Normal (y = 0).
3. Leak-Free Scaling: RobustStandardScaler is fit strictly on the 80% train split.
"""

from typing import Tuple, List, Dict, Any, Optional
import os
import json
import csv
from datetime import datetime
import numpy as np

from src.preprocessing.scaler import RobustStandardScaler


class SWaTLoader:
    """
    Robust, leak-free loader for the Secure Water Treatment (SWaT) industrial testbed dataset.
    """

    DATE_FORMAT = "%d/%m/%Y %I:%M:%S %p"
    OFFICIAL_NORMAL_ROWS = 496800
    OFFICIAL_NORMAL_OFFSET = 395298
    OFFICIAL_TEST_ROWS = 449919

    def __init__(
        self,
        raw_dir: str = "reference/swat_data",
        processed_dir: str = "data/processed/SWaT",
        downsample_rate: int = 10,
        train_ratio: float = 0.80,
    ):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.downsample_rate = downsample_rate
        self.train_ratio = train_ratio

        self.sensor_names: List[str] = []
        self.scaler: Optional[RobustStandardScaler] = None

    def load_and_preprocess(
        self,
        force_recompute: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads SWaT, normalizes strictly on training split, and returns
        (train, val, test, test_labels).
        """
        train_path = os.path.join(self.processed_dir, "train.npy")
        val_path = os.path.join(self.processed_dir, "val.npy")
        test_path = os.path.join(self.processed_dir, "test.npy")
        test_labels_path = os.path.join(self.processed_dir, "test_labels.npy")

        if (
            not force_recompute
            and os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)
            and os.path.exists(test_labels_path)
        ):
            print(f"[SWaTLoader] Loading cached processed SWaT from {self.processed_dir}")
            train = np.load(train_path)
            val = np.load(val_path)
            test = np.load(test_path)
            test_labels = np.load(test_labels_path)

            names_path = os.path.join(self.processed_dir, "sensor_names.json")
            if os.path.exists(names_path):
                with open(names_path, "r", encoding="utf-8") as f:
                    self.sensor_names = json.load(f)
            return train, val, test, test_labels

        print(f"[SWaTLoader] Extracting official SWaT A1/A2 splits (Downsample: {self.downsample_rate}x)...")
        os.makedirs(self.processed_dir, exist_ok=True)

        normal_file = os.path.join(self.raw_dir, "normal.csv")
        attack_file = os.path.join(self.raw_dir, "attack.csv")

        if not os.path.exists(normal_file) or not os.path.exists(attack_file):
            raise FileNotFoundError(
                f"Missing SWaT raw files in {self.raw_dir}. Required: normal.csv, attack.csv"
            )

        # -------------------------------------------------------------
        # STEP 1: Extract Sensor Names & Normal Test vs Train Sequences
        # -------------------------------------------------------------
        print("  Reading normal.csv...")
        normal_test_records = []
        official_train_rows = []

        with open(normal_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            raw_header = next(reader)
            cleaned_header = [col.strip() for col in raw_header]
            self.sensor_names = cleaned_header[1:-1]
            num_sensors = len(self.sensor_names)

            # Part 1: First 395,298 rows belong to the 4-day attack test phase (Normal intervals)
            last_valid = [0.0] * num_sensors
            for i in range(self.OFFICIAL_NORMAL_OFFSET):
                row = next(reader)
                row_vals = []
                for col_idx, raw_val in enumerate(row[1:-1]):
                    s = raw_val.strip()
                    if s != "":
                        try:
                            v = float(s)
                            last_valid[col_idx] = v
                        except ValueError:
                            v = last_valid[col_idx]
                    else:
                        v = last_valid[col_idx]
                    row_vals.append(v)

                dt = datetime.strptime(row[0].strip(), self.DATE_FORMAT)
                normal_test_records.append((dt.timestamp(), row_vals, 0))

            print(f"  Extracted {len(normal_test_records)} normal intervals for test set.")

            # Part 2: Next 496,800 rows represent the official 7-day normal training run
            for i in range(self.OFFICIAL_NORMAL_ROWS):
                row = next(reader)
                row_vals = []
                for col_idx, raw_val in enumerate(row[1:-1]):
                    s = raw_val.strip()
                    if s != "":
                        try:
                            v = float(s)
                            last_valid[col_idx] = v
                        except ValueError:
                            v = last_valid[col_idx]
                    else:
                        v = last_valid[col_idx]
                    row_vals.append(v)
                official_train_rows.append(row_vals)

            print(f"  Extracted {len(official_train_rows)} rows for official 7-day training run.")

        # -------------------------------------------------------------
        # STEP 2: Extract Attack Test Records
        # -------------------------------------------------------------
        print("  Reading attack.csv...")
        attack_test_records = []
        with open(attack_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Header
            last_valid_atk = [0.0] * num_sensors
            for row in reader:
                if not row:
                    continue
                row_vals = []
                for col_idx, raw_val in enumerate(row[1:-1]):
                    s = raw_val.strip()
                    if s != "":
                        try:
                            v = float(s)
                            last_valid_atk[col_idx] = v
                        except ValueError:
                            v = last_valid_atk[col_idx]
                        else:
                            v = last_valid_atk[col_idx]
                    row_vals.append(v)

                dt = datetime.strptime(row[0].strip(), self.DATE_FORMAT)
                attack_test_records.append((dt.timestamp(), row_vals, 1))

        print(f"  Extracted {len(attack_test_records)} attack intervals for test set.")

        # -------------------------------------------------------------
        # STEP 3: Reconstruct Chronological Mixed Test Set
        # -------------------------------------------------------------
        all_test_records = normal_test_records + attack_test_records
        assert len(all_test_records) == self.OFFICIAL_TEST_ROWS, (
            f"Expected {self.OFFICIAL_TEST_ROWS} test rows, got {len(all_test_records)}"
        )
        print("  Sorting test set chronologically by timestamp...")
        all_test_records.sort(key=lambda x: x[0])

        test_data_full = np.array([r[1] for r in all_test_records], dtype=np.float32)
        test_labels_full = np.array([r[2] for r in all_test_records], dtype=np.int32)
        print(f"  Test set sorted: {len(test_data_full)} rows (Attacks: {test_labels_full.sum()} = {test_labels_full.mean()*100:.2f}%).")

        # -------------------------------------------------------------
        # STEP 4: Chronological Train / Val Split on Pure Normal Data
        # -------------------------------------------------------------
        train_data_full = np.array(official_train_rows, dtype=np.float32)
        n_train_total = len(train_data_full)
        n_train = int(n_train_total * self.train_ratio)

        raw_train = train_data_full[:n_train]
        raw_val = train_data_full[n_train:]
        raw_test = test_data_full

        # Downsample
        if self.downsample_rate > 1:
            raw_train = raw_train[::self.downsample_rate]
            raw_val = raw_val[::self.downsample_rate]
            raw_test = raw_test[::self.downsample_rate]
            test_labels = test_labels_full[::self.downsample_rate]
        else:
            test_labels = test_labels_full

        print(f"  Downsampled: Train={len(raw_train)}, Val={len(raw_val)}, Test={len(raw_test)}")

        # -------------------------------------------------------------
        # STEP 5: Strict Leak-Free Normalization
        # -------------------------------------------------------------
        # Fit scaler ONLY on raw_train (zero test or validation leakage)
        self.scaler = RobustStandardScaler()
        train = self.scaler.fit_transform(raw_train)
        val = self.scaler.transform(raw_val)
        test = self.scaler.transform(raw_test)

        assert not np.isnan(train).any(), "NaN in train"
        assert not np.isnan(val).any(), "NaN in val"
        assert not np.isnan(test).any(), "NaN in test"
        assert len(test) == len(test_labels)

        # -------------------------------------------------------------
        # STEP 6: Save Processed Outputs
        # -------------------------------------------------------------
        np.save(train_path, train)
        np.save(val_path, val)
        np.save(test_path, test)
        np.save(test_labels_path, test_labels)

        with open(os.path.join(self.processed_dir, "sensor_names.json"), "w", encoding="utf-8") as f:
            json.dump(self.sensor_names, f, indent=2)

        self.scaler.save_json(os.path.join(self.processed_dir, "scaler_params.json"))

        metadata = {
            "dataset": "SWaT_A1_A2_Dec2015",
            "downsample_rate": self.downsample_rate,
            "train_ratio": self.train_ratio,
            "num_sensors": len(self.sensor_names),
            "train_samples": int(len(train)),
            "train_anomalies": 0,
            "val_samples": int(len(val)),
            "val_anomalies": 0,
            "test_samples": int(len(test)),
            "test_anomalies": int(np.sum(test_labels)),
            "test_attack_rate": float(np.mean(test_labels)),
            "raw_official_train_rows": self.OFFICIAL_NORMAL_ROWS,
            "raw_official_test_rows": self.OFFICIAL_TEST_ROWS,
        }
        with open(os.path.join(self.processed_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[SWaTLoader] Successfully generated and saved processed datasets to {self.processed_dir}")
        return train, val, test, test_labels
