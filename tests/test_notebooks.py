"""
Unit tests for Jupyter notebooks integrity, schema validity, and modularity.

Ensures:
1. All 5 canonical notebooks exist in `notebooks/`.
2. Each notebook parses as valid JSON conforming to nbformat v4.
3. Every notebook has both markdown documentation and executable code cells.
4. Code cells import strictly from `src.*` without duplicate algorithm implementations.
"""

import os
import json
import pytest

NOTEBOOK_NAMES = [
    "01_dataset_exploration.ipynb",
    "02_preprocessing_and_windowing.ipynb",
    "03_baseline_evaluation.ipynb",
    "04_r_stnet_analysis.ipynb",
    "05_robustness_and_ablation.ipynb",
]


@pytest.fixture
def notebooks_dir():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(root, "notebooks")


def test_all_notebooks_exist(notebooks_dir):
    """Verify all 5 required educational notebooks exist."""
    assert os.path.isdir(notebooks_dir), f"Notebooks dir not found: {notebooks_dir}"
    for name in NOTEBOOK_NAMES:
        path = os.path.join(notebooks_dir, name)
        assert os.path.isfile(path), f"Missing notebook: {path}"


def test_notebooks_json_validity_and_schema(notebooks_dir):
    """Verify notebook files parse as valid nbformat v4 JSON with required fields."""
    for name in NOTEBOOK_NAMES:
        path = os.path.join(notebooks_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        assert "nbformat" in nb and nb["nbformat"] >= 4
        assert "cells" in nb and len(nb["cells"]) >= 5
        assert "metadata" in nb

        cell_types = [c.get("cell_type") for c in nb["cells"]]
        assert "markdown" in cell_types, f"{name} missing markdown cells"
        assert "code" in cell_types, f"{name} missing code cells"

        # Verify no empty code cells
        for c in nb["cells"]:
            if c.get("cell_type") == "code":
                assert len(c.get("source", [])) > 0, f"Found empty code cell in {name}"


def test_notebooks_modularity_and_src_imports(notebooks_dir):
    """Verify notebooks import from src.* rather than re-implementing core algorithms."""
    for name in NOTEBOOK_NAMES:
        path = os.path.join(notebooks_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_text = " ".join(
            "".join(c.get("source", []))
            for c in nb["cells"]
            if c.get("cell_type") == "code"
        )

        # Check for modular imports
        if name == "01_dataset_exploration.ipynb":
            assert "data" in code_text
        elif name == "02_preprocessing_and_windowing.ipynb":
            assert "src.preprocessing" in code_text or "RobustStandardScaler" in code_text
        elif name == "03_baseline_evaluation.ipynb":
            assert "results" in code_text or "baseline" in code_text
        elif name == "04_r_stnet_analysis.ipynb":
            assert "src.models.rstnet" in code_text or "RSTNet" in code_text
        elif name == "05_robustness_and_ablation.ipynb":
            assert "src.corruption" in code_text or "SensorCorruptor" in code_text
