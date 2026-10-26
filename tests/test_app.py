"""Smoke test: the deployed Streamlit demo renders from the committed demo_assets/."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "demo_assets"


def test_demo_assets_are_consistent():
    queue = pd.read_parquet(ASSETS / "queue.parquet")
    validation = pd.read_parquet(ASSETS / "test_validation.parquet")
    summary = json.loads((ASSETS / "summary.json").read_text())
    with np.load(ASSETS / "images.npz") as npz:
        image_ids = set(npz.files)

    assert summary["n_flagged"] == len(queue)
    assert set(queue["global_id"].astype(str)) <= image_ids
    reviewed = validation.loc[validation["gt_reviewed"], "global_id"].astype(str)
    assert set(reviewed) <= image_ids


def test_app_renders_without_exceptions():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
