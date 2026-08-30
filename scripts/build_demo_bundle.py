"""One-time, dev-machine-only export: bundles a small, self-contained slice of the
notebook pipeline's cached artifacts into demo_assets/ so the Streamlit demo can run
without CIFAR-10, torch, or cleanlab installed at deploy time.

Run once from a fresh checkout that still has data/cache/ and data/raw/cifar10/
populated (i.e. after running notebooks 01-03 locally). Not part of the deployed
app's runtime path.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import load_cifar10, get_image_array  # noqa: E402

OUT_DIR = PROJECT_ROOT / "demo_assets"
QUEUE_COLS = [
    "global_id", "split", "true_label_name", "predicted_label_name",
    "margin", "cl_quality_score", "review_category", "review_priority",
]
VALIDATION_COLS = [
    "global_id", "true_label_name", "predicted_label_name", "margin",
    "cl_flagged", "gt_reviewed", "gt_confirmed_error", "gt_guessed_label_name",
]


def main():
    OUT_DIR.mkdir(exist_ok=True)

    queue_df = pd.read_parquet(PROJECT_ROOT / "data/cache/queue_df.parquet")
    qa_df = pd.read_parquet(PROJECT_ROOT / "data/cache/qa_df.parquet")
    test_df = qa_df[qa_df["split"] == "test"].copy()

    queue_out = queue_df[QUEUE_COLS].copy()
    queue_out.to_parquet(OUT_DIR / "queue.parquet", index=False)

    validation_out = test_df[VALIDATION_COLS].copy()
    validation_out.to_parquet(OUT_DIR / "test_validation.parquet", index=False)

    needed_ids = sorted(set(queue_df["global_id"]) | set(test_df.loc[test_df["gt_reviewed"], "global_id"]))
    print(f"Exporting {len(needed_ids)} images...")

    train_ds, test_ds = load_cifar10()
    id_to_ds = {}
    for gid in needed_ids:
        row = qa_df.loc[qa_df["global_id"] == gid].iloc[0]
        ds = train_ds if row["split"] == "train" else test_ds
        id_to_ds[gid] = get_image_array(ds, int(row["split_index"]))

    np.savez_compressed(OUT_DIR / "images.npz", **{str(gid): arr for gid, arr in id_to_ds.items()})

    impact_df = pd.read_csv(PROJECT_ROOT / "reports/figures/04_impact_table.csv")
    summary = {
        "precision_reviewed_pct": 20.2,
        "precision_reviewed_frac": "23/114",
        "recall_known_errors_pct": 42.6,
        "recall_known_errors_frac": "23/54",
        "baseline_accuracy_pct": 82.9,
        "n_flagged": int(len(queue_df)),
        "n_total": int(len(qa_df)),
        "before_after_impact": impact_df.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    images_size_mb = (OUT_DIR / "images.npz").stat().st_size / 1e6
    print(f"Wrote demo_assets/: images.npz ({images_size_mb:.1f} MB), "
          f"queue.parquet ({len(queue_out)} rows), "
          f"test_validation.parquet ({len(validation_out)} rows), summary.json")


if __name__ == "__main__":
    main()
