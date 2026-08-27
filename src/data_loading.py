"""Loading CIFAR-10 and the published cleanlab/label-errors ground truth.

CIFAR-10 class order (standard, matches torchvision.datasets.CIFAR10.classes):
0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from torchvision.datasets import CIFAR10

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CIFAR_DIR = PROJECT_ROOT / "data" / "raw" / "cifar10"
GT_DIR = PROJECT_ROOT / "data" / "raw" / "label_errors_ground_truth"


def load_cifar10(root: Path = CIFAR_DIR):
    """Return (train_dataset, test_dataset) — torchvision CIFAR10, PIL images, no transform."""
    train_ds = CIFAR10(root=str(root), train=True, download=False)
    test_ds = CIFAR10(root=str(root), train=False, download=False)
    return train_ds, test_ds


def load_ground_truth(gt_dir: Path = GT_DIR):
    """Load the published cleanlab/label-errors CIFAR-10 test-set ground truth.

    Returns
    -------
    mturk_df : DataFrame indexed by `split_index` (0..9999, matches the CIFAR-10
        test set's native order) with the raw MTurk vote counts, cleanlab's 2021
        guessed label, and a `confirmed_error` boolean (majority of 5 MTurk
        workers disagreed with the dataset's given label).
    original_labels : np.ndarray, shape (10000,) — the given label for every
        test-set image in native order (used only as an index-alignment check
        against torchvision's own labels).
    """
    with open(gt_dir / "cifar10_mturk.json") as f:
        records = json.load(f)

    rows = []
    for r in records:
        given_votes = r["mturk"]["given"]
        rows.append({
            "split_index": r["id"],
            "given_original_label": r["given_original_label"],
            "given_original_label_name": r["given_original_label_name"],
            "our_guessed_label": r["our_guessed_label"],
            "our_guessed_label_name": r["our_guessed_label_name"],
            "mturk_given_votes": given_votes,
            "mturk_guessed_votes": r["mturk"]["guessed"],
            "mturk_neither_votes": r["mturk"]["neither"],
            "mturk_both_votes": r["mturk"]["both"],
            # Ground-truth label: majority (>=3 of 5) of MTurk workers disagreed
            # with the dataset's given label -> confirmed label error.
            "confirmed_error": given_votes < 3,
        })
    mturk_df = pd.DataFrame(rows).set_index("split_index").sort_index()

    original_labels = np.load(gt_dir / "cifar10_test_set_original_labels.npy")
    return mturk_df, original_labels


def build_master_dataframe(train_ds, test_ds, mturk_df: pd.DataFrame) -> pd.DataFrame:
    """One row per CIFAR-10 image (60,000 total) with a stable global_id.

    global_id: 0..49999 = train set (in torchvision order),
               50000..59999 = test set (in torchvision order == label-errors' `id`).
    """
    train_labels = np.array(train_ds.targets)
    test_labels = np.array(test_ds.targets)

    n_train, n_test = len(train_labels), len(test_labels)
    df = pd.DataFrame({
        "global_id": np.arange(n_train + n_test),
        "split": (["train"] * n_train) + (["test"] * n_test),
        "split_index": np.concatenate([np.arange(n_train), np.arange(n_test)]),
        "true_label": np.concatenate([train_labels, test_labels]),
    })
    df["true_label_name"] = df["true_label"].map(lambda i: CLASS_NAMES[i])

    # Attach ground-truth review status (test set only — that's what was audited
    # and human-verified by the cleanlab/label-errors project).
    df["gt_reviewed"] = False
    df["gt_confirmed_error"] = pd.NA
    df["gt_mturk_given_votes"] = pd.NA
    df["gt_guessed_label"] = pd.NA
    df["gt_guessed_label_name"] = pd.NA

    test_mask = df["split"] == "test"
    test_idx_in_df = df.index[test_mask]
    split_index_for_test = df.loc[test_mask, "split_index"].values

    reviewed_lookup = mturk_df.to_dict("index")
    for row_pos, s_idx in zip(test_idx_in_df, split_index_for_test):
        rec = reviewed_lookup.get(int(s_idx))
        if rec is not None:
            df.at[row_pos, "gt_reviewed"] = True
            df.at[row_pos, "gt_confirmed_error"] = bool(rec["confirmed_error"])
            df.at[row_pos, "gt_mturk_given_votes"] = int(rec["mturk_given_votes"])
            df.at[row_pos, "gt_guessed_label"] = int(rec["our_guessed_label"])
            df.at[row_pos, "gt_guessed_label_name"] = rec["our_guessed_label_name"]

    return df


def get_image_array(ds, split_index: int) -> np.ndarray:
    """Return the raw HxWx3 uint8 numpy array for one image in a torchvision CIFAR10 dataset."""
    return ds.data[split_index]
