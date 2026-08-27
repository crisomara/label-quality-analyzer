"""SQLite recordkeeping for the label-quality audit.

Schema
------
images            one row per CIFAR-10 image (60,000 rows): true label, model
                   prediction, confidence metrics, confident-learning flag, and
                   (for the 10,000 test-set images) the published ground-truth
                   review status from cleanlab/label-errors.
review_queue       the prioritized subset of `images` flagged by confident
                   learning, ranked by confidence gap, with a review category.
near_duplicates    pairs of images whose frozen embeddings are near-identical
                   (candidate duplicate images within the same class).
"""
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    global_id             INTEGER PRIMARY KEY,
    split                 TEXT NOT NULL,
    split_index           INTEGER NOT NULL,
    true_label            INTEGER NOT NULL,
    true_label_name       TEXT NOT NULL,
    predicted_label       INTEGER NOT NULL,
    predicted_label_name  TEXT NOT NULL,
    pred_prob_true_label  REAL NOT NULL,
    pred_prob_max         REAL NOT NULL,
    margin                REAL NOT NULL,
    cl_flagged            INTEGER NOT NULL,
    cl_quality_score       REAL NOT NULL,
    gt_reviewed           INTEGER NOT NULL,
    gt_confirmed_error    INTEGER,
    gt_mturk_given_votes  INTEGER,
    gt_guessed_label      INTEGER,
    gt_guessed_label_name TEXT
);

CREATE TABLE IF NOT EXISTS review_queue (
    review_priority   INTEGER PRIMARY KEY,
    global_id         INTEGER NOT NULL REFERENCES images(global_id),
    review_category   TEXT NOT NULL,
    margin            REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS near_duplicates (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    global_id_a         INTEGER NOT NULL REFERENCES images(global_id),
    global_id_b         INTEGER NOT NULL REFERENCES images(global_id),
    class_label         INTEGER NOT NULL,
    cosine_similarity   REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_images_flagged ON images(cl_flagged);
CREATE INDEX IF NOT EXISTS idx_images_split ON images(split);
CREATE INDEX IF NOT EXISTS idx_images_true_label ON images(true_label);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    return conn


def _to_nullable_int(series: pd.Series) -> pd.Series:
    return series.astype("Int64")


def load_images(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    cols = [
        "global_id", "split", "split_index", "true_label", "true_label_name",
        "predicted_label", "predicted_label_name", "pred_prob_true_label",
        "pred_prob_max", "margin", "cl_flagged", "cl_quality_score",
        "gt_reviewed", "gt_confirmed_error", "gt_mturk_given_votes",
        "gt_guessed_label", "gt_guessed_label_name",
    ]
    out = df[cols].copy()
    out["cl_flagged"] = out["cl_flagged"].astype(int)
    out["gt_reviewed"] = out["gt_reviewed"].astype(int)
    out["gt_confirmed_error"] = out["gt_confirmed_error"].map(
        lambda v: int(v) if pd.notna(v) else None
    )
    out["gt_mturk_given_votes"] = out["gt_mturk_given_votes"].map(
        lambda v: int(v) if pd.notna(v) else None
    )
    out["gt_guessed_label"] = out["gt_guessed_label"].map(
        lambda v: int(v) if pd.notna(v) else None
    )
    conn.execute("DELETE FROM images")
    out.to_sql("images", conn, if_exists="append", index=False)
    conn.commit()


def load_review_queue(conn: sqlite3.Connection, queue_df: pd.DataFrame) -> None:
    cols = ["review_priority", "global_id", "review_category", "margin"]
    conn.execute("DELETE FROM review_queue")
    queue_df[cols].to_sql("review_queue", conn, if_exists="append", index=False)
    conn.commit()


def load_near_duplicates(conn: sqlite3.Connection, dup_df: pd.DataFrame) -> None:
    conn.execute("DELETE FROM near_duplicates")
    if len(dup_df):
        dup_df[["global_id_a", "global_id_b", "class_label", "cosine_similarity"]].to_sql(
            "near_duplicates", conn, if_exists="append", index=False
        )
    conn.commit()
