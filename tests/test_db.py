import pandas as pd

from src.db import get_connection, load_images, load_near_duplicates, load_review_queue


def _images_df():
    return pd.DataFrame(
        {
            "global_id": [0, 1],
            "split": ["train", "test"],
            "split_index": [0, 0],
            "true_label": [3, 5],
            "true_label_name": ["cat", "dog"],
            "predicted_label": [3, 3],
            "predicted_label_name": ["cat", "cat"],
            "pred_prob_true_label": [0.9, 0.1],
            "pred_prob_max": [0.9, 0.8],
            "margin": [0.0, 0.7],
            "cl_flagged": [False, True],
            "cl_quality_score": [0.9, 0.1],
            "gt_reviewed": [False, True],
            "gt_confirmed_error": [pd.NA, True],
            "gt_mturk_given_votes": [pd.NA, 1],
            "gt_guessed_label": [pd.NA, 3],
            "gt_guessed_label_name": [pd.NA, "cat"],
        }
    )


def test_round_trip_and_idempotent_reload(tmp_path):
    conn = get_connection(tmp_path / "lq.db")
    load_images(conn, _images_df())
    load_images(conn, _images_df())  # reload replaces, never duplicates

    rows = conn.execute(
        "SELECT global_id, cl_flagged, gt_reviewed, gt_confirmed_error, gt_mturk_given_votes "
        "FROM images ORDER BY global_id"
    ).fetchall()
    assert rows == [(0, 0, 0, None, None), (1, 1, 1, 1, 1)]


def test_review_queue_and_duplicates(tmp_path):
    conn = get_connection(tmp_path / "lq.db")
    load_images(conn, _images_df())
    load_review_queue(
        conn,
        pd.DataFrame({"review_priority": [1], "global_id": [1], "review_category": ["likely_mislabel"], "margin": [0.7]}),
    )
    load_near_duplicates(conn, pd.DataFrame(columns=["global_id_a", "global_id_b", "class_label", "cosine_similarity"]))
    load_near_duplicates(
        conn,
        pd.DataFrame({"global_id_a": [0], "global_id_b": [1], "class_label": [3], "cosine_similarity": [0.999]}),
    )

    joined = conn.execute(
        "SELECT q.review_priority, i.true_label_name FROM review_queue q JOIN images i USING (global_id)"
    ).fetchall()
    assert joined == [(1, "dog")]
    assert conn.execute("SELECT COUNT(*) FROM near_duplicates").fetchone() == (1,)
