import numpy as np
import pandas as pd

from src.qa import (
    build_qa_table,
    build_review_queue,
    detect_near_duplicates,
    find_label_issues,
    get_out_of_fold_pred_probs,
)


def _master_df(labels):
    return pd.DataFrame({"global_id": np.arange(len(labels)), "true_label": labels})


def test_build_qa_table_margin_and_flags():
    labels = np.array([0, 1, 1])
    pred_probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
    out = build_qa_table(_master_df(labels), pred_probs, np.array([False, True, False]), np.array([0.9, 0.2, 0.7]))

    assert out["predicted_label"].tolist() == [0, 0, 1]
    np.testing.assert_allclose(out["pred_prob_true_label"], [0.9, 0.2, 0.7])
    # margin = how much the model prefers its top guess over the given label
    np.testing.assert_allclose(out["margin"], [0.0, 0.6, 0.0])
    assert out["cl_flagged"].tolist() == [False, True, False]


def test_build_qa_table_does_not_mutate_input():
    df = _master_df(np.array([0, 1]))
    build_qa_table(df, np.eye(2), np.array([False, False]), np.ones(2))
    assert list(df.columns) == ["global_id", "true_label"]


def test_detect_near_duplicates_same_class_only():
    emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0001, 0.0],  # near-duplicate of 0, same class
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0001],  # identical direction to 0 but a different class
        ]
    )
    labels = np.array([0, 0, 1, 1])
    dups = detect_near_duplicates(emb, labels, global_ids=np.array([10, 11, 12, 13]))

    assert len(dups) == 1
    row = dups.iloc[0]
    assert (row["global_id_a"], row["global_id_b"], row["class_label"]) == (10, 11, 0)
    assert row["cosine_similarity"] > 0.995


def test_detect_near_duplicates_empty_result_has_columns():
    emb = np.eye(3)
    dups = detect_near_duplicates(emb, np.array([0, 0, 0]), np.arange(3))
    assert dups.empty
    assert list(dups.columns) == ["global_id_a", "global_id_b", "class_label", "cosine_similarity"]


def test_build_review_queue_orders_and_categorises():
    df = pd.DataFrame(
        {
            "global_id": [1, 2, 3, 4],
            "cl_flagged": [True, True, False, True],
            "margin": [0.10, 0.80, 0.99, 0.50],
        }
    )
    queue = build_review_queue(df, duplicate_ids={4})

    assert queue["global_id"].tolist() == [2, 4, 1]  # biggest margin first, unflagged dropped
    assert queue["review_priority"].tolist() == [1, 2, 3]
    assert queue["review_category"].tolist() == ["likely_mislabel", "duplicate", "ambiguous"]


def test_confident_learning_finds_planted_label_errors():
    rng = np.random.default_rng(0)
    n_per_class = 60
    centers = np.array([[0.0, 0.0], [6.0, 6.0], [0.0, 6.0]])
    x = np.vstack([c + rng.normal(scale=0.6, size=(n_per_class, 2)) for c in centers])
    labels = np.repeat(np.arange(3), n_per_class)
    noisy = labels.copy()
    planted = np.array([0, 1, 2, 60, 61, 120])
    noisy[planted] = (labels[planted] + 1) % 3

    pred_probs = get_out_of_fold_pred_probs(x, noisy, n_splits=3)
    assert pred_probs.shape == (len(x), 3)
    np.testing.assert_allclose(pred_probs.sum(axis=1), 1.0, rtol=1e-6)

    issue_mask, scores = find_label_issues(noisy, pred_probs)
    assert scores.shape == (len(x),)
    assert issue_mask[planted].sum() >= len(planted) - 1
    assert issue_mask.sum() <= len(planted) + 3
