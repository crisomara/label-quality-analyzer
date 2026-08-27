"""Confident-learning QA: cross-validated baseline classifier, cleanlab label-issue
flagging, near-duplicate detection, and review-queue construction."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def get_out_of_fold_pred_probs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Cross-validated logistic regression on frozen embeddings -> out-of-fold
    predicted class probabilities, shape (N, num_classes). This is the standard
    cleanlab-recommended way to get the well-calibrated `pred_probs` confident
    learning needs, without any data leakage (every prediction comes from a
    fold the example itself was held out of)."""
    clf = LogisticRegression(max_iter=2000, C=1.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pred_probs = cross_val_predict(
        clf, embeddings, labels, cv=skf, method="predict_proba", n_jobs=-1
    )
    return pred_probs


def find_label_issues(labels: np.ndarray, pred_probs: np.ndarray):
    """Run cleanlab confident learning. Returns (issue_mask, label_quality_scores)."""
    from cleanlab.filter import find_label_issues as _find
    from cleanlab.rank import get_label_quality_scores

    issue_mask = _find(labels=labels, pred_probs=pred_probs, filter_by="prune_by_noise_rate")
    quality_scores = get_label_quality_scores(labels=labels, pred_probs=pred_probs)
    return issue_mask, quality_scores


def build_qa_table(df: pd.DataFrame, pred_probs: np.ndarray, issue_mask: np.ndarray,
                    quality_scores: np.ndarray) -> pd.DataFrame:
    """Attach model predictions, confidence metrics, and CL flags to the master dataframe."""
    df = df.copy()
    true_label = df["true_label"].to_numpy()

    predicted_label = pred_probs.argmax(axis=1)
    pred_prob_true_label = pred_probs[np.arange(len(df)), true_label]
    pred_prob_max = pred_probs.max(axis=1)

    df["predicted_label"] = predicted_label
    df["pred_prob_true_label"] = pred_prob_true_label
    df["pred_prob_max"] = pred_prob_max
    # Confidence gap: how much more the model prefers its own top guess over
    # the label the dataset gave this image. This is the review-queue priority
    # signal -- a big gap means "model is confident this image is something
    # else entirely," a small gap means a borderline/ambiguous call.
    df["margin"] = df["pred_prob_max"] - df["pred_prob_true_label"]
    df["cl_flagged"] = issue_mask.astype(bool)
    df["cl_quality_score"] = quality_scores
    return df


def detect_near_duplicates(embeddings: np.ndarray, labels: np.ndarray,
                            global_ids: np.ndarray, threshold: float = 0.995,
                            max_pairs_per_class: int = 5000) -> pd.DataFrame:
    """Flag near-duplicate image pairs via cosine similarity of frozen embeddings,
    restricted to same-class pairs (duplicates almost always share a label, and
    this keeps the pairwise comparison tractable on CPU: per-class matrix
    multiply instead of a 60,000 x 60,000 comparison)."""
    norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    rows = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if len(idx) < 2:
            continue
        sub = norm[idx]
        sim = sub @ sub.T
        iu = np.triu_indices(len(idx), k=1)
        sim_vals = sim[iu]
        mask = sim_vals >= threshold
        if not mask.any():
            continue
        pair_i = idx[iu[0][mask]]
        pair_j = idx[iu[1][mask]]
        pair_sim = sim_vals[mask]
        order = np.argsort(-pair_sim)[:max_pairs_per_class]
        for k in order:
            rows.append({
                "global_id_a": int(global_ids[pair_i[k]]),
                "global_id_b": int(global_ids[pair_j[k]]),
                "class_label": int(cls),
                "cosine_similarity": float(pair_sim[k]),
            })
    cols = ["global_id_a", "global_id_b", "class_label", "cosine_similarity"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


def build_review_queue(df: pd.DataFrame, duplicate_ids: set) -> pd.DataFrame:
    """The actual deliverable: a prioritized queue of flagged items for a human
    labeling team, ranked by confidence gap (biggest disagreement first), with
    a review category per item."""
    flagged = df[df["cl_flagged"]].copy()

    def categorize(row):
        if row["global_id"] in duplicate_ids:
            return "duplicate"
        if row["margin"] < 0.15:
            return "ambiguous"
        return "likely_mislabel"

    flagged["review_category"] = flagged.apply(categorize, axis=1)
    flagged = flagged.sort_values("margin", ascending=False).reset_index(drop=True)
    flagged["review_priority"] = np.arange(1, len(flagged) + 1)
    return flagged
