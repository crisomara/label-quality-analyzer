"""Live demo for the Label Quality Analyzer case study.

Loads only the bundled slice of the pipeline's output in demo_assets/ (built once,
offline, by scripts/build_demo_bundle.py) — no CIFAR-10 download, no torch, no
cleanlab at runtime. Precision/recall are computed live from the bundled ground-truth
join, not hardcoded, so the headline numbers are reproducible from what's on screen.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "demo_assets"

st.set_page_config(
    page_title="Label Quality Analyzer",
    page_icon="\U0001F50D",
    layout="wide",
)


@st.cache_data
def load_data():
    queue = pd.read_parquet(ASSETS / "queue.parquet")
    validation = pd.read_parquet(ASSETS / "test_validation.parquet")
    summary = json.loads((ASSETS / "summary.json").read_text())
    with np.load(ASSETS / "images.npz") as npz:
        images = {k: npz[k] for k in npz.files}
    return queue, validation, summary, images


queue_df, val_df, summary, images = load_data()


def get_image(global_id: int) -> Image.Image:
    arr = images[str(global_id)]
    return Image.fromarray(arr).resize((96, 96), Image.NEAREST)


def live_precision_recall(df: pd.DataFrame):
    reviewed_flagged = df[df["gt_reviewed"] & df["cl_flagged"]]
    confirmed = df[df["gt_confirmed_error"] == True]  # noqa: E712
    precision = reviewed_flagged["gt_confirmed_error"].sum() / len(reviewed_flagged)
    recall = confirmed["cl_flagged"].sum() / len(confirmed)
    return precision, recall, len(reviewed_flagged), int(reviewed_flagged["gt_confirmed_error"].sum()), \
        len(confirmed), int(confirmed["cl_flagged"].sum())


def live_classifier_accuracy(df: pd.DataFrame):
    correct = df["true_label_name"] == df["predicted_label_name"]
    overall = correct.mean()
    per_class = (
        pd.DataFrame({"true_label_name": df["true_label_name"], "correct": correct})
        .groupby("true_label_name")["correct"].mean()
        .sort_values()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
    )
    return overall, per_class


st.title("Label Quality Analyzer")
st.caption(
    "A confident-learning QA pipeline that flags likely mislabeled CIFAR-10 images "
    "and scores its own output against a published, human-verified answer key. "
    "[Full case study](https://github.com/crisomara/label-quality-analyzer/blob/master/reports/case_study.md) · "
    "[Source](https://github.com/crisomara/label-quality-analyzer)"
)
st.info(
    "This project asks two separate questions, and it's easy to conflate them: "
    "**(1) how well does the model classify images?** (see the *Classifier "
    "performance* tab) and **(2) how well does an automated QA process catch "
    "mislabeled images in an already-clean dataset?** (see *Review queue explorer* "
    "and *Where the errors live*). A low QA precision/recall doesn't mean the "
    "classifier itself is bad — it means finding a 0.5%-error-rate needle in a "
    "haystack is a much harder task than ordinary classification.",
    icon="ℹ️",
)

tab_classifier, tab_queue, tab_findings = st.tabs(
    ["Classifier performance", "Review queue explorer", "Where the errors live"]
)

with tab_classifier:
    st.subheader("How well does the model classify images?")
    overall_test_acc, per_class_acc = live_classifier_accuracy(val_df)
    c1, c2 = st.columns(2)
    c1.metric(
        "Baseline accuracy (case study, full 60k, 5-fold CV)",
        f"{summary['baseline_accuracy_pct']}%",
        "frozen MobileNetV2 embeddings + logistic regression",
    )
    c2.metric(
        "Accuracy, live on this demo's test-set slice",
        f"{overall_test_acc:.1%}",
        f"{int((val_df['true_label_name'] == val_df['predicted_label_name']).sum()):,} / {len(val_df):,} correct",
    )
    fig_acc = px.bar(
        per_class_acc, x="accuracy", y="true_label_name", orientation="h",
        title="Per-class accuracy (test-set slice, computed live)",
        color_discrete_sequence=["#3E7C74"],
    )
    fig_acc.update_layout(
        xaxis_tickformat=".0%", yaxis_title="", xaxis_title="Accuracy",
        height=380,
    )
    st.plotly_chart(fig_acc, width="stretch")
    st.caption(
        "CPU-only, frozen embeddings, no fine-tuning — a fully fine-tuned CNN "
        "typically reaches 90-95%+ on CIFAR-10, so this is the honest cost of that "
        "trade-off, not a state-of-the-art result. **Cat and dog are the hardest "
        "classes** (69.5% / 75.7% here), **automobile and ship the easiest** "
        "(89.5% / 88.2%) — the classic animal-vs-vehicle split for CIFAR-10. This "
        "per-class weakness is also why cat/dog dominate the flagged review queue: "
        "see the *Where the errors live* tab."
    )

with tab_queue:
    st.subheader("How well does the QA process catch label errors?")
    precision, recall, n_reviewed_flagged, n_confirmed_in_reviewed, n_confirmed, n_confirmed_caught = \
        live_precision_recall(val_df)
    k1, k2, k3 = st.columns(3)
    k1.metric("Precision (reviewed pool)", f"{precision:.1%}", f"{n_confirmed_in_reviewed}/{n_reviewed_flagged}")
    k2.metric("Recall (known errors)", f"{recall:.1%}", f"{n_confirmed_caught}/{n_confirmed}")
    k3.metric("Images flagged", f"{summary['n_flagged']:,}", f"of {summary['n_total']:,} total")
    st.caption(
        "Computed live, in this app, by joining the review queue against the "
        "published `cleanlab/label-errors` ground truth bundled in `demo_assets/` — "
        "not hardcoded from the case study. Note this measures something different "
        "from classifier accuracy (previous tab): only 54 of CIFAR-10's 10,000 test "
        "images have a confirmed label error, so this is a needle-in-a-haystack "
        "search, not an ordinary classification task."
    )

    st.subheader("Prioritized review queue")
    st.caption(
        "The pipeline's actual deliverable: every flagged image, ranked by confidence "
        "gap, with a review category — the triage list a QA analyst would hand to a "
        "labeling team."
    )

    classes = sorted(queue_df["true_label_name"].unique())
    categories = sorted(queue_df["review_category"].unique())

    fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 1])
    class_filter = fc1.multiselect("Given label", classes)
    category_filter = fc2.multiselect("Review category", categories)
    split_filter = fc3.selectbox("Split", ["all", "train", "test"])
    sort_by = fc4.selectbox("Sort by", ["Confidence gap (margin)", "Quality score (worst first)"])

    filtered = queue_df.copy()
    if class_filter:
        filtered = filtered[filtered["true_label_name"].isin(class_filter)]
    if category_filter:
        filtered = filtered[filtered["review_category"].isin(category_filter)]
    if split_filter != "all":
        filtered = filtered[filtered["split"] == split_filter]
    if sort_by == "Confidence gap (margin)":
        filtered = filtered.sort_values("margin", ascending=False)
    else:
        filtered = filtered.sort_values("cl_quality_score", ascending=True)

    st.write(f"{len(filtered):,} images match")

    if "page" not in st.session_state:
        st.session_state.page = 0
    if "selected_id" not in st.session_state:
        st.session_state.selected_id = None

    page_size = 18
    n_pages = max(1, (len(filtered) - 1) // page_size + 1)
    st.session_state.page = min(st.session_state.page, n_pages - 1)
    nav1, nav2, nav3 = st.columns([1, 2, 1])
    prev_clicked = nav1.button("← Previous", disabled=st.session_state.page == 0)
    next_clicked = nav3.button("Next →", disabled=st.session_state.page >= n_pages - 1)
    # Rerun immediately on click rather than falling through: st.button() only
    # returns True for the single rerun right after the click, so the label and
    # the other button's disabled state below would otherwise render with the
    # pre-click page number, one step behind the page_df computed further down.
    if prev_clicked:
        st.session_state.page -= 1
        st.rerun()
    if next_clicked:
        st.session_state.page += 1
        st.rerun()
    nav2.markdown(
        f"<div style='text-align:center'>Page {st.session_state.page + 1} of {n_pages}</div>",
        unsafe_allow_html=True,
    )

    page_df = filtered.iloc[
        st.session_state.page * page_size: (st.session_state.page + 1) * page_size
    ]

    if st.session_state.selected_id is not None:
        sel = queue_df[queue_df["global_id"] == st.session_state.selected_id]
        if len(sel):
            row = sel.iloc[0]
            with st.container(border=True):
                d1, d2 = st.columns([1, 3])
                d1.image(get_image(row["global_id"]), width=140)
                d2.markdown(f"**Given label:** {row['true_label_name']}")
                d2.markdown(f"**Model prediction:** {row['predicted_label_name']}")
                d2.markdown(f"**Confidence gap (margin):** {row['margin']:.3f}")
                d2.markdown(f"**Review category:** `{row['review_category']}`")
                gt_row = val_df[val_df["global_id"] == row["global_id"]]
                if len(gt_row) and bool(gt_row.iloc[0]["gt_reviewed"]):
                    confirmed = bool(gt_row.iloc[0]["gt_confirmed_error"])
                    d2.markdown(
                        f"**Ground truth (MTurk-verified):** "
                        f"{'✅ confirmed label error' if confirmed else '❌ not a confirmed error'}"
                    )
                else:
                    d2.markdown("**Ground truth:** not independently reviewed (train-set image)")
                if st.button("Close", key="close_detail"):
                    st.session_state.selected_id = None
                    st.rerun()

    cols = st.columns(6)
    for i, (_, row) in enumerate(page_df.iterrows()):
        with cols[i % 6]:
            st.image(get_image(row["global_id"]))
            st.caption(f"{row['true_label_name']} → {row['predicted_label_name']}")
            if st.button("Inspect", key=f"inspect_{row['global_id']}", width="stretch"):
                st.session_state.selected_id = int(row["global_id"])
                st.rerun()

with tab_findings:
    st.subheader("Which classes get confused")
    confirmed_only = val_df[val_df["gt_confirmed_error"] == True]  # noqa: E712
    pair_counts = (
        confirmed_only.groupby(["true_label_name", "gt_guessed_label_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )
    pair_counts["pair"] = pair_counts["true_label_name"] + " → " + pair_counts["gt_guessed_label_name"]
    fig = px.bar(
        pair_counts, x="count", y="pair", orientation="h",
        title="Top confirmed-error label pairs (given → MTurk-guessed true label)",
        color_discrete_sequence=["#B5652A"],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "cat→dog and automobile→truck were the hypothesized confusions going in; "
        "airplane↔ship shows up too, likely driven by shared sky/water backgrounds "
        "rather than the object itself — see the case study's “what surprised us” section."
    )

    st.subheader("Does correcting the flagged queue actually help?")
    impact = pd.DataFrame(summary["before_after_impact"])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=impact["k_corrected"], y=impact["acc_on_corrected_full_test"],
        mode="lines+markers", line_color="#3E7C74", name="Accuracy after correcting top-K",
    ))
    fig2.update_layout(
        title="Before/after retraining simulation: correcting the top-K flagged items",
        xaxis_title="Images corrected (K)", yaxis_title="Accuracy on corrected full test set",
        height=380,
    )
    st.plotly_chart(fig2, width="stretch")
    st.caption(
        "Correcting a small slice of the queue (up to 1,000 images) didn't move accuracy. "
        "Only correcting nearly the entire flagged set (5,320 images) produced a clear "
        "improvement — at ~20% flagged-item precision, a human review step matters more "
        "than naively auto-accepting the model's guess for a handful of top flags."
    )
