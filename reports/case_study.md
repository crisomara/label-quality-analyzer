# Can an Automated Process Actually Catch Bad Labels?

*A confident-learning case study: flagging likely mislabeled CIFAR-10 images with
`cleanlab`, then checking the flags against a published, human-verified answer key.*

## The question

Every ML model is downstream of its training labels, and that part of the pipeline
is easy to take on faith. This project starts from a simple curiosity: how would you
actually know if your training data's labels are wrong, and can an automated QA
process be trusted to catch it? Confident learning (the technique behind `cleanlab`)
promises exactly that — flag likely label errors using nothing but a model's own
predicted probabilities. This case study builds that pipeline end to end and then
does the thing that's easy to skip: score its output against a real, independently
human-verified answer key, and report what actually happened, including the parts
that didn't go as expected.

## The data

- **CIFAR-10**: 60,000 32x32 images across 10 classes (50,000 train / 10,000 test),
  loaded via `torchvision.datasets.CIFAR10`.
- **[`cleanlab/label-errors`](https://github.com/cleanlab/label-errors)** (GitHub):
  the published, human-verified ground truth for CIFAR-10's test set, from the
  peer-reviewed paper *"Pervasive Label Errors in Test Sets Destabilize ML
  Benchmarks"* (Northcutt et al., 2021). The original 2021 cleanlab v1.0 run flagged
  **275** CIFAR-10 test images as candidate label errors; each was independently
  reviewed by 5 Amazon Mechanical Turk workers who voted on whether the dataset's
  given label was correct, the algorithm's guessed label was correct, neither, or
  both. **54** of the 275 were confirmed as real errors (majority of workers
  disagreed with the given label) — a number that matches what's publicly reported
  for this dataset, which was itself a first useful integrity check on this project's
  data loading (`notebooks/01_data_audit.ipynb`).

**Why CIFAR-10 instead of a more original dataset?** It's a well-worn benchmark, but
that's exactly the point: it's one of the only datasets where an independent,
peer-reviewed, human-verified answer key for label errors already exists.
A more original dataset would look more novel but this project would have no way to
check whether its own QA pipeline actually works — only to assert that it does.
Verifiability was chosen over novelty.

**Index alignment was verified directly, not assumed** (`notebooks/01_data_audit.ipynb`):
the `cleanlab/label-errors` `id` field is supposed to line up with torchvision's
native CIFAR-10 test-set order. Checked two ways — torchvision's own labels against
the repo's independent `original_test_labels` array, and every MTurk-reviewed
item's `given_original_label` against torchvision's label at that same index — both
came back with **0 mismatches out of 10,000 / 275**. If this had failed, every
downstream precision/recall number would have been comparing the wrong images.

## Building the QA pipeline

**Compute approach, CPU-only by design.** A full end-to-end CNN training loop
caused an out-of-memory crash on this same Windows machine during an earlier
project. To avoid repeating that, this pipeline never trains a CNN: it takes a
small ImageNet-pretrained model (MobileNetV2, 3,504,872 parameters), freezes it
completely, and does a single forward pass per image to get a 1280-dimensional
embedding. A cheap classifier (logistic regression) is then cross-validated on top
of those frozen embeddings to get the out-of-fold predicted probabilities confident
learning needs. Before writing any of this, a bare CPU-only `torch`/`torchvision`
install was verified end-to-end in a fresh venv — import, tensor ops, and a full
forward pass through pretrained MobileNetV2 — confirming no CUDA dependency and no
OOM, which is what justified committing to the approach at all.

**A real mid-project correction: native 32x32 input doesn't work well.**
MobileNetV2's global average pool technically accepts any input size, so feeding
CIFAR-10's native 32x32 images straight in is a valid forward pass. It's also a bad
one: a first full run at native resolution produced a baseline classifier accuracy
of only **58.5%**, and confident learning subsequently flagged **29.4%** of the
entire dataset — an implausibly high rate that signaled the underlying predicted
probabilities were too noisy to trust. A targeted diagnostic (`src/embeddings.py`,
also reproduced live in `notebooks/02_embeddings_and_baseline.ipynb`) confirmed why:
MobileNetV2's BatchNorm statistics were computed at 224x224, and 32x32 inputs sit
far enough outside that distribution to badly degrade the features.

| resize | accuracy (subsample, cross-validated) | extraction time (60k images, CPU) |
|---|---|---|
| 32x32 (native, no resize) | 48.9-50.2% | ~1 min |
| 96x96 | 75.1% | ~7 min |
| 128x128 | 77.1-79.1% | ~12 min |
| 224x224 (full ImageNet size) | 82.8% | ~39 min |

**128x128 was chosen** as the resize target for the full pipeline: it recovers most
of 224x224's accuracy gain at roughly a third of the compute cost. This is still a
fully frozen, CPU-only forward pass — resizing the input is a preprocessing step,
not fine-tuning, and the same discipline used to validate the CPU-only torch
install was applied again here before committing to a design choice.

**Final baseline, full 60,000-image dataset:** 5-fold cross-validated logistic
regression on the 128x128-resized MobileNetV2 embeddings reaches **82.86%**
out-of-fold accuracy (`notebooks/02_embeddings_and_baseline.ipynb`). A CNN trained
end-to-end on CIFAR-10 typically reaches 90-95%+; the gap is the honest cost of a
CPU-only, no-fine-tuning approach. **Cat (71.6%) and dog (77.3%) are the hardest
classes for this baseline classifier**; automobile (89.7%), ship (88.6%), and truck
(87.7%) are the easiest — a pattern that turns out to matter directly for where
confident learning ends up flagging images (see below).

**Confident learning, rule-based checks, and the review queue**
(`notebooks/03_confident_learning_qa.ipynb`): `cleanlab.filter.find_label_issues`
flagged **6,439 of 60,000 images (10.73%)** as likely mislabeled. Two supporting
rule-based checks ran alongside it: near-duplicate detection (cosine similarity
&ge;0.995 between same-class frozen embeddings) found **zero** near-duplicate pairs
in CIFAR-10 at this threshold — a real negative result, not a bug (near-identical
duplicate photos are simply not common in this dataset) — and a low-confidence
outlier check (predicted probability on the model's own top class below 0.3) as an
independent cross-check against the CL flag. All 6,439 flagged images, their
predictions, confidence metrics, and (for the 10,000 test images) the published
ground-truth review status were loaded into `data/label_quality.db`, a SQLite
database with three tables (`images`, `review_queue`, `near_duplicates`) — the
recordkeeping layer the rest of this analysis runs SQL against. The review queue
itself — the actual deliverable — ranks all 6,439 flagged images by **margin**
(the gap between the model's top predicted probability and the probability it
assigns the image's *given* label): a large gap means the model is confident the
image is something else entirely; a small gap (&lt;0.15) is downgraded to an
`ambiguous` review category rather than `likely_mislabel`.

## Does it actually work?

This is the project's central, load-bearing result
(`notebooks/04_validation_and_impact.ipynb`), computed with real SQL queries
against `data/label_quality.db`:

| Metric | Value | Definition |
|---|---|---|
| **Precision** (within the reviewed pool) | **20.2%** (23/114) | of flagged test images with known ground truth, how many are real confirmed errors |
| **Recall** (of all known errors) | **42.6%** (23/54) | of the 54 published confirmed errors, how many our pipeline also flagged |

Of the 10,000 test images, this pipeline flagged **1,119**. Ground truth only
exists for the 275 images the *original 2021* cleanlab v1.0 run flagged and sent to
MTurk — this project's pipeline is an independent re-implementation (different
embedding model, different classifier, different cleanlab version), so it
inevitably flags a different, only partially-overlapping set of images. Of the
1,119 flags, 114 fall inside that reviewed-275 pool, and 23 of those are confirmed
real errors — hence precision is reported *within the reviewed pool*, the only
place ground truth actually exists, rather than as an unverifiable "overall"
number.

**Reading the recall number honestly:** 42.6% means well over half of the known,
human-confirmed label errors in CIFAR-10's test set were *not* caught. That's a
real, meaningful limitation of this specific pipeline (CPU-only frozen embeddings +
linear classifier), not a data or measurement problem — see "What surprised us"
below for a specific, counterintuitive result about why a *more accurate* baseline
classifier actually caught *fewer* known errors than a noisier one.

### The 1,005 test-set flags with no ground-truth review

1,005 of the 1,119 test-set flags fall outside the reviewed-275 pool entirely —
there is no independent human judgment on record for them. A manual spot-check of
the top 10 (ranked by confidence gap, all with margin &ge;0.9999,
`reports/figures/04_unreviewed_flags_spotcheck.png`) shows a consistent pattern:
**automobile/truck boundary cases** (a vintage car shaped like a small truck, a
green pickup labeled "automobile," a red hot-rod truck guessed as "automobile") and
**context-driven confusions** where the background rather than the object seems to
be driving the model's guess (a truck against a plain sky/water background guessed
as "ship," a bird in flight silhouetted like small aircraft guessed as "airplane").
None of the 10 look like a blatant, obviously-wrong false alarm (e.g. an actual boat
labeled "bird") — they read as defensible disagreements about a fuzzy
category boundary or genuinely hard, ambiguous images, which is closer to the
plan's original hypothesis than the "does it work at all" precision number might
suggest on its own.

## Where the errors live

**The confusable-classes hypothesis holds, unevenly.** The hypothesis going in:
visually-confusable class pairs (cat/dog, automobile/truck, deer/horse,
airplane/bird) should show disproportionately higher error/flag rates.

- Among the **54 published, confirmed** label errors, cat&harr;dog is the single
  largest confusion pair (7 of 54), and automobile&harr;truck is second (5 of 54).
  **33.3%** of all 54 confirmed errors fall into one of the four hypothesized pairs.
- Among **our own 6,439 CL-flagged images**, the pattern is even sharper:
  dog&rarr;cat (560) and cat&rarr;dog (457) dominate, with truck&rarr;automobile
  (283) and automobile&rarr;truck (264) close behind. **54.6%** of our top-15
  confusion pairs by count fall into one of the four hypothesized pairs.
- **A pair nobody hypothesized shows up strongly in both**: airplane&harr;ship
  (4th-largest confirmed-error pair, and 5th/7th-largest in our own flags,
  ship&rarr;airplane 246 and airplane&rarr;ship 221). In hindsight this makes
  sense — both classes are commonly photographed against open sky or water, and at
  32x32 resolution a frozen ImageNet-pretrained CNN may be picking up on that shared
  background context as much as the object itself.

**But raw flag rate by class is misleading on its own, and it's important to say
so.** Cat (18.0% of all cat images flagged) and dog (15.55%) are by far the two
most-flagged classes — but cat and dog are also, coincidentally, the two *hardest*
classes for the baseline classifier itself (71.6% and 77.3% accuracy, vs. 87-90%
for automobile/ship/truck). Some of that elevated flag rate is very likely the
baseline classifier being weaker on cat/dog specifically (a harder visual category)
rather than purely reflecting a higher true rate of genuine label errors. The
ground truth partially corroborates a real signal regardless (cat&harr;dog is
still the largest *confirmed*-error pair), but the magnitude of the raw
flag-rate gap between classes should not be read as a direct, unbiased estimate of
where labeling actually went wrong most often — it's entangled with which classes
this particular CPU-only baseline classifier finds hardest.

**Confidence gap by class tells a related but distinct story**
(`reports/figures/04_margin_by_class.png`): among *flagged* images specifically,
automobile, truck, and ship average the largest confidence gaps (~0.89-0.92) —
when the model flags one of these, it's usually very sure. Cat and dog, despite
being flagged most *often*, have the smallest average confidence gap among their
flagged images (~0.75-0.79) — even when flagged, these calls tend to be more
borderline. So automobile/truck/ship get flagged less often, but with more
conviction; cat/dog get flagged constantly, but more equivocally.

## What surprised us

**A more accurate baseline classifier caught *fewer* known errors, not more.**
Before settling on the 128x128 resize, this pipeline was fully run once at native
32x32 resolution: a noisy 58.5%-accuracy baseline flagged 29.4% of the dataset and
recovered **62.96%** recall against the same 54 confirmed errors (34/54), at
similar precision (20.86%, 34/163). After fixing the resize issue, baseline
accuracy jumped to 82.9%, but recall *dropped* to 42.6% (23/54) with the pipeline
now flagging a far more conservative 10.7% of the dataset. A better classifier
disagrees with given labels *less* often overall — including, it turns out, for a
meaningful share of images where the disagreement would have been correct. This
is a genuine methodological lesson, not a reason to prefer the worse classifier:
confident learning's flag rate (and therefore its recall) is directly coupled to
how confidently the underlying classifier behaves, and a more accurate, better
calibrated classifier is not automatically a strictly better error-finder in every
respect — it trades recall for a much higher-precision-*feeling*, lower-volume flag
set that a human reviewer would actually be able to work through. Both runs are
real, honest results from the same pipeline design; only the embedding resolution
changed between them.

**Correcting a small slice of the review queue did *not* show a clean benefit —
only correcting nearly all of it did.** The plan's working hypothesis was that
correcting even a modest top-K slice of the review queue should yield a
disproportionate accuracy improvement. The actual before/after simulation
(retrain a fresh logistic regression on the training embeddings with the queue's
top-K items relabeled to the model's own prediction, evaluate on two untouched
test-set slices) tells a different story:

| items corrected (of 5,320 flagged train images) | accuracy, 221-image confirmed-clean subset | accuracy, full corrected test set |
|---|---|---|
| 0 (baseline) | 46.2% | 82.3% |
| 50 | 46.2% | 82.5% |
| 100 | 43.4% | 82.4% |
| 250 | 44.8% | 82.5% |
| 500 | 43.9% | 82.9% |
| 1,000 | 44.8% | 83.3% |
| 5,320 (all flagged train images) | **56.1%** | **86.8%** |

For every K from 50 to 1,000, accuracy on the small 221-image confirmed-clean
subset is flat or *slightly worse* than the uncorrected baseline — not the
disproportionate small-K win the plan hypothesized. Only correcting the *entire*
flagged training set (5,320 images, 10.6% of the 50,000-image training set)
produces a clear, sizable improvement on both eval sets (+9.9 points on the
221-image subset, +4.5 points on the full corrected test set). The most likely
explanation: with only 20.2% flagged-item precision, "correcting" a small,
high-margin top slice mixes a modest number of genuine fixes with a larger number
of confident-but-wrong relabelings, and at small K those two effects roughly wash
out (and the 221-image eval subset is small enough — a ~3.4-point standard error
at n=221 — that swings of a couple points between K values are within noise
either way). It's only once nearly the whole flagged pool is corrected that the
genuine fixes outweigh the noise introduced by the ~80% of flags that aren't real
errors. **The practical implication is not "QA doesn't help"** — the full-queue
correction clearly does — **it's that a human-in-the-loop review step (confirming
each flag before correcting it, rather than auto-accepting the model's guess) is
not optional at this pipeline's current precision level**; naively auto-correcting
a small top slice is not obviously better than doing nothing.

## Limitations & next steps

- **Single dataset, single domain.** Everything here is CIFAR-10 test/train
  images; the pipeline hasn't been checked against a second image dataset, let
  alone a non-image modality.
- **The correction simulation assumes perfect human relabeling accuracy for
  simplicity** (every "corrected" label is the model's own top guess, accepted
  without a real reviewer). The results above show exactly why that simplification
  matters: at this pipeline's ~20% flagged-item precision, a real labeling team
  reviewing each flag before correcting it — not auto-accepting the model's
  guess — would very likely outperform the naive top-K simulation, especially at
  small K.
- **Ground truth only exists for 275 of 60,000 images** (the ones the *original*
  2021 run happened to flag). Precision is necessarily measured within that pool
  only; the 1,005 unreviewed test-set flags (and every flagged train-set image)
  can only be spot-checked qualitatively, not scored.
- **No inter-annotator agreement tracking or SLA monitoring over time** — this is
  a static, one-shot dataset with no real labeling team or time dimension, so a
  production labeling-QA pipeline's ongoing-monitoring half (are labelers agreeing
  with each other over time? is queue turnaround meeting an SLA?) simply isn't
  something this project can demonstrate. That's the natural next piece: a live
  labeling pipeline with a real annotator pool, not a fixed benchmark.
- **The baseline classifier is a single linear model on frozen features** — no
  hyperparameter search on `C`, no comparison against a different frozen backbone
  (e.g. ResNet18), no ensembling. The 32x32-vs-128x128 result above suggests
  classifier quality has an outsized, non-obvious effect on confident learning's
  output; a systematic sweep over backbone/resolution/classifier choices would be
  the natural next step before trusting any single precision/recall number too
  literally.
