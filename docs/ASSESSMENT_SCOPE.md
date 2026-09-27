# Frontal landmark assessment: definitions and limits

This is a research-oriented **measurement report**, not an attractiveness
rating, a clinical report, or a substitute for a human-reviewed facial
assessment. Its layout follows the useful principle of explaining individual
features and their evidence before drawing conclusions. It is independent of,
and not affiliated with, QOVES. Their publicly described service uses multiple
views and additional measurements that this static application does not have:
[QOVES service overview](https://www.qoves.com/) and
[QOVES press description](https://www.qoves.com/press-kit).

The original 27-page APA report documents the synthetic Generalized
Procrustes Analysis (GPA) engine. This addendum documents the **separate**
`core/frontal_assessment.py` measurement module. Neither the simulated GPA
reference distribution nor its 0–10 Geometric Displacement Index calibrates
the frontal report. The latter contains no aesthetic target, rank, score,
population percentile, or comparison with an ideal face.

## What is observed

The input is one Dlib-order 68-point configuration of image-plane `(x, y)`
coordinates. The browser obtains these from a deterministic synthetic sample,
a user-selected coordinate file, or an explicit, browser-local MediaPipe
landmark pass on one photo. Each metric is computed after centering and scaling
the configuration; distances and ratios are invariant to translation, uniform
scale, and in-plane rotation. This mathematical invariance does **not** remove
3D head-pose, lens-distance, perspective, expression, or detection error.

| Displayed quantity | Formula or points | Unit | Important limit |
|---|---|---|---|
| Inner-eye / eye width | `d(39,42) / mean(d(36,39), d(42,45))` | ratio | Uses visible 2D eye corners. |
| Nose / inner-eye span | `d(31,35) / d(39,42)` | ratio | Perspective changes projected nose size. |
| Mouth / nose width | `d(48,54) / d(31,35)` | ratio | Expression changes mouth span. |
| Mouth / jaw span | `d(48,54) / d(0,16)` | ratio | Jaw outline changes with pose and occlusion. |
| Lower visible-face fraction | Projected `(33→8) / (27→8)` along the nose-bridge-to-chin axis | ratio | No hairline: **not** classical facial thirds. |
| Eyes, jaw, mouth paired discrepancy | Mean reflected-pair distance divided by `d(0,16)`, multiplied by 100 | % of jaw span | Midline `(27→8)` and paired points are 2D; not a symmetry or beauty grade. |

`d(i,j)` is Euclidean distance between landmarks `i` and `j`. Reflected pairs
for the three regions are specified in `PAIRS` in the Python source. If a
required span is zero or smaller than `1e-12` of the unit-centroid-size shape,
that metric is **unavailable**. If the nose-base point is not between the
bridge and chin along the chosen axis, the lower-face fraction is unavailable.
The rest of a valid report remains visible. We do not impute missing metrics or
turn an unavailable value into zero.

## What remains unmeasured

- Frontal pose, neutral expression, image quality, landmark error, and camera
  distance are not automatically verified. A centered, frontal, evenly lit
  image taken from farther away is a better input than a close selfie.
- Side-profile projection and 3D shape cannot be recovered from this view.
- The 68-point schema has no reliable hairline landmark; the app cannot infer
  forehead thirds. It also does not assess skin texture, pigmentation, aging,
  or health.
- The ratio values have no empirical reference intervals or validated
  association with human judgments of beauty. There is no universal optimum,
  personalized treatment plan, or percentage of people who are "more attractive."

Camera perspective can materially change apparent facial proportions, as
shown by [Ward et al. (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5876805/).
Reviews of appearance perception address symmetry and averageness but do not
provide a universal conversion from these eight 2D measurements to an
individual beauty score; see
[Rhodes (2006)](https://pubmed.ncbi.nlm.nih.gov/16318594/).

## Requirements before normative interpretation

Any future scoring model would need a consented empirical sample spanning
relevant presentation conditions and voluntary population descriptors, a
prespecified target and rater protocol, measurement repeatability studies,
calibration and uncertainty estimates, and a genuinely independent holdout.
Selection and thresholds must be frozen before that holdout. Synthetic samples
cannot serve as beauty-label training data. The current project has none of
these validation ingredients and accordingly reports only observable geometry.

All selected images remain in browser memory; the local report includes
measurements and the selected source kind, not image pixels. JSON export is an
explicit user action and can contain the 68 input-derived measurements, so
handle a downloaded report as personal data when a real photo was analyzed.

## AME preview: additional projected measurements

The separate Next.js preview renders the eight definitions above and four
additional observations from its browser-local 478-point mesh. Coordinates
are converted from image-normalized x/y to pixels before any distance, so a
non-square image does not distort its ratios. Eye-corner angles are measured
relative to the line joining the two outer eye corners, which removes in-plane
roll from the reported angle but does not estimate 3D head pose.

| Quantity | MediaPipe indices and definition | Limit |
|---|---|---|
| Inner / outer eye span | `d(133,362) / d(33,263)` | Projected eye corners; no target range. |
| Viewer-left and viewer-right eye tilt | Signed medial-to-lateral corner angle relative to the outer-corner line; positive if the lateral corner is higher | Image roll is removed; yaw, pitch, perspective, and expression remain. |
| Lower outline / cheek width | `d(148,377) / d(234,454)` | The lower outline points are **not** anatomical Gonion; this is not a clinical bigonial ratio. |

The preview also shows partial Procrustes distance to the **simulated pooled**
GPA consensus. This is a shape-space displacement, not a population z-score,
attractiveness rating, or individual diagnostic. The 2D alignment removes
translation, uniform scale, and in-plane rotation but cannot undo pose and
camera perspective. MediaPipe identifies its landmark output as normalized
image coordinates: [FaceLandmarkerResult](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerResult).

The suggested `1:1:1` facial thirds, `0.38–0.42` intercanthal interval, and
sex-specific gonial-angle ideals are **not** built into the engine. Trichion,
Nasion, Articulare, and Gonion are not validated by this single projected mesh.
Nor is a fixed `±3°` yaw/pitch/roll gate or a focal-length estimator claimed:
no camera calibration or pose-error validation is available. A comparative
study by [Le et al. (2002)](https://pubmed.ncbi.nlm.nih.gov/11891603/) found
that equal facial profile thirds did not describe its sampled groups, which
illustrates why a classical canon cannot be substituted for a measured norm.
The public [QOVES press description](https://www.qoves.com/press-kit) describes
personal reports and evidence-cited planning; its proprietary internal
algorithms and reference data are not available to this project.
