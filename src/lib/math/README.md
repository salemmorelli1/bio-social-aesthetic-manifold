# AME descriptive geometry (Pass 2)

`morphometrics.ts` maps a single 478-point MediaPipe result into the existing
68-point correspondence in image pixels. It computes the same five 2D ratios
and three paired discrepancies as `core/frontal_assessment.py`, and a proper
rotation Procrustes distance from one of three **simulated** GPA consensus
shapes. The Python engine supplies those shapes through
`scripts/export_ame_references.py`; CI verifies the export and a Python-made
cross-language example. The module does not implement the Python tangent PCA,
covariance, Mahalanobis distance, or photographic pose correction.

The preview renders those eight measurements and four further image-plane
observations from the dense mesh: inner/outer eye span, two roll-relative eye
corner angles, and lower-outline/cheek width. Each is an uncalibrated projected
measurement with stated point indices and missingness. The lower-outline
points are not anatomical Gonion. The Python frontal module and the 27-page
GPA report keep their original eight-metric scope.

`simulated-references.json` contains algorithm-generated method examples with
sample sizes, seeds, engine version, and canonical mesh provenance. Keys `a`,
`b`, and `pooled` do **not** denote male/female groups, beauty ideals, clinical
standards, population percentiles, or observed people. Procrustes distance is
a geometric displacement, not an attractiveness score. A projected frontal
image and a relative mesh depth do not provide calibrated 3D anatomy, a
hairline landmark, or a verified anatomical sagittal plane. Hence no classical
facial thirds, fifths, clinical facial index, dimorphism, dermatology,
recommendation, or 0–100 aesthetic score is returned.

A real population comparison would require a licensed or consented empirical
cohort, documented recruitment and measurement protocols, provenance,
coverage and uncertainty analysis, independent validation, and a declared
scientific purpose. Adding a sex or ancestry label to this synthetic JSON is
not a valid substitute. A future UI should preserve the source and missingness
labels adjacent to every quantity and keep photo pixels local.

Large coordinate offsets are handled by translating by an observed point
before dividing by extent. This preserves all detail still representable in
float64; it cannot recover bits lost before the coordinates reached the app.

Regenerate on a pinned Python environment:

```bash
python scripts/export_ame_references.py --write
python scripts/export_ame_references.py --check
npm test
```
