# AME descriptive geometry (Pass 2)

`morphometrics.ts` maps a single 478-point MediaPipe result into the existing
68-point correspondence in image pixels. It computes the same five 2D ratios
and three paired discrepancies as `core/frontal_assessment.py`, and a proper
rotation Procrustes distance from one of three **simulated** GPA consensus
shapes. The Python engine supplies those shapes through
`scripts/export_ame_references.py`; CI verifies the export and a Python-made
cross-language example. The module does not implement the Python tangent PCA,
covariance, Mahalanobis distance, or photographic pose correction.

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

Regenerate on a pinned Python environment:

```bash
python scripts/export_ame_references.py --write
python scripts/export_ame_references.py --check
npm test
```
