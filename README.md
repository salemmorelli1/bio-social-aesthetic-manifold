# bio-social-aesthetic-manifold

A serverless, browser-based demonstration of descriptive geometric
morphometrics. The application uses iterative Generalized Procrustes Analysis
(GPA), a 132-dimensional Kendall tangent representation, tangent-space PCA,
and shrinkage-regularized Mahalanobis distance to compare a 68-point landmark
configuration with explicitly simulated reference data.

The complete, page-controlled methods record is available as a
[27-page APA-style technical report](docs/bio_social_aesthetic_manifold_apa_report.pdf).

## Scientific scope

This repository implements a **synthetic demonstration**, not a biometric
population model.

- Reference A, Reference B, and the pooled reference are deterministic
  simulations.
- The application does not estimate male, female, sex, gender, ancestry,
  identity, health, attractiveness, or any other personal category.
- It does not rank configurations, produce population percentiles, prescribe
  coordinate changes, or optimize an observed configuration toward a reference.
- Residual arrows are descriptive differences between aligned configurations.
- Study-context sliders are exportable annotations and have no effect on any
  statistic.
- JPG, PNG, and WebP images can be displayed in a local preview workspace. An
  explicit button can run one-face landmark detection and a texture warp in
  browser memory. Pixels are not transmitted or stored by the application.
- The displayed 0–10 Geometric Displacement Index is only a bounded rescaling
  of partial Procrustes distance. It is not an attractiveness, quality,
  psychology, sociology, health, or identity score.

The word *aesthetic* remains in the repository name as project provenance; no
aesthetic response is operationalized by the software.

## Application capabilities

- Runs NumPy and SciPy locally through WebAssembly using Pyodide.
- Opens with an interactive photo-confirmation workspace supporting drag and
  drop, full-surface panning (including direct dragging on the image), zoom,
  90-degree rotation, replacement, removal, and image metadata.
- Uses the project's shared dark scientific design language: Inter interface
  text, Georgia display accents, cyan/amber highlights, wide desktop panels,
  and a safe-area-aware bottom navigation bar on phones.
- Separates Photo Preview, Shape Laboratory, and Methods into accessible,
  keyboard-navigable views.
- Provides three deterministic synthetic configurations for immediate use.
- Accepts JSON, CSV, and plain-text landmark configurations.
- Optionally detects one dense face mesh locally with a pinned MediaPipe Face
  Landmarker and samples it into the fixed 68-point correspondence.
- Enforces the conventional 68-point index topology.
- Fits a true multi-configuration GPA reference consensus.
- Removes translation, uniform scale, and proper planar rotation.
- Projects shape information into a deterministic 132-dimensional tangent
  basis constructed by modified Gram-Schmidt for the fitted consensus.
- Computes PCA scores and explained-variance ratios with a symmetric
  eigendecomposition.
- Estimates diagonal-target covariance shrinkage analytically and adds a small
  numerical ridge.
- Computes partial/full Procrustes and regularized Mahalanobis distances.
- Draws input, consensus, descriptive residual vectors, and an optional
  proportional warp mesh on an HTML canvas.
- Applies those residuals to photo texture with a bounded, piecewise-affine
  Delaunay-triangle warp, shown as Original, Split, or Warped, with
  independently switchable displacement arrows and triangulation mesh.
- Displays a neutral 0–10 Geometric Displacement Index with fixed mathematical
  endpoints, an in-sample simulated-reference position, and an explicit
  non-normative interpretation.
- Returns visible warnings when an input exceeds the simulated reference range
  or the declared small-distortion tangent-chart region.
- Explains Sample A, Sample B, Blend, every displayed metric, and the current
  run in plain language within the interface.
- Exports the analysis and study-context metadata as JSON.
- Deploys as a static GitHub Pages application without a server or database.

## Repository structure

```text
bio-social-aesthetic-manifold/
├── .github/workflows/deploy.yml   # GitHub Pages deployment
├── assets/
│   ├── css/main.css               # Responsive scientific interface
│   ├── js/app.js                  # Pyodide bridge and interface controller
│   └── js/photo-warp.mjs          # Local landmark adapter and texture warp
├── core/analytics.py              # NumPy/SciPy statistical engine
├── docs/bio_social_aesthetic_manifold_apa_report.pdf # 27-page report
├── docs/mathematical_theory.md    # Detailed mathematical specification
├── LICENSE                        # Project MIT license and attribution pointer
├── LICENSES/Apache-2.0.txt        # MediaPipe source license
├── requirements-dev.txt           # Pinned test/report authoring environment
├── scripts/build_apa_report.py    # Reproducible report generator
├── tests/test_analytics.py        # Property, regression, and contract tests
├── THIRD_PARTY_NOTICES.md         # Canonical-model provenance
├── index.html                     # Application entry point
└── README.md                      # Architecture and literature guide
```

The primary implementation files are
[`index.html`](index.html),
[`assets/css/main.css`](assets/css/main.css),
[`assets/js/app.js`](assets/js/app.js), and
[`core/analytics.py`](core/analytics.py). The extended derivations are in
[`docs/mathematical_theory.md`](docs/mathematical_theory.md), and the
page-controlled implementation narrative is in the
[APA report](docs/bio_social_aesthetic_manifold_apa_report.pdf).

## Built-in research inputs

- **Sample A** is one deterministic simulated configuration generated around
  Synthetic Template A with seed 90011.
- **Sample B** is a separate deterministic simulated configuration generated
  around the deliberately altered Synthetic Template B with seed 90021.
- **Blend** starts from a landmark-wise 55% Template A and 45% Template B
  combination, is centered and normalized, and then receives a new simulated
  perturbation with seed 90031.

These controls never blend the selected photo. The reference selector is also
separate: it chooses whether distances and covariance are calculated against
Reference A, Reference B, or the pooled 320-configuration reference.

## Runtime architecture

```mermaid
flowchart TB
    A["Local browser session"] --> B["Image input"]
    A --> C["Synthetic or coordinate input"]
    B --> D["Preview or explicit local landmarks"]
    C --> E["68-point validation"]
    D --> E
    E --> F["Pyodide statistical engine"]
    F --> G["Metrics, shape plot, and optional photo warp"]
```

1. `index.html` loads the three-view interface, the pinned Pyodide
   distribution, and the JavaScript bridge.
2. A selected image receives a temporary browser object URL and is shown in an
   isolated preview. No detection occurs until the user presses **Detect
   landmarks & render warp**.
3. On that explicit action, `assets/js/photo-warp.mjs` loads MediaPipe Face
   Landmarker 1.0.1, detects one dense mesh locally, and samples 68 points in
   the project's Dlib-style ordering. Blendshape and personal-attribute outputs
   are disabled.
4. `assets/js/app.js` loads NumPy and SciPy, fetches `core/analytics.py`, and
   executes it within Pyodide.
5. A JavaScript `Float64Array` containing 136 coordinate values is placed in
   the Pyodide global namespace with `pyodide.globals.set()`.
6. JavaScript calls `run_pipeline_from_js()` and parses its JSON result.
7. For a photo-derived configuration, unit-shape residuals are restored to
   pixels with the input centroid size and applied by a piecewise-affine mesh.
8. The browser renders aligned configurations, optional source-to-destination
   arrows over the photo warp, the neutral displacement index, distance
   statistics, PCA scores, and GPA diagnostics.
9. Neither images nor coordinate data are transmitted to an application
   server.

The browser requests Pyodide, NumPy, and SciPy assets from the pinned CDN when
the runtime initializes. The Face Landmarker code and model are requested only
on first photo-analysis use. Those resource requests are distinct from analysis
data: the application does not attach the photo or coordinates to them.

## Local setup

No Python or JavaScript package installation is needed to use the web
application. A local HTTP server is required because browsers normally block
`fetch()` for pages opened directly from `file://`.

### Python server

From the repository root:

```bash
python -m http.server 8000
```

Open `http://localhost:8000/` in a current browser. The first load downloads the
pinned Pyodide runtime and scientific packages and can take longer than later
loads.

### Alternative Node server

If Node.js is available:

```bash
npx --yes serve .
```

Use the local URL printed by the command.

### Rebuild the 27-page report

The checked-in PDF is ready to read without installing anything. To reproduce
it from the current analytics engine, install the authoring dependencies and
run the page-controlled builder from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python scripts/build_apa_report.py
```

The builder recalculates the worked Blend example, refuses to continue if its
verified results drift, checks that the output contains exactly 27 pages, and
prints the finished file's SHA-256 digest. It embeds Nimbus Roman when available
and otherwise aliases its report font names to ReportLab's metrically compatible
base-14 Times family.

## Input formats

Each coordinate-file input must contain exactly 68 ordered two-dimensional
landmarks, or 136 finite numbers in total.

### JSON matrix

```json
[
  [120.1, 212.7],
  [126.4, 225.0]
]
```

The abbreviated example shows the structure; a valid file contains all 68
rows.

### JSON object

```json
{
  "landmarks": [[120.1, 212.7], [126.4, 225.0]]
}
```

Again, the `landmarks` array must contain 68 complete rows.

### Delimited text

```csv
x,y
120.1,212.7
126.4,225.0
```

CSV, semicolon-separated, tab-separated, and whitespace-separated coordinates
are accepted. A single header row is allowed. Input files are limited to 1 MB.

## Dlib 68-point correspondence

| Index interval | Anatomical label used for correspondence |
|---:|---|
| 0–16 | Jawline |
| 17–21 | Right eyebrow |
| 22–26 | Left eyebrow |
| 27–30 | Nose bridge |
| 31–35 | Nose base |
| 36–41 | Right eye |
| 42–47 | Left eye |
| 48–59 | Outer lip |
| 60–67 | Inner lip |

The names identify point ordering; they do not imply native Dlib detector
output. Synthetic Template A is a centered, unit-scaled 2D projection of the
MediaPipe canonical face-model vertices at the same 68 adapter indices used in
the browser. The source is pinned to MediaPipe commit
`a908d668c730da128dfa8d9f6bd25d519d006692` and documented in
`THIRD_PARTY_NOTICES.md`. Template B is a deliberate synthetic deformation of
Template A. Neither template is a person, biometric population, biological
norm, or appearance standard.

For the optional photo route, `MEDIAPIPE_TO_DLIB_68` in
`assets/js/photo-warp.mjs` is a fixed topology adapter from selected dense-mesh
vertices to this order. It is an engineering correspondence, not native Dlib
detector output and not a claim that two landmark definitions are anatomically
identical.

## Statistical pipeline

### 1. Preshape normalization

For configuration \(X\in\mathbb R^{68\times2}\), let

\[
C=I-\frac{1}{68}\mathbf1\mathbf1^\mathsf T,
\qquad
Z=\frac{CX}{\|CX\|_F}.
\]

This removes translation and centroid size.

### 2. Generalized Procrustes Analysis

For simulated reference preshapes \(Z_1,\ldots,Z_n\), GPA minimizes

\[
\min_{M,R_1,\ldots,R_n}
\sum_{i=1}^{n}\|Z_iR_i-M\|_F^2,
\qquad R_i\in SO(2),\quad\|M\|_F=1.
\]

Given \(Z_i^\mathsf TM=U\Sigma V^\mathsf T\), the proper SVD rotation is

\[
R_i=U\operatorname{diag}\!\left(1,\det(UV^\mathsf T)\right)V^\mathsf T.
\]

The mean is recomputed and renormalized until its Frobenius change is below
\(10^{-10}\).

### 3. Kendall tangent coordinates

Translation, scale, and planar rotation remove four degrees of freedom. The
local dimension is therefore

\[
2(68)-4=132.
\]

For aligned unit preshape \(Y\), consensus \(M\), and a 132-column orthonormal
tangent basis \(B\), the engine uses the central projection

\[
z=B^\mathsf T\left[
\frac{\operatorname{vec}(Y)}
{\langle\operatorname{vec}(Y),\operatorname{vec}(M)\rangle}
-\operatorname{vec}(M)
\right].
\]

The engine constructs \(B\) by orthogonalizing the four similarity directions
and then scanning the 136 standard basis vectors in index order with modified
Gram-Schmidt and one reorthogonalization pass. This fixes the chart orientation
conditional on the fitted consensus. PCA eigenvector signs are fixed by making
each largest-magnitude loading positive; this resolves sign ambiguity but does
not claim whole-pipeline bitwise equality across arbitrary numerical stacks.

### 4. PCA

For reference tangent covariance \(S\),

\[
S=V\Lambda V^\mathsf T.
\]

`scipy.linalg.eigh` is used because \(S\) is symmetric. Eigenvalues are ordered
from largest to smallest, and the input score for component \(j\) is
\(v_j^\mathsf T(z-\bar z)\).

### 5. Covariance regularization

The synthetic reference covariance is

\[
\widehat\Sigma_\lambda
=(1-\widehat\lambda)S
+\widehat\lambda\operatorname{diag}(S)+\varepsilon I,
\]

where the Schäfer-Strimmer diagonal-target intensity is estimated as

\[
\widehat\lambda
=\min\!\left(1,
\max\!\left(0,
\frac{\sum_{i\ne j}\widehat{\operatorname{Var}}(s_{ij})}
{\sum_{i\ne j}s_{ij}^{2}}
\right)\right).
\]

The diagonal cancels from numerator and denominator because the target exactly
reproduces it. The pooled deterministic reference currently gives
\(\widehat\lambda=0.0320069\). Unlike a scalar-multiple identity target, this
estimator preserves coordinate-specific variances and attenuates rather than
erases estimated cross-landmark covariance. The ridge \(\varepsilon I\)
guarantees stable Cholesky factorization. Set
`COVARIANCE_SHRINKAGE_OVERRIDE` only for a declared sensitivity analysis.

### 6. Distances

The partial Procrustes distance is

\[
d_P=\|Y-M\|_F.
\]

If \(a=\langle\operatorname{vec}(Y),\operatorname{vec}(M)\rangle\), the full
distance is

\[
d_F=\sqrt{\max(0,1-a^2)}.
\]

The regularized Mahalanobis distance in tangent coordinates is

\[
D_M=\sqrt{(z-\bar z)^\mathsf T
\widehat\Sigma^{-1}(z-\bar z)}.
\]

The engine solves the quadratic form with a Cholesky factor rather than
forming \(\widehat\Sigma^{-1}\) explicitly.

### 7. Geometric Displacement Index

The interface places partial Procrustes distance on a fixed display interval:

\[
\operatorname{GDI}=10\min\left(1,\frac{d_P}{\sqrt 2}\right).
\]

For optimally aligned unit preshapes, 0 indicates coincidence and \(\sqrt2\)
is the maximum chord separation under this proper-rotation convention. Thus,
GDI 0 means no aligned displacement and GDI 10 means the algebraic maximum on
this normalization. It is a magnitude index only: higher never means better or
worse, and no psychology or sociology slider enters the formula.

### 8. Residuals, gradients, and photo rendering

The residual at landmark \(j\) is

\[
r_j=M_j-Y_j.
\]

The shape canvas can display \(r_j\) literally or magnify it for visibility.
For a photo-derived configuration, the residual is returned to input
orientation and pixels:

\[
\Delta x_j=s\,c(X)\,r_j,
\qquad x'_j=x_j+\Delta x_j,
\]

where \(s\) is the selected visualization scale and \(c(X)\) is centroid size.
Delaunay triangles define local affine maps from \(x_j\) to \(x'_j\). Eight
zero-displacement anchors surround the detected face region, individual shifts
are capped relative to face size, and a line search reduces \(s\) if any
triangle would reverse orientation. These safeguards affect rendering only;
the reported statistics never change.

For reference, the derivative of squared Mahalanobis distance is

\[
\nabla_zD_M^2=2\widehat\Sigma^{-1}(z-\bar z).
\]

The application deliberately does not optimize this expression. With no
validated response model or scientific estimand, minimization would merely
collapse configurations toward a simulated mean.

Full derivations and numerical safeguards are provided in
[`docs/mathematical_theory.md`](docs/mathematical_theory.md).

## Splines and kernels

Hastie, Tibshirani, and Friedman formulate a cubic smoothing spline as the
solution to

\[
\min_f\left\{
\sum_{i=1}^{n}[y_i-f(x_i)]^2
+\lambda\int[f''(t)]^2dt
\right\}.
\]

The residual term measures data fit, while the integrated squared second
derivative penalizes curvature. In a future longitudinal morphometric study,
this construction could smooth a tangent coordinate through time, provided
within-unit dependence and landmark measurement error were modeled.

A positive-semidefinite kernel satisfies

\[
\sum_{i=1}^{n}\sum_{j=1}^{n}a_i a_jK(x_i,x_j)\ge0
\]

for every finite collection and coefficient vector. Equivalently,
\(K(x,x')=\langle\phi(x),\phi(x')\rangle_{\mathcal H}\) in an RKHS. A Gaussian
kernel on tangent coordinates is

\[
K(z,z')=\exp[-\gamma\|z-z'\|_2^2].
\]

Splines and kernels are documented but not executed in the present
demonstration.

## Study-context metadata

The three sliders are included to preserve a clean distinction between
**measurement** and **modeling**:

| Field | Interface range | Analytic role |
|---|---:|---|
| Environmental Stress Index | 0.00–1.00 | Export annotation only |
| Pathogen Prevalence Index | 0.00–1.00 | Export annotation only |
| Operational Sex Ratio | 0.50–1.50 | Export annotation only |

No coefficient, interaction, gradient, prior, or covariance term is generated
from these sliders. A scientifically defensible population-level study would
require validated measurements, a defined outcome, a sampling frame,
hierarchical dependence, missing-data assumptions, uncertainty propagation,
and external validation.

## JSON output

The Python bridge returns these principal blocks:

- `reference`: simulation status, key, sample size, and deterministic seed;
- `landmark_schema`: exact inclusive index intervals;
- `input_geometry`: raw centroid and centroid size;
- `gpa`: convergence and ensemble-alignment diagnostics;
- `distances`: partial/full Procrustes and regularized Mahalanobis distances;
- `chart_diagnostics` and `warnings`: reference-range and tangent-chart checks;
- `reference_calibration`: explicitly in-sample positions within the simulated
  ensemble, not population inference;
- `geometric_displacement_index`: bounded value, formula, range, and explicit
  non-normative metadata;
- `tangent_space`: all 132 coordinates and the first ten PCA summaries;
- `residual_shape_difference`: aligned and input-orientation vectors;
- `covariance_diagnostics`: shrinkage, ridge, condition number, and retained
  off-diagonal structure;
- `interpretation`: explicit non-normative and non-prescriptive metadata.

Exported browser files additionally include the three study-context annotations,
source type, optional photo-render diagnostics, and an ISO 8601 UTC timestamp.
No image pixels are included.

## Validation

Run Python syntax validation:

```bash
python -m py_compile core/analytics.py
```

Run all property, regression, error-handling, and front-end contract tests:

```bash
python -m pytest tests/ -q
```

Run JavaScript syntax validation:

```bash
node --check assets/js/app.js
node --check assets/js/photo-warp.mjs
```

Serve the site and confirm that:

1. JPG, PNG, and WebP files appear immediately in Photo Preview;
2. zoom, pan, rotate, fit, replace, and remove work in the local preview;
3. the runtime completes all three initialization stages;
4. each built-in sample produces 132 tangent coordinates and 68 residuals;
5. changing Reference A/B/Pooled recomputes the metrics;
6. changing the warp/residual display scale changes only the drawing;
7. the Geometric Displacement Index equals
   \(10\min(1,d_P/\sqrt2)\) and is unchanged by context sliders;
8. after explicit local photo analysis, Original/Split/Warped modes display a
   texture render; the vector and mesh switches work independently; and
   changing the visualization scale redraws it without changing any metric;
9. dragging from either the image pixels or the surrounding stage pans the
   preview and does not trigger native browser image dragging;
10. malformed image or coordinate files produce a readable error;
11. exported JSON marks the reference as simulated and context as
   `annotation_only`.

The engine has also been checked for translation, uniform-scale, and
proper-rotation invariance to floating-point precision.

## GitHub Pages deployment

1. Create a GitHub repository named `bio-social-aesthetic-manifold`.
2. Copy this repository tree into it.
3. Commit and push to the `main` branch.
4. In **Settings → Pages**, select **GitHub Actions** as the source.
5. The workflow in `.github/workflows/deploy.yml` uploads the static repository
   and deploys it through the GitHub Pages environment.

The workflow grants `contents: read`, `pages: write`, and `id-token: write` and
uses deployment concurrency so an obsolete in-progress deployment is canceled
when a newer commit arrives.

## Browser and privacy notes

- A current Chromium, Firefox, or Safari browser with WebAssembly support is
  required.
- Pyodide is pinned to version 314.0.6 using the complete official jsDelivr
  path documented by the [Pyodide project](https://pyodide.org/en/stable/usage/quickstart.html).
- MediaPipe Tasks Vision is pinned to 1.0.1 and follows the official
  [Face Landmarker web guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js).
- The Content Security Policy limits executable resources to the repository and
  the pinned jsDelivr host; model fetches are limited to Google's model host.
- Image files are decoded through a temporary `blob:` URL, remain in browser
  memory, and are released when replaced, removed, or when the page closes.
- The preview accepts JPG, PNG, and WebP images up to 20 MB. SVG is excluded
  from the accepted image types.
- Photo pixels are decoded and, after explicit action, passed only to the local
  MediaPipe task and canvas renderer. The Python engine receives 136 numbers,
  not image pixels. The export includes a source label and render diagnostics,
  but never pixel data.
- Coordinate inputs are read through the browser File API and are not uploaded
  by application code.
- Nothing is stored in cookies, local storage, a database, or an analytics
  service.
- Export occurs only when the user presses **Export analysis JSON**.

## Limitations

- The synthetic ensembles cannot support empirical inference or claims about a
  real population.
- The analytic shrinkage intensity is estimated from simulated data and is not
  externally validated or cross-validated for an empirical target.
- Mahalanobis distances have no empirical chi-square calibration here. The
  held-out simulation test is only a numerical scale check.
- A 2D configuration is affected by pose, perspective, acquisition, and
  landmarking error.
- The method assumes complete homologous correspondence and no missing points.
- Tangent projection is local. The engine warns beyond its declared diagnostic
  cutoff, but very distant configurations may require another chart or an
  intrinsic method.
- The photo landmark adapter is a convenience visualization without a
  reliability study, uncertainty estimate, pose correction, repeated-measures
  model, causal model, outcome model, or external validation sample.
- Piecewise-affine warping can show seams or artifacts and is not a prediction,
  recommendation, treatment simulation, or depiction of an attainable result.

## Path from demonstration to empirical research

An empirical extension should not relabel these synthetic templates as
population estimates. It should instead:

1. define a target population, estimand, and inclusion criteria;
2. obtain appropriately consented, de-identified adult research data;
3. document landmark acquisition and inter-/intra-rater reliability;
4. separate training, validation, and locked external test samples;
5. estimate GPA consensus and covariance entirely within training folds;
6. tune shrinkage and dimension reduction without test-data leakage;
7. model participant, stimulus, rater, session, and cultural clustering when
   those levels exist;
8. propagate landmark uncertainty into downstream intervals;
9. control multiplicity for regionwise or componentwise inference;
10. report calibration, sensitivity analyses, missingness assumptions, and
    limitations before making population-level claims.

## Literature and implementation map

The bibliography is grouped by the four methodological areas requested for the
project. Each entry includes its role in the architecture.

### 1. Geometric Morphometrics and Procrustes Subspaces

- Adams, D. C., Rohlf, F. J., & Slice, D. E. (2004). Geometric morphometrics:
  Ten years of progress following the “revolution.” *Italian Journal of
  Zoology, 71*(1), 5–16.
  [https://doi.org/10.1080/11250000409356545](https://doi.org/10.1080/11250000409356545)
  — Historical synthesis of GPA-based morphometrics and the Stony Brook/SUNY
  research lineage. The associated
  [Stony Brook Morphometrics review](https://www.sbmorphometrics.org/review/review.html)
  describes centering, centroid-size scaling, and least-squares rotation.

- Bookstein, F. L. (1991). *Morphometric tools for landmark data: Geometry and
  biology*. Cambridge University Press. — Establishes landmark geometry,
  coordinate transformations, and thin-plate-spline interpretation.

- Dryden, I. L., & Mardia, K. V. (2016). *Statistical shape analysis: With
  applications in R* (2nd ed.). Wiley. — Formal source for preshape spheres,
  shape spaces, tangent coordinates, Procrustes distances, and statistical
  analysis of configurations.

- Gower, J. C. (1975). Generalized Procrustes analysis. *Psychometrika, 40*,
  33–51.
  [https://doi.org/10.1007/BF02291478](https://doi.org/10.1007/BF02291478)
  — Foundational generalized least-squares alignment formulation.

- Kendall, D. G. (1984). Shape manifolds, Procrustean metrics, and complex
  projective spaces. *Bulletin of the London Mathematical Society, 16*(2),
  81–121.
  [https://doi.org/10.1112/blms/16.2.81](https://doi.org/10.1112/blms/16.2.81)
  — Geometric foundation for quotient shape spaces and tangent approximations.

- Rohlf, F. J., & Slice, D. E. (1990). Extensions of the Procrustes method for
  the optimal superimposition of landmarks. *Systematic Zoology, 39*(1),
  40–59.
  [https://doi.org/10.2307/2992207](https://doi.org/10.2307/2992207)
  — Iterative consensus alignment used by the engine.

- Slice, D. E. (2007). Geometric morphometrics. *Annual Review of Anthropology,
  36*, 261–281.
  [https://doi.org/10.1146/annurev.anthro.34.081804.120613](https://doi.org/10.1146/annurev.anthro.34.081804.120613)
  — Reviews GPA, tangent-space analysis, allometry, asymmetry, semilandmarks,
  and singular covariance constraints.

- Zelditch, M. L., Swiderski, D. L., & Sheets, H. D. (2012). *Geometric
  morphometrics for biologists: A primer* (2nd ed.). Academic Press. — Practical
  experimental design, landmark acquisition, visualization, and biological
  interpretation.

### 2. High-Dimensional Dimension Reduction and Covariance Structures

- Bishop, C. M. (2006). *Pattern recognition and machine learning*. Springer.
  [https://doi.org/10.1007/978-0-387-45528-0](https://doi.org/10.1007/978-0-387-45528-0)
  — Probability, latent-variable models, PCA, kernels, graphical models,
  sampling, and approximate inference.

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of
  statistical learning: Data mining, inference, and prediction* (2nd ed.).
  Springer.
  [https://doi.org/10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  — Basis for the spline penalty, positive-semidefinite kernels, regularization,
  model assessment, and the bias–variance framing used in the documentation.

- Jolliffe, I. T. (2002). *Principal component analysis* (2nd ed.). Springer.
  [https://doi.org/10.1007/b98835](https://doi.org/10.1007/b98835)
  — Core eigenspace theory and interpretation for PCA.

- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for
  large-dimensional covariance matrices. *Journal of Multivariate Analysis,
  88*(2), 365–411.
  [https://doi.org/10.1016/S0047-259X(03)00096-4](https://doi.org/10.1016/S0047-259X(03)00096-4)
  — Motivates covariance shrinkage when dimension is large relative to sample
  size.

- May, P. (2021). *Methods for high-dimensional spatial data: Dimension
  reduction and covariance approximation* (Doctoral dissertation, South Dakota
  State University). Open PRAIRIE.
  [https://openprairie.sdstate.edu/etd2/223](https://openprairie.sdstate.edu/etd2/223)
  — Connects dimension reduction, spatial dependence, envelope models, Gaussian
  processes, and scalable covariance approximation.

- Schäfer, J., & Strimmer, K. (2005). A shrinkage approach to large-scale
  covariance matrix estimation and implications for functional genomics.
  *Statistical Applications in Genetics and Molecular Biology, 4*(1), Article
  32.
  [https://doi.org/10.2202/1544-6115.1175](https://doi.org/10.2202/1544-6115.1175)
  — Provides a practical statistical rationale for stabilizing high-dimensional
  covariance estimates.

- Wold, S., Sjöström, M., & Eriksson, L. (2001). PLS-regression: A basic tool of
  chemometrics. *Chemometrics and Intelligent Laboratory Systems, 58*(2),
  109–130.
  [https://doi.org/10.1016/S0169-7439(01)00155-1](https://doi.org/10.1016/S0169-7439(01)00155-1)
  — Reference for supervised covariance-based dimension reduction. PLS is not
  run here because the demonstration has no empirical response variable.

### 3. Distance Metrics, Multivariate Inference, and Neural Mapping

- Anderson, M. J. (2001). A new method for non-parametric multivariate analysis
  of variance. *Austral Ecology, 26*(1), 32–46.
  [https://doi.org/10.1111/j.1442-9993.2001.01070.pp.x](https://doi.org/10.1111/j.1442-9993.2001.01070.pp.x)
  — Permutation-based inference for multivariate distance structures.

- Klingenberg, C. P., & Monteiro, L. R. (2005). Distances and directions in
  multidimensional shape spaces: Implications for morphometric applications.
  *Systematic Biology, 54*(4), 678–688.
  [https://doi.org/10.1080/10635150590947258](https://doi.org/10.1080/10635150590947258)
  — Clarifies the geometry and interpretation of directions and distances in
  shape space.

- Mardia, K. V., Kent, J. T., & Bibby, J. M. (1979). *Multivariate analysis*.
  Academic Press. — Classical covariance, quadratic-distance, discrimination,
  and multivariate modeling foundations.

- McArdle, B. H., & Anderson, M. J. (2001). Fitting multivariate models to
  community data: A comment on distance-based redundancy analysis. *Ecology,
  82*(1), 290–297.
  [https://doi.org/10.1890/0012-9658(2001)082%5B0290:FMMTCD%5D2.0.CO;2](https://doi.org/10.1890/0012-9658(2001)082%5B0290:FMMTCD%5D2.0.CO;2)
  — Regression-style partitioning of multivariate distance matrices.

- Shehzad, Z., Kelly, C., Reiss, P. T., Craddock, R. C., Emerson, J. W.,
  McMahon, K., Copland, D. A., Castellanos, F. X., & Milham, M. P. (2014). A
  multivariate distance-based analytic framework for connectome-wide
  association studies. *NeuroImage, 93*, 74–94.
  [https://doi.org/10.1016/j.neuroimage.2014.02.024](https://doi.org/10.1016/j.neuroimage.2014.02.024)
  — Demonstrates MDMR for relating whole high-dimensional patterns to external
  variables through permutation inference.

- Tomlinson, C. E., Laurienti, P. J., Lyday, R. G., & Simpson, S. L. (2022). A
  regression framework for brain network distance metrics. *Network
  Neuroscience, 6*(1), 49–68.
  [https://doi.org/10.1162/netn_a_00214](https://doi.org/10.1162/netn_a_00214)
  — Shows how dependence among pairwise distances must be handled when relating
  complex networks to continuous or categorical covariates.

- Kanwisher, N., McDermott, J., & Chun, M. M. (1997). The fusiform face area: A
  module in human extrastriate cortex specialized for face perception. *Journal
  of Neuroscience, 17*(11), 4302–4311.
  [https://doi.org/10.1523/JNEUROSCI.17-11-04302.1997](https://doi.org/10.1523/JNEUROSCI.17-11-04302.1997)
  — Supports the broader premise that facial configurations are complex visual
  stimuli; it does not supply a value model or a morphometric population prior.

### 4. Density Estimation and Manifold Methods

- Belkin, M., & Niyogi, P. (2003). Laplacian eigenmaps for dimensionality
  reduction and data representation. *Neural Computation, 15*(6), 1373–1396.
  [https://doi.org/10.1162/089976603321780317](https://doi.org/10.1162/089976603321780317)
  — Graph-Laplacian construction for learning low-dimensional geometry.

- Berenfeld, C., Rosa, P., & Rousseau, J. (2024). Estimating a density near an
  unknown manifold: A Bayesian nonparametric approach. *The Annals of
  Statistics, 52*(5), 2081–2111.
  [https://doi.org/10.1214/24-AOS2423](https://doi.org/10.1214/24-AOS2423)
  — Bayesian mixture modeling for observations concentrated in a noisy tube
  around an unknown manifold, with adaptive contraction theory.

- Berry, T., & Sauer, T. (2017). Density estimation on manifolds with boundary.
  *Computational Statistics & Data Analysis, 107*, 1–17.
  [https://doi.org/10.1016/j.csda.2016.09.011](https://doi.org/10.1016/j.csda.2016.09.011)
  — Intrinsic-dimension kernel density estimation, boundary-direction
  estimation, and cut-and-normalize correction.

- Coifman, R. R., & Lafon, S. (2006). Diffusion maps. *Applied and Computational
  Harmonic Analysis, 21*(1), 5–30.
  [https://doi.org/10.1016/j.acha.2006.04.006](https://doi.org/10.1016/j.acha.2006.04.006)
  — Diffusion geometry for nonlinear coordinates and multiscale structure.

- Pelletier, B. (2005). Kernel density estimation on Riemannian manifolds.
  *Statistics & Probability Letters, 73*(3), 297–304.
  [https://doi.org/10.1016/j.spl.2005.04.004](https://doi.org/10.1016/j.spl.2005.04.004)
  — Formulates KDE with the geometry and volume measure of a known manifold.

- Silverman, B. W. (1986). *Density estimation for statistics and data
  analysis*. Chapman & Hall.
  [https://doi.org/10.1201/9781315140919](https://doi.org/10.1201/9781315140919)
  — Classical bandwidth, kernel, bias, variance, and boundary foundations.

- Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000). A global geometric
  framework for nonlinear dimensionality reduction. *Science, 290*(5500),
  2319–2323.
  [https://doi.org/10.1126/science.290.5500.2319](https://doi.org/10.1126/science.290.5500.2319)
  — Isomap and geodesic-distance preservation for nonlinear manifolds.

## License

The project source is released under the root [MIT License](LICENSE).
MediaPipe-derived canonical-model coordinates retain their Apache-2.0
attribution; see [Third-Party Notices](THIRD_PARTY_NOTICES.md) and the included
[Apache License 2.0](LICENSES/Apache-2.0.txt). Other third-party projects and
publications remain under their respective licenses and copyrights.
