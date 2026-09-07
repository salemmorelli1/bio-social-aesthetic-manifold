"""Build the page-controlled APA-style technical report.

The report is deliberately generated from code so its page count, worked
example, equations, and figures remain synchronized with the application.
Run from the repository root with:

    python scripts/build_apa_report.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import reportlab
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.analytics import (  # noqa: E402
    SIMULATED_TEMPLATE_A,
    get_simulated_demo_json,
    run_pipeline_from_js,
)

OUTPUT = ROOT / "docs" / "bio_social_aesthetic_manifold_apa_report.pdf"
PAGE_WIDTH, PAGE_HEIGHT = letter
LEFT = 72.0
RIGHT = PAGE_WIDTH - 72.0
BODY_WIDTH = RIGHT - LEFT
FONT_DIRECTORY = Path("/usr/share/fonts")
NIMBUS_AFM_DIRECTORY = FONT_DIRECTORY / "type1" / "urw-base35"
NIMBUS_PFB_DIRECTORY = FONT_DIRECTORY / "X11" / "Type1"
BODY_FONT = "ReportNimbusRoman"
BODY_BOLD = "ReportNimbusRomanBold"
BODY_ITALIC = "ReportNimbusRomanItalic"
BODY_SIZE = 12.0
LEADING = 24.0
RUNNING_HEAD = "SHAPE MANIFOLD STATISTICS"

INK = colors.HexColor("#171717")
MUTED = colors.HexColor("#555555")
CYAN = colors.HexColor("#237A8A")
VIOLET = colors.HexColor("#6D55A3")
AMBER = colors.HexColor("#9B641D")
PALE_CYAN = colors.HexColor("#E9F5F7")
PALE_VIOLET = colors.HexColor("#F1EDF8")
GRID = colors.HexColor("#D8D8D8")


def register_report_fonts() -> None:
    """Embed a metrically stable Times-compatible family in the PDF."""
    variants = (
        (BODY_FONT, "NimbusRoman-Regular"),
        (BODY_BOLD, "NimbusRoman-Bold"),
        (BODY_ITALIC, "NimbusRoman-Italic"),
    )
    nimbus_sources = [
        (
            public_name,
            NIMBUS_AFM_DIRECTORY / f"{source_name}.afm",
            NIMBUS_PFB_DIRECTORY / f"{source_name}.pfb",
        )
        for public_name, source_name in variants
    ]
    if all(afm.exists() and pfb.exists() for _, afm, pfb in nimbus_sources):
        for public_name, afm, pfb in nimbus_sources:
            face = pdfmetrics.EmbeddedType1Face(str(afm), str(pfb))
            pdfmetrics.registerTypeFace(face)
            pdfmetrics.registerFont(
                pdfmetrics.Font(public_name, face.name, "WinAnsiEncoding")
            )
        return

    # Fallback. The page layout is tuned against Times metrics, and Nimbus Roman
    # is metrically compatible with Times, so the substitute must be too. The
    # bundled Bitstream Vera family is NOT: it is a sans face roughly eight
    # percent wider, which lengthens every wrapped line and overflows the body
    # margin on hosts without the URW fonts. ReportLab always ships the base-14
    # Type 1 metrics, so alias the public names onto Times instead.
    for public_name, core_name in (
        (BODY_FONT, "Times-Roman"),
        (BODY_BOLD, "Times-Bold"),
        (BODY_ITALIC, "Times-Italic"),
    ):
        pdfmetrics.registerFont(
            pdfmetrics.Font(public_name, core_name, "WinAnsiEncoding")
        )


def paragraph(text: str) -> dict[str, Any]:
    return {"kind": "paragraph", "text": " ".join(text.split())}


def heading(text: str) -> dict[str, Any]:
    return {"kind": "heading", "text": text}


def equation(text: str) -> dict[str, Any]:
    return {"kind": "equation", "text": text}


def bullets(*items: str) -> dict[str, Any]:
    return {"kind": "bullets", "items": [" ".join(item.split()) for item in items]}


def figure(name: str, height: float) -> dict[str, Any]:
    return {"kind": "figure", "name": name, "height": height}


def table(headers: list[str], rows: list[list[str]], widths: list[float]) -> dict[str, Any]:
    return {
        "kind": "table",
        "headers": headers,
        "rows": rows,
        "widths": widths,
    }


def reference(text: str) -> dict[str, Any]:
    return {"kind": "reference", "text": " ".join(text.split())}


# Single source of truth for every worked-example value printed in the report.
# The build-time drift check below compares this fixture with the live engine.
WORKED_EXAMPLE: dict[str, Any] = {
    "engine_version": "2.0.0",
    "gpa_iterations": 4,
    "reference_sample_size": 320,
    "shrinkage_intensity": 0.03200686389237932,
    "pc1_explained_variance_ratio": 0.2727981812317445,
    "inputs": {
        "a": {
            "label": "Sample A",
            "partial_procrustes": 0.054565314792775904,
            "full_procrustes": 0.05454500334658048,
            "regularized_mahalanobis": 10.201310283098197,
            "centroid_size": 0.8956481395542113,
            "residual_rms": 0.006617016364285085,
            "geometric_displacement_index": 0.3858350410755047,
            "pc1_score": 0.01342480535539741,
        },
        "b": {
            "label": "Sample B",
            "partial_procrustes": 0.0499959877547803,
            "full_procrustes": 0.04998036407476828,
            "regularized_mahalanobis": 12.765025588550127,
            "centroid_size": 0.8981573560192828,
            "residual_rms": 0.0060629040697072435,
            "geometric_displacement_index": 0.3535250197352474,
            "pc1_score": -0.004838824382566118,
        },
        "blend": {
            "label": "Blend",
            "partial_procrustes": 0.06440561718765489,
            "full_procrustes": 0.06437221354039414,
            "regularized_mahalanobis": 9.75919181210427,
            "centroid_size": 0.8397417779747599,
            "residual_rms": 0.007810328310229334,
            "geometric_displacement_index": 0.4554164865989563,
            "pc1_score": -0.007410982688225582,
        },
    },
}
WORKED_EXAMPLE_TOLERANCE = 1.0e-12


def worked(input_key: str, field: str) -> float:
    """Return one numeric value from the report's pinned worked fixture."""

    return float(WORKED_EXAMPLE["inputs"][input_key][field])


PAGES: list[dict[str, Any]] = [
    {
        "number": 2,
        "title": "Abstract",
        "blocks": [
            paragraph(
                "This technical report documents a serverless demonstration of descriptive "
                "geometric morphometrics implemented with NumPy and SciPy inside a Pyodide "
                "WebAssembly runtime. The application accepts either one of three deterministic "
                "synthetic configurations, a de-identified matrix, or an explicitly detected local "
                "photo mesh sampled into 68 ordered two-dimensional landmarks. It removes translation "
                "and centroid size, aligns configurations by "
                "proper singular-value-decomposition rotations, constructs a 132-dimensional "
                "Kendall tangent representation, calculates principal-component scores, and "
                "reports Procrustes, shrinkage-regularized Mahalanobis, and residual distances."
            ),
            paragraph(
                "The implementation is intentionally descriptive. Reference A, Reference B, and "
                "their pooled ensemble are simulations, not estimates of biological populations. "
                "Optional photo landmarking and texture warping occur only after an explicit action "
                "and remain in browser memory; Python receives coordinates, not pixels. A proportional "
                "warp visualizes the same residual field used by the numerical output. The worked "
                "pooled-reference Blend example yields partial "
                f"Procrustes distance {worked('blend', 'partial_procrustes'):.5f}, full Procrustes "
                f"distance {worked('blend', 'full_procrustes'):.5f}, regularized Mahalanobis "
                f"distance {worked('blend', 'regularized_mahalanobis'):.3f}, residual root mean "
                f"square {worked('blend', 'residual_rms'):.5f}, and Geometric Displacement Index "
                f"{worked('blend', 'geometric_displacement_index'):.2f} of 10. These are "
                "uncalibrated geometric magnitudes rather "
                "than percentiles, probabilities, classifications, or appearance ratings."
            ),
            heading("Keywords"),
            paragraph(
                "geometric morphometrics, Generalized Procrustes Analysis, tangent space, "
                "principal component analysis, covariance shrinkage, Mahalanobis distance, "
                "WebAssembly, local face landmarks, piecewise-affine warp, reproducible simulation"
            ),
        ],
    },
    {
        "number": 3,
        "title": "Descriptive Geometric Morphometrics in a Browser",
        "blocks": [
            paragraph(
                "Geometric morphometrics studies configurations of corresponding landmarks while "
                "preserving the geometry among those landmarks. Unlike a list of isolated lengths, "
                "an ordered coordinate configuration retains information about relative position, "
                "joint displacement, and multivariate shape. The central statistical difficulty is "
                "that raw coordinates also contain arbitrary translation, overall scale, and "
                "rotation. Those nuisance transformations must be separated from the shape signal "
                "before distances or covariance summaries are meaningful (Dryden & Mardia, 2016; "
                "Rohlf & Slice, 1990)."
            ),
            paragraph(
                "The present application turns this sequence into an inspectable browser workflow. "
                "A configuration is validated, centered, scaled to unit centroid size, rotated into "
                "a reference frame, projected into a tangent plane, and compared with a structured "
                "simulated ensemble. Each transformation is deterministic. JavaScript manages the "
                "interface and transfers only numeric coordinates to Python; the Python module "
                "performs the linear algebra and returns strict JSON."
            ),
            paragraph(
                "This report has two purposes. First, it states exactly what every displayed number "
                "means and how it is calculated. Second, it establishes the inferential boundary. "
                "The software does not operationalize attractiveness, health, identity, sex, "
                "gender, ancestry, or clinical status. It does not optimize a configuration toward "
                "a mean. Its output is a reproducible demonstration of shape-space mechanics, "
                "consistent with the distinction between geometric description and substantive "
                "biological inference emphasized by Adams et al. (2004) and Zelditch et al. (2012)."
            ),
        ],
    },
    {
        "number": 4,
        "title": "Scientific Scope, Unit of Analysis, and Estimand",
        "blocks": [
            paragraph(
                "The unit of analysis is one 68 by 2 matrix X whose rows are ordered landmark "
                "coordinates. A valid row has one horizontal and one vertical coordinate, and every "
                "configuration must use the same row-to-feature correspondence. The engine's "
                "estimand is descriptive separation between the analyzed configuration and a "
                "selected simulated reference consensus after translation, uniform scale, and "
                "proper rotation have been removed."
            ),
            paragraph(
                "Three families of output address different geometric questions. Procrustes "
                "distances summarize unweighted separation in aligned unit shape space. Tangent "
                "coordinates and PCA scores describe direction relative to the simulated reference "
                "covariance axes. Regularized Mahalanobis distance weights the tangent displacement "
                "by the inverse simulated covariance, so a displacement along a low-variance "
                "direction contributes more than an equally sized displacement along a high-variance "
                "direction (Mardia et al., 1979). Residual vectors retain landmark-level direction, "
                "and residual RMS compresses their magnitudes into one scalar."
            ),
            paragraph(
                "None of these quantities has an empirical sampling interpretation here. The "
                "reference covariance is generated rather than observed, the shrinkage amount is "
                "estimated within that simulation rather than validated against an external "
                "objective, and the analyzed "
                "configuration is not sampled from a declared population. Consequently, a distance "
                "cannot be labeled typical, unusual, desirable, or undesirable. The correct "
                "conclusion is comparative and conditional: for this input, this simulated "
                "reference, and this implementation, the reported magnitude equals the stated "
                "calculation."
            ),
        ],
    },
    {
        "number": 5,
        "title": "Serverless and Local-First Statistical Architecture",
        "blocks": [
            paragraph(
                "The application is a static GitHub Pages site. HTML defines the accessible three-tab "
                "workspace, CSS provides the responsive phone-like presentation, JavaScript controls "
                "state and validation, and Pyodide instantiates Python inside the browser. NumPy and "
                "SciPy are loaded from a pinned content-delivery-network path. No application server, "
                "database, session store, or telemetry endpoint is required."
            ),
            figure("pipeline", 126.0),
            paragraph(
                "A selected JPG, PNG, or WebP file receives a temporary browser object URL for viewing, "
                "panning, zooming, and rotation. That object URL is revoked when the image is replaced, "
                "removed, or the page closes. No detection occurs until an explicit button press. The "
                "pinned MediaPipe task then obtains one dense mesh in browser memory and a fixed adapter "
                "samples 68 vertices. Blendshape outputs are disabled, and pixels are never sent to "
                "Python or an application server."
            ),
            paragraph(
                "For an analysis, JavaScript creates a Float64Array of 136 values and assigns it to "
                "the Pyodide global namespace. The top-level Python function validates and reshapes "
                "the values, fits or retrieves the selected cached reference model, and returns JSON. "
                "A photo-derived run uses only those 136 numbers in Python. Returned residual vectors "
                "may drive a browser-canvas texture warp, while exports omit pixel data. Synthetic and "
                "de-identified coordinate routes remain independent alternatives."
            ),
        ],
    },
    {
        "number": 6,
        "title": "The 68-Point Coordinate Schema",
        "blocks": [
            paragraph(
                "Landmark methods require correspondence: row j must represent the same labeled "
                "location in every configuration. The application follows the conventional 68-point "
                "ordering used by the Dlib schema: 0-16 jawline, 17-21 right brow, 22-26 left brow, "
                "27-30 nose bridge, 31-35 nose base, 36-41 right eye, 42-47 left eye, 48-59 outer "
                "lip, and 60-67 inner lip. These labels establish topology only; they do not make the "
                "synthetic template a biometric standard."
            ),
            figure("landmarks", 260.0),
            paragraph(
                "Open feature paths such as the jaw and brows are drawn without connecting their "
                "endpoints. Eye and lip paths are closed. Coordinate files may be JSON, CSV, or "
                "plain text, but after parsing they must contain exactly 136 finite numbers. The photo "
                "route uses a fixed list of 68 unique MediaPipe mesh indices as an engineering "
                "correspondence; it is not native Dlib detector output. Template A samples those same "
                "indices from MediaPipe's canonical OBJ at commit a908d668, retains x and y, reverses "
                "the image y-axis, then centers and unit-scales the result. The Apache-2.0 provenance "
                "is recorded in THIRD_PARTY_NOTICES.md. This source match does not make the adapter a "
                "validated measurement model. Missing or reordered landmarks invalidate direct "
                "comparison (Bookstein, 1991; Zelditch et al., 2012)."
            ),
        ],
    },
    {
        "number": 7,
        "title": "Translation Removal and Centroid Size",
        "blocks": [
            paragraph(
                "Let X contain k = 68 landmark rows. The centroid is the row-wise average x-bar, and "
                "the centered configuration is Xc = X - 1(x-bar)^T. Subtracting the same centroid "
                "from every row removes the two degrees of freedom associated with horizontal and "
                "vertical translation. The operation preserves all pairwise landmark differences."
            ),
            equation("c(X) = ||Xc||_F = sqrt[ sum_j ||x_j - x-bar||_2^2 ]"),
            paragraph(
                "Centroid size c(X) is the Frobenius norm of the centered configuration. It is not "
                "face area, image resolution, or a biological size estimate. It depends on the units "
                "of the submitted coordinates and on the complete landmark configuration. The "
                "reported centroid-size card records this value before normalization so the user can "
                "audit the scale that was removed."
            ),
            equation("Z = Xc / c(X),     ||Z||_F = 1"),
            paragraph(
                "Dividing by centroid size produces a unit preshape Z and removes one uniform-scale "
                "degree of freedom. A configuration with zero or numerically negligible centroid "
                "size is rejected because normalization would be undefined. Translation and scale "
                "invariance are tested by applying arbitrary shifts and positive multipliers to an "
                "input and confirming that its post-alignment shape distances remain equal within "
                "floating-point tolerance (Dryden & Mardia, 2016)."
            ),
        ],
    },
    {
        "number": 8,
        "title": "Generalized Procrustes Analysis",
        "blocks": [
            paragraph(
                "Generalized Procrustes Analysis aligns an ensemble of preshapes to a jointly "
                "estimated consensus. If Zi is configuration i, Ri is a proper two-dimensional "
                "rotation, and M is a unit-norm consensus, the engine minimizes the summed squared "
                "Frobenius residuals over all rotations and the consensus."
            ),
            equation("min_(M,Ri)  sum_i ||Zi Ri - M||_F^2,     det(Ri) = 1"),
            paragraph(
                "The algorithm begins with the first preshape as the working frame. Every preshape is "
                "rotated toward that frame, the aligned configurations are averaged, the mean is "
                "recentered and normalized, and the new mean is aligned back to the previous frame. "
                "These steps repeat until the consensus change is below 1e-10 or 100 iterations have "
                "been attempted. A final alignment and normalized average ensure that the returned "
                "ensemble and consensus share one frame."
            ),
            paragraph(
                "This is a multi-configuration fit rather than a single pairwise alignment. The "
                "reported generalized sum of squares is the sum of squared configuration-to-consensus "
                "distances. The application also reports the mean and maximum reference distances, "
                "the number of iterations, and whether the convergence tolerance was met. Gower "
                "(1975) established the generalized formulation, and Rohlf and Slice (1990) developed "
                "the iterative landmark superimposition approach used throughout modern geometric "
                "morphometrics."
            ),
        ],
    },
    {
        "number": 9,
        "title": "Proper Rotation by Singular Value Decomposition",
        "blocks": [
            paragraph(
                "For one preshape Z and a current reference M, the rotation problem is solved from "
                "the 2 by 2 cross-product H = Z^T M. SciPy computes the singular value decomposition "
                "H = U S V^T. The orthogonal matrix U V^T minimizes the least-squares discrepancy, "
                "subject to a determinant correction."
            ),
            equation("R = U diag[1, det(U V^T)] V^T"),
            paragraph(
                "If det(U V^T) is negative, the unconstrained solution includes a reflection. The "
                "engine changes the final singular-vector sign so det(R) = 1. This restriction matters "
                "because a reflected configuration reverses handedness and should not be treated as "
                "an ordinary planar rotation. The aligned preshape is Y = ZR."
            ),
            paragraph(
                "SVD is numerically preferable to solving for an angle through ad hoc trigonometric "
                "expressions. It supplies the least-squares orthogonal factor directly and generalizes "
                "to higher-dimensional Procrustes problems. The implementation uses "
                "scipy.linalg.svd with finite-value checking handled during input validation. Proper "
                "rotation invariance is tested by multiplying an input by a known rotation matrix and "
                "confirming that the resulting distance measures agree with those from the original "
                "configuration (Goodall, 1991)."
            ),
        ],
    },
    {
        "number": 10,
        "title": "Consensus Convergence and Alignment Diagnostics",
        "blocks": [
            paragraph(
                "A small consensus change is a numerical stopping condition, not proof that a "
                "reference represents an external population. At iteration t the engine evaluates "
                "the Frobenius norm ||M(t+1) - M(t)||. Convergence is recorded when that norm is less "
                "than 1e-10. If the maximum of 100 iterations is reached first, the analysis still "
                "returns a status that exposes the failure."
            ),
            equation("G* = sum_i ||Yi - M-hat||_F^2"),
            paragraph(
                "The diagnostics shown in the interface serve distinct purposes. Iterations describe "
                "algorithmic effort. Reference n gives 160 for Reference A or B and 320 for the pooled "
                "reference. The generalized sum of squares measures total within-reference aligned "
                "dispersion. The covariance condition number, calculated later, describes sensitivity "
                "of linear solves after shrinkage and ridge stabilization."
            ),
            paragraph(
                "These diagnostics should be read before interpreting distances. A nonconverged GPA, "
                "nonfinite output, wrong tangent dimension, or failed Cholesky factorization would "
                "make downstream summaries unreliable. In the deterministic pooled model used for "
                "the worked example, GPA converges in four iterations. Fixed RandomState seeds stabilize "
                "the simulated draw stream, but SVD and eigendecomposition remain numerical-library "
                "dependent. The regression fixture therefore uses a tight tolerance, and exact artifact "
                "reproduction requires the pinned development stack."
            ),
        ],
    },
    {
        "number": 11,
        "title": "Preshape Sphere and Kendall Shape Space",
        "blocks": [
            paragraph(
                "After centering and unit scaling, a planar configuration lies on a preshape sphere. "
                "However, two preshapes that differ only by a proper planar rotation represent the same "
                "shape. Kendall shape space is therefore a quotient space: rotational orbits on the "
                "preshape sphere are identified. This curvature is why ordinary multivariate analysis "
                "cannot be applied naively to raw coordinates (Kendall, 1984)."
            ),
            paragraph(
                "For k planar landmarks, the unconstrained coordinate vector has 2k entries. Removing "
                "two translations, one uniform scale, and one planar rotation leaves 2k - 4 shape "
                "degrees of freedom. With k = 68, the resulting dimension is 132. The interface reports "
                "this number as a structural check."
            ),
            equation("dimension = 2(68) - 2 translation - 1 scale - 1 rotation = 132"),
            paragraph(
                "Statistical calculations proceed in a tangent plane at the fitted consensus rather "
                "than directly on the curved quotient. This is a local approximation. It is reliable "
                "when configurations remain sufficiently close to the tangent point, but it should "
                "not be treated as a globally distance-preserving map. The engine guards against a "
                "nearly orthogonal preshape and consensus because central projection would then divide "
                "by a value near zero (Dryden & Mardia, 2016; Klingenberg & Monteiro, 2005)."
            ),
        ],
    },
    {
        "number": 12,
        "title": "Constructing the 132-Dimensional Tangent Basis",
        "blocks": [
            paragraph(
                "The engine explicitly constructs four nuisance directions in the 136-dimensional "
                "ambient coordinate vector. Translation-x alternates 1 and 0, translation-y alternates "
                "0 and 1, the radial direction equals the vectorized consensus, and the infinitesimal "
                "rotation direction rotates every consensus row by 90 degrees."
            ),
            equation("N = [t_x, t_y, vec(M), vec(JM)],     B = canonical MGS(N-perp)"),
            paragraph(
                "The engine first orthonormalizes the four nuisance directions, then applies modified "
                "Gram-Schmidt with one reorthogonalization pass to the 136 standard basis vectors in "
                "index order. The accepted 132 columns form B, with B^T B = I and B orthogonal to each "
                "nuisance direction. This fixes the chart orientation conditional on the fitted "
                "consensus instead of accepting an arbitrary null-space basis."
            ),
            paragraph(
                "For aligned preshape Y, let a be the inner product between vec(Y) and vec(M). Central "
                "projection first forms vec(Y)/a - vec(M), which lies in the affine tangent plane at "
                "M. Multiplication by B^T gives the nonredundant coordinate vector z. Because B is "
                "orthonormal, Euclidean operations on z correspond to operations within the selected "
                "tangent subspace. The basis depends on the selected reference consensus, so changing "
                "Reference A, Reference B, or Pooled A+B recomputes the coordinate system."
            ),
            equation("z = B^T [ vec(Y)/<vec(Y),vec(M)> - vec(M) ]"),
        ],
    },
    {
        "number": 13,
        "title": "Tangent-Space Principal Component Analysis",
        "blocks": [
            paragraph(
                "PCA summarizes covariance directions in the simulated tangent sample. Let zi denote "
                "reference configuration i and z-bar its sample mean. The sample covariance is the "
                "sum of outer products of centered tangent vectors divided by n - 1. It is symmetric "
                "and positive semidefinite."
            ),
            equation("S = (1/(n-1)) sum_i (zi - z-bar)(zi - z-bar)^T = V Lambda V^T"),
            paragraph(
                "The engine uses scipy.linalg.eigh, which is designed for symmetric matrices. "
                "Eigenpairs are sorted from largest to smallest eigenvalue, and small negative values "
                "caused by floating-point roundoff are truncated to zero. For an analyzed tangent "
                "vector z, score j is vj^T(z - z-bar). The displayed percentage is lambda-j divided by "
                "the sum of all 132 eigenvalues (Jolliffe, 2002)."
            ),
            paragraph(
                "A PCA sign is arbitrary: multiplying an eigenvector and its score by -1 describes the "
                "same axis. Therefore, a positive score is not better than a negative score. Score "
                "magnitude describes coordinate displacement along that simulated axis, while the "
                "percentage describes how much simulated reference variance the axis accounts for. "
                "For stable presentation, each eigenvector is signed so its largest-magnitude loading "
                "is positive; exactly repeated eigenvalues can still leave a multidimensional eigenspace "
                "without a unique axis orientation. The JSON response returns the first 10 scores, "
                "eigenvalues, and ratios."
            ),
        ],
    },
    {
        "number": 14,
        "title": "Structured Synthetic Reference Ensembles",
        "blocks": [
            paragraph(
                "A diagonal noise model would make landmarks vary independently and would fail to "
                "demonstrate covariance-aware geometry. Instead, the engine defines 12 smooth tangent "
                "deformation modes spanning jaw width and depth, brow position and curvature, eye "
                "width and height, nose width and length, lip width and height, bilateral asymmetry, "
                "and a local lower-contour mode. Each raw mode is projected into the 132-dimensional "
                "tangent subspace and normalized."
            ),
            paragraph(
                "For every simulated configuration, independent normal coefficients multiply the 12 "
                "modes at fixed amplitudes. Small landmark-level normal noise is also projected into "
                "the tangent subspace. The perturbed configuration is returned to unit preshape form, "
                "then random translation, positive scale, and rotation are added as nuisance changes. "
                "GPA must recover from these intentionally introduced transformations."
            ),
            paragraph(
                "Reference A uses 160 shapes around Synthetic Template A with seed 31041. Reference B "
                "uses 160 around Synthetic Template B with seed 72107. The pooled model concatenates "
                "both ensembles and therefore has n = 320. This construction produces cross-landmark "
                "covariance and a stable legacy random stream within the declared stack. It is not "
                "fitted to Dlib users, a "
                "biological group, or any empirical sample. Synthetic structure supports software "
                "demonstration, not population inference (Bishop, 2006)."
            ),
        ],
    },
    {
        "number": 15,
        "title": "Covariance Estimation, Shrinkage, and Ridge Stabilization",
        "blocks": [
            paragraph(
                "The tangent sample covariance S contains 132 by 132 entries. Even with 160 or 320 "
                "simulated shapes, correlations can make S ill-conditioned. Direct inversion would "
                "amplify numerical error and could make Mahalanobis distance unstable. The application "
                "therefore shrinks S toward its own diagonal rather than toward a scalar identity "
                "matrix."
            ),
            equation("Sigma-hat = (1 - lam) S + lam diag(S) + epsilon I"),
            equation("lam* = clip_[0,1]{ sum_(i != j) Var-hat(s_ij) / sum_(i != j) s_ij^2 }"),
            figure("covariance", 104.0),
            paragraph(
                "Diagonal-target shrinkage preserves coordinate-specific variances while reducing "
                "off-diagonal magnitude by lam. The intensity is not fixed by hand: lam is the "
                "analytic minimum-risk estimate for this target, which for the pooled reference is "
                f"{WORKED_EXAMPLE['shrinkage_intensity']:.4f}. The ridge epsilon equals 1e-6 times "
                "the average post-shrinkage variance, "
                "bounded below by machine precision, and the matrix is symmetrized before Cholesky "
                "factorization (Ledoit & Wolf, 2004; Schafer & Strimmer, 2005)."
            ),
            paragraph(
                "The diagnostics report the ridge, the condition number, and the fraction of the "
                "Frobenius norm carried by off-diagonal entries. Those quantities show that covariance "
                "structure remains present and that the distance is not merely rescaled Euclidean "
                "distance."
            ),
        ],
    },
    {
        "number": 16,
        "title": "Regularized Mahalanobis Distance",
        "blocks": [
            paragraph(
                "Let z be the analyzed tangent coordinate and z-bar the simulated tangent mean. Their "
                "difference delta = z - z-bar is measured relative to the regularized covariance. The "
                "squared Mahalanobis distance is a quadratic form."
            ),
            equation("D_M^2 = delta^T Sigma-hat^(-1) delta,     D_M = sqrt(max(0,D_M^2))"),
            paragraph(
                "The engine does not explicitly form Sigma-hat inverse. SciPy first computes a lower "
                "Cholesky factor and then solves the two triangular systems needed for Sigma-hat "
                "inverse times delta. The final inner product delta^T solution yields the squared "
                "distance. A maximum with zero protects the square root from a tiny negative value "
                "caused by roundoff."
            ),
            paragraph(
                "Mahalanobis distance changes the geometry: displacement along a direction with small "
                "simulated variance receives greater weight, while displacement along a direction "
                "with larger simulated variance receives less. However, the value is not compared with "
                "a chi-square distribution because the reference is simulated, regularized, and not "
                "linked to a sampling model for the input. The displayed value therefore answers only "
                "how large this tangent displacement is under the selected simulated covariance metric "
                "(Mahalanobis, 1936; Mardia et al., 1979). The returned reference position is an "
                "in-sample rank among simulated reference distances, not a population percentile or "
                "p-value; held-out simulations serve only as a numerical scale check."
            ),
        ],
    },
    {
        "number": 17,
        "title": "Partial and Full Procrustes Distances",
        "blocks": [
            paragraph(
                "After optimal rotation, the partial Procrustes distance is the Frobenius norm between "
                "aligned input Y and consensus M. It is also called a chord distance because it measures "
                "the straight chord between unit preshapes in the embedding space."
            ),
            equation("d_P(Y,M) = ||Y - M||_F"),
            paragraph(
                "The full Procrustes measure allows an additional scalar adjustment. With unit "
                "preshapes and inner product a = <vec(Y),vec(M)>, the implemented expression is the "
                "square root of max(0, 1 - a^2). Both measures are nonnegative and approach zero as "
                "the aligned configurations agree."
            ),
            equation("d_F(Y,M) = sqrt[max(0, 1 - a^2)]"),
            paragraph(
                "These measures should be compared only under the same landmark schema and preprocessing "
                "definition. They contain no covariance weighting and no automatic threshold. A smaller "
                "value means closer geometric agreement with the selected consensus in its defined "
                "shape space; it does not mean a more desirable, healthy, or representative appearance. "
                "The proximity of partial and full values in the worked examples reflects the small "
                "local separations after unit normalization (Dryden & Mardia, 2016; Klingenberg & "
                "Monteiro, 2005)."
            ),
        ],
    },
    {
        "number": 18,
        "title": "Landmark Residuals and Root Mean Square",
        "blocks": [
            paragraph(
                "The aligned residual at landmark j is rj = Mj - Yj. Each rj is a two-dimensional "
                "vector pointing from the analyzed aligned coordinate toward the selected simulated "
                "consensus coordinate. Its Euclidean length records landmark-level separation in unit "
                "shape space. Multiplying the aligned residual matrix by the transpose of the fitted "
                "rotation also expresses the same vectors in normalized input orientation."
            ),
            equation("r_RMS = sqrt[(1/68) sum_j ||rj||_2^2]"),
            paragraph(
                "Residual RMS aggregates the 68 vector magnitudes. It is sensitive to distributed "
                "small differences and to a few larger differences, but it discards direction and "
                "regional identity. The canvas preserves direction by drawing arrows, while the JSON "
                "retains both vector components and all 68 magnitudes."
            ),
            paragraph(
                "The arrows can be rendered at 1x, 3x, or 6x. For photo landmarks, multiplying the "
                "input-orientation residual by centroid size restores pixel units before rendering. "
                "This occurs in JavaScript after Python returns; RMS and every other statistic remain "
                "unchanged. Residuals diagnose where aligned coordinate systems differ. They are not "
                "suggested edits, and the engine contains no optimization routine."
            ),
        ],
    },
    {
        "number": 19,
        "title": "Sample A, Sample B, and the Synthetic Blend",
        "blocks": [
            paragraph(
                "The three buttons in Research Input select teaching configurations rather than "
                "categories of people. Sample A is the first shape generated around Synthetic Template "
                "A from seed 90011. Sample B is the first shape generated around the deliberately "
                "altered Synthetic Template B from seed 90021. Each includes structured tangent "
                "variation plus nuisance translation, scale, and rotation."
            ),
            equation("base_blend = preshape(0.55 Template_A + 0.45 Template_B)"),
            paragraph(
                "Blend begins with a landmark-by-landmark convex combination of the two synthetic "
                "templates. The 55-to-45 weighting is explicit in core/analytics.py. The combined "
                "coordinates are centered and scaled to a unit preshape, then a new deterministic "
                "configuration is simulated around that base with seed 90031. The displayed Blend is "
                "therefore not the exact midpoint of the displayed Samples A and B."
            ),
            paragraph(
                "The input choice and reference choice answer different questions. Selecting Blend "
                "chooses the configuration being analyzed. Selecting Pooled A+B chooses a reference "
                "ensemble containing 160 A-based and 160 B-based simulated shapes. Selecting Reference "
                "A or B changes the consensus, tangent basis, covariance, PCA axes, and distances. None "
                "of these three sample buttons blends a photograph; the explicit local-photo route is "
                "a separate input control and does not produce an appearance score."
            ),
        ],
    },
    {
        "number": 20,
        "title": "Worked Deterministic Example",
        "blocks": [
            paragraph(
                f"Table 1 records engine version {WORKED_EXAMPLE['engine_version']} results for all "
                "three built-in configurations "
                "against Pooled A+B. The values were produced by run_pipeline_from_js with the fixed "
                "seeds documented above. Rebuilding the report imports the current analytics module "
                "and verifies the Blend values before writing the PDF."
            ),
            table(
                ["Input", "Partial", "Full", "Mahalanobis", "Centroid", "RMS"],
                [
                    [
                        str(WORKED_EXAMPLE["inputs"][key]["label"]),
                        f"{worked(key, 'partial_procrustes'):.5f}",
                        f"{worked(key, 'full_procrustes'):.5f}",
                        f"{worked(key, 'regularized_mahalanobis'):.3f}",
                        f"{worked(key, 'centroid_size'):.3f}",
                        f"{worked(key, 'residual_rms'):.5f}",
                    ]
                    for key in ("a", "b", "blend")
                ],
                [86, 69, 69, 82, 76, 70],
            ),
            paragraph(
                f"For Blend, the pooled reference GPA converges in {WORKED_EXAMPLE['gpa_iterations']} "
                f"iterations with n = {WORKED_EXAMPLE['reference_sample_size']}. The partial distance "
                f"{worked('blend', 'partial_procrustes'):.5f} and RMS "
                f"{worked('blend', 'residual_rms'):.5f} summarize aligned unit-space separation. "
                f"Regularized Mahalanobis {worked('blend', 'regularized_mahalanobis'):.3f} is larger "
                "numerically because it measures the same "
                "tangent displacement in standardized covariance geometry; it is not on the same scale "
                f"as a Procrustes distance. PC1 accounts for "
                f"{100 * float(WORKED_EXAMPLE['pc1_explained_variance_ratio']):.1f}% of pooled "
                f"simulated variance and the Blend PC1 score is {worked('blend', 'pc1_score'):.5f}. "
                "Rescaling its partial distance by 10 min(1, dP / sqrt(2)) gives a Geometric "
                f"Displacement Index of {worked('blend', 'geometric_displacement_index'):.2f}."
            ),
            paragraph(
                "The only defensible conclusion is that the generated Blend differs from the pooled "
                "simulated consensus by the reported amounts under this deterministic pipeline. The "
                "table cannot establish typicality, preference, identity, or a biological group "
                "difference. It is a reproducibility fixture that helps readers trace interface values "
                "back to equations and code."
            ),
        ],
    },
    {
        "number": 21,
        "title": "Landmark and Photo Texture Warp Visualization",
        "blocks": [
            paragraph(
                "The proportional warp makes the residual field visible. In aligned shape space, "
                "Wj(s) = Yj + s(Mj - Yj). For a photo input, the residual is rotated back to input "
                "orientation and multiplied by centroid size, giving the pixel destination "
                "xj' = xj + s c(X) rj. The selected scale changes rendering only."
            ),
            equation("GDI = 10 min[1, d_P / sqrt(2)]"),
            figure("warp", 205.0),
            paragraph(
                "The shape-space contour and the photo texture are rendering aids, not new statistical "
                "models. The photo renderer adds eight fixed boundary anchors, computes a Delaunay "
                "triangulation, and uses one affine pixel map per triangle. Vertex shifts are capped at "
                "16% of the face-box diagonal, and a line search reduces the requested scale if a "
                "triangle would reverse orientation."
            ),
            paragraph(
                "Original, Split, and Warped views use the same locally decoded image. Pixels are not "
                "uploaded or placed in the JSON export. Changing warp scale or mesh visibility redraws "
                "the canvas without changing any metric. The render is a geometric comparison, not a "
                "recommendation, prediction, improvement, or appearance evaluation."
            ),
        ],
    },
    {
        "number": 22,
        "title": "Study-Context Annotations and Future Research",
        "blocks": [
            paragraph(
                "Environmental Stress Index, Pathogen Prevalence Index, and Operational Sex Ratio are "
                "optional metadata sliders. The first two range from 0 to 1, and the third ranges from "
                "0.50 to 1.50 around a neutral code of 1.00. Their values are stored under "
                "study_context_metadata with analytic_role equal to annotation_only."
            ),
            paragraph(
                "Moving a slider never changes GPA, the tangent basis, PCA, covariance, distances, the "
                "Geometric Displacement Index, or residuals. The interface says this explicitly because "
                "adding a context variable to a "
                "screen can otherwise create the false impression that a causal or associational model "
                "has been fitted. The current values have no operational definition outside a future "
                "protocol. In particular, an operational ratio would require a declared numerator, "
                "denominator, population, and observation window."
            ),
            paragraph(
                "A genuine bio-social study would need a target population, sampling frame, measurement "
                "protocol, rater reliability analysis, missingness plan, predeclared estimand, and an "
                "outcome that is scientifically and ethically defensible. Repeated observations would "
                "require within-subject dependence to be modeled. Multivariate distance-based regression "
                "could relate a distance matrix to covariates, but permutation structure and dependence "
                "must be respected (Anderson, 2001; McArdle & Anderson, 2001; Shehzad et al., 2014). "
                "Those requirements are beyond this synthetic demonstration."
            ),
        ],
    },
    {
        "number": 23,
        "title": "Numerical Safeguards and Runtime Diagnostics",
        "blocks": [
            paragraph(
                "Input validation requires exactly 136 finite values, a 68 by 2 shape after reshaping, "
                "and positive centroid size. Photo files and coordinate files have separate accepted "
                "extensions and size limits. Malformed JSON, nonnumeric tokens, NaN, infinity, "
                "near-zero geometry, invalid reference keys, and inappropriate tangent projection are "
                "returned as bounded error messages."
            ),
            bullets(
                "GPA prohibits reflections and exposes convergence status and iteration count.",
                "The canonical tangent basis must have exactly 132 orthonormal columns.",
                "Covariance is shrunk, ridge-stabilized, symmetrized, and Cholesky-factorized.",
                "Linear solves replace an explicitly constructed covariance inverse.",
                "Reference-range and tangent-chart warnings accompany extrapolating inputs.",
                "The reported reference position is explicitly labeled in-sample and simulated.",
                "JSON rejects nonfinite values, and fixed RandomState seeds stabilize test streams.",
                "Photo shifts are capped and triangle orientation is protected by a scale line search.",
            ),
            paragraph(
                "The covariance condition number is a sensitivity diagnostic rather than a pass-fail "
                "proof. A very large value would suggest that small numeric perturbations could change "
                "the solution. The off-diagonal Frobenius fraction confirms that the covariance retains "
                "structured dependence. These values should be interpreted jointly with reference n, "
                "ridge magnitude, and GPA convergence."
            ),
            paragraph(
                "At the web layer, the runtime modal reports WebAssembly initialization, NumPy/SciPy "
                "loading, and analytics-module loading separately. Controls are disabled while an "
                "analysis is running to prevent state races. Photo landmarking is an explicit opt-in "
                "action; a fixed 68-index map is validated before use. Temporary Pyodide globals are "
                "deleted in a finally block, and an analysis token prevents an obsolete asynchronous "
                "response from overwriting a newer configuration."
            ),
        ],
    },
    {
        "number": 24,
        "title": "Validation, Invariance, and Reproducibility",
        "blocks": [
            paragraph(
                "A reproducible implementation must test properties, not only example outputs. "
                "Translation invariance is checked by adding a constant vector to all landmarks; "
                "uniform-scale invariance by multiplying every centered coordinate by a positive "
                "constant; and rotation invariance by applying a proper 2 by 2 rotation. Partial and "
                "full Procrustes distances, tangent summaries, and covariance-aware distance should "
                "remain equal within floating-point tolerance."
            ),
            paragraph(
                "Schema tests verify 68 residual vectors, 132 tangent coordinates, 10 returned PCA "
                "summaries, finite JSON, and explicit simulated-reference metadata. Negative tests "
                "cover wrong row counts, all-equal coordinates, nonfinite entries, reflection, invalid "
                "reference names, and malformed browser files. Interface tests verify direct panning "
                "from image pixels, fit/reset behavior, opt-in landmark status, 68 unique mapped mesh "
                "indices, triangulation, Original/Split/Warped redraws, the exact index formula, and "
                "the absence of image pixels from the Python and export payloads."
            ),
            paragraph(
                "Reproducibility also depends on provenance. The engine version, schema version, "
                "reference key, reference n, and deterministic seeds are returned with each result. "
                "Seventy-five tests cover invariances, algebraic identities, the canonical template "
                "digest, tangent-basis properties, malformed inputs, warning paths, and the browser JSON "
                "contract. The report builder verifies its worked fixture and exact 27-page count before "
                "printing a SHA-256 digest. requirements-dev.txt pins the authoring stack; neither seeds "
                "nor sign conventions alone promise cross-platform bitwise equality."
            ),
        ],
    },
    {
        "number": 25,
        "title": "Discussion, Limitations, and Conclusion",
        "blocks": [
            paragraph(
                "The application demonstrates that sophisticated shape statistics can run entirely in "
                "a static browser site while remaining auditable. Iterative GPA removes nuisance "
                "similarity transformations, the tangent basis expresses 132 independent local shape "
                "coordinates, PCA summarizes simulated covariance axes, regularization stabilizes the "
                "quadratic metric, and residuals preserve landmark-level differences."
            ),
            paragraph(
                "Its strongest design choice is also its main limitation: all reference distributions "
                "are synthetic. The system cannot estimate prevalence, group means, uncertainty for a "
                "target population, or external validity. It assumes exact landmark correspondence and "
                "does not model landmark acquisition error, perspective, expression, occlusion, "
                "missingness, or repeated measurements. The MediaPipe-to-68 adapter is approximate, "
                "and piecewise-affine image warping can contain seams or artifacts. The analytic "
                "shrinkage estimate is fitted to a simulated ensemble and has no external validation. "
                "Tangent coordinates are local and may distort large "
                "geodesic separations."
            ),
            paragraph(
                "For the deterministic pooled Blend example, the conclusion is narrow: after centering, "
                "unit scaling, and proper rotation, the synthetic Blend is separated from the pooled "
                f"synthetic consensus by partial Procrustes "
                f"{worked('blend', 'partial_procrustes'):.5f} and residual RMS "
                f"{worked('blend', 'residual_rms'):.5f}, while its covariance-weighted tangent "
                f"distance is {worked('blend', 'regularized_mahalanobis'):.3f}. Its neutral Geometric "
                f"Displacement Index is {worked('blend', 'geometric_displacement_index'):.2f} of 10. "
                "The proportional warp visualizes this residual direction without "
                "altering the calculation; none of these values evaluates appearance."
            ),
            paragraph(
                "Accordingly, the project should be used as an educational and engineering foundation. "
                "Any transition to empirical research requires representative de-identified data, "
                "measurement reliability, governance, uncertainty propagation, ethical review, and "
                "out-of-sample validation. Until those conditions are met, descriptive geometry is the "
                "appropriate endpoint."
            ),
        ],
    },
    {
        "number": 26,
        "title": "References",
        "blocks": [
            reference(
                "Adams, D. C., Rohlf, F. J., & Slice, D. E. (2004). Geometric morphometrics: "
                "Ten years of progress following the revolution. Italian Journal of Zoology, "
                "71(1), 5-16. https://doi.org/10.1080/11250000409356545"
            ),
            reference(
                "Anderson, M. J. (2001). A new method for non-parametric multivariate analysis "
                "of variance. Austral Ecology, 26(1), 32-46. "
                "https://doi.org/10.1111/j.1442-9993.2001.01070.pp.x"
            ),
            reference(
                "Berenfeld, C., Rosa, P., & Rousseau, J. (2024). Estimating a density near an "
                "unknown manifold: A Bayesian nonparametric approach. The Annals of Statistics, "
                "52(5), 2081-2111. https://doi.org/10.1214/24-AOS2423"
            ),
            reference(
                "Berry, T., & Sauer, T. (2017). Density estimation on manifolds with boundary. "
                "Computational Statistics & Data Analysis, 107, 1-17. "
                "https://doi.org/10.1016/j.csda.2016.09.011"
            ),
            reference(
                "Bishop, C. M. (2006). Pattern recognition and machine learning. Springer. "
                "https://doi.org/10.1007/978-0-387-45528-0"
            ),
            reference(
                "Bookstein, F. L. (1991). Morphometric tools for landmark data: Geometry and "
                "biology. Cambridge University Press."
            ),
            reference(
                "Dryden, I. L., & Mardia, K. V. (2016). Statistical shape analysis: With "
                "applications in R (2nd ed.). Wiley."
            ),
            reference(
                "Goodall, C. (1991). Procrustes methods in the statistical analysis of shape. "
                "Journal of the Royal Statistical Society: Series B, 53(2), 285-321."
            ),
            reference(
                "Gower, J. C. (1975). Generalized Procrustes analysis. Psychometrika, 40, "
                "33-51. https://doi.org/10.1007/BF02291478"
            ),
            reference(
                "Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical "
                "learning: Data mining, inference, and prediction (2nd ed.). Springer. "
                "https://doi.org/10.1007/978-0-387-84858-7"
            ),
        ],
    },
    {
        "number": 27,
        "title": "References (continued)",
        "blocks": [
            reference(
                "Google. (n.d.). Face Landmarker for Web. Google AI Edge. Retrieved "
                "September 6, 2026, from https://ai.google.dev/edge/mediapipe/solutions/"
                "vision/face_landmarker/web_js"
            ),
            reference(
                "MediaPipe Authors. (2020). Canonical face model [OBJ data file]. Google. "
                "Git commit a908d668c730da128dfa8d9f6bd25d519d006692. "
                "https://github.com/google-ai-edge/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/"
                "mediapipe/modules/face_geometry/data/canonical_face_model.obj"
            ),
            reference(
                "Jolliffe, I. T. (2002). Principal component analysis (2nd ed.). Springer. "
                "https://doi.org/10.1007/b98835"
            ),
            reference(
                "Kendall, D. G. (1984). Shape manifolds, Procrustean metrics, and complex "
                "projective spaces. Bulletin of the London Mathematical Society, 16(2), "
                "81-121. https://doi.org/10.1112/blms/16.2.81"
            ),
            reference(
                "Klingenberg, C. P., & Monteiro, L. R. (2005). Distances and directions in "
                "multidimensional shape spaces: Implications for morphometric applications. "
                "Systematic Biology, 54(4), 678-688. https://doi.org/10.1080/10635150590947258"
            ),
            reference(
                "Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional "
                "covariance matrices. Journal of Multivariate Analysis, 88(2), 365-411. "
                "https://doi.org/10.1016/S0047-259X(03)00096-4"
            ),
            reference(
                "Mahalanobis, P. C. (1936). On the generalized distance in statistics. "
                "Proceedings of the National Institute of Sciences of India, 2(1), 49-55."
            ),
            reference(
                "Mardia, K. V., Kent, J. T., & Bibby, J. M. (1979). Multivariate analysis. "
                "Academic Press."
            ),
            reference(
                "Rohlf, F. J., & Slice, D. E. (1990). Extensions of the Procrustes method for "
                "the optimal superimposition of landmarks. Systematic Zoology, 39(1), 40-59. "
                "https://doi.org/10.2307/2992207"
            ),
            reference(
                "Schafer, J., & Strimmer, K. (2005). A shrinkage approach to large-scale "
                "covariance matrix estimation and implications for functional genomics. "
                "Statistical Applications in Genetics and Molecular Biology, 4(1), Article 32. "
                "https://doi.org/10.2202/1544-6115.1175"
            ),
            reference(
                "Slice, D. E. (2007). Geometric morphometrics. Annual Review of Anthropology, "
                "36, 261-281. https://doi.org/10.1146/annurev.anthro.34.081804.120613"
            ),
            reference(
                "Zelditch, M. L., Swiderski, D. L., & Sheets, H. D. (2012). Geometric "
                "morphometrics for biologists: A primer (2nd ed.). Academic Press."
            ),
        ],
    },
]


def split_long_token(token: str, font: str, size: float, width: float) -> list[str]:
    if stringWidth(token, font, size) <= width:
        return [token]
    pieces: list[str] = []
    current = ""
    for character in token:
        candidate = current + character
        if current and stringWidth(candidate, font, size) > width:
            pieces.append(current)
            current = character
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def wrap_lines(text: str, font: str, size: float, width: float) -> list[str]:
    tokens: list[str] = []
    for token in text.split():
        tokens.extend(split_long_token(token, font, size, width))
    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = token if not current else f"{current} {token}"
        if current and stringWidth(candidate, font, size) > width:
            lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def draw_header(pdf: canvas.Canvas, page_number: int) -> None:
    pdf.setFillColor(MUTED)
    pdf.setFont(BODY_FONT, 9)
    pdf.drawString(LEFT, PAGE_HEIGHT - 44, RUNNING_HEAD)
    pdf.drawRightString(RIGHT, PAGE_HEIGHT - 44, str(page_number))
    pdf.setStrokeColor(colors.HexColor("#B7B7B7"))
    pdf.setLineWidth(0.35)
    pdf.line(LEFT, PAGE_HEIGHT - 52, RIGHT, PAGE_HEIGHT - 52)


def draw_footer(pdf: canvas.Canvas) -> None:
    pdf.setFillColor(colors.HexColor("#777777"))
    pdf.setFont(BODY_ITALIC, 8)
    pdf.drawCentredString(
        PAGE_WIDTH / 2,
        38,
        "Local geometric demonstration - no appearance or personal inference.",
    )


def draw_title_page(pdf: canvas.Canvas) -> None:
    draw_header(pdf, 1)
    pdf.setFillColor(INK)
    pdf.setFont(BODY_BOLD, 17)
    title_lines = [
        "Descriptive Geometric Morphometrics in a Browser:",
        "Statistical Architecture, Synthetic Demonstration,",
        "and Reproducible Interpretation",
    ]
    y = 520
    for line in title_lines:
        pdf.drawCentredString(PAGE_WIDTH / 2, y, line)
        y -= 26
    y -= 26
    pdf.setFont(BODY_FONT, BODY_SIZE)
    for line in [
        "Salem Morelli",
        "Bio-Social-Aesthetic-Manifold Project",
        "Technical Report",
        "September 6, 2026",
    ]:
        pdf.drawCentredString(PAGE_WIDTH / 2, y, line)
        y -= 24
    y -= 42
    pdf.setFont(BODY_ITALIC, 11)
    pdf.setFillColor(MUTED)
    pdf.drawCentredString(
        PAGE_WIDTH / 2,
        y,
        "A synthetic, descriptive computing demonstration",
    )
    draw_footer(pdf)
    pdf.showPage()


def draw_paragraph(
    pdf: canvas.Canvas,
    text: str,
    y: float,
    *,
    first_indent: float = 36.0,
    font: str = BODY_FONT,
    size: float = BODY_SIZE,
    leading: float = LEADING,
) -> float:
    words = text.split()
    if not words:
        return y
    first_width = BODY_WIDTH - first_indent
    first_tokens: list[str] = []
    while words:
        candidate = " ".join(first_tokens + [words[0]])
        if first_tokens and stringWidth(candidate, font, size) > first_width:
            break
        first_tokens.append(words.pop(0))
    lines = [" ".join(first_tokens)]
    lines.extend(wrap_lines(" ".join(words), font, size, BODY_WIDTH))
    pdf.setFillColor(INK)
    pdf.setFont(font, size)
    for index, line in enumerate(lines):
        x = LEFT + first_indent if index == 0 else LEFT
        pdf.drawString(x, y, line)
        y -= leading
    return y


def draw_heading(pdf: canvas.Canvas, text: str, y: float) -> float:
    pdf.setFillColor(INK)
    pdf.setFont(BODY_BOLD, BODY_SIZE)
    pdf.drawString(LEFT, y, text)
    return y - LEADING


def draw_equation(pdf: canvas.Canvas, text: str, y: float) -> float:
    box_height = 40.0
    pdf.setFillColor(PALE_CYAN)
    pdf.roundRect(LEFT, y - box_height + 8, BODY_WIDTH, box_height, 6, fill=1, stroke=0)
    pdf.setFillColor(colors.HexColor("#153E46"))
    pdf.setFont(BODY_ITALIC, 11.5)
    line = text
    if stringWidth(line, BODY_ITALIC, 11.5) > BODY_WIDTH - 28:
        line = wrap_lines(line, BODY_ITALIC, 11.5, BODY_WIDTH - 28)[0]
    pdf.drawCentredString(PAGE_WIDTH / 2, y - 16, line)
    return y - box_height - 4


def draw_bullets(pdf: canvas.Canvas, items: list[str], y: float) -> float:
    pdf.setFont(BODY_FONT, 11)
    for item in items:
        lines = wrap_lines(item, BODY_FONT, 11, BODY_WIDTH - 28)
        pdf.setFillColor(CYAN)
        pdf.circle(LEFT + 4, y + 3, 2, fill=1, stroke=0)
        pdf.setFillColor(INK)
        for line in lines:
            pdf.drawString(LEFT + 18, y, line)
            y -= 19
        y -= 2
    return y


def draw_reference(pdf: canvas.Canvas, text: str, y: float) -> float:
    first_width = BODY_WIDTH
    lines = wrap_lines(text, BODY_FONT, 11, first_width)
    pdf.setFillColor(INK)
    pdf.setFont(BODY_FONT, 11)
    for index, line in enumerate(lines):
        pdf.drawString(LEFT + (0 if index == 0 else 36), y, line)
        y -= 19
    return y - 6


def draw_table(pdf: canvas.Canvas, block: dict[str, Any], y: float) -> float:
    widths = block["widths"]
    headers = block["headers"]
    rows = block["rows"]
    row_height = 26
    total_width = sum(widths)
    x0 = LEFT + (BODY_WIDTH - total_width) / 2
    pdf.setFillColor(colors.HexColor("#244D59"))
    pdf.rect(x0, y - row_height, total_width, row_height, fill=1, stroke=0)
    x = x0
    pdf.setFillColor(colors.white)
    pdf.setFont(BODY_BOLD, 9)
    for label, width in zip(headers, widths, strict=True):
        pdf.drawCentredString(x + width / 2, y - 17, label)
        x += width
    y -= row_height
    pdf.setFont(BODY_FONT, 9.5)
    for row_index, row in enumerate(rows):
        pdf.setFillColor(PALE_CYAN if row_index % 2 == 0 else colors.white)
        pdf.rect(x0, y - row_height, total_width, row_height, fill=1, stroke=0)
        x = x0
        pdf.setFillColor(INK)
        for value, width in zip(row, widths, strict=True):
            pdf.drawCentredString(x + width / 2, y - 17, value)
            x += width
        y -= row_height
    pdf.setStrokeColor(GRID)
    pdf.rect(x0, y, total_width, row_height * (len(rows) + 1), fill=0, stroke=1)
    pdf.setFillColor(MUTED)
    pdf.setFont(BODY_ITALIC, 9)
    pdf.drawString(LEFT, y - 16, "Note. All reference configurations are simulated.")
    return y - 30


def point_transform(points: np.ndarray, x: float, y: float, width: float, height: float):
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    range_x = max(max_x - min_x, 1e-9)
    range_y = max(max_y - min_y, 1e-9)
    scale = min(width / range_x, height / range_y)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    def transform(point: np.ndarray) -> tuple[float, float]:
        return (
            x + width / 2 + (float(point[0]) - center_x) * scale,
            y + height / 2 - (float(point[1]) - center_y) * scale,
        )

    return transform


PATHS = [
    (0, 16, False), (17, 21, False), (22, 26, False), (27, 30, False),
    (31, 35, False), (36, 41, True), (42, 47, True), (48, 59, True),
    (60, 67, True),
]


def draw_configuration(
    pdf: canvas.Canvas,
    points: np.ndarray,
    transform,
    stroke: colors.Color,
    *,
    width: float = 1.0,
    dots: bool = True,
) -> None:
    pdf.setStrokeColor(stroke)
    pdf.setFillColor(stroke)
    pdf.setLineWidth(width)
    for start, end, closed in PATHS:
        path = pdf.beginPath()
        for index in range(start, end + 1):
            px, py = transform(points[index])
            if index == start:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        if closed:
            path.close()
        pdf.drawPath(path, stroke=1, fill=0)
    if dots:
        for point in points:
            px, py = transform(point)
            pdf.circle(px, py, 1.5, stroke=0, fill=1)


def draw_pipeline_figure(pdf: canvas.Canvas, y: float, height: float) -> None:
    labels = [
        ("1", "Acquire", "local 68 x 2 values"),
        ("2", "Normalize", "center and unit scale"),
        ("3", "Align", "proper SVD rotations"),
        ("4", "Project", "132 tangent values"),
        ("5", "Describe", "distances and residuals"),
    ]
    gap = 7
    box_width = (BODY_WIDTH - gap * 4) / 5
    box_height = 70
    base_y = y - 86
    for index, (number, title, detail) in enumerate(labels):
        x = LEFT + index * (box_width + gap)
        pdf.setFillColor(PALE_CYAN if index % 2 == 0 else PALE_VIOLET)
        pdf.setStrokeColor(CYAN if index % 2 == 0 else VIOLET)
        pdf.roundRect(x, base_y, box_width, box_height, 5, fill=1, stroke=1)
        pdf.setFillColor(MUTED)
        pdf.setFont(BODY_BOLD, 8)
        pdf.drawString(x + 7, base_y + 53, number)
        pdf.setFillColor(INK)
        pdf.setFont(BODY_BOLD, 9.5)
        pdf.drawCentredString(x + box_width / 2, base_y + 35, title)
        pdf.setFont(BODY_FONT, 7.5)
        pdf.drawCentredString(x + box_width / 2, base_y + 18, detail)


def draw_landmark_figure(pdf: canvas.Canvas, y: float, height: float) -> None:
    x = LEFT + 96
    width = BODY_WIDTH - 192
    bottom = y - height + 8
    transform = point_transform(SIMULATED_TEMPLATE_A, x, bottom, width, height - 18)
    draw_configuration(pdf, SIMULATED_TEMPLATE_A, transform, CYAN, width=1.1)
    pdf.setFont(BODY_FONT, 8)
    pdf.setFillColor(MUTED)
    labels = [(0, "0"), (8, "8"), (16, "16"), (27, "27"), (30, "30"),
              (36, "36"), (42, "42"), (48, "48"), (60, "60")]
    for index, label in labels:
        px, py = transform(SIMULATED_TEMPLATE_A[index])
        pdf.drawString(px + 4, py + 2, label)


def draw_covariance_figure(pdf: canvas.Canvas, y: float, height: float) -> None:
    size = 10
    cell = 10
    total = size * cell
    x0 = PAGE_WIDTH / 2 - total / 2
    bottom = y - height + 18
    for row in range(size):
        for column in range(size):
            if row == column:
                fill = colors.HexColor("#2E7C88")
            else:
                strength = ((row * 7 + column * 11) % 8) / 7
                fill = colors.Color(
                    0.88 - 0.25 * strength,
                    0.93 - 0.20 * strength,
                    0.97,
                )
            pdf.setFillColor(fill)
            pdf.rect(x0 + column * cell, bottom + (size - row - 1) * cell, cell, cell, fill=1, stroke=0)
    pdf.setStrokeColor(GRID)
    pdf.rect(x0, bottom, total, total, fill=0, stroke=1)
    pdf.setFillColor(MUTED)
    pdf.setFont(BODY_ITALIC, 8)
    pdf.drawCentredString(
        PAGE_WIDTH / 2,
        bottom - 13,
        "Schematic covariance structure: diagonal variance plus retained cross-coordinate dependence",
    )


def draw_warp_figure(pdf: canvas.Canvas, y: float, height: float) -> None:
    payload = json.loads(get_simulated_demo_json("blend"))
    result = json.loads(run_pipeline_from_js(payload["landmarks"], "pooled"))
    input_points = np.asarray(result["aligned_configuration"], dtype=float)
    reference_points = np.asarray(result["reference_consensus"], dtype=float)
    warp_points = input_points + 3.0 * (reference_points - input_points)
    all_points = np.concatenate([input_points, reference_points, warp_points], axis=0)
    transform = point_transform(
        all_points,
        LEFT + 72,
        y - height + 6,
        BODY_WIDTH - 144,
        height - 18,
    )
    draw_configuration(pdf, warp_points, transform, VIOLET, width=1.5, dots=False)
    draw_configuration(pdf, reference_points, transform, CYAN, width=0.9, dots=False)
    draw_configuration(pdf, input_points, transform, INK, width=1.0)
    pdf.setFont(BODY_FONT, 8)
    legend_y = y - height + 3
    for index, (label, color) in enumerate(
        [("Input", INK), ("Consensus", CYAN), ("3x warp", VIOLET)]
    ):
        start_x = LEFT + 112 + index * 110
        pdf.setStrokeColor(color)
        pdf.setLineWidth(2)
        pdf.line(start_x, legend_y, start_x + 18, legend_y)
        pdf.setFillColor(MUTED)
        pdf.drawString(start_x + 23, legend_y - 3, label)


FIGURE_DRAWERS = {
    "pipeline": draw_pipeline_figure,
    "landmarks": draw_landmark_figure,
    "covariance": draw_covariance_figure,
    "warp": draw_warp_figure,
}


def draw_content_page(pdf: canvas.Canvas, page: dict[str, Any]) -> None:
    number = int(page["number"])
    draw_header(pdf, number)
    pdf.setFillColor(INK)
    pdf.setFont(BODY_BOLD, 14)
    title_lines = wrap_lines(page["title"], BODY_BOLD, 14, BODY_WIDTH)
    y = PAGE_HEIGHT - 86
    for line in title_lines:
        pdf.drawCentredString(PAGE_WIDTH / 2, y, line)
        y -= 21
    y -= 8

    for block in page["blocks"]:
        kind = block["kind"]
        if kind == "paragraph":
            y = draw_paragraph(pdf, block["text"], y)
            y -= 3
        elif kind == "heading":
            y = draw_heading(pdf, block["text"], y)
        elif kind == "equation":
            y = draw_equation(pdf, block["text"], y)
        elif kind == "bullets":
            y = draw_bullets(pdf, block["items"], y)
        elif kind == "reference":
            y = draw_reference(pdf, block["text"], y)
        elif kind == "table":
            y = draw_table(pdf, block, y)
        elif kind == "figure":
            height = float(block["height"])
            FIGURE_DRAWERS[block["name"]](pdf, y, height)
            y -= height
            if block["name"] in {"covariance", "warp"}:
                y -= 18
        else:
            raise ValueError(f"Unknown report block kind: {kind}")

    if y < 56:
        raise RuntimeError(
            f"Page {number} content overflowed the one-inch body margin (y={y:.1f})."
        )
    draw_footer(pdf)
    pdf.showPage()


def verify_worked_example() -> None:
    """Refuse to build if the engine no longer reproduces the quoted values.

    ``RandomState`` stabilizes the simulated draw stream for this test fixture,
    while the tolerance accommodates ordinary floating-point variation. Exact
    artifact reproduction additionally requires the pinned development stack;
    NEP 19 does not promise whole-program bitwise equality across platforms.
    """

    mappings = {
        "partial_procrustes": ("distances", "partial_procrustes"),
        "full_procrustes": ("distances", "full_procrustes"),
        "regularized_mahalanobis": ("distances", "regularized_mahalanobis"),
        "centroid_size": ("input_geometry", "centroid_size"),
        "residual_rms": (
            "residual_shape_difference",
            "root_mean_square_magnitude",
        ),
        "geometric_displacement_index": (
            "geometric_displacement_index",
            "value",
        ),
        "pc1_score": ("tangent_space", "pca_scores"),
    }
    for input_key, expected in WORKED_EXAMPLE["inputs"].items():
        payload = json.loads(get_simulated_demo_json(input_key))
        result = json.loads(run_pipeline_from_js(payload["landmarks"], "pooled"))
        if "error" in result:
            raise RuntimeError(f"Engine returned an error: {result['error']}")
        if result["engine_version"] != WORKED_EXAMPLE["engine_version"]:
            raise RuntimeError("Worked example engine version drifted.")
        if result["gpa"]["iterations"] != WORKED_EXAMPLE["gpa_iterations"]:
            raise RuntimeError("Worked example GPA iteration count drifted.")

        for field, path in mappings.items():
            value = result[path[0]][path[1]]
            observed = float(value[0] if field == "pc1_score" else value)
            expected_value = float(expected[field])
            if not np.isclose(
                observed,
                expected_value,
                rtol=0.0,
                atol=WORKED_EXAMPLE_TOLERANCE,
            ):
                raise RuntimeError(
                    f"Worked example drift for {input_key}.{field}: "
                    f"{observed} != {expected_value}"
                )

        if result["warnings"]:
            raise RuntimeError(
                f"The {input_key} example raised chart warnings: "
                f"{result['warnings']}"
            )

    blend = json.loads(
        run_pipeline_from_js(
            json.loads(get_simulated_demo_json("blend"))["landmarks"],
            "pooled",
        )
    )
    observed_ratio = float(
        blend["tangent_space"]["pca_explained_variance_ratio"][0]
    )
    observed_shrinkage = float(
        blend["covariance_diagnostics"]["shrinkage_to_diagonal"]
    )
    for label, observed, expected in (
        (
            "pc1_explained_variance_ratio",
            observed_ratio,
            float(WORKED_EXAMPLE["pc1_explained_variance_ratio"]),
        ),
        (
            "shrinkage_intensity",
            observed_shrinkage,
            float(WORKED_EXAMPLE["shrinkage_intensity"]),
        ),
    ):
        if not np.isclose(
            observed,
            expected,
            rtol=0.0,
            atol=WORKED_EXAMPLE_TOLERANCE,
        ):
            raise RuntimeError(
                f"Worked example drift for {label}: {observed} != {expected}"
            )


def build_report() -> None:
    register_report_fonts()
    verify_worked_example()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUTPUT), pagesize=letter, pageCompression=1)
    pdf.setTitle(
        "Descriptive Geometric Morphometrics in a Browser: Technical Report"
    )
    pdf.setAuthor("Salem Morelli")
    pdf.setSubject(
        "Generalized Procrustes Analysis, tangent PCA, covariance regularization, "
        "Mahalanobis distance, local photo warping, and reproducible browser computing"
    )
    pdf.setKeywords(
        "geometric morphometrics, GPA, tangent space, PCA, Mahalanobis, photo warp, WebAssembly"
    )

    draw_title_page(pdf)
    for page in PAGES:
        draw_content_page(pdf, page)
    pdf.save()

    reader = PdfReader(str(OUTPUT))
    if len(reader.pages) != 27:
        raise RuntimeError(f"Expected 27 pages; generated {len(reader.pages)}.")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"Created {OUTPUT}")
    print(f"Pages: {len(reader.pages)}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    build_report()
