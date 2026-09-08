"""Client-side descriptive geometric-morphometrics engine.

SPDX-License-Identifier: MIT

MIT License

Copyright (c) 2026 Salem Morelli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Scientific scope
----------------
The reference configurations and reference samples in this module are entirely
simulated. They are not empirical population estimates, biological norms,
clinical standards, attractiveness measures, or classification targets.

Synthetic Template A is a geometric mean-face *model*, not a person and not a
sample from any population. Its 68 vertices are sampled from the MediaPipe
canonical face model at exactly the mesh indices used by the browser adapter in
``assets/js/photo-warp.mjs`` (see ``MEDIAPIPE_TO_DLIB_68``). This gives the
reference and photo route the same vertex correspondence and a compatible
frontal 2D coordinate convention. In the audited canonical-model comparison,
the earlier hand-authored seed fell beyond the prior simulated reference
maximum; the source-matched seed removes that known template mismatch. It does
not validate photo landmarking or make the reference empirical.

The canonical mesh coordinates below are derived from the MediaPipe project
file ``mediapipe/modules/face_geometry/data/canonical_face_model.obj`` at
commit ``a908d668c730da128dfa8d9f6bd25d519d006692``. The source is Copyright
2020 The MediaPipe Authors and distributed under the Apache License 2.0. See
``THIRD_PARTY_NOTICES.md`` and ``LICENSES/Apache-2.0.txt``.

Implemented methods
-------------------
1. Deterministic synthetic reference ensembles following the Dlib 68-point
   landmark indexing convention.
2. Iterative Generalized Procrustes Analysis (GPA) without reflections.
3. A deterministic 132-dimensional Kendall tangent basis, conditional on the
   fitted consensus, for 68 two-dimensional landmarks after removing
   translation, scale, and rotation.
4. Tangent-space PCA using ``scipy.linalg.eigh`` with a fixed sign convention.
5. A descriptive Mahalanobis distance based on a structured covariance matrix
   whose shrinkage intensity is estimated analytically (Schaefer & Strimmer,
   2005; Ledoit & Wolf, 2004) rather than fixed by hand.
6. Residual shape-difference vectors in normalized input orientation.
7. A bounded geometric displacement index that rescales partial Procrustes
   distance for display without ranking, valuation, or appearance inference,
   reported alongside its position in the simulated reference distribution
   because the absolute 0-10 interval can have little resolution locally.
8. Explicit tangent-chart validity diagnostics, so that a mirrored, malformed,
   or otherwise distant configuration is flagged instead of silently returning
   an authoritative-looking distance.

Reproducibility
---------------
Simulation uses an explicitly instantiated ``numpy.random.RandomState``. NEP 19
retains its legacy distribution implementations for stable test streams, while
also warning that this does not make a whole numerical program bitwise
reproducible across NumPy versions, operating systems, or LAPACK builds. The
report therefore uses a tight numerical tolerance, and exact artifact
reproduction additionally requires the dependency versions in
``requirements-dev.txt``.

The public ``run_pipeline_from_js`` function accepts JSON, ordinary Python
sequences, NumPy arrays, or Pyodide JavaScript proxies and returns JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final, Mapping, Sequence

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

ENGINE_VERSION: Final[str] = "2.0.0"
SCHEMA_VERSION: Final[str] = "2.0"
LANDMARK_COUNT: Final[int] = 68
SPATIAL_DIMENSIONS: Final[int] = 2
AMBIENT_DIMENSION: Final[int] = LANDMARK_COUNT * SPATIAL_DIMENSIONS
TANGENT_DIMENSION: Final[int] = AMBIENT_DIMENSION - 4
REFERENCE_SAMPLE_SIZE: Final[int] = 160
RIDGE_RELATIVE: Final[float] = 1.0e-6
NUM_PCS_RETURNED: Final[int] = 10
CANONICAL_TEMPLATE_SOURCE_COMMIT: Final[str] = (
    "a908d668c730da128dfa8d9f6bd25d519d006692"
)
CANONICAL_TEMPLATE_SOURCE_PATH: Final[str] = (
    "mediapipe/modules/face_geometry/data/canonical_face_model.obj"
)

# Maximum partial Procrustes distance between optimally aligned unit preshapes.
# With proper rotations only the optimal inner product is sigma_1 - sigma_2 >= 0,
# so d_P^2 = 2 - 2a <= 2.
MAXIMUM_PARTIAL_PROCRUSTES: Final[float] = float(np.sqrt(2.0))

# Beyond this partial Procrustes distance the central tangent projection is no
# longer treated as a small-distortion chart. At d_P = 0.35 the spherical
# geodesic is about 20 degrees and central-projection radial magnitude differs
# from geodesic magnitude by about 4 percent. The cutoff is a transparent
# diagnostic convention, not an inferential threshold. Results past it are
# still returned, but they are flagged.
TANGENT_CHART_LIMIT: Final[float] = 0.35

# Centroid sizes below this IEEE-754 boundary are subnormal. Analysis remains
# mathematically defined, but the input has fewer significant bits than a
# normal float64 value. This is a representation diagnostic, not a geometric
# rejection threshold: subnormal configurations are analyzed and visibly
# flagged so callers can decide whether to re-export them at a larger scale.
FLOAT64_NORMAL_FLOOR: Final[float] = float(np.finfo(np.float64).tiny)

# Optional manual override for the covariance shrinkage intensity. ``None``
# selects the analytic Schaefer-Strimmer estimate, which is what the cited
# literature actually prescribes.
COVARIANCE_SHRINKAGE_OVERRIDE: Final[float | None] = None

LANDMARK_SCHEMA: Final[Mapping[str, tuple[int, int]]] = {
    "jawline": (0, 16),
    "right_eyebrow": (17, 21),
    "left_eyebrow": (22, 26),
    "nose_bridge": (27, 30),
    "nose_base": (31, 35),
    "right_eye": (36, 41),
    "left_eye": (42, 47),
    "outer_lip": (48, 59),
    "inner_lip": (60, 67),
}

# Canonical face-model vertices sampled at the 68 adapter indices, expressed in
# image convention (x to the right, y downward), centred, and scaled to unit
# centroid size. Provenance and licence are documented in the module docstring.
CANONICAL_TEMPLATE: Final[FloatArray] = np.array(
    [
        # jawline 0-16
        [-0.16446661, -0.04594342],
        [-0.16184993, -0.00898193],
        [-0.15602702, +0.03053794],
        [-0.14419847, +0.07126151],
        [-0.10912555, +0.12254763],
        [-0.08731558, +0.14002649],
        [-0.06889775, +0.15161804],
        [-0.02774501, +0.16798367],
        [+0.00000000, +0.17028963],
        [+0.02774501, +0.16798367],
        [+0.06889775, +0.15161804],
        [+0.08731558, +0.14002649],
        [+0.10912555, +0.12254763],
        [+0.14419847, +0.07126151],
        [+0.15602702, +0.03053794],
        [+0.16184993, -0.00898193],
        [+0.16446661, -0.04594342],
        # right eyebrow 17-21
        [-0.12276694, -0.12279823],
        [-0.10699290, -0.13455519],
        [-0.08554812, -0.14114370],
        [-0.05923344, -0.14096096],
        [-0.02994908, -0.13905092],
        # left eyebrow 22-26
        [+0.02994908, -0.13905092],
        [+0.05923344, -0.14096096],
        [+0.08554812, -0.14114370],
        [+0.10699290, -0.13455519],
        [+0.12276694, -0.12279823],
        # nose bridge 27-30
        [+0.00000000, -0.10169197],
        [+0.00000000, -0.08457248],
        [+0.00000000, -0.06858788],
        [+0.00000000, -0.02155938],
        # nose base 31-35
        [-0.03016352, +0.00528654],
        [-0.01282058, +0.01171333],
        [+0.00000000, +0.01333002],
        [+0.01282058, +0.01171333],
        [+0.03016352, +0.00528654],
        # right eye 36-41
        [-0.09540423, -0.08866551],
        [-0.07875658, -0.09432477],
        [-0.05845533, -0.09505644],
        [-0.03983740, -0.08697569],
        [-0.05845533, -0.08119368],
        [-0.07875658, -0.08214541],
        # left eye 42-47
        [+0.03983740, -0.08697569],
        [+0.05845533, -0.09505644],
        [+0.07875658, -0.09432477],
        [+0.09540423, -0.08866551],
        [+0.07875658, -0.08214541],
        [+0.05845533, -0.08119368],
        # outer lip 48-59
        [-0.05270802, +0.06169023],
        [-0.04109229, +0.05011357],
        [-0.01526713, +0.03994643],
        [+0.00000000, +0.04159983],
        [+0.01526713, +0.03994643],
        [+0.04109229, +0.05011357],
        [+0.05270802, +0.06169023],
        [+0.03945526, +0.07212204],
        [+0.01501293, +0.08205985],
        [+0.00000000, +0.08363222],
        [-0.01501293, +0.08205985],
        [-0.03945526, +0.07212204],
        # inner lip 60-67
        [-0.04620329, +0.06026751],
        [-0.02198144, +0.05412009],
        [+0.00000000, +0.05421848],
        [+0.02198144, +0.05412009],
        [+0.04620329, +0.06026751],
        [+0.02357969, +0.06418307],
        [+0.00000000, +0.06597731],
        [-0.02357969, +0.06418307],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class GPAResult:
    """Result of a multi-configuration Generalized Procrustes fit."""

    aligned_shapes: FloatArray
    consensus: FloatArray
    iterations: int
    converged: bool
    per_shape_distances: FloatArray
    generalized_sum_of_squares: float


@dataclass(frozen=True)
class ReferenceModel:
    """Fitted descriptive model for one simulated reference ensemble."""

    key: str
    sample_size: int
    seed_description: str
    gpa: GPAResult
    tangent_basis: FloatArray
    tangent_mean: FloatArray
    covariance: FloatArray
    covariance_cholesky: tuple[FloatArray, bool]
    covariance_ridge: float
    covariance_condition_number: float
    covariance_off_diagonal_fraction: float
    shrinkage_intensity: float
    shrinkage_method: str
    pca_eigenvalues: FloatArray
    pca_eigenvectors: FloatArray
    reference_partial_distances: FloatArray
    reference_mahalanobis_squared: FloatArray


def _centroid_size(shape: FloatArray) -> float:
    """Return centroid size, equal to the Frobenius norm after centering."""

    centered = shape - np.mean(shape, axis=0, keepdims=True)
    return float(la.norm(centered, ord="fro", check_finite=False))


def _to_preshape(shape: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    """Remove translation and scale and return preshape, centroid, and size."""

    array = np.asarray(shape, dtype=np.float64)
    if array.shape != (LANDMARK_COUNT, SPATIAL_DIMENSIONS):
        raise ValueError(
            f"Each configuration must have shape ({LANDMARK_COUNT}, "
            f"{SPATIAL_DIMENSIONS}); received {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Landmark coordinates must all be finite numbers.")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        centroid = np.mean(array, axis=0)
        centered = array - centroid
        size = float(la.norm(centered, ord="fro", check_finite=False))

    # The direct norm is fastest and preserves the pinned ordinary-scale
    # results, but squaring can overflow above roughly sqrt(float_max) or
    # underflow below roughly sqrt(float_tiny). Recompute in scaled coordinates
    # whenever that path is unsafe. This retains similarity invariance across
    # the representable range instead of applying a dimensionful "near-zero"
    # cutoff to otherwise valid configurations.
    direct_norm_is_unsafe = (
        not np.all(np.isfinite(centroid))
        or not np.all(np.isfinite(centered))
        or not np.isfinite(size)
        or size < np.sqrt(np.finfo(np.float64).tiny)
    )
    if direct_norm_is_unsafe:
        coordinate_scale = float(np.max(np.abs(array)))
        if coordinate_scale == 0.0:
            raise ValueError("The configuration has zero centroid size.")

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scaled_array = array / coordinate_scale
            scaled_centroid = np.mean(scaled_array, axis=0)
            scaled_centered = scaled_array - scaled_centroid
            centered_peak = float(np.max(np.abs(scaled_centered)))

        if centered_peak == 0.0:
            raise ValueError("The configuration has zero centroid size.")

        norm_input = scaled_centered / centered_peak
        norm_factor = float(la.norm(norm_input, ord="fro", check_finite=False))
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            centroid = scaled_centroid * coordinate_scale
            size = coordinate_scale * (centered_peak * norm_factor)
        preshape = norm_input / norm_factor

        if (
            not np.all(np.isfinite(centroid))
            or not np.isfinite(size)
            or size <= 0.0
            or not np.all(np.isfinite(preshape))
        ):
            raise ValueError(
                "Coordinate magnitudes are too large to compute a finite "
                "centroid size in double precision. Rescale the configuration "
                "before analysis."
            )
        return preshape, centroid, size

    if size == 0.0:
        raise ValueError("The configuration has zero centroid size.")
    return centered / size, centroid, size


def _optimal_rotation(shape: FloatArray, reference: FloatArray) -> FloatArray:
    """Return the proper rotation minimizing ``||shape @ R - reference||``."""

    cross_product = shape.T @ reference
    u_matrix, _, vt_matrix = la.svd(
        cross_product,
        full_matrices=False,
        check_finite=False,
    )
    rotation = u_matrix @ vt_matrix
    if la.det(rotation) < 0.0:
        u_matrix[:, -1] *= -1.0
        rotation = u_matrix @ vt_matrix
    return np.asarray(rotation, dtype=np.float64)


def _align_to_reference(
    preshape: FloatArray,
    reference: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    rotation = _optimal_rotation(preshape, reference)
    return preshape @ rotation, rotation


def iterative_generalized_procrustes(
    shapes: FloatArray,
    max_iterations: int = 100,
    tolerance: float = 1.0e-10,
) -> GPAResult:
    """Fit an iterative GPA consensus to a sample of 2D configurations.

    Translation and centroid size are removed from every configuration. Proper
    rotations are estimated by SVD; reflections are prohibited. The algorithm
    iterates until the Frobenius change in the unit-norm consensus is below
    ``tolerance`` or ``max_iterations`` is reached.
    """

    sample = np.asarray(shapes, dtype=np.float64)
    expected_tail = (LANDMARK_COUNT, SPATIAL_DIMENSIONS)
    if sample.ndim != 3 or sample.shape[1:] != expected_tail:
        raise ValueError(
            f"GPA input must have shape (n, 68, 2); received {sample.shape}."
        )
    if sample.shape[0] < 2:
        raise ValueError("GPA requires at least two configurations.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    preshapes = np.empty_like(sample, dtype=np.float64)
    for index, shape in enumerate(sample):
        preshapes[index], _, _ = _to_preshape(shape)

    consensus = np.array(preshapes[0], copy=True)
    aligned = np.empty_like(preshapes)
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        for index, preshape in enumerate(preshapes):
            aligned[index], _ = _align_to_reference(preshape, consensus)

        raw_mean = np.mean(aligned, axis=0)
        new_consensus, _, _ = _to_preshape(raw_mean)
        new_consensus, frame_rotation = _align_to_reference(
            new_consensus,
            consensus,
        )
        aligned = aligned @ frame_rotation

        consensus_change = float(
            la.norm(new_consensus - consensus, ord="fro", check_finite=False)
        )
        consensus = new_consensus
        if consensus_change < tolerance:
            converged = True
            break

    for index, preshape in enumerate(preshapes):
        aligned[index], _ = _align_to_reference(preshape, consensus)

    final_raw_mean = np.mean(aligned, axis=0)
    final_consensus, _, _ = _to_preshape(final_raw_mean)
    final_consensus, frame_rotation = _align_to_reference(
        final_consensus,
        consensus,
    )
    aligned = aligned @ frame_rotation

    distances = np.asarray(
        [
            la.norm(shape - final_consensus, ord="fro", check_finite=False)
            for shape in aligned
        ],
        dtype=np.float64,
    )
    sum_of_squares = float(distances @ distances)

    return GPAResult(
        aligned_shapes=aligned,
        consensus=final_consensus,
        iterations=iterations,
        converged=converged,
        per_shape_distances=distances,
        generalized_sum_of_squares=sum_of_squares,
    )


def generate_dlib_68_template() -> FloatArray:
    """Return Synthetic Template A as a unit-centroid-size preshape.

    The template is the canonical face-model geometry sampled at the same mesh
    indices the browser adapter uses, so the simulated reference and any
    photo-derived configuration occupy the same shape space.
    """

    preshape, _, _ = _to_preshape(CANONICAL_TEMPLATE)
    return preshape


def _make_template_b(template_a: FloatArray) -> FloatArray:
    """Create a second synthetic configuration for demonstration.

    The deformations are deliberately arbitrary. They exist so that Reference A
    and Reference B are distinguishable; they do not encode any group, class,
    or population contrast.
    """

    template_b = np.array(template_a, copy=True)
    template_b[0:17, 0] *= 1.055
    template_b[5:12, 1] *= 1.035
    template_b[17:27, 1] -= 0.010
    template_b[31:36, 0] *= 1.045
    template_b[48:68, 0] *= 0.970
    template_b, _, _ = _to_preshape(template_b)
    return template_b


SIMULATED_TEMPLATE_A: Final[FloatArray] = generate_dlib_68_template()
SIMULATED_TEMPLATE_B: Final[FloatArray] = _make_template_b(SIMULATED_TEMPLATE_A)


def _tangent_basis(consensus: FloatArray) -> FloatArray:
    """Construct a canonical orthonormal 132D basis excluding similarity modes.

    ``scipy.linalg.null_space`` returns *an* orthonormal basis for the required
    subspace, and its orientation may depend on the LAPACK build. Modified
    Gram-Schmidt over the standard basis vectors in index order fixes one chart
    conditional on the supplied consensus and floating-point arithmetic. This
    removes arbitrary null-space orientation; it does not promise bitwise
    equality for the complete LAPACK-dependent pipeline across platforms.
    """

    translation_x = np.zeros_like(consensus)
    translation_y = np.zeros_like(consensus)
    translation_x[:, 0] = 1.0
    translation_y[:, 1] = 1.0
    radial = consensus
    rotational = np.column_stack((-consensus[:, 1], consensus[:, 0]))

    nuisance = [
        translation_x.ravel(),
        translation_y.ravel(),
        radial.ravel(),
        rotational.ravel(),
    ]

    def orthonormalize(vector: FloatArray, accepted: list[FloatArray]) -> FloatArray | None:
        residual = np.array(vector, dtype=np.float64, copy=True)
        initial_norm = float(la.norm(residual, check_finite=False))
        if initial_norm <= 0.0:
            return None
        for _ in range(2):  # reorthogonalize once for numerical stability
            for direction in accepted:
                residual -= float(residual @ direction) * direction
        residual_norm = float(la.norm(residual, check_finite=False))
        if residual_norm <= 1.0e-8 * initial_norm:
            return None
        return residual / residual_norm

    similarity_frame: list[FloatArray] = []
    for vector in nuisance:
        direction = orthonormalize(vector, similarity_frame)
        if direction is None:
            raise la.LinAlgError(
                "The four similarity directions were not independent at this "
                "consensus."
            )
        similarity_frame.append(direction)

    basis_vectors: list[FloatArray] = []
    spanning = list(similarity_frame)
    for axis in range(AMBIENT_DIMENSION):
        if len(basis_vectors) == TANGENT_DIMENSION:
            break
        candidate = np.zeros(AMBIENT_DIMENSION, dtype=np.float64)
        candidate[axis] = 1.0
        direction = orthonormalize(candidate, spanning)
        if direction is None:
            continue
        basis_vectors.append(direction)
        spanning.append(direction)

    if len(basis_vectors) != TANGENT_DIMENSION:
        raise la.LinAlgError(
            "The tangent-space basis did not have the expected dimension."
        )
    return np.column_stack(basis_vectors)


def _project_to_tangent(
    aligned_shape: FloatArray,
    consensus: FloatArray,
    basis: FloatArray,
) -> tuple[FloatArray, FloatArray, float]:
    """Centrally project a preshape onto the tangent plane at the consensus."""

    inner_product = float(aligned_shape.ravel() @ consensus.ravel())
    if inner_product <= np.finfo(np.float64).eps * 100.0:
        raise ValueError(
            "The configuration is too far from the reference for this local "
            "tangent-plane approximation."
        )
    ambient_tangent = aligned_shape.ravel() / inner_product - consensus.ravel()
    coordinates = basis.T @ ambient_tangent
    return coordinates, ambient_tangent, inner_product


def _raw_simulation_modes(base: FloatArray, basis: FloatArray) -> FloatArray:
    """Build correlated anatomical deformation modes for simulated sampling.

    ``basis`` is supplied by the caller so that the canonical tangent basis is
    constructed once per ensemble rather than once per helper. Gram-Schmidt is
    cheap but the browser runs this under WebAssembly.
    """

    modes = np.zeros((12, LANDMARK_COUNT, SPATIAL_DIMENSIONS), dtype=np.float64)

    modes[0, 0:17, 0] = base[0:17, 0]
    modes[1, 4:13, 1] = np.sin(np.linspace(0.0, np.pi, 9))
    modes[2, 17:27, 1] = 1.0
    modes[3, 17:22, 1] = np.array([0.5, -0.2, -0.7, -0.2, 0.5])
    modes[3, 22:27, 1] = np.array([0.5, -0.2, -0.7, -0.2, 0.5])

    for eye_slice in (slice(36, 42), slice(42, 48)):
        eye = base[eye_slice]
        eye_center = np.mean(eye, axis=0)
        modes[4, eye_slice, 0] = eye[:, 0] - eye_center[0]
        modes[5, eye_slice, 1] = eye[:, 1] - eye_center[1]

    nose_base = base[31:36]
    modes[6, 31:36, 0] = nose_base[:, 0] - np.mean(nose_base[:, 0])
    modes[7, 27:36, 1] = np.linspace(-0.6, 1.0, 9)

    for lip_slice in (slice(48, 60), slice(60, 68)):
        lip = base[lip_slice]
        lip_center = np.mean(lip, axis=0)
        modes[8, lip_slice, 0] = lip[:, 0] - lip_center[0]
        modes[9, lip_slice, 1] = lip[:, 1] - lip_center[1]

    left_indices = np.array([0, 1, 2, 3, 4, 17, 18, 19, 20, 21, 36, 37, 38, 41])
    right_indices = np.array([12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 43, 44, 45, 46])
    modes[10, left_indices, 1] = 1.0
    modes[10, right_indices, 1] = -1.0
    modes[11, 6:11, 0] = np.array([-0.2, -0.5, -0.8, -0.5, -0.2])

    projected_modes = np.empty_like(modes)
    for index, mode in enumerate(modes):
        projected = basis @ (basis.T @ mode.ravel())
        mode_norm = float(la.norm(projected, check_finite=False))
        if mode_norm <= np.finfo(np.float64).eps * 100.0:
            raise la.LinAlgError("A simulated deformation mode became degenerate.")
        projected_modes[index] = (projected / mode_norm).reshape(
            LANDMARK_COUNT,
            SPATIAL_DIMENSIONS,
        )
    return projected_modes


def _simulate_reference_ensemble(
    base: FloatArray,
    sample_size: int,
    seed: int,
) -> FloatArray:
    """Generate a deterministic structured synthetic reference ensemble."""

    if sample_size < 3:
        raise ValueError("A simulated reference ensemble requires at least 3 shapes.")

    rng = np.random.RandomState(seed)
    base_basis = _tangent_basis(base)
    modes = _raw_simulation_modes(base, base_basis)
    mode_scales = np.array(
        [0.032, 0.026, 0.019, 0.016, 0.018, 0.014,
         0.016, 0.019, 0.021, 0.015, 0.011, 0.009],
        dtype=np.float64,
    )
    ensemble = np.empty(
        (sample_size, LANDMARK_COUNT, SPATIAL_DIMENSIONS),
        dtype=np.float64,
    )

    for index in range(sample_size):
        coefficients = rng.normal(size=modes.shape[0]) * mode_scales
        structured_deformation = np.tensordot(coefficients, modes, axes=(0, 0))
        local_noise = rng.normal(scale=0.0015, size=AMBIENT_DIMENSION)
        tangent_noise = (base_basis @ (base_basis.T @ local_noise)).reshape(
            LANDMARK_COUNT,
            SPATIAL_DIMENSIONS,
        )
        preshape, _, _ = _to_preshape(base + structured_deformation + tangent_noise)

        angle = float(rng.normal(scale=0.12))
        cosine = np.cos(angle)
        sine = np.sin(angle)
        nuisance_rotation = np.array(
            [[cosine, -sine], [sine, cosine]],
            dtype=np.float64,
        )
        nuisance_scale = float(np.exp(rng.normal(scale=0.08)))
        nuisance_translation = rng.normal(scale=0.15, size=2)
        ensemble[index] = (
            preshape @ nuisance_rotation * nuisance_scale + nuisance_translation
        )

    return ensemble


def _analytic_shrinkage_intensity(centered: FloatArray, sample: FloatArray) -> float:
    """Estimate the optimal intensity for shrinkage toward ``diag(S)``.

    This is the Schaefer and Strimmer (2005) closed form of the Ledoit and Wolf
    (2004) estimator for target D = diag(S). Because the target reproduces the
    diagonal of S exactly, the diagonal terms cancel from both sums and only
    off-diagonal entries contribute:

        lambda* = sum_{i != j} Var-hat(s_ij) / sum_{i != j} s_ij^2.

    The variance is accumulated with two matrix products rather than an
    n x p x p array, because the browser runtime has a limited heap.
    """

    observations = centered.shape[0]
    if observations < 4:
        return 1.0

    squared = centered ** 2
    sum_of_squared_products = squared.T @ squared
    scale = observations / float((observations - 1) ** 3)
    entry_variance = scale * (
        sum_of_squared_products
        - (float((observations - 1) ** 2) / observations) * sample ** 2
    )

    off_diagonal = ~np.eye(sample.shape[0], dtype=bool)
    numerator = float(np.sum(entry_variance[off_diagonal]))
    denominator = float(np.sum(sample[off_diagonal] ** 2))
    if denominator <= 0.0:
        return 1.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


@lru_cache(maxsize=3)
def _fit_reference_model(population: str) -> ReferenceModel:
    """Fit GPA, tangent covariance, and PCA for a simulated reference choice."""

    normalized_key = population.lower().strip()
    aliases = {
        "a": "a",
        "simulated_a": "a",
        "reference_a": "a",
        "b": "b",
        "simulated_b": "b",
        "reference_b": "b",
        "pooled": "pooled",
        "neutral": "pooled",
    }
    if normalized_key not in aliases:
        raise ValueError("reference_population must be 'a', 'b', or 'pooled'.")
    key = aliases[normalized_key]

    if key == "a":
        ensemble = _simulate_reference_ensemble(
            SIMULATED_TEMPLATE_A, REFERENCE_SAMPLE_SIZE, seed=31_041
        )
        seed_description = "31041"
    elif key == "b":
        ensemble = _simulate_reference_ensemble(
            SIMULATED_TEMPLATE_B, REFERENCE_SAMPLE_SIZE, seed=72_107
        )
        seed_description = "72107"
    else:
        ensemble_a = _simulate_reference_ensemble(
            SIMULATED_TEMPLATE_A, REFERENCE_SAMPLE_SIZE, seed=31_041
        )
        ensemble_b = _simulate_reference_ensemble(
            SIMULATED_TEMPLATE_B, REFERENCE_SAMPLE_SIZE, seed=72_107
        )
        ensemble = np.concatenate((ensemble_a, ensemble_b), axis=0)
        seed_description = "31041+72107"

    gpa = iterative_generalized_procrustes(ensemble)
    basis = _tangent_basis(gpa.consensus)
    tangent_rows = np.asarray(
        [
            _project_to_tangent(shape, gpa.consensus, basis)[0]
            for shape in gpa.aligned_shapes
        ],
        dtype=np.float64,
    )
    tangent_mean = np.mean(tangent_rows, axis=0)
    centered_tangent = tangent_rows - tangent_mean
    sample_covariance = (
        centered_tangent.T @ centered_tangent / (tangent_rows.shape[0] - 1)
    )
    sample_covariance = (sample_covariance + sample_covariance.T) * 0.5

    if COVARIANCE_SHRINKAGE_OVERRIDE is None:
        intensity = _analytic_shrinkage_intensity(centered_tangent, sample_covariance)
        shrinkage_method = "analytic_schaefer_strimmer_diagonal_target"
    else:
        intensity = float(np.clip(COVARIANCE_SHRINKAGE_OVERRIDE, 0.0, 1.0))
        shrinkage_method = "fixed_override"

    diagonal_target = np.diag(np.diag(sample_covariance))
    covariance = (
        (1.0 - intensity) * sample_covariance + intensity * diagonal_target
    )
    average_variance = float(np.trace(covariance) / TANGENT_DIMENSION)
    ridge = max(average_variance * RIDGE_RELATIVE, np.finfo(np.float64).eps)
    covariance = covariance + np.eye(TANGENT_DIMENSION) * ridge
    covariance = (covariance + covariance.T) * 0.5
    covariance_cholesky = la.cho_factor(covariance, lower=True, check_finite=False)

    eigenvalues, eigenvectors = la.eigh(sample_covariance, check_finite=False)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    # LAPACK leaves eigenvector signs arbitrary. Fix them so that exported PCA
    # scores reproduce across platforms.
    dominant = np.argmax(np.abs(eigenvectors), axis=0)
    signs = np.sign(eigenvectors[dominant, np.arange(eigenvectors.shape[1])])
    signs[signs == 0.0] = 1.0
    eigenvectors = eigenvectors * signs

    diagonal = np.diag(np.diag(covariance))
    off_diagonal = covariance - diagonal
    covariance_norm = float(la.norm(covariance, ord="fro", check_finite=False))
    off_diagonal_fraction = float(
        la.norm(off_diagonal, ord="fro", check_finite=False) / covariance_norm
    )

    # In-sample distribution of the reference ensemble itself. This is what the
    # absolute 0-10 display index lacks: without it a user cannot tell whether a
    # value of 0.6 is ordinary or extreme. It is in-sample and therefore
    # optimistic; it is reported as descriptive context, not as a test.
    reference_solved = la.cho_solve(
        covariance_cholesky, centered_tangent.T, check_finite=False
    )
    reference_mahalanobis_squared = np.maximum(
        np.einsum("ij,ji->i", centered_tangent, reference_solved), 0.0
    )

    return ReferenceModel(
        key=key,
        sample_size=int(ensemble.shape[0]),
        seed_description=seed_description,
        gpa=gpa,
        tangent_basis=basis,
        tangent_mean=tangent_mean,
        covariance=covariance,
        covariance_cholesky=covariance_cholesky,
        covariance_ridge=ridge,
        covariance_condition_number=float(np.linalg.cond(covariance)),
        covariance_off_diagonal_fraction=off_diagonal_fraction,
        shrinkage_intensity=float(intensity),
        shrinkage_method=shrinkage_method,
        pca_eigenvalues=eigenvalues,
        pca_eigenvectors=eigenvectors,
        reference_partial_distances=gpa.per_shape_distances,
        reference_mahalanobis_squared=np.asarray(
            reference_mahalanobis_squared, dtype=np.float64
        ),
    )


def _percentile_of(value: float, distribution: FloatArray) -> float:
    """Return the fraction of the reference distribution below ``value``."""

    if distribution.size == 0:
        return float("nan")
    return float(np.mean(distribution < value) * 100.0)


class DescriptiveMorphometricEngine:
    """Analyze configurations against an explicitly simulated reference model."""

    def __init__(self, reference_population: str = "pooled") -> None:
        self.reference_model = _fit_reference_model(reference_population)

    def analyze_configuration(self, input_landmarks: FloatArray) -> dict[str, Any]:
        """Return descriptive shape-space measurements without ranking or advice."""

        model = self.reference_model
        preshape, centroid, centroid_size = _to_preshape(input_landmarks)
        aligned, rotation = _align_to_reference(preshape, model.gpa.consensus)

        inner_product = float(np.clip(
            aligned.ravel() @ model.gpa.consensus.ravel(), -1.0, 1.0
        ))
        partial_procrustes = float(
            la.norm(aligned - model.gpa.consensus, ord="fro", check_finite=False)
        )
        full_procrustes = float(np.sqrt(max(0.0, 1.0 - inner_product**2)))
        geometric_displacement_index = float(
            10.0 * min(1.0, partial_procrustes / MAXIMUM_PARTIAL_PROCRUSTES)
        )

        tangent_coordinates, _, _ = _project_to_tangent(
            aligned, model.gpa.consensus, model.tangent_basis
        )
        tangent_delta = tangent_coordinates - model.tangent_mean
        solved = la.cho_solve(
            model.covariance_cholesky, tangent_delta, check_finite=False
        )
        mahalanobis_squared = max(0.0, float(tangent_delta @ solved))

        reference_maximum = float(np.max(model.reference_partial_distances))
        within_chart = partial_procrustes <= TANGENT_CHART_LIMIT
        within_reference_range = partial_procrustes <= reference_maximum

        warnings: list[str] = []
        if 0.0 < centroid_size < FLOAT64_NORMAL_FLOOR:
            warnings.append(
                "The configuration's centroid size is subnormal in double "
                "precision, so its coordinates carry fewer significant digits "
                "than normal float64 values. Analysis continues, but the "
                "reported distances may lose accuracy. Re-export the source "
                "coordinates at a larger numerical scale to preserve full "
                "precision."
            )

        if not within_chart:
            warnings.append(
                "The configuration lies outside the small-distortion region of "
                "the tangent chart (partial Procrustes distance "
                f"{partial_procrustes:.4f} exceeds {TANGENT_CHART_LIMIT}). The "
                "tangent coordinates, PCA scores, and Mahalanobis distance are "
                "reported but should not be interpreted quantitatively. "
                "Possible causes include reflection, landmark-order mismatch, "
                "or geometry well outside the simulated ensemble."
            )
        elif not within_reference_range:
            warnings.append(
                "The configuration is farther from the consensus than every "
                "member of the simulated reference ensemble (maximum "
                f"{reference_maximum:.4f}). The covariance was not estimated in "
                "this region, so the Mahalanobis distance is an extrapolation."
            )

        residual_aligned = model.gpa.consensus - aligned
        residual_input_orientation = residual_aligned @ rotation.T
        residual_magnitudes = la.norm(
            residual_input_orientation, axis=1, check_finite=False
        )

        component_count = min(NUM_PCS_RETURNED, model.pca_eigenvalues.size)
        component_vectors = model.pca_eigenvectors[:, :component_count]
        pca_scores = tangent_delta @ component_vectors
        total_variance = float(np.sum(model.pca_eigenvalues))
        if total_variance <= np.finfo(np.float64).eps:
            explained_ratios = np.zeros(component_count, dtype=np.float64)
        else:
            explained_ratios = (
                model.pca_eigenvalues[:component_count] / total_variance
            )

        gpa_distances = model.gpa.per_shape_distances
        return {
            "schema_version": SCHEMA_VERSION,
            "engine_version": ENGINE_VERSION,
            "analysis_kind": "descriptive_shape_comparison_with_simulated_reference",
            "warnings": warnings,
            "reference": {
                "key": model.key,
                "is_simulated": True,
                "is_empirical_population_estimate": False,
                "sample_size": model.sample_size,
                "deterministic_seed": model.seed_description,
                "template_provenance": (
                    "MediaPipe canonical face model at commit "
                    f"{CANONICAL_TEMPLATE_SOURCE_COMMIT}, sampled at the 68 "
                    "adapter indices and projected to centered unit-size 2D; "
                    "a geometric model, not a person or a population"
                ),
            },
            "landmark_schema": {
                name: {"start": bounds[0], "end": bounds[1]}
                for name, bounds in LANDMARK_SCHEMA.items()
            },
            "input_geometry": {
                "centroid": centroid.tolist(),
                "centroid_size": centroid_size,
            },
            "gpa": {
                "iterations": model.gpa.iterations,
                "converged": model.gpa.converged,
                "generalized_sum_of_squares": model.gpa.generalized_sum_of_squares,
                "mean_reference_distance": float(np.mean(gpa_distances)),
                "maximum_reference_distance": float(np.max(gpa_distances)),
            },
            "distances": {
                "partial_procrustes": partial_procrustes,
                "full_procrustes": full_procrustes,
                "regularized_mahalanobis": float(np.sqrt(mahalanobis_squared)),
                "regularized_mahalanobis_squared": mahalanobis_squared,
            },
            "chart_diagnostics": {
                "tangent_projection_within_small_distortion_region": within_chart,
                "within_reference_distance_range": within_reference_range,
                "partial_procrustes_chart_limit": TANGENT_CHART_LIMIT,
                "partial_procrustes_theoretical_maximum": (
                    MAXIMUM_PARTIAL_PROCRUSTES
                ),
                "reference_maximum_partial_procrustes": reference_maximum,
            },
            "reference_calibration": {
                "note": (
                    "In-sample position of this configuration within the "
                    "simulated reference ensemble. Descriptive context only: "
                    "these are not p-values, percentiles of any population, or "
                    "calibrated probabilities."
                ),
                "partial_procrustes_reference_percentile": _percentile_of(
                    partial_procrustes, model.reference_partial_distances
                ),
                "mahalanobis_squared_reference_percentile": _percentile_of(
                    mahalanobis_squared, model.reference_mahalanobis_squared
                ),
                "reference_partial_procrustes_median": float(
                    np.median(model.reference_partial_distances)
                ),
                "reference_partial_procrustes_maximum": reference_maximum,
                "reference_mahalanobis_squared_mean": float(
                    np.mean(model.reference_mahalanobis_squared)
                ),
                "tangent_dimension_for_reference": TANGENT_DIMENSION,
            },
            "geometric_displacement_index": {
                "value": geometric_displacement_index,
                "minimum": 0.0,
                "maximum": 10.0,
                "formula": "10 * min(1, partial_procrustes / sqrt(2))",
                "direction": (
                    "larger means greater aligned geometric separation only"
                ),
                "normative": False,
                "appearance_rating": False,
                "resolution_note": (
                    "The endpoints are mathematical, not empirical. Face-like "
                    "configurations generated by this simulator occupy a "
                    "narrow region of this interval, so read the in-sample "
                    "reference position alongside it."
                ),
                "reference_percentile": _percentile_of(
                    partial_procrustes, model.reference_partial_distances
                ),
            },
            "tangent_space": {
                "dimension": TANGENT_DIMENSION,
                "basis_convention": (
                    "modified Gram-Schmidt over standard basis vectors in index "
                    "order, orthogonal to the four similarity directions"
                ),
                "coordinates": tangent_coordinates.tolist(),
                "pca_scores": pca_scores.tolist(),
                "pca_eigenvalues": (
                    model.pca_eigenvalues[:component_count].tolist()
                ),
                "pca_explained_variance_ratio": explained_ratios.tolist(),
            },
            "residual_shape_difference": {
                "coordinate_system": "centered unit-centroid-size shape space",
                "aligned_vectors": residual_aligned.tolist(),
                "input_orientation_vectors": residual_input_orientation.tolist(),
                "magnitudes": residual_magnitudes.tolist(),
                "root_mean_square_magnitude": float(
                    np.sqrt(np.mean(residual_magnitudes**2))
                ),
            },
            "aligned_configuration": aligned.tolist(),
            "reference_consensus": model.gpa.consensus.tolist(),
            "covariance_diagnostics": {
                "dimension": TANGENT_DIMENSION,
                "shrinkage_to_diagonal": model.shrinkage_intensity,
                "shrinkage_method": model.shrinkage_method,
                "ridge_added": model.covariance_ridge,
                "condition_number": model.covariance_condition_number,
                "off_diagonal_frobenius_fraction": (
                    model.covariance_off_diagonal_fraction
                ),
            },
            "interpretation": {
                "normative": False,
                "classification": False,
                "prescriptive_modification": False,
                "statement": (
                    "All outputs are descriptive comparisons with a simulated "
                    "reference and must not be interpreted as biological, "
                    "clinical, identity, or aesthetic judgments."
                ),
            },
        }


def get_simulated_demo_json(demo_key: str = "a") -> str:
    """Return one deterministic, explicitly simulated 68-point configuration.

    The helper exists for the browser demonstration so the application can run
    without accepting a photograph or making inferences about a person.
    """

    normalized_key = demo_key.lower().strip()
    if normalized_key in {"a", "demo_a", "sample_a"}:
        ensemble = _simulate_reference_ensemble(
            SIMULATED_TEMPLATE_A, sample_size=3, seed=90_011
        )
        key = "a"
    elif normalized_key in {"b", "demo_b", "sample_b"}:
        ensemble = _simulate_reference_ensemble(
            SIMULATED_TEMPLATE_B, sample_size=3, seed=90_021
        )
        key = "b"
    elif normalized_key in {"blend", "pooled", "demo_blend"}:
        blended, _, _ = _to_preshape(
            0.55 * SIMULATED_TEMPLATE_A + 0.45 * SIMULATED_TEMPLATE_B
        )
        ensemble = _simulate_reference_ensemble(
            blended, sample_size=3, seed=90_031
        )
        key = "blend"
    else:
        raise ValueError("demo_key must be 'a', 'b', or 'blend'.")

    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "demo_key": key,
            "is_simulated": True,
            "landmarks": ensemble[0].tolist(),
        },
        allow_nan=False,
        separators=(",", ":"),
    )


def _coerce_landmarks(value: Any) -> FloatArray:
    """Convert JSON, a Python sequence, NumPy data, or a Pyodide proxy."""

    raw: Any = value
    if isinstance(raw, str):
        raw = json.loads(raw)
    elif hasattr(raw, "to_py"):
        raw = raw.to_py()

    array = np.asarray(raw, dtype=np.float64)
    if array.size != AMBIENT_DIMENSION:
        raise ValueError(
            f"Expected {AMBIENT_DIMENSION} coordinate values for 68x2 "
            f"landmarks; received {array.size}."
        )
    array = array.reshape(LANDMARK_COUNT, SPATIAL_DIMENSIONS)
    if not np.all(np.isfinite(array)):
        raise ValueError("Landmark coordinates must all be finite numbers.")
    return array


def run_pipeline_from_js(
    landmarks_flat: str | Sequence[float] | FloatArray | Any,
    reference_population: str = "pooled",
) -> str:
    """Run the descriptive pipeline and return a strict JSON response.

    ``reference_population`` accepts ``"a"``, ``"b"``, or ``"pooled"``.
    All three choices refer only to simulated reference ensembles.
    """

    try:
        landmarks = _coerce_landmarks(landmarks_flat)
        engine = DescriptiveMorphometricEngine(str(reference_population))
        result = engine.analyze_configuration(landmarks)
        return json.dumps(result, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, json.JSONDecodeError, la.LinAlgError) as error:
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "error": str(error),
                "error_type": type(error).__name__,
            },
            allow_nan=False,
            separators=(",", ":"),
        )
    except Exception:
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "error": "Unexpected morphometric analysis failure.",
                "error_type": "RuntimeError",
            },
            allow_nan=False,
            separators=(",", ":"),
        )
