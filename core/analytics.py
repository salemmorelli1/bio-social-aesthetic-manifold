"""Client-side descriptive geometric-morphometrics engine.

The reference configurations and reference samples in this module are entirely
simulated. They are not empirical population estimates, biological norms,
clinical standards, attractiveness measures, or classification targets.

Implemented methods
-------------------
1. Deterministic synthetic reference ensembles following the Dlib 68-point
   landmark indexing convention.
2. Iterative Generalized Procrustes Analysis (GPA) without reflections.
3. A 132-dimensional Kendall tangent-space basis for 68 two-dimensional
   landmarks after removing translation, scale, and rotation.
4. Tangent-space PCA using ``scipy.linalg.eigh``.
5. A descriptive Mahalanobis distance based on a structured, shrinkage-
   regularized covariance matrix.
6. Residual shape-difference vectors in normalized input orientation.

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

ENGINE_VERSION: Final[str] = "1.0.0"
LANDMARK_COUNT: Final[int] = 68
SPATIAL_DIMENSIONS: Final[int] = 2
AMBIENT_DIMENSION: Final[int] = LANDMARK_COUNT * SPATIAL_DIMENSIONS
TANGENT_DIMENSION: Final[int] = AMBIENT_DIMENSION - 4
REFERENCE_SAMPLE_SIZE: Final[int] = 160
COVARIANCE_SHRINKAGE: Final[float] = 0.20
RIDGE_RELATIVE: Final[float] = 1.0e-6
NUM_PCS_RETURNED: Final[int] = 10

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
    pca_eigenvalues: FloatArray
    pca_eigenvectors: FloatArray


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

    centroid = np.mean(array, axis=0)
    centered = array - centroid
    size = float(la.norm(centered, ord="fro", check_finite=False))
    if size <= np.finfo(np.float64).eps * 100.0:
        raise ValueError("The configuration has zero or near-zero centroid size.")
    return centered / size, centroid, size


def _optimal_rotation(
    shape: FloatArray,
    reference: FloatArray,
) -> FloatArray:
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
            "GPA input must have shape (n, 68, 2); "
            f"received {sample.shape}."
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
    """Generate a coherent synthetic template in Dlib 68-point index order."""

    template = np.zeros((LANDMARK_COUNT, SPATIAL_DIMENSIONS), dtype=np.float64)

    jaw_x = np.linspace(-0.92, 0.92, 17)
    jaw_y = 0.02 + 0.84 * (1.0 - (np.abs(jaw_x) / 0.92) ** 1.75)
    template[0:17] = np.column_stack((jaw_x, jaw_y))

    template[17:22] = np.array(
        [
            [-0.62, -0.34],
            [-0.51, -0.40],
            [-0.38, -0.43],
            [-0.25, -0.41],
            [-0.14, -0.36],
        ],
        dtype=np.float64,
    )
    template[22:27] = np.array(
        [
            [0.14, -0.36],
            [0.25, -0.41],
            [0.38, -0.43],
            [0.51, -0.40],
            [0.62, -0.34],
        ],
        dtype=np.float64,
    )

    template[27:31] = np.array(
        [[0.00, -0.30], [0.00, -0.14], [0.00, 0.02], [0.00, 0.17]],
        dtype=np.float64,
    )
    template[31:36] = np.array(
        [
            [-0.22, 0.22],
            [-0.13, 0.27],
            [0.00, 0.29],
            [0.13, 0.27],
            [0.22, 0.22],
        ],
        dtype=np.float64,
    )

    template[36:42] = np.array(
        [
            [-0.48, -0.18],
            [-0.40, -0.23],
            [-0.29, -0.23],
            [-0.20, -0.18],
            [-0.29, -0.14],
            [-0.40, -0.14],
        ],
        dtype=np.float64,
    )
    template[42:48] = np.array(
        [
            [0.20, -0.18],
            [0.29, -0.23],
            [0.40, -0.23],
            [0.48, -0.18],
            [0.40, -0.14],
            [0.29, -0.14],
        ],
        dtype=np.float64,
    )

    template[48:60] = np.array(
        [
            [-0.31, 0.43],
            [-0.22, 0.37],
            [-0.11, 0.34],
            [0.00, 0.33],
            [0.11, 0.34],
            [0.22, 0.37],
            [0.31, 0.43],
            [0.22, 0.49],
            [0.11, 0.53],
            [0.00, 0.54],
            [-0.11, 0.53],
            [-0.22, 0.49],
        ],
        dtype=np.float64,
    )
    template[60:68] = np.array(
        [
            [-0.19, 0.43],
            [-0.10, 0.39],
            [0.00, 0.38],
            [0.10, 0.39],
            [0.19, 0.43],
            [0.10, 0.47],
            [0.00, 0.49],
            [-0.10, 0.47],
        ],
        dtype=np.float64,
    )

    preshape, _, _ = _to_preshape(template)
    return preshape


def _make_template_b(template_a: FloatArray) -> FloatArray:
    """Create a second arbitrary synthetic configuration for demonstration."""

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
    """Construct an orthonormal 132D basis excluding four similarity modes."""

    translation_x = np.zeros_like(consensus)
    translation_y = np.zeros_like(consensus)
    translation_x[:, 0] = 1.0
    translation_y[:, 1] = 1.0
    radial = consensus
    rotational = np.column_stack((-consensus[:, 1], consensus[:, 0]))

    nuisance = np.column_stack(
        (
            translation_x.ravel(),
            translation_y.ravel(),
            radial.ravel(),
            rotational.ravel(),
        )
    )
    basis = la.null_space(nuisance.T, rcond=1.0e-12)
    if basis.shape != (AMBIENT_DIMENSION, TANGENT_DIMENSION):
        raise la.LinAlgError(
            "The tangent-space basis did not have the expected dimension."
        )
    return np.asarray(basis, dtype=np.float64)


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


def _raw_simulation_modes(base: FloatArray) -> FloatArray:
    """Build correlated anatomical deformation modes for simulated sampling."""

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

    basis = _tangent_basis(base)
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

    rng = np.random.default_rng(seed)
    modes = _raw_simulation_modes(base)
    mode_scales = np.array(
        [0.032, 0.026, 0.019, 0.016, 0.018, 0.014,
         0.016, 0.019, 0.021, 0.015, 0.011, 0.009],
        dtype=np.float64,
    )
    base_basis = _tangent_basis(base)
    ensemble = np.empty(
        (sample_size, LANDMARK_COUNT, SPATIAL_DIMENSIONS),
        dtype=np.float64,
    )

    for index in range(sample_size):
        coefficients = rng.normal(size=modes.shape[0]) * mode_scales
        structured_deformation = np.tensordot(
            coefficients,
            modes,
            axes=(0, 0),
        )
        local_noise = rng.normal(scale=0.0015, size=AMBIENT_DIMENSION)
        tangent_noise = (base_basis @ (base_basis.T @ local_noise)).reshape(
            LANDMARK_COUNT,
            SPATIAL_DIMENSIONS,
        )
        preshape, _, _ = _to_preshape(
            base + structured_deformation + tangent_noise
        )

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
            preshape @ nuisance_rotation * nuisance_scale
            + nuisance_translation
        )

    return ensemble


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
        raise ValueError(
            "reference_population must be 'a', 'b', or 'pooled'."
        )
    key = aliases[normalized_key]

    if key == "a":
        ensemble = _simulate_reference_ensemble(
            SIMULATED_TEMPLATE_A,
            REFERENCE_SAMPLE_SIZE,
            seed=31_041,
        )
        seed_description = "31041"
    elif key == "b":
        ensemble = _simulate_reference_ensemble(
            SIMULATED_TEMPLATE_B,
            REFERENCE_SAMPLE_SIZE,
            seed=72_107,
        )
        seed_description = "72107"
    else:
        ensemble_a = _simulate_reference_ensemble(
            SIMULATED_TEMPLATE_A,
            REFERENCE_SAMPLE_SIZE,
            seed=31_041,
        )
        ensemble_b = _simulate_reference_ensemble(
            SIMULATED_TEMPLATE_B,
            REFERENCE_SAMPLE_SIZE,
            seed=72_107,
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

    diagonal_target = np.diag(np.diag(sample_covariance))
    covariance = (
        (1.0 - COVARIANCE_SHRINKAGE) * sample_covariance
        + COVARIANCE_SHRINKAGE * diagonal_target
    )
    average_variance = float(np.trace(covariance) / TANGENT_DIMENSION)
    ridge = max(average_variance * RIDGE_RELATIVE, np.finfo(np.float64).eps)
    covariance = covariance + np.eye(TANGENT_DIMENSION) * ridge
    covariance = (covariance + covariance.T) * 0.5
    covariance_cholesky = la.cho_factor(
        covariance,
        lower=True,
        check_finite=False,
    )

    eigenvalues, eigenvectors = la.eigh(
        sample_covariance,
        check_finite=False,
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    diagonal = np.diag(np.diag(covariance))
    off_diagonal = covariance - diagonal
    covariance_norm = float(la.norm(covariance, ord="fro", check_finite=False))
    off_diagonal_fraction = float(
        la.norm(off_diagonal, ord="fro", check_finite=False) / covariance_norm
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
        pca_eigenvalues=eigenvalues,
        pca_eigenvectors=eigenvectors,
    )


class DescriptiveMorphometricEngine:
    """Analyze configurations against an explicitly simulated reference model."""

    def __init__(self, reference_population: str = "pooled") -> None:
        self.reference_model = _fit_reference_model(reference_population)

    def analyze_configuration(self, input_landmarks: FloatArray) -> dict[str, Any]:
        """Return descriptive shape-space measurements without ranking or advice."""

        model = self.reference_model
        preshape, centroid, centroid_size = _to_preshape(input_landmarks)
        aligned, rotation = _align_to_reference(preshape, model.gpa.consensus)

        inner_product = float(aligned.ravel() @ model.gpa.consensus.ravel())
        inner_product = float(np.clip(inner_product, -1.0, 1.0))
        partial_procrustes = float(
            la.norm(aligned - model.gpa.consensus, ord="fro", check_finite=False)
        )
        full_procrustes = float(np.sqrt(max(0.0, 1.0 - inner_product**2)))

        tangent_coordinates, _, _ = _project_to_tangent(
            aligned,
            model.gpa.consensus,
            model.tangent_basis,
        )
        tangent_delta = tangent_coordinates - model.tangent_mean
        solved = la.cho_solve(
            model.covariance_cholesky,
            tangent_delta,
            check_finite=False,
        )
        mahalanobis_squared = max(0.0, float(tangent_delta @ solved))

        residual_aligned = model.gpa.consensus - aligned
        residual_input_orientation = residual_aligned @ rotation.T
        residual_magnitudes = la.norm(
            residual_input_orientation,
            axis=1,
            check_finite=False,
        )

        component_count = min(
            NUM_PCS_RETURNED,
            model.pca_eigenvalues.size,
        )
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
            "schema_version": "1.0",
            "engine_version": ENGINE_VERSION,
            "analysis_kind": "descriptive_synthetic_shape_comparison",
            "reference": {
                "key": model.key,
                "is_simulated": True,
                "is_empirical_population_estimate": False,
                "sample_size": model.sample_size,
                "deterministic_seed": model.seed_description,
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
                "generalized_sum_of_squares": (
                    model.gpa.generalized_sum_of_squares
                ),
                "mean_reference_distance": float(np.mean(gpa_distances)),
                "maximum_reference_distance": float(np.max(gpa_distances)),
            },
            "distances": {
                "partial_procrustes": partial_procrustes,
                "full_procrustes": full_procrustes,
                "regularized_mahalanobis": float(
                    np.sqrt(mahalanobis_squared)
                ),
                "regularized_mahalanobis_squared": mahalanobis_squared,
            },
            "tangent_space": {
                "dimension": TANGENT_DIMENSION,
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
                "shrinkage_to_diagonal": COVARIANCE_SHRINKAGE,
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
            SIMULATED_TEMPLATE_A,
            sample_size=3,
            seed=90_011,
        )
        key = "a"
    elif normalized_key in {"b", "demo_b", "sample_b"}:
        ensemble = _simulate_reference_ensemble(
            SIMULATED_TEMPLATE_B,
            sample_size=3,
            seed=90_021,
        )
        key = "b"
    elif normalized_key in {"blend", "pooled", "demo_blend"}:
        blended, _, _ = _to_preshape(
            0.55 * SIMULATED_TEMPLATE_A + 0.45 * SIMULATED_TEMPLATE_B
        )
        ensemble = _simulate_reference_ensemble(
            blended,
            sample_size=3,
            seed=90_031,
        )
        key = "blend"
    else:
        raise ValueError("demo_key must be 'a', 'b', or 'blend'.")

    return json.dumps(
        {
            "schema_version": "1.0",
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
        return json.dumps(
            result,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, json.JSONDecodeError, la.LinAlgError) as error:
        return json.dumps(
            {
                "schema_version": "1.0",
                "error": str(error),
                "error_type": type(error).__name__,
            },
            allow_nan=False,
            separators=(",", ":"),
        )
    except Exception:
        return json.dumps(
            {
                "schema_version": "1.0",
                "error": "Unexpected morphometric analysis failure.",
                "error_type": "RuntimeError",
            },
            allow_nan=False,
            separators=(",", ":"),
        )
