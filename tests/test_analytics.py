"""Property and regression tests for the descriptive morphometric engine.

SPDX-License-Identifier: MIT

The README asserts that the engine "has also been checked for translation,
uniform-scale, and proper-rotation invariance to floating-point precision."
These tests are that check. Each test named ``test_regression_*`` pins a defect
found in the audit of the 1.1.0 engine so it cannot return silently.

Run with::

    python -m pytest tests/ -q
"""

from __future__ import annotations

import json
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.analytics import (  # noqa: E402
    AMBIENT_DIMENSION,
    CANONICAL_TEMPLATE,
    CANONICAL_TEMPLATE_SOURCE_COMMIT,
    LANDMARK_COUNT,
    MAXIMUM_PARTIAL_PROCRUSTES,
    SIMULATED_TEMPLATE_A,
    SIMULATED_TEMPLATE_B,
    TANGENT_CHART_LIMIT,
    TANGENT_DIMENSION,
    DescriptiveMorphometricEngine,
    _analytic_shrinkage_intensity,
    _fit_reference_model,
    _simulate_reference_ensemble,
    _tangent_basis,
    _to_preshape,
    get_simulated_demo_json,
    iterative_generalized_procrustes,
    run_pipeline_from_js,
)

TOLERANCE = 1.0e-10


def rotation_matrix(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def demo(key: str) -> np.ndarray:
    return np.asarray(json.loads(get_simulated_demo_json(key))["landmarks"])


def analyze(landmarks: np.ndarray, reference: str = "pooled") -> dict:
    return json.loads(
        run_pipeline_from_js(np.asarray(landmarks).ravel().tolist(), reference)
    )


@pytest.fixture(scope="module")
def pooled():
    return _fit_reference_model("pooled")


# ---------------------------------------------------------------- invariance


@pytest.mark.parametrize("key", ["a", "b", "blend"])
@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(lambda x: x + np.array([137.0, -45.0]), id="translation"),
        pytest.param(lambda x: x * 1.0e3, id="scale-up"),
        pytest.param(lambda x: x * 1.0e-4, id="scale-down"),
        pytest.param(lambda x: x @ rotation_matrix(0.7), id="rotation"),
        pytest.param(lambda x: x @ rotation_matrix(np.pi), id="rotation-pi"),
        pytest.param(
            lambda x: x @ rotation_matrix(1.1) * 22.0 + np.array([-9.0, 4.0]),
            id="similarity",
        ),
    ],
)
def test_similarity_invariance(key, transform):
    """Distances are invariant under translation, uniform scale, and rotation."""

    baseline = analyze(demo(key))["distances"]
    transformed = analyze(transform(demo(key)))["distances"]
    for name, value in baseline.items():
        assert transformed[name] == pytest.approx(value, abs=TOLERANCE)


def test_reflection_is_not_an_invariance():
    """Reflections are prohibited, so a mirrored configuration must differ."""

    baseline = analyze(demo("a"))["distances"]["partial_procrustes"]
    mirrored = analyze(demo("a") * np.array([-1.0, 1.0]))["distances"][
        "partial_procrustes"
    ]
    assert mirrored > baseline * 5.0


# ----------------------------------------------------------------- distances


@pytest.mark.parametrize("key", ["a", "b", "blend"])
def test_partial_procrustes_respects_its_bound(key):
    """d_P = 2 sin(rho/2) <= sqrt(2), because a = sigma_1 - sigma_2 >= 0."""

    result = analyze(demo(key))
    assert 0.0 <= result["distances"]["partial_procrustes"] <= MAXIMUM_PARTIAL_PROCRUSTES


def test_procrustes_distance_identities():
    """d_P^2 = 2 - 2a and d_F = sin(rho) for the same alignment (Dryden & Mardia)."""

    model = _fit_reference_model("pooled")
    engine = DescriptiveMorphometricEngine("pooled")
    result = engine.analyze_configuration(demo("blend"))

    aligned = np.asarray(result["aligned_configuration"])
    inner = float(aligned.ravel() @ model.gpa.consensus.ravel())
    partial = result["distances"]["partial_procrustes"]
    full = result["distances"]["full_procrustes"]

    assert partial**2 == pytest.approx(2.0 - 2.0 * inner, abs=TOLERANCE)
    assert full == pytest.approx(np.sqrt(max(0.0, 1.0 - inner**2)), abs=TOLERANCE)
    assert inner >= 0.0


def test_identical_configuration_has_zero_distance():
    model = _fit_reference_model("pooled")
    result = analyze(model.gpa.consensus)
    assert result["distances"]["partial_procrustes"] == pytest.approx(0.0, abs=1e-9)
    assert result["geometric_displacement_index"]["value"] == pytest.approx(
        0.0, abs=1e-8
    )


def test_displacement_index_matches_its_published_formula():
    for key in ("a", "b", "blend"):
        result = analyze(demo(key))
        expected = 10.0 * min(
            1.0,
            result["distances"]["partial_procrustes"] / MAXIMUM_PARTIAL_PROCRUSTES,
        )
        assert result["geometric_displacement_index"]["value"] == pytest.approx(
            expected, abs=TOLERANCE
        )


# ----------------------------------------------------------- tangent geometry


def test_tangent_basis_is_orthonormal_and_correctly_sized(pooled):
    basis = pooled.tangent_basis
    assert basis.shape == (AMBIENT_DIMENSION, TANGENT_DIMENSION)
    assert np.allclose(basis.T @ basis, np.eye(TANGENT_DIMENSION), atol=1e-10)


def test_tangent_basis_excludes_the_four_similarity_modes(pooled):
    consensus = pooled.gpa.consensus
    similarity = np.column_stack(
        (
            np.tile([1.0, 0.0], LANDMARK_COUNT),
            np.tile([0.0, 1.0], LANDMARK_COUNT),
            consensus.ravel(),
            np.column_stack((-consensus[:, 1], consensus[:, 0])).ravel(),
        )
    )
    assert np.allclose(pooled.tangent_basis.T @ similarity, 0.0, atol=1e-10)


def test_regression_tangent_basis_is_canonical(pooled):
    """Audit finding 8: remove arbitrary null-space basis orientation.

    ``scipy.linalg.null_space`` returns an arbitrary orthonormal basis for the
    subspace. The replacement is deterministic for a fixed consensus and
    arithmetic environment; this test does not claim cross-LAPACK bitwise
    equality for the complete pipeline.
    """

    first = _tangent_basis(pooled.gpa.consensus)
    second = _tangent_basis(pooled.gpa.consensus)
    assert np.array_equal(first, second)
    assert np.array_equal(first, pooled.tangent_basis)


def test_regression_pca_eigenvector_signs_are_fixed(pooled):
    """Eigenvector signs are arbitrary in LAPACK; the engine must pin them."""

    vectors = pooled.pca_eigenvectors
    dominant = np.argmax(np.abs(vectors), axis=0)
    assert np.all(vectors[dominant, np.arange(vectors.shape[1])] > 0.0)


def test_pca_eigenvalues_are_ordered_and_nonnegative(pooled):
    values = pooled.pca_eigenvalues
    assert np.all(np.diff(values) <= 1e-15)
    assert np.all(values >= 0.0)


# -------------------------------------------------------------------- GPA


def test_gpa_converges_and_produces_a_unit_consensus(pooled):
    assert pooled.gpa.converged
    assert np.linalg.norm(pooled.gpa.consensus) == pytest.approx(1.0, abs=TOLERANCE)
    assert np.allclose(pooled.gpa.consensus.mean(axis=0), 0.0, atol=TOLERANCE)


def test_gpa_rejects_degenerate_input():
    with pytest.raises(ValueError):
        iterative_generalized_procrustes(np.zeros((1, LANDMARK_COUNT, 2)))
    with pytest.raises(ValueError):
        iterative_generalized_procrustes(np.zeros((4, 12, 2)))


# ---------------------------------------------------------- audit regressions


def test_regression_template_is_face_shaped():
    """Audit finding 1: pin the audited canonical-model projection.

    This digest was independently calculated after selecting the browser's 68
    indices from MediaPipe's canonical OBJ at the declared source commit,
    reversing the vertical axis, centering, and normalizing centroid size.
    """

    # Hash the checked-in source coordinates, not the normalized derivative.
    # Re-normalization calls a BLAS-backed norm and can legitimately differ in
    # the final bits across numerical-library builds even when the geometry is
    # unchanged.
    digest = hashlib.sha256(
        np.asarray(CANONICAL_TEMPLATE, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    assert CANONICAL_TEMPLATE_SOURCE_COMMIT == (
        "a908d668c730da128dfa8d9f6bd25d519d006692"
    )
    assert digest == "bc32a46f49a41071b2d4ded4bdad26f215de4be5377f74811358978ce64861b3"
    assert np.allclose(SIMULATED_TEMPLATE_A, CANONICAL_TEMPLATE, atol=2.0e-9)

    result = analyze(SIMULATED_TEMPLATE_A)
    model = _fit_reference_model("pooled")
    assert result["distances"]["partial_procrustes"] < float(
        np.median(model.reference_partial_distances)
    )
    assert result["chart_diagnostics"]["within_reference_distance_range"]
    assert result["distances"]["regularized_mahalanobis_squared"] < TANGENT_DIMENSION


def test_regression_shrinkage_is_estimated_not_hard_coded(pooled):
    """Audit finding 4: the fixed 0.20 was six times the analytic optimum."""

    assert pooled.shrinkage_method == "analytic_schaefer_strimmer_diagonal_target"
    assert 0.0 <= pooled.shrinkage_intensity <= 1.0
    assert pooled.shrinkage_intensity < 0.10


def test_analytic_shrinkage_saturates_on_independent_coordinates():
    """With no reproducible cross-structure the optimum moves toward the target."""

    rng = np.random.RandomState(7)
    data = rng.normal(size=(8, 40))
    centered = data - data.mean(axis=0)
    sample = centered.T @ centered / (centered.shape[0] - 1)
    intensity = _analytic_shrinkage_intensity(centered, sample)
    assert intensity > 0.5


def test_regression_covariance_is_positive_definite_and_well_conditioned(pooled):
    eigenvalues = np.linalg.eigvalsh(pooled.covariance)
    assert eigenvalues.min() > 0.0
    assert pooled.covariance_condition_number < 1.0e8
    assert np.allclose(pooled.covariance, pooled.covariance.T, atol=1e-15)


def test_regression_held_out_mahalanobis_scale_is_not_collapsed():
    """Audit finding 6: a held-out simulation guards against scale collapse.

    The broad dimension-based interval is a numerical sanity check. It is not
    evidence of chi-square calibration because the reference is a regularized
    finite simulation and the pooled ensemble is a mixture.
    """

    held_out = _simulate_reference_ensemble(SIMULATED_TEMPLATE_A, 60, seed=555_001)
    engine = DescriptiveMorphometricEngine("pooled")
    squared = np.array(
        [
            engine.analyze_configuration(shape)["distances"][
                "regularized_mahalanobis_squared"
            ]
            for shape in held_out
        ]
    )
    assert 0.75 < squared.mean() / TANGENT_DIMENSION < 1.35


def test_regression_distant_configuration_is_flagged():
    """Audit finding 7: a mirrored input used to return D = 16690 with no warning."""

    result = analyze(demo("a") * np.array([-1.0, 1.0]))
    assert result["warnings"]
    assert not result["chart_diagnostics"][
        "tangent_projection_within_small_distortion_region"
    ]
    assert result["distances"]["partial_procrustes"] > TANGENT_CHART_LIMIT


def test_in_range_configuration_is_not_flagged():
    for key in ("a", "b", "blend"):
        result = analyze(demo(key))
        assert result["warnings"] == []
        assert result["chart_diagnostics"][
            "tangent_projection_within_small_distortion_region"
        ]


def test_regression_index_is_reported_with_reference_context():
    """Audit finding 3: the absolute 0-10 interval has no usable resolution."""

    result = analyze(demo("blend"))
    percentile = result["geometric_displacement_index"]["reference_percentile"]
    assert 0.0 <= percentile <= 100.0
    assert result["geometric_displacement_index"]["value"] < 1.5
    assert "resolution_note" in result["geometric_displacement_index"]


def test_regression_simulation_uses_a_stream_stable_generator():
    """Audit finding 9: pin the intentionally selected legacy test stream."""

    first = _simulate_reference_ensemble(SIMULATED_TEMPLATE_A, 5, seed=4242)
    second = _simulate_reference_ensemble(SIMULATED_TEMPLATE_A, 5, seed=4242)
    assert np.array_equal(first, second)

    expected = np.array(
        [
            0.43010114523104764,
            -0.17095125770225938,
            -0.24394639306625945,
            0.4730687718067827,
            0.64489206739365312,
            1.1556490036534721,
            -2.0595256529154695,
            0.36845887425885732,
            0.22778656631571534,
            1.7718175789887662,
            -1.0164603043906757,
            0.25844521229393053,
        ],
        dtype=np.float64,
    )
    assert np.array_equal(np.random.RandomState(4242).normal(size=12), expected)


def test_module_declares_its_licence():
    source = (Path(__file__).resolve().parents[1] / "core" / "analytics.py").read_text()
    assert "SPDX-License-Identifier: MIT" in source
    assert "MIT License" in source


# ------------------------------------------------------------ input handling


def test_templates_are_valid_unit_preshapes():
    for template in (SIMULATED_TEMPLATE_A, SIMULATED_TEMPLATE_B):
        assert template.shape == (LANDMARK_COUNT, 2)
        assert np.linalg.norm(template) == pytest.approx(1.0, abs=TOLERANCE)
        assert np.allclose(template.mean(axis=0), 0.0, atol=TOLERANCE)


def test_templates_are_distinguishable():
    preshape_a, _, _ = _to_preshape(SIMULATED_TEMPLATE_A)
    preshape_b, _, _ = _to_preshape(SIMULATED_TEMPLATE_B)
    assert np.linalg.norm(preshape_a - preshape_b) > 0.01


def test_overflowing_coordinates_get_their_own_message():
    """Centroid-size overflow must not be misattributed to the tangent chart."""

    maximum = np.finfo(np.float64).max
    landmarks = np.tile(
        [[maximum, 0.0], [-maximum, 0.0]],
        (LANDMARK_COUNT // 2, 1),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = json.loads(run_pipeline_from_js(landmarks.ravel().tolist()))
    assert result["error_type"] == "ValueError"
    assert "too large" in result["error"]
    assert caught == []


def test_scale_invariance_spans_representable_magnitudes():
    baseline = analyze(demo("a"))["distances"]
    for scale in (1.0e-200, 1.0e-20, 1.0e150, 1.0e300):
        scaled = analyze(demo("a") * scale)["distances"]
        for name, value in baseline.items():
            assert scaled[name] == pytest.approx(value, abs=TOLERANCE)


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        json.dumps([[1.0, 2.0]] * 67),
        json.dumps([[1.0, 2.0]] * 69),
        json.dumps([[0.0, 0.0]] * LANDMARK_COUNT),
        json.dumps([[float("nan"), 0.0]] * LANDMARK_COUNT),
    ],
)
def test_malformed_input_returns_a_structured_error(payload):
    result = json.loads(run_pipeline_from_js(payload))
    assert "error" in result
    assert "error_type" in result


def test_unknown_reference_population_is_rejected():
    result = analyze(demo("a"), reference="nonexistent")
    assert "error" in result


def test_accepts_flat_nested_and_ndarray_input():
    landmarks = demo("a")
    flat = json.loads(run_pipeline_from_js(landmarks.ravel().tolist()))
    nested = json.loads(run_pipeline_from_js(landmarks.tolist()))
    array = json.loads(run_pipeline_from_js(landmarks))
    text = json.loads(run_pipeline_from_js(json.dumps(landmarks.tolist())))
    for other in (nested, array, text):
        assert other["distances"] == flat["distances"]


# ----------------------------------------------------------- output contract


REQUIRED_FIELDS = (
    ("distances", "partial_procrustes"),
    ("distances", "full_procrustes"),
    ("distances", "regularized_mahalanobis"),
    ("gpa", "iterations"),
    ("gpa", "converged"),
    ("input_geometry", "centroid_size"),
    ("reference", "key"),
    ("reference", "sample_size"),
    ("residual_shape_difference", "aligned_vectors"),
    ("residual_shape_difference", "input_orientation_vectors"),
    ("residual_shape_difference", "root_mean_square_magnitude"),
    ("tangent_space", "dimension"),
    ("tangent_space", "pca_scores"),
    ("tangent_space", "pca_explained_variance_ratio"),
    ("covariance_diagnostics", "condition_number"),
    ("covariance_diagnostics", "shrinkage_to_diagonal"),
    ("chart_diagnostics", "tangent_projection_within_small_distortion_region"),
    ("geometric_displacement_index", "reference_percentile"),
)


@pytest.mark.parametrize("block,field", REQUIRED_FIELDS)
def test_front_end_contract_is_preserved(block, field):
    """assets/js/app.js reads these paths; adding fields is fine, removing is not."""

    assert field in analyze(demo("a"))[block]


def test_top_level_arrays_have_the_expected_shapes():
    result = analyze(demo("a"))
    assert isinstance(result["warnings"], list)
    assert np.asarray(result["aligned_configuration"]).shape == (LANDMARK_COUNT, 2)
    assert np.asarray(result["reference_consensus"]).shape == (LANDMARK_COUNT, 2)
    assert len(result["tangent_space"]["coordinates"]) == TANGENT_DIMENSION
    assert len(result["residual_shape_difference"]["magnitudes"]) == LANDMARK_COUNT


def test_output_is_strict_json_without_nonfinite_values():
    payload = run_pipeline_from_js(demo("blend").ravel().tolist())
    assert "NaN" not in payload and "Infinity" not in payload
    json.loads(payload)


def test_interpretation_metadata_stays_non_normative():
    result = analyze(demo("a"))
    assert result["interpretation"]["normative"] is False
    assert result["interpretation"]["classification"] is False
    assert result["interpretation"]["prescriptive_modification"] is False
    assert result["geometric_displacement_index"]["appearance_rating"] is False
    assert result["reference"]["is_simulated"] is True
    assert result["reference"]["is_empirical_population_estimate"] is False


@pytest.mark.parametrize("reference", ["a", "b", "pooled"])
def test_every_reference_choice_produces_a_complete_result(reference):
    result = analyze(demo("a"), reference=reference)
    assert "error" not in result
    assert result["reference"]["key"] == reference
    assert result["reference"]["sample_size"] in (160, 320)
