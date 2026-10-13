"""Regression oracles for input integrity and representable shape geometry."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.analytics import (  # noqa: E402
    CANONICAL_TEMPLATE,
    DescriptiveMorphometricEngine,
    _fit_reference_model,
    _to_preshape,
    iterative_generalized_procrustes,
    run_pipeline_from_js,
)
from core.frontal_assessment import assess_frontal_landmarks_json  # noqa: E402


@pytest.mark.parametrize("offset", [1.0e10, 1.0e12, 1.0e13, 3.0e13])
def test_core_preserves_quantized_geometry_when_coordinate_origin_is_large(offset):
    moved = CANONICAL_TEMPLATE + np.array([offset, -offset])
    # Compare exactly the same quantized data, not an unrecoverable original.
    local = moved - np.array([offset, -offset])
    actual, _, size = _to_preshape(moved)
    expected, _, expected_size = _to_preshape(local)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(actual.mean(axis=0), 0.0, rtol=0.0, atol=1.0e-15)
    assert size == pytest.approx(expected_size, rel=1.0e-14)
    engine = DescriptiveMorphometricEngine("pooled")
    observed = engine.analyze_configuration(moved)["distances"]
    oracle = engine.analyze_configuration(local)["distances"]
    for key in oracle:
        assert observed[key] == pytest.approx(oracle[key], abs=1.0e-12), key


def malformed_inputs():
    masked = np.ma.array(CANONICAL_TEMPLATE.copy(), mask=False)
    masked.mask[0, 0] = True
    mixed = CANONICAL_TEMPLATE.tolist()
    mixed[0][0] = True
    giant = CANONICAL_TEMPLATE.tolist()
    giant[0][0] = 10 ** 1000
    return [
        pytest.param(CANONICAL_TEMPLATE.T, id="transposed"),
        pytest.param(CANONICAL_TEMPLATE.reshape(1, 68, 2), id="extra-axis"),
        pytest.param(CANONICAL_TEMPLATE.reshape(34, 4), id="wrong-row-width"),
        pytest.param(CANONICAL_TEMPLATE.astype(str), id="numeric-strings"),
        pytest.param(CANONICAL_TEMPLATE.astype(complex) + 1j, id="complex"),
        pytest.param(CANONICAL_TEMPLATE > 0, id="booleans"),
        pytest.param(mixed, id="mixed-boolean"),
        pytest.param(masked, id="missing-mask"),
        pytest.param(giant, id="unrepresentable-integer"),
    ]


@pytest.mark.parametrize("payload", malformed_inputs())
def test_both_python_bridges_reject_ambiguous_or_coerced_coordinate_data(payload):
    for bridge in (run_pipeline_from_js, assess_frontal_landmarks_json):
        result = json.loads(bridge(payload))
        assert "error" in result, bridge.__name__


@pytest.mark.parametrize("tolerance", [np.nan, np.inf, True, False, "0.01", 0, -1, 10 ** 1000])
def test_gpa_requires_a_positive_finite_real_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        iterative_generalized_procrustes(
            np.stack([CANONICAL_TEMPLATE] * 2), tolerance=tolerance
        )


@pytest.mark.parametrize("iterations", [True, False, 1.5, np.nan, np.inf, "10", 0])
def test_gpa_requires_a_positive_integer_iteration_limit(iterations):
    with pytest.raises(ValueError, match="max_iterations"):
        iterative_generalized_procrustes(
            np.stack([CANONICAL_TEMPLATE] * 2), max_iterations=iterations
        )


def test_cached_reference_arrays_cannot_be_accidentally_changed():
    model = _fit_reference_model("pooled")
    for array in (
        model.gpa.consensus, model.gpa.aligned_shapes, model.gpa.per_shape_distances,
        model.tangent_basis, model.tangent_mean, model.covariance,
        model.covariance_cholesky[0], model.pca_eigenvalues, model.pca_eigenvectors,
        model.reference_partial_distances, model.reference_mahalanobis_squared,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = array.flat[0] + 1


def test_numpy_integer_limits_cannot_wrap_the_gpa_loop_to_zero_iterations():
    result = iterative_generalized_procrustes(
        np.stack([CANONICAL_TEMPLATE] * 2), max_iterations=np.iinfo(np.int64).max
    )
    assert result.iterations == 1 and result.converged


def test_gpa_does_not_drop_missing_coordinate_masks():
    sample = np.ma.array(np.stack([CANONICAL_TEMPLATE] * 2), mask=False)
    sample.mask[0, 0, 0] = True
    with pytest.raises(ValueError, match="Missing landmark"):
        iterative_generalized_procrustes(sample)
