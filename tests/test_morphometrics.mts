import assert from "node:assert/strict";
import { test } from "node:test";
import {
  MEDIAPIPE_TO_DLIB_68,
  assessDenseImageMeasurements,
  assessFrontalLandmarks,
  compareWithSimulatedReference,
  denseMeshToDlib68,
  type Point2D,
} from "../src/lib/math/morphometrics.ts";
import fixture from "../src/lib/math/simulated-references.json" with { type: "json" };

const canonical: Point2D[] = fixture.cross_language_check.input.map(([x, y]) => [x, y]);

test("reference provenance exposes simulated samples without demographic labels", () => {
  assert.equal(fixture.kind, "simulated");
  assert.deepEqual(Object.keys(fixture.references).sort(), ["a", "b", "pooled"]);
  assert.deepEqual(Object.values(fixture.references).map((reference) => reference.sample_size), [160, 160, 320]);
  assert.match(fixture.interpretation, /No empirical cohort/);
  for (const reference of Object.values(fixture.references)) {
    assert.equal(reference.kind, "simulated");
    assert.equal(reference.consensus.length, 68);
    const size = Math.hypot(...reference.consensus.flat());
    assert.ok(Math.abs(size - 1) < 1e-12);
  }
});

test("the 478-to-68 correspondence and image aspect ratio match the Pages adapter", async () => {
  const legacy = await import(new URL("../assets/js/photo-warp.mjs", import.meta.url).href);
  assert.deepEqual([...MEDIAPIPE_TO_DLIB_68], [...legacy.MEDIAPIPE_TO_DLIB_68]);
  const mesh = Array.from({ length: 478 }, (_, index) => ({
    x: index / 700, y: index / 800, z: index / 900,
  }));
  const result = denseMeshToDlib68(mesh, 1200, 600);
  assert.deepEqual(result, legacy.denseMeshToDlib68(mesh, 1200, 600));
  assert.equal(result[8][0], mesh[152].x * 1200);
  assert.equal(result[8][1], mesh[152].y * 600);
  assert.throws(() => denseMeshToDlib68(mesh.slice(1), 1200, 600), /478-point/);
  assert.throws(() => denseMeshToDlib68(mesh, 0, 600), /dimensions/);
  mesh[152] = { x: Infinity, y: 0, z: 0 };
  assert.throws(() => denseMeshToDlib68(mesh, 1200, 600), /finite/);
});

test("all eight frontal values match the Python engine's exported check", () => {
  const report = assessFrontalLandmarks(canonical);
  assert.equal(report.reference_population, null);
  assert.equal(report.attractiveness_score, null);
  assert.equal(report.pose_verified, false);
  assert.equal(report.metrics.length, 8);
  for (const metric of report.metrics) {
    const expected = fixture.cross_language_check.frontal_values[
      metric.key as keyof typeof fixture.cross_language_check.frontal_values
    ];
    assert.equal(metric.status, "measured");
    assert.ok(Math.abs((metric.value ?? NaN) - expected) < 1e-11, metric.key);
  }
});

test("proper Procrustes distance matches the Python SVD alignment", () => {
  for (const key of ["a", "b", "pooled"] as const) {
    const result = compareWithSimulatedReference(canonical, key);
    assert.equal(result.reference_kind, "simulated");
    assert.equal(result.attractiveness_score, null);
    assert.ok(Math.abs(result.partial_procrustes_distance -
      fixture.cross_language_check.partial_procrustes[key]) < 1e-11, key);
    assert.ok(result.full_procrustes_distance >= 0);
  }
});

test("translation, scale and rotation leave the descriptive quantities unchanged", () => {
  const base = assessFrontalLandmarks(canonical).metrics;
  const baseDistance = compareWithSimulatedReference(canonical, "pooled").partial_procrustes_distance;
  const angle = 0.91;
  const transformed: Point2D[] = canonical.map(([x, y]) => [
    103 * (x * Math.cos(angle) + y * Math.sin(angle)) + 200,
    103 * (-x * Math.sin(angle) + y * Math.cos(angle)) - 173,
  ]);
  const moved = assessFrontalLandmarks(transformed).metrics;
  for (let index = 0; index < base.length; index += 1) {
    assert.ok(Math.abs((base[index].value ?? NaN) - (moved[index].value ?? NaN)) < 1e-11);
  }
  assert.ok(Math.abs(compareWithSimulatedReference(transformed, "pooled")
    .partial_procrustes_distance - baseDistance) < 1e-11);
});

test("large coordinate origins preserve the representable geometry", () => {
  const shifted: Point2D[] = canonical.map(([x, y]) => [x + 1e10, y - 1e10]);
  const recentered: Point2D[] = shifted.map(([x, y]) => [x - 1e10, y + 1e10]);
  const actual = assessFrontalLandmarks(shifted).metrics;
  const expected = assessFrontalLandmarks(recentered).metrics;
  for (let index = 0; index < actual.length; index += 1) {
    assert.ok(Math.abs((actual[index].value ?? NaN) - (expected[index].value ?? NaN)) < 1e-12,
      actual[index].key);
  }
  assert.ok(Math.abs(compareWithSimulatedReference(shifted, "pooled").partial_procrustes_distance -
    compareWithSimulatedReference(recentered, "pooled").partial_procrustes_distance) < 1e-12);
});

test("dense image observations use eye and outline points without anatomical targets", () => {
  const mesh = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  for (const [index, x, y] of [
    [33, .2, .4], [263, .8, .4], [133, .35, .42], [362, .65, .42],
    [234, .1, .5], [454, .9, .5], [148, .25, .8], [377, .75, .8],
  ]) mesh[index] = { x, y, z: 0 };
  const measured = assessDenseImageMeasurements(mesh, 1000, 1000);
  const values = Object.fromEntries(measured.metrics.map((metric) => [metric.key, metric.value]));
  assert.equal(measured.pose_verified, false);
  assert.ok(Math.abs((values.intercanthal_to_outer_eye_span ?? NaN) - .5) < 1e-12);
  assert.ok(Math.abs((values.lower_outline_to_cheek_width ?? NaN) - .625) < 1e-12);
  assert.ok((values.viewer_left_eye_tilt_deg ?? 0) > 0);
  assert.ok((values.viewer_right_eye_tilt_deg ?? 0) > 0);
  assert.ok(measured.not_assessed.some((item) => item.includes("Gonion")));

  const angle = .62;
  const rotated = mesh.map(({ x, y, z }) => ({
    x: .5 + (x - .5) * Math.cos(angle) + (y - .5) * Math.sin(angle),
    y: .5 - (x - .5) * Math.sin(angle) + (y - .5) * Math.cos(angle), z,
  }));
  const moved = assessDenseImageMeasurements(rotated, 1000, 1000);
  for (const metric of measured.metrics) {
    const rotatedValue = moved.metrics.find((other) => other.key === metric.key)?.value;
    assert.ok(Math.abs((metric.value ?? NaN) - (rotatedValue ?? NaN)) < 1e-10, metric.key);
  }

  mesh[263] = mesh[33];
  const missing = assessDenseImageMeasurements(mesh, 1000, 1000).metrics;
  assert.equal(missing.find((metric) => metric.key === "intercanthal_to_outer_eye_span")?.status, "unavailable");
  assert.equal(missing.find((metric) => metric.key === "viewer_left_eye_tilt_deg")?.value, null);
});

test("reflection is not an allowed Procrustes rotation", () => {
  const exemplar: Point2D[] = fixture.references.a.consensus.map(([x, y]) => [x, y]);
  const mirrored: Point2D[] = exemplar.map(([x, y]) => [-x, y]);
  assert.ok(compareWithSimulatedReference(exemplar, "a").partial_procrustes_distance < 1e-12);
  assert.ok(compareWithSimulatedReference(mirrored, "a").partial_procrustes_distance > 0.02);
});

test("missing spans are unavailable and malformed configurations fail", () => {
  const modified = canonical.map(([x, y]) => [x, y] as Point2D);
  modified[35] = modified[31];
  const mouthNose = assessFrontalLandmarks(modified).metrics.find((metric) => metric.key === "mouth_to_nose");
  assert.equal(mouthNose?.status, "unavailable");
  assert.equal(mouthNose?.value, null);
  assert.throws(() => assessFrontalLandmarks(canonical.slice(1)), /68 ordered/);
  assert.throws(() => assessFrontalLandmarks(Array.from({ length: 68 }, () => [2, 3])), /zero centroid/);
  assert.throws(() => compareWithSimulatedReference(canonical, "male" as "a"), /simulated reference/);
});
