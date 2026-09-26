import assert from "node:assert/strict";
import { test } from "node:test";
import {
  MEDIAPIPE_TO_DLIB_68,
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
