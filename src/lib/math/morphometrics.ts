/** Descriptive, image-plane facial geometry for the experimental AME preview.
 *
 * No biological sex, gender, ancestry, age, skin quality, health, or aesthetic
 * value is inferred. Reference shapes below are simulations, never clinical
 * or demographic baselines. MediaPipe x/y coordinates are image normalized;
 * its z value is not a calibrated anatomical depth measurement.
 */

import type { MeshPoint } from "../vision/landmarker";
import fixture from "./simulated-references.json" with { type: "json" };

export type Point2D = readonly [number, number];
export type SimulatedReferenceKey = "a" | "b" | "pooled";
export type MeasuredMetric = Readonly<{
  key: string;
  value: number | null;
  unit: "ratio" | "% of jaw span";
  status: "measured" | "unavailable";
  landmark_indices: readonly number[];
  definition: string;
  reason: string | null;
}>;

const MIN_SPAN = 1e-12;

// Identical to assets/js/photo-warp.mjs, which maps the 478-point model into
// the Dlib-style index order. A contract test compares both arrays.
export const MEDIAPIPE_TO_DLIB_68 = [
  234, 93, 132, 58, 136, 150, 149, 148, 152, 377, 378, 379, 365, 288, 361, 323, 454,
  70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 168, 6, 197, 4,
  98, 97, 2, 326, 327, 33, 160, 158, 133, 153, 144,
  362, 385, 387, 263, 373, 380, 61, 40, 37, 0, 267, 270, 291,
  321, 314, 17, 84, 91, 78, 81, 13, 311, 308, 402, 14, 178,
] as const;

export function denseMeshToDlib68(
  mesh: readonly MeshPoint[], imageWidth: number, imageHeight: number,
): Point2D[] {
  if (!Array.isArray(mesh) || mesh.length !== 478 ||
      mesh.some((point) => !point || ![point.x, point.y, point.z].every(Number.isFinite))) {
    throw new Error("Expected a single 478-point finite face mesh.");
  }
  if (![imageWidth, imageHeight].every((size) => Number.isSafeInteger(size) && size > 0)) {
    throw new Error("Image dimensions must be positive finite integers.");
  }
  return MEDIAPIPE_TO_DLIB_68.map((index) => {
    const x = mesh[index].x * imageWidth;
    const y = mesh[index].y * imageHeight;
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      throw new Error(`Projected mesh coordinate ${index} is not finite.`);
    }
    return [x, y];
  });
}

function preshape(landmarks: readonly Point2D[]): Point2D[] {
  if (!Array.isArray(landmarks) || landmarks.length !== 68 ||
      landmarks.some((point) => !Array.isArray(point) || point.length !== 2 ||
        !Number.isFinite(point[0]) || !Number.isFinite(point[1]))) {
    throw new Error("Expected 68 ordered finite two-dimensional landmarks.");
  }
  const peak = Math.max(...landmarks.flatMap(([x, y]) => [Math.abs(x), Math.abs(y)]));
  if (peak === 0) throw new Error("The configuration has zero centroid size.");
  const scaled = landmarks.map(([x, y]) => [x / peak, y / peak] as const);
  if (scaled.every(([x, y]) => x === scaled[0][0] && y === scaled[0][1])) {
    throw new Error("The configuration has zero centroid size.");
  }
  const meanX = scaled.reduce((sum, [x]) => sum + x, 0) / 68;
  const meanY = scaled.reduce((sum, [, y]) => sum + y, 0) / 68;
  const centered = scaled.map(([x, y]) => [x - meanX, y - meanY] as const);
  const size = Math.hypot(...centered.flatMap(([x, y]) => [x, y]));
  if (size === 0) throw new Error("The configuration has zero centroid size.");
  return centered.map(([x, y]) => [x / size, y / size]);
}

function distance(points: readonly Point2D[], a: number, b: number): number {
  return Math.hypot(points[a][0] - points[b][0], points[a][1] - points[b][1]);
}

function metric(
  key: string, numerator: number | null, denominator: number,
  indices: readonly number[], definition: string, unit: MeasuredMetric["unit"] = "ratio",
): MeasuredMetric {
  const available = numerator !== null && Number.isFinite(numerator) &&
    Number.isFinite(denominator) && denominator > MIN_SPAN;
  return {
    key, value: available ? numerator / denominator : null, unit,
    status: available ? "measured" : "unavailable", landmark_indices: indices,
    definition, reason: available ? null : "Required span is missing or too small.",
  };
}

const PAIRS = {
  eyes: [[36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46]],
  jaw: Array.from({ length: 8 }, (_, index) => [index, 16 - index]),
  mouth: [[48, 54], [49, 53], [50, 52], [55, 59], [56, 58]],
} as const;

/** Match core/frontal_assessment.py; values describe this 2D projection only. */
export function assessFrontalLandmarks(landmarks: readonly Point2D[]): Readonly<{
  schema_version: "frontal-landmarks-v1";
  analysis_kind: "exploratory_2d_frontal_measurements";
  reference_population: null;
  attractiveness_score: null;
  pose_verified: false;
  image_quality_verified: false;
  metrics: readonly MeasuredMetric[];
}> {
  const points = preshape(landmarks);
  const eyeWidth = (distance(points, 36, 39) + distance(points, 42, 45)) / 2;
  const innerEye = distance(points, 39, 42);
  const nose = distance(points, 31, 35);
  const mouth = distance(points, 48, 54);
  const jaw = distance(points, 0, 16);
  const axisX = points[8][0] - points[27][0];
  const axisY = points[8][1] - points[27][1];
  const axisLength = Math.hypot(axisX, axisY);
  const direction = axisLength > MIN_SPAN ? [axisX / axisLength, axisY / axisLength] : null;
  let lower: number | null = null;
  if (direction) {
    const projection = (points[33][0] - points[27][0]) * direction[0] +
      (points[33][1] - points[27][1]) * direction[1];
    if (projection >= 0 && projection <= axisLength) lower = axisLength - projection;
  }

  const metrics: MeasuredMetric[] = [
    metric("inner_eye_to_eye_width", innerEye, eyeWidth, [36, 39, 42, 45],
      "Distance between inner eye corners divided by mean visible eye width."),
    metric("nose_to_inner_eye", nose, innerEye, [31, 35, 39, 42],
      "Visible nose-base width divided by inner eye-corner distance."),
    metric("mouth_to_nose", mouth, nose, [31, 35, 48, 54],
      "Mouth-corner span divided by visible nose-base width."),
    metric("mouth_to_jaw", mouth, jaw, [0, 16, 48, 54],
      "Mouth-corner span divided by jaw-outline span in this image."),
    metric("lower_visible_face_fraction", lower, axisLength, [8, 27, 33],
      "Projected nose-base-to-chin length divided by nose-bridge-to-chin length. " +
      "The hairline is not detected, so this is not a classical facial-third measure."),
  ];
  for (const [region, pairs] of Object.entries(PAIRS)) {
    let discrepancy: number | null = null;
    if (direction) {
      const [dx, dy] = direction;
      discrepancy = pairs.reduce((sum, [first, second]) => {
        const offsetX = points[first][0] - points[27][0];
        const offsetY = points[first][1] - points[27][1];
        const projection = offsetX * dx + offsetY * dy;
        const reflectedX = points[27][0] + 2 * projection * dx - offsetX;
        const reflectedY = points[27][1] + 2 * projection * dy - offsetY;
        return sum + Math.hypot(reflectedX - points[second][0], reflectedY - points[second][1]);
      }, 0) / pairs.length;
    }
    metrics.push(metric(`${region}_paired_discrepancy_pct`,
      discrepancy === null ? null : 100 * discrepancy, jaw,
      [...pairs.flat(), 27, 8, 0, 16],
      "Mean distance after reflecting paired landmarks across the nose-bridge-to-chin " +
      "axis, divided by jaw-outline span; percent of image span, not a beauty score.",
      "% of jaw span"));
  }
  return {
    schema_version: "frontal-landmarks-v1",
    analysis_kind: "exploratory_2d_frontal_measurements",
    reference_population: null,
    attractiveness_score: null,
    pose_verified: false,
    image_quality_verified: false,
    metrics,
  };
}

export function compareWithSimulatedReference(
  landmarks: readonly Point2D[], key: SimulatedReferenceKey,
): Readonly<{
  analysis_kind: "descriptive_simulated_reference_comparison";
  reference_kind: "simulated";
  reference_key: SimulatedReferenceKey;
  reference_sample_size: number;
  partial_procrustes_distance: number;
  full_procrustes_distance: number;
  attractiveness_score: null;
}> {
  if (fixture.kind !== "simulated" || fixture.schema_version !== "ame-simulated-consensus-v1") {
    throw new Error("The simulated-reference fixture is invalid.");
  }
  if (!Object.hasOwn(fixture.references, key)) {
    throw new Error("Choose a simulated reference: a, b, or pooled.");
  }
  const reference = fixture.references[key];
  const input = preshape(landmarks);
  const target = preshape(reference.consensus.map(([x, y]) => [x, y]));
  // Minimize ||input R - target|| with det(R)=+1. Reflection is prohibited.
  let cosineTerm = 0;
  let sineTerm = 0;
  for (let index = 0; index < 68; index += 1) {
    const [x, y] = input[index];
    const [referenceX, referenceY] = target[index];
    cosineTerm += x * referenceX + y * referenceY;
    sineTerm += y * referenceX - x * referenceY;
  }
  const angle = Math.atan2(sineTerm, cosineTerm);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  let squaredDistance = 0;
  let alignment = 0;
  for (let index = 0; index < 68; index += 1) {
    const [x, y] = input[index];
    const [referenceX, referenceY] = target[index];
    const rotatedX = x * cos + y * sin;
    const rotatedY = -x * sin + y * cos;
    squaredDistance += (rotatedX - referenceX) ** 2 + (rotatedY - referenceY) ** 2;
    alignment += rotatedX * referenceX + rotatedY * referenceY;
  }
  return {
    analysis_kind: "descriptive_simulated_reference_comparison",
    reference_kind: "simulated",
    reference_key: key,
    reference_sample_size: reference.sample_size,
    partial_procrustes_distance: Math.sqrt(squaredDistance),
    full_procrustes_distance: Math.sqrt(Math.max(0, 1 - Math.min(1, alignment) ** 2)),
    attractiveness_score: null,
  };
}
