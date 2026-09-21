import assert from "node:assert/strict";
import test from "node:test";

import {
  FACE_LANDMARKER_MODEL_SHA256,
  MEDIAPIPE_TO_DLIB_68,
  buildPhotoWarpGeometry,
  delaunayTriangulate,
  geometricDisplacementIndex,
} from "../assets/js/photo-warp.mjs";


test("MediaPipe adapter is a fixed 68-vertex correspondence", () => {
  assert.equal(MEDIAPIPE_TO_DLIB_68.length, 68);
  assert.equal(new Set(MEDIAPIPE_TO_DLIB_68).size, 68);
  assert.ok(MEDIAPIPE_TO_DLIB_68.every(Number.isSafeInteger));
  assert.match(FACE_LANDMARKER_MODEL_SHA256, /^[0-9a-f]{64}$/u);
});


test("display index implements the published bounded formula", () => {
  assert.equal(geometricDisplacementIndex(0), 0);
  assert.equal(geometricDisplacementIndex(Math.SQRT2), 10);
  assert.equal(geometricDisplacementIndex(10), 10);
  assert.equal(geometricDisplacementIndex(-1), 0);
  assert.throws(() => geometricDisplacementIndex(Number.NaN), /finite number/u);
});


test("Delaunay triangulation covers a simple square without degenerate faces", () => {
  const points = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const triangles = delaunayTriangulate(points);
  assert.equal(triangles.length, 2);
  for (const triangle of triangles) {
    assert.equal(new Set(triangle).size, 3);
    assert.ok(triangle.every((index) => index >= 0 && index < points.length));
  }
});


test("zero residual field produces an identity photo warp", () => {
  const sourceLandmarks = Array.from({ length: 68 }, (_, index) => {
    const row = Math.floor(index / 17);
    const column = index % 17;
    return [80 + column * 12 + row * 0.37, 70 + row * 34 + column * 0.11];
  });
  const residualVectors = sourceLandmarks.map(() => [0, 0]);
  const geometry = buildPhotoWarpGeometry({
    sourceLandmarks,
    residualVectors,
    centroidSize: 300,
    scale: 4,
    width: 400,
    height: 300,
  });

  assert.ok(geometry.triangles.length > 0);
  assert.equal(geometry.effectiveScaleFactor, 1);
  assert.deepEqual(geometry.destinationPoints, geometry.sourcePoints);
});
