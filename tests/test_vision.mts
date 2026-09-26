import assert from "node:assert/strict";
import { test } from "node:test";
import {
  MESH_LANDMARK_COUNT,
  MODEL_SHA256,
  drawMeshOverlay,
  scaledDimensions,
  singleFaceMesh,
  validatePhotoFile,
  verifyModelBuffer,
} from "../src/lib/vision/landmarker.ts";

const validFile = { name: "portrait.jpg", type: "image/jpeg", size: 120_000 };

test("the model pin is an exact SHA-256 digest", () => {
  assert.match(MODEL_SHA256, /^[a-f0-9]{64}$/);
});

test("only supported, bounded image files enter the capture route", () => {
  assert.doesNotThrow(() => validatePhotoFile(validFile));
  assert.throws(() => validatePhotoFile({ ...validFile, name: "portrait.svg", type: "image/svg+xml" }), /JPEG, PNG, or WebP/);
  assert.throws(() => validatePhotoFile({ ...validFile, name: "portrait.exe" }), /JPEG, PNG, or WebP/);
  assert.throws(() => validatePhotoFile({ ...validFile, size: 20 * 1024 * 1024 + 1 }), /20 MiB/);
  assert.throws(() => validatePhotoFile({ ...validFile, size: 0 }), /nonempty/);
});

test("large uploads are bounded before making the analysis canvas", () => {
  assert.deepEqual(scaledDimensions(4000, 3000), { width: 1600, height: 1200 });
  assert.deepEqual(scaledDimensions(600, 450), { width: 600, height: 450 });
  assert.throws(() => scaledDimensions(10_000, 10_000), /36 megapixels/);
  assert.throws(() => scaledDimensions(0, 100), /dimensions/);
});

test("zero faces are explicit, and the 478-point contract is strict", () => {
  assert.equal(singleFaceMesh({ faceLandmarks: [] }), null);
  const mesh = Array.from({ length: MESH_LANDMARK_COUNT }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  assert.equal(singleFaceMesh({ faceLandmarks: [mesh] }), mesh);
  assert.throws(() => singleFaceMesh({ faceLandmarks: [mesh.slice(1)] }), /478 finite/);
  assert.throws(() => singleFaceMesh({ faceLandmarks: [mesh, mesh] }), /exactly one/);
  mesh[10] = { x: NaN, y: 0.5, z: 0 };
  assert.throws(() => singleFaceMesh({ faceLandmarks: [mesh] }), /478 finite/);
});

test("model bytes pass only for the expected digest", async () => {
  const bytes = new TextEncoder().encode("abc").buffer;
  const abcSha256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
  await assert.doesNotReject(verifyModelBuffer(bytes, abcSha256));
  await assert.rejects(verifyModelBuffer(bytes), /integrity failed/);
  await assert.rejects(verifyModelBuffer(bytes, abcSha256, {} as Pick<Crypto, "subtle">), /HTTPS or localhost/);
});

test("overlay paints the source, mesh edges, all 478 points, and a three-point guide", () => {
  const calls = { image: 0, line: 0, arc: 0 };
  const context = {
    drawImage: () => { calls.image += 1; },
    beginPath: () => {},
    moveTo: () => {},
    lineTo: () => { calls.line += 1; },
    arc: () => { calls.arc += 1; },
    stroke: () => {},
    fill: () => {},
  };
  const canvas = { width: 640, height: 480, getContext: () => context } as unknown as HTMLCanvasElement;
  const mesh = Array.from({ length: MESH_LANDMARK_COUNT }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  drawMeshOverlay(canvas, canvas, mesh, [{ start: 0, end: 1 }, { start: 1, end: 2 }]);
  assert.deepEqual(calls, { image: 1, line: 4, arc: 478 });
  drawMeshOverlay(canvas, canvas, null, []);
  assert.equal(calls.image, 2);
});
