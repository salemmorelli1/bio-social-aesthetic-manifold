"use strict";

/**
 * Local photo-landmark and piecewise-affine texture-warp utilities.
 *
 * Image pixels remain in the browser. MediaPipe is used only to locate one
 * dense face mesh; this adapter samples that mesh into the fixed 68-point
 * topology used by the descriptive Python engine. No blendshape, identity,
 * demographic, emotion, health, or appearance inference is requested.
 */

export const MEDIAPIPE_VERSION = "1.0.1";
export const MEDIAPIPE_MODULE_URL =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`;
export const MEDIAPIPE_WASM_ROOT =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
export const FACE_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
export const FACE_LANDMARKER_MODEL_SHA256 =
  "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff";

// A documented correspondence adapter from MediaPipe's dense mesh to the
// canonical Dlib-style 68-point order. It is not Dlib detector output and does
// not claim anatomical equivalence between the two landmark definitions.
export const MEDIAPIPE_TO_DLIB_68 = Object.freeze([
  // 0-16: jawline
  234, 93, 132, 58, 136, 150, 149, 148, 152, 377, 378, 379, 365, 288, 361, 323, 454,
  // 17-21: right eyebrow (viewer left)
  70, 63, 105, 66, 107,
  // 22-26: left eyebrow (viewer right)
  336, 296, 334, 293, 300,
  // 27-30: nose bridge and tip
  168, 6, 197, 4,
  // 31-35: nose base
  98, 97, 2, 326, 327,
  // 36-41: right eye (viewer left)
  33, 160, 158, 133, 153, 144,
  // 42-47: left eye (viewer right)
  362, 385, 387, 263, 373, 380,
  // 48-59: outer lip
  61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
  // 60-67: inner lip
  78, 81, 13, 311, 308, 402, 14, 178,
]);

export const DLIB_LANDMARK_PATHS = Object.freeze([
  Object.freeze({ start: 0, end: 16, closed: false }),
  Object.freeze({ start: 17, end: 21, closed: false }),
  Object.freeze({ start: 22, end: 26, closed: false }),
  Object.freeze({ start: 27, end: 30, closed: false }),
  Object.freeze({ start: 31, end: 35, closed: false }),
  Object.freeze({ start: 36, end: 41, closed: true }),
  Object.freeze({ start: 42, end: 47, closed: true }),
  Object.freeze({ start: 48, end: 59, closed: true }),
  Object.freeze({ start: 60, end: 67, closed: true }),
]);

let faceLandmarkerPromise = null;

function bytesToHex(bytes) {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
}

async function fetchVerifiedFaceLandmarkerModel() {
  if (!globalThis.crypto?.subtle) {
    throw new Error(
      "Model verification requires Web Crypto in a secure browser context. Use HTTPS or localhost."
    );
  }
  const response = await fetch(FACE_LANDMARKER_MODEL_URL, {
    mode: "cors",
    credentials: "omit",
    cache: "force-cache",
    referrerPolicy: "no-referrer",
  });
  if (!response.ok) {
    throw new Error(`Failed to download the face-landmark model (${response.status}).`);
  }
  const modelBuffer = await response.arrayBuffer();
  const digestBuffer = await globalThis.crypto.subtle.digest("SHA-256", modelBuffer);
  const observedDigest = bytesToHex(new Uint8Array(digestBuffer));
  if (observedDigest !== FACE_LANDMARKER_MODEL_SHA256) {
    throw new Error(
      "Face-landmark model integrity verification failed. The remote model bytes do not match the reviewed SHA-256."
    );
  }
  return new Uint8Array(modelBuffer);
}

function finiteNumber(value, name) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    throw new Error(`${name} must be a finite number.`);
  }
  return numeric;
}

function validatePointArray(points, expectedLength, name) {
  if (!Array.isArray(points) || points.length !== expectedLength) {
    throw new Error(`${name} must contain exactly ${expectedLength} coordinate pairs.`);
  }
  return points.map((point, index) => {
    if (!Array.isArray(point) || point.length !== 2) {
      throw new Error(`${name}[${index}] must be an [x, y] pair.`);
    }
    return [
      finiteNumber(point[0], `${name}[${index}][0]`),
      finiteNumber(point[1], `${name}[${index}][1]`),
    ];
  });
}

export async function loadLocalFaceLandmarker() {
  if (!faceLandmarkerPromise) {
    faceLandmarkerPromise = (async () => {
      const [vision, modelAssetBuffer] = await Promise.all([
        import(MEDIAPIPE_MODULE_URL),
        fetchVerifiedFaceLandmarkerModel(),
      ]);
      const wasmFileset = await vision.FilesetResolver.forVisionTasks(
        MEDIAPIPE_WASM_ROOT
      );
      return vision.FaceLandmarker.createFromOptions(wasmFileset, {
        baseOptions: {
          modelAssetBuffer,
          delegate: "CPU",
        },
        runningMode: "IMAGE",
        numFaces: 1,
        minFaceDetectionConfidence: 0.55,
        minFacePresenceConfidence: 0.55,
        minTrackingConfidence: 0.55,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
      });
    })().catch((error) => {
      faceLandmarkerPromise = null;
      throw error;
    });
  }
  return faceLandmarkerPromise;
}

export function createOrientedImageCanvas(image, rotationDegrees = 0, maximumDimension = 1600) {
  if (!image || !Number.isFinite(image.naturalWidth) || !Number.isFinite(image.naturalHeight)
      || image.naturalWidth < 1 || image.naturalHeight < 1) {
    throw new Error("A decoded image is required before local landmark detection.");
  }

  const normalizedRotation = ((Math.round(Number(rotationDegrees) / 90) * 90) % 360 + 360) % 360;
  const swapsAxes = normalizedRotation === 90 || normalizedRotation === 270;
  const unscaledWidth = swapsAxes ? image.naturalHeight : image.naturalWidth;
  const unscaledHeight = swapsAxes ? image.naturalWidth : image.naturalHeight;
  const limit = Math.max(320, finiteNumber(maximumDimension, "maximumDimension"));
  const scale = Math.min(1, limit / Math.max(unscaledWidth, unscaledHeight));
  const outputWidth = Math.max(1, Math.round(unscaledWidth * scale));
  const outputHeight = Math.max(1, Math.round(unscaledHeight * scale));
  const sourceScale = Math.min(1, limit / Math.max(image.naturalWidth, image.naturalHeight));
  const sourceWidth = Math.max(1, Math.round(image.naturalWidth * sourceScale));
  const sourceHeight = Math.max(1, Math.round(image.naturalHeight * sourceScale));

  const canvas = document.createElement("canvas");
  canvas.width = outputWidth;
  canvas.height = outputHeight;
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) {
    throw new Error("This browser does not provide a 2D canvas context.");
  }

  context.fillStyle = "#050a12";
  context.fillRect(0, 0, outputWidth, outputHeight);
  context.save();
  context.translate(outputWidth / 2, outputHeight / 2);
  context.rotate(normalizedRotation * Math.PI / 180);
  context.drawImage(
    image,
    -sourceWidth / 2,
    -sourceHeight / 2,
    sourceWidth,
    sourceHeight
  );
  context.restore();
  return canvas;
}

export function denseMeshToDlib68(denseLandmarks, width, height) {
  if (!Array.isArray(denseLandmarks)) {
    throw new Error("The face landmarker did not return a dense landmark array.");
  }
  const maximumIndex = Math.max(...MEDIAPIPE_TO_DLIB_68);
  if (denseLandmarks.length <= maximumIndex) {
    throw new Error(
      `The detected mesh contains ${denseLandmarks.length} points; at least ${maximumIndex + 1} are required.`
    );
  }
  const pixelWidth = finiteNumber(width, "image width");
  const pixelHeight = finiteNumber(height, "image height");
  return MEDIAPIPE_TO_DLIB_68.map((meshIndex) => {
    const point = denseLandmarks[meshIndex];
    if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      throw new Error(`Dense landmark ${meshIndex} is missing or invalid.`);
    }
    return [point.x * pixelWidth, point.y * pixelHeight];
  });
}

export async function detectDlib68FromImage(faceLandmarker, imageSource) {
  if (!faceLandmarker || typeof faceLandmarker.detect !== "function") {
    throw new Error("The local face landmark model is unavailable.");
  }
  if (!imageSource || imageSource.width < 1 || imageSource.height < 1) {
    throw new Error("A prepared image canvas is required for detection.");
  }

  const result = faceLandmarker.detect(imageSource);
  const faces = result?.faceLandmarks;
  if (!Array.isArray(faces) || faces.length === 0) {
    throw new Error("No face mesh was detected. Try a clear, front-facing image with the full face visible.");
  }
  const landmarks = denseMeshToDlib68(faces[0], imageSource.width, imageSource.height);
  return {
    landmarks,
    denseLandmarkCount: faces[0].length,
    detectedFaceCount: faces.length,
  };
}

export function geometricDisplacementIndex(partialProcrustesDistance) {
  const distance = Math.max(0, finiteNumber(partialProcrustesDistance, "partial Procrustes distance"));
  return Math.min(10, (10 * distance) / Math.SQRT2);
}

function signedDoubleArea(a, b, c) {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function circumcircle(a, b, c) {
  const ax = a[0];
  const ay = a[1];
  const bx = b[0];
  const by = b[1];
  const cx = c[0];
  const cy = c[1];
  const divisor = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  if (Math.abs(divisor) < 1e-9) return null;

  const a2 = ax * ax + ay * ay;
  const b2 = bx * bx + by * by;
  const c2 = cx * cx + cy * cy;
  const x = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / divisor;
  const y = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / divisor;
  return { x, y, radiusSquared: (x - ax) ** 2 + (y - ay) ** 2 };
}

function orientedTriangle(aIndex, bIndex, cIndex, points) {
  return signedDoubleArea(points[aIndex], points[bIndex], points[cIndex]) >= 0
    ? [aIndex, bIndex, cIndex]
    : [aIndex, cIndex, bIndex];
}

export function delaunayTriangulate(inputPoints) {
  const points = inputPoints.map((point, index) => {
    if (!Array.isArray(point) || point.length !== 2) {
      throw new Error(`Triangulation point ${index} must be an [x, y] pair.`);
    }
    return [finiteNumber(point[0], `point ${index} x`), finiteNumber(point[1], `point ${index} y`)];
  });
  if (points.length < 3) return [];

  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const span = Math.max(maxX - minX, maxY - minY, 1);
  const midpointX = (minX + maxX) / 2;
  const midpointY = (minY + maxY) / 2;
  const originalCount = points.length;
  points.push(
    [midpointX - 20 * span, midpointY - span],
    [midpointX, midpointY + 20 * span],
    [midpointX + 20 * span, midpointY - span]
  );

  let triangles = [orientedTriangle(originalCount, originalCount + 1, originalCount + 2, points)];

  for (let pointIndex = 0; pointIndex < originalCount; pointIndex += 1) {
    const point = points[pointIndex];
    const badTriangleIndexes = [];
    const edgeCounts = new Map();

    triangles.forEach((triangle, triangleIndex) => {
      const circle = circumcircle(
        points[triangle[0]],
        points[triangle[1]],
        points[triangle[2]]
      );
      if (!circle) return;
      const distanceSquared = (point[0] - circle.x) ** 2 + (point[1] - circle.y) ** 2;
      const tolerance = Math.max(1e-8, circle.radiusSquared * 1e-10);
      if (distanceSquared <= circle.radiusSquared + tolerance) {
        badTriangleIndexes.push(triangleIndex);
        [[triangle[0], triangle[1]], [triangle[1], triangle[2]], [triangle[2], triangle[0]]]
          .forEach(([start, end]) => {
            const key = start < end ? `${start}:${end}` : `${end}:${start}`;
            const record = edgeCounts.get(key);
            if (record) record.count += 1;
            else edgeCounts.set(key, { count: 1, start, end });
          });
      }
    });

    const badSet = new Set(badTriangleIndexes);
    triangles = triangles.filter((_, index) => !badSet.has(index));
    edgeCounts.forEach((edge) => {
      if (edge.count === 1) {
        triangles.push(orientedTriangle(edge.start, edge.end, pointIndex, points));
      }
    });
  }

  return triangles.filter((triangle) =>
    triangle.every((index) => index < originalCount)
    && Math.abs(signedDoubleArea(
      points[triangle[0]],
      points[triangle[1]],
      points[triangle[2]]
    )) > 1e-7
  );
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function createBoundaryAnchors(points, width, height) {
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const faceLeft = Math.min(...xs);
  const faceRight = Math.max(...xs);
  const faceTop = Math.min(...ys);
  const faceBottom = Math.max(...ys);
  const faceWidth = Math.max(faceRight - faceLeft, 1);
  const faceHeight = Math.max(faceBottom - faceTop, 1);
  const left = clamp(faceLeft - faceWidth * 0.24, 0, width - 1);
  const right = clamp(faceRight + faceWidth * 0.24, 0, width - 1);
  const top = clamp(faceTop - faceHeight * 0.58, 0, height - 1);
  const bottom = clamp(faceBottom + faceHeight * 0.16, 0, height - 1);
  const middleX = (left + right) / 2;
  const middleY = (top + bottom) / 2;
  return [
    [left, top], [middleX, top], [right, top], [right, middleY],
    [right, bottom], [middleX, bottom], [left, bottom], [left, middleY],
  ];
}

function capDisplacements(points, displacements, scale) {
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const diagonal = Math.max(
    Math.hypot(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)),
    1
  );
  const maximumShift = diagonal * 0.16;
  const scaled = displacements.map((vector) => {
    const x = vector[0] * scale;
    const y = vector[1] * scale;
    const magnitude = Math.hypot(x, y);
    if (magnitude <= maximumShift || magnitude === 0) return [x, y];
    const factor = maximumShift / magnitude;
    return [x * factor, y * factor];
  });
  return { displacements: scaled, faceDiagonal: diagonal, maximumShift };
}

function deformationIsSafe(sourcePoints, destinationPoints, triangles) {
  return triangles.every(([aIndex, bIndex, cIndex]) => {
    const sourceArea = signedDoubleArea(
      sourcePoints[aIndex], sourcePoints[bIndex], sourcePoints[cIndex]
    );
    const destinationArea = signedDoubleArea(
      destinationPoints[aIndex], destinationPoints[bIndex], destinationPoints[cIndex]
    );
    if (Math.abs(sourceArea) < 1e-7) return true;
    const ratio = Math.abs(destinationArea / sourceArea);
    return sourceArea * destinationArea > 0 && ratio >= 0.08 && ratio <= 8;
  });
}

function safeDestinationPoints(sourcePoints, displacementVectors, triangles) {
  let factor = 1;
  while (factor >= 0.05) {
    const destination = sourcePoints.map((point, index) => [
      point[0] + displacementVectors[index][0] * factor,
      point[1] + displacementVectors[index][1] * factor,
    ]);
    if (deformationIsSafe(sourcePoints, destination, triangles)) {
      return { destination, factor };
    }
    factor *= 0.82;
  }
  return { destination: sourcePoints.map((point) => [...point]), factor: 0 };
}

function affineTransform(sourceTriangle, destinationTriangle) {
  const [[x0, y0], [x1, y1], [x2, y2]] = sourceTriangle;
  const [[u0, v0], [u1, v1], [u2, v2]] = destinationTriangle;
  const denominator = x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1);
  if (Math.abs(denominator) < 1e-9) return null;

  const solve = (q0, q1, q2) => ({
    x: (q0 * (y1 - y2) + q1 * (y2 - y0) + q2 * (y0 - y1)) / denominator,
    y: (q0 * (x2 - x1) + q1 * (x0 - x2) + q2 * (x1 - x0)) / denominator,
    offset: (
      q0 * (x1 * y2 - x2 * y1)
      + q1 * (x2 * y0 - x0 * y2)
      + q2 * (x0 * y1 - x1 * y0)
    ) / denominator,
  });
  const horizontal = solve(u0, u1, u2);
  const vertical = solve(v0, v1, v2);
  return {
    a: horizontal.x,
    b: vertical.x,
    c: horizontal.y,
    d: vertical.y,
    e: horizontal.offset,
    f: vertical.offset,
  };
}

function expandedTriangle(points, expansion = 0.7) {
  const center = points.reduce(
    (sum, point) => [sum[0] + point[0] / 3, sum[1] + point[1] / 3],
    [0, 0]
  );
  return points.map((point) => {
    const distance = Math.hypot(point[0] - center[0], point[1] - center[1]);
    if (distance < 1e-9) return [...point];
    return [
      point[0] + expansion * (point[0] - center[0]) / distance,
      point[1] + expansion * (point[1] - center[1]) / distance,
    ];
  });
}

function drawWarpedTexture(context, imageSource, sourcePoints, destinationPoints, triangles) {
  triangles.forEach((triangle) => {
    const sourceTriangle = triangle.map((index) => sourcePoints[index]);
    const destinationTriangle = triangle.map((index) => destinationPoints[index]);
    const transform = affineTransform(sourceTriangle, destinationTriangle);
    if (!transform) return;

    const clipTriangle = expandedTriangle(destinationTriangle);
    context.save();
    context.beginPath();
    context.moveTo(clipTriangle[0][0], clipTriangle[0][1]);
    context.lineTo(clipTriangle[1][0], clipTriangle[1][1]);
    context.lineTo(clipTriangle[2][0], clipTriangle[2][1]);
    context.closePath();
    context.clip();
    context.setTransform(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f);
    context.drawImage(imageSource, 0, 0);
    context.restore();
  });
}

function drawLandmarkOverlay(context, points, triangles) {
  context.save();
  context.strokeStyle = "rgba(85, 228, 243, 0.34)";
  context.fillStyle = "rgba(255, 209, 125, 0.9)";
  context.lineWidth = Math.max(0.8, context.canvas.width / 1200);
  triangles.forEach((triangle) => {
    context.beginPath();
    context.moveTo(points[triangle[0]][0], points[triangle[0]][1]);
    context.lineTo(points[triangle[1]][0], points[triangle[1]][1]);
    context.lineTo(points[triangle[2]][0], points[triangle[2]][1]);
    context.closePath();
    context.stroke();
  });
  points.slice(0, 68).forEach(([x, y]) => {
    context.beginPath();
    context.arc(x, y, Math.max(1.3, context.canvas.width / 900), 0, Math.PI * 2);
    context.fill();
  });
  context.restore();
}

function drawDisplacementVectorOverlay(context, sourcePoints, destinationPoints) {
  const source = sourcePoints.slice(0, 68);
  const destination = destinationPoints.slice(0, 68);
  const lineWidth = Math.max(1.25, context.canvas.width / 960);
  const outlineWidth = lineWidth + Math.max(1.4, context.canvas.width / 900);
  const endpointRadius = Math.max(1.8, context.canvas.width / 720);
  const visibleVectors = source.map((point, index) => {
    const target = destination[index];
    const dx = target[0] - point[0];
    const dy = target[1] - point[1];
    return { point, target, dx, dy, length: Math.hypot(dx, dy) };
  }).filter((vector) => vector.length >= 0.75);

  const drawVectorPath = ({ point, target, dx, dy, length }) => {
    const directionX = dx / length;
    const directionY = dy / length;
    const headLength = Math.min(
      Math.max(4.5, context.canvas.width / 210),
      Math.max(2.5, length * 0.72),
    );
    const headWidth = headLength * 0.55;
    const baseX = target[0] - directionX * headLength;
    const baseY = target[1] - directionY * headLength;
    const perpendicularX = -directionY;
    const perpendicularY = directionX;

    context.beginPath();
    context.moveTo(point[0], point[1]);
    context.lineTo(target[0], target[1]);
    context.stroke();

    context.beginPath();
    context.moveTo(target[0], target[1]);
    context.lineTo(baseX + perpendicularX * headWidth, baseY + perpendicularY * headWidth);
    context.lineTo(baseX - perpendicularX * headWidth, baseY - perpendicularY * headWidth);
    context.closePath();
    context.fill();
  };

  context.save();
  context.lineCap = "round";
  context.lineJoin = "round";

  context.strokeStyle = "rgba(2, 8, 14, 0.78)";
  context.fillStyle = "rgba(2, 8, 14, 0.84)";
  context.lineWidth = outlineWidth;
  visibleVectors.forEach(drawVectorPath);

  context.strokeStyle = "rgba(255, 194, 86, 0.96)";
  context.fillStyle = "rgba(255, 194, 86, 0.98)";
  context.lineWidth = lineWidth;
  visibleVectors.forEach(drawVectorPath);

  context.fillStyle = "rgba(85, 228, 243, 0.94)";
  visibleVectors.forEach(({ point }) => {
    context.beginPath();
    context.arc(point[0], point[1], endpointRadius, 0, Math.PI * 2);
    context.fill();
  });
  context.restore();

  return visibleVectors.length;
}

export function buildPhotoWarpGeometry({
  sourceLandmarks,
  residualVectors,
  centroidSize,
  scale = 1,
  width,
  height,
}) {
  const landmarks = validatePointArray(sourceLandmarks, 68, "sourceLandmarks");
  const unitResiduals = validatePointArray(residualVectors, 68, "residualVectors");
  const size = Math.max(0, finiteNumber(centroidSize, "centroidSize"));
  const visualizationScale = Math.max(0, finiteNumber(scale, "scale"));
  const imageWidth = Math.max(1, finiteNumber(width, "width"));
  const imageHeight = Math.max(1, finiteNumber(height, "height"));
  const pixelResiduals = unitResiduals.map((vector) => [
    vector[0] * size,
    vector[1] * size,
  ]);
  const capped = capDisplacements(landmarks, pixelResiduals, visualizationScale);
  const anchors = createBoundaryAnchors(landmarks, imageWidth, imageHeight);
  const sourcePoints = [...landmarks, ...anchors];
  const displacementVectors = [
    ...capped.displacements,
    ...anchors.map(() => [0, 0]),
  ];
  const triangles = delaunayTriangulate(sourcePoints);
  if (triangles.length === 0) {
    throw new Error("The detected landmark geometry could not be triangulated.");
  }
  const safe = safeDestinationPoints(sourcePoints, displacementVectors, triangles);
  return {
    landmarks,
    sourcePoints,
    destinationPoints: safe.destination,
    triangles,
    visualizationScale,
    effectiveScaleFactor: safe.factor,
    faceDiagonalPixels: capped.faceDiagonal,
    maximumLandmarkShiftPixels: capped.maximumShift,
  };
}

export function renderPhotoWarp({
  canvas,
  imageSource,
  sourceLandmarks,
  residualVectors,
  centroidSize,
  scale = 1,
  mode = "split",
  showMesh = false,
  showVectors = true,
}) {
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new Error("A destination HTML canvas is required for the photo warp.");
  }
  if (!imageSource || imageSource.width < 1 || imageSource.height < 1) {
    throw new Error("A prepared source image is required for the photo warp.");
  }
  const selectedMode = ["original", "split", "warped"].includes(mode) ? mode : "split";
  const geometry = buildPhotoWarpGeometry({
    sourceLandmarks,
    residualVectors,
    centroidSize,
    scale,
    width: imageSource.width,
    height: imageSource.height,
  });
  const {
    landmarks,
    sourcePoints,
    destinationPoints,
    triangles,
    visualizationScale,
    effectiveScaleFactor,
  } = geometry;

  canvas.width = imageSource.width;
  canvas.height = imageSource.height;
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) throw new Error("This browser does not provide a 2D canvas context.");

  const warpedCanvas = document.createElement("canvas");
  warpedCanvas.width = canvas.width;
  warpedCanvas.height = canvas.height;
  const warpedContext = warpedCanvas.getContext("2d", { alpha: false });
  warpedContext.drawImage(imageSource, 0, 0);
  drawWarpedTexture(warpedContext, imageSource, sourcePoints, destinationPoints, triangles);
  if (showMesh) drawLandmarkOverlay(warpedContext, destinationPoints, triangles);
  const renderedVectorCount = showVectors
    ? drawDisplacementVectorOverlay(warpedContext, landmarks, destinationPoints)
    : 0;

  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (selectedMode === "original") {
    context.drawImage(imageSource, 0, 0);
    if (showMesh) drawLandmarkOverlay(context, sourcePoints, triangles);
    if (showVectors) drawDisplacementVectorOverlay(context, landmarks, destinationPoints);
  } else if (selectedMode === "warped") {
    context.drawImage(warpedCanvas, 0, 0);
  } else {
    const divider = Math.round(canvas.width / 2);
    context.drawImage(imageSource, 0, 0);
    context.save();
    context.beginPath();
    context.rect(divider, 0, canvas.width - divider, canvas.height);
    context.clip();
    context.drawImage(warpedCanvas, 0, 0);
    context.restore();
    context.save();
    context.strokeStyle = "rgba(255, 255, 255, 0.86)";
    context.lineWidth = Math.max(2, canvas.width / 500);
    context.beginPath();
    context.moveTo(divider, 0);
    context.lineTo(divider, canvas.height);
    context.stroke();
    context.fillStyle = "rgba(3, 9, 16, 0.78)";
    context.fillRect(Math.max(0, divider - 52), 12, 104, 28);
    context.fillStyle = "#edf4fb";
    context.font = `${Math.max(11, Math.round(canvas.width / 90))}px system-ui, sans-serif`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText("original | warp", divider, 26);
    context.restore();
  }

  const appliedDisplacements = destinationPoints.slice(0, 68).map((point, index) => [
    point[0] - landmarks[index][0],
    point[1] - landmarks[index][1],
  ]);
  const rmsPixels = Math.sqrt(
    appliedDisplacements.reduce((sum, vector) => sum + vector[0] ** 2 + vector[1] ** 2, 0)
    / appliedDisplacements.length
  );

  return {
    triangleCount: triangles.length,
    effectiveScaleFactor,
    requestedVisualizationScale: visualizationScale,
    appliedVisualizationScale: visualizationScale * effectiveScaleFactor,
    faceDiagonalPixels: geometry.faceDiagonalPixels,
    maximumLandmarkShiftPixels: geometry.maximumLandmarkShiftPixels,
    rmsAppliedDisplacementPixels: rmsPixels,
    mode: selectedMode,
    showMesh: Boolean(showMesh),
    showVectors: Boolean(showVectors),
    renderedVectorCount,
  };
}
