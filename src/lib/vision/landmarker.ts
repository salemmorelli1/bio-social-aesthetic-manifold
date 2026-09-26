import type { FaceLandmarker } from "@mediapipe/tasks-vision";

export const MESH_LANDMARK_COUNT = 478;
export const MEDIAPIPE_VERSION = "1.0.1";
export const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
export const MODEL_SHA256 =
  "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff";
export const MODEL_BYTES = 3_758_596;
const WASM_ROOT =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;

export type MeshPoint = Readonly<{ x: number; y: number; z: number }>;
type MeshConnection = Readonly<{ start: number; end: number }>;
export type LocalLandmarker = Readonly<{
  task: FaceLandmarker;
  connections: readonly MeshConnection[];
}>;

export function validatePhotoFile(file: Pick<File, "name" | "type" | "size">): void {
  const allowedTypes = new Set(["image/jpeg", "image/png", "image/webp"]);
  const allowedExtensions = /\.(jpe?g|png|webp)$/i;
  if (!allowedTypes.has(file.type) || !allowedExtensions.test(file.name)) {
    throw new Error("Choose a JPEG, PNG, or WebP image.");
  }
  if (!Number.isFinite(file.size) || file.size <= 0 || file.size > 20 * 1024 * 1024) {
    throw new Error("The image must be nonempty and no larger than 20 MiB.");
  }
}

export function scaledDimensions(width: number, height: number, maximum = 1600): {
  width: number;
  height: number;
} {
  if (!Number.isSafeInteger(maximum) || maximum < 1 ||
      !Number.isSafeInteger(width) || !Number.isSafeInteger(height) ||
      width < 1 || height < 1 || width * height > 36_000_000) {
    throw new Error("The decoded image dimensions are unsupported (maximum 36 megapixels).");
  }
  const factor = Math.min(1, maximum / Math.max(width, height));
  return {
    width: Math.max(1, Math.round(width * factor)),
    height: Math.max(1, Math.round(height * factor)),
  };
}

export function singleFaceMesh(result: { faceLandmarks: readonly (readonly MeshPoint[])[] }):
  readonly MeshPoint[] | null {
  if (result.faceLandmarks.length === 0) return null;
  if (result.faceLandmarks.length !== 1) {
    throw new Error("This capture stage supports exactly one visible face.");
  }
  const mesh = result.faceLandmarks[0];
  if (mesh.length !== MESH_LANDMARK_COUNT ||
      mesh.some((point) => !point || ![point.x, point.y, point.z].every(Number.isFinite))) {
    throw new Error("The face mesh did not contain 478 finite landmarks.");
  }
  return mesh;
}

export async function verifyModelBuffer(
  bytes: ArrayBuffer,
  expectedSha256 = MODEL_SHA256,
  cryptoProvider: Pick<Crypto, "subtle"> | undefined = globalThis.crypto,
): Promise<void> {
  if (!cryptoProvider?.subtle) {
    throw new Error("Model verification requires HTTPS or localhost (Web Crypto).");
  }
  const digest = await cryptoProvider.subtle.digest("SHA-256", bytes);
  const observed = Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")).join("");
  if (observed !== expectedSha256) {
    throw new Error("Face-landmark model integrity failed; the upstream model may have changed. No photo was analyzed.");
  }
}

async function verifiedModelBytes(): Promise<Uint8Array<ArrayBuffer>> {
  const response = await fetch(MODEL_URL, {
    mode: "cors",
    credentials: "omit",
    cache: "force-cache",
    referrerPolicy: "no-referrer",
  });
  if (!response.ok) throw new Error(`Face-landmark model download failed (${response.status}).`);
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength !== MODEL_BYTES) {
    throw new Error("Face-landmark model length changed. No photo was analyzed.");
  }
  await verifyModelBuffer(bytes);
  return new Uint8Array(bytes);
}

// Create only in an explicitly requested browser capture session. Call close()
// on the returned task when the component unmounts. No image pixels are sent
// to a Next.js route, Python, analytics service, or report payload.
export async function createLocalFaceLandmarker(): Promise<LocalLandmarker> {
  const [vision, modelAssetBuffer] = await Promise.all([
    import("@mediapipe/tasks-vision"),
    verifiedModelBytes(),
  ]);
  const fileset = await vision.FilesetResolver.forVisionTasks(WASM_ROOT);
  const task = await vision.FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetBuffer, delegate: "CPU" },
    runningMode: "IMAGE",
    // Ask for two so a multi-face frame is rejected rather than silently
    // selecting whichever face the detector happened to return first.
    numFaces: 2,
    minFaceDetectionConfidence: 0.55,
    minFacePresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });
  return { task, connections: vision.FaceLandmarker.FACE_LANDMARKS_TESSELATION };
}

export function drawMeshOverlay(
  canvas: HTMLCanvasElement,
  source: CanvasImageSource,
  mesh: readonly MeshPoint[] | null,
  connections: readonly MeshConnection[],
): void {
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) throw new Error("A 2D canvas context is unavailable.");
  context.drawImage(source, 0, 0, canvas.width, canvas.height);
  if (!mesh) return;

  context.lineWidth = Math.max(0.55, canvas.width / 1600);
  context.strokeStyle = "rgba(112, 222, 213, 0.34)";
  context.beginPath();
  for (const { start, end } of connections) {
    if (start >= mesh.length || end >= mesh.length) continue;
    context.moveTo(mesh[start].x * canvas.width, mesh[start].y * canvas.height);
    context.lineTo(mesh[end].x * canvas.width, mesh[end].y * canvas.height);
  }
  context.stroke();

  context.fillStyle = "rgba(153, 251, 241, 0.76)";
  context.beginPath();
  const radius = Math.max(0.65, canvas.width / 1350);
  for (const point of mesh) {
    context.moveTo(point.x * canvas.width + radius, point.y * canvas.height);
    context.arc(point.x * canvas.width, point.y * canvas.height, radius, 0, Math.PI * 2);
  }
  context.fill();

  // These three points are an overlay guide, not an anatomical sagittal fit.
  context.strokeStyle = "#f4bb79";
  context.lineWidth = Math.max(1.5, canvas.width / 550);
  context.beginPath();
  for (const [index, meshIndex] of [10, 2, 152].entries()) {
    const point = mesh[meshIndex];
    if (index === 0) context.moveTo(point.x * canvas.width, point.y * canvas.height);
    else context.lineTo(point.x * canvas.width, point.y * canvas.height);
  }
  context.stroke();
}
