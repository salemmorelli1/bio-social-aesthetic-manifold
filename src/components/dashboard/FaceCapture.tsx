"use client";

import { useCallback, useEffect, useRef, useState, type ChangeEvent } from "react";
import { Camera, ImageUp, ShieldCheck, Square } from "lucide-react";
import {
  createLocalFaceLandmarker,
  drawMeshOverlay,
  scaledDimensions,
  singleFaceMesh,
  validatePhotoFile,
  type LocalLandmarker,
} from "@/lib/vision/landmarker";

function messageFor(error: unknown): string {
  return error instanceof Error ? error.message : "The local capture could not finish.";
}

export default function FaceCapture() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameRef = useRef<number | null>(null);
  const sessionRef = useRef(0);
  const landmarkerRef = useRef<Promise<LocalLandmarker> | null>(null);
  const [busy, setBusy] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [faceCount, setFaceCount] = useState<number | null>(null);
  const [status, setStatus] = useState("Choose an image or start the camera. Nothing runs until you choose.");

  const releaseCamera = useCallback(() => {
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
    for (const track of streamRef.current?.getTracks() ?? []) track.stop();
    streamRef.current = null;
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
    }
  }, []);

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      // Resizing discards the displayed pixels, including the last webcam frame.
      canvas.width = 0;
      canvas.height = 0;
    }
  }, []);

  const stopCamera = useCallback(() => {
    sessionRef.current += 1;
    releaseCamera();
    clearCanvas();
    setCameraActive(false);
    setFaceCount(null);
    setStatus("Camera off. Media tracks stopped and the displayed frame cleared.");
  }, [clearCanvas, releaseCamera]);

  const getLandmarker = useCallback(() => {
    if (!landmarkerRef.current) {
      landmarkerRef.current = createLocalFaceLandmarker().catch((error: unknown) => {
        landmarkerRef.current = null; // a transient network failure can be retried
        throw error;
      });
    }
    return landmarkerRef.current;
  }, []);

  useEffect(() => () => {
    sessionRef.current += 1;
    releaseCamera();
    clearCanvas();
    void landmarkerRef.current?.then(({ task }) => task.close()).catch(() => {});
  }, [clearCanvas, releaseCamera]);

  useEffect(() => {
    const stopWhenHidden = () => {
      if (document.visibilityState === "hidden" && streamRef.current) stopCamera();
    };
    document.addEventListener("visibilitychange", stopWhenHidden);
    return () => document.removeEventListener("visibilitychange", stopWhenHidden);
  }, [stopCamera]);

  async function onUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (!file) return;
    const session = ++sessionRef.current;
    releaseCamera();
    clearCanvas();
    setCameraActive(false);
    setFaceCount(null);
    setBusy(true);
    setStatus("Checking the model and processing the image locally…");
    let bitmap: ImageBitmap | null = null;
    try {
      validatePhotoFile(file);
      bitmap = await createImageBitmap(file);
      const size = scaledDimensions(bitmap.width, bitmap.height);
      const source = document.createElement("canvas");
      source.width = size.width;
      source.height = size.height;
      const context = source.getContext("2d", { alpha: false });
      if (!context) throw new Error("A 2D canvas context is unavailable.");
      context.drawImage(bitmap, 0, 0, size.width, size.height);
      bitmap.close();
      bitmap = null;

      const handle = await getLandmarker();
      if (session !== sessionRef.current) return;
      await handle.task.setOptions({ runningMode: "IMAGE" });
      if (session !== sessionRef.current) return;
      const mesh = singleFaceMesh(handle.task.detect(source));
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = size.width;
      canvas.height = size.height;
      drawMeshOverlay(canvas, source, mesh, handle.connections);
      setFaceCount(mesh ? 1 : 0);
      setStatus(mesh
        ? "478 local mesh landmarks mapped. This is geometry, not an appearance rating."
        : "No face found. Try a clear, front-facing image with the whole face visible.");
    } catch (error) {
      if (session === sessionRef.current) setStatus(messageFor(error));
    } finally {
      bitmap?.close();
      if (session === sessionRef.current) setBusy(false);
    }
  }

  async function startCamera() {
    if (busy || cameraActive) return;
    const session = ++sessionRef.current;
    setFaceCount(null);
    setBusy(true);
    setStatus("Requesting camera permission and verifying the local model…");
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera capture requires HTTPS or localhost and a compatible browser.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      if (session !== sessionRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      stream.getVideoTracks().forEach((track) =>
        track.addEventListener("ended", stopCamera, { once: true }));
      const video = videoRef.current;
      if (!video) throw new Error("The camera view is unavailable.");
      video.srcObject = stream;
      await video.play();
      const handle = await getLandmarker();
      if (session !== sessionRef.current) return;
      await handle.task.setOptions({ runningMode: "VIDEO" });
      if (session !== sessionRef.current) return;
      setCameraActive(true);
      setStatus("Live mesh runs locally. Stop camera to end the stream.");

      let lastFrame = -Infinity;
      const nextFrame = (now: number) => {
        if (session !== sessionRef.current) return;
        try {
          if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && now - lastFrame >= 100) {
            lastFrame = now;
            const size = scaledDimensions(video.videoWidth, video.videoHeight);
            const canvas = canvasRef.current;
            if (!canvas) return;
            if (canvas.width !== size.width || canvas.height !== size.height) {
              canvas.width = size.width;
              canvas.height = size.height;
            }
            const mesh = singleFaceMesh(handle.task.detectForVideo(video, now));
            drawMeshOverlay(canvas, video, mesh, handle.connections);
            setFaceCount(mesh ? 1 : 0);
          }
          frameRef.current = requestAnimationFrame(nextFrame);
        } catch (error) {
          stopCamera();
          setStatus(messageFor(error));
        }
      };
      frameRef.current = requestAnimationFrame(nextFrame);
    } catch (error) {
      if (session === sessionRef.current) {
        releaseCamera();
        setStatus(messageFor(error));
      }
    } finally {
      if (session === sessionRef.current) setBusy(false);
    }
  }

  return (
    <section className="capture-panel" aria-labelledby="capture-title">
      <div className="capture-heading">
        <div>
          <p className="eyebrow">Step 02 / local landmark capture</p>
          <h2 id="capture-title">See the mesh, not a verdict.</h2>
          <p className="copy">Upload a photo or use your camera with permission. The 478-point model runs in this browser; the overlay is an observation, not a diagnostic or beauty score.</p>
        </div>
        <span className="privacy-chip"><ShieldCheck size={16} aria-hidden="true" /> Local processing</span>
      </div>

      <div className="capture-stage">
        <canvas ref={canvasRef} className="capture-canvas" aria-label="Photo or live camera with a face-mesh overlay" />
        <div className="stage-caption"><span className="signal" /> {faceCount === null ? "Awaiting input" : faceCount === 1 ? "One 478-point mesh" : "No face detected"}</div>
      </div>
      <video ref={videoRef} className="sr-only" muted playsInline aria-hidden="true" />

      <div className="capture-actions">
        <label className={`action action-primary ${busy || cameraActive ? "action-disabled" : ""}`}>
          <ImageUp size={18} aria-hidden="true" /> Upload image
          <input type="file" accept="image/jpeg,image/png,image/webp" onChange={onUpload}
            disabled={busy || cameraActive} aria-label="Upload a JPEG, PNG, or WebP facial photo" />
        </label>
        {cameraActive ? (
          <button className="action action-secondary" type="button" onClick={stopCamera}>
            <Square size={16} aria-hidden="true" /> Stop camera
          </button>
        ) : (
          <button className="action action-secondary" type="button" onClick={startCamera} disabled={busy}>
            <Camera size={18} aria-hidden="true" /> Start camera
          </button>
        )}
      </div>
      <p className="capture-status" role="status" aria-live="polite">{status}</p>
      <p className="capture-footnote">No account, server upload, demographic estimate, or report export is used in this capture stage. The model and WASM files are fetched from pinned third-party paths; model bytes are checked against a reviewed SHA-256.</p>
    </section>
  );
}
