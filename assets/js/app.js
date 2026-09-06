"use strict";

const PYODIDE_VERSION = "314.0.6";
const PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const ANALYTICS_PATH = "./core/analytics.py";
const EXPECTED_COORDINATE_COUNT = 136;
const MAX_INPUT_FILE_BYTES = 1024 * 1024;
const MAX_PHOTO_FILE_BYTES = 20 * 1024 * 1024;
const ACCEPTED_PHOTO_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const ACCEPTED_PHOTO_EXTENSIONS = new Set(["jpg", "jpeg", "png", "webp"]);

const LANDMARK_PATHS = [
  { start: 0, end: 16, closed: false },
  { start: 17, end: 21, closed: false },
  { start: 22, end: 26, closed: false },
  { start: 27, end: 30, closed: false },
  { start: 31, end: 35, closed: false },
  { start: 36, end: 41, closed: true },
  { start: 42, end: 47, closed: true },
  { start: 48, end: 59, closed: true },
  { start: 60, end: 67, closed: true },
];

const state = {
  pyodide: null,
  runtimeReady: false,
  runtimeLoading: false,
  currentLandmarks: null,
  currentSource: "Synthetic Sample A",
  currentDemo: "a",
  result: null,
  analysisToken: 0,
  analysisRunning: false,
  activeView: "photo",
  photoObjectUrl: null,
  photoZoom: 1,
  photoRotation: 0,
  photoPanX: 0,
  photoPanY: 0,
  photoPointerId: null,
  photoPointerStartX: 0,
  photoPointerStartY: 0,
  photoPanStartX: 0,
  photoPanStartY: 0,
  scopeReturnFocus: null,
};

const ui = {};

function getElement(id) {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Required interface element #${id} was not found.`);
  }
  return element;
}

function cacheInterface() {
  ui.viewTabs = Array.from(document.querySelectorAll("[data-view]"));
  ui.viewPanels = Array.from(document.querySelectorAll("[data-panel]"));
  ui.openLab = getElement("open-lab");
  ui.openScope = getElement("open-scope");
  ui.scopeModal = getElement("scope-modal");
  ui.closeScope = getElement("close-scope");
  ui.acknowledgeScope = getElement("acknowledge-scope");
  ui.runtimeModal = getElement("runtime-modal");
  ui.runtimeTitle = getElement("runtime-modal-title");
  ui.runtimeMessage = getElement("runtime-modal-message");
  ui.runtimeProgress = getElement("runtime-progress");
  ui.dismissRuntime = getElement("dismiss-runtime");
  ui.continueRuntime = getElement("continue-runtime");
  ui.retryRuntime = getElement("retry-runtime");
  ui.runtimeDot = getElement("runtime-dot");
  ui.runtimeLabel = getElement("runtime-label");
  ui.configurationState = getElement("configuration-state");
  ui.analysisState = getElement("analysis-state");
  ui.analyzeButton = getElement("analyze-button");
  ui.exportButton = getElement("export-button");
  ui.referencePopulation = getElement("reference-population");
  ui.vectorScale = getElement("vector-scale");
  ui.landmarkFile = getElement("landmark-file");
  ui.dropZone = getElement("drop-zone");
  ui.fileState = getElement("file-state");
  ui.sourceLabel = getElement("source-label");
  ui.canvas = getElement("shape-canvas");
  ui.canvasShell = getElement("canvas-shell");
  ui.canvasEmpty = getElement("canvas-empty");
  ui.liveRegion = getElement("live-region");
  ui.pcaBars = getElement("pca-bars");
  ui.tangentDimension = getElement("tangent-dimension");
  ui.referenceKey = getElement("reference-key");
  ui.metricPartial = getElement("metric-partial");
  ui.metricFull = getElement("metric-full");
  ui.metricMahalanobis = getElement("metric-mahalanobis");
  ui.metricCentroid = getElement("metric-centroid");
  ui.metricRms = getElement("metric-rms");
  ui.gpaConverged = getElement("gpa-converged");
  ui.gpaIterations = getElement("gpa-iterations");
  ui.referenceSize = getElement("reference-size");
  ui.covarianceCondition = getElement("covariance-condition");
  ui.environmentalStress = getElement("environmental-stress");
  ui.environmentalStressValue = getElement("environmental-stress-value");
  ui.pathogenPrevalence = getElement("pathogen-prevalence");
  ui.pathogenPrevalenceValue = getElement("pathogen-prevalence-value");
  ui.operationalSexRatio = getElement("operational-sex-ratio");
  ui.operationalSexRatioValue = getElement("operational-sex-ratio-value");
  ui.photoFile = getElement("photo-file");
  ui.photoStage = getElement("photo-stage");
  ui.photoEmpty = getElement("photo-empty");
  ui.photoViewer = getElement("photo-viewer");
  ui.photoPreview = getElement("photo-preview");
  ui.photoState = getElement("photo-state");
  ui.photoUploadLabel = getElement("photo-upload-label");
  ui.photoFit = getElement("photo-fit");
  ui.photoRotateLeft = getElement("photo-rotate-left");
  ui.photoRotateRight = getElement("photo-rotate-right");
  ui.photoZoom = getElement("photo-zoom");
  ui.photoZoomValue = getElement("photo-zoom-value");
  ui.photoRemove = getElement("photo-remove");
  ui.photoName = getElement("photo-name");
  ui.photoDimensions = getElement("photo-dimensions");
  ui.photoSize = getElement("photo-size");
  ui.photoProcessing = getElement("photo-processing");
  ui.demoButtons = Array.from(document.querySelectorAll("[data-demo]"));
  ui.runtimeSteps = Array.from(document.querySelectorAll("[data-runtime-step]"));
}

function announce(message) {
  ui.liveRegion.textContent = "";
  window.setTimeout(() => {
    ui.liveRegion.textContent = message;
  }, 20);
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function selectView(viewName, focusTab = false) {
  const selectedTab = ui.viewTabs.find((tab) => tab.dataset.view === viewName);
  const selectedPanel = ui.viewPanels.find((panel) => panel.dataset.panel === viewName);
  if (!selectedTab || !selectedPanel) {
    return;
  }

  state.activeView = viewName;
  ui.viewTabs.forEach((tab) => {
    const active = tab === selectedTab;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
    if (active) {
      tab.setAttribute("aria-current", "page");
    } else {
      tab.removeAttribute("aria-current");
    }
    tab.tabIndex = active ? 0 : -1;
  });
  ui.viewPanels.forEach((panel) => {
    const active = panel === selectedPanel;
    panel.hidden = !active;
    panel.classList.toggle("is-active", active);
  });

  if (focusTab) {
    selectedTab.focus();
  }

  const prefersReducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  window.scrollTo({ top: 0, behavior: prefersReducedMotion ? "auto" : "smooth" });

  if (viewName === "lab") {
    window.requestAnimationFrame(drawCurrentState);
  }
}

function handleTabKeydown(event) {
  const currentIndex = ui.viewTabs.indexOf(event.currentTarget);
  let nextIndex = null;
  if (event.key === "ArrowRight") nextIndex = (currentIndex + 1) % ui.viewTabs.length;
  if (event.key === "ArrowLeft") nextIndex = (currentIndex - 1 + ui.viewTabs.length) % ui.viewTabs.length;
  if (event.key === "Home") nextIndex = 0;
  if (event.key === "End") nextIndex = ui.viewTabs.length - 1;
  if (nextIndex === null) return;
  event.preventDefault();
  selectView(ui.viewTabs[nextIndex].dataset.view, true);
}

function setBodyModalState() {
  const modalOpen = ui.runtimeModal.classList.contains("is-visible")
    || ui.scopeModal.classList.contains("is-visible");
  document.body.style.overflow = modalOpen ? "hidden" : "";
}

function openScopeModal() {
  state.scopeReturnFocus = document.activeElement;
  ui.scopeModal.hidden = false;
  window.requestAnimationFrame(() => {
    ui.scopeModal.classList.add("is-visible");
    setBodyModalState();
    ui.closeScope.focus();
  });
}

function closeScopeModal() {
  ui.scopeModal.classList.remove("is-visible");
  setBodyModalState();
  window.setTimeout(() => {
    ui.scopeModal.hidden = true;
    if (state.scopeReturnFocus instanceof HTMLElement) {
      state.scopeReturnFocus.focus();
    }
  }, 190);
}

function formatFileSize(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(2)} MB`;
}

function photoExtension(fileName) {
  const pieces = String(fileName).toLowerCase().split(".");
  return pieces.length > 1 ? pieces.pop() : "";
}

function validatePhotoFile(file) {
  const recognizedType = ACCEPTED_PHOTO_TYPES.has(file.type);
  const recognizedExtension = ACCEPTED_PHOTO_EXTENSIONS.has(photoExtension(file.name));
  if (!recognizedType && !recognizedExtension) {
    throw new Error("Choose a JPG, PNG, or WebP image.");
  }
  if (file.size <= 0) {
    throw new Error("The selected image is empty.");
  }
  if (file.size > MAX_PHOTO_FILE_BYTES) {
    throw new Error("Images must be no larger than 20 MB.");
  }
}

function updatePhotoTransform() {
  ui.photoPreview.style.transform = [
    `translate(${state.photoPanX}px, ${state.photoPanY}px)`,
    `scale(${state.photoZoom})`,
    `rotate(${state.photoRotation}deg)`,
  ].join(" ");
  ui.photoZoom.value = String(state.photoZoom);
  ui.photoZoomValue.value = `${Math.round(state.photoZoom * 100)}%`;
  ui.photoZoomValue.textContent = ui.photoZoomValue.value;
}

function resetPhotoTransform() {
  state.photoZoom = 1;
  state.photoRotation = 0;
  state.photoPanX = 0;
  state.photoPanY = 0;
  updatePhotoTransform();
}

function setPhotoControlsEnabled(enabled) {
  ui.photoFit.disabled = !enabled;
  ui.photoRotateLeft.disabled = !enabled;
  ui.photoRotateRight.disabled = !enabled;
  ui.photoZoom.disabled = !enabled;
  ui.photoRemove.disabled = !enabled;
}

function waitForImage(image) {
  return new Promise((resolve, reject) => {
    image.addEventListener("load", resolve, { once: true });
    image.addEventListener("error", () => reject(new Error("The browser could not decode this image.")), { once: true });
  });
}

async function loadPhotoFile(file) {
  try {
    validatePhotoFile(file);
    ui.photoState.textContent = "Loading";
    const nextUrl = URL.createObjectURL(file);
    const previousUrl = state.photoObjectUrl;
    const loaded = waitForImage(ui.photoPreview);
    ui.photoPreview.src = nextUrl;

    try {
      await loaded;
    } catch (error) {
      URL.revokeObjectURL(nextUrl);
      if (previousUrl) ui.photoPreview.src = previousUrl;
      throw error;
    }

    if (previousUrl) URL.revokeObjectURL(previousUrl);
    state.photoObjectUrl = nextUrl;
    ui.photoPreview.alt = `Local preview of ${file.name}`;
    ui.photoEmpty.hidden = true;
    ui.photoViewer.hidden = false;
    ui.photoStage.classList.add("has-image");
    ui.photoState.textContent = "Ready";
    ui.photoState.style.color = "var(--mint)";
    ui.photoUploadLabel.textContent = "Replace image";
    ui.photoName.textContent = file.name;
    ui.photoName.title = file.name;
    ui.photoDimensions.textContent = `${ui.photoPreview.naturalWidth} × ${ui.photoPreview.naturalHeight}`;
    ui.photoSize.textContent = formatFileSize(file.size);
    ui.photoProcessing.textContent = "Local preview only";
    setPhotoControlsEnabled(true);
    resetPhotoTransform();
    announce(`${file.name} is displayed in the local photo preview.`);
  } catch (error) {
    ui.photoState.textContent = state.photoObjectUrl ? "Ready" : "Invalid image";
    ui.photoState.style.color = state.photoObjectUrl ? "var(--mint)" : "var(--danger)";
    announce(error instanceof Error ? error.message : String(error));
  } finally {
    ui.photoFile.value = "";
  }
}

function removePhoto() {
  if (state.photoObjectUrl) {
    URL.revokeObjectURL(state.photoObjectUrl);
  }
  state.photoObjectUrl = null;
  ui.photoPreview.removeAttribute("src");
  ui.photoPreview.alt = "Locally selected preview";
  ui.photoViewer.hidden = true;
  ui.photoEmpty.hidden = false;
  ui.photoStage.classList.remove("has-image", "is-panning", "is-dragging");
  ui.photoState.textContent = "No image";
  ui.photoState.style.color = "";
  ui.photoUploadLabel.textContent = "Choose image";
  ui.photoName.textContent = "—";
  ui.photoName.removeAttribute("title");
  ui.photoDimensions.textContent = "—";
  ui.photoSize.textContent = "—";
  ui.photoProcessing.textContent = "Local preview";
  setPhotoControlsEnabled(false);
  resetPhotoTransform();
  announce("The local photo preview was cleared.");
}

function adjustPhotoZoom(nextZoom) {
  state.photoZoom = Math.min(3, Math.max(0.5, Number(nextZoom)));
  updatePhotoTransform();
}

function beginPhotoPan(event) {
  if (!state.photoObjectUrl || event.button !== 0) return;
  state.photoPointerId = event.pointerId;
  state.photoPointerStartX = event.clientX;
  state.photoPointerStartY = event.clientY;
  state.photoPanStartX = state.photoPanX;
  state.photoPanStartY = state.photoPanY;
  ui.photoStage.classList.add("is-panning");
  ui.photoStage.setPointerCapture(event.pointerId);
}

function continuePhotoPan(event) {
  if (event.pointerId !== state.photoPointerId) return;
  state.photoPanX = state.photoPanStartX + event.clientX - state.photoPointerStartX;
  state.photoPanY = state.photoPanStartY + event.clientY - state.photoPointerStartY;
  updatePhotoTransform();
}

function endPhotoPan(event) {
  if (event.pointerId !== state.photoPointerId) return;
  if (ui.photoStage.hasPointerCapture(event.pointerId)) {
    ui.photoStage.releasePointerCapture(event.pointerId);
  }
  state.photoPointerId = null;
  ui.photoStage.classList.remove("is-panning");
}

function handlePhotoStageKeydown(event) {
  if (!state.photoObjectUrl) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      ui.photoFile.click();
    }
    return;
  }

  const panStep = event.shiftKey ? 40 : 12;
  const actions = {
    ArrowLeft: () => { state.photoPanX -= panStep; },
    ArrowRight: () => { state.photoPanX += panStep; },
    ArrowUp: () => { state.photoPanY -= panStep; },
    ArrowDown: () => { state.photoPanY += panStep; },
    "+": () => adjustPhotoZoom(state.photoZoom + 0.1),
    "=": () => adjustPhotoZoom(state.photoZoom + 0.1),
    "-": () => adjustPhotoZoom(state.photoZoom - 0.1),
    "0": resetPhotoTransform,
  };
  const action = actions[event.key];
  if (!action) return;
  event.preventDefault();
  action();
  updatePhotoTransform();
}

function setRuntimeStage(stage, message, progress) {
  const order = ["wasm", "packages", "engine"];
  const activeIndex = order.indexOf(stage);
  ui.runtimeSteps.forEach((step) => {
    const stepIndex = order.indexOf(step.dataset.runtimeStep);
    step.classList.toggle("is-complete", stepIndex < activeIndex);
    step.classList.toggle("is-active", stepIndex === activeIndex);
    step.classList.remove("is-error");
  });
  ui.runtimeMessage.textContent = message;
  ui.runtimeProgress.style.width = `${progress}%`;
}

function setRuntimeReady() {
  ui.runtimeSteps.forEach((step) => {
    step.classList.remove("is-active", "is-error");
    step.classList.add("is-complete");
  });
  ui.runtimeProgress.style.width = "100%";
  ui.runtimeTitle.textContent = "Local engine ready";
  ui.runtimeMessage.textContent = "The simulated demonstration is ready to analyze.";
  ui.runtimeDot.className = "status-dot status-dot--ready";
  ui.runtimeLabel.textContent = "Runtime ready";
}

function setRuntimeError(error) {
  ui.runtimeSteps.forEach((step) => step.classList.remove("is-active"));
  const unfinished = ui.runtimeSteps.find((step) => !step.classList.contains("is-complete"));
  if (unfinished) {
    unfinished.classList.add("is-error");
  }
  ui.runtimeTitle.textContent = "Runtime initialization failed";
  ui.runtimeMessage.textContent = error instanceof Error ? error.message : String(error);
  ui.runtimeProgress.style.width = "100%";
  ui.runtimeProgress.style.background = "var(--danger)";
  ui.retryRuntime.hidden = false;
  ui.runtimeDot.className = "status-dot status-dot--error";
  ui.runtimeLabel.textContent = "Runtime error";
  showRuntimeModal();
  announce("The local analysis runtime could not be initialized.");
}

function showRuntimeModal() {
  ui.runtimeModal.removeAttribute("aria-hidden");
  ui.runtimeModal.classList.add("is-visible");
  setBodyModalState();
}

function hideRuntimeModal() {
  ui.runtimeModal.classList.remove("is-visible");
  ui.runtimeModal.setAttribute("aria-hidden", "true");
  setBodyModalState();
}

async function initializeRuntime() {
  if (state.runtimeLoading) {
    return;
  }

  state.runtimeLoading = true;
  state.runtimeReady = false;
  ui.retryRuntime.hidden = true;
  ui.runtimeProgress.style.background = "";
  ui.runtimeTitle.textContent = "Starting analysis runtime";
  ui.runtimeDot.className = "status-dot status-dot--loading";
  ui.runtimeLabel.textContent = "Runtime loading";
  showRuntimeModal();

  try {
    setRuntimeStage("wasm", "Instantiating the WebAssembly sandbox.", 12);
    if (typeof window.loadPyodide !== "function") {
      throw new Error("The pinned Pyodide runtime did not load. Check the network connection and retry.");
    }

    state.pyodide = await window.loadPyodide({ indexURL: PYODIDE_INDEX_URL });

    setRuntimeStage("packages", "Loading NumPy and SciPy into browser memory.", 42);
    await state.pyodide.loadPackage(["numpy", "scipy"]);

    setRuntimeStage("engine", "Loading the descriptive morphometric engine.", 76);
    const response = await fetch(ANALYTICS_PATH, { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`Unable to load ${ANALYTICS_PATH} (${response.status}).`);
    }
    const analyticsSource = await response.text();
    await state.pyodide.runPythonAsync(analyticsSource);

    if (!state.pyodide.globals.has("run_pipeline_from_js")) {
      throw new Error("The Python bridge function was not defined by the analytics engine.");
    }

    state.runtimeReady = true;
    setRuntimeReady();
    await loadSimulatedDemo("a", false);
    await runAnalysis();
    await delay(260);
    hideRuntimeModal();
    announce("Local morphometric runtime ready. Synthetic Sample A analyzed.");
  } catch (error) {
    state.runtimeReady = false;
    setRuntimeError(error);
  } finally {
    state.runtimeLoading = false;
  }
}

function deletePythonGlobal(name) {
  if (state.pyodide && state.pyodide.globals.has(name)) {
    state.pyodide.globals.delete(name);
  }
}

async function loadSimulatedDemo(demoKey, runAfterLoad = true) {
  if (!state.runtimeReady || !state.pyodide) {
    return;
  }
  if (state.analysisRunning) {
    announce("Wait for the current analysis to finish before changing the configuration.");
    return;
  }

  ui.configurationState.textContent = "Loading";
  state.pyodide.globals.set("__demo_key", demoKey);
  try {
    const response = await state.pyodide.runPythonAsync(
      "get_simulated_demo_json(__demo_key)"
    );
    const payload = JSON.parse(String(response));
    const flattened = payload.landmarks.flat();
    validateCoordinateArray(flattened);

    state.currentLandmarks = new Float64Array(flattened);
    state.currentDemo = demoKey;
    state.currentSource = demoKey === "blend"
      ? "Synthetic Blend"
      : `Synthetic Sample ${demoKey.toUpperCase()}`;
    state.result = null;

    ui.demoButtons.forEach((button) => {
      button.classList.toggle("is-active", button.dataset.demo === demoKey);
    });
    ui.landmarkFile.value = "";
    ui.fileState.textContent = `Using built-in ${state.currentSource}.`;
    ui.sourceLabel.textContent = state.currentSource;
    ui.configurationState.textContent = "Loaded";
    ui.analyzeButton.disabled = false;
    ui.exportButton.disabled = true;
    resetResultDisplay();
    drawCurrentState();

    if (runAfterLoad) {
      await runAnalysis();
    }
  } finally {
    deletePythonGlobal("__demo_key");
  }
}

function validateCoordinateArray(values) {
  if (!Array.isArray(values) && !(values instanceof Float64Array)) {
    throw new Error("The landmark payload must be an array.");
  }
  if (values.length !== EXPECTED_COORDINATE_COUNT) {
    throw new Error(
      `Expected ${EXPECTED_COORDINATE_COUNT} numeric values for 68 × 2 landmarks; received ${values.length}.`
    );
  }
  for (const value of values) {
    if (!Number.isFinite(Number(value))) {
      throw new Error("Every landmark coordinate must be a finite number.");
    }
  }
}

function flattenJsonLandmarks(payload) {
  let candidate = payload;
  if (candidate && typeof candidate === "object" && !Array.isArray(candidate)) {
    if (!("landmarks" in candidate)) {
      throw new Error("A JSON object must contain a 'landmarks' array.");
    }
    candidate = candidate.landmarks;
  }
  if (!Array.isArray(candidate)) {
    throw new Error("JSON landmark data must be an array or an object with a landmarks array.");
  }
  return candidate.flat(Infinity).map(Number);
}

function parseDelimitedCoordinates(text) {
  const lines = text
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"));

  const coordinates = [];
  lines.forEach((line, lineIndex) => {
    const tokens = line.split(/[\s,;\t]+/u).filter(Boolean);
    const numericTokens = tokens.map(Number);
    if (numericTokens.some((value) => !Number.isFinite(value))) {
      if (lineIndex === 0) {
        return;
      }
      throw new Error(`Non-numeric coordinate found on data line ${lineIndex + 1}.`);
    }
    coordinates.push(...numericTokens);
  });
  return coordinates;
}

async function parseLandmarkFile(file) {
  if (file.size > MAX_INPUT_FILE_BYTES) {
    throw new Error("Coordinate files must be no larger than 1 MB.");
  }
  const text = await file.text();
  if (!text.trim()) {
    throw new Error("The selected coordinate file is empty.");
  }

  let values;
  const lowerName = file.name.toLowerCase();
  if (lowerName.endsWith(".json") || /^[\s]*[\[{]/u.test(text)) {
    values = flattenJsonLandmarks(JSON.parse(text));
  } else {
    values = parseDelimitedCoordinates(text);
  }

  validateCoordinateArray(values);
  return new Float64Array(values.map(Number));
}

async function handleLandmarkFile(file) {
  if (state.analysisRunning) {
    announce("Wait for the current analysis to finish before loading another configuration.");
    return;
  }

  try {
    ui.configurationState.textContent = "Reading";
    const coordinates = await parseLandmarkFile(file);
    state.currentLandmarks = coordinates;
    state.currentSource = file.name;
    state.currentDemo = null;
    state.result = null;

    ui.demoButtons.forEach((button) => button.classList.remove("is-active"));
    ui.fileState.textContent = `${file.name} · ${coordinates.length / 2} landmarks loaded locally.`;
    ui.sourceLabel.textContent = file.name;
    ui.configurationState.textContent = "Loaded";
    ui.analyzeButton.disabled = !state.runtimeReady;
    ui.exportButton.disabled = true;
    resetResultDisplay();
    drawCurrentState();

    if (state.runtimeReady) {
      await runAnalysis();
    }
  } catch (error) {
    ui.configurationState.textContent = "Invalid";
    ui.fileState.textContent = error instanceof Error ? error.message : String(error);
    announce(ui.fileState.textContent);
  }
}

async function runAnalysis() {
  if (
    state.analysisRunning
    || !state.runtimeReady
    || !state.pyodide
    || !state.currentLandmarks
  ) {
    return;
  }

  state.analysisRunning = true;
  const token = ++state.analysisToken;
  ui.analysisState.textContent = "Computing";
  ui.configurationState.textContent = "Locked";
  ui.analyzeButton.disabled = true;
  ui.exportButton.disabled = true;
  ui.referencePopulation.disabled = true;
  ui.landmarkFile.disabled = true;
  ui.demoButtons.forEach((button) => {
    button.disabled = true;
  });

  const typedCoordinates = new Float64Array(state.currentLandmarks);
  state.pyodide.globals.set("__landmark_coordinates", typedCoordinates);
  state.pyodide.globals.set("__reference_population", ui.referencePopulation.value);

  try {
    const response = await state.pyodide.runPythonAsync(
      "run_pipeline_from_js(__landmark_coordinates, __reference_population)"
    );
    const result = JSON.parse(String(response));
    if (result.error) {
      throw new Error(result.error);
    }
    if (token !== state.analysisToken) {
      return;
    }

    state.result = result;
    renderResult(result);
    drawCurrentState();
    ui.analysisState.textContent = "Complete";
    ui.configurationState.textContent = "Loaded";
    ui.exportButton.disabled = false;
    announce("Descriptive shape-space analysis complete.");
  } catch (error) {
    if (token === state.analysisToken) {
      ui.analysisState.textContent = "Error";
      ui.configurationState.textContent = "Loaded";
      announce(error instanceof Error ? error.message : String(error));
    }
  } finally {
    deletePythonGlobal("__landmark_coordinates");
    deletePythonGlobal("__reference_population");
    if (token === state.analysisToken) {
      ui.analyzeButton.disabled = false;
      ui.referencePopulation.disabled = false;
      ui.landmarkFile.disabled = false;
      ui.demoButtons.forEach((button) => {
        button.disabled = false;
      });
    }
    state.analysisRunning = false;
  }
}

function formatMetric(value, digits = 5) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "—";
  }
  const absolute = Math.abs(numeric);
  if ((absolute > 0 && absolute < 0.0001) || absolute >= 100000) {
    return numeric.toExponential(3);
  }
  return numeric.toFixed(digits);
}

function formatReferenceKey(key) {
  if (key === "a") return "Reference A";
  if (key === "b") return "Reference B";
  return "Pooled A + B";
}

function resetResultDisplay() {
  ui.metricPartial.textContent = "—";
  ui.metricFull.textContent = "—";
  ui.metricMahalanobis.textContent = "—";
  ui.metricCentroid.textContent = "—";
  ui.metricRms.textContent = "—";
  ui.gpaConverged.textContent = "—";
  ui.gpaIterations.textContent = "—";
  ui.referenceSize.textContent = "—";
  ui.covarianceCondition.textContent = "—";
  ui.analysisState.textContent = "Not run";
  ui.pcaBars.replaceChildren();
  const message = document.createElement("p");
  message.className = "empty-copy";
  message.textContent = "Principal-component scores will appear here.";
  ui.pcaBars.append(message);
}

function renderResult(result) {
  ui.metricPartial.textContent = formatMetric(result.distances.partial_procrustes);
  ui.metricFull.textContent = formatMetric(result.distances.full_procrustes);
  ui.metricMahalanobis.textContent = formatMetric(
    result.distances.regularized_mahalanobis,
    3
  );
  ui.metricCentroid.textContent = formatMetric(result.input_geometry.centroid_size, 3);
  ui.metricRms.textContent = formatMetric(
    result.residual_shape_difference.root_mean_square_magnitude
  );
  ui.gpaConverged.textContent = result.gpa.converged ? "Yes" : "No";
  ui.gpaIterations.textContent = String(result.gpa.iterations);
  ui.referenceSize.textContent = String(result.reference.sample_size);
  ui.covarianceCondition.textContent = formatMetric(
    result.covariance_diagnostics.condition_number,
    2
  );
  ui.tangentDimension.textContent = `${result.tangent_space.dimension} dimensions`;
  ui.referenceKey.textContent = formatReferenceKey(result.reference.key);
  renderPcaScores(
    result.tangent_space.pca_scores,
    result.tangent_space.pca_explained_variance_ratio
  );
}

function renderPcaScores(scores, explainedRatios) {
  ui.pcaBars.replaceChildren();
  const visibleScores = scores.slice(0, 6).map(Number);
  const maximum = Math.max(...visibleScores.map((value) => Math.abs(value)), 1e-12);

  visibleScores.forEach((score, index) => {
    const row = document.createElement("div");
    row.className = "pca-row";

    const label = document.createElement("span");
    label.textContent = `PC${index + 1}`;

    const track = document.createElement("div");
    track.className = "pca-track";
    const bar = document.createElement("span");
    const width = Math.min(48, Math.abs(score) / maximum * 48);
    bar.style.width = `${width}%`;
    bar.style.left = score < 0 ? `${50 - width}%` : "50%";
    bar.style.right = "auto";
    bar.style.transform = "none";
    bar.style.background = score < 0 ? "var(--violet)" : "var(--cyan)";
    track.append(bar);

    const value = document.createElement("span");
    const variancePercent = Number(explainedRatios[index] || 0) * 100;
    value.textContent = `${score >= 0 ? "+" : ""}${formatMetric(score, 3)} · ${variancePercent.toFixed(1)}%`;

    row.append(label, track, value);
    ui.pcaBars.append(row);
  });
}

function coordinatesToPairs(flatCoordinates) {
  const pairs = [];
  for (let index = 0; index < flatCoordinates.length; index += 2) {
    pairs.push([Number(flatCoordinates[index]), Number(flatCoordinates[index + 1])]);
  }
  return pairs;
}

function normalizeForPreview(points) {
  const centroid = points.reduce(
    (sum, point) => [sum[0] + point[0], sum[1] + point[1]],
    [0, 0]
  ).map((value) => value / points.length);
  const centered = points.map((point) => [point[0] - centroid[0], point[1] - centroid[1]]);
  const norm = Math.sqrt(
    centered.reduce((sum, point) => sum + point[0] ** 2 + point[1] ** 2, 0)
  );
  if (!Number.isFinite(norm) || norm <= Number.EPSILON) {
    return points;
  }
  return centered.map((point) => [point[0] / norm, point[1] / norm]);
}

function resizeCanvas() {
  const rect = ui.canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const targetWidth = Math.max(1, Math.round(rect.width * ratio));
  const targetHeight = Math.max(1, Math.round(rect.height * ratio));
  if (ui.canvas.width !== targetWidth || ui.canvas.height !== targetHeight) {
    ui.canvas.width = targetWidth;
    ui.canvas.height = targetHeight;
  }
  const context = ui.canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width: rect.width, height: rect.height };
}

function drawBackgroundAxes(context, width, height) {
  context.save();
  context.strokeStyle = "rgba(103, 215, 247, 0.12)";
  context.lineWidth = 1;
  context.setLineDash([4, 6]);
  context.beginPath();
  context.moveTo(width / 2, 24);
  context.lineTo(width / 2, height - 24);
  context.moveTo(24, height / 2);
  context.lineTo(width - 24, height / 2);
  context.stroke();
  context.restore();
}

function createScreenTransform(pointCollections, width, height) {
  const points = pointCollections.flat();
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const rangeX = Math.max(maxX - minX, 1e-6);
  const rangeY = Math.max(maxY - minY, 1e-6);
  const margin = Math.max(38, Math.min(width, height) * 0.09);
  const scale = Math.min(
    (width - margin * 2) / rangeX,
    (height - margin * 2) / rangeY
  );
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  return (point) => [
    width / 2 + (point[0] - centerX) * scale,
    height / 2 + (point[1] - centerY) * scale,
  ];
}

function drawLandmarkConfiguration(context, points, transform, strokeStyle, pointStyle, lineWidth) {
  context.save();
  context.strokeStyle = strokeStyle;
  context.fillStyle = pointStyle;
  context.lineWidth = lineWidth;
  context.lineJoin = "round";
  context.lineCap = "round";

  LANDMARK_PATHS.forEach((path) => {
    context.beginPath();
    for (let index = path.start; index <= path.end; index += 1) {
      const [x, y] = transform(points[index]);
      if (index === path.start) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    if (path.closed) context.closePath();
    context.stroke();
  });

  points.forEach((point) => {
    const [x, y] = transform(point);
    context.beginPath();
    context.arc(x, y, 1.8, 0, Math.PI * 2);
    context.fill();
  });
  context.restore();
}

function drawArrow(context, start, end, transform) {
  const [startX, startY] = transform(start);
  const [endX, endY] = transform(end);
  const deltaX = endX - startX;
  const deltaY = endY - startY;
  const length = Math.hypot(deltaX, deltaY);
  if (length < 0.8) return;

  context.beginPath();
  context.moveTo(startX, startY);
  context.lineTo(endX, endY);
  context.stroke();

  if (length >= 3.5) {
    const angle = Math.atan2(deltaY, deltaX);
    const headLength = Math.min(5, Math.max(2.5, length * 0.32));
    context.beginPath();
    context.moveTo(endX, endY);
    context.lineTo(
      endX - headLength * Math.cos(angle - Math.PI / 6),
      endY - headLength * Math.sin(angle - Math.PI / 6)
    );
    context.lineTo(
      endX - headLength * Math.cos(angle + Math.PI / 6),
      endY - headLength * Math.sin(angle + Math.PI / 6)
    );
    context.closePath();
    context.fill();
  }
}

function drawCurrentState() {
  const { context, width, height } = resizeCanvas();
  if (width < 2 || height < 2) {
    return;
  }
  context.clearRect(0, 0, width, height);
  drawBackgroundAxes(context, width, height);

  if (!state.currentLandmarks) {
    ui.canvasEmpty.hidden = false;
    return;
  }
  ui.canvasEmpty.hidden = true;

  if (!state.result) {
    const preview = normalizeForPreview(coordinatesToPairs(state.currentLandmarks));
    const transform = createScreenTransform([preview], width, height);
    drawLandmarkConfiguration(
      context,
      preview,
      transform,
      "rgba(237, 244, 251, 0.88)",
      "#edf4fb",
      1.2
    );
    return;
  }

  const input = state.result.aligned_configuration;
  const reference = state.result.reference_consensus;
  const residuals = state.result.residual_shape_difference.aligned_vectors;
  const vectorScale = Number(ui.vectorScale.value);
  const vectorEndpoints = input.map((point, index) => [
    point[0] + residuals[index][0] * vectorScale,
    point[1] + residuals[index][1] * vectorScale,
  ]);
  const transform = createScreenTransform([input, reference, vectorEndpoints], width, height);

  drawLandmarkConfiguration(
    context,
    reference,
    transform,
    "rgba(103, 215, 247, 0.72)",
    "#67d7f7",
    1.05
  );

  context.save();
  context.strokeStyle = "rgba(245, 184, 107, 0.48)";
  context.fillStyle = "rgba(245, 184, 107, 0.78)";
  context.lineWidth = 0.9;
  input.forEach((point, index) => {
    drawArrow(context, point, vectorEndpoints[index], transform);
  });
  context.restore();

  drawLandmarkConfiguration(
    context,
    input,
    transform,
    "rgba(237, 244, 251, 0.9)",
    "#edf4fb",
    1.35
  );

  context.save();
  context.fillStyle = "rgba(147, 165, 184, 0.86)";
  context.font = "12px SFMono-Regular, Consolas, monospace";
  context.fillText(`Residual vectors ×${vectorScale}`, 18, 25);
  context.restore();
}

function getStudyContextMetadata() {
  return {
    analytic_role: "annotation_only",
    environmental_stress_index: Number(ui.environmentalStress.value),
    pathogen_prevalence_index: Number(ui.pathogenPrevalence.value),
    operational_sex_ratio: Number(ui.operationalSexRatio.value),
  };
}

function exportAnalysis() {
  if (!state.result) return;
  const payload = {
    export_schema_version: "1.0",
    application: "bio-social-aesthetic-manifold",
    exported_at_utc: new Date().toISOString(),
    source_label: state.currentSource,
    study_context_metadata: getStudyContextMetadata(),
    analysis: state.result,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "shape-manifold-analysis.json";
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  announce("Analysis JSON exported.");
}

function updateRangeOutput(input, output) {
  output.value = Number(input.value).toFixed(2);
  output.textContent = output.value;
}

function bindInterfaceEvents() {
  ui.viewTabs.forEach((tab) => {
    tab.addEventListener("click", () => selectView(tab.dataset.view));
    tab.addEventListener("keydown", handleTabKeydown);
  });
  ui.openLab.addEventListener("click", () => selectView("lab", true));

  ui.openScope.addEventListener("click", openScopeModal);
  ui.closeScope.addEventListener("click", closeScopeModal);
  ui.acknowledgeScope.addEventListener("click", closeScopeModal);
  ui.scopeModal.addEventListener("click", (event) => {
    if (event.target === ui.scopeModal) closeScopeModal();
  });

  ui.dismissRuntime.addEventListener("click", hideRuntimeModal);
  ui.continueRuntime.addEventListener("click", hideRuntimeModal);

  ui.photoFile.addEventListener("change", () => {
    const [file] = ui.photoFile.files;
    if (file) loadPhotoFile(file);
  });
  ui.photoFit.addEventListener("click", resetPhotoTransform);
  ui.photoRemove.addEventListener("click", removePhoto);
  ui.photoRotateLeft.addEventListener("click", () => {
    state.photoRotation -= 90;
    updatePhotoTransform();
  });
  ui.photoRotateRight.addEventListener("click", () => {
    state.photoRotation += 90;
    updatePhotoTransform();
  });
  ui.photoZoom.addEventListener("input", () => adjustPhotoZoom(ui.photoZoom.value));
  ui.photoStage.addEventListener("wheel", (event) => {
    if (!state.photoObjectUrl) return;
    event.preventDefault();
    adjustPhotoZoom(state.photoZoom + (event.deltaY < 0 ? 0.1 : -0.1));
  }, { passive: false });
  ui.photoStage.addEventListener("pointerdown", beginPhotoPan);
  ui.photoStage.addEventListener("pointermove", continuePhotoPan);
  ui.photoStage.addEventListener("pointerup", endPhotoPan);
  ui.photoStage.addEventListener("pointercancel", endPhotoPan);
  ui.photoStage.addEventListener("keydown", handlePhotoStageKeydown);

  ["dragenter", "dragover"].forEach((eventName) => {
    ui.photoStage.addEventListener(eventName, (event) => {
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
      ui.photoStage.classList.add("is-dragging");
    });
  });
  ui.photoStage.addEventListener("dragleave", (event) => {
    event.preventDefault();
    if (event.relatedTarget && ui.photoStage.contains(event.relatedTarget)) return;
    ui.photoStage.classList.remove("is-dragging");
  });
  ui.photoStage.addEventListener("drop", (event) => {
    event.preventDefault();
    ui.photoStage.classList.remove("is-dragging");
    const [file] = event.dataTransfer.files;
    if (file) loadPhotoFile(file);
  });

  ui.demoButtons.forEach((button) => {
    button.addEventListener("click", () => loadSimulatedDemo(button.dataset.demo));
  });

  ui.analyzeButton.addEventListener("click", runAnalysis);
  ui.exportButton.addEventListener("click", exportAnalysis);
  ui.referencePopulation.addEventListener("change", () => {
    if (state.runtimeReady && state.currentLandmarks) runAnalysis();
  });
  ui.vectorScale.addEventListener("change", drawCurrentState);
  ui.retryRuntime.addEventListener("click", initializeRuntime);

  ui.landmarkFile.addEventListener("change", () => {
    const [file] = ui.landmarkFile.files;
    if (file) handleLandmarkFile(file);
  });

  ui.dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      ui.landmarkFile.click();
    }
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    ui.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      ui.dropZone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    ui.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      ui.dropZone.classList.remove("is-dragging");
    });
  });
  ui.dropZone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    if (file) handleLandmarkFile(file);
  });

  const ranges = [
    [ui.environmentalStress, ui.environmentalStressValue],
    [ui.pathogenPrevalence, ui.pathogenPrevalenceValue],
    [ui.operationalSexRatio, ui.operationalSexRatioValue],
  ];
  ranges.forEach(([input, output]) => {
    updateRangeOutput(input, output);
    input.addEventListener("input", () => updateRangeOutput(input, output));
  });

  let resizeFrame = null;
  const requestCanvasRedraw = () => {
    if (resizeFrame !== null) window.cancelAnimationFrame(resizeFrame);
    resizeFrame = window.requestAnimationFrame(() => {
      resizeFrame = null;
      drawCurrentState();
    });
  };
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(requestCanvasRedraw);
    observer.observe(ui.canvasShell);
  } else {
    window.addEventListener("resize", requestCanvasRedraw);
  }

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (ui.scopeModal.classList.contains("is-visible")) {
      closeScopeModal();
    } else if (ui.runtimeModal.classList.contains("is-visible")) {
      hideRuntimeModal();
    }
  });

  window.addEventListener("pagehide", () => {
    if (state.photoObjectUrl) URL.revokeObjectURL(state.photoObjectUrl);
  });
}

async function startApplication() {
  cacheInterface();
  bindInterfaceEvents();
  resetPhotoTransform();
  resetResultDisplay();
  await initializeRuntime();
}

window.addEventListener("DOMContentLoaded", () => {
  startApplication().catch((error) => {
    if (ui.runtimeModal) setRuntimeError(error);
  });
});
