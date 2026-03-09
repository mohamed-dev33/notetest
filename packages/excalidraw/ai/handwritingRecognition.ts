/**
 * Handwriting Recognition Module
 *
 * Engine priority (best → worst):
 * 1. WICG Handwriting Recognition API (native browser, Chrome 99+)
 *    — highest accuracy, multi-language, zero model download
 * 2. TrOCR (transformer-based via Web Worker)
 * 3. Tesseract.js (WASM OCR, always available fallback)
 *
 * All engines are non-blocking — the main thread stays responsive.
 *
 * Key optimizations for accuracy:
 * - Pressure simulation (speed-based stroke width variation)
 * - High-resolution rendering (120px height for Tesseract, 384px for TrOCR)
 * - Multiple PSM modes tried in parallel for best coverage
 * - Smart post-processing to clean OCR artifacts
 * - No restrictive char whitelist (hurts more than helps)
 */

import type { LocalPoint } from "@excalidraw/math";

export interface HandwritingResult {
  text: string;
  confidence: number; // 0-100
}

// =====================================================================
// WICG Handwriting Recognition API (Chrome 99+)
// https://wicg.github.io/handwriting-recognition/
// =====================================================================

/** Declare WICG Handwriting Recognition types (experimental API) */
interface HandwritingPoint {
  x: number;
  y: number;
  t?: number;
}

interface HandwritingStroke {
  addPoint(point: HandwritingPoint): void;
}

interface HandwritingPrediction {
  text: string;
  segmentationResult?: unknown[];
}

interface HandwritingDrawing {
  addStroke(stroke: HandwritingStroke): void;
  getPrediction(): Promise<HandwritingPrediction[]>;
  clear(): void;
}

interface HandwritingRecognizerConstraint {
  languages: string[];
}

interface HandwritingRecognizer {
  startDrawing(hints?: Record<string, unknown>): HandwritingDrawing;
  finish(): void;
}

interface NavigatorHWR extends Navigator {
  createHandwritingRecognizer?: (
    constraint: HandwritingRecognizerConstraint,
  ) => Promise<HandwritingRecognizer>;
  queryHandwritingRecognizer?: (
    constraint: HandwritingRecognizerConstraint,
  ) => Promise<{ languages?: string[] } | null>;
}

let nativeHWRAvailable: boolean | null = null;
let nativeRecognizer: HandwritingRecognizer | null = null;

let nativeArabicHWRAvailable: boolean | null = null;
let nativeArabicRecognizer: HandwritingRecognizer | null = null;

async function checkNativeHWR(): Promise<boolean> {
  if (nativeHWRAvailable !== null) { return nativeHWRAvailable; }
  try {
    const nav = navigator as NavigatorHWR;
    if (!nav.createHandwritingRecognizer) {
      nativeHWRAvailable = false;
      return false;
    }
    const result = await nav.queryHandwritingRecognizer?.({ languages: ["en"] });
    nativeHWRAvailable = !!result;
    return nativeHWRAvailable;
  } catch {
    nativeHWRAvailable = false;
    return false;
  }
}

async function ensureNativeRecognizer(): Promise<HandwritingRecognizer | null> {
  if (nativeRecognizer) { return nativeRecognizer; }
  const available = await checkNativeHWR();
  if (!available) { return null; }
  try {
    const nav = navigator as NavigatorHWR;
    nativeRecognizer = await nav.createHandwritingRecognizer!({
      languages: ["en"],
    });
    console.log("[AI] Native Handwriting Recognition API initialized");
    return nativeRecognizer;
  } catch (e) {
    console.warn("[AI] Failed to initialize native HWR:", e);
    nativeHWRAvailable = false;
    return null;
  }
}

// --- Arabic Native HWR ---

async function checkNativeArabicHWR(): Promise<boolean> {
  if (nativeArabicHWRAvailable !== null) { return nativeArabicHWRAvailable; }
  try {
    const nav = navigator as NavigatorHWR;
    if (!nav.createHandwritingRecognizer) {
      nativeArabicHWRAvailable = false;
      return false;
    }
    const result = await nav.queryHandwritingRecognizer?.({ languages: ["ar"] });
    nativeArabicHWRAvailable = !!result;
    return nativeArabicHWRAvailable;
  } catch {
    nativeArabicHWRAvailable = false;
    return false;
  }
}

async function ensureNativeArabicRecognizer(): Promise<HandwritingRecognizer | null> {
  if (nativeArabicRecognizer) { return nativeArabicRecognizer; }
  const available = await checkNativeArabicHWR();
  if (!available) { return null; }
  try {
    const nav = navigator as NavigatorHWR;
    nativeArabicRecognizer = await nav.createHandwritingRecognizer!({
      languages: ["ar"],
    });
    console.log("[AI] Native Arabic Handwriting Recognition API initialized");
    return nativeArabicRecognizer;
  } catch (e) {
    console.warn("[AI] Failed to initialize native Arabic HWR:", e);
    nativeArabicHWRAvailable = false;
    return null;
  }
}

/**
 * Recognize Arabic using native HWR API.
 */
async function recognizeArabicWithNativeAPI(
  points: readonly LocalPoint[],
): Promise<HandwritingResult | null> {
  const recognizer = await ensureNativeArabicRecognizer();
  if (!recognizer) { return null; }

  try {
    const drawing = recognizer.startDrawing({
      recognitionType: "text",
      inputType: "mouse",
      textContext: "",
      alternatives: 3,
    });

    const stroke = {
      _points: [] as HandwritingPoint[],
      addPoint(p: HandwritingPoint) { this._points.push(p); },
      getPoints() { return this._points; },
    };

    const startTime = Date.now();
    for (let i = 0; i < points.length; i++) {
      stroke.addPoint({
        x: points[i][0],
        y: points[i][1],
        t: startTime + i * 10,
      });
    }

    drawing.addStroke(stroke as unknown as HandwritingStroke);

    const predictions = await drawing.getPrediction();
    drawing.clear();

    if (predictions.length > 0 && predictions[0].text.length > 0) {
      const text = cleanArabicOcrText(predictions[0].text);
      if (text.length > 0 && !isArabicGarbageText(text)) {
        console.log(`[AI] Native Arabic HWR: "${text}" (top prediction)`);
        return { text, confidence: 85 };
      }
    }
    return null;
  } catch (e) {
    console.warn("[AI] Native Arabic HWR failed:", e);
    return null;
  }
}

/**
 * Recognize Arabic multi-stroke with native API.
 */
async function recognizeArabicMultiStrokeNativeAPI(
  strokes: readonly { points: readonly LocalPoint[]; offsetX: number; offsetY: number }[],
): Promise<HandwritingResult | null> {
  const recognizer = await ensureNativeArabicRecognizer();
  if (!recognizer) { return null; }

  try {
    const drawing = recognizer.startDrawing({
      recognitionType: "text",
      inputType: "mouse",
      textContext: "",
      alternatives: 3,
    });

    const startTime = Date.now();
    let timeOffset = 0;

    for (const s of strokes) {
      const stroke = {
        _points: [] as HandwritingPoint[],
        addPoint(p: HandwritingPoint) { this._points.push(p); },
        getPoints() { return this._points; },
      };

      for (let i = 0; i < s.points.length; i++) {
        stroke.addPoint({
          x: s.points[i][0] + s.offsetX,
          y: s.points[i][1] + s.offsetY,
          t: startTime + timeOffset + i * 10,
        });
      }
      timeOffset += s.points.length * 10 + 50;

      drawing.addStroke(stroke as unknown as HandwritingStroke);
    }

    const predictions = await drawing.getPrediction();
    drawing.clear();

    if (predictions.length > 0 && predictions[0].text.length > 0) {
      const text = cleanArabicOcrText(predictions[0].text);
      if (text.length > 0 && !isArabicGarbageText(text)) {
        console.log(`[AI] Native Arabic HWR multi-stroke: "${text}"`);
        return { text, confidence: 85 };
      }
    }
    return null;
  } catch (e) {
    console.warn("[AI] Native Arabic HWR multi-stroke failed:", e);
    return null;
  }
}

/**
 * Returns high-confidence result or null if unavailable/failed.
 */
async function recognizeWithNativeAPI(
  points: readonly LocalPoint[],
): Promise<HandwritingResult | null> {
  const recognizer = await ensureNativeRecognizer();
  if (!recognizer) { return null; }

  try {
    const drawing = recognizer.startDrawing({
      recognitionType: "text",
      inputType: "mouse",
      textContext: "",
      alternatives: 3,
    });

    // Convert excalidraw points to HWR stroke
    const stroke = {
      _points: [] as HandwritingPoint[],
      addPoint(p: HandwritingPoint) { this._points.push(p); },
      getPoints() { return this._points; },
    };

    const startTime = Date.now();
    for (let i = 0; i < points.length; i++) {
      stroke.addPoint({
        x: points[i][0],
        y: points[i][1],
        t: startTime + i * 10,
      });
    }

    drawing.addStroke(stroke as unknown as HandwritingStroke);

    const predictions = await drawing.getPrediction();
    drawing.clear();

    if (predictions.length > 0 && predictions[0].text.length > 0) {
      const text = cleanOcrText(predictions[0].text);
      if (text.length > 0 && !isGarbageText(text)) {
        console.log(`[AI] Native HWR: "${text}" (top prediction)`);
        return { text, confidence: 85 };
      }
    }
    return null;
  } catch (e) {
    console.warn("[AI] Native HWR recognition failed:", e);
    return null;
  }
}

/**
 * Recognize multiple strokes using native API.
 */
async function recognizeMultiStrokeNativeAPI(
  strokes: readonly { points: readonly LocalPoint[]; offsetX: number; offsetY: number }[],
): Promise<HandwritingResult | null> {
  const recognizer = await ensureNativeRecognizer();
  if (!recognizer) { return null; }

  try {
    const drawing = recognizer.startDrawing({
      recognitionType: "text",
      inputType: "mouse",
      textContext: "",
      alternatives: 3,
    });

    const startTime = Date.now();
    let timeOffset = 0;

    for (const s of strokes) {
      const stroke = {
        _points: [] as HandwritingPoint[],
        addPoint(p: HandwritingPoint) { this._points.push(p); },
        getPoints() { return this._points; },
      };

      for (let i = 0; i < s.points.length; i++) {
        stroke.addPoint({
          x: s.points[i][0] + s.offsetX,
          y: s.points[i][1] + s.offsetY,
          t: startTime + timeOffset + i * 10,
        });
      }
      timeOffset += s.points.length * 10 + 50;

      drawing.addStroke(stroke as unknown as HandwritingStroke);
    }

    const predictions = await drawing.getPrediction();
    drawing.clear();

    if (predictions.length > 0 && predictions[0].text.length > 0) {
      const text = cleanOcrText(predictions[0].text);
      if (text.length > 0 && !isGarbageText(text)) {
        console.log(`[AI] Native HWR multi-stroke: "${text}"`);
        return { text, confidence: 85 };
      }
    }
    return null;
  } catch (e) {
    console.warn("[AI] Native HWR multi-stroke failed:", e);
    return null;
  }
}

// =====================================================================
// Canvas Preprocessing — separate rendering for each engine
// =====================================================================

/**
 * Apply blur to soften rendered strokes — makes mouse-drawn strokes
 * look more like scanned handwriting.
 */
function applyBlur(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  radius: number,
): void {
  if (radius <= 0) { return; }
  // Use OffscreenCanvas pattern: draw to temp, then draw back with blur
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = w;
  tempCanvas.height = h;
  const tempCtx = tempCanvas.getContext("2d")!;
  tempCtx.drawImage(ctx.canvas, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, w, h);
  ctx.filter = `blur(${radius}px)`;
  ctx.drawImage(tempCanvas, 0, 0);
  ctx.filter = "none";
}

/**
 * Compute stroke speed at each point to simulate pen pressure.
 * Returns an array of widths (one per point).
 * Fast strokes → thin lines, slow strokes → thick lines.
 */
function computePressureWidths(
  points: readonly LocalPoint[],
  baseWidth: number,
  minFactor: number = 0.5,
  maxFactor: number = 1.5,
): number[] {
  if (points.length < 2) { return [baseWidth]; }

  // Compute distances between consecutive points
  const speeds: number[] = [0];
  for (let i = 1; i < points.length; i++) {
    const dx = points[i][0] - points[i - 1][0];
    const dy = points[i][1] - points[i - 1][1];
    speeds.push(Math.sqrt(dx * dx + dy * dy));
  }

  // Smooth speeds with a window average
  const windowSize = Math.max(3, Math.floor(points.length / 10));
  const smoothed: number[] = [];
  for (let i = 0; i < speeds.length; i++) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - windowSize); j <= Math.min(speeds.length - 1, i + windowSize); j++) {
      sum += speeds[j];
      count++;
    }
    smoothed.push(sum / count);
  }

  // Normalize: fast → thin, slow → thick
  const maxSpeed = Math.max(...smoothed, 0.001);
  return smoothed.map((s) => {
    const normalized = s / maxSpeed; // 0 = slow, 1 = fast
    const factor = maxFactor - normalized * (maxFactor - minFactor);
    return baseWidth * factor;
  });
}

/**
 * Draw a stroke with pressure-variable width using overlapping circles.
 */
function drawPressureStroke(
  ctx: CanvasRenderingContext2D,
  points: readonly LocalPoint[],
  widths: number[],
  offsetX: number,
  offsetY: number,
  scale: number,
  padX: number,
  padY: number,
): void {
  for (let i = 0; i < points.length; i++) {
    const x = (points[i][0] + offsetX) * scale + padX;
    const y = (points[i][1] + offsetY) * scale + padY;
    const r = (widths[i] ?? widths[widths.length - 1]) * scale * 0.5;

    ctx.beginPath();
    ctx.arc(x, y, Math.max(0.5, r), 0, Math.PI * 2);
    ctx.fill();

    // Also draw line segment to previous point for continuity
    if (i > 0) {
      const px = (points[i - 1][0] + offsetX) * scale + padX;
      const py = (points[i - 1][1] + offsetY) * scale + padY;
      const avgWidth = ((widths[i] ?? 1) + (widths[i - 1] ?? 1)) * 0.5 * scale;
      ctx.lineWidth = Math.max(1, avgWidth);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(x, y);
      ctx.stroke();
    }
  }
}

/**
 * Render points for TrOCR — natural-looking handwriting.
 * TrOCR was trained on IAM dataset: dark gray ink on light background,
 * anti-aliased, ~384px height images. NO binarization.
 */
function renderForTrOCR(
  points: readonly LocalPoint[],
  strokeWidth: number = 3,
): HTMLCanvasElement | null {
  if (points.length < 2) { return null; }

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of points) {
    minX = Math.min(minX, p[0]);
    minY = Math.min(minY, p[1]);
    maxX = Math.max(maxX, p[0]);
    maxY = Math.max(maxY, p[1]);
  }

  const rawW = maxX - minX;
  const rawH = maxY - minY;
  if (rawW < 3 && rawH < 3) { return null; }

  // TrOCR expects ~384px height with proper aspect ratio
  const targetH = 384;
  const padding = 40;
  const scale = Math.max(2.0, targetH / Math.max(rawH, 1));
  const canvasW = Math.ceil(rawW * scale + padding * 2);
  const canvasH = Math.ceil(rawH * scale + padding * 2);

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(canvasW, 384);
  canvas.height = Math.max(canvasH, 384);

  const ctx = canvas.getContext("2d")!;
  if (!ctx) { return null; }

  // Light gray background (like paper)
  ctx.fillStyle = "#F8F8F8";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Dark gray strokes (like ink pen) — anti-aliased naturally
  ctx.strokeStyle = "#1A1A1A";
  ctx.lineWidth = Math.max(3, strokeWidth * scale * 0.8);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    const x = (points[i][0] - minX) * scale + padding;
    const y = (points[i][1] - minY) * scale + padding;
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      const prev = points[i - 1];
      const prevX = (prev[0] - minX) * scale + padding;
      const prevY = (prev[1] - minY) * scale + padding;
      const midX = (prevX + x) / 2;
      const midY = (prevY + y) / 2;
      ctx.quadraticCurveTo(prevX, prevY, midX, midY);
    }
  }
  ctx.stroke();

  // NO binarization — TrOCR handles anti-aliased text better
  return canvas;
}

/**
 * Render points for Tesseract — high-contrast at optimal resolution.
 * Key insights for Tesseract accuracy:
 * - Tesseract works best at ~300 DPI equivalent (~120px text height)
 * - Stroke width should be proportional (~8-12% of text height)
 * - Pressure simulation makes mouse strokes look like pen strokes
 * - Slight blur softens jagged edges from mouse input
 * - NO binarization — Tesseract handles internally
 */
function renderForTesseract(
  points: readonly LocalPoint[],
  strokeWidth: number = 3,
): HTMLCanvasElement | null {
  if (points.length < 2) { return null; }

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of points) {
    minX = Math.min(minX, p[0]);
    minY = Math.min(minY, p[1]);
    maxX = Math.max(maxX, p[0]);
    maxY = Math.max(maxY, p[1]);
  }

  const rawW = maxX - minX;
  const rawH = maxY - minY;
  if (rawW < 3 && rawH < 3) { return null; }

  // Higher resolution: 120px text height (optimal for Tesseract at ~300 DPI)
  const targetH = 120;
  const padding = 24;
  const scale = Math.max(1.5, targetH / Math.max(rawH, 1));
  const canvasW = Math.ceil(rawW * scale + padding * 2);
  const canvasH = Math.ceil(rawH * scale + padding * 2);

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(canvasW, 100);
  canvas.height = Math.max(canvasH, 100);

  const ctx = canvas.getContext("2d")!;
  if (!ctx) { return null; }

  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Stroke width: ~10% of target height
  const baseStrokeW = Math.max(2, Math.min(14, targetH * 0.10));

  // Pressure-variable rendering for natural look
  const widths = computePressureWidths(points, baseStrokeW, 0.6, 1.4);

  ctx.fillStyle = "#000000";
  ctx.strokeStyle = "#000000";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  drawPressureStroke(ctx, points, widths, -minX, -minY, scale, padding, padding);

  // Gentle blur to soften jagged mouse strokes
  applyBlur(ctx, canvas.width, canvas.height, 0.6);

  return canvas;
}

/**
 * Multi-stroke rendering for TrOCR
 */
function renderMultiStrokeForTrOCR(
  strokes: readonly { points: readonly LocalPoint[]; offsetX: number; offsetY: number }[],
  strokeWidth: number = 3,
): HTMLCanvasElement | null {
  if (strokes.length === 0) { return null; }

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const stroke of strokes) {
    for (const p of stroke.points) {
      const gx = p[0] + stroke.offsetX;
      const gy = p[1] + stroke.offsetY;
      minX = Math.min(minX, gx);
      minY = Math.min(minY, gy);
      maxX = Math.max(maxX, gx);
      maxY = Math.max(maxY, gy);
    }
  }

  const rawW = maxX - minX;
  const rawH = maxY - minY;
  if (rawW < 3 && rawH < 3) { return null; }

  const targetH = 384;
  const padding = 40;
  const scale = Math.max(2.0, targetH / Math.max(rawH, 1));
  const canvasW = Math.ceil(rawW * scale + padding * 2);
  const canvasH = Math.ceil(rawH * scale + padding * 2);

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(canvasW, 384);
  canvas.height = Math.max(canvasH, 384);

  const ctx = canvas.getContext("2d")!;
  if (!ctx) { return null; }

  ctx.fillStyle = "#F8F8F8";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "#1A1A1A";
  ctx.lineWidth = Math.max(3, strokeWidth * scale * 0.8);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (const stroke of strokes) {
    if (stroke.points.length < 2) { continue; }
    ctx.beginPath();
    for (let i = 0; i < stroke.points.length; i++) {
      const x = (stroke.points[i][0] + stroke.offsetX - minX) * scale + padding;
      const y = (stroke.points[i][1] + stroke.offsetY - minY) * scale + padding;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        const prev = stroke.points[i - 1];
        const prevX = (prev[0] + stroke.offsetX - minX) * scale + padding;
        const prevY = (prev[1] + stroke.offsetY - minY) * scale + padding;
        const midX = (prevX + x) / 2;
        const midY = (prevY + y) / 2;
        ctx.quadraticCurveTo(prevX, prevY, midX, midY);
      }
    }
    ctx.stroke();
  }

  return canvas;
}

/**
 * Multi-stroke rendering for Tesseract — high resolution with pressure simulation.
 */
function renderMultiStrokeForTesseract(
  strokes: readonly { points: readonly LocalPoint[]; offsetX: number; offsetY: number }[],
  strokeWidth: number = 3,
): HTMLCanvasElement | null {
  if (strokes.length === 0) { return null; }

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const stroke of strokes) {
    for (const p of stroke.points) {
      const gx = p[0] + stroke.offsetX;
      const gy = p[1] + stroke.offsetY;
      minX = Math.min(minX, gx);
      minY = Math.min(minY, gy);
      maxX = Math.max(maxX, gx);
      maxY = Math.max(maxY, gy);
    }
  }

  const rawW = maxX - minX;
  const rawH = maxY - minY;
  if (rawW < 3 && rawH < 3) { return null; }

  const targetH = 120;
  const padding = 24;
  const scale = Math.max(1.5, targetH / Math.max(rawH, 1));
  const canvasW = Math.ceil(rawW * scale + padding * 2);
  const canvasH = Math.ceil(rawH * scale + padding * 2);

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(canvasW, 100);
  canvas.height = Math.max(canvasH, 100);

  const ctx = canvas.getContext("2d")!;
  if (!ctx) { return null; }

  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const baseStrokeW = Math.max(2, Math.min(14, targetH * 0.10));
  ctx.fillStyle = "#000000";
  ctx.strokeStyle = "#000000";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (const stroke of strokes) {
    if (stroke.points.length < 2) { continue; }
    const widths = computePressureWidths(stroke.points, baseStrokeW, 0.6, 1.4);
    drawPressureStroke(
      ctx, stroke.points, widths,
      stroke.offsetX - minX, stroke.offsetY - minY,
      scale, padding, padding,
    );
  }

  applyBlur(ctx, canvas.width, canvas.height, 0.6);

  return canvas;
}

// =====================================================================
// TrOCR Engine via Web Worker (Primary — non-blocking, accurate)
// =====================================================================

let trOcrWorker: Worker | null = null;
let trOcrReady = false;
let trOcrLoading = false;
let trOcrLoadPromise: Promise<boolean> | null = null;
let trOcrRequestId = 0;

function ensureTrOcrWorker(): Promise<boolean> {
  if (trOcrReady) { return Promise.resolve(true); }
  if (trOcrLoading && trOcrLoadPromise) { return trOcrLoadPromise; }

  trOcrLoading = true;
  trOcrLoadPromise = new Promise<boolean>((resolve) => {
    try {
      trOcrWorker = new Worker(
        new URL("./trOcrWorker.ts", import.meta.url),
        { type: "module" },
      );

      const timeout = setTimeout(() => {
        console.warn("[AI] TrOCR worker init timed out (120s)");
        trOcrLoading = false;
        resolve(false);
      }, 120000);

      trOcrWorker.onmessage = (e) => {
        if (e.data.type === "ready") {
          clearTimeout(timeout);
          trOcrReady = true;
          trOcrLoading = false;
          console.log("[AI] TrOCR Web Worker ready");
          resolve(true);
        } else if (e.data.type === "error" && !trOcrReady) {
          clearTimeout(timeout);
          console.warn("[AI] TrOCR worker init error:", e.data.message);
          trOcrLoading = false;
          resolve(false);
        }
      };

      trOcrWorker.onerror = (err) => {
        clearTimeout(timeout);
        console.warn("[AI] TrOCR worker error:", err);
        trOcrLoading = false;
        resolve(false);
      };

      trOcrWorker.postMessage({ type: "init" });
    } catch (err) {
      console.warn("[AI] TrOCR worker creation failed:", err);
      trOcrLoading = false;
      resolve(false);
    }
  });

  return trOcrLoadPromise;
}

function recognizeWithTrOcrWorker(
  canvas: HTMLCanvasElement,
): Promise<HandwritingResult> {
  return new Promise((resolve) => {
    if (!trOcrWorker || !trOcrReady) {
      resolve({ text: "", confidence: 0 });
      return;
    }

    const id = ++trOcrRequestId;
    const ctx = canvas.getContext("2d");
    if (!ctx) { resolve({ text: "", confidence: 0 }); return; }

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    const timeout = setTimeout(() => {
      resolve({ text: "", confidence: 0 });
    }, 30000);

    const handler = (e: MessageEvent) => {
      if (e.data.id !== id) { return; }
      clearTimeout(timeout);
      trOcrWorker?.removeEventListener("message", handler);

      if (e.data.type === "result") {
        resolve({ text: e.data.text || "", confidence: e.data.confidence || 0 });
      } else {
        resolve({ text: "", confidence: 0 });
      }
    };

    trOcrWorker.addEventListener("message", handler);
    trOcrWorker.postMessage({ type: "recognize", imageData, id });
  });
}

// =====================================================================
// Tesseract.js Engine (Fallback — also non-blocking via internal worker)
// =====================================================================

let tesseractWorker: any = null;
let tesseractLoading = false;
let tesseractLoadPromise: Promise<void> | null = null;

async function ensureTesseract(): Promise<any> {
  if (tesseractWorker) { return tesseractWorker; }
  if (tesseractLoading && tesseractLoadPromise) {
    await tesseractLoadPromise;
    return tesseractWorker;
  }

  tesseractLoading = true;
  tesseractLoadPromise = (async () => {
    try {
      const Tesseract = await import("tesseract.js");
      tesseractWorker = await Tesseract.createWorker("eng", 1, {
        logger: () => {},
      });
      // Use PSM 7 (single text line) with dictionary enabled.
      // NO char whitelist — it hurts recognition for handwriting because
      // Tesseract can't form letter combinations it doesn't see in the whitelist.
      await tesseractWorker.setParameters({
        tessedit_pageseg_mode: "7",
        preserve_interword_spaces: "1",
      });
      console.log("[AI] Tesseract worker ready (PSM 7, no whitelist)");
    } catch (e) {
      console.error("[AI] Tesseract initialization failed:", e);
      tesseractWorker = null;
    }
  })();

  await tesseractLoadPromise;
  tesseractLoading = false;
  return tesseractWorker;
}

/**
 * Recognize with Tesseract using multiple PSM modes for best coverage.
 * PSM 7: single text line (good for sentences)
 * PSM 8: single word (good for names, labels)
 * PSM 13: raw line (no dictionary, catches unusual words)
 */
async function recognizeWithTesseract(
  canvas: HTMLCanvasElement,
): Promise<HandwritingResult> {
  const noResult: HandwritingResult = { text: "", confidence: 0 };

  try {
    const worker = await ensureTesseract();
    if (!worker) { return noResult; }

    // Try PSM 7 (single line) first — best for general handwriting
    const { data: data7 } = await worker.recognize(canvas);
    const text7 = cleanOcrText(data7.text);
    const conf7 = data7.confidence;
    console.log(`[AI] Tesseract PSM7: "${text7}" conf=${conf7}`);

    // Try PSM 8 (single word) for short inputs
    let text8 = "";
    let conf8 = 0;
    try {
      await worker.setParameters({ tessedit_pageseg_mode: "8" });
      const { data: d8 } = await worker.recognize(canvas);
      text8 = cleanOcrText(d8.text);
      conf8 = d8.confidence;
      console.log(`[AI] Tesseract PSM8: "${text8}" conf=${conf8}`);
      // Reset to PSM 7
      await worker.setParameters({ tessedit_pageseg_mode: "7" });
    } catch { /* PSM 8 failed, continue with PSM 7 result */ }

    // Pick the best result that isn't garbage
    const candidates: HandwritingResult[] = [];

    if (text7.length > 0 && !isGarbageText(text7) && conf7 >= 15) {
      candidates.push({ text: text7, confidence: conf7 });
    }
    if (text8.length > 0 && !isGarbageText(text8) && conf8 >= 15) {
      candidates.push({ text: text8, confidence: conf8 });
    }

    if (candidates.length === 0) { return noResult; }

    // Prefer the result with higher confidence, but boost single-word
    // results that look more like real words
    candidates.sort((a, b) => {
      const scoreA = a.confidence + (isLikelyWord(a.text) ? 10 : 0);
      const scoreB = b.confidence + (isLikelyWord(b.text) ? 10 : 0);
      return scoreB - scoreA;
    });

    return candidates[0];
  } catch (e) {
    console.error("[AI] Tesseract recognition error:", e);
    return noResult;
  }
}

// =====================================================================
// Arabic Tesseract.js Engine
// =====================================================================

let tesseractArabicWorker: any = null;
let tesseractArabicLoading = false;
let tesseractArabicLoadPromise: Promise<void> | null = null;

async function ensureTesseractArabic(): Promise<any> {
  if (tesseractArabicWorker) { return tesseractArabicWorker; }
  if (tesseractArabicLoading && tesseractArabicLoadPromise) {
    await tesseractArabicLoadPromise;
    return tesseractArabicWorker;
  }

  tesseractArabicLoading = true;
  tesseractArabicLoadPromise = (async () => {
    try {
      const Tesseract = await import("tesseract.js");
      tesseractArabicWorker = await Tesseract.createWorker("ara", 1, {
        logger: () => {},
      });
      // PSM 6: assume uniform block of text — better for RTL Arabic
      await tesseractArabicWorker.setParameters({
        tessedit_pageseg_mode: "6",
        preserve_interword_spaces: "1",
      });
      console.log("[AI] Arabic Tesseract worker ready (PSM 6)");
    } catch (e) {
      console.error("[AI] Arabic Tesseract initialization failed:", e);
      tesseractArabicWorker = null;
    }
  })();

  await tesseractArabicLoadPromise;
  tesseractArabicLoading = false;
  return tesseractArabicWorker;
}

/**
 * Recognize Arabic text with Tesseract using multiple PSM modes.
 */
async function recognizeArabicWithTesseract(
  canvas: HTMLCanvasElement,
): Promise<HandwritingResult> {
  const noResult: HandwritingResult = { text: "", confidence: 0 };

  try {
    const worker = await ensureTesseractArabic();
    if (!worker) { return noResult; }

    // PSM 6: uniform block
    const { data: data6 } = await worker.recognize(canvas);
    const text6 = cleanArabicOcrText(data6.text);
    const conf6 = data6.confidence;
    console.log(`[AI] Arabic Tesseract PSM6: "${text6}" conf=${conf6}`);

    // PSM 7: single line
    let text7 = "";
    let conf7 = 0;
    try {
      await worker.setParameters({ tessedit_pageseg_mode: "7" });
      const { data: d7 } = await worker.recognize(canvas);
      text7 = cleanArabicOcrText(d7.text);
      conf7 = d7.confidence;
      console.log(`[AI] Arabic Tesseract PSM7: "${text7}" conf=${conf7}`);
      await worker.setParameters({ tessedit_pageseg_mode: "6" });
    } catch { /* fallback to PSM 6 result */ }

    // PSM 8: single word
    let text8 = "";
    let conf8 = 0;
    try {
      await worker.setParameters({ tessedit_pageseg_mode: "8" });
      const { data: d8 } = await worker.recognize(canvas);
      text8 = cleanArabicOcrText(d8.text);
      conf8 = d8.confidence;
      console.log(`[AI] Arabic Tesseract PSM8: "${text8}" conf=${conf8}`);
      await worker.setParameters({ tessedit_pageseg_mode: "6" });
    } catch { /* fallback */ }

    const candidates: HandwritingResult[] = [];

    if (text6.length > 0 && !isArabicGarbageText(text6) && conf6 >= 10) {
      candidates.push({ text: text6, confidence: conf6 });
    }
    if (text7.length > 0 && !isArabicGarbageText(text7) && conf7 >= 10) {
      candidates.push({ text: text7, confidence: conf7 });
    }
    if (text8.length > 0 && !isArabicGarbageText(text8) && conf8 >= 10) {
      candidates.push({ text: text8, confidence: conf8 });
    }

    if (candidates.length === 0) { return noResult; }

    // Prefer result with more Arabic characters
    candidates.sort((a, b) => {
      const arabicA = (a.text.match(/[\u0600-\u06FF]/g) || []).length;
      const arabicB = (b.text.match(/[\u0600-\u06FF]/g) || []).length;
      const scoreA = a.confidence + arabicA * 2;
      const scoreB = b.confidence + arabicB * 2;
      return scoreB - scoreA;
    });

    return candidates[0];
  } catch (e) {
    console.error("[AI] Arabic Tesseract recognition error:", e);
    return noResult;
  }
}

/** Clean common OCR artifacts from text */
function cleanOcrText(raw: string): string {
  return raw
    .trim()
    .replace(/\n+/g, " ")
    .replace(/[|]/g, "l")
    .replace(/[{}[\]]/g, "")
    .replace(/[_~^`]/g, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

/** Check if text looks like a real word (has vowels, reasonable structure) */
function isLikelyWord(text: string): boolean {
  const cleaned = text.replace(/[^a-zA-Z]/g, "");
  if (cleaned.length < 2) { return false; }
  // Has at least one vowel
  if (/[aeiouAEIOU]/.test(cleaned)) { return true; }
  // Common consonant-only words (hmm, etc.) — rare but valid
  return false;
}

/**
 * Detect garbage OCR output: repeated characters, digit strings,
 * non-printable characters, or text that's clearly not meaningful.
 */
function isGarbageText(text: string): boolean {
  if (text.length === 0) { return true; }

  const cleaned = text.replace(/\s/g, "");
  if (cleaned.length === 0) { return true; }

  // Single char is valid
  if (cleaned.length === 1) { return false; }

  // All same character
  if (new Set(cleaned).size <= 1 && cleaned.length > 1) { return true; }

  // 3+ consecutive same characters (was 4, tightened)
  if (/(.)\1{2,}/.test(cleaned)) { return true; }

  // More than 50% digits with length > 2 (unlikely handwritten text)
  const digits = (cleaned.match(/\d/g) || []).length;
  if (digits > cleaned.length * 0.5 && cleaned.length > 2) { return true; }

  // All punctuation / symbols — no actual letters or digits
  if (!/[a-zA-Z0-9]/.test(cleaned)) { return true; }

  // Mostly non-alphanumeric (> 70% symbols)
  const alphaNum = (cleaned.match(/[a-zA-Z0-9]/g) || []).length;
  if (alphaNum < cleaned.length * 0.3 && cleaned.length > 3) { return true; }

  // Very short text that's just consonants (likely misrecognized shape)
  if (cleaned.length <= 3) {
    const letters = cleaned.replace(/[^a-zA-Z]/g, "");
    if (letters.length >= 2 && !/[aeiouAEIOU]/.test(letters)) {
      // Exception: common consonant abbreviations
      const upper = letters.toUpperCase();
      const validAbbrevs = new Set(["OK", "HW", "AI", "ML", "DB", "UI", "UX", "PR", "QA", "JS", "TS", "FN"]);
      if (!validAbbrevs.has(upper)) { return true; }
    }
  }

  return false;
}

// =====================================================================
// Arabic Text Cleaning & Validation
// =====================================================================

/** Clean Arabic OCR artifacts */
function cleanArabicOcrText(raw: string): string {
  return raw
    .trim()
    .replace(/\n+/g, " ")
    .replace(/[{}[\]|_~^`]/g, "")
    .replace(/\s{2,}/g, " ")
    // Remove isolated Latin characters mixed into Arabic (common OCR noise)
    .replace(/(?<=[\u0600-\u06FF])\s*[a-zA-Z]\s*(?=[\u0600-\u06FF])/g, "")
    .trim();
}

/** Check if Arabic OCR output is garbage */
function isArabicGarbageText(text: string): boolean {
  if (text.length === 0) { return true; }

  const cleaned = text.replace(/\s/g, "");
  if (cleaned.length === 0) { return true; }

  // Single Arabic char is valid
  if (cleaned.length === 1 && /[\u0600-\u06FF]/.test(cleaned)) { return false; }

  // All same character
  if (new Set(cleaned).size <= 1 && cleaned.length > 1) { return true; }

  // 3+ consecutive same characters
  if (/(.)\1{2,}/.test(cleaned)) { return true; }

  // Must contain at least one Arabic character
  const arabicChars = (cleaned.match(/[\u0600-\u06FF]/g) || []).length;
  if (arabicChars === 0) { return true; }

  // Mostly non-Arabic/non-space (> 70% noise)
  const meaningful = arabicChars + (cleaned.match(/[\u0660-\u0669\d\s]/g) || []).length;
  if (meaningful < cleaned.length * 0.3 && cleaned.length > 3) { return true; }

  return false;
}

/** Check if Arabic text looks like a real word */
function isLikelyArabicWord(text: string): boolean {
  const arabicChars = (text.match(/[\u0600-\u06FF]/g) || []).length;
  return arabicChars >= 2;
}

// =====================================================================
// Text Detection Heuristic
// =====================================================================

export function looksLikeText(points: readonly LocalPoint[]): boolean {
  if (points.length < 6) { return false; }

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of points) {
    minX = Math.min(minX, p[0]);
    minY = Math.min(minY, p[1]);
    maxX = Math.max(maxX, p[0]);
    maxY = Math.max(maxY, p[1]);
  }

  const width = maxX - minX;
  const height = maxY - minY;
  if (width < 8 || height < 4) { return false; }

  let dirChanges = 0;
  for (let i = 2; i < points.length; i++) {
    const dx1 = points[i - 1][0] - points[i - 2][0];
    const dy1 = points[i - 1][1] - points[i - 2][1];
    const dx2 = points[i][0] - points[i - 1][0];
    const dy2 = points[i][1] - points[i - 1][1];
    const cross = dx1 * dy2 - dy1 * dx2;
    if (i > 2) {
      const prevDx1 = points[i - 2][0] - points[i - 3][0];
      const prevDy1 = points[i - 2][1] - points[i - 3][1];
      const prevCross = prevDx1 * dy1 - prevDy1 * dx1;
      if (Math.sign(cross) !== Math.sign(prevCross) && Math.abs(cross) > 0.2) {
        dirChanges++;
      }
    }
  }

  return dirChanges / points.length > 0.05;
}

// =====================================================================
// Public API
// =====================================================================

/**
 * Recognize handwritten text from freedraw points (single stroke).
 * Priority: Native HWR API → TrOCR → Tesseract. Best result wins.
 */
export async function recognizeHandwriting(
  points: readonly LocalPoint[],
  strokeWidth: number = 2,
  skipHeuristic: boolean = false,
): Promise<HandwritingResult> {
  const noResult: HandwritingResult = { text: "", confidence: 0 };

  if (!skipHeuristic && !looksLikeText(points)) {
    return noResult;
  }

  const results: HandwritingResult[] = [];

  // Engine 1: Native Handwriting Recognition API (fastest, most accurate)
  const nativePromise = (async () => {
    const r = await recognizeWithNativeAPI(points);
    if (r) { results.push(r); }
  })();

  // Engine 2: TrOCR with IAM-style rendering (384px, no binarization)
  const trOcrPromise = trOcrReady
    ? (async () => {
        const trCanvas = renderForTrOCR(points, strokeWidth);
        if (trCanvas) {
          const r = await recognizeWithTrOcrWorker(trCanvas);
          console.log(`[AI] TrOCR single: "${r.text}" conf=${r.confidence}`);
          if (r.text.length > 0 && !isGarbageText(r.text)) { results.push(r); }
        }
      })()
    : Promise.resolve();

  // Engine 3: Tesseract with high-res pressure-simulated rendering (120px)
  const tessPromise = (async () => {
    const tessCanvas = renderForTesseract(points, strokeWidth);
    if (tessCanvas) {
      const r = await recognizeWithTesseract(tessCanvas);
      if (r.text.length > 0) { results.push(r); }
    }
  })();

  await Promise.all([nativePromise, trOcrPromise, tessPromise]);

  if (results.length > 0) {
    // Sort by confidence, boost results that look like real words
    results.sort((a, b) => {
      const scoreA = a.confidence + (isLikelyWord(a.text) ? 8 : 0);
      const scoreB = b.confidence + (isLikelyWord(b.text) ? 8 : 0);
      return scoreB - scoreA;
    });
    if (results[0].confidence >= 20) {
      return results[0];
    }
  }

  return noResult;
}

/**
 * Recognize handwritten text from multiple strokes combined.
 * Priority: Native HWR API → TrOCR → Tesseract. Best result wins.
 */
export async function recognizeHandwritingMultiStroke(
  strokes: readonly { points: readonly LocalPoint[]; offsetX: number; offsetY: number }[],
  strokeWidth: number = 2,
): Promise<HandwritingResult> {
  const noResult: HandwritingResult = { text: "", confidence: 0 };

  if (strokes.length === 0) { return noResult; }

  const results: HandwritingResult[] = [];

  // Engine 1: Native API
  const nativePromise = (async () => {
    const r = await recognizeMultiStrokeNativeAPI(strokes);
    if (r) { results.push(r); }
  })();

  // Engine 2: TrOCR
  const trOcrPromise = trOcrReady
    ? (async () => {
        const trCanvas = renderMultiStrokeForTrOCR(strokes, strokeWidth);
        if (trCanvas) {
          const r = await recognizeWithTrOcrWorker(trCanvas);
          console.log(`[AI] TrOCR multi: "${r.text}" conf=${r.confidence}`);
          if (r.text.length > 0 && !isGarbageText(r.text)) { results.push(r); }
        }
      })()
    : (() => {
        console.log("[AI] TrOCR not ready, skipping");
        return Promise.resolve();
      })();

  // Engine 3: Tesseract
  const tessPromise = (async () => {
    const tessCanvas = renderMultiStrokeForTesseract(strokes, strokeWidth);
    if (tessCanvas) {
      const r = await recognizeWithTesseract(tessCanvas);
      if (r.text.length > 0) { results.push(r); }
    }
  })();

  await Promise.all([nativePromise, trOcrPromise, tessPromise]);

  if (results.length > 0) {
    results.sort((a, b) => {
      const scoreA = a.confidence + (isLikelyWord(a.text) ? 8 : 0);
      const scoreB = b.confidence + (isLikelyWord(b.text) ? 8 : 0);
      return scoreB - scoreA;
    });
    if (results[0].confidence >= 20) {
      return results[0];
    }
  }

  return noResult;
}

// =====================================================================
// Arabic Handwriting Recognition (exported)
// =====================================================================

/**
 * Recognize Arabic handwritten text from a single stroke.
 * Priority: Native HWR API → Tesseract (ara). Best result wins.
 */
export async function recognizeArabicHandwriting(
  points: readonly LocalPoint[],
  strokeWidth: number = 2,
): Promise<HandwritingResult> {
  const noResult: HandwritingResult = { text: "", confidence: 0 };

  if (points.length < 5) { return noResult; }

  const results: HandwritingResult[] = [];

  // Engine 1: Native Arabic HWR API (fastest, most accurate)
  const nativePromise = (async () => {
    const r = await recognizeArabicWithNativeAPI(points);
    if (r) { results.push(r); }
  })();

  // Engine 2: Arabic Tesseract
  const tessPromise = (async () => {
    const tessCanvas = renderForTesseract(points, strokeWidth);
    if (tessCanvas) {
      const r = await recognizeArabicWithTesseract(tessCanvas);
      if (r.text.length > 0) { results.push(r); }
    }
  })();

  await Promise.all([nativePromise, tessPromise]);

  if (results.length > 0) {
    results.sort((a, b) => {
      const scoreA = a.confidence + (isLikelyArabicWord(a.text) ? 8 : 0);
      const scoreB = b.confidence + (isLikelyArabicWord(b.text) ? 8 : 0);
      return scoreB - scoreA;
    });
    if (results[0].confidence >= 15) {
      return results[0];
    }
  }

  return noResult;
}

/**
 * Recognize Arabic handwritten text from multiple strokes.
 */
export async function recognizeArabicHandwritingMultiStroke(
  strokes: readonly { points: readonly LocalPoint[]; offsetX: number; offsetY: number }[],
  strokeWidth: number = 2,
): Promise<HandwritingResult> {
  const noResult: HandwritingResult = { text: "", confidence: 0 };

  if (strokes.length === 0) { return noResult; }

  const results: HandwritingResult[] = [];

  // Engine 1: Native Arabic API
  const nativePromise = (async () => {
    const r = await recognizeArabicMultiStrokeNativeAPI(strokes);
    if (r) { results.push(r); }
  })();

  // Engine 2: Arabic Tesseract
  const tessPromise = (async () => {
    const tessCanvas = renderMultiStrokeForTesseract(strokes, strokeWidth);
    if (tessCanvas) {
      const r = await recognizeArabicWithTesseract(tessCanvas);
      if (r.text.length > 0) { results.push(r); }
    }
  })();

  await Promise.all([nativePromise, tessPromise]);

  if (results.length > 0) {
    results.sort((a, b) => {
      const scoreA = a.confidence + (isLikelyArabicWord(a.text) ? 8 : 0);
      const scoreB = b.confidence + (isLikelyArabicWord(b.text) ? 8 : 0);
      return scoreB - scoreA;
    });
    if (results[0].confidence >= 15) {
      return results[0];
    }
  }

  return noResult;
}

/**
 * Pre-initialize engines (call when handwriting recognition is enabled).
 * Native HWR checked first, TrOCR loads in background, Tesseract available immediately.
 */
export async function preloadHandwritingEngine(): Promise<void> {
  ensureNativeRecognizer().catch(() => {});
  ensureTrOcrWorker().catch(() => {});
  await ensureTesseract().catch(() => {});
}

/**
 * Pre-initialize Arabic engines.
 */
export async function preloadArabicHandwritingEngine(): Promise<void> {
  ensureNativeArabicRecognizer().catch(() => {});
  await ensureTesseractArabic().catch(() => {});
}

/**
 * Terminate engines when no longer needed
 */
export async function terminateHandwritingEngine(): Promise<void> {
  if (tesseractWorker) {
    try { await tesseractWorker.terminate(); } catch { /* ignore */ }
    tesseractWorker = null;
  }
  if (trOcrWorker) {
    trOcrWorker.terminate();
    trOcrWorker = null;
    trOcrReady = false;
    trOcrLoading = false;
    trOcrLoadPromise = null;
  }
  if (nativeRecognizer) {
    try { nativeRecognizer.finish(); } catch { /* ignore */ }
    nativeRecognizer = null;
  }
}

/**
 * Terminate Arabic engines when no longer needed
 */
export async function terminateArabicHandwritingEngine(): Promise<void> {
  if (tesseractArabicWorker) {
    try { await tesseractArabicWorker.terminate(); } catch { /* ignore */ }
    tesseractArabicWorker = null;
  }
  if (nativeArabicRecognizer) {
    try { nativeArabicRecognizer.finish(); } catch { /* ignore */ }
    nativeArabicRecognizer = null;
  }
}
