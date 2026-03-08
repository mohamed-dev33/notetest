/**
 * Handwriting Recognition Module — TrOCR + Tesseract.js
 *
 * Uses Microsoft TrOCR (via @xenova/transformers / ONNX Runtime Web) as
 * the primary handwriting recognizer. Falls back to Tesseract.js if
 * TrOCR fails or is unavailable. Both run entirely in the browser.
 *
 * TrOCR is a transformer-based model specifically trained on handwritten
 * text (IAM dataset) — far more accurate than traditional OCR for
 * freehand writing.
 */

import type { LocalPoint } from "@excalidraw/math";

export interface HandwritingResult {
  text: string;
  confidence: number; // 0-100
}

// =====================================================================
// Canvas Preprocessing — Critical for OCR accuracy
// =====================================================================

/**
 * Render freedraw points to a high-quality canvas optimized for OCR.
 * - Scales to standard height (64px for TrOCR, 300 DPI equivalent)
 * - White background, black strokes
 * - Generous padding
 * - Anti-aliased smooth strokes
 */
function renderPointsToCanvas(
  points: readonly LocalPoint[],
  strokeWidth: number = 3,
  targetHeight: number = 64,
): HTMLCanvasElement | null {
  if (points.length < 2) {
    return null;
  }

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of points) {
    minX = Math.min(minX, p[0]);
    minY = Math.min(minY, p[1]);
    maxX = Math.max(maxX, p[0]);
    maxY = Math.max(maxY, p[1]);
  }

  const rawW = maxX - minX;
  const rawH = maxY - minY;
  if (rawW < 5 && rawH < 5) { return null; }

  // Scale so height = targetHeight, with generous padding
  const padding = 16;
  const scale = Math.max(1, targetHeight / Math.max(rawH, 1));
  const canvasW = Math.ceil(rawW * scale + padding * 2);
  const canvasH = Math.ceil(rawH * scale + padding * 2);

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(canvasW, targetHeight);
  canvas.height = Math.max(canvasH, targetHeight);

  const ctx = canvas.getContext("2d");
  if (!ctx) { return null; }

  // White background
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw stroke with smooth anti-aliased lines
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = Math.max(2, strokeWidth * scale * 0.8);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.imageSmoothingEnabled = true;

  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    const x = (points[i][0] - minX) * scale + padding;
    const y = (points[i][1] - minY) * scale + padding;
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      // Use quadratic curves for smoother rendering
      const prev = points[i - 1];
      const prevX = (prev[0] - minX) * scale + padding;
      const prevY = (prev[1] - minY) * scale + padding;
      const midX = (prevX + x) / 2;
      const midY = (prevY + y) / 2;
      ctx.quadraticCurveTo(prevX, prevY, midX, midY);
    }
  }
  ctx.stroke();

  return canvas;
}

/**
 * Render to a larger canvas for Tesseract (needs higher res)
 */
function renderForTesseract(
  points: readonly LocalPoint[],
  strokeWidth: number = 3,
): HTMLCanvasElement | null {
  return renderPointsToCanvas(points, strokeWidth, 200);
}

// =====================================================================
// TrOCR Engine (Primary — State of the Art)
// =====================================================================

let trOcrPipeline: any = null;
let trOcrLoading = false;
let trOcrLoadPromise: Promise<any> | null = null;
let trOcrAvailable: boolean | null = null; // null = not yet checked

async function ensureTrOcr(): Promise<any> {
  if (trOcrPipeline) { return trOcrPipeline; }
  if (trOcrAvailable === false) { return null; }

  if (trOcrLoading && trOcrLoadPromise) {
    await trOcrLoadPromise;
    return trOcrPipeline;
  }

  trOcrLoading = true;
  trOcrLoadPromise = (async () => {
    try {
      const { pipeline, env } = await import("@xenova/transformers");
      // Configure environment for browser usage
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      // Ensure remote model fetching is enabled
      env.allowRemoteModels = true;
      // Use CDN for ONNX wasm files
      if (env.backends?.onnx?.wasm) {
        env.backends.onnx.wasm.numThreads = 1;
      }
      console.log("[AI] Loading TrOCR model from HuggingFace...");
      // Load the small TrOCR model for handwritten text
      trOcrPipeline = await pipeline(
        "image-to-text",
        "Xenova/trocr-small-handwritten",
        {
          quantized: true,
        },
      );
      trOcrAvailable = true;
      console.log("[AI] TrOCR handwriting model loaded successfully");
    } catch (e) {
      console.warn("[AI] TrOCR not available, will use Tesseract fallback:", e);
      trOcrAvailable = false;
      trOcrPipeline = null;
    }
  })();

  await trOcrLoadPromise;
  trOcrLoading = false;
  return trOcrPipeline;
}

async function recognizeWithTrOcr(
  canvas: HTMLCanvasElement,
): Promise<HandwritingResult> {
  const noResult: HandwritingResult = { text: "", confidence: 0 };

  try {
    const pipe = await ensureTrOcr();
    if (!pipe) { return noResult; }

    // Convert canvas to blob URL for the pipeline
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/png"),
    );
    if (!blob) { return noResult; }

    const url = URL.createObjectURL(blob);

    try {
      const results = await pipe(url);
      if (results && results.length > 0) {
        const text = (results[0].generated_text || "").trim();
        if (text.length > 0) {
          // TrOCR doesn't return confidence directly;
          // estimate based on text quality heuristics
          const hasAlphanumeric = /[a-zA-Z0-9]/.test(text);
          const confidence = hasAlphanumeric ? 85 : 40;
          return { text, confidence };
        }
      }
    } finally {
      URL.revokeObjectURL(url);
    }
  } catch (e) {
    console.warn("[AI] TrOCR recognition error:", e);
  }

  return noResult;
}

// =====================================================================
// Tesseract.js Engine (Fallback)
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
      // Try multiple page segmentation modes for best results
      await tesseractWorker.setParameters({
        tessedit_pageseg_mode: "7", // single text line
        preserve_interword_spaces: "1",
      });
    } catch (e) {
      console.error("[AI] Tesseract initialization failed:", e);
      tesseractWorker = null;
    }
  })();

  await tesseractLoadPromise;
  tesseractLoading = false;
  return tesseractWorker;
}

async function recognizeWithTesseract(
  canvas: HTMLCanvasElement,
): Promise<HandwritingResult> {
  const noResult: HandwritingResult = { text: "", confidence: 0 };

  try {
    const worker = await ensureTesseract();
    if (!worker) { return noResult; }

    // Try PSM 7 (single text line) first
    const { data } = await worker.recognize(canvas);
    let text = data.text.trim().replace(/\n+/g, " ");
    let confidence = data.confidence;

    // If low confidence, try PSM 8 (single word)
    if (confidence < 50 && text.length > 0) {
      await worker.setParameters({ tessedit_pageseg_mode: "8" });
      const { data: data2 } = await worker.recognize(canvas);
      if (data2.confidence > confidence) {
        text = data2.text.trim().replace(/\n+/g, " ");
        confidence = data2.confidence;
      }
      // Reset to PSM 7
      await worker.setParameters({ tessedit_pageseg_mode: "7" });
    }

    if (text.length === 0 || confidence < 25) {
      return noResult;
    }

    return { text, confidence };
  } catch (e) {
    console.error("[AI] Tesseract recognition error:", e);
    return noResult;
  }
}

// =====================================================================
// Text Detection Heuristic
// =====================================================================

/**
 * Heuristic to detect if a stroke likely represents handwritten text.
 * Checks direction changes, aspect ratio, and path complexity.
 */
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
  if (width < 10 || height < 5) { return false; }

  // Count direction changes — text has many
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
      if (Math.sign(cross) !== Math.sign(prevCross) && Math.abs(cross) > 0.3) {
        dirChanges++;
      }
    }
  }

  return dirChanges / points.length > 0.06;
}

// =====================================================================
// Multi-Stroke Canvas Rendering
// =====================================================================

/**
 * Render multiple freedraw strokes onto a single canvas for OCR.
 * Each stroke has its own points and a global offset (element x, y).
 * Strokes are combined into one image preserving spatial relationships.
 */
function renderMultiStrokeToCanvas(
  strokes: readonly { points: readonly LocalPoint[]; offsetX: number; offsetY: number }[],
  strokeWidth: number = 3,
  targetHeight: number = 64,
): HTMLCanvasElement | null {
  if (strokes.length === 0) { return null; }

  // Compute global bounding box across all strokes
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
  if (rawW < 5 && rawH < 5) { return null; }

  const padding = 16;
  const scale = Math.max(1, targetHeight / Math.max(rawH, 1));
  const canvasW = Math.ceil(rawW * scale + padding * 2);
  const canvasH = Math.ceil(rawH * scale + padding * 2);

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(canvasW, targetHeight);
  canvas.height = Math.max(canvasH, targetHeight);

  const ctx = canvas.getContext("2d");
  if (!ctx) { return null; }

  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "#000000";
  ctx.lineWidth = Math.max(2, strokeWidth * scale * 0.8);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.imageSmoothingEnabled = true;

  // Draw each stroke
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

// =====================================================================
// Public API
// =====================================================================

/**
 * Recognize handwritten text from freedraw points (single stroke).
 * Uses TrOCR (primary) with Tesseract.js (fallback).
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

  // Try TrOCR first
  const trOcrCanvas = renderPointsToCanvas(points, strokeWidth, 64);
  if (trOcrCanvas) {
    const trOcrResult = await recognizeWithTrOcr(trOcrCanvas);
    if (trOcrResult.text.length > 0 && trOcrResult.confidence >= 40) {
      return trOcrResult;
    }
  }

  // Fallback to Tesseract.js
  const tessCanvas = renderForTesseract(points, strokeWidth);
  if (tessCanvas) {
    const tessResult = await recognizeWithTesseract(tessCanvas);
    if (tessResult.text.length > 0 && tessResult.confidence >= 30) {
      return tessResult;
    }
  }

  return noResult;
}

/**
 * Recognize handwritten text from multiple strokes combined.
 * All strokes are rendered onto a single canvas preserving
 * their spatial positions, then fed to OCR.
 */
export async function recognizeHandwritingMultiStroke(
  strokes: readonly { points: readonly LocalPoint[]; offsetX: number; offsetY: number }[],
  strokeWidth: number = 2,
): Promise<HandwritingResult> {
  const noResult: HandwritingResult = { text: "", confidence: 0 };

  if (strokes.length === 0) { return noResult; }

  // TrOCR with combined canvas
  const trOcrCanvas = renderMultiStrokeToCanvas(strokes, strokeWidth, 64);
  if (trOcrCanvas) {
    const trOcrResult = await recognizeWithTrOcr(trOcrCanvas);
    if (trOcrResult.text.length > 0 && trOcrResult.confidence >= 40) {
      return trOcrResult;
    }
  }

  // Tesseract fallback with larger canvas
  const tessCanvas = renderMultiStrokeToCanvas(strokes, strokeWidth, 200);
  if (tessCanvas) {
    const tessResult = await recognizeWithTesseract(tessCanvas);
    if (tessResult.text.length > 0 && tessResult.confidence >= 30) {
      return tessResult;
    }
  }

  return noResult;
}

/**
 * Pre-initialize engines (call when handwriting recognition is enabled)
 */
export async function preloadHandwritingEngine(): Promise<void> {
  // Start loading both engines in parallel
  const promises: Promise<any>[] = [];
  promises.push(ensureTrOcr().catch(() => {}));
  promises.push(ensureTesseract().catch(() => {}));
  await Promise.all(promises);
}

/**
 * Terminate engines when no longer needed
 */
export async function terminateHandwritingEngine(): Promise<void> {
  if (tesseractWorker) {
    try { await tesseractWorker.terminate(); } catch { /* ignore */ }
    tesseractWorker = null;
  }
  // TrOCR pipeline doesn't need explicit termination
  trOcrPipeline = null;
  trOcrAvailable = null;
}
