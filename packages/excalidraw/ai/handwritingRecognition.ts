/**
 * Handwriting Recognition Module — TrOCR (Web Worker) + Tesseract.js
 *
 * Primary engine: TrOCR (transformer-based, trained on IAM handwriting).
 * Runs in a dedicated Web Worker so inference never blocks the UI.
 *
 * Fallback engine: Tesseract.js (also runs in a Web Worker via its own
 * internal worker mechanism).
 *
 * Both engines are non-blocking — the main thread stays responsive.
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
 * Uses binarization with thick strokes on white background.
 */
function renderPointsToCanvas(
  points: readonly LocalPoint[],
  strokeWidth: number = 3,
  targetHeight: number = 300,
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
  if (rawW < 3 && rawH < 3) { return null; }

  const padding = 32;
  const scale = Math.max(1.5, targetHeight / Math.max(rawH, 1));
  const canvasW = Math.ceil(rawW * scale + padding * 2);
  const canvasH = Math.ceil(rawH * scale + padding * 2);

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(canvasW, targetHeight);
  canvas.height = Math.max(canvasH, targetHeight);

  const ctx = canvas.getContext("2d")!;
  if (!ctx) { return null; }

  // White background
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Thick black strokes — thicker = clearer for OCR
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = Math.max(4, strokeWidth * scale * 1.2);
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
      // CRITICAL: Disable dictionary-based word correction
      // This prevents Tesseract from "fixing" handwritten text
      // (e.g. "youssef" → "you say" or "gogst")
      await tesseractWorker.setParameters({
        tessedit_pageseg_mode: "7",
        preserve_interword_spaces: "1",
        load_system_dawg: "0",
        load_freq_dawg: "0",
        language_model_penalty_non_dict_word: "0",
        language_model_penalty_non_freq_dict_word: "0",
      });
      console.log("[AI] Tesseract worker ready (dictionary disabled)");
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

    let bestText = "";
    let bestConf = 0;

    // Try multiple page segmentation modes and pick the best result
    const psmModes = [
      "7",   // single text line
      "8",   // single word
      "6",   // uniform block of text
      "13",  // raw line (no language model)
    ];

    for (const psm of psmModes) {
      try {
        await worker.setParameters({
          tessedit_pageseg_mode: psm,
          load_system_dawg: "0",
          load_freq_dawg: "0",
        });
        const { data } = await worker.recognize(canvas);
        const text = data.text.trim().replace(/\n+/g, " ");
        const conf = data.confidence;

        if (text.length > 0 && conf > bestConf) {
          bestText = text;
          bestConf = conf;
        }

        // Good enough — stop early
        if (conf >= 75) { break; }
      } catch {
        // PSM mode failed — continue with next
      }
    }

    // Reset to default PSM
    await worker.setParameters({
      tessedit_pageseg_mode: "7",
      load_system_dawg: "0",
      load_freq_dawg: "0",
    });

    if (bestText.length === 0 || bestConf < 20) {
      return noResult;
    }

    // Clean up common OCR artifacts
    bestText = bestText
      .replace(/[|]/g, "l")
      .replace(/[{}[\]]/g, "")
      .replace(/\s{2,}/g, " ")
      .trim();

    return { text: bestText, confidence: bestConf };
  } catch (e) {
    console.error("[AI] Tesseract recognition error:", e);
    return noResult;
  }
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
// Multi-Stroke Canvas Rendering
// =====================================================================

function renderMultiStrokeToCanvas(
  strokes: readonly { points: readonly LocalPoint[]; offsetX: number; offsetY: number }[],
  strokeWidth: number = 3,
  targetHeight: number = 200,
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

  const padding = 32;
  const scale = Math.max(1.5, targetHeight / Math.max(rawH, 1));
  const canvasW = Math.ceil(rawW * scale + padding * 2);
  const canvasH = Math.ceil(rawH * scale + padding * 2);

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(canvasW, targetHeight);
  canvas.height = Math.max(canvasH, targetHeight);

  const ctx = canvas.getContext("2d")!;
  if (!ctx) { return null; }

  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "#000000";
  ctx.lineWidth = Math.max(4, strokeWidth * scale * 1.2);
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

// =====================================================================
// Public API
// =====================================================================

/**
 * Recognize handwritten text from freedraw points (single stroke).
 * TrOCR (Web Worker, non-blocking) → Tesseract fallback.
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

  // Run BOTH engines in parallel and pick the best result
  const results: HandwritingResult[] = [];

  // TrOCR (128px height, optimized for transformer model)
  const trOcrPromise = trOcrReady
    ? (async () => {
        const trCanvas = renderPointsToCanvas(points, strokeWidth, 128);
        if (trCanvas) {
          const r = await recognizeWithTrOcrWorker(trCanvas);
          if (r.text.length > 0) { results.push(r); }
        }
      })()
    : Promise.resolve();

  // Tesseract (300px height for traditional OCR)
  const tessPromise = (async () => {
    const tessCanvas = renderPointsToCanvas(points, strokeWidth, 300);
    if (tessCanvas) {
      const r = await recognizeWithTesseract(tessCanvas);
      if (r.text.length > 0) { results.push(r); }
    }
  })();

  await Promise.all([trOcrPromise, tessPromise]);

  // Pick highest confidence result
  if (results.length > 0) {
    results.sort((a, b) => b.confidence - a.confidence);
    if (results[0].confidence >= 20) {
      return results[0];
    }
  }

  return noResult;
}

/**
 * Recognize handwritten text from multiple strokes combined.
 * TrOCR (Web Worker) → Tesseract fallback.
 */
export async function recognizeHandwritingMultiStroke(
  strokes: readonly { points: readonly LocalPoint[]; offsetX: number; offsetY: number }[],
  strokeWidth: number = 2,
): Promise<HandwritingResult> {
  const noResult: HandwritingResult = { text: "", confidence: 0 };

  if (strokes.length === 0) { return noResult; }

  // Run BOTH engines in parallel and pick the best result
  const results: HandwritingResult[] = [];

  const trOcrPromise = trOcrReady
    ? (async () => {
        const trCanvas = renderMultiStrokeToCanvas(strokes, strokeWidth, 128);
        if (trCanvas) {
          const r = await recognizeWithTrOcrWorker(trCanvas);
          if (r.text.length > 0) { results.push(r); }
        }
      })()
    : Promise.resolve();

  const tessPromise = (async () => {
    const tessCanvas = renderMultiStrokeToCanvas(strokes, strokeWidth, 300);
    if (tessCanvas) {
      const r = await recognizeWithTesseract(tessCanvas);
      if (r.text.length > 0) { results.push(r); }
    }
  })();

  await Promise.all([trOcrPromise, tessPromise]);

  if (results.length > 0) {
    results.sort((a, b) => b.confidence - a.confidence);
    if (results[0].confidence >= 20) {
      return results[0];
    }
  }

  return noResult;
}

/**
 * Pre-initialize both engines (call when handwriting recognition is enabled).
 * TrOCR loads in background — Tesseract is available immediately as fallback.
 */
export async function preloadHandwritingEngine(): Promise<void> {
  // Start both in parallel — neither blocks the main thread
  ensureTrOcrWorker().catch(() => {});
  await ensureTesseract().catch(() => {});
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
}
