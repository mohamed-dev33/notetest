/**
 * AI Recognition Manager
 *
 * Orchestrates shape and handwriting recognition for freedraw elements.
 * Called after a freedraw element is finalized to optionally replace it
 * with a recognized shape or text element.
 *
 * Strategy:
 * - Single stroke: shape runs first (fast), line/arrow returns immediately.
 *   For other shapes, handwriting runs in parallel; best wins.
 * - Multi-stroke: check if strokes look like separate characters (text)
 *   vs connected parts of one shape.
 *   For text-like input, only run handwriting.
 *   For shape-like input, run both and pick best.
 */

import type { ExcalidrawFreeDrawElement } from "@excalidraw/element/types";
import type { LocalPoint } from "@excalidraw/math";

import {
  recognizeShape,
  type RecognizedShape,
  type RecognizedShapeType,
} from "./shapeRecognition";
import {
  recognizeHandwriting,
  recognizeHandwritingMultiStroke,
  recognizeArabicHandwriting,
  recognizeArabicHandwritingMultiStroke,
  looksLikeText,
  preloadHandwritingEngine,
  preloadArabicHandwritingEngine,
  terminateHandwritingEngine,
  terminateArabicHandwritingEngine,
  type HandwritingResult,
} from "./handwritingRecognition";

export interface AIRecognitionResult {
  type: "shape" | "text" | "none";
  shape?: RecognizedShape;
  handwriting?: HandwritingResult;
  consumedElements?: ExcalidrawFreeDrawElement[];
}

// Higher thresholds = fewer false positives
const SHAPE_CONFIDENCE_THRESHOLD = 0.68;
const HANDWRITING_CONFIDENCE_THRESHOLD = 25;

/**
 * Process a single freedraw element.
 * Runs shape + handwriting in parallel, picks the best.
 */
export async function processFreedrawElement(
  element: ExcalidrawFreeDrawElement,
  shapeRecognitionEnabled: boolean,
  handwritingRecognitionEnabled: boolean,
  arabicHandwritingEnabled: boolean = false,
): Promise<AIRecognitionResult> {
  const noResult: AIRecognitionResult = { type: "none" };

  if (!shapeRecognitionEnabled && !handwritingRecognitionEnabled && !arabicHandwritingEnabled) {
    return noResult;
  }

  const points = element.points as readonly LocalPoint[];

  if (points.length < 5) {
    return noResult;
  }

  // Run shape recognition first (synchronous, fast)
  let shapeResult: RecognizedShape | null = null;

  if (shapeRecognitionEnabled) {
    shapeResult = recognizeShape(points, element.x, element.y);
    if (shapeResult.type === "freedraw" || shapeResult.confidence < SHAPE_CONFIDENCE_THRESHOLD) {
      shapeResult = null;
    }
  }

  // Line/arrow: return immediately without waiting for slow handwriting OCR
  if (
    shapeResult &&
    (shapeResult.type === "line" || shapeResult.type === "arrow") &&
    shapeResult.confidence >= SHAPE_CONFIDENCE_THRESHOLD
  ) {
    return { type: "shape", shape: shapeResult, consumedElements: [element] };
  }

  // High-confidence closed shape (≥ 88%): return without handwriting
  if (shapeResult && shapeResult.confidence >= 0.88) {
    return { type: "shape", shape: shapeResult, consumedElements: [element] };
  }

  // Run handwriting recognition (async, slower)
  let hwResult: HandwritingResult | null = null;

  if (handwritingRecognitionEnabled) {
    const r = await recognizeHandwriting(points, element.strokeWidth, true);
    if (r.text.length > 0 && r.confidence >= HANDWRITING_CONFIDENCE_THRESHOLD) {
      hwResult = r;
    }
  }

  // Run Arabic handwriting recognition
  if (arabicHandwritingEnabled) {
    const r = await recognizeArabicHandwriting(points, element.strokeWidth);
    if (r.text.length > 0 && r.confidence >= HANDWRITING_CONFIDENCE_THRESHOLD) {
      // Prefer Arabic result if it has Arabic characters
      if (!hwResult || r.confidence >= hwResult.confidence) {
        hwResult = r;
      }
    }
  }

  // Decision logic: pick the best result
  const sr = shapeResult;
  const hr = hwResult;

  if (sr && hr) {
    if (sr.confidence >= 0.85) {
      return { type: "shape", shape: sr, consumedElements: [element] };
    }
    if (looksLikeText(points)) {
      return { type: "text", handwriting: hr, consumedElements: [element] };
    }
    if (sr.confidence * 100 > hr.confidence) {
      return { type: "shape", shape: sr, consumedElements: [element] };
    }
    return { type: "text", handwriting: hr, consumedElements: [element] };
  }

  if (sr) {
    return { type: "shape", shape: sr, consumedElements: [element] };
  }

  if (hr) {
    return { type: "text", handwriting: hr, consumedElements: [element] };
  }

  return noResult;
}

/**
 * Check if a set of strokes look like separate characters (text)
 * vs parts of a single connected shape.
 *
 * Text strokes: similar heights, horizontally separated, on same baseline.
 * Shape strokes: overlap, connect, or form a single larger pattern.
 */
function checkStrokesAreSeparateChars(
  elements: ExcalidrawFreeDrawElement[],
): boolean {
  if (elements.length < 2) { return false; }
  if (elements.length > 25) { return false; }

  const boxes = elements.map((el) => {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of el.points as readonly LocalPoint[]) {
      minX = Math.min(minX, el.x + p[0]);
      minY = Math.min(minY, el.y + p[1]);
      maxX = Math.max(maxX, el.x + p[0]);
      maxY = Math.max(maxY, el.y + p[1]);
    }
    return { minX, minY, maxX, maxY, w: maxX - minX, h: maxY - minY };
  });

  const allMinY = Math.min(...boxes.map((b) => b.minY));
  const allMaxY = Math.max(...boxes.map((b) => b.maxY));
  const totalHeight = allMaxY - allMinY;

  if (totalHeight < 5) { return false; }

  // Count strokes sharing a similar vertical baseline
  let sameLineCount = 0;
  for (const box of boxes) {
    const vertOverlap = (Math.min(allMaxY, box.maxY) - Math.max(allMinY, box.minY)) / totalHeight;
    if (vertOverlap > 0.3 || box.h < totalHeight * 0.3) {
      sameLineCount++;
    }
  }

  // Check horizontal separation — text characters don't heavily overlap
  const sortedByX = [...boxes].sort((a, b) => a.minX - b.minX);
  let overlapCount = 0;
  for (let i = 1; i < sortedByX.length; i++) {
    const prevRight = sortedByX[i - 1].maxX;
    const currLeft = sortedByX[i].minX;
    const overlapFrac = (prevRight - currLeft) / Math.max(sortedByX[i].w, 1);
    if (overlapFrac > 0.5) { overlapCount++; }
  }

  // Text: most strokes on same line, limited horizontal overlap
  const sameLineFrac = sameLineCount / elements.length;
  const overlapFrac = overlapCount / Math.max(elements.length - 1, 1);

  return sameLineFrac >= 0.6 && overlapFrac < 0.4;
}

/**
 * Check if multi-stroke input forms a single shape (e.g., arrow = shaft + head).
 * Analyzes spatial connectivity between strokes.
 */
function checkStrokesFormOneShape(
  elements: ExcalidrawFreeDrawElement[],
): boolean {
  if (elements.length < 2 || elements.length > 5) { return false; }

  // Get start/end points of each stroke
  const strokeEnds = elements.map((el) => {
    const pts = el.points as readonly LocalPoint[];
    return {
      start: { x: el.x + pts[0][0], y: el.y + pts[0][1] },
      end: { x: el.x + pts[pts.length - 1][0], y: el.y + pts[pts.length - 1][1] },
    };
  });

  // Check if strokes connect — an end of one is near a start/end of another
  let connectionCount = 0;
  const threshold = 30; // pixels

  for (let i = 0; i < strokeEnds.length; i++) {
    for (let j = i + 1; j < strokeEnds.length; j++) {
      const distances = [
        Math.hypot(strokeEnds[i].end.x - strokeEnds[j].start.x, strokeEnds[i].end.y - strokeEnds[j].start.y),
        Math.hypot(strokeEnds[i].end.x - strokeEnds[j].end.x, strokeEnds[i].end.y - strokeEnds[j].end.y),
        Math.hypot(strokeEnds[i].start.x - strokeEnds[j].start.x, strokeEnds[i].start.y - strokeEnds[j].start.y),
        Math.hypot(strokeEnds[i].start.x - strokeEnds[j].end.x, strokeEnds[i].start.y - strokeEnds[j].end.y),
      ];
      if (Math.min(...distances) < threshold) {
        connectionCount++;
      }
    }
  }

  // At least one connection between strokes
  return connectionCount > 0;
}

/**
 * Process a batch of freedraw elements drawn in quick succession.
 *
 * For multi-stroke input:
 * - If strokes look like separate characters → handwriting recognition
 * - If strokes form one connected shape → shape recognition on combined points
 * - Otherwise → run both and pick best
 */
export async function processFreedrawBatch(
  elements: ExcalidrawFreeDrawElement[],
  shapeRecognitionEnabled: boolean,
  handwritingRecognitionEnabled: boolean,
  arabicHandwritingEnabled: boolean = false,
): Promise<AIRecognitionResult> {
  const noResult: AIRecognitionResult = { type: "none" };

  if (elements.length === 0) {
    return noResult;
  }

  if (elements.length === 1) {
    return processFreedrawElement(
      elements[0],
      shapeRecognitionEnabled,
      handwritingRecognitionEnabled,
      arabicHandwritingEnabled,
    );
  }

  // --- Analyze multi-stroke pattern ---
  const strokesLikeSeparateChars = checkStrokesAreSeparateChars(elements);
  const strokesFormOneShape = !strokesLikeSeparateChars && checkStrokesFormOneShape(elements);

  // --- Run both in parallel ---
  let shapeResult: RecognizedShape | null = null;
  let hwResult: HandwritingResult | null = null;

  // Shape recognition: combine all points
  const shapePromise = (shapeRecognitionEnabled && !strokesLikeSeparateChars)
    ? Promise.resolve().then(() => {
        const combinedPoints: LocalPoint[] = [];
        let globalMinX = Infinity, globalMinY = Infinity;

        for (const el of elements) {
          for (const p of el.points as readonly LocalPoint[]) {
            globalMinX = Math.min(globalMinX, el.x + p[0]);
            globalMinY = Math.min(globalMinY, el.y + p[1]);
          }
        }

        for (const el of elements) {
          for (const p of el.points as readonly LocalPoint[]) {
            combinedPoints.push(
              [p[0] + el.x - globalMinX, p[1] + el.y - globalMinY] as LocalPoint,
            );
          }
        }

        if (combinedPoints.length >= 5) {
          const result = recognizeShape(combinedPoints, globalMinX, globalMinY);
          if (
            result.type !== "freedraw" &&
            result.confidence >= SHAPE_CONFIDENCE_THRESHOLD
          ) {
            shapeResult = result;
          }
        }
      })
    : Promise.resolve();

  // Handwriting: multi-stroke recognition
  const hwPromise = (handwritingRecognitionEnabled || arabicHandwritingEnabled)
    ? (async () => {
        const allStrokes: { points: readonly LocalPoint[]; offsetX: number; offsetY: number }[] = [];
        for (const el of elements) {
          if (el.points.length >= 2) {
            allStrokes.push({
              points: el.points as readonly LocalPoint[],
              offsetX: el.x,
              offsetY: el.y,
            });
          }
        }

        if (allStrokes.length > 0) {
          const avgSW = elements.reduce((s, e) => s + e.strokeWidth, 0) / elements.length;

          if (arabicHandwritingEnabled) {
            const result = await recognizeArabicHandwritingMultiStroke(allStrokes, avgSW);
            if (result.text.length > 0 && result.confidence >= HANDWRITING_CONFIDENCE_THRESHOLD) {
              hwResult = result;
            }
          } else {
            const result = await recognizeHandwritingMultiStroke(allStrokes, avgSW);
            if (result.text.length > 0 && result.confidence >= HANDWRITING_CONFIDENCE_THRESHOLD) {
              hwResult = result;
            }
          }
        }
      })()
    : Promise.resolve();

  await Promise.all([shapePromise, hwPromise]);

  const sr = shapeResult as RecognizedShape | null;
  const hr = hwResult as HandwritingResult | null;

  // Decision logic for multi-stroke
  if (hr && sr) {
    // If strokes form one connected shape, prefer shape
    if (strokesFormOneShape && sr.confidence >= 0.75) {
      return { type: "shape", shape: sr, consumedElements: elements };
    }
    // Very high confidence shape wins regardless
    if (sr.confidence >= 0.90) {
      return { type: "shape", shape: sr, consumedElements: elements };
    }
    // Otherwise prefer handwriting for multi-stroke (text is usually multi-stroke)
    return { type: "text", handwriting: hr, consumedElements: elements };
  }

  if (hr) {
    return { type: "text", handwriting: hr, consumedElements: elements };
  }

  if (sr) {
    return { type: "shape", shape: sr, consumedElements: elements };
  }

  // Fallback: try each element individually for shape
  if (shapeRecognitionEnabled) {
    for (const el of elements) {
      const points = el.points as readonly LocalPoint[];
      if (points.length >= 5) {
        const result = recognizeShape(points, el.x, el.y);
        if (
          result.type !== "freedraw" &&
          result.confidence >= SHAPE_CONFIDENCE_THRESHOLD
        ) {
          return { type: "shape", shape: result, consumedElements: [el] };
        }
      }
    }
  }

  return noResult;
}

export {
  preloadHandwritingEngine,
  preloadArabicHandwritingEngine,
  terminateHandwritingEngine,
  terminateArabicHandwritingEngine,
  type RecognizedShape,
  type RecognizedShapeType,
  type HandwritingResult,
};
