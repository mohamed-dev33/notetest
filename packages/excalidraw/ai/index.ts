/**
 * AI Recognition Manager
 *
 * Orchestrates shape and handwriting recognition for freedraw elements.
 * Called after a freedraw element is finalized to optionally replace it
 * with a recognized shape or text element.
 *
 * Strategy: run both engines in parallel, pick the best result.
 * For multi-stroke input, prefer handwriting (text is usually multi-stroke).
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
  looksLikeText,
  preloadHandwritingEngine,
  terminateHandwritingEngine,
  type HandwritingResult,
} from "./handwritingRecognition";

export interface AIRecognitionResult {
  type: "shape" | "text" | "none";
  shape?: RecognizedShape;
  handwriting?: HandwritingResult;
  consumedElements?: ExcalidrawFreeDrawElement[];
}

// Higher thresholds = fewer false positives
const SHAPE_CONFIDENCE_THRESHOLD = 0.70;
const HANDWRITING_CONFIDENCE_THRESHOLD = 30;

/**
 * Process a single freedraw element.
 * Runs shape + handwriting in parallel, picks the best.
 */
export async function processFreedrawElement(
  element: ExcalidrawFreeDrawElement,
  shapeRecognitionEnabled: boolean,
  handwritingRecognitionEnabled: boolean,
): Promise<AIRecognitionResult> {
  const noResult: AIRecognitionResult = { type: "none" };

  if (!shapeRecognitionEnabled && !handwritingRecognitionEnabled) {
    return noResult;
  }

  const points = element.points as readonly LocalPoint[];

  if (points.length < 5) {
    return noResult;
  }

  // Run both in parallel
  let shapeResult: RecognizedShape | null = null;
  let hwResult: HandwritingResult | null = null;

  const shapePromise = shapeRecognitionEnabled
    ? Promise.resolve().then(() => {
        shapeResult = recognizeShape(points, element.x, element.y);
      })
    : Promise.resolve();

  const hwPromise = handwritingRecognitionEnabled
    ? recognizeHandwriting(points, element.strokeWidth, true).then((r) => {
        if (r.text.length > 0 && r.confidence >= HANDWRITING_CONFIDENCE_THRESHOLD) {
          hwResult = r;
        }
      })
    : Promise.resolve();

  await Promise.all([shapePromise, hwPromise]);

  const sr = shapeResult as RecognizedShape | null;
  const hr = hwResult as HandwritingResult | null;

  const shapeValid =
    sr !== null &&
    sr.type !== "freedraw" &&
    sr.confidence >= SHAPE_CONFIDENCE_THRESHOLD;

  // If we have both, compare quality
  if (shapeValid && hr) {
    if (sr!.confidence >= 0.85) {
      return { type: "shape", shape: sr!, consumedElements: [element] };
    }
    if (looksLikeText(points)) {
      return { type: "text", handwriting: hr, consumedElements: [element] };
    }
  }

  if (shapeValid) {
    return { type: "shape", shape: sr!, consumedElements: [element] };
  }

  if (hr) {
    return { type: "text", handwriting: hr, consumedElements: [element] };
  }

  return noResult;
}

/**
 * Process a batch of freedraw elements drawn in quick succession.
 *
 * For multi-stroke input:
 * - Run handwriting recognition (multi-stroke text is common)
 * - Run shape recognition on combined points
 * - Pick the best result
 */
export async function processFreedrawBatch(
  elements: ExcalidrawFreeDrawElement[],
  shapeRecognitionEnabled: boolean,
  handwritingRecognitionEnabled: boolean,
): Promise<AIRecognitionResult> {
  const noResult: AIRecognitionResult = { type: "none" };

  if (elements.length === 0) {
    return noResult;
  }

  // Single element — delegate
  if (elements.length === 1) {
    return processFreedrawElement(
      elements[0],
      shapeRecognitionEnabled,
      handwritingRecognitionEnabled,
    );
  }

  // --- Run both in parallel for multi-stroke ---
  let shapeResult: RecognizedShape | null = null;
  let hwResult: HandwritingResult | null = null;

  // Shape: combine all strokes into one path
  const shapePromise = shapeRecognitionEnabled
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
  const hwPromise = handwritingRecognitionEnabled
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
          const result = await recognizeHandwritingMultiStroke(allStrokes, avgSW);
          if (
            result.text.length > 0 &&
            result.confidence >= HANDWRITING_CONFIDENCE_THRESHOLD
          ) {
            hwResult = result;
          }
        }
      })()
    : Promise.resolve();

  await Promise.all([shapePromise, hwPromise]);

  const sr = shapeResult as RecognizedShape | null;
  const hr = hwResult as HandwritingResult | null;

  // Multi-stroke: prefer handwriting over shape (text is usually multi-stroke)
  if (hr && sr) {
    // Only let shape win if it's very confident (>90%)
    if (sr.confidence >= 0.90) {
      return { type: "shape", shape: sr, consumedElements: elements };
    }
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
  terminateHandwritingEngine,
  type RecognizedShape,
  type RecognizedShapeType,
  type HandwritingResult,
};
