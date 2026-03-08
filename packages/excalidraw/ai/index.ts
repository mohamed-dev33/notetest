/**
 * AI Recognition Manager
 *
 * Orchestrates shape and handwriting recognition for freedraw elements.
 * Called after a freedraw element is finalized to optionally replace it
 * with a recognized shape or text element.
 *
 * Supports batch processing: multiple strokes drawn within a time window
 * are collected and recognized together (critical for multi-stroke
 * handwriting and complex shapes).
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
  // Which freedraw elements were consumed (for batch mode)
  consumedElements?: ExcalidrawFreeDrawElement[];
}

// Confidence thresholds
const SHAPE_CONFIDENCE_THRESHOLD = 0.45;
const HANDWRITING_CONFIDENCE_THRESHOLD = 35;

/**
 * Process a single freedraw element.
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

  // Try shape recognition first (synchronous and fast)
  let shapeResult: RecognizedShape | null = null;
  if (shapeRecognitionEnabled) {
    shapeResult = recognizeShape(points, element.x, element.y);
  }

  const shapeMatched =
    shapeResult &&
    shapeResult.type !== "freedraw" &&
    shapeResult.confidence >= SHAPE_CONFIDENCE_THRESHOLD;

  const isClosedShape =
    shapeMatched &&
    shapeResult!.type !== "line" &&
    shapeResult!.type !== "arrow";

  if (isClosedShape) {
    return {
      type: "shape",
      shape: shapeResult!,
      consumedElements: [element],
    };
  }

  // For open strokes, try handwriting first
  if (handwritingRecognitionEnabled) {
    const textLikely = looksLikeText(points);
    if (textLikely || !isClosedShape) {
      const hwResult = await recognizeHandwriting(
        points,
        element.strokeWidth,
        true,
      );

      if (
        hwResult.text.length > 0 &&
        hwResult.confidence >= HANDWRITING_CONFIDENCE_THRESHOLD
      ) {
        return {
          type: "text",
          handwriting: hwResult,
          consumedElements: [element],
        };
      }
    }
  }

  // Fall back to shape result (line/arrow)
  if (shapeMatched) {
    return {
      type: "shape",
      shape: shapeResult!,
      consumedElements: [element],
    };
  }

  return noResult;
}

/**
 * Process a batch of freedraw elements drawn in quick succession.
 * Tries multi-stroke handwriting recognition first, then falls back
 * to individual shape recognition.
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

  // Single element — delegate to single-element processor
  if (elements.length === 1) {
    return processFreedrawElement(
      elements[0],
      shapeRecognitionEnabled,
      handwritingRecognitionEnabled,
    );
  }

  // Multiple strokes — likely handwriting (multi-stroke letters/words)
  // Try combined handwriting recognition first
  if (handwritingRecognitionEnabled) {
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
      const avgStrokeWidth = elements.reduce((s, e) => s + e.strokeWidth, 0) / elements.length;
      const hwResult = await recognizeHandwritingMultiStroke(
        allStrokes,
        avgStrokeWidth,
      );

      if (
        hwResult.text.length > 0 &&
        hwResult.confidence >= HANDWRITING_CONFIDENCE_THRESHOLD
      ) {
        return {
          type: "text",
          handwriting: hwResult,
          consumedElements: elements,
        };
      }
    }
  }

  // Multi-stroke didn't match as text — try individual shape recognition
  // on the last element (most likely the completed shape)
  if (shapeRecognitionEnabled) {
    const lastEl = elements[elements.length - 1];
    const points = lastEl.points as readonly LocalPoint[];
    if (points.length >= 5) {
      const shapeResult = recognizeShape(points, lastEl.x, lastEl.y);
      if (
        shapeResult.type !== "freedraw" &&
        shapeResult.confidence >= SHAPE_CONFIDENCE_THRESHOLD
      ) {
        return {
          type: "shape",
          shape: shapeResult,
          consumedElements: [lastEl],
        };
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
