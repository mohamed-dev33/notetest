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
 *
 * Strategy:
 * 1. Combine all strokes' points into one path and try shape recognition
 *    (handles multi-stroke shapes like arrow = line + head)
 * 2. Try multi-stroke handwriting recognition
 * 3. Fall back to single-element shape recognition on each element
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

  // --- Multi-stroke shape recognition ---
  // Combine all strokes into a single point sequence (preserving global coords)
  // so Protractor can recognize multi-stroke shapes (e.g. arrow = line + head)
  if (shapeRecognitionEnabled) {
    const combinedPoints: LocalPoint[] = [];
    let globalMinX = Infinity, globalMinY = Infinity;

    // Compute global min to use as reference origin
    for (const el of elements) {
      for (const p of el.points as readonly LocalPoint[]) {
        globalMinX = Math.min(globalMinX, el.x + p[0]);
        globalMinY = Math.min(globalMinY, el.y + p[1]);
      }
    }

    // Combine all points into one sequence, translated to local coords
    for (const el of elements) {
      for (const p of el.points as readonly LocalPoint[]) {
        combinedPoints.push(
          [p[0] + el.x - globalMinX, p[1] + el.y - globalMinY] as LocalPoint,
        );
      }
    }

    if (combinedPoints.length >= 5) {
      const shapeResult = recognizeShape(combinedPoints, globalMinX, globalMinY);
      if (
        shapeResult.type !== "freedraw" &&
        shapeResult.confidence >= SHAPE_CONFIDENCE_THRESHOLD
      ) {
        return {
          type: "shape",
          shape: shapeResult,
          consumedElements: elements,
        };
      }
    }
  }

  // --- Multi-stroke handwriting recognition ---
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

  // --- Fallback: try each element individually for shape ---
  if (shapeRecognitionEnabled) {
    for (const el of elements) {
      const points = el.points as readonly LocalPoint[];
      if (points.length >= 5) {
        const shapeResult = recognizeShape(points, el.x, el.y);
        if (
          shapeResult.type !== "freedraw" &&
          shapeResult.confidence >= SHAPE_CONFIDENCE_THRESHOLD
        ) {
          return {
            type: "shape",
            shape: shapeResult,
            consumedElements: [el],
          };
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
