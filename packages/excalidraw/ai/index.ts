/**
 * AI Recognition Manager
 *
 * Orchestrates shape recognition for freedraw elements.
 * Called after a freedraw element is finalized to optionally replace it
 * with a recognized shape element.
 *
 * Strategy:
 * - Single stroke: shape recognition runs, returns top-N candidates.
 * - Multi-stroke: first try each stroke individually. Only combine if
 *   strokes are spatially connected AND individual recognition fails.
 *   This prevents two separate lines from being recognized as an arrow.
 */

import type { ExcalidrawFreeDrawElement } from "@excalidraw/element/types";
import type { LocalPoint } from "@excalidraw/math";

import {
  recognizeShape,
  recognizeShapeTopN,
  type RecognizedShape,
  type RecognizedShapeType,
} from "./shapeRecognition";

export interface AIRecognitionResult {
  type: "shape" | "none";
  shape?: RecognizedShape;
  /** Alternative shape suggestions (for non-line/arrow shapes) */
  alternatives?: RecognizedShape[];
  consumedElements?: ExcalidrawFreeDrawElement[];
}

// Higher thresholds = fewer false positives
const SHAPE_CONFIDENCE_THRESHOLD = 0.68;

/**
 * Process a single freedraw element for shape recognition.
 * Returns the best match plus alternatives for closed shapes.
 */
export async function processFreedrawElement(
  element: ExcalidrawFreeDrawElement,
  shapeRecognitionEnabled: boolean,
): Promise<AIRecognitionResult> {
  const noResult: AIRecognitionResult = { type: "none" };

  if (!shapeRecognitionEnabled) {
    return noResult;
  }

  const points = element.points as readonly LocalPoint[];

  if (points.length < 5) {
    return noResult;
  }

  const shapeResult = recognizeShape(points, element.x, element.y);
  if (shapeResult.type === "freedraw" || shapeResult.confidence < SHAPE_CONFIDENCE_THRESHOLD) {
    return noResult;
  }

  // For lines/arrows, no alternatives needed
  if (shapeResult.type === "line" || shapeResult.type === "arrow") {
    return { type: "shape", shape: shapeResult, consumedElements: [element] };
  }

  // For closed shapes, get top-N alternatives
  const topN = recognizeShapeTopN(points, element.x, element.y, 3);
  const alternatives = topN.length > 1 ? topN.slice(1) : undefined;

  return {
    type: "shape",
    shape: shapeResult,
    alternatives,
    consumedElements: [element],
  };
}

/**
 * Check if multi-stroke input forms a single connected shape.
 * Requires strokes to have endpoints within threshold distance.
 */
function checkStrokesFormOneShape(
  elements: ExcalidrawFreeDrawElement[],
): boolean {
  if (elements.length < 2 || elements.length > 5) { return false; }

  const strokeEnds = elements.map((el) => {
    const pts = el.points as readonly LocalPoint[];
    return {
      start: { x: el.x + pts[0][0], y: el.y + pts[0][1] },
      end: { x: el.x + pts[pts.length - 1][0], y: el.y + pts[pts.length - 1][1] },
    };
  });

  let connectionCount = 0;
  const threshold = 30;

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

  return connectionCount > 0;
}

/**
 * Check if strokes overlap spatially (bounding boxes intersect).
 * Used to decide if strokes are part of one shape vs separate drawings.
 */
function strokesOverlap(elements: ExcalidrawFreeDrawElement[]): boolean {
  if (elements.length < 2) { return false; }

  const bboxes = elements.map((el) => {
    const pts = el.points as readonly LocalPoint[];
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of pts) {
      const ax = el.x + p[0];
      const ay = el.y + p[1];
      if (ax < minX) { minX = ax; }
      if (ay < minY) { minY = ay; }
      if (ax > maxX) { maxX = ax; }
      if (ay > maxY) { maxY = ay; }
    }
    // Pad bbox by 20px to catch nearby strokes
    return { minX: minX - 20, minY: minY - 20, maxX: maxX + 20, maxY: maxY + 20 };
  });

  // Check if all bboxes overlap with at least one other
  for (let i = 0; i < bboxes.length; i++) {
    let overlapsAny = false;
    for (let j = 0; j < bboxes.length; j++) {
      if (i === j) { continue; }
      if (
        bboxes[i].minX <= bboxes[j].maxX &&
        bboxes[i].maxX >= bboxes[j].minX &&
        bboxes[i].minY <= bboxes[j].maxY &&
        bboxes[i].maxY >= bboxes[j].minY
      ) {
        overlapsAny = true;
        break;
      }
    }
    if (!overlapsAny) { return false; }
  }
  return true;
}

/**
 * Process a batch of freedraw elements drawn in quick succession.
 *
 * Returns ALL individual recognition results (one per stroke that was
 * successfully recognized). This ensures that drawing an arrow and a line
 * quickly results in BOTH being recognized, not just the best one.
 *
 * Multi-stroke strategy:
 * 1. Try each stroke individually — collect ALL successful recognitions.
 * 2. For remaining unrecognized strokes, try combining spatially connected ones.
 */
export async function processFreedrawBatch(
  elements: ExcalidrawFreeDrawElement[],
  shapeRecognitionEnabled: boolean,
): Promise<AIRecognitionResult[]> {
  if (elements.length === 0) {
    return [];
  }

  if (!shapeRecognitionEnabled) {
    return [];
  }

  if (elements.length === 1) {
    const r = await processFreedrawElement(elements[0], shapeRecognitionEnabled);
    return r.type === "none" ? [] : [r];
  }

  const results: AIRecognitionResult[] = [];
  const unrecognized: ExcalidrawFreeDrawElement[] = [];

  // Phase 1: Try each element individually — collect ALL results
  for (const el of elements) {
    const points = el.points as readonly LocalPoint[];
    if (points.length < 5) {
      unrecognized.push(el);
      continue;
    }
    const shape = recognizeShape(points, el.x, el.y);
    if (shape.type !== "freedraw" && shape.confidence >= SHAPE_CONFIDENCE_THRESHOLD) {
      let alternatives: RecognizedShape[] | undefined;
      if (shape.type !== "line" && shape.type !== "arrow") {
        const topN = recognizeShapeTopN(points, el.x, el.y, 3);
        alternatives = topN.length > 1 ? topN.slice(1) : undefined;
      }
      results.push({ type: "shape", shape, alternatives, consumedElements: [el] });
    } else {
      unrecognized.push(el);
    }
  }

  // Phase 2: For unrecognized strokes, try combining IF spatially connected
  if (unrecognized.length >= 2) {
    const areConnected = checkStrokesFormOneShape(unrecognized);
    const doOverlap = strokesOverlap(unrecognized);

    if (areConnected || doOverlap) {
      const combinedPoints: LocalPoint[] = [];
      let globalMinX = Infinity, globalMinY = Infinity;

      for (const el of unrecognized) {
        for (const p of el.points as readonly LocalPoint[]) {
          globalMinX = Math.min(globalMinX, el.x + p[0]);
          globalMinY = Math.min(globalMinY, el.y + p[1]);
        }
      }

      for (const el of unrecognized) {
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
          let alternatives: RecognizedShape[] | undefined;
          if (result.type !== "line" && result.type !== "arrow") {
            const topN = recognizeShapeTopN(combinedPoints, globalMinX, globalMinY, 3);
            alternatives = topN.length > 1 ? topN.slice(1) : undefined;
          }
          results.push({ type: "shape", shape: result, alternatives, consumedElements: unrecognized });
        }
      }
    }
  }

  return results;
}

export {
  type RecognizedShape,
  type RecognizedShapeType,
};
