/**
 * Shape Recognition Module — Protractor Algorithm
 *
 * Implements the Protractor gesture recognizer (Li, 2010) which achieves
 * 99% accuracy for single-stroke shape recognition. Uses cosine distance
 * with optimal angular alignment — no iterative search needed.
 *
 * Also includes geometric heuristics for line/arrow detection where
 * template matching is less appropriate.
 */

import { type LocalPoint } from "@excalidraw/math";

export type RecognizedShapeType =
  | "line"
  | "rectangle"
  | "ellipse"
  | "triangle"
  | "diamond"
  | "arrow"
  | "freedraw";

export interface RecognizedShape {
  type: RecognizedShapeType;
  confidence: number; // 0-1
  bounds: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  startPoint?: { x: number; y: number };
  endPoint?: { x: number; y: number };
  corners?: { x: number; y: number }[];
}

// =====================================================================
// Protractor Algorithm Core (Li, 2010)
// =====================================================================

interface Point {
  x: number;
  y: number;
}

const NUM_POINTS = 64; // resample to this many points
const ORIGIN: Point = { x: 0, y: 0 };

/** Euclidean distance between two points */
function dist(a: Point, b: Point): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

/** Total path length of a polyline */
function pathLength(pts: Point[]): number {
  let d = 0;
  for (let i = 1; i < pts.length; i++) {
    d += dist(pts[i - 1], pts[i]);
  }
  return d;
}

/** Resample the point path to N equally-spaced points */
function resample(points: Point[], n: number): Point[] {
  const interval = pathLength(points) / (n - 1);
  let D = 0;
  const newPts: Point[] = [{ ...points[0] }];

  for (let i = 1; i < points.length; i++) {
    const d = dist(points[i - 1], points[i]);
    if (D + d >= interval) {
      const qx =
        points[i - 1].x + ((interval - D) / d) * (points[i].x - points[i - 1].x);
      const qy =
        points[i - 1].y + ((interval - D) / d) * (points[i].y - points[i - 1].y);
      const q: Point = { x: qx, y: qy };
      newPts.push(q);
      points.splice(i, 0, q);
      D = 0;
    } else {
      D += d;
    }
  }

  // Edge case: rounding may leave us one short
  while (newPts.length < n) {
    newPts.push({ ...points[points.length - 1] });
  }

  return newPts.slice(0, n);
}

/** Compute centroid of points */
function centroid(pts: Point[]): Point {
  let cx = 0,
    cy = 0;
  for (const p of pts) {
    cx += p.x;
    cy += p.y;
  }
  return { x: cx / pts.length, y: cy / pts.length };
}

/** Translate so centroid is at origin */
function translateTo(pts: Point[], target: Point): Point[] {
  const c = centroid(pts);
  return pts.map((p) => ({ x: p.x + target.x - c.x, y: p.y + target.y - c.y }));
}

/** Compute indicative angle (angle from centroid to first point) */
function indicativeAngle(pts: Point[]): number {
  const c = centroid(pts);
  return Math.atan2(c.y - pts[0].y, c.x - pts[0].x);
}

/** Rotate points by angle around centroid */
function rotateBy(pts: Point[], radians: number): Point[] {
  const c = centroid(pts);
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  return pts.map((p) => {
    const dx = p.x - c.x;
    const dy = p.y - c.y;
    return { x: dx * cos - dy * sin + c.x, y: dx * sin + dy * cos + c.y };
  });
}

/**
 * Vectorize: create a 2N-element vector from resampled, rotated,
 * translated points. This is the Protractor representation.
 */
function vectorize(pts: Point[], orientationSensitive: boolean): number[] {
  const iAngle = indicativeAngle(pts);
  let delta = 0;

  if (orientationSensitive) {
    // Snap to nearest base orientation
    const baseOrientation = (Math.PI / 4) * Math.round(iAngle / (Math.PI / 4));
    delta = baseOrientation - iAngle;
  } else {
    delta = -iAngle;
  }

  let sum = 0;
  const vec: number[] = new Array(pts.length * 2);

  for (let i = 0; i < pts.length; i++) {
    const dx = pts[i].x - centroid(pts).x;
    const dy = pts[i].y - centroid(pts).y;
    const cos = Math.cos(delta);
    const sin = Math.sin(delta);
    vec[i * 2] = dx * cos - dy * sin;
    vec[i * 2 + 1] = dx * sin + dy * cos;
    sum += vec[i * 2] ** 2 + vec[i * 2 + 1] ** 2;
  }

  // Normalize
  const magnitude = Math.sqrt(sum);
  for (let i = 0; i < vec.length; i++) {
    vec[i] /= magnitude;
  }

  return vec;
}

/**
 * Protractor's optimal cosine distance.
 * Returns a score between 0 and 1 (1 = perfect match).
 */
function optimalCosineDistance(v1: number[], v2: number[]): number {
  let a = 0,
    b = 0;
  for (let i = 0; i < v1.length; i += 2) {
    a += v1[i] * v2[i] + v1[i + 1] * v2[i + 1];
    b += v1[i] * v2[i + 1] - v1[i + 1] * v2[i];
  }
  const angle = Math.atan(b / a);
  const score = a * Math.cos(angle) + b * Math.sin(angle);
  return Math.max(0, score); // clamp to [0,1]
}

// =====================================================================
// Template Generation
// =====================================================================

interface GestureTemplate {
  name: RecognizedShapeType;
  vector: number[];
}

/** Generate points along a rectangle path */
function makeRectangle(w: number, h: number): Point[] {
  const pts: Point[] = [];
  const steps = 16;
  // Top edge
  for (let i = 0; i <= steps; i++) { pts.push({ x: (i / steps) * w, y: 0 }); }
  // Right edge
  for (let i = 1; i <= steps; i++) { pts.push({ x: w, y: (i / steps) * h }); }
  // Bottom edge
  for (let i = 1; i <= steps; i++) { pts.push({ x: w - (i / steps) * w, y: h }); }
  // Left edge
  for (let i = 1; i <= steps; i++) { pts.push({ x: 0, y: h - (i / steps) * h }); }
  return pts;
}

/** Generate points along a circle path */
function makeCircle(r: number): Point[] {
  const pts: Point[] = [];
  const steps = 64;
  for (let i = 0; i <= steps; i++) {
    const angle = (i / steps) * Math.PI * 2;
    pts.push({ x: r + r * Math.cos(angle), y: r + r * Math.sin(angle) });
  }
  return pts;
}

/** Generate points along an ellipse path */
function makeEllipse(rx: number, ry: number): Point[] {
  const pts: Point[] = [];
  const steps = 64;
  for (let i = 0; i <= steps; i++) {
    const angle = (i / steps) * Math.PI * 2;
    pts.push({ x: rx + rx * Math.cos(angle), y: ry + ry * Math.sin(angle) });
  }
  return pts;
}

/** Generate points along a triangle path */
function makeTriangle(w: number, h: number): Point[] {
  const pts: Point[] = [];
  const steps = 21;
  const top: Point = { x: w / 2, y: 0 };
  const bl: Point = { x: 0, y: h };
  const br: Point = { x: w, y: h };
  // Top to bottom-right
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    pts.push({ x: top.x + (br.x - top.x) * t, y: top.y + (br.y - top.y) * t });
  }
  // Bottom-right to bottom-left
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    pts.push({ x: br.x + (bl.x - br.x) * t, y: br.y + (bl.y - br.y) * t });
  }
  // Bottom-left to top
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    pts.push({ x: bl.x + (top.x - bl.x) * t, y: bl.y + (top.y - bl.y) * t });
  }
  return pts;
}

/** Generate triangle starting from bottom-left vertex */
function makeTriangleBL(w: number, h: number): Point[] {
  const pts: Point[] = [];
  const steps = 21;
  const top: Point = { x: w / 2, y: 0 };
  const bl: Point = { x: 0, y: h };
  const br: Point = { x: w, y: h };
  for (let i = 0; i <= steps; i++) { const t = i/steps; pts.push({ x: bl.x+(br.x-bl.x)*t, y: bl.y+(br.y-bl.y)*t }); }
  for (let i = 1; i <= steps; i++) { const t = i/steps; pts.push({ x: br.x+(top.x-br.x)*t, y: br.y+(top.y-br.y)*t }); }
  for (let i = 1; i <= steps; i++) { const t = i/steps; pts.push({ x: top.x+(bl.x-top.x)*t, y: top.y+(bl.y-top.y)*t }); }
  return pts;
}

/** Right-pointing triangle */
function makeTriangleRight(w: number, h: number): Point[] {
  const pts: Point[] = [];
  const steps = 21;
  const right: Point = { x: w, y: h / 2 };
  const tl: Point = { x: 0, y: 0 };
  const bl: Point = { x: 0, y: h };
  for (let i = 0; i <= steps; i++) { const t = i/steps; pts.push({ x: tl.x+(right.x-tl.x)*t, y: tl.y+(right.y-tl.y)*t }); }
  for (let i = 1; i <= steps; i++) { const t = i/steps; pts.push({ x: right.x+(bl.x-right.x)*t, y: right.y+(bl.y-right.y)*t }); }
  for (let i = 1; i <= steps; i++) { const t = i/steps; pts.push({ x: bl.x+(tl.x-bl.x)*t, y: bl.y+(tl.y-bl.y)*t }); }
  return pts;
}

/** Generate points along a diamond path */
function makeDiamond(w: number, h: number): Point[] {
  const pts: Point[] = [];
  const steps = 16;
  const top: Point = { x: w / 2, y: 0 };
  const right: Point = { x: w, y: h / 2 };
  const bottom: Point = { x: w / 2, y: h };
  const left: Point = { x: 0, y: h / 2 };
  const edges = [
    [top, right],
    [right, bottom],
    [bottom, left],
    [left, top],
  ];
  for (const [from, to] of edges) {
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      pts.push({ x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t });
    }
  }
  return pts;
}

/** Generate points along a line path */
function makeLine(len: number, angle: number): Point[] {
  const pts: Point[] = [];
  const steps = 32;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    pts.push({ x: t * len * Math.cos(angle), y: t * len * Math.sin(angle) });
  }
  return pts;
}

/** Generate points along an arrow path (line + V head) */
function makeArrow(len: number, angle: number): Point[] {
  const pts: Point[] = [];
  const steps = 24;
  const headLen = len * 0.25;
  const headAngle = Math.PI / 6;

  // Shaft
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    pts.push({ x: t * len * Math.cos(angle), y: t * len * Math.sin(angle) });
  }

  const tip = pts[pts.length - 1];

  // Head: tip → left barb
  for (let i = 1; i <= 8; i++) {
    const t = i / 8;
    pts.push({
      x: tip.x - headLen * t * Math.cos(angle - headAngle),
      y: tip.y - headLen * t * Math.sin(angle - headAngle),
    });
  }
  // Back to tip
  for (let i = 1; i <= 4; i++) {
    const t = i / 4;
    const last = pts[pts.length - 1];
    pts.push({
      x: last.x + (tip.x - last.x) * t,
      y: last.y + (tip.y - last.y) * t,
    });
  }
  // Head: tip → right barb
  for (let i = 1; i <= 8; i++) {
    const t = i / 8;
    pts.push({
      x: tip.x - headLen * t * Math.cos(angle + headAngle),
      y: tip.y - headLen * t * Math.sin(angle + headAngle),
    });
  }

  return pts;
}

function buildVector(pts: Point[], orientationSensitive: boolean): number[] {
  const resampled = resample([...pts], NUM_POINTS);
  const translated = translateTo(resampled, ORIGIN);
  return vectorize(translated, orientationSensitive);
}

/** Create all shape templates with multiple variants */
function buildTemplates(): GestureTemplate[] {
  const templates: GestureTemplate[] = [];
  const add = (name: RecognizedShapeType, pts: Point[], orientSensitive = false) => {
    templates.push({ name, vector: buildVector(pts, orientSensitive) });
  };

  // Rectangles — various aspect ratios, CW and CCW
  for (const [w, h] of [[100, 100], [150, 80], [80, 150], [200, 100], [100, 200], [200, 60], [60, 200]]) {
    add("rectangle", makeRectangle(w, h));
    add("rectangle", makeRectangle(w, h).reverse()); // CCW
  }

  // Circles and Ellipses — CW and CCW
  for (const r of [50, 80, 120]) {
    add("ellipse", makeCircle(r));
    add("ellipse", makeCircle(r).reverse());
  }
  for (const [rx, ry] of [[80, 50], [50, 80], [120, 60], [60, 120]]) {
    add("ellipse", makeEllipse(rx, ry));
    add("ellipse", makeEllipse(rx, ry).reverse());
  }

  // Triangles — many variants, start vertices, orientations
  for (const [w, h] of [[100, 100], [120, 80], [80, 120], [150, 100], [200, 180], [100, 180], [180, 100]]) {
    add("triangle", makeTriangle(w, h));
    add("triangle", makeTriangle(w, h).reverse());
    add("triangle", makeTriangleBL(w, h));
    add("triangle", makeTriangleBL(w, h).reverse());
    add("triangle", makeTriangleRight(w, h));
    add("triangle", makeTriangleRight(w, h).reverse());
  }

  // Diamonds
  for (const [w, h] of [[100, 100], [80, 120], [120, 80], [100, 150]]) {
    add("diamond", makeDiamond(w, h));
    add("diamond", makeDiamond(w, h).reverse());
  }

  // Lines — various angles (orientation-sensitive to distinguish from arrows)
  for (const angle of [0, Math.PI / 6, Math.PI / 4, Math.PI / 3, Math.PI / 2,
    -Math.PI / 6, -Math.PI / 4, -Math.PI / 3]) {
    for (const len of [100, 200]) {
      add("line", makeLine(len, angle), true);
    }
  }

  // Arrows — various angles
  for (const angle of [0, Math.PI / 6, Math.PI / 4, -Math.PI / 6, -Math.PI / 4, Math.PI / 2]) {
    for (const len of [120, 200]) {
      add("arrow", makeArrow(len, angle), true);
    }
  }

  return templates;
}

// Lazy-init templates (built once on first use)
let _templates: GestureTemplate[] | null = null;
function getTemplates(): GestureTemplate[] {
  if (!_templates) {
    _templates = buildTemplates();
  }
  return _templates;
}

// =====================================================================
// Geometric helpers for computing bounds, closedness, etc.
// =====================================================================

function getBounds(points: Point[]): {
  x: number; y: number; width: number; height: number;
} {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) { minX = p.x; }
    if (p.y < minY) { minY = p.y; }
    if (p.x > maxX) { maxX = p.x; }
    if (p.y > maxY) { maxY = p.y; }
  }
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

function isClosed(points: Point[]): boolean {
  if (points.length < 3) { return false; }
  const d = dist(points[0], points[points.length - 1]);
  const len = pathLength(points);
  return d < len * 0.15;
}

/** Shoelace area of a polygon */
function shoelaceArea(pts: Point[]): number {
  let area = 0;
  for (let i = 0; i < pts.length; i++) {
    const j = (i + 1) % pts.length;
    area += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
  }
  return Math.abs(area) / 2;
}

// =====================================================================
// Main Recognition Function
// =====================================================================

const MIN_POINTS = 5;
const MIN_STROKE_SIZE = 20;

/**
 * Recognize a shape from freedraw points using the Protractor algorithm
 * combined with geometric post-validation.
 */
export function recognizeShape(
  rawPoints: readonly LocalPoint[],
  elementX: number,
  elementY: number,
): RecognizedShape {
  const fallback: RecognizedShape = {
    type: "freedraw",
    confidence: 0,
    bounds: { x: elementX, y: elementY, width: 0, height: 0 },
  };

  if (rawPoints.length < MIN_POINTS) {
    return fallback;
  }

  const points: Point[] = rawPoints.map((p) => ({ x: p[0], y: p[1] }));
  const bounds = getBounds(points);

  if (bounds.width < MIN_STROKE_SIZE && bounds.height < MIN_STROKE_SIZE) {
    return fallback;
  }

  const closed = isClosed(points);
  const totalLen = pathLength(points);

  // --- Quick line/arrow check via straightness ratio ---
  // Lines and arrows are better detected geometrically since their
  // Protractor templates are orientation-sensitive
  if (!closed) {
    const directDist = dist(points[0], points[points.length - 1]);
    const straightness = directDist / totalLen;

    if (straightness > 0.88) {
      // Very straight stroke → line
      const result: RecognizedShape = {
        type: "line",
        confidence: straightness,
        bounds: { x: elementX + bounds.x, y: elementY + bounds.y, width: bounds.width, height: bounds.height },
        startPoint: { x: elementX + points[0].x, y: elementY + points[0].y },
        endPoint: { x: elementX + points[points.length - 1].x, y: elementY + points[points.length - 1].y },
      };
      return result;
    }

    // Arrow detection: mostly straight shaft with a sharp turn near the end
    if (straightness > 0.55 && points.length >= 8) {
      const shaftEnd = Math.floor(points.length * 0.7);
      const shaftStraightness = dist(points[0], points[shaftEnd]) / pathLength(points.slice(0, shaftEnd + 1));

      if (shaftStraightness > 0.85) {
        // Check for direction change in last 30%
        const tailPts = points.slice(shaftEnd);
        if (tailPts.length >= 3) {
          const tailDirect = dist(tailPts[0], tailPts[tailPts.length - 1]);
          const tailLen = pathLength(tailPts);
          const tailStraightness = tailLen > 0 ? tailDirect / tailLen : 1;

          if (tailStraightness < 0.7) {
            return {
              type: "arrow",
              confidence: shaftStraightness * 0.95,
              bounds: { x: elementX + bounds.x, y: elementY + bounds.y, width: bounds.width, height: bounds.height },
              startPoint: { x: elementX + points[0].x, y: elementY + points[0].y },
              endPoint: { x: elementX + points[shaftEnd].x, y: elementY + points[shaftEnd].y },
            };
          }
        }
      }
    }
  }

  // --- Protractor matching for closed shapes ---
  const templates = getTemplates();

  // Build vector from input
  const inputVector = buildVector([...points], false);

  // Find best matching template
  let bestScore = -Infinity;
  let bestName: RecognizedShapeType = "freedraw";

  // Accumulate scores per shape type for voting
  const scoreMap: Record<string, number[]> = {};

  for (const tmpl of templates) {
    const score = optimalCosineDistance(inputVector, tmpl.vector);
    if (!scoreMap[tmpl.name]) {
      scoreMap[tmpl.name] = [];
    }
    scoreMap[tmpl.name].push(score);

    if (score > bestScore) {
      bestScore = score;
      bestName = tmpl.name;
    }
  }

  // Compute average of top-3 scores per shape type for robustness
  const avgScores: Record<string, number> = {};
  for (const [name, scores] of Object.entries(scoreMap)) {
    const sorted = scores.sort((a, b) => b - a);
    const top3 = sorted.slice(0, 3);
    avgScores[name] = top3.reduce((s, v) => s + v, 0) / top3.length;
  }

  // Pick the shape with highest average top-3 score
  let finalName: RecognizedShapeType = "freedraw";
  let finalScore = 0;
  for (const [name, avg] of Object.entries(avgScores)) {
    if (avg > finalScore) {
      finalScore = avg;
      finalName = name as RecognizedShapeType;
    }
  }

  // --- Geometric post-validation ---
  // Apply sanity checks to prevent misclassification
  if (closed) {
    const area = shoelaceArea(points);
    const bbArea = bounds.width * bounds.height;
    const fillRatio = bbArea > 0 ? area / bbArea : 0;
    const bbPerimeter = 2 * (bounds.width + bounds.height);
    const perimRatio = totalLen / bbPerimeter;

    // Use fill ratio to disambiguate shapes:
    // Rectangle: ~0.85-1.0, Ellipse: ~0.78, Diamond: ~0.5-0.6, Triangle: ~0.4-0.55

    // Rectangle validation: must fill its bounding box well
    if (finalName === "rectangle") {
      if (fillRatio < 0.65) {
        // Too hollow for a rectangle — use fill ratio to pick correct shape
        if (fillRatio < 0.56) {
          finalName = "triangle";
        } else {
          finalName = "diamond";
        }
      }
    }

    // Ellipse validation
    if (finalName === "ellipse") {
      if (fillRatio > 0.9 && perimRatio < 1.15) {
        // Too boxy for an ellipse — probably a rectangle
        finalName = "rectangle";
        finalScore = Math.max(finalScore, 0.85);
      }
    }

    // Triangle validation: should have low fill ratio
    if (finalName === "triangle") {
      if (fillRatio > 0.8) {
        // Too full for triangle → rectangle
        finalName = "rectangle";
      }
    }

    // Diamond validation
    if (finalName === "diamond") {
      if (fillRatio > 0.85) {
        // Too full for diamond → rectangle
        finalName = "rectangle";
      } else if (fillRatio < 0.35) {
        // Too hollow for diamond → maybe triangle
        finalName = "triangle";
      }
    }

    // If Protractor is unsure (low score), use fill ratio as primary signal
    if (finalScore < 0.6) {
      if (fillRatio > 0.82) {
        finalName = "rectangle";
      } else if (fillRatio > 0.7) {
        finalName = "ellipse";
      } else if (fillRatio > 0.55) {
        finalName = "diamond";
      } else if (fillRatio > 0.3) {
        finalName = "triangle";
      }
      finalScore = Math.max(finalScore, 0.5);
    }
  } else {
    // Open strokes should not be closed shapes
    if (finalName === "rectangle" || finalName === "ellipse" || finalName === "triangle" || finalName === "diamond") {
      // Check if it's actually close enough to closed
      const closeDist = dist(points[0], points[points.length - 1]);
      if (closeDist > totalLen * 0.2) {
        // Not closed enough for a closed shape
        finalName = "line";
        finalScore = dist(points[0], points[points.length - 1]) / totalLen;
      }
    }
  }

  // Minimum confidence threshold
  if (finalScore < 0.45) {
    return fallback;
  }

  const result: RecognizedShape = {
    type: finalName,
    confidence: Math.min(1, finalScore),
    bounds: {
      x: elementX + bounds.x,
      y: elementY + bounds.y,
      width: bounds.width,
      height: bounds.height,
    },
  };

  // Add start/end points for lines and arrows
  if (finalName === "line" || finalName === "arrow") {
    result.startPoint = { x: elementX + points[0].x, y: elementY + points[0].y };
    result.endPoint = {
      x: elementX + points[points.length - 1].x,
      y: elementY + points[points.length - 1].y,
    };
  }

  return result;
}
