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
  return d < len * 0.18;
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

/**
 * Circularity metric: 4π × area / perimeter²
 * Perfect circle = 1.0, square ≈ 0.785, triangle ≈ 0.60
 */
function circularity(pts: Point[]): number {
  const area = shoelaceArea(pts);
  const perim = pathLength(pts);
  if (perim === 0) { return 0; }
  return (4 * Math.PI * area) / (perim * perim);
}

/**
 * Detect corners using angle-based method.
 * Returns indices of significant corners in the point array.
 */
function detectCorners(pts: Point[], angleThrDeg: number = 30): number[] {
  if (pts.length < 5) { return []; }

  // Compute angle at each point using a window
  const windowSize = Math.max(3, Math.floor(pts.length / 20));
  const corners: number[] = [];
  const angles: number[] = [];

  for (let i = windowSize; i < pts.length - windowSize; i++) {
    const prev = pts[i - windowSize];
    const curr = pts[i];
    const next = pts[i + windowSize];

    const v1x = prev.x - curr.x;
    const v1y = prev.y - curr.y;
    const v2x = next.x - curr.x;
    const v2y = next.y - curr.y;

    const dot = v1x * v2x + v1y * v2y;
    const mag1 = Math.sqrt(v1x * v1x + v1y * v1y);
    const mag2 = Math.sqrt(v2x * v2x + v2y * v2y);
    if (mag1 < 0.001 || mag2 < 0.001) { continue; }

    const cosAngle = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
    const angleDeg = Math.acos(cosAngle) * 180 / Math.PI;
    angles.push(angleDeg);

    if (angleDeg < (180 - angleThrDeg)) {
      corners.push(i);
    }
  }

  // Remove close-together corners (non-maximum suppression)
  const minDist = pts.length / 8;
  const filtered: number[] = [];
  for (const c of corners) {
    if (filtered.length === 0 || c - filtered[filtered.length - 1] >= minDist) {
      filtered.push(c);
    } else {
      // Keep the sharper corner
      const prevIdx = filtered[filtered.length - 1];
      const prevAngle = angles[prevIdx - Math.max(3, Math.floor(pts.length / 20))] ?? 180;
      const currAngle = angles[c - Math.max(3, Math.floor(pts.length / 20))] ?? 180;
      if (currAngle < prevAngle) {
        filtered[filtered.length - 1] = c;
      }
    }
  }

  return filtered;
}

/**
 * Ramer-Douglas-Peucker simplification — reduces points to key vertices.
 */
function rdpSimplify(pts: Point[], epsilon: number): Point[] {
  if (pts.length <= 2) { return [...pts]; }

  let maxDist = 0;
  let maxIdx = 0;
  const first = pts[0];
  const last = pts[pts.length - 1];

  for (let i = 1; i < pts.length - 1; i++) {
    const d = pointToLineDistance(pts[i], first, last);
    if (d > maxDist) {
      maxDist = d;
      maxIdx = i;
    }
  }

  if (maxDist > epsilon) {
    const left = rdpSimplify(pts.slice(0, maxIdx + 1), epsilon);
    const right = rdpSimplify(pts.slice(maxIdx), epsilon);
    return [...left.slice(0, -1), ...right];
  }

  return [first, last];
}

function pointToLineDistance(p: Point, a: Point, b: Point): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) { return dist(p, a); }
  const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / lenSq));
  return dist(p, { x: a.x + t * dx, y: a.y + t * dy });
}

/**
 * Compute convex hull (Andrew's monotone chain).
 */
function convexHull(pts: Point[]): Point[] {
  const sorted = [...pts].sort((a, b) => a.x - b.x || a.y - b.y);
  if (sorted.length <= 2) { return sorted; }

  const cross = (o: Point, a: Point, b: Point) =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const lower: Point[] = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }

  const upper: Point[] = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], sorted[i]) <= 0) {
      upper.pop();
    }
    upper.push(sorted[i]);
  }

  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

// =====================================================================
// Main Recognition Function — Hybrid Geometric + Protractor
// =====================================================================

const MIN_POINTS = 5;
const MIN_STROKE_SIZE = 15;

/**
 * Recognize a shape from freedraw points using a hybrid approach:
 * 1. Geometric analysis (corners, circularity, straightness) — primary
 * 2. Protractor template matching — secondary confirmation
 * 3. Cross-validation between both for high accuracy
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

  const totalLen = pathLength(points);
  const closeDist = dist(points[0], points[points.length - 1]);
  const closeness = totalLen > 0 ? closeDist / totalLen : 1;
  const effectivelyClosed = closeness < 0.20;

  const makeBounds = () => ({
    x: elementX + bounds.x,
    y: elementY + bounds.y,
    width: bounds.width,
    height: bounds.height,
  });

  // ========== Phase 0: ALWAYS try line/arrow detection first ==========
  // This must run before closed-shape analysis because arrows with V-heads
  // can appear "closed" but should still be recognized as arrows.
  const straightness = closeDist / totalLen;

  if (straightness > 0.85) {
    return {
      type: "line",
      confidence: Math.min(1, straightness * 1.05),
      bounds: makeBounds(),
      startPoint: { x: elementX + points[0].x, y: elementY + points[0].y },
      endPoint: { x: elementX + points[points.length - 1].x, y: elementY + points[points.length - 1].y },
    };
  }

  // Arrow detection — skip for clearly closed shapes (circles, rectangles)
  // For moderately open shapes, check arrow pattern
  // For clearly open shapes, always check
  if (points.length >= 6 && closeness > 0.10) {
    const arrowResult = detectArrow(points, totalLen, elementX, elementY, bounds);
    if (arrowResult) { return arrowResult; }
  }
  // Even for nearly-closed shapes, check elongated arrow pattern (V-head curling back)
  if (points.length >= 6 && closeness <= 0.10) {
    const elongation = Math.max(bounds.width, bounds.height) / Math.max(Math.min(bounds.width, bounds.height), 1);
    if (elongation > 2.0) {
      const arrowResult = detectArrow(points, totalLen, elementX, elementY, bounds);
      if (arrowResult && arrowResult.confidence > 0.6) { return arrowResult; }
    }
  }

  // Moderately straight open stroke → line
  if (!effectivelyClosed && straightness > 0.75) {
    return {
      type: "line",
      confidence: straightness * 0.9,
      bounds: makeBounds(),
      startPoint: { x: elementX + points[0].x, y: elementY + points[0].y },
      endPoint: { x: elementX + points[points.length - 1].x, y: elementY + points[points.length - 1].y },
    };
  }

  // ========== Phase 1: Closed-shape geometric analysis ==========
  // Compute metrics for shapes that are closed or nearly closed
  const analyzePoints = effectivelyClosed ? points : [...points, { ...points[0] }];
  const area = shoelaceArea(analyzePoints);
  const bbArea = bounds.width * bounds.height;
  const fillRatio = bbArea > 0 ? area / bbArea : 0;
  const circ = circularity(analyzePoints);
  const aspectRatio = bounds.width / Math.max(bounds.height, 1);

  // Corner detection via RDP simplification
  const epsilon = Math.max(bounds.width, bounds.height) * 0.04;
  const simplified = rdpSimplify(analyzePoints, epsilon);
  const vertexCount = simplified.length - (dist(simplified[0], simplified[simplified.length - 1]) < epsilon * 2 ? 1 : 0);

  // Convex hull analysis
  const hull = convexHull(analyzePoints);
  const hullArea = shoelaceArea(hull);
  const convexity = hullArea > 0 ? area / hullArea : 0;

  // Corner detection via angle changes
  const corners = detectCorners(analyzePoints, 35);
  const cornerCount = corners.length;

  // Confidence multiplier for nearly-closed (not perfectly closed) strokes
  const closedConf = effectivelyClosed ? 1.0 : 0.85;

  // ---- Ellipse/Circle detection ----
  // High circularity, few/no sharp corners, high convexity
  if (circ > 0.68 && convexity > 0.88 && cornerCount <= 2) {
    return {
      type: "ellipse",
      confidence: Math.min(1, circ * 1.1 * closedConf),
      bounds: makeBounds(),
    };
  }

  // ---- Rectangle detection ----
  // ~4 corners, high fill ratio
  if (
    cornerCount >= 3 &&
    fillRatio > 0.68 &&
    convexity > 0.82 &&
    aspectRatio > 0.2 && aspectRatio < 5.0
  ) {
    return {
      type: "rectangle",
      confidence: Math.min(1, fillRatio * 1.05 * closedConf),
      bounds: makeBounds(),
    };
  }

  // ---- Diamond detection (BEFORE triangle — both have ~4 corners but different fill) ----
  // Diamond has ~4 corners, fill ratio ~0.5 (half of bounding box), near-square aspect
  if (
    cornerCount >= 3 &&
    fillRatio > 0.32 && fillRatio < 0.68 &&
    convexity > 0.82 &&
    aspectRatio > 0.4 && aspectRatio < 2.5
  ) {
    return {
      type: "diamond",
      confidence: Math.min(1, convexity * 0.9 * closedConf),
      bounds: makeBounds(),
    };
  }

  // ---- Triangle detection ----
  // ~3 corners, lower fill ratio (~0.5), NOT elongated (elongated = arrow/line)
  const elongationRatio = Math.max(bounds.width, bounds.height) / Math.max(Math.min(bounds.width, bounds.height), 1);
  if (
    cornerCount >= 2 && cornerCount <= 4 &&
    fillRatio > 0.20 && fillRatio < 0.68 &&
    convexity > 0.75 &&
    aspectRatio > 0.35 && aspectRatio < 2.5 &&
    elongationRatio < 2.5
  ) {
    return {
      type: "triangle",
      confidence: Math.min(1, convexity * 0.9 * closedConf),
      bounds: makeBounds(),
    };
  }

  // ========== Phase 3: Protractor template fallback ==========
  if (effectivelyClosed || closeness < 0.30) {
    const templates = getTemplates();
    const inputVector = buildVector([...points], false);

    const scoreMap: Record<string, number[]> = {};
    for (const tmpl of templates) {
      const score = optimalCosineDistance(inputVector, tmpl.vector);
      if (!scoreMap[tmpl.name]) { scoreMap[tmpl.name] = []; }
      scoreMap[tmpl.name].push(score);
    }

    const avgScores: Record<string, number> = {};
    for (const [name, scores] of Object.entries(scoreMap)) {
      const sorted = scores.sort((a, b) => b - a);
      const top3 = sorted.slice(0, 3);
      avgScores[name] = top3.reduce((s, v) => s + v, 0) / top3.length;
    }

    let protoName: RecognizedShapeType = "freedraw";
    let protoScore = 0;
    for (const [name, avg] of Object.entries(avgScores)) {
      if (avg > protoScore) {
        protoScore = avg;
        protoName = name as RecognizedShapeType;
      }
    }

    // Cross-validate Protractor with geometry
    if (protoScore >= 0.45) {
      let validated = protoName;

      if (protoName === "rectangle" && fillRatio < 0.55) {
        validated = fillRatio < 0.40 ? "triangle" : "diamond";
      } else if (protoName === "ellipse" && circ < 0.55) {
        validated = cornerCount >= 3 ? (fillRatio > 0.65 ? "rectangle" : "diamond") : "ellipse";
      } else if (protoName === "triangle" && fillRatio > 0.75) {
        validated = "rectangle";
      } else if (protoName === "diamond" && fillRatio > 0.78) {
        validated = "rectangle";
      }

      return {
        type: validated,
        confidence: Math.min(1, protoScore * closedConf),
        bounds: makeBounds(),
      };
    }

    // Phase 4: Pure geometric fallback
    if (circ > 0.65 && cornerCount <= 2) {
      return { type: "ellipse", confidence: circ * closedConf, bounds: makeBounds() };
    }
    if (fillRatio > 0.68) {
      return { type: "rectangle", confidence: fillRatio * closedConf, bounds: makeBounds() };
    }
    if (fillRatio > 0.50 && aspectRatio > 0.5 && aspectRatio < 2.0) {
      return { type: "diamond", confidence: fillRatio * closedConf, bounds: makeBounds() };
    }
    if (fillRatio > 0.20 && cornerCount <= 4 && elongationRatio < 2.5) {
      return { type: "triangle", confidence: Math.max(0.5, convexity * 0.8) * closedConf, bounds: makeBounds() };
    }
  }

  return fallback;
}

/**
 * Detect arrows from open stroke — looks for V-shaped head at either end.
 */
function detectArrow(
  points: Point[],
  totalLen: number,
  elementX: number,
  elementY: number,
  bounds: { x: number; y: number; width: number; height: number },
): RecognizedShape | null {
  const makeBounds = () => ({
    x: elementX + bounds.x,
    y: elementY + bounds.y,
    width: bounds.width,
    height: bounds.height,
  });

  // Method 1: Shaft+head split — scan for a straight shaft with non-straight tail
  const tryDirection = (pts: Point[], reverse: boolean): RecognizedShape | null => {
    for (const shaftFrac of [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]) {
      const shaftEnd = Math.floor(pts.length * shaftFrac);
      if (shaftEnd < 3 || pts.length - shaftEnd < 3) { continue; }

      const shaftPts = pts.slice(0, shaftEnd + 1);
      const shaftLen = pathLength(shaftPts);
      const shaftDirect = dist(pts[0], pts[shaftEnd]);
      const shaftStraight = shaftLen > 0 ? shaftDirect / shaftLen : 0;

      if (shaftStraight > 0.72) {
        const tailPts = pts.slice(shaftEnd);
        const tailDirect = dist(tailPts[0], tailPts[tailPts.length - 1]);
        const tailLen = pathLength(tailPts);
        const tailStraight = tailLen > 0 ? tailDirect / tailLen : 1;

        if (tailStraight < 0.82 && tailLen > totalLen * 0.04) {
          const startPt = reverse ? points[points.length - 1] : points[0];
          const endPt = reverse
            ? points[points.length - 1 - shaftEnd]
            : points[shaftEnd];

          return {
            type: "arrow",
            confidence: Math.min(1, shaftStraight * 0.95),
            bounds: makeBounds(),
            startPoint: { x: elementX + startPt.x, y: elementY + startPt.y },
            endPoint: { x: elementX + endPt.x, y: elementY + endPt.y },
          };
        }
      }
    }
    return null;
  };

  // Try arrowhead at end, then at start
  const fwd = tryDirection(points, false);
  if (fwd) { return fwd; }
  const rev = tryDirection([...points].reverse(), true);
  if (rev) { return rev; }

  // Method 2: Elongated shape with sharp angle change — likely arrow even if V curls back
  // An arrow's bounding box is typically much wider than tall (or much taller than wide)
  const elongation = Math.max(bounds.width, bounds.height) / Math.max(Math.min(bounds.width, bounds.height), 1);
  if (elongation > 2.5 && points.length >= 8) {
    // Check if majority of path is straight (80%+ of points along a main axis)
    // Compute principal direction via first and last 25% of points
    const q1End = Math.floor(points.length * 0.25);
    const q3Start = Math.floor(points.length * 0.75);
    const startCenter = {
      x: points.slice(0, q1End).reduce((s, p) => s + p.x, 0) / q1End,
      y: points.slice(0, q1End).reduce((s, p) => s + p.y, 0) / q1End,
    };
    const endCenter = {
      x: points.slice(q3Start).reduce((s, p) => s + p.x, 0) / (points.length - q3Start),
      y: points.slice(q3Start).reduce((s, p) => s + p.y, 0) / (points.length - q3Start),
    };
    const mainAxisDist = dist(startCenter, endCenter);
    const mainAxisRatio = mainAxisDist / totalLen;

    if (mainAxisRatio > 0.30) {
      // Elongated with directional flow → likely arrow
      return {
        type: "arrow",
        confidence: Math.min(1, mainAxisRatio * 1.2),
        bounds: makeBounds(),
        startPoint: { x: elementX + points[0].x, y: elementY + points[0].y },
        endPoint: { x: elementX + points[points.length - 1].x, y: elementY + points[points.length - 1].y },
      };
    }
  }

  return null;
}
