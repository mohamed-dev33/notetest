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

/** Squared distance (avoids sqrt for comparisons) */
function distSq(a: Point, b: Point): number {
  return (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
}

/** Euclidean distance between two points */
function dist(a: Point, b: Point): number {
  return Math.sqrt(distSq(a, b));
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
function translateTo(pts: Point[], target: Point, c?: Point): Point[] {
  const ct = c ?? centroid(pts);
  return pts.map((p) => ({ x: p.x + target.x - ct.x, y: p.y + target.y - ct.y }));
}

/** Compute indicative angle (angle from centroid to first point) */
function indicativeAngle(pts: Point[], c?: Point): number {
  const ct = c ?? centroid(pts);
  return Math.atan2(ct.y - pts[0].y, ct.x - pts[0].x);
}

/** Rotate points by angle around centroid */
function rotateBy(pts: Point[], radians: number, c?: Point): Point[] {
  const ct = c ?? centroid(pts);
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  return pts.map((p) => {
    const dx = p.x - ct.x;
    const dy = p.y - ct.y;
    return { x: dx * cos - dy * sin + ct.x, y: dx * sin + dy * cos + ct.y };
  });
}

/**
 * Vectorize: create a 2N-element vector from resampled, rotated,
 * translated points. This is the Protractor representation.
 */
function vectorize(pts: Point[], orientationSensitive: boolean): number[] {
  const c = centroid(pts);
  const iAngle = indicativeAngle(pts, c);
  let delta = 0;

  if (orientationSensitive) {
    const baseOrientation = (Math.PI / 4) * Math.round(iAngle / (Math.PI / 4));
    delta = baseOrientation - iAngle;
  } else {
    delta = -iAngle;
  }

  const cos = Math.cos(delta);
  const sin = Math.sin(delta);

  let sum = 0;
  const vec: number[] = new Array(pts.length * 2);

  for (let i = 0; i < pts.length; i++) {
    const dx = pts[i].x - c.x;
    const dy = pts[i].y - c.y;
    vec[i * 2] = dx * cos - dy * sin;
    vec[i * 2 + 1] = dx * sin + dy * cos;
    sum += vec[i * 2] ** 2 + vec[i * 2 + 1] ** 2;
  }

  const magnitude = Math.sqrt(sum);
  if (magnitude > 0) {
    for (let i = 0; i < vec.length; i++) {
      vec[i] /= magnitude;
    }
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

  // NOTE: Lines and arrows are NOT included in Protractor templates.
  // They are handled entirely by geometric heuristics (Phase 0) which are
  // more accurate for open strokes. Including them here with different
  // orientationSensitive flags would cause score comparison mismatches.

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

  const windowSize = Math.max(3, Math.floor(pts.length / 20));
  const cornerCandidates: { idx: number; angle: number }[] = [];
  const minMagSq = 0.000001; // 0.001²

  for (let i = windowSize; i < pts.length - windowSize; i++) {
    const prev = pts[i - windowSize];
    const curr = pts[i];
    const next = pts[i + windowSize];

    const v1x = prev.x - curr.x;
    const v1y = prev.y - curr.y;
    const v2x = next.x - curr.x;
    const v2y = next.y - curr.y;

    const magSq1 = v1x * v1x + v1y * v1y;
    const magSq2 = v2x * v2x + v2y * v2y;
    if (magSq1 < minMagSq || magSq2 < minMagSq) { continue; }

    const dot = v1x * v2x + v1y * v2y;
    // Combined sqrt: sqrt(a)*sqrt(b) = sqrt(a*b)
    const cosAngle = Math.max(-1, Math.min(1, dot / Math.sqrt(magSq1 * magSq2)));
    const angleDeg = Math.acos(cosAngle) * 180 / Math.PI;

    if (angleDeg < (180 - angleThrDeg)) {
      cornerCandidates.push({ idx: i, angle: angleDeg });
    }
  }

  // Non-maximum suppression — keep the sharpest corner in each neighborhood
  const minDist = Math.max(4, pts.length / 8);
  const filtered: { idx: number; angle: number }[] = [];
  for (const candidate of cornerCandidates) {
    if (filtered.length === 0 || candidate.idx - filtered[filtered.length - 1].idx >= minDist) {
      filtered.push(candidate);
    } else {
      const prev = filtered[filtered.length - 1];
      if (candidate.angle < prev.angle) {
        filtered[filtered.length - 1] = candidate;
      }
    }
  }

  return filtered.map((c) => c.idx);
}

/**
 * Ramer-Douglas-Peucker simplification — iterative (no .slice() allocations).
 */
function rdpSimplify(pts: Point[], epsilon: number): Point[] {
  if (pts.length <= 2) { return [...pts]; }

  // Use a stack-based iterative approach to avoid O(n) slice per recursion
  const keep = new Uint8Array(pts.length);
  keep[0] = 1;
  keep[pts.length - 1] = 1;

  const stack: [number, number][] = [[0, pts.length - 1]];
  while (stack.length > 0) {
    const [start, end] = stack.pop()!;
    let maxDist = 0;
    let maxIdx = start;

    for (let i = start + 1; i < end; i++) {
      const d = pointToLineDistance(pts[i], pts[start], pts[end]);
      if (d > maxDist) {
        maxDist = d;
        maxIdx = i;
      }
    }

    if (maxDist > epsilon) {
      keep[maxIdx] = 1;
      if (maxIdx - start > 1) { stack.push([start, maxIdx]); }
      if (end - maxIdx > 1) { stack.push([maxIdx, end]); }
    }
  }

  const result: Point[] = [];
  for (let i = 0; i < pts.length; i++) {
    if (keep[i]) { result.push(pts[i]); }
  }
  return result;
}

function pointToLineDistance(p: Point, a: Point, b: Point): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) { return dist(p, a); }
  const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / lenSq));
  const px = a.x + t * dx - p.x;
  const py = a.y + t * dy - p.y;
  return Math.sqrt(px * px + py * py);
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

  // Avoid spread+slice: concat in-place
  lower.length--; // remove last (same as first of upper)
  upper.length--; // remove last (same as first of lower)
  return lower.concat(upper);
}

// =====================================================================
// Main Recognition Function — Hybrid Geometric + Protractor
// =====================================================================

const MIN_POINTS = 5;
const MIN_STROKE_SIZE = 15;

/**
 * Measure the angle (in degrees) at a corner point given its neighbors.
 */
function measureCornerAngle(pts: Point[], cornerIdx: number, windowSize: number): number {
  const pIdx = Math.max(0, cornerIdx - windowSize);
  const nIdx = Math.min(pts.length - 1, cornerIdx + windowSize);
  const prev = pts[pIdx];
  const curr = pts[cornerIdx];
  const next = pts[nIdx];
  const v1x = prev.x - curr.x;
  const v1y = prev.y - curr.y;
  const v2x = next.x - curr.x;
  const v2y = next.y - curr.y;
  const magSq1 = v1x * v1x + v1y * v1y;
  const magSq2 = v2x * v2x + v2y * v2y;
  if (magSq1 < 0.000001 || magSq2 < 0.000001) { return 180; }
  const dot = v1x * v2x + v1y * v2y;
  const cosAngle = Math.max(-1, Math.min(1, dot / Math.sqrt(magSq1 * magSq2)));
  return Math.acos(cosAngle) * 180 / Math.PI;
}

/**
 * Compute average curvature along the path.
 * Uses squared lengths to avoid sqrt in inner loop.
 */
function averageCurvature(pts: Point[]): number {
  if (pts.length < 3) { return 0; }
  let totalAngle = 0;
  let count = 0;
  const step = Math.max(1, Math.floor(pts.length / 40));
  const minLenSq = 0.25; // 0.5²
  for (let i = step; i < pts.length - step; i += step) {
    const dx1 = pts[i].x - pts[i - step].x;
    const dy1 = pts[i].y - pts[i - step].y;
    const dx2 = pts[i + step].x - pts[i].x;
    const dy2 = pts[i + step].y - pts[i].y;
    const lenSq1 = dx1 * dx1 + dy1 * dy1;
    const lenSq2 = dx2 * dx2 + dy2 * dy2;
    if (lenSq1 < minLenSq || lenSq2 < minLenSq) { continue; }
    const dot = (dx1 * dx2 + dy1 * dy2) / Math.sqrt(lenSq1 * lenSq2);
    totalAngle += Math.acos(Math.max(-1, Math.min(1, dot)));
    count++;
  }
  return count > 0 ? totalAngle / count : 0;
}

/**
 * Check if corners have approximately right angles (~90°).
 * Returns fraction of corners that are near-right-angle.
 */
function rightAngleFraction(pts: Point[], cornerIndices: number[]): number {
  if (cornerIndices.length === 0) { return 0; }
  const windowSize = Math.max(3, Math.floor(pts.length / 16));
  let rightCount = 0;
  for (const ci of cornerIndices) {
    const angle = measureCornerAngle(pts, ci, windowSize);
    // Right angle = 90° ± 30°
    if (angle >= 60 && angle <= 120) { rightCount++; }
  }
  return rightCount / cornerIndices.length;
}

/**
 * Check if the stroke has consistent curvature (circle-like)
 * vs sharp changes (polygon-like). Uses squared lengths.
 */
function curvatureVariance(pts: Point[]): number {
  if (pts.length < 6) { return 0; }
  const curvatures: number[] = [];
  const step = Math.max(1, Math.floor(pts.length / 30));
  const minLenSq = 0.25; // 0.5²
  for (let i = step; i < pts.length - step; i += step) {
    const dx1 = pts[i].x - pts[i - step].x;
    const dy1 = pts[i].y - pts[i - step].y;
    const dx2 = pts[i + step].x - pts[i].x;
    const dy2 = pts[i + step].y - pts[i].y;
    const lenSq1 = dx1 * dx1 + dy1 * dy1;
    const lenSq2 = dx2 * dx2 + dy2 * dy2;
    if (lenSq1 < minLenSq || lenSq2 < minLenSq) { continue; }
    const dot = (dx1 * dx2 + dy1 * dy2) / Math.sqrt(lenSq1 * lenSq2);
    curvatures.push(Math.acos(Math.max(-1, Math.min(1, dot))));
  }
  if (curvatures.length < 2) { return 0; }
  const mean = curvatures.reduce((s, v) => s + v, 0) / curvatures.length;
  const variance = curvatures.reduce((s, v) => s + (v - mean) ** 2, 0) / curvatures.length;
  return variance;
}

/**
 * Compute perimeter-to-diameter ratio.
 * Circle ≈ π (3.14), square ≈ 4, triangle ≈ depends on shape
 */
function perimeterDiameterRatio(pts: Point[], bounds: { width: number; height: number }): number {
  const perim = pathLength(pts);
  const diameter = Math.sqrt(bounds.width ** 2 + bounds.height ** 2);
  return diameter > 0 ? perim / diameter : 0;
}

/**
 * Pre-process raw stroke: remove duplicate points, smooth pen jitter.
 * Pen input produces more noise than mouse — this reduces false corners.
 */
function preprocessStroke(pts: Point[]): Point[] {
  if (pts.length < 3) { return [...pts]; }

  // Remove duplicate/near-duplicate consecutive points (pen pressure stutters)
  const deduped: Point[] = [pts[0]];
  const minDistSq = 1.0; // 1px² threshold
  for (let i = 1; i < pts.length; i++) {
    if (distSq(pts[i], deduped[deduped.length - 1]) >= minDistSq) {
      deduped.push(pts[i]);
    }
  }
  if (deduped.length < 3) { return deduped; }

  // Light Gaussian-like smoothing (3-point weighted average) to reduce pen jitter
  // Preserves start/end points and overall shape, but reduces micro-wobble
  const smoothed: Point[] = [deduped[0]];
  for (let i = 1; i < deduped.length - 1; i++) {
    smoothed.push({
      x: deduped[i - 1].x * 0.2 + deduped[i].x * 0.6 + deduped[i + 1].x * 0.2,
      y: deduped[i - 1].y * 0.2 + deduped[i].y * 0.6 + deduped[i + 1].y * 0.2,
    });
  }
  smoothed.push(deduped[deduped.length - 1]);
  return smoothed;
}

/**
 * Classify detected corners as near bounding-box corners (rectangle-like)
 * vs near bounding-box edge midpoints (diamond-like).
 *
 * Returns a score from -1 (strongly diamond) to +1 (strongly rectangle).
 * 0 = ambiguous.
 *
 * Uses squared distances to avoid sqrt. Also considers the distribution
 * pattern: rectangles have 4 corners in 4 quadrants of the bbox, while
 * diamonds have corners at N/S/E/W midpoints.
 */
function cornerPositionScore(
  pts: Point[],
  cornerIndices: number[],
  bounds: { x: number; y: number; width: number; height: number },
): number {
  if (cornerIndices.length < 2 || bounds.width < 1 || bounds.height < 1) {
    return 0;
  }

  let rectScore = 0;
  let diamondScore = 0;

  // Track which quadrants/midpoints are covered
  let rectQuadrants = 0; // bits: TL=1, TR=2, BR=4, BL=8
  let diamondMids = 0;   // bits: T=1, R=2, B=4, L=8

  for (const ci of cornerIndices) {
    const nx = (pts[ci].x - bounds.x) / bounds.width;  // 0..1
    const ny = (pts[ci].y - bounds.y) / bounds.height;

    // Squared distance to nearest bbox corner (rectangle pattern)
    const dCornerSq = Math.min(
      nx * nx + ny * ny,                            // top-left
      (1 - nx) * (1 - nx) + ny * ny,                // top-right
      (1 - nx) * (1 - nx) + (1 - ny) * (1 - ny),   // bottom-right
      nx * nx + (1 - ny) * (1 - ny),                // bottom-left
    );

    // Squared distance to nearest bbox edge midpoint (diamond pattern)
    const dMidSq = Math.min(
      (0.5 - nx) * (0.5 - nx) + ny * ny,                // top-mid
      (1 - nx) * (1 - nx) + (0.5 - ny) * (0.5 - ny),   // right-mid
      (0.5 - nx) * (0.5 - nx) + (1 - ny) * (1 - ny),   // bottom-mid
      nx * nx + (0.5 - ny) * (0.5 - ny),                // left-mid
    );

    if (dCornerSq < dMidSq) {
      rectScore += Math.sqrt(dMidSq) - Math.sqrt(dCornerSq);
      // Track which quadrant
      if (nx < 0.5 && ny < 0.5) { rectQuadrants |= 1; }
      if (nx >= 0.5 && ny < 0.5) { rectQuadrants |= 2; }
      if (nx >= 0.5 && ny >= 0.5) { rectQuadrants |= 4; }
      if (nx < 0.5 && ny >= 0.5) { rectQuadrants |= 8; }
    } else {
      diamondScore += Math.sqrt(dCornerSq) - Math.sqrt(dMidSq);
      // Track which midpoint
      if (ny < 0.3) { diamondMids |= 1; }  // top
      if (nx > 0.7) { diamondMids |= 2; }  // right
      if (ny > 0.7) { diamondMids |= 4; }  // bottom
      if (nx < 0.3) { diamondMids |= 8; }  // left
    }
  }

  // Bonus for having corners in all 4 quadrants (strong rectangle signal)
  const rectCoverage = popcount(rectQuadrants) / 4;
  const diamondCoverage = popcount(diamondMids) / 4;
  rectScore += rectCoverage * 0.3;
  diamondScore += diamondCoverage * 0.3;

  const total = rectScore + diamondScore;
  if (total < 0.001) { return 0; }
  return (rectScore - diamondScore) / total; // -1..+1
}

/** Count set bits (popcount) */
function popcount(n: number): number {
  let count = 0;
  while (n) { count += n & 1; n >>= 1; }
  return count;
}

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

  // Preprocess: remove duplicates and smooth pen jitter
  const rawPts: Point[] = rawPoints.map((p) => ({ x: p[0], y: p[1] }));
  const points = preprocessStroke(rawPts);

  if (points.length < MIN_POINTS) {
    return fallback;
  }

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
  const straightness = totalLen > 0 ? closeDist / totalLen : 0;

  // Very straight stroke → line
  if (straightness > 0.88) {
    return {
      type: "line",
      confidence: Math.min(1, straightness * 1.02),
      bounds: makeBounds(),
      startPoint: { x: elementX + points[0].x, y: elementY + points[0].y },
      endPoint: { x: elementX + points[points.length - 1].x, y: elementY + points[points.length - 1].y },
    };
  }

  // Arrow detection — for open or semi-closed shapes
  if (points.length >= 6 && closeness > 0.08) {
    const arrowResult = detectArrow(points, totalLen, elementX, elementY, bounds);
    if (arrowResult) { return arrowResult; }
  }
  // For nearly-closed elongated shapes, check arrow pattern (V-head curling back)
  if (points.length >= 6 && closeness <= 0.08) {
    const elongation = Math.max(bounds.width, bounds.height) / Math.max(Math.min(bounds.width, bounds.height), 1);
    if (elongation > 2.5) {
      const arrowResult = detectArrow(points, totalLen, elementX, elementY, bounds);
      if (arrowResult && arrowResult.confidence > 0.65) { return arrowResult; }
    }
  }

  // Moderately straight open stroke → line
  if (!effectivelyClosed && straightness > 0.78) {
    return {
      type: "line",
      confidence: straightness * 0.92,
      bounds: makeBounds(),
      startPoint: { x: elementX + points[0].x, y: elementY + points[0].y },
      endPoint: { x: elementX + points[points.length - 1].x, y: elementY + points[points.length - 1].y },
    };
  }

  // ========== Phase 1: Closed-shape geometric analysis ==========
  const analyzePoints = effectivelyClosed ? points : [...points, { ...points[0] }];
  const area = shoelaceArea(analyzePoints);
  const bbArea = bounds.width * bounds.height;
  const fillRatio = bbArea > 0 ? area / bbArea : 0;
  const circ = circularity(analyzePoints);
  const aspectRatio = bounds.width / Math.max(bounds.height, 1);
  const elongationRatio = Math.max(bounds.width, bounds.height) / Math.max(Math.min(bounds.width, bounds.height), 1);

  // Corner detection
  const epsilon = Math.max(bounds.width, bounds.height) * 0.04;
  const simplified = rdpSimplify(analyzePoints, epsilon);
  const vertexCount = simplified.length - (dist(simplified[0], simplified[simplified.length - 1]) < epsilon * 2 ? 1 : 0);

  const hull = convexHull(analyzePoints);
  const hullArea = shoelaceArea(hull);
  const convexity = hullArea > 0 ? area / hullArea : 0;

  const corners = detectCorners(analyzePoints, 35);
  const cornerCount = corners.length;

  // Advanced metrics
  const rightAngleFrac = cornerCount > 0 ? rightAngleFraction(analyzePoints, corners) : 0;
  const curvVar = curvatureVariance(analyzePoints);
  const pdRatio = perimeterDiameterRatio(analyzePoints, bounds);
  const cpScore = cornerPositionScore(analyzePoints, corners, bounds);

  const closedConf = effectivelyClosed ? 1.0 : 0.85;

  // ---- Ellipse/Circle detection ----
  // High circularity, low curvature variance (smooth curve), few sharp corners
  // Key: rightAngleFrac should be LOW (ellipses don't have right angles)
  // BUT with very few corners (<=1), rightAngleFrac is unreliable — trust circularity
  if (circ > 0.62 && convexity > 0.85 && cornerCount <= 3 && curvVar < 0.20
      && (cornerCount <= 1 || rightAngleFrac < 0.5)) {
    return {
      type: "ellipse",
      confidence: Math.min(1, circ * 1.10 * closedConf),
      bounds: makeBounds(),
    };
  }
  // Fallback ellipse: good circularity even with some corners (hand-drawn wobble)
  if (circ > 0.72 && convexity > 0.88 && curvVar < 0.12 && (cornerCount <= 1 || rightAngleFrac < 0.4)) {
    return {
      type: "ellipse",
      confidence: Math.min(1, circ * 1.05 * closedConf),
      bounds: makeBounds(),
    };
  }

  // ---- Rectangle detection ----
  // Rectangles: corners at bbox corners → cpScore > 0, high fillRatio, right angles
  // Lower fillRatio threshold to 0.60 to catch hand-drawn rectangles with loose corners
  // Use cpScore to disambiguate from diamond in the overlap zone
  if (
    cornerCount >= 3 && cornerCount <= 6 &&
    fillRatio > 0.60 &&
    convexity > 0.82 &&
    rightAngleFrac >= 0.35 &&
    circ < 0.82 &&
    aspectRatio > 0.15 && aspectRatio < 6.0 &&
    cpScore > -0.3 // corners should NOT be at edge midpoints (diamond pattern)
  ) {
    const conf = Math.min(1, (fillRatio * 0.4 + rightAngleFrac * 0.4 + Math.max(0, cpScore) * 0.2) * 1.05 * closedConf);
    return {
      type: "rectangle",
      confidence: conf,
      bounds: makeBounds(),
    };
  }
  // Relaxed rectangle: high fill, not smooth, corners near bbox corners
  if (
    cornerCount >= 2 &&
    fillRatio > 0.68 &&
    convexity > 0.85 &&
    circ < 0.80 &&
    curvVar > 0.02 &&
    aspectRatio > 0.2 && aspectRatio < 5.0 &&
    cpScore > -0.2
  ) {
    return {
      type: "rectangle",
      confidence: Math.min(1, fillRatio * 1.02 * closedConf),
      bounds: makeBounds(),
    };
  }

  // ---- Diamond detection ----
  // Diamond: ~4 vertices, corners at bbox edge midpoints, fillRatio ≈ 0.50
  // Near-square aspect ratio (rotated square), 4 clear corners
  if (
    cornerCount >= 2 && cornerCount <= 6 &&
    fillRatio > 0.30 && fillRatio < 0.68 &&
    convexity > 0.82 &&
    vertexCount >= 4 &&
    aspectRatio > 0.35 && aspectRatio < 2.8 &&
    !(rightAngleFrac > 0.5 && fillRatio > 0.55) // not a rectangle
  ) {
    const cpBonus = Math.max(0, -cpScore) * 0.1;
    return {
      type: "diamond",
      confidence: Math.min(1, (convexity * 0.90 + cpBonus) * closedConf),
      bounds: makeBounds(),
    };
  }

  // ---- Triangle detection ----
  // Triangles: ~3 corners, fill ratio ~0.50, NOT elongated
  // Key distinguisher from diamond: fewer vertices (3 vs 4), lower corner count
  if (
    cornerCount >= 2 && cornerCount <= 4 &&
    fillRatio > 0.25 && fillRatio < 0.68 &&
    convexity > 0.78 &&
    aspectRatio > 0.35 && aspectRatio < 2.5 &&
    elongationRatio < 2.5 &&
    vertexCount >= 3 && vertexCount <= 5
  ) {
    const triBonus = vertexCount === 3 ? 0.05 : 0;
    return {
      type: "triangle",
      confidence: Math.min(1, (convexity * 0.88 + triBonus) * closedConf),
      bounds: makeBounds(),
    };
  }

  // Relaxed diamond: fewer constraints, catches diamonds with fewer vertices
  if (
    cornerCount >= 2 && cornerCount <= 6 &&
    fillRatio > 0.28 && fillRatio < 0.65 &&
    convexity > 0.78 &&
    aspectRatio > 0.35 && aspectRatio < 2.8 &&
    !(rightAngleFrac > 0.5 && fillRatio > 0.50) &&
    cpScore < 0.4
  ) {
    return {
      type: "diamond",
      confidence: Math.min(1, convexity * 0.85 * closedConf),
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

    // Cross-validate Protractor result with geometric metrics + corner positions
    if (protoScore >= 0.50) {
      let validated = protoName;

      // Protractor says rectangle but fill is low → diamond or triangle
      if (protoName === "rectangle" && fillRatio < 0.55) {
        validated = fillRatio < 0.40 ? "triangle" : "diamond";
      }
      // Protractor says rectangle but corners at edge midpoints → diamond
      else if (protoName === "rectangle" && cpScore < -0.3 && fillRatio < 0.68) {
        validated = "diamond";
      }
      // Protractor says ellipse but low circularity → polygon
      else if (protoName === "ellipse" && circ < 0.60) {
        if (cornerCount >= 3 && fillRatio > 0.60 && cpScore > -0.2) { validated = "rectangle"; }
        else if (cornerCount >= 3 && fillRatio > 0.35) { validated = "diamond"; }
      }
      // Protractor says triangle but high fill → rectangle
      else if (protoName === "triangle" && fillRatio > 0.72) {
        validated = "rectangle";
      }
      // Protractor says diamond but high fill + corners at bbox corners → rectangle
      else if (protoName === "diamond" && fillRatio > 0.68 && cpScore > 0) {
        validated = "rectangle";
      }
      else if (protoName === "diamond" && fillRatio > 0.78) {
        validated = "rectangle";
      }
      // Rectangle needs some right-angle corners — use cpScore as tiebreaker
      if (validated === "rectangle" && rightAngleFrac < 0.20 && cornerCount >= 3) {
        if (fillRatio < 0.55 || cpScore < -0.3) {
          validated = "diamond";
        }
      }

      return {
        type: validated,
        confidence: Math.min(1, protoScore * closedConf),
        bounds: makeBounds(),
      };
    }

    // Phase 4: Pure geometric fallback — more conservative thresholds
    if (circ > 0.60 && cornerCount <= 3 && curvVar < 0.18 && (cornerCount <= 1 || rightAngleFrac < 0.5)) {
      return { type: "ellipse", confidence: Math.min(1, circ * 1.05) * closedConf, bounds: makeBounds() };
    }
    if (fillRatio > 0.60 && rightAngleFrac > 0.25 && circ < 0.80 && cpScore > -0.2) {
      return { type: "rectangle", confidence: fillRatio * closedConf, bounds: makeBounds() };
    }
    if (fillRatio > 0.30 && fillRatio < 0.68 && aspectRatio > 0.5 && aspectRatio < 2.0 && cornerCount >= 3 && vertexCount >= 4 && cpScore < 0.3) {
      return { type: "diamond", confidence: Math.max(0.5, fillRatio) * closedConf, bounds: makeBounds() };
    }
    if (fillRatio > 0.25 && cornerCount >= 2 && cornerCount <= 4 && elongationRatio < 2.2 && vertexCount <= 4) {
      return { type: "triangle", confidence: Math.max(0.5, convexity * 0.8) * closedConf, bounds: makeBounds() };
    }
  }

  return fallback;
}

/**
 * Recognize top-N shape candidates from freedraw points.
 * Returns up to `maxResults` candidates ranked by confidence.
 * Used for showing suggestions UI so user can pick the right shape.
 */
export function recognizeShapeTopN(
  rawPoints: readonly LocalPoint[],
  elementX: number,
  elementY: number,
  maxResults: number = 3,
): RecognizedShape[] {
  // First get the primary result
  const primary = recognizeShape(rawPoints, elementX, elementY);

  // For lines/arrows/freedraw, just return the single result
  if (primary.type === "line" || primary.type === "arrow" || primary.type === "freedraw") {
    return primary.type === "freedraw" ? [] : [primary];
  }

  if (rawPoints.length < MIN_POINTS) {
    return [];
  }

  const points: Point[] = rawPoints.map((p) => ({ x: p[0], y: p[1] }));
  const smoothed = preprocessStroke(points);
  const pts = smoothed.length >= MIN_POINTS ? smoothed : points;
  const bounds = getBounds(pts);

  if (bounds.width < MIN_STROKE_SIZE && bounds.height < MIN_STROKE_SIZE) {
    return [];
  }

  const totalLen = pathLength(pts);
  const closeDist = dist(pts[0], pts[pts.length - 1]);
  const closeness = totalLen > 0 ? closeDist / totalLen : 1;
  const effectivelyClosed = closeness < 0.20;

  const makeBounds = () => ({
    x: elementX + bounds.x,
    y: elementY + bounds.y,
    width: bounds.width,
    height: bounds.height,
  });

  // Compute geometric features
  const analyzePoints = effectivelyClosed ? pts : [...pts, { ...pts[0] }];
  const area = shoelaceArea(analyzePoints);
  const bbArea = bounds.width * bounds.height;
  const fillRatio = bbArea > 0 ? area / bbArea : 0;
  const circ = circularity(analyzePoints);
  const aspectRatio = bounds.width / Math.max(bounds.height, 1);

  const epsilon = Math.max(bounds.width, bounds.height) * 0.04;
  const simplified = rdpSimplify(analyzePoints, epsilon);
  const vertexCount = simplified.length - (dist(simplified[0], simplified[simplified.length - 1]) < epsilon * 2 ? 1 : 0);

  const hull = convexHull(analyzePoints);
  const hullArea = shoelaceArea(hull);
  const convexity = hullArea > 0 ? area / hullArea : 0;

  const corners = detectCorners(analyzePoints, 35);
  const cornerCount = corners.length;
  const rightAngleFrac = cornerCount > 0 ? rightAngleFraction(analyzePoints, corners) : 0;
  const curvVar = curvatureVariance(analyzePoints);
  const cpScore = cornerPositionScore(analyzePoints, corners, bounds);
  const closedConf = effectivelyClosed ? 1.0 : 0.85;

  // Score each closed shape type independently
  const candidates: RecognizedShape[] = [];

  // Ellipse score
  const ellipseScore = computeEllipseScore(circ, convexity, cornerCount, curvVar, rightAngleFrac, closedConf);
  if (ellipseScore > 0.3) {
    candidates.push({ type: "ellipse", confidence: ellipseScore, bounds: makeBounds() });
  }

  // Rectangle score
  const rectScore = computeRectangleScore(cornerCount, fillRatio, convexity, rightAngleFrac, circ, aspectRatio, cpScore, closedConf);
  if (rectScore > 0.3) {
    candidates.push({ type: "rectangle", confidence: rectScore, bounds: makeBounds() });
  }

  // Diamond score
  const diamondScoreVal = computeDiamondScore(cornerCount, fillRatio, convexity, vertexCount, aspectRatio, rightAngleFrac, cpScore, closedConf);
  if (diamondScoreVal > 0.3) {
    candidates.push({ type: "diamond", confidence: diamondScoreVal, bounds: makeBounds() });
  }

  // Triangle score
  const triScore = computeTriangleScore(cornerCount, fillRatio, convexity, aspectRatio, vertexCount, closedConf);
  if (triScore > 0.3) {
    candidates.push({ type: "triangle", confidence: triScore, bounds: makeBounds() });
  }

  // Sort by confidence descending
  candidates.sort((a, b) => b.confidence - a.confidence);

  // Ensure primary is first if it matches a candidate
  const primaryIdx = candidates.findIndex((c) => c.type === primary.type);
  if (primaryIdx > 0) {
    // Move primary to front but keep its score from recognizeShape
    const [moved] = candidates.splice(primaryIdx, 1);
    moved.confidence = Math.max(moved.confidence, primary.confidence);
    candidates.unshift(moved);
  } else if (primaryIdx === -1) {
    // Primary not in candidates — add it first
    candidates.unshift(primary);
  }

  // Deduplicate and cap
  const seen = new Set<string>();
  const result: RecognizedShape[] = [];
  for (const c of candidates) {
    if (!seen.has(c.type) && result.length < maxResults) {
      seen.add(c.type);
      result.push(c);
    }
  }

  return result;
}

// ---- Independent scoring functions for each shape type ----

function computeEllipseScore(
  circ: number, convexity: number, cornerCount: number,
  curvVar: number, rightAngleFrac: number, closedConf: number,
): number {
  let score = 0;
  // High circularity is the primary signal
  if (circ > 0.55) { score += circ * 0.5; }
  // Low curvature variance = smooth curve
  if (curvVar < 0.20) { score += (0.20 - curvVar) * 1.0; }
  // Few corners
  if (cornerCount <= 2) { score += 0.15; }
  else if (cornerCount <= 3) { score += 0.05; }
  // High convexity
  if (convexity > 0.85) { score += 0.1; }
  // Penalty for right angles (ellipses don't have them)
  if (rightAngleFrac > 0.4) { score -= 0.2; }
  return Math.min(1, score * closedConf);
}

function computeRectangleScore(
  cornerCount: number, fillRatio: number, convexity: number,
  rightAngleFrac: number, circ: number, aspectRatio: number,
  cpScore: number, closedConf: number,
): number {
  let score = 0;
  // Right angles are the strongest rectangle signal
  if (rightAngleFrac >= 0.5) { score += rightAngleFrac * 0.35; }
  else if (rightAngleFrac >= 0.3) { score += rightAngleFrac * 0.25; }
  // High fill ratio
  if (fillRatio > 0.60) { score += fillRatio * 0.25; }
  // Corners near bbox corners (positive cpScore)
  if (cpScore > 0) { score += cpScore * 0.2; }
  // 3-6 corners expected
  if (cornerCount >= 3 && cornerCount <= 6) { score += 0.1; }
  // Not too circular
  if (circ < 0.80) { score += 0.05; }
  // Reasonable aspect ratio
  if (aspectRatio > 0.15 && aspectRatio < 6.0) { score += 0.05; }
  // Good convexity
  if (convexity > 0.82) { score += 0.05; }
  // Penalty for corners at edge midpoints (diamond signal)
  if (cpScore < -0.3) { score -= 0.15; }
  return Math.min(1, score * closedConf);
}

function computeDiamondScore(
  cornerCount: number, fillRatio: number, convexity: number,
  vertexCount: number, aspectRatio: number, rightAngleFrac: number,
  cpScore: number, closedConf: number,
): number {
  let score = 0;
  // Fill ratio around 0.50 is ideal for diamonds
  if (fillRatio > 0.30 && fillRatio < 0.68) {
    score += (1 - Math.abs(fillRatio - 0.50) * 3) * 0.3;
  }
  // Corners at edge midpoints (negative cpScore)
  if (cpScore < 0) { score += Math.abs(cpScore) * 0.25; }
  // ~4 vertices
  if (vertexCount >= 4 && vertexCount <= 6) { score += 0.15; }
  // 2-6 corners
  if (cornerCount >= 2 && cornerCount <= 6) { score += 0.1; }
  // Near-square aspect
  if (aspectRatio > 0.35 && aspectRatio < 2.8) { score += 0.05; }
  // Convexity
  if (convexity > 0.82) { score += 0.05; }
  // Penalty for strong right angles (rectangle signal)
  if (rightAngleFrac > 0.5) { score -= 0.15; }
  // Penalty for very high fill (rectangle territory)
  if (fillRatio > 0.72) { score -= 0.1; }
  return Math.min(1, score * closedConf);
}

function computeTriangleScore(
  cornerCount: number, fillRatio: number, convexity: number,
  aspectRatio: number, vertexCount: number, closedConf: number,
): number {
  let score = 0;
  // ~3 corners
  if (cornerCount >= 2 && cornerCount <= 4) { score += 0.2; }
  if (cornerCount === 3) { score += 0.1; }
  // Fill ratio ~0.50
  if (fillRatio > 0.25 && fillRatio < 0.68) { score += 0.2; }
  // 3 vertices ideal
  if (vertexCount >= 3 && vertexCount <= 5) { score += 0.15; }
  if (vertexCount === 3) { score += 0.1; }
  // Convexity
  if (convexity > 0.78) { score += 0.1; }
  // Reasonable aspect
  if (aspectRatio > 0.35 && aspectRatio < 2.5) { score += 0.05; }
  // Penalty for too many corners (rectangle/diamond)
  if (cornerCount > 4) { score -= 0.15; }
  return Math.min(1, score * closedConf);
}

/**
 * Detect arrows from open stroke — looks for V-shaped head at either end.
 * Uses 3 methods for robustness:
 *   1. Shaft+head split: find straight shaft with non-straight tail
 *   2. Sharp angle detection: find V-turn in stroke
 *   3. Elongated directional flow: elongated shape with consistent direction
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
    let bestConf = 0;
    let bestResult: RecognizedShape | null = null;

    for (const shaftFrac of [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]) {
      const shaftEnd = Math.floor(pts.length * shaftFrac);
      if (shaftEnd < 4 || pts.length - shaftEnd < 4) { continue; }

      const shaftPts = pts.slice(0, shaftEnd + 1);
      const shaftLen = pathLength(shaftPts);
      const shaftDirect = dist(pts[0], pts[shaftEnd]);
      const shaftStraight = shaftLen > 0 ? shaftDirect / shaftLen : 0;

      if (shaftStraight > 0.70) {
        const tailPts = pts.slice(shaftEnd);
        const tailDirect = dist(tailPts[0], tailPts[tailPts.length - 1]);
        const tailLen = pathLength(tailPts);
        const tailStraight = tailLen > 0 ? tailDirect / tailLen : 1;

        // Tail should bend (arrowhead) and be at least 5% of total
        if (tailStraight < 0.80 && tailLen > totalLen * 0.05) {
          // Verify V-head: the tail should spread away from shaft direction
          const shaftAngle = Math.atan2(
            pts[shaftEnd].y - pts[0].y,
            pts[shaftEnd].x - pts[0].x,
          );
          const tailEndAngle = Math.atan2(
            tailPts[tailPts.length - 1].y - tailPts[0].y,
            tailPts[tailPts.length - 1].x - tailPts[0].x,
          );
          const headDeflection = Math.abs(shaftAngle - tailEndAngle);
          const normalizedDeflection = headDeflection > Math.PI
            ? 2 * Math.PI - headDeflection
            : headDeflection;

          // Head should deflect at least 30° from shaft
          if (normalizedDeflection > Math.PI / 6) {
            const startPt = reverse ? points[points.length - 1] : points[0];
            const endPt = reverse
              ? points[points.length - 1 - shaftEnd]
              : points[shaftEnd];

            const conf = Math.min(1,
              shaftStraight * 0.7 +
              (1 - tailStraight) * 0.15 +
              Math.min(normalizedDeflection / Math.PI, 0.15),
            );
            if (conf > bestConf) {
              bestConf = conf;
              bestResult = {
                type: "arrow",
                confidence: conf,
                bounds: makeBounds(),
                startPoint: { x: elementX + startPt.x, y: elementY + startPt.y },
                endPoint: { x: elementX + endPt.x, y: elementY + endPt.y },
              };
            }
          }
        }
      }
    }
    return bestResult;
  };

  // Try arrowhead at end, then at start
  const fwd = tryDirection(points, false);
  if (fwd) { return fwd; }
  const rev = tryDirection([...points].reverse(), true);
  if (rev) { return rev; }

  // Method 2: Detect sharp angle change in stroke
  const trySharpAngle = (pts: Point[], reverse: boolean): RecognizedShape | null => {
    if (pts.length < 10) { return null; }

    const checkStart = Math.floor(pts.length * 0.45);
    let maxAngleChange = 0;
    let maxAngleIdx = -1;
    const step = Math.max(2, Math.floor(pts.length / 40));

    for (let i = checkStart + step; i < pts.length - step; i++) {
      const dx1 = pts[i].x - pts[i - step].x;
      const dy1 = pts[i].y - pts[i - step].y;
      const dx2 = pts[i + step].x - pts[i].x;
      const dy2 = pts[i + step].y - pts[i].y;
      const lenSq1 = dx1 * dx1 + dy1 * dy1;
      const lenSq2 = dx2 * dx2 + dy2 * dy2;
      if (lenSq1 < 1 || lenSq2 < 1) { continue; }
      const dot = (dx1 * dx2 + dy1 * dy2) / Math.sqrt(lenSq1 * lenSq2);
      const angle = Math.acos(Math.max(-1, Math.min(1, dot)));
      if (angle > maxAngleChange) {
        maxAngleChange = angle;
        maxAngleIdx = i;
      }
    }

    // Need a sharp angle (> 70°) — tighter than before to avoid false positives
    if (maxAngleChange > Math.PI * 70 / 180 && maxAngleIdx > 0) {
      const shaftPts = pts.slice(0, maxAngleIdx + 1);
      const shaftDirect = dist(shaftPts[0], shaftPts[shaftPts.length - 1]);
      const shaftLen = pathLength(shaftPts);
      const shaftStraight = shaftLen > 0 ? shaftDirect / shaftLen : 0;

      if (shaftStraight > 0.65) {
        const startPt = reverse ? points[points.length - 1] : points[0];
        const endIdx = reverse ? points.length - 1 - maxAngleIdx : maxAngleIdx;
        const endPt = points[endIdx];

        return {
          type: "arrow",
          confidence: Math.min(1, shaftStraight * 0.80 + maxAngleChange / Math.PI * 0.20),
          bounds: makeBounds(),
          startPoint: { x: elementX + startPt.x, y: elementY + startPt.y },
          endPoint: { x: elementX + endPt.x, y: elementY + endPt.y },
        };
      }
    }
    return null;
  };

  const sharp = trySharpAngle(points, false);
  if (sharp) { return sharp; }
  const sharpRev = trySharpAngle([...points].reverse(), true);
  if (sharpRev) { return sharpRev; }

  // Method 3: Elongated shape with directional flow
  // TIGHTER than before — require higher elongation AND verify the stroke
  // has a clear main axis with limited perpendicular spread
  const elongation = Math.max(bounds.width, bounds.height) / Math.max(Math.min(bounds.width, bounds.height), 1);
  if (elongation > 3.0 && points.length >= 10) {
    // Verify perpendicular spread is small relative to length
    const mainAxis = bounds.width > bounds.height ? "horizontal" : "vertical";
    const mainLen = mainAxis === "horizontal" ? bounds.width : bounds.height;
    const crossLen = mainAxis === "horizontal" ? bounds.height : bounds.width;

    // Perpendicular spread must be < 30% of main axis
    if (crossLen < mainLen * 0.30) {
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

      if (mainAxisRatio > 0.35) {
        return {
          type: "arrow",
          confidence: Math.min(0.85, mainAxisRatio * 1.1),
          bounds: makeBounds(),
          startPoint: { x: elementX + points[0].x, y: elementY + points[0].y },
          endPoint: { x: elementX + points[points.length - 1].x, y: elementY + points[points.length - 1].y },
        };
      }
    }
  }

  return null;
}
