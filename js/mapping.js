/*
 * Copyright (c) 2026 CNCKitchen (Stefan Hermann) and contributors
 * SPDX-License-Identifier: AGPL-3.0-only
 */

/**
 * CPU-side UV mapping — exact JavaScript mirror of the GLSL in previewMaterial.js.
 * All functions take Three.js Vector3 objects for position/normal and
 * a bounds object { min, max, center, size } (all THREE.Vector3).
 */

import { QuantizedPointMap } from './meshIndex.js';

export const MODE_PLANAR_XY   = 0;
export const MODE_PLANAR_XZ   = 1;
export const MODE_PLANAR_YZ   = 2;
export const MODE_CYLINDRICAL = 3;
export const MODE_SPHERICAL   = 4;
export const MODE_TRIPLANAR   = 5;
export const MODE_CUBIC       = 6;

const TWO_PI = Math.PI * 2;

/**
 * Reference lengths (mm) that one full UV unit spans along U and V, per
 * mapping mode. settings.scaleU/scaleV are absolute tile sizes in mm; the
 * internal math still works in normalized coordinates, so consumers divide
 * mm by these lengths to get the relative scale factor:
 *
 *   planar / triplanar / cubic: largest bbox edge (cancels against the
 *     coordinate normalization → texture is truly world-anchored)
 *   cylindrical: circumference of the projection cylinder (U = arc length,
 *     V is normalized by the same C in computeUV)
 *   spherical: equator arc for U, pole-to-pole meridian arc for V
 */
export function getScaleReferenceLengths(mode, settings, bounds) {
  const { size } = bounds;
  const md = Math.max(size.x, size.y, size.z, 1e-6);
  switch (mode) {
    case MODE_CYLINDRICAL: {
      const r = Math.max(settings.cylinderRadius ?? Math.max(size.x, size.y) * 0.5, 1e-6);
      const C = TWO_PI * r;
      return { refU: C, refV: C };
    }
    case MODE_SPHERICAL: {
      const R = Math.max(0.5 * md, 1e-6);
      return { refU: TWO_PI * R, refV: Math.PI * R };
    }
    default:
      return { refU: md, refV: md };
  }
}

/** Convert absolute mm tile sizes to the mode's relative scale factors. */
export function scaleMmToRelative(mode, settings, bounds) {
  const { refU, refV } = getScaleReferenceLengths(mode, settings, bounds);
  const u = Math.max(Number(settings.scaleU) || 1e-6, 1e-6) / refU;
  const v = Math.max(Number(settings.scaleV) || 1e-6, 1e-6) / refV;
  return { u, v };
}
const CUBIC_AXIS_EPSILON = 1e-4;

export function getDominantCubicAxis(normal) {
  const ax = Math.abs(normal.x);
  const ay = Math.abs(normal.y);
  const az = Math.abs(normal.z);

  // Treat near-ties as an intentional tie so 45° faces pick one stable axis
  // instead of flipping projection due to tiny normal jitter between triangles.
  if (ax >= ay - CUBIC_AXIS_EPSILON && ax >= az - CUBIC_AXIS_EPSILON) return 'x';
  if (ay >= az - CUBIC_AXIS_EPSILON) return 'y';
  return 'z';
}

export function getCubicBlendWeights(normal, blend, seamBandWidth = 0.35) {
  const axis = getDominantCubicAxis(normal);
  const ax = Math.abs(normal.x);
  const ay = Math.abs(normal.y);
  const az = Math.abs(normal.z);
  const primary = axis === 'x' ? ax : axis === 'y' ? ay : az;
  const secondary = axis === 'x' ? Math.max(ay, az) : axis === 'y' ? Math.max(ax, az) : Math.max(ax, ay);

  // blend=0: hard one-hot for sharp seams. Do NOT also short-circuit at the
  // primary≈secondary tie when blend>0 — the smooth weight branch handles
  // an exact 45° normal correctly (it produces 0.5/0.5), and short-circuiting
  // to one-hot there creates a single-vertex spike on fillets where the
  // smooth normal sweeps continuously across the cube-face boundary.
  if (blend <= 0.001) {
    return {
      x: axis === 'x' ? 1 : 0,
      y: axis === 'y' ? 1 : 0,
      z: axis === 'z' ? 1 : 0,
    };
  }

  const oneHot = {
    x: axis === 'x' ? 1 : 0,
    y: axis === 'y' ? 1 : 0,
    z: axis === 'z' ? 1 : 0,
  };

  // Only blend inside a seam band around the cube-face boundary. This keeps
  // strongly dominant faces fully textured even when the slider is barely on.
  const seamWidth = Math.max(seamBandWidth, CUBIC_AXIS_EPSILON * 2);
  const seamMixRaw = 1 - Math.min(1, Math.max(0, (primary - secondary) / seamWidth));
  const seamMix = blend * seamMixRaw * seamMixRaw * (3 - 2 * seamMixRaw);
  if (seamMix <= 0.001) return oneHot;

  // blend=1 should produce a genuinely soft triplanar-style transition.
  // Lower blend values progressively sharpen the weights back toward a single
  // dominant axis without snapping until the slider reaches zero.
  const power = 1 + (1 - seamMix) * 11;
  const sx = Math.pow(ax, power);
  const sy = Math.pow(ay, power);
  const sz = Math.pow(az, power);
  const smoothSum = sx + sy + sz + 1e-6;
  const smooth = {
    x: sx / smoothSum,
    y: sy / smoothSum,
    z: sz / smoothSum,
  };

  const mx = oneHot.x * (1 - seamMix) + smooth.x * seamMix;
  const my = oneHot.y * (1 - seamMix) + smooth.y * seamMix;
  const mz = oneHot.z * (1 - seamMix) + smooth.z * seamMix;
  const sum = mx + my + mz;

  return {
    x: mx / sum,
    y: my / sum,
    z: mz / sum,
  };
}

/**
 * Compute normalised UV coordinates [0, 1) (tiling) for a vertex.
 *
 * @param {{ x:number, y:number, z:number }} pos      vertex position
 * @param {{ x:number, y:number, z:number }} normal   vertex normal (unit)
 * @param {number}  mode    one of the MODE_* constants
 * @param {{ scaleU:number, scaleV:number, offsetU:number, offsetV:number }} settings
 * @param {{ min, max, center, size }} bounds           THREE.Vector3 fields
 * @returns {{ u:number, v:number }}                    tiled UV after scale+offset
 */
export function computeUV(pos, normal, mode, settings, bounds) {
  const { min, size, center } = bounds;
  // Compensate for non-square textures: divide scale by aspect correction
  // so equal world-space distances produce equal physical texture distances.
  const aU = settings.textureAspectU ?? 1;
  const aV = settings.textureAspectV ?? 1;
  // settings.scaleU/scaleV are absolute mm — convert to the relative factors
  // the normalized-coordinate math below expects.
  const rel = scaleMmToRelative(mode, settings, bounds);
  const scaleU = rel.u / aU;
  const scaleV = rel.v / aV;
  const { offsetU, offsetV } = settings;
  const rotRad = (settings.rotation ?? 0) * Math.PI / 180;
  const cosR = Math.cos(rotRad);
  const sinR = Math.sin(rotRad);
  const md = Math.max(size.x, size.y, size.z, 1e-6);

  let u = 0, v = 0;

  switch (mode) {

    case MODE_PLANAR_XY: {
      u = (pos.x - min.x) / md;
      v = (pos.y - min.y) / md;
      break;
    }

    case MODE_PLANAR_XZ: {
      u = (pos.x - min.x) / md;
      v = (pos.z - min.z) / md;
      break;
    }

    case MODE_PLANAR_YZ: {
      u = (pos.y - min.y) / md;
      v = (pos.z - min.z) / md;
      break;
    }

    case MODE_CYLINDRICAL: {
      // mappingBlend=0 → pure side projection for all faces (original behaviour, no cap seam).
      // mappingBlend>0 → smooth side↔cap blend.
      // Cylinder axis is +Z. Center XY and radius default to the AABB but can
      // be overridden so partial cylinders (pie slices) project undistorted.
      const cx = settings.cylinderCenterX ?? center.x;
      const cy = settings.cylinderCenterY ?? center.y;
      const r  = Math.max(settings.cylinderRadius ?? Math.max(size.x, size.y) * 0.5, 1e-6);
      const C  = TWO_PI * r;
      const rx = pos.x - cx;
      const ry = pos.y - cy;
      const blend = settings.mappingBlend ?? 0.0;
      const theta = Math.atan2(ry, rx);
      const uRaw = (theta / TWO_PI) + 0.5;
      const vSide = (pos.z - min.z) / C;

      // Seam smoothing: cross-fade between left-side and right-side texture
      // continuations at the atan2 wrap. Both sides use smoothly varying UVs
      // (shifted by ±1.0 in raw space), preserving full texture detail.
      const seamBand = (settings.seamBandWidth ?? 0.5) * 0.1;
      const seamDist = Math.min(uRaw, 1.0 - uRaw);
      const inSeamZone = seamBand > 0.001 && seamDist < seamBand;

      let sideSamples;
      if (inSeamZone) {
        const d = uRaw < 0.5 ? uRaw : uRaw - 1.0;
        const tRaw = (d + seamBand) / (2.0 * seamBand);
        const t = tRaw * tRaw * (3 - 2 * tRaw); // smoothstep
        const tLeft  = applyTransform(1.0 + d, vSide, scaleU, scaleV, offsetU, offsetV, cosR, sinR);
        const tRight = applyTransform(d,       vSide, scaleU, scaleV, offsetU, offsetV, cosR, sinR);
        sideSamples = [
          { u: tRight.u, v: tRight.v, w: t },
          { u: tLeft.u,  v: tLeft.v,  w: 1 - t },
        ];
      } else {
        const tSide = applyTransform(uRaw, vSide, scaleU, scaleV, offsetU, offsetV, cosR, sinR);
        sideSamples = [{ u: tSide.u, v: tSide.v, w: 1 }];
      }

      if (blend <= 0.001) {
        if (sideSamples.length === 1 && sideSamples[0].w === 1) return sideSamples[0];
        return { triplanar: true, samples: sideSamples };
      }

      const capThreshold = Math.cos((settings.capAngle ?? 20) * Math.PI / 180);
      const blendHalf = (settings.seamBandWidth ?? 0.5) * 0.5;
      const absnz = Math.abs(normal.z);
      const capRaw = Math.max(0, Math.min(1, (absnz - (capThreshold - blendHalf)) / (2 * blendHalf + 1e-6)));
      const capW = capRaw * capRaw * (3 - 2 * capRaw); // smoothstep, as in the preview shader

      if (capW <= 0) {
        if (sideSamples.length === 1 && sideSamples[0].w === 1) return sideSamples[0];
        return { triplanar: true, samples: sideSamples };
      }

      const uCap  = rx / C + 0.5;
      const vCap  = ry / C + 0.5;
      const tCap = applyTransform(uCap, vCap, scaleU, scaleV, offsetU, offsetV, cosR, sinR);

      if (capW >= 1) {
        return tCap;
      }

      // Combine seam-blended side samples with cap sample
      const samples = sideSamples.map(s => ({ u: s.u, v: s.v, w: s.w * (1 - capW) }));
      samples.push({ u: tCap.u, v: tCap.v, w: capW });
      return { triplanar: true, samples };
    }

    case MODE_SPHERICAL: {
      const rx = pos.x - center.x;
      const ry = pos.y - center.y;
      const rz = pos.z - center.z;
      const r  = Math.sqrt(rx*rx + ry*ry + rz*rz);
      const phi   = Math.acos(Math.max(-1, Math.min(1, rz / Math.max(r, 1e-6)))); // [0, PI], Z is up
      const theta = Math.atan2(ry, rx);              // [-PI, PI]
      const uRaw = (theta / TWO_PI) + 0.5;
      const vRaw = phi / Math.PI;

      // Seam smoothing: cross-fade at the atan2 wrap
      const seamBand = (settings.seamBandWidth ?? 0.5) * 0.1;
      const seamDist = Math.min(uRaw, 1.0 - uRaw);
      if (seamBand > 0.001 && seamDist < seamBand) {
        const d = uRaw < 0.5 ? uRaw : uRaw - 1.0;
        const tRaw = (d + seamBand) / (2.0 * seamBand);
        const t = tRaw * tRaw * (3 - 2 * tRaw); // smoothstep
        const tLeft  = applyTransform(1.0 + d, vRaw, scaleU, scaleV, offsetU, offsetV, cosR, sinR);
        const tRight = applyTransform(d,       vRaw, scaleU, scaleV, offsetU, offsetV, cosR, sinR);
        return {
          triplanar: true,
          samples: [
            { u: tRight.u, v: tRight.v, w: t },
            { u: tLeft.u,  v: tLeft.v,  w: 1 - t },
          ],
        };
      }

      u = uRaw;
      v = vRaw;
      break;
    }

    case MODE_CUBIC: {
      const weights = getCubicBlendWeights(normal, settings.mappingBlend ?? 0.0, settings.seamBandWidth ?? 0.35);
      // Flip U based on normal sign so opposite faces show correct (non-mirrored) text.
      // Derived from camera right = view_dir × up_dir for each face orientation (Z-up).
      let yzU = (pos.y - min.y) / md;
      if (normal.x < 0) yzU = -yzU;
      let xzU = (pos.x - min.x) / md;
      if (normal.y > 0) xzU = -xzU;
      let xyU = (pos.x - min.x) / md;
      if (normal.z < 0) xyU = -xyU;
      const tYZ = applyTransform(yzU, (pos.z - min.z) / md, scaleU, scaleV, offsetU, offsetV, cosR, sinR);
      const tXZ = applyTransform(xzU, (pos.z - min.z) / md, scaleU, scaleV, offsetU, offsetV, cosR, sinR);
      const tXY = applyTransform(xyU, (pos.y - min.y) / md, scaleU, scaleV, offsetU, offsetV, cosR, sinR);

      if (weights.x > 0.999) return tYZ;
      if (weights.y > 0.999) return tXZ;
      if (weights.z > 0.999) return tXY;

      return {
        triplanar: true,
        samples: [
          { u: tXY.u, v: tXY.v, w: weights.z },
          { u: tXZ.u, v: tXZ.v, w: weights.y },
          { u: tYZ.u, v: tYZ.v, w: weights.x },
        ],
      };
    }

    case MODE_TRIPLANAR:
    default: {
      // World-space normal blending
      const ax = Math.abs(normal.x);
      const ay = Math.abs(normal.y);
      const az = Math.abs(normal.z);
      const ax2 = ax * ax, ay2 = ay * ay, az2 = az * az;
      const bx = ax2 * ax2;
      const by = ay2 * ay2;
      const bz = az2 * az2;
      const sum = bx + by + bz + 1e-6;
      const wx = bx / sum;
      const wy = by / sum;
      const wz = bz / sum;

      // Flip U based on normal sign so opposite faces show correct (non-mirrored) text.
      let yzU = (pos.y - min.y) / md;
      if (normal.x < 0) yzU = -yzU;
      let xzU = (pos.x - min.x) / md;
      if (normal.y > 0) xzU = -xzU;
      let xyU = (pos.x - min.x) / md;
      if (normal.z < 0) xyU = -xyU;
      const uvXY = {
        u: xyU,
        v: (pos.y - min.y) / md,
        w: wz,
      };
      const uvXZ = {
        u: xzU,
        v: (pos.z - min.z) / md,
        w: wy,
      };
      const uvYZ = {
        u: yzU,
        v: (pos.z - min.z) / md,
        w: wx,
      };

      // Apply scale+offset+rotation and tile each independently
      return {
        triplanar: true,
        samples: [
          { ...applyTransform(uvXY.u, uvXY.v, scaleU, scaleV, offsetU, offsetV, cosR, sinR), w: uvXY.w },
          { ...applyTransform(uvXZ.u, uvXZ.v, scaleU, scaleV, offsetU, offsetV, cosR, sinR), w: uvXZ.w },
          { ...applyTransform(uvYZ.u, uvYZ.v, scaleU, scaleV, offsetU, offsetV, cosR, sinR), w: uvYZ.w },
        ],
      };
    }
  }

  return applyTransform(u, v, scaleU, scaleV, offsetU, offsetV, cosR, sinR);
}

// ── Stamp ("place once") ─────────────────────────────────────────────────────
// A stamp layer shows its image exactly once instead of tiling: centred on a
// picked surface point, sized by scaleU/scaleV (mm, aspect-corrected like a
// tile), rotated about its centre, and only on surfaces that face the stamp
// side. Outside that footprint the layer has zero coverage — it neither
// displaces nor covers the layers below. Mirrored by stampFrame /
// stampCoverAt in the GLSL of previewMaterial.js.

// Facing fade: coverage ramps from 0 at FACING_LO (≈ 81° from the projection
// axis) to 1 at FACING_HI (≈ 73°), so the stamp fades out instead of smearing
// down walls the projection hits edge-on.
export const STAMP_FACING_LO = 0.15;
export const STAMP_FACING_HI = 0.30;

/** True when this layer's settings place the image once (planar and cylindrical modes only). */
export function isStampActive(settings) {
  return !!settings.stampOnce && settings.mappingMode >= MODE_PLANAR_XY && settings.mappingMode <= MODE_CYLINDRICAL;
}

/**
 * Default stamp side per mode: the side each projection shows unmirrored
 * (top for XY, front (−Y) for XZ, +X for YZ, outside for cylindrical).
 */
export function defaultStampSide(mode) {
  return mode === MODE_PLANAR_XZ ? -1 : 1;
}

/**
 * Per-layer stamp placement, resolved once: centre (model mm), facing side,
 * footprint size (mm), rotation and the cylinder axis. A null centre falls
 * back to the bounds centre — for cylindrical, the front (−Y) point of the
 * projection cylinder at mid height.
 */
export function stampFrame(settings, bounds) {
  const mode = settings.mappingMode;
  const ccx = settings.cylinderCenterX ?? bounds.center.x;
  const ccy = settings.cylinderCenterY ?? bounds.center.y;
  const r   = Math.max(settings.cylinderRadius ?? Math.max(bounds.size.x, bounds.size.y) * 0.5, 1e-6);
  let cx = settings.stampCenterX, cy = settings.stampCenterY, cz = settings.stampCenterZ;
  if (cx == null || cy == null || cz == null) {
    cx = bounds.center.x; cy = bounds.center.y; cz = bounds.center.z;
    if (mode === MODE_CYLINDRICAL) { cx = ccx; cy = ccy - r; }
  }
  const side = (settings.stampSide === 1 || settings.stampSide === -1) ? settings.stampSide : defaultStampSide(mode);
  const rotRad = (settings.rotation ?? 0) * Math.PI / 180;
  return {
    mode, cx, cy, cz, side, ccx, ccy, r,
    sizeU: Math.max(Number(settings.scaleU) || 1e-6, 1e-6) / (settings.textureAspectU ?? 1),
    sizeV: Math.max(Number(settings.scaleV) || 1e-6, 1e-6) / (settings.textureAspectV ?? 1),
    cosR: Math.cos(rotRad), sinR: Math.sin(rotRad),
    // Smooth Mask: the layer's falloff distance (mm) and curve also fade the
    // stamp's own border.
    fade: Math.max(Number(settings.boundaryFalloff) || 0, 0),
    fadeCurve: settings.boundaryFalloffCurve ?? 'linear',
  };
}

/** The Smooth Mask ramp shapes — must match applyFalloffCurve in main.js and the shader. */
function _falloffCurve(t, mode) {
  if (mode === 'scurve') return t * t * (3 - 2 * t);
  if (mode === 'ease')   return t * t;
  return t;
}

/**
 * Stamp UV at a point: { u, v } in [0, 1] inside the footprint (not wrapped)
 * and `cover` in [0, 1] — 0 outside the footprint or on surfaces facing away,
 * ramping up from the footprint's edge when the layer has a Smooth Mask.
 * Mirroring follows the triplanar convention, so the image reads correctly
 * from whichever side it is placed on.
 */
export function stampUV(pos, normal, frame) {
  const ab = _stampAB(pos.x, pos.y, pos.z, frame, _ab);
  const facing = _stampFacing(normal.x, normal.y, normal.z, pos.x, pos.y, pos.z, frame);
  const ra = frame.cosR * ab.a - frame.sinR * ab.b;
  const rb = frame.sinR * ab.a + frame.cosR * ab.b;
  const u = ra / frame.sizeU + 0.5;
  const v = rb / frame.sizeV + 0.5;
  const inside = u >= 0 && u <= 1 && v >= 0 && v <= 1;
  let cover = 0;
  if (inside) {
    const t = Math.max(0, Math.min(1, (facing - STAMP_FACING_LO) / (STAMP_FACING_HI - STAMP_FACING_LO)));
    cover = t * t * (3 - 2 * t);
    if (frame.fade > 0) {
      // Distance (mm) to the nearest footprint edge, in the stamp's own frame.
      const edge = Math.min(Math.min(u, 1 - u) * frame.sizeU, Math.min(v, 1 - v) * frame.sizeV);
      cover *= _falloffCurve(Math.min(1, edge / frame.fade), frame.fadeCurve);
    }
  }
  return { u, v, cover };
}

const _ab = { a: 0, b: 0, dth: 0 };

/**
 * Stamp-plane coordinates (mm, mirrored for the side, before rotation) of a
 * point: planar axes, or arc length at the projection radius and height for
 * cylindrical (dth = angle from the centre, in (−π, π]).
 */
function _stampAB(x, y, z, frame, out) {
  const { mode, side } = frame;
  let a, b;
  if (mode === MODE_PLANAR_XY) {
    a = x - frame.cx; b = y - frame.cy;
    if (side < 0) a = -a;
  } else if (mode === MODE_PLANAR_XZ) {
    a = x - frame.cx; b = z - frame.cz;
    if (side > 0) a = -a;
  } else if (mode === MODE_PLANAR_YZ) {
    a = y - frame.cy; b = z - frame.cz;
    if (side < 0) a = -a;
  } else {
    const rpx = x - frame.ccx, rpy = y - frame.ccy;
    const rcx = frame.cx - frame.ccx, rcy = frame.cy - frame.ccy;
    out.dth = Math.atan2(rcx * rpy - rcy * rpx, rcx * rpx + rcy * rpy);
    a = out.dth * frame.r; b = z - frame.cz;
    if (side < 0) a = -a;
  }
  out.a = a; out.b = b;
  return out;
}

/** Cosine between a normal and the stamp axis on the stamp side (outward radial for cylindrical). */
function _stampFacing(nx, ny, nz, x, y, z, frame) {
  const { mode, side } = frame;
  const nLen = Math.hypot(nx, ny, nz) || 1;
  let facing;
  if (mode === MODE_PLANAR_XY) facing = nz * side;
  else if (mode === MODE_PLANAR_XZ) facing = ny * side;
  else if (mode === MODE_PLANAR_YZ) facing = nx * side;
  else {
    const rpx = x - frame.ccx, rpy = y - frame.ccy;
    const lp = Math.hypot(rpx, rpy);
    facing = lp > 1e-6 ? (nx * rpx + ny * rpy) / lp * side : 0;
  }
  return facing / nLen;
}

/**
 * Per-triangle flags for a non-indexed position array: 1 where the stamp
 * cannot cover any point of the triangle, so the export can skip refining it
 * and Preserve Untextured keeps it as it is. A face counts as reached when
 * the bounding box of its stamp coordinates overlaps the footprint — widened
 * by the Smooth Mask distance, so the mask boundary these faces create never
 * ramps the stamp itself down — and one of its corners touches a face that
 * points toward the stamp. Faces straddling the cylinder axis or its back
 * seam count as overlapping. Every corner of an unreached face has zero
 * coverage (it is outside the footprint, or all its faces point away), so
 * masking these faces never changes the displacement.
 */
export function stampUntouchedFaces(pos, frame) {
  const triCount = pos.length / 9;
  const cyl = frame.mode === MODE_CYLINDRICAL;
  const ab = { a: 0, b: 0, dth: 0 };
  const cr = frame.cosR, sr = frame.sinR;
  const margin = frame.fade * 1.25;
  const mu = margin / frame.sizeU, mv = margin / frame.sizeV;

  // Welded corners; per corner the most stamp-facing of its faces.
  const welded = new QuantizedPointMap(1e5, Math.min(triCount * 3, 1 << 22));
  const vid = new Int32Array(triCount * 3);
  let next = 0;
  for (let i = 0; i < triCount * 3; i++) {
    vid[i] = welded.getOrSet(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2], next);
    if (welded.inserted) next++;
  }
  const facesToward = new Uint8Array(next);
  for (let t = 0; t < triCount; t++) {
    const o = t * 9;
    const ax = pos[o], ay = pos[o + 1], az = pos[o + 2];
    const e1x = pos[o + 3] - ax, e1y = pos[o + 4] - ay, e1z = pos[o + 5] - az;
    const e2x = pos[o + 6] - ax, e2y = pos[o + 7] - ay, e2z = pos[o + 8] - az;
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    const facing = _stampFacing(nx, ny, nz,
      ax + (e1x + e2x) / 3, ay + (e1y + e2y) / 3, az + (e1z + e2z) / 3, frame);
    if (facing > 0) { facesToward[vid[t * 3]] = 1; facesToward[vid[t * 3 + 1]] = 1; facesToward[vid[t * 3 + 2]] = 1; }
  }

  const out = new Uint8Array(triCount);
  for (let t = 0; t < triCount; t++) {
    if (!facesToward[vid[t * 3]] && !facesToward[vid[t * 3 + 1]] && !facesToward[vid[t * 3 + 2]]) { out[t] = 1; continue; }
    const o = t * 9;
    let amin = Infinity, amax = -Infinity, bmin = Infinity, bmax = -Infinity;
    let thMin = Infinity, thMax = -Infinity;
    for (let k = 0; k < 3; k++) {
      _stampAB(pos[o + k * 3], pos[o + k * 3 + 1], pos[o + k * 3 + 2], frame, ab);
      if (ab.a < amin) amin = ab.a; if (ab.a > amax) amax = ab.a;
      if (ab.b < bmin) bmin = ab.b; if (ab.b > bmax) bmax = ab.b;
      if (cyl) { if (ab.dth < thMin) thMin = ab.dth; if (ab.dth > thMax) thMax = ab.dth; }
    }
    if (cyl && (thMax - thMin > Math.PI || _triContainsAxis(pos, o, frame.ccx, frame.ccy))) continue;
    // Rotate the (a, b) box's corners: bounds the face's stamp UV even where
    // the cylindrical mapping bends its edges.
    let umin = Infinity, umax = -Infinity, vmin = Infinity, vmax = -Infinity;
    for (let c = 0; c < 4; c++) {
      const a = c & 1 ? amax : amin, b = c & 2 ? bmax : bmin;
      const u = (cr * a - sr * b) / frame.sizeU + 0.5;
      const v = (sr * a + cr * b) / frame.sizeV + 0.5;
      if (u < umin) umin = u; if (u > umax) umax = u;
      if (v < vmin) vmin = v; if (v > vmax) vmax = v;
    }
    if (umax < -mu || umin > 1 + mu || vmax < -mv || vmin > 1 + mv) out[t] = 1;
  }
  return out;
}

/** True when the XY projection of triangle o (offset into pos) contains the point (cx, cy). */
function _triContainsAxis(pos, o, cx, cy) {
  const d = (x1, y1, x2, y2) => (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1);
  const d1 = d(pos[o], pos[o + 1], pos[o + 3], pos[o + 4]);
  const d2 = d(pos[o + 3], pos[o + 4], pos[o + 6], pos[o + 7]);
  const d3 = d(pos[o + 6], pos[o + 7], pos[o], pos[o + 1]);
  return !((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0));
}

function applyTransform(u, v, scaleU, scaleV, offsetU, offsetV, cosR, sinR) {
  let uu = u / scaleU + offsetU;
  let vv = v / scaleV + offsetV;
  if (cosR !== 1 || sinR !== 0) {
    uu -= 0.5; vv -= 0.5;
    const ru = cosR * uu - sinR * vv;
    const rv = sinR * uu + cosR * vv;
    uu = ru + 0.5; vv = rv + 0.5;
  }
  return { triplanar: false, u: fract(uu), v: fract(vv) };
}

/** Fractional part, always positive (mirrors GLSL fract) */
function fract(x) { return x - Math.floor(x); }
