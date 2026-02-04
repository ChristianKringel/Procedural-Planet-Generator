import { hashStringToUint, mulberry32 } from "./random";
import type { NoiseType } from "@shared/schema";

// Lightweight procedural noise (no external libs).
// - perlin: classic gradient noise (2D) sampled on sphere via lat/long
// - simplex: "simplex-ish" value noise w/ skewing (approx) — good enough visually
// - random: hash-based value noise

function fade(t: number) {
  return t * t * t * (t * (t * 6 - 15) + 10);
}
function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function hash2i(x: number, y: number, seed: number) {
  // integer hash -> [0,1)
  let h = seed ^ (x * 374761393) ^ (y * 668265263);
  h = (h ^ (h >>> 13)) * 1274126177;
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

function grad2(x: number, y: number, seed: number) {
  const a = hash2i(x, y, seed) * Math.PI * 2;
  return [Math.cos(a), Math.sin(a)] as const;
}

function perlin2(x: number, y: number, seed: number) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = x0 + 1;
  const y1 = y0 + 1;

  const sx = fade(x - x0);
  const sy = fade(y - y0);

  const g00 = grad2(x0, y0, seed);
  const g10 = grad2(x1, y0, seed);
  const g01 = grad2(x0, y1, seed);
  const g11 = grad2(x1, y1, seed);

  const dx0 = x - x0;
  const dy0 = y - y0;
  const dx1 = x - x1;
  const dy1 = y - y1;

  const n00 = g00[0] * dx0 + g00[1] * dy0;
  const n10 = g10[0] * dx1 + g10[1] * dy0;
  const n01 = g01[0] * dx0 + g01[1] * dy1;
  const n11 = g11[0] * dx1 + g11[1] * dy1;

  const ix0 = lerp(n00, n10, sx);
  const ix1 = lerp(n01, n11, sx);
  const v = lerp(ix0, ix1, sy);

  // normalize-ish to [-1,1]
  return v * 1.6;
}

function valueNoise2(x: number, y: number, seed: number) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = x0 + 1;
  const y1 = y0 + 1;

  const sx = fade(x - x0);
  const sy = fade(y - y0);

  const v00 = hash2i(x0, y0, seed) * 2 - 1;
  const v10 = hash2i(x1, y0, seed) * 2 - 1;
  const v01 = hash2i(x0, y1, seed) * 2 - 1;
  const v11 = hash2i(x1, y1, seed) * 2 - 1;

  const ix0 = lerp(v00, v10, sx);
  const ix1 = lerp(v01, v11, sx);
  return lerp(ix0, ix1, sy);
}

function fbm2(
  base: (x: number, y: number, seed: number) => number,
  x: number,
  y: number,
  seed: number,
  octaves = 5,
) {
  let amp = 0.5;
  let freq = 1;
  let sum = 0;
  let norm = 0;
  for (let i = 0; i < octaves; i++) {
    sum += amp * base(x * freq, y * freq, seed + i * 1013);
    norm += amp;
    amp *= 0.5;
    freq *= 2;
  }
  return sum / (norm || 1);
}

export function makeNoiseSampler(seedStr: string, type: NoiseType) {
  const seed = hashStringToUint(seedStr);
  const rng = mulberry32(seed ^ 0x9e3779b9);

  return function sample(nlat: number, nlon: number) {
    // nlat: [-pi/2..pi/2], nlon: [-pi..pi]
    // Ensure seamless wrapping at longitude boundaries
    // Map to [0, 1] range with proper wrapping
    let u = (nlon / (Math.PI * 2)) + 0.5;
    const v = (nlat / Math.PI) + 0.5;
    
    // Normalize u to [0, 1) to ensure continuity
    u = u - Math.floor(u);

    // Scale for noise sampling
    const x = u * 6.0;
    const y = v * 3.5;

    if (type === "perlin") {
      return fbm2(perlin2, x, y, seed, 6);
    }
    if (type === "simplex") {
      // "simplex-ish": skew coordinates to create diagonal features; fbm on value noise
      const s = (x + y) * 0.3660254;
      const xs = x + s;
      const ys = y + s;
      return fbm2(valueNoise2, xs, ys, seed ^ 0x51ed270b, 6);
    }

    // random: hash-based value noise with fewer octaves (chunkier)
    const t = fbm2(valueNoise2, x, y, (seed ^ 0xa2c2a) + Math.floor(rng() * 10000), 3);
    return t;
  };
}
