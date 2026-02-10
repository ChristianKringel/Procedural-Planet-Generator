import { hashStringToUint, mulberry32 } from "./random";
import type { NoiseType } from "@shared/schema";

// Lightweight procedural 3D noise (no external libs).
// Sampled in Cartesian coordinates on the sphere surface to avoid
// longitude-seam and pole-convergence artifacts inherent to 2D lat/lon mapping.

function fade(t: number) {
  return t * t * t * (t * (t * 6 - 15) + 10);
}
function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function hash3i(x: number, y: number, z: number, seed: number) {
  let h = seed ^ (x * 374761393) ^ (y * 668265263) ^ (z * 1013904223);
  h = (h ^ (h >>> 13)) * 1274126177;
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

const GRAD3: readonly (readonly [number, number, number])[] = [
  [1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],
  [1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],
  [0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1],
];

function grad3(ix: number, iy: number, iz: number, seed: number) {
  let h = seed ^ (ix * 374761393) ^ (iy * 668265263) ^ (iz * 1013904223);
  h = (h ^ (h >>> 13)) * 1274126177;
  h ^= h >>> 16;
  return GRAD3[(h >>> 0) % 12];
}

function perlin3(x: number, y: number, z: number, seed: number) {
  const x0 = Math.floor(x), y0 = Math.floor(y), z0 = Math.floor(z);
  const x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
  const sx = fade(x - x0), sy = fade(y - y0), sz = fade(z - z0);
  const dx0 = x - x0, dy0 = y - y0, dz0 = z - z0;
  const dx1 = x - x1, dy1 = y - y1, dz1 = z - z1;

  const g000 = grad3(x0,y0,z0,seed); const n000 = g000[0]*dx0 + g000[1]*dy0 + g000[2]*dz0;
  const g100 = grad3(x1,y0,z0,seed); const n100 = g100[0]*dx1 + g100[1]*dy0 + g100[2]*dz0;
  const g010 = grad3(x0,y1,z0,seed); const n010 = g010[0]*dx0 + g010[1]*dy1 + g010[2]*dz0;
  const g110 = grad3(x1,y1,z0,seed); const n110 = g110[0]*dx1 + g110[1]*dy1 + g110[2]*dz0;
  const g001 = grad3(x0,y0,z1,seed); const n001 = g001[0]*dx0 + g001[1]*dy0 + g001[2]*dz1;
  const g101 = grad3(x1,y0,z1,seed); const n101 = g101[0]*dx1 + g101[1]*dy0 + g101[2]*dz1;
  const g011 = grad3(x0,y1,z1,seed); const n011 = g011[0]*dx0 + g011[1]*dy1 + g011[2]*dz1;
  const g111 = grad3(x1,y1,z1,seed); const n111 = g111[0]*dx1 + g111[1]*dy1 + g111[2]*dz1;

  return lerp(
    lerp(lerp(n000, n100, sx), lerp(n010, n110, sx), sy),
    lerp(lerp(n001, n101, sx), lerp(n011, n111, sx), sy),
    sz,
  ) * 1.4;
}

function valueNoise3(x: number, y: number, z: number, seed: number) {
  const x0 = Math.floor(x), y0 = Math.floor(y), z0 = Math.floor(z);
  const x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
  const sx = fade(x - x0), sy = fade(y - y0), sz = fade(z - z0);

  return lerp(
    lerp(
      lerp(hash3i(x0,y0,z0,seed)*2-1, hash3i(x1,y0,z0,seed)*2-1, sx),
      lerp(hash3i(x0,y1,z0,seed)*2-1, hash3i(x1,y1,z0,seed)*2-1, sx),
      sy,
    ),
    lerp(
      lerp(hash3i(x0,y0,z1,seed)*2-1, hash3i(x1,y0,z1,seed)*2-1, sx),
      lerp(hash3i(x0,y1,z1,seed)*2-1, hash3i(x1,y1,z1,seed)*2-1, sx),
      sy,
    ),
    sz,
  );
}

function fbm3(
  base: (x: number, y: number, z: number, seed: number) => number,
  x: number,
  y: number,
  z: number,
  seed: number,
  octaves = 5,
) {
  let amp = 0.5;
  let freq = 1;
  let sum = 0;
  let norm = 0;
  for (let i = 0; i < octaves; i++) {
    sum += amp * base(x * freq, y * freq, z * freq, seed + i * 1013);
    norm += amp;
    amp *= 0.5;
    freq *= 2;
  }
  return sum / (norm || 1);
}

export function makeNoiseSampler(seedStr: string, type: NoiseType) {
  const seed = hashStringToUint(seedStr);
  const rng = mulberry32(seed ^ 0x9e3779b9);

  return function sample(x: number, y: number, z: number) {
    if (type === "perlin") {
      return fbm3(perlin3, x, y, z, seed, 6);
    }
    if (type === "simplex") {
      const s = (x + y + z) * 0.3660254;
      return fbm3(valueNoise3, x + s, y + s, z + s, seed ^ 0x51ed270b, 6);
    }
    // random: hash-based value noise with fewer octaves (chunkier)
    return fbm3(valueNoise3, x, y, z, (seed ^ 0xa2c2a) + Math.floor(rng() * 10000), 3);
  };
}
