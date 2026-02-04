import type { ObjModel } from "@shared/schema";
import { v3 } from "./math";

type Vec3 = [number, number, number];
type Vec2 = [number, number];

function parseIndex(token: string, count: number) {
  const i = parseInt(token, 10);
  if (Number.isNaN(i)) return 0;
  return i < 0 ? count + i : i - 1;
}

export function parseObj(objText: string): ObjModel {
  const positions: Vec3[] = [];
  const normals: Vec3[] = [];
  const uvs: Vec2[] = [];

  // expanded vertices
  const outPos: number[] = [];
  const outNor: number[] = [];
  const outUv: number[] = [];
  const outIdx: number[] = [];

  const vertMap = new Map<string, number>();

  const lines = objText.split(/\r?\n/);
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const parts = line.split(/\s+/);
    const head = parts[0];

    if (head === "v") {
      positions.push([parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3])]);
    } else if (head === "vn") {
      normals.push([parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3])]);
    } else if (head === "vt") {
      uvs.push([parseFloat(parts[1]), parseFloat(parts[2])]);
    } else if (head === "f") {
      const face = parts.slice(1);
      // triangulate fan
      for (let i = 1; i + 1 < face.length; i++) {
        const tri = [face[0], face[i], face[i + 1]];
        for (const vtx of tri) {
          const key = vtx;
          let idx = vertMap.get(key);
          if (idx === undefined) {
            const [p, t, n] = vtx.split("/");
            const pi = parseIndex(p, positions.length);
            const ti = t ? parseIndex(t, uvs.length) : -1;
            const ni = n ? parseIndex(n, normals.length) : -1;

            const pv = positions[pi] ?? v3();
            const tv = ti >= 0 ? (uvs[ti] ?? [0, 0]) : ([0, 0] as const);
            const nv = ni >= 0 ? (normals[ni] ?? [0, 1, 0]) : ([0, 1, 0] as const);

            idx = outPos.length / 3;
            vertMap.set(key, idx);

            outPos.push(pv[0], pv[1], pv[2]);
            outUv.push(tv[0], tv[1]);
            outNor.push(nv[0], nv[1], nv[2]);
          }
          outIdx.push(idx);
        }
      }
    }
  }

  // bounds
  let min: Vec3 = [Infinity, Infinity, Infinity];
  let max: Vec3 = [-Infinity, -Infinity, -Infinity];

  for (let i = 0; i < outPos.length; i += 3) {
    const x = outPos[i];
    const y = outPos[i + 1];
    const z = outPos[i + 2];
    min = [Math.min(min[0], x), Math.min(min[1], y), Math.min(min[2], z)];
    max = [Math.max(max[0], x), Math.max(max[1], y), Math.max(max[2], z)];
  }

  return {
    positions: new Float32Array(outPos),
    normals: new Float32Array(outNor),
    uvs: new Float32Array(outUv),
    indices: new Uint32Array(outIdx),
    bounds: { min, max },
  };
}
