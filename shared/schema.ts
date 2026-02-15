import { z } from "zod";

export type NoiseType = "perlin" | "simplex" | "random";

export const planetSettingsSchema = z.object({
  seed: z.string().min(1),
  subdivisions: z.number().int().min(20).max(200),
  waterThreshold: z.number().min(-1).max(1),
  noiseStrength: z.number().min(0).max(5),
  noiseType: z.union([z.literal("perlin"), z.literal("simplex"), z.literal("random")]),
  objectCount: z.number().int().min(0).max(2000),
  shadowsEnabled: z.boolean(),
});

export type PlanetSettings = z.infer<typeof planetSettingsSchema>;

export interface PickResult {
  hit: boolean;
  worldPos?: [number, number, number];
  worldNormal?: [number, number, number];
  localDir?: [number, number, number];
  height?: number;
  isWater?: boolean;
  isSnow?: boolean;
}

export interface ObjModel {
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  indices: Uint32Array;
  bounds: {
    min: [number, number, number];
    max: [number, number, number];
  };
}

export type ObjectKind = "tree" | "boat" | "snow_tree";

export interface PlacedObject {
  id: string;
  kind: ObjectKind;
  position: [number, number, number];
  normal: [number, number, number];
  tangent: [number, number, number];
  bitangent: [number, number, number];
  scale: number;
  phase: number;
}
