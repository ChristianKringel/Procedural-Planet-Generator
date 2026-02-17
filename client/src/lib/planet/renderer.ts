import type { NoiseType, PickResult, PlanetSettings, PlacedObject, ObjectKind, ObjModel } from "@shared/schema";
import { makeNoiseSampler } from "./noise";
import { parseObj } from "./obj";
import { BOAT_OBJ, TREE_OBJ } from "./models";
import {
  anyPerpendicular,
  clamp,
  cross,
  dot,
  len,
  mat4Identity,
  mat4LookAt,
  mat4Mul,
  mat4Ortho,
  mat4Perspective,
  mat4RotateY,
  mat4Scale,
  mat4Translate,
  mulScalar,
  normalize,
  raySphereIntersect,
  sub,
  type Mat4,
  type Vec3,
  add,
} from "./math";
import { hashStringToUint, mulberry32, randRange } from "./random";

type GL = WebGL2RenderingContext;

function createShader(gl: GL, type: number, source: string) {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, source);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(s);
    gl.deleteShader(s);
    throw new Error(log || "Shader compile error");
  }
  return s;
}
function createProgram(gl: GL, vs: string, fs: string) {
  const p = gl.createProgram()!;
  gl.attachShader(p, createShader(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, createShader(gl, gl.FRAGMENT_SHADER, fs));
  gl.bindAttribLocation(p, 0, "a_position");
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(p);
    gl.deleteProgram(p);
    throw new Error(log || "Program link error");
  }
  return p;
}

function createVao(
  gl: GL,
  prog: WebGLProgram,
  attribs: {
    name: string;
    size: number;
    data: BufferSource;
    type?: number;
    normalized?: boolean;
  }[],
  indices?: BufferSource,
) {
  const vao = gl.createVertexArray()!;
  gl.bindVertexArray(vao);

  for (const a of attribs) {
    const loc = gl.getAttribLocation(prog, a.name);
    if (loc < 0) continue;
    const buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, a.data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(
      loc,
      a.size,
      a.type ?? gl.FLOAT,
      a.normalized ?? false,
      0,
      0,
    );
  }

  let ibo: WebGLBuffer | null = null;
  if (indices) {
    ibo = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
  }

  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  return { vao, ibo };
}

const PLANET_VS = `#version 300 es
precision highp float;

in vec3 a_position;
in vec3 a_normal;
in float a_craterMask;

uniform mat4 u_world;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat4 u_lightVP;

out vec3 v_worldPos;
out vec3 v_worldNormal;
out vec4 v_lightClip;
out float v_height;
out float v_craterMask;

void main() {
  vec4 wp = u_world * vec4(a_position, 1.0);
  v_worldPos = wp.xyz;
  v_worldNormal = mat3(u_world) * a_normal;
  v_lightClip = u_lightVP * wp;
  v_height = length(a_position); // height relative to radius 1.0
  v_craterMask = a_craterMask;
  gl_Position = u_proj * u_view * wp;
}
`;

const PLANET_FS = `#version 300 es
precision highp float;

in vec3 v_worldPos;
in vec3 v_worldNormal;
in vec4 v_lightClip;
in float v_height;
in float v_craterMask;

uniform vec3 u_lightDir;     // direction towards light (sun)
uniform vec3 u_cameraPos;
uniform float u_waterThreshold;
uniform float u_noiseStrength;
uniform float u_time;
uniform bool u_shadowsEnabled;
uniform bool u_debugShadow;

uniform sampler2D u_shadowMap;

out vec4 outColor;

float sat(float x){ return clamp(x, 0.0, 1.0); }

// Simple hash for noise-based texture variation
float hash(vec3 p) {
    p = fract(p * 0.3183099 + .1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

// Multi-octave noise for water depth variation
float waterDepthNoise(vec3 p) {
    // Multiple scales of noise to create natural depth patterns
    float n1 = hash(p * 2.5);        // Large ocean patterns
    float n2 = hash(p * 7.0);        // Medium patterns  
    float n3 = hash(p * 18.0);       // Fine details
    float n4 = hash(p * 45.0);       // Very fine ripples
    
    // Combine with different weights
    return n1 * 0.5 + n2 * 0.25 + n3 * 0.15 + n4 * 0.1;
}

float sampleShadow(vec4 lightClip, vec3 normal, vec3 lightDir, float craterMask) {
  vec3 proj = lightClip.xyz / lightClip.w;
  vec2 uv = proj.xy * 0.5 + 0.5;
  float current = proj.z * 0.5 + 0.5;

  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1.0;

  // Surfaces facing away from the light get no shadow lookup.
  // The orthographic projection overlaps both hemispheres, so the
  // near-hemisphere front face is stored in the shadow map.  Back-facing
  // fragments on the far hemisphere would always fail the depth test and
  // appear fully shadowed.  We fade the shadow contribution to zero for
  // those fragments so that darkening is handled solely by ndl.
  float cosTheta = sat(dot(normal, -lightDir));
  if (cosTheta < 0.01) return 1.0;
  float shadowFade = smoothstep(0.01, 0.1, cosTheta);

  // Shadow map stores back-face depth of the planet (~0.74 in NDC).
  // Objects are rendered with a small lift so their depth (~0.10) is
  // reliably shallower than the back face.  Standard comparison:
  // if current > depth → in shadow.
  float bias = 0.0005;

  float shadow = 0.0;
  vec2 texel = 1.0 / vec2(textureSize(u_shadowMap, 0));
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      float depth = texture(u_shadowMap, uv + vec2(float(x), float(y)) * texel).r;
      shadow += (current - bias > depth) ? 0.0 : 1.0;
    }
  }
  shadow = shadow / 9.0;

  // Blend shadow out near terminator to avoid hard edge
  shadow = mix(1.0, shadow, shadowFade);

  return shadow;
}

void main() {
  vec3 N = normalize(v_worldNormal);
  vec3 L = normalize(-u_lightDir);
  vec3 V = normalize(u_cameraPos - v_worldPos);

  float ndl = sat(dot(N, L));
  float rim = pow(sat(1.0 - dot(N, V)), 2.2);

  // Coloração baseada em altura e threshold
  float h = (v_height - 1.0) * 10.0; // scale for threshold comparison
  float threshold = u_waterThreshold;

  vec3 color;
  float landMask = 0.0;

  // Gradientes de biomas
  if (v_craterMask > 0.01) {
    // CRATERA — rocha exposta / terra queimada
    float cm = v_craterMask;
    vec3 outerRock = vec3(0.45, 0.38, 0.32);    // borda: rocha clara
    vec3 innerRock = vec3(0.35, 0.28, 0.22);    // meio: rocha média
    vec3 deepCenter = vec3(0.20, 0.14, 0.10);   // fundo: terra escura queimada
    float rockNoise = hash(v_worldPos * 60.0);
    vec3 craterColor = mix(outerRock, innerRock, sat(cm * 2.0));
    craterColor = mix(craterColor, deepCenter, sat(cm * 1.8 - 0.5));
    // Add some rocky texture variation
    craterColor *= 0.85 + 0.3 * rockNoise;
    color = craterColor;
    landMask = 1.0;
  } else if (h < threshold) {
    // ÁGUA - varied colors from shallow beach to deep ocean
    
    // Normalize world position to get consistent noise pattern
    vec3 noisePos = normalize(v_worldPos);
    
    // Get multi-scale depth noise
    float depthNoise = waterDepthNoise(noisePos);
    
    // Distance from shore (how far below threshold)
    // Now with 40% geometric height variation for better depth colors
    float shoreDistance = sat((threshold - h) * 5.0); // 0 at shore, 1 far from shore
    
    // Combine shore distance with noise for natural depth variation
    // Geometry provides the base, noise adds texture
    float apparentDepth = shoreDistance * 0.7 + depthNoise * 0.3;
    
    // Water color gradient from beach to deep ocean
    vec3 beachWater = vec3(0.15, 0.60, 0.85);    // Very shallow - bright cyan
    vec3 shallowWater = vec3(0.08, 0.50, 0.78);  // Shallow - light blue
    vec3 mediumWater = vec3(0.05, 0.38, 0.65);   // Medium depth - blue
    vec3 deepWater = vec3(0.03, 0.25, 0.50);     // Deep - dark blue
    vec3 oceanWater = vec3(0.01, 0.12, 0.30);    // Very deep ocean - very dark blue
    
    // Progressive color mixing based on depth - more aggressive transitions
    vec3 waterColor = mix(beachWater, shallowWater, sat(apparentDepth * 3.0));
    waterColor = mix(waterColor, mediumWater, sat(apparentDepth * 2.5 - 0.1));
    waterColor = mix(waterColor, deepWater, sat(apparentDepth * 2.0 - 0.2));
    waterColor = mix(waterColor, oceanWater, sat(apparentDepth * 1.5 - 0.3));
    
    color = waterColor;

    // Specular highlight na água
    vec3 H_water = normalize(L + V);
    float spec_water = pow(sat(dot(N, H_water)), 120.0) * 0.8;
    color += spec_water * vec3(0.8, 0.9, 1.0);
  } else if (h < threshold + 0.05) {
    // PRAIA
    color = vec3(0.76, 0.70, 0.50);
    landMask = 1.0;
  } else if (h < 0.6) {
    // PLANÍCIE / GRAMA
    float g = sat((h - (threshold + 0.05)) / (0.6 - (threshold + 0.05)));
    vec3 forest = vec3(0.1, 0.3, 0.1);
    vec3 meadow = vec3(0.2, 0.6, 0.2);
    color = mix(meadow, forest, g);

    // Noise de vegetação
    float vegNoise = hash(v_worldPos * 200.0);
    color = mix(color, color * (0.85 + 0.3 * vegNoise), 0.4);
    landMask = 1.0;
  } else if (h < 0.8) {
    // MONTANHA
    float m = sat((h - 0.6) / 0.2);
    vec3 rock = vec3(0.35, 0.35, 0.38);
    vec3 dirt = vec3(0.25, 0.22, 0.20);
    color = mix(dirt, rock, m);
    landMask = 1.0;
  } else {
    // NEVE
    float s = sat((h - 0.8) * 5.0);
    vec3 rockUpper = vec3(0.4, 0.4, 0.42);
    vec3 snow = vec3(0.95, 0.95, 1.0);
    color = mix(rockUpper, snow, s);
    landMask = 1.0;
  }

  if (u_shadowsEnabled) {
    float shadow = sampleShadow(v_lightClip, N, u_lightDir, v_craterMask);

    // DEBUG: visualize raw shadow value
    if (u_debugShadow) {
      // Red = in shadow (shadow=0), Green = lit (shadow=1)
      outColor = vec4(1.0 - shadow, shadow, 0.0, 1.0);
      return;
    }

    // Specular (less pronounced than water)
    vec3 H = normalize(L + V);
    float spec = pow(sat(dot(N, H)), 40.0) * 0.2 * landMask;

    // Craters get full uniform lighting without shadows or directional lighting
    if (v_craterMask > 0.01) {
      vec3 col = color * 0.95; // Bright uniform lighting for craters
      col += rim * vec3(0.4, 0.6, 1.0) * 0.15;
      outColor = vec4(col, 1.0);
    } else {
      // Enhanced shadow contrast for better visibility
      // ambient: 0.2, diffuse with shadow: 0.8
      vec3 col = color * (0.2 + 0.8 * ndl * shadow);
      col += spec * vec3(1.0) * shadow;
      col += rim * vec3(0.4, 0.6, 1.0) * 0.15; // Atmosfera
      outColor = vec4(col, 1.0);
    }
  } else {
    // Without shadows: uniform ambient lighting
    vec3 col = color * 0.85;
    col += rim * vec3(0.4, 0.6, 1.0) * 0.1; // Subtle atmosphere

    outColor = vec4(col, 1.0);
  }
}
`;

const SHADOW_VS = `#version 300 es
precision highp float;

in vec3 a_position;
uniform mat4 u_world;
uniform mat4 u_lightVP;

void main() {
  gl_Position = u_lightVP * u_world * vec4(a_position, 1.0);
}
`;

const SHADOW_FS = `#version 300 es
precision highp float;
out vec4 outColor;
void main() {
  // depth only
  outColor = vec4(1.0);
}
`;

const OBJ_VS = `#version 300 es
precision highp float;

in vec3 a_position;
in vec3 a_normal;

uniform mat4 u_world;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat4 u_lightVP;

out vec3 v_worldPos;
out vec3 v_worldNormal;
out vec4 v_lightClip;
out vec3 v_localPos;

void main() {
  vec4 wp = u_world * vec4(a_position, 1.0);
  v_worldPos = wp.xyz;
  v_worldNormal = mat3(u_world) * a_normal;
  v_lightClip = u_lightVP * wp;
  v_localPos = a_position;
  gl_Position = u_proj * u_view * wp;
}
`;

const OBJ_FS = `#version 300 es
precision highp float;

in vec3 v_worldPos;
in vec3 v_worldNormal;
in vec4 v_lightClip;
in vec3 v_localPos;

uniform vec3 u_lightDir;
uniform vec3 u_cameraPos;
uniform vec3 u_albedo;
uniform vec3 u_albedo2;
uniform vec3 u_albedo3;
uniform float u_isBoat;
uniform bool u_shadowsEnabled;
uniform sampler2D u_shadowMap;

out vec4 outColor;

float sat(float x){ return clamp(x, 0.0, 1.0); }

float sampleShadow(vec4 lightClip, vec3 normal, vec3 lightDir) {
  vec3 proj = lightClip.xyz / lightClip.w;
  vec2 uv = proj.xy * 0.5 + 0.5;
  float current = proj.z * 0.5 + 0.5;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1.0;

  float cosTheta = clamp(dot(normal, -lightDir), 0.0, 1.0);
  float bias = 0.0005 * tan(acos(cosTheta));
  bias = clamp(bias, 0.0001, 0.001);

  // Max shadow distance: shadows fade when caster is far from surface
  // Depth range is ~3.5 units (near=6, far=9.5), objects are ~0.1 tall
  float maxDepthDiff = 0.04;

  vec2 texel = 1.0 / vec2(textureSize(u_shadowMap, 0));
  float shadow = 0.0;
  for (int y=-1; y<=1; y++){
    for (int x=-1; x<=1; x++){
      float depth = texture(u_shadowMap, uv + vec2(float(x), float(y)) * texel).r;
      float diff = current - bias - depth;
      shadow += (diff > 0.0 && diff < maxDepthDiff) ? 0.0 : 1.0;
    }
  }
  shadow = shadow / 9.0;
  
  return shadow;
}

void main() {
  vec3 N = normalize(v_worldNormal);
  vec3 L = normalize(-u_lightDir);
  vec3 V = normalize(u_cameraPos - v_worldPos);

  float ndl = sat(dot(N, L));

  // Multi-color variation for boats based on vertical position
  vec3 baseColor = u_albedo;
  if (u_isBoat > 0.5) {
    float height = v_localPos.y;
    // Lower hull
    if (height < 0.15) {
      baseColor = u_albedo;
    }
    // Mid section / deck
    else if (height < 0.45) {
      baseColor = u_albedo2;
    }
    // Superstructure
    else {
      baseColor = u_albedo3;
    }
  }

  // Objects CAST shadows (drawn in shadow pass) but do NOT receive them.
  // Standard directional lighting only.
  vec3 H = normalize(L + V);
  float spec = pow(sat(dot(N, H)), 80.0);

  vec3 col = baseColor * (0.25 + 0.75 * ndl);
  col += spec * vec3(0.9, 1.0, 1.0) * 0.25;

  outColor = vec4(col, 1.0);
}
`;

const PARTICLE_VS = `#version 300 es
precision highp float;

in vec3 a_position;
in vec4 a_color;
in float a_size;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_screenHeight;

out vec4 v_color;

void main() {
  v_color = a_color;
  vec4 viewPos = u_view * vec4(a_position, 1.0);
  gl_Position = u_proj * viewPos;
  float dist = length(viewPos.xyz);
  gl_PointSize = clamp(a_size * u_screenHeight * 0.5 / max(dist, 0.1), 1.0, 300.0);
}
`;

const PARTICLE_FS = `#version 300 es
precision highp float;

in vec4 v_color;
out vec4 outColor;

void main() {
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float d = dot(uv, uv);
  if (d > 1.0) discard;
  float alpha = v_color.a * smoothstep(1.0, 0.1, d);
  outColor = vec4(v_color.rgb, alpha);
}
`;

function transformVec3(m: Mat4, v: Vec3): Vec3 {
  return [
    m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12],
    m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13],
    m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14],
  ];
}

/**
 * Calculate terrain displacement for a given direction on the sphere.
 * This function applies noise layers and reliefAggressiveness consistently
 * across terrain mesh, object placement, and ray picking.
 * 
 * @param dir - Unit direction vector on the sphere
 * @param noiseBase - Base noise sampler
 * @param noiseDetail - Detail noise sampler
 * @param noiseMicro - Micro noise sampler
 * @param settings - Planet settings including noiseStrength, reliefAggressiveness, waterThreshold
 * @returns Displacement value (added to base radius 1.0)
 */
function calculateDisplacement(
  dir: Vec3,
  noiseBase: (x: number, y: number, z: number) => number,
  noiseDetail: (x: number, y: number, z: number) => number,
  noiseMicro: (x: number, y: number, z: number) => number,
  settings: PlanetSettings
): number {
  const [dx, dy, dz] = dir;
  const n1 = noiseBase(dx, dy, dz);
  const n2 = noiseDetail(dx * 3.0, dy * 3.0, dz * 3.0);
  const n3 = noiseMicro(dx * 8.0, dy * 8.0, dz * 8.0);

  const h = (n1 * 1.0 + n2 * 0.5 + n3 * 0.25) * settings.noiseStrength;
  let displacement = h * 0.10;
  
  // Apply relief aggressiveness to create more/fewer peaks and mountains
  // This is separate from noise strength - it amplifies the extremes
  const aggressiveness = settings.reliefAggressiveness ?? 1.0;
  if (aggressiveness !== 1.0) {
    // Preserve the sign (for valleys) but apply power to the magnitude
    const sign = displacement >= 0 ? 1 : -1;
    const magnitude = Math.abs(displacement);
    // Scale by a reference to keep similar range
    const scaled = magnitude / 0.15; // normalize to ~0-1 range
    const amplified = Math.pow(scaled, aggressiveness) * 0.15;
    displacement = sign * amplified;
  }
  
  // Calculate normalized height for this noise value
  // This represents height relative to the base sphere (radius 1.0)
  const normalizedHeight = displacement * 10.0; // scale to match shader calculations
  
  // If below water threshold, reduce displacement but keep enough variation for color depth
  // This makes water calmer while land keeps its terrain
  if (normalizedHeight < settings.waterThreshold) {
    // Smooth transition near waterline to avoid sharp edge
    const fadeRange = 0.15; // transition zone width
    const fade = clamp((normalizedHeight - (settings.waterThreshold - fadeRange)) / fadeRange, 0, 1);
    // Keep 40% of displacement for water (enough for depth colors, but calmer than land)
    return displacement * (0.40 + fade * 0.60);
  }
  
  return displacement;
}

function sphereMesh(subdiv: number, settings: PlanetSettings) {
  const latSeg = subdiv;
  const lonSeg = subdiv * 2;

  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];

  const noiseBase = makeNoiseSampler(settings.seed, settings.noiseType as NoiseType);
  const noiseDetail = makeNoiseSampler(settings.seed + "_detail", settings.noiseType as NoiseType);
  const noiseMicro = makeNoiseSampler(settings.seed + "_micro", settings.noiseType as NoiseType);

  // Use shared displacement calculation for consistency with object placement
  const getDisplacement = (dir: Vec3) => calculateDisplacement(dir, noiseBase, noiseDetail, noiseMicro, settings);

  const getPos = (lat: number, lon: number) => {
    const cl = Math.cos(lat);
    const sl = Math.sin(lat);
    const cx = Math.cos(lon);
    const sx = Math.sin(lon);
    const dir: Vec3 = [cl * cx, sl, cl * sx];
    const d = getDisplacement(dir);
    return mulScalar(dir, 1.0 + d);
  };

  const eps = 0.01;

  for (let y = 0; y <= latSeg; y++) {
    const v = y / latSeg;
    const lat = (v - 0.5) * Math.PI;
    const rowFromPole = Math.min(y, latSeg - y);

    for (let x = 0; x <= lonSeg; x++) {
      const u = x / lonSeg;
      const lon = u * Math.PI * 2 - Math.PI;

      const p = getPos(lat, lon);

      // At the poles the longitude derivative degenerates (cos(lat)≈0 collapses
      // all longitudes to the same point), so use the radial normal instead.
      let n: Vec3;
      if (rowFromPole < 1) {
        n = normalize(p);
      } else {
        const pLat = getPos(lat + eps, lon);
        const pLon = getPos(lat, lon + eps);
        const vLat = sub(pLat, p);
        const vLon = sub(pLon, p);
        n = normalize(cross(vLon, vLat));
      }

      positions.push(p[0], p[1], p[2]);
      normals.push(n[0], n[1], n[2]);
    }
  }

  const stride = lonSeg + 1;
  for (let y = 0; y < latSeg; y++) {
    for (let x = 0; x < lonSeg; x++) {
      const i0 = y * stride + x;
      const i1 = i0 + 1;
      const i2 = i0 + stride;
      const i3 = i2 + 1;

      indices.push(i0, i2, i1);
      indices.push(i1, i2, i3);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array(indices),
    triCount: indices.length / 3,
  };
}

function createShadowTarget(gl: GL, size: number) {
  const depthTex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, depthTex);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.DEPTH_COMPONENT24,
    size,
    size,
    0,
    gl.DEPTH_COMPONENT,
    gl.UNSIGNED_INT,
    null,
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  const fb = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depthTex, 0);
  gl.drawBuffers([gl.NONE]);
  gl.readBuffer(gl.NONE);

  const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  if (!ok) throw new Error("Shadow framebuffer incomplete");

  return { fb, depthTex, size };
}

function nowMs() {
  return performance.now();
}

export type RendererStats = {
  fps: number;
  triPlanet: number;
  triObjects: number;
  seed: string;
};

interface Crater {
  localDir: Vec3;
  radius: number;
  depth: number;
  timeCreated: number;
}

interface MissileState {
  localDir: Vec3;
  targetHeight: number;
  startDist: number;
  progress: number;
  duration: number;
}

interface ImpactParticle {
  localPos: Vec3;
  localVel: Vec3;
  life: number;
  maxLife: number;
  size: number;
  color: [number, number, number, number];
  growRate: number;
  isAdditive: boolean;
}

const MAX_PARTICLES = 2000;

export class PlanetRenderer {
  private canvas: HTMLCanvasElement;
  private gl: GL;

  private planetProg: WebGLProgram;
  private shadowProg: WebGLProgram;
  private objProg: WebGLProgram;

  private shadow: { fb: WebGLFramebuffer; depthTex: WebGLTexture; size: number };

  private planetVao?: WebGLVertexArrayObject;
  private planetIndexCount = 0;
  private planetTriCount = 0;

  private treeModel?: ObjModel;
  private boatModel?: ObjModel;
  private snowTreeModel?: ObjModel;
  private rocketModel?: ObjModel;

  private treeVao?: WebGLVertexArrayObject;
  private boatVao?: WebGLVertexArrayObject;
  private snowTreeVao?: WebGLVertexArrayObject;
  private rocketVao?: WebGLVertexArrayObject;

  private treeIndexCount = 0;
  private boatIndexCount = 0;
  private snowTreeIndexCount = 0;
  private rocketIndexCount = 0;

  private settings: PlanetSettings;

  private placed: PlacedObject[] = [];

  private raf = 0;
  private disposed = false;

  private rot = 0; // planet rotation
  private cameraRot = 0; // camera rotation around planet
  private lastT = nowMs();
  private fpsSmoothed = 60;

  private cameraDist = 3.15;
  private autoRotate = true;
  private rotationSpeed = 1.0;
  private objectDrawLogged = false;
  private initialized = false;

  private dragging = false;
  private lastPointerX = 0;
  private lastPointerY = 0;
  private mouseClientX = 0;
  private mouseClientY = 0;
  private boundOnKeyDown: ((e: KeyboardEvent) => void) | null = null;
  private cameraElevation = 0.11; // polar angle in radians

  private shadowVP: Mat4 = mat4Identity();
  private view: Mat4 = mat4Identity();
  private proj: Mat4 = mat4Identity();

  private lightDir: Vec3 = normalize([-0.35, -0.85, -0.35]);

  // Missile & impact system
  private craters: Crater[] = [];
  private missiles: MissileState[] = [];
  private particles: ImpactParticle[] = [];
  private particleProg!: WebGLProgram;
  private particleVao!: WebGLVertexArrayObject;
  private particlePosBuf!: WebGLBuffer;
  private particleColorBuf!: WebGLBuffer;
  private particleSizeBuf!: WebGLBuffer;
  private particlePosData = new Float32Array(MAX_PARTICLES * 3);
  private particleColorData = new Float32Array(MAX_PARTICLES * 4);
  private particleSizeData = new Float32Array(MAX_PARTICLES);

  // Planet mesh data (for crater deformation)
  private planetPositions!: Float32Array;
  private planetNormals!: Float32Array;
  private planetCraterMask!: Float32Array;
  private planetIndices!: Uint32Array;
  private planetPosBuf!: WebGLBuffer;
  private planetNorBuf!: WebGLBuffer;
  private planetCraterMaskBuf!: WebGLBuffer;

  constructor(canvas: HTMLCanvasElement, settings: PlanetSettings) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", {
      antialias: true,
      alpha: false,
      depth: true,
      stencil: false,
      powerPreference: "high-performance",
    }) as GL | null;
    if (!gl) throw new Error("WebGL2 not supported");
    this.gl = gl;

    this.settings = settings;
    this.rotationSpeed = settings.rotationSpeed ?? 1.0;

    this.planetProg = createProgram(gl, PLANET_VS, PLANET_FS);
    this.shadowProg = createProgram(gl, SHADOW_VS, SHADOW_FS);
    this.objProg = createProgram(gl, OBJ_VS, OBJ_FS);
    this.particleProg = createProgram(gl, PARTICLE_VS, PARTICLE_FS);

    this.shadow = createShadowTarget(gl, 4096);

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);

    this.initParticleBuffers();
    this.resize();
    this.loop = this.loop.bind(this);
    this.setupEventListeners();

    // Initialize asynchronously then start loop
    this.initRenderer().then(() => {
      console.log("✓ Renderer initialized, starting render loop");
      this.raf = requestAnimationFrame(this.loop);
    });
  }

  private setupEventListeners() {
    this.canvas.addEventListener('wheel', this.onWheel.bind(this), { passive: false });
    this.canvas.addEventListener('pointerdown', this.onPointerDown.bind(this));
    this.canvas.addEventListener('pointermove', this.onPointerMove.bind(this));
    this.canvas.addEventListener('pointerup', this.onPointerUp.bind(this));
    this.canvas.addEventListener('pointerleave', this.onPointerUp.bind(this));
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    // Track mouse position over canvas
    this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
    // Listen for M key to launch missile
    this.boundOnKeyDown = this.onKeyDown.bind(this);
    window.addEventListener('keydown', this.boundOnKeyDown);
  }

  private onMouseMove(e: MouseEvent) {
    this.mouseClientX = e.clientX;
    this.mouseClientY = e.clientY;
  }

  private onKeyDown(e: KeyboardEvent) {
    if (e.key === 'm' || e.key === 'M') {
      console.log('[MISSILE] M key pressed! mouseX:', this.mouseClientX, 'mouseY:', this.mouseClientY, 'initialized:', this.initialized);
      if (!this.initialized) {
        console.log('[MISSILE] ⏳ Renderer not initialized yet');
        return;
      }
      const pick = this.pick(this.mouseClientX, this.mouseClientY);
      console.log('[MISSILE] 🎯 Pick result:', { hit: pick.hit, localDir: pick.localDir, height: pick.height, isWater: pick.isWater });
      if (pick.hit) {
        this.launchMissile(pick);
        console.log('[MISSILE] 🚀 Missile launched! Total missiles:', this.missiles.length);
      } else {
        console.log('[MISSILE] ❌ Pick missed planet - aim cursor at the planet!');
      }
    }
  }

  private onWheel(e: WheelEvent) {
    e.preventDefault();
    const delta = e.deltaY * 0.001;
    this.cameraDist = clamp(this.cameraDist + delta, 1.5, 6.0);
  }

  private onPointerDown(e: PointerEvent) {
    if (e.button !== 0) return; // only left button for drag
    this.dragging = true;
    this.lastPointerX = e.clientX;
    this.lastPointerY = e.clientY;
    this.canvas.setPointerCapture(e.pointerId);
  }

  private onPointerMove(e: PointerEvent) {
    if (!this.dragging) return;
    const dx = e.clientX - this.lastPointerX;
    const dy = e.clientY - this.lastPointerY;
    this.lastPointerX = e.clientX;
    this.lastPointerY = e.clientY;

    this.cameraRot -= dx * 0.005;
    this.cameraElevation = clamp(this.cameraElevation - dy * 0.004, -1.4, 1.4);
  }

  private onPointerUp(e: PointerEvent) {
    if (e.button !== 0 && e.type === 'pointerup') return;
    this.dragging = false;
  }

  private async initRenderer() {
    console.log("🚀 Starting renderer initialization...");
    const modelsLoaded = await this.loadModels();
    if (!modelsLoaded) {
      console.error("❌ Failed to load models, objects will not render");
    }
    this.initObjects();
    this.rebuildPlanet();
    this.redistributeObjects();
    this.initialized = true;
    console.log("✅ Renderer initialization complete");
  }

  private async loadModels(): Promise<boolean> {
    try {
      console.log("📦 Fetching models...");
      const base = import.meta.env.BASE_URL;
      const [treeRes, boatRes, snowTreeRes, rocketRes] = await Promise.all([
        fetch(`${base}models/new_tree.obj`),
        fetch(`${base}models/12219_boat_v2_L2.obj`),
        fetch(`${base}models/snow_tree.obj`),
        fetch(`${base}models/rocket.obj`)
      ]);

      if (!treeRes.ok || !boatRes.ok || !snowTreeRes.ok || !rocketRes.ok) {
        console.error("❌ Failed to fetch models:", {
          tree: treeRes.status,
          boat: boatRes.status,
          snow_tree: snowTreeRes.status,
          rocket: rocketRes.status
        });
        return false;
      }

      const [treeText, boatText, snowTreeText, rocketText] = await Promise.all([
        treeRes.text(),
        boatRes.text(),
        snowTreeRes.text(),
        rocketRes.text()
      ]);

      this.treeModel = parseObj(treeText);
      this.rocketModel = parseObj(rocketText);
      this.boatModel = parseObj(boatText);
      this.snowTreeModel = parseObj(snowTreeText);

      // Normalize tree model: center XZ at origin, base (Y min) at 0, scale to unit cube
      {
        const b = this.treeModel.bounds;
        const cx = (b.min[0] + b.max[0]) / 2;
        const cy = b.min[1]; // base of trunk at Y=0
        const cz = (b.min[2] + b.max[2]) / 2;
        const maxExt = Math.max(
          b.max[0] - b.min[0],
          b.max[1] - b.min[1],
          b.max[2] - b.min[2],
        );
        const s = 1.0 / maxExt;
        const pos = this.treeModel.positions;
        for (let i = 0; i < pos.length; i += 3) {
          pos[i]     = (pos[i] - cx) * s;
          pos[i + 1] = (pos[i + 1] - cy) * s;
          pos[i + 2] = (pos[i + 2] - cz) * s;
        }
      }

      // Normalize snow tree model: center XZ at origin, base (Y min) at 0, scale to unit cube
      {
        const b = this.snowTreeModel.bounds;
        const cx = (b.min[0] + b.max[0]) / 2;
        const cy = b.min[1]; // base of trunk at Y=0
        const cz = (b.min[2] + b.max[2]) / 2;
        const maxExt = Math.max(
          b.max[0] - b.min[0],
          b.max[1] - b.min[1],
          b.max[2] - b.min[2],
        );
        const s = 1.0 / maxExt;
        const pos = this.snowTreeModel.positions;
        for (let i = 0; i < pos.length; i += 3) {
          pos[i]     = (pos[i] - cx) * s;
          pos[i + 1] = (pos[i + 1] - cy) * s;
          pos[i + 2] = (pos[i + 2] - cz) * s;
        }
      }

      // Normalize boat model: center XY at origin, align keel (Z min) to 0, scale to unit cube, rotate Z-up → Y-up
      {
        const b = this.boatModel.bounds;
        const cx = (b.min[0] + b.max[0]) / 2;
        const cy = (b.min[1] + b.max[1]) / 2;
        const cz = b.min[2]; // Keel at Z=0 so boat sits above origin after rotation
        const maxExt = Math.max(
          b.max[0] - b.min[0],
          b.max[1] - b.min[1],
          b.max[2] - b.min[2],
        );
        const s = 1.0 / maxExt;
        const pos = this.boatModel.positions;
        for (let i = 0; i < pos.length; i += 3) {
          const x = (pos[i] - cx) * s;
          const y = (pos[i + 1] - cy) * s;
          const z = (pos[i + 2] - cz) * s;
          // Rotate 90° around X: (x, y, z) → (x, z, -y)
          pos[i]     = x;
          pos[i + 1] = z;
          pos[i + 2] = -y;
        }
        const nor = this.boatModel.normals;
        for (let i = 0; i < nor.length; i += 3) {
          const ny = nor[i + 1];
          const nz = nor[i + 2];
          nor[i + 1] = nz;
          nor[i + 2] = -ny;
        }
      }

      // Normalize rocket model: rotate to point forward, center at origin, scale to unit size
      {
        const b = this.rocketModel.bounds;
        const cx = (b.min[0] + b.max[0]) / 2;
        const cy = (b.min[1] + b.max[1]) / 2;
        const cz = (b.min[2] + b.max[2]) / 2;
        const maxExt = Math.max(
          b.max[0] - b.min[0],
          b.max[1] - b.min[1],
          b.max[2] - b.min[2],
        );
        const s = 1.0 / maxExt;
        const pos = this.rocketModel.positions;
        // Original rocket points in -Y direction (nose down)
        // We want it to point in +Y (nose up) for easier orientation
        // Rotate 180° around X: (x, y, z) → (x, -y, -z), then center and scale
        for (let i = 0; i < pos.length; i += 3) {
          const x = (pos[i] - cx) * s;
          const y = (pos[i + 1] - cy) * s;
          const z = (pos[i + 2] - cz) * s;
          // Rotate 180° around X to flip Y and Z
          pos[i]     = x;
          pos[i + 1] = -y;
          pos[i + 2] = -z;
        }
        const nor = this.rocketModel.normals;
        for (let i = 0; i < nor.length; i += 3) {
          const ny = nor[i + 1];
          const nz = nor[i + 2];
          nor[i + 1] = -ny;
          nor[i + 2] = -nz;
        }
      }

      console.log("✅ Models loaded:", {
        tree: `${this.treeModel.positions.length / 3} verts, ${this.treeModel.indices.length / 3} tris`,
        boat: `${this.boatModel.positions.length / 3} verts, ${this.boatModel.indices.length / 3} tris`,
        snow_tree: `${this.snowTreeModel.positions.length / 3} verts, ${this.snowTreeModel.indices.length / 3} tris`,
        rocket: `${this.rocketModel.positions.length / 3} verts, ${this.rocketModel.indices.length / 3} tris`
      });
      return true;
    } catch (e) {
      console.error("❌ Exception loading models:", e);
      return false;
    }
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    if (this.boundOnKeyDown) {
      window.removeEventListener('keydown', this.boundOnKeyDown);
      this.boundOnKeyDown = null;
    }
  }

  setSettings(next: PlanetSettings, opts?: { rebuild?: boolean; redistribute?: boolean }) {
    this.settings = next;
    this.rotationSpeed = next.rotationSpeed ?? 1.0;
    if (opts?.rebuild) this.rebuildPlanet();
    if (opts?.redistribute) this.redistributeObjects();
  }

  setAutoRotate(enabled: boolean) {
    this.autoRotate = enabled;
  }

  getAutoRotate(): boolean {
    return this.autoRotate;
  }

  setZoom(distance: number) {
    this.cameraDist = clamp(distance, 1.5, 6.0);
  }

  getZoom(): number {
    return this.cameraDist;
  }

  getSettings() {
    return this.settings;
  }

  getStats(): RendererStats {
    const triObjects =
      this.placed.reduce((acc, o) => {
        const idxCount = o.kind === "tree" ? this.treeIndexCount : this.boatIndexCount;
        return acc + idxCount / 3;
      }, 0) | 0;

    return {
      fps: this.fpsSmoothed,
      triPlanet: this.planetTriCount,
      triObjects,
      seed: this.settings.seed,
    };
  }

  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    const aspect = w / h;
    this.proj = mat4Perspective((55 * Math.PI) / 180, aspect, 0.1, 50);
  }

  private initObjects() {
    const gl = this.gl;
    if (!this.treeModel || !this.boatModel || !this.snowTreeModel || !this.rocketModel) {
      console.warn("⚠️ Cannot init objects: models not loaded", {
        treeModel: !!this.treeModel,
        boatModel: !!this.boatModel,
        snowTreeModel: !!this.snowTreeModel,
        rocketModel: !!this.rocketModel
      });
      return;
    }

    console.log("🔧 Initializing object VAOs...", {
      treeVerts: this.treeModel.positions.length / 3,
      treeTris: this.treeModel.indices.length / 3,
      boatVerts: this.boatModel.positions.length / 3,
      boatTris: this.boatModel.indices.length / 3,
      snowTreeVerts: this.snowTreeModel.positions.length / 3,
      snowTreeTris: this.snowTreeModel.indices.length / 3,
      rocketVerts: this.rocketModel.positions.length / 3,
      rocketTris: this.rocketModel.indices.length / 3
    });

    // tree
    {
      const vaoInfo = createVao(
        gl,
        this.objProg,
        [
          { name: "a_position", size: 3, data: this.treeModel.positions },
          { name: "a_normal", size: 3, data: this.treeModel.normals },
        ],
        this.treeModel.indices,
      );
      this.treeVao = vaoInfo.vao;
      this.treeIndexCount = this.treeModel.indices.length;
      console.log("✓ Tree VAO created, indices:", this.treeIndexCount);
    }

    // boat
    {
      const vaoInfo = createVao(
        gl,
        this.objProg,
        [
          { name: "a_position", size: 3, data: this.boatModel.positions },
          { name: "a_normal", size: 3, data: this.boatModel.normals },
        ],
        this.boatModel.indices,
      );
      this.boatVao = vaoInfo.vao;
      this.boatIndexCount = this.boatModel.indices.length;
      console.log("✓ Boat VAO created, indices:", this.boatIndexCount);
    }

    // snow_tree
    {
      const vaoInfo = createVao(
        gl,
        this.objProg,
        [
          { name: "a_position", size: 3, data: this.snowTreeModel.positions },
          { name: "a_normal", size: 3, data: this.snowTreeModel.normals },
        ],
        this.snowTreeModel.indices,
      );
      this.snowTreeVao = vaoInfo.vao;
      this.snowTreeIndexCount = this.snowTreeModel.indices.length;
      console.log("✓ Snow Tree VAO created, indices:", this.snowTreeIndexCount);
    }

    // rocket
    {
      const vaoInfo = createVao(
        gl,
        this.objProg,
        [
          { name: "a_position", size: 3, data: this.rocketModel.positions },
          { name: "a_normal", size: 3, data: this.rocketModel.normals },
        ],
        this.rocketModel.indices,
      );
      this.rocketVao = vaoInfo.vao;
      this.rocketIndexCount = this.rocketModel.indices.length;
      console.log("✓ Rocket VAO created, indices:", this.rocketIndexCount);
    }
  }

  private rebuildPlanet() {
    const gl = this.gl;
    const mesh = sphereMesh(this.settings.subdivisions, this.settings);

    if (this.planetVao) {
      gl.deleteVertexArray(this.planetVao);
      this.planetVao = undefined;
    }

    // Store mesh data for crater deformation
    this.planetPositions = new Float32Array(mesh.positions);
    this.planetNormals = new Float32Array(mesh.normals);
    this.planetCraterMask = new Float32Array(mesh.positions.length / 3); // one float per vertex, starts at 0
    this.planetIndices = new Uint32Array(mesh.indices);

    // Create VAO with DYNAMIC_DRAW buffers for position/normal updates
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);

    this.planetPosBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.planetPosBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.planetPositions, gl.DYNAMIC_DRAW);
    const posLoc = gl.getAttribLocation(this.planetProg, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

    this.planetNorBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.planetNorBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.planetNormals, gl.DYNAMIC_DRAW);
    const norLoc = gl.getAttribLocation(this.planetProg, 'a_normal');
    gl.enableVertexAttribArray(norLoc);
    gl.vertexAttribPointer(norLoc, 3, gl.FLOAT, false, 0, 0);

    this.planetCraterMaskBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.planetCraterMaskBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.planetCraterMask, gl.DYNAMIC_DRAW);
    const craterLoc = gl.getAttribLocation(this.planetProg, 'a_craterMask');
    if (craterLoc >= 0) {
      gl.enableVertexAttribArray(craterLoc);
      gl.vertexAttribPointer(craterLoc, 1, gl.FLOAT, false, 0, 0);
    }

    const ibo = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.planetIndices, gl.STATIC_DRAW);

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    this.planetVao = vao;
    this.planetIndexCount = mesh.indices.length;
    this.planetTriCount = mesh.triCount;

    // Re-apply existing craters after rebuilding
    for (const crater of this.craters) {
      this.applyCraterToMesh(crater);
    }
    if (this.craters.length > 0) {
      this.uploadPlanetBuffers();
    }
  }

  private worldFromBasis(pos: Vec3, n: Vec3, t: Vec3, b: Vec3, scale: number): Mat4 {
    // Construct world matrix with basis vectors as columns (t, n, b)
    // We'll make a 4x4 then apply scale and translation.
    const m = mat4Identity();
    // column-major:
    m[0] = t[0]; m[1] = t[1]; m[2] = t[2];
    m[4] = n[0]; m[5] = n[1]; m[6] = n[2];
    m[8] = b[0]; m[9] = b[1]; m[10] = b[2];
    m[12] = pos[0];
    m[13] = pos[1];
    m[14] = pos[2];

    const s = mat4Scale([scale, scale, scale]);
    return mat4Mul(m, s);
  }

  private computeLightVP() {
    // Fixed directional light that rotates WITH the planet so shadows
    // stay anchored to objects as the planet spins.
    //
    // The light direction has a strong horizontal component so that
    // shadows on the visible hemisphere (equator) stay compact and close
    // to the objects that cast them.  A nearly-vertical light (large |Y|)
    // would hit the equator at a grazing angle, stretching shadows across
    // almost the entire planet radius.
    const baseLightDir: Vec3 = normalize([-0.4, -0.6, -0.7]);

    // Rotate baseLightDir around Y by the current planet rotation angle
    const cosR = Math.cos(this.rot);
    const sinR = Math.sin(this.rot);
    this.lightDir = normalize([
      baseLightDir[0] * cosR + baseLightDir[2] * sinR,
      baseLightDir[1],
      -baseLightDir[0] * sinR + baseLightDir[2] * cosR,
    ]);

    const lightPos: Vec3 = mulScalar(this.lightDir, -7.5);
    const center: Vec3 = [0, 0, 0];
    // Standard up vector; works as long as the light isn't straight along Y
    const up: Vec3 = [0, 1, 0];

    const lightView = mat4LookAt(lightPos, center, up);
    const lightProj = mat4Ortho(-1.5, 1.5, -1.5, 1.5, 6.0, 9.5);
    this.shadowVP = mat4Mul(lightProj, lightView);
  }

  private redistributeObjects() {
    const rng = mulberry32(hashStringToUint(this.settings.seed) ^ 0x1337abcd);
    const noiseBase = makeNoiseSampler(this.settings.seed, this.settings.noiseType as NoiseType);
    const noiseDetail = makeNoiseSampler(this.settings.seed + "_detail", this.settings.noiseType as NoiseType);
    const noiseMicro = makeNoiseSampler(this.settings.seed + "_micro", this.settings.noiseType as NoiseType);

    const count = this.settings.objectCount;
    const maxBoats = 5;
    const maxTrees = Math.floor(count * 0.45);
    const trees: PlacedObject[] = [];
    const snowTrees: PlacedObject[] = [];
    const boats: PlacedObject[] = [];

    for (let i = 0; i < count; i++) {
      const z = randRange(rng, -1, 1);
      const t = randRange(rng, 0, Math.PI * 2);
      const r = Math.sqrt(1 - z * z);
      const base: Vec3 = [r * Math.cos(t), z, r * Math.sin(t)];

      // Use shared displacement calculation to account for reliefAggressiveness
      const height = calculateDisplacement(base, noiseBase, noiseDetail, noiseMicro, this.settings);
      const isWater = height < this.settings.waterThreshold * 0.10;
      const isSnow = (height * 10.0) >= 0.8; // Snow starts at h=0.8 in normalized range

      const normal = normalize(base);
      const tangent = anyPerpendicular(normal);
      const bitangent = normalize(cross(normal, tangent));

      let kind: ObjectKind;
      if (isWater) {
        kind = "boat";
      } else if (isSnow) {
        kind = "snow_tree";
      } else {
        kind = "tree";
      }

      const scale = isWater ? randRange(rng, 0.10, 0.15) : randRange(rng, 0.06, 0.12);

      const pos = mulScalar(normal, 1.0 + height + (isWater ? -0.003 : 0.0));

      const obj: PlacedObject = {
        id: `${i}-${Math.floor(rng() * 1e9)}`,
        kind,
        position: pos,
        normal,
        tangent,
        bitangent,
        scale,
        phase: rng() * Math.PI * 2,
      };

      if (kind === "boat") {
        if (boats.length < maxBoats) boats.push(obj);
      } else if (kind === "snow_tree") {
        if (snowTrees.length < maxTrees) snowTrees.push(obj);
      } else {
        if (trees.length < maxTrees) trees.push(obj);
      }
    }

    this.placed = [...trees, ...snowTrees, ...boats];
    console.log(`✓ Redistributed ${this.placed.length} objects:`, {
      trees: trees.length,
      snow_trees: snowTrees.length,
      boats: boats.length
    });
  }

  pick(clientX: number, clientY: number): PickResult {
    const rect = this.canvas.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * 2 - 1;
    const y = -(((clientY - rect.top) / rect.height) * 2 - 1);

    // Unproject: build ray in view space then transform to world space.
    const aspect = this.canvas.width / this.canvas.height;
    const fovy = (55 * Math.PI) / 180;
    const tan = Math.tan(fovy / 2);

    const dirView: Vec3 = normalize([x * aspect * tan, y * tan, -1]);

    // Camera at spherical coordinates (azimuth=cameraRot, elevation=cameraElevation)
    const cosElev = Math.cos(this.cameraElevation);
    const sinElev = Math.sin(this.cameraElevation);
    const camPos: Vec3 = [
      Math.sin(this.cameraRot) * cosElev * this.cameraDist,
      sinElev * this.cameraDist,
      Math.cos(this.cameraRot) * cosElev * this.cameraDist,
    ];
    const center: Vec3 = [0, 0, 0];
    const up: Vec3 = [0, 1, 0];
    // Derive camera basis
    const f = normalize(sub(center, camPos));
    const s = normalize(cross(f, up));
    const u = cross(s, f);

    // view -> world: dir = s*x + u*y + (-f)*z (since view forward is -Z)
    const dirWorld = normalize(add(add(mulScalar(s, dirView[0]), mulScalar(u, dirView[1])), mulScalar(f, -dirView[2])));

    // Compute tight bounding sphere from actual noiseStrength
    const maxHeight = 1.75 * this.settings.noiseStrength * 0.10;
    const rMax = 1.0 + Math.max(maxHeight, 0.05);

    // Get both intersections with bounding sphere
    const bCoeff = 2 * dot(camPos, dirWorld);
    const cCoeff = dot(camPos, camPos) - rMax * rMax;
    const disc = bCoeff * bCoeff - 4 * cCoeff;
    if (disc < 0) return { hit: false };

    const sqrtDisc = Math.sqrt(disc);
    const t0 = (-bCoeff - sqrtDisc) / 2;
    const t1 = (-bCoeff + sqrtDisc) / 2;
    if (t1 < 0) return { hit: false }; // sphere entirely behind camera

    const tStart = Math.max(t0, 0.001); // if camera is inside sphere, start near camera
    const tEnd = t1;

    // Un-rotate from world space to planet-local space (inverse of mat4RotateY(rot))
    // Inverse of rotation matrix = transpose, so sin terms are negated vs forward transform.
    const cosR = Math.cos(this.rot);
    const sinR = Math.sin(this.rot);
    const toLocal = (d: Vec3): Vec3 => [
      d[0] * cosR - d[2] * sinR,
      d[1],
      d[0] * sinR + d[2] * cosR,
    ];

    const noiseBase = makeNoiseSampler(this.settings.seed, this.settings.noiseType as NoiseType);
    const noiseDetail = makeNoiseSampler(this.settings.seed + "_detail", this.settings.noiseType as NoiseType);
    const noiseMicro = makeNoiseSampler(this.settings.seed + "_micro", this.settings.noiseType as NoiseType);

    // Use shared displacement calculation to account for reliefAggressiveness
    const computeHeight = (ld: Vec3) => calculateDisplacement(ld, noiseBase, noiseDetail, noiseMicro, this.settings);

    // Helper: signed distance from the terrain surface at ray parameter t.
    // Positive = above terrain, negative = inside terrain.
    const sdfAt = (t: number) => {
      const p = add(camPos, mulScalar(dirWorld, t));
      const r = len(p);
      const dir = normalize(p);
      const localDir = toLocal(dir);
      const h = computeHeight(localDir);
      return r - (1.0 + h);
    };

    // Ray march: step along the ray and detect the first sign change
    // (above terrain → inside terrain). This is precise even at grazing angles.
    const STEPS = 48;
    const dt = (tEnd - tStart) / STEPS;

    let prevSdf = sdfAt(tStart);
    let tCross = -1;
    let tPrev = tStart;

    for (let i = 1; i <= STEPS; i++) {
      const t = tStart + dt * i;
      const curSdf = sdfAt(t);

      if (curSdf <= 0 && prevSdf > 0) {
        // The ray crossed the surface between tPrev and t
        tCross = tPrev;
        break;
      }
      prevSdf = curSdf;
      tPrev = t;
    }

    if (tCross < 0) return { hit: false };

    // Binary search to refine the exact crossing point
    let tLo = tCross;
    let tHi = tCross + dt;
    for (let i = 0; i < 16; i++) {
      const tMid = (tLo + tHi) / 2;
      if (sdfAt(tMid) > 0) {
        tLo = tMid;
      } else {
        tHi = tMid;
      }
    }

    const tFinal = (tLo + tHi) / 2;
    const hitPoint = add(camPos, mulScalar(dirWorld, tFinal));
    const worldDir = normalize(hitPoint);
    const localDir = toLocal(worldDir);
    const height = computeHeight(localDir);
    const isWater = height < this.settings.waterThreshold * 0.10;
    const isSnow = (height * 10.0) >= 0.8; // Snow starts at h=0.8 in normalized range

    return {
      hit: true,
      worldPos: mulScalar(worldDir, 1.0 + height),
      worldNormal: worldDir,
      localDir,
      height,
      isWater,
      isSnow,
    };
  }

  placeObjectFromPick(pick: PickResult) {
    if (!pick.hit || !pick.localDir || pick.height == null) {
      console.log("❌ Cannot place: invalid pick", pick);
      return;
    }

    // Determine object kind: boat for water, snow_tree for snow, tree for everything else
    let kind: ObjectKind;
    if (pick.isWater) {
      kind = "boat";
    } else if (pick.isSnow) {
      kind = "snow_tree";
    } else {
      kind = "tree";
    }

    const rng = mulberry32(hashStringToUint(this.settings.seed + "_place") ^ (this.placed.length * 2654435761));
    const scale = kind === "boat" ? randRange(rng, 0.10, 0.15) : randRange(rng, 0.06, 0.12);

    // Use planet-local direction and analytical noise height (not ray hit distance)
    const normal = normalize(pick.localDir);
    const tangent = anyPerpendicular(normal);
    const bitangent = normalize(cross(normal, tangent));
    const pos = mulScalar(normal, 1.0 + pick.height + (kind === "boat" ? -0.003 : 0.0));

    this.placed = [
      ...this.placed,
      {
        id: `p-${Date.now()}-${Math.floor(rng() * 1e9)}`,
        kind,
        position: pos,
        normal,
        tangent,
        bitangent,
        scale,
        phase: rng() * Math.PI * 2,
      },
    ];
  }

  // ─── Missile & Impact System ───────────────────────────────────────────

  launchMissile(pick: PickResult) {
    if (!pick.hit || !pick.localDir || pick.height == null) return;

    const localDir = normalize(pick.localDir);
    const targetHeight = pick.height;
    const startDist = 1.0 + targetHeight + 1.8; // start well above surface

    this.missiles.push({
      localDir,
      targetHeight,
      startDist,
      progress: 0,
      duration: this.settings.missileDuration || 1.2,
    });
  }

  private initParticleBuffers() {
    const gl = this.gl;

    this.particleVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.particleVao);

    // Position buffer
    this.particlePosBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particlePosBuf);
    gl.bufferData(gl.ARRAY_BUFFER, MAX_PARTICLES * 3 * 4, gl.DYNAMIC_DRAW);
    const posLoc = gl.getAttribLocation(this.particleProg, 'a_position');
    if (posLoc >= 0) {
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);
    }

    // Color buffer
    this.particleColorBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleColorBuf);
    gl.bufferData(gl.ARRAY_BUFFER, MAX_PARTICLES * 4 * 4, gl.DYNAMIC_DRAW);
    const colorLoc = gl.getAttribLocation(this.particleProg, 'a_color');
    if (colorLoc >= 0) {
      gl.enableVertexAttribArray(colorLoc);
      gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, 0, 0);
    }

    // Size buffer
    this.particleSizeBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleSizeBuf);
    gl.bufferData(gl.ARRAY_BUFFER, MAX_PARTICLES * 4, gl.DYNAMIC_DRAW);
    const sizeLoc = gl.getAttribLocation(this.particleProg, 'a_size');
    if (sizeLoc >= 0) {
      gl.enableVertexAttribArray(sizeLoc);
      gl.vertexAttribPointer(sizeLoc, 1, gl.FLOAT, false, 0, 0);
    }

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  private applyCraterToMesh(crater: Crater) {
    const { localDir, radius, depth } = crater;
    const cosR = Math.cos(radius);
    const pos = this.planetPositions;
    const mask = this.planetCraterMask;
    const dirtyVertices = new Set<number>();

    for (let i = 0; i < pos.length; i += 3) {
      const px = pos[i], py = pos[i + 1], pz = pos[i + 2];
      const r = Math.sqrt(px * px + py * py + pz * pz);
      if (r < 0.001) continue;

      const dx = px / r, dy = py / r, dz = pz / r;
      const d = dx * localDir[0] + dy * localDir[1] + dz * localDir[2];
      if (d < cosR) continue;

      const angDist = Math.acos(Math.min(1, d));
      const t = angDist / radius;

      let displacement: number;
      let craterIntensity: number;
      if (t < 0.7) {
        // Bowl interior: smooth cosine depression
        displacement = -depth * (Math.cos((t / 0.7) * Math.PI) * 0.5 + 0.5);
        craterIntensity = 1.0 - t / 0.7; // 1.0 at center, 0.0 at bowl edge
      } else {
        // Raised rim
        const rimT = (t - 0.7) / 0.3;
        displacement = depth * 0.2 * Math.sin(rimT * Math.PI);
        craterIntensity = 0.3 * Math.sin(rimT * Math.PI); // slight marking on rim
      }

      const newR = r + displacement;
      pos[i] = dx * newR;
      pos[i + 1] = dy * newR;
      pos[i + 2] = dz * newR;

      // Update crater mask (take max so overlapping craters work)
      const vi = i / 3;
      mask[vi] = Math.max(mask[vi], craterIntensity);
      dirtyVertices.add(vi);
    }

    // Only recompute normals for vertices affected by this crater,
    // preserving the original analytic normals for the rest of the planet.
    this.recomputeNormalsPartial(dirtyVertices);
  }

  /**
   * Recompute normals ONLY for the given set of vertex indices.
   * Non-dirty vertices keep their original normals (analytic finite-difference
   * normals from sphereMesh), preventing a global shading shift when a crater
   * is first created.
   */
  /**
   * Recompute normals for affected vertices and their neighbors to ensure
   * smooth shading transitions at crater boundaries.
   */
  private recomputeNormalsPartial(dirtyVerts: Set<number>) {
    if (dirtyVerts.size === 0) return;

    const pos = this.planetPositions;
    const idx = this.planetIndices;
    const nor = this.planetNormals;

    // Expand dirty set to include all vertices that share triangles with
    // dirty vertices, ensuring smooth shading at crater boundaries
    const expandedDirty = new Set<number>(dirtyVerts);
    for (let i = 0; i < idx.length; i += 3) {
      const a = idx[i], b = idx[i + 1], c = idx[i + 2];
      if (dirtyVerts.has(a) || dirtyVerts.has(b) || dirtyVerts.has(c)) {
        expandedDirty.add(a);
        expandedDirty.add(b);
        expandedDirty.add(c);
      }
    }

    // Zero out normals for all expanded dirty vertices
    expandedDirty.forEach(vi => {
      const i = vi * 3;
      nor[i] = 0;
      nor[i + 1] = 0;
      nor[i + 2] = 0;
    });

    // Accumulate face normals from every triangle that touches any dirty vertex
    for (let i = 0; i < idx.length; i += 3) {
      const a = idx[i], b = idx[i + 1], c = idx[i + 2];
      const aD = expandedDirty.has(a), bD = expandedDirty.has(b), cD = expandedDirty.has(c);
      if (!aD && !bD && !cD) continue;

      const i0 = a * 3, i1 = b * 3, i2 = c * 3;

      const ax = pos[i1] - pos[i0], ay = pos[i1 + 1] - pos[i0 + 1], az = pos[i1 + 2] - pos[i0 + 2];
      const bx = pos[i2] - pos[i0], by = pos[i2 + 1] - pos[i0 + 1], bz = pos[i2 + 2] - pos[i0 + 2];

      const nx = ay * bz - az * by;
      const ny = az * bx - ax * bz;
      const nz = ax * by - ay * bx;

      if (aD) { nor[i0] += nx; nor[i0 + 1] += ny; nor[i0 + 2] += nz; }
      if (bD) { nor[i1] += nx; nor[i1 + 1] += ny; nor[i1 + 2] += nz; }
      if (cD) { nor[i2] += nx; nor[i2 + 1] += ny; nor[i2 + 2] += nz; }
    }

    // Re-normalize all expanded dirty vertices
    expandedDirty.forEach(vi => {
      const i = vi * 3;
      const l = Math.sqrt(nor[i] * nor[i] + nor[i + 1] * nor[i + 1] + nor[i + 2] * nor[i + 2]);
      if (l > 0.0001) { 
        nor[i] /= l; 
        nor[i + 1] /= l; 
        nor[i + 2] /= l; 
      } else {
        // Fallback to radial direction if normal is degenerate
        const px = pos[i], py = pos[i + 1], pz = pos[i + 2];
        const r = Math.sqrt(px * px + py * py + pz * pz);
        if (r > 0.0001) {
          nor[i] = px / r;
          nor[i + 1] = py / r;
          nor[i + 2] = pz / r;
        } else {
          // Ultimate fallback (should never happen)
          nor[i] = 0;
          nor[i + 1] = 1;
          nor[i + 2] = 0;
        }
      }
    });
  }

  private uploadPlanetBuffers() {
    const gl = this.gl;
    
    // Save current VAO state to prevent corruption
    const prevVao = gl.getParameter(gl.VERTEX_ARRAY_BINDING);
    
    // Unbind any active VAO before updating buffers
    gl.bindVertexArray(null);
    
    // Update planet buffers
    gl.bindBuffer(gl.ARRAY_BUFFER, this.planetPosBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.planetPositions);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.planetNorBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.planetNormals);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.planetCraterMaskBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.planetCraterMask);
    
    // Clean up state
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    
    // Force WebGL to recognize the buffer updates by briefly binding the planet VAO
    if (this.planetVao) {
      gl.bindVertexArray(this.planetVao);
      gl.bindVertexArray(null);
    }
    
    // Restore previous VAO if any
    if (prevVao && prevVao !== this.planetVao) {
      gl.bindVertexArray(prevVao as WebGLVertexArrayObject);
    }
  }

  private removeObjectsNearCrater(localDir: Vec3, radius: number) {
    const cosR = Math.cos(radius * 1.5);
    this.placed = this.placed.filter(obj => {
      const p = obj.position as Vec3;
      const r = len(p);
      if (r < 0.001) return true;
      const n: Vec3 = [p[0] / r, p[1] / r, p[2] / r];
      return dot(n, localDir) < cosR;
    });
  }

  private spawnImpactParticles(localDir: Vec3, _timeS: number) {
    const tangent = anyPerpendicular(localDir);
    const bitangent = normalize(cross(localDir, tangent));

    // Compute surface position from noise
    const noiseBase = makeNoiseSampler(this.settings.seed, this.settings.noiseType as NoiseType);
    const noiseDetail = makeNoiseSampler(this.settings.seed + "_detail", this.settings.noiseType as NoiseType);
    const noiseMicro = makeNoiseSampler(this.settings.seed + "_micro", this.settings.noiseType as NoiseType);

    const n1 = noiseBase(localDir[0], localDir[1], localDir[2]);
    const n2 = noiseDetail(localDir[0] * 3.0, localDir[1] * 3.0, localDir[2] * 3.0);
    const n3 = noiseMicro(localDir[0] * 8.0, localDir[1] * 8.0, localDir[2] * 8.0);
    const height = (n1 * 1.0 + n2 * 0.5 + n3 * 0.25) * this.settings.noiseStrength * 0.10;

    const surfacePos = mulScalar(localDir, 1.0 + height);
    const rng = mulberry32(hashStringToUint(`impact-${Date.now()}`));

    // Fire particles (bright, additive glow)
    for (let i = 0; i < 45; i++) {
      const angle = rng() * Math.PI * 2;
      const speed = 0.04 + rng() * 0.12;
      const upSpeed = 0.02 + rng() * 0.08;
      const dir = add(
        add(mulScalar(tangent, Math.cos(angle) * speed), mulScalar(bitangent, Math.sin(angle) * speed)),
        mulScalar(localDir, upSpeed)
      );

      const life = 1.5 + rng() * 2.5;
      const r = 0.9 + rng() * 0.1;
      const g = 0.3 + rng() * 0.5;
      const b = rng() * 0.1;

      this.particles.push({
        localPos: [...surfacePos] as Vec3,
        localVel: dir,
        life,
        maxLife: life,
        size: 0.02 + rng() * 0.03,
        color: [r, g, b, 1.0],
        growRate: -0.4,
        isAdditive: true,
      });
    }

    // Smoke particles (gray, slow, expanding)
    for (let i = 0; i < 30; i++) {
      const angle = rng() * Math.PI * 2;
      const speed = 0.008 + rng() * 0.025;
      const upSpeed = 0.025 + rng() * 0.05;
      const dir = add(
        add(mulScalar(tangent, Math.cos(angle) * speed), mulScalar(bitangent, Math.sin(angle) * speed)),
        mulScalar(localDir, upSpeed)
      );

      const life = 3.5 + rng() * 3.5;
      const gray = 0.35 + rng() * 0.35;

      this.particles.push({
        localPos: [...surfacePos] as Vec3,
        localVel: dir,
        life,
        maxLife: life,
        size: 0.012 + rng() * 0.018,
        color: [gray, gray, gray * 0.9, 0.5],
        growRate: 0.6,
        isAdditive: false,
      });
    }

    // Debris particles (brown/dark, fast, short-lived)
    for (let i = 0; i < 20; i++) {
      const angle = rng() * Math.PI * 2;
      const speed = 0.1 + rng() * 0.2;
      const upSpeed = 0.08 + rng() * 0.15;
      const dir = add(
        add(mulScalar(tangent, Math.cos(angle) * speed), mulScalar(bitangent, Math.sin(angle) * speed)),
        mulScalar(localDir, upSpeed)
      );

      const life = 0.6 + rng() * 0.8;
      const r = 0.3 + rng() * 0.25;
      const g = 0.18 + rng() * 0.12;

      this.particles.push({
        localPos: [...surfacePos] as Vec3,
        localVel: dir,
        life,
        maxLife: life,
        size: 0.006 + rng() * 0.012,
        color: [r, g, 0.05, 1.0],
        growRate: -0.5,
        isAdditive: true,
      });
    }

    // Shockwave ring particles (bright flash ring)
    for (let i = 0; i < 20; i++) {
      const angle = (i / 20) * Math.PI * 2;
      const speed = 0.2 + rng() * 0.05;
      const dir = add(
        mulScalar(tangent, Math.cos(angle) * speed),
        mulScalar(bitangent, Math.sin(angle) * speed)
      );

      this.particles.push({
        localPos: [...surfacePos] as Vec3,
        localVel: dir,
        life: 0.4,
        maxLife: 0.4,
        size: 0.015 + rng() * 0.01,
        color: [1.0, 0.9, 0.5, 1.0],
        growRate: -2.0,
        isAdditive: true,
      });
    }
  }

  private updateMissiles(dt: number, timeS: number) {
    for (let i = this.missiles.length - 1; i >= 0; i--) {
      const m = this.missiles[i];
      m.progress += dt / m.duration;

      // Spawn trail particles during flight
      if (m.progress < 1.0 && Math.random() < 0.4) {
        const dist = m.startDist + (1.0 + m.targetHeight - m.startDist) * m.progress;
        const trailPos = mulScalar(m.localDir, dist);
        this.particles.push({
          localPos: [...trailPos] as Vec3,
          localVel: mulScalar(m.localDir, -0.01),
          life: 0.3 + Math.random() * 0.2,
          maxLife: 0.5,
          size: 0.008 + Math.random() * 0.005,
          color: [1.0, 0.6, 0.1, 0.7],
          growRate: -1.0,
          isAdditive: true,
        });
      }

      if (m.progress >= 1.0) {
        // IMPACT!
        const craterRadius = 0.12;
        const craterDepth = 0.025;

        const crater: Crater = {
          localDir: m.localDir,
          radius: craterRadius,
          depth: craterDepth,
          timeCreated: timeS,
        };
        this.craters.push(crater);
        this.applyCraterToMesh(crater);
        this.uploadPlanetBuffers();

        // Remove objects near crater
        this.removeObjectsNearCrater(m.localDir, craterRadius);

        // Spawn explosion particles
        this.spawnImpactParticles(m.localDir, timeS);

        this.missiles.splice(i, 1);
      }
    }
  }

  private updateParticles(dt: number) {
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.life -= dt;

      if (p.life <= 0) {
        this.particles.splice(i, 1);
        continue;
      }

      // Update position
      p.localPos = add(p.localPos, mulScalar(p.localVel, dt));

      // Dampen velocity (drag)
      p.localVel = mulScalar(p.localVel, Math.max(0, 1 - dt * 1.8));

      // Gravity toward planet center for debris
      const r = len(p.localPos);
      if (r > 0.01) {
        const gravity = mulScalar(normalize(p.localPos), -dt * 0.04);
        p.localVel = add(p.localVel, gravity);
      }

      // Update size based on grow rate
      if (p.growRate > 0) {
        p.size += p.growRate * dt * 0.01;
      } else {
        p.size = Math.max(0.001, p.size + p.growRate * dt * p.size);
      }

      // Fade alpha over lifetime
      const lifeRatio = p.life / p.maxLife;
      p.color[3] = lifeRatio * (p.isAdditive ? 0.9 : 0.4);
    }

    // Cap total particles
    if (this.particles.length > MAX_PARTICLES) {
      this.particles.splice(0, this.particles.length - MAX_PARTICLES);
    }
  }

  private renderParticles(timeS: number) {
    const gl = this.gl;
    const totalCount = this.particles.length;
    if (totalCount === 0) return;

    const planetRot = mat4RotateY(this.rot);

    const positions = this.particlePosData;
    const colors = this.particleColorData;
    const sizes = this.particleSizeData;

    // Separate into additive and non-additive counts
    let additiveStart = 0;
    let idx = 0;

    // First: non-additive particles (smoke) with regular alpha blending
    for (const p of this.particles) {
      if (p.isAdditive) continue;
      const wp = transformVec3(planetRot, p.localPos);
      positions[idx * 3] = wp[0];
      positions[idx * 3 + 1] = wp[1];
      positions[idx * 3 + 2] = wp[2];
      colors[idx * 4] = p.color[0];
      colors[idx * 4 + 1] = p.color[1];
      colors[idx * 4 + 2] = p.color[2];
      colors[idx * 4 + 3] = p.color[3];
      sizes[idx] = p.size;
      idx++;
    }
    additiveStart = idx;

    // Then: additive particles (fire, debris, shockwave)
    for (const p of this.particles) {
      if (!p.isAdditive) continue;
      const wp = transformVec3(planetRot, p.localPos);
      positions[idx * 3] = wp[0];
      positions[idx * 3 + 1] = wp[1];
      positions[idx * 3 + 2] = wp[2];
      colors[idx * 4] = p.color[0];
      colors[idx * 4 + 1] = p.color[1];
      colors[idx * 4 + 2] = p.color[2];
      colors[idx * 4 + 3] = p.color[3];
      sizes[idx] = p.size;
      idx++;
    }

    const totalReal = idx;
    if (totalReal === 0) return;

    // Upload buffers
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particlePosBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, positions.subarray(0, totalReal * 3));
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleColorBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, colors.subarray(0, totalReal * 4));
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleSizeBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, sizes.subarray(0, totalReal));
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // Setup render state
    gl.useProgram(this.particleProg);
    gl.bindVertexArray(this.particleVao);

    gl.uniformMatrix4fv(gl.getUniformLocation(this.particleProg, 'u_view'), false, this.view);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.particleProg, 'u_proj'), false, this.proj);
    gl.uniform1f(gl.getUniformLocation(this.particleProg, 'u_screenHeight'), this.canvas.height);

    gl.depthMask(false);
    gl.disable(gl.CULL_FACE);
    gl.enable(gl.BLEND);

    // Pass 1: non-additive (smoke) with regular alpha blending
    if (additiveStart > 0) {
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.drawArrays(gl.POINTS, 0, additiveStart);
    }

    // Pass 2: additive (fire, debris, missiles, shockwave)
    if (totalReal > additiveStart) {
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      gl.drawArrays(gl.POINTS, additiveStart, totalReal - additiveStart);
    }

    // Restore state
    gl.disable(gl.BLEND);
    gl.depthMask(true);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.bindVertexArray(null);
  }

  private renderMissiles(timeS: number) {
    const gl = this.gl;
    if (this.missiles.length === 0 || !this.rocketVao) return;

    const planetRot = mat4RotateY(this.rot);

    gl.useProgram(this.objProg);
    gl.bindVertexArray(this.rocketVao);

    const uWorld = gl.getUniformLocation(this.objProg, "u_world");
    const uView = gl.getUniformLocation(this.objProg, "u_view");
    const uProj = gl.getUniformLocation(this.objProg, "u_proj");
    const uNormal = gl.getUniformLocation(this.objProg, "u_normal");
    const uLightDir = gl.getUniformLocation(this.objProg, "u_lightDir");
    const uCameraPos = gl.getUniformLocation(this.objProg, "u_cameraPos");
    const uLightVP = gl.getUniformLocation(this.objProg, "u_lightVP");
    const uAlbedo = gl.getUniformLocation(this.objProg, "u_albedo");
    const uAlbedo2 = gl.getUniformLocation(this.objProg, "u_albedo2");
    const uAlbedo3 = gl.getUniformLocation(this.objProg, "u_albedo3");
    const uIsBoat = gl.getUniformLocation(this.objProg, "u_isBoat");
    const uShadowsEnabled = gl.getUniformLocation(this.objProg, "u_shadowsEnabled");

    gl.uniformMatrix4fv(uView, false, this.view);
    gl.uniformMatrix4fv(uProj, false, this.proj);
    gl.uniformMatrix4fv(uLightVP, false, this.shadowVP);
    gl.uniform3fv(uLightDir, this.lightDir);
    gl.uniform1i(uShadowsEnabled, this.settings.shadowsEnabled ? 1 : 0);
    gl.uniform1f(uIsBoat, 0.0); // Not a boat
    
    // Camera position
    const cosElev = Math.cos(this.cameraElevation);
    const sinElev = Math.sin(this.cameraElevation);
    const camPos: Vec3 = [
      Math.sin(this.cameraRot) * cosElev * this.cameraDist,
      sinElev * this.cameraDist,
      Math.cos(this.cameraRot) * cosElev * this.cameraDist,
    ];
    gl.uniform3fv(uCameraPos, camPos);

    // Bind shadow map
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.shadow.depthTex);
    gl.uniform1i(gl.getUniformLocation(this.objProg, "u_shadowMap"), 0);

    // Rocket colors (corpo cinza escuro, cone branco, detalhes amarelo/dourado)
    const bodyColor: Vec3 = [0.35, 0.35, 0.36]; // cinza escuro metálico
    const coneColor: Vec3 = [0.9, 0.9, 0.92]; // branco
    const detailColor: Vec3 = [0.95, 0.75, 0.1]; // amarelo dourado
    
    gl.uniform3fv(uAlbedo, bodyColor);
    gl.uniform3fv(uAlbedo2, coneColor);
    gl.uniform3fv(uAlbedo3, detailColor);

    // Render each missile
    for (const m of this.missiles) {
      const dist = m.startDist + (1.0 + m.targetHeight - m.startDist) * m.progress;
      const localPos = mulScalar(m.localDir, dist);

      // Direction vector pointing down toward surface (inward toward planet center)
      // Rocket's Y-axis (tip after 180° normalization) should point this way
      const direction = normalize(m.localDir); // Points toward center (down)

      // Build orientation matrix: rocket's Y-axis (tip) points along direction
      const yAxis = direction;  // Rocket's local Y (tip direction)
      const xAxis = anyPerpendicular(yAxis);  // Rocket's local X
      const zAxis = normalize(cross(xAxis, yAxis));  // Rocket's local Z

      // Scale the rocket
      const scale = 0.08;

      // World matrix: orientation + position + scale
      // Columns are the basis vectors scaled
      const rocketLocal = mat4Identity();
      // Column 0: X-axis
      rocketLocal[0] = xAxis[0] * scale;
      rocketLocal[1] = xAxis[1] * scale;
      rocketLocal[2] = xAxis[2] * scale;
      // Column 1: Y-axis (rocket tip direction)
      rocketLocal[4] = yAxis[0] * scale;
      rocketLocal[5] = yAxis[1] * scale;
      rocketLocal[6] = yAxis[2] * scale;
      // Column 2: Z-axis
      rocketLocal[8] = zAxis[0] * scale;
      rocketLocal[9] = zAxis[1] * scale;
      rocketLocal[10] = zAxis[2] * scale;
      // Column 3: position
      rocketLocal[12] = localPos[0];
      rocketLocal[13] = localPos[1];
      rocketLocal[14] = localPos[2];

      const world = mat4Mul(planetRot, rocketLocal);

      gl.uniformMatrix4fv(uWorld, false, world);
      gl.uniformMatrix3fv(uNormal, false, [
        world[0], world[1], world[2],
        world[4], world[5], world[6],
        world[8], world[9], world[10]
      ]);

      gl.drawElements(gl.TRIANGLES, this.rocketIndexCount, gl.UNSIGNED_INT, 0);
    }

    gl.bindVertexArray(null);
  }

  private renderShadowPass(timeS: number) {
    const gl = this.gl;
    this.computeLightVP();

    // Unbind shadow texture from TEXTURE0 before binding shadow FBO to
    // prevent a rendering feedback loop (same texture attached to FBO and
    // bound for sampling). ANGLE on Windows may produce undefined results.
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, null);

    // Ensure completely clean GL state
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    
    // Reset render state
    gl.disable(gl.BLEND);
    gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.enable(gl.CULL_FACE);

    // Bind shadow framebuffer and clear
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadow.fb);
    gl.viewport(0, 0, this.shadow.size, this.shadow.size);
    gl.colorMask(false, false, false, false);
    gl.clear(gl.DEPTH_BUFFER_BIT);

    gl.useProgram(this.shadowProg);

    const uWorld = gl.getUniformLocation(this.shadowProg, "u_world");
    const uLightVP = gl.getUniformLocation(this.shadowProg, "u_lightVP");

    // Planet: cull FRONT faces → renders the planet's back-face depth
    // (the far hemisphere surface as seen from the light).  The back face
    // depth (~0.74) is much larger than the front face depth (~0.11), so
    // front-face fragments in the main pass always satisfy
    //   current < back_depth → lit.
    // Objects rendered without culling write their true depth (~0.11 minus
    // a small lift), which is shallower than the back face.  Surface
    // fragments behind the object then satisfy
    //   current > obj_depth → in shadow.
    gl.cullFace(gl.FRONT);
    gl.bindVertexArray(this.planetVao!);
    gl.uniformMatrix4fv(uWorld, false, mat4RotateY(this.rot));
    gl.uniformMatrix4fv(uLightVP, false, this.shadowVP);
    gl.drawElements(gl.TRIANGLES, this.planetIndexCount, gl.UNSIGNED_INT, 0);

    // objects — no polygon offset, no face culling
    gl.disable(gl.CULL_FACE);

    const planetRot = mat4RotateY(this.rot);

    const drawObj = (o: PlacedObject) => {
      let pos = o.position as Vec3;

      // boat bob (gentle)
      if (o.kind === "boat") {
        const bob = Math.sin(timeS * 1.8 + o.phase) * 0.005;
        pos = add(pos, mulScalar(o.normal as Vec3, bob));
      }

      // Lift the object slightly above the surface in the SHADOW PASS
      // ONLY.  This creates a reliable depth gap between the object
      // (closer to the light) and the planet surface (further from the
      // light) so that the depth comparison in the main pass can detect
      // the shadow even at the object's base where object depth ≈ surface
      // depth.  The lift is invisible because the shadow pass only writes
      // depth — not color.
      const shadowLift = 0.04;
      pos = add(pos, mulScalar(o.normal as Vec3, shadowLift));

      const objLocal = this.worldFromBasis(pos, o.normal as Vec3, o.tangent as Vec3, o.bitangent as Vec3, o.scale);
      const world = mat4Mul(planetRot, objLocal);
      gl.uniformMatrix4fv(uWorld, false, world);
      gl.uniformMatrix4fv(uLightVP, false, this.shadowVP);

      if (o.kind === "tree") {
        if (!this.treeVao) return;
        gl.bindVertexArray(this.treeVao);
        gl.drawElements(gl.TRIANGLES, this.treeIndexCount, gl.UNSIGNED_INT, 0);
      } else if (o.kind === "snow_tree") {
        if (!this.snowTreeVao) return;
        gl.bindVertexArray(this.snowTreeVao);
        gl.drawElements(gl.TRIANGLES, this.snowTreeIndexCount, gl.UNSIGNED_INT, 0);
      } else {
        if (!this.boatVao) return;
        gl.bindVertexArray(this.boatVao);
        gl.drawElements(gl.TRIANGLES, this.boatIndexCount, gl.UNSIGNED_INT, 0);
      }
    };

    for (const o of this.placed) drawObj(o);

    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.colorMask(true, true, true, true);
  }

  private renderMainPass(timeS: number) {
    const gl = this.gl;

    // Ensure clean state for main pass (shadow pass may leave colorMask off)
    gl.colorMask(true, true, true, true);
    gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.disable(gl.BLEND);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.03, 0.04, 0.07, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // camera (spherical coords: cameraRot = azimuth, cameraElevation = polar angle)
    const cosElev = Math.cos(this.cameraElevation);
    const sinElev = Math.sin(this.cameraElevation);
    const camPos: Vec3 = [
      Math.sin(this.cameraRot) * cosElev * this.cameraDist,
      sinElev * this.cameraDist,
      Math.cos(this.cameraRot) * cosElev * this.cameraDist,
    ];
    this.view = mat4LookAt(camPos, [0, 0, 0], [0, 1, 0]);

    // Bind shadow map
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.shadow.depthTex);

    // planet
    gl.useProgram(this.planetProg);
    gl.bindVertexArray(this.planetVao!);

    gl.uniformMatrix4fv(gl.getUniformLocation(this.planetProg, "u_world"), false, mat4RotateY(this.rot));
    gl.uniformMatrix4fv(gl.getUniformLocation(this.planetProg, "u_view"), false, this.view);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.planetProg, "u_proj"), false, this.proj);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.planetProg, "u_lightVP"), false, this.shadowVP);

    gl.uniform3f(gl.getUniformLocation(this.planetProg, "u_lightDir"), this.lightDir[0], this.lightDir[1], this.lightDir[2]);
    gl.uniform3f(gl.getUniformLocation(this.planetProg, "u_cameraPos"), camPos[0], camPos[1], camPos[2]);

    gl.uniform1f(gl.getUniformLocation(this.planetProg, "u_waterThreshold"), this.settings.waterThreshold);
    gl.uniform1f(gl.getUniformLocation(this.planetProg, "u_noiseStrength"), this.settings.noiseStrength);
    gl.uniform1f(gl.getUniformLocation(this.planetProg, "u_time"), timeS);
    gl.uniform1i(gl.getUniformLocation(this.planetProg, "u_shadowMap"), 0);
    gl.uniform1i(gl.getUniformLocation(this.planetProg, "u_shadowsEnabled"), this.settings.shadowsEnabled ? 1 : 0);
    // DEBUG: set to 1 to see raw shadow map values (red=shadow, green=lit)
    gl.uniform1i(gl.getUniformLocation(this.planetProg, "u_debugShadow"), 0);

    gl.drawElements(gl.TRIANGLES, this.planetIndexCount, gl.UNSIGNED_INT, 0);

    // objects
    gl.useProgram(this.objProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.objProg, "u_view"), false, this.view);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.objProg, "u_proj"), false, this.proj);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.objProg, "u_lightVP"), false, this.shadowVP);
    gl.uniform3f(gl.getUniformLocation(this.objProg, "u_lightDir"), this.lightDir[0], this.lightDir[1], this.lightDir[2]);
    gl.uniform3f(gl.getUniformLocation(this.objProg, "u_cameraPos"), camPos[0], camPos[1], camPos[2]);
    gl.uniform1i(gl.getUniformLocation(this.objProg, "u_shadowMap"), 0);
    gl.uniform1i(gl.getUniformLocation(this.objProg, "u_shadowsEnabled"), this.settings.shadowsEnabled ? 1 : 0);

    const planetRot = mat4RotateY(this.rot);

    let drawnCount = 0;
    const drawObj = (o: PlacedObject) => {
      drawnCount++;
      if (!this.objectDrawLogged && drawnCount === 1) {
        this.objectDrawLogged = true;
        console.log("🖌️ Drawing objects. First:", {
          kind: o.kind,
          scale: o.scale,
          position: o.position,
          posLength: len(o.position as Vec3),
          hasTreeVao: !!this.treeVao,
          hasBoatVao: !!this.boatVao,
          hasSnowTreeVao: !!this.snowTreeVao
        });
      }
      let pos = o.position as Vec3;
      let albedo: Vec3;
      let albedo2: Vec3 = [0.8, 0.8, 0.8]; // Default deck color
      let albedo3: Vec3 = [0.9, 0.9, 0.9]; // Default superstructure color
      let isBoat = 0.0;
      let t = o.tangent as Vec3;
      let b = o.bitangent as Vec3;

      if (o.kind === "boat") {
        isBoat = 1.0;
        // Elevate boat above water surface to prevent sinking
        const bob = Math.sin(timeS * 1.8 + o.phase) * 0.005;
        const floatHeight = 0.018; // Fine-tuned for realistic floating
        pos = add(pos, mulScalar(o.normal as Vec3, bob + floatHeight));
        // Varied boat colors based on phase (different boat color schemes)
        // Each boat gets 3 colors: hull, deck, and superstructure
        const tp = o.phase / (Math.PI * 2); // 0–1
        if (tp < 0.20) {
          albedo = [0.18, 0.25, 0.35]; // Dark blue/gray hull
          albedo2 = [0.82, 0.78, 0.72]; // Cream deck
          albedo3 = [0.72, 0.25, 0.22]; // Red superstructure
        } else if (tp < 0.35) {
          albedo = [0.75, 0.28, 0.22]; // Red/orange hull
          albedo2 = [0.88, 0.85, 0.80]; // White deck
          albedo3 = [0.22, 0.22, 0.25]; // Dark superstructure
        } else if (tp < 0.50) {
          albedo = [0.85, 0.82, 0.75]; // White/cream hull
          albedo2 = [0.35, 0.48, 0.62]; // Blue deck
          albedo3 = [0.88, 0.85, 0.82]; // Light superstructure
        } else if (tp < 0.65) {
          albedo = [0.25, 0.52, 0.65]; // Teal/aqua hull
          albedo2 = [0.88, 0.88, 0.90]; // White deck
          albedo3 = [0.65, 0.28, 0.25]; // Red-brown superstructure
        } else if (tp < 0.80) {
          albedo = [0.68, 0.58, 0.32]; // Sandy/tan hull
          albedo2 = [0.45, 0.35, 0.28]; // Brown deck
          albedo3 = [0.85, 0.82, 0.75]; // Cream superstructure
        } else {
          albedo = [0.35, 0.42, 0.38]; // Green/military hull
          albedo2 = [0.28, 0.32, 0.30]; // Dark green deck
          albedo3 = [0.82, 0.78, 0.70]; // Tan superstructure
        }
      } else if (o.kind === "snow_tree") {
        // snow tree with tiny sway
        const sway = Math.sin(timeS * 1.6 + o.phase) * 0.015;
        t = normalize(add(o.tangent as Vec3, mulScalar(o.bitangent as Vec3, sway)));
        b = normalize(cross(o.normal as Vec3, t));
        // Snow tree colors - white to light blue tones
        const tp = o.phase / (Math.PI * 2); // 0–1
        if (tp < 0.33) {
          albedo = [0.85, 0.90, 0.95]; // white-blue
        } else if (tp < 0.66) {
          albedo = [0.90, 0.92, 0.95]; // bright white
        } else {
          albedo = [0.80, 0.85, 0.90]; // light blue-gray
        }
      } else {
        // tree with tiny sway
        const sway = Math.sin(timeS * 1.6 + o.phase) * 0.015;
        t = normalize(add(o.tangent as Vec3, mulScalar(o.bitangent as Vec3, sway)));
        b = normalize(cross(o.normal as Vec3, t));
        // varied foliage colors based on phase
        const tp = o.phase / (Math.PI * 2); // 0–1
        if (tp < 0.30) {
          albedo = [0.13, 0.42, 0.14]; // dark forest
        } else if (tp < 0.55) {
          albedo = [0.22, 0.58, 0.20]; // green
        } else if (tp < 0.75) {
          albedo = [0.38, 0.62, 0.18]; // light green
        } else if (tp < 0.88) {
          albedo = [0.56, 0.58, 0.12]; // yellow-green
        } else {
          albedo = [0.68, 0.42, 0.10]; // autumn orange
        }
      }

      const objLocal = this.worldFromBasis(pos, o.normal as Vec3, t, b, o.scale);
      const world = mat4Mul(planetRot, objLocal);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.objProg, "u_world"), false, world);
      gl.uniform3f(gl.getUniformLocation(this.objProg, "u_albedo"), albedo[0], albedo[1], albedo[2]);
      gl.uniform3f(gl.getUniformLocation(this.objProg, "u_albedo2"), albedo2[0], albedo2[1], albedo2[2]);
      gl.uniform3f(gl.getUniformLocation(this.objProg, "u_albedo3"), albedo3[0], albedo3[1], albedo3[2]);
      gl.uniform1f(gl.getUniformLocation(this.objProg, "u_isBoat"), isBoat);

      if (o.kind === "tree") {
        if (!this.treeVao) {
          if (drawnCount === 1) console.log("❌ Tree VAO missing!");
          return;
        }
        gl.bindVertexArray(this.treeVao);
        gl.drawElements(gl.TRIANGLES, this.treeIndexCount, gl.UNSIGNED_INT, 0);
      } else if (o.kind === "snow_tree") {
        if (!this.snowTreeVao) {
          if (drawnCount === 1) console.log("❌ Snow Tree VAO missing!");
          return;
        }
        gl.bindVertexArray(this.snowTreeVao);
        gl.drawElements(gl.TRIANGLES, this.snowTreeIndexCount, gl.UNSIGNED_INT, 0);
      } else {
        if (!this.boatVao) {
          if (drawnCount === 1) console.log("❌ Boat VAO missing!");
          return;
        }
        gl.bindVertexArray(this.boatVao);
        gl.drawElements(gl.TRIANGLES, this.boatIndexCount, gl.UNSIGNED_INT, 0);
      }
    };

    for (const o of this.placed) drawObj(o);

    if (drawnCount > 0 && drawnCount !== this.placed.length) {
      console.log("⚠️ Drew", drawnCount, "of", this.placed.length, "objects");
    }

    gl.bindVertexArray(null);

    // Unbind shadow texture so it's not bound when next frame's shadow pass
    // binds the shadow FBO (prevents texture feedback loop)
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  private loop() {
    if (this.disposed) return;

    const t = nowMs();
    const dt = Math.max(0.001, (t - this.lastT) / 1000);
    this.lastT = t;

    // smooth rotation
    if (this.autoRotate) {
      this.rot += dt * 0.12 * this.rotationSpeed; // planet spins
      this.cameraRot += dt * 0.06 * this.rotationSpeed; // camera orbits
    }

    // fps smoothing
    const fps = 1 / dt;
    this.fpsSmoothed = this.fpsSmoothed * 0.92 + fps * 0.08;

    const timeS = t / 1000;

    // Update missile & particle systems
    this.updateMissiles(dt, timeS);
    this.updateParticles(dt);

    this.resize();

    // render shadow then main
    if (this.settings.shadowsEnabled) this.renderShadowPass(timeS);
    else this.computeLightVP(); // still compute matrix for shading; map still used but ignored

    this.renderMainPass(timeS);
    this.renderMissiles(timeS);
    this.renderParticles(timeS);

    this.raf = requestAnimationFrame(this.loop);
  }
}
