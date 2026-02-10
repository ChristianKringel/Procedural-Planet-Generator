import type { NoiseType, PickResult, PlanetSettings, PlacedObject, ObjectKind } from "@shared/schema";
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

uniform mat4 u_world;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat4 u_lightVP;

out vec3 v_worldPos;
out vec3 v_worldNormal;
out vec4 v_lightClip;
out float v_height;

void main() {
  vec4 wp = u_world * vec4(a_position, 1.0);
  v_worldPos = wp.xyz;
  v_worldNormal = mat3(u_world) * a_normal;
  v_lightClip = u_lightVP * wp;
  v_height = length(a_position); // height relative to radius 1.0
  gl_Position = u_proj * u_view * wp;
}
`;

const PLANET_FS = `#version 300 es
precision highp float;

in vec3 v_worldPos;
in vec3 v_worldNormal;
in vec4 v_lightClip;
in float v_height;

uniform vec3 u_lightDir;     // direction towards light (sun)
uniform vec3 u_cameraPos;
uniform float u_waterThreshold;
uniform float u_noiseStrength;
uniform float u_time;
uniform bool u_shadowsEnabled;

uniform sampler2D u_shadowMap;

out vec4 outColor;

float sat(float x){ return clamp(x, 0.0, 1.0); }

// Simple hash for noise-based texture variation
float hash(vec3 p) {
    p = fract(p * 0.3183099 + .1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float sampleShadow(vec4 lightClip) {
  vec3 proj = lightClip.xyz / lightClip.w;
  vec2 uv = proj.xy * 0.5 + 0.5;
  float current = proj.z * 0.5 + 0.5;

  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1.0;

  float shadow = 0.0;
  vec2 texel = 1.0 / vec2(textureSize(u_shadowMap, 0));
  float bias = 0.001;
  for (int y = -2; y <= 2; y++) {
    for (int x = -2; x <= 2; x++) {
      float depth = texture(u_shadowMap, uv + vec2(float(x), float(y)) * texel).r;
      shadow += (current - bias) <= depth ? 1.0 : 0.0;
    }
  }
  return shadow / 25.0;
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
  if (h < threshold) {
    // ÁGUA
    float depth = sat((threshold - h) * 4.0);
    vec3 deepBlue = vec3(0.01, 0.05, 0.15);
    vec3 shallowBlue = vec3(0.05, 0.45, 0.75);
    color = mix(shallowBlue, deepBlue, depth);

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
    float shadow = sampleShadow(v_lightClip);

    // Specular (less pronounced than water)
    vec3 H = normalize(L + V);
    float spec = pow(sat(dot(N, H)), 40.0) * 0.2 * landMask;

    // Shadow only affects direct lighting, ambient is always preserved
    vec3 col = color * (0.2 + 0.8 * ndl * shadow);
    col += spec * vec3(1.0) * shadow;
    col += rim * vec3(0.4, 0.6, 1.0) * 0.15; // Atmosfera

    outColor = vec4(col, 1.0);
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

void main() {
  vec4 wp = u_world * vec4(a_position, 1.0);
  v_worldPos = wp.xyz;
  v_worldNormal = mat3(u_world) * a_normal;
  v_lightClip = u_lightVP * wp;
  gl_Position = u_proj * u_view * wp;
}
`;

const OBJ_FS = `#version 300 es
precision highp float;

in vec3 v_worldPos;
in vec3 v_worldNormal;
in vec4 v_lightClip;

uniform vec3 u_lightDir;
uniform vec3 u_cameraPos;
uniform vec3 u_albedo;
uniform bool u_shadowsEnabled;
uniform sampler2D u_shadowMap;

out vec4 outColor;

float sat(float x){ return clamp(x, 0.0, 1.0); }

float sampleShadow(vec4 lightClip) {
  vec3 proj = lightClip.xyz / lightClip.w;
  vec2 uv = proj.xy * 0.5 + 0.5;
  float current = proj.z * 0.5 + 0.5;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1.0;
  vec2 texel = 1.0 / vec2(textureSize(u_shadowMap, 0));
  float bias = 0.001;
  float shadow = 0.0;
  for (int y=-2; y<=2; y++){
    for (int x=-2; x<=2; x++){
      float depth = texture(u_shadowMap, uv + vec2(float(x), float(y)) * texel).r;
      shadow += (current - bias) <= depth ? 1.0 : 0.0;
    }
  }
  return shadow / 25.0;
}

void main() {
  vec3 N = normalize(v_worldNormal);
  vec3 L = normalize(-u_lightDir);
  vec3 V = normalize(u_cameraPos - v_worldPos);

  float ndl = sat(dot(N, L));

  if (u_shadowsEnabled) {
    float shadow = sampleShadow(v_lightClip);
    vec3 H = normalize(L + V);
    float spec = pow(sat(dot(N, H)), 80.0);

    // Shadow only affects direct lighting, ambient is always preserved
    vec3 col = u_albedo * (0.25 + 0.75 * ndl * shadow);
    col += spec * vec3(0.9, 1.0, 1.0) * 0.25 * shadow;

    outColor = vec4(col, 1.0);
  } else {
    // Without shadows: uniform ambient lighting
    vec3 col = u_albedo * 0.8;
    outColor = vec4(col, 1.0);
  }
}
`;

function sphereMesh(subdiv: number, settings: PlanetSettings) {
  const latSeg = subdiv;
  const lonSeg = subdiv * 2;

  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];

  const noiseBase = makeNoiseSampler(settings.seed, settings.noiseType as NoiseType);
  const noiseDetail = makeNoiseSampler(settings.seed + "_detail", settings.noiseType as NoiseType);
  const noiseMicro = makeNoiseSampler(settings.seed + "_micro", settings.noiseType as NoiseType);

  // Compute displacement from 3D Cartesian sphere direction (unit vector).
  // Sampling noise in 3D eliminates the longitude seam and pole convergence
  // artifacts that 2D lat/lon mapping causes.
  const getDisplacement = (dir: Vec3) => {
    const [dx, dy, dz] = dir;
    const n1 = noiseBase(dx, dy, dz);
    const n2 = noiseDetail(dx * 3.0, dy * 3.0, dz * 3.0);
    const n3 = noiseMicro(dx * 8.0, dy * 8.0, dz * 8.0);

    const h = (n1 * 1.0 + n2 * 0.5 + n3 * 0.25) * settings.noiseStrength;
    return h * 0.10;
  };

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

  private treeVao?: WebGLVertexArrayObject;
  private boatVao?: WebGLVertexArrayObject;

  private treeIndexCount = 0;
  private boatIndexCount = 0;

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
  private objectDrawLogged = false;

  private dragging = false;
  private lastPointerX = 0;
  private lastPointerY = 0;
  private cameraElevation = 0.11; // polar angle in radians

  private shadowVP: Mat4 = mat4Identity();
  private view: Mat4 = mat4Identity();
  private proj: Mat4 = mat4Identity();

  private lightDir: Vec3 = normalize([-0.35, -0.75, -0.55]);

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

    this.planetProg = createProgram(gl, PLANET_VS, PLANET_FS);
    this.shadowProg = createProgram(gl, SHADOW_VS, SHADOW_FS);
    this.objProg = createProgram(gl, OBJ_VS, OBJ_FS);

    this.shadow = createShadowTarget(gl, 2048);

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);

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
    console.log("✅ Renderer initialization complete");
  }

  private async loadModels(): Promise<boolean> {
    try {
      console.log("📦 Fetching models...");
      const [treeRes, boatRes] = await Promise.all([
        fetch("/models/tree.obj"),
        fetch("/models/12219_boat_v2_L2.obj")
      ]);

      if (!treeRes.ok || !boatRes.ok) {
        console.error("❌ Failed to fetch models:", {
          tree: treeRes.status,
          boat: boatRes.status
        });
        return false;
      }

      const [treeText, boatText] = await Promise.all([
        treeRes.text(),
        boatRes.text()
      ]);

      this.treeModel = parseObj(treeText);
      this.boatModel = parseObj(boatText);

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

      console.log("✅ Models loaded:", {
        tree: `${this.treeModel.positions.length / 3} verts, ${this.treeModel.indices.length / 3} tris`,
        boat: `${this.boatModel.positions.length / 3} verts, ${this.boatModel.indices.length / 3} tris`
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
    // Note: bound listeners from setupEventListeners are GC'd with the canvas
  }

  setSettings(next: PlanetSettings, opts?: { rebuild?: boolean; redistribute?: boolean }) {
    this.settings = next;
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
    if (!this.treeModel || !this.boatModel) {
      console.warn("⚠️ Cannot init objects: models not loaded", {
        treeModel: !!this.treeModel,
        boatModel: !!this.boatModel
      });
      return;
    }

    console.log("🔧 Initializing object VAOs...", {
      treeVerts: this.treeModel.positions.length / 3,
      treeTris: this.treeModel.indices.length / 3,
      boatVerts: this.boatModel.positions.length / 3,
      boatTris: this.boatModel.indices.length / 3
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
  }

  private rebuildPlanet() {
    const gl = this.gl;
    const mesh = sphereMesh(this.settings.subdivisions, this.settings);

    if (this.planetVao) {
      gl.deleteVertexArray(this.planetVao);
      this.planetVao = undefined;
    }

    const vaoInfo = createVao(
      gl,
      this.planetProg,
      [
        { name: "a_position", size: 3, data: mesh.positions },
        { name: "a_normal", size: 3, data: mesh.normals },
      ],
      mesh.indices,
    );

    this.planetVao = vaoInfo.vao;
    this.planetIndexCount = mesh.indices.length;
    this.planetTriCount = mesh.triCount;
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
    // directional light: ortho tightly around planet
    const lightPos: Vec3 = mulScalar(this.lightDir, -7.5);
    const center: Vec3 = [0, 0, 0];
    const up: Vec3 = [0, 1, 0];

    const lightView = mat4LookAt(lightPos, center, up);
    // Tight frustum: planet radius ~1.15, so ±1.5 covers it with margin.
    // Near/far: light at 7.5, planet surface at ~6.3–8.7 from light.
    const lightProj = mat4Ortho(-1.5, 1.5, -1.5, 1.5, 5.5, 10.0);
    this.shadowVP = mat4Mul(lightProj, lightView);
  }

  private redistributeObjects() {
    const rng = mulberry32(hashStringToUint(this.settings.seed) ^ 0x1337abcd);
    const noiseBase = makeNoiseSampler(this.settings.seed, this.settings.noiseType as NoiseType);
    const noiseDetail = makeNoiseSampler(this.settings.seed + "_detail", this.settings.noiseType as NoiseType);
    const noiseMicro = makeNoiseSampler(this.settings.seed + "_micro", this.settings.noiseType as NoiseType);

    const count = this.settings.objectCount;
    const maxBoats = 5;
    const trees: PlacedObject[] = [];
    const boats: PlacedObject[] = [];

    for (let i = 0; i < count; i++) {
      const z = randRange(rng, -1, 1);
      const t = randRange(rng, 0, Math.PI * 2);
      const r = Math.sqrt(1 - z * z);
      const base: Vec3 = [r * Math.cos(t), z, r * Math.sin(t)];

      const n1 = noiseBase(base[0], base[1], base[2]);
      const n2 = noiseDetail(base[0] * 3.0, base[1] * 3.0, base[2] * 3.0);
      const n3 = noiseMicro(base[0] * 8.0, base[1] * 8.0, base[2] * 8.0);
      const height = (n1 * 1.0 + n2 * 0.5 + n3 * 0.25) * this.settings.noiseStrength * 0.10;
      const isWater = height < this.settings.waterThreshold * 0.10;

      const normal = normalize(base);
      const tangent = anyPerpendicular(normal);
      const bitangent = normalize(cross(normal, tangent));

      const kind: ObjectKind = isWater ? "boat" : "tree";
      const scale = isWater ? randRange(rng, 0.10, 0.15) : randRange(rng, 0.22, 0.34);

      // position on surface (boat keel is at Y=0 so it sits above naturally; sink slightly for realism)
      const pos = mulScalar(normal, 1.0 + height + (isWater ? -0.005 : 0.01));

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
      } else {
        trees.push(obj);
      }
    }

    this.placed = [...trees, ...boats];
    console.log(`✓ Redistributed ${this.placed.length} objects:`, {
      trees: trees.length,
      boats: boats.length
    });
  }

  pick(clientX: number, clientY: number): PickResult {
    const rect = this.canvas.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * 2 - 1;
    const y = -(((clientY - rect.top) / rect.height) * 2 - 1);

    // Build inverse of proj*view approximately by unproject at near/far with manual method
    // We'll generate a ray in view space then rotate by camera orbit (only Y rotation).
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

    const tHit = raySphereIntersect(camPos, dirWorld, 1.25); // loose radius for displaced sphere
    if (tHit == null) return { hit: false };

    const hitPos = add(camPos, mulScalar(dirWorld, tHit));
    const normal = normalize(hitPos);

    // estimate water by noise at that spot
    const noiseBase = makeNoiseSampler(this.settings.seed, this.settings.noiseType as NoiseType);
    const noiseDetail = makeNoiseSampler(this.settings.seed + "_detail", this.settings.noiseType as NoiseType);
    const noiseMicro = makeNoiseSampler(this.settings.seed + "_micro", this.settings.noiseType as NoiseType);

    const n1 = noiseBase(normal[0], normal[1], normal[2]);
    const n2 = noiseDetail(normal[0] * 3.0, normal[1] * 3.0, normal[2] * 3.0);
    const n3 = noiseMicro(normal[0] * 8.0, normal[1] * 8.0, normal[2] * 8.0);
    const height = (n1 * 1.0 + n2 * 0.5 + n3 * 0.25) * this.settings.noiseStrength * 0.10;
    const isWater = height < this.settings.waterThreshold * 0.10;

    return {
      hit: true,
      worldPos: hitPos,
      worldNormal: normal,
      isWater,
    };
  }

  placeObjectFromPick(pick: PickResult) {
    if (!pick.hit || !pick.worldPos || !pick.worldNormal) {
      console.log("❌ Cannot place: invalid pick", pick);
      return;
    }

    console.log("✓ Placing object:", pick.isWater ? "boat" : "tree", "at", pick.worldPos);
    console.log("  VAOs ready?", { tree: !!this.treeVao, boat: !!this.boatVao });
    console.log("  Placed count before:", this.placed.length);

    const normal = normalize(pick.worldNormal);
    const tangent = anyPerpendicular(normal);
    const bitangent = normalize(cross(normal, tangent));

    const kind: ObjectKind = pick.isWater ? "boat" : "tree";
    const rng = mulberry32(hashStringToUint(this.settings.seed + "_place") ^ (this.placed.length * 2654435761));
    const scale = kind === "boat" ? randRange(rng, 0.10, 0.15) : randRange(rng, 0.22, 0.34);

    // Keep them hugging the surface (boat keel at Y=0, sinks slightly for realism)
    const pos = mulScalar(normal, len(pick.worldPos) + (kind === "boat" ? -0.005 : 0.012));

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

    console.log("✅ Object added! Total placed:", this.placed.length, "Last object:", {
      kind,
      scale,
      position: pos,
      posLength: len(pos)
    });
  }

  private renderShadowPass(timeS: number) {
    const gl = this.gl;
    this.computeLightVP();

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadow.fb);
    gl.viewport(0, 0, this.shadow.size, this.shadow.size);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.colorMask(false, false, false, false);

    gl.useProgram(this.shadowProg);

    const uWorld = gl.getUniformLocation(this.shadowProg, "u_world");
    const uLightVP = gl.getUniformLocation(this.shadowProg, "u_lightVP");

    // Hardware polygon offset bias — more precise than shader-only bias
    gl.enable(gl.POLYGON_OFFSET_FILL);
    gl.polygonOffset(1.5, 2.0);

    // planet — cull front faces so shadow map stores back-face depth (natural bias)
    gl.cullFace(gl.FRONT);
    gl.bindVertexArray(this.planetVao!);
    gl.uniformMatrix4fv(uWorld, false, mat4RotateY(this.rot));
    gl.uniformMatrix4fv(uLightVP, false, this.shadowVP);
    gl.drawElements(gl.TRIANGLES, this.planetIndexCount, gl.UNSIGNED_INT, 0);

    // objects — disable culling so left-handed basis doesn't hide faces from light
    gl.disable(gl.CULL_FACE);

    const planetRot = mat4RotateY(this.rot);

    const drawObj = (o: PlacedObject) => {
      let pos = o.position as Vec3;

      // boat bob (gentle)
      if (o.kind === "boat") {
        const bob = Math.sin(timeS * 1.8 + o.phase) * 0.005;
        pos = add(pos, mulScalar(o.normal as Vec3, bob));
      }

      const objLocal = this.worldFromBasis(pos, o.normal as Vec3, o.tangent as Vec3, o.bitangent as Vec3, o.scale);
      const world = mat4Mul(planetRot, objLocal);
      gl.uniformMatrix4fv(uWorld, false, world);
      gl.uniformMatrix4fv(uLightVP, false, this.shadowVP);

      if (o.kind === "tree") {
        if (!this.treeVao) return;
        gl.bindVertexArray(this.treeVao);
        gl.drawElements(gl.TRIANGLES, this.treeIndexCount, gl.UNSIGNED_INT, 0);
      } else {
        if (!this.boatVao) return;
        gl.bindVertexArray(this.boatVao);
        gl.drawElements(gl.TRIANGLES, this.boatIndexCount, gl.UNSIGNED_INT, 0);
      }
    };

    for (const o of this.placed) drawObj(o);

    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.disable(gl.POLYGON_OFFSET_FILL);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.colorMask(true, true, true, true);
  }

  private renderMainPass(timeS: number) {
    const gl = this.gl;

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
          hasBoatVao: !!this.boatVao
        });
      }
      let pos = o.position as Vec3;
      let albedo: Vec3;
      let t = o.tangent as Vec3;
      let b = o.bitangent as Vec3;

      if (o.kind === "boat") {
        const bob = Math.sin(timeS * 1.8 + o.phase) * 0.005;
        pos = add(pos, mulScalar(o.normal as Vec3, bob));
        albedo = [0.12, 0.62, 0.78];
      } else {
        // tree with tiny sway
        const sway = Math.sin(timeS * 1.6 + o.phase) * 0.015;
        t = normalize(add(o.tangent as Vec3, mulScalar(o.bitangent as Vec3, sway)));
        b = normalize(cross(o.normal as Vec3, t));
        albedo = [0.18, 0.58, 0.26];
      }

      const objLocal = this.worldFromBasis(pos, o.normal as Vec3, t, b, o.scale);
      const world = mat4Mul(planetRot, objLocal);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.objProg, "u_world"), false, world);
      gl.uniform3f(gl.getUniformLocation(this.objProg, "u_albedo"), albedo[0], albedo[1], albedo[2]);

      if (o.kind === "tree") {
        if (!this.treeVao) {
          if (drawnCount === 1) console.log("❌ Tree VAO missing!");
          return;
        }
        gl.bindVertexArray(this.treeVao);
        gl.drawElements(gl.TRIANGLES, this.treeIndexCount, gl.UNSIGNED_INT, 0);
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
  }

  private loop() {
    if (this.disposed) return;

    const t = nowMs();
    const dt = Math.max(0.001, (t - this.lastT) / 1000);
    this.lastT = t;

    // smooth rotation
    if (this.autoRotate) {
      this.rot += dt * 0.3; // planet spins
      this.cameraRot += dt * 0.15; // camera orbits
    }

    // fps smoothing
    const fps = 1 / dt;
    this.fpsSmoothed = this.fpsSmoothed * 0.92 + fps * 0.08;

    const timeS = t / 1000;

    this.resize();

    // render shadow then main
    if (this.settings.shadowsEnabled) this.renderShadowPass(timeS);
    else this.computeLightVP(); // still compute matrix for shading; map still used but ignored

    this.renderMainPass(timeS);

    this.raf = requestAnimationFrame(this.loop);
  }
}
