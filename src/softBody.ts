import * as THREE from 'three';

/**
 * Simulates the soft parts of a rigged character. The rig script stores the layout in the
 * root bone's `userData`. There are two kinds:
 *
 * Sheets (a cape): a grid of bones laid over the part. Each grid point is a particle: it is
 * pulled by gravity and wind, held to its neighbours by fixed-length links, kept out of the
 * body's collision capsules, and gently drawn back to where the part would sit if it were
 * rigid. Row 0 of a grid has no bone: those particles are the attachment line.
 *
 * Bodies (a shirt, hair): the part keeps its ordinary skin weights, so it bends with the body
 * exactly like the rest of the mesh. Control points are spread over it. Each one knows where it
 * would be if the part were rigid and is a particle on a spring to that place, linked to its
 * neighbours by springs that keep the modelled shape. How far each particle has strayed is
 * handed to the vertex shader, which adds it to the nearby vertices after skinning. Unlike a
 * sheet the part never hangs freely: it wobbles, lags and settles back.
 */
export interface SoftBodies {
  update(dt: number): void;
  /** Names and particle counts, for the on-screen readout. */
  summary: string;
  /** Tell the simulation that a joint's rest position was moved (viewer joint editing). */
  moveJoint(name: string, pos: [number, number, number]): void;
  /** Makes a material add the soft-body wobble (needed for any extra material shown on the model). */
  patch(material: THREE.Material): void;
}

interface SimSettings {
  gravity?: number; // downward acceleration, model units / s^2
  damping?: number; // share of velocity lost per step
  followTop?: number; // pull toward the rigid shape per step, at the first simulated row...
  followBottom?: number; // ...and at the last row
  wind?: number; // strength of the built-in breeze
  sideLinks?: number; // 0..1 stiffness of links between neighbouring columns (1 = cloth, low = separate strands)
}
interface NodeData {
  r: number;
  c: number;
  rest: [number, number, number];
  bone: string | null;
}
interface SoftData {
  name: string;
  parent: string;
  parentPos: [number, number, number];
  rows: number;
  columns: number;
  sim: SimSettings;
  nodes: NodeData[];
}
interface ColliderData {
  from: string;
  to?: string;
  offset?: [number, number, number];
  radius: number;
}
interface BodySettings {
  gravity?: number; // downward acceleration, model units / s^2
  damping?: number; // share of velocity lost per step
  follow?: number; // pull back to the modelled shape per step, where the part is fully soft
  stiffness?: number; // 0..1 strength of the springs between control points
  wind?: number; // strength of the built-in breeze
  maxOffset?: number; // furthest a fully soft point may drift from its modelled place
}
interface BodyNodeData {
  rest: [number, number, number];
  soft: number; // 0 = held rigidly, 1 = fully simulated
  skin: [string, number][]; // the ordinary bones this point follows, with weights
}
interface BodyData {
  name: string;
  sim: BodySettings;
  /** Number of this body's first control point in the shared list the shader reads. */
  first: number;
  nodes: BodyNodeData[];
  links: [number, number][];
}
interface RigData {
  restPositions: Record<string, [number, number, number]>;
  colliders: ColliderData[];
  soft: SoftData[];
  bodies?: BodyData[];
}

const STEP = 1 / 60;
const MAX_STEPS = 4;
const ITERATIONS = 4;

interface Link {
  a: number;
  b: number;
  length: number;
  stiffness: number;
}

class Sheet {
  readonly count: number;
  private pos: Float32Array;
  private prev: Float32Array;
  private target: Float32Array;
  private local: Float32Array; // rest position relative to the parent bone
  private follow: Float32Array; // 1 = pinned
  private rowShare: Float32Array; // 0 at the attachment, 1 at the free end
  private phase: Float32Array;
  private links: Link[] = [];
  private bones: (THREE.Bone | null)[];
  private parent: THREE.Bone;
  /** Allowed radius of each collider for each particle (never more than its distance at rest). */
  private reach: Float32Array;
  private started = false;
  private readonly inv = new THREE.Matrix4();

  constructor(
    private data: SoftData,
    bones: Map<string, THREE.Bone>,
    private colliders: Collider[],
  ) {
    const n = (this.count = data.nodes.length);
    this.pos = new Float32Array(3 * n);
    this.prev = new Float32Array(3 * n);
    this.target = new Float32Array(3 * n);
    this.local = new Float32Array(3 * n);
    this.follow = new Float32Array(n);
    this.rowShare = new Float32Array(n);
    this.phase = new Float32Array(n);
    this.parent = bones.get(data.parent)!;
    this.bones = data.nodes.map((node) => (node.bone ? (bones.get(node.bone) ?? null) : null));

    const sim = data.sim;
    const top = sim.followTop ?? 0.4;
    const bottom = sim.followBottom ?? 0.02;
    const index = new Map<string, number>();
    data.nodes.forEach((node, i) => {
      index.set(`${node.r}_${node.c}`, i);
      for (let k = 0; k < 3; k++) this.local[3 * i + k] = node.rest[k] - data.parentPos[k];
      const share = data.rows > 2 ? (node.r - 1) / (data.rows - 2) : 1;
      this.rowShare[i] = node.r === 0 ? 0 : Math.max(0, share);
      // Stiff near the attachment, loose toward the free end.
      this.follow[i] = node.bone ? top * Math.pow(bottom / top, Math.max(0, share)) : 1;
      this.phase[i] = node.c * 0.9 + node.r * 0.35;
    });

    const side = sim.sideLinks ?? 1;
    const link = (i: number, r: number, c: number, stiffness: number) => {
      const j = index.get(`${r}_${c}`);
      if (j === undefined || stiffness <= 0) return;
      const a = data.nodes[i].rest;
      const b = data.nodes[j].rest;
      this.links.push({ a: i, b: j, length: Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]), stiffness });
    };
    data.nodes.forEach((node, i) => {
      link(i, node.r + 1, node.c, 1); // down
      link(i, node.r, node.c + 1, side); // across
      link(i, node.r + 1, node.c + 1, 0.5 * side); // diagonals resist shearing
      link(i, node.r + 1, node.c - 1, 0.5 * side);
      link(i, node.r + 2, node.c, 0.25); // skip-one links resist sharp folds
    });

    // A collider may never push a particle further out than it already sits at rest,
    // otherwise the modelled shape would be distorted before anything moves.
    this.reach = new Float32Array(n * colliders.length);
    data.nodes.forEach((node, i) => {
      colliders.forEach((col, k) => {
        const d = distanceToSegment(node.rest, col.restA, col.restB);
        this.reach[i * colliders.length + k] = Math.min(col.radius, d - 0.004);
      });
    });
  }

  private computeTargets() {
    const m = this.parent.matrixWorld.elements;
    for (let i = 0; i < this.count; i++) {
      const x = this.local[3 * i], y = this.local[3 * i + 1], z = this.local[3 * i + 2];
      this.target[3 * i] = m[0] * x + m[4] * y + m[8] * z + m[12];
      this.target[3 * i + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
      this.target[3 * i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    }
  }

  /** Call once per frame before stepping: where the part would be if it were rigid. */
  prepare() {
    this.computeTargets();
    if (!this.started) {
      this.pos.set(this.target);
      this.prev.set(this.target);
      this.started = true;
    }
  }

  step(time: number) {
    const { pos, prev, target, follow, count } = this;
    const sim = this.data.sim;
    const g = (sim.gravity ?? 3) * STEP * STEP;
    const keep = 1 - (sim.damping ?? 0.04);
    const wind = (sim.wind ?? 0) * STEP * STEP;

    for (let i = 0; i < count; i++) {
      const o = 3 * i;
      if (follow[i] >= 1) {
        for (let k = 0; k < 3; k++) prev[o + k] = pos[o + k] = target[o + k];
        continue;
      }
      // A slow breeze from behind with gusts that travel across the part.
      const p = this.phase[i];
      const gust = 0.55 * Math.sin(0.7 * time + p) + 0.45 * Math.sin(1.9 * time + 1.7 * p);
      const wz = wind * this.rowShare[i] * (0.4 + gust);
      const wx = wind * this.rowShare[i] * 0.5 * Math.sin(1.1 * time + 0.6 * p);
      for (let k = 0; k < 3; k++) {
        const now = pos[o + k];
        let next = now + (now - prev[o + k]) * keep;
        if (k === 0) next += wx;
        else if (k === 1) next -= g;
        else next += wz;
        next += (target[o + k] - next) * follow[i];
        prev[o + k] = now;
        pos[o + k] = next;
      }
    }

    for (let it = 0; it < ITERATIONS; it++) {
      for (const l of this.links) {
        const a = 3 * l.a, b = 3 * l.b;
        const dx = pos[b] - pos[a], dy = pos[b + 1] - pos[a + 1], dz = pos[b + 2] - pos[a + 2];
        const d = Math.hypot(dx, dy, dz) || 1e-6;
        const wa = follow[l.a] >= 1 ? 0 : 1;
        const wb = follow[l.b] >= 1 ? 0 : 1;
        if (wa + wb === 0) continue;
        const f = ((d - l.length) / d) * l.stiffness / (wa + wb);
        pos[a] += dx * f * wa; pos[a + 1] += dy * f * wa; pos[a + 2] += dz * f * wa;
        pos[b] -= dx * f * wb; pos[b + 1] -= dy * f * wb; pos[b + 2] -= dz * f * wb;
      }
      this.collide();
    }
  }

  private collide() {
    const { pos, count, colliders } = this;
    const nc = colliders.length;
    for (let i = 0; i < count; i++) {
      if (this.follow[i] >= 1) continue;
      const o = 3 * i;
      for (let k = 0; k < nc; k++) {
        const radius = this.reach[i * nc + k];
        if (radius <= 0) continue;
        const col = colliders[k];
        const ax = col.a.x, ay = col.a.y, az = col.a.z;
        const abx = col.b.x - ax, aby = col.b.y - ay, abz = col.b.z - az;
        const len2 = abx * abx + aby * aby + abz * abz;
        let t = len2 > 0 ? ((pos[o] - ax) * abx + (pos[o + 1] - ay) * aby + (pos[o + 2] - az) * abz) / len2 : 0;
        t = Math.min(1, Math.max(0, t));
        const dx = pos[o] - (ax + abx * t), dy = pos[o + 1] - (ay + aby * t), dz = pos[o + 2] - (az + abz * t);
        const d = Math.hypot(dx, dy, dz);
        if (d >= radius || d < 1e-6) continue;
        const push = (radius - d) / d;
        pos[o] += dx * push; pos[o + 1] += dy * push; pos[o + 2] += dz * push;
      }
    }
  }

  /** Move the grid bones to the particles. */
  apply() {
    this.inv.copy(this.parent.matrixWorld).invert();
    for (let i = 0; i < this.count; i++) {
      const bone = this.bones[i];
      if (!bone) continue;
      if (!Number.isFinite(this.pos[3 * i])) {
        // Should never happen; recover by restarting from the rigid shape.
        this.started = false;
        return;
      }
      bone.position.set(this.pos[3 * i], this.pos[3 * i + 1], this.pos[3 * i + 2]).applyMatrix4(this.inv);
    }
  }
}

/** A soft part that keeps its modelled shape: control points on springs. */
class Volume {
  readonly count: number;
  private pos: Float32Array;
  private prev: Float32Array;
  private target: Float32Array;
  /** Per point: the blended transform of the bones it follows (rotation columns, then translation). */
  private frame: Float32Array;
  private follow: Float32Array; // 1 = held
  private softness: Float32Array;
  private phase: Float32Array;
  private links: Link[] = [];
  private skins: { bone: THREE.Bone; weight: number; offset: [number, number, number] }[][];
  private reach: Float32Array;
  private started = false;

  constructor(
    private data: BodyData,
    bones: Map<string, THREE.Bone>,
    rest: Record<string, [number, number, number]>,
    private colliders: Collider[],
    /** Shared list of how far every control point has strayed (x, y, z each), read by the shader. */
    private offsets: Float32Array,
  ) {
    const n = (this.count = data.nodes.length);
    this.pos = new Float32Array(3 * n);
    this.prev = new Float32Array(3 * n);
    this.target = new Float32Array(3 * n);
    this.frame = new Float32Array(12 * n);
    this.follow = new Float32Array(n);
    this.softness = new Float32Array(n);
    this.phase = new Float32Array(n);
    this.skins = [];
    this.rebase(bones, rest);
    const pull = data.sim.follow ?? 0.1;
    data.nodes.forEach((node, i) => {
      const soft = node.soft < 0.05 ? 0 : Math.min(1, node.soft);
      this.softness[i] = soft;
      this.follow[i] = 1 - soft * (1 - pull);
      this.phase[i] = node.rest[0] * 9 + node.rest[1] * 4;
    });
    const stiffness = data.sim.stiffness ?? 1;
    for (const [a, b] of data.links) {
      const p = data.nodes[a].rest, q = data.nodes[b].rest;
      this.links.push({ a, b, length: Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]), stiffness });
    }
    this.reach = new Float32Array(n * colliders.length);
    data.nodes.forEach((node, i) => {
      colliders.forEach((col, k) => {
        const d = distanceToSegment(node.rest, col.restA, col.restB);
        this.reach[i * colliders.length + k] = Math.min(col.radius, d - 0.004);
      });
    });
  }

  /** Works out where each control point sits relative to the bones it follows, at rest. */
  rebase(bones: Map<string, THREE.Bone>, rest: Record<string, [number, number, number]>) {
    this.skins = this.data.nodes.map((node) =>
      node.skin
        .filter(([name]) => bones.has(name) && rest[name])
        .map(([name, weight]) => ({
          bone: bones.get(name)!,
          weight,
          offset: [node.rest[0] - rest[name][0], node.rest[1] - rest[name][1], node.rest[2] - rest[name][2]] as [number, number, number],
        })),
    );
  }

  /** Call once per frame before stepping: where each point would be if the part were rigid. */
  prepare() {
    const { frame, target } = this;
    for (let i = 0; i < this.count; i++) {
      const f = 12 * i;
      frame.fill(0, f, f + 12);
      for (const { bone, weight, offset } of this.skins[i]) {
        const m = bone.matrixWorld.elements;
        for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) frame[f + 3 * c + r] += weight * m[4 * c + r];
        for (let r = 0; r < 3; r++) {
          frame[f + 9 + r] += weight * (m[r] * offset[0] + m[4 + r] * offset[1] + m[8 + r] * offset[2] + m[12 + r]);
        }
      }
      for (let r = 0; r < 3; r++) target[3 * i + r] = frame[f + 9 + r];
    }
    if (!this.started) {
      this.pos.set(target);
      this.prev.set(target);
      this.started = true;
    }
  }

  step(time: number) {
    const { pos, prev, target, follow, softness, count } = this;
    const sim = this.data.sim;
    const g = (sim.gravity ?? 3) * STEP * STEP;
    const keep = 1 - (sim.damping ?? 0.05);
    const wind = (sim.wind ?? 0) * STEP * STEP;
    const limit = sim.maxOffset ?? 0.25;

    for (let i = 0; i < count; i++) {
      const o = 3 * i;
      const soft = softness[i];
      if (soft <= 0) {
        for (let k = 0; k < 3; k++) prev[o + k] = pos[o + k] = target[o + k];
        continue;
      }
      const p = this.phase[i];
      const gust = 0.55 * Math.sin(0.7 * time + p) + 0.45 * Math.sin(1.9 * time + 1.7 * p);
      const push = [wind * soft * 0.5 * Math.sin(1.1 * time + 0.6 * p), -g * soft, wind * soft * (0.4 + gust)];
      for (let k = 0; k < 3; k++) {
        const now = pos[o + k];
        let next = now + (now - prev[o + k]) * keep + push[k];
        next += (target[o + k] - next) * follow[i];
        prev[o + k] = now;
        pos[o + k] = next;
      }
    }

    for (let it = 0; it < ITERATIONS; it++) {
      for (const l of this.links) {
        const a = 3 * l.a, b = 3 * l.b;
        const dx = pos[b] - pos[a], dy = pos[b + 1] - pos[a + 1], dz = pos[b + 2] - pos[a + 2];
        const d = Math.hypot(dx, dy, dz) || 1e-6;
        const wa = softness[l.a], wb = softness[l.b]; // held points do not give way
        if (wa + wb <= 0) continue;
        const f = (((d - l.length) / d) * l.stiffness) / (wa + wb);
        pos[a] += dx * f * wa; pos[a + 1] += dy * f * wa; pos[a + 2] += dz * f * wa;
        pos[b] -= dx * f * wb; pos[b + 1] -= dy * f * wb; pos[b + 2] -= dz * f * wb;
      }
      this.collide();
    }

    // Never drift further from the modelled place than allowed.
    for (let i = 0; i < count; i++) {
      const o = 3 * i;
      const dx = pos[o] - target[o], dy = pos[o + 1] - target[o + 1], dz = pos[o + 2] - target[o + 2];
      const d = Math.hypot(dx, dy, dz);
      const max = limit * softness[i];
      if (d > max && d > 0) {
        const k = max / d;
        pos[o] = target[o] + dx * k; pos[o + 1] = target[o + 1] + dy * k; pos[o + 2] = target[o + 2] + dz * k;
      }
    }
  }

  private collide() {
    const { pos, count, colliders } = this;
    const nc = colliders.length;
    for (let i = 0; i < count; i++) {
      if (this.softness[i] <= 0) continue;
      const o = 3 * i;
      for (let k = 0; k < nc; k++) {
        const radius = this.reach[i * nc + k];
        if (radius <= 0) continue;
        const col = colliders[k];
        const ax = col.a.x, ay = col.a.y, az = col.a.z;
        const abx = col.b.x - ax, aby = col.b.y - ay, abz = col.b.z - az;
        const len2 = abx * abx + aby * aby + abz * abz;
        let t = len2 > 0 ? ((pos[o] - ax) * abx + (pos[o + 1] - ay) * aby + (pos[o + 2] - az) * abz) / len2 : 0;
        t = Math.min(1, Math.max(0, t));
        const dx = pos[o] - (ax + abx * t), dy = pos[o + 1] - (ay + aby * t), dz = pos[o + 2] - (az + abz * t);
        const d = Math.hypot(dx, dy, dz);
        if (d >= radius || d < 1e-6) continue;
        const push = (radius - d) / d;
        pos[o] += dx * push; pos[o + 1] += dy * push; pos[o + 2] += dz * push;
      }
    }
  }

  /** Publish how far each control point has strayed. `simulate` false = not at all (plain skin). */
  apply(simulate: boolean) {
    const { pos, target, offsets } = this;
    const first = 3 * this.data.first;
    for (let i = 0; i < 3 * this.count; i++) {
      if (simulate && !Number.isFinite(pos[i])) {
        this.started = false; // should never happen; restart from the rigid shape
        offsets.fill(0, first, first + 3 * this.count);
        return;
      }
      offsets[first + i] = simulate ? pos[i] - target[i] : 0;
    }
  }
}

interface Collider {
  radius: number;
  restA: [number, number, number];
  restB: [number, number, number];
  a: THREE.Vector3;
  b: THREE.Vector3;
  update(): void;
}

function distanceToSegment(p: number[], a: number[], b: number[]) {
  const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
  const len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
  const t = len2 > 0 ? Math.min(1, Math.max(0, (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / len2)) : 0;
  return Math.hypot(ap[0] - ab[0] * t, ap[1] - ab[1] * t, ap[2] - ab[2] * t);
}

/** `simulate` false keeps every soft part rigid on the body (used by the viewer's ?still). */
export function createSoftBodies(model: THREE.Object3D, bones: Map<string, THREE.Bone>, simulate = true): SoftBodies | null {
  let rig: RigData | null = null;
  for (const bone of bones.values()) {
    if (bone.userData.soft) rig = bone.userData as RigData;
  }
  if (!rig || !(rig.soft.length + (rig.bodies?.length ?? 0))) return null;
  const rest = rig.restPositions;

  const colliders: Collider[] = [];
  for (const c of rig.colliders ?? []) {
    const from = bones.get(c.from);
    const to = c.to ? bones.get(c.to) : null;
    if (!from || !rest[c.from] || (c.to && (!to || !rest[c.to]))) continue;
    const offset = new THREE.Vector3(...(c.offset ?? [0, 0, 0]));
    const restA = rest[c.from];
    const restB: [number, number, number] = c.to ? rest[c.to] : [restA[0] + offset.x, restA[1] + offset.y, restA[2] + offset.z];
    const col: Collider = {
      radius: c.radius,
      restA,
      restB,
      a: new THREE.Vector3(),
      b: new THREE.Vector3(),
      update() {
        col.a.setFromMatrixPosition(from.matrixWorld);
        if (to) col.b.setFromMatrixPosition(to.matrixWorld);
        else col.b.copy(offset).applyMatrix4(from.matrixWorld);
      },
    };
    colliders.push(col);
  }

  const sheets = simulate ? rig.soft.filter((s) => bones.has(s.parent)).map((s) => new Sheet(s, bones, colliders)) : [];
  const pointCount = (rig.bodies ?? []).reduce((n, b) => Math.max(n, b.first + b.nodes.length), 0);
  const offsets = new Float32Array(3 * Math.max(1, pointCount));
  const volumes = (rig.bodies ?? []).map((b) => new Volume(b, bones, rest, colliders, offsets));

  // The vertex shader adds each vertex's share of its control points' wobble after skinning.
  // (The meshes sit at the origin with no transform, so skinned positions are world positions.)
  const patched = new WeakSet<THREE.Material>();
  const patch = (material: THREE.Material) => {
    if (!pointCount || patched.has(material)) return;
    patched.add(material);
    material.onBeforeCompile = (shader) => {
      shader.uniforms.softOffsets = { value: offsets };
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', `#include <common>\nattribute vec4 _softindex;\nattribute vec4 _softweight;\nuniform vec3 softOffsets[${pointCount}];`)
        .replace(
          '#include <skinning_vertex>',
          `#include <skinning_vertex>\nfor (int k = 0; k < 4; k++) transformed += _softweight[k] * softOffsets[int(_softindex[k] + 0.5)];`,
        );
    };
    material.customProgramCacheKey = () => `soft${pointCount}`;
    material.needsUpdate = true;
  };
  model.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry.getAttribute('_softindex')) return;
    for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) patch(material);
  });
  let time = 0;
  let pending = 0;
  return {
    summary: [...rig.soft, ...(rig.bodies ?? [])].map((s) => s.name).join(', '),
    patch,
    moveJoint(name, pos) {
      rest[name] = pos;
      for (const volume of volumes) volume.rebase(bones, rest);
    },
    update(dt) {
      model.updateMatrixWorld(true);
      for (const col of colliders) col.update();
      for (const sheet of sheets) sheet.prepare();
      for (const volume of volumes) volume.prepare();
      pending = Math.min(pending + dt, MAX_STEPS * STEP);
      while (simulate && pending >= STEP) {
        pending -= STEP;
        time += STEP;
        for (const sheet of sheets) sheet.step(time);
        for (const volume of volumes) volume.step(time);
      }
      for (const sheet of sheets) sheet.apply();
      for (const volume of volumes) volume.apply(simulate);
    },
  };
}
