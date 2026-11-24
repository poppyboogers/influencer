// Turns a static character mesh into a rigged one:
//   1. optionally cuts the mouth open along a plane so the lower jaw can separate,
//   2. optionally adds a mouth interior (cavity, tongue, teeth),
//   3. builds a skeleton from characters/<id>/<id>.rig.json (body, jaw, brows, mouth corners, fingers),
//   4. computes skin weights (nearest bone, then smoothed across the surface),
//   5. sets up soft parts that the app simulates. Two kinds:
//        "sheet" (default): a rows x columns grid of bones, hangs like cloth (cape),
//        "body": the part keeps its ordinary skin weights; control points spread over it
//                wobble around the shape they were modelled in and spring back, and each
//                vertex adds the wobble of its nearest control points on top (shirt, hair),
//   6. fixes rigid parts (teeth, glasses) to a single bone,
//   7. builds the facial expression shapes ("morphs" in the rig file) as morph targets.
// Parts are found by object name when the GLB has separate objects ("parts" in the rig file),
// otherwise by the colour / position "select" rules of each soft part.
// Usage: npm run rig -- vanduude [--parts]   ->  characters/<id>/<id>.rigged.glb
//        --parts also writes characters/<id>/<id>.parts.glb, coloured by part, to check the selection.
import fs from 'node:fs';
import { NodeIO } from '@gltf-transform/core';

const id = process.argv[2] ?? 'vanduude';
// Everything that belongs to a character lives in its own folder; file names in the rig file are relative to it.
const dir = `characters/${id}`;
const cfg = JSON.parse(fs.readFileSync(`${dir}/${id}.rig.json`, 'utf8'));
const M = cfg.mouth ?? { cut: false, interior: false };
const softCfg = cfg.soft ?? [];
const debugParts = process.argv.includes('--parts');
// Vertex-group weights saved from Blender by tools/blend_weights.py (the GLB export drops them).
const weightData = cfg.weights && fs.existsSync(`${dir}/${cfg.weights}`) ? JSON.parse(fs.readFileSync(`${dir}/${cfg.weights}`, 'utf8')) : {};

const io = new NodeIO();
const doc = await io.read(`${dir}/${cfg.source}`);
const root = doc.getRoot();
const buffer = root.listBuffers()[0];

// ---- pull every primitive into shared arrays we can grow ----------------------
const prims = [];
for (const node of root.listNodes()) {
  const mesh = node.getMesh();
  if (!mesh) continue;
  const t = node.getTranslation(), r = node.getRotation(), s = node.getScale();
  if (Math.hypot(...t) > 1e-6 || Math.abs(r[3]) < 0.999999 || Math.abs(s[0] - 1) + Math.abs(s[1] - 1) + Math.abs(s[2] - 1) > 1e-6) {
    console.warn(`Warning: object "${node.getName()}" has its own transform; run scripts/normalize-glb.mjs first.`);
  }
  for (const prim of mesh.listPrimitives()) prims.push({ prim, mesh, node, name: `${node.getName()} ${mesh.getName()}`.toLowerCase() });
}
const P = [], N = [], UV = [];
const primOf = []; // vertex -> primitive
const idx = []; // triangles (global vertex numbers)
const triPrim = []; // triangle -> primitive
const srcTris = [];
prims.forEach((p, k) => {
  const base = P.length / 3;
  const pos = p.prim.getAttribute('POSITION').getArray();
  const nrm = p.prim.getAttribute('NORMAL')?.getArray();
  const uv = p.prim.getAttribute('TEXCOORD_0')?.getArray();
  p.hasUV = !!uv;
  const count = pos.length / 3;
  for (let i = 0; i < count; i++) {
    P.push(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    N.push(nrm ? nrm[3 * i] : 0, nrm ? nrm[3 * i + 1] : 1, nrm ? nrm[3 * i + 2] : 0);
    UV.push(uv ? uv[2 * i] : 0, uv ? uv[2 * i + 1] : 0);
    primOf.push(k);
  }
  const ind = p.prim.getIndices()?.getArray() ?? Array.from({ length: count }, (_, i) => i);
  for (let t = 0; t < ind.length; t += 3) srcTris.push([ind[t] + base, ind[t + 1] + base, ind[t + 2] + base, k]);
});
/** 0 = ordinary vertex, +1 / -1 = upper / lower copy of a vertex created on the mouth cut. */
const cutSide = new Array(P.length / 3).fill(0);

const smoothstep = (a, b, v) => {
  const t = Math.min(1, Math.max(0, (v - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

// ---- skeleton (needed early: parts refer to bones by name) ---------------------
const bones = [];
for (const b of cfg.bones) {
  bones.push({ ...b, sideSign: b.side ? 1 : 0 });
  if (b.mirror) {
    const flip = (v) => v && [-v[0], v[1], v[2]];
    const rename = (n) => n && n.replace(/^left/, 'right');
    const keep = b.keep && { ...b.keep, normal: flip(b.keep.normal) };
    bones.push({ ...b, name: rename(b.name), parent: rename(b.parent), pos: flip(b.pos), tail: flip(b.tail), keep, sideSign: b.side ? -1 : 0 });
  }
}
// Joint positions edited in the viewer (characters/<id>/<id>.joints.json) replace the ones in the rig file.
// A bone that ends where the moved joint began (upper arm -> elbow) has its end moved along with it.
const jointsFile = `${dir}/${id}.joints.json`;
const jointEdits = fs.existsSync(jointsFile) ? JSON.parse(fs.readFileSync(jointsFile, 'utf8')) : {};
{
  const gap = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const before = new Map(bones.map((b) => [b.name, { pos: b.pos, tail: b.tail }]));
  for (const b of bones) {
    const to = jointEdits[b.name];
    if (!to) continue;
    const from = before.get(b.name).pos;
    b.pos = to;
    const parent = bones.find((p) => p.name === b.parent);
    if (parent?.tail && gap(before.get(parent.name).tail, from) < 0.05) parent.tail = to;
    // A bone whose end is not another joint (a hand, a toe) keeps its length and direction.
    const endsAtChild = bones.some((c) => c.parent === b.name && b.tail && gap(before.get(c.name).pos, before.get(b.name).tail) < 0.05);
    if (b.tail && !endsAtChild) b.tail = b.tail.map((v, k) => v + to[k] - from[k]);
  }
}
const boneIndex = new Map(bones.map((b, i) => [b.name, i]));
const mustBone = (name) => {
  if (!boneIndex.has(name)) throw new Error(`Rig file refers to unknown bone "${name}"`);
  return boneIndex.get(name);
};

// ---- which part does each primitive belong to? ----------------------------------
// partSoft[k] = index into cfg.soft, partBone[k] = bone index, both -1 for the body.
const partSoft = prims.map(() => -1);
const partBone = prims.map(() => -1);
prims.forEach((p, k) => {
  for (const rule of cfg.parts ?? []) {
    if (!new RegExp(rule.match, 'i').test(p.name)) continue;
    if (rule.soft) partSoft[k] = softCfg.findIndex((s) => s.name === rule.soft);
    else if (rule.bone) partBone[k] = mustBone(rule.bone);
    // "offset": [x, y, z] moves the whole object by that much (model units) before rigging.
    if (rule.offset) {
      for (let i = 0; i < primOf.length; i++) {
        if (primOf[i] === k) for (let c = 0; c < 3; c++) P[3 * i + c] += rule.offset[c];
      }
    }
    break;
  }
});
const softHasOwnMesh = softCfg.map((_, s) => partSoft.includes(s));

// ---- 1. cut the mouth --------------------------------------------------------
// Triangles crossing the mouth plane inside the cut region are split along the plane,
// and the new edge vertices are duplicated so the upper and lower lips are not joined.
// Set "cut": false in the rig file when the mouth has been modelled by hand.
const isBodyVertex = (i) => partSoft[primOf[i]] < 0 && partBone[primOf[i]] < 0;
const inCut = (i) =>
  M.cut !== false && isBodyVertex(i) && P[3 * i + 2] > M.minZ && Math.abs(P[3 * i]) < M.maxAbsX && Math.abs(P[3 * i + 1] - M.planeY) < 0.06;
const sideOf = (i) => (P[3 * i + 1] >= M.planeY ? 1 : -1);
const edgeVerts = new Map();

function splitVertex(a, b, side) {
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  const key = `${lo}_${hi}_${side}`;
  let v = edgeVerts.get(key);
  if (v !== undefined) return v;
  const t = (M.planeY - P[3 * lo + 1]) / (P[3 * hi + 1] - P[3 * lo + 1]);
  const lerp = (arr, n, k) => arr[n * lo + k] + (arr[n * hi + k] - arr[n * lo + k]) * t;
  v = P.length / 3;
  P.push(lerp(P, 3, 0), M.planeY, lerp(P, 3, 2));
  const n = [lerp(N, 3, 0), lerp(N, 3, 1), lerp(N, 3, 2)];
  const len = Math.hypot(...n) || 1;
  N.push(n[0] / len, n[1] / len, n[2] / len);
  UV.push(lerp(UV, 2, 0), lerp(UV, 2, 1));
  cutSide.push(side);
  primOf.push(primOf[a]);
  edgeVerts.set(key, v);
  return v;
}

let cutTriangles = 0;
for (const [v0, v1, v2, k] of srcTris) {
  let tri = [v0, v1, v2];
  const s = tri.map(sideOf);
  if ((s[0] === s[1] && s[1] === s[2]) || !tri.every(inCut)) {
    idx.push(...tri);
    triPrim.push(k);
    continue;
  }
  // Rotate so tri[0] is the vertex alone on its side (keeps the winding order).
  if (s[1] !== s[0] && s[1] !== s[2]) tri = [tri[1], tri[2], tri[0]];
  else if (s[2] !== s[0] && s[2] !== s[1]) tri = [tri[2], tri[0], tri[1]];
  const [a, b, c] = tri;
  const sa = sideOf(a);
  const sb = -sa;
  idx.push(a, splitVertex(a, b, sa), splitVertex(a, c, sa));
  const pab = splitVertex(a, b, sb);
  idx.push(pab, b, c, pab, c, splitVertex(a, c, sb));
  triPrim.push(k, k, k);
  cutTriangles++;
}
const isLower = (i) => (cutSide[i] !== 0 ? cutSide[i] < 0 : P[3 * i + 1] < M.planeY);

// ---- group vertices that share a position (UV seams duplicate them) -----------
const nVerts = P.length / 3;
const groupOf = new Int32Array(nVerts);
const groupRep = [];
{
  const seen = new Map();
  for (let i = 0; i < nVerts; i++) {
    const key = `${Math.round(P[3 * i] * 1e5)}_${Math.round(P[3 * i + 1] * 1e5)}_${Math.round(P[3 * i + 2] * 1e5)}_${cutSide[i]}_${primOf[i]}`;
    let g = seen.get(key);
    if (g === undefined) {
      g = groupRep.length;
      groupRep.push(i);
      seen.set(key, g);
    }
    groupOf[i] = g;
  }
}
const nGroups = groupRep.length;
const neighbours = Array.from({ length: nGroups }, () => new Set());
for (let t = 0; t < idx.length; t += 3) {
  const g = [groupOf[idx[t]], groupOf[idx[t + 1]], groupOf[idx[t + 2]]];
  for (let k = 0; k < 3; k++) {
    const a = g[k];
    const b = g[(k + 1) % 3];
    if (a !== b) {
      neighbours[a].add(b);
      neighbours[b].add(a);
    }
  }
}

// ---- 5a. label soft and rigid vertices ----------------------------------------
// softOf[g] = index into cfg.soft or -1; rigidOf[g] = bone index or -1.
const softOf = new Int32Array(nGroups).fill(-1);
const rigidOf = new Int32Array(nGroups).fill(-1);
for (let g = 0; g < nGroups; g++) {
  const k = primOf[groupRep[g]];
  softOf[g] = partSoft[k];
  rigidOf[g] = partBone[k];
}

// Fallback for a fused mesh: pick the part by texture colour and position.
async function loadTextures() {
  let sharp;
  try {
    sharp = (await import('sharp')).default;
  } catch {
    console.warn('Warning: "sharp" is not installed, colour-based part selection is skipped.');
    return null;
  }
  return Promise.all(
    prims.map(async (p) => {
      const tex = p.prim.getMaterial()?.getBaseColorTexture();
      if (!tex || !p.hasUV) return null;
      const { data, info } = await sharp(Buffer.from(tex.getImage())).removeAlpha().raw().toBuffer({ resolveWithObject: true });
      return { data, w: info.width, h: info.height };
    }),
  );
}
const needColour = softCfg.some((s, k) => !softHasOwnMesh[k] && s.select);
const textures = needColour ? await loadTextures() : null;
function colourAt(i) {
  const tex = textures?.[primOf[i]];
  if (!tex) return null;
  const u = UV[2 * i] - Math.floor(UV[2 * i]);
  const v = UV[2 * i + 1] - Math.floor(UV[2 * i + 1]);
  // Average a small patch: a single texel is too noisy on a detailed texture.
  const cx = Math.floor(u * tex.w), cy = Math.floor(v * tex.h);
  const sum = [0, 0, 0];
  let n = 0;
  for (let dy = -2; dy <= 2; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      const x = cx + dx, y = cy + dy;
      if (x < 0 || y < 0 || x >= tex.w || y >= tex.h) continue;
      const q = 3 * (y * tex.w + x);
      sum[0] += tex.data[q]; sum[1] += tex.data[q + 1]; sum[2] += tex.data[q + 2];
      n++;
    }
  }
  return [sum[0] / n, sum[1] / n, sum[2] / n];
}
const colourTests = {
  // Any colour: the part is picked by its boxes alone (hair the same colour as the fur under it).
  any: () => true,
  red: ([r, g, b]) => r > 20 && r > 1.8 * g && r > 1.8 * b,
  notRed: ([r, g, b]) => !(r > 20 && r > 1.8 * g && r > 1.8 * b),
  dark: ([r, g, b], limit = 40) => Math.max(r, g, b) < limit && !(r > 20 && r > 1.8 * g && r > 1.8 * b),
  // Printed fabric against grey-mauve skin: strongly coloured, or leaning blue, green or yellow
  // (skin is pinkish: its green and blue are about equal).
  // `limit` is the saturation (0..100) above which a colour counts as fabric.
  fabric: ([r, g, b], limit = 30) => {
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    return (max - min) / (max || 1) > limit / 100 || b > 1.08 * r || g > 1.12 * b;
  },
};
function inBox(box, x, y, z) {
  const within = (range, v) => !range || (v >= range[0] && v <= range[1]);
  return within(box.x, x) && within(box.y, y) && within(box.z, z) && within(box.absX, Math.abs(x));
}
softCfg.forEach((s, k) => {
  if (softHasOwnMesh[k] || !s.select || !textures) return;
  const test = colourTests[s.select.color];
  const picked = new Uint8Array(nGroups);
  for (let g = 0; g < nGroups; g++) {
    if (softOf[g] >= 0 || rigidOf[g] >= 0) continue;
    const i = groupRep[g];
    const [x, y, z] = [P[3 * i], P[3 * i + 1], P[3 * i + 2]];
    if (!s.select.boxes.some((box) => inBox(box, x, y, z))) continue;
    // A position can have several texture samples (UV seams): any match counts.
    picked[g] = 2; // in the region
  }
  for (let i = 0; i < nVerts; i++) {
    const g = groupOf[i];
    if (picked[g] !== 2) continue;
    const c = colourAt(i);
    if (c && test(c, s.select.limit)) picked[g] = 1;
  }
  // Optionally spread from the confident picks through matching neighbours inside wider boxes
  // (hair strands hang well below the head, where position alone cannot tell them from armour).
  if (s.select.growBoxes) {
    const match = new Uint8Array(nGroups);
    for (let i = 0; i < nVerts; i++) {
      const c = colourAt(i);
      if (c && test(c, s.select.limit)) match[groupOf[i]] = 1;
    }
    let frontier = [];
    for (let g = 0; g < nGroups; g++) if (picked[g] === 1) frontier.push(g);
    while (frontier.length) {
      const next = [];
      for (const g of frontier) {
        for (const h of neighbours[g]) {
          if (picked[h] === 1 || !match[h] || softOf[h] >= 0 || rigidOf[h] >= 0) continue;
          const i = groupRep[h];
          if (!s.select.growBoxes.some((box) => inBox(box, P[3 * i], P[3 * i + 1], P[3 * i + 2]))) continue;
          picked[h] = 1;
          next.push(h);
        }
      }
      frontier = next;
    }
  }
  // Tidy up speckle: a vertex takes the majority label of its neighbours (region vertices only).
  for (let pass = 0; pass < (s.select.fill ?? 2); pass++) {
    const next = picked.slice();
    for (let g = 0; g < nGroups; g++) {
      if (!picked[g]) continue;
      let yes = 0, all = 0;
      for (const h of neighbours[g]) {
        all++;
        if (picked[h] === 1) yes++;
      }
      if (all) next[g] = yes * 2 > all ? 1 : yes * 2 < all ? 2 : picked[g];
    }
    picked.set(next);
  }
  // Optionally fill holes: patches inside the region that the colour test missed and that are
  // completely surrounded by the part (a pale flower on a dark shirt).
  if (s.select.fillHoles) {
    const seen = new Uint8Array(nGroups);
    for (let g = 0; g < nGroups; g++) {
      if (picked[g] !== 2 || seen[g]) continue;
      const patch = [];
      const stack = [g];
      seen[g] = 1;
      let enclosed = true;
      while (stack.length) {
        const a = stack.pop();
        patch.push(a);
        for (const h of neighbours[a]) {
          if (picked[h] === 0) enclosed = false; // touches something outside the region
          else if (picked[h] === 2 && !seen[h]) {
            seen[h] = 1;
            stack.push(h);
          }
        }
      }
      if (enclosed) for (const a of patch) picked[a] = 1;
    }
  }
  // Optionally keep only the largest connected pieces (drops stray patches such as jewellery).
  if (s.select.pieces) {
    const label = new Int32Array(nGroups).fill(-1);
    const sizes = [];
    for (let g = 0; g < nGroups; g++) {
      if (picked[g] !== 1 || label[g] >= 0) continue;
      const id = sizes.length;
      let size = 0;
      const stack = [g];
      label[g] = id;
      while (stack.length) {
        const a = stack.pop();
        size++;
        for (const h of neighbours[a]) {
          if (picked[h] === 1 && label[h] < 0) {
            label[h] = id;
            stack.push(h);
          }
        }
      }
      sizes.push(size);
    }
    const keep = new Set(sizes.map((size, id) => [size, id]).sort((a, b) => b[0] - a[0]).slice(0, s.select.pieces).map((e) => e[1]));
    for (let g = 0; g < nGroups; g++) if (picked[g] === 1 && !keep.has(label[g])) picked[g] = 2;
  }
  let count = 0;
  for (let g = 0; g < nGroups; g++) {
    if (picked[g] === 1) {
      softOf[g] = k;
      count++;
    }
  }
  console.log(`Soft part "${s.name}": no separate object found, picked ${count} points by colour (${s.select.color}).`);
});

// ---- 5b. bone grids for the soft parts -------------------------------------------
// Each soft part gets rows x columns of bones laid over it (x across, y down). Row 0 is
// not a bone: it is where the part is attached, and follows the ordinary skin.
const isBody = (s) => s.type === 'body';
const softGrids = softCfg.map((s, k) => {
  if (isBody(s)) return { cfg: s, nodes: [], corners: () => [] };
  const [x0, x1] = s.x;
  const dx = (x1 - x0) / (s.columns - 1);
  const dy = (s.yTop - s.yBottom) / (s.rows - 1);
  const nodes = Array.from({ length: s.rows * s.columns }, () => ({ w: 0, z: 0, bone: -1 }));
  /** Bilinear corners of a point: [[node, weight], ...]. Empty above the top row. */
  const corners = (x, y, z) => {
    const rf = Math.min(s.rows - 1, (s.yTop - y) / dy);
    if (rf <= 0) return [];
    const cf = Math.min(s.columns - 1, Math.max(0, (x - x0) / dx));
    const r0 = Math.min(s.rows - 2, Math.floor(rf));
    const c0 = Math.min(s.columns - 2, Math.floor(cf));
    const fr = rf - r0;
    const fc = cf - c0;
    const fade = s.zFade ? smoothstep(s.zFade[0], s.zFade[1], z) : 1;
    return [
      [r0 * s.columns + c0, (1 - fr) * (1 - fc), fade],
      [r0 * s.columns + c0 + 1, (1 - fr) * fc, fade],
      [(r0 + 1) * s.columns + c0, fr * (1 - fc), fade],
      [(r0 + 1) * s.columns + c0 + 1, fr * fc, fade],
    ].filter((c) => c[1] > 1e-5);
  };
  for (let g = 0; g < nGroups; g++) {
    if (softOf[g] !== k) continue;
    const i = groupRep[g];
    for (const [n, w, fade] of corners(P[3 * i], P[3 * i + 1], P[3 * i + 2])) {
      nodes[n].w += w * fade;
      nodes[n].z += w * fade * P[3 * i + 2];
    }
  }
  nodes.forEach((n, q) => {
    const r = Math.floor(q / s.columns);
    const c = q % s.columns;
    n.used = n.w > 1e-4;
    n.pos = [x0 + c * dx, s.yTop - r * dy, n.used ? n.z / n.w : 0];
    n.r = r;
    n.c = c;
  });
  return { cfg: s, nodes, corners };
});
// Add the grid bones to the skeleton (after the ordinary bones, so their indices are stable).
softGrids.forEach((grid) => {
  for (const n of grid.nodes) {
    if (!n.used || n.r === 0) continue;
    n.bone = bones.length;
    bones.push({ name: `${grid.cfg.name}_${n.r}_${n.c}`, parent: grid.cfg.parent, pos: n.pos, soft: true });
    boneIndex.set(bones[n.bone].name, n.bone);
  }
});

// ---- 5c. control points for the "body" soft parts ------------------------------------
// Spread evenly over the part: each new point is the one farthest, measured along the surface,
// from those already chosen (so every hair strand gets its own points however close the strands lie).
// The app moves each control point as a springy particle.
const rootName = bones.find((b) => !b.parent).name;
const at = (g) => [P[3 * groupRep[g]], P[3 * groupRep[g] + 1], P[3 * groupRep[g] + 2]];
const softBodies = softCfg.map((s, k) => {
  if (!isBody(s)) return null;
  const members = [];
  for (let g = 0; g < nGroups; g++) if (softOf[g] === k) members.push(g);
  if (!members.length) {
    console.warn(`Warning: soft part "${s.name}" has no vertices.`);
    return null;
  }
  const local = new Map(members.map((g, j) => [g, j]));
  const adj = members.map((g) => {
    const a = at(g);
    const out = [];
    for (const h of neighbours[g]) {
      const j = local.get(h);
      if (j === undefined) continue;
      const b = at(h);
      out.push([j, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])]);
    }
    return out;
  });
  const count = Math.min(s.points ?? 40, members.length);
  const far = new Float64Array(members.length).fill(Infinity);
  let current = 0;
  members.forEach((g, j) => {
    if (at(g)[1] > at(members[current])[1]) current = j;
  });
  const points = [];
  let spacing = 0.05;
  for (let c = 0; c < count; c++) {
    const g = members[current];
    const pos = at(g);
    if (Number.isFinite(far[current])) spacing = far[current]; // gap to the points chosen before it
    points.push({ g, pos });
    const d = distancesFrom(adj, [current], Infinity);
    let best = -1;
    for (let j = 0; j < members.length; j++) {
      far[j] = Math.min(far[j], d[j]);
      if (best < 0 || far[j] > far[best]) best = j;
    }
    current = best;
  }
  return { cfg: s, members, points, local, adj, spacing };
});
const NB = bones.length;

// ---- 4. skin weights ----------------------------------------------------------
function distToSegment(px, py, pz, a, b) {
  const abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
  const apx = px - a[0], apy = py - a[1], apz = pz - a[2];
  const t = Math.min(1, Math.max(0, (apx * abx + apy * aby + apz * abz) / (abx * abx + aby * aby + abz * abz)));
  return Math.hypot(apx - abx * t, apy - aby * t, apz - abz * t);
}
// Soft parts only ever hang from their "base" bones (a cape must not follow the arms).
const softBase = softCfg.map((s) => new Set((s.base ?? [s.parent]).map(mustBone)));

let W = new Float32Array(nGroups * NB);
/** How many smoothing passes a point takes part in (fingers are small: fewer passes). */
const passes = new Uint8Array(nGroups).fill(cfg.smoothIterations);
for (let g = 0; g < nGroups; g++) {
  if (rigidOf[g] >= 0) {
    W[g * NB + rigidOf[g]] = 1;
    passes[g] = 0;
    continue;
  }
  const i = groupRep[g];
  const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
  const allowed = softOf[g] >= 0 ? softBase[softOf[g]] : null;
  let best = -1;
  let bestD = Infinity;
  for (let k = 0; k < NB; k++) {
    const b = bones[k];
    if (b.face || b.soft) continue; // layered on afterwards
    if (allowed && !allowed.has(k)) continue;
    if (b.sideSign && x * b.sideSign < -0.02) continue; // left bones never grab the right side
    if (b.leg && z < cfg.capeZ) continue; // whatever hangs behind the legs should not follow them
    if (b.reach && distToSegment(x, y, z, b.pos, b.tail) > b.reach) continue;
    // "keep": the bone only takes points on one side of a plane (an arm that hangs close to the
    // torso must not take the side of the torso just because it is nearer than the spine).
    if (b.keep && b.keep.normal[0] * x + b.keep.normal[1] * y + b.keep.normal[2] * z < b.keep.offset) continue;
    const d = distToSegment(x, y, z, b.pos, b.tail);
    if (d < bestD) {
      bestD = d;
      best = k;
    }
  }
  W[g * NB + best] = 1;
  if (bones[best].smooth !== undefined) passes[g] = bones[best].smooth;
}
// Smooth across the surface so joints bend gradually instead of creasing.
for (let it = 0; it < cfg.smoothIterations; it++) {
  const next = new Float32Array(W);
  for (let g = 0; g < nGroups; g++) {
    if (it >= passes[g]) continue;
    const nb = neighbours[g];
    if (!nb.size) continue;
    for (let k = 0; k < NB; k++) {
      let sum = 0;
      for (const h of nb) sum += W[h * NB + k];
      next[g * NB + k] = 0.5 * W[g * NB + k] + (0.5 * sum) / nb.size;
    }
  }
  W = next;
}
// Layer a bone on top: scale the existing weights down and give the remainder to the bone.
function layer(g, bone, w) {
  if (w <= 0) return;
  for (let k = 0; k < NB; k++) W[g * NB + k] *= 1 - w;
  W[g * NB + bone] += w;
}
let jawGroups = 0;
for (let g = 0; g < nGroups; g++) {
  if (softOf[g] >= 0 || rigidOf[g] >= 0) continue;
  const i = groupRep[g];
  const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
  for (let k = 0; k < NB; k++) {
    const b = bones[k];
    if (b.face === 'jaw') {
      const lower = isLower(i);
      let w;
      if (M.bodyZ) {
        // Whole lower jaw swings from the hinge: nothing at the joint, everything by bodyZ[1].
        // Around the mouth corner the skin above and below the mouth line shares the movement
        // (half at the corner itself), so the corners rise and fall with the jaw. In front of
        // the corner the lips separate: lower lip fully on the jaw, upper lip on the head.
        const [up, down] = M.cornerBand ?? [0.03, 0.03];
        const back = smoothstep(-up, down, M.planeY - y);
        const front = smoothstep(M.cornerZ, M.cornerZ + (M.cornerFade ?? 0.035), z);
        w = smoothstep(M.bodyZ[0], M.bodyZ[1], z) * (back + ((lower ? 1 : 0) - back) * front);
      } else {
        if (!lower) continue;
        w = smoothstep(M.fadeZ[0], M.fadeZ[1], z);
      }
      w *=
        smoothstep(M.fadeY[0], M.fadeY[1], y) *
        (1 - smoothstep(M.maxAbsX * 0.85, M.maxAbsX, Math.abs(x)));
      if (w > 0.5) jawGroups++;
      layer(g, k, w);
    } else if (b.face === 'blob') {
      const d2 = (x - b.pos[0]) ** 2 + (y - b.pos[1]) ** 2 + (z - b.pos[2]) ** 2;
      const w = b.strength * Math.exp(-d2 / (2 * b.radius * b.radius));
      if (w > 0.02) layer(g, k, w);
    }
  }
}
// Soft parts: keep only the base bones from the ordinary skin, then hand over to the grid bones.
const softCounts = softCfg.map(() => 0);
for (let g = 0; g < nGroups; g++) {
  const k = softOf[g];
  if (k < 0) continue;
  softCounts[k]++;
  if (isBody(softCfg[k])) continue; // handled below
  const row = W.subarray(g * NB, (g + 1) * NB);
  let total = 0;
  for (let b = 0; b < NB; b++) {
    if (!softBase[k].has(b)) row[b] = 0;
    total += row[b];
  }
  if (total < 1e-6) {
    row[mustBone(softCfg[k].parent)] = 1;
    total = 1;
  }
  const i = groupRep[g];
  let gridShare = 0;
  const grid = [];
  for (const [n, w, fade] of softGrids[k].corners(P[3 * i], P[3 * i + 1], P[3 * i + 2])) {
    const bone = softGrids[k].nodes[n].bone;
    if (bone < 0) continue; // row 0, or a corner nothing hangs from: stays with the base skin
    grid.push([bone, w * fade]);
    gridShare += w * fade;
  }
  for (let b = 0; b < NB; b++) row[b] *= (1 - gridShare) / total;
  for (const [bone, w] of grid) row[bone] += w;
}

// "Body" soft parts: every vertex is carried by its nearest control points (measured along the
// surface, so neighbouring hair strands or the two front panels of a shirt stay independent).
// Each control point remembers which ordinary bones the surface under it followed; the app uses
// that to work out where the point would be if the part were rigid.
function distancesFrom(adj, sources, limit) {
  const dist = new Float64Array(adj.length).fill(Infinity);
  const heap = [];
  const push = (d, j) => {
    heap.push([d, j]);
    for (let i = heap.length - 1; i > 0; ) {
      const up = (i - 1) >> 1;
      if (heap[up][0] <= heap[i][0]) break;
      [heap[up], heap[i]] = [heap[i], heap[up]];
      i = up;
    }
  };
  const pop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length) {
      heap[0] = last;
      for (let i = 0; ; ) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < heap.length && heap[l][0] < heap[m][0]) m = l;
        if (r < heap.length && heap[r][0] < heap[m][0]) m = r;
        if (m === i) break;
        [heap[m], heap[i]] = [heap[i], heap[m]];
        i = m;
      }
    }
    return top;
  };
  for (const j of sources) {
    dist[j] = 0;
    push(0, j);
  }
  while (heap.length) {
    const [d, j] = pop();
    if (d > dist[j]) continue;
    for (const [h, len] of adj[j]) {
      const nd = d + len;
      if (nd < dist[h] && nd <= limit) {
        dist[h] = nd;
        push(nd, h);
      }
    }
  }
  return dist;
}
/** Nearest saved Blender weight to a position, for one object / vertex group. */
function savedWeight(object, group) {
  const key = Object.keys(weightData).find((name) => name.toLowerCase() === object.toLowerCase());
  const flat = key && weightData[key][group];
  if (!flat) return null;
  return (p) => {
    let best = Infinity, w = 0;
    for (let q = 0; q < flat.length; q += 4) {
      const d = (flat[q] - p[0]) ** 2 + (flat[q + 1] - p[1]) ** 2 + (flat[q + 2] - p[2]) ** 2;
      if (d < best) {
        best = d;
        w = flat[q + 3];
      }
    }
    return w;
  };
}
const bodyNotes = [];
const softnessOfGroup = new Float32Array(nGroups).fill(-1); // for the --parts preview
// Per vertex group: up to four control points (numbered across all soft bodies) and their weights.
const softPoints = new Map();
let firstPoint = 0;
softBodies.forEach((body, k) => {
  if (!body) return;
  const { cfg: s, members, points, local, adj, spacing } = body;
  body.first = firstPoint;
  firstPoint += points.length;

  // How soft each vertex is: 0 = moves rigidly with the body, 1 = fully simulated.
  const softness = new Float32Array(members.length).fill(1);
  const rule = s.softness ?? {};
  let softnessNote = 'everywhere fully soft';
  if (rule.group) {
    const lookup = savedWeight(rule.object ?? s.name, rule.group);
    if (lookup) {
      members.forEach((g, j) => (softness[j] = rule.invert ? 1 - lookup(at(g)) : lookup(at(g))));
      softnessNote = `softness from Blender vertex group "${rule.group}"`;
    } else {
      console.warn(`Warning: vertex group "${rule.group}" not found in ${cfg.weights ?? '(no weights file)'}; "${s.name}" is fully soft.`);
    }
  } else if (rule.pin) {
    const seeds = [];
    members.forEach((g, j) => {
      if (rule.pin.some((box) => inBox(box, ...at(g)))) seeds.push(j);
    });
    const d = distancesFrom(adj, seeds, Infinity);
    for (let j = 0; j < members.length; j++) softness[j] = Number.isFinite(d[j]) ? smoothstep(0, rule.fade ?? 0.3, d[j]) : 1;
    softnessNote = `${seeds.length} points held, softening over ${rule.fade ?? 0.3} units`;
  }
  const floor = rule.min ?? 0;
  for (let j = 0; j < members.length; j++) softness[j] = floor + (1 - floor) * softness[j];
  members.forEach((g, j) => (softnessOfGroup[g] = softness[j]));

  // Reach of a control point: a few times the gap between control points (along the surface).
  const reach = (s.reach ?? 2.2) * spacing;
  const dist = points.map((p) => distancesFrom(adj, [local.get(p.g)], reach));
  const baseSet = s.base ? new Set(s.base.map(mustBone)) : null;
  const skin = points.map(() => new Float64Array(NB));
  const softSum = points.map(() => [0, 0]);
  let strays = 0;
  members.forEach((g, j) => {
    const near = [];
    points.forEach((p, i) => {
      if (dist[i][j] <= reach) near.push([dist[i][j], i]);
    });
    near.sort((a, b) => a[0] - b[0]);
    let weights;
    if (!near.length) {
      // Not connected to any control point (a stray fragment): follow the nearest one in a straight line.
      strays++;
      const a = at(g);
      const gap = (p) => Math.hypot(a[0] - p.pos[0], a[1] - p.pos[1], a[2] - p.pos[2]);
      let best = 0;
      points.forEach((p, i) => {
        if (gap(p) < gap(points[best])) best = i;
      });
      weights = [[best, 1]];
    } else {
      // Weights fall to zero at the distance of the fifth-nearest point, so they change smoothly.
      const edge = near.length > 4 ? near[4][0] : reach;
      weights = near.slice(0, 4).map(([d, i]) => [i, (1 - d / (edge || 1)) ** 2]);
      if (weights.every((w) => w[1] < 1e-6)) weights = [[near[0][1], 1]];
    }
    const total = weights.reduce((sum, w) => sum + w[1], 0);
    const row = W.subarray(g * NB, (g + 1) * NB);
    // The vertex keeps its ordinary skin (limited to the part's base bones). The same skin,
    // averaged, tells each control point where it would be if the part were rigid.
    let baseTotal = 0;
    for (let b = 0; b < NB; b++) {
      if (bones[b].soft || (baseSet && !baseSet.has(b))) row[b] = 0;
      baseTotal += row[b];
    }
    if (baseTotal < 1e-6) {
      row[mustBone(s.parent ?? rootName)] = 1;
      baseTotal = 1;
    }
    for (let b = 0; b < NB; b++) row[b] /= baseTotal;
    for (const [i, w] of weights) {
      const share = w / total;
      for (let b = 0; b < NB; b++) skin[i][b] += share * row[b];
      softSum[i][0] += share * softness[j];
      softSum[i][1] += share;
    }
    softPoints.set(g, weights.map(([i, w]) => [body.first + i, w / total]));
  });

  const fallback = mustBone(s.parent ?? rootName);
  points.forEach((p, i) => {
    const t = top4(skin[i]);
    p.skin = t.joints.length ? t.joints.map((b, q) => [bones[b].name, +t.weights[q].toFixed(4)]) : [[bones[fallback].name, 1]];
    p.soft = softSum[i][1] > 0 ? +(softSum[i][0] / softSum[i][1]).toFixed(3) : 1;
  });
  // Springs between neighbouring control points keep the part's shape.
  const links = new Set();
  points.forEach((p, i) => {
    const near = [];
    points.forEach((q, j) => {
      const d = dist[i][local.get(q.g)];
      if (i !== j && d <= reach) near.push([d, j]);
    });
    near.sort((a, b) => a[0] - b[0]);
    for (const [, j] of near.slice(0, s.links ?? 4)) links.add(i < j ? `${i}_${j}` : `${j}_${i}`);
  });
  body.links = [...links].map((key) => key.split('_').map(Number));
  bodyNotes.push(
    `Soft part "${s.name}" (body): ${members.length} points, ${points.length} control points (${points.filter((p) => p.soft < 0.05).length} held), ` +
      `${body.links.length} springs, ${softnessNote}${strays ? `, ${strays} stray points` : ''}`,
  );
});

// Keep the four strongest bones per vertex (the glTF limit) and normalise.
function top4(row) {
  const order = [];
  for (let k = 0; k < row.length; k++) if (row[k] > 1e-5) order.push(k);
  order.sort((a, b) => row[b] - row[a]);
  order.length = Math.min(order.length, 4);
  const total = order.reduce((s, k) => s + row[k], 0) || 1;
  return { joints: order, weights: order.map((k) => row[k] / total) };
}
const J = new Uint16Array(nVerts * 4);
const WT = new Float32Array(nVerts * 4);
{
  const cache = new Array(nGroups);
  for (let i = 0; i < nVerts; i++) {
    const g = groupOf[i];
    const r = (cache[g] ??= top4(W.subarray(g * NB, (g + 1) * NB)));
    for (let k = 0; k < r.joints.length; k++) {
      J[4 * i + k] = r.joints[k];
      WT[4 * i + k] = r.weights[k];
    }
  }
}

// ---- 2. mouth interior, sized to the cross-section of the snout at the cut -----
const HEAD = boneIndex.get('head');
const JAW = boneIndex.get('jaw');
function part() {
  return { pos: [], nrm: [], idx: [], joints: [], weights: [] };
}
function addVertex(p, pos, nrm, bone) {
  p.pos.push(...pos);
  p.nrm.push(...nrm);
  p.joints.push(bone, 0, 0, 0);
  p.weights.push(1, 0, 0, 0);
  return p.pos.length / 3 - 1;
}
/** Ellipsoid; `keep(centroid)` can drop faces (used to open the front of the cavity). */
function ellipsoid(p, c, r, boneFor, inward = false, keep = () => true, rings = 12, segs = 20) {
  const grid = [];
  for (let a = 0; a <= rings; a++) {
    const theta = (a / rings) * Math.PI;
    const row = [];
    for (let s = 0; s < segs; s++) {
      const phi = (s / segs) * Math.PI * 2;
      const d = [Math.sin(theta) * Math.cos(phi), Math.cos(theta), Math.sin(theta) * Math.sin(phi)];
      const pos = [c[0] + r[0] * d[0], c[1] + r[1] * d[1], c[2] + r[2] * d[2]];
      const n = [d[0] / r[0], d[1] / r[1], d[2] / r[2]];
      const len = Math.hypot(...n) * (inward ? -1 : 1);
      row.push(addVertex(p, pos, [n[0] / len, n[1] / len, n[2] / len], boneFor(pos)));
    }
    grid.push(row);
  }
  for (let a = 0; a < rings; a++) {
    for (let s = 0; s < segs; s++) {
      const q = [grid[a][s], grid[a][(s + 1) % segs], grid[a + 1][(s + 1) % segs], grid[a + 1][s]];
      const cen = [0, 1, 2].map((k) => q.reduce((sum, v) => sum + p.pos[3 * v + k], 0) / 4);
      if (!keep(cen)) continue;
      p.idx.push(q[0], q[1], q[2], q[0], q[2], q[3]);
    }
  }
}
function cone(p, base, tip, radius, bone, segs = 6) {
  const apex = addVertex(p, tip, [0, Math.sign(tip[1] - base[1]), 0], bone);
  const ring = [];
  for (let s = 0; s < segs; s++) {
    const a = (s / segs) * Math.PI * 2;
    ring.push(addVertex(p, [base[0] + radius * Math.cos(a), base[1], base[2] + radius * Math.sin(a)], [Math.cos(a), 0, Math.sin(a)], bone));
  }
  for (let s = 0; s < segs; s++) p.idx.push(apex, ring[s], ring[(s + 1) % segs]);
}

const accessor = (array, type) => doc.createAccessor().setArray(array).setType(type).setBuffer(buffer);
let interiorNote = 'none';
// Set "interior": false in the rig file to leave out the generated cavity, tongue and teeth.
if (M.interior !== false) {
  let zFront = -Infinity;
  for (let i = 0; i < nVerts; i++) {
    if (Math.abs(P[3 * i + 1] - M.planeY) < 0.008 && Math.abs(P[3 * i]) < M.maxAbsX && P[3 * i + 2] > M.minZ) {
      zFront = Math.max(zFront, P[3 * i + 2]);
    }
  }
  const cz = M.minZ + 0.55 * (zFront - M.minZ);
  const rz = 0.4 * (zFront - M.minZ);
  let halfWidth = 0;
  for (let i = 0; i < nVerts; i++) {
    if (Math.abs(P[3 * i + 1] - M.planeY) < 0.008 && Math.abs(P[3 * i + 2] - cz) < 0.012 && Math.abs(P[3 * i]) < M.maxAbsX) {
      halfWidth = Math.max(halfWidth, Math.abs(P[3 * i]));
    }
  }
  const rx = 0.7 * halfWidth;
  const ry = M.cavityHeight;
  const cavity = part();
  // Upper half follows the head, lower half follows the jaw, so the cavity stretches open with the mouth.
  // The front cap is removed so you look into it rather than at it.
  ellipsoid(cavity, [0, M.planeY, cz], [rx, ry, rz], (pos) => (pos[1] >= M.planeY ? HEAD : JAW), true, (cen) => cen[2] < cz + 0.6 * rz);
  const tongue = part();
  ellipsoid(tongue, [0, M.planeY - 0.6 * ry, cz + 0.05 * rz], [0.55 * rx, 0.3 * ry, 0.75 * rz], () => JAW, false, () => true, 8, 14);
  const teeth = part();
  for (let k = 0; k < M.teeth; k++) {
    const a = ((25 + (130 * k) / (M.teeth - 1)) * Math.PI) / 180;
    const x = 0.8 * rx * Math.cos(a);
    const z = cz + 0.8 * rz * Math.sin(a);
    const size = 0.75 + 0.5 * Math.sin(a); // longer toward the front
    cone(teeth, [x, M.planeY + 0.003, z], [x, M.planeY - 0.012 * size, z], 0.0045, HEAD);
    if (k < M.teeth - 1) {
      const a2 = a + ((65 / (M.teeth - 1)) * Math.PI) / 180; // lower teeth sit between the upper ones
      const x2 = 0.74 * rx * Math.cos(a2);
      const z2 = cz + 0.74 * rz * Math.sin(a2);
      cone(teeth, [x2, M.planeY - 0.006, z2], [x2, M.planeY + 0.008 * size, z2], 0.004, JAW);
    }
  }
  const addPart = (p, name, color, roughness) => {
    const material = doc
      .createMaterial(name)
      .setBaseColorFactor([...color, 1])
      .setRoughnessFactor(roughness)
      .setMetallicFactor(0)
      .setDoubleSided(true);
    prims[0].mesh.addPrimitive(
      doc
        .createPrimitive()
        .setMaterial(material)
        .setAttribute('POSITION', accessor(new Float32Array(p.pos), 'VEC3'))
        .setAttribute('NORMAL', accessor(new Float32Array(p.nrm), 'VEC3'))
        .setAttribute('JOINTS_0', accessor(new Uint16Array(p.joints), 'VEC4'))
        .setAttribute('WEIGHTS_0', accessor(new Float32Array(p.weights), 'VEC4'))
        .setIndices(accessor(new Uint32Array(p.idx), 'SCALAR')),
    );
  };
  addPart(cavity, 'mouthCavity', [0.012, 0.002, 0.003], 1);
  addPart(tongue, 'tongue', [0.42, 0.1, 0.11], 0.55);
  addPart(teeth, 'teeth', [0.82, 0.76, 0.62], 0.45);
  interiorNote = `centre z ${cz.toFixed(3)}, half-width ${rx.toFixed(3)}, depth ${rz.toFixed(3)}`;
}

// ---- 7. facial expression shapes ------------------------------------------------------
// Each shape is a list of small pushes on the face, described in the rig file:
//   { "at": [x,y,z], "r": radius, "move": [dx,dy,dz] }   push the surface around a point
//   { "at": ..., "r": ..., "squash": amount, "axis": "y" } flatten toward the point (1 = closed at the centre,
//                                                          more = closed over a wider zone; negative widens)
//   { "at": ..., "r": ..., "curl": radians }             roll the surface around a left-right line through the point
//                                                          (negative rolls what is below the line forward and up,
//                                                          positive rolls what is above it forward and down)
// "lip": "upper" | "lower" limits a push to one lip, "mirror": true repeats it on the other side.
// A shape with "sides": true is built twice, as <name>Left and <name>Right (the character's own sides).
// "clothes": true also moves the simulated clothing over the area (body corrections such as the
// shrug bulge, where the collar has to rise with the muscle under it).
// "guardTeeth": true on a shape keeps the inside of the mouth still while the lips move. The pushes are
// round, so a push on a lip also reaches the floor and roof of the mouth behind it; those would rise or
// drop over the teeth, which are rigid and do not move with a shape. Seen from above, each row of teeth
// is an arch: skin behind the front of that arch (the floor for the lower row, the roof for the upper)
// is held in place, fading to free just in front of the teeth. Distances can be changed with
// "guard" under "mouth": front = [held, free] distance in front of the teeth, below / above = height
// band relative to the lowest point of the teeth in which the hold fades out.
const guard = { front: [0.001, 0.007], below: [-0.008, -0.002], above: [0.01, 0.018], ...(M.guard ?? {}) };
const GUARD_BINS = 36;
const teethRows = (() => {
  const rows = { lower: [], upper: [] };
  for (let i = 0; i < nVerts; i++) {
    const k = primOf[i];
    if (partBone[k] < 0 || !/teeth|tooth/i.test(prims[k].name)) continue;
    const bone = bones[partBone[k]];
    (bone.face === 'jaw' || bone.name === 'jaw' ? rows.lower : rows.upper).push(i);
  }
  const all = [...rows.lower, ...rows.upper];
  if (!all.length) return null;
  // Centre of the arches: a little behind the rearmost tooth, on the middle line.
  const zc = all.reduce((m, i) => Math.min(m, P[3 * i + 2]), Infinity) - 0.03;
  const arch = (list) => {
    if (!list.length) return null;
    const R = new Float32Array(GUARD_BINS), Y = new Float32Array(GUARD_BINS).fill(Infinity);
    for (const i of list) {
      const x = P[3 * i], z = P[3 * i + 2] - zc;
      const b = Math.floor(((Math.atan2(x, z) + Math.PI) / (2 * Math.PI)) * GUARD_BINS) % GUARD_BINS;
      R[b] = Math.max(R[b], Math.hypot(x, z));
      Y[b] = Math.min(Y[b], P[3 * i + 1]);
    }
    // Directions with no teeth: the hold shrinks away over the two directions past the end of the row.
    const R2 = Float32Array.from(R), Y2 = Float32Array.from(Y);
    for (let b = 0; b < GUARD_BINS; b++) {
      if (R[b] > 0) continue;
      for (let step = 1; step <= 2 && R2[b] === 0; step++) {
        for (const o of [b + step, b - step + GUARD_BINS]) {
          if (R[o % GUARD_BINS] > 0) {
            R2[b] = R[o % GUARD_BINS] * (1 - step / 3);
            Y2[b] = Y[o % GUARD_BINS];
          }
        }
      }
    }
    return { R: R2, Y: Y2 };
  };
  return { zc, lower: arch(rows.lower), upper: arch(rows.upper) };
})();
/** 0 = free to move with a shape, 1 = held still (inside the mouth, behind the teeth). */
function teethHold(i) {
  const row = teethRows && (isLower(i) ? teethRows.lower : teethRows.upper);
  if (!row) return 0;
  const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2] - teethRows.zc;
  const f = ((Math.atan2(x, z) + Math.PI) / (2 * Math.PI)) * GUARD_BINS - 0.5;
  const b0 = (Math.floor(f) + GUARD_BINS) % GUARD_BINS, b1 = (b0 + 1) % GUARD_BINS, t = f - Math.floor(f);
  const R = row.R[b0] * (1 - t) + row.R[b1] * t;
  if (R <= 0) return 0;
  const y0 = Math.min(row.Y[b0], row.Y[b1]);
  if (!Number.isFinite(y0)) return 0;
  const r = Math.hypot(x, z);
  const behind = 1 - smoothstep(R + guard.front[0], R + guard.front[1], r);
  const level = isLower(i) ? smoothstep(y0 + guard.below[0], y0 + guard.below[1], y) : 1 - smoothstep(y0 + guard.above[0], y0 + guard.above[1], y);
  return behind * level;
}

const morphs = [];
function buildMorph(name, ops, flip, clothes = false, guardTeeth = false, split = 0) {
  const data = new Float32Array(nVerts * 3);
  // How far each point is flattened by this shape's squashes (0 = not at all, 1 = fully closed).
  const collapse = new Float32Array(nVerts);
  let touched = 0;
  for (let i = 0; i < nVerts; i++) {
    if (!isBodyVertex(i) || (!clothes && softOf[groupOf[i]] >= 0)) continue;
    const p = [P[3 * i], P[3 * i + 1], P[3 * i + 2]];
    let moved = false;
    for (const op of ops) {
      if (op.lip === 'upper' && isLower(i)) continue;
      if (op.lip === 'lower' && !isLower(i)) continue;
      for (const sx of op.mirror ? [1, -1] : [flip]) {
        const c = [op.at[0] * sx, op.at[1], op.at[2]];
        const d2 = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2;
        if (d2 > 9 * op.r * op.r) continue;
        const w = Math.exp(-d2 / (2 * op.r * op.r));
        if (op.move) {
          data[3 * i] += w * op.move[0] * sx;
          data[3 * i + 1] += w * op.move[1];
          data[3 * i + 2] += w * op.move[2];
        }
        if (op.curl) {
          // Roll the surface around a left-right line through the centre (a lip curling outward).
          const angle = op.curl * w, dy = p[1] - c[1], dz = p[2] - c[2];
          data[3 * i + 1] += dy * Math.cos(angle) - dz * Math.sin(angle) - dy;
          data[3 * i + 2] += dy * Math.sin(angle) + dz * Math.cos(angle) - dz;
        }
        if (op.squash) {
          const axis = { x: 0, y: 1, z: 2 }[op.axis ?? 'y'];
          // Above 1 the middle closes completely and the closed zone widens.
          data[3 * i + axis] -= Math.min(1, op.squash * w) * (p[axis] - c[axis]);
          if (op.squash > 0) collapse[i] = Math.max(collapse[i], Math.min(1, op.squash * w));
        }
        moved = true;
      }
    }
    if (moved && guardTeeth) {
      const free = 1 - teethHold(i);
      for (let a = 0; a < 3; a++) data[3 * i + a] *= free;
    }
    if (moved && split > 0) {
      // Left and Right versions of a shape overlap on the middle of the face and would add up to
      // double there. With "split" each side fades out across the middle line (half at the centre).
      const share = smoothstep(-split, split, p[0] * flip);
      for (let a = 0; a < 3; a++) data[3 * i + a] *= share;
    }
    if (moved) touched++;
  }
  morphs.push({ name, data, touched, collapse });
}
// A "combo" shape corrects two shapes that are on together. Shapes simply add up, so skin that one
// shape lifts toward an eye is pushed straight through the lid line when a blink has flattened that
// same skin onto it, and the surface folds into itself. { "combo": ["eyeBlink", "noseSneer"] } builds
// combo_eyeBlinkLeft_noseSneerLeft (and Right): it takes the second shape's movement back out
// wherever the first shape has flattened the skin (fully where the eye is fully closed). The app
// turns it on by the product of the two shapes (src/rigAvatar.ts), so each shape alone is unchanged.
function buildCombo(first, second, amount = 1) {
  const a = morphs.find((m) => m.name === first), b = morphs.find((m) => m.name === second);
  if (!a || !b) return console.warn(`combo: no shape called ${a ? second : first}`);
  const data = new Float32Array(nVerts * 3);
  let touched = 0;
  for (let i = 0; i < nVerts; i++) {
    if (!a.collapse[i]) continue;
    let moved = false;
    for (let k = 0; k < 3; k++) {
      data[3 * i + k] = -amount * a.collapse[i] * b.data[3 * i + k];
      moved ||= data[3 * i + k] !== 0;
    }
    if (moved) touched++;
  }
  morphs.push({ name: `combo_${first}_${second}`, data, touched, collapse: new Float32Array(nVerts) });
}
for (const m of (cfg.morphs ?? []).filter((m) => !m.combo)) {
  if (m.sides) {
    buildMorph(`${m.name}Left`, m.ops, 1, m.clothes, m.guardTeeth, m.split);
    buildMorph(`${m.name}Right`, m.ops, -1, m.clothes, m.guardTeeth, m.split);
  } else buildMorph(m.name, m.ops, 1, m.clothes, m.guardTeeth);
}
for (const m of (cfg.morphs ?? []).filter((m) => m.combo)) {
  for (const side of ['Left', 'Right']) buildCombo(`${m.combo[0]}${side}`, `${m.combo[1]}${side}`, m.amount);
}

// ---- write the geometry back, one primitive at a time ------------------------------
prims.forEach((p, k) => {
  const local = new Map();
  const pos = [], nrm = [], uv = [], joints = [], weights = [], softIndex = [], softWeight = [];
  for (let i = 0; i < nVerts; i++) {
    if (primOf[i] !== k) continue;
    local.set(i, pos.length / 3);
    pos.push(P[3 * i], P[3 * i + 1], P[3 * i + 2]);
    nrm.push(N[3 * i], N[3 * i + 1], N[3 * i + 2]);
    uv.push(UV[2 * i], UV[2 * i + 1]);
    for (let q = 0; q < 4; q++) {
      joints.push(J[4 * i + q]);
      weights.push(WT[4 * i + q]);
    }
    const sp = softPoints.get(groupOf[i]) ?? [];
    for (let q = 0; q < 4; q++) {
      softIndex.push(sp[q]?.[0] ?? 0);
      softWeight.push(sp[q]?.[1] ?? 0);
    }
  }
  const ind = [];
  for (let t = 0; t < triPrim.length; t++) {
    if (triPrim[t] === k) ind.push(local.get(idx[3 * t]), local.get(idx[3 * t + 1]), local.get(idx[3 * t + 2]));
  }
  for (const name of ['POSITION', 'NORMAL', 'TEXCOORD_0']) p.prim.getAttribute(name)?.dispose();
  p.prim.getIndices()?.dispose();
  p.prim
    .setAttribute('POSITION', accessor(new Float32Array(pos), 'VEC3'))
    .setAttribute('NORMAL', accessor(new Float32Array(nrm), 'VEC3'))
    .setAttribute('JOINTS_0', accessor(new Uint16Array(joints), 'VEC4'))
    .setAttribute('WEIGHTS_0', accessor(new Float32Array(weights), 'VEC4'))
    .setIndices(accessor(new Uint32Array(ind), 'SCALAR'));
  if (p.hasUV) p.prim.setAttribute('TEXCOORD_0', accessor(new Float32Array(uv), 'VEC2'));
  // Which soft-body control points move each vertex, and by how much (all zero outside soft parts).
  // Written on every object so they can share one material.
  if (firstPoint > 0) {
    p.prim
      .setAttribute('_SOFTINDEX', accessor(new Float32Array(softIndex), 'VEC4'))
      .setAttribute('_SOFTWEIGHT', accessor(new Float32Array(softWeight), 'VEC4'));
  }
  // Expression shapes go on the body only (not on teeth, hair or other separate parts).
  if (morphs.length && partSoft[k] < 0 && partBone[k] < 0) {
    for (const morph of morphs) {
      const delta = new Float32Array(pos.length);
      for (const [i, j] of local) for (let a = 0; a < 3; a++) delta[3 * j + a] = morph.data[3 * i + a];
      p.prim.addTarget(doc.createPrimitiveTarget(morph.name).setAttribute('POSITION', accessor(delta, 'VEC3')));
    }
    p.mesh.setWeights(morphs.map(() => 0)).setExtras({ ...p.mesh.getExtras(), targetNames: morphs.map((m) => m.name) });
  }
});

// ---- skeleton nodes ---------------------------------------------------------------
const skin = doc.createSkin(`${id}-rig`);
const nodes = [];
const ibm = new Float32Array(16 * NB);
bones.forEach((b, k) => {
  const parent = b.parent ? bones[mustBone(b.parent)] : null;
  const node = doc.createNode(b.name).setTranslation(parent ? b.pos.map((v, a) => v - parent.pos[a]) : b.pos);
  // A mouth modelled open at rest: tell the app how far to rotate the jaw back to close it.
  if (b.face === 'jaw' && M.restOpen) node.setExtras({ restOpen: M.restOpen });
  if (b.detached) node.setExtras({ ...node.getExtras(), detached: true });
  nodes.push(node);
  // Bones are axis-aligned at rest, so the inverse bind matrix is just a translation.
  ibm.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -b.pos[0], -b.pos[1], -b.pos[2], 1], 16 * k);
});
bones.forEach((b, k) => {
  if (b.parent) nodes[mustBone(b.parent)].addChild(nodes[k]);
  skin.addJoint(nodes[k]);
});
const rootIndex = bones.findIndex((b) => !b.parent);
const rootBone = nodes[rootIndex];
skin.setSkeleton(rootBone).setInverseBindMatrices(accessor(ibm, 'MAT4'));

// Everything the app needs to simulate the soft parts travels in the root bone's extras.
const mirrored = (list) =>
  (list ?? []).flatMap((c) => {
    if (!c.mirror) return [c];
    const rename = (n) => n && n.replace(/^left/, 'right');
    const { mirror: _m, ...rest } = c;
    return [rest, { ...rest, from: rename(c.from), to: rename(c.to), offset: c.offset && [-c.offset[0], c.offset[1], c.offset[2]] }];
  });
rootBone.setExtras({
  restPositions: Object.fromEntries(bones.filter((b) => !b.soft).map((b) => [b.name, b.pos])),
  restTails: Object.fromEntries(bones.filter((b) => b.tail).map((b) => [b.name, b.tail])),
  colliders: mirrored(cfg.colliders),
  // Optional camera framing for the app: { "target": [x, y, z], "distance": n }.
  ...(cfg.view ? { view: cfg.view } : {}),
  // Optional directions the arms hang in while untracked (left arm; mirrored for the right):
  // { "upper": [x, y, z], "lower": [x, y, z] }. For a character whose skirt is in the way.
  ...(cfg.relaxedArms ? { relaxedArms: cfg.relaxedArms } : {}),
  // Optional: the operator's own head from the webcam in place of a helmet (src/webcamHead.ts):
  // { "hide": "pattern for the helmet's object names", "scale": 1, "offset": [x, y, z] }.
  ...(cfg.webcamHead ? { webcamHead: cfg.webcamHead } : {}),  bodies: softBodies.filter(Boolean).map((body) => ({
    name: body.cfg.name,
    sim: body.cfg.sim ?? {},
    first: body.first,
    nodes: body.points.map((p) => ({ rest: p.pos.map((v) => +v.toFixed(5)), soft: p.soft, skin: p.skin })),
    links: body.links,
  })),
  soft: softGrids.filter((grid) => !isBody(grid.cfg)).map((grid) => ({
    name: grid.cfg.name,
    parent: grid.cfg.parent,
    parentPos: bones[mustBone(grid.cfg.parent)].pos,
    rows: grid.cfg.rows,
    columns: grid.cfg.columns,
    sim: grid.cfg.sim ?? {},
    nodes: grid.nodes.filter((n) => n.used).map((n) => ({ r: n.r, c: n.c, rest: n.pos.map((v) => +v.toFixed(5)), bone: n.bone >= 0 ? bones[n.bone].name : null })),
  })),
});

// Optional material overrides: { "metallic": 0, "roughness": 1 } scale what the textures say.
// (Generated models often mark cloth as metal, which renders as dark chrome without a studio around it.)
if (cfg.material) {
  for (const material of root.listMaterials()) {
    if (cfg.material.metallic !== undefined) material.setMetallicFactor(cfg.material.metallic);
    if (cfg.material.roughness !== undefined) material.setRoughnessFactor(cfg.material.roughness);
  }
}

const seenNodes = new Set();
for (const p of prims) {
  if (seenNodes.has(p.node)) continue;
  seenNodes.add(p.node);
  p.node.setSkin(skin).setName(prims.length === 1 ? `${id}-mesh` : p.node.getName());
}
root.listScenes()[0].addChild(rootBone);

await io.write(`${dir}/${cfg.output}`, doc);
console.log(`Objects: ${prims.map((p, k) => `${p.node.getName() || 'mesh'}${partSoft[k] >= 0 ? ` (soft: ${softCfg[partSoft[k]].name})` : partBone[k] >= 0 ? ` (fixed to ${bones[partBone[k]].name})` : ''}`).join(', ')}`);
console.log(`Mouth cut: ${cutTriangles} triangles split, ${edgeVerts.size} vertices added, ${jawGroups} points on the jaw`);
console.log(`Mouth interior: ${interiorNote}`);
softGrids.forEach((grid, k) => {
  if (!isBody(grid.cfg)) console.log(`Soft part "${grid.cfg.name}" (sheet): ${softCounts[k]} points, ${grid.nodes.filter((n) => n.bone >= 0).length} simulated bones`);
});
bodyNotes.forEach((note) => console.log(note));
if (morphs.length) console.log(`Expression shapes: ${morphs.length} (${morphs.filter((m) => !m.touched).map((m) => m.name + ' moves nothing!').join(', ') || 'all move some of the face'})`);
if (Object.keys(jointEdits).length) console.log(`Joint positions from ${jointsFile}: ${Object.keys(jointEdits).join(', ')}`);
console.log(`Skeleton: ${NB} bones (${bones.filter((b) => !b.soft).map((b) => b.name).join(', ')})`);
console.log(`Saved ${dir}/${cfg.output}: ${nVerts} vertices, ${idx.length / 3} triangles`);

// ---- optional preview: the same mesh with flat colours per part -------------------------------
// Body grey, each soft part its own colour (dark where it is held, bright where it is fully soft),
// rigid parts white. Open it in the viewer: /viewer.html?model=<id>.parts.glb (found in the character's folder)
if (debugParts) {
  const palette = [[1, 0.25, 0.2], [0.2, 0.6, 1], [0.3, 0.9, 0.4], [1, 0.8, 0.2]];
  prims.forEach((p, k) => {
    const colours = [];
    for (let i = 0; i < nVerts; i++) {
      if (primOf[i] !== k) continue;
      const g = groupOf[i];
      let c = [0.45, 0.45, 0.45];
      if (rigidOf[g] >= 0) c = [1, 1, 1];
      else if (softOf[g] >= 0) {
        const shade = 0.25 + 0.75 * (softnessOfGroup[g] >= 0 ? softnessOfGroup[g] : 1);
        c = palette[softOf[g] % palette.length].map((v) => v * shade);
      }
      colours.push(...c);
    }
    p.prim.setAttribute('COLOR_0', accessor(new Float32Array(colours), 'VEC3'));
    p.prim.setMaterial(null);
    for (const target of p.prim.listTargets()) target.dispose();
    p.mesh.setWeights([]);
  });
  await io.write(`${dir}/${id}.parts.glb`, doc);
  console.log(`Saved ${dir}/${id}.parts.glb (parts preview)`);
}
