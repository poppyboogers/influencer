// Gets a Blender GLB export ready for rigging:
//   - drops objects that should not ship (default: any object with "highpoly" in its name) and empties,
//   - bakes every object's position / rotation / scale into its vertices (mirrored objects keep facing outward),
//   - shrinks the textures (default: 2048 px on the long side, JPEG).
//   node scripts/prepare-glb.mjs <in.glb> <out.glb> [--drop <regex>] [--texture <px>]
import { NodeIO } from '@gltf-transform/core';
import fs from 'node:fs';
import sharp from 'sharp';

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args.splice(i, 2)[1] : fallback;
};
const drop = new RegExp(flag('--drop', 'highpoly'), 'i');
const maxTexture = Number(flag('--texture', 2048));
// --textures <folder>: a picture there named like one of the model's textures (texture_0.png) is used
// in its place when it is newer than the model file, so a texture can be edited without re-exporting.
const textureDir = flag('--textures', null);
// --refine {"object":"lowpoly","levels":2,"x":[-0.1,0.1],"y":[2.05,2.17],"z":[0.02,1]}
// adds points inside a box (used for the face, so expression shapes have something to move).
const refine = JSON.parse(flag('--refine', 'null'));
const [input, output] = args;
if (!input || !output) {
  console.error('usage: node scripts/prepare-glb.mjs <in.glb> <out.glb> [--drop <regex>] [--texture <px>]');
  process.exit(1);
}

const io = new NodeIO();
const doc = await io.read(input);
const root = doc.getRoot();

// ---- drop unwanted objects, then whatever only they used --------------------------
const dropped = [];
for (const node of root.listNodes()) {
  if (!node.getMesh() || drop.test(node.getName())) {
    dropped.push(node.getName());
    node.dispose();
  }
}
const unused = (prop) => prop.listParents().every((p) => p === root);
for (const mesh of root.listMeshes()) {
  if (!unused(mesh)) continue;
  for (const prim of mesh.listPrimitives()) prim.dispose();
  mesh.dispose();
}
for (const material of root.listMaterials()) if (unused(material)) material.dispose();
for (const texture of root.listTextures()) if (unused(texture)) texture.dispose();
for (const accessor of root.listAccessors()) if (unused(accessor)) accessor.dispose();

// ---- bake transforms -----------------------------------------------------------------
const kept = [];
for (const node of root.listNodes()) {
  const mesh = node.getMesh();
  const m = node.getWorldMatrix();
  const det =
    m[0] * (m[5] * m[10] - m[6] * m[9]) - m[4] * (m[1] * m[10] - m[2] * m[9]) + m[8] * (m[1] * m[6] - m[2] * m[5]);
  let verts = 0, tris = 0;
  for (const prim of mesh.listPrimitives()) {
    // Each object gets its own copies, so shared mesh data is never transformed twice.
    const pos = prim.getAttribute('POSITION').clone();
    const a = pos.getArray();
    for (let i = 0; i < a.length; i += 3) {
      const x = a[i], y = a[i + 1], z = a[i + 2];
      a[i] = m[0] * x + m[4] * y + m[8] * z + m[12];
      a[i + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
      a[i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    }
    prim.setAttribute('POSITION', pos.setArray(a));
    verts += a.length / 3;

    let ind = prim.getIndices()?.clone();
    if (!ind) ind = doc.createAccessor().setType('SCALAR').setArray(Uint32Array.from({ length: a.length / 3 }, (_, i) => i)).setBuffer(root.listBuffers()[0]);
    const tri = ind.getArray();
    // A mirrored object (negative scale) would end up inside out: reverse its triangles.
    if (det < 0) for (let t = 0; t < tri.length; t += 3) [tri[t + 1], tri[t + 2]] = [tri[t + 2], tri[t + 1]];
    prim.setIndices(ind.setArray(tri));
    tris += tri.length / 3;

    // An object that was only moved keeps the normals it was exported with.
    const linear = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
    if (linear.every((v, i) => Math.abs(v - (i % 4 === 0 ? 1 : 0)) < 1e-6) && prim.getAttribute('NORMAL')) continue;
    // Otherwise they are rebuilt from the baked triangles (area-weighted), which also handles uneven scale.
    const n = new Float32Array(a.length);
    for (let t = 0; t < tri.length; t += 3) {
      const [i0, i1, i2] = [3 * tri[t], 3 * tri[t + 1], 3 * tri[t + 2]];
      const e1 = [a[i1] - a[i0], a[i1 + 1] - a[i0 + 1], a[i1 + 2] - a[i0 + 2]];
      const e2 = [a[i2] - a[i0], a[i2 + 1] - a[i0 + 1], a[i2 + 2] - a[i0 + 2]];
      const c = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      for (const i of [i0, i1, i2]) for (let k = 0; k < 3; k++) n[i + k] += c[k];
    }
    // Vertices split by a UV seam share a position: let them share a normal too, so no seam shows.
    const sums = new Map();
    const keyOf = (i) => `${Math.round(a[i] * 1e5)}_${Math.round(a[i + 1] * 1e5)}_${Math.round(a[i + 2] * 1e5)}`;
    for (let i = 0; i < a.length; i += 3) {
      const s = sums.get(keyOf(i)) ?? [0, 0, 0];
      for (let k = 0; k < 3; k++) s[k] += n[i + k];
      sums.set(keyOf(i), s);
    }
    for (let i = 0; i < a.length; i += 3) {
      const s = sums.get(keyOf(i));
      const len = Math.hypot(...s) || 1;
      for (let k = 0; k < 3; k++) n[i + k] = s[k] / len;
    }
    const old = prim.getAttribute('NORMAL');
    prim.setAttribute('NORMAL', (old ? old.clone() : doc.createAccessor().setType('VEC3').setBuffer(root.listBuffers()[0])).setArray(n));
  }
  node.setTranslation([0, 0, 0]).setRotation([0, 0, 0, 1]).setScale([1, 1, 1]);
  kept.push(`${node.getName()} (${verts} vertices, ${tris} triangles${det < 0 ? ', mirrored' : ''})`);
}
for (const accessor of root.listAccessors()) if (unused(accessor)) accessor.dispose();

// ---- add detail where the expression shapes need it -----------------------------------
// Each level splits every triangle inside the box into four (new points sit on the old edges, so
// the surface keeps its exact shape and texture); triangles on the border are split to match.
if (refine) {
  const match = new RegExp(refine.object ?? '.', 'i');
  const inside = (a, i) =>
    ['x', 'y', 'z'].every((axis, k) => !refine[axis] || (a[3 * i + k] >= refine[axis][0] && a[3 * i + k] <= refine[axis][1]));
  for (const node of root.listNodes()) {
    if (!match.test(node.getName())) continue;
    for (const prim of node.getMesh().listPrimitives()) {
      const before = prim.getIndices().getCount() / 3;
      for (let level = 0; level < (refine.levels ?? 1); level++) {
        const names = prim.listSemantics();
        const data = names.map((name) => {
          const acc = prim.getAttribute(name);
          return { name, acc, size: acc.getElementSize(), values: Array.from(acc.getArray()) };
        });
        const P = data.find((d) => d.name === 'POSITION').values;
        const tri = prim.getIndices().getArray();
        const posKey = (i) => `${P[3 * i]},${P[3 * i + 1]},${P[3 * i + 2]}`;
        const edgeKey = (i, j) => (posKey(i) < posKey(j) ? `${posKey(i)}|${posKey(j)}` : `${posKey(j)}|${posKey(i)}`);
        // Edges are marked by position, so both sides of a texture seam split at the same place.
        const split = new Set();
        for (let t = 0; t < tri.length; t += 3) {
          const v = [tri[t], tri[t + 1], tri[t + 2]];
          if (!v.every((i) => inside(P, i))) continue;
          for (let k = 0; k < 3; k++) split.add(edgeKey(v[k], v[(k + 1) % 3]));
        }
        const made = new Map();
        const midpoint = (i, j) => {
          const key = i < j ? `${i}_${j}` : `${j}_${i}`;
          if (made.has(key)) return made.get(key);
          const index = P.length / 3;
          for (const d of data) {
            const mid = [];
            for (let k = 0; k < d.size; k++) mid.push((d.values[d.size * i + k] + d.values[d.size * j + k]) / 2);
            if (d.name === 'NORMAL') {
              const len = Math.hypot(mid[0], mid[1], mid[2]) || 1;
              for (let k = 0; k < 3; k++) mid[k] /= len;
            }
            d.values.push(...mid);
          }
          made.set(key, index);
          return index;
        };
        const out = [];
        for (let t = 0; t < tri.length; t += 3) {
          const v = [tri[t], tri[t + 1], tri[t + 2]];
          const mid = v.map((i, k) => (split.has(edgeKey(i, v[(k + 1) % 3])) ? midpoint(i, v[(k + 1) % 3]) : -1)); // on edge k -> k+1
          const count = mid.filter((x) => x >= 0).length;
          if (count === 0) out.push(...v);
          else if (count === 3) out.push(v[0], mid[0], mid[2], mid[0], v[1], mid[1], mid[2], mid[1], v[2], mid[0], mid[1], mid[2]);
          else if (count === 1) {
            const k = mid.findIndex((x) => x >= 0);
            const [a, b, c] = [v[k], v[(k + 1) % 3], v[(k + 2) % 3]];
            out.push(a, mid[k], c, mid[k], b, c);
          } else {
            const k = mid.findIndex((x) => x < 0); // the one edge left whole: a -> b
            const [a, b, c] = [v[k], v[(k + 1) % 3], v[(k + 2) % 3]];
            const mbc = mid[(k + 1) % 3], mca = mid[(k + 2) % 3];
            out.push(a, b, mbc, a, mbc, mca, mca, mbc, c);
          }
        }
        for (const d of data) {
          const Type = d.acc.getArray().constructor;
          prim.setAttribute(d.name, d.acc.clone().setArray(Type.from(d.values)));
        }
        prim.setIndices(prim.getIndices().clone().setArray(Uint32Array.from(out)));
      }
      console.log(`refined ${node.getName()}: ${before} -> ${prim.getIndices().getCount() / 3} triangles (${refine.levels ?? 1} level(s) inside the box)`);
    }
  }
  for (const accessor of root.listAccessors()) if (unused(accessor)) accessor.dispose();
}

// ---- textures -------------------------------------------------------------------------
for (const texture of root.listTextures()) {
  let before = texture.getImage();
  const edited = textureDir && ['png', 'jpg', 'jpeg'].map((ext) => `${textureDir}/${texture.getName()}.${ext}`).find((f) => fs.existsSync(f));
  if (edited && fs.statSync(edited).mtimeMs > fs.statSync(input).mtimeMs) {
    before = fs.readFileSync(edited);
    console.log(`texture ${texture.getName()}: using the edited file ${edited}`);
  }
  const image = sharp(Buffer.from(before));
  const meta = await image.metadata();
  const out = await image
    .resize({ width: maxTexture, height: maxTexture, fit: 'inside', withoutEnlargement: true })
    .jpeg({ quality: 90 })
    .toBuffer({ resolveWithObject: true });
  texture.setImage(new Uint8Array(out.data)).setMimeType('image/jpeg');
  console.log(
    `texture ${texture.getName() || '(unnamed)'}: ${meta.width}x${meta.height} ${(before.byteLength / 1e6).toFixed(1)} MB -> ` +
      `${out.info.width}x${out.info.height} ${(out.data.byteLength / 1e6).toFixed(1)} MB`,
  );
}

await io.write(output, doc);
console.log(`dropped: ${dropped.join(', ') || 'nothing'}`);
console.log(`kept: ${kept.join('; ')}`);
console.log(`-> ${output}`);
