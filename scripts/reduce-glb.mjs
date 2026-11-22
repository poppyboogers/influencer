// Reduces a very dense model (a raw image-to-3D export) to a triangle count the app can run,
// keeping more detail inside one box than everywhere else (the face, which is what a webcam
// avatar shows most and what the expression shapes have to move).
// Also stands the model on y = 0 at a given height, like normalize-glb.mjs.
//   node scripts/reduce-glb.mjs <in.glb> <out.glb> [--triangles 60000] [--height 2.25]
//        [--detail '{"x":[-0.3,0.3],"y":[1.47,1.9],"z":[0.1,1],"triangles":30000}']
// The detail box is in the rescaled model's units (x = character's left, y = up, z = forward).
// Triangles wholly inside the box are reduced to their own budget; the points where the two
// regions meet are left alone, so no gap opens between them.
import { NodeIO } from '@gltf-transform/core';
import { MeshoptSimplifier } from 'meshoptimizer';

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args.splice(i, 2)[1] : fallback;
};
const target = Number(flag('--triangles', 60000));
const height = Number(flag('--height', 2.25));
const detail = JSON.parse(flag('--detail', 'null'));
const [input, output] = args;
if (!input || !output) {
  console.error('usage: node scripts/reduce-glb.mjs <in.glb> <out.glb> [--triangles n] [--height h] [--detail {...}]');
  process.exit(1);
}

await MeshoptSimplifier.ready;
const io = new NodeIO();
const doc = await io.read(input);
const root = doc.getRoot();
const prims = root.listNodes().flatMap((node) => (node.getMesh() ? node.getMesh().listPrimitives().map((prim) => ({ node, prim })) : []));
if (prims.length !== 1) throw new Error(`Expected one object with one material, found ${prims.length}. Reduce the parts separately.`);
const { node, prim } = prims[0];

// ---- stand the model on the ground at the wanted height --------------------------------
const pos = prim.getAttribute('POSITION').getArray().slice();
const count = pos.length / 3;
{
  const m = node.getWorldMatrix();
  let minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < pos.length; i += 3) {
    const x = pos[i], y = pos[i + 1], z = pos[i + 2];
    pos[i] = m[0] * x + m[4] * y + m[8] * z + m[12];
    pos[i + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
    pos[i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    minY = Math.min(minY, pos[i + 1]);
    maxY = Math.max(maxY, pos[i + 1]);
  }
  const scale = height / (maxY - minY);
  for (let i = 0; i < pos.length; i += 3) {
    pos[i] *= scale;
    pos[i + 1] = (pos[i + 1] - minY) * scale;
    pos[i + 2] *= scale;
  }
  node.setTranslation([0, 0, 0]).setRotation([0, 0, 0, 1]).setScale([1, 1, 1]);
  console.log(`rescaled by ${scale.toFixed(4)} to ${height} tall`);
}
const indices = prim.getIndices() ? Uint32Array.from(prim.getIndices().getArray()) : Uint32Array.from({ length: count }, (_, i) => i);
const normals = prim.getAttribute('NORMAL')?.getArray();

// ---- split the triangles into the detail box and the rest -------------------------------
const within = (range, v) => !range || (v >= range[0] && v <= range[1]);
const inBox = (i) => !!detail && within(detail.x, pos[3 * i]) && within(detail.y, pos[3 * i + 1]) && within(detail.z, pos[3 * i + 2]);
const inside = [], outside = [];
for (let t = 0; t < indices.length; t += 3) {
  const tri = [indices[t], indices[t + 1], indices[t + 2]];
  (tri.every(inBox) ? inside : outside).push(...tri);
}
// A point used by both regions must not move. Points are compared by position, because the
// same place can be stored several times (once per texture seam).
const samePlace = MeshoptSimplifier.generatePositionRemap(pos, 3);
const usedIn = new Uint8Array(count);
for (const i of inside) usedIn[samePlace[i]] |= 1;
for (const i of outside) usedIn[samePlace[i]] |= 2;
const lock = new Uint8Array(count);
let locked = 0;
for (let i = 0; i < count; i++) {
  if (usedIn[samePlace[i]] === 3) {
    lock[i] = 1;
    locked++;
  }
}

const attributes = normals ? Float32Array.from(normals) : new Float32Array(count * 3);
function reduce(list, triangles, label) {
  if (!list.length) return new Uint32Array(0);
  const wanted = Math.min(list.length, triangles * 3);
  const [result, error] = MeshoptSimplifier.simplifyWithAttributes(Uint32Array.from(list), pos, 3, attributes, 3, [0.5, 0.5, 0.5], lock, wanted, 0.02);
  console.log(`${label}: ${(list.length / 3).toLocaleString()} -> ${(result.length / 3).toLocaleString()} triangles (error ${error.toFixed(5)})`);
  return result;
}
const reducedOutside = reduce(outside, target, 'model');
const reducedInside = detail ? reduce(inside, detail.triangles ?? 20000, 'detail box') : new Uint32Array(0);
if (detail) console.log(`${locked.toLocaleString()} points held where the two regions meet`);

// ---- keep only the points still in use ----------------------------------------------------
const merged = new Uint32Array(reducedOutside.length + reducedInside.length);
merged.set(reducedOutside);
merged.set(reducedInside, reducedOutside.length);
const newIndex = new Int32Array(count).fill(-1);
let kept = 0;
for (let k = 0; k < merged.length; k++) {
  if (newIndex[merged[k]] < 0) newIndex[merged[k]] = kept++;
  merged[k] = newIndex[merged[k]];
}
const buffer = root.listBuffers()[0];
for (const name of prim.listSemantics()) {
  const old = prim.getAttribute(name);
  const source = name === 'POSITION' ? pos : old.getArray();
  const size = old.getElementSize();
  const next = new source.constructor(kept * size);
  for (let i = 0; i < count; i++) {
    if (newIndex[i] < 0) continue;
    for (let k = 0; k < size; k++) next[newIndex[i] * size + k] = source[i * size + k];
  }
  prim.setAttribute(name, doc.createAccessor().setType(old.getType()).setNormalized(old.getNormalized()).setArray(next).setBuffer(buffer));
  old.dispose();
}
const oldIndices = prim.getIndices();
prim.setIndices(doc.createAccessor().setType('SCALAR').setArray(merged).setBuffer(buffer));
oldIndices?.dispose();

await io.write(output, doc);
console.log(`-> ${output}: ${(merged.length / 3).toLocaleString()} triangles, ${kept.toLocaleString()} points`);
