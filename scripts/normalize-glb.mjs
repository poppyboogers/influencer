// Rescale a GLB so the feet sit at y = 0 and the model is a given height (default 2.25).
// Bakes the change into the vertices. x and z are scaled about the origin, not re-centred.
//   node scripts/normalize-glb.mjs <in.glb> <out.glb> [height]
import { NodeIO } from '@gltf-transform/core';

const [input, output, heightArg] = process.argv.slice(2);
if (!input || !output) {
  console.error('usage: node scripts/normalize-glb.mjs <in.glb> <out.glb> [height]');
  process.exit(1);
}
const height = Number(heightArg ?? 2.25);

const io = new NodeIO();
const doc = await io.read(input);
const root = doc.getRoot();

// Bake node transforms into world-space positions/normals first, so the result is unambiguous.
const done = new Set();
const accessors = [];
for (const node of root.listNodes()) {
  const mesh = node.getMesh();
  if (!mesh) continue;
  const m = node.getWorldMatrix();
  for (const prim of mesh.listPrimitives()) {
    const pos = prim.getAttribute('POSITION');
    if (done.has(pos)) continue;
    done.add(pos);
    const a = pos.getArray();
    for (let i = 0; i < a.length; i += 3) {
      const x = a[i], y = a[i + 1], z = a[i + 2];
      a[i] = m[0] * x + m[4] * y + m[8] * z + m[12];
      a[i + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
      a[i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    }
    const nrm = prim.getAttribute('NORMAL');
    if (nrm && !done.has(nrm)) {
      done.add(nrm);
      const n = nrm.getArray();
      for (let i = 0; i < n.length; i += 3) {
        const x = n[i], y = n[i + 1], z = n[i + 2];
        const nx = m[0] * x + m[4] * y + m[8] * z;
        const ny = m[1] * x + m[5] * y + m[9] * z;
        const nz = m[2] * x + m[6] * y + m[10] * z;
        const len = Math.hypot(nx, ny, nz) || 1;
        n[i] = nx / len; n[i + 1] = ny / len; n[i + 2] = nz / len;
      }
    }
    accessors.push(pos);
  }
}
for (const node of root.listNodes()) {
  node.setTranslation([0, 0, 0]).setRotation([0, 0, 0, 1]).setScale([1, 1, 1]);
}

let minY = Infinity, maxY = -Infinity;
for (const pos of accessors) {
  const a = pos.getArray();
  for (let i = 1; i < a.length; i += 3) {
    if (a[i] < minY) minY = a[i];
    if (a[i] > maxY) maxY = a[i];
  }
}
const scale = height / (maxY - minY);
for (const pos of accessors) {
  const a = pos.getArray();
  for (let i = 0; i < a.length; i += 3) {
    a[i] *= scale;
    a[i + 1] = (a[i + 1] - minY) * scale;
    a[i + 2] *= scale;
  }
  pos.setArray(a);
}

await io.write(output, doc);
console.log(`scale ${scale.toFixed(5)}, lifted by ${(-minY * scale).toFixed(5)} -> ${output}`);
