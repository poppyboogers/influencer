// Draws a flat (no perspective) picture of a GLB with a labelled grid in model units, for
// measuring where joints and face features are when writing a rig file.
//   node scripts/measure-glb.mjs <in.glb> <out.png> [--view front|back|left|right|top]
//        [--box h0,h1,v0,v1]   area to show, in the units printed on the grid
//        [--slice lo,hi]       only draw what lies between these depths (along the viewing axis)
//        [--grid 0.1] [--size 1400] [--shade] (grey shaded surface instead of the texture)
//        [--probe "h,v;h,v"]   print how far toward the viewer the surface is at these grid positions
// A model without a texture is drawn in its vertex colours (the --parts preview of the rig script).
// Axes: x = character's left, y = up, z = forward. Views: front looks at the face (x to the right),
// left looks at the character's left side (z to the left), top looks down (x right, z down the page).
import { NodeIO } from '@gltf-transform/core';
import sharp from 'sharp';

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args.splice(i, 2)[1] : fallback;
};
const has = (name) => {
  const i = args.indexOf(name);
  if (i >= 0) args.splice(i, 1);
  return i >= 0;
};
const view = flag('--view', 'front');
const boxArg = flag('--box', null);
const sliceArg = flag('--slice', null);
const gridStep = Number(flag('--grid', 0.1));
const size = Number(flag('--size', 1400));
const shade = has('--shade');
const plain = has('--plain'); // no grid or labels (used to make a picker picture)
const probes = (flag('--probe', '') || '').split(';').filter(Boolean).map((t) => t.split(',').map(Number));
const [input, output] = args;
if (!input || !output) {
  console.error('usage: node scripts/measure-glb.mjs <in.glb> <out.png> [--view front|back|left|right|top] [--box h0,h1,v0,v1] [--slice lo,hi] [--grid 0.1] [--shade]');
  process.exit(1);
}

// h = across the page (left to right), v = up the page, d = toward the viewer.
const views = {
  front: { h: [0, 1], v: [1, 1], d: [2, 1], hName: 'x', vName: 'y' },
  back: { h: [0, -1], v: [1, 1], d: [2, -1], hName: 'x', vName: 'y' },
  left: { h: [2, -1], v: [1, 1], d: [0, 1], hName: 'z', vName: 'y' },
  right: { h: [2, 1], v: [1, 1], d: [0, -1], hName: 'z', vName: 'y' },
  top: { h: [0, 1], v: [2, -1], d: [1, 1], hName: 'x', vName: 'z' },
}[view];
if (!views) throw new Error(`Unknown view "${view}"`);

const doc = await new NodeIO().read(input);
const prims = [];
for (const node of doc.getRoot().listNodes()) {
  const mesh = node.getMesh();
  if (!mesh) continue;
  for (const prim of mesh.listPrimitives()) {
    const pos = prim.getAttribute('POSITION').getArray();
    const uv = prim.getAttribute('TEXCOORD_0')?.getArray();
    const ind = prim.getIndices()?.getArray() ?? Uint32Array.from({ length: pos.length / 3 }, (_, i) => i);
    const tex = prim.getMaterial()?.getBaseColorTexture();
    let image = null;
    if (tex && uv && !shade) {
      const { data, info } = await sharp(Buffer.from(tex.getImage())).removeAlpha().raw().toBuffer({ resolveWithObject: true });
      image = { data, w: info.width, h: info.height };
    }
    const colours = prim.getAttribute('COLOR_0');
    prims.push({ pos, uv, ind, image, colours: !image && !shade && colours ? colours.getArray() : null, colourSize: colours?.getElementSize() ?? 3, factor: prim.getMaterial()?.getBaseColorFactor() ?? [0.7, 0.7, 0.7, 1] });
  }
}

// Model-space coordinate -> (h, v, d) in grid units (h and v carry the sign of the model axis).
const H = (p, i) => p[3 * i + views.h[0]];
const V = (p, i) => p[3 * i + views.v[0]];
const D = (p, i) => p[3 * i + views.d[0]] * views.d[1];
const slice = sliceArg ? sliceArg.split(',').map(Number) : null;

let box = boxArg ? boxArg.split(',').map(Number) : null;
if (!box) {
  box = [Infinity, -Infinity, Infinity, -Infinity];
  for (const p of prims) {
    for (let i = 0; i < p.pos.length / 3; i++) {
      box[0] = Math.min(box[0], H(p.pos, i)); box[1] = Math.max(box[1], H(p.pos, i));
      box[2] = Math.min(box[2], V(p.pos, i)); box[3] = Math.max(box[3], V(p.pos, i));
    }
  }
  const pad = 0.04 * Math.max(box[1] - box[0], box[3] - box[2]);
  box = [box[0] - pad, box[1] + pad, box[2] - pad, box[3] + pad];
}
const scale = size / Math.max(box[1] - box[0], box[3] - box[2]);
const width = Math.round((box[1] - box[0]) * scale);
const height = Math.round((box[3] - box[2]) * scale);
// Page position of a grid value. An axis drawn backwards (sign -1) runs the other way across the page.
const px = (h) => (views.h[1] > 0 ? (h - box[0]) * scale : (box[1] - h) * scale);
const py = (v) => (views.v[1] > 0 ? (box[3] - v) * scale : (v - box[2]) * scale);

const rgb = new Uint8Array(width * height * 3).fill(255);
const depth = new Float32Array(width * height).fill(-Infinity);
// The slice is measured along the model axis itself (not flipped), so its numbers match the grid.
const sliceOf = (p, i) => p[3 * i + views.d[0]];
for (const p of prims) {
  for (let t = 0; t < p.ind.length; t += 3) {
    const [a, b, c] = [p.ind[t], p.ind[t + 1], p.ind[t + 2]];
    if (slice && [a, b, c].some((i) => sliceOf(p.pos, i) < slice[0] || sliceOf(p.pos, i) > slice[1])) continue;
    const x = [px(H(p.pos, a)), px(H(p.pos, b)), px(H(p.pos, c))];
    const y = [py(V(p.pos, a)), py(V(p.pos, b)), py(V(p.pos, c))];
    const z = [D(p.pos, a), D(p.pos, b), D(p.pos, c)];
    const area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
    if (Math.abs(area) < 1e-9) continue;
    // Light from the viewer: how squarely the triangle faces us.
    const e1 = [0, 1, 2].map((k) => p.pos[3 * b + k] - p.pos[3 * a + k]);
    const e2 = [0, 1, 2].map((k) => p.pos[3 * c + k] - p.pos[3 * a + k]);
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const facing = Math.abs(n[views.d[0]]) / (Math.hypot(...n) || 1);
    const light = shade ? 0.25 + 0.75 * facing : 0.75 + 0.25 * facing;
    const x0 = Math.max(0, Math.floor(Math.min(...x))), x1 = Math.min(width - 1, Math.ceil(Math.max(...x)));
    const y0 = Math.max(0, Math.floor(Math.min(...y))), y1 = Math.min(height - 1, Math.ceil(Math.max(...y)));
    for (let yy = y0; yy <= y1; yy++) {
      for (let xx = x0; xx <= x1; xx++) {
        const cx = xx + 0.5, cy = yy + 0.5;
        const w0 = ((x[1] - cx) * (y[2] - cy) - (x[2] - cx) * (y[1] - cy)) / area;
        const w1 = ((x[2] - cx) * (y[0] - cy) - (x[0] - cx) * (y[2] - cy)) / area;
        const w2 = 1 - w0 - w1;
        if (w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4) continue;
        const zz = w0 * z[0] + w1 * z[1] + w2 * z[2];
        const q = yy * width + xx;
        if (zz <= depth[q]) continue;
        depth[q] = zz;
        let col = [p.factor[0] * 255, p.factor[1] * 255, p.factor[2] * 255];
        if (shade) col = [200, 200, 200];
        else if (p.image) {
          const u = w0 * p.uv[2 * a] + w1 * p.uv[2 * b] + w2 * p.uv[2 * c];
          const v = w0 * p.uv[2 * a + 1] + w1 * p.uv[2 * b + 1] + w2 * p.uv[2 * c + 1];
          const tx = Math.min(p.image.w - 1, Math.max(0, Math.floor((u - Math.floor(u)) * p.image.w)));
          const ty = Math.min(p.image.h - 1, Math.max(0, Math.floor((v - Math.floor(v)) * p.image.h)));
          const s = 3 * (ty * p.image.w + tx);
          col = [p.image.data[s], p.image.data[s + 1], p.image.data[s + 2]];
        } else if (p.colours) {
          const n = p.colourSize;
          col = [0, 1, 2].map((k) => 255 * (w0 * p.colours[n * a + k] + w1 * p.colours[n * b + k] + w2 * p.colours[n * c + k]));
        }
        rgb[3 * q] = col[0] * light; rgb[3 * q + 1] = col[1] * light; rgb[3 * q + 2] = col[2] * light;
      }
    }
  }
}

for (const [h, v] of probes) {
  const q = Math.floor(py(v)) * width + Math.floor(px(h));
  const d = depth[q];
  console.log(`probe ${views.hName} ${h}, ${views.vName} ${v}: ${Number.isFinite(d) ? `${'xyz'[views.d[0]]} = ${(d * views.d[1]).toFixed(3)}` : 'nothing there'}`);
}

// Grid and labels as an overlay.
const lines = [];
const label = (v) => (Math.abs(v) < 1e-9 ? '0' : v.toFixed(gridStep < 0.1 ? 2 : 1).replace(/^(-?)0\./, '$1.'));
const font = Math.max(11, Math.round(size / 95));
for (let k = Math.ceil(box[0] / gridStep); k * gridStep <= box[1]; k++) {
  const v = k * gridStep, x = px(v).toFixed(1);
  const strong = k % 5 === 0;
  lines.push(`<line x1="${x}" y1="0" x2="${x}" y2="${height}" stroke="${v === 0 ? '#d00' : '#06c'}" stroke-opacity="${strong ? 0.75 : 0.3}" stroke-width="1"/>`);
  lines.push(`<text x="${+x + 2}" y="${font}" font-size="${font}" fill="#024">${label(v)}</text>`);
  lines.push(`<text x="${+x + 2}" y="${height - 3}" font-size="${font}" fill="#024">${label(v)}</text>`);
}
for (let k = Math.ceil(box[2] / gridStep); k * gridStep <= box[3]; k++) {
  const v = k * gridStep, y = py(v).toFixed(1);
  const strong = k % 5 === 0;
  lines.push(`<line x1="0" y1="${y}" x2="${width}" y2="${y}" stroke="#06c" stroke-opacity="${strong ? 0.75 : 0.3}" stroke-width="1"/>`);
  lines.push(`<text x="2" y="${+y - 2}" font-size="${font}" fill="#024">${label(v)}</text>`);
  lines.push(`<text x="${width - font * 2.6}" y="${+y - 2}" font-size="${font}" fill="#024">${label(v)}</text>`);
}
if (plain) lines.length = 0;
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" font-family="Arial">${lines.join('')}</svg>`;
await sharp(Buffer.from(rgb.buffer), { raw: { width, height, channels: 3 } })
  .composite([{ input: Buffer.from(svg) }])
  .png()
  .toFile(output);
console.log(`${view} view, across = ${views.hName} ${box[0].toFixed(3)}..${box[1].toFixed(3)}, up = ${views.vName} ${box[2].toFixed(3)}..${box[3].toFixed(3)}${slice ? `, slice ${slice}` : ''} -> ${output}`);
