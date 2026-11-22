// Print mesh / texture statistics for GLB files: node scripts/inspect-glb.mjs <file> [...]
import { NodeIO } from '@gltf-transform/core';
import { statSync } from 'node:fs';

const io = new NodeIO();
for (const file of process.argv.slice(2)) {
  let doc;
  try {
    doc = await io.read(file);
  } catch (e) {
    console.log(`${file}: FAILED ${e.message}`);
    continue;
  }
  const root = doc.getRoot();
  console.log(`\n${file}  (${(statSync(file).size / 1e6).toFixed(1)} MB)`);
  console.log('  extensions:', root.listExtensionsUsed().map((e) => e.extensionName).join(', ') || 'none');
  console.log('  nodes:', root.listNodes().map((n) => n.getName()).slice(0, 20).join(', '));
  console.log('  skins:', root.listSkins().length, ' animations:', root.listAnimations().length);
  let total = 0;
  for (const mesh of root.listMeshes()) {
    for (const prim of mesh.listPrimitives()) {
      const pos = prim.getAttribute('POSITION');
      const idx = prim.getIndices();
      const tris = (idx ? idx.getCount() : pos.getCount()) / 3;
      total += tris;
      const min = pos.getMin([]).map((v) => v.toFixed(3));
      const max = pos.getMax([]).map((v) => v.toFixed(3));
      console.log(
        `  mesh "${mesh.getName()}" tris=${tris} verts=${pos.getCount()} attrs=[${prim.listSemantics().join(',')}] ` +
          `mat="${prim.getMaterial()?.getName()}" min=[${min}] max=[${max}]`,
      );
    }
  }
  console.log('  total triangles:', total);
  for (const tex of root.listTextures()) {
    const size = tex.getSize();
    console.log(`  texture "${tex.getName()}" ${tex.getMimeType()} ${size?.join('x')} ${(tex.getImage().byteLength / 1e6).toFixed(2)} MB`);
  }
  for (const mat of root.listMaterials()) {
    console.log(
      `  material "${mat.getName()}" base=${!!mat.getBaseColorTexture()} normal=${!!mat.getNormalTexture()} ` +
        `mr=${!!mat.getMetallicRoughnessTexture()} emissive=${!!mat.getEmissiveTexture()} occ=${!!mat.getOcclusionTexture()}`,
    );
  }
}
