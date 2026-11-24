// Rebuilds a character after its model was edited and re-exported from Blender:
//   1. prepare the exported GLB (drop the high-poly, bake transforms, shrink textures),
//   2. read the vertex groups from the .blend (skipped when Blender or the file is missing),
//   3. rig it again with the same rig file and saved joint positions.
// Usage: npm run rebuild -- vanduude
// File names come from "export", "blend", "source" and "weights" in characters/<id>/<id>.rig.json, relative to that folder.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';

const id = process.argv[2] ?? 'vanduude';
const dir = `characters/${id}`;
const cfg = JSON.parse(fs.readFileSync(`${dir}/${id}.rig.json`, 'utf8'));
const run = (file, args) => execFileSync(file, args, { stdio: 'inherit' });

if (cfg.export) {
  console.log(`\n== 1. Preparing ${cfg.export}`);
  run(process.execPath, ['scripts/prepare-glb.mjs', `${dir}/${cfg.export}`, `${dir}/${cfg.source}`, ...(cfg.refine ? ['--refine', JSON.stringify(cfg.refine)] : []), ...(cfg.textures ? ['--textures', `${dir}/${cfg.textures}`] : [])]);
  // "height" in the rig file: stand the prepared model on y = 0 at that height (the app expects 2.25).
  if (cfg.height) run(process.execPath, ['scripts/normalize-glb.mjs', `${dir}/${cfg.source}`, `${dir}/${cfg.source}`, String(cfg.height)]);
}
if (cfg.blend && cfg.weights) {
  const roots = ['C:/Program Files/Blender Foundation', 'C:/Program Files (x86)/Blender Foundation'];
  const blender = process.env.BLENDER ?? roots.flatMap((root) => (fs.existsSync(root) ? fs.readdirSync(root).map((dir) => `${root}/${dir}/blender.exe`) : [])).filter((f) => fs.existsSync(f)).sort().pop();
  if (blender && fs.existsSync(`${dir}/${cfg.blend}`)) {
    console.log(`\n== 2. Reading vertex groups from ${cfg.blend}`);
    execFileSync(blender, ['-b', `${dir}/${cfg.blend}`, '--python', 'tools/blend_weights.py', '--', `${dir}/${cfg.weights}`], { stdio: ['ignore', 'ignore', 'inherit'] });
  } else {
    console.log('\n== 2. Skipped vertex groups (Blender or the .blend file was not found); keeping the saved ones.');
  }
}
console.log('\n== 3. Rigging');
run(process.execPath, ['scripts/rig-character.mjs', id]);
