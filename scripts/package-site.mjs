// Builds the standalone site for ordinary web hosting (cPanel and the like):
//   npm run package
// Output: site/            the folder to upload (everything inside it goes into public_html or a subfolder)
//         character-mocap-site.zip   the same, zipped for cPanel's File Manager (Upload, then Extract)
//
// The site is the webcam app plus a character picker. It holds no editor, no scripts and no source
// models: only each character's rigged model and its tuning files. All addresses are relative, so
// it works from the top of a domain or from any subfolder.
//
//   npm run package -- --only loveheart   refreshes just that character's folder in site/ (no zip),
//                                         for uploading one new or changed character
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { NodeIO } from '@gltf-transform/core';
import { filesOf, listCharacters } from './characters.mjs';

const OUT = 'site';
const ZIP = 'character-mocap-site.zip';
const only = process.argv.includes('--only') ? process.argv[process.argv.indexOf('--only') + 1] : null;
const mb = (bytes) => `${(bytes / 1e6).toFixed(1)} MB`;

const characters = listCharacters('characters');
if (!characters.length) throw new Error('No characters/<id>/<id>.rigged.glb found.');
if (only && !characters.some((c) => c.id === only)) throw new Error(`No rigged character "${only}".`);

if (!only) {
  console.log('== Building the app');
  execFileSync(process.execPath, ['node_modules/vite/bin/vite.js', 'build', '--base', './', '--outDir', OUT, '--emptyOutDir'], { stdio: 'inherit' });
}

/**
 * Expression shapes move a small part of the face, but the rig script stores a full-size table of
 * (mostly zero) offsets for each. Storing only the points that move makes the file several times
 * smaller with no change to what is drawn.
 */
async function writeCompactModel(from, to) {
  const io = new NodeIO();
  const doc = await io.read(from);
  for (const mesh of doc.getRoot().listMeshes()) {
    for (const prim of mesh.listPrimitives()) {
      for (const target of prim.listTargets()) {
        for (const semantic of target.listSemantics()) target.getAttribute(semantic).setSparse(true);
      }
    }
  }
  await io.write(to, doc);
}

console.log('== Characters');
for (const c of characters) {
  if (only && c.id !== only) continue;
  fs.rmSync(path.join(OUT, 'characters', c.id), { recursive: true, force: true });
  for (const file of filesOf(c)) {
    const from = path.join('characters', file);
    const to = path.join(OUT, 'characters', file);
    fs.mkdirSync(path.dirname(to), { recursive: true });
    if (file === c.model) await writeCompactModel(from, to);
    else fs.copyFileSync(from, to);
  }
  const model = path.join(OUT, 'characters', c.model);
  console.log(`  ${c.id.padEnd(14)} ${c.name.padEnd(14)} model ${mb(fs.statSync(path.join('characters', c.model)).size)} -> ${mb(fs.statSync(model).size)}`);
}
fs.writeFileSync(path.join(OUT, 'characters', 'characters.json'), JSON.stringify(characters, null, 2));

if (!only) {
  // Apache settings. Every block is optional, so a host without a module just skips it.
  fs.writeFileSync(
    path.join(OUT, '.htaccess'),
    `# Character Mocap: file types and compression for Apache (cPanel) hosting.
# Serve index.html for the folder address itself. When the site sits in a subfolder of another
# website, these stop that website's own rules (or its Node / Python app) from taking the request.
DirectoryIndex index.html
<IfModule mod_rewrite.c>
  RewriteEngine On
</IfModule>
<IfModule mod_passenger.c>
  PassengerEnabled off
</IfModule>
<IfModule mod_mime.c>
  AddType application/wasm .wasm
  AddType model/gltf-binary .glb
  AddType application/octet-stream .task .tflite
</IfModule>
<IfModule mod_deflate.c>
  AddOutputFilterByType DEFLATE text/html text/css application/javascript text/javascript application/json application/wasm model/gltf-binary
</IfModule>
<IfModule mod_headers.c>
  # Files in assets/ carry a fingerprint in their name, so they can be kept for a year.
  <FilesMatch "-[A-Za-z0-9_-]{8}\\.(js|css)$">
    Header set Cache-Control "public, max-age=31536000, immutable"
  </FilesMatch>
  # The character list is always fetched fresh, so a newly uploaded character shows up at once.
  <FilesMatch "^characters\\.json$">
    Header set Cache-Control "no-cache"
  </FilesMatch>
</IfModule>
`,
  );

  console.log('== Zipping');
  fs.rmSync(ZIP, { force: true });
  // Windows' own tar writes zip files with forward-slash paths, which cPanel extracts correctly.
  const tar = process.platform === 'win32' ? path.join(process.env.SystemRoot ?? 'C:/Windows', 'System32', 'tar.exe') : 'tar';
  try {
    execFileSync(tar, ['-a', '-c', '-f', ZIP, '-C', OUT, '.'], { stdio: 'inherit' });
  } catch (err) {
    console.log(`  Could not make the zip (${err.message}); upload the ${OUT}/ folder itself instead.`);
  }
}

const size = (dir) => fs.readdirSync(dir, { withFileTypes: true }).reduce((sum, e) => sum + (e.isDirectory() ? size(path.join(dir, e.name)) : fs.statSync(path.join(dir, e.name)).size), 0);
console.log(`\nDone. ${OUT}/ is ${mb(size(OUT))}${!only && fs.existsSync(ZIP) ? `, ${ZIP} is ${mb(fs.statSync(ZIP).size)}` : ''}.`);
console.log(only ? `Upload ${OUT}/characters/${only}/ and ${OUT}/characters/characters.json into the site's characters/ folder.` : `Upload ${ZIP} with cPanel's File Manager and extract it where the site should live.`);
