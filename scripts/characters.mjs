// The list of characters the app can show, read from the characters/ folder.
// Used by the dev server (vite.config.ts) and by the hosting package (scripts/package-site.mjs),
// so both describe a character the same way.
//
// A character is any folder characters/<id>/ that holds <id>.rigged.glb. Optional beside it:
//   <id>.face.json                      expression tuning
//   <id>.roughness.png, <id>.emissive.png   generated material maps
//   <id>.seat.png                       seat behind the character in stream mode, on a green background
//   <id>.thumb.jpg|png|webp             picture for the character picker; otherwise the first
//                                       image with "front" in its name is used
// "displayName" in <id>.rig.json sets the name shown in the picker (default: the id, capitalised).
import fs from 'node:fs';
import path from 'node:path';

/**
 * @returns {{ id: string, name: string, model: string, face?: string, roughness?: string,
 *   emissive?: string, seat?: string, thumb?: string, v: string }[]} file paths are relative to `root`
 */
export function listCharacters(root = 'characters') {
  if (!fs.existsSync(root)) return [];
  const out = [];
  for (const id of fs.readdirSync(root).sort()) {
    const dir = path.join(root, id);
    // Letters, digits, "_", "-" and single dots inside the name (sc.defiance.red).
    if (!/^[\w-]+(\.[\w-]+)*$/.test(id) || !fs.statSync(dir).isDirectory()) continue;
    const names = fs.readdirSync(dir);
    if (!names.includes(`${id}.rigged.glb`)) continue;
    const has = (name) => (names.includes(name) ? `${id}/${name}` : undefined);
    const thumbName =
      ['jpg', 'png', 'webp'].map((ext) => `${id}.thumb.${ext}`).find((n) => names.includes(n)) ??
      names.find((n) => /front/i.test(n) && /\.(jpe?g|png|webp)$/i.test(n));
    let name = id.charAt(0).toUpperCase() + id.slice(1);
    try {
      name = JSON.parse(fs.readFileSync(path.join(dir, `${id}.rig.json`), 'utf8')).displayName ?? name;
    } catch {
      /* no rig file (an uploaded character): keep the id as its name */
    }
    const entry = {
      id,
      name,
      model: `${id}/${id}.rigged.glb`,
      face: has(`${id}.face.json`),
      roughness: has(`${id}.roughness.png`),
      emissive: has(`${id}.emissive.png`),
      seat: has(`${id}.seat.png`),
      thumb: thumbName && `${id}/${thumbName}`,
    };
    // Changes whenever one of the character's files does, so browsers fetch the new copy.
    const files = [entry.model, entry.face, entry.roughness, entry.emissive, entry.seat, entry.thumb].filter(Boolean);
    entry.v = Math.max(...files.map((f) => fs.statSync(path.join(root, f)).mtimeMs)).toString(36).replace('.', '');
    out.push(entry);
  }
  return out;
}

/** The files of one list entry, relative to the characters folder. */
export const filesOf = (entry) => [entry.model, entry.face, entry.roughness, entry.emissive, entry.seat, entry.thumb].filter(Boolean);
