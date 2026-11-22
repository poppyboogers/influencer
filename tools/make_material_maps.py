"""Generate a roughness map and an eye-glow (emissive) map for a rigged character.

    tools/venv/Scripts/python.exe tools/make_material_maps.py vanduude

Reads characters/<id>/<id>.rigged.glb (mesh, colour texture, existing roughness texture) and
<id>.rig.json (eye position, from the eyeBlink shape). Writes, next to them:

    <id>.roughness.png   green = roughness, blue = metal (0), the glTF layout
    <id>.emissive.png    the glowing part of the eyes, black elsewhere

The app picks both up when they exist (src/rigAvatar.ts); delete them to go back to the model's
own textures. A rebuild of the model does not overwrite them; re-run this if the UV layout changes.

How surfaces are told apart:
  teeth / gums / hair   separate objects, found by object name
  cloth                 where the model's own roughness map is already high
  eyes                  triangles near the eye position whose texture colour is orange
  mouth interior        dark red skin
  skin                  everything else
Skin and cloth also get fine variation from the colour texture: raised, lighter detail is a
little glossier and creases a little rougher, so highlights break up instead of looking plastic.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage

# Roughness per surface: 0 = mirror, 1 = chalk.
ROUGH = {
    'skin': 0.6, 'skin_range': (0.44, 0.78), 'skin_detail': 1.2,
    'cloth': 0.88, 'cloth_range': (0.8, 0.97), 'cloth_detail': 0.5,
    'teeth': 0.2, 'gums': 0.28, 'hair': 0.38,
    'eye': 0.06, 'eyelid': 0.42, 'mouth': 0.25,
}
CLOTH_ABOVE = 0.85   # the model's own roughness above this is treated as cloth
EYE_RADIUS = 0.03    # model units around the eye position
DETAIL_BLUR = 6      # texels; detail finer than this modulates roughness


def main(char_id: str) -> None:
    folder = Path(__file__).resolve().parent.parent / 'characters' / char_id
    rig = json.loads((folder / f'{char_id}.rig.json').read_text(encoding='utf-8-sig'))
    scene = trimesh.load(folder / f'{char_id}.rigged.glb', process=False)

    # Object name -> geometry. The body is the object with the most triangles.
    parts = {node: scene.geometry[scene.graph[node][1]] for node in scene.graph.nodes_geometry}
    body_name = max(parts, key=lambda n: len(parts[n].faces))
    body = parts[body_name]
    material = body.visual.material
    colour_img = material.baseColorTexture.convert('RGB')
    size = colour_img.width
    colour = np.asarray(colour_img).astype(np.float32) / 255
    old_img = material.metallicRoughnessTexture
    old = (np.asarray(old_img.convert('RGB').resize((size, size)))[..., 1].astype(np.float32) / 255
           if old_img is not None else np.full((size, size), 0.8, np.float32))

    def raster(geom, faces=None, grow=0):
        """Mask of the texels covered by these triangles (trimesh's v runs bottom-up)."""
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        uv = np.asarray(geom.visual.uv)
        for tri in (geom.faces if faces is None else faces):
            draw.polygon([(uv[i, 0] * size, (1 - uv[i, 1]) * size) for i in tri], fill=255, outline=255)
        mask = np.asarray(img) > 0
        return ndimage.binary_dilation(mask, iterations=grow) if grow else mask

    r, g, b = colour[..., 0], colour[..., 1], colour[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    detail = lum - ndimage.gaussian_filter(lum, DETAIL_BLUR)

    # Skin and cloth, blended across a soft edge.
    cloth = ndimage.gaussian_filter((old > CLOTH_ABOVE).astype(np.float32), 2)
    skin_r = np.clip(ROUGH['skin'] - ROUGH['skin_detail'] * detail, *ROUGH['skin_range'])
    cloth_r = np.clip(ROUGH['cloth'] - ROUGH['cloth_detail'] * detail, *ROUGH['cloth_range'])
    rough = skin_r * (1 - cloth) + cloth_r * cloth

    # Inside of the mouth and tongue: dark red, and not cloth (the shirt has red flowers).
    mouth = ((r - g > 0.24) & (g < 0.39) & (r > 0.3)).astype(np.float32) * (1 - cloth)
    mouth = ndimage.gaussian_filter(mouth, 1.5)
    rough = rough * (1 - mouth) + ROUGH['mouth'] * mouth

    # Separate objects.
    for name, geom in parts.items():
        if name == body_name:
            continue
        kind = ('teeth' if re.search('teeth|tooth', name) else 'gums' if 'gum' in name
                else 'hair' if re.search('hair|dread|tentacle', name) else None)
        if kind:
            rough[raster(geom, grow=6)] = ROUGH[kind]
            print(f'  {name}: {kind} {ROUGH[kind]}')

    # Eyes.
    emissive = np.zeros_like(colour)
    blink = next((m for m in rig.get('morphs', []) if m.get('name') == 'eyeBlink'), None)
    if blink:
        at = np.array(blink['ops'][0]['at'], np.float32)
        centres = np.asarray(body.vertices)[body.faces].mean(axis=1)
        near = np.zeros(len(centres), bool)
        for side in (1, -1):
            near |= np.linalg.norm(centres - at * [side, 1, 1], axis=1) < EYE_RADIUS
        region = raster(body, body.faces[near], grow=4)
        # The lit part of the eye is saturated orange; the lids around it are skin colour.
        glow = region & (r > 0.5) & (r - b > 0.3) & (g > 0.2)
        soft_region = ndimage.gaussian_filter(region.astype(np.float32), 2)
        rough = rough * (1 - soft_region) + ROUGH['eyelid'] * soft_region
        soft_glow = np.clip(ndimage.gaussian_filter(ndimage.binary_dilation(glow, iterations=1).astype(np.float32), 1.2) * 1.5, 0, 1)
        rough = rough * (1 - soft_glow) + ROUGH['eye'] * soft_glow
        emissive = colour * soft_glow[..., None]
        print(f'  eyes: {int(near.sum())} triangles, {int(glow.sum())} glowing texels')
    else:
        print('  no eyeBlink shape in the rig file: eyes not found, emissive map is black')

    out = np.zeros((size, size, 3), np.uint8)
    out[..., 0] = 255  # red is ambient occlusion in the glTF layout; unused, left at "none"
    out[..., 1] = np.clip(rough * 255, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(folder / f'{char_id}.roughness.png', optimize=True)
    Image.fromarray(np.clip(emissive * 255, 0, 255).astype(np.uint8)).save(folder / f'{char_id}.emissive.png', optimize=True)
    print(f'wrote {char_id}.roughness.png and {char_id}.emissive.png ({size} px) in {folder}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'vanduude')
