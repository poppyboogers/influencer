# Writes the vertex-group weights of every mesh in a .blend to a JSON file, because the GLB
# exporter drops vertex groups when there is no armature. Read-only for the .blend.
#   blender -b characters/<file>.blend --python tools/blend_weights.py -- characters/<id>.weights.json
# Output: { "<object name>": { "<group name>": [x, y, z, weight, ...] } } with positions in
# glTF axes (x right, y up, z forward), world space, so the rig script can match them by position.
import bpy, json, sys

out = sys.argv[sys.argv.index("--") + 1]
data = {}
for ob in bpy.data.objects:
    if ob.type != 'MESH' or not ob.vertex_groups:
        continue
    groups = {}
    for vg in ob.vertex_groups:
        flat = []
        for v in ob.data.vertices:
            w = 0.0
            for g in v.groups:
                if g.group == vg.index:
                    w = g.weight
            p = ob.matrix_world @ v.co
            flat += [round(p.x, 5), round(p.z, 5), round(-p.y, 5), round(w, 4)]
        groups[vg.name] = flat
    data[ob.name] = groups
with open(out, "w") as f:
    json.dump(data, f)
print("WROTE", out, {k: {g: len(v) // 4 for g, v in gs.items()} for k, gs in data.items()})
