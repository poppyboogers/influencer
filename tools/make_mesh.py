"""
Builds an untextured 3D mesh of a character from its T-pose views, locally,
with Tencent's open-source Hunyuan3D-2mv model (low-memory build).

Usage (from the app folder):
    tools/venv/Scripts/python.exe tools/make_mesh.py vandude

Reads   characters/<id>_t-pose_front.png, _side.png, _back.png
Writes  characters/<id>.mesh.glb

Model weights are downloaded once into tools/.cache (several GB).
"""
import argparse
import os
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
APP = TOOLS.parent
# Keep every download on this drive rather than the user profile on C:.
os.environ.setdefault("HF_HOME", str(TOOLS / ".cache" / "hf"))
os.environ.setdefault("U2NET_HOME", str(TOOLS / ".cache" / "u2net"))
sys.path.insert(0, str(TOOLS / "Hunyuan3D-2GP"))

parser = argparse.ArgumentParser()
parser.add_argument("character", nargs="?", default="vandude")
parser.add_argument("--quality", action="store_true", help="full model (slower) instead of the turbo model")
parser.add_argument("--resolution", type=int, default=256, help="surface grid resolution; 380 gives more detail")
parser.add_argument("--steps", type=int, default=0, help="override sampling steps")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--faces", type=int, default=120000, help="reduce the mesh to about this many triangles")
parser.add_argument("--side-is", choices=["left", "right"], default="left",
                    help="which side of the character the side image shows")
parser.add_argument("--smooth", type=int, default=12,
                    help="Taubin smoothing passes to remove grid stair-stepping; 0 turns it off")
parser.add_argument("--out", default="", help="output file name inside characters/ (default <id>.mesh.glb)")
parser.add_argument("--profile", type=int, default=4, help="mmgp memory profile; 5 uses the least memory")
args = parser.parse_args()

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from mmgp import offload  # noqa: E402

from hy3dgen.rembg import BackgroundRemover  # noqa: E402
from hy3dgen.shapegen import (  # noqa: E402
    DegenerateFaceRemover,
    FaceReducer,
    FloaterRemover,
    Hunyuan3DDiTFlowMatchingPipeline,
)
from hy3dgen.shapegen.pipelines import export_to_trimesh  # noqa: E402


def replace_property_getter(instance, property_name, new_getter):
    """Same trick as the upstream app: lets mmgp move the model between RAM and GPU."""
    original_class = type(instance)
    original_property = getattr(original_class, property_name)
    custom_class = type(f"Custom{original_class.__name__}", (original_class,), {})
    setattr(custom_class, property_name, property(new_getter, original_property.fset))
    instance.__class__ = custom_class
    return instance


chars = APP / "characters"
views = {
    "front": chars / f"{args.character}_t-pose_front.png",
    "back": chars / f"{args.character}_t-pose_back.png",
}
# Side views: prefer explicit _side-left / _side-right files; fall back to a single _side file.
for side in ("left", "right"):
    p = chars / f"{args.character}_t-pose_side-{side}.png"
    if p.exists():
        views[side] = p
if "left" not in views and "right" not in views:
    views[args.side_is] = chars / f"{args.character}_t-pose_side.png"
print("Views:", ", ".join(f"{k}={v.name}" for k, v in views.items()))
missing = [str(p) for p in views.values() if not p.exists()]
if missing:
    sys.exit("Missing view images:\n  " + "\n  ".join(missing))

t0 = time.time()
print("Removing backgrounds ...")
rembg = BackgroundRemover()
images = {}
debug_dir = TOOLS / ".cache" / "views"
debug_dir.mkdir(parents=True, exist_ok=True)
for name, path in views.items():
    cut = rembg(Image.open(path).convert("RGB"))
    cut.save(debug_dir / f"{args.character}_{name}.png")
    images[name] = cut
del rembg

subfolder = "hunyuan3d-dit-v2-mv" if args.quality else "hunyuan3d-dit-v2-mv-turbo"
steps = args.steps or (30 if args.quality else 8)
print(f"Loading tencent/Hunyuan3D-2mv/{subfolder} ...")
torch.set_default_device("cpu")
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2mv", subfolder=subfolder, use_safetensors=True, device="cuda"
)
if not args.quality:
    pipe.enable_flashvdm(mc_algo="mc")

replace_property_getter(pipe, "_execution_device", lambda self: "cuda")
models = offload.extract_models("i23d_worker", pipe)
kwargs = {"budgets": {"*": 2200}}
if args.profile < 5:
    kwargs["pinnedMemory"] = "i23d_worker/model"
offload.profile(models, profile_no=args.profile, verboseLevel=1, **kwargs)

print(f"Generating shape: {steps} steps, resolution {args.resolution} ...")
t1 = time.time()
outputs = pipe(
    image=images,
    num_inference_steps=steps,
    guidance_scale=5.0,
    generator=torch.Generator().manual_seed(args.seed),
    octree_resolution=args.resolution,
    num_chunks=20000,  # small chunks keep the surface decode inside 4 GB
    output_type="mesh",
)
mesh = export_to_trimesh(outputs)[0]
print(f"Shape generated in {time.time() - t1:.0f}s: {len(mesh.faces)} faces")

mesh = FloaterRemover()(mesh)
mesh = DegenerateFaceRemover()(mesh)
if args.smooth > 0:
    # The surface is extracted from a voxel grid, which leaves terraces on slanted faces.
    # Taubin smoothing removes them without shrinking the model the way plain smoothing does.
    import trimesh
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.53, iterations=args.smooth)
    print(f"Smoothed with {args.smooth} Taubin passes")
mesh = FaceReducer()(mesh, max_facenum=args.faces)

out = chars / (args.out or f"{args.character}.mesh.glb")
mesh.export(out)
print(f"Saved {out} ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces) in {time.time() - t0:.0f}s total")
if torch.cuda.is_available():
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
