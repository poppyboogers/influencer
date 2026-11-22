// Standalone model viewer: /viewer.html?model=vanduude.rigged.glb (looked for in characters/vanduude/).
// Any other file: give its path inside characters/, e.g. model=loveheart/loveheart.lowpoly.glb
// Drag to orbit, scroll to zoom, right-drag to pan.
// Keys: W wireframe mode, S skeleton, T texture, R reset pose.
//
// Optional address parameters (mainly for automated checks):
//   yaw, pitch            starting camera angle in degrees
//   wire=0|1|2            0 shaded, 1 shaded + wireframe, 2 wireframe only
//   skel=1                show the skeleton
//   ortho=a,b,half&dir=front|side   flat (no perspective) view centred on (x,y) or (z,y)
//   pose=jaw:20,0,0;head:0,30,0     bone rotations in degrees (x,y,z)
//   shape=mouthPucker:1,browInnerUp:0.5   expression shapes, 0..1
//   still                           no shirt / hair simulation
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { createBoneOverlay, type BoneOverlay } from './boneOverlay';
import { createSoftBodies, type SoftBodies } from './softBody';

const params = new URLSearchParams(location.search);
const asked = params.get('model') ?? 'vanduude.rigged.glb';
// A character's own files (<id>.rigged.glb, <id>.parts.glb) are found in its folder without spelling the folder out.
const file = asked.includes('/') ? asked : asked.replace(/^([\w-]+(?:\.[\w-]+)*)\.(rigged|parts)\.glb$/, '$1/$&');
const yaw = THREE.MathUtils.degToRad(Number(params.get('yaw') ?? 0));
const pitch = THREE.MathUtils.degToRad(Number(params.get('pitch') ?? 0));

const canvas = document.querySelector<HTMLCanvasElement>('#stage')!;
const info = document.querySelector<HTMLDivElement>('#info')!;
const panel = document.querySelector<HTMLDivElement>('#panel')!;

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x15161a);

const ortho = params.get('ortho')?.split(',').map(Number);
const camera: THREE.PerspectiveCamera | THREE.OrthographicCamera = ortho
  ? new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100)
  : new THREE.PerspectiveCamera(35, 1, 0.01, 100);
const controls = new OrbitControls(camera, canvas);

scene.add(new THREE.HemisphereLight(0xffffff, 0x30343c, 1.6));
const key = new THREE.DirectionalLight(0xffffff, 2.4);
key.position.set(2, 3, 4);
scene.add(key);
const rim = new THREE.DirectionalLight(0x88aaff, 1.2);
rim.position.set(-3, 2, -3);
scene.add(rim);

function resize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  if (camera instanceof THREE.PerspectiveCamera) {
    camera.aspect = w / h;
  } else if (ortho) {
    const half = ortho[2];
    camera.top = half;
    camera.bottom = -half;
    camera.left = (-half * w) / h;
    camera.right = (half * w) / h;
  }
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

// ---- display state ---------------------------------------------------------
let wireMode = Number(params.get('wire') ?? 0); // 0 shaded, 1 shaded + wire, 2 wire only
let showSkeleton = params.has('skel');
let showTexture = true;
const shaded: THREE.Mesh[] = [];
const wires: THREE.Mesh[] = [];
const texturedMaterials = new Map<THREE.Mesh, THREE.Material | THREE.Material[]>();
const plain = new THREE.MeshStandardMaterial({ color: 0xb9b2a6, roughness: 0.75, metalness: 0.05 });
let skeletonHelper: BoneOverlay | null = null;
let softBodies: SoftBodies | null = null;
let modelRoot: THREE.Object3D | null = null;
const bones = new Map<string, THREE.Bone>();
const restPose = new Map<string, THREE.Quaternion>();
const boneAngles = new Map<string, [number, number, number]>();

function applyDisplay() {
  for (const m of shaded) {
    m.visible = wireMode !== 2;
    m.material = showTexture ? texturedMaterials.get(m)! : plain;
  }
  for (const w of wires) w.visible = wireMode !== 0;
  if (skeletonHelper) skeletonHelper.visible = showSkeleton;
  (panel.querySelector('#wire') as HTMLSelectElement).value = String(wireMode);
  (panel.querySelector('#skel') as HTMLInputElement).checked = showSkeleton;
  (panel.querySelector('#tex') as HTMLInputElement).checked = showTexture;
}

function applyBone(name: string) {
  const bone = bones.get(name);
  if (!bone) return;
  const [x, y, z] = boneAngles.get(name) ?? [0, 0, 0];
  const d = THREE.MathUtils.degToRad;
  const delta = new THREE.Quaternion().setFromEuler(new THREE.Euler(d(x), d(y), d(z), 'YXZ'));
  // Rotate on top of the rest pose, in the bone's parent space.
  bone.quaternion.copy(restPose.get(name)!).premultiply(delta);
}

/** Expression shapes (morph targets) by name -> 0..1. */
const shapeValues = new Map<string, number>();
const shapeNames: string[] = [];
function applyShape(name: string) {
  modelRoot?.traverse((o) => {
    const mesh = o as THREE.Mesh;
    const index = mesh.morphTargetDictionary?.[name];
    if (index !== undefined && mesh.morphTargetInfluences) mesh.morphTargetInfluences[index] = shapeValues.get(name) ?? 0;
  });
}

// ---- joint editing -----------------------------------------------------------
// Rest positions of the ordinary bones in model axes (x left, y up, z forward). Moving a joint
// here changes where the bone pivots, without moving the mesh; the skin weights are only
// recomputed when the joints are saved and the model is re-rigged.
const jointPos = new Map<string, [number, number, number]>();
const jointMoved = new Set<string>();
const marker = new THREE.Mesh(
  new THREE.SphereGeometry(0.012, 16, 12),
  new THREE.MeshBasicMaterial({ color: 0xff3b30, depthTest: false, transparent: true }),
);
marker.renderOrder = 20;
marker.visible = false;
scene.add(marker);
const otherSide = (name: string) => (/^left/.test(name) ? name.replace(/^left/, 'right') : /^right/.test(name) ? name.replace(/^right/, 'left') : null);

function moveJoint(name: string, pos: [number, number, number]) {
  const bone = bones.get(name);
  if (!bone || !jointPos.has(name)) return;
  jointPos.set(name, pos);
  jointMoved.add(name);
  const parent = jointPos.get(bone.parent?.name ?? '');
  bone.position.set(pos[0] - (parent?.[0] ?? 0), pos[1] - (parent?.[1] ?? 0), pos[2] - (parent?.[2] ?? 0));
  // Bones below it stay where they are.
  for (const child of bone.children) {
    const c = jointPos.get(child.name);
    if (c) child.position.set(c[0] - pos[0], c[1] - pos[1], c[2] - pos[2]);
  }
  // The mesh is bound to the joint's new place, so it does not move.
  modelRoot?.traverse((o) => {
    const skeleton = (o as THREE.SkinnedMesh).skeleton;
    const index = skeleton?.bones.indexOf(bone) ?? -1;
    if (index >= 0) skeleton.boneInverses[index].makeTranslation(-pos[0], -pos[1], -pos[2]);
  });
  softBodies?.moveJoint(name, pos);
}

// The move handle (red / green / blue arrows) that sits on the selected joint.
const handleTarget = new THREE.Object3D();
scene.add(handleTarget);
const handle = new TransformControls(camera, canvas);
handle.setMode('translate');
handle.setSpace('world');
handle.setSize(0.7);
handle.enabled = false;
const handleGizmo = handle.getHelper();
handleGizmo.visible = false;
scene.add(handleGizmo);
let draggingHandle = false;
handle.addEventListener('dragging-changed', (e) => {
  draggingHandle = e.value as boolean;
  controls.enabled = !draggingHandle; // do not orbit while a joint is being dragged
});
const lastHandle = new THREE.Vector3();
const handleDelta = new THREE.Vector3();
const parentTurn = new THREE.Quaternion();
handle.addEventListener('objectChange', () => {
  const name = (panel.querySelector('#joint') as HTMLSelectElement).value;
  const bone = bones.get(name);
  const pos = jointPos.get(name);
  if (!bone || !pos) return;
  // The drag is measured on screen in the current pose; turn it back into the rest pose the joints are stored in.
  handleDelta.copy(handleTarget.position).sub(lastHandle);
  lastHandle.copy(handleTarget.position);
  if (bone.parent) handleDelta.applyQuaternion(bone.parent.getWorldQuaternion(parentTurn).invert());
  editJoint(name, [pos[0] + handleDelta.x, pos[1] + handleDelta.y, pos[2] + handleDelta.z]);
  syncJoint();
});

/**
 * Moves a joint. With "Mirror X" on, the matching joint on the other side moves the opposite way
 * left/right (both in or both out); with "Mirror Y and Z" on it takes the same depth and height.
 */
function editJoint(name: string, pos: [number, number, number]) {
  moveJoint(name, pos);
  const twin = otherSide(name);
  const twinPos = twin && jointPos.get(twin);
  if (twin && twinPos) {
    const mirrorX = (panel.querySelector('#jmirrorx') as HTMLInputElement).checked;
    const mirrorYZ = (panel.querySelector('#jmirror') as HTMLInputElement).checked;
    const next: [number, number, number] = [mirrorX ? -pos[0] : twinPos[0], mirrorYZ ? pos[1] : twinPos[1], mirrorYZ ? pos[2] : twinPos[2]];
    if (next.some((v, k) => Math.abs(v - twinPos[k]) > 1e-7)) moveJoint(twin, next);
  }
  (panel.querySelector('#jstatus') as HTMLDivElement).textContent = `${jointMoved.size} joint(s) moved, not saved yet.`;
}

/** Shows the handle on a joint and selects it in the panel. */
function selectJoint(name: string) {
  if (!jointPos.has(name)) return;
  (panel.querySelector('#joint') as HTMLSelectElement).value = name;
  marker.visible = true;
  showSkeleton = true;
  applyDisplay();
  handle.attach(handleTarget);
  handle.enabled = true;
  handleGizmo.visible = true;
  syncJoint();
  placeHandle();
}
const jointWorld = new THREE.Vector3();
/** Keeps the handle and the red dot on the selected joint as the model is posed. */
function placeHandle() {
  const bone = bones.get((panel.querySelector('#joint') as HTMLSelectElement).value);
  if (!bone || !marker.visible) return;
  modelRoot?.updateMatrixWorld(true);
  bone.getWorldPosition(jointWorld);
  marker.position.copy(jointWorld);
  if (!draggingHandle) {
    handleTarget.position.copy(jointWorld);
    lastHandle.copy(jointWorld);
  }
}

// Click a joint dot to select it (a click, not a drag: dragging still orbits the view).
let pressAt: [number, number] | null = null;
canvas.addEventListener('pointerdown', (e) => (pressAt = [e.clientX, e.clientY]));
canvas.addEventListener('pointerup', (e) => {
  if (!pressAt || draggingHandle || handle.axis || !jointPos.size) return;
  if (Math.hypot(e.clientX - pressAt[0], e.clientY - pressAt[1]) > 4) return;
  const rect = canvas.getBoundingClientRect();
  let best: string | null = null;
  let bestGap = 16; // pixels
  for (const name of jointPos.keys()) {
    const bone = bones.get(name);
    if (!bone) continue;
    bone.getWorldPosition(jointWorld).project(camera);
    const gap = Math.hypot(rect.left + ((jointWorld.x + 1) / 2) * rect.width - e.clientX, rect.top + ((1 - jointWorld.y) / 2) * rect.height - e.clientY);
    if (gap < bestGap) {
      bestGap = gap;
      best = name;
    }
  }
  if (best) selectJoint(best);
});

function syncJoint() {
  const name = (panel.querySelector('#joint') as HTMLSelectElement).value;
  const pos = jointPos.get(name);
  if (!pos) return;
  // Shown in Blender's axes: X = left/right, Y = depth (forward is negative), Z = up.
  (panel.querySelector('#jx') as HTMLInputElement).value = pos[0].toFixed(3);
  (panel.querySelector('#jy') as HTMLInputElement).value = (-pos[2]).toFixed(3);
  (panel.querySelector('#jz') as HTMLInputElement).value = pos[1].toFixed(3);
}

function resetPose() {
  boneAngles.clear();
  for (const name of bones.keys()) applyBone(name);
  shapeValues.clear();
  for (const name of shapeNames) applyShape(name);
  syncSliders();
}

function syncSliders() {
  const name = (panel.querySelector('#bone') as HTMLSelectElement).value;
  const angles = boneAngles.get(name) ?? [0, 0, 0];
  ['rx', 'ry', 'rz'].forEach((id, i) => {
    (panel.querySelector(`#${id}`) as HTMLInputElement).value = String(angles[i]);
    (panel.querySelector(`#${id}v`) as HTMLSpanElement).textContent = `${angles[i]}°`;
  });
  const shape = (panel.querySelector('#shape') as HTMLSelectElement).value;
  const amount = Math.round((shapeValues.get(shape) ?? 0) * 100);
  (panel.querySelector('#amount') as HTMLInputElement).value = String(amount);
  (panel.querySelector('#amountv') as HTMLSpanElement).textContent = `${amount}%`;
}

panel.innerHTML = `
  <label>View <select id="wire">
    <option value="0">Shaded</option><option value="1">Shaded + wireframe</option><option value="2">Wireframe only</option>
  </select></label>
  <label><input type="checkbox" id="tex" checked> Texture</label>
  <label><input type="checkbox" id="skel"> Skeleton</label>
  <div id="rig" hidden>
    <hr>
    <label>Bone <select id="bone"></select></label>
    <label>X <input type="range" id="rx" min="-90" max="90" value="0"> <span id="rxv">0°</span></label>
    <label>Y <input type="range" id="ry" min="-90" max="90" value="0"> <span id="ryv">0°</span></label>
    <label>Z <input type="range" id="rz" min="-90" max="90" value="0"> <span id="rzv">0°</span></label>
    <div id="face" hidden>
      <hr>
      <label>Expression <select id="shape"></select></label>
      <label>Amount <input type="range" id="amount" min="0" max="100" value="0"> <span id="amountv">0%</span></label>
    </div>
    <button id="reset">Reset pose</button>
    <div id="joints" hidden>
      <hr>
      <b>Move joints</b>
      <label>Joint <select id="joint"></select></label>
      <label>X left/right <input type="number" id="jx" step="0.005"></label>
      <label>Y depth <input type="number" id="jy" step="0.005"></label>
      <label>Z up <input type="number" id="jz" step="0.005"></label>
      <label><input type="checkbox" id="jmirrorx" checked> Mirror X (other side moves opposite)</label>
      <label><input type="checkbox" id="jmirror" checked> Mirror Y and Z to the other side</label>
      <div class="hint" id="jstatus">Click a green joint dot on the model, then drag the arrows that appear on it (or type values here). Axes as in Blender.</div>
      <button id="jsave">Save joints and re-rig</button>
      <button id="jundo">Undo this joint's saved edit</button>
    </div>
  </div>`;
panel.querySelector('#shape')!.addEventListener('change', syncSliders);
panel.querySelector('#amount')!.addEventListener('input', (e) => {
  const name = (panel.querySelector('#shape') as HTMLSelectElement).value;
  shapeValues.set(name, Number((e.target as HTMLInputElement).value) / 100);
  applyShape(name);
  syncSliders();
});
panel.querySelector('#wire')!.addEventListener('change', (e) => {
  wireMode = Number((e.target as HTMLSelectElement).value);
  applyDisplay();
});
panel.querySelector('#tex')!.addEventListener('change', (e) => {
  showTexture = (e.target as HTMLInputElement).checked;
  applyDisplay();
});
panel.querySelector('#skel')!.addEventListener('change', (e) => {
  showSkeleton = (e.target as HTMLInputElement).checked;
  applyDisplay();
});
panel.querySelector('#bone')!.addEventListener('change', syncSliders);
panel.querySelector('#reset')!.addEventListener('click', resetPose);
['rx', 'ry', 'rz'].forEach((id, i) => {
  panel.querySelector(`#${id}`)!.addEventListener('input', (e) => {
    const name = (panel.querySelector('#bone') as HTMLSelectElement).value;
    const angles = boneAngles.get(name) ?? [0, 0, 0];
    angles[i] = Number((e.target as HTMLInputElement).value);
    boneAngles.set(name, angles);
    applyBone(name);
    syncSliders();
  });
});
window.addEventListener('keydown', (e) => {
  if ((e.target as HTMLElement).tagName === 'SELECT') return;
  const k = e.key.toLowerCase();
  if (k === 'w') wireMode = (wireMode + 1) % 3;
  else if (k === 's') showSkeleton = !showSkeleton;
  else if (k === 't') showTexture = !showTexture;
  else if (k === 'r') resetPose();
  else return;
  applyDisplay();
});

// ---- load ------------------------------------------------------------------
new GLTFLoader().load(
  `/characters/${file}`,
  (gltf) => {
    const model = gltf.scene;
    let triangles = 0;
    let vertices = 0;
    let textured = false;
    const meshes: THREE.Mesh[] = [];
    model.traverse((o) => {
      const bone = o as THREE.Bone;
      if (bone.isBone) {
        bones.set(bone.name, bone);
        restPose.set(bone.name, bone.quaternion.clone());
      }
      if ((o as THREE.Mesh).isMesh) meshes.push(o as THREE.Mesh);
    });
    for (const mesh of meshes) {
      const g = mesh.geometry;
      triangles += (g.index ? g.index.count : g.attributes.position.count) / 3;
      vertices += g.attributes.position.count;
      const mat = mesh.material as THREE.MeshStandardMaterial;
      if (mat.map) textured = true;
      mat.polygonOffset = true; // keeps the wireframe overlay from flickering against the surface
      mat.polygonOffsetFactor = 1;
      mat.polygonOffsetUnits = 1;
      texturedMaterials.set(mesh, mesh.material);
      shaded.push(mesh);

      const wire = mesh.clone(); // a SkinnedMesh clone shares the skeleton, so it deforms too
      wire.material = new THREE.MeshBasicMaterial({ color: 0x35e0ff, wireframe: true, transparent: true, opacity: 0.45 });
      mesh.parent!.add(wire);
      wires.push(wire);
    }
    scene.add(model);

    if (bones.size > 0) {
      skeletonHelper = createBoneOverlay(bones.values());
      scene.add(skeletonHelper.object);
      modelRoot = model;
      // ?still turns the shirt / hair simulation off (they then move as ordinary skin), to inspect the mesh exactly as rigged.
      softBodies = createSoftBodies(model, bones, !params.has('still'));
      // The wireframe copies and the untextured material are not part of the loaded model.
      for (const w of wires) softBodies?.patch(w.material as THREE.Material);
      softBodies?.patch(plain);
      const select = panel.querySelector('#bone') as HTMLSelectElement;
      for (const name of bones.keys()) select.add(new Option(name, name));
      (panel.querySelector('#rig') as HTMLDivElement).hidden = false;
    }

    // Joint editor: every ordinary bone, at the rest position the rig script recorded.
    let rigRest: Record<string, [number, number, number]> | undefined;
    for (const b of bones.values()) if (b.userData.restPositions) rigRest = b.userData.restPositions;
    const characterId = file.replace(/^.*\//, '').replace(/\.rigged\.glb$/, '');
    if (rigRest && file.endsWith('.rigged.glb')) {
      const select = panel.querySelector('#joint') as HTMLSelectElement;
      for (const [name, pos] of Object.entries(rigRest)) {
        jointPos.set(name, [pos[0], pos[1], pos[2]]);
        select.add(new Option(name, name));
      }
      select.value = jointPos.has('leftUpperArm') ? 'leftUpperArm' : select.value;
      (panel.querySelector('#joints') as HTMLDivElement).hidden = false;
      const status = panel.querySelector('#jstatus') as HTMLDivElement;
      select.addEventListener('change', () => selectJoint(select.value));
      ['jx', 'jy', 'jz'].forEach((inputId) => {
        panel.querySelector(`#${inputId}`)!.addEventListener('input', () => {
          const value = (q: string) => Number((panel.querySelector(q) as HTMLInputElement).value);
          const name = select.value;
          if (![value('#jx'), value('#jy'), value('#jz')].every(Number.isFinite)) return;
          if (!marker.visible) selectJoint(name);
          editJoint(name, [value('#jx'), value('#jz'), -value('#jy')]);
        });
      });
      const send = async (body: Record<string, number[] | null>) => {
        status.textContent = 'Saving and re-rigging (a few seconds)...';
        try {
          const res = await fetch(`/api/joints/${characterId}`, { method: 'POST', body: JSON.stringify(body) });
          const out = (await res.json()) as { ok: boolean; log: string };
          if (!out.ok) throw new Error(out.log.trim().split('\n').pop());
          location.reload();
        } catch (err) {
          status.textContent = `Could not save: ${(err as Error).message}`;
        }
      };
      panel.querySelector('#jsave')!.addEventListener('click', () => {
        void send(Object.fromEntries([...jointMoved].map((name) => [name, jointPos.get(name)!])));
      });
      panel.querySelector('#jundo')!.addEventListener('click', () => {
        const name = select.value;
        const twin = otherSide(name);
        const mirror = (panel.querySelector('#jmirror') as HTMLInputElement).checked || (panel.querySelector('#jmirrorx') as HTMLInputElement).checked;
        void send({ [name]: null, ...(twin && mirror ? { [twin]: null } : {}) });
      });
      syncJoint();
      // ?joint=leftUpperArm opens with that joint selected.
      if (params.get('joint')) selectJoint(params.get('joint')!);
    }

    // Expression shapes, after the wireframe copies exist so both get them.
    modelRoot = model;
    for (const mesh of meshes) for (const name of Object.keys(mesh.morphTargetDictionary ?? {})) if (!shapeNames.includes(name)) shapeNames.push(name);
    if (shapeNames.length) {
      const select = panel.querySelector('#shape') as HTMLSelectElement;
      for (const name of shapeNames) select.add(new Option(name, name));
      (panel.querySelector('#face') as HTMLDivElement).hidden = false;
      (panel.querySelector('#rig') as HTMLDivElement).hidden = false;
    }
    for (const part of (params.get('shape') ?? '').split(',').filter(Boolean)) {
      const [name, value] = part.split(':');
      shapeValues.set(name, Number(value ?? 1));
      applyShape(name);
    }

    // Pose from the address, e.g. pose=jaw:20,0,0;head:0,30,0
    for (const part of (params.get('pose') ?? '').split(';').filter(Boolean)) {
      const [name, values] = part.split(':');
      const [x = 0, y = 0, z = 0] = (values ?? '').split(',').map(Number);
      boneAngles.set(name, [x, y, z]);
      applyBone(name);
    }
    syncSliders();
    applyDisplay();

    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    if (ortho) {
      const side = params.get('dir') === 'side';
      const target = side ? new THREE.Vector3(0, ortho[1], ortho[0]) : new THREE.Vector3(ortho[0], ortho[1], 0);
      camera.position.copy(target).add(side ? new THREE.Vector3(20, 0, 0) : new THREE.Vector3(0, 0, 20));
      controls.target.copy(target);
    } else {
      const radius = Math.max(size.x, size.y, size.z) / 2;
      const fov = (camera as THREE.PerspectiveCamera).fov;
      const dist = (radius / Math.tan(THREE.MathUtils.degToRad(fov / 2))) * 1.15;
      camera.position.set(
        center.x + dist * Math.sin(yaw) * Math.cos(pitch),
        center.y + dist * Math.sin(pitch),
        center.z + dist * Math.cos(yaw) * Math.cos(pitch),
      );
      controls.target.copy(center);
    }
    controls.update();

    info.textContent =
      `${file}\n${Math.round(triangles).toLocaleString()} triangles, ${vertices.toLocaleString()} vertices\n` +
      `texture: ${textured ? 'yes' : 'none'}   skeleton: ${bones.size ? bones.size + ' bones' : 'none'}   expressions: ${shapeNames.length || 'none'}\n` +
      `size ${size.x.toFixed(2)} x ${size.y.toFixed(2)} x ${size.z.toFixed(2)}\n` +
      `[W] wireframe  [S] skeleton  [T] texture  [R] reset pose`;
    // Lets automated screenshots wait until the model is on screen.
    requestAnimationFrame(() => requestAnimationFrame(() => ((window as unknown as { __ready: boolean }).__ready = true)));
    if (params.has('bare')) {
      info.style.display = 'none';
      panel.style.display = 'none';
    }
  },
  undefined,
  (err) => {
    info.textContent = `Could not load characters/${file}\n${(err as Error).message ?? err}`;
  },
);

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = Math.min(0.1, clock.getDelta());
  controls.update();
  softBodies?.update(dt);
  placeHandle();
  if (skeletonHelper?.visible && modelRoot) {
    modelRoot.updateMatrixWorld(true);
    skeletonHelper.update();
  }
  renderer.render(scene, camera);
});
