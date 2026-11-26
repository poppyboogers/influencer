import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { createLook, type StreamSetup } from './lighting';
import type { Vec3 } from './body';
import { createBoneOverlay } from './boneOverlay';
import { characterUrl, type CharacterInfo } from './characters';
import { createSoftBodies } from './softBody';
import type { TorsoPose } from './torso';
import { createWrinkles, type WrinkleConfig } from './wrinkles';
import { createWebcamHead, type WebcamHead, type WebcamHeadConfig } from './webcamHead';

/** Unit directions in avatar space: x = screen right, y = up, z = toward the viewer. */
export interface ArmPose {
  upper: Vec3;
  lower: Vec3;
}

/** A hand in avatar space. Fingers are thumb, index, middle, ring, pinky. */
export interface HandPose {
  forward: Vec3; // wrist -> middle knuckle
  side: Vec3; // pinky knuckle -> index knuckle
  fingers: { prox: Vec3; dist: Vec3 }[];
}

/** What the trackers ask the avatar to do. "left"/"right" are the avatar's own sides. */
export interface RigPose {
  yaw: number; // radians
  pitch: number;
  roll: number;
  jaw: number; // 0..1
  browLeft: number; // -1 (down) .. 1 (up)
  browRight: number;
  smileLeft: number; // 0..1
  smileRight: number;
  /** Expression shapes by name (the face tracker's names, e.g. mouthPucker, eyeBlinkLeft), each 0..1. */
  shapes: Record<string, number>;
  jawSide: number; // -1..1, positive = toward the avatar's left
  jawForward: number; // 0..1
  leftArm: ArmPose | null; // null = arm not tracked, hang relaxed
  rightArm: ArmPose | null;
  leftHand: HandPose | null; // null = hand not tracked, rest shape
  rightHand: HandPose | null;
  /** null = torso not tracked: stand upright with relaxed shoulders. */
  torso: TorsoPose | null;
}

export interface RigAvatar {
  setPose(pose: RigPose, dt: number): void;
  render(): void;
  /** Shows / hides the skeleton overlay; returns the new state. */
  toggleBones(): boolean;
  /** True when the model has finger bones. */
  hasFingers: boolean;
  /** The model's arm, shoulder to wrist, in shoulder widths; null when it has no arm bones. */
  armPerShoulder: number | null;
  /** Stream mode: where the camera is from the middle of the model's shoulders, in its shoulder widths; null otherwise. */
  cameraFrom: Vec3 | null;
  /** Names of the simulated soft parts, or '' when there are none. */
  softParts: string;
  /** How closed each hand is being read as, for the read-out ('' when no hand is tracked). */
  handState(): { left: string; right: string };
  /** Names of the expression shapes built into the model (empty when it has none). */
  shapeNames: string[];
  /** The operator's own head in place of the helmet; null when the character does not offer it. */
  webcamHead: WebcamHead | null;}

const JAW_OPEN = 0.45; // radians at jaw = 1
const JAW_SIDE = 0.14; // radians of sideways jaw swing at jawSide = 1
const JAW_FORWARD = 0.012; // model units of jaw thrust at jawForward = 1
const NECK_SHARE = 0.35; // the neck takes this share of the head turn, the head the rest
const CHEST_SHARE = 0.12;
/** Of the torso's lean / bend / twist, the lower spine takes this share and the chest the rest. The hips stay put. */
const SPINE_SHARE = 0.45;
/**
 * Shrugs. A shoulder rises partly by the shoulder bone swinging up and partly by the whole bone
 * lifting. Swinging alone folds the mesh between shoulder and neck, because the bone pivots
 * close to the neck and its far end comes up and in over the pivot.
 *   SHRUG_GAIN   how much further the avatar's shoulder rises than the operator's (1 = the same share of the shoulder width)
 *   SHRUG_RISE   the furthest a shoulder joint may drop / rise, in shoulder widths
 *   SHRUG_LIFT   the share of the rise made by lifting the bone instead of swinging it, 0..1
 *   SHRUG_BULGE  strength of the bunched-muscle shape at a full shrug, 0..1
 */
const SHRUG_GAIN = 1.5;
const SHRUG_RISE = [-0.04, 0.18];
const SHRUG_LIFT = 0.5;
const SHRUG_BULGE = 0.6;
/**
 * How far fingers may bend backwards, radians (applies to every character with finger bones):
 * [at the knuckle, past the back of the hand; at the middle joint, past the first finger bone].
 * 0 = never past straight. The thumb's base is looser because a real thumb swings well back.
 */
const FINGER_BACK_LIMIT = [0.17, 0.05];
const THUMB_BACK_LIMIT = 0.7;
/**
 * Closing the hand. The model's fingers have two bones where a real finger has three, and may be
 * long claws, so copying the tracked directions leaves the hand half open when the operator makes
 * a fist. Instead each finger's bend is measured as two angles (at the knuckle, and from there to
 * the tip), multiplied up, and capped where the model's finger is fully folded into the palm.
 *   knuckle / curl        how much further the model bends than the operator, at each of the two joints
 *   knuckleMax / curlMax  the bend of a tight fist at each joint, radians
 *   spread                how much of the fingers' sideways splay is kept (it fades out as the finger closes)
 *   gather                how far the closing fingers' knuckles slide toward the middle of the hand, as a share of the
 *                         way there, so the fingers lie side by side and pressed together (0 = stay where they are)
 *   palmSwing             how far the thumb's own part of the palm swings round in front of the fingers' part, radians
 *                         (a quarter turn = square to it). See THUMB_PALM.
 *   inflate               how much the whole hand swells as it balls up (0.06 = 6%)
 *   thumbAcross / thumbWrap  the thumb in a fist, as [into the palm, along the fingers, toward the little finger]:
 *                         its palm section swings out in front of the palm, nearly square to the fingers' palm
 *                         section, and then its tip section lies across the folded fingers
 */
const FIST = {
  knuckle: 1.3,
  curl: 1.45,
  knuckleMax: 1.8,
  curlMax: 2.5,
  spread: 0.8,
  gather: 0.6,
  palmSwing: 0.95,
  inflate: 0.07,
  thumbAcross: [0.1, 0.95, 0.25],
  thumbWrap: [-0.55, 0.8, 0.3],
};
/**
 * The thumb's part of the palm. In a real hand the thumb is hinged at the wrist, and the fleshy
 * pad at its base swings across the palm when a fist is made; the fingers fold down against it.
 * The model's thumb bones start half way up the thumb, so that pad is part of one rigid palm and
 * cannot move. An extra bone is added for it when the model loads.
 *   pivot   where the hinge is, as a share of the way from the wrist to the thumb's first bone
 *   reach   how much nearer the thumb's line than the fingers' a point of the palm must be to belong
 *           to the thumb: [not at all, entirely], model units. Wider apart = a softer join.
 */
const THUMB_PALM = { pivot: 0.15, reach: [-0.015, 0.02] };
const FINGER_NAMES =['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'];

/**
 * `character.roughness` / `.emissive` are the maps generated by tools/make_material_maps.py; optional.
 * `onProgress` is told how much of the model file has arrived, 0..1.
 */
export async function createRigAvatar(
  canvas: HTMLCanvasElement,
  character: CharacterInfo,
  onProgress?: (fraction: number) => void,
  /** Expression wrinkles drawn in the skin's shader ("wrinkles" in the character's face.json). */
  wrinkleConfig?: WrinkleConfig,
  /** Stream mode (?stream): webcam framing, green screen, seat picture, lens and monitor light. See stream.ts. */
  stream?: StreamSetup,
): Promise<RigAvatar> {
  const url = characterUrl(character, character.model);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  // Stream mode draws exactly the pixels asked for (1024 x 1024), whatever the display's scaling.
  renderer.setPixelRatio(stream ? 1 : Math.min(window.devicePixelRatio, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.05, 50);
  const controls = new OrbitControls(camera, canvas);

  // Lights, reflections, shadows and tone mapping: src/lighting.ts. ?plain gives the old flat lighting.
  const look = createLook(renderer, scene, new URLSearchParams(location.search).has('plain'), stream);

  const gltf = await new GLTFLoader().loadAsync(url, (e) => e.total && onProgress?.(e.loaded / e.total));
  scene.add(gltf.scene);
  const bones = new Map<string, THREE.Bone>();
  /** Meshes that carry expression shapes (morph targets), made by the rig script. */
  const shapeMeshes: THREE.Mesh[] = [];
  gltf.scene.traverse((o) => {
    if ((o as THREE.Bone).isBone) bones.set(o.name, o as THREE.Bone);
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    // Posed limbs can leave the rest-pose bounding box; never cull the character.
    mesh.frustumCulled = false;
    if (mesh.morphTargetDictionary && mesh.morphTargetInfluences) shapeMeshes.push(mesh);
  });
  // Shapes named body... are corrections driven by the body pose (bodyShrugLeft), not face expressions.
  const shapeNames = [...new Set(shapeMeshes.flatMap((m) => Object.keys(m.morphTargetDictionary!)))].filter(
    (name) => !name.startsWith('body') && !name.startsWith('combo_'),
  );
  // combo_<first>_<second> shapes are corrections for two shapes on together (see rig-character.mjs):
  // each is turned on by the product of its two shapes.
  const combos = [...new Set(shapeMeshes.flatMap((m) => Object.keys(m.morphTargetDictionary!)))]
    .filter((name) => name.startsWith('combo_'))
    .map((name) => ({ name, first: name.split('_')[1], second: name.split('_')[2] }));
  const bodyShapes: Record<string, number> = {};
  const bone = (name: string) => bones.get(name) ?? null;

  const soft = createSoftBodies(gltf.scene, bones);
  await look.dress(gltf.scene, soft?.patch, {
    roughness: character.roughness && characterUrl(character, character.roughness),
    emissive: character.emissive && characterUrl(character, character.emissive),
  });
  // After the soft-body patch, which it keeps.
  const wrinkles = createWrinkles(wrinkleConfig, shapeMeshes);
  const overlay = createBoneOverlay(bones.values());
  scene.add(overlay.object);

  // Frame the head and upper body, where a webcam operator's motion shows.
  const head = bone('head');
  // A character with "webcamHead" in its rig file can swap its helmet for the operator's own head.
  let webcamHeadConfig: WebcamHeadConfig | undefined;
  for (const b of bones.values()) webcamHeadConfig ??= b.userData.webcamHead;
  const webcamHead = webcamHeadConfig && head ? createWebcamHead(scene, gltf.scene, head, webcamHeadConfig) : null;
  const focus = head ? head.getWorldPosition(new THREE.Vector3()) : new THREE.Vector3(0, 1.6, 0);
  focus.y -= 0.22;
  // A character with other proportions (a big head) sets its own framing: "view" in the rig file.
  let view: { target?: Vec3; distance?: number } | undefined;
  for (const b of bones.values()) view ??= b.userData.view;
  if (view?.target) focus.set(...view.target);
  controls.target.copy(focus);
  camera.position.set(focus.x, focus.y + 0.05, focus.z + (view?.distance ?? 2.6));
  // ?look=x,y,z,distance,turn frames a close-up for checking: the point looked at, how far from it, degrees round to the side.
  const lookAt = new URLSearchParams(location.search).get('look')?.split(',').map(Number);
  if (lookAt?.length && lookAt.every(Number.isFinite)) {
    const [x, y, z, distance = 0.4, turn = 0] = lookAt;
    const angle = THREE.MathUtils.degToRad(turn);
    controls.target.set(x, y, z);
    camera.position.set(x + Math.sin(angle) * distance, y, z + Math.cos(angle) * distance);
  }
  let cameraFrom: Vec3 | null = null;
  if (stream && head) {
    // A webcam on top of the monitor: close, wide, a little above the head and looking slightly down.
    const at = head.getWorldPosition(new THREE.Vector3());
    const cam = stream.config.camera;
    camera.fov = cam.fov;
    camera.position.set(at.x, at.y + cam.height, at.z + cam.distance);
    controls.target.set(at.x, at.y - cam.lookBelow, at.z);
    controls.enabled = false; // the seat picture is flat: it only works from this one angle
    const l = bone('leftUpperArm');
    const r = bone('rightUpperArm');
    if (l && r) {
      const a = l.getWorldPosition(new THREE.Vector3());
      const b = r.getWorldPosition(new THREE.Vector3());
      const span = a.distanceTo(b);
      const mid = a.add(b).multiplyScalar(0.5);
      if (span > 1e-4) cameraFrom = [(camera.position.x - mid.x) / span, (camera.position.y - mid.y) / span, (camera.position.z - mid.z) / span];
    }
  }
  controls.update();

  function resize() {
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    camera.aspect = canvas.clientWidth / canvas.clientHeight;
    camera.updateProjectionMatrix();
    look.resize();
  }
  window.addEventListener('resize', resize);
  resize();

  // ---- face ------------------------------------------------------------------
  const neck = bone('neck');
  const chest = bone('chest');
  const jaw = bone('jaw');
  const jawRest = jaw?.position.clone();
  const faceRest = new Map<string, THREE.Vector3>();
  for (const name of ['leftBrow', 'rightBrow', 'leftMouthCorner', 'rightMouthCorner']) {
    const b = bone(name);
    if (b) faceRest.set(name, b.position.clone());
  }
  const euler = new THREE.Euler();

  // ---- torso and shoulders ---------------------------------------------------
  const spine = bone('spine');
  const shoulders = { left: bone('leftShoulder'), right: bone('rightShoulder') };
  /** Torso pose being shown; eases toward the tracked one, or back to upright when tracking is lost. */
  const torso: TorsoPose = { lean: 0, bend: 0, twist: 0, shrugLeft: 0, shrugRight: 0 };
  // A shrug arrives as a fraction of the operator's shoulder width; the same fraction of the
  // avatar's shoulder width is how far its shoulder joint should rise.
  gltf.scene.updateMatrixWorld(true);
  const shoulderSpan = (() => {
    const l = bone('leftUpperArm');
    const r = bone('rightUpperArm');
    return l && r ? l.getWorldPosition(new THREE.Vector3()).distanceTo(r.getWorldPosition(new THREE.Vector3())) : 0.5;
  })();
  const shoulderRest = {
    left: shoulders.left?.position.clone() ?? null,
    right: shoulders.right?.position.clone() ?? null,
  };
  const vLift = new THREE.Vector3();
  const qLift = new THREE.Quaternion();
  function shrug(side: 'left' | 'right', amount: number) {
    const b = shoulders[side];
    const rest = shoulderRest[side];
    const arm = bone(`${side}UpperArm`);
    if (!b || !rest || !arm) return;
    // How far the shoulder joint rises, in model units.
    const rise = THREE.MathUtils.clamp(amount * SHRUG_GAIN, SHRUG_RISE[0], SHRUG_RISE[1]) * shoulderSpan;
    // Part of it by lifting the whole bone straight up (worked out in the chest's own axes)...
    vLift.set(0, rise * SHRUG_LIFT, 0).applyQuaternion(b.parent!.getWorldQuaternion(qLift).invert());
    b.position.copy(rest).add(vLift);
    // ...the rest by swinging it up about the front-to-back axis, a little forward as it rises.
    const angle = Math.asin(THREE.MathUtils.clamp((rise * (1 - SHRUG_LIFT)) / arm.position.length(), -1, 1));
    const s = side === 'left' ? 1 : -1;
    b.rotation.set(0, -s * angle * 0.2, s * angle);
    // The muscle between neck and shoulder bunches up as the shoulder rises (shape made by the rig script).
    bodyShapes[side === 'left' ? 'bodyShrugLeft' : 'bodyShrugRight'] =
      THREE.MathUtils.clamp(rise / (SHRUG_RISE[1] * shoulderSpan), 0, 1) * SHRUG_BULGE;
  }
  const qTorso = new THREE.Quaternion();
  const qHead = new THREE.Quaternion();
  const qPart = new THREE.Quaternion();
  function offsetBone(name: string, dx: number, dy: number) {
    const b = bone(name);
    const rest = faceRest.get(name);
    if (b && rest) b.position.set(rest.x + dx, rest.y + dy, rest.z);
  }

  // ---- arms ------------------------------------------------------------------
  // Each arm bone points along the offset to its child at rest. To pose it, rotate that
  // rest direction onto the tracked direction, expressed in the parent's space.
  interface ArmRig {
    upper: THREE.Bone;
    lower: THREE.Bone;
    upperRest: THREE.Vector3;
    lowerRest: THREE.Vector3;
    relaxed: ArmPose;
  }
  // Where untracked arms hang; a character can set its own in the rig file ("relaxedArms", left arm).
  let relaxedArms: { upper: Vec3; lower: Vec3 } | undefined;
  for (const b of bones.values()) relaxedArms ??= b.userData.relaxedArms;
  // Stream mode: the arms rest forward, as if on a keyboard and mouse below the picture.
  if (stream) relaxedArms = stream.config.restArms;
  function armRig(side: 'left' | 'right'): ArmRig | null {
    const upper = bone(`${side}UpperArm`);
    const lower = bone(`${side}LowerArm`);
    const hand = bone(`${side}Hand`);
    if (!upper || !lower || !hand) return null;
    const s = side === 'left' ? 1 : -1; // the avatar's left is +x
    const unit = (x: number, y: number, z: number): Vec3 => {
      const len = Math.hypot(x, y, z);
      return [x / len, y / len, z / len];
    };
    return {
      upper,
      lower,
      upperRest: lower.position.clone().normalize(),
      lowerRest: hand.position.clone().normalize(),
      relaxed: relaxedArms
        ? { upper: unit(relaxedArms.upper[0] * s, relaxedArms.upper[1], relaxedArms.upper[2]), lower: unit(relaxedArms.lower[0] * s, relaxedArms.lower[1], relaxedArms.lower[2]) }
        : { upper: unit(0.3 * s, -0.95, 0.05), lower: unit(0.18 * s, -0.95, 0.25) },
    };
  }
  const arms = { left: armRig('left'), right: armRig('right') };
  // When the hand turns, a real forearm turns with it all the way along: fully at the wrist, less
  // toward the elbow, not at all at the elbow, and it keeps its thickness, because the flesh is
  // carried round the bones rather than wrung. The model has one rigid forearm bone, so all of the
  // turn landed in the skin right at the wrist.
  //
  // Each forearm gets a ladder of extra bones along its length, each turned a step further than
  // the last, and every point of the forearm's skin is shared between the two rungs either side
  // of it. Sharing skin between two bones that differ by a big turn pulls it in toward the middle
  // (the "sweet wrapper" pinch); between rungs a small step apart it does not, so nothing is lost.
  /** Where along the forearm (0 = elbow, 1 = wrist) the turn starts and where it is complete. */
  const FOREARM_TWIST = [0.1, 0.95];
  /** Rungs per forearm. More = less thinning at a big turn (8 keeps 98% of the thickness at a half turn). */
  const FOREARM_RUNGS = 8;
  function addForearmTwist(arm: ArmRig | null, side: 'left' | 'right'): THREE.Bone[] | null {
    const hand = bone(`${side}Hand`);
    if (!arm || !hand || hand.parent !== arm.lower) return null;
    const rungs = Array.from({ length: FOREARM_RUNGS }, (_, i) => {
      const b = new THREE.Bone();
      b.name = `${side}ForearmTwist${i + 1}`;
      arm.lower.add(b); // sits exactly on the forearm bone until it is turned
      return b;
    });
    gltf.scene.updateMatrixWorld(true);
    const elbow = arm.lower.getWorldPosition(new THREE.Vector3());
    const along = hand.getWorldPosition(new THREE.Vector3()).sub(elbow);
    const length = along.length();
    if (length < 1e-4) return null;
    along.divideScalar(length);
    const v = new THREE.Vector3();
    const done = new Set<THREE.BufferGeometry>();
    const skeletons = new Set<THREE.Skeleton>();
    gltf.scene.traverse((o) => {
      const mesh = o as THREE.SkinnedMesh;
      if (!mesh.isSkinnedMesh) return;
      const skeleton = mesh.skeleton;
      const from = skeleton.bones.indexOf(arm.lower);
      if (from < 0) return;
      if (!skeletons.has(skeleton)) {
        for (const b of rungs) {
          skeleton.bones.push(b);
          skeleton.boneInverses.push(skeleton.boneInverses[from].clone());
        }
        skeletons.add(skeleton);
      }
      // Rung 0 is the forearm bone itself (no turn); the rest are the new bones.
      const ladder = [from, ...rungs.map((b) => skeleton.bones.indexOf(b))];
      const geometry = mesh.geometry;
      if (done.has(geometry)) return;
      done.add(geometry);
      const position = geometry.getAttribute('position');
      let index = geometry.getAttribute('skinIndex') as THREE.BufferAttribute;
      const weight = geometry.getAttribute('skinWeight') as THREE.BufferAttribute;
      if (!position || !index || !weight) return;
      if (index.array instanceof Uint8Array && skeleton.bones.length > 255) {
        index = new THREE.Uint16BufferAttribute(Array.from(index.array), 4);
        geometry.setAttribute('skinIndex', index);
      }
      for (let i = 0; i < position.count; i++) {
        for (let k = 0; k < 4; k++) {
          const w = weight.getComponent(i, k);
          if (w <= 0 || index.getComponent(i, k) !== from) continue;
          v.fromBufferAttribute(position, i).applyMatrix4(mesh.bindMatrix);
          const share = THREE.MathUtils.smoothstep(v.sub(elbow).dot(along) / length, FOREARM_TWIST[0], FOREARM_TWIST[1]);
          if (share <= 0) break;
          // Between which two rungs this point sits, and how far from the lower to the upper.
          const at = share * FOREARM_RUNGS;
          const lower = Math.min(Math.floor(at), FOREARM_RUNGS - 1);
          const up = at - lower;
          let spare = -1;
          for (let j = 0; j < 4; j++) if (j !== k && weight.getComponent(i, j) <= 0) spare = j;
          if (spare >= 0 && up > 0.001 && up < 0.999) {
            index.setComponent(i, k, ladder[lower]);
            weight.setComponent(i, k, w * (1 - up));
            index.setComponent(i, spare, ladder[lower + 1]);
            weight.setComponent(i, spare, w * up);
          } else index.setComponent(i, k, ladder[up >= 0.5 ? lower + 1 : lower]); // no room for both: the nearer rung
          break; // the forearm holds a point through one slot only
        }
      }
      index.needsUpdate = true;
      weight.needsUpdate = true;
    });
    for (const skeleton of skeletons) {
      skeleton.boneMatrices = new Float32Array(skeleton.bones.length * 16);
      skeleton.boneTexture?.dispose();
      skeleton.boneTexture = null;
    }
    return rungs;
  }
  const forearmTwist = { left: addForearmTwist(arms.left, 'left'), right: addForearmTwist(arms.right, 'right') };
  /** Turns each rung its share of however far the hand is turned about the forearm's length. */
  function twistForearm(side: 'left' | 'right') {
    const rungs = forearmTwist[side];
    const arm = arms[side];
    const hand = bone(`${side}Hand`);
    if (!rungs || !arm || !hand) return;
    const q = hand.quaternion;
    const axis = arm.lowerRest;
    // The part of the hand's rotation that is a turn about the forearm (the rest is the wrist bending),
    // as an angle between a half turn one way and a half turn the other.
    const sign = q.w < 0 ? -1 : 1;
    const turn = 2 * Math.atan2((q.x * axis.x + q.y * axis.y + q.z * axis.z) * sign, q.w * sign);
    rungs.forEach((b, i) => b.quaternion.setFromAxisAngle(axis, (turn * (i + 1)) / FOREARM_RUNGS));
  }
  // The model's own proportions, so a tracked hand can be placed against its body rather than the operator's.
  let armPerShoulder: number | null = null;
  {
    const l = arms.left, r = arms.right, lHand = bone('leftHand');
    if (l && r && lHand) {
      gltf.scene.updateMatrixWorld(true);
      const at = (b: THREE.Object3D) => b.getWorldPosition(new THREE.Vector3());
      const span = at(l.upper).distanceTo(at(r.upper));
      const arm = at(l.upper).distanceTo(at(l.lower)) + at(l.lower).distanceTo(at(lHand));
      if (span > 1e-4) armPerShoulder = arm / span;
    }
  }
  const qWorld = new THREE.Quaternion();
  const qParent = new THREE.Quaternion();
  const qTarget = new THREE.Quaternion();
  const vTarget = new THREE.Vector3();

  const vRelaxed = new THREE.Vector3();
  const qFromRelaxed = new THREE.Quaternion();
  /**
   * Points a bone along `dir`. Many orientations do that, differing by how the bone is rolled
   * about its own length; the one chosen is the smallest turn away from how the bone sits in the
   * relaxed pose, so the limb never rolls further than the move needs and the skin is not wrung.
   */
  function aim(b: THREE.Bone, rest: THREE.Vector3, dir: Vec3, blend: number, relaxed: Vec3) {
    vRelaxed.set(relaxed[0], relaxed[1], relaxed[2]).normalize();
    qWorld.setFromUnitVectors(rest, vRelaxed); // the bone as it sits relaxed
    qFromRelaxed.setFromUnitVectors(vRelaxed, vTarget.set(dir[0], dir[1], dir[2]).normalize()); // the smallest turn from there
    qWorld.premultiply(qFromRelaxed);
    b.parent!.getWorldQuaternion(qParent);
    qTarget.copy(qParent).invert().multiply(qWorld);
    b.quaternion.slerp(qTarget, blend);
  }
  function poseArm(rig: ArmRig | null, pose: ArmPose | null, dt: number) {
    if (!rig) return;
    const target = pose ?? rig.relaxed;
    // Ease toward the target: quick while tracked, slower when falling back to relaxed.
    const blend = 1 - Math.exp(-dt / (pose ? 0.06 : 0.3));
    aim(rig.upper, rig.upperRest, target.upper, blend, rig.relaxed.upper);
    aim(rig.lower, rig.lowerRest, target.lower, blend, rig.relaxed.lower);
  }

  // ---- hands and fingers -------------------------------------------------------
  // The hand takes its whole orientation from the tracked palm (forward + side directions).
  // Each finger bone is then aimed along the tracked finger, measured inside its parent's
  // frame so the claws keep their natural roll as the hand turns.
  interface FingerRig {
    prox: THREE.Bone;
    dist: THREE.Bone;
    proxRest: THREE.Vector3;
    distRest: THREE.Vector3;
  }
  interface HandRig {
    hand: THREE.Bone;
    restInverse: THREE.Quaternion;
    fingers: (FingerRig | null)[];
    /** 1 or -1: which way along (palm forward x palm side) the back of this hand faces. */
    back: number;
  }
  let rigData: { restPositions?: Record<string, Vec3>; restTails?: Record<string, Vec3> } = {};
  for (const b of bones.values()) if (b.userData.restPositions) rigData = b.userData;
  const basisMatrix = new THREE.Matrix4();
  const bf = new THREE.Vector3();
  const bs = new THREE.Vector3();
  const bn = new THREE.Vector3();
  /** Orientation of a palm from its forward and side directions. */
  function palmQuaternion(forward: Vec3, side: Vec3, out: THREE.Quaternion) {
    bf.set(forward[0], forward[1], forward[2]).normalize();
    bs.set(side[0], side[1], side[2]);
    bn.crossVectors(bf, bs).normalize();
    bs.crossVectors(bn, bf).normalize();
    return out.setFromRotationMatrix(basisMatrix.makeBasis(bf, bs, bn));
  }
  function handRig(side: 'left' | 'right'): HandRig | null {
    const hand = bone(`${side}Hand`);
    const rest = rigData.restPositions;
    const tails = rigData.restTails;
    if (!hand || !rest || !tails) return null;
    const at = (name: string) => rest[`${side}${name}`];
    const [wrist, index, pinky] = [at('Hand'), at('Index1'), at('Pinky1')];
    if (!wrist || !index || !pinky) return null;
    // A hand with fewer fingers has no middle knuckle: aim between the first and last finger instead.
    const middle = at('Middle1') ?? ([0, 1, 2].map((k) => (index[k] + pinky[k]) / 2) as Vec3);
    const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const restInverse = palmQuaternion(sub(middle, wrist), sub(index, pinky), new THREE.Quaternion()).invert();
    const fingers = FINGER_NAMES.map((name): FingerRig | null => {
      const prox = bone(`${side}${name}1`);
      const dist = bone(`${side}${name}2`);
      const tail = tails[`${side}${name}2`];
      const distPos = at(`${name}2`);
      if (!prox || !dist || !tail || !distPos) return null;
      return {
        prox,
        dist,
        proxRest: dist.position.clone().normalize(),
        distRest: new THREE.Vector3(...sub(tail, distPos)).normalize(),
      };
    });
    // Palm forward x side points out of the back of a right hand and out of the palm of a left hand.
    return { hand, restInverse, fingers, back: side === 'right' ? 1 : -1 };
  }
  const hands = { left: handRig('left'), right: handRig('right') };

  /** What a hand needs, beyond its finger bones, to ball into a fist. */
  interface FistRig {
    /** The hinge for the thumb's part of the palm, the axis it swings about and which way is across the palm. */
    thumbPalm: { bone: THREE.Bone; axis: THREE.Vector3; sign: number } | null;
    /** Each finger's knuckle: where it sits, and the way to the middle of the knuckles. */
    knuckles: ({ rest: THREE.Vector3; toMiddle: THREE.Vector3 } | null)[];
    /** For the read-out. */
    state: string;
  }
  function fistRig(side: 'left' | 'right'): FistRig | null {
    const rig = hands[side];
    if (!rig) return null;
    const hand = rig.hand;
    // Knuckles of the fingers (not the thumb), in the hand's own space.
    const firsts = rig.fingers.map((f, i) => (f && i > 0 && f.prox.parent === hand ? f.prox : null));
    const present = firsts.filter(Boolean) as THREE.Bone[];
    const middle = new THREE.Vector3();
    for (const b of present) middle.addScaledVector(b.position, 1 / present.length);
    const knuckles = firsts.map((b) => (b ? { rest: b.position.clone(), toMiddle: middle.clone().sub(b.position) } : null));

    let thumbPalm: FistRig['thumbPalm'] = null;
    const thumb = rig.fingers[0]?.prox;
    if (thumb && thumb.parent === hand && present.length) {
      const palm = new THREE.Bone();
      palm.name = `${side}ThumbPalm`;
      palm.position.copy(thumb.position).multiplyScalar(THUMB_PALM.pivot);
      hand.add(palm);
      gltf.scene.updateMatrixWorld(true);
      const at = (b: THREE.Object3D) => b.getWorldPosition(new THREE.Vector3());
      const wrist = at(hand);
      const hinge = at(palm);
      const thumbAt = at(thumb);
      const fingersAt = new THREE.Vector3();
      for (const b of present) fingersAt.addScaledVector(at(b), 1 / present.length);
      const thumbLine = new THREE.Line3(hinge, thumbAt);
      const fingerLine = new THREE.Line3(wrist, fingersAt);
      const v = new THREE.Vector3();
      const near = new THREE.Vector3();
      const done = new Set<THREE.BufferGeometry>();
      const skeletons = new Set<THREE.Skeleton>();
      gltf.scene.traverse((o) => {
        const mesh = o as THREE.SkinnedMesh;
        if (!mesh.isSkinnedMesh) return;
        const skeleton = mesh.skeleton;
        const from = skeleton.bones.indexOf(hand);
        if (from < 0) return;
        if (!skeletons.has(skeleton)) {
          // Where the new bone was when the skin was bound: its own offset on top of the hand's.
          palm.updateMatrix();
          skeleton.bones.push(palm);
          skeleton.boneInverses.push(palm.matrix.clone().invert().multiply(skeleton.boneInverses[from]));
          skeletons.add(skeleton);
        }
        const to = skeleton.bones.indexOf(palm);
        const geometry = mesh.geometry;
        if (done.has(geometry)) return;
        done.add(geometry);
        const position = geometry.getAttribute('position');
        let index = geometry.getAttribute('skinIndex') as THREE.BufferAttribute;
        const weight = geometry.getAttribute('skinWeight') as THREE.BufferAttribute;
        if (!position || !index || !weight) return;
        if (index.array instanceof Uint8Array && skeleton.bones.length > 255) {
          index = new THREE.Uint16BufferAttribute(Array.from(index.array), 4);
          geometry.setAttribute('skinIndex', index);
        }
        for (let i = 0; i < position.count; i++) {
          for (let k = 0; k < 4; k++) {
            const w = weight.getComponent(i, k);
            if (w <= 0 || index.getComponent(i, k) !== from) continue;
            v.fromBufferAttribute(position, i).applyMatrix4(mesh.bindMatrix);
            // The thumb takes the palm that is nearer its own line than the fingers', fading in from the hinge.
            const along = thumbLine.closestPointToPointParameter(v, true);
            const toThumb = thumbLine.closestPointToPoint(v, true, near).distanceTo(v);
            const toFingers = fingerLine.closestPointToPoint(v, true, near).distanceTo(v);
            const share =
              THREE.MathUtils.smoothstep(toFingers - toThumb, THUMB_PALM.reach[0], THUMB_PALM.reach[1]) * THREE.MathUtils.smoothstep(along, 0, 0.35);
            if (share <= 0) break;
            let spare = -1;
            for (let j = 0; j < 4; j++) if (j !== k && weight.getComponent(i, j) <= 0) spare = j;
            if (spare >= 0 && share < 0.999) {
              index.setComponent(i, spare, to);
              weight.setComponent(i, spare, w * share);
              weight.setComponent(i, k, w * (1 - share));
            } else if (share >= 0.5) index.setComponent(i, k, to);
            break;
          }
        }
        index.needsUpdate = true;
        weight.needsUpdate = true;
      });
      for (const skeleton of skeletons) {
        skeleton.boneMatrices = new Float32Array(skeleton.bones.length * 16);
        skeleton.boneTexture?.dispose();
        skeleton.boneTexture = null;
      }
      // The thumb now hangs off its own part of the palm (it stays exactly where it was).
      palm.attach(thumb);
      // It swings about the palm's length. Which way is across the front of the palm: the rest pose's
      // palm frame, as handRig measured it, turned into the hand's own space.
      const toLocal = hand.getWorldQuaternion(new THREE.Quaternion()).invert();
      const axis = fingersAt.clone().sub(wrist).normalize();
      const [index1, pinky1] = [rigData.restPositions?.[`${side}Index1`], rigData.restPositions?.[`${side}Pinky1`]];
      let sign = 1;
      if (index1 && pinky1) {
        const sideways = new THREE.Vector3(index1[0] - pinky1[0], index1[1] - pinky1[1], index1[2] - pinky1[2]);
        const front = new THREE.Vector3().crossVectors(axis, sideways).multiplyScalar(-rig.back); // out of the palm
        const swung = new THREE.Vector3().crossVectors(axis, thumbAt.clone().sub(hinge));
        sign = swung.dot(front) >= 0 ? 1 : -1;
      }
      thumbPalm = { bone: palm, axis: axis.applyQuaternion(toLocal), sign };
    }
    return { thumbPalm, knuckles, state: '' };
  }
  const fists = { left: fistRig('left'), right: fistRig('right') };
  const qPalmSwing = new THREE.Quaternion();
  const IDENTITY = new THREE.Quaternion();
  const qPalm = new THREE.Quaternion();

  const vBack = new THREE.Vector3();
  const vSegBack = new THREE.Vector3();
  const vProx = new THREE.Vector3();
  const vDir = new THREE.Vector3();
  /**
   * Keeps a tracked direction from leaning toward `back` (a unit vector) by more than `limit`
   * radians; a direction within the limit is returned untouched. `fallback` is used when the
   * direction points straight backwards.
   */
  function limitBack(dir: Vec3, back: THREE.Vector3, limit: number, fallback: THREE.Vector3): Vec3 {
    vDir.set(dir[0], dir[1], dir[2]).normalize();
    const lean = vDir.dot(back);
    const most = Math.sin(limit);
    if (lean <= most) return dir;
    vDir.addScaledVector(back, -lean);
    if (vDir.lengthSq() < 1e-6) vDir.copy(fallback);
    vDir.normalize().multiplyScalar(Math.cos(limit)).addScaledVector(back, most);
    return [vDir.x, vDir.y, vDir.z];
  }

  function aimInParent(b: THREE.Bone, rest: THREE.Vector3, dir: Vec3, blend: number) {
    b.parent!.getWorldQuaternion(qParent).invert();
    vTarget.set(dir[0], dir[1], dir[2]).normalize().applyQuaternion(qParent);
    b.quaternion.slerp(qTarget.setFromUnitVectors(rest, vTarget), blend);
  }
  function poseHand(rig: HandRig | null, pose: HandPose | null, dt: number, ball: FistRig | null) {
    if (!rig) return;
    const blend = 1 - Math.exp(-dt / (pose ? 0.06 : 0.3));
    if (!pose) {
      rig.hand.quaternion.slerp(IDENTITY, blend);
      rig.hand.scale.setScalar(rig.hand.scale.x + (1 - rig.hand.scale.x) * blend);
      if (ball) {
        ball.state = '';
        ball.thumbPalm?.bone.quaternion.slerp(IDENTITY, blend);
        ball.knuckles.forEach((k, i) => k && rig.fingers[i]?.prox.position.lerp(k.rest, blend));
      }
      for (const f of rig.fingers) {
        f?.prox.quaternion.slerp(IDENTITY, blend);
        f?.dist.quaternion.slerp(IDENTITY, blend);
      }
      return;
    }
    palmQuaternion(pose.forward, pose.side, qPalm).multiply(rig.restInverse);
    rig.hand.parent!.getWorldQuaternion(qParent);
    rig.hand.quaternion.slerp(qTarget.copy(qParent).invert().multiply(qPalm), blend);
    // palmQuaternion left the palm's forward, side and normal in bf, bs, bn.
    vBack.copy(bn).multiplyScalar(rig.back);
    // The fingers first (the thumb's fist pose depends on how far they have closed).
    let closed = 0;
    let closedOf = 0;
    let indexClosed = 0;
    rig.fingers.forEach((f, i) => {
      if (!f || i === 0) return;
      // This finger stands in for any of the operator's fingers the model does not have that lie
      // between it and the next one it does have: a three-fingered hand closes its last finger
      // when the operator's middle, ring and little fingers close.
      let knuckle = 0;
      let curl = 0;
      let lean = 0;
      let count = 0;
      for (let j = i; j >= 1 && (j === i || !rig.fingers[j]); j--) {
        const t = pose.fingers[j];
        if (!t) continue;
        // In the plane the finger folds in: 0 = straight out along the palm, a quarter turn = straight into the palm.
        const at = (d: Vec3) => Math.atan2(-(d[0] * vBack.x + d[1] * vBack.y + d[2] * vBack.z), d[0] * bf.x + d[1] * bf.y + d[2] * bf.z);
        const first = at(t.prox);
        let second = at(t.dist) - first;
        if (second < -Math.PI) second += 2 * Math.PI; // a tip folded right round past the wrist
        knuckle += first;
        curl += second;
        if (j === i) lean = t.prox[0] * bs.x + t.prox[1] * bs.y + t.prox[2] * bs.z;
        count++;
      }
      if (!count) return;
      // Fingers do not bend backwards past their limits; forwards they stop at a tight fist.
      const a1 = THREE.MathUtils.clamp((knuckle / count) * FIST.knuckle, -FINGER_BACK_LIMIT[0], FIST.knuckleMax);
      const a2 = THREE.MathUtils.clamp((curl / count) * FIST.curl, -FINGER_BACK_LIMIT[1], FIST.curlMax);
      // How far shut this finger is, 0..1.
      const shut = (THREE.MathUtils.clamp(a1 / FIST.knuckleMax, 0, 1) + THREE.MathUtils.clamp(a2 / FIST.curlMax, 0, 1)) / 2;
      if (i === 1) indexClosed = shut;
      else {
        closed += shut;
        closedOf++;
      }
      // Open, a finger keeps its tracked splay. Closing, it straightens up and its knuckle slides
      // toward the middle of the hand, so the fingers end up side by side and pressed together.
      const sideways = lean * FIST.spread * (1 - shut);
      const knuckleAt = ball?.knuckles[i];
      if (knuckleAt) {
        vKnuckle.copy(knuckleAt.rest).addScaledVector(knuckleAt.toMiddle, FIST.gather * THREE.MathUtils.smoothstep(shut, 0.2, 0.9));
        f.prox.position.lerp(vKnuckle, blend);
      }
      const along = (angle: number, out: THREE.Vector3): Vec3 => {
        out.copy(bf).multiplyScalar(Math.cos(angle)).addScaledVector(vBack, -Math.sin(angle)).addScaledVector(bs, sideways).normalize();
        return [out.x, out.y, out.z];
      };
      aimInParent(f.prox, f.proxRest, along(a1, vProx), blend);
      aimInParent(f.dist, f.distRest, along(a1 + a2, vSegBack), blend);
    });
    // A fist: the fingers other than the index are shut (the index may be out, pointing).
    const fist = closedOf ? closed / closedOf : indexClosed;

    const thumb = rig.fingers[0];
    const tracked = pose.fingers[0];
    if (thumb && tracked) {
      // The thumb only joins the fist if the operator's own thumb is folded across (not for a thumbs up):
      // its tip section then points toward the little finger's side.
      const across = -(tracked.dist[0] * bs.x + tracked.dist[1] * bs.y + tracked.dist[2] * bs.z);
      const folded = THREE.MathUtils.smoothstep(across, -0.3, 0.3);
      // First its palm section swings out in front of the palm, then its tip section closes over the fingers.
      const swing = THREE.MathUtils.smoothstep(fist, 0.2, 0.65) * folded;
      const wrap = THREE.MathUtils.smoothstep(fist, 0.5, 0.95) * folded;
      // The thumb's part of the palm swings round in front of the fingers' part, taking the thumb with it.
      const hinge = ball?.thumbPalm;
      if (hinge) hinge.bone.quaternion.slerp(qPalmSwing.setFromAxisAngle(hinge.axis, hinge.sign * FIST.palmSwing * swing), blend);
      if (ball) ball.state = `fist ${fist.toFixed(2)} (index ${indexClosed.toFixed(2)})  thumb across ${across.toFixed(2)} -> folded ${folded.toFixed(2)}  palm swing ${(FIST.palmSwing * swing * 57.3).toFixed(0)} deg`;
      const toward = (dir: Vec3, shape: number[], mix: number, out: THREE.Vector3): Vec3 => {
        // shape = [into the palm, along the fingers, toward the little finger]
        vThumb.set(0, 0, 0).addScaledVector(vBack, -shape[0]).addScaledVector(bf, shape[1]).addScaledVector(bs, -shape[2]).normalize();
        out.set(dir[0], dir[1], dir[2]).normalize().lerp(vThumb, mix).normalize();
        return [out.x, out.y, out.z];
      };
      aimInParent(thumb.prox, thumb.proxRest, toward(limitBack(tracked.prox, vBack, THUMB_BACK_LIMIT, bf), FIST.thumbAcross, swing, vProx), blend);
      aimInParent(thumb.dist, thumb.distRest, toward(tracked.dist, FIST.thumbWrap, wrap, vSegBack), blend);
    }
    // Flesh bunches up in a fist: the hand swells a little as it balls.
    const swell = 1 + FIST.inflate * THREE.MathUtils.smoothstep(fist, 0.4, 1);
    rig.hand.scale.setScalar(rig.hand.scale.x + (swell - rig.hand.scale.x) * blend);
  }
  const vThumb = new THREE.Vector3();
  const vKnuckle = new THREE.Vector3();

  // Start with the arms hanging rather than in the pose the model was built in.
  for (let i = 0; i < 2; i++) {
    poseArm(arms.left, null, 10);
    poseArm(arms.right, null, 10);
  }

  return {
    hasFingers: !!(hands.left ?? hands.right)?.fingers.some(Boolean),
    armPerShoulder,
    cameraFrom,
    softParts: soft?.summary ?? '',
    handState: () => ({ left: fists.left?.state ?? '', right: fists.right?.state ?? '' }),
    shapeNames,
    webcamHead,
    toggleBones() {
      overlay.visible = !overlay.visible;
      return overlay.visible;
    },
    setPose(pose, dt) {
      // Torso: ease toward the tracked pose, then bend the spine and chest from the fixed hips.
      const blend = 1 - Math.exp(-dt / (pose.torso ? 0.06 : 0.3));
      for (const key of Object.keys(torso) as (keyof TorsoPose)[]) {
        torso[key] += ((pose.torso?.[key] ?? 0) - torso[key]) * blend;
      }
      qTorso.identity();
      const bendTorso = (b: THREE.Bone | null, share: number, headShare: number) => {
        if (!b) return;
        // The chest also follows the head a little, as it did before the torso was tracked.
        euler.set(
          torso.bend * share + pose.pitch * headShare,
          torso.twist * share + pose.yaw * headShare,
          -torso.lean * share + pose.roll * headShare,
          'YXZ',
        );
        b.quaternion.setFromEuler(euler);
        qTorso.multiply(b.quaternion);
      };
      bendTorso(spine, SPINE_SHARE, 0);
      bendTorso(chest, spine ? 1 - SPINE_SHARE : 1, CHEST_SHARE);
      shrug('left', torso.shrugLeft);
      shrug('right', torso.shrugRight);

      // The face tracker gives the head's direction in the room, so the neck and head make up
      // whatever the torso has not already turned: the head keeps facing where the operator's does.
      qHead.setFromEuler(euler.set(pose.pitch, pose.yaw, pose.roll, 'YXZ'));
      qHead.premultiply(qTorso.invert());
      if (neck) {
        neck.quaternion.identity().slerp(qHead, NECK_SHARE / (1 - CHEST_SHARE));
        qHead.premultiply(qPart.copy(neck.quaternion).invert());
      }
      head?.quaternion.copy(qHead);
      // restOpen (radians) is set by the rig script when the mouth was modelled open:
      // a closed mouth then means rotating the jaw back up by that much.
      if (jaw && jawRest) {
        jaw.rotation.set(pose.jaw * JAW_OPEN - ((jaw.userData.restOpen as number) ?? 0), pose.jawSide * JAW_SIDE, 0);
        jaw.position.set(jawRest.x, jawRest.y, jawRest.z + pose.jawForward * JAW_FORWARD);
      }
      for (const c of combos) bodyShapes[c.name] = (pose.shapes[c.first] ?? 0) * (pose.shapes[c.second] ?? 0);
      for (const mesh of shapeMeshes) {
        const dict = mesh.morphTargetDictionary!;
        for (const name in dict) mesh.morphTargetInfluences![dict[name]] = bodyShapes[name] ?? pose.shapes[name] ?? 0;
      }
      wrinkles?.update(pose.shapes);

      offsetBone('leftBrow', 0, pose.browLeft * 0.012);
      offsetBone('rightBrow', 0, pose.browRight * 0.012);
      offsetBone('leftMouthCorner', pose.smileLeft * 0.004, pose.smileLeft * 0.008);
      offsetBone('rightMouthCorner', -pose.smileRight * 0.004, pose.smileRight * 0.008);

      poseArm(arms.left, pose.leftArm, dt);
      poseArm(arms.right, pose.rightArm, dt);
      poseHand(hands.left, pose.leftHand, dt, fists.left);
      poseHand(hands.right, pose.rightHand, dt, fists.right);
      twistForearm('left');
      twistForearm('right');

      // Cape and hair react to wherever the body has just been put.
      soft?.update(dt);
    },
    render() {
      controls.update();
      if (overlay.visible) {
        gltf.scene.updateMatrixWorld(true);
        overlay.update();
      }
      webcamHead?.place(camera);
      look.render(scene, camera);
    },
  };
}
