import './style.css';
import { MathUtils } from 'three';
import { createBodyTracker, type ArmSample, type BodySample, type BodyTracker, type Vec3 } from './body';
import { startCamera, type CameraSession } from './camera';
import { characterUrl, loadCharacters, showPicker, type CharacterInfo } from './characters';
import { createHandTracker, type HandSample, type HandTracker } from './hands';
import { createRigAvatar, type ArmPose, type HandPose, type RigPose } from './rigAvatar';
import { OneEuro, Steady, clamp, ease } from './smoothing';
import { TorsoFilter, type TorsoPose } from './torso';
import { createTracker, type FaceSample, type Tracker } from './tracker';

const params = new URLSearchParams(location.search);
const askedFor = params.get('character');
/** ?demo drives the avatar with synthetic motion and never opens the webcam. */
const demo = params.has('demo');

const canvas = document.querySelector<HTMLCanvasElement>('#stage')!;
const video = document.querySelector<HTMLVideoElement>('#cam')!;
const overlayEl = document.querySelector<HTMLPreElement>('#overlay')!;
const messageEl = document.querySelector<HTMLDivElement>('#message')!;

function showMessage(text: string) {
  messageEl.textContent = text;
  messageEl.hidden = false;
}

const CALIBRATION_KEY = 'mocap.calibration';
const LOST_AFTER_MS = 250;
const ARM_LOST_AFTER_MS = 400;
/** An arm has to be seen for this long without a break before the avatar follows it. */
const ARM_ACQUIRE_MS = 250;
/** A hand counts as being at the body tracker's wrist within this distance (fraction of the picture). */
const HAND_AT_WRIST = 0.15;
/** An arm stays confirmed this long after its hand was last found (fists and fast moves drop out briefly). */
const HAND_CONFIRMS_ARM_MS = 800;
const HAND_LOST_AFTER_MS = 350;
/** Maximum head rotation in degrees. */
const LIMITS = { yaw: 55, pitch: 35, roll: 30 };

interface Calibration {
  yaw: number;
  pitch: number;
  roll: number;
  jaw: number;
  /** The tracker's expression signals while the operator's face is relaxed; subtracted before use. */
  blend?: Record<string, number>;
}

/**
 * How strongly each tracker signal drives the avatar's shape of the same name. The tracker
 * reports some expressions timidly (a real sneer or frown rarely scores above 0.4), so those
 * are boosted. Matched by the start of the name; anything not listed uses DEFAULT_GAIN.
 */
const SHAPE_GAIN: [string, number][] = [
  ['eyeBlink', 1.4],
  ['eyeSquint', 1.3],
  ['eyeWide', 2.2],
  ['browDown', 1.6],
  ['browInnerUp', 1.5],
  ['browOuterUp', 1.6],
  ['cheekPuff', 3.0],
  ['cheekSquint', 1.5],
  ['noseSneer', 3.0],
  ['mouthSmile', 1.3],
  ['mouthFrown', 2.5],
  ['mouthPucker', 1.4],
  ['mouthFunnel', 2.2],
  ['mouthLeft', 2.2],
  ['mouthRight', 2.2],
  ['mouthUpperUp', 2.2],
  ['mouthLowerDown', 1.8],
  ['mouthStretch', 2.2],
  ['mouthDimple', 2.0],
  ['mouthPress', 1.6],
  ['mouthRoll', 1.6],
  ['mouthShrug', 1.6],
];
const DEFAULT_GAIN = 1.5;
/** Signal below this (after removing the relaxed-face level) counts as nothing. */
const SHAPE_DEAD_ZONE = 0.04;
/**
 * How far a tracker signal (0..1, before gain) must change before the avatar's face moves at all,
 * so tracking noise on a still face does not make the lips quiver. Blinks are exempt.
 * Raise if the face still twitches at rest; lower if subtle expressions are being missed.
 */
const SHAPE_STEADY = 0.02;
const JAW_STEADY = 0.025;
/**
 * The jaw only opens when the lips have actually parted. The tracker's jawOpen score leans on the
 * chin outline, which wanders under a beard, so with the lips together it is ignored altogether.
 * Numbers are the lip gap beyond the relaxed face, as a fraction of the distance between the outer
 * eye corners (about 90 mm): jaw fully held shut below the first, fully free above the second.
 * Raise the first if the jaw still bobs at rest; lower the second if small mouth openings are lost.
 */
const LIP_GATE = [0.02, 0.07] as const;
/**
 * The jaw also follows the measured lip gap directly, and takes whichever of the two asks for more.
 * The jawOpen score stays small during speech (and smaller under a beard), so on its own the lip
 * shapes parted the lips over closed teeth. Lip gap is a fraction of the outer-eye-corner distance.
 *   deadZone: gap beyond the relaxed face that still counts as closed
 *   gain:     jaw travel (0..1) per unit of gap; about 2 matches the real opening, higher exaggerates
 *   steady:   gap changes smaller than this are ignored (stops the jaw trembling)
 * Overridden by "jaw": { "fromLips", "lipDeadZone" } in the character's face.json; fromLips 0 = off.
 */
const JAW_FROM_LIPS = { deadZone: 0.012, gain: 4.5, steady: 0.008 };
/**
 * Per-character overrides from characters/<id>/<id>.face.json (optional). Saving that file reloads the app
 * (on the dev server; on the hosted site, upload the new file and reload the page).
 *   gain:  how strongly a tracker signal drives the shape (0 switches the shape off)
 *   limit: the furthest the shape may go, 0..1
 * Keys match by the start of the shape name, longest first (mouthSmile covers Left and Right).
 */
interface FaceTuning {
  gain?: Record<string, number>;
  limit?: Record<string, number>;
  /** Strength of the landmark-measured nose scrunch and brow frown (0 = off). */
  landmarks?: { noseScrunch?: number; browFrown?: number };
  /** Jaw opening taken from the measured lip gap (see JAW_FROM_LIPS). */
  jaw?: { fromLips?: number; lipDeadZone?: number };
}
let tuning: FaceTuning = {};
const tuned = (table: Record<string, number> | undefined, name: string) => {
  const key = Object.keys(table ?? {})
    .filter((prefix) => typeof table![prefix] === 'number' && name.startsWith(prefix))
    .sort((p, q) => q.length - p.length)[0];
  return key === undefined ? undefined : table![key];
};
const gainOf = (name: string) =>
  tuned(tuning.gain, name) ?? SHAPE_GAIN.find(([prefix]) => name.startsWith(prefix))?.[1] ?? DEFAULT_GAIN;
/**
 * The tracker's own noseSneer and browDown scores are weak, so these two expressions are also
 * measured from the face landmarks (see tracker.ts). Value = how much that measurement has shrunk
 * from the relaxed face, as a fraction; these turn it into shape strength.
 */
const landmarkGain = () => ({ noseScrunch: tuning.landmarks?.noseScrunch ?? 10, browFrown: tuning.landmarks?.browFrown ?? 8 });
const LANDMARK_DEAD_ZONE = 0.02;
const limitOf = (name: string) => clamp(tuned(tuning.limit, name) ?? 1, 0, 1);
/** mouthSmileLeft <-> mouthSmileRight, mouthLeft <-> mouthRight, ... */
const swapSides = (name: string) => name.replace(/Left|Right/, (side) => (side === 'Left' ? 'Right' : 'Left'));

function loadCalibration(): Calibration {
  try {
    const raw = localStorage.getItem(CALIBRATION_KEY);
    if (raw) return JSON.parse(raw) as Calibration;
  } catch {
    /* storage unavailable: fall through to defaults */
  }
  return { yaw: 0, pitch: 0, roll: 0, jaw: 0 };
}

/** Smooths the six numbers of one arm's two direction vectors. */
class ArmFilter {
  private f = Array.from({ length: 6 }, () => new OneEuro(1.5, 0.4));
  value: ArmPose | null = null;
  lastSeen = -Infinity;
  /** Start of the current unbroken run of sightings. */
  since = Infinity;

  update(upper: Vec3, lower: Vec3, dt: number, now: number) {
    if (now - this.lastSeen > ARM_LOST_AFTER_MS) this.since = now;
    const v = [...upper, ...lower].map((x, i) => this.f[i].filter(x, dt));
    this.value = { upper: [v[0], v[1], v[2]], lower: [v[3], v[4], v[5]] };
    this.lastSeen = now;
  }
}

/** The parts of a hand sample that drive the avatar (directions only). */
type HandShape = Pick<HandSample, 'forward' | 'side' | 'fingers'>;

/** Smooths the twelve direction vectors of one hand (palm forward, palm side, two per finger). */
class HandFilter {
  private f = Array.from({ length: 36 }, () => new OneEuro(1.5, 0.5));
  value: HandPose | null = null;
  lastSeen = -Infinity;

  update(hand: HandShape, map: (v: Vec3) => Vec3, dt: number, now: number) {
    let n = 0;
    const smooth = (v: Vec3): Vec3 => {
      const m = map(v);
      return [this.f[n++].filter(m[0], dt), this.f[n++].filter(m[1], dt), this.f[n++].filter(m[2], dt)];
    };
    this.value = {
      forward: smooth(hand.forward),
      side: smooth(hand.side),
      fingers: hand.fingers.map((finger) => ({ prox: smooth(finger.prox), dist: smooth(finger.dist) })),
    };
    this.lastSeen = now;
  }
}

async function main() {
  const characters = await loadCharacters();
  if (!characters.length) throw new Error('No characters found in the characters folder.');
  // No character in the address: go straight in when there is only one, otherwise ask.
  const character: CharacterInfo | undefined = askedFor
    ? characters.find((c) => c.id === askedFor)
    : characters.length === 1 ? characters[0] : undefined;
  if (!character) {
    video.style.display = 'none';
    overlayEl.hidden = true;
    if (askedFor) {
      showMessage(`There is no character called "${askedFor}". Choose one below.`);
      messageEl.style.top = '24px';
      messageEl.style.transform = 'translate(-50%, 0)';
      messageEl.style.zIndex = '11';
    }
    showPicker(characters);
    return;
  }
  const characterId = character.id;
  document.title = `${character.name} - Character Mocap`;
  if (character.face) {
    try {
      tuning = (await (await fetch(characterUrl(character, character.face))).json()) as FaceTuning;
    } catch (err) {
      console.warn('Face tuning file did not load, using the defaults:', err);
    }
  }
  const LANDMARK_GAIN = landmarkGain();

  overlayEl.textContent = `Loading ${character.name}...`;
  const avatar = await createRigAvatar(canvas, character, (fraction) => {
    overlayEl.textContent = `Loading ${character.name}... ${Math.round(fraction * 100)}%`;
  });

  // Character picker: the button top right, or P.
  let picker: { close(): void } | null = null;
  const togglePicker = () => {
    if (document.querySelector('#picker')) picker?.close();
    else picker = showPicker(characters, characterId);
  };
  if (characters.length > 1) {
    const button = document.createElement('button');
    button.id = 'pick';
    button.textContent = 'Change character';
    button.addEventListener('click', togglePicker);
    document.body.append(button);
  }
  // Helmet / webcam head (only a character with "webcamHead" in its rig file): the button, or V.
  const liveHead = avatar.webcamHead;
  let helmetButton: HTMLButtonElement | null = null;
  const setHelmet = (on: boolean) => {
    if (!liveHead) return;
    liveHead.set(!on);
    if (helmetButton) helmetButton.textContent = on ? 'Helmet: on' : 'Helmet: off (webcam head)';
  };
  if (liveHead) {
    helmetButton = document.createElement('button');
    helmetButton.id = 'helmet';
    helmetButton.title = 'Swap the helmet for your own head from the webcam (V)';
    helmetButton.addEventListener('click', () => setHelmet(liveHead.wanted));
    document.body.append(helmetButton);
    if (demo) liveHead.demo();
    // The helmet always starts off (the operator's own head shows); ?helmet=on in the address starts with it on.
    setHelmet(params.get('helmet') === 'on');
  }
  if (params.has('bones')) avatar.toggleBones(); // same as pressing S
  // The app opens with the read-out and camera preview hidden (H shows them; the buttons always
  // stay). ?ui in the address opens with them showing.
  if (!params.has('ui')) document.body.classList.add('clean');
  avatar.render();

  let tracker: Tracker | null = null;
  let bodyTracker: BodyTracker | null = null;
  let handTracker: HandTracker | null = null;
  let camera: CameraSession | null = null;
  if (demo) {
    video.style.display = 'none';
  } else {
    overlayEl.textContent = 'Waiting for webcam permission...';
    camera = await startCamera(video);
    overlayEl.textContent = 'Loading face tracker...';
    tracker = await createTracker();
    overlayEl.textContent = 'Loading body tracker...';
    try {
      bodyTracker = await createBodyTracker();
    } catch (err) {
      console.warn('Body tracker unavailable, continuing with face only:', err);
    }
    if (avatar.hasFingers) {
      overlayEl.textContent = 'Loading hand tracker...';
      try {
        handTracker = await createHandTracker();
      } catch (err) {
        console.warn('Hand tracker unavailable, continuing without fingers:', err);
      }
    }
  }

  // --- state -------------------------------------------------------------
  let calibration = loadCalibration();
  let mirror = true;
  let bodyOn = true;
  let handsOn = true;
  let handMs = 0;
  let lastHandDetect = performance.now();
  /** Where the body tracker last saw the operator's wrists in the image. */
  let wrists: NonNullable<BodySample['wrists']> = { left: null, right: null };
  let lastRaw: FaceSample | null = null;
  let lastSeen = -Infinity;
  let presence = 0; // 0 = neutral pose, 1 = fully following the operator
  let lastVideoTime = -1;
  let frameCount = 0;
  let inferMs = 0;
  let bodyMs = 0;
  let lastFrame = performance.now();
  let lastDetect = lastFrame;
  let lastBodyDetect = lastFrame;
  let fps = 0;
  let trackFps = 0;

  const filters = {
    yaw: new OneEuro(1.2, 0.6),
    pitch: new OneEuro(1.2, 0.6),
    roll: new OneEuro(1.2, 0.6),
    jaw: new OneEuro(3.0, 1.5), // mouths move fast: smooth less
    browLeft: new OneEuro(2.0, 0.8),
    browRight: new OneEuro(2.0, 0.8),
    smileLeft: new OneEuro(2.0, 0.8),
    smileRight: new OneEuro(2.0, 0.8),
  };
  const face = { yaw: 0, pitch: 0, roll: 0, jaw: 0, browLeft: 0, browRight: 0, smileLeft: 0, smileRight: 0, jawSide: 0, jawForward: 0 };
  /** Expression shapes the model has, by the avatar's own left / right, each 0..1. */
  const shapes: Record<string, number> = {};
  // Blinks are over in a tenth of a second: smooth them far less than the other shapes.
  const shapeFilters = new Map(
    avatar.shapeNames.map((name) => [name, name.startsWith('eyeBlink') ? new OneEuro(12, 2.0) : new OneEuro(2.5, 1.0)]),
  );
  /** Relaxed-face landmark measurements when the saved calibration has none: the first ones seen. */
  const firstSeen: Record<string, number> = {};
  const jawFilters = { side: new OneEuro(2.0, 0.8), forward: new OneEuro(2.0, 0.8) };
  const jawSteady = new Steady(JAW_STEADY);
  const lipGapSteady = new Steady(JAW_FROM_LIPS.steady);
  const shapeSteady = new Map<string, Steady>();
  /** The tracker signal for one shape with its at-rest noise held still. */
  const steadied = (name: string, value: number) => {
    if (name.startsWith('eyeBlink')) return value;
    let s = shapeSteady.get(name);
    if (!s) shapeSteady.set(name, (s = new Steady(SHAPE_STEADY)));
    return s.follow(value);
  };
  /** Keyed by the avatar's own side. */
  const armFilters = { left: new ArmFilter(), right: new ArmFilter() };
  const handFilters = { left: new HandFilter(), right: new HandFilter() };
  /** When the hand tracker last found a hand at the body tracker's wrist (the operator's own left / right). */
  const handAtWrist = { left: -Infinity, right: -Infinity };
  const torsoFilter = new TorsoFilter();

  window.addEventListener('keydown', (e) => {
    const key = e.key.toLowerCase();
    if (key === 'c' && lastRaw) {
      calibration = {
        yaw: lastRaw.yaw,
        pitch: lastRaw.pitch,
        roll: lastRaw.roll,
        jaw: lastRaw.blend.jawOpen ?? 0,
        blend: { ...lastRaw.blend },
      };
      try {
        localStorage.setItem(CALIBRATION_KEY, JSON.stringify(calibration));
      } catch {
        /* calibration just won't persist */
      }
      torsoFilter.recalibrate(); // the way you are sitting now becomes upright and relaxed
    } else if (key === 'm') {
      mirror = !mirror;
      torsoFilter.recalibrate();
    } else if (key === 'b') {
      bodyOn = !bodyOn;
    } else if (key === 'f') {
      handsOn = !handsOn;
    } else if (key === 's') {
      avatar.toggleBones();
    } else if (key === 'k' && camera) {
      void camera.next();
    } else if (key === 'h') {
      document.body.classList.toggle('clean');
    } else if ((key === 'v' || key === 'y') && liveHead) {
      setHelmet(liveHead.wanted);
    } else if (key === 'p' && characters.length > 1) {
      togglePicker();
    }
  });

  // ?demo&pose=yaw,pitch,roll,jaw (degrees, jaw 0..1) holds a fixed head pose, for checking the rig.
  const fixedPose = params.get('pose')?.split(',').map(Number);
  // ?demo&shape=mouthPucker:1,browInnerUp:0.5 holds expression shapes. Without it the demo plays
  // through every shape in turn.
  const fixedShapes = params.get('shape')
    ? Object.fromEntries(params.get('shape')!.split(',').map((part) => [part.split(':')[0], Number(part.split(':')[1] ?? 1)]))
    : null;
  function demoShapes(t: number): Record<string, number> {
    if (fixedShapes) return fixedShapes;
    if (fixedPose || !avatar.shapeNames.length) return {};
    const beat = t / 1.2;
    const name = avatar.shapeNames[Math.floor(beat) % avatar.shapeNames.length];
    return { [name]: Math.sin(Math.PI * (beat % 1)) };
  }

  function demoFace(t: number): FaceSample {
    if (fixedPose) {
      const [yaw = 0, pitch = 0, roll = 0, jaw = 0] = fixedPose;
      return {
        // Pre-undo the mirror flip and jaw gain so the numbers in the address are what you see.
        yaw: MathUtils.degToRad(yaw) * (mirror ? -1 : 1),
        pitch: MathUtils.degToRad(pitch),
        roll: MathUtils.degToRad(roll) * (mirror ? -1 : 1),
        blend: { jawOpen: jaw / 1.8 + 0.03, browInnerUp: 0.8, mouthSmileLeft: 0.9, mouthSmileRight: 0.9 },
        direct: demoShapes(t),
      };
    }
    return {
      yaw: Math.sin(t * 0.9) * 0.6,
      pitch: Math.sin(t * 0.6 + 1) * 0.25,
      roll: Math.sin(t * 0.5 + 2) * 0.15,
      blend: {
        jawOpen: Math.max(0, Math.sin(t * 3.0)) * 0.7,
        browInnerUp: Math.max(0, Math.sin(t * 0.8)),
        mouthSmileLeft: Math.max(0, Math.sin(t * 0.7 + 1)),
        mouthSmileRight: Math.max(0, Math.sin(t * 0.7 + 1)),
      },
      direct: demoShapes(t),
    };
  }

  // ?demo&torso=lean,bend,twist,shrugLeft,shrugRight (degrees; shrugs 0..1) holds a torso pose.
  // Without it the demo sways, twists and shrugs in turn.
  const fixedTorso = params.get('torso')?.split(',').map(Number);
  function demoTorso(t: number): TorsoPose {
    const rad = MathUtils.degToRad;
    if (fixedTorso) {
      const [lean = 0, bend = 0, twist = 0, shrugLeft = 0, shrugRight = 0] = fixedTorso;
      return { lean: rad(lean), bend: rad(bend), twist: rad(twist), shrugLeft: shrugLeft * 0.25, shrugRight: shrugRight * 0.25 };
    }
    if (fixedPose) return { lean: 0, bend: 0, twist: 0, shrugLeft: 0, shrugRight: 0 };
    const shrug = Math.max(0, Math.sin(t * 1.7)) * 0.22;
    return {
      lean: Math.sin(t * 0.7) * 0.2,
      bend: Math.sin(t * 0.45 + 1) * 0.15,
      twist: Math.sin(t * 0.55 + 2) * 0.3,
      shrugLeft: shrug,
      shrugRight: Math.sin(t * 0.4) > 0 ? shrug : 0, // both shoulders, then the left alone
    };
  }

  /** Synthetic arms in MediaPipe space: one arm waves, the other hangs. */
  function demoBody(t: number): BodySample {
    const wave = fixedPose ? 0.5 : Math.sin(t * 2.2) * 0.5;
    const norm = (v: Vec3): Vec3 => {
      const len = Math.hypot(...v);
      return [v[0] / len, v[1] / len, v[2] / len];
    };
    return {
      // Operator's left arm: raised out to their left (image right), forearm up and waving.
      left: { upper: norm([0.9, -0.2, -0.1]), lower: norm([0.3 + wave, -1, -0.2]) },
      right: null,
    };
  }

  /** A synthetic hand on the waving arm, opening and closing. */
  function demoHand(t: number): HandShape {
    const norm = (v: Vec3): Vec3 => {
      const len = Math.hypot(...v);
      return [v[0] / len, v[1] / len, v[2] / len];
    };
    const forward = norm([0.3 + (fixedPose ? 0.5 : Math.sin(t * 2.2) * 0.5), -1, -0.2]);
    const side = norm([forward[1], -forward[0], 0]);
    const normal: Vec3 = [
      forward[1] * side[2] - forward[2] * side[1],
      forward[2] * side[0] - forward[0] * side[2],
      forward[0] * side[1] - forward[1] * side[0],
    ];
    const curl = fixedPose ? 0.5 : 0.5 + 0.5 * Math.sin(t * 2.6);
    const bent = (angle: number): Vec3 =>
      norm([0, 1, 2].map((k) => forward[k] * Math.cos(angle) + normal[k] * Math.sin(angle)) as Vec3);
    return { forward, side, fingers: Array.from({ length: 5 }, () => ({ prox: bent(curl * 1.2), dist: bent(curl * 2.3) })) };
  }

  function consumeFace(sample: FaceSample, dt: number) {
    lastRaw = sample;
    const sign = mirror ? -1 : 1; // mirrored: the avatar behaves like your reflection
    const lim = (deg: number) => MathUtils.degToRad(deg);
    const b = sample.blend;
    const yaw = clamp((sample.yaw - calibration.yaw) * sign, -lim(LIMITS.yaw), lim(LIMITS.yaw));
    const pitch = clamp(sample.pitch - calibration.pitch, -lim(LIMITS.pitch), lim(LIMITS.pitch));
    const roll = clamp((sample.roll - calibration.roll) * sign, -lim(LIMITS.roll), lim(LIMITS.roll));
    // Dead zone at the bottom, gain so a normal talking mouth reaches full travel.
    let lipsApart = 1; // 0 = lips together, 1 = clearly parted (or no lip measurement, as in the demo)
    let jawFromLips = 0; // jaw opening implied by the measured lip gap
    if (b.geoMouth !== undefined) {
      const restGap = calibration.blend?.geoMouth ?? (firstSeen.geoMouth ??= b.geoMouth);
      const t = clamp((b.geoMouth - restGap - LIP_GATE[0]) / (LIP_GATE[1] - LIP_GATE[0]), 0, 1);
      lipsApart = t * t * (3 - 2 * t);
      const gap = lipGapSteady.follow(b.geoMouth - restGap);
      const deadZone = tuning.jaw?.lipDeadZone ?? JAW_FROM_LIPS.deadZone;
      jawFromLips = clamp((gap - deadZone) * (tuning.jaw?.fromLips ?? JAW_FROM_LIPS.gain), 0, 1);
    }
    const jawFromScore = clamp((jawSteady.follow(b.jawOpen ?? 0) - calibration.jaw - 0.03) * 1.8, 0, 1) * lipsApart;
    const jaw = Math.max(jawFromScore, jawFromLips);

    // The operator's own left/right. Brows: raised is positive, frowning negative.
    const brow = (side: 'Left' | 'Right') =>
      clamp(Math.max(b.browInnerUp ?? 0, b[`browOuterUp${side}`] ?? 0) - (b[`browDown${side}`] ?? 0), -1, 1);
    const smile = (side: 'Left' | 'Right') => clamp((b[`mouthSmile${side}`] ?? 0) * 1.3, 0, 1);
    // Mirrored, your left side drives the avatar's right side (the one on your left on screen).
    const [avatarLeft, avatarRight] = mirror ? (['Right', 'Left'] as const) : (['Left', 'Right'] as const);

    face.yaw = filters.yaw.filter(yaw, dt);
    face.pitch = filters.pitch.filter(pitch, dt);
    face.roll = filters.roll.filter(roll, dt);
    face.jaw = filters.jaw.filter(jaw, dt);
    face.browLeft = filters.browLeft.filter(brow(avatarLeft), dt);
    face.browRight = filters.browRight.filter(brow(avatarRight), dt);
    face.smileLeft = filters.smileLeft.filter(smile(avatarLeft), dt);
    face.smileRight = filters.smileRight.filter(smile(avatarRight), dt);

    // Expression shapes: each of the model's shapes follows the tracker signal of the same name.
    // Mirrored, the operator's left drives the avatar's right (as in a mirror).
    const relaxed = calibration.blend ?? {};
    /** How much a landmark measurement has shrunk from the relaxed face (fraction, 0 = relaxed). */
    const shrink = (key: string) => {
      const now = b[key];
      if (now === undefined) return 0;
      const base = relaxed[key] ?? (firstSeen[key] ??= now);
      return Math.max(0, (base - now) / base - LANDMARK_DEAD_ZONE);
    };
    const signal = (name: string) => {
      const source = mirror ? swapSides(name) : name;
      const side = source.endsWith('Left') ? 'Left' : 'Right';
      let value = Math.max(0, (steadied(name, b[source] ?? 0) - (relaxed[source] ?? 0) - SHAPE_DEAD_ZONE) * gainOf(name));
      if (gainOf(name) > 0) {
        if (name.startsWith('noseSneer')) value += shrink(`geoNose${side}`) * LANDMARK_GAIN.noseScrunch;
        if (name.startsWith('browDown')) value += shrink(`geoBrow${side}`) * LANDMARK_GAIN.browFrown;
      }
      return clamp(value, 0, limitOf(name));
    };
    for (const [name, filter] of shapeFilters) {
      shapes[name] = sample.direct ? (sample.direct[name] ?? 0) : clamp(filter.filter(signal(name), dt), 0, 1);
    }
    face.jawSide = jawFilters.side.filter(signal('jawLeft') - signal('jawRight'), dt);
    face.jawForward = jawFilters.forward.filter(signal('jawForward'), dt);
  }

  function consumeBody(sample: BodySample, dt: number, now: number) {
    // MediaPipe: x = image right, y = down, z = away from camera.
    // Avatar:    x = screen right, y = up, z = toward the viewer.
    const map = (v: Vec3): Vec3 => [mirror ? -v[0] : v[0], -v[1], -v[2]];
    const feed = (filter: ArmFilter, arm: ArmSample | null) => {
      if (arm) filter.update(map(arm.upper), map(arm.lower), dt, now);
    };
    // The body tracker sometimes invents a wrist inside the picture for an arm that is out of
    // sight. When the hand tracker is running, an arm only counts while it has found a real hand
    // at that wrist.
    const real = (side: 'left' | 'right') =>
      demo || !handTracker || now - handAtWrist[side] < HAND_CONFIRMS_ARM_MS ? sample[side] : null;
    feed(armFilters.left, mirror ? real('right') : real('left'));
    feed(armFilters.right, mirror ? real('left') : real('right'));
    if (sample.wrists) wrists = sample.wrists;
    if (sample.torso) torsoFilter.update(sample.torso, mirror, dt, now);
  }

  /** `left` / `right` are the operator's own hands. */
  function consumeHands(left: HandShape | null, right: HandShape | null, dt: number, now: number) {
    const map = (v: Vec3): Vec3 => [mirror ? -v[0] : v[0], -v[1], -v[2]];
    const forLeft = mirror ? right : left;
    const forRight = mirror ? left : right;
    if (forLeft) handFilters.left.update(forLeft, map, dt, now);
    if (forRight) handFilters.right.update(forRight, map, dt, now);
  }

  /** Decides which of the operator's hands each detection is, preferring the body tracker's wrists. */
  function sortHands(found: HandSample[], now: number) {
    const out: { left: HandSample | null; right: HandSample | null } = { left: null, right: null };
    const gap = (hand: HandSample, wrist: [number, number] | null) =>
      wrist ? Math.hypot(hand.wrist[0] - wrist[0], hand.wrist[1] - wrist[1]) : Infinity;
    for (const hand of found) {
      const dl = gap(hand, wrists.left);
      const dr = gap(hand, wrists.right);
      let side: 'left' | 'right';
      if (dl < HAND_AT_WRIST) handAtWrist.left = now;
      if (dr < HAND_AT_WRIST) handAtWrist.right = now;
      if (Math.min(dl, dr) < 0.25) side = dl <= dr ? 'left' : 'right';
      // MediaPipe's label assumes a mirrored picture; the webcam frame is not mirrored, so swap it.
      else side = hand.label === 'Left' ? 'right' : 'left';
      const other = side === 'left' ? 'right' : 'left';
      if (!out[side]) out[side] = hand;
      else if (!out[other]) out[other] = hand;
    }
    return out;
  }

  function frame(now: number) {
    const dt = Math.min(0.1, (now - lastFrame) / 1000);
    lastFrame = now;
    if (dt > 0) fps = ease(fps, 1 / dt, dt, 0.5);

    if (demo) {
      consumeFace(demoFace(now / 1000), dt);
      if (bodyOn) consumeBody(demoBody(now / 1000), dt, now);
      if (handsOn) consumeHands(demoHand(now / 1000), null, dt, now);
      lastSeen = now;
    } else if (tracker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      // Only run inference when the camera has delivered a new frame.
      lastVideoTime = video.currentTime;
      frameCount++;
      const t0 = performance.now();
      const sample = tracker.detect(video, now);
      inferMs = ease(inferMs, performance.now() - t0, dt, 0.5);
      const detectDt = (now - lastDetect) / 1000;
      lastDetect = now;
      if (detectDt > 0) trackFps = ease(trackFps, 1 / detectDt, detectDt, 0.5);
      if (sample) {
        consumeFace(sample, detectDt);
        lastSeen = now;
        if (liveHead?.wanted && sample.head) liveHead.update(video, sample.head, mirror, now);
      }
      // Arms change more slowly than the face: run the body model on every other frame.
      if (bodyTracker && bodyOn && frameCount % 2 === 0) {
        const t1 = performance.now();
        const body = bodyTracker.detect(video, now);
        bodyMs = ease(bodyMs, performance.now() - t1, dt, 0.5);
        if (body) consumeBody(body, (now - lastBodyDetect) / 1000, now);
        lastBodyDetect = now;
      }
      // Hands take the frames the body model skips, so the two never run together.
      // They also run with fingers switched off, because a found hand is what confirms an arm.
      if (handTracker && (handsOn || bodyOn) && frameCount % 2 === 1) {
        const t2 = performance.now();
        const found = sortHands(handTracker.detect(video, now), now);
        handMs = ease(handMs, performance.now() - t2, dt, 0.5);
        if (handsOn) consumeHands(found.left, found.right, (now - lastHandDetect) / 1000, now);
        lastHandDetect = now;
      }
    }

    // Tracking loss: fade the whole pose back to neutral instead of freezing on the last frame.
    const tracked = now - lastSeen < LOST_AFTER_MS;
    presence = ease(presence, tracked ? 1 : 0, dt, tracked ? 0.15 : 0.35);
    liveHead?.setFound(tracked);
    // An arm must be seen steadily for a moment before it is followed, so a joint flickering at
    // the picture's edge does not make the arm jump. A hand is only followed on a followed arm.
    const armPose = (f: ArmFilter) =>
      bodyOn && now - f.lastSeen < ARM_LOST_AFTER_MS && now - f.since >= ARM_ACQUIRE_MS ? f.value : null;
    const handPose = (f: HandFilter, arm: ArmFilter) =>
      handsOn && (!bodyOn || armPose(arm)) && now - f.lastSeen < HAND_LOST_AFTER_MS ? f.value : null;
    const torso = !bodyOn ? null : demo ? demoTorso(now / 1000) : now - torsoFilter.lastSeen < ARM_LOST_AFTER_MS ? torsoFilter.value : null;

    const pose: RigPose = {
      yaw: face.yaw * presence,
      pitch: face.pitch * presence,
      roll: face.roll * presence,
      jaw: face.jaw * presence,
      browLeft: face.browLeft * presence,
      browRight: face.browRight * presence,
      smileLeft: face.smileLeft * presence,
      smileRight: face.smileRight * presence,
      shapes: Object.fromEntries(Object.entries(shapes).map(([name, v]) => [name, v * presence])),
      jawSide: face.jawSide * presence,
      jawForward: face.jawForward * presence,
      leftArm: armPose(armFilters.left),
      rightArm: armPose(armFilters.right),
      leftHand: handPose(handFilters.left, armFilters.left),
      rightHand: handPose(handFilters.right, armFilters.right),
      torso,
    };
    avatar.setPose(pose, dt);
    avatar.render();

    const deg = (r: number) => MathUtils.radToDeg(r).toFixed(0).padStart(4);
    const armState = (p: ArmPose | HandPose | null) => (p ? 'tracked' : 'relaxed');
    const lines = [
      `${characterId}${demo ? '  [demo motion, no webcam]' : ''}`,
      `render   ${fps.toFixed(0).padStart(3)} fps`,
    ];
    if (!demo) {
      lines.push(
        `tracking ${trackFps.toFixed(0).padStart(3)} fps   face ${inferMs.toFixed(1)} ms (${tracker?.delegate})` +
          (bodyTracker && bodyOn ? `   body ${bodyMs.toFixed(1)} ms` : '') +
          (handTracker && handsOn ? `   hands ${handMs.toFixed(1)} ms` : ''),
        `camera   ${camera?.label ?? ''}`,
      );
    }
    lines.push(
      `tracking ${tracked ? 'FACE FOUND' : 'FACE LOST - returning to neutral'}`,
      `arms     ${!bodyOn ? 'off' : `left ${armState(pose.leftArm)}, right ${armState(pose.rightArm)}`}` +
        (!demo && !bodyTracker ? '  (body tracker failed to load)' : ''),
      `torso    ${
        !pose.torso
          ? bodyOn ? 'not tracked - upright' : 'off'
          : `lean ${deg(pose.torso.lean)}  bend ${deg(pose.torso.bend)}  twist ${deg(pose.torso.twist)}  shrug L ${pose.torso.shrugLeft.toFixed(2)} R ${pose.torso.shrugRight.toFixed(2)}`
      }`,
      `hands    ${!avatar.hasFingers ? 'no finger bones in this model' : !handsOn ? 'off' : `left ${armState(pose.leftHand)}, right ${armState(pose.rightHand)}`}` +
        (!demo && avatar.hasFingers && !handTracker ? '  (hand tracker failed to load)' : ''),
      ...(avatar.softParts ? [`soft     ${avatar.softParts} (simulated)`] : []),
      ...(liveHead ? [`head     ${liveHead.status()}`] : []),
      `face     ${
        !avatar.shapeNames.length
          ? 'no expression shapes in this model'
          : Object.entries(shapes)
              .filter(([, v]) => v > 0.15)
              .sort((x, y) => y[1] - x[1])
              .slice(0, 4)
              .map(([name, v]) => `${name} ${v.toFixed(1)}`)
              .join('  ') || 'relaxed'
      }`,
      `yaw ${deg(face.yaw)}  pitch ${deg(face.pitch)}  roll ${deg(face.roll)}  jaw ${face.jaw.toFixed(2)}`,
      '',
      `[C] calibrate (sit upright, relaxed face and shoulders)   [M] mirror ${mirror ? 'on' : 'off'}   [B] body ${bodyOn ? 'on' : 'off'}   [F] fingers ${handsOn ? 'on' : 'off'}`,
      `[S] show bones   [K] next camera   [H] hide UI${characters.length > 1 ? '   [P] change character' : ''}${liveHead ? '   [V] helmet on / off' : ''}   drag to orbit, scroll to zoom`,
    );
    overlayEl.textContent = lines.join('\n');

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  (window as unknown as { __ready: boolean }).__ready = true;
}

main().catch((err: unknown) => {
  console.error(err);
  const e = err as Error;
  const denied = e.name === 'NotAllowedError' || e.name === 'NotFoundError';
  showMessage(
    denied
      ? 'No webcam available or permission was denied.\nAllow camera access and reload, or add ?demo to the address to see the avatar move without a camera.'
      : `Could not start: ${e.message}`,
  );
});
