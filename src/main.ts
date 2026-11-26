import './style.css';
import { MathUtils } from 'three';
import { createBodyTracker, type ArmSample, type BodySample, type BodyTracker, type Vec3 } from './body';
import { startCamera, type CameraSession } from './camera';
import { characterUrl, loadCharacters, showPicker, type CharacterInfo } from './characters';
import { FV_SOUND, createFvListener, type FvListener, type FvSound } from './fricative';
import { createHandTracker, type HandSample, type HandTracker } from './hands';
import type { WrinkleConfig } from './wrinkles';
import type { StreamSetup } from './lighting';
import { createRigAvatar, type ArmPose, type HandPose, type RigPose } from './rigAvatar';
import { OneEuro, Steady, clamp, ease } from './smoothing';
import { streamConfig, type StreamConfig } from './stream';
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

/** Dev server only: adds a line to the test log (see debugLog in vite.config.ts). */
function report(text: string) {
  if (!import.meta.env.DEV) return;
  void fetch('/api/debuglog', { method: 'POST', body: `${new Date().toISOString().slice(11, 23)} # ${text}` }).catch(() => {});
}
if (import.meta.env.DEV) {
  report(`page opened: ${location.search}`);
  window.addEventListener('error', (e) => report(`ERROR ${e.message} at ${e.filename}:${e.lineno}`));
  window.addEventListener('unhandledrejection', (e) => report(`ERROR (promise) ${String(e.reason)}`));
  document.addEventListener('visibilitychange', () => report(`tab ${document.visibilityState}`));
}

const CALIBRATION_KEY = 'mocap.calibration';
const LOST_AFTER_MS = 250;
const ARM_LOST_AFTER_MS = 400;
/** An arm has to be seen for this long without a break before the avatar follows it. */
const ARM_ACQUIRE_MS = 100;
/** A hand counts as being at the body tracker's wrist within this distance (fraction of the picture). */
const HAND_AT_WRIST = 0.15;
/** An arm stays confirmed this long after its hand was last found (fists and fast moves drop out briefly). */
const HAND_CONFIRMS_ARM_MS = 800;
const HAND_LOST_AFTER_MS = 350;
/** A hand's measured size is used for this long after it was last seen. */
const HAND_SIZE_LASTS_MS = 500;
/** Assumed horizontal field of view of the webcam, for turning sizes in the picture into distances. */
const WEBCAM_FOV = (60 * Math.PI) / 180;
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
 * The "f / v" trigger: lower lip tucked under the upper teeth. It fires only while BOTH are true:
 * the face tracker sees the lower lip pushed up against the top teeth, and the microphone hears
 * an "f" hiss or a "v" voice (see fricative.ts). While it fires, the model's mouthRollLower shape
 * (lower lip rolled in) is driven directly by the trigger, and its mouthShrugLower shape (lower
 * lip pushed up and forward) is faded out so the two do not fight.
 *
 * The lip half is "the lips are together" (the measured gap between the inner lips, geoMouth),
 * not a lip-roll score. Measured on the operator's face (bearded, 24 November, four logged
 * tests): no tracker score separates an "f" from a resting closed mouth. mouthRollLower reads
 * 0.03 during an "f" and 0.15 at rest (the wrong way round), and mouthShrugLower reads about 0.5
 * for both. What does hold is that the lips stay together through an "f" or "v" (gap about
 * 0.013) while an "s" or a vowel parts them, so lips together + the sound is the test.
 *   strength: how far the roll shape goes while the trigger is fully on, 0..1 (0 = trigger off,
 *             microphone not opened)
 *   lips:     lip gap as a fraction of the outer-eye-corner distance: [fully on at or below,
 *             off at or above]
 *   voiced:   how far a voice alone (no hiss) opens the trigger, 0..1; this is what lets "v"
 *             through (and "m" / "b", which are also voiced with the lips together)
 *   release:  seconds the trigger stays on after the sound stops (the picture lags the sound)
 * Overridden by "fv" in the character's face.json, which may also carry the microphone levels
 * "hiss", "tilt" and "voice" (each [from, to] in decibels, see FV_SOUND in fricative.ts).
 */
const FV = { strength: 1, jaw: 0.5, lips: [0.022, 0.035] as readonly [number, number], voiced: 0.7, release: 0.25 };
/**
 * The model shape the trigger drives: mouthFV when the model has it (lower lip up and rolled back
 * over the lower teeth, upper lip lifted; made for this and driven by nothing else), otherwise
 * the plain lower-lip roll. With mouthFV the jaw also opens by FV.jaw (0..1 of full travel) while
 * the trigger fires, so the upper teeth come to rest on the raised lip instead of the mouth
 * sealing shut. FV_LIP is the model shape faded out while it fires (it pushes the lip forward).
 */
const FV_SHAPES = ['mouthFV', 'mouthRollLower'];
const FV_LIP = 'mouthShrugLower';
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
  landmarks?: { noseScrunch?: number; browFrown?: number; lipRaise?: number };
  /** One-sided mouth expressions (see SIDES). */
  sides?: { strength?: number; deadZone?: number; full?: number };
  /** Jaw opening taken from the measured lip gap (see JAW_FROM_LIPS). */
  jaw?: { fromLips?: number; lipDeadZone?: number };
  /** Blinking from the measured gap between the eyelids (see BLINK). */
  blink?: { lids?: [number, number]; shutAt?: number; hold?: number; together?: number };
  /** Expression wrinkles drawn in the skin's shader (see wrinkles.ts). */
  wrinkles?: WrinkleConfig;
  /** The "f / v" trigger (see FV). */
  fv?: {
    strength?: number;
    jaw?: number;
    lips?: [number, number];
    voiced?: number;
    release?: number;
    hiss?: [number, number];
    tilt?: [number, number];
    voice?: [number, number];
  };
  /** Arms: maxRaise = how far above shoulder height the upper arm may point, degrees (0 = never above the shoulder). Unset = no limit. */
  arms?: { maxRaise?: number };
  /** Stream mode, ?stream in the address (see stream.ts). */
  stream?: Partial<StreamConfig>;
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
const landmarkGain = () => ({
  noseScrunch: tuning.landmarks?.noseScrunch ?? 10,
  browFrown: tuning.landmarks?.browFrown ?? 8,
  lipRaise: tuning.landmarks?.lipRaise ?? 6,
});
const LANDMARK_DEAD_ZONE = 0.02;
/**
 * Blinking. The tracker's eyeBlink score alone gave no clear blink: it often stops well short of 1
 * on a real blink, and a blink is over within two or three camera frames. So the gap between the
 * eyelids is also measured from the face landmarks (geoEyeLeft / geoEyeRight in tracker.ts) and
 * compared with that eye's usual open gap, which is learned while the app runs (no calibration
 * needed: it follows a wider eye quickly and a narrower one over about 15 seconds, so looking down
 * for a while stops counting as closed). The eye takes whichever asks for more, score or lids.
 *   lids:     how far the lid gap has shrunk, as a fraction of the open gap: [starts closing, fully shut].
 *             Lower both if blinks are still missed; raise the first if smiling or squinting shuts the eyes.
 *   shutAt:   at or above this (0..1) the eye snaps fully shut ...
 *   hold:     ... and stays shut for at least this many seconds from that moment (the shortest blink), so a
 *             very quick blink is still seen. It is never added on after your eye has reopened.
 *   together: when both eyes are above this they close together by the larger amount (a wink stays one-sided).
 * Overridden by "blink" in the character's face.json; lids [1, 1] = lid measurement off.
 */
const BLINK = { lids: [0.3, 0.55] as [number, number], shutAt: 0.75, hold: 0.05, together: 0.35 };
/** Seconds for the learned open-eye gap to come down to a narrower eye. */
const EYE_OPEN_FORGET = 15;
/**
 * One-sided mouth expressions. The tracker's Left / Right mouth scores rise and fall together
 * even when only one side of the face moves, so a one-sided snarl or smile came out on both sides
 * of the avatar. Each side of the mouth is therefore also measured from the face landmarks (see
 * tracker.ts), and when one side has moved more than the other, the quieter side's shape is
 * turned down. An even expression is left exactly as it was.
 *   strength: how far the quieter side is turned down, 0..1 (0 = off, sides as the tracker reports)
 *   deadZone: difference between the sides (fraction of the relaxed measurement) that still counts as even
 *   full:     difference at which the quieter side is fully turned down
 * Overridden by "sides" in the character's face.json.
 */
const SIDES = { strength: 1, deadZone: 0.015, full: 0.06 };
/** Shape name (start) -> the landmark measurement that tells its sides apart, and which way it moves when the shape is made. */
const SIDE_MEASURE: [string, string, 1 | -1][] = [
  ['mouthUpperUp', 'geoLipUp', -1],
  ['mouthSmile', 'geoCorner', -1],
  ['cheekSquint', 'geoCorner', -1],
  ['mouthFrown', 'geoCorner', 1],
  ['mouthStretch', 'geoCornerOut', 1],
  ['mouthDimple', 'geoCornerOut', 1],
  ['mouthLowerDown', 'geoLipDown', 1],
];
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

/** A person's arm, shoulder to wrist, is about this many shoulder widths long. */
const ARM_PER_SHOULDER = 1.45;

/**
 * Upper arm and forearm directions that put the avatar's hand where the operator's is. A hand
 * position leaves the elbow free to swing round the line from shoulder to hand; it is put at the
 * point of that circle nearest to where the elbow sits in the resting pose.
 *   reach  where the hand should be from the middle of the avatar's shoulders, avatar axes, in its shoulder widths
 *   side   1 for the avatar's left arm (+x), -1 for its right
 *   armPerShoulder  the avatar's arm length in its own shoulder widths
 *   restUpper  the resting pose's upper arm direction for this side (where the elbow is at rest)
 *   gain   above 1 makes the avatar reach further than the operator does
 *   inward below 1 squeezes the hand's sideways position toward the middle of the chest
 *   limit  keeps the upper arm within what the model can do; the forearm then makes up the difference
 * The hand is placed in shoulder widths from the middle of the chest, so on a broad avatar it
 * lands in the same place against its body as the operator's hand does against theirs.
 */
function armForReach(
  reach: Vec3,
  side: 1 | -1,
  armPerShoulder: number,
  restUpper: Vec3,
  gain = 1,
  inward = 1,
  limit: (upper: Vec3) => Vec3 = (v) => v,
): ArmPose {
  // From the avatar's own shoulder (half a shoulder width out from the middle), in shoulder widths.
  const from: Vec3 = [reach[0] * gain * inward - 0.5 * side, reach[1] * gain, reach[2] * gain];
  const length = Math.hypot(...from);
  const dir: Vec3 = length > 1e-4 ? [from[0] / length, from[1] / length, from[2] / length] : [0, -1, 0];
  // How far the hand is from the shoulder, as a fraction of the straight arm. Both arm bones count as half.
  // Never closer than a third of the arm: with the hand almost on the shoulder the elbow has no steady place to be.
  const f = clamp(length / armPerShoulder, 0.35, 0.99);
  // The elbow sits off the shoulder-to-hand line, on the side the resting elbow is.
  const pole = restUpper;
  const along = pole[0] * dir[0] + pole[1] * dir[1] + pole[2] * dir[2];
  let perp: Vec3 = [pole[0] - dir[0] * along, pole[1] - dir[1] * along, pole[2] - dir[2] * along];
  const perpLength = Math.hypot(...perp);
  perp = perpLength > 1e-3 ? [perp[0] / perpLength, perp[1] / perpLength, perp[2] / perpLength] : [side, 0, 0];
  const out = Math.sqrt(Math.max(0, 0.25 - (f * f) / 4));
  const elbow: Vec3 = [dir[0] * (f / 2) + perp[0] * out, dir[1] * (f / 2) + perp[1] * out, dir[2] * (f / 2) + perp[2] * out];
  const upper = limit([elbow[0] * 2, elbow[1] * 2, elbow[2] * 2]);
  // From wherever the elbow ended up, the forearm still points at the hand.
  const lower: Vec3 = [dir[0] * f - upper[0] / 2, dir[1] * f - upper[1] / 2, dir[2] * f - upper[2] / 2];
  const lowerLength = Math.hypot(...lower) || 1;
  return { upper, lower: [lower[0] / lowerLength, lower[1] / lowerLength, lower[2] / lowerLength] };
}

/**
 * Smooths the six numbers of one arm's two direction vectors. The first number is how hard it
 * smooths a still arm (higher = less smoothing, more jitter); the second is how quickly the
 * smoothing lets go once the arm moves (higher = less lag on a moving arm).
 */
class ArmFilter {
  private f = Array.from({ length: 6 }, () => new OneEuro(1.5, 0.8));
  // Stream mode steadies the hand's position before the elbow is worked out from it: a small
  // wobble in where the hand is becomes a large swing of the elbow, so it has to go first.
  // Sideways and up / down come from the picture; depth is the tracker's guess and gets a wide
  // still band and slow smoothing.
  private hand = [new OneEuro(1.2, 1.0), new OneEuro(1.2, 1.0), new OneEuro(0.8, 0.6)];
  private still = [new Steady(0.02), new Steady(0.02), new Steady(0.03)];

  /** The hand's position (across and up in shoulder widths, then the share of the way to the lens) with the tracker's wobble taken out. */
  place(reach: Vec3, dt: number, now: number): Vec3 {
    if (now - this.lastSeen > ARM_LOST_AFTER_MS) {
      this.hand = [new OneEuro(1.2, 1.0), new OneEuro(1.2, 1.0), new OneEuro(0.8, 0.6)];
      this.still = [new Steady(0.02), new Steady(0.02), new Steady(0.03)];
    }
    return reach.map((v, i) => this.hand[i].filter(this.still[i].follow(v), dt)) as Vec3;
  }
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
  // ?stream: the webcam-on-the-monitor view on a green screen, for OBS. The picture is a fixed
  // square in the window's top left corner (1024 pixels; ?size= changes it), whatever the window's size.
  // ?key=alpha leaves the background see-through instead of green (used by stream-preview.html).
  let streamSetup: StreamSetup | undefined;
  if (params.has('stream')) {
    const config = streamConfig(tuning.stream);
    const size = clamp(Number(params.get('size')) || 1024, 128, 4096);
    const transparent = params.get('key') === 'alpha';
    streamSetup = { config, transparent, seatUrl: character.seat && characterUrl(character, character.seat) };
    document.body.classList.add('stream');
    document.documentElement.style.background = document.body.style.background = transparent ? 'transparent' : config.green;
    Object.assign(canvas.style, { inset: '0 auto auto 0', width: `${size}px`, height: `${size}px` });
  }
  const avatar = await createRigAvatar(canvas, character, (fraction) => {
    overlayEl.textContent = `Loading ${character.name}... ${Math.round(fraction * 100)}%`;
    }, tuning.wrinkles,
    streamSetup,
  );

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
  // The "f / v" trigger needs the microphone; only asked for when this model has the shape.
  const fv = { ...FV, ...tuning.fv };
  let fvListener: FvListener | null = null;
  let fvProblem = '';
  const FV_SHAPE = FV_SHAPES.find((name) => avatar.shapeNames.includes(name)) ?? '';
  const fvWanted = fv.strength > 0 && FV_SHAPE !== '';
  /** How far the jaw opens while the trigger fires (only with the purpose-made shape). */
  const fvJaw = FV_SHAPE === FV_SHAPES[0] ? fv.jaw : 0;
  if (fvWanted && !demo) {
    // Not waited for: the avatar starts moving at once and the trigger joins in when the
    // microphone has been allowed.
    report('microphone: asking');
    createFvListener({
      hiss: tuning.fv?.hiss ?? FV_SOUND.hiss,
      tilt: tuning.fv?.tilt ?? FV_SOUND.tilt,
      voice: tuning.fv?.voice ?? FV_SOUND.voice,
    }).then(
      (listener) => {
        fvListener = listener;
        report(`microphone: open (${listener.label})`);
      },
      (err: unknown) => {
        console.warn('Microphone unavailable, the f / v trigger is off:', err);
        fvProblem = 'no microphone (permission denied or none found)';
        report(`microphone: failed (${String(err)})`);
      },
    );
  } else {
    report(`f / v trigger not started: demo ${demo}, strength ${fv.strength}, model shape: ${FV_SHAPE || 'none'}`);
  }

  // --- state -------------------------------------------------------------
  let calibration = loadCalibration();
  let mirror = true;
  let bodyOn = params.get('body') !== 'off'; // ?body=off starts with arm and torso tracking off (same as pressing B)
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
  /** One-sided mouth expressions: settings, and the last measured lean per measurement (+ = the operator's left). */
  const sides = { ...SIDES, ...tuning.sides };
  const sideLean: Record<string, number> = {};
  /** Blinking: settings, each eye's learned open lid gap (the operator's own side), and the read-out numbers. */
  const blink = { ...BLINK, ...tuning.blink };
  const eyeOpen: Record<string, number> = {};
  const eyeShut: Record<string, number> = {};
  const blinkShutUntil: Record<string, number> = {};
  const blinkOver: Record<string, boolean> = {};
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
  /** How large each hand was last drawn (the operator's own left / right), for judging how far forward it is. */
  const handSize: Record<'left' | 'right', { perMetre: number; at: number } | null> = { left: null, right: null };
  const torsoFilter = new TorsoFilter();
  /** The f / v trigger: the last microphone reading, the held sound level, the lip level and the two together. */
  let fvSound: FvSound | null = null;
  let fvHeard = 0;
  let fvLip = 0;
  let fvLevel = 0;
  const fvLog: string[] = [];
  let fvLogged = 0;

  /** Takes the face and posture seen right now as upright and relaxed. */
  const calibrate = () => {
    if (!lastRaw) return;
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
  };
  /** How long after C the calibration is taken a second time, once the hand is back from the keyboard. */
  const CALIBRATE_AGAIN_MS = 2000;
  let calibrateAgain = 0;

  window.addEventListener('keydown', (e) => {
    const key = e.key.toLowerCase();
    if (key === 'c') report(`C pressed (face ${lastRaw ? 'seen' : 'not seen yet'})`);
    if (key === 'c') {
      // Now, and again in two seconds: reaching for the key shifts the shoulders, so the first
      // reading alone would be of the wrong posture.
      calibrate();
      clearTimeout(calibrateAgain);
      calibrateAgain = window.setTimeout(calibrate, CALIBRATE_AGAIN_MS);
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
    // ?demo&arm=ux,uy,uz,lx,ly,lz holds the operator's left arm in one pose, for checking the rig and the
    // framing: upper arm and forearm directions as the tracker gives them (x = picture right, y = down, z = away from the camera).
    // ?demo&reach=x,y,z does the same by hand position: the left wrist from the middle of the shoulders, in metres.
    const fixedReach = params.get('reach')?.split(',').map(Number);
    if (fixedReach?.length === 3 && fixedReach.every(Number.isFinite)) {
      return { left: { upper: [0, 1, 0], lower: [0, 1, 0], elbowSeen: false, reach: fixedReach.map((v) => v / 0.38) as Vec3 }, right: null };
    }
    const fixedArm = params.get('arm')?.split(',').map(Number);
    if (fixedArm?.length === 6 && fixedArm.every(Number.isFinite)) {
      return { left: { upper: norm(fixedArm.slice(0, 3) as Vec3), lower: norm(fixedArm.slice(3) as Vec3) }, right: null };
    }
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
    // ?demo&turn=degrees turns the demo hand about its own length, for checking how the forearm's skin follows.
    const turn = MathUtils.degToRad(Number(params.get('turn')) || 0);
    const flat = norm([forward[1], -forward[0], 0]);
    const out: Vec3 = [forward[1] * flat[2] - forward[2] * flat[1], forward[2] * flat[0] - forward[0] * flat[2], forward[0] * flat[1] - forward[1] * flat[0]];
    const side = norm([0, 1, 2].map((k) => flat[k] * Math.cos(turn) + out[k] * Math.sin(turn)) as Vec3);
    const normal: Vec3 = [
      forward[1] * side[2] - forward[2] * side[1],
      forward[2] * side[0] - forward[0] * side[2],
      forward[0] * side[1] - forward[1] * side[0],
    ];
    // ?demo&curl=0..1 holds the fingers that far closed (1 = a person's fist); &point leaves the index finger straight.
    const asked = params.get('curl');
    const curl = asked !== null ? Number(asked) || 0 : fixedPose ? 0.5 : 0.5 + 0.5 * Math.sin(t * 2.6);
    const bent = (angle: number): Vec3 =>
      norm([0, 1, 2].map((k) => forward[k] * Math.cos(angle) + normal[k] * Math.sin(angle)) as Vec3);
    return {
      forward,
      side,
      fingers: Array.from({ length: 5 }, (_, i) => {
        if (i === 1 && params.has('point')) return { prox: bent(0.05), dist: bent(0.1) };
        if (i > 0) return { prox: bent(curl * 1.35), dist: bent(curl * 3.3) };
        // A thumb that starts out to the index side and folds across the palm as the hand closes.
        const mixed = (f: number, sd: number, n: number): Vec3 => norm([0, 1, 2].map((k) => forward[k] * f + side[k] * sd + normal[k] * n) as Vec3);
        return { prox: mixed(1 - 0.6 * curl, 0.7 - 0.9 * curl, 0.6 * curl), dist: mixed(0.6 - 0.3 * curl, 0.6 - 1.4 * curl, 0.3 * curl) };
      }),
    };
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
    // fvLevel is from the previous frame here, which is close enough: it is held and smoothed.
    const jaw = Math.max(jawFromScore, jawFromLips, fvWanted ? fvLevel * fvJaw : 0);

    // The operator's own left/right. Brows: raised is positive, frowning negative.
    const brow = (side: 'Left' | 'Right') =>
      clamp(Math.max(b.browInnerUp ?? 0, b[`browOuterUp${side}`] ?? 0) - (b[`browDown${side}`] ?? 0), -1, 1);
    /** How far a landmark measurement has moved from the relaxed face (signed fraction). */
    const moved = (key: string) => {
      const now = b[key];
      if (now === undefined) return 0;
      const base = calibration.blend?.[key] ?? (firstSeen[key] ??= now);
      return base > 0 ? (now - base) / base : 0;
    };
    /**
     * 1 = leave this shape alone, toward 0 = the other side of the mouth is doing more of it.
     * `side` is the operator's own side.
     */
    const sideFactor = (name: string, side: 'Left' | 'Right') => {
      const measure = SIDE_MEASURE.find(([prefix]) => name.startsWith(prefix));
      if (!measure || sides.strength <= 0) return 1;
      const [, key, direction] = measure;
      // Positive = the operator's left side is making more of this shape than the right.
      const lean = direction * (moved(`${key}Left`) - moved(`${key}Right`));
      // For the read-out: in the sense of the first shape listed for this measurement.
      sideLean[key] = lean * direction * SIDE_MEASURE.find(([, k]) => k === key)![2];
      const against = side === 'Left' ? -lean : lean;
      const t = clamp((against - sides.deadZone) / Math.max(1e-4, sides.full - sides.deadZone), 0, 1);
      return 1 - clamp(sides.strength, 0, 1) * t * t * (3 - 2 * t);
    };
    const smile = (side: 'Left' | 'Right') => clamp((b[`mouthSmile${side}`] ?? 0) * 1.3, 0, 1) * sideFactor('mouthSmile', side);
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
    // The f / v trigger: the lip has to be rolled in AND the microphone has to hear the sound.
    const lipT = b.geoMouth === undefined ? 0 : clamp((fv.lips[1] - b.geoMouth) / (fv.lips[1] - fv.lips[0]), 0, 1);
    fvLip = lipT * lipT * (3 - 2 * lipT);
    fvLevel = fvHeard * fvLip;
    // Blinking from the eyelids: how far each lid gap has shrunk from that eye's usual open gap.
    for (const side of ['Left', 'Right'] as const) {
      const gap = b[`geoEye${side}`];
      if (gap === undefined) continue;
      const open = eyeOpen[side] ?? gap;
      eyeOpen[side] = gap > open ? ease(open, gap, dt, 0.3) : ease(open, gap, dt, EYE_OPEN_FORGET);
      const shrunk = eyeOpen[side] > 0 ? 1 - gap / eyeOpen[side] : 0;
      eyeShut[side] = clamp((shrunk - blink.lids[0]) / Math.max(1e-4, blink.lids[1] - blink.lids[0]), 0, 1);
    }
    const signal = (name: string) => {
      const source = mirror ? swapSides(name) : name;
      const side = source.endsWith('Left') ? 'Left' : 'Right';
      let value = Math.max(0, (steadied(name, b[source] ?? 0) - (relaxed[source] ?? 0) - SHAPE_DEAD_ZONE) * gainOf(name));
      if (fvWanted) {
        if (name === FV_SHAPE) value = Math.max(value, fvLevel * fv.strength);
        if (name === FV_LIP) value *= 1 - fvLevel;
      }
      if (gainOf(name) > 0) {
        if (name.startsWith('noseSneer')) value += shrink(`geoNose${side}`) * LANDMARK_GAIN.noseScrunch;
        if (name.startsWith('browDown')) value += shrink(`geoBrow${side}`) * LANDMARK_GAIN.browFrown;
        // A one-sided snarl barely moves the tracker's own score, so the lift of each side of
        // the upper lip is measured too.
        if (name.startsWith('mouthUpperUp')) value += shrink(`geoLipUp${side}`) * LANDMARK_GAIN.lipRaise;
        if (name.startsWith('eyeBlink')) value = Math.max(value, eyeShut[side] ?? 0);
      }
      if (/(Left|Right)$/.test(source)) value *= sideFactor(name, side);
      return clamp(value, 0, limitOf(name));
    };
    // Both eyes: a blink closes them together by the larger amount; a wink stays one-sided.
    const blinkNames = [...shapeFilters.keys()].filter((name) => name.startsWith('eyeBlink'));
    const blinkWanted = Object.fromEntries(blinkNames.map((name) => [name, signal(name)]));
    const blinkValues = Object.values(blinkWanted);
    if (blinkValues.length > 1 && Math.min(...blinkValues) >= blink.together) {
      for (const name of blinkNames) blinkWanted[name] = Math.max(...blinkValues);
    }
    const clock = performance.now();
    for (const [name, filter] of shapeFilters) {
      if (sample.direct) shapes[name] = sample.direct[name] ?? 0;
      else if (name in blinkWanted) {
        // Past `shutAt` the eye snaps fully shut and stays shut for `hold`, so a quick blink shows.
        // The hold is counted from the moment the eye shuts (a shortest blink length), never added
        // on after it: the eye reopens as soon as yours does.
        const over = blinkWanted[name] >= blink.shutAt;
        if (over && !blinkOver[name]) blinkShutUntil[name] = clock + blink.hold * 1000;
        blinkOver[name] = over;
        const shut = over || clock < (blinkShutUntil[name] ?? 0);
        shapes[name] = clamp(filter.filter(shut ? 1 : blinkWanted[name], dt), 0, 1);
        if (shut) shapes[name] = limitOf(name);
      } else shapes[name] = clamp(filter.filter(signal(name), dt), 0, 1);
    }
    face.jawSide = jawFilters.side.filter(signal('jawLeft') - signal('jawRight'), dt);
    face.jawForward = jawFilters.forward.filter(signal('jawForward'), dt);
  }

  function consumeBody(sample: BodySample, dt: number, now: number) {
    // MediaPipe: x = image right, y = down, z = away from camera.
    // Avatar:    x = screen right, y = up, z = toward the viewer.
    const map = (v: Vec3): Vec3 => [mirror ? -v[0] : v[0], -v[1], -v[2]];
    // A character whose shoulders deform badly with the arm overhead keeps its upper arm at or
    // below a set angle ("arms": { "maxRaise" } in face.json). The forearm is left free.
    const maxRaise = tuning.arms?.maxRaise;
    const lowered = (v: Vec3): Vec3 => {
      if (maxRaise === undefined) return v;
      const flat = Math.max(Math.hypot(v[0], v[2]), 0.2);
      const top = flat * Math.tan(MathUtils.degToRad(clamp(maxRaise, -80, 80)));
      if (v[1] <= top) return v;
      // Straight up has no sideways direction of its own: send it out to the arm's own side.
      const x = Math.hypot(v[0], v[2]) < 0.2 ? Math.sign(v[0] || 1) * 0.2 : v[0];
      const len = Math.hypot(x, top, v[2]);
      return [x / len, top / len, v[2] / len];
    };
    /**
     * How far a hand is held toward the webcam, as a share of the way from the shoulders to the lens
     * (0 = level with the shoulders), from how large it is drawn against them: nearer is larger.
     * null when the hand has not just been seen.
     */
    const forward = (own: 'left' | 'right'): number | null => {
      const hand = handSize[own];
      if (!hand || now - hand.at > HAND_SIZE_LASTS_MS || !sample.shoulders) return null;
      // Distance from the camera, hand against shoulders (under 1 = the hand is nearer).
      return 1 - clamp(sample.shoulders.perMetre / hand.perMetre, 0.3, 1.2);
    };
    /** The operator's shoulders' distance from the webcam, in shoulder widths, from how wide they are drawn. */
    const webcamAway = () => (sample.shoulders ? 1 / (2 * Math.tan(WEBCAM_FOV / 2) * sample.shoulders.across) : 2.2);
    const feed = (filter: ArmFilter, arm: ArmSample | null, side: 1 | -1, own: 'left' | 'right') => {
      if (!arm) return;
      if (streamSetup && arm.reach) {
        // Stream mode: put the hand where the operator's hand is, and work the elbow out from that.
        const { armReach, armInward, armDepth } = streamSetup.config;
        // Depth, as a share of the way to the lens: from the hand's size when there is one, the body
        // tracker's own guess otherwise. Never more than most of the way.
        const share = clamp((forward(own) ?? -arm.reach[2] / webcamAway()) * armDepth, -0.1, 0.7);
        // Steadied: across and up in shoulder widths (avatar axes), and that share.
        const across = map([arm.reach[0], arm.reach[1], 0]);
        const [x, y, toLens] = filter.place([across[0] * armInward, across[1], share], dt, now);
        // A hand moving toward the webcam stays where it is in the picture and grows. So the avatar's
        // hand moves along the line from where it would be beside the body to the avatar's own camera.
        const lens = avatar.cameraFrom ?? [0, 0.5, webcamAway()];
        const hand: Vec3 = [x + (lens[0] - x) * toLens, y + (lens[1] - y) * toLens, lens[2] * toLens];
        // Of all the elbow positions that reach the hand, take the one nearest the resting arm's.
        const rest = streamSetup.config.restArms.upper;
        const placed = armForReach(hand, side, avatar.armPerShoulder ?? ARM_PER_SHOULDER, [rest[0] * side, rest[1], rest[2]], armReach, 1, lowered);
        filter.update(placed.upper, placed.lower, dt, now);
      } else if (arm.elbowSeen !== false) filter.update(lowered(map(arm.upper)), map(arm.lower), dt, now);
    };
    // The body tracker sometimes invents a wrist inside the picture for an arm that is out of
    // sight. When the hand tracker is running, an arm only counts while it has found a real hand
    // at that wrist.
    const real = (side: 'left' | 'right') =>
      demo || !handTracker || now - handAtWrist[side] < HAND_CONFIRMS_ARM_MS ? sample[side] : null;
    feed(armFilters.left, mirror ? real('right') : real('left'), 1, mirror ? 'right' : 'left'); // the avatar's left is +x
    feed(armFilters.right, mirror ? real('left') : real('right'), -1, mirror ? 'left' : 'right');
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
      if (hand.perMetre && Math.min(dl, dr) < HAND_AT_WRIST) handSize[dl <= dr ? 'left' : 'right'] = { perMetre: hand.perMetre, at: now };
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

    // Microphone: rises at once with the sound, then lets go over `release` seconds.
    if (fvListener) {
      fvSound = fvListener.read(now);
      fvHeard = Math.max(fvSound.hiss, fvSound.voice * fv.voiced, fvHeard - dt / Math.max(0.01, fv.release));
    }
    // Dev server only: report the f / v numbers ten times a second so a test can be watched from outside.
    if (import.meta.env.DEV && fvWanted && !demo && now - fvLogged > 100) {
      fvLogged = now;
      const n = (v: number | undefined) => (v ?? 0).toFixed(2);
      fvLog.push(
        `${new Date().toISOString().slice(11, 23)} trig ${n(fvLevel)} lip ${n(fvLip)} sound ${n(fvHeard)} | lipGap ${(lastRaw?.blend.geoMouth ?? 0).toFixed(3)} ` +
          `lipGapRelaxed ${(calibration.blend?.geoMouth ?? 0).toFixed(3)} shape ${n(shapes[FV_SHAPE])} | hiss ${n(fvSound?.hiss)} voice ${n(fvSound?.voice)} ` +
          `rise ${n(fvSound?.rise)} tilt ${n(fvSound?.tilt)} voiceRise ${n(fvSound?.voiceRise)} | ` +
          `mic ${fvListener ? fvListener.label : fvProblem || 'none'} face ${now - lastSeen < LOST_AFTER_MS ? 'found' : 'lost'} | ` +
          // Every mouth, jaw and cheek signal the tracker reports, to find which ones mark an "f".
          `ALL ${JSON.stringify(
            Object.fromEntries(
              Object.entries(lastRaw?.blend ?? {})
                .filter(([name]) => /^(mouth|jaw|cheek|geo|nose|eyeBlink|eyeSquint)/.test(name))
                .map(([name, v]) => [name, Math.round(v * 1000) / 1000]),
            ),
          )}`,
      );
      if (fvLog.length >= 10) void fetch('/api/debuglog', { method: 'POST', body: fvLog.splice(0).join('\n') }).catch(() => {});
    }

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
      ...(['left', 'right'] as const).flatMap((side) => (avatar.handState()[side] ? [`${side === 'left' ? 'l' : 'r'} hand   ${avatar.handState()[side]}`] : [])),
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
      ...(!demo && lastRaw?.blend.geoEyeLeft !== undefined && avatar.shapeNames.some((name) => name.startsWith('eyeBlink'))
        ? [
            `blink    ${(['Left', 'Right'] as const)
              .map((side) => {
                const raw = lastRaw!.blend;
                const shrunk = eyeOpen[side] > 0 ? 1 - raw[`geoEye${side}`] / eyeOpen[side] : 0;
                return `${side.toLowerCase()} lids shrunk ${shrunk.toFixed(2)} score ${(raw[`eyeBlink${side}`] ?? 0).toFixed(2)}`;
              })
              .join('   ')}   (your own side; lids start at ${blink.lids[0]}, shut at ${blink.lids[1]})`,
          ]
        : []),
      ...(!demo && avatar.shapeNames.length && sides.strength > 0
        ? [
            `sides    ${[
              ['lip raise', 'geoLipUp'],
              ['corner up', 'geoCorner'],
              ['corner out', 'geoCornerOut'],
              ['lower lip', 'geoLipDown'],
            ]
              .map(([label, key]) => {
                const lean = sideLean[key] ?? 0;
                return `${label} ${Math.abs(lean) < sides.deadZone ? 'even ' : lean > 0 ? 'LEFT ' : 'RIGHT'} ${Math.abs(lean).toFixed(3)}`;
              })
              .join('   ')}   (your own side; acts from ${sides.deadZone}, full at ${sides.full})`,
          ]
        : []),
      ...(fvWanted && !demo
        ? [
            `f / v    ${
              !fvListener || !fvSound
                ? fvProblem || 'waiting for the microphone (click the page)'
                : `${fvLevel > 0.5 ? 'TRIGGERED' : 'off      '} ${fvLevel.toFixed(2)}   lips together ${fvLip.toFixed(2)}   sound ${fvHeard.toFixed(2)}` +
                  `   (hiss +${fvSound.rise.toFixed(0)} dB, tilt ${fvSound.tilt.toFixed(0)}, voice +${fvSound.voiceRise.toFixed(0)} dB)   ${fvListener.label}`
            }`,
          ]
        : []),
      `yaw ${deg(face.yaw)}  pitch ${deg(face.pitch)}  roll ${deg(face.roll)}  jaw ${face.jaw.toFixed(2)}`,
      '',
      `[C] calibrate (sit upright, relaxed face and shoulders)   [M] mirror ${mirror ? 'on' : 'off'}   [B] body ${bodyOn ? 'on' : 'off'}   [F] fingers ${handsOn ? 'on' : 'off'}`,
      `[S] show bones   [K] next camera   [H] hide UI${characters.length > 1 ? '   [P] change character' : ''}${liveHead ? '   [V] helmet on / off' : ''}   drag to orbit, scroll to zoom`,
    );
    overlayEl.textContent = lines.join('\n');

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  report('trackers loaded, avatar running');
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
