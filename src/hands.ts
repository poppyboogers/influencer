import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { Vec3 } from './body';
import { BASE } from './characters';

/** One finger as two unit directions: knuckle -> middle joint, middle joint -> tip. */
export interface FingerSample {
  prox: Vec3;
  dist: Vec3;
}

/** One detected hand. Directions are in MediaPipe space: x = image right, y = down, z = away from the camera. */
export interface HandSample {
  /** Wrist position in the image, 0..1, used to tell which arm the hand belongs to. */
  wrist: [number, number];
  /** MediaPipe's own guess. It assumes a mirrored image, so on a raw webcam frame it is swapped. */
  label: 'Left' | 'Right';
  /** Wrist -> middle knuckle. */
  forward: Vec3;
  /** Pinky knuckle -> index knuckle. */
  side: Vec3;
  /** Thumb, index, middle, ring, pinky. */
  fingers: FingerSample[];
  /**
   * How large the hand is drawn: picture widths per metre of hand. A hand nearer the camera is
   * drawn larger, which is the one dependable sign of how far forward it is held.
   */
  perMetre?: number;
}

export interface HandTracker {
  detect(video: HTMLVideoElement, timestampMs: number): HandSample[];
}

/** MediaPipe hand landmark numbers: [knuckle, middle joint, tip] per finger. */
const FINGERS = [
  [2, 3, 4],
  [5, 6, 8],
  [9, 10, 12],
  [13, 14, 16],
  [17, 18, 20],
];

/** Pairs of palm landmarks whose distance does not change as the fingers move: wrist and knuckles. */
const PALM = [
  [0, 5],
  [0, 9],
  [0, 17],
  [5, 17],
  [5, 13],
  [9, 17],
];

export async function createHandTracker(): Promise<HandTracker> {
  const fileset = await FilesetResolver.forVisionTasks(`${BASE}wasm`);
  const make = (delegate: 'GPU' | 'CPU') =>
    HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: `${BASE}models/hand_landmarker.task`, delegate },
      runningMode: 'VIDEO',
      numHands: 2,
    });
  let landmarker: HandLandmarker;
  try {
    landmarker = await make('GPU');
  } catch {
    landmarker = await make('CPU');
  }

  return {
    detect(video, timestampMs) {
      const result = landmarker.detectForVideo(video, timestampMs);
      const hands: HandSample[] = [];
      for (let h = 0; h < (result.worldLandmarks?.length ?? 0); h++) {
        const lm = result.worldLandmarks[h];
        const image = result.landmarks[h];
        const dir = (from: number, to: number): Vec3 => {
          const x = lm[to].x - lm[from].x;
          const y = lm[to].y - lm[from].y;
          const z = lm[to].z - lm[from].z;
          const len = Math.hypot(x, y, z) || 1;
          return [x / len, y / len, z / len];
        };
        // The hand's size in the picture against its real size, over the palm's own bones. The real
        // lengths are taken flat-on to the camera, so a tilted hand is not mistaken for a far one.
        const tall = video.videoHeight / (video.videoWidth || 1) || 0.75;
        let drawn = 0;
        let real = 0;
        for (const [a, b] of PALM) {
          drawn += Math.hypot(image[a].x - image[b].x, (image[a].y - image[b].y) * tall);
          real += Math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y);
        }
        hands.push({
          perMetre: real > 0.03 ? drawn / real : undefined,
          wrist: [image[0].x, image[0].y],
          label: result.handedness?.[h]?.[0]?.categoryName === 'Left' ? 'Left' : 'Right',
          forward: dir(0, 9),
          side: dir(17, 5),
          fingers: FINGERS.map(([a, b, c]) => ({ prox: dir(a, b), dist: dir(b, c) })),
        });
      }
      return hands;
    },
  };
}
