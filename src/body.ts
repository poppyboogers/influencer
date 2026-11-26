import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import type { TorsoSample } from './torso';
import { BASE } from './characters';

export type Vec3 = [number, number, number];

/** Unit directions of one arm in MediaPipe world space: x = image right, y = down, z = away from the camera. */
export interface ArmSample {
  upper: Vec3; // shoulder -> elbow
  lower: Vec3; // elbow -> wrist
  /** False when the elbow is out of sight: upper and lower are then built on a guessed elbow. */
  elbowSeen?: boolean;
  /**
   * Where the hand is, whatever the elbow is doing: middle of the shoulders -> wrist, in shoulder
   * widths. Sideways and up / down are measured in the picture, where the tracker is steady;
   * only the depth comes from its 3D guess, which is its noisiest output.
   */
  reach?: Vec3;
}

export interface BodySample {
  /** The operator's own left / right arm, or null when it is out of frame. */
  left: ArmSample | null;
  right: ArmSample | null;
  /** Wrist positions in the image (0..1), used to match detected hands to arms. */
  wrists?: { left: [number, number] | null; right: [number, number] | null };
  /** Shoulders, hips and ear line, or null when the shoulders are not both in view. */
  torso?: TorsoSample | null;
  /**
   * How large the shoulders are drawn, when both are in sight: `across` is the distance between
   * them in picture widths, `perMetre` is picture widths per metre at their distance from the camera.
   */
  shoulders?: { across: number; perMetre: number } | null;
}

export interface BodyTracker {
  detect(video: HTMLVideoElement, timestampMs: number): BodySample | null;
}

const VISIBLE = 0.6;
/** Below this the tracker is guessing a shoulder's place: too unsure to measure a shrug from. */
const SHOULDER_SURE = 0.8;
/** A hand this close to a shoulder (as a fraction of the shoulder-to-shoulder distance in the picture) is covering it. */
const HAND_COVERS = 0.35;
/** A joint closer to the picture's edge than this (fraction of the picture) counts as outside it. */
const EDGE = 0.04;

export async function createBodyTracker(): Promise<BodyTracker> {
  const fileset = await FilesetResolver.forVisionTasks(`${BASE}wasm`);
  const make = (delegate: 'GPU' | 'CPU') =>
    PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: `${BASE}models/pose_landmarker_lite.task`, delegate },
      runningMode: 'VIDEO',
      numPoses: 1,
    });
  let landmarker: PoseLandmarker;
  try {
    landmarker = await make('GPU');
  } catch {
    landmarker = await make('CPU');
  }

  return {
    detect(video, timestampMs) {
      const result = landmarker.detectForVideo(video, timestampMs);
      const lm = result.worldLandmarks?.[0];
      if (!lm) return null;

      const seen = (i: number) => (lm[i].visibility ?? 0) > VISIBLE;
      const dir = (from: number, to: number): Vec3 => {
        const x = lm[to].x - lm[from].x;
        const y = lm[to].y - lm[from].y;
        const z = lm[to].z - lm[from].z;
        const len = Math.hypot(x, y, z) || 1;
        return [x / len, y / len, z / len];
      };
      // The model still reports joints that are outside the picture, as guesses that wander and
      // often with a high visibility score. Only a joint that is really in the picture counts.
      const image = result.landmarks?.[0];
      const inPictureAt = (i: number) =>
        !!image && image[i].x > EDGE && image[i].x < 1 - EDGE && image[i].y > EDGE && image[i].y < 1 - EDGE;
      const shown = (i: number) => seen(i) && inPictureAt(i);
      const reachOf = (wrist: number): Vec3 | undefined => {
        if (!image) return undefined;
        const tall = video.videoHeight / (video.videoWidth || 1) || 0.75; // picture heights into picture widths
        const across = Math.hypot(image[11].x - image[12].x, (image[11].y - image[12].y) * tall);
        const deep = Math.hypot(lm[11].x - lm[12].x, lm[11].y - lm[12].y, lm[11].z - lm[12].z);
        if (across < 0.02 || deep < 0.05) return undefined;
        return [
          (image[wrist].x - (image[11].x + image[12].x) / 2) / across,
          ((image[wrist].y - (image[11].y + image[12].y) / 2) * tall) / across,
          (lm[wrist].z - (lm[11].z + lm[12].z) / 2) / deep,
        ];
      };
      const arm = (shoulder: number, elbow: number, wrist: number): ArmSample | null => {
        // Head-and-shoulders framing: elbow and wrist are both below the picture, so the arm is
        // not tracked and hangs relaxed. A raised hand with the elbow out of the picture still counts.
        // The wrist has to be in the picture: an elbow sitting at the bottom edge with the forearm
        // out of sight is a hanging arm, and following its guessed direction made the arm lift.
        // Stream mode places the hand from the shoulder and wrist alone (`reach`), so an arm whose
        // elbow is out of sight is still reported; `elbowSeen` tells the direction-following mode to skip it.
        if (!seen(shoulder) || !shown(wrist)) return null;
        return {
          upper: dir(shoulder, elbow),
          lower: dir(elbow, wrist),
          elbowSeen: seen(elbow),
          reach: reachOf(wrist),
        };
      };
      // MediaPipe indices: 11/13/15 = left shoulder/elbow/wrist, 12/14/16 = right.
      const wrist = (i: number): [number, number] | null => (image && shown(i) ? [image[i].x, image[i].y] : null);
      // 7/8 = ears, 23/24 = hips. Picture positions are turned into picture widths on both axes.
      const aspect = video.videoHeight / (video.videoWidth || 1) || 0.75;
      const at = (i: number): Vec3 => [lm[i].x, lm[i].y, lm[i].z];
      const inPicture = (i: number): [number, number] => [image![i].x, image![i].y * aspect];
      // A hand held in front of a shoulder hides it; the tracker then invents the shoulder's position.
      // 15-22 = both wrists and the pinky, index and thumb points of both hands.
      const covered = (shoulder: number) => {
        if (!image) return false;
        if ((lm[shoulder].visibility ?? 0) < SHOULDER_SURE) return true;
        const [sx, sy] = inPicture(shoulder);
        const [lx, ly] = inPicture(11);
        const [rx, ry] = inPicture(12);
        const reach = Math.hypot(lx - rx, ly - ry) * HAND_COVERS;
        for (let i = 15; i <= 22; i++) {
          if ((lm[i].visibility ?? 0) < 0.5 || !inPictureAt(i)) continue;
          const [hx, hy] = inPicture(i);
          if (Math.hypot(hx - sx, hy - sy) < reach) return true;
        }
        return false;
      };
      const torso: TorsoSample | null =
        // One shoulder in sight is enough to keep the torso held; with one hidden it counts as covered.
        image && (seen(11) || seen(12))
          ? {
              shoulderLeft: at(11),
              shoulderRight: at(12),
              hipMid:
                seen(23) && seen(24)
                  ? [(lm[23].x + lm[24].x) / 2, (lm[23].y + lm[24].y) / 2, (lm[23].z + lm[24].z) / 2]
                  : null,
              image: {
                left: inPicture(11),
                right: inPicture(12),
                ears: [(image[7].x + image[8].x) / 2, ((image[7].y + image[8].y) / 2) * aspect],
              },
              covered: { left: covered(11), right: covered(12) },
            }
          : null;
      let shoulders: BodySample['shoulders'] = null;
      if (image && seen(11) && seen(12)) {
        const across = Math.hypot(image[11].x - image[12].x, (image[11].y - image[12].y) * aspect);
        const flat = Math.hypot(lm[11].x - lm[12].x, lm[11].y - lm[12].y);
        if (across > 0.05 && flat > 0.12) shoulders = { across, perMetre: across / flat };
      }
      return { left: arm(11, 13, 15), right: arm(12, 14, 16), wrists: { left: wrist(15), right: wrist(16) }, torso, shoulders };
    },
  };
}
