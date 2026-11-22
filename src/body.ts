import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import type { TorsoSample } from './torso';
import { BASE } from './characters';

export type Vec3 = [number, number, number];

/** Unit directions of one arm in MediaPipe world space: x = image right, y = down, z = away from the camera. */
export interface ArmSample {
  upper: Vec3; // shoulder -> elbow
  lower: Vec3; // elbow -> wrist
}

export interface BodySample {
  /** The operator's own left / right arm, or null when it is out of frame. */
  left: ArmSample | null;
  right: ArmSample | null;
  /** Wrist positions in the image (0..1), used to match detected hands to arms. */
  wrists?: { left: [number, number] | null; right: [number, number] | null };
  /** Shoulders, hips and ear line, or null when the shoulders are not both in view. */
  torso?: TorsoSample | null;
}

export interface BodyTracker {
  detect(video: HTMLVideoElement, timestampMs: number): BodySample | null;
}

const VISIBLE = 0.6;
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
      const arm = (shoulder: number, elbow: number, wrist: number): ArmSample | null => {
        // Head-and-shoulders framing: elbow and wrist are both below the picture, so the arm is
        // not tracked and hangs relaxed. A raised hand with the elbow out of the picture still counts.
        // The wrist has to be in the picture: an elbow sitting at the bottom edge with the forearm
        // out of sight is a hanging arm, and following its guessed direction made the arm lift.
        if (!seen(shoulder) || !seen(elbow) || !shown(wrist)) return null;
        return { upper: dir(shoulder, elbow), lower: dir(elbow, wrist) };
      };
      // MediaPipe indices: 11/13/15 = left shoulder/elbow/wrist, 12/14/16 = right.
      const wrist = (i: number): [number, number] | null => (image && shown(i) ? [image[i].x, image[i].y] : null);
      // 7/8 = ears, 23/24 = hips. Picture positions are turned into picture widths on both axes.
      const aspect = video.videoHeight / (video.videoWidth || 1) || 0.75;
      const at = (i: number): Vec3 => [lm[i].x, lm[i].y, lm[i].z];
      const inPicture = (i: number): [number, number] => [image![i].x, image![i].y * aspect];
      const torso: TorsoSample | null =
        image && seen(11) && seen(12)
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
            }
          : null;
      return { left: arm(11, 13, 15), right: arm(12, 14, 16), wrists: { left: wrist(15), right: wrist(16) }, torso };
    },
  };
}
