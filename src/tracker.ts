import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { Euler, Matrix4, Quaternion, Vector3 } from 'three';
import { BASE } from './characters';

export interface FaceSample {
  /** Head rotation in radians, camera space: +yaw = nose toward image right, +pitch = nod down, +roll = anticlockwise in the image. */
  yaw: number;
  pitch: number;
  roll: number;
  /**
   * MediaPipe blendshape scores (0..1) by ARKit-style name, e.g. jawOpen. Also four measurements
   * taken straight from the face landmarks, because the tracker's own nose-scrunch and brow-frown
   * scores barely move: geoNoseLeft / geoNoseRight (nostril wing to inner eye corner) and
   * geoBrowLeft / geoBrowRight (brow to eye corners), each divided by the distance between the
   * eyes. They shrink when the nose scrunches / the brow comes down. Left = the operator's left.
   * geoMouth is the gap between the inner lips on the same scale (grows as the mouth opens).
   */
  blend: Record<string, number>;
  /** Demo only: finished expression-shape values that bypass calibration and gain. */
  direct?: Record<string, number>;
  /** Where the face is in the video frame (used by the webcam head, src/webcamHead.ts). */
  head?: HeadBox;
}

/** The face in the video frame, in pixels (y down). */
export interface HeadBox {
  /** Middle of the face: halfway between the top of the forehead and the chin. */
  x: number;
  y: number;
  /** Unit direction from the chin to the forehead. */
  upX: number;
  upY: number;
  /** Distance from the top of the forehead to the chin. */
  size: number;
  /** The jawline from ear to ear through the chin, as [x, y] points. */
  jaw: [number, number][];
}

/** Face-mesh points along the jawline, from one ear down through the chin (152) to the other ear. */
const JAWLINE = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454];

export interface Tracker {
  delegate: 'GPU' | 'CPU';
  /** Runs detection on the current video frame. Returns null when no face is found. */
  detect(video: HTMLVideoElement, timestampMs: number): FaceSample | null;
}


export async function createTracker(): Promise<Tracker> {
  // WASM runtime and model are served by this app (see scripts/setup-assets.mjs), not a CDN.
  const fileset = await FilesetResolver.forVisionTasks(`${BASE}wasm`);
  const make = (delegate: 'GPU' | 'CPU') =>
    FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: `${BASE}models/face_landmarker.task`, delegate },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });

  let delegate: 'GPU' | 'CPU' = 'GPU';
  let landmarker: FaceLandmarker;
  try {
    landmarker = await make('GPU');
  } catch {
    delegate = 'CPU';
    landmarker = await make('CPU');
  }

  const m = new Matrix4();
  const q = new Quaternion();
  const pos = new Vector3();
  const scale = new Vector3();
  const euler = new Euler();

  return {
    delegate,
    detect(video, timestampMs) {
      const result = landmarker.detectForVideo(video, timestampMs);
      const matrix = result.facialTransformationMatrixes?.[0];
      const shapes = result.faceBlendshapes?.[0];
      if (!matrix || !shapes) return null;

      m.fromArray(matrix.data);
      m.decompose(pos, q, scale);
      euler.setFromQuaternion(q, 'YXZ');

      const blend: Record<string, number> = {};
      for (const c of shapes.categories) blend[c.categoryName] = c.score;

      const points = result.faceLandmarks?.[0];
      let head: HeadBox | undefined;
      if (points && points.length > 362) {
        const w = video.videoWidth || 1, h = video.videoHeight || 1;
        // 10 = top of the forehead, 152 = chin.
        const ux = (points[10].x - points[152].x) * w, uy = (points[10].y - points[152].y) * h;
        const size = Math.hypot(ux, uy) || 1;
        head = { x: ((points[10].x + points[152].x) / 2) * w, y: ((points[10].y + points[152].y) / 2) * h, upX: ux / size, upY: uy / size, size, jaw: JAWLINE.map((i) => [points[i].x * w, points[i].y * h]) };
        const dist = (a: number, b: number) =>
          Math.hypot((points[a].x - points[b].x) * w, (points[a].y - points[b].y) * h, (points[a].z - points[b].z) * w);
        const eyes = dist(33, 263) || 1; // outer eye corners
        blend.geoNoseLeft = dist(358, 362) / eyes;
        blend.geoNoseRight = dist(129, 133) / eyes;
        blend.geoBrowLeft = (dist(285, 362) + dist(334, 263)) / eyes;
        blend.geoBrowRight = (dist(55, 133) + dist(105, 33)) / eyes;
        // Gap between the inner edges of the lips. Unlike jawOpen it does not rely on the chin
        // outline, which the tracker places poorly under a beard.
        blend.geoMouth = dist(13, 14) / eyes;
        // Lip measurements for the "f / v" mouth (lower lip tucked under the top teeth):
        // visible height of the lower lip and of the upper lip, lower lip to chin, mouth width.
        blend.geoLipLower = dist(14, 17) / eyes;
        blend.geoLipUpper = dist(0, 13) / eyes;
        blend.geoLipChin = dist(17, 152) / eyes;
        blend.geoLipWidth = dist(61, 291) / eyes;
      }

      return { yaw: euler.y, pitch: euler.x, roll: euler.z, blend, head };
    },
  };
}
