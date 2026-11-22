// The operator's own head, cut out of the webcam picture and shown on the character in place of
// its helmet. Only for a character whose rig file has "webcamHead" (see sc.defiance.red.rig.json).
// The picture is a flat card at the helmet's position that always faces the viewer; the head's
// turn and tilt are already in the picture. The background is removed with MediaPipe's selfie
// segmenter (public/models/selfie_segmenter.tflite, run in this page), limited to an oval around
// the face so the shoulders stay out. Without that model the oval alone is used.
import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';
import * as THREE from 'three';
import { BASE } from './characters';
import type { HeadBox } from './tracker';

/** "webcamHead" in the rig file. */
export interface WebcamHeadConfig {
  /** Pattern for the names of the objects the webcam head replaces (the helmet). */
  hide: string;
  /** Size of the webcam head against the helmet; 1 = a head that would fit inside it. */
  scale?: number;
  /** Moves the webcam head from the helmet's centre, model units [left, up, forward]. */
  offset?: [number, number, number];
  /** Contrast of the webcam picture: 1 = as the camera gives it, 1.3 = punchier. */
  contrast?: number;
}

export interface WebcamHead {
  /** True when the operator has chosen the webcam head over the helmet. */
  readonly wanted: boolean;
  set(wanted: boolean): void;
  /** Call with each tracked video frame while the webcam head is wanted. */
  update(video: HTMLVideoElement, head: HeadBox, mirror: boolean, now: number): void;
  /** The helmet comes back while no face is found, so the character is never headless. */
  setFound(found: boolean): void;
  /** Puts the card on the head, facing the viewer. Call before rendering. */
  place(camera: THREE.Camera): void;
  /** A drawn stand-in face, for ?demo where there is no webcam. */
  demo(): void;
  /** One line for the on-screen read-out. */
  status(): string;
}

/** Card side, in face heights (forehead top to chin). Leaves room for hair and a tilted head. */
const CROP = 2.6;
/** Face height as a share of the helmet's height at scale 1. */
const FACE_SHARE = 0.62;
/**
 * The oval kept around the face, in face heights, measured from the middle of the face along the
 * face's own up direction. Raise halfHeight / up if tall hair is cut off; lower halfHeight if
 * the shoulders show.
 */
/**
 * The bottom of the cut-out follows the jawline (ear, chin, ear) with a hard edge: no neck.
 * This is how far below the tracked jawline is kept, in face heights. Raise it if the chin looks
 * shaved; lower it (0 or a little negative) if a strip of neck shows.
 */
const JAW_MARGIN = 0.02;
/**
 * Smoothing of the cut-out's outline.
 *   radius: blur in mask pixels (the mask is 256 across the card). 0 = off; higher = rounder outline.
 *   edge:   after the blur, mask values from the first to the second number become the edge.
 *           Closer together = crisper edge; both higher = the cut-out shrinks inward a little.
 *   hold:   share of the previous frame's mask kept each frame (0 = none). Higher = steadier
 *           edge, but it trails behind fast movement.
 */
const MASK_SMOOTH = { radius: 3, edge: [0.3, 0.7], hold: 0.4 };
const OVAL = { up: 0.18, halfWidth: 0.8, halfHeight: 1.02, feather: 0.2 };

export function createWebcamHead(scene: THREE.Scene, root: THREE.Object3D, headBone: THREE.Bone, cfg: WebcamHeadConfig): WebcamHead | null {
  const pattern = new RegExp(cfg.hide);
  const helmet: THREE.Mesh[] = [];
  root.traverse((o) => {
    // The loader strips dots from names; the name as written in the file is kept in userData.
    if ((o as THREE.Mesh).isMesh && (pattern.test(o.name) || pattern.test((o.userData.name as string) ?? ''))) helmet.push(o as THREE.Mesh);
  });
  if (!helmet.length) {
    console.warn(`webcamHead: no object matches "${cfg.hide}"; the webcam head is off.`);
    return null;
  }
  root.updateMatrixWorld(true);
  const box = new THREE.Box3();
  for (const mesh of helmet) {
    mesh.geometry.computeBoundingBox();
    box.union(mesh.geometry.boundingBox!.clone().applyMatrix4(mesh.matrixWorld));
  }
  const centre = headBone.worldToLocal(box.getCenter(new THREE.Vector3()).add(new THREE.Vector3(...(cfg.offset ?? [0, 0, 0]))));
  const side = (box.max.y - box.min.y) * FACE_SHARE * (cfg.scale ?? 1) * CROP;

  const material = new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false });
  // Contrast: push the picture's tones away from mid grey.
  const contrast = { value: cfg.contrast ?? 1 };
  material.onBeforeCompile = (shader) => {
    shader.uniforms.webcamContrast = contrast;
    shader.fragmentShader = shader.fragmentShader
      .replace('void main() {', 'uniform float webcamContrast;\nvoid main() {')
      .replace('#include <map_fragment>', '#include <map_fragment>\n\tdiffuseColor.rgb = clamp((diffuseColor.rgb - 0.18) * webcamContrast + 0.18, 0.0, 1.0);');
  };
  const card = new THREE.Mesh(new THREE.PlaneGeometry(side, side), material);
  card.visible = false;
  card.renderOrder = 10;
  card.frustumCulled = false;
  scene.add(card);

  let wanted = false;
  let found = false;
  let hasPicture = false;
  let isDemo = false;
  const apply = () => {
    const shown = wanted && found && hasPicture;
    card.visible = shown;
    for (const mesh of helmet) mesh.visible = !shown;
  };

  // ---- background removal -------------------------------------------------------
  let segmenter: ImageSegmenter | null = null;
  let segmenterState: 'none' | 'loading' | 'ready' | 'failed' = 'none';
  async function loadSegmenter() {
    segmenterState = 'loading';
    try {
      const fileset = await FilesetResolver.forVisionTasks(`${BASE}wasm`);
      const make = (delegate: 'GPU' | 'CPU') =>
        ImageSegmenter.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: `${BASE}models/selfie_segmenter.tflite`, delegate },
          runningMode: 'VIDEO',
          outputCategoryMask: false,
          outputConfidenceMasks: true,
        });
      try {
        segmenter = await make('GPU');
      } catch {
        segmenter = await make('CPU');
      }
      segmenterState = 'ready';
    } catch (err) {
      console.warn('Background removal for the webcam head did not load; using the oval only:', err);
      segmenterState = 'failed';
    }
  }

  let videoTexture: THREE.VideoTexture | null = null;
  let maskTexture: THREE.DataTexture | null = null;
  const MASK_SIZE = 256; // the segmenter's own working size
  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = cropCanvas.height = MASK_SIZE;
  const cropContext = cropCanvas.getContext('2d', { willReadFrequently: false })!;
  let work = new Float32Array(0);
  let scratch = new Float32Array(0);
  let held = new Float32Array(0); // the mask shown last frame
  let size = 0; // face height in video pixels, smoothed so the head does not pulse
  const smooth = (t: number) => t * t * (3 - 2 * t);

  /** Writes the cut-out (person and inside the oval) into the mask texture. `probs` null = oval only. */
  // The mask covers only the square around the face (the card), not the whole video frame, so
  // its edge is several times finer than a mask of the whole picture would give.
  function paint(probs: Float32Array | null, mw: number, mh: number, head: HeadBox) {
    if (!maskTexture || maskTexture.image.width !== mw || maskTexture.image.height !== mh) {
      maskTexture?.dispose();
      maskTexture = new THREE.DataTexture(new Uint8Array(mw * mh * 4), mw, mh, THREE.RGBAFormat);
      maskTexture.minFilter = maskTexture.magFilter = THREE.LinearFilter;
      material.alphaMap = maskTexture;
      material.needsUpdate = true;
    }
    const data = maskTexture.image.data as Uint8Array;
    if (work.length !== mw * mh) {
      work = new Float32Array(mw * mh);
      scratch = new Float32Array(mw * mh);
      held = new Float32Array(mw * mh);
    }
    work.fill(0);
    // The model's output is "how likely a person" on some versions and "how likely background"
    // on others: whichever makes the middle of the face a person is the right way round.
    const invert = probs ? probs[(mh >> 1) * mw + (mw >> 1)] < 0.5 : false;
    // The jawline in the face's own frame (across, height), and from it the lowest height kept
    // in each column across the face. Beside the jaw (past the ears) nothing below ear level is kept.
    const jaw = head.jaw
      .map(([x, y]) => {
        const jx = (x - head.x) / head.size, jy = (y - head.y) / head.size;
        return [jx * head.upY - jy * head.upX, jx * head.upX + jy * head.upY];
      })
      .sort((p, q) => p[0] - q[0]);
    const BINS = 128;
    const floor = new Float32Array(BINS);
    for (let b = 0, k = 0; b < BINS; b++) {
      const across = ((b + 0.5) / BINS - 0.5) * CROP * 1.5;
      while (k < jaw.length - 2 && jaw[k + 1][0] < across) k++;
      const [p, q] = [jaw[k], jaw[k + 1]];
      const t = Math.min(1, Math.max(0, (across - p[0]) / (q[0] - p[0] || 1)));
      floor[b] = p[1] + (q[1] - p[1]) * t - JAW_MARGIN;
    }
    const edge = (1.5 * CROP) / mw; // the edge is one and a half mask pixels wide, so it is not jagged
    // Distances below are in face heights; the mask spans CROP of them, centred on the face.
    for (let j = 1; j < mh - 1; j++) {
      const dy = ((j + 0.5) / mh - 0.5) * CROP;
      for (let i = 1; i < mw - 1; i++) {
        const dx = ((i + 0.5) / mw - 0.5) * CROP;
        const height = dx * head.upX + dy * head.upY; // 0 = middle of the face, -0.5 = chin
        const across = dx * head.upY - dy * head.upX;
        const bin = Math.min(BINS - 1, Math.max(0, Math.floor((across / (CROP * 1.5) + 0.5) * BINS)));
        const above = (height - floor[bin]) / edge; // how far above the jawline, in edge widths
        if (above <= 0) continue;
        const along = height - OVAL.up;
        const q = Math.hypot(across / OVAL.halfWidth, along / OVAL.halfHeight);
        if (q >= 1) continue;
        let a = (q < 1 - OVAL.feather ? 1 : smooth((1 - q) / OVAL.feather)) * Math.min(1, above);
        if (probs) {
          const p = invert ? 1 - probs[j * mw + i] : probs[j * mw + i];
          a *= p <= 0.3 ? 0 : p >= 0.7 ? 1 : smooth((p - 0.3) / 0.4);
        }
        work[j * mw + i] = a;
      }
    }
    // Smooth the outline: blur the cut-out (twice across, twice down), then tighten it again so
    // the edge stays where it was but loses its steps and specks. Each frame is also blended
    // with the one before, which stops the edge shimmering.
    const r = MASK_SMOOTH.radius;
    if (r > 0) {
      for (let pass = 0; pass < 2; pass++) {
        blur(work, scratch, mw, mh, 1, mw, r);
        blur(scratch, work, mh, mw, mw, 1, r);
      }
    }
    const keep = MASK_SMOOTH.hold;
    for (let j = 0; j < mh; j++) {
      for (let i = 0; i < mw; i++) {
        const k = j * mw + i;
        const t = Math.min(1, Math.max(0, (work[k] - MASK_SMOOTH.edge[0]) / (MASK_SMOOTH.edge[1] - MASK_SMOOTH.edge[0])));
        const a = (held[k] = held[k] * keep + smooth(t) * (1 - keep));
        // Row 0 of the texture is the bottom of the picture.
        const o = 4 * ((mh - 1 - j) * mw + i);
        data[o] = data[o + 1] = data[o + 2] = data[o + 3] = a * 255;
      }
    }
    maskTexture.needsUpdate = true;
  }

  /** Running-average blur along one direction: `lines` lines of `n` values, `step` apart. */
  function blur(from: Float32Array, to: Float32Array, n: number, lines: number, step: number, lineStep: number, r: number) {
    for (let l = 0; l < lines; l++) {
      const base = l * lineStep;
      let sum = 0;
      for (let k = -r; k <= r; k++) sum += from[base + Math.min(n - 1, Math.max(0, k)) * step];
      for (let k = 0; k < n; k++) {
        to[base + k * step] = sum / (2 * r + 1);
        sum += from[base + Math.min(n - 1, k + r + 1) * step] - from[base + Math.max(0, k - r) * step];
      }
    }
  }

  return {
    get wanted() {
      return wanted;
    },
    set(on) {
      wanted = on;
      apply();
    },
    setFound(on) {
      if (on === found) return;
      found = on;
      apply();
    },
    update(video, head, mirror, now) {
      const w = video.videoWidth, h = video.videoHeight;
      if (!w || !h || isDemo) return;
      if (segmenterState === 'none') void loadSegmenter();
      if (!videoTexture || videoTexture.image !== video) {
        videoTexture?.dispose();
        videoTexture = new THREE.VideoTexture(video);
        videoTexture.colorSpace = THREE.SRGBColorSpace;
        material.map = videoTexture;
        material.needsUpdate = true;
      }
      size = size ? size + (head.size - size) * 0.3 : head.size;

      const span = size * CROP; // side of the square around the face, video pixels
      if (segmenter) {
        // Give the model just that square, enlarged, instead of the whole frame.
        cropContext.fillRect(0, 0, MASK_SIZE, MASK_SIZE);
        cropContext.drawImage(video, head.x - span / 2, head.y - span / 2, span, span, 0, 0, MASK_SIZE, MASK_SIZE);
        segmenter.segmentForVideo(cropCanvas, now, (result) => {
          const mask = result.confidenceMasks?.[result.confidenceMasks.length - 1];
          if (mask) paint(mask.getAsFloat32Array(), mask.width, mask.height, head);
        });
      } else {
        paint(null, MASK_SIZE, MASK_SIZE, head);
      }

      // Show the square around the face. Mirrored, the picture is flipped like the avatar's motion.
      const sx = span / w, sy = span / h;
      videoTexture.repeat.set(mirror ? -sx : sx, sy);
      videoTexture.offset.set(head.x / w + (mirror ? sx : -sx) / 2, 1 - head.y / h - sy / 2);
      maskTexture?.repeat.set(mirror ? -1 : 1, 1);
      maskTexture?.offset.set(mirror ? 1 : 0, 0);
      if (!hasPicture) {
        hasPicture = true;
        apply();
      }
    },
    place(camera) {
      if (!card.visible) return;
      headBone.updateWorldMatrix(true, false);
      headBone.localToWorld(card.position.copy(centre));
      card.quaternion.copy(camera.quaternion);
    },
    demo() {
      isDemo = true;
      const c = document.createElement('canvas');
      c.width = c.height = 256;
      const g = c.getContext('2d')!;
      const u = 256 / CROP; // one face height
      const oval = (x: number, y: number, rx: number, ry: number, fill: string) => {
        g.fillStyle = fill;
        g.beginPath();
        g.ellipse(128 + x * u, 128 - y * u, rx * u, ry * u, 0, 0, Math.PI * 2);
        g.fill();
      };
      oval(0, 0.22, 0.52, 0.62, '#4a3424'); // hair
      oval(0, -0.6, 0.2, 0.3, '#c99a7c'); // neck
      oval(0, 0, 0.4, 0.5, '#e0b392'); // face, forehead top to chin
      oval(-0.17, 0.1, 0.06, 0.04, '#222');
      oval(0.17, 0.1, 0.06, 0.04, '#222');
      oval(0, -0.25, 0.14, 0.03, '#8a3b3b');
      const texture = new THREE.CanvasTexture(c);
      texture.colorSpace = THREE.SRGBColorSpace;
      material.map = texture;
      material.needsUpdate = true;
      hasPicture = true;
      apply();
    },
    status() {
      if (!wanted) return 'helmet on';
      if (isDemo) return 'webcam head (demo: drawn stand-in face)';
      if (!found || !hasPicture) return 'webcam head - no face found, helmet shown';
      return `webcam head${segmenterState === 'loading' ? ' (background removal loading)' : segmenterState === 'failed' ? ' (background removal failed to load: oval only)' : ''}`;
    },
  };
}
