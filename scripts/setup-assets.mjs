// Copies the MediaPipe WASM runtime and downloads the tracking models into public/,
// so the app serves everything itself and runs offline afterwards.
import { cp, mkdir, access, writeFile } from 'node:fs/promises';

const MODELS = {
  'face_landmarker.task':
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  'pose_landmarker_lite.task':
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
  'hand_landmarker.task':
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
  // Removes the background behind the operator's head for the webcam head (src/webcamHead.ts).
  'selfie_segmenter.tflite':
    'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite',
};

await mkdir('public/models', { recursive: true });
await cp('node_modules/@mediapipe/tasks-vision/wasm', 'public/wasm', { recursive: true });
console.log('Copied MediaPipe WASM runtime to public/wasm');

for (const [file, url] of Object.entries(MODELS)) {
  const path = `public/models/${file}`;
  try {
    await access(path);
    console.log(`${file} already present`);
  } catch {
    console.log(`Downloading ${file} ...`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Download of ${file} failed: HTTP ${res.status}`);
    await writeFile(path, Buffer.from(await res.arrayBuffer()));
    console.log(`Saved ${path}`);
  }
}
