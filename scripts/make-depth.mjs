// One-time step per character: infer a depth map from characters/<id>.png and
// save it as characters/<id>.depth.png (brighter = nearer to the camera).
// Usage: npm run depth -- vandude
import { pipeline, RawImage } from '@huggingface/transformers';

const id = process.argv[2] ?? 'vandude';
const src = `characters/${id}.png`;
const out = `characters/${id}.depth.png`;

console.log(`Estimating depth for ${src} ...`);
const estimator = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small');
const image = await RawImage.read(src);
const { depth } = await estimator(image);
const sized =
  depth.width === image.width && depth.height === image.height
    ? depth
    : await depth.resize(image.width, image.height);
await sized.save(out);
console.log(`Saved ${out} (${sized.width}x${sized.height})`);
