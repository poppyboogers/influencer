import * as THREE from 'three';

/**
 * Expression wrinkles drawn in the skin's shader: fine creases that deepen as an expression shape
 * comes on (crow's feet when the eye shuts, lines across the nose in a scrunch, the furrows
 * between the brows in a frown). The face mesh is far too coarse to fold into creases (its points
 * are about 5 mm apart), so each crease is a groove in the lighting instead: the surface direction
 * is tilted as if the groove were there, and the bottom of it is darkened a little.
 *
 * Creases are lines on the front view of the model at rest (x = the character's left, y = up, in
 * model units), set under "wrinkles" in the character's face.json. They are painted on the skin,
 * so they move and squash with it. Each set is drawn on both sides of the face; the left side
 * follows the Left shapes and the right side the Right shapes.
 */
export interface WrinkleConfig {
  /** How deep the grooves are, model units. 0 = off. */
  depth?: number;
  /** How much the bottom of a groove is darkened, 0..1. */
  dark?: number;
  /** Half-width of a groove, model units. */
  width?: number;
  /**
   * As an eye shuts, its glow goes out and the painted eye darkens to a lash line (otherwise the glow
   * map squashes into a bright streak that still shines through a closed eye). false = leave the glow alone.
   */
  eyeGlowShuts?: boolean;
  /** Only skin in front of this depth (z) gets wrinkles, so they do not show through on the back of the head. */
  minZ?: number;
  sets?: WrinkleSet[];
}

export interface WrinkleSet {
  name?: string;
  /** Shape name (without Left / Right) -> how strongly it brings the creases on. The total is capped at 1. */
  drive: Record<string, number>;
  /** Strength of this set against the others (default 1). */
  strength?: number;
  /** Width of this set's grooves against "width" (default 1). */
  width?: number;
  /** Each line is [x, y, x, y, ...] on the character's left side (x positive); it is mirrored to the right. */
  lines: number[][];
}

export interface Wrinkles {
  /** Call every frame with the current value (0..1) of each expression shape. */
  update(shapes: Record<string, number>): void;
}

const MAX_SETS = 8;
const MAX_SEGMENTS = 64;

export function createWrinkles(config: WrinkleConfig | undefined, meshes: THREE.Mesh[]): Wrinkles | null {
  const depth = config?.depth ?? 0;
  const sets = (config?.sets ?? []).filter((s) => s.lines?.length && s.drive).slice(0, MAX_SETS);
  if (!(depth > 0) || !sets.length || !meshes.length) return null;
  const baseWidth = config?.width ?? 0.0013;

  // Every line is cut into straight pieces: [x0, y0, x1, y1] and [width, start, end, set], where
  // start / end say how far along the whole line the piece runs (a crease fades out at its ends).
  const seg: number[] = [];
  const par: number[] = [];
  const box = [Infinity, Infinity, -Infinity, -Infinity];
  sets.forEach((set, index) => {
    for (const line of set.lines) {
      const points: [number, number][] = [];
      for (let i = 0; i + 1 < line.length; i += 2) points.push([line[i], line[i + 1]]);
      const lengths = points.slice(1).map((p, i) => Math.hypot(p[0] - points[i][0], p[1] - points[i][1]));
      const total = lengths.reduce((a, c) => a + c, 0) || 1;
      let done = 0;
      for (let i = 0; i + 1 < points.length && seg.length < 4 * MAX_SEGMENTS; i++) {
        seg.push(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1]);
        // A line that starts on the middle of the face joins its mirror image there, so it must not fade in.
        const from = points[0][0] === 0 ? 0.2 : 0;
        par.push(baseWidth * (set.width ?? 1), from + ((1 - from) * done) / total, from + ((1 - from) * (done + lengths[i])) / total, index);
        done += lengths[i];
      }
      for (const [x, y] of points) {
        box[0] = Math.min(box[0], x);
        box[1] = Math.min(box[1], y);
        box[2] = Math.max(box[2], x);
        box[3] = Math.max(box[3], y);
      }
    }
  });
  const count = seg.length / 4;
  if (!count) return null;
  const margin = baseWidth * 8;

  const uniforms = {
    wrSeg: { value: Array.from({ length: count }, (_, i) => new THREE.Vector4(...seg.slice(4 * i, 4 * i + 4))) },
    wrPar: { value: Array.from({ length: count }, (_, i) => new THREE.Vector4(...par.slice(4 * i, 4 * i + 4))) },
    wrLeft: { value: new Float32Array(MAX_SETS) },
    wrRight: { value: new Float32Array(MAX_SETS) },
    wrBox: { value: new THREE.Vector4(box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin) },
    wrLids: { value: new THREE.Vector2() },
    wrLook: { value: new THREE.Vector3(depth, config?.dark ?? 0.3, config?.minZ ?? 0.02) },
  };

  const declarations = `
    varying vec3 vWrRest;
    uniform vec4 wrSeg[${count}];
    uniform vec4 wrPar[${count}];
    uniform float wrLeft[${MAX_SETS}];
    uniform float wrRight[${MAX_SETS}];
    uniform vec4 wrBox;
    uniform vec3 wrLook;
    uniform vec2 wrLids;
    /** x = height of the skin surface (negative in a groove), yz = its slope across the front view. */
    vec3 wrinkleHeight(vec3 rest) {
      vec2 p = vec2(abs(rest.x), rest.y);
      if (rest.z < wrLook.z || p.x < wrBox.x || p.y < wrBox.y || p.x > wrBox.z || p.y > wrBox.w) return vec3(0.0);
      // Only the strongest crease at this spot counts: where two pieces of a line meet (or two
      // lines cross) the groove must not come out twice as deep.
      float best = 0.0, bestAmp = 0.0, bestSigma = 1.0;
      vec2 bestD = vec2(0.0);
      for (int i = 0; i < ${count}; i++) {
        int set = int(wrPar[i].w + 0.5);
        float on = rest.x >= 0.0 ? wrLeft[set] : wrRight[set];
        if (on <= 0.001) continue;
        vec2 a = wrSeg[i].xy, ba = wrSeg[i].zw - wrSeg[i].xy;
        float t = clamp(dot(p - a, ba) / max(dot(ba, ba), 1e-9), 0.0, 1.0);
        vec2 d = p - (a + ba * t);
        float sigma = wrPar[i].x;
        float d2 = dot(d, d);
        if (d2 > 40.0 * sigma * sigma) continue;
        // Fades in quickly at the start of the line and out slowly toward its end.
        float along = mix(wrPar[i].y, wrPar[i].z, t);
        float amp = on * wrLook.x * smoothstep(0.0, 0.15, along) * (1.0 - smoothstep(0.55, 1.0, along));
        float reach = amp * exp(-d2 / (2.0 * 6.76 * sigma * sigma));
        if (reach > best) { best = reach; bestAmp = amp; bestSigma = sigma; bestD = d; }
      }
      if (best <= 0.0) return vec3(0.0);
      // A narrow groove with a low wide swell either side of it, like bunched skin.
      float d2 = dot(bestD, bestD);
      float groove = exp(-d2 / (2.0 * bestSigma * bestSigma));
      float wide = 2.6 * bestSigma;
      float swell = 0.4 * exp(-d2 / (2.0 * wide * wide));
      float h = bestAmp * (swell - groove);
      vec2 slope = bestAmp * bestD * (groove / (bestSigma * bestSigma) - swell / (wide * wide));
      slope.x *= rest.x >= 0.0 ? 1.0 : -1.0;
      return vec3(h, slope);
    }`;

  const patched = new WeakSet<THREE.Material>();
  for (const mesh of meshes) {
    for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
      if (patched.has(material) || !(material as THREE.MeshStandardMaterial).isMeshStandardMaterial) continue;
      patched.add(material);
      // Other patches (the soft-body wobble) are kept: this one runs after them.
      const earlier = material.onBeforeCompile;
      const earlierKey = material.customProgramCacheKey;
      material.onBeforeCompile = (shader, renderer) => {
        earlier?.call(material, shader, renderer);
        Object.assign(shader.uniforms, uniforms);
        shader.vertexShader = shader.vertexShader
          .replace('#include <common>', '#include <common>\nvarying vec3 vWrRest;')
          .replace('#include <begin_vertex>', '#include <begin_vertex>\nvWrRest = position;');
        shader.fragmentShader = shader.fragmentShader
          .replace('#include <common>', `#include <common>\n${declarations}`)
          .replace(
            '#include <map_fragment>',
            `#include <map_fragment>
            vec3 wr = wrinkleHeight(vWrRest);
            diffuseColor.rgb *= 1.0 - wrLook.y * clamp(-wr.x / wrLook.x, 0.0, 1.0);`,
          )
          .replace(
            '#include <emissivemap_fragment>',
            `#include <emissivemap_fragment>
            #ifdef USE_EMISSIVEMAP
            {
              // The glowing part of the eye (where the glow map is lit) goes dark as that eye shuts.
              float lid = smoothstep(0.35, 0.9, vWrRest.x >= 0.0 ? wrLids.x : wrLids.y);
              float lit = smoothstep(0.02, 0.2, max(emissiveColor.r, max(emissiveColor.g, emissiveColor.b)));
              totalEmissiveRadiance *= 1.0 - lid;
              diffuseColor.rgb *= 1.0 - 0.93 * lid * lit;
            }
            #endif`,
          )
          .replace(
            '#include <normal_fragment_maps>',
            `#include <normal_fragment_maps>
            {
              // Tilt the surface direction by the groove's slope (the same sum a bump map uses).
              vec2 dH = vec2(dot(wr.yz, dFdx(vWrRest.xy)), dot(wr.yz, dFdy(vWrRest.xy)));
              vec3 sx = dFdx(-vViewPosition), sy = dFdy(-vViewPosition);
              vec3 r1 = cross(sy, normal), r2 = cross(normal, sx);
              float det = dot(sx, r1) * faceDirection;
              vec3 grad = sign(det) * (dH.x * r1 + dH.y * r2);
              // Where the skin is squashed (a shut eye) the creases bunch up and the tilt would run away
              // into sparkling highlights, so it is capped at about 40 degrees.
              float tilt = length(grad) / max(abs(det), 1e-12);
              if (tilt > 0.0) normal = normalize(normal - grad / max(abs(det), 1e-12) * min(1.0, 0.85 / tilt));
            }`,
          );
      };
      material.customProgramCacheKey = () => `${earlierKey?.call(material) ?? ''}|wrinkles${count}`;
      material.needsUpdate = true;
    }
  }

  return {
    update(shapes) {
      if (config?.eyeGlowShuts !== false) uniforms.wrLids.value.set(shapes.eyeBlinkLeft ?? 0, shapes.eyeBlinkRight ?? 0);
      sets.forEach((set, index) => {
        let left = 0, right = 0;
        for (const [name, amount] of Object.entries(set.drive)) {
          left += (shapes[`${name}Left`] ?? shapes[name] ?? 0) * amount;
          right += (shapes[`${name}Right`] ?? shapes[name] ?? 0) * amount;
        }
        const strength = set.strength ?? 1;
        uniforms.wrLeft.value[index] = THREE.MathUtils.clamp(left, 0, 1) * strength;
        uniforms.wrRight.value[index] = THREE.MathUtils.clamp(right, 0, 1) * strength;
      });
    },
  };
}
