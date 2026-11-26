import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { SavePass } from 'three/addons/postprocessing/SavePass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { createStreamLens, monitorLight, type StreamConfig } from './stream';

/**
 * The app's look: studio surroundings for reflections, a shadow-casting key light, fill and rim
 * lights, and film-style tone mapping. `?plain` in the address gives the old three-light setup.
 * All the numbers to tune are here.
 */
const LOOK = {
  exposure: 1.0,
  /** Strength of the studio surroundings: reflections plus soft all-round light. Much above 0.3 flattens the key light's shading. */
  environment: 0.2,
  ambient: 0.1,
  key: { color: 0xfff0dc, intensity: 3.2, from: [2.6, 3.6, 3.2] },
  fill: { color: 0xcfe0ff, intensity: 0.5, from: [-3.5, 0.8, 2.5] },
  rim: { color: 0x88aaff, intensity: 3.5, from: [-2.5, 2.2, -3.5] },
  shadowMap: 2048,
  /** How far the shadow is blurred, in shadow-map texels. */
  shadowBlur: 4,
  /** Brightness of the eye-glow map (<id>.emissive.png). Above 1 it is brighter than anything lit, which is what blooms. */
  eyeGlow: 2.2,
  /** The halo around anything brighter than `threshold` (only the eyes are). strength 0 turns the pass off. */
  bloom: { strength: 0.7, radius: 0.5, threshold: 1.1 },
};

/** Optional per-character texture files that replace or add to what is inside the model. */
export interface MaterialMaps {
  /** glTF layout: green = roughness, blue = metal. */
  roughness?: string;
  /** Colour of the parts that glow (eyes); black elsewhere. */
  emissive?: string;
}

export interface Look {
  /** Prepare a loaded model: shadows on, sharper textures. `patch` is the soft-body shader patch. */
  dress(model: THREE.Object3D, patch?: (material: THREE.Material) => void, maps?: MaterialMaps): Promise<void>;
  render(scene: THREE.Scene, camera: THREE.Camera): void;
  /** Call after renderer.setSize. */
  resize(): void;
}

/** Stream mode (see stream.ts): its settings and the character's seat picture, if it has one. */
export interface StreamSetup {
  config: StreamConfig;
  seatUrl?: string;
  /** Leave the background see-through instead of green. */
  transparent?: boolean;
}

export function createLook(renderer: THREE.WebGLRenderer, scene: THREE.Scene, plain = false, stream?: StreamSetup): Look {
  if (plain && !stream) {
    scene.add(new THREE.HemisphereLight(0xffffff, 0x30343c, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.set(2, 3, 4);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x88aaff, 1.2);
    rim.position.set(-3, 2, -3);
    scene.add(rim);
    return { async dress() {}, render: (s, c) => renderer.render(s, c), resize() {} };
  }

  // Khronos neutral keeps the painted texture colours; highlights roll off instead of clipping.
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.toneMappingExposure = LOOK.exposure;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;

  // Metal and glossy surfaces need something to reflect; without surroundings they render near black.
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  // Stream mode dims the room so the monitor is the main light (stream.ts, "room").
  const room = stream?.config.room ?? { environment: 1, ambient: 1, key: 1, fill: 1, rim: 1 };
  scene.environmentIntensity = LOOK.environment * room.environment;
  pmrem.dispose();

  scene.add(new THREE.HemisphereLight(0xffffff, 0x30343c, LOOK.ambient * room.ambient));

  const centre = new THREE.Vector3(0, 1.3, 0); // the character stands at the origin, 2.25 tall
  const light = (spec: { color: number; intensity: number; from: number[] }) => {
    const l = new THREE.DirectionalLight(spec.color, spec.intensity);
    l.position.set(spec.from[0], spec.from[1], spec.from[2]).normalize().multiplyScalar(6).add(centre);
    l.target.position.copy(centre);
    scene.add(l, l.target);
    return l;
  };
  const key = light({ ...LOOK.key, intensity: LOOK.key.intensity * room.key });
  light({ ...LOOK.fill, intensity: LOOK.fill.intensity * room.fill });
  light({ ...LOOK.rim, intensity: LOOK.rim.intensity * room.rim });

  key.castShadow = true;
  key.shadow.mapSize.set(LOOK.shadowMap, LOOK.shadowMap);
  key.shadow.radius = LOOK.shadowBlur;
  key.shadow.bias = -0.0004;
  key.shadow.normalBias = 0.025;
  const cam = key.shadow.camera;
  cam.left = cam.bottom = -1.5; // covers the whole figure with raised arms
  cam.right = cam.top = 1.5;
  cam.near = 2;
  cam.far = 10;
  cam.updateProjectionMatrix();

  // Glow: the picture is drawn into a buffer that can hold values brighter than white, the
  // brightest parts are blurred and added back, then tone mapping is applied once at the end.
  let composer: EffectComposer | null = null;
  let renderPass: RenderPass | null = null;
  // Stream mode: the screen's light on the character, and the webcam lens as the last step.
  const lens = stream ? createStreamLens(stream.config, stream.seatUrl, stream.transparent) : null;
  const monitor = stream && stream.config.monitor.intensity > 0 ? new THREE.PointLight(0xffffff, 0, 6, 2) : null;
  if (monitor) scene.add(monitor);
  if (LOOK.bloom.strength > 0 || lens) {
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    const target = new THREE.WebGLRenderTarget(size.x, size.y, { type: THREE.HalfFloatType, samples: 4 });
    composer = new EffectComposer(renderer, target);
    composer.setPixelRatio(1);
    renderPass = new RenderPass(scene, new THREE.PerspectiveCamera());
    composer.addPass(renderPass);
    if (lens) {
      // Keep a copy from before the glow: its alpha is where the character is. (The glow pass does not keep alpha.)
      const save = new SavePass();
      composer.addPass(save);
      lens.setMask(save.renderTarget.texture);
    }
    if (LOOK.bloom.strength > 0) {
      composer.addPass(new UnrealBloomPass(size.clone().multiplyScalar(0.5), LOOK.bloom.strength, LOOK.bloom.radius, LOOK.bloom.threshold));
    }
    composer.addPass(new OutputPass());
    if (lens) {
      composer.addPass(lens.pass);
      lens.resize(size.x, size.y);
    }
  }

  const anisotropy = renderer.capabilities.getMaxAnisotropy();
  return {
    async dress(model, patch, maps) {
      const loader = new THREE.TextureLoader();
      const load = async (url: string | undefined, like: THREE.Texture | null, colour: boolean) => {
        if (!url) return null;
        const texture = await loader.loadAsync(url);
        texture.flipY = false; // glTF texture coordinates start at the top
        texture.wrapS = like?.wrapS ?? THREE.RepeatWrapping;
        texture.wrapT = like?.wrapT ?? THREE.RepeatWrapping;
        texture.colorSpace = colour ? THREE.SRGBColorSpace : THREE.NoColorSpace;
        return texture;
      };
      // The shadow pass uses its own material; it needs the soft-body wobble too or the
      // shirt and hair would shadow themselves from where they are not.
      const depth = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
      patch?.(depth);
      const meshes: THREE.Mesh[] = [];
      model.traverse((o) => {
        if ((o as THREE.Mesh).isMesh) meshes.push(o as THREE.Mesh);
      });
      const done = new Set<THREE.Material>();
      let roughness: THREE.Texture | null = null;
      let emissive: THREE.Texture | null = null;
      for (const mesh of meshes) {
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        if (mesh.geometry.getAttribute('_softindex')) mesh.customDepthMaterial = depth;
        for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
          const m = material as THREE.MeshStandardMaterial;
          if (!done.has(m) && m.isMeshStandardMaterial && m.map) {
            done.add(m);
            roughness ??= await load(maps?.roughness, m.map, false);
            emissive ??= await load(maps?.emissive, m.map, true);
            if (roughness) {
              m.roughnessMap = m.metalnessMap = roughness;
              m.roughness = m.metalness = 1; // the map holds the actual values
            }
            if (emissive) {
              m.emissiveMap = emissive;
              m.emissive.set(0xffffff);
              m.emissiveIntensity = LOOK.eyeGlow;
            }
            m.needsUpdate = true;
          }
          for (const texture of [m.map, m.normalMap, m.roughnessMap, m.metalnessMap]) {
            if (!texture) continue;
            texture.anisotropy = anisotropy; // keeps textures sharp on surfaces seen at a slant
            texture.needsUpdate = true;
          }
        }
      }
    },
    render(s, c) {
      if (stream && lens) {
        const time = performance.now() / 1000;
        const light = monitorLight(time, stream.config.monitor);
        lens.update(time, light);
        if (monitor) {
          // The screen is just below the webcam, facing the character.
          const [right, up, towards] = stream.config.monitor.offset;
          monitor.position.set(right, up, -towards).applyMatrix4(c.matrixWorld);
          const level = Math.max(light[0], light[1], light[2], 1e-4);
          monitor.color.setRGB(light[0] / level, light[1] / level, light[2] / level);
          monitor.intensity = stream.config.monitor.intensity * level;
        }
      }
      if (composer) {
        if (renderPass!.scene !== s || renderPass!.camera !== c) Object.assign(renderPass!, { scene: s, camera: c });
        composer.render();
      } else renderer.render(s, c);
    },
    resize() {
      const size = renderer.getDrawingBufferSize(new THREE.Vector2());
      composer?.setPixelRatio(1);
      composer?.setSize(size.x, size.y);
      lens?.resize(size.x, size.y);
    },
  };
}
