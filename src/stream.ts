import * as THREE from 'three';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

/**
 * Stream mode (`?stream` in the address): the character as seen by a webcam mounted on top of
 * their monitor, for cutting out in OBS.
 *
 *   - the camera sits just above head height, close, with a wide lens: head, upper torso and arms
 *   - a flat seat picture (characters/<id>/<id>.seat.png, drawn on a green background) sits behind
 *   - everything that is neither character nor seat is one flat green for a chroma key
 *   - the room is dim and the monitor is the main light, changing colour and brightness like a screen
 *   - the picture is bent like a fisheye lens, its colours split towards the edges, and it is
 *     softened, given more contrast and covered in sensor grain like a cheap webcam's
 *
 * The green itself is never touched by the lens effects, so the key stays clean.
 * All the numbers to tune are in STREAM; a character overrides them under "stream" in its face.json.
 */
export interface StreamConfig {
  /** The key colour. Leave it pure green unless the character itself is green. */
  green: string;
  camera: {
    /** Vertical field of view, degrees. Wide, like a webcam. */
    fov: number;
    /** In front of the head bone, model units. */
    distance: number;
    /** Above the head bone: a webcam on top of a monitor looks slightly down. */
    height: number;
    /** The point looked at, this far below the head bone. */
    lookBelow: number;
  };
  /**
   * Where the arms rest while the body tracker does not see them (left arm; the right is mirrored):
   * directions of the upper arm and forearm as [out to the side, up, towards the camera].
   * Forward and down, as if on a keyboard and mouse just below the picture.
   */
  restArms: { upper: [number, number, number]; lower: [number, number, number] };
  /** How far the avatar reaches for a tracked hand against the operator's own reach: 1 = the same share of the arm's length. */
  armReach: number;
  /**
   * Squeezes a tracked hand's sideways position toward the middle. The webcam sees space either side
   * of the operator's shoulders; this tighter picture does not, so without it a hand held beside the
   * shoulder would be out of shot. 1 = true to the body, 0.65 = a hand at the edge of the webcam
   * picture is at the edge of this one.
   */
  armInward: number;
  /** How strongly a hand moving toward the webcam moves the avatar's hand toward this camera: 1 = true to life, 0 = no depth. */
  armDepth: number;
  /** Fisheye bend: 0 = none, 0.25 = a wide webcam, 0.5 = a door viewer. */
  fisheye: number;
  /** Colour split at the frame's edge, as a fraction of the half-width. 0 = none. */
  aberration: number;
  /** Sensor grain, 0 = none, 0.1 = a webcam in a dim room, 0.25 = very noisy. */
  grain: number;
  /** How much of the grain is coloured speckle rather than grey, 0..1. */
  grainColour: number;
  /** Blur, in pixels of the picture: a cheap lens and video compression. 0 = sharp. */
  soften: number;
  /** Contrast of the finished picture: 1 = as lit, 1.3 = punchy, deeper shadows. */
  contrast: number;
  /** Colour strength: 1 = as lit, below 1 = washed out like a webcam. */
  saturation: number;
  /** The room's own lights, as multiples of the app's normal look (lighting.ts). Low = the monitor dominates. */
  room: { environment: number; ambient: number; key: number; fill: number; rim: number };
  seat: {
    /** Size of the picture against the frame's height (1 = the picture's height fills the frame). */
    scale: number;
    /** Moves the picture right / up, in frame heights. */
    offset: [number, number];
    /** Brightness of the picture: above 1 lifts a dark seat so the contrast setting does not crush it to black. */
    brightness: number;
    /** How dark the character's soft shadow on the seat is, 0..1. */
    shadow: number;
    /** How much of the monitor's light colour the seat picks up, 0..1. */
    monitor: number;
  };
  monitor: {
    /** Overall brightness of the screen's light on the character. 0 = no monitor light. */
    intensity: number;
    /** Where the screen is from the stream camera: [right, up, towards the character]. */
    offset: [number, number, number];
    /** Seconds between possible changes of what is "on screen"; a change happens on about one in three. */
    beat: number;
    /** Seconds a change takes. */
    fade: number;
    /** Size of the fast shimmer, 0..1. */
    flicker: number;
  };
}

export const STREAM: StreamConfig = {
  green: '#00ff00',
  camera: { fov: 58, distance: 0.95, height: 0.14, lookBelow: 0.3 },
  restArms: { upper: [0.22, -0.72, 0.66], lower: [-0.04, -0.42, 0.9] },
  armReach: 1,
  armInward: 0.65,
  armDepth: 1,
  fisheye: 0.22,
  aberration: 0.006,
  grain: 0.1,
  grainColour: 0.35,
  soften: 1.2,
  contrast: 1.2,
  saturation: 0.92,
  room: { environment: 0.3, ambient: 0.3, key: 0.45, fill: 0.15, rim: 0.9 },
  seat: { scale: 1.0, offset: [0, 0], brightness: 1.5, shadow: 0.45, monitor: 0.35 },
  monitor: { intensity: 4.0, offset: [0, -0.3, 0.05], beat: 0.6, fade: 0.25, flicker: 0.05 },
};

/** Merge a character's partial "stream" settings over the defaults. */
export function streamConfig(over?: Partial<StreamConfig>): StreamConfig {
  return {
    ...STREAM,
    ...over,
    camera: { ...STREAM.camera, ...over?.camera },
    room: { ...STREAM.room, ...over?.room },
    seat: { ...STREAM.seat, ...over?.seat },
    monitor: { ...STREAM.monitor, ...over?.monitor },
  };
}

// ---- the monitor's light --------------------------------------------------------------------

/** Screen colours. No greens: green light on the character would be cut out by the chroma key. */
const SCREEN: [number, number, number][] = [
  [0.78, 0.86, 1.0], // a white page
  [0.42, 0.58, 1.0], // deep space blue
  [1.0, 0.68, 0.42], // engine glow, a sunrise
  [0.92, 0.5, 0.95], // nebula magenta
  [0.5, 0.88, 1.0], // HUD cyan
  [1.0, 0.94, 0.84], // warm white
  [1.0, 0.4, 0.3], // alarm red
];

const hash = (n: number) => {
  const s = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return s - Math.floor(s);
};

/** What is "on screen" during beat k: the colour and brightness chosen at the last change. */
function screenAt(k: number): { start: number; colour: [number, number, number]; level: number } {
  let start = k;
  while (start > k - 40 && hash(start) > 0.34) start--;
  return { start, colour: SCREEN[Math.floor(hash(start + 0.5) * SCREEN.length)], level: 0.45 + 0.55 * hash(start + 0.25) };
}

/** The screen's light at time t (seconds): linear colour, already scaled by its brightness (0..1). */
export function monitorLight(t: number, config: StreamConfig['monitor']): [number, number, number] {
  const beat = t / config.beat;
  const now = screenAt(Math.floor(beat));
  const before = screenAt(now.start - 1);
  // Ease from the previous picture into this one over `fade` seconds after the cut.
  const since = (beat - now.start) * config.beat;
  const mix = THREE.MathUtils.smoothstep(since, 0, config.fade);
  const shimmer = 1 + config.flicker * (0.6 * Math.sin(t * 37.0) + 0.4 * Math.sin(t * 91.3 + 1.7));
  // A slow swell under everything, as if the picture were moving.
  const swell = 0.9 + 0.1 * Math.sin(t * 0.8) * Math.sin(t * 0.23 + 2.0);
  return [0, 1, 2].map(
    (i) => (before.colour[i] * before.level * (1 - mix) + now.colour[i] * now.level * mix) * shimmer * swell,
  ) as [number, number, number];
}

// ---- the lens -------------------------------------------------------------------------------

const LensShader = {
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    tMask: { value: null as THREE.Texture | null },
    tSeat: { value: null as THREE.Texture | null },
    uAspect: { value: 1 },
    uPixel: { value: new THREE.Vector2(1 / 1024, 1 / 1024) },
    uTime: { value: 0 },
    uFisheye: { value: 0 },
    uAberration: { value: 0 },
    uGrain: { value: 0 },
    uGrainColour: { value: 0 },
    uSoften: { value: 0 },
    uContrast: { value: 1 },
    uSaturation: { value: 1 },
    uGreen: { value: new THREE.Vector3(0, 1, 0) },
    uSeatOn: { value: 0 },
    uSeatAspect: { value: 1 },
    uSeatScale: { value: 1 },
    uSeatOffset: { value: new THREE.Vector2() },
    uSeatLight: { value: new THREE.Vector3(1, 1, 1) },
    uShadow: { value: 0.5 },
    uKey: { value: 1 },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse; // the lit character, already tone mapped, black where there is none
    uniform sampler2D tMask;    // alpha = where the character is
    uniform sampler2D tSeat;    // the seat picture on its green background
    uniform float uAspect, uTime, uFisheye, uAberration, uGrain, uGrainColour, uSoften, uContrast, uSaturation;
    uniform float uSeatOn, uSeatAspect, uSeatScale, uShadow, uKey;
    uniform vec2 uPixel, uSeatOffset;
    uniform vec3 uGreen, uSeatLight;
    varying vec2 vUv;

    // Where in the straight picture this bent pixel looks. A fisheye squeezes the edges and
    // enlarges the middle; shift moves one colour slightly further out or in.
    vec2 lens(vec2 uv, float shift) {
      vec2 p = (uv - 0.5) * 2.0;
      p.x *= uAspect;
      float r2 = dot(p, p);
      float corner2 = uAspect * uAspect + 1.0;
      p *= (1.0 + uFisheye * r2) / (1.0 + uFisheye * corner2) * (1.0 + shift * r2);
      p.x /= uAspect;
      return p * 0.5 + 0.5;
    }

    // The seat picture with its own green removed.
    vec4 seat(vec2 uv) {
      vec2 q = uv - 0.5;
      q.x *= uAspect / uSeatAspect;
      q = (q - uSeatOffset) / uSeatScale + 0.5;
      if (q.x < 0.0 || q.x > 1.0 || q.y > 1.0) return vec4(0.0);
      q.y = max(q.y, 0.0); // the seat carries on below the picture
      vec3 c = texture2D(tSeat, q).rgb;
      float spill = c.g - max(c.r, c.b);
      c.g = min(c.g, max(c.r, c.b) + 0.03);
      return vec4(c, uSeatOn * (1.0 - smoothstep(0.06, 0.28, spill)));
    }

    // Seat with the character over it at one point of the straight picture: colour already
    // multiplied by how much of the pixel it covers (alpha), nothing where there is only background.
    vec4 layer(vec2 uv, float shade) {
      vec4 s = seat(uv);
      float m = texture2D(tMask, uv).a;
      return vec4(s.rgb * uSeatLight * shade * s.a * (1.0 - m) + texture2D(tDiffuse, uv).rgb, 1.0 - (1.0 - s.a) * (1.0 - m));
    }

    // The same, slightly out of focus.
    vec4 soft(vec2 uv, float shade) {
      vec2 d = uPixel * uSoften;
      return (layer(uv, shade) * 2.0
        + layer(uv + vec2(d.x, d.y), shade) + layer(uv + vec2(-d.x, d.y), shade)
        + layer(uv + vec2(d.x, -d.y), shade) + layer(uv + vec2(-d.x, -d.y), shade)) / 6.0;
    }

    float rand(vec2 p) {
      return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    void main() {
      vec2 uv = lens(vUv, 0.0);
      // A soft shadow of the character on the seat: how much character is nearby.
      float near = 0.0;
      for (int i = 0; i < 8; i++) {
        float a = float(i) * 0.7854;
        near += texture2D(tMask, uv + vec2(cos(a) / uAspect, sin(a)) * 0.035 + vec2(0.0, 0.012)).a;
      }
      float shade = 1.0 - uShadow * near / 8.0;

      // Each colour looks through the lens at a slightly different size.
      vec4 r = soft(lens(vUv, uAberration), shade);
      vec4 g = soft(uv, shade);
      vec4 b = soft(lens(vUv, -uAberration), shade);
      vec3 cover = vec3(r.a, g.a, b.a);
      vec3 colour = vec3(r.r, g.g, b.b) / max(cover, 0.001);

      // The camera's own picture processing: contrast about mid grey, then colour strength.
      colour = (colour - 0.5) * uContrast + 0.5;
      float grey = dot(colour, vec3(0.299, 0.587, 0.114));
      colour = mix(vec3(grey), colour, uSaturation);

      // Sensor grain: new every 1/24 s, two pixels wide, strongest in the shadows.
      vec2 cell = floor(vUv / (uPixel * 2.0)) + floor(uTime * 24.0) * vec2(17.0, 31.0);
      float n = rand(cell) - 0.5;
      vec3 speckle = vec3(rand(cell + 7.1), rand(cell + 13.7), rand(cell + 29.3)) - 0.5;
      colour += mix(vec3(n), speckle, uGrainColour) * uGrain * (1.4 - clamp(grey, 0.0, 1.0));
      colour = clamp(colour, 0.0, 1.0);

      // Over the key colour, which nothing above has touched. uKey 0: see-through instead.
      gl_FragColor = vec4(uGreen * uKey * (1.0 - cover) + colour * cover, max(uKey, g.a));
    }`,
};

export interface StreamLens {
  pass: ShaderPass;
  /** The texture whose alpha says where the character is (the frame saved before the glow). */
  setMask(texture: THREE.Texture): void;
  /** Once a frame: the time in seconds and the monitor's light, for the grain and for tinting the seat. */
  update(time: number, light: [number, number, number]): void;
  resize(width: number, height: number): void;
}

export function createStreamLens(config: StreamConfig, seatUrl?: string, transparent = false): StreamLens {
  const pass = new ShaderPass(LensShader);
  const u = pass.uniforms;
  // This pass works in screen colours (it runs after tone mapping), so the key colour and the
  // seat picture are used exactly as stored.
  const green = new THREE.Color(config.green).convertLinearToSRGB();
  u.uGreen.value.set(green.r, green.g, green.b);
  u.uFisheye.value = config.fisheye;
  u.uAberration.value = config.aberration;
  u.uGrain.value = config.grain;
  u.uGrainColour.value = config.grainColour;
  u.uSoften.value = config.soften;
  u.uContrast.value = config.contrast;
  u.uSaturation.value = config.saturation;
  u.uSeatScale.value = config.seat.scale;
  u.uSeatOffset.value.set(...config.seat.offset);
  u.uShadow.value = config.seat.shadow;
  u.uKey.value = transparent ? 0 : 1;
  if (seatUrl) {
    new THREE.TextureLoader().load(seatUrl, (texture) => {
      texture.colorSpace = THREE.NoColorSpace;
      texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
      const image = texture.image as { width: number; height: number };
      u.uSeatAspect.value = image.width / image.height;
      u.tSeat.value = texture;
      u.uSeatOn.value = 1;
    });
  }
  return {
    pass,
    setMask(texture) {
      u.tMask.value = texture;
    },
    update(time, light) {
      u.uTime.value = time % 1000;
      const k = config.seat.monitor;
      const b = config.seat.brightness;
      u.uSeatLight.value.set(b * (1 - k + k * light[0] * 1.6), b * (1 - k + k * light[1] * 1.6), b * (1 - k + k * light[2] * 1.6));
    },
    resize(width, height) {
      u.uAspect.value = width / height;
      u.uPixel.value.set(1 / width, 1 / height);
    },
  };
}
