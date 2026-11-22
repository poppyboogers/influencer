import type { Vec3 } from './body';
import { OneEuro, Steady, clamp, ease } from './smoothing';

/** What the body tracker saw of the operator's torso in one frame. */
export interface TorsoSample {
  /** Shoulders in MediaPipe world space (metres; x = image right, y = down, z = away from the camera). */
  shoulderLeft: Vec3;
  shoulderRight: Vec3;
  /** Midpoint of the hips in the same space, or null when the hips are out of frame (sitting at a desk). */
  hipMid: Vec3 | null;
  /** The same shoulders and the midpoint of the ears in the picture: x right, y down, both in picture widths. */
  image: { left: [number, number]; right: [number, number]; ears: [number, number] };
}

/** Torso motion for the avatar, measured from the pose it was calibrated in. Angles in radians. */
export interface TorsoPose {
  lean: number; // sideways, positive = toward the avatar's left (+x)
  bend: number; // positive = forward, toward the viewer
  twist: number; // about the vertical, positive = the avatar's left shoulder moves back
  /** How far each shoulder is raised, as a fraction of the shoulder width (a full shrug is about 0.2). */
  shrugLeft: number;
  shrugRight: number;
}

/** Hip to shoulder distance of an adult, metres: turns "the shoulders moved this far" into a lean angle. */
const TORSO_LENGTH = 0.5;
/** Assumed horizontal field of view of the webcam, used only to judge leaning toward the camera. */
const CAMERA_FOV = (60 * Math.PI) / 180;
const CALIBRATION_FRAMES = 12;
const LIMIT = { lean: 0.45, bend: [-0.3, 0.5], twist: 0.7, shrug: [-0.05, 0.32] } as const;
const SHRUG_DEAD_ZONE = 0.02;
/**
 * Radians the measured torso angle must change before the avatar's torso moves at all, so tracking
 * noise does not rock the chest while the operator sits still. Twist comes from the tracker's
 * depth guess for each shoulder, by far its noisiest output, so it gets the widest band.
 * Raise a number if that motion still wobbles at rest; lower it if small movements are being missed.
 */
const STEADY = { lean: 0.012, bend: 0.02, twist: 0.045 } as const;

/** The average of the first few values it is given; fixed after that. */
class Baseline {
  private sum = 0;
  private n = 0;
  add(value: number) {
    if (this.n < CALIBRATION_FRAMES) {
      this.sum += value;
      this.n++;
    }
  }
  get ready() {
    return this.n >= CALIBRATION_FRAMES;
  }
  get value() {
    return this.sum / Math.max(1, this.n);
  }
}

/**
 * Turns torso samples into a smoothed TorsoPose. The pose the operator holds for the first
 * moment (or when recalibrate() is called) counts as upright and relaxed.
 *
 * With the hips in view, lean and bend are the tilt of the hip-to-shoulder line. Seated at a
 * desk the hips are hidden, so the avatar's hips stay put and the torso leans by however far
 * the shoulders have moved sideways (in the picture) or toward the camera (they look wider).
 * A shrug is the shoulder rising toward the ear line.
 */
export class TorsoFilter {
  value: TorsoPose | null = null;
  lastSeen = -Infinity;
  private base: Record<string, Baseline> = {};
  private hipMix = 0;
  private f = {
    lean: new OneEuro(0.8, 0.5),
    bend: new OneEuro(0.5, 0.4),
    twist: new OneEuro(0.5, 0.4),
    shrugLeft: new OneEuro(1.5, 1.2),
    shrugRight: new OneEuro(1.5, 1.2),
  };
  private steady = { lean: new Steady(STEADY.lean), bend: new Steady(STEADY.bend), twist: new Steady(STEADY.twist) };

  recalibrate() {
    this.base = {};
  }

  /** Difference between a measurement and its calibrated value (0 while still calibrating). */
  private delta(key: string, value: number): number {
    const b = (this.base[key] ??= new Baseline());
    b.add(value);
    return b.ready ? value - b.value : 0;
  }

  update(s: TorsoSample, mirror: boolean, dt: number, now: number) {
    const L = s.shoulderLeft;
    const R = s.shoulderRight;
    const across: Vec3 = [L[0] - R[0], L[1] - R[1], L[2] - R[2]];
    const width = Math.hypot(...across);
    const flat = Math.hypot(across[0], across[1]);
    const inPicture = Math.hypot(s.image.left[0] - s.image.right[0], s.image.left[1] - s.image.right[1]);
    // Side-on to the camera the shoulders overlap and nothing below can be measured.
    if (flat < 0.12 || inPicture < 0.02) return;
    const scale = inPicture / flat; // picture widths per metre at the operator
    const distance = 1 / (2 * Math.tan(CAMERA_FOV / 2) * scale); // metres from the camera

    // Sideways lean (toward picture right) and forward bend (toward the camera).
    const midX = ((s.image.left[0] + s.image.right[0]) / 2 - 0.5) / scale; // metres from the middle of the picture
    let lean = Math.asin(clamp(this.delta('midX', midX) / TORSO_LENGTH, -0.7, 0.7));
    let bend = Math.asin(clamp(-this.delta('distance', distance) / TORSO_LENGTH, -0.7, 0.7));
    this.hipMix = ease(this.hipMix, s.hipMid ? 1 : 0, dt, 0.4);
    if (s.hipMid) {
      const up: Vec3 = [
        (L[0] + R[0]) / 2 - s.hipMid[0],
        (L[1] + R[1]) / 2 - s.hipMid[1],
        (L[2] + R[2]) / 2 - s.hipMid[2],
      ];
      const hipLean = this.delta('hipLean', Math.atan2(up[0], -up[1]));
      const hipBend = this.delta('hipBend', Math.atan2(-up[2], -up[1]));
      lean += (hipLean - lean) * this.hipMix;
      bend += (hipBend - bend) * this.hipMix;
    }
    const twist = this.delta('twist', Math.atan2(across[2], across[0]));

    // Shrug: each shoulder's rise toward the ear line, as a fraction of the shoulder width.
    const gap = (shoulder: [number, number]) => (shoulder[1] - s.image.ears[1]) / scale;
    let raiseLeft = -this.delta('gapLeft', gap(s.image.left)) / width;
    let raiseRight = -this.delta('gapRight', gap(s.image.right)) / width;
    // Leaning sideways drops one shoulder and lifts the other without any shrug. Take that back
    // out, but never more than the lean explains (sliding sideways does not tilt the shoulders).
    const uneven = (raiseLeft - raiseRight) / 2;
    const fromLean = -Math.sin(lean) / 2;
    if (uneven * fromLean > 0) {
      const remove = Math.sign(uneven) * Math.min(Math.abs(uneven), Math.abs(fromLean));
      raiseLeft -= remove;
      raiseRight += remove;
    }
    // Ignore the last few millimetres: tracking noise should not make the shoulders twitch.
    const settle = (v: number) => Math.sign(v) * Math.max(0, Math.abs(v) - SHRUG_DEAD_ZONE);
    raiseLeft = settle(raiseLeft);
    raiseRight = settle(raiseRight);

    // Mirrored, the avatar behaves like the operator's reflection: sides swap.
    const sx = mirror ? -1 : 1;
    const [shrugLeft, shrugRight] = mirror ? [raiseRight, raiseLeft] : [raiseLeft, raiseRight];
    this.value = {
      lean: this.f.lean.filter(this.steady.lean.follow(clamp(lean * sx, -LIMIT.lean, LIMIT.lean)), dt),
      bend: this.f.bend.filter(this.steady.bend.follow(clamp(bend, LIMIT.bend[0], LIMIT.bend[1])), dt),
      twist: this.f.twist.filter(this.steady.twist.follow(clamp(twist * sx, -LIMIT.twist, LIMIT.twist)), dt),
      shrugLeft: this.f.shrugLeft.filter(clamp(shrugLeft, LIMIT.shrug[0], LIMIT.shrug[1]), dt),
      shrugRight: this.f.shrugRight.filter(clamp(shrugRight, LIMIT.shrug[0], LIMIT.shrug[1]), dt),
    };
    this.lastSeen = now;
  }
}
