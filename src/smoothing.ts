/**
 * One Euro filter: heavy smoothing when the signal is still (kills jitter),
 * light smoothing when it moves fast (keeps lag low).
 */
export class OneEuro {
  private x: number | null = null;
  private dx = 0;

  constructor(
    private minCutoff = 1.5,
    private beta = 0.02,
    private dCutoff = 1.0,
  ) {}

  private static alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(value: number, dt: number): number {
    if (this.x === null || dt <= 0) {
      this.x = value;
      return value;
    }
    const rawDx = (value - this.x) / dt;
    this.dx += OneEuro.alpha(this.dCutoff, dt) * (rawDx - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += OneEuro.alpha(cutoff, dt) * (value - this.x);
    return this.x;
  }

  get value(): number {
    return this.x ?? 0;
  }
}

/**
 * Holds still until the input has moved more than `band` away, then follows it at that distance.
 * Tracking noise smaller than the band produces no movement at all (a filter only shrinks it);
 * the cost is that small deliberate movements under the band are ignored too.
 */
export class Steady {
  private out: number | null = null;

  constructor(private band: number) {}

  follow(value: number): number {
    if (this.out === null) this.out = value;
    else if (value > this.out + this.band) this.out = value - this.band;
    else if (value < this.out - this.band) this.out = value + this.band;
    return this.out;
  }
}

/** Moves `current` toward `target` at a rate that reaches ~63% in `timeConstant` seconds. */
export function ease(current: number, target: number, dt: number, timeConstant: number): number {
  return current + (target - current) * (1 - Math.exp(-dt / timeConstant));
}

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
