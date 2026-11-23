/**
 * Listens on the microphone for the two sounds made with the lower lip against the upper teeth:
 * the hiss of an "f" and the voiced buzz of a "v". The sound alone cannot tell an "f" from an
 * "s" or a "v" from a vowel, so main.ts only acts on it while the face tracker also sees the
 * lower lip rolled in.
 *
 * Measured from the live spectrum, in decibels:
 *   rise      how far the high band (hiss) stands above the room's background level
 *   tilt      high band minus low band: near 0 for a bare hiss, very negative for a vowel
 *   voiceRise how far the low band (voice) stands above the background
 */

export interface FvSound {
  /** 0..1: an unvoiced hiss is being heard ("f"). */
  hiss: number;
  /** 0..1: a voice is being heard ("v", but also any vowel). */
  voice: number;
  rise: number;
  tilt: number;
  voiceRise: number;
}

export interface FvListener {
  read(now: number): FvSound;
  readonly label: string;
}

/** [nothing below, full above], in decibels. Overridden by "fv" in the character's face.json. */
export interface FvSoundSettings {
  hiss: readonly [number, number];
  tilt: readonly [number, number];
  voice: readonly [number, number];
}
export const FV_SOUND: FvSoundSettings = { hiss: [6, 14], tilt: [-24, -12], voice: [5, 12] };

/** Hiss lives above the vowel formants; voice below them. Hz. */
const HIGH_BAND = [4000, 11000] as const;
const LOW_BAND = [150, 1200] as const;
/** The background level drops to a quieter reading at once, but creeps up this slowly (dB per second). */
const FLOOR_RISE = 2;
const FLOOR_FALL_S = 0.1;
/** How long after the first sound the background level is taken as whatever is heard. */
const WARM_UP_MS = 1500;
/** Loopback and virtual inputs that are not a real microphone. */
const VIRTUAL = /ndi|obs|virtual|vb-audio|cable|stereo mix|voicemeeter/i;

const smooth = (value: number, [from, to]: readonly [number, number]) => {
  const t = Math.min(1, Math.max(0, (value - from) / (to - from)));
  return t * t * (3 - 2 * t);
};

export async function createFvListener(settings: FvSoundSettings = FV_SOUND): Promise<FvListener> {
  // The browser's own clean-up filters treat a hiss as noise and remove it: switch them off.
  const open = (deviceId?: string) =>
    navigator.mediaDevices.getUserMedia({
      audio: {
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
      video: false,
    });
  let stream = await open();
  let track = stream.getAudioTracks()[0];
  if (VIRTUAL.test(track.label)) {
    const real = (await navigator.mediaDevices.enumerateDevices()).find(
      (d) => d.kind === 'audioinput' && d.deviceId !== 'default' && d.deviceId !== 'communications' && !VIRTUAL.test(d.label),
    );
    if (real) {
      track.stop();
      stream = await open(real.deviceId);
      track = stream.getAudioTracks()[0];
    }
  }

  const context = new AudioContext();
  const analyser = context.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.3;
  context.createMediaStreamSource(stream).connect(analyser); // analysed only, never played back
  // A browser may hold the audio back until the page has been clicked or a key pressed.
  const wake = () => void context.resume();
  wake();
  window.addEventListener('pointerdown', wake);
  window.addEventListener('keydown', wake);

  const bins = new Float32Array(analyser.frequencyBinCount);
  const hzPerBin = context.sampleRate / analyser.fftSize;
  /** Mean power of a band, in decibels. */
  const band = ([from, to]: readonly [number, number]) => {
    const first = Math.max(1, Math.round(from / hzPerBin));
    const last = Math.min(bins.length - 1, Math.round(to / hzPerBin));
    let power = 0;
    for (let i = first; i <= last; i++) power += 10 ** (Math.max(bins[i], -120) / 10);
    return 10 * Math.log10(power / Math.max(1, last - first + 1));
  };
  const settle = (floor: number, value: number, dt: number) =>
    value < floor ? floor + (value - floor) * Math.min(1, dt / FLOOR_FALL_S) : Math.min(value, floor + FLOOR_RISE * dt);

  let highFloor: number | null = null;
  let lowFloor = 0;
  let started = 0;
  let lastRead = -Infinity;
  let last: FvSound = { hiss: 0, voice: 0, rise: 0, tilt: 0, voiceRise: 0 };

  return {
    label: track.label || 'microphone',
    read(now) {
      const dt = Math.min(0.1, (now - lastRead) / 1000);
      if (dt < 0.005 || context.state !== 'running') return last;
      lastRead = now;
      analyser.getFloatFrequencyData(bins);
      const high = band(HIGH_BAND);
      const low = band(LOW_BAND);
      if (low < -115 && high < -115) return last; // the microphone has not delivered any sound yet
      if (highFloor === null) {
        highFloor = high;
        lowFloor = low;
        started = now;
      }
      if (now - started < WARM_UP_MS) {
        // The first readings arrive while the microphone is still opening and are far too quiet:
        // follow the level both ways until it has settled.
        highFloor += (high - highFloor) * Math.min(1, dt / 0.3);
        lowFloor += (low - lowFloor) * Math.min(1, dt / 0.3);
      } else {
        highFloor = settle(highFloor, high, dt);
        lowFloor = settle(lowFloor, low, dt);
      }
      const rise = high - highFloor;
      const tilt = high - low;
      const voiceRise = low - lowFloor;
      last = {
        hiss: smooth(rise, settings.hiss) * smooth(tilt, settings.tilt),
        voice: smooth(voiceRise, settings.voice),
        rise,
        tilt,
        voiceRise,
      };
      return last;
    },
  };
}
