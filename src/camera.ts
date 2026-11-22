const PREFERRED_KEY = 'mocap.camera';
/** Virtual cameras show up as webcams but deliver nothing unless another app is feeding them. */
const VIRTUAL = /ndi|virtual|obs|broadcast|snap|droidcam|iriun|epoccam|manycam/i;

export interface CameraSession {
  /** Name of the camera currently in use. */
  label: string;
  /** Switches to the next camera in the list and remembers the choice. */
  next(): Promise<void>;
}

function remember(deviceId: string) {
  try {
    localStorage.setItem(PREFERRED_KEY, deviceId);
  } catch {
    /* choice just won't persist */
  }
}

function preferredId(): string | null {
  try {
    return localStorage.getItem(PREFERRED_KEY);
  } catch {
    return null;
  }
}

async function listCameras(): Promise<MediaDeviceInfo[]> {
  const all = await navigator.mediaDevices.enumerateDevices();
  return all.filter((d) => d.kind === 'videoinput');
}

async function open(video: HTMLVideoElement, device: MediaDeviceInfo): Promise<void> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { deviceId: { exact: device.deviceId }, width: { ideal: 640 }, height: { ideal: 480 } },
  });
  (video.srcObject as MediaStream | null)?.getTracks().forEach((t) => t.stop());
  video.srcObject = stream;
  await video.play();
}

/**
 * Opens a webcam and plays it into the given <video>. Frames stay in this page.
 * Tries the remembered camera first, then real cameras, then virtual ones, so a
 * dead virtual camera (NDI, OBS, ...) set as the system default does not block startup.
 */
export async function startCamera(video: HTMLVideoElement): Promise<CameraSession> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('This browser does not expose a webcam API (open the app on http://localhost).');
  }

  let cameras = await listCameras();
  if (cameras.length === 0 || cameras.every((c) => c.label === '')) {
    // No permission yet: names are hidden until the user grants access once.
    try {
      const probe = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      probe.getTracks().forEach((t) => t.stop());
    } catch (err) {
      const name = (err as Error).name;
      if (name === 'NotAllowedError' || name === 'NotFoundError') throw err;
      // The default camera failed to start, but permission was granted; carry on and try each one.
    }
    cameras = await listCameras();
  }
  if (cameras.length === 0) {
    const err = new Error('No webcam found.');
    err.name = 'NotFoundError';
    throw err;
  }

  const preferred = preferredId();
  const rank = (c: MediaDeviceInfo) => (c.deviceId === preferred ? 0 : VIRTUAL.test(c.label) ? 2 : 1);
  cameras.sort((a, b) => rank(a) - rank(b));

  // Never fall back to a virtual camera on its own: it can "start" and deliver nothing,
  // which hides the real problem. Virtual cameras are used only if the user picked one
  // with [K] earlier, or if the machine has no real camera at all.
  const hasReal = cameras.some((c) => !VIRTUAL.test(c.label));
  const autoStart = (c: MediaDeviceInfo) => !hasReal || c.deviceId === preferred || !VIRTUAL.test(c.label);

  const failures: string[] = [];
  let index = -1;
  for (let i = 0; i < cameras.length; i++) {
    if (!autoStart(cameras[i])) continue;
    try {
      await open(video, cameras[i]);
      index = i;
      break;
    } catch (err) {
      failures.push(`- ${cameras[i].label || 'Unnamed camera'}: ${(err as Error).message}`);
    }
  }
  if (index < 0) {
    const skipped = cameras.length - failures.length;
    throw new Error(
      `the webcam would not start.\n${failures.join('\n')}\n` +
        (skipped > 0 ? `(${skipped} virtual cameras such as NDI were skipped on purpose.)\n` : '') +
        '\n' +
        'A real webcam that times out is usually held by another app (Teams, Zoom, the Windows Camera app) ' +
        'or blocked by a privacy shutter or keyboard camera switch. Close those apps and reload.',
    );
  }

  const session: CameraSession = {
    label: cameras[index].label || 'Camera',
    async next() {
      for (let step = 1; step <= cameras.length; step++) {
        const i = (index + step) % cameras.length;
        try {
          await open(video, cameras[i]);
          index = i;
          session.label = cameras[i].label || 'Camera';
          remember(cameras[i].deviceId);
          return;
        } catch {
          /* skip cameras that will not start */
        }
      }
    },
  };
  return session;
}
