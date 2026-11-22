# Webcam-to-Character Motion Capture — Feasibility and Proof

**Research date:** 17 November 2026  
**Conclusion:** Feasible as a small, local browser application. A shader is suitable for rendering/deformation, but landmark detection requires a computer-vision model or equivalent tracking logic.

## 1. Proposed application

Use an ordinary webcam to track the operator and drive a character in real time. The initial proof of concept should use a rigged 3D avatar; driving an arbitrary single PNG is possible only with additional artwork segmentation, mesh rigging, or image-warping, and will not recreate hidden views of the character.

**Pipeline:** webcam → MediaPipe face/pose/hand tracking → landmark filtering and retargeting → Three.js avatar animation and shaders → browser canvas / optional recording.

## 2. Evidence of feasibility

| Claim | Supporting evidence | Qualification |
|---|---|---|
| Webcam facial tracking works in a browser | Google MediaPipe Face Landmarker supports video/live camera frames, 478 face landmarks, 52 facial blendshape scores and facial transformation matrices. [1] | Expression outputs must be explicitly enabled. |
| A webcam can drive a rendered 3D character | Browser VTuber is an existing browser project combining MediaPipe tracking, Kalidokit rigging, VRM avatars and Three.js. [2] | This demonstrates an existing implementation, not guaranteed performance on every machine. |
| Face geometry can be turned into a morph-target-capable model | Face-to-Blendshape-3D describes photo-to-GLB output with ARKit-style morph targets and Three.js rendering. [3] | Separate from live motion capture; evaluate generated mesh quality. |
| The tracking code is open source | Google AI Edge maintains the MediaPipe repository. [4] | Review current licenses and model distribution terms before redistribution. |

**Important correction:** MediaPipe's 478/52 figures specifically describe the **Face Landmarker**, not a blanket guarantee for every Holistic API/version. Choose and validate the individual Tasks APIs against the installed version.

## 3. Existing repositories and documentation

1. **Google MediaPipe (official):** https://github.com/google-ai-edge/mediapipe  
   Core tracking framework and task implementations. [4]
2. **MediaPipe Face Landmarker documentation (official):** https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker  
   Live video, blendshapes, face transformation matrices, and browser configuration. [1]
3. **Browser VTuber:** https://github.com/Galantryas/browser-vtuber  
   Closest directly relevant example: webcam → MediaPipe/Kalidokit → VRM avatar in Three.js. [2]
4. **Face-to-Blendshape-3D:** https://github.com/arturwyroslak/face-to-blendshape-3d  
   Experimental image-to-3D-face asset pipeline, useful for preparing character geometry. [3]
5. **Three.js:** https://github.com/mrdoob/three.js  
   WebGL renderer, morph targets, bones, materials and shader customization.
6. **three-vrm:** https://github.com/pixiv/three-vrm  
   VRM avatar loader and animation support for Three.js.
7. **Kalidokit:** https://github.com/yeemachine/kalidokit  
   Landmark-to-avatar-rig calculation utilities; check compatibility and maintenance status before adopting.

## 4. Recommended proof of concept

**Stack:** Vite + TypeScript, `@mediapipe/tasks-vision`, Three.js, `@pixiv/three-vrm`, and optionally Kalidokit (subject to compatibility checks).

**MVP acceptance tests:**

- Webcam permission is requested and video frames are captured locally.
- Facial expressions (blink, mouth open, smile) update corresponding avatar morph targets.
- Head turning updates the avatar's head orientation.
- Tracking loss returns the avatar smoothly to neutral instead of freezing unnaturally.
- A basic FPS/latency overlay is shown; target approximately 30 FPS, measured on the actual machine rather than promised.
- No webcam images are uploaded to a server in the default configuration.

**Implementation sequence:**

1. Run the official Face Landmarker web demo and confirm webcam tracking.
2. Load a licensed VRM model in Three.js.
3. Map blendshape names and facial pose into the avatar's expression and head controls.
4. Add smoothing, calibration and tracking-loss handling.
5. Add body and hand tracking only after facial animation is stable.
6. Optionally add shader-driven 2D mesh warping for illustrated characters.

## 5. Constraints and risks

- **Shader-only approach:** not recommended for the tracker; shaders can deform or display the avatar but are not a substitute for robust pose inference.
- **Single flat character image:** blinking/mouth deformation is practical with prepared layers or a deformable mesh; realistic head rotation and occluded anatomy cannot be reliably recovered from a single flat picture.
- **3D avatar:** requires a compatible rig and expression mappings. A generic unrigged GLB will not animate automatically.
- **Tracking quality:** occlusion, lighting, camera framing and motion blur can degrade results; monocular depth is estimated, not measured precisely.
- **Performance:** inference and rendering compete for resources. Use resolution limits, frame skipping and possibly a worker; benchmark rather than assume 30–60 FPS.
- **Licensing/privacy:** check each repository, model and avatar license. Browser webcam access typically requires localhost or HTTPS and user permission.

## 6. Effort estimate

| Scope | Indicative effort, assuming an experienced developer and prepared assets |
|---|---|
| Face-tracking + existing rigged 3D avatar prototype | 1–3 development days |
| Polished facial tracking, calibration, UI, recording | Several additional days to weeks |
| Arbitrary single-image 2D character animation | Highly dependent on artwork and deformation quality; separate feasibility prototype required |

These are engineering estimates, **not** benchmarked measurements or delivery guarantees.

## 7. Decision

**Proceed with a local browser proof of concept using MediaPipe Face Landmarker and a VRM avatar rendered with Three.js.** Reuse Browser VTuber as a working architectural reference rather than implementing the full pipeline from scratch. If the actual goal is animating a single uploaded illustration, first validate a face-only 2D mesh-warping prototype before committing to full-body motion capture.

## Sources

[1] Google AI Edge, *Face Landmarker*: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker  
[2] Galantryas, *browser-vtuber*: https://github.com/Galantryas/browser-vtuber  
[3] arturwyroslak, *face-to-blendshape-3d*: https://github.com/arturwyroslak/face-to-blendshape-3d  
[4] Google AI Edge, *MediaPipe repository*: https://github.com/google-ai-edge/mediapipe
