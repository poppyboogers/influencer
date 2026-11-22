import * as THREE from 'three';

/** Draws the skeleton over the model: green lines for bones, spheres at the joints. */
export interface BoneOverlay {
  object: THREE.Object3D;
  visible: boolean;
  /** Call every frame after the bones have been posed. */
  update(): void;
}

export function createBoneOverlay(bones: Iterable<THREE.Bone>): BoneOverlay {
  const list = [...bones];
  const color = 0x35ff5a;

  // Ordinary bones are drawn to their parent. The simulated cape / hair bones all hang off one
  // parent, which would draw as a fan, so those are drawn as the grid they form instead.
  const byName = new Map(list.map((b) => [b.name, b]));
  const pairs: [THREE.Object3D, THREE.Object3D][] = [];
  // Where each bone ends at rest, relative to its joint (the rig script records these).
  let rig: { restPositions?: Record<string, number[]>; restTails?: Record<string, number[]> } = {};
  for (const b of list) if (b.userData.restPositions) rig = b.userData;
  /** End bones (a toe tip, a hand, the head) have no child to draw a line to: draw them to their own end. */
  const tails: [THREE.Bone, THREE.Vector3][] = [];
  for (const b of list) {
    const pos = rig.restPositions?.[b.name];
    const tail = rig.restTails?.[b.name];
    if (pos && tail && !b.children.some((c) => (c as THREE.Bone).isBone)) {
      tails.push([b, new THREE.Vector3(tail[0] - pos[0], tail[1] - pos[1], tail[2] - pos[2])]);
    }
  }
  for (const b of list) {
    const grid = /^(.+)_(\d+)_(\d+)$/.exec(b.name);
    if (grid) {
      const [, part, r, c] = grid;
      const above = byName.get(`${part}_${Number(r) - 1}_${c}`);
      const beside = byName.get(`${part}_${r}_${Number(c) - 1}`);
      if (above) pairs.push([above, b]);
      if (beside) pairs.push([beside, b]);
    } else if (/_p\d+$/.test(b.name)) {
      // A control point of a shirt / hair soft body: shown as a dot only.
    } else if (b.userData.detached) {
      // Starts at its own place (a toe at its own base): no line back to the parent's joint.
    } else if ((b.parent as THREE.Bone | null)?.isBone) {
      pairs.push([b.parent!, b]);
    }
  }

  const positions = new Float32Array((pairs.length + tails.length) * 6);
  const lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const lines = new THREE.LineSegments(
    lineGeometry,
    new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true }),
  );
  lines.frustumCulled = false;
  lines.renderOrder = 10;

  const joints = new THREE.InstancedMesh(
    new THREE.SphereGeometry(1, 10, 8),
    new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true }),
    list.length,
  );
  joints.frustumCulled = false;
  joints.renderOrder = 11;
  // Simulated cape / hair grid bones and fingers are numerous and close together: draw them smaller.
  const sizes = list.map((b) => (/^.+_\d+_\d+$|_p\d+$/.test(b.name) ? 0.007 : /Thumb|Index|Middle|Ring|Pinky|Brow|MouthCorner|Toe(Inner|Middle|Outer)/.test(b.name) ? 0.006 : 0.014));

  const group = new THREE.Group();
  group.add(lines, joints);
  group.visible = false;

  const a = new THREE.Vector3();
  const m = new THREE.Matrix4();
  return {
    object: group,
    get visible() {
      return group.visible;
    },
    set visible(v: boolean) {
      group.visible = v;
    },
    update() {
      if (!group.visible) return;
      pairs.forEach(([from, to], i) => {
        a.setFromMatrixPosition(from.matrixWorld).toArray(positions, 6 * i);
        a.setFromMatrixPosition(to.matrixWorld).toArray(positions, 6 * i + 3);
      });
      tails.forEach(([bone, end], i) => {
        const o = 6 * (pairs.length + i);
        a.setFromMatrixPosition(bone.matrixWorld).toArray(positions, o);
        a.copy(end).applyMatrix4(bone.matrixWorld).toArray(positions, o + 3);
      });
      lineGeometry.attributes.position.needsUpdate = true;
      list.forEach((bone, i) => {
        a.setFromMatrixPosition(bone.matrixWorld);
        m.makeScale(sizes[i], sizes[i], sizes[i]).setPosition(a);
        joints.setMatrixAt(i, m);
      });
      joints.instanceMatrix.needsUpdate = true;
    },
  };
}
