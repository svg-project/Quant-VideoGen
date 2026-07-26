#!/usr/bin/env python
"""Build the shiba_wasd_1min example: WASD-only camera actions, one per 5s.

The causal_fast model is driven by poses.npy (c2w, OpenCV: +x right, +y down,
+z forward) + intrinsics.npy. WASD = pure translation, identity rotation:
    W = +z (push forward)   S = -z (pull back)
    A = -x (drift left)     D = +x (drift right)
Framewise translation is normalized to unit std downstream
(wan/utils/cam_utils.compute_relative_poses), so only direction matters.

973 frames @ 16 fps = 12 x 5s segments (80 frames) + 13 tail frames that
continue the last action. Segment boundaries are smoothed with a ~0.5s
moving average on velocity to avoid jerks, matching camera_actions.composite.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

N = 973
FPS = 16
SEG_FRAMES = 5 * FPS  # 80
SCHEDULE = ["W", "A", "W", "D", "S", "A", "W", "D", "S", "W", "A", "S"]
DIRS = {
    "W": np.array([0.0, 0.0, 1.0]),
    "S": np.array([0.0, 0.0, -1.0]),
    "A": np.array([-1.0, 0.0, 0.0]),
    "D": np.array([1.0, 0.0, 0.0]),
}
T_STEP = 0.02  # arbitrary; normalized to unit std downstream

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True,
                   help="Source example dir supplying image.jpg + intrinsics.npy "
                        "(e.g. an everframe task dir).")
    p.add_argument("--dst", type=Path,
                   default=Path("experiments/LingBot-v2/examples/shiba_wasd_1min"),
                   help="Example dir to write.")
    args = p.parse_args()
    SRC, DST = args.src, args.dst

    vel = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        seg = min(i // SEG_FRAMES, len(SCHEDULE) - 1)
        vel[i] = DIRS[SCHEDULE[seg]] * T_STEP
    k = np.ones(8) / 8.0  # ~0.5s smoothing, same as camera_actions.composite
    for c in range(3):
        vel[:, c] = np.convolve(vel[:, c], k, mode="same")

    pos = np.cumsum(vel, axis=0)
    pos -= pos[0]  # frame 0 at origin
    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    poses[:, :3, 3] = pos.astype(np.float32)

    DST.mkdir(parents=True, exist_ok=True)
    np.save(DST / "poses.npy", poses)
    shutil.copy(SRC / "intrinsics.npy", DST / "intrinsics.npy")
    shutil.copy(SRC / "image.jpg", DST / "image.jpg")

    prompt = (
        "A Shiba Inu trots toward the camera, mid-stride with one paw lifted, "
        "ears perked, tail curled. Golden sunlight casts long shadows as it moves "
        "along a dirt lane lined with a wooden fence and poplar trees receding into "
        "the distance. Shallow depth of field, 85mm telephoto look, ultra-detailed "
        "fur. Throughout the shot the camera moves through the scene as if "
        "exploring it: it pushes forward and eases back, and occasionally drifts "
        "sideways to the left and right, always facing straight ahead."
    )
    (DST / "prompt.txt").write_text(prompt)
    (DST / "meta.json").write_text(json.dumps({
        "id": "shiba_wasd_1min",
        "source": "everframe ef_A-01 (image), WASD-only synthetic camera",
        "action": "wasd_5s",
        "schedule": SCHEDULE,
        "seg_seconds": 5,
        "fps": FPS,
        "frames": N,
        "prompt": prompt,
    }, indent=2, ensure_ascii=False))
    print(f"wrote {DST}: poses {poses.shape}, schedule {'-'.join(SCHEDULE)}")

if __name__ == "__main__":
    main()
