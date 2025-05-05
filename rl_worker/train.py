"""
rl_worker/train.py – minimal smoke‑test loop
--------------------------------------------
* Spawns a MineRLTreechop‑v0 env
* Records 20 random steps to MP4
* Logs a dummy scalar to TensorBoard so we can verify TB wiring
"""

import os
import datetime as dt
import random
import gym
import minerl
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from torch.utils.tensorboard import SummaryWriter

# ── Runtime identifiers ──────────────────────────────────────────────
WORLD = os.getenv("WORLD", "default-world")
# fallback to container hostname if WORKER_ID env var is absent
WID   = os.getenv("WORKER_ID", os.uname()[1])

# ── Output directories ───────────────────────────────────────────────
BASE_DIR = f"/workspace/logs/{WORLD}/{WID}"
os.makedirs(BASE_DIR, exist_ok=True)

print(f"[{WID}] logging to {BASE_DIR}")

# ── TensorBoard writer ───────────────────────────────────────────────
writer = SummaryWriter(log_dir=BASE_DIR)

# ── MineRL environment ───────────────────────────────────────────────
env = gym.make("MineRLTreechop-v0")
obs = env.reset()
print(f"[{WID}] reset OK, obs['pov'].shape = {obs['pov'].shape}")

# ── Video recorder ───────────────────────────────────────────────────
video_path = f"{BASE_DIR}/run_{dt.datetime.now():%Y%m%d_%H%M%S}.mp4"
video      = VideoRecorder(env, video_path)

# ── Random‑agent loop (20 steps) ─────────────────────────────────────
for step in range(20):
    video.capture_frame()
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)

    # dummy scalar so TensorBoard UI has content
    writer.add_scalar("random_reward", random.random(), step)

    if done:
        env.reset()

# ── Cleanup ──────────────────────────────────────────────────────────
video.close()
writer.close()
env.close()

print(f"[{WID}] finished 20 steps, video saved → {video_path}")
