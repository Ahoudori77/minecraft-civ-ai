import os, datetime as dt
import gym, minerl
from gym.wrappers.monitoring.video_recorder import VideoRecorder

WORLD  = os.getenv("WORLD", "default-world")
WID    = os.getenv("WORKER_ID", "worker-0")
BASE   = f"/workspace/logs/{WORLD}/{WID}"
os.makedirs(BASE, exist_ok=True)

print(f"[{WID}] logging to {BASE}")

env = gym.make("MineRLTreechop-v0")
obs = env.reset()
print(f"[{WID}] reset OK, obs['pov'].shape = {obs['pov'].shape}")

video = VideoRecorder(env, f"{BASE}/run_{dt.datetime.now():%Y%m%d_%H%M%S}.mp4")
for _ in range(20):                       # 20 ステップだけ録画
    video.capture_frame()
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    if done:
        env.reset()
video.close()
env.close()
print(f"[{WID}] finished 20 steps, video saved")
