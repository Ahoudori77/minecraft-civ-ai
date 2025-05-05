#!/usr/bin/env python3
"""
rl_worker/train.py  – Headless MineRL Treechop PPO loop
------------------------------------------------------
* Xvfb (:99) 上で Malmo をヘッドレス起動
* Dict‑action → Discrete に潰す ActionFlattenWrapper を使用
* 画像 (64 × 64 × 3) を VecTransposeImage で channel‑first に変換
* TensorBoard に episode_reward / episode_len を書き込む
"""

import os
import gym
import minerl
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from torch.utils.tensorboard import SummaryWriter

from wrappers import ActionFlattenWrapper

# ── headless / logging path ──────────────────────────────────────────
os.environ["MINERL_HEADLESS"] = "1"        # MineRL 0.4.4+ で有効
os.environ["DISPLAY"] = ":99"              # Xvfb は entrypoint で起動済み

WORLD = os.getenv("WORLD", "main-sim-001")
WID   = os.getenv("WORKER_ID", "worker-local")
BASE  = f"/workspace/logs/{WORLD}/{WID}"
os.makedirs(BASE, exist_ok=True)

print(f"[{WID}] logging to {BASE}", flush=True)
tb = SummaryWriter(BASE)

# ── env factory ─────────────────────────────────────────────────────
def make_env():
    env = gym.make("MineRLTreechop-v0")
    env = ActionFlattenWrapper(env)        # Dict‑action → Discrete
    env = VecTransposeImage(env)           # (H,W,C) → (C,H,W)
    return env

vec_env = DummyVecEnv([make_env])

# ── PPO (CNN policy) ────────────────────────────────────────────────
model = PPO(
    policy="CnnPolicy",
    env=vec_env,
    n_steps=1024,
    batch_size=256,
    learning_rate=2.5e-4,
    gamma=0.99,
    verbose=1,
    tensorboard_log=BASE,
)

# ── online learning loop ────────────────────────────────────────────
TOTAL_STEPS = 50_000                      # smoke‑test scale
obs         = vec_env.reset()
ep_reward   = 0.0
ep_len      = 0
episodes    = 0

for _ in range(TOTAL_STEPS):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, _ = vec_env.step(action)

    ep_reward += reward[0]
    ep_len    += 1

    # 1 step ごとに mini‑update
    model.learn(total_timesteps=1, reset_num_timesteps=False)

    if done[0]:
        episodes += 1
        tb.add_scalar("episode_reward", ep_reward, episodes)
        tb.add_scalar("episode_len",    ep_len,    episodes)
        ep_reward, ep_len = 0.0, 0

tb.close()
model.save(f"{BASE}/ppo_minerl_treechop.zip")
print(f"[{WID}] training finished, model saved → ppo_minerl_treechop.zip")
