#!/usr/bin/env python
"""
rl_worker/train.py – MineRL Treechop PPO (headless)
---------------------------------------------------
* LWJGL をソフトウェアレンダリングで動かし X11 依存を排除
* Dict 観測 → SB3 MultiInputPolicy
* episode_reward / episode_len を TensorBoard に書き込む
"""

import os, datetime as dt

# ── ① ここで“完全ヘッドレス”を宣言 ─────────────────────────────
os.environ["MINERL_RENDER_MODE"] = "headless"       # Malmo ≥0.4
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"      # llvmpipe GL3.3
# ------------------------------------------------------------------

import gym, minerl
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from wrappers import ActionFlattenWrapper

# ── ログ出力パス ───────────────────────────────────────────────
WORLD = os.getenv("WORLD", "default-world")
WID   = os.getenv("WORKER_ID", os.uname()[1])
BASE  = f"/workspace/logs/{WORLD}/{WID}"
os.makedirs(BASE, exist_ok=True)
print(f"[{WID}] logging to {BASE}")

# ── 環境ファクトリ ─────────────────────────────────────────────
def make_env():
    env = gym.make("MineRLTreechop-v0")
    env = ActionFlattenWrapper(env)     # Dict→MultiDiscrete (action)
    return env

vec_env = DummyVecEnv([make_env])

# ── TensorBoard writer ────────────────────────────────────────────
writer = SummaryWriter(log_dir=BASE)

# ── PPO モデル ──────────────────────────────────────────────────
model = PPO(
    policy="MultiInputPolicy",          # Dict 観測に対応
    env=vec_env,
    n_steps=2048,
    batch_size=512,
    learning_rate=2.5e-4,
    gamma=0.99,
    verbose=1,
)

# ── 学習ループ ────────────────────────────────────────────────
TOTAL_STEPS   = 50_000          # PoC 用
reward_sum, ep_len = 0.0, 0
obs = vec_env.reset()

for _ in range(TOTAL_STEPS):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, _ = vec_env.step(action)

    reward_sum += float(reward[0])
    ep_len     += 1

    # on‑policy: 1 step ごとにバッファへ追加 & 即学習
    model.learn(total_timesteps=1, reset_num_timesteps=False)

    if done[0]:
        ts = model.num_timesteps
        writer.add_scalar("episode_reward", reward_sum, ts)
        writer.add_scalar("episode_len",    ep_len,    ts)
        reward_sum, ep_len = 0.0, 0

writer.close()
vec_env.close()

model_path = f"{BASE}/ppo_minerl_treechop.zip"
model.save(model_path)
print(f"[{WID}] training finished, model saved → {model_path}")
