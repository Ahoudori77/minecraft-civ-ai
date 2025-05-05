"""
rl_worker/train.py – MineRL Treechop PPO PoC
--------------------------------------------
* ActionFlattenWrapper で Dict→MultiDiscrete に変換
* SB3 1.6 PPO を online 学習
* episode_reward / episode_len を TensorBoard に書き込む
"""

import os, random
import gym, minerl
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from wrappers import ActionFlattenWrapper      # ← 新規ファイル

# ── ID & 出力パス ──────────────────────────────────────────
WORLD = os.getenv("WORLD", "default-world")
WID   = os.getenv("WORKER_ID", os.uname()[1])
BASE  = f"/workspace/logs/{WORLD}/{WID}"
os.makedirs(BASE, exist_ok=True)
print(f"[{WID}] logging to {BASE}")

writer = SummaryWriter(log_dir=BASE)

# ── 環境作成 ────────────────────────────────────────────────
def make_env():
    env = gym.make("MineRLTreechop-v0")
    env = ActionFlattenWrapper(env)
    return env

vec_env = DummyVecEnv([make_env])

# ── PPO モデル ─────────────────────────────────────────────
model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    n_steps=2048,
    batch_size=512,
    learning_rate=2.5e-4,
    gamma=0.99,
    verbose=1,
)

# ── 学習ループ ─────────────────────────────────────────────
TOTAL_STEPS   = 50_000      # PoC 用
reward_sum    = 0.0
episode_len   = 0
obs = vec_env.reset()

for step in range(TOTAL_STEPS):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, _ = vec_env.step(action)

    reward_sum  += reward[0]
    episode_len += 1

    # online 学習: 1 step 毎に更新
    model.learn(total_timesteps=1, reset_num_timesteps=False)

    if done[0]:
        writer.add_scalar("episode_reward", reward_sum, model.num_timesteps)
        writer.add_scalar("episode_len",    episode_len, model.num_timesteps)
        reward_sum, episode_len = 0.0, 0

writer.close()
model.save(f"{BASE}/ppo_minerl_treechop.zip")
print(f"[{WID}] training finished, model saved")
