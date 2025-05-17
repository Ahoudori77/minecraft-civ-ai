"""
MineRL Treechop PPO (headless)  
--------------------------------
* ActionFlattenWrapper で MineRL の辞書型アクション → MultiDiscrete へ変換  
* Docker 内で xvfb を使った仮想 X サーバに接続 (`DISPLAY=:99`)  
* TensorBoard へ episode_reward / episode_len を出力  
"""

import os, gym, minerl
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_worker.wrappers.action_flatten_wrapper import ActionFlattenWrapper

# ── Headless ──────────────────────────────────────────────────────────
os.environ.setdefault("DISPLAY", ":99")            # docker-entrypoint で Xvfb :99 が起動済み
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # off‑screen OpenGL

# ── ログ出力先 ────────────────────────────────────────────────────────
WORLD = os.getenv("WORLD", "default-world")
WID   = os.getenv("WORKER_ID", os.uname()[1])
BASE  = f"/workspace/logs/{WORLD}/{WID}"
os.makedirs(BASE, exist_ok=True)
print(f"[{WID}] logging to {BASE}")
writer = SummaryWriter(BASE)

# ── 環境ファクトリ ───────────────────────────────────────────────────
def make_env():
    env = gym.make("MineRLTreechop-v0")
    env = ActionFlattenWrapper(env)
    return env

vec_env = DummyVecEnv([make_env])

# ── PPO モデル ───────────────────────────────────────────────────────
model = PPO(
    policy="MultiInputPolicy",          # Dict 観測なので必ず MultiInputPolicy
    env=vec_env,
    n_steps=2048,
    batch_size=512,
    learning_rate=2.5e-4,
    gamma=0.99,
    verbose=1,
)

# ── 学習 ────────────────────────────────────────────────────────────
TOTAL_STEPS = int(os.getenv("TOTAL_STEPS", "50000"))
model.learn(total_timesteps=TOTAL_STEPS)
model.save(f"{BASE}/ppo_minerl_treechop.zip")
print(f"[{WID}] training finished, model saved")
