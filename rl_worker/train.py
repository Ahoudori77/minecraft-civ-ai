"""
Smoke PPO training compatible with cleanrl 0.4.x + gym 0.19
• --smoke-test で 2_000 step だけ回す
"""
import argparse, gym, minerl, os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    def _init():
        return gym.make("MineRLTreechop-v0")
    return _init

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    if args.smoke_test:
        args.total_timesteps = 2_000

    env = DummyVecEnv([make_env()])
    model = PPO("CnnPolicy", env, verbose=0, n_steps=256, batch_size=256)
    model.learn(total_timesteps=args.total_timesteps)
    env.close()

if __name__ == "__main__":
    main()
