"""
ActionFlattenWrapper
--------------------
MineRL の Dict‑action を Discrete N に畳み込み、SB3 がそのまま扱えるようにする。
Treechop‑v0 で頻出の最小操作セットだけを離散化している。
"""

import gym

# 0/1 のフラグ操作
_BINARY_KEYS = ["attack", "forward", "jump"]
# カメラ(Yaw, Pitch) Δ ∈ {‑1, 0, 1} ²  → 9 通り
_CAMERA_DELTAS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

class ActionFlattenWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        # 2³ × 9 = 72 通りのアクションを列挙
        self._actions = []
        for bits in range(1 << len(_BINARY_KEYS)):
            flags = [(bits >> i) & 1 for i in range(len(_BINARY_KEYS))]
            for cam in _CAMERA_DELTAS:
                act = {k: v for k, v in zip(_BINARY_KEYS, flags)}
                act["camera"] = cam
                self._actions.append(act)

        self.action_space = gym.spaces.Discrete(len(self._actions))

    # SB3 → MineRL
    def action(self, act_idx: int):
        return self._actions[int(act_idx)]

    # MineRL → SB3（今回不要）
    def reverse_action(self, x):
        raise NotImplementedError
