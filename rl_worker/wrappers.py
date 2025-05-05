"""
ActionFlattenWrapper  
MineRL の辞書型アクション空間 → gym.spaces.MultiDiscrete に圧縮
"""

import gym
import numpy as np
from gym import spaces

# カメラ回転量を粗く量子化（° 単位）
CAM_ANGLES = np.array([-40, -20, -10, -5, 0, 5, 10, 20, 40], dtype=np.int32)

class ActionFlattenWrapper(gym.ActionWrapper):
    """
    MineRL の Dict アクションを以下 8 要素の MultiDiscrete ベクトルへ：

    0: forward/back  (-1,0,+1)
    1: left/right    (-1,0,+1)
    2: jump          (0/1)
    3: sneak         (0/1)
    4: sprint        (0/1)
    5: camera pitch  (index in CAM_ANGLES)
    6: camera yaw    (index in CAM_ANGLES)
    7: attack        (0/1)
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.action_space = spaces.MultiDiscrete([
            3,  # forward/back
            3,  # left/right
            2,  # jump
            2,  # sneak
            2,  # sprint
            len(CAM_ANGLES),  # pitch
            len(CAM_ANGLES),  # yaw
            2,  # attack
        ])

    # ── flat → Dict ────────────────────────────────────────────────
    def action(self, act):
        fb, lr, jump, sneak, sprint, pitch_idx, yaw_idx, attack = act

        act_dict = {
            "forward": fb == 2,
            "back":    fb == 0,
            "left":    lr == 0,
            "right":   lr == 2,
            "jump":    jump == 1,
            "sneak":   sneak == 1,
            "sprint":  sprint == 1,
            "camera":  np.array([CAM_ANGLES[pitch_idx],
                                 CAM_ANGLES[yaw_idx]], dtype=np.float32),
            "attack":  attack == 1,
        }
        return act_dict
