import gym
import numpy as np
from gym.spaces import Dict, MultiDiscrete, Box

BTN_KEYS = [
    "attack", "back", "forward", "jump",
    "left", "right", "sneak", "sprint"
]  # Discrete(2) → 0/1 ボタン

CAMERA_BINS = (-10, 0, 10)  # pitch / yaw を粗く3分割

class ActionFlattenWrapper(gym.ActionWrapper):
    """
    Convert MineRL Dict action to MultiDiscrete:
      [ btn0 … btn7,  camera_pitch_bin, camera_yaw_bin ]
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._orig_space: Dict = env.action_space

        # 8 buttons (0/1) + 2 camera axes (len(bins)) each
        n_bins = len(CAMERA_BINS)
        self.action_space = MultiDiscrete([2]*len(BTN_KEYS) + [n_bins, n_bins])

    def action(self, act):
        # act: NDArray shape (10,)
        d = {}
        # 8 buttons
        for i, k in enumerate(BTN_KEYS):
            d[k] = int(act[i])

        # camera
        pitch_bin = int(act[8])
        yaw_bin   = int(act[9])
        d["camera"] = np.array([
            CAMERA_BINS[pitch_bin],
            CAMERA_BINS[yaw_bin]
        ], dtype=np.float32)

        return d
