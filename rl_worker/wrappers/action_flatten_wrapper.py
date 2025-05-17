"""
ActionFlattenWrapper
--------------------
Dict / MultiDiscrete などの複雑な行動空間を Discrete にフラット化する簡単ラッパー。
"""
import gym
import numpy as np
from gym.spaces import Discrete, Dict


class ActionFlattenWrapper(gym.ActionWrapper):
    """Flattens a Dict action space into Discrete.

    Each key of the Dict must be Discrete( n ); the flattened index is computed
    with mixed-radix encoding.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        if not isinstance(env.action_space, Dict):
            raise TypeError("ActionFlattenWrapper expects Dict action space")

        self._subspaces = list(env.action_space.spaces.values())
        self._radix = np.cumprod([1] + [sp.n for sp in self._subspaces[:-1]])
        self.action_space = Discrete(int(np.prod([sp.n for sp in self._subspaces])))

    # --- wrap/unwrap --------------------------------------------------------
    def action(self, flat_idx: int):
        vals = {}
        for key, sp, r in zip(self.env.action_space.spaces.keys(),
                              self._subspaces, self._radix):
            vals[key] = int((flat_idx // r) % sp.n)
        return vals

    def reverse_action(self, action_dict):
        flat = 0
        for (val, r) in zip(action_dict.values(), self._radix):
            flat += int(val) * r
        return flat
