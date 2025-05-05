import gym

class ActionFlattenWrapper(gym.ActionWrapper):
    """
    MineRL の Dict ActionSpace → MultiDiscrete に変換
    """
    def __init__(self, env):
        super().__init__(env)
        self._keys = list(self.action_space.spaces.keys())
        sizes = [self.action_space.spaces[k].n for k in self._keys]
        self.action_space = gym.spaces.MultiDiscrete(sizes)

    def action(self, flat_action):
        # MultiDiscrete -> Dict に戻して環境へ渡す
        return {key: int(flat_action[i]) for i, key in enumerate(self._keys)}
