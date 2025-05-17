"""
ActionFlattenWrapper
====================
Dict アクション空間を 1 次元 Discrete に射影する。

* Discrete   -> そのまま
* MultiBinary/MultiDiscrete -> そのまま
* Box        -> MultiDiscrete へ量子化してから離散化
"""
from __future__ import annotations
import gym
import numpy as np
from gym.spaces import (
    Discrete,
    MultiBinary,
    MultiDiscrete,
    Box,
    Dict,
)


class ActionFlattenWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, box_bins: int = 11):
        """
        Parameters
        ----------
        env : gym.Env
        box_bins : int
            Box 空間を何分割で量子化するか（奇数が望ましい）
        """
        super().__init__(env)

        if not isinstance(env.action_space, Dict):
            raise TypeError("ActionFlattenWrapper expects a Dict action space")

        self.box_bins = box_bins
        self._keys: list[str] = []
        self._subspaces: list[Discrete | MultiDiscrete | MultiBinary] = []

        for k, sp in env.action_space.spaces.items():
            if isinstance(sp, Discrete):
                new_sp = sp
            elif isinstance(sp, MultiBinary):
                new_sp = MultiBinary(sp.n)
            elif isinstance(sp, MultiDiscrete):
                new_sp = sp
            elif isinstance(sp, Box):
                # [-x,x] → linspace で量子化し MultiDiscrete へ
                low, high = sp.low, sp.high
                bins = np.linspace(low, high, box_bins)
                nvec = np.full(sp.shape, box_bins, dtype=int)
                new_sp = MultiDiscrete(nvec)
                # 逆変換で使用
                self.__dict__.setdefault("_box_bins", {})[k] = bins
            else:
                raise TypeError(f"Unsupported sub-space type: {type(sp)} (key={k})")

            self._keys.append(k)
            self._subspaces.append(new_sp)

        # 変換後の各サブ空間サイズ
        self._sizes = [
            int(np.prod(sp.n if isinstance(sp, Discrete) else sp.nvec))
            for sp in self._subspaces
        ]
        self._radix = np.cumprod([1] + self._sizes[:-1])
        self.action_space = Discrete(int(np.prod(self._sizes)))

    # -------- Flat index → Dict --------
    def action(self, flat_idx: int):
        d = {}
        for k, sp, r in zip(self._keys, self._subspaces, self._radix):
            local = (flat_idx // r) % self._size(sp)
            d[k] = self._unflatten_component(k, sp, local)
        return d

    # -------- Dict → Flat index --------
    def reverse_action(self, d: dict):
        idx = 0
        for k, sp, r in zip(self._keys, self._subspaces, self._radix):
            local = self._flatten_component(k, sp, d[k])
            idx += local * r
        return int(idx)

    # -------- helpers --------
    @staticmethod
    def _size(sp):
        return sp.n if isinstance(sp, Discrete) else int(np.prod(sp.nvec))

    def _flatten_component(self, k, sp, v):
        if isinstance(sp, Discrete):
            return int(v)
        elif isinstance(sp, MultiBinary):
            return int("".join(map(str, v.astype(int))), 2)
        elif isinstance(sp, MultiDiscrete):
            if k in getattr(self, "_box_bins", {}):
                # Box→MultiDiscrete の場合 v は実数 → 最近傍ビンへ
                bins = self._box_bins[k]
                v = np.digitize(v, bins) - 1  # 0-based
            return np.ravel_multi_index(v, sp.nvec)
        else:  # pragma: no cover
            raise TypeError

    def _unflatten_component(self, k, sp, idx):
        if isinstance(sp, Discrete):
            return int(idx)
        elif isinstance(sp, MultiBinary):
            bits = np.array(list(np.binary_repr(idx, width=sp.n)), dtype=int)
            return bits
        elif isinstance(sp, MultiDiscrete):
            multi = np.array(np.unravel_index(idx, sp.nvec), dtype=int)
            if k in getattr(self, "_box_bins", {}):
                bins = self._box_bins[k]
                return bins[multi]  # 離散→実数
            return multi
        else:  # pragma: no cover
            raise TypeError
