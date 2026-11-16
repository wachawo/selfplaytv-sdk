#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rollout storage, reward attribution and GAE.

Each model gets a rectangular (T, arenas) buffer, and a validity mask beside it.
The mask is what makes the shape safe: when the roster is larger than the
field a model holds a slot in only some arenas, and its rows elsewhere are
meaningless. When the roster is exactly the field the mask is all true.

Reward attribution is the subtle part. A paintball fired at 15 m lands about
0.6 s later; crediting the shooter at impact means the signal has to survive a
dozen steps of discounting before it reaches the decision that earned it. So a
landed hit is written back to the step on which the trigger was pulled.
"""

import logging
from typing import Dict

import numpy as np
import torch

# Local imports
from engine import config

logger = logging.getLogger(__name__)

class RolloutBuffer:
    """Fixed-size storage for one model's slice of the rollout."""

    def __init__(self, steps: int, arenas: int, obs_size: int, device: torch.device):
        self.steps = steps
        self.arenas = arenas
        self.device = device
        self.pos = 0

        shape = (steps, arenas)
        self.obs        = np.zeros(shape + (obs_size,), dtype=np.float32)
        self.privileged = np.zeros(shape + (obs_size,), dtype=np.float32)
        self.move       = np.zeros(shape, dtype=np.int64)
        self.turn       = np.zeros(shape, dtype=np.int64)
        self.shoot      = np.zeros(shape, dtype=np.int64)
        self.can_shoot  = np.zeros(shape, dtype=bool)
        self.log_prob   = np.zeros(shape, dtype=np.float32)
        self.value      = np.zeros(shape, dtype=np.float32)
        self.reward     = np.zeros(shape, dtype=np.float32)
        self.done       = np.zeros(shape, dtype=np.float32)
        # With a roster larger than the field, a model sits out some arenas. The
        # buffer stays rectangular and those cells are simply not learned from.
        self.valid      = np.zeros(shape, dtype=bool)

    def reset(self) -> None:
        self.pos = 0
        self.reward[:] = 0.0
        self.done[:] = 0.0
        self.valid[:] = False

    def full(self) -> bool:
        return self.pos >= self.steps


def add_reward(buffer: RolloutBuffer, step: int, arenas: np.ndarray, amount: np.ndarray) -> None:
    """Accumulate reward at an arbitrary earlier step; out-of-window credit is dropped."""
    if arenas.size == 0:
        return
    if step < 0 or step >= buffer.steps:
        return
    np.add.at(buffer.reward[step], arenas, amount)


def add_reward_scattered(buffer: RolloutBuffer, steps: np.ndarray, arenas: np.ndarray,
                         amount: np.ndarray) -> int:
    """Credit rewards at per-event step indices. Returns how many fell outside the window."""
    if arenas.size == 0:
        return 0
    inside = (steps >= 0) & (steps < buffer.steps)
    if inside.any():
        np.add.at(buffer.reward, (steps[inside], arenas[inside]), amount[inside])
    return int((~inside).sum())


def compute_gae(buffer: RolloutBuffer, last_value: np.ndarray,
                gamma: float = config.GAMMA, lam: float = config.GAE_LAMBDA):
    """Generalised advantage estimation over the stored rollout."""
    advantages = np.zeros_like(buffer.reward)
    running = np.zeros(buffer.arenas, dtype=np.float32)
    next_value = last_value

    for t in reversed(range(buffer.steps)):
        not_done = 1.0 - buffer.done[t]
        delta = buffer.reward[t] + gamma * next_value * not_done - buffer.value[t]
        running = delta + gamma * lam * not_done * running
        advantages[t] = running
        next_value = buffer.value[t]
    returns = advantages + buffer.value
    return advantages, returns


def flatten(buffer: RolloutBuffer, advantages: np.ndarray, returns: np.ndarray,
            device: torch.device) -> Dict[str, torch.Tensor]:
    """Flatten (T, arenas, ...) into a batch, keeping only the steps this model played."""
    keep = buffer.valid.reshape(-1)

    def pack(array, dtype):
        flat = array.reshape(-1, *array.shape[2:])
        return torch.as_tensor(flat[keep], dtype=dtype, device=device)

    return {
        "obs":        pack(buffer.obs, torch.float32),
        "privileged": pack(buffer.privileged, torch.float32),
        "move":       pack(buffer.move, torch.long),
        "turn":       pack(buffer.turn, torch.long),
        "shoot":      pack(buffer.shoot, torch.long),
        "can_shoot":  pack(buffer.can_shoot, torch.bool),
        "log_prob":   pack(buffer.log_prob, torch.float32),
        "value":      pack(buffer.value, torch.float32),
        "advantage":  pack(advantages, torch.float32),
        "return":     pack(returns, torch.float32),
    }


class ReturnNormalizer:
    """Running standard deviation of returns.

    With gamma at 0.997 and rewards spanning 0.02 to 2.0, unnormalised returns
    make the value loss dominate the policy gradient.
    """

    def __init__(self, epsilon: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, values: np.ndarray) -> None:
        batch_mean = float(values.mean())
        batch_var = float(values.var())
        batch_count = values.size

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta * delta * self.count * batch_count / total) / total
        self.count = total

    def std(self) -> float:
        return max(float(np.sqrt(self.var)), 1e-4)


def main():
    pass


if __name__ == "__main__":
    main()
