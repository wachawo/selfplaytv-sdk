#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PPO update for a single model.

One of these runs per model, as independent learners sharing one self-play
population. Entropy is weighted per head because the three heads carry very
different amounts of decision: movement is chosen every step, the trigger only
about forty-five times per episode.
"""

import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

# Local imports
from engine import config
from engine.rl import policy

logger = logging.getLogger(__name__)


def anneal(start: float, end: float, progress: float) -> float:
    """Linear interpolation clamped to [0, 1] of progress."""
    t = min(max(progress, 0.0), 1.0)
    return start + (end - start) * t


def update(net: policy.ActorCritic, optimizer: torch.optim.Optimizer,
           batch: Dict[str, torch.Tensor], entropy_scale: float = 1.0,
           progress: float = 0.0) -> Dict[str, float]:
    """Run the PPO epochs over one model's rollout and return diagnostics."""
    total = batch["obs"].shape[0]
    minibatch = max(1, total // config.N_MINIBATCHES)
    indices = np.arange(total)

    entropy_decay = anneal(1.0, config.ENTROPY_FINAL_SCALE, progress) * entropy_scale
    learning_rate = anneal(config.LEARNING_RATE, config.LEARNING_RATE_FLOOR, progress)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "approx_kl": 0.0, "clip_fraction": 0.0, "updates": 0}

    for unused_epoch in range(config.PPO_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, total, minibatch):
            selection = torch.as_tensor(indices[start:start + minibatch], device=batch["obs"].device)
            if selection.numel() < 2:
                continue

            evaluation = policy.evaluate(
                net,
                batch["obs"][selection], batch["privileged"][selection],
                batch["move"][selection], batch["turn"][selection],
                batch["shoot"][selection], batch["can_shoot"][selection],
            )

            advantage = batch["advantage"][selection]
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            ratio = torch.exp(evaluation["log_prob"] - batch["log_prob"][selection])
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - config.CLIP_EPS, 1.0 + config.CLIP_EPS) * advantage
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = nn.functional.mse_loss(evaluation["value"], batch["return"][selection])

            entropy = (config.ENTROPY_MOVE * evaluation["entropy_move"].mean()
                       + config.ENTROPY_TURN * evaluation["entropy_turn"].mean()
                       + config.ENTROPY_SHOOT * evaluation["entropy_shoot"].mean())

            loss = policy_loss + config.VALUE_COEF * value_loss - entropy_decay * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), config.GRAD_CLIP)
            optimizer.step()

            with torch.no_grad():
                log_ratio = evaluation["log_prob"] - batch["log_prob"][selection]
                approx_kl = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio)
                clip_fraction = torch.mean(((ratio - 1.0).abs() > config.CLIP_EPS).float())

            stats["policy_loss"] += float(policy_loss.detach())
            stats["value_loss"] += float(value_loss.detach())
            stats["entropy"] += float(entropy.detach())
            stats["approx_kl"] += float(approx_kl)
            stats["clip_fraction"] += float(clip_fraction)
            stats["updates"] += 1

    count = max(stats["updates"], 1)
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
        stats[key] /= count
    stats["learning_rate"] = learning_rate
    return stats


def main():
    pass


if __name__ == "__main__":
    main()
