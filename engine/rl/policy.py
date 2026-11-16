#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Actor-critic network for one paintball model.

Two details are specific to this game:

* The shoot head is masked while the marker is on cooldown. At 20 Hz physics
  with a one-second cap, roughly 95% of trigger decisions cannot fire at all;
  left unmasked they contribute gradient and entropy that mean nothing, and
  they drown the ~45 real decisions an episode contains.
* Actor and critic have separate trunks, and the critic is fed the ungated
  state. The actor must live with not seeing through walls, which is what makes
  cover valuable; a value function given the same censored input would be
  estimating returns for states it cannot distinguish.
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn

# Local imports
from engine import config

logger = logging.getLogger(__name__)

HEAD_SIZES = (config.N_MOVE_ACTIONS, config.N_TURN_ACTIONS, config.N_SHOOT_ACTIONS)


def init_layer(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    """Orthogonal init; small gain on policy heads keeps the initial policy near uniform."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def build_trunk(obs_size: int, hidden: int) -> nn.Sequential:
    """Two-layer MLP trunk."""
    return nn.Sequential(
        init_layer(nn.Linear(obs_size, hidden), 2 ** 0.5), nn.ReLU(),
        init_layer(nn.Linear(hidden, hidden), 2 ** 0.5), nn.ReLU(),
    )


class ActorCritic(nn.Module):
    """Policy and value network for a single model in the roster."""

    def __init__(self, obs_size: int, hidden: int = 256):
        super().__init__()
        self.actor_trunk = build_trunk(obs_size, hidden)
        self.critic_trunk = build_trunk(obs_size, hidden)
        self.move_head = init_layer(nn.Linear(hidden, config.N_MOVE_ACTIONS), 0.01)
        self.turn_head = init_layer(nn.Linear(hidden, config.N_TURN_ACTIONS), 0.01)
        self.shoot_head = init_layer(nn.Linear(hidden, config.N_SHOOT_ACTIONS), 0.01)
        self.value_head = init_layer(nn.Linear(hidden, 1), 1.0)

    def logits(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.actor_trunk(obs)
        return self.move_head(features), self.turn_head(features), self.shoot_head(features)

    def value(self, privileged: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.critic_trunk(privileged)).squeeze(-1)


def distributions(net: ActorCritic, obs: torch.Tensor):
    """Return the three categorical distributions for a batch of observations."""
    move_logits, turn_logits, shoot_logits = net.logits(obs)
    return (torch.distributions.Categorical(logits=move_logits),
            torch.distributions.Categorical(logits=turn_logits),
            torch.distributions.Categorical(logits=shoot_logits))


@torch.no_grad()
def act(net: ActorCritic, obs: torch.Tensor, privileged: torch.Tensor,
        can_shoot: torch.Tensor, deterministic: bool = False,
        with_value: bool = True) -> Dict[str, torch.Tensor]:
    """Sample actions. The shoot head is bypassed entirely while on cooldown.

    `with_value` off skips the critic entirely, and the key is then absent rather
    than None -- a caller that wants it should fail where it asked, not where it
    reads. Only the rollout needs it; play that learns nothing, a demo match
    for one, would throw away a full forward pass over the privileged observation.
    """
    move_dist, turn_dist, shoot_dist = distributions(net, obs)
    if deterministic:
        move = move_dist.probs.argmax(dim=-1)
        turn = turn_dist.probs.argmax(dim=-1)
        shoot = shoot_dist.probs.argmax(dim=-1)
    else:
        move = move_dist.sample()
        turn = turn_dist.sample()
        shoot = shoot_dist.sample()

    shoot = torch.where(can_shoot, shoot, torch.zeros_like(shoot))
    log_prob = move_dist.log_prob(move) + turn_dist.log_prob(turn)
    log_prob = log_prob + torch.where(can_shoot, shoot_dist.log_prob(shoot),
                                      torch.zeros_like(log_prob))
    result = {"move": move, "turn": turn, "shoot": shoot, "log_prob": log_prob}
    if with_value:
        result["value"] = net.value(privileged)
    return result


def evaluate(net: ActorCritic, obs: torch.Tensor, privileged: torch.Tensor,
             move: torch.Tensor, turn: torch.Tensor, shoot: torch.Tensor,
             can_shoot: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Re-evaluate stored actions under the current parameters."""
    move_dist, turn_dist, shoot_dist = distributions(net, obs)
    zero = torch.zeros_like(move)

    log_prob = move_dist.log_prob(move) + turn_dist.log_prob(turn)
    log_prob = log_prob + torch.where(can_shoot, shoot_dist.log_prob(shoot), zero.float())

    entropy_move = move_dist.entropy()
    entropy_turn = turn_dist.entropy()
    entropy_shoot = torch.where(can_shoot, shoot_dist.entropy(), zero.float())
    return {
        "log_prob": log_prob,
        "value": net.value(privileged),
        "entropy_move": entropy_move,
        "entropy_turn": entropy_turn,
        "entropy_shoot": entropy_shoot,
    }


def resolve_device(preference: str = None) -> torch.device:
    """Pick the torch device, honouring TORCH_DEVICE but falling back gracefully."""
    preference = preference or config.TORCH_DEVICE
    if preference and preference != "auto":
        return torch.device(preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    pass


if __name__ == "__main__":
    main()
