#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training curriculum.

Arena size is the strongest single lever available. On a 16x11 m field an enemy
subtends roughly 6 degrees instead of 3, which quadruples the hit rate of early
random play and gives the policy something to learn from immediately.

Team size is not one of the levers here: every phase is played at the team
size the caller fields, and the observation layout is the same in all of them.
"""

import logging
from typing import Any, Dict, List

# Local imports
from engine import config

logger = logging.getLogger(__name__)

# capture_seconds is annealed the same way the arena is grown. Holding a strip
# for the full five seconds is 50 control steps of standing still in the enemy
# spawn, which random exploration essentially never produces, so the rule would
# go the whole run without once being experienced. A short hold early is
# something a wandering policy does stumble into, and once it has, the reward
# tells it what happened.
PHASES: List[Dict[str, Any]] = [
    {
        "name": "close quarters",
        "until": 10_000_000,
        "width": 16.0, "height": 11.0, "obstacles": 4,
        "match_seconds": 20.0,
        "capture_seconds": 1.5,
        "autofire": True,
    },
    {
        "name": "trigger handover",
        "until": 30_000_000,
        "width": 20.0, "height": 14.0, "obstacles": 8,
        "match_seconds": 30.0,
        "capture_seconds": 2.5,
        "autofire": False,
    },
    {
        "name": "open field",
        "until": 80_000_000,
        "width": 26.0, "height": 17.0, "obstacles": 14,
        "match_seconds": 45.0,
        "capture_seconds": 3.5,
        "autofire": False,
    },
    {
        "name": "full arena",
        "until": None,
        "width": config.ARENA_W, "height": config.ARENA_H, "obstacles": config.N_OBSTACLES,
        "match_seconds": config.MATCH_SECONDS,
        "capture_seconds": config.CAPTURE_SECONDS,
        "autofire": False,
    },
]


def phase_for(steps: int) -> Dict[str, Any]:
    """Return the curriculum phase covering this many elapsed agent-steps.

    Episode length is clamped to the configured match limit: the ramp is written
    for the default limit, and lowering MATCH_SECONDS must not leave an earlier
    phase running longer than a real match ever can.
    """
    chosen = PHASES[-1]
    for phase in PHASES:
        if phase["until"] is None or steps < phase["until"]:
            chosen = phase
            break
    if chosen["match_seconds"] <= config.MATCH_SECONDS:
        return chosen
    return {**chosen, "match_seconds": config.MATCH_SECONDS}


def phase_index(steps: int) -> int:
    """Index of the active phase, used to detect transitions."""
    for i, phase in enumerate(PHASES):
        if phase["until"] is None or steps < phase["until"]:
            return i
    return len(PHASES) - 1


def aim_shaping_weight(steps: int) -> float:
    """Potential-based aim shaping, annealed towards a floor as the policy matures.

    Towards a floor and not to zero: a hit lands only about once every few
    hundred player-steps, so with this term switched off the only dense thing
    left in the reward is the capture potential, which points at the enemy
    strip and never anneals. Aiming was left with nothing continuous to learn
    from at exactly the point the arena grew and shots got longer.
    """
    if steps <= config.AIM_ANNEAL_FROM:
        return config.AIM_SHAPING_K
    if steps >= config.AIM_ANNEAL_TO:
        return config.AIM_SHAPING_FLOOR
    span = config.AIM_ANNEAL_TO - config.AIM_ANNEAL_FROM
    share = (steps - config.AIM_ANNEAL_FROM) / span
    return config.AIM_SHAPING_K + (config.AIM_SHAPING_FLOOR - config.AIM_SHAPING_K) * share


def shot_penalty(steps: int) -> float:
    """Wasted-shot penalty, held at zero until shooting is established.

    Charged too early it turns the expected value of a speculative shot negative
    and both sides quietly stop firing.
    """
    if steps < config.SHOT_PENALTY_FROM:
        return 0.0
    return config.SHOT_PENALTY


def describe(steps: int) -> str:
    """One-line summary of the active phase for the training log."""
    phase = phase_for(steps)
    return (f"{phase['name']} ({phase['width']:.0f}x{phase['height']:.0f}m, "
            f"{phase['obstacles']} obstacles, {phase['match_seconds']:.0f}s, "
            f"hold {phase['capture_seconds']:.1f}s, "
            f"autofire={'on' if phase['autofire'] else 'off'})")


def main():
    pass


if __name__ == "__main__":
    main()
