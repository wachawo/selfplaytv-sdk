#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The PPO trainer: independent learners, one network each, sharing the arenas.

Every roster model has a network, an optimiser, a rollout buffer and a return
normaliser of its own. The roster can be larger than the field: an arena fields
two teams and the rest of the roster sits that arena out, so buffers stay
rectangular and carry a validity mask. The steps a model spent off the field
are simply not learned from, and its rewards are masked so it is never paid for
somebody else's play. When the roster exactly fills the field the mask is all
true, which is the ordinary case of nobody sitting out.

This module is the rollout, the PPO update and the checkpoint, and nothing
around them: it has no loop and no command line. sdk/train.py drives it and
decides the three things this module only asks for - who stands where in each
arena (Trainer.draw_lineups), which step count the schedules are read against
(Trainer.read_clock) and which models the gradient reaches (Trainer.learning).

The observation layout is fixed to config.MAX_TEAM_SIZE, which is what keeps a
checkpoint loadable anywhere that layout is the same.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Local imports
from engine import config
from engine.env import arena, obs, sim
from engine.rl import buffer as buffer_mod
from engine.rl import curriculum, persist, policy, ppo

logger = logging.getLogger(__name__)

TEAM_BLEND_SHARE = 0.3


class Trainer:
    """Container for the whole training run."""

    def __init__(self, arenas: int, device: torch.device, seed: int = 0,
                 team_size: Optional[int] = None):
        self.device = device
        self.arenas = arenas
        self.rng = np.random.default_rng(seed)
        self.n_models = config.N_MODELS
        # Fixed layout, sized for config.MAX_TEAM_SIZE whatever the team size on
        # the field: it is what every checkpoint's first layer is built for.
        self.obs_size = obs.observation_size()
        self.aim_at = obs.aim_offset()
        self.capture_at = obs.capture_offset()

        self.phase_idx = -1
        # One map theme by key, or None for all of them. Read by apply_phase
        # when it builds the arena pool, so the caller sets it before that.
        self.theme = None
        # Carried from the manifest a run was restored from into the one it
        # saves, untouched: the checkpoint format has the field, and a run that
        # dropped it could not be told from one written by an older build.
        self.generation = 0
        self.state: sim.SimState = None
        self.pool: arena.ArenaPool = None

        self.nets: List[policy.ActorCritic] = []
        self.optimizers: List[torch.optim.Optimizer] = []
        for i in range(self.n_models):
            personality = config.personality(i)
            net = policy.ActorCritic(self.obs_size, personality["hidden"]).to(device)
            self.nets.append(net)
            self.optimizers.append(torch.optim.Adam(net.parameters(),
                                                    lr=config.LEARNING_RATE, eps=config.ADAM_EPS))

        self.buffers = [buffer_mod.RolloutBuffer(config.ROLLOUT_STEPS, arenas, self.obs_size, device)
                        for unused in range(self.n_models)]
        self.normalizers = [buffer_mod.ReturnNormalizer() for unused in range(self.n_models)]

        shape_for_mode(self, team_size if team_size is not None else config.TEAM_SIZE)

        self.steps_done = 0
        # PER-MODEL LEARNING STEPS, which steps_done cannot answer: that one
        # counts every agent-step on the field, and dividing it by the roster
        # credits every model with the same figure whether or not the gradient
        # reached it. With one team learning and another frozen, the divided
        # figure would say both had learned the same amount.
        #
        # Counted in the agent-steps a model actually stood on the field for,
        # which is what `plays` records: with a roster larger than the field
        # only some of it plays each arena, so an equal split would be wrong
        # even with everybody learning. Summed over the roster the rollout's
        # play is exactly steps_done's, because plays has one true per slot per
        # arena; learn() then credits it only to the models that learned.
        self.steps_by_model = np.zeros(self.n_models, dtype=np.int64)
        # What each model PLAYED in the rollout being collected. Kept apart from
        # the total because the two are decided at different moments: playing is
        # known while the rollout runs, and whether it counts as LEARNING is not
        # known until learn() reads who learns. Zeroed at the top of every
        # rollout.
        self.rollout_steps = np.zeros(self.n_models, dtype=np.int64)
        # THE FIELD: the roster indices the per-model loops walk, which is the
        # whole roster - the caller narrows the roster to the teams that play
        # before this is built. Kept as an array because it is indexed into
        # buffers, and iterated through .tolist() so the hot loops hand plain
        # ints to everything that indexes a list with them.
        self.field = np.arange(self.n_models)
        self.updates = 0
        # THE CALLER'S THREE DECISIONS, set after construction and before the
        # first apply_phase. This module asks for them and never makes them.
        #
        # draw_lineups(trainer, rows) puts two teams on the field of each of the
        # given arenas by writing model_of_slot, plays and slot_of_model for
        # those rows; see assign_slots for what the three arrays have to hold.
        #
        # read_clock(trainer) answers the step count every schedule is read
        # against: the curriculum phase, the aim shaping, the shot penalty and
        # PPO's annealing. Not steps_done, which counts frozen opponents' play
        # as well: see curriculum_clock.
        #
        # learning holds the nicknames the gradient reaches; EMPTY MEANS EVERY
        # MODEL, so a caller that wants nothing to learn must not call learn().
        self.draw_lineups = None
        self.read_clock = None
        self.learning: List[str] = []
        # Called as on_step(trainer, done, winner) once per control step of
        # collect_rollout, after the round endings are known and before
        # finish_episodes resets the finished arenas - the only moment a caller
        # can still read how they ended. Optional: None means nobody is asking.
        self.on_step = None
        self.started = time.time()
        # Steps already done when this process started, so the rate reports
        # this run's throughput and not a resumed counter divided by seconds.
        self.steps_at_start = 0
        self.tally = new_tally(self.n_models)


def shape_for_mode(trainer: Trainer, team_size: int) -> None:
    """Size everything that follows the field to a team size.

    Only the arrays indexed by slot are built here. The networks, the optimiser
    moments and the rollout buffers are sized by the observation layout, which
    does not move with the team size.
    """
    if trainer.n_models < team_size * 2:
        raise ValueError(f"{trainer.n_models} models cannot field {team_size * 2} players; "
                         f"the roster needs at least that many models")

    trainer.team_size = int(team_size)
    trainer.n_players = trainer.team_size * 2
    trainer.mates, trainer.foes = obs.team_index_tables(trainer.team_size)
    # A roster larger than the field means a model holds a slot in only some
    # arenas; `plays` says where, and the slot index is meaningless elsewhere.
    # With two teams the roster is exactly the field and `plays` is all true.
    trainer.slot_of_model = np.zeros((trainer.arenas, trainer.n_models), dtype=np.int64)
    trainer.model_of_slot = np.zeros((trainer.arenas, trainer.n_players), dtype=np.int64)
    trainer.plays = np.zeros((trainer.arenas, trainer.n_models), dtype=bool)
    trainer.prev_phi = np.zeros((trainer.arenas, trainer.n_players), dtype=np.float32)


def new_tally(n_models: int) -> Dict[str, np.ndarray]:
    """Fresh accumulator for the per-model counters of the scoreboard."""
    return {
        "points": np.zeros(n_models, dtype=np.int64),
        "matches": np.zeros(n_models, dtype=np.int64),
        "hits": np.zeros(n_models, dtype=np.int64),
        "deaths": np.zeros(n_models, dtype=np.int64),
        "shots": np.zeros(n_models, dtype=np.int64),
        "wins": np.zeros(n_models, dtype=np.int64),
        "losses": np.zeros(n_models, dtype=np.int64),
        "draws": np.zeros(n_models, dtype=np.int64),
        # Captures completed while standing on the strip. Counted because a
        # rule nothing counts can go a whole run without being learned, and
        # nobody would notice.
        "captures": np.zeros(n_models, dtype=np.int64),
        "reward": np.zeros(n_models, dtype=np.float64),
    }


def assign_slots(trainer: Trainer, rows: np.ndarray) -> None:
    """Have the caller draw which models take the field in the given arenas, and where.

    Called when the arenas are built and again for every arena that restarts.
    What the caller's draw_lineups has to leave behind, for each row: in
    model_of_slot the model standing in each slot, slots 0..team_size-1 being
    one end of the field and the rest the other; in plays, true for exactly
    those models; in slot_of_model, each of those models' slot. A model off the
    field keeps a slot index of 0, which is never read, because its buffer cells
    are marked invalid. One model cannot stand in two slots of one arena:
    slot_of_model holds one slot per arena and model.
    """
    trainer.draw_lineups(trainer, rows)


def curriculum_clock(trainer: Trainer) -> int:
    """The step count every training schedule in this file is read against.

    The caller's answer (Trainer.read_clock), and deliberately neither
    steps_done nor the whole field's own steps. The schedules - the curriculum
    phase, the aim shaping, the shot penalty and PPO's annealing - all measure
    how far the networks that are LEARNING have come. steps_done counts the
    frozen opponents' play as well, and belongs to the run, which may be far
    younger or far older than the networks in it. The field's own steps would
    bring a veteran opponent's hundreds of millions with them: five fresh
    networks would start on the full arena with the learning rate at its floor
    and the entropy at its final fifth, everything a beginner needs annealed
    away before its first update.

    steps_done itself is left alone: the checkpoint guard and the log line are
    about the run, not about who is learning in it.
    """
    return int(trainer.read_clock(trainer))


def apply_phase(trainer: Trainer, phase: Dict[str, Any]) -> None:
    """Rebuild the arena pool and simulation state for a curriculum phase."""
    trainer.pool = arena.build_pool(
        128, width=phase["width"], height=phase["height"],
        n_obstacles=phase["obstacles"], theme_key=trainer.theme,
        seed=int(trainer.rng.integers(1 << 30)),
    )
    keep_step = trainer.state.step_idx.copy() if trainer.state is not None else None
    trainer.state = sim.SimState(trainer.arenas, trainer.pool, trainer.team_size,
                                 seed=int(trainer.rng.integers(1 << 30)))
    trainer.state.capture_needed = float(phase["capture_seconds"])
    if keep_step is not None:
        trainer.state.step_idx[:] = keep_step
    sim.reset_arenas(trainer.state, np.ones(trainer.arenas, dtype=bool))
    assign_slots(trainer, np.arange(trainer.arenas))
    trainer.prev_phi[:] = 0.0
    logger.info(f"Phase: {curriculum.describe(curriculum_clock(trainer))}, "
                f"mode: {trainer.team_size}v{trainer.team_size}, "
                f"maps: {trainer.theme or 'all themes'}")


def match_ticks_for(phase: Dict[str, Any]) -> int:
    """Episode length in physics ticks for a phase."""
    return int(round(phase["match_seconds"] / config.DT))


def real_ticks_for(phase: Dict[str, Any]) -> int:
    """Hard ceiling on the real elapsed ticks of one episode of a phase.

    The phase limit above is measured on the match clock, which stops while a
    capture is running and can therefore be held still for as long as an
    attacker keeps stepping in and out of the strip. Left at that, an arena
    could run an episode that never terminates, and a rollout buffer that never
    sees a `done` never pays out a terminal result for it. The same reasoning
    and the same factor as config.MATCH_TICKS_MAX, applied to the shorter
    episodes the curriculum uses early on.
    """
    return int(round(match_ticks_for(phase) * config.MATCH_OVERTIME))


def gather_by_model(values: np.ndarray, slot_of_model: np.ndarray) -> np.ndarray:
    """Reorder a per-slot array (N, P, ...) into per-model order (M, N, ...)."""
    rows = np.arange(values.shape[0])[:, None]
    picked = values[rows, slot_of_model]
    return np.swapaxes(picked, 0, 1)


def scatter_to_slots(per_model: np.ndarray, model_of_slot: np.ndarray) -> np.ndarray:
    """Inverse of gather_by_model: (M, N) actions back into (N, P) slot order.

    Driven by which model holds each slot, not by each model's slot: with a
    roster larger than the field the benched models have no slot, and writing
    by their placeholder index would overwrite slot 0.
    """
    rows = np.arange(model_of_slot.shape[0])[:, None]
    return per_model[model_of_slot, rows]


def aim_potential(observations: np.ndarray, aim_at: int, weight: float) -> np.ndarray:
    """Potential-based aim shaping term, phi(s), shape (N, P).

    Potential-based shaping provably leaves the optimal policy unchanged, so it
    cannot bribe the agent into standing in the open staring at enemies -- which
    is exactly what a plain "small aim error is good" bonus would do.
    """
    if weight <= 0.0:
        return np.zeros(observations.shape[:2], dtype=np.float32)
    cos_error = observations[..., aim_at + obs.AIM_COS]
    has_target = observations[..., aim_at + obs.AIM_HAS_TARGET]
    # Faded out past the range limit, because tracking a target the paint cannot
    # reach is not the skill this term exists to teach. AIM_FLIGHT is scaled by
    # the projectile lifetime, so 1.0 is the limit itself and 2.0 is twice it;
    # fading between the two rather than cutting at 1.0 keeps phi continuous, so
    # crossing the boundary does not fire a spike of shaping reward.
    reach = np.clip(2.0 - observations[..., aim_at + obs.AIM_FLIGHT], 0.0, 1.0)
    return (-weight * (1.0 - cos_error) * has_target * reach).astype(np.float32)


# How much of the progress term is the player's own accrual rather than the best
# on its team. Both matter: the team term is what makes a defender care about a
# capture it is not carrying, the own term is what makes stepping out cost the
# player who steps out.
CAPTURE_OWN_SHARE = 0.5

# Largest return that can be reached from the reward scale in use. The per-step
# rewards are a few units at most and gamma is below one, so anything past this
# is corruption rather than an unusually good episode.
RETURN_SANITY_LIMIT = 5_000.0

# Largest magnitude an observation feature can legitimately carry. Every feature
# in the vector is normalised by construction -- a ratio, a sine, a clipped
# distance or a flag -- and the widest of them is the velocity clip at 2.0, so
# this sits five times above anything the game can produce.
OBS_SANITY_LIMIT = 10.0


def capture_potential(observations: np.ndarray, capture_at: int, weight: float) -> np.ndarray:
    """Potential term for holding ground, phi(s), shape (N, P).

    Three terms, all read straight out of the capture block.

    Progress: how far my side has carried a capture against how far the enemy
    has carried theirs. Blended with my OWN progress, because the team-best term
    alone is flat for a player who walks out of the strip while a teammate is
    still standing in it -- leaving costs that player nothing, which is exactly
    the behaviour this is supposed to price.

    Approach: how close I am to the strip I have to take. Without it the whole
    potential is identically zero everywhere outside the objective, so its
    gradient is zero too and nothing anywhere on the field points at the thing
    the rule is about. Reaching the strip was left entirely to chance, and
    holding it for five seconds by chance is not a thing that happens.

    Potential-based like the aim term, so none of it can invent a strategy of
    its own -- it changes how fast the rule is learned, not what the optimal
    play is.
    """
    if weight <= 0.0:
        return np.zeros(observations.shape[:2], dtype=np.float32)
    mine = observations[..., capture_at + obs.CAP_TEAM_BEST]
    own = observations[..., capture_at + obs.CAP_OWN]
    theirs = observations[..., capture_at + obs.CAP_FOE_BEST]
    approach = 1.0 - observations[..., capture_at + obs.CAP_ATK_DIST]
    progress = CAPTURE_OWN_SHARE * own + (1.0 - CAPTURE_OWN_SHARE) * mine
    return (weight * (progress - theirs + config.CAPTURE_APPROACH_K * approach)).astype(np.float32)


def clamp_observations(*arrays: np.ndarray) -> None:
    """Pull corrupt observation features back into range, in place.

    One bad feature is enough to cost a run. The critic is a linear map on its
    input, so a feature arriving twelve orders of magnitude out of range leaves
    the value head at that same scale; GAE carries it into the neighbouring
    return, and the value loss squares it. Nothing downstream can tell that is
    wrong, because a return of 1e10 is an ordinary float -- which is exactly why
    it has to be caught where the number is still recognisable as impossible.

    Clamping rather than dropping keeps the step: every other feature of that
    player is still the truth, and a rollout is not worth discarding over one of
    them. It is logged as a warning every time, because a clamped observation is
    an upstream fault and not a normal event.
    """
    for array in arrays:
        out = ~(np.abs(array) <= OBS_SANITY_LIMIT)
        if not out.any():
            continue
        worst = np.abs(np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0))
        column = int(np.unravel_index(worst.argmax(), worst.shape)[-1])
        logger.warning(f"Observation out of range at {int(out.sum())} cells, worst "
                       f"{worst.max():.3g} in feature column {column}; clamped to "
                       f"+/-{OBS_SANITY_LIMIT:g}")
        np.nan_to_num(array, nan=0.0, posinf=OBS_SANITY_LIMIT,
                      neginf=-OBS_SANITY_LIMIT, copy=False)
        np.clip(array, -OBS_SANITY_LIMIT, OBS_SANITY_LIMIT, out=array)


def collect_rollout(trainer: Trainer, phase: Dict[str, Any]) -> Dict[str, float]:
    """Fill every model's buffer with ROLLOUT_STEPS of self-play."""
    state = trainer.state
    n, p = trainer.arenas, trainer.n_players
    # This rollout's play, counted from zero. It is credited to the models that
    # LEARN from it, which learn() decides after this returns.
    trainer.rollout_steps[:] = 0
    clock = curriculum_clock(trainer)
    aim_weight = curriculum.aim_shaping_weight(clock)
    penalty = curriculum.shot_penalty(clock)
    ticks_limit = match_ticks_for(phase)
    real_limit = real_ticks_for(phase)
    # Plain ints, once per rollout, for the per-model loop below.
    field = trainer.field.tolist()

    # Every buffer starts the rollout with its valid mask cleared, so a cell is
    # valid only where this rollout's play marks it. Three small fills each.
    for buf in trainer.buffers:
        buf.reset()

    dropped_credit = 0
    actor_obs, critic_obs, unused_vis = obs.build_pair(state, trainer.mates, trainer.foes)
    clamp_observations(actor_obs, critic_obs)

    for t in range(config.ROLLOUT_STEPS):
        can_shoot = (state.cooldown <= 0) & state.alive

        obs_by_model = gather_by_model(actor_obs, trainer.slot_of_model)
        critic_by_model = gather_by_model(critic_obs, trainer.slot_of_model)
        shoot_by_model = gather_by_model(can_shoot, trainer.slot_of_model)

        moves = np.zeros((trainer.n_models, n), dtype=np.int64)
        turns = np.zeros((trainer.n_models, n), dtype=np.int64)
        shoots = np.zeros((trainer.n_models, n), dtype=np.int64)

        for m in field:
            buf = trainer.buffers[m]
            # Only the arenas this model is actually in. Across the roster this
            # still adds up to one forward per slot on the field, so a bigger
            # roster costs no extra inference.
            playing = np.nonzero(trainer.plays[:, m])[0]
            buf.valid[t] = trainer.plays[:, m]
            if playing.size == 0:
                continue

            model_obs = torch.as_tensor(obs_by_model[m][playing], device=trainer.device)
            model_critic = torch.as_tensor(critic_by_model[m][playing], device=trainer.device)
            model_can_shoot = torch.as_tensor(shoot_by_model[m][playing], device=trainer.device)

            out = policy.act(trainer.nets[m], model_obs, model_critic, model_can_shoot)
            moves[m, playing] = out["move"].cpu().numpy()
            turns[m, playing] = out["turn"].cpu().numpy()
            shoots[m, playing] = out["shoot"].cpu().numpy()

            buf.obs[t, playing] = obs_by_model[m][playing]
            buf.privileged[t, playing] = critic_by_model[m][playing]
            buf.move[t, playing] = moves[m, playing]
            buf.turn[t, playing] = turns[m, playing]
            buf.shoot[t, playing] = shoots[m, playing]
            buf.can_shoot[t, playing] = shoot_by_model[m][playing]
            buf.log_prob[t, playing] = out["log_prob"].cpu().numpy()
            buf.value[t, playing] = out["value"].cpu().numpy()

        move = scatter_to_slots(moves, trainer.model_of_slot)
        turn = scatter_to_slots(turns, trainer.model_of_slot)
        shoot = scatter_to_slots(shoots, trainer.model_of_slot).astype(bool)

        step_base = int(state.step_idx[0])
        result = sim.step(state, move, turn, shoot, autofire=phase["autofire"])
        dropped_credit += award_hits(trainer, result["hits"], t, step_base)

        # Shots are tallied as they are fired, matching how hits are counted; a
        # per-episode readout would lag hits and push accuracy above 100%.
        fired_by_model = (gather_by_model(result["fired"], trainer.slot_of_model)
                          * trainer.plays.T).sum(axis=1)
        trainer.tally["shots"] += fired_by_model

        if penalty < 0.0:
            charge_shots(trainer, result["fired"], t, penalty)

        # Both clocks, for the reason spelled out in real_ticks_for: the match
        # clock decides the normal ending and stops during a capture, the real
        # tick counter guarantees there is an ending at all.
        done = result["done"] | (state.tick >= ticks_limit) | (state.real_tick >= real_limit)
        if trainer.on_step is not None:
            trainer.on_step(trainer, done, result["winner"])
        if done.any():
            finish_episodes(trainer, done, result["winner"], t)

        # One observation pass per step, taken after any restart so the next step
        # sees fresh spawns. Arenas that just restarted get no shaping this step:
        # the difference across an episode boundary is meaningless, and they have
        # already been paid the terminal team reward.
        actor_obs, critic_obs, unused_vis = obs.build_pair(state, trainer.mates, trainer.foes)
        clamp_observations(actor_obs, critic_obs)
        phi = (aim_potential(actor_obs, trainer.aim_at, aim_weight)
               + capture_potential(actor_obs, trainer.capture_at, config.CAPTURE_SHAPING_K))
        shaping = config.GAMMA * phi - trainer.prev_phi
        shaping[done] = 0.0
        trainer.prev_phi = phi
        add_slot_rewards(trainer, shaping, t)

        trainer.steps_done += n * p
        # The same agent-steps, split by who was actually standing in them. One
        # true per slot per arena, so this sums across models to exactly the
        # n * p above and the two counters can never drift apart.
        trainer.rollout_steps += trainer.plays.sum(axis=0)

    return {"dropped_credit": dropped_credit}


def add_slot_rewards(trainer: Trainer, per_slot: np.ndarray, t: int) -> None:
    """Add a (N, P) reward array into every model's buffer at step t.

    Masked by who is on the field: a benched model's gathered value is whatever
    sat in slot 0, and paying it out would be crediting somebody else's play.
    """
    per_model = gather_by_model(per_slot, trainer.slot_of_model)
    for m in trainer.field.tolist():
        trainer.buffers[m].reward[t] += per_model[m] * trainer.plays[:, m]


def award_hits(trainer: Trainer, hits: Dict[str, np.ndarray], t: int, step_base: int) -> int:
    """Credit hits to the shooter at the step the trigger was pulled.

    The victim's penalty stays at impact, which is genuinely when its situation
    went wrong, but the shooter's reward belongs to the decision that earned it.
    """
    if hits["arena"].size == 0:
        return 0

    arenas = hits["arena"]
    shooters = hits["shooter"]
    victims = hits["victim"]
    fire_steps = hits["fire_step"] - step_base + t

    shooter_models = trainer.model_of_slot[arenas, shooters]
    victim_models = trainer.model_of_slot[arenas, victims]
    dropped = 0

    # A hit on an intruder in our own strip is priced separately, level with the
    # ordinary hit unless REWARD_DEFENCE says otherwise. It carried a premium and
    # no longer does: that made the easiest shot in the game the best paid.
    defence = hits["defence"]
    payout = np.where(defence, config.REWARD_DEFENCE, config.REWARD_HIT)

    for m in trainer.field.tolist():
        buf = trainer.buffers[m]
        shot = shooter_models == m
        if shot.any():
            dropped += buffer_mod.add_reward_scattered(
                buf, fire_steps[shot], arenas[shot], payout[shot].astype(np.float32))
        hurt = victim_models == m
        if hurt.any():
            buffer_mod.add_reward(buf, t, arenas[hurt],
                                  np.full(int(hurt.sum()), config.REWARD_HIT_TAKEN, dtype=np.float32))

    award_teammates(trainer, arenas, shooters, victims, fire_steps, t)
    trainer.tally["hits"] += np.bincount(shooter_models, minlength=trainer.n_models)
    trainer.tally["deaths"] += np.bincount(victim_models, minlength=trainer.n_models)
    trainer.tally["points"] += (np.bincount(shooter_models[~defence], minlength=trainer.n_models)
                                * config.POINTS_PER_HIT)
    trainer.tally["points"] += (np.bincount(shooter_models[defence], minlength=trainer.n_models)
                                * config.POINTS_PER_DEFENCE)
    return dropped


def award_teammates(trainer: Trainer, arenas: np.ndarray, shooters: np.ndarray,
                    victims: np.ndarray, fire_steps: np.ndarray, t: int) -> None:
    """Spread a share of every hit across the squad.

    Without this the players on a side are independent duellists: nothing in the
    reward tells them that a teammate's success is their success too.
    """
    team = trainer.state.team
    share = np.float32(TEAM_BLEND_SHARE)
    for k in range(arenas.size):
        row = arenas[k]
        for slot in range(trainer.n_players):
            if team[slot] == team[shooters[k]] and slot != shooters[k]:
                model = trainer.model_of_slot[row, slot]
                buffer_mod.add_reward_scattered(
                    trainer.buffers[model], np.array([fire_steps[k]]), np.array([row]),
                    np.array([share * config.REWARD_HIT], dtype=np.float32))
            elif team[slot] == team[victims[k]] and slot != victims[k]:
                model = trainer.model_of_slot[row, slot]
                buffer_mod.add_reward(trainer.buffers[model], t, np.array([row]),
                                      np.array([share * config.REWARD_HIT_TAKEN], dtype=np.float32))


def charge_shots(trainer: Trainer, fired: np.ndarray, t: int, penalty: float) -> None:
    """Small cost per shot, enabled only once shooting is well established."""
    per_model = gather_by_model(fired, trainer.slot_of_model)
    for m in trainer.field.tolist():
        trainer.buffers[m].reward[t] += per_model[m] * trainer.plays[:, m] * penalty


def finish_episodes(trainer: Trainer, done: np.ndarray, winner: np.ndarray, t: int) -> None:
    """Pay out the team result, record the scoreboard, restart those arenas."""
    state = trainer.state
    rows = np.nonzero(done)[0]
    field = trainer.field.tolist()
    # Read here, at the top, because reset_arenas below wipes the capture latch
    # and revives the dead, and the ending can no longer be told after that.
    # Charged to every slot of both sides, as REWARD_DRAW is, so waiting for the
    # clock costs something even on a round won on survivors.
    reasons = sim.end_reasons(state, rows)
    timed_out = np.where(reasons == sim.REASON_TIME, config.REWARD_TIMEOUT, 0.0)

    for slot in range(trainer.n_players):
        team = int(state.team[slot])
        models = trainer.model_of_slot[rows, slot]
        outcome = winner[rows]
        won = outcome == team
        lost = (outcome >= 0) & ~won

        # Covering a capture pays too, but less than carrying it, and only to
        # the players who were not on the strip: the holders draw REWARD_CAPTURE
        # below and must not collect both.
        captured = state.capture_winner[rows] == team
        held = state.capture_credit[rows, slot]
        payout = np.where(won, config.REWARD_WIN,
                          np.where(lost, config.REWARD_LOSS, config.REWARD_DRAW))
        payout = payout + np.where(captured & ~held, config.REWARD_CAPTURE_COVER, 0.0)
        payout = payout + timed_out
        for m in field:
            picked = models == m
            if picked.any():
                buf = trainer.buffers[m]
                buffer_mod.add_reward(buf, t, rows[picked], payout[picked].astype(np.float32))
                buf.done[t, rows[picked]] = 1.0

        # Being on the objective when it falls is the act the bonus buys, so it
        # is paid per player standing there and not per member of the winning
        # side. capture_credit was latched on the tick the strip fell.
        credited = held
        if credited.any():
            bonus = np.where(credited, config.REWARD_CAPTURE, 0.0)
            for m in field:
                picked = (models == m) & credited
                if picked.any():
                    buffer_mod.add_reward(trainer.buffers[m], t, rows[picked],
                                          bonus[picked].astype(np.float32))
            trainer.tally["points"] += (np.bincount(models[credited], minlength=trainer.n_models)
                                        * config.POINTS_PER_CAPTURE)
            trainer.tally["captures"] += np.bincount(models[credited], minlength=trainer.n_models)

        # The rest of that side is paid for the capture too -- one point, against
        # the holders' three above. Both are totals, so the branch pays one or
        # the other and the team's point is already inside the holder's price.
        covering = captured & ~credited
        if covering.any():
            trainer.tally["points"] += (np.bincount(models[covering], minlength=trainer.n_models)
                                        * config.POINTS_PER_CAPTURE_COVER)

        trainer.tally["points"] += np.bincount(models[won], minlength=trainer.n_models) * config.POINTS_PER_WIN
        trainer.tally["wins"] += np.bincount(models[won], minlength=trainer.n_models)
        trainer.tally["losses"] += np.bincount(models[lost], minlength=trainer.n_models)
        trainer.tally["draws"] += np.bincount(models[outcome < 0], minlength=trainer.n_models)
        trainer.tally["matches"] += np.bincount(models, minlength=trainer.n_models)

    # Every buffer is cut here, not just the ones that played: the line-up is
    # redrawn on restart, and GAE must not run across a change of who is in the
    # arena at all.
    for buf in trainer.buffers:
        buf.done[t, rows] = 1.0

    # Exploring starts, for the share of restarts EXPLORING_START_FRACTION asks
    # for: none by default.
    scatter = np.zeros(state.n, dtype=bool)
    if config.EXPLORING_START_FRACTION > 0.0:
        scatter[rows] = trainer.rng.random(rows.size) < config.EXPLORING_START_FRACTION
    sim.reset_arenas(state, done, scatter=scatter)
    assign_slots(trainer, rows)


def learning_models(trainer: Trainer) -> set:
    """Which model indices the gradient is allowed to reach, this update.

    Trainer.learning, resolved; an empty list is every model.

    THE TEST IS ON THE LIST ITSELF, not on what it resolves to. A non-empty list
    that maps to nothing -- none of its names in the roster -- freezes the whole
    roster, which is the conservative reading: the caller named models, none of
    them are here, and quietly training everybody instead is the one answer that
    cannot be undone. It is worth a line in the log because it is otherwise
    invisible: every figure the update prints reads 0.000, which is also what a
    healthy frozen roster prints.

    Resolved by NICKNAME through config.MODEL_NICKS at call time: the roster
    order is the caller's, and a name cannot point at the wrong model the way an
    index can.
    """
    wanted = trainer.learning
    if not wanted:
        return set(range(trainer.n_models))
    learning = set(m for m in range(trainer.n_models) if config.MODEL_NICKS[m] in wanted)
    if not learning:
        logger.warning(f"Learning is set to {', '.join(wanted)}, and the roster holds "
                       f"none of them; nothing is being trained")
    return learning


def learn(trainer: Trainer) -> Dict[str, float]:
    """Run one PPO update over the collected rollouts, for the models that learn.

    Which models those are is Trainer.learning, and the empty list means all of
    them.

    What freezing the rest does NOT do: they still act, still fill a rollout
    buffer of their own every step and still have that buffer discarded here, so
    it buys no simulation time whatsoever. It is a change to the GRADIENT and not
    to the cost. What it buys is a controlled comparison -- weights held still at
    a known point while one model goes on learning against them, which does not
    exist if everything keeps learning.
    """
    state = trainer.state
    critic_obs, unused = obs.build(state, trainer.mates, trainer.foes, gate=False)
    # The bootstrap value is the one critic call whose input never passes through
    # collect_rollout, and a corrupt feature here lands on every step of the tail.
    clamp_observations(critic_obs)
    critic_by_model = gather_by_model(critic_obs, trainer.slot_of_model)

    # Resolved once per call. A name the roster does not hold is dropped rather
    # than raised on; should the list name ONLY such models, nothing learns and
    # every diagnostic below reads 0.000 -- which is the honest report of an
    # update that did not happen.
    learning = learning_models(trainer)

    # Against the annealing horizon, never against the length of the run: an
    # open-ended run has no length to measure against, and dividing by it left
    # the schedule pinned at the start for ever.
    # The learners' clock and not the run's; see curriculum_clock.
    progress = min(1.0, curriculum_clock(trainer) / max(config.ANNEAL_STEPS, 1))
    merged = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
              "approx_kl": 0.0, "clip_fraction": 0.0, "learning_rate": 0.0}
    # Summed here and divided at the bottom by the updates that actually ran,
    # not by the roster. With one team learning and one frozen that would report
    # half of the only real updates there were, and the KL and the clip fraction
    # are exactly the numbers a run is watched by.
    updated = 0

    for m in trainer.field.tolist():
        buf = trainer.buffers[m]
        # BEFORE the critic and GAE. A model that played no step of this
        # rollout -- benched in every arena, which a roster larger than the
        # field allows -- would otherwise cost a forward pass over every arena,
        # a device sync to bring the answer back and a GAE sweep, all of it
        # thrown away.
        if not buf.valid.any():
            continue
        with torch.no_grad():
            tail = torch.as_tensor(critic_by_model[m], device=trainer.device)
            last_value = trainer.nets[m].value(tail).cpu().numpy()

        advantages, returns = buffer_mod.compute_gae(buf, last_value)
        # One update on impossible returns is enough to wreck a run that has
        # been going for hours: the value head chases the outlier, the policy
        # follows the value head, and nothing downstream can tell that the
        # numbers it is fitting were never reachable from the rewards. Rewards
        # are bounded by a few units and gamma is under one, so a return in the
        # thousands is arithmetically impossible and means something upstream
        # produced a non-finite or corrupted value.
        peak = float(np.abs(returns[buf.valid]).max())
        if not np.isfinite(peak) or peak > RETURN_SANITY_LIMIT:
            logger.error(f"Model {config.MODEL_NICKS[m]}: returns peaked at {peak:.3g} "
                         f"(reward peak {float(np.abs(buf.reward[buf.valid]).max()):.3g}, "
                         f"value peak {float(np.abs(buf.value[buf.valid]).max()):.3g}); "
                         f"update skipped")
            continue
        # Above the freeze below, and the order is the whole point: a frozen
        # model is still playing every rollout and still earning, and this
        # figure is what it earned.
        trainer.tally["reward"][m] += float(buf.reward[buf.valid].sum())

        # Neither ppo.update NOR the normaliser. The normaliser is the return
        # scale this model's value head is fitted against; moving it under a
        # policy that is not being updated is a slow drift in the one number that
        # survives to the next time the model does learn, so a frozen copy would
        # come back holding a scale nothing it did produced.
        if m not in learning:
            continue

        # CREDITED HERE, past the freeze skip, so the counter measures learning
        # and not play. A frozen model reached the line above and is turned back
        # before this one, so its figure holds exactly where it was.
        #
        # Past the two guards above it as well, deliberately. A rollout thrown
        # out for an impossible return, or one too short to make a minibatch,
        # moved no weights, so crediting it would count steps that taught this
        # model nothing.
        trainer.steps_by_model[m] += int(trainer.rollout_steps[m])

        trainer.normalizers[m].update(returns[buf.valid])
        scale = trainer.normalizers[m].std()
        advantages = advantages / scale
        returns = returns / scale

        batch = buffer_mod.flatten(buf, advantages, returns, trainer.device)
        if batch["obs"].shape[0] < config.N_MINIBATCHES:
            continue
        personality = config.personality(m)
        stats = ppo.update(trainer.nets[m], trainer.optimizers[m], batch,
                           entropy_scale=personality["entropy_scale"], progress=progress)
        updated += 1
        for key in merged:
            merged[key] += stats[key]

    # Left at zero when nothing updated, rather than divided by a floor of one:
    # zeros are what an update that did not happen produced, and a floor would
    # dress the last model's numbers up as the population's.
    if updated:
        for key in merged:
            merged[key] /= updated

    trainer.updates += 1
    return merged


def save_checkpoints(trainer: Trainer, directory: str) -> None:
    """Persist the whole run: weights, optimiser moments, normalisers, counters.

    Refuses to write over a checkpoint that is FURTHER ALONG than the run in
    memory, and that refusal is the whole of this function's caution.

    A process can come up holding less than the directory does in several ways:
    an observation layout that no longer matches, a checkpoint that would not
    load, a restore that was skipped. Every one of them reaches this function
    with a young run and a manifest saying otherwise, and the periodic save
    would then bury the run on disk under untrained weights a few updates later.
    Here rather than at the call sites, because the point is to cover the caller
    nobody has written yet.

    Refusing rather than exiting: a trainer that cannot save is still learning,
    and the session it loses is worth far less than the run on disk.
    """
    saved = read_saved_steps(directory)
    if trainer.steps_done < saved:
        logger.error(
            f"Checkpoint NOT written: this run holds {trainer.steps_done / 1e6:.2f}M steps "
            f"and the one on disk is further along at {saved / 1e6:.2f}M -- so this process "
            f"came up without the run in {persist.MANIFEST_NAME} and would bury it. Check the "
            f"lines above for why the restore found nothing. Nothing on disk was touched.")
        return

    persist.save_run(
        directory, trainer.nets, trainer.optimizers, trainer.normalizers,
        list(config.MODEL_NICKS), trainer.obs_size, trainer.steps_done, trainer.updates,
        trainer.generation, team_size=trainer.team_size,
        # The phase being played, by the learners' clock. Not steps_done: with
        # fresh networks facing veterans that could name the full arena while
        # every arena is close quarters.
        extra={"phase": curriculum.describe(curriculum_clock(trainer))},
        steps_by_model=[int(v) for v in trainer.steps_by_model],
    )


def drop_tally(trainer: Trainer) -> None:
    """Start the tally again from zero. A caller that wants the old one takes it first."""
    trainer.tally = new_tally(trainer.n_models)


def adopt_model_steps(trainer: Trainer, report: Optional[Dict[str, Any]]) -> None:
    """Take the per-model learning steps back off a restore, seeding what is missing.

    Read by NICKNAME, which is how restore_run reports them: the roster order is
    the caller's own, so a positional read could hand one model's history to
    another.

    A checkpoint that carries no count of its own is seeded with the run's
    share, steps_done divided by the roster. It is a seed and not a measurement,
    and it is why a checkpoint written for a NEW network has to say
    "model_steps": 0 rather than leave the key out: left out, a network that has
    learned nothing is credited with its share of a long run and starts on the
    full arena with its learning rate at the floor.
    """
    known = (report or {}).get("model_steps") or {}
    share = int(trainer.steps_done) // max(trainer.n_models, 1)
    seeded = []
    # A model with no checkpoint of its own starts at zero, never at the share.
    # Its network is the one the constructor just built and has learned
    # nothing; the share is for a file without the count, not for no file at
    # all.
    fresh = set((report or {}).get("fresh") or [])
    for index, nick in enumerate(config.MODEL_NICKS):
        if nick in known:
            trainer.steps_by_model[index] = int(known[nick])
        elif nick in fresh:
            trainer.steps_by_model[index] = 0
        else:
            trainer.steps_by_model[index] = share
            seeded.append(nick)
    if seeded:
        logger.info(f"Nothing recorded for {', '.join(seeded)}; seeded from the "
                    f"run's share at {share / 1e6:.2f}M each")


def read_saved_steps(directory: str) -> int:
    """Steps behind the checkpoint on disk, or 0 when there is none to read."""
    try:
        manifest = persist.read_manifest(directory)
    except Exception:
        return 0
    return int((manifest or {}).get("steps", 0))


def window_metrics(tally: Dict[str, np.ndarray]) -> Dict[str, float]:
    """The three headline figures over the window this tally covers.

    Taken from the tally's window and never from run-long counters: the
    question these answer is what the teams are doing now, and a run-long
    average is dominated by every round already played, so it would read as a
    flat line through both an improvement and a collapse.

    What the tally calls a match is one ROUND, and rounds are tallied per player,
    so points per match is what one model scores in one round.
    """
    matches = max(int(tally["matches"].sum()), 1)
    deaths = int(tally["deaths"].sum())
    return {
        "avg_points_per_match": int(tally["points"].sum()) / matches,
        "kills_per_death": int(tally["hits"].sum()) / max(deaths, 1),
        "deaths_per_match": deaths / matches,
    }


def log_progress(trainer: Trainer, stats: Dict[str, float], extra: Dict[str, float]) -> None:
    """Emit one line per update with the numbers worth watching.

    Points per round, hit rate and accuracy lead, because those are the columns
    actually read while a run is going. There is no win-rate column here: the
    line sums both ends of every round, so it would read 50% whatever happens.
    A team's results are in the match scores the caller records.
    """
    elapsed = time.time() - trainer.started
    rate = (trainer.steps_done - trainer.steps_at_start) / max(elapsed, 1e-6)
    tally = trainer.tally
    window = window_metrics(tally)
    accuracy = tally["hits"].sum() / max(tally["shots"].sum(), 1)

    logger.info(
        f"upd {trainer.updates:5d} | steps {trainer.steps_done / 1e6:7.2f}M | {rate:8,.0f}/s | "
        f"avg {window['avg_points_per_match']:5.2f} | hr {window['kills_per_death']:4.2f} | "
        f"dpm {window['deaths_per_match']:4.2f} | acc {accuracy:5.1%} | "
        f"matches {int(tally['matches'].sum()):7d} | hits {int(tally['hits'].sum()):7d} | "
        f"cap {int(tally['captures'].sum()):5d} | "
        f"kl {stats['approx_kl']:.4f} | clip {stats['clip_fraction']:.2f} | "
        f"vloss {stats['value_loss']:.3f} | ent {stats['entropy']:.3f}"
    )
    if extra.get("dropped_credit"):
        logger.debug(f"hit credit outside rollout window: {extra['dropped_credit']}")


def main():
    raise SystemExit("trainer.selfplaytv has no command line of its own; see python -m sdk.train --help")


if __name__ == "__main__":
    main()
