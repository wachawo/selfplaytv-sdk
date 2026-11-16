#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observation assembly.

Everything an agent sees is expressed in its own frame. In world coordinates a
network has to learn the product of heading and relative position before it can
aim at all; egocentrically, "turn until the bearing is zero" is close to a
linear readout, and aiming stops being the hard part of the problem.

The same builder produces the critic's input with line-of-sight gating switched
off. The actor must cope with not seeing through walls - that is what makes
cover worth anything - but a value function fed the same censored state would
be estimating returns for states it cannot tell apart.

The layout is fixed to config.MAX_TEAM_SIZE and does not change with the mode
being played. A 3v3 fills the first two teammate and first three enemy slots and
pads the rest. A layout that shrank with the match would give 3v3 and 5v5
different observation widths, and every switch between them would invalidate
every checkpoint on disk.
"""

import logging
import math
from typing import Dict, Tuple

import numpy as np

# Local imports
from engine import config
from engine.env import sim

logger = logging.getLogger(__name__)

# How far a paintball in flight is noticed from, and how many of them count as a
# firefight rather than a shot. The range is well inside MAX_SIGHT because this
# cue is not checked against walls.
TRACER_RANGE = 14.0
TRACER_VOLUME_FULL = 4.0
INF_DIST = 1e6

RAY_ANGLES = np.linspace(-math.pi, math.pi, config.N_RAYS, endpoint=False).astype(np.float32)

# Entity slots in the observation, fixed for every mode. The teammate block is
# one short of the enemy block because the observer is not its own teammate.
MAX_MATES = config.MAX_TEAM_SIZE - 1
MAX_FOES = config.MAX_TEAM_SIZE

# Where the two entity blocks start. Everything downstream derives from these,
# so a feature added to the self block moves the rest instead of silently
# shifting what the trainer reads by index.
MATE_OFFSET = config.SELF_FEATURES
FOE_OFFSET = MATE_OFFSET + MAX_MATES * config.ENTITY_FEATURES

# Named columns inside one entity row, used by the padding below.
ENT_DIST = 0
ENT_ALIVE = 7
ENT_VISIBLE = 8


def team_index_tables(team_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-player lists of teammate and enemy slots, shapes (P, S-1) and (P, S)."""
    p = team_size * 2
    team = np.zeros(p, dtype=np.int8)
    team[team_size:] = 1
    mates = np.zeros((p, team_size - 1), dtype=np.int64)
    foes = np.zeros((p, team_size), dtype=np.int64)
    for i in range(p):
        mates[i] = [j for j in range(p) if j != i and team[j] == team[i]]
        foes[i] = [j for j in range(p) if team[j] != team[i]]
    return mates, foes


def aim_offset() -> int:
    """Index of the aim block inside the observation vector.

    The potential-based shaping term reads cos(aim error) straight out of the
    observation instead of recomputing the intercept, so this offset has to
    stay derived rather than hard-coded. It no longer depends on the team size:
    the entity blocks are padded to their fixed width before this point.
    """
    return FOE_OFFSET + MAX_FOES * config.ENTITY_FEATURES


AIM_SIN = 0
AIM_COS = 1
AIM_FLIGHT = 2
AIM_HAS_TARGET = 3

# Indices inside the capture block, read by the trainer's capture shaping.
CAP_OWN = 0
CAP_TEAM_BEST = 1
CAP_FOE_BEST = 2
CAP_STANDING = 3
CAP_ALERT_SIN = 4
CAP_ALERT_COS = 5
CAP_CONTESTED = 6
# Where the objective is. Without these the capture block states the score of a
# game whose board the player cannot locate: the only positional cue it had was
# its own world coordinate, from which the strip to attack has to be inferred
# through which half it spawned in, and there is no team flag in the vector at
# all. The bearing makes "go there" a readout instead of an inference.
CAP_ATK_SIN  = 7
CAP_ATK_COS  = 8
CAP_ATK_DIST = 9
CAP_HOME_DIST = 10


def capture_offset() -> int:
    """Index of the capture block inside the observation vector.

    Derived the same way aim_offset is, so a block added earlier moves it
    automatically instead of silently shifting what the trainer reads.
    """
    return (aim_offset()
            + 4
            + config.N_RAYS * config.RAY_FEATURES
            + config.BALL_FEATURES
            + config.SOUND_FEATURES
            + config.TRACER_FEATURES)


def observation_size() -> int:
    """Total observation width. The same in every mode, by construction."""
    return (capture_offset()
            + config.CAPTURE_FEATURES
            + config.GLOBAL_FEATURES)


def to_egocentric(vx: np.ndarray, vy: np.ndarray, heading: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate a world-frame vector into the observer's heading frame."""
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    return vx * cos_h + vy * sin_h, -vx * sin_h + vy * cos_h


def update_memory(state: sim.SimState, visible: np.ndarray) -> None:
    """Refresh each observer's last-known position of every other player."""
    seen = visible[..., None]
    target = np.broadcast_to(state.pos[:, None, :, :], state.last_seen_pos.shape)
    state.last_seen_pos = np.where(seen, target, state.last_seen_pos).astype(np.float32)
    state.last_seen_age = np.where(visible, 0.0, state.last_seen_age + config.CONTROL_DT).astype(np.float32)


def pad_entities(block: np.ndarray, slots: int) -> np.ndarray:
    """Widen a sorted entity block to the fixed slot count, surplus slots last.

    Appended after the sort, never mixed into it: the block is ordered nearest
    first, so a filler row that sorted by its own contents would push real
    players down the list and change what "the closest enemy" means between
    modes.

    The filler is zeros except for the distance column. Read a row of pure
    zeros back through this layout and it says alive 0, visible 0 - and
    distance 0, which is the near end of the channel, not the far end. That is
    a corpse lying on the observer's feet, which is a thing the game can
    actually produce, so the empty slot would be claiming something rather than
    nothing. Distance 1.0 is the saturated end of clip(dist / MAX_SIGHT), i.e.
    "at least as far away as anything can be seen", which is what an absent
    player should read as. The remaining columns are genuinely zero: no
    bearing, no relative heading, no velocity, not alive, not visible, no
    memory of ever having been seen.
    """
    missing = slots - block.shape[2]
    if missing <= 0:
        return block
    pad = np.zeros(block.shape[:2] + (missing, block.shape[3]), dtype=block.dtype)
    pad[..., ENT_DIST] = 1.0
    return np.concatenate([block, pad], axis=2)


def entity_block(state: sim.SimState, idx: np.ndarray, visible: np.ndarray,
                 gate: bool, slots: int) -> np.ndarray:
    """Build the (N, P, slots, ENTITY_FEATURES) block for a set of related players.

    `slots` is the fixed width of the block; the arena fills the first
    idx.shape[1] of them and pad_entities fills the rest.
    """
    n, p = state.n, state.p
    observer = np.arange(p)[:, None]

    true_pos = state.pos[:, idx, :]
    seen = visible[:, observer, idx]
    remembered = state.last_seen_pos[:, observer, idx, :]
    age = state.last_seen_age[:, observer, idx]

    # When gated, an unseen enemy is reported at its last known spot with the age
    # of that memory attached; ungated (critic) input always uses the truth.
    known = np.broadcast_to(seen[..., None], true_pos.shape)
    stale = np.broadcast_to((age < 1e2)[..., None], true_pos.shape)
    pos = true_pos if not gate else np.where(known, true_pos, np.where(stale, remembered, true_pos))

    rel = pos - state.pos[:, :, None, :]
    dist = sim.vec_length(rel)
    bearing = np.arctan2(rel[..., 1], rel[..., 0]) - state.heading[:, :, None]

    rel_heading = state.heading[:, idx] - state.heading[:, :, None]
    vel = state.vel[:, idx, :] - state.vel[:, :, None, :]
    vx_ego, vy_ego = to_egocentric(vel[..., 0], vel[..., 1], state.heading[:, :, None])

    alive = state.alive[:, idx].astype(np.float32)
    visible_f = seen.astype(np.float32) if gate else alive
    memory = np.clip(age, 0.0, config.MEMORY_SECONDS) / config.MEMORY_SECONDS

    hidden = np.ones_like(dist) if not gate else (seen | (age < 1e2)).astype(np.float32)
    block = np.stack([
        np.clip(dist / config.MAX_SIGHT, 0.0, 1.0) * hidden,
        np.sin(bearing) * hidden,
        np.cos(bearing) * hidden,
        np.sin(rel_heading) * visible_f,
        np.cos(rel_heading) * visible_f,
        np.clip(vx_ego / config.SPEED_FORWARD, -2.0, 2.0) * visible_f,
        np.clip(vy_ego / config.SPEED_FORWARD, -2.0, 2.0) * visible_f,
        alive,
        visible_f,
        memory,
    ], axis=-1)

    # Nearest first, so the slot ordering carries meaning instead of raw index.
    #
    # Sorted on what this observer is allowed to know, not on the truth. The
    # values of an unseen enemy's row are already blanked, but its POSITION in
    # the block was not: ordering every enemy by real distance told the actor how
    # many enemies it cannot see are nearer than the one it can, which is most of
    # what seeing them would have given it. Cover is only worth something if
    # standing behind it removes information, and the slot index was handing a
    # large part of it straight back.
    knowable = dist if not gate else np.where(hidden > 0, dist, 1e6)
    order = np.argsort(np.where(alive > 0, knowable, 1e6), axis=-1)
    rows = np.arange(n)[:, None, None]
    cols = np.arange(p)[None, :, None]
    return pad_entities(block[rows, cols, order], slots)


def aim_block(state: sim.SimState, visible: np.ndarray, bearing: np.ndarray,
              flight: np.ndarray, foes: np.ndarray) -> np.ndarray:
    """Bearing error to the intercept point of the nearest visible enemy."""
    n, p = state.n, state.p
    observer = np.arange(p)[:, None]
    foe_bearing = bearing[:, observer, foes]
    foe_flight = flight[:, observer, foes]
    foe_visible = visible[:, observer, foes]
    rel = state.pos[:, foes, :] - state.pos[:, :, None, :]
    foe_dist = np.where(foe_visible, sim.vec_length(rel), 1e6)

    pick = foe_dist.argmin(axis=-1)
    rows = np.arange(n)[:, None]
    cols = np.arange(p)[None, :]
    chosen_bearing = foe_bearing[rows, cols, pick]
    chosen_flight = foe_flight[rows, cols, pick]
    has_target = foe_visible.any(axis=-1).astype(np.float32)

    return np.stack([
        np.sin(chosen_bearing) * has_target,
        np.cos(chosen_bearing) * has_target,
        # Scaled by the projectile's own lifetime, so 1.0 is exactly the range
        # limit: the paint expires on arrival. A hardcoded divisor encodes the
        # boundary at whatever value it happened to land on for one lifetime and
        # silently moves it the moment the lifetime changes -- which is how a
        # policy that had learned to fire below 0.8 ended up firing at targets
        # twice as far as it could reach. Saturating at 2.0 keeps everything
        # past double the range indistinguishable, which it is.
        np.clip(chosen_flight / config.BALL_LIFETIME, 0.0, 2.0) * has_target,
        has_target,
    ], axis=-1)


def incoming_block(state: sim.SimState) -> np.ndarray:
    """Threat features for the most dangerous enemy paintball in flight."""
    n, p, b = state.n, state.p, state.b
    ball_pos = state.ball_pos.reshape(n, p * b, 2)
    ball_vel = state.ball_vel.reshape(n, p * b, 2)
    ball_on = state.ball_active.reshape(n, p * b)
    shooter = np.repeat(np.arange(p), b)

    rel = ball_pos[:, None, :, :] - state.pos[:, :, None, :]
    rel_vel = ball_vel[:, None, :, :]
    closing = -sim.vec_dot(rel, rel_vel)
    speed_sq = np.maximum(sim.vec_dot(rel_vel, rel_vel), 1e-6)
    time_to_impact = np.clip(closing / speed_sq, 0.0, config.BALL_LIFETIME)
    closest = rel + rel_vel * time_to_impact[..., None]
    miss = sim.vec_length(closest)

    friendly = state.team[shooter][None, None, :] == state.team[None, :, None]
    threat = ball_on[:, None, :] & ~friendly & (closing > 0)
    ranked = np.where(threat, miss, 1e6)

    pick = ranked.argmin(axis=-1)
    rows = np.arange(n)[:, None]
    cols = np.arange(p)[None, :]
    any_threat = threat.any(axis=-1).astype(np.float32)
    bearing = np.arctan2(rel[rows, cols, pick, 1], rel[rows, cols, pick, 0]) - state.heading

    return np.stack([
        np.clip(time_to_impact[rows, cols, pick] / config.BALL_LIFETIME, 0.0, 1.0) * any_threat,
        np.clip(miss[rows, cols, pick] / 3.0, 0.0, 1.0) * any_threat,
        np.sin(bearing) * any_threat,
        np.cos(bearing) * any_threat,
    ], axis=-1)


def sound_block(state: sim.SimState) -> np.ndarray:
    """Bearing, staleness and loudness of the last enemy gunshot heard, (N, P, 5).

    Sound is not gated by line of sight - that is the whole point of it. Firing
    from behind cover now costs the shooter a bearing, so the trade between
    taking the shot and staying unheard is a real one.
    """
    rel = state.last_sound_pos - state.pos
    dist = sim.vec_length(rel)
    bearing = np.arctan2(rel[..., 1], rel[..., 0]) - state.heading

    age = state.last_sound_age
    has_sound = (age < config.SOUND_MEMORY_SECONDS).astype(np.float32)
    staleness = np.clip(age, 0.0, config.SOUND_MEMORY_SECONDS) / config.SOUND_MEMORY_SECONDS
    loudness = 1.0 - np.clip(dist / config.SOUND_RANGE, 0.0, 1.0)

    centre_rel = state.sound_centre - state.pos
    centre_dist = sim.vec_length(centre_rel)
    centre_bearing = np.arctan2(centre_rel[..., 1], centre_rel[..., 0]) - state.heading
    heat = np.clip(state.sound_heat / sim.SOUND_HEAT_MAX, 0.0, 1.0)
    # The centre only means anything while there is heat behind it; with none, it
    # is the last place anyone fired from an unknown time ago, which the four
    # features above already say better.
    has_heat = (state.sound_heat > 0.05).astype(np.float32)
    diagonal = math.hypot(state.pool.width, state.pool.height)

    return np.stack([
        np.sin(bearing) * has_sound,
        np.cos(bearing) * has_sound,
        staleness * has_sound,
        loudness * has_sound,
        has_sound,
        np.sin(centre_bearing) * has_heat,
        np.cos(centre_bearing) * has_heat,
        np.clip(centre_dist / diagonal, 0.0, 1.0) * has_heat,
        heat.astype(np.float32),
    ], axis=-1).astype(np.float32)


def tracer_block(state: sim.SimState) -> np.ndarray:
    """The enemy paint in the air that is not aimed at me, (N, P, 4).

    incoming_block answers "what is about to hit me". This answers "where is the
    shooting happening", which is a different question and the one a player asks
    before deciding which way round to go. A firefight two team-mates are having
    is a stream of paint crossing a place on the map, and none of it appears in
    any other block unless a shooter happens to be in view.

    Range-limited rather than gated by line of sight: paint in flight is seen,
    not heard, so cross-map x-ray vision would be wrong -- but a ray per ball per
    player is not affordable at 20 Hz, and a distance cut is the cheap
    approximation. It is deliberately shorter than the sight range, so anything
    this reports is close enough that a wall between is the exception.
    """
    n, p, b = state.n, state.p, state.b
    ball_pos = state.ball_pos.reshape(n, p * b, 2)
    ball_on = state.ball_active.reshape(n, p * b)
    shooter = np.repeat(np.arange(p), b)

    rel = ball_pos[:, None, :, :] - state.pos[:, :, None, :]
    dist = sim.vec_length(rel)
    enemy = state.team[shooter][None, None, :] != state.team[None, :, None]
    near = ball_on[:, None, :] & enemy & (dist < TRACER_RANGE)
    ranked = np.where(near, dist, INF_DIST)

    pick = ranked.argmin(axis=-1)
    rows = np.arange(n)[:, None]
    cols = np.arange(p)[None, :]
    seen = near.any(axis=-1).astype(np.float32)
    bearing = np.arctan2(rel[rows, cols, pick, 1], rel[rows, cols, pick, 0]) - state.heading
    # How much paint, not just the nearest piece: one ball is a shot, four is a
    # position being fought over.
    volume = np.clip(near.sum(axis=-1) / TRACER_VOLUME_FULL, 0.0, 1.0)

    return np.stack([
        np.sin(bearing) * seen,
        np.cos(bearing) * seen,
        np.clip(dist[rows, cols, pick] / TRACER_RANGE, 0.0, 1.0) * seen,
        volume.astype(np.float32),
    ], axis=-1).astype(np.float32)


def intruder_bearing(state: sim.SimState) -> Tuple[np.ndarray, np.ndarray]:
    """Egocentric sin/cos toward the nearest enemy inside the observer's own strip.

    Both are zero when no enemy is intruding, so the pair doubles as its own
    presence flag: a bearing of exactly (0, 0) is not a direction any live
    intruder can produce.

    capture_box holds the *enemy* strip for every slot, so an enemy standing
    inside its own capture box is by definition standing inside mine. Reusing
    sim.inside_capture_box keeps the alarm and the rule reading the same
    rectangle, and reads the position of this instant rather than the timer,
    which only starts accruing on the next physics tick.
    """
    intruding = sim.inside_capture_box(state) & state.alive
    enemy = state.team[:, None] != state.team[None, :]
    valid = intruding[:, None, :] & enemy[None, :, :]

    rel = state.pos[:, None, :, :] - state.pos[:, :, None, :]
    dist = sim.vec_length(rel)
    ranked = np.where(valid, dist, 1e6)

    pick = ranked.argmin(axis=-1)
    rows = np.arange(state.n)[:, None]
    cols = np.arange(state.p)[None, :]
    has_intruder = valid.any(axis=-1).astype(np.float32)
    bearing = np.arctan2(rel[rows, cols, pick, 1], rel[rows, cols, pick, 0]) - state.heading

    return np.sin(bearing) * has_intruder, np.cos(bearing) * has_intruder


def contested_flags(state: sim.SimState) -> np.ndarray:
    """(N, P) float: is the capture this player cares about being contested?

    Which capture that is depends on where the player is standing. Inside the
    enemy strip it is my own team's attempt, and the flag says a defender has
    turned up to pause my countdown. Anywhere else it is the enemy attempt on
    my own strip, and the flag says somebody on my side is already in there
    holding it - which is exactly the fact a defender needs in order to learn
    that walking in is what stops the clock, and to know whether a teammate has
    beaten it to the job.
    """
    attacking = sim.inside_capture_box(state) & state.alive
    is_red = state.team[None, :] == 0
    red_contested = state.capture_contested[:, 0:1]
    blue_contested = state.capture_contested[:, 1:2]

    own_attempt = np.where(is_red, red_contested, blue_contested)
    foe_attempt = np.where(is_red, blue_contested, red_contested)
    return np.where(attacking, own_attempt, foe_attempt).astype(np.float32)


def capture_block(state: sim.SimState) -> np.ndarray:
    """Capture progress and objective geometry seen from each player, (N, P, 11).

    Order is fixed, every reader of this block indexes it: own progress, the best
    progress on my team, the best progress among my enemies, whether I am
    standing in the enemy strip right now - all in 0..1 - then the sine and
    cosine of the egocentric bearing to the nearest enemy currently standing
    inside my own strip, or 0, 0 when nobody is intruding, the contested flag
    described in contested_flags, and last the egocentric bearing and distance to
    the strip I have to take plus the distance back to the one I defend.

    Not gated by line of sight, and deliberately so, for both halves. The enemy
    figure is what a defender reacts to: hiding it would leave the rule
    exploitable and never contested, and the score is public knowledge in any
    game that has one. The bearing is an alarm rather than eyesight, so it
    reaches every member of the defending team through walls, on the same
    reasoning as the gunshot cue - a defender that can read the number but not
    the direction knows it is losing the strip and still cannot go and contest
    it.
    """
    progress = sim.capture_progress(state)
    team_size = state.team_size
    red_best = progress[:, :team_size].max(axis=1, keepdims=True)
    blue_best = progress[:, team_size:].max(axis=1, keepdims=True)

    is_red = state.team[None, :] == 0
    own_best = np.broadcast_to(np.where(is_red, red_best, blue_best), progress.shape)
    foe_best = np.broadcast_to(np.where(is_red, blue_best, red_best), progress.shape)
    # A non-zero timer means the player was inside the enemy strip at the end of
    # the last physics tick, which is exactly the flag wanted here.
    # Read the box test, not the timer: a contested attacker accrues nothing, so
    # deriving this from capture_time would report "not inside" for exactly the
    # player the flag exists to describe -- the one standing on a contested
    # objective, which is when knowing it matters most.
    standing = (sim.inside_capture_box(state) & state.alive).astype(np.float32)
    alarm_sin, alarm_cos = intruder_bearing(state)
    contested = np.broadcast_to(contested_flags(state), progress.shape)
    atk_sin, atk_cos, atk_dist = box_bearing(state, state.capture_box)
    unused_sin, unused_cos, home_dist = box_bearing(state, state.home_box)

    return np.stack([progress, own_best, foe_best, standing,
                     alarm_sin, alarm_cos, contested,
                     atk_sin, atk_cos, atk_dist, home_dist], axis=-1).astype(np.float32)


def box_bearing(state: sim.SimState, box: np.ndarray):
    """Egocentric bearing and normalised distance to the nearest point of a box.

    Measured to the nearest point rather than to the centre, so the distance
    reaches zero the moment the player is inside and the bearing keeps pointing
    at the closest way in from outside. A bearing to the centre would instead
    keep pulling a player who is already standing on the objective towards its
    middle, which is a place the rule does not care about.
    """
    px = state.pos[:, :, 0]
    py = state.pos[:, :, 1]
    near_x = np.clip(px, box[None, :, 0], box[None, :, 1])
    near_y = np.clip(py, box[None, :, 2], box[None, :, 3])
    dx = near_x - px
    dy = near_y - py

    distance = np.sqrt(dx * dx + dy * dy)
    diagonal = math.hypot(state.pool.width, state.pool.height)
    bearing = np.arctan2(dy, dx) - state.heading
    # Inside the box the bearing has no meaning -- every direction is equally
    # "towards" a place the player already occupies -- so it is zeroed rather
    # than left to whatever arctan2 makes of a zero vector.
    outside = (distance > 1e-6).astype(np.float32)
    return (np.sin(bearing) * outside, np.cos(bearing) * outside,
            (distance / diagonal).astype(np.float32))


def build_pair(state: sim.SimState, mates: np.ndarray,
               foes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build actor and critic observations in one pass.

    The two differ only in whether enemy information is gated by line of sight;
    rays, memory, geometry and the intercept solution are shared, so computing
    them twice was pure waste on the training hot path.
    """
    visible = sim.compute_los(state)
    update_memory(state, visible)

    shared = shared_blocks(state, mates, foes, visible)
    gated_foes = entity_block(state, foes, visible, gate=True, slots=MAX_FOES)
    open_foes = entity_block(state, foes, visible, gate=False, slots=MAX_FOES)

    actor = assemble(state, shared, gated_foes)
    critic = assemble(state, shared, open_foes)
    return actor, critic, visible


def shared_blocks(state: sim.SimState, mates: np.ndarray, foes: np.ndarray,
                  visible: np.ndarray) -> Dict[str, np.ndarray]:
    """Blocks that are identical for the actor and the critic."""
    bearing, flight = sim.lead_solution(state)

    dead = ~state.alive
    vx_ego, vy_ego = to_egocentric(state.vel[:, :, 0], state.vel[:, :, 1], state.heading)
    clock = (state.tick[:, None] / config.MATCH_TICKS).astype(np.float32)
    own = np.stack([
        state.pos[:, :, 0] / state.pool.width,
        state.pos[:, :, 1] / state.pool.height,
        np.sin(state.heading),
        np.cos(state.heading),
        np.clip(vx_ego / config.SPEED_FORWARD, -2.0, 2.0),
        np.clip(vy_ego / config.SPEED_FORWARD, -2.0, 2.0),
        state.cooldown / config.COOLDOWN_TICKS,
        (state.cooldown <= 0).astype(np.float32),
        dead.astype(np.float32),
        np.broadcast_to(clock, dead.shape),
    ], axis=-1)

    mate_block = entity_block(state, mates, visible, gate=False, slots=MAX_MATES)
    aim = aim_block(state, visible, bearing, flight, foes)
    incoming = incoming_block(state)

    ray_dist, ray_kind = sim.cast_rays(state, RAY_ANGLES)
    rays = np.stack([
        ray_dist / config.MAX_SIGHT,
        (ray_kind == 0).astype(np.float32),
        (ray_kind == 1).astype(np.float32),
        (ray_kind == 2).astype(np.float32),
    ], axis=-1)

    alive_red = state.alive[:, :state.team_size].sum(axis=1, keepdims=True)
    alive_blue = state.alive[:, state.team_size:].sum(axis=1, keepdims=True)
    own_alive = np.where(state.team[None, :] == 0, alive_red, alive_blue) / state.team_size
    foe_alive = np.where(state.team[None, :] == 0, alive_blue, alive_red) / state.team_size
    glob = np.stack([
        np.broadcast_to(clock, dead.shape),
        own_alive.astype(np.float32),
        foe_alive.astype(np.float32),
        np.broadcast_to((state.team[None, :] == 0).astype(np.float32), dead.shape),
        np.broadcast_to(1.0 - clock, dead.shape),
        # Signed, and not two counts to be differenced: whether we are ahead is
        # the question a decision to go for the strip rests on, and a ReLU layer
        # reads a sign far more directly than it reconstructs one. It replaces a
        # constant 1.0 that the layer's own bias already provided.
        (own_alive - foe_alive).astype(np.float32),
    ], axis=-1)

    n, p = state.n, state.p
    return {
        # Cast here rather than left to the concatenation in assemble. This is the
        # one block that comes out in float64, and a single float64 column widens
        # the whole 186-wide observation: it is then built at twice the size and
        # copied back down. Each value is rounded to float32 either way and by the
        # same rule, so the answer does not move -- only the pass over three
        # quarters of a million floats goes.
        "own": own.astype(np.float32, copy=False),
        "mate": mate_block.reshape(n, p, -1),
        "aim": aim,
        "rays": rays.reshape(n, p, -1),
        "incoming": incoming,
        "sound": sound_block(state),
        "tracer": tracer_block(state),
        "capture": capture_block(state),
        "glob": glob,
        "dead": dead,
    }


def assemble(state: sim.SimState, shared: Dict[str, np.ndarray],
             foe_block: np.ndarray) -> np.ndarray:
    """Concatenate the blocks into the final observation vector."""
    n, p = state.n, state.p
    dead = shared["dead"]
    obs = np.concatenate([
        shared["own"],
        shared["mate"],
        foe_block.reshape(n, p, -1),
        shared["aim"],
        shared["rays"],
        shared["incoming"],
        shared["sound"],
        shared["tracer"],
        shared["capture"],
        shared["glob"],
    ], axis=-1).astype(np.float32, copy=False)

    # A dead player is an absorbing state: it keeps its slot and its terminal team
    # reward, but stops carrying information that could still be acted on.
    obs[dead] = 0.0
    obs[..., 8] = dead.astype(np.float32)
    return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def build(state: sim.SimState, mates: np.ndarray, foes: np.ndarray,
          gate: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble one observation variant. Returns (obs, visible) of shape (N, P, D)."""
    visible = sim.compute_los(state)
    if gate:
        update_memory(state, visible)
    shared = shared_blocks(state, mates, foes, visible)
    foe_block = entity_block(state, foes, visible, gate=gate, slots=MAX_FOES)
    return assemble(state, shared, foe_block), visible


def main():
    pass


if __name__ == "__main__":
    main()
