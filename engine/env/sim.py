#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batched paintball physics, vectorized with numpy across many arenas at once.

Every array carries a leading arena axis, so one call advances hundreds of
matches. Shapes used throughout:

    N = arenas, P = players per arena, B = projectile slots per player,
    M = obstacle slots per arena, R = rays per player

Three details here are load-bearing for learning rather than for realism:

* Projectiles are swept as segments. At 25 m/s a ball covers 1.25 m per tick
  against a 0.45 m target, so a naive point test would fly straight through.
* Hits are reported with the control step on which the shot was *fired*, so the
  trainer can credit the decision instead of the moment of impact half a second
  later.
* Dead players stay in every array as absorbing states. They stop moving and
  shooting but keep their slot, which keeps all tensors rectangular and lets
  them still receive the terminal team reward.
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Local imports
from engine import config
from engine.env import kernels
from engine.env.arena import ArenaPool, SHAPE_CIRCLE, spawn_box

logger = logging.getLogger(__name__)

INF = np.float32(1e9)

# How hard each heard shot pulls the remembered centre of fire towards itself,
# and the ceiling and per-step decay of the heat that counts them. The decay is
# set so heat halves in about the time the single-shot memory takes to go stale.
SOUND_CENTRE_PULL = 0.35
SOUND_HEAT_MAX = 6.0
SOUND_HEAT_DECAY = float(0.5 ** (config.CONTROL_DT / max(config.SOUND_MEMORY_SECONDS, 1e-6)))

# Egocentric move directions: index 0 is idle, 1..8 are 45-degree steps starting
# straight ahead and going counter-clockwise.
MOVE_ANGLES = np.array([0.0] + [k * math.pi / 4 for k in range(8)], dtype=np.float32)
MOVE_ACTIVE = np.array([0.0] + [1.0] * 8, dtype=np.float32)


class SimState:
    """Plain container for the batched simulation arrays."""

    def __init__(self, n_arenas: int, pool: ArenaPool, team_size: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.pool = pool
        self.n = n_arenas
        self.team_size = team_size
        self.p = team_size * 2
        self.b = config.MAX_BALLS_PER_PLAYER

        n, p, b = self.n, self.p, self.b
        self.team = np.zeros(p, dtype=np.int8)
        self.team[team_size:] = 1

        self.arena_idx = np.zeros(n, dtype=np.int32)
        self.pos      = np.zeros((n, p, 2), dtype=np.float32)
        self.vel      = np.zeros((n, p, 2), dtype=np.float32)
        self.heading  = np.zeros((n, p), dtype=np.float32)
        self.alive    = np.ones((n, p), dtype=bool)
        self.cooldown = np.zeros((n, p), dtype=np.int16)

        self.ball_active = np.zeros((n, p, b), dtype=bool)
        self.ball_pos    = np.zeros((n, p, b, 2), dtype=np.float32)
        self.ball_vel    = np.zeros((n, p, b, 2), dtype=np.float32)
        self.ball_ttl    = np.zeros((n, p, b), dtype=np.int16)
        self.ball_step   = np.zeros((n, p, b), dtype=np.int32)

        # Two clocks. `tick` is the match clock and stops while a capture is
        # running; `real_tick` counts physics ticks unconditionally and is the
        # only thing that can bound an episode once the match clock can stop.
        self.tick      = np.zeros(n, dtype=np.int32)
        self.real_tick = np.zeros(n, dtype=np.int32)
        self.step_idx  = np.zeros(n, dtype=np.int32)
        self.episode   = np.zeros(n, dtype=np.int32)

        # Per-episode kinematic jitter keeps policies from overfitting exact numbers.
        self.speed_scale = np.ones((n, p), dtype=np.float32)
        self.turn_scale  = np.ones((n, p), dtype=np.float32)
        self.ball_scale  = np.ones((n, p), dtype=np.float32)

        # Match bookkeeping surfaced to the trainer and to whatever draws the match.
        self.hits   = np.zeros((n, p), dtype=np.int32)
        self.deaths = np.zeros((n, p), dtype=np.int32)
        self.shots  = np.zeros((n, p), dtype=np.int32)

        # Explicit belief state: where i last saw j, and how long ago. This is the
        # cheap stand-in for recurrence under line-of-sight gating.
        self.last_seen_pos = np.zeros((n, p, p, 2), dtype=np.float32)
        self.last_seen_age = np.full((n, p, p), 1e3, dtype=np.float32)

        # Where each player last heard an enemy gunshot, and how long ago. Unlike
        # sight this ignores walls, so it is the one thing that survives cover.
        # Which team has already completed a capture, latched the tick it
        # happens. Reading it back off capture_time instead would let a win
        # evaporate when the capturer is shot later in the same control step.
        self.capture_winner = np.full(n, -1, dtype=np.int8)
        # Who was standing on the strip when it fell, latched on the same tick.
        # A capture is worth taking risk for, and the bonus is paid for being
        # there when it lands rather than for the team it happened to be on.
        self.capture_credit = np.zeros((n, p), dtype=bool)

        self.last_sound_pos = np.zeros((n, p, 2), dtype=np.float32)
        self.last_sound_age = np.full((n, p), 1e3, dtype=np.float32)
        # Where the shooting has been, rather than where the last shot was. The
        # single remembered muzzle answers "who just fired at me"; a fight two
        # players are having on the far side of the field is a stream of shots
        # from roughly one place, and the smoothed centre of them is what says
        # where not to walk -- or which way round to go. Heat is how much of it
        # there has been, decaying, so a burst and a lone shot are told apart.
        self.sound_centre = np.zeros((n, p, 2), dtype=np.float32)
        self.sound_heat = np.zeros((n, p), dtype=np.float32)

        # Unbroken seconds each player has spent inside the opposing spawn strip.
        # Reaching capture_needed wins the match for that player's team.
        self.capture_time = np.zeros((n, p), dtype=np.float32)
        # Every second spent on the enemy strip this match, added up. Distinct
        # from capture_time, which is the current unbroken hold and goes back to
        # zero the moment the player steps out or is shot: this is the work done
        # over the match, and it is what says who has been playing the objective
        # rather than who happens to be standing on it right now.
        #
        # A statistic and nothing else: nothing in the simulation reads it. It
        # is kept here for whoever reports on a round, outside this file.
        self.zone_time = np.zeros((n, p), dtype=np.float32)
        # Seconds a strip has to be held. Held on the state rather than read from
        # config so training can start with a hold short enough to be stumbled
        # into: five seconds of standing still is 50 control steps, and a policy
        # that has never done it by accident is never told it is worth doing.
        self.capture_needed = float(config.CAPTURE_SECONDS)
        # The rectangle each player has to hold, i.e. the *other* team's spawn
        # box, as (P, 4) of x_min, x_max, y_min, y_max. Arena size is a property
        # of the pool rather than of the layout, so these never change and the
        # hot loop never has to call spawn_box.
        self.capture_box = np.array(
            [spawn_box(1 - int(t), pool.width, pool.height) for t in self.team],
            dtype=np.float32)
        # The rectangle each player defends, i.e. its *own* spawn box. Standing
        # in it is what contests an enemy hold, so the defence needs the same
        # cheap cached test the attack already has.
        self.home_box = np.array(
            [spawn_box(int(t), pool.width, pool.height) for t in self.team],
            dtype=np.float32)

        # Per team, latest values from update_capture: whether the attempt this
        # team is carrying is being contested by a defender, and whether any
        # attempt in this arena is actually accruing (which is what stops the
        # match clock).
        self.capture_contested = np.zeros((n, 2), dtype=bool)
        self.capture_running   = np.zeros(n, dtype=bool)

        self.refresh_obstacles()

    def refresh_obstacles(self) -> None:
        """Cache this batch's obstacle arrays. Only changes when layouts change."""
        idx = self.arena_idx
        self.ob_shape = np.ascontiguousarray(self.pool.shape[idx])
        self.ob_cx    = np.ascontiguousarray(self.pool.cx[idx])
        self.ob_cy    = np.ascontiguousarray(self.pool.cy[idx])
        self.ob_hw    = np.ascontiguousarray(self.pool.hw[idx])
        self.ob_hh    = np.ascontiguousarray(self.pool.hh[idx])
        self.ob_active = np.ascontiguousarray(self.pool.active[idx])
        self.ob_count = np.ascontiguousarray(self.pool.count[idx])

    def obstacles(self) -> Tuple[np.ndarray, ...]:
        """Return this batch's cached obstacle arrays."""
        return (self.ob_shape, self.ob_cx, self.ob_cy, self.ob_hw, self.ob_hh, self.ob_active)


def sample_spawns(state: SimState, mask: np.ndarray, scatter: Optional[np.ndarray] = None) -> None:
    """Place every player of the selected arenas in a free cell of their spawn strip.

    Arenas flagged in `scatter` are placed anywhere free on the map instead --
    exploring starts. A policy that has never stood on the enemy strip cannot be
    told by any reward what standing there is worth, and from an ordinary spawn
    it will not get there by accident: the measured time on the objective is a
    fifth of a second per match, so the state is simply never sampled. Starting
    some episodes already spread over the field puts the critic in those states
    often enough to value them. Nothing here dictates an action; only where the
    episode begins.
    """
    width, height = state.pool.width, state.pool.height
    rows = np.nonzero(mask)[0]
    if rows.size == 0:
        return
    loose = np.zeros(state.n, dtype=bool) if scatter is None else scatter

    shape, cx, cy, hw, hh, active = state.obstacles()
    for player in range(state.p):
        team = int(state.team[player])
        home = spawn_box(team, width, height)
        whole = (config.PLAYER_RADIUS, width - config.PLAYER_RADIUS,
                 config.PLAYER_RADIUS, height - config.PLAYER_RADIUS)
        for box, subset in ((home, rows[~loose[rows]]), (whole, rows[loose[rows]])):
            if subset.size:
                place_player(state, subset, player, box, shape, cx, cy, hw, hh, active)

    # Face the enemy half, with a little slack so openings are not memorised.
    facing = np.where(state.team[None, :] == 0, 0.0, math.pi).astype(np.float32)
    state.heading[rows] = facing[0][None, :] + state.rng.uniform(-0.6, 0.6, (rows.size, state.p)).astype(np.float32)


def place_player(state: SimState, rows: np.ndarray, player: int, box, shape, cx, cy,
                 hw, hh, active) -> None:
    """Rejection-sample one player's position inside `box` for the given arenas."""
    x_min, x_max, y_min, y_max = box
    if rows.size:
        pending = rows.copy()
        for unused_attempt in range(24):
            if pending.size == 0:
                break
            trial_x = state.rng.uniform(x_min + config.PLAYER_RADIUS, x_max - config.PLAYER_RADIUS, pending.size)
            trial_y = state.rng.uniform(y_min + config.PLAYER_RADIUS, y_max - config.PLAYER_RADIUS, pending.size)
            free = free_of_obstacles(trial_x, trial_y, shape[pending], cx[pending], cy[pending],
                                     hw[pending], hh[pending], active[pending], config.PLAYER_RADIUS)
            ok = np.nonzero(free)[0]
            if ok.size:
                state.pos[pending[ok], player, 0] = trial_x[ok].astype(np.float32)
                state.pos[pending[ok], player, 1] = trial_y[ok].astype(np.float32)
            pending = pending[~free]
        if pending.size:
            state.pos[pending, player, 0] = (x_min + x_max) * 0.5
            state.pos[pending, player, 1] = (y_min + y_max) * 0.5


def free_of_obstacles(px: np.ndarray, py: np.ndarray, shape: np.ndarray, cx: np.ndarray,
                      cy: np.ndarray, hw: np.ndarray, hh: np.ndarray, active: np.ndarray,
                      radius: float) -> np.ndarray:
    """Test candidate points against a per-row obstacle set. Inputs are (K,) and (K, M)."""
    dx = np.abs(px[:, None] - cx)
    dy = np.abs(py[:, None] - cy)
    circle_hit = (dx * dx + dy * dy) < (hw + radius) ** 2
    rect_hit = (dx < hw + radius) & (dy < hh + radius)
    hit = np.where(shape == SHAPE_CIRCLE, circle_hit, rect_hit) & active
    return ~hit.any(axis=1)


def reset_arenas(state: SimState, mask: np.ndarray, arena_choices: Optional[np.ndarray] = None,
                 scatter: Optional[np.ndarray] = None) -> None:
    """Restart the selected arenas: new layout, new spawns, cleared projectiles.

    `scatter` marks arenas to start as exploring starts -- players spread over
    the whole map rather than in their own strips. See sample_spawns.
    """
    if not mask.any():
        return
    rows = np.nonzero(mask)[0]
    if arena_choices is None:
        arena_choices = state.rng.integers(0, len(state.pool), rows.size)
    state.arena_idx[rows] = arena_choices.astype(np.int32)
    state.refresh_obstacles()

    state.vel[rows] = 0.0
    state.alive[rows] = True
    state.cooldown[rows] = 0
    state.ball_active[rows] = False
    state.ball_ttl[rows] = 0
    state.tick[rows] = 0
    state.real_tick[rows] = 0
    state.episode[rows] += 1
    # step_idx is deliberately NOT reset: it is the monotonic clock that lets a
    # hit be credited back to the buffer slot where the shot was decided.
    state.hits[rows] = 0
    state.deaths[rows] = 0
    state.shots[rows] = 0
    state.last_seen_pos[rows] = 0.0
    state.last_seen_age[rows] = 1e3
    state.capture_winner[rows] = -1
    state.capture_credit[rows] = False
    state.zone_time[rows] = 0.0
    state.sound_centre[rows] = 0.0
    state.sound_heat[rows] = 0.0
    state.last_sound_pos[rows] = 0.0
    state.last_sound_age[rows] = 1e3
    state.capture_time[rows] = 0.0
    state.capture_contested[rows] = False
    state.capture_running[rows] = False

    shape = (rows.size, state.p)
    low, high = config.PLAYER_SCALE_RANGE
    state.speed_scale[rows] = state.rng.uniform(low, high, shape).astype(np.float32)
    state.turn_scale[rows] = state.rng.uniform(low, high, shape).astype(np.float32)
    state.ball_scale[rows] = state.rng.uniform(0.95, 1.05, shape).astype(np.float32)

    sample_spawns(state, mask, scatter)


def vec_length(vectors: np.ndarray) -> np.ndarray:
    """Length of the 2-vectors along the trailing axis.

    Written out rather than left to `np.linalg.norm(v, axis=-1)`. The axis being
    reduced is two elements long, so the reduction machinery is set up and torn
    down once per output element to add two numbers -- measured at ten times the
    cost of the arithmetic on the arrays this is called with, and it is called
    about fifteen times per control step on the hot path.

    Term for term the same sum in the same order, so the answer is bit-identical
    in float32, which is the bar: this feeds the observation a trained policy
    reads.
    """
    x = vectors[..., 0]
    y = vectors[..., 1]
    return np.sqrt(x * x + y * y)


def vec_dot(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Dot product of the 2-vectors along the trailing axis.

    For the reason vec_length is written out, and one more: `(a * b).sum(-1)`
    builds the whole product array before reducing it, so it also pays for a
    temporary the size of both inputs.
    """
    return first[..., 0] * second[..., 0] + first[..., 1] * second[..., 1]


def apply_movement(state: SimState, move: np.ndarray, turn: np.ndarray) -> None:
    """Turn, accelerate toward the commanded velocity, integrate, resolve collisions."""
    turn_bins = np.asarray(config.TURN_BINS, dtype=np.float32)
    state.heading += turn_bins[turn] * config.TURN_RATE * state.turn_scale * config.DT * state.alive
    state.heading = np.mod(state.heading + math.pi, 2 * math.pi) - math.pi

    angle = MOVE_ANGLES[move]
    activity = MOVE_ACTIVE[move] * state.alive
    world_angle = state.heading + angle
    forwardness = np.maximum(0.0, np.cos(angle))
    speed = (config.SPEED_LATERAL + (config.SPEED_FORWARD - config.SPEED_LATERAL) * forwardness)
    speed = speed * state.speed_scale * activity

    desired = np.stack([np.cos(world_angle) * speed, np.sin(world_angle) * speed], axis=-1)
    delta = desired - state.vel
    delta_mag = vec_length(delta)[..., None]
    max_delta = config.ACCELERATION * config.DT
    scale = np.minimum(1.0, max_delta / np.maximum(delta_mag, 1e-6))
    state.vel += delta * scale

    state.pos += state.vel * config.DT
    # Players first, then the geometry: a player can be shoved by another, but
    # not into a wall or an obstacle. Whatever overlap is left over after the
    # geometry has had the last word is parted again on the next tick.
    separate_players(state)
    resolve_collisions(state)


def separate_players(state: SimState) -> None:
    """Part players who overlap and share the speed they are closing at."""
    kernels.separate_players(state.pos, state.vel, state.alive,
                             np.float32(config.PLAYER_RADIUS))


def separate_players_numpy(state: SimState) -> None:
    """Reference implementation of separate_players, kept for equivalence checks."""
    want = np.float32(2.0 * config.PLAYER_RADIUS)
    px = state.pos[:, :, 0]
    py = state.pos[:, :, 1]
    dx = px[:, :, None] - px[:, None, :]
    dy = py[:, :, None] - py[:, None, :]
    dist = np.sqrt(dx * dx + dy * dy)

    # A player never collides with itself, and the dead are off the field: a body
    # that stopped where it was hit would otherwise go on barging the living
    # around for the rest of the round.
    slots = np.arange(state.p)
    pair = state.alive[:, :, None] & state.alive[:, None, :]
    pair &= slots[:, None] != slots[None, :]
    touching = pair & (dist < want)

    # Exactly on top of one another: no line of centres to part along, so the
    # lower slot goes one way and the higher the other.
    order = np.sign(slots[:, None] - slots[None, :]).astype(np.float32)
    flat = dist < 1e-6
    safe = np.maximum(dist, 1e-6)
    nx = np.where(flat, order, dx / safe)
    ny = np.where(flat, np.float32(0.0), dy / safe)

    # Every figure below is read from the state this tick arrived with, and the
    # positions are written only once the normals have been taken off them.
    depth = np.where(touching, want - dist, np.float32(0.0))
    rvx = state.vel[:, :, 0][:, :, None] - state.vel[:, :, 0][:, None, :]
    rvy = state.vel[:, :, 1][:, :, None] - state.vel[:, :, 1][:, None, :]
    approach = rvx * nx + rvy * ny
    share = np.where(touching & (approach < 0.0), -0.5 * approach, np.float32(0.0))

    state.pos[:, :, 0] += (0.5 * depth * nx).sum(axis=2)
    state.pos[:, :, 1] += (0.5 * depth * ny).sum(axis=2)
    state.vel[:, :, 0] += (share * nx).sum(axis=2)
    state.vel[:, :, 1] += (share * ny).sum(axis=2)


def resolve_collisions(state: SimState) -> None:
    """Push players out of obstacles and clamp them inside the arena walls."""
    kernels.resolve_collisions(state.pos, state.vel, state.ob_shape, state.ob_cx, state.ob_cy,
                               state.ob_hw, state.ob_hh, state.ob_count, config.PLAYER_RADIUS,
                               np.float32(state.pool.width), np.float32(state.pool.height), 4)


def resolve_collisions_numpy(state: SimState) -> None:
    """Reference implementation of resolve_collisions, kept for equivalence checks."""
    shape, cx, cy, hw, hh, active = state.obstacles()
    radius = config.PLAYER_RADIUS

    # Corners between two obstacles need several passes: the push-out from one
    # can shove the player into the other.
    for unused_pass in range(4):
        px = state.pos[:, :, 0][:, :, None]
        py = state.pos[:, :, 1][:, :, None]
        ox, oy = cx[:, None, :], cy[:, None, :]
        ohw, ohh = hw[:, None, :], hh[:, None, :]
        is_circle = (shape == SHAPE_CIRCLE)[:, None, :]
        live = active[:, None, :]

        # Closest point on each obstacle: the circle case reuses the rect maths by
        # clamping to a degenerate box, then normalising the offset.
        clamp_x = np.clip(px - ox, -ohw, ohw)
        clamp_y = np.clip(py - oy, -ohh, ohh)
        near_x = np.where(is_circle, ox, ox + clamp_x)
        near_y = np.where(is_circle, oy, oy + clamp_y)

        dx = px - near_x
        dy = py - near_y
        dist = np.sqrt(dx * dx + dy * dy)
        want = np.where(is_circle, ohw + radius, radius)

        # Resolve only the deepest contact per pass. Summing every overlap at once
        # overshoots in corners: each obstacle's correction shoves the player into
        # its neighbour, and the two never settle.
        penetration = np.where(live, want - dist, -1.0)
        deepest = penetration.argmax(axis=2)
        rows = np.arange(state.n)[:, None]
        cols = np.arange(state.p)[None, :]
        depth = penetration[rows, cols, deepest]
        if not (depth > 0.0).any():
            break

        pick_dx = dx[rows, cols, deepest]
        pick_dy = dy[rows, cols, deepest]
        pick_dist = dist[rows, cols, deepest]
        safe = np.maximum(pick_dist, 1e-6)
        scale = np.where(depth > 0.0, depth / safe, 0.0)

        # A player exactly on an obstacle centre has no direction to escape along;
        # nudge it along +x so the next pass can resolve it properly.
        degenerate = (depth > 0.0) & (pick_dist < 1e-6)
        state.pos[:, :, 0] += np.where(degenerate, depth, pick_dx * scale)
        state.pos[:, :, 1] += np.where(degenerate, 0.0, pick_dy * scale)
        # Clamp inside the loop, not after it: a wall clamp applied last would
        # shove the player straight back into the obstacle it was just freed from.
        clamp_to_walls(state)

    clamp_to_walls(state)


def clamp_to_walls(state: SimState) -> None:
    """Keep players inside the arena rectangle and kill the velocity into the wall."""
    radius = config.PLAYER_RADIUS
    width, height = state.pool.width, state.pool.height
    clipped_x = np.clip(state.pos[:, :, 0], radius, width - radius)
    clipped_y = np.clip(state.pos[:, :, 1], radius, height - radius)
    state.vel[:, :, 0] = np.where(clipped_x != state.pos[:, :, 0], 0.0, state.vel[:, :, 0])
    state.vel[:, :, 1] = np.where(clipped_y != state.pos[:, :, 1], 0.0, state.vel[:, :, 1])
    state.pos[:, :, 0] = clipped_x
    state.pos[:, :, 1] = clipped_y


def spawn_balls(state: SimState, shoot: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Fire for every player whose trigger is pulled, alive and off cooldown.

    Returns the (N, P) firing mask and the shots fired on this tick, the latter
    as parallel arrays of arena, shooter slot and the muzzle position the report
    came from.
    """
    can_fire = shoot & state.alive & (state.cooldown <= 0)
    free_slot = state.ball_active.argmin(axis=2)
    has_slot = ~state.ball_active.all(axis=2)
    firing = can_fire & has_slot
    if not firing.any():
        return firing, empty_shots()

    rows, players = np.nonzero(firing)
    slots = free_slot[rows, players]
    spread = state.rng.normal(0.0, config.BALL_SPREAD, rows.size).astype(np.float32)
    angle = state.heading[rows, players] + spread
    direction = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    speed = config.BALL_SPEED * state.ball_scale[rows, players]

    offset = config.PLAYER_RADIUS + config.BALL_RADIUS + 0.05
    state.ball_pos[rows, players, slots] = state.pos[rows, players] + direction * offset
    state.ball_vel[rows, players, slots] = direction * speed[:, None]
    state.ball_ttl[rows, players, slots] = config.BALL_TICKS
    state.ball_step[rows, players, slots] = state.step_idx[rows]
    state.ball_active[rows, players, slots] = True

    state.cooldown[rows, players] = config.COOLDOWN_TICKS
    state.shots[rows, players] += 1

    muzzle = state.pos[rows, players]
    hear_shots(state, rows, players, muzzle)
    return firing, {"arena": rows.astype(np.int32), "shooter": players.astype(np.int32),
                    "x": muzzle[:, 0].copy(), "y": muzzle[:, 1].copy()}


def empty_shots() -> Dict[str, np.ndarray]:
    """Empty shots-fired record, shaped like the one spawn_balls returns."""
    return {"arena": np.zeros(0, dtype=np.int32), "shooter": np.zeros(0, dtype=np.int32),
            "x": np.zeros(0, dtype=np.float32), "y": np.zeros(0, dtype=np.float32)}


def hear_shots(state: SimState, rows: np.ndarray, players: np.ndarray,
               muzzle: np.ndarray) -> None:
    """Give every enemy of a shooter the position the shot was fired from.

    Teammates learn nothing: the cue exists to locate opponents, and a friendly
    muzzle flash would only drown it out. Listeners are walked slot by slot so
    two shots landing in one arena resolve to the later one rather than summing.
    """
    for listener in range(state.p):
        heard = state.team[players] != state.team[listener]
        if not heard.any():
            continue
        arenas = rows[heard]
        state.last_sound_pos[arenas, listener] = muzzle[heard]
        state.last_sound_age[arenas, listener] = 0.0
        # The centre is pulled towards each shot rather than set to it, so a
        # second shooter somewhere else moves it a little instead of erasing the
        # first. Heat saturates: past a certain rate of fire, more of it says
        # nothing new about where to walk.
        centre = state.sound_centre[arenas, listener]
        heat = state.sound_heat[arenas, listener][:, None]
        fresh = heat <= 1e-3
        pull = np.where(fresh, 1.0, SOUND_CENTRE_PULL)
        state.sound_centre[arenas, listener] = centre + (muzzle[heard] - centre) * pull
        state.sound_heat[arenas, listener] = np.minimum(
            state.sound_heat[arenas, listener] + 1.0, SOUND_HEAT_MAX)


def segment_circle_t(p0: np.ndarray, d: np.ndarray, centre: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Earliest t in [0,1] where segment p0+t*d enters a circle, else INF.

    Broadcasting: p0/d are (..., 2), centre is (..., K, 2), radius is (..., K).
    """
    m = p0[..., None, :] - centre
    a = (d * d).sum(axis=-1)[..., None]
    b = 2.0 * (m * d[..., None, :]).sum(axis=-1)
    c = (m * m).sum(axis=-1) - radius * radius

    disc = b * b - 4.0 * a * c
    valid = disc > 0.0
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    t = (-b - sqrt_disc) / np.maximum(2.0 * a, 1e-12)
    # A segment starting inside the circle counts as an immediate hit.
    t = np.where(c < 0.0, 0.0, t)
    return np.where(valid & (t >= 0.0) & (t <= 1.0), t, INF)


def segment_rect_t(p0: np.ndarray, d: np.ndarray, centre: np.ndarray, half: np.ndarray,
                   pad: float) -> np.ndarray:
    """Earliest t in [0,1] where a segment enters an axis-aligned box, else INF."""
    lo = centre - (half + pad)
    hi = centre + (half + pad)
    origin = p0[..., None, :]
    direction = d[..., None, :]
    inv = 1.0 / np.where(np.abs(direction) < 1e-12, 1e-12, direction)

    t1 = (lo - origin) * inv
    t2 = (hi - origin) * inv
    t_near = np.maximum(np.minimum(t1, t2).max(axis=-1), 0.0)
    t_far = np.maximum(t1, t2).min(axis=-1)

    inside = ((origin >= lo) & (origin <= hi)).all(axis=-1)
    hit = (t_near <= t_far) & (t_far >= 0.0) & (t_near <= 1.0)
    return np.where(inside, 0.0, np.where(hit, t_near, INF))


def advance_balls(state: SimState) -> Dict[str, np.ndarray]:
    """Sweep every projectile one tick and resolve the earliest blocking contact."""
    victim = np.empty((state.n, state.p, state.b), dtype=np.int8)
    when = np.empty((state.n, state.p, state.b), dtype=np.float32)
    kernels.sweep_balls(state.ball_pos, state.ball_vel, state.ball_active, state.ball_ttl,
                        state.pos, state.alive, state.team, state.ob_shape, state.ob_cx,
                        state.ob_cy, state.ob_hw, state.ob_hh, state.ob_count,
                        np.float32(config.DT), np.float32(config.BALL_RADIUS),
                        np.float32(config.PLAYER_RADIUS), bool(config.FRIENDLY_FIRE),
                        victim, when)
    return register_hits(state, victim, when)


def register_hits(state: SimState, victim: np.ndarray, when: np.ndarray) -> Dict[str, np.ndarray]:
    """Deduplicate simultaneous contacts, apply eliminations, emit hit events."""
    empty = np.zeros(0, dtype=np.int32)
    struck = victim >= 0
    if not struck.any():
        return {"arena": empty, "shooter": empty, "victim": empty, "fire_step": empty,
                "defence": np.zeros(0, dtype=bool)}
    # Where the victim was standing has to be read now, while it is still
    # standing there: eliminating it moves nothing, but the strip test runs
    # again later in the tick and an intruder shot on its own last step would
    # otherwise be indistinguishable from one killed in the open.
    intruding = inside_capture_box(state)

    rows, shooters, slots = np.nonzero(struck)
    victims = victim[rows, shooters, slots].astype(np.int32)
    times = when[rows, shooters, slots]
    fire_steps = state.ball_step[rows, shooters, slots]

    # Two balls can reach the same player within one tick; only the earlier one
    # counts, otherwise a single elimination would pay out twice.
    order = np.lexsort((times, victims, rows))
    rows, victims, shooters, fire_steps = rows[order], victims[order], shooters[order], fire_steps[order]
    keep = np.ones(rows.size, dtype=bool)
    keep[1:] = (rows[1:] != rows[:-1]) | (victims[1:] != victims[:-1])

    rows, victims, shooters, fire_steps = rows[keep], victims[keep], shooters[keep], fire_steps[keep]
    state.alive[rows, victims] = False
    state.vel[rows, victims] = 0.0
    state.hits[rows, shooters] += 1
    state.deaths[rows, victims] += 1
    # A victim inside the box it was trying to capture is, by construction,
    # inside the shooter's own spawn strip: the two rectangles are the same one
    # seen from opposite sides. So this is exactly "I shot an intruder off my
    # zone", with no second test and no way for the two to disagree.
    defence = intruding[rows, victims]
    return {"arena": rows.astype(np.int32), "shooter": shooters.astype(np.int32),
            "victim": victims, "fire_step": fire_steps, "defence": defence}


def advance_balls_numpy(state: SimState) -> Dict[str, np.ndarray]:
    """Reference implementation of advance_balls, kept for equivalence checks."""
    n, p, b = state.n, state.p, state.b
    p0 = state.ball_pos.reshape(n, p * b, 2)
    d = state.ball_vel.reshape(n, p * b, 2) * config.DT
    active = state.ball_active.reshape(n, p * b)

    shape, cx, cy, hw, hh, obst_active = state.obstacles()
    centre = np.stack([cx, cy], axis=-1)[:, None, :, :]
    half = np.stack([hw, hh], axis=-1)[:, None, :, :]
    radius = hw[:, None, :]

    t_circle = segment_circle_t(p0, d, centre, radius + config.BALL_RADIUS)
    t_rect = segment_rect_t(p0, d, centre, half, config.BALL_RADIUS)
    is_circle = (shape == SHAPE_CIRCLE)[:, None, :]
    t_obstacle = np.where(obst_active[:, None, :], np.where(is_circle, t_circle, t_rect), INF)
    t_wall = t_obstacle.min(axis=2)

    target_pos = state.pos[:, None, :, :]
    hit_radius = np.full((n, 1, p), config.PLAYER_RADIUS + config.BALL_RADIUS, dtype=np.float32)
    t_player = segment_circle_t(p0, d, target_pos, hit_radius)

    shooter = np.repeat(np.arange(p), b)
    same_shooter = np.zeros((p * b, p), dtype=bool)
    same_shooter[np.arange(p * b), shooter] = True
    blocked = same_shooter[None, :, :] | ~state.alive[:, None, :]
    if not config.FRIENDLY_FIRE:
        same_team = state.team[shooter][:, None] == state.team[None, :]
        blocked = blocked | same_team[None, :, :]
    t_player = np.where(blocked, INF, t_player)

    t_best = t_player.min(axis=2)
    victim = t_player.argmin(axis=2)
    struck = active & (t_best < INF) & (t_best <= t_wall)
    perished = active & (t_wall < INF) & (t_wall < t_best)

    events = collect_hits(state, struck, victim, t_best, shooter)

    moved = p0 + d
    state.ball_pos = moved.reshape(n, p, b, 2)
    state.ball_ttl -= 1
    expired = (state.ball_ttl <= 0)
    state.ball_active &= ~(expired | struck.reshape(n, p, b) | perished.reshape(n, p, b))
    return events


def collect_hits(state: SimState, struck: np.ndarray, victim: np.ndarray, t_best: np.ndarray,
                 shooter_of_slot: np.ndarray) -> Dict[str, np.ndarray]:
    """Turn per-slot contacts into deduplicated hit events and apply eliminations."""
    empty = np.zeros(0, dtype=np.int32)
    if not struck.any():
        return {"arena": empty, "shooter": empty, "victim": empty, "fire_step": empty,
                "defence": np.zeros(0, dtype=bool)}

    intruding = inside_capture_box(state)
    rows, slots = np.nonzero(struck)
    victims = victim[rows, slots].astype(np.int32)
    times = t_best[rows, slots]
    shooters = shooter_of_slot[slots].astype(np.int32)
    ball_slots = slots % state.b
    fire_steps = state.ball_step[rows, shooters, ball_slots]

    # Two balls can reach the same player within one tick; only the earlier one
    # counts, otherwise a single elimination would pay out twice.
    order = np.lexsort((times, victims, rows))
    rows, victims, shooters, fire_steps = rows[order], victims[order], shooters[order], fire_steps[order]
    keep = np.ones(rows.size, dtype=bool)
    keep[1:] = (rows[1:] != rows[:-1]) | (victims[1:] != victims[:-1])

    rows, victims, shooters, fire_steps = rows[keep], victims[keep], shooters[keep], fire_steps[keep]
    state.alive[rows, victims] = False
    state.vel[rows, victims] = 0.0
    state.hits[rows, shooters] += 1
    state.deaths[rows, victims] += 1
    return {"arena": rows.astype(np.int32), "shooter": shooters, "victim": victims,
            "fire_step": fire_steps, "defence": intruding[rows, victims]}


def cast_rays(state: SimState, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cast rays from every player. Returns (distance, class) arrays of shape (N, P, R).

    Class encoding: 0 wall or obstacle, 1 enemy, 2 teammate.
    """
    dist = np.empty((state.n, state.p, angles.size), dtype=np.float32)
    kind = np.empty((state.n, state.p, angles.size), dtype=np.int8)
    kernels.cast_rays(state.pos, state.heading, state.alive, state.team, angles,
                      state.ob_shape, state.ob_cx, state.ob_cy, state.ob_hw, state.ob_hh,
                      state.ob_count, np.float32(state.pool.width), np.float32(state.pool.height),
                      np.float32(config.MAX_SIGHT), np.float32(config.PLAYER_RADIUS), dist, kind)
    return dist, kind


def cast_rays_numpy(state: SimState, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reference implementation of cast_rays, kept for equivalence checks."""
    n, p = state.n, state.p
    r = angles.size
    world = state.heading[:, :, None] + angles[None, None, :]
    d = np.stack([np.cos(world), np.sin(world)], axis=-1) * config.MAX_SIGHT
    origin = np.repeat(state.pos[:, :, None, :], r, axis=2).reshape(n, p * r, 2)
    d = d.reshape(n, p * r, 2)

    shape, cx, cy, hw, hh, obst_active = state.obstacles()
    centre = np.stack([cx, cy], axis=-1)[:, None, :, :]
    half = np.stack([hw, hh], axis=-1)[:, None, :, :]
    t_circle = segment_circle_t(origin, d, centre, hw[:, None, :])
    t_rect = segment_rect_t(origin, d, centre, half, 0.0)
    is_circle = (shape == SHAPE_CIRCLE)[:, None, :]
    t_obst = np.where(obst_active[:, None, :], np.where(is_circle, t_circle, t_rect), INF).min(axis=2)
    t_bounds = segment_box_exit(origin, d, state.pool.width, state.pool.height)
    t_static = np.minimum(t_obst, t_bounds)

    body = np.full((n, 1, p), config.PLAYER_RADIUS, dtype=np.float32)
    t_body = segment_circle_t(origin, d, state.pos[:, None, :, :], body)
    self_mask = np.zeros((p * r, p), dtype=bool)
    self_mask[np.arange(p * r), np.repeat(np.arange(p), r)] = True
    t_body = np.where(self_mask[None, :, :] | ~state.alive[:, None, :], INF, t_body)

    t_player = t_body.min(axis=2)
    nearest = t_body.argmin(axis=2)
    sees_player = t_player < t_static

    distance = np.where(sees_player, t_player, t_static) * config.MAX_SIGHT
    shooter_team = np.repeat(state.team, r)
    is_mate = state.team[nearest] == shooter_team[None, :]
    kind = np.where(sees_player, np.where(is_mate, 2, 1), 0).astype(np.int8)
    return distance.reshape(n, p, r), kind.reshape(n, p, r)


def segment_box_exit(origin: np.ndarray, d: np.ndarray, width: float, height: float) -> np.ndarray:
    """Fraction of the ray at which it leaves the arena rectangle."""
    inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
    hi = np.stack([np.full(origin.shape[:-1], width, dtype=np.float32),
                   np.full(origin.shape[:-1], height, dtype=np.float32)], axis=-1)
    t1 = (0.0 - origin) * inv
    t2 = (hi - origin) * inv
    return np.maximum(t1, t2).min(axis=-1)


def compute_los(state: SimState) -> np.ndarray:
    """Line-of-sight matrix (N, P, P): can player i currently see player j?

    Teammates block sight even though they do not block paint, so a body in the
    doorway actually costs you the angle.
    """
    visible = np.empty((state.n, state.p, state.p), dtype=np.bool_)
    kernels.compute_los(state.pos, state.alive, state.ob_shape, state.ob_cx, state.ob_cy,
                        state.ob_hw, state.ob_hh, state.ob_count,
                        np.float32(config.PLAYER_RADIUS), visible)
    return visible


def compute_los_numpy(state: SimState) -> np.ndarray:
    """Reference implementation of compute_los, kept for equivalence checks."""
    n, p = state.n, state.p
    origin = np.repeat(state.pos[:, :, None, :], p, axis=2).reshape(n, p * p, 2)
    target = np.tile(state.pos[:, None, :, :], (1, p, 1, 1)).reshape(n, p * p, 2)
    d = target - origin

    shape, cx, cy, hw, hh, obst_active = state.obstacles()
    centre = np.stack([cx, cy], axis=-1)[:, None, :, :]
    half = np.stack([hw, hh], axis=-1)[:, None, :, :]
    t_circle = segment_circle_t(origin, d, centre, hw[:, None, :])
    t_rect = segment_rect_t(origin, d, centre, half, 0.0)
    is_circle = (shape == SHAPE_CIRCLE)[:, None, :]
    blocked = np.where(obst_active[:, None, :], np.where(is_circle, t_circle, t_rect), INF).min(axis=2) < 1.0

    body = np.full((n, 1, p), config.PLAYER_RADIUS * 0.9, dtype=np.float32)
    t_body = segment_circle_t(origin, d, state.pos[:, None, :, :], body)
    pair = np.arange(p * p)
    endpoints = np.zeros((p * p, p), dtype=bool)
    endpoints[pair, pair // p] = True
    endpoints[pair, pair % p] = True
    t_body = np.where(endpoints[None, :, :] | ~state.alive[:, None, :], INF, t_body)
    blocked |= (t_body < 0.999).any(axis=2)

    visible = (~blocked).reshape(n, p, p)
    visible &= state.alive[:, None, :] & state.alive[:, :, None]
    # The self pair has a zero-length segment and would otherwise read as visible.
    visible[:, np.arange(p), np.arange(p)] = False
    return visible


def lead_solution(state: SimState) -> Tuple[np.ndarray, np.ndarray]:
    """Bearing error to the intercept point of every target, shape (N, P, P).

    Solves |target + v*t - shooter| = ball_speed * t for the flight time, then
    reports the angle between the shooter's heading and that aim point. This is
    what turns "learn ballistics" into "learn when to pull the trigger".
    """
    rel = state.pos[:, None, :, :] - state.pos[:, :, None, :]
    vel = state.vel[:, None, :, :]
    speed = config.BALL_SPEED

    a = vec_dot(vel, vel) - speed * speed
    b = 2.0 * vec_dot(rel, vel)
    c = vec_dot(rel, rel)
    disc = b * b - 4.0 * a * c
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    t1 = (-b + sqrt_disc) / np.where(np.abs(2.0 * a) < 1e-9, 1e-9, 2.0 * a)
    t2 = (-b - sqrt_disc) / np.where(np.abs(2.0 * a) < 1e-9, 1e-9, 2.0 * a)
    flight = np.where(t1 > 0, t1, t2)
    flight = np.where((disc < 0) | (flight < 0), np.sqrt(c) / speed, flight)

    aim = rel + vel * flight[..., None]
    bearing = np.arctan2(aim[..., 1], aim[..., 0]) - state.heading[:, :, None]
    bearing = np.mod(bearing + math.pi, 2 * math.pi) - math.pi
    return bearing.astype(np.float32), flight.astype(np.float32)


def autofire_decision(state: SimState, visible: np.ndarray, bearing: np.ndarray) -> np.ndarray:
    """Environment-controlled trigger used during the warm-up phase."""
    enemy = state.team[None, :, None] != state.team[None, None, :]
    on_target = visible & enemy & (np.abs(bearing) < config.AUTOFIRE_CONE)
    return on_target.any(axis=2) & state.alive & (state.cooldown <= 0)


def inside_capture_box(state: SimState) -> np.ndarray:
    """(N, P) mask: is this player standing inside the opposing spawn strip?

    Own strip never counts - capture_box holds the *enemy* rectangle for every
    slot, so standing at home is indistinguishable from standing anywhere else.
    """
    return inside_box(state, state.capture_box)


def inside_home_box(state: SimState) -> np.ndarray:
    """(N, P) mask: is this player standing inside its own team's spawn strip?

    The mirror image of inside_capture_box, and the test that decides whether a
    hold is contested: the strip team t attacks is the strip team 1-t defends,
    so the two functions read the same rectangle from opposite sides.
    """
    return inside_box(state, state.home_box)


def inside_box(state: SimState, box: np.ndarray) -> np.ndarray:
    """(N, P) mask: is each player inside its own (P, 4) rectangle?"""
    x, y = state.pos[:, :, 0], state.pos[:, :, 1]
    return ((x >= box[None, :, 0]) & (x <= box[None, :, 1]) &
            (y >= box[None, :, 2]) & (y <= box[None, :, 3]))


def latch_capture(state: SimState) -> None:
    """Record the first team to complete a hold, once and for good.

    The win is decided here, on the physics tick the bar fills, and not later
    from whatever the state happens to look like: episode_status only runs once
    per control step, so a capturer eliminated on the following sub-tick would
    otherwise have its finished capture quietly undone, and two captures a tick
    apart inside one control step would read as simultaneous and score a draw.
    """
    filled = state.capture_time >= state.capture_needed
    if not filled.any():
        return
    team_size = state.team_size
    red_done = filled[:, :team_size].any(axis=1)
    blue_done = filled[:, team_size:].any(axis=1)

    # Undecided arenas only: the first completion stands. Both on the very same
    # tick is a genuine tie, with no earlier arrival to separate them.
    fresh = state.capture_winner < 0
    state.capture_winner = np.where(fresh & red_done & ~blue_done, 0, state.capture_winner)
    state.capture_winner = np.where(fresh & blue_done & ~red_done, 1, state.capture_winner)
    state.capture_winner = np.where(fresh & red_done & blue_done, 2, state.capture_winner)

    # Who was standing on the objective as it fell. Recorded here rather than
    # read back at the end of the episode for the same reason the winner is: by
    # then the holders can have been shot, and a bonus for taking the strip that
    # evaporates when the enemy shoots you off it a tick later rewards the wrong
    # thing -- it pays for surviving, not for taking ground.
    decided = fresh & (state.capture_winner >= 0) & (state.capture_winner < 2)
    if not decided.any():
        return
    on_objective = inside_capture_box(state) & state.alive
    winners = state.team[None, :] == state.capture_winner[:, None]
    state.capture_credit |= decided[:, None] & winners & on_objective


def update_capture(state: SimState) -> None:
    """Accrue one physics tick of standing time, hold it, or reset it to zero.

    Run per physics tick rather than once per control step: the tick loop is
    what moves the player, so "continuously" means continuously on the physics
    timeline. Sampling once per control step would let a player leave the strip
    and come back between two samples and still be paid for the gap. The test
    is a handful of comparisons on an (N, P) array, which is nothing next to
    the ball sweep already running in the same loop.

    Called after hits are resolved, so a player eliminated on this tick is
    already dead here and loses its progress on the tick it dies.

    A hold is *contested* while a member of the owning team also stands in the
    strip, and a contested countdown pauses rather than resets. Pausing is the
    whole point of the rule: a reset would make one defender's brief step into
    the strip worth as much as killing the attacker, so defence would be free
    and touch-and-go, whereas holding the value makes the strip somewhere both
    sides have to keep standing. Ground is only given back by leaving or dying.
    """
    holding = inside_capture_box(state) & state.alive
    defending = inside_home_box(state) & state.alive
    team_size = state.team_size

    # Counted on the same test and the same tick as the hold itself, so the two
    # can never disagree about who was standing where. Contested time counts:
    # the countdown pauses, but the player is still holding ground under fire,
    # which is the thing this number is for.
    #
    # AND IT STOPS THE TICK THE STRIP FALLS. This is a stopwatch on the work of
    # taking the objective, and the objective is taken the moment latch_capture
    # -- the next call in the step -- sets the winner.
    #
    # The countdown below is deliberately NOT gated with it. capture_time is
    # drawn as a bar that is already full, and freezing it would change nothing
    # anybody can see; this is a figure, and a figure is read.
    state.zone_time += holding * (state.capture_winner < 0)[:, None] * config.DT

    # Per team: is anyone carrying an attempt, and is anyone at home to answer
    # one. Reversing at_home lines each team's attempt up with the defenders of
    # the strip it is attacking.
    attempt = np.stack([holding[:, :team_size].any(axis=1),
                        holding[:, team_size:].any(axis=1)], axis=1)
    at_home = np.stack([defending[:, :team_size].any(axis=1),
                        defending[:, team_size:].any(axis=1)], axis=1)
    defended = at_home[:, ::-1]

    state.capture_contested = attempt & defended
    # An attempt that is accruing, i.e. one that is neither absent nor paused.
    # This is what freezes the match clock, so it is stored rather than
    # recomputed by the caller.
    state.capture_running = (attempt & ~defended).any(axis=1)

    is_red = (state.team == 0)[None, :]
    paused = np.where(is_red, defended[:, 0:1], defended[:, 1:2])

    # The countdown runs at one second per attacker standing on the strip, so a
    # squad takes it in a fraction of the time one player would. Every holder on
    # the team accrues at the same shared rate, which is what makes them a group
    # rather than three separate five-second attempts that never finish.
    crowd = np.stack([holding[:, :team_size].sum(axis=1),
                      holding[:, team_size:].sum(axis=1)], axis=1)
    own_crowd = np.where(is_red, crowd[:, 0:1], crowd[:, 1:2])
    rate = np.minimum(own_crowd, config.CAPTURE_RATE_CAP)

    gain = np.where(paused, 0.0, config.DT * rate)
    state.capture_time = np.where(holding, state.capture_time + gain, 0.0).astype(np.float32)


def capture_progress(state: SimState) -> np.ndarray:
    """Per-player capture progress in 0..1, shape (N, P)."""
    return np.clip(state.capture_time / state.capture_needed, 0.0, 1.0).astype(np.float32)


# Why a round ended, as end_reasons codes it.
REASON_CAPTURE = 0
REASON_ELIMINATION = 1
REASON_TIME = 2


def end_reasons(state: SimState, rows: np.ndarray) -> np.ndarray:
    """Why each of these arenas' rounds ended, as int8 REASON_* codes.

    One precedence, so everything that tallies endings names an ending the
    same way: the capture latch first, then a wiped side, and anything else ran
    out of time -- the real-tick ceiling and a phase's shorter clock included.
    Must be read before reset_arenas, which zeroes the latch and revives
    everybody.
    """
    rows = np.asarray(rows)
    side_one = (state.team == 1)[None, :]
    alive = state.alive[rows]
    red = (alive & ~side_one).sum(axis=1)
    blue = (alive & side_one).sum(axis=1)
    reasons = np.full(rows.shape[0], REASON_TIME, dtype=np.int8)
    reasons = np.where((red == 0) | (blue == 0), REASON_ELIMINATION, reasons)
    reasons = np.where(state.capture_winner[rows] >= 0, REASON_CAPTURE, reasons)
    return reasons.astype(np.int8)


def episode_status(state: SimState) -> Tuple[np.ndarray, np.ndarray]:
    """Return (done, winner) where winner is 0 red, 1 blue, -1 draw or unfinished.

    Precedence when several endings land on the same tick:

    1. A completed capture outranks everything else. It is the only thing in
       the game a team can deliberately achieve, so it must not be overturned
       by the survivor count it happens to coincide with: red holding the blue
       strip with one player left beats blue's three survivors.
    2. Capture beats the clock. A capture that completes on the very tick the
       match times out still wins; otherwise the objective would be worth
       nothing in the last second, which is exactly when it is contested.
    3. A capture cannot coexist with the capturing side being wiped out.
       update_capture runs after hit resolution and zeroes the timer of a
       player that has just been eliminated, so whoever fills the bar is
       provably alive at the end of that tick, and its team is therefore not
       wiped. The awkward case simply cannot arise.
    4. Both teams completing on the same tick is a draw. Progress accrues in
       whole ticks of DT and the match stops the instant either bar fills, so
       two simultaneous captures are exactly tied at CAPTURE_SECONDS with no
       earlier arrival to separate them; inventing a tiebreak would be
       arbitrary.
    5. More survivors wins and equal is a draw, but a wipeout does not settle
       the match while the wiped side still has paint in the air. Two players
       who shoot each other are both out, and awarding the match to whichever
       ball landed first decides it on a metre of flight time.
    6. The real-tick ceiling ends the match on the same terms as the clock. The
       match clock stops during a capture and can therefore be held still
       indefinitely, so it is no longer on its own enough to guarantee that an
       episode ends; the ceiling is, and an episode that reaches it is simply
       out of time by another name.
    7. A round that runs out of time is settled by the fight and by nothing
       else. Seconds accumulated on the enemy strip used to break it first; they
       no longer count for anything on their own, and holding the strip pays
       only by finishing the capture, which ends the round outright.
    """
    red = (state.alive[:, :state.team_size]).sum(axis=1)
    blue = (state.alive[:, state.team_size:]).sum(axis=1)
    red_out = red == 0
    blue_out = blue == 0

    # A side that has just been wiped out has not lost while its paint is still
    # in the air. Two players who shoot each other are both out, and which ball
    # happens to land first is a matter of a metre of flight; settling the match
    # on the earlier landing hands the win to whoever stood marginally closer.
    #
    # Bounded by the projectile lifetime, so a match cannot be held open by this:
    # the balls either land or expire within BALL_LIFETIME.
    red_paint = state.ball_active[:, :state.team_size].any(axis=(1, 2))
    blue_paint = state.ball_active[:, state.team_size:].any(axis=(1, 2))
    pending = ((red_out & ~blue_out & red_paint) | (blue_out & ~red_out & blue_paint))

    wiped = (red_out | blue_out) & ~pending
    timeout = (state.tick >= config.MATCH_TICKS) | (state.real_tick >= config.MATCH_TICKS_MAX)

    # Read the latch, not the live timer: a capture that completed earlier in
    # this control step counts even if the capturer has since been eliminated,
    # which also keeps `done` monotonic within an episode.
    red_capture = state.capture_winner == 0
    blue_capture = state.capture_winner == 1
    tied_capture = state.capture_winner == 2
    done = wiped | timeout | red_capture | blue_capture | tied_capture

    winner = np.full(state.n, -1, dtype=np.int8)
    winner = np.where(done & (red > blue), 0, winner)
    winner = np.where(done & (blue > red), 1, winner)

    # A round that simply ran out of time is decided by what was scored and then
    # by who is left standing -- and here those are one comparison, not two.
    # Points on a timed-out round can only be hit points: a capture ends the
    # round, so there are none of those, and the win bonus is what is being
    # decided. With friendly fire off every hit is exactly one elimination and
    # nobody is eliminated twice, so a side's hits are the enemy's dead: more
    # points and more survivors are the same fact. Written as the survivor count
    # because that is what it is; a second line comparing hits would read as a
    # tiebreak and compute nothing.
    #
    # They part only if the two prices part -- POINTS_PER_DEFENCE is invited to
    # rise above POINTS_PER_HIT -- and a rule that read the scoreboard's prices
    # would put them in the gradient, which this project keeps them out of on
    # purpose. If that day comes the sim needs its own defence counter and its
    # own weights, not config.POINTS_*.
    #
    # Ground taken used to decide this, and the argument for it was that the
    # survivor count rewards the side that committed to least -- a team that
    # never left its own half winning by standing still. That is a real cost and
    # it is being paid deliberately: the objective is still the only thing worth
    # more than a kill, through REWARD_CAPTURE and the capture shaping, and it
    # ends the round outright. What goes is the consolation prize for having
    # stood on the strip without finishing the job. The price is more drawn
    # rounds among the ones that run out of time, because a level fight is now
    # level.
    survivors = np.where(red > blue, 0, np.where(blue > red, 1, -1))
    clock_only = timeout & ~wiped & (state.capture_winner < 0)
    winner = np.where(clock_only, survivors, winner)

    # Applied last so a capture overrides the survivor comparison above.
    winner = np.where(red_capture & ~blue_capture, 0, winner)
    winner = np.where(blue_capture & ~red_capture, 1, winner)
    winner = np.where(tied_capture, -1, winner)
    return done, winner


def step(state: SimState, move: np.ndarray, turn: np.ndarray, shoot: np.ndarray,
         autofire: bool = False) -> Dict[str, Any]:
    """Advance one control step: ACTION_REPEAT physics ticks under held actions."""
    events = {"arena": [], "shooter": [], "victim": [], "fire_step": [], "defence": []}
    shots = {"arena": [], "shooter": [], "x": [], "y": []}
    fired = np.zeros((state.n, state.p), dtype=bool)

    # Age the sound memory before the triggers are pulled, so a shot fired on
    # this step is still zero seconds old in the observation built after it.
    state.last_sound_age = np.minimum(state.last_sound_age + config.CONTROL_DT, 1e3)
    # Heat fades over roughly the same window the single muzzle memory keeps, so
    # the two cues describe the same stretch of time.
    state.sound_heat *= SOUND_HEAT_DECAY

    for sub in range(config.ACTION_REPEAT):
        if autofire and sub == 0:
            visible = compute_los(state)
            bearing, unused_flight = lead_solution(state)
            trigger = autofire_decision(state, visible, bearing)
        else:
            trigger = shoot if sub == 0 else np.zeros_like(shoot)

        apply_movement(state, move, turn)
        tick_firing, tick_shots = spawn_balls(state, trigger)
        fired |= tick_firing
        for key in shots:
            shots[key].append(tick_shots[key])
        tick_events = advance_balls(state)
        for key in events:
            events[key].append(tick_events[key])

        update_capture(state)
        latch_capture(state)
        state.cooldown = np.maximum(state.cooldown - 1, 0)
        # The match clock stops while a capture is actually accruing, so a side
        # that is a second from taking the strip cannot simply be saved by the
        # buzzer. Everything else - physics, projectiles, cooldowns, the agents
        # acting - keeps running; only the clock and its timeout hold still.
        state.tick += ~state.capture_running
        state.real_tick += 1

    state.step_idx += 1
    merged = {key: np.concatenate(value) if value else np.zeros(0, dtype=np.int32)
              for key, value in events.items()}
    merged_shots = {key: np.concatenate(value) for key, value in shots.items()}
    done, winner = episode_status(state)
    return {"hits": merged, "shots_fired": merged_shots, "fired": fired,
            "done": done, "winner": winner}


def main():
    pass


if __name__ == "__main__":
    main()
