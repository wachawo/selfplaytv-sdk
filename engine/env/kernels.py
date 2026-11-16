#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compiled inner loops for the simulation hot path.

The vectorized numpy formulation of raycasting and line-of-sight allocates
intermediate arrays of shape (arenas, rays, obstacles, 2) and ends up bound by
memory bandwidth rather than arithmetic. Written as explicit loops and compiled
with numba, the same maths runs from registers and parallelises across arenas.

Every kernel here is a drop-in replacement for a numpy version that is kept in
sim.py, as the reference the kernel was written against.
"""

import math

import numpy as np
from numba import njit, prange

SHAPE_CIRCLE = 1
INF = 1e9

JIT = {"cache": True, "fastmath": True, "nogil": True}


@njit(inline="always")
def ray_vs_obstacles(ox, oy, dx, dy, shape, cx, cy, hw, hh, count, max_t):
    """Nearest hit fraction of a ray against one arena's obstacle set."""
    best = max_t
    for k in range(count):
        if shape[k] == SHAPE_CIRCLE:
            mx = ox - cx[k]
            my = oy - cy[k]
            a = dx * dx + dy * dy
            b = 2.0 * (mx * dx + my * dy)
            c = mx * mx + my * my - hw[k] * hw[k]
            disc = b * b - 4.0 * a * c
            if disc <= 0.0 or a <= 0.0:
                continue
            root = math.sqrt(disc)
            t = (-b - root) / (2.0 * a)
            if c < 0.0:
                t = 0.0
            if 0.0 <= t < best:
                best = t
        else:
            lo_x = cx[k] - hw[k]
            hi_x = cx[k] + hw[k]
            lo_y = cy[k] - hh[k]
            hi_y = cy[k] + hh[k]
            inv_x = 1.0 / dx if abs(dx) > 1e-12 else 1e12
            inv_y = 1.0 / dy if abs(dy) > 1e-12 else 1e12
            t1x = (lo_x - ox) * inv_x
            t2x = (hi_x - ox) * inv_x
            t1y = (lo_y - oy) * inv_y
            t2y = (hi_y - oy) * inv_y
            near_x = t1x if t1x < t2x else t2x
            far_x = t2x if t1x < t2x else t1x
            near_y = t1y if t1y < t2y else t2y
            far_y = t2y if t1y < t2y else t1y
            t_near = near_x if near_x > near_y else near_y
            t_far = far_x if far_x < far_y else far_y
            if t_near < 0.0:
                t_near = 0.0
            if lo_x <= ox <= hi_x and lo_y <= oy <= hi_y:
                return 0.0
            if t_near <= t_far and t_far >= 0.0 and t_near < best:
                best = t_near
    return best


@njit(inline="always")
def ray_exit_box(ox, oy, dx, dy, width, height):
    """Fraction of the ray at which it leaves the arena rectangle.

    Both slab crossings are computed and the FAR one kept per axis, then the
    nearer of the two axes. Picking the crossing by the sign of the direction
    instead breaks on a ray exactly parallel to an axis: the sentinel that
    stands in for the infinite inverse is positive, the numerator it multiplies
    is negative, and the axis the ray never leaves returns a large negative
    exit fraction that then wins the minimum. The distance came out around
    -1e12, and ray distance is the one observation feature that is not clipped,
    so it reached the critic intact and took the value function with it.
    Taking the far crossing sends the sentinel to +1e12, where the minimum
    correctly ignores it. This is the same formulation as the numpy reference
    in sim.segment_box_exit, which was always right.
    """
    inv_x = 1.0 / dx if abs(dx) > 1e-12 else 1e12
    inv_y = 1.0 / dy if abs(dy) > 1e-12 else 1e12
    tx1 = (0.0 - ox) * inv_x
    tx2 = (width - ox) * inv_x
    ty1 = (0.0 - oy) * inv_y
    ty2 = (height - oy) * inv_y
    tx = tx1 if tx1 > tx2 else tx2
    ty = ty1 if ty1 > ty2 else ty2
    return tx if tx < ty else ty


@njit(parallel=True, **JIT)
def cast_rays(pos, heading, alive, team, angles, shape, cx, cy, hw, hh, counts,
              width, height, max_sight, player_radius, out_dist, out_kind):
    """Cast R rays per player. Fills (N, P, R) distance and class arrays."""
    n, p = pos.shape[0], pos.shape[1]
    r = angles.shape[0]
    for i in prange(n):
        count = counts[i]
        for a in range(p):
            ox = pos[i, a, 0]
            oy = pos[i, a, 1]
            for q in range(r):
                theta = heading[i, a] + angles[q]
                dx = math.cos(theta) * max_sight
                dy = math.sin(theta) * max_sight

                t_static = ray_vs_obstacles(ox, oy, dx, dy, shape[i], cx[i], cy[i],
                                            hw[i], hh[i], count, 1.0)
                t_bounds = ray_exit_box(ox, oy, dx, dy, width, height)
                if t_bounds < t_static:
                    t_static = t_bounds

                best_body = t_static
                nearest = -1
                for b in range(p):
                    if b == a or not alive[i, b]:
                        continue
                    mx = ox - pos[i, b, 0]
                    my = oy - pos[i, b, 1]
                    aa = dx * dx + dy * dy
                    bb = 2.0 * (mx * dx + my * dy)
                    cc = mx * mx + my * my - player_radius * player_radius
                    disc = bb * bb - 4.0 * aa * cc
                    if disc <= 0.0:
                        continue
                    root = math.sqrt(disc)
                    t = (-bb - root) / (2.0 * aa)
                    if cc < 0.0:
                        t = 0.0
                    if 0.0 <= t < best_body:
                        best_body = t
                        nearest = b

                out_dist[i, a, q] = best_body * max_sight
                if nearest < 0:
                    out_kind[i, a, q] = 0
                elif team[nearest] == team[a]:
                    out_kind[i, a, q] = 2
                else:
                    out_kind[i, a, q] = 1


@njit(parallel=True, **JIT)
def compute_los(pos, alive, shape, cx, cy, hw, hh, counts, player_radius, out):
    """Fill the (N, P, P) visibility matrix. Symmetric, so each pair is done once."""
    n, p = pos.shape[0], pos.shape[1]
    for i in prange(n):
        count = counts[i]
        for a in range(p):
            for b in range(p):
                out[i, a, b] = False
        for a in range(p):
            if not alive[i, a]:
                continue
            for b in range(a + 1, p):
                if not alive[i, b]:
                    continue
                ox = pos[i, a, 0]
                oy = pos[i, a, 1]
                dx = pos[i, b, 0] - ox
                dy = pos[i, b, 1] - oy
                if ray_vs_obstacles(ox, oy, dx, dy, shape[i], cx[i], cy[i],
                                    hw[i], hh[i], count, 1.0) < 1.0:
                    continue

                # A third body standing in the lane blocks sight even though paint
                # passes through teammates.
                occluded = False
                for c in range(p):
                    if c == a or c == b or not alive[i, c]:
                        continue
                    mx = ox - pos[i, c, 0]
                    my = oy - pos[i, c, 1]
                    aa = dx * dx + dy * dy
                    bb = 2.0 * (mx * dx + my * dy)
                    cc = mx * mx + my * my - (player_radius * 0.9) ** 2
                    disc = bb * bb - 4.0 * aa * cc
                    if disc <= 0.0 or aa <= 0.0:
                        continue
                    root = math.sqrt(disc)
                    t = (-bb - root) / (2.0 * aa)
                    if cc < 0.0:
                        t = 0.0
                    if 0.0 <= t < 0.999:
                        occluded = True
                        break
                if not occluded:
                    out[i, a, b] = True
                    out[i, b, a] = True


@njit(parallel=True, **JIT)
def separate_players(pos, vel, alive, radius):
    """Part players who overlap, and let the one driving in carry the other along.

    Every displacement is read out of the positions this tick arrived with and
    banked, then applied once at the end. Resolved in place instead, the answer
    would depend on the order the slots are walked in -- and the slots are the
    two teams in a row, so the red team would be settled against a blue team
    that had not moved yet, every tick, on a field built to be mirror-symmetric.

    The push alone would only stop the overlap, not the walking: both would keep
    their speed, drive back in and be parted again next tick. So the speed the
    two are closing at is shared between them, which is what one body pushing
    another actually does -- the one behind slows, the one in front is carried
    off, and neither passes through.
    """
    n, p = pos.shape[0], pos.shape[1]
    want = 2.0 * radius
    for i in prange(n):
        push = np.zeros((p, 2), dtype=np.float32)
        kick = np.zeros((p, 2), dtype=np.float32)
        for a in range(p):
            if not alive[i, a]:
                continue
            for b in range(p):
                if b == a or not alive[i, b]:
                    continue
                dx = pos[i, a, 0] - pos[i, b, 0]
                dy = pos[i, a, 1] - pos[i, b, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist >= want:
                    continue
                if dist < 1e-6:
                    # Exactly on top of one another: no line of centres to part
                    # along, so the lower slot goes one way and the higher the
                    # other. Asked of either of the two, that is the same answer.
                    nx = 1.0 if a > b else -1.0
                    ny = 0.0
                else:
                    nx = dx / dist
                    ny = dy / dist
                depth = want - dist
                push[a, 0] += 0.5 * depth * nx
                push[a, 1] += 0.5 * depth * ny
                approach = ((vel[i, a, 0] - vel[i, b, 0]) * nx
                            + (vel[i, a, 1] - vel[i, b, 1]) * ny)
                # Only while they are still closing. Taking it from a pair already
                # moving apart would glue them together.
                if approach < 0.0:
                    kick[a, 0] -= 0.5 * approach * nx
                    kick[a, 1] -= 0.5 * approach * ny
        for a in range(p):
            pos[i, a, 0] += push[a, 0]
            pos[i, a, 1] += push[a, 1]
            vel[i, a, 0] += kick[a, 0]
            vel[i, a, 1] += kick[a, 1]


@njit(parallel=True, **JIT)
def resolve_collisions(pos, vel, shape, cx, cy, hw, hh, counts, radius, width, height, passes):
    """Push players out of obstacles, then clamp them inside the arena walls."""
    n, p = pos.shape[0], pos.shape[1]
    for i in prange(n):
        count = counts[i]
        for a in range(p):
            for unused_pass in range(passes):
                deepest = -1.0
                push_x = 0.0
                push_y = 0.0
                px = pos[i, a, 0]
                py = pos[i, a, 1]
                for k in range(count):
                    if shape[i, k] == SHAPE_CIRCLE:
                        dx = px - cx[i, k]
                        dy = py - cy[i, k]
                        dist = math.sqrt(dx * dx + dy * dy)
                        want = hw[i, k] + radius
                    else:
                        clamp_x = px - cx[i, k]
                        clamp_y = py - cy[i, k]
                        if clamp_x < -hw[i, k]:
                            clamp_x = -hw[i, k]
                        elif clamp_x > hw[i, k]:
                            clamp_x = hw[i, k]
                        if clamp_y < -hh[i, k]:
                            clamp_y = -hh[i, k]
                        elif clamp_y > hh[i, k]:
                            clamp_y = hh[i, k]
                        dx = px - (cx[i, k] + clamp_x)
                        dy = py - (cy[i, k] + clamp_y)
                        dist = math.sqrt(dx * dx + dy * dy)
                        want = radius
                    depth = want - dist
                    if depth > deepest:
                        deepest = depth
                        if dist < 1e-6:
                            push_x = want
                            push_y = 0.0
                        else:
                            push_x = dx / dist * depth
                            push_y = dy / dist * depth
                if deepest <= 0.0:
                    break
                pos[i, a, 0] += push_x
                pos[i, a, 1] += push_y

                if pos[i, a, 0] < radius:
                    pos[i, a, 0] = radius
                    vel[i, a, 0] = 0.0
                elif pos[i, a, 0] > width - radius:
                    pos[i, a, 0] = width - radius
                    vel[i, a, 0] = 0.0
                if pos[i, a, 1] < radius:
                    pos[i, a, 1] = radius
                    vel[i, a, 1] = 0.0
                elif pos[i, a, 1] > height - radius:
                    pos[i, a, 1] = height - radius
                    vel[i, a, 1] = 0.0

            if pos[i, a, 0] < radius:
                pos[i, a, 0] = radius
                vel[i, a, 0] = 0.0
            elif pos[i, a, 0] > width - radius:
                pos[i, a, 0] = width - radius
                vel[i, a, 0] = 0.0
            if pos[i, a, 1] < radius:
                pos[i, a, 1] = radius
                vel[i, a, 1] = 0.0
            elif pos[i, a, 1] > height - radius:
                pos[i, a, 1] = height - radius
                vel[i, a, 1] = 0.0


@njit(parallel=True, **JIT)
def sweep_balls(ball_pos, ball_vel, ball_active, ball_ttl, pos, alive, team,
                shape, cx, cy, hw, hh, counts, dt, ball_radius, player_radius,
                friendly_fire, hit_victim, hit_time):
    """Advance every projectile one tick, recording the earliest player contact.

    Writes -1 into hit_victim for slots that hit nothing, and deactivates balls
    that struck an obstacle or expired. Positions are integrated in place.
    """
    n, p, b = ball_pos.shape[0], ball_pos.shape[1], ball_pos.shape[2]
    for i in prange(n):
        count = counts[i]
        for s in range(p):
            for j in range(b):
                hit_victim[i, s, j] = -1
                hit_time[i, s, j] = INF
                if not ball_active[i, s, j]:
                    continue
                ox = ball_pos[i, s, j, 0]
                oy = ball_pos[i, s, j, 1]
                dx = ball_vel[i, s, j, 0] * dt
                dy = ball_vel[i, s, j, 1] * dt

                t_wall = ray_vs_obstacles(ox, oy, dx, dy, shape[i], cx[i], cy[i],
                                          hw[i], hh[i], count, INF)

                best = INF
                victim = -1
                reach = player_radius + ball_radius
                for v in range(p):
                    if v == s or not alive[i, v]:
                        continue
                    if not friendly_fire and team[v] == team[s]:
                        continue
                    mx = ox - pos[i, v, 0]
                    my = oy - pos[i, v, 1]
                    aa = dx * dx + dy * dy
                    bb = 2.0 * (mx * dx + my * dy)
                    cc = mx * mx + my * my - reach * reach
                    disc = bb * bb - 4.0 * aa * cc
                    if disc <= 0.0 or aa <= 0.0:
                        continue
                    root = math.sqrt(disc)
                    t = (-bb - root) / (2.0 * aa)
                    if cc < 0.0:
                        t = 0.0
                    if 0.0 <= t <= 1.0 and t < best:
                        best = t
                        victim = v

                ball_pos[i, s, j, 0] = ox + dx
                ball_pos[i, s, j, 1] = oy + dy
                ball_ttl[i, s, j] -= 1

                if victim >= 0 and best <= t_wall:
                    hit_victim[i, s, j] = victim
                    hit_time[i, s, j] = best
                    ball_active[i, s, j] = False
                elif t_wall <= 1.0:
                    ball_active[i, s, j] = False
                elif ball_ttl[i, s, j] <= 0:
                    ball_active[i, s, j] = False


def warmup(team_size: int = 3) -> None:
    """Force compilation up front so the first training step is not a stall."""
    n, p, b, m, r = 2, team_size * 2, 2, 4, 4
    pos = np.zeros((n, p, 2), dtype=np.float32)
    vel = np.zeros((n, p, 2), dtype=np.float32)
    heading = np.zeros((n, p), dtype=np.float32)
    alive = np.ones((n, p), dtype=np.bool_)
    team = np.zeros(p, dtype=np.int8)
    angles = np.zeros(r, dtype=np.float32)
    shape = np.zeros((n, m), dtype=np.int8)
    cx = np.zeros((n, m), dtype=np.float32)
    cy = np.zeros((n, m), dtype=np.float32)
    hw = np.ones((n, m), dtype=np.float32)
    hh = np.ones((n, m), dtype=np.float32)
    counts = np.zeros(n, dtype=np.int64)
    out_dist = np.zeros((n, p, r), dtype=np.float32)
    out_kind = np.zeros((n, p, r), dtype=np.int8)
    los = np.zeros((n, p, p), dtype=np.bool_)

    cast_rays(pos, heading, alive, team, angles, shape, cx, cy, hw, hh, counts,
              30.0, 20.0, 40.0, 0.45, out_dist, out_kind)
    compute_los(pos, alive, shape, cx, cy, hw, hh, counts, 0.45, los)
    resolve_collisions(pos, vel, shape, cx, cy, hw, hh, counts, 0.45, 30.0, 20.0, 4)

    ball_pos = np.zeros((n, p, b, 2), dtype=np.float32)
    ball_vel = np.zeros((n, p, b, 2), dtype=np.float32)
    ball_active = np.zeros((n, p, b), dtype=np.bool_)
    ball_ttl = np.zeros((n, p, b), dtype=np.int16)
    hit_victim = np.zeros((n, p, b), dtype=np.int8)
    hit_time = np.zeros((n, p, b), dtype=np.float32)
    sweep_balls(ball_pos, ball_vel, ball_active, ball_ttl, pos, alive, team,
                shape, cx, cy, hw, hh, counts, 0.05, 0.1, 0.45, False,
                hit_victim, hit_time)


def main():
    warmup()


if __name__ == "__main__":
    main()
