#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Procedural arena generation.

Layouts are mirror-symmetric about the vertical centre line: whatever the red
half contains, the blue half contains reflected. Without that the two spawn
sides are not equally defensible and the agents learn the asymmetry instead of
the game.

Obstacles collapse to two physics primitives: axis-aligned rectangles and
circles. A pool of layouts is generated once and reused across episodes, which
keeps generation off the hot path and lets a raycast cache be attached later.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Local imports
from engine import config
from engine.env import themes

logger = logging.getLogger(__name__)

SHAPE_RECT = 0
SHAPE_CIRCLE = 1

# Obstacles keep clear of the walls by more than a player diameter, so the lane
# along the wall stays comfortably walkable instead of pinching the player.
SPAWN_MARGIN = 1.6
SPAWN_DEPTH_FRACTION = 0.18
CENTER_KEEPOUT = 1.2
GRID_STEP = 0.25

# How far an obstacle's extent has to stay clear of a spawn strip edge. Nothing
# may reach into the strip at all, and flush against the line is still too
# close: a wall grown straight off the edge is cover an attacker can stand
# behind while already touching the strip. More than a player radius (0.45), so
# a defender standing on his own spawn line has his whole body in the open.
SPAWN_CLEARANCE = 0.6

# Per-layout obstacle count, as a share of the requested count. A pool where
# every layout holds the same number of pieces reads as one arena with the
# furniture moved, and density is exactly the variable that decides whether a
# map plays as cover or as a maze. Expressed as a share rather than a fixed
# +/- so it follows the curriculum, whose early phases ask for as few as 4.
#
# The upper end stops at the requested count, which makes N_OBSTACLES a ceiling
# rather than a target: excluding both spawn strips leaves 63% of the area the
# old generator had, and rejection placement delivers about 0.71 of what it is
# asked for, so the default 18 realises as 8-18 with a mean near 12.6. The
# region does not actually saturate there - asking 26 measures 15.4 placed - so
# raising this band does buy density if a denser field is ever wanted. It is
# left at 1.0 because a knob that overshoots to hit its own number is worse than
# one that under-delivers visibly.
COUNT_BAND = (0.55, 1.0)


def pick_kind(theme: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """Sample one obstacle kind from a theme according to its weights."""
    kinds = theme["kinds"]
    weights = np.array([k["weight"] for k in kinds], dtype=np.float64)
    return kinds[int(rng.choice(len(kinds), p=weights / weights.sum()))]


def sample_obstacle(theme: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """Sample one unplaced obstacle (shape and half-extents, no position yet)."""
    kind = pick_kind(theme, rng)
    if kind["shape"] == "circle":
        radius = rng.uniform(*kind["r"])
        return {"kind": kind["kind"], "shape": SHAPE_CIRCLE, "hw": radius, "hh": radius}
    half_w = rng.uniform(*kind["w"]) * 0.5
    half_h = rng.uniform(*kind["h"]) * 0.5
    if rng.random() < 0.5:
        half_w, half_h = half_h, half_w
    return {"kind": kind["kind"], "shape": SHAPE_RECT, "hw": half_w, "hh": half_h}


def overlaps(a: Dict[str, Any], b: Dict[str, Any], gap: float) -> bool:
    """Conservative AABB overlap test with a gap, used during placement only."""
    return (abs(a["cx"] - b["cx"]) < a["hw"] + b["hw"] + gap and
            abs(a["cy"] - b["cy"]) < a["hh"] + b["hh"] + gap)


def placement_bounds(width: float, height: float) -> Tuple[float, float, float, float]:
    """Bounds an obstacle's full extent must stay inside, for the left half.

    Returns (x_min, x_max, y_min, y_max). The left bound is the red spawn strip
    plus clearance, and mirroring carries that same clearance to the blue strip:
    an obstacle whose left edge sits at red_x_max + SPAWN_CLEARANCE reflects to
    one whose right edge sits at blue_x_min - SPAWN_CLEARANCE. So constraining
    the half is enough to keep both strips clear.
    """
    spawn_x_max = spawn_box(0, width, height)[1]
    # The top bound is whichever keep-out is deeper. Taking the lane alone would
    # let ARENA_TOP_LANE=0 put obstacles flush against the top wall while the
    # bottom still keeps SPAWN_MARGIN, and an obstacle inflated by the player
    # radius then seals the very lane the wall margin exists to keep walkable.
    return (spawn_x_max + SPAWN_CLEARANCE,
            width * 0.5 - CENTER_KEEPOUT,
            max(SPAWN_MARGIN, config.ARENA_TOP_LANE),
            height - SPAWN_MARGIN)


def fits_region(item: Dict[str, Any], bounds: Tuple[float, float, float, float]) -> bool:
    """Is there any position at all where this obstacle's extent stays in bounds?"""
    x_min, x_max, y_min, y_max = bounds
    return x_min + item["hw"] < x_max - item["hw"] and y_min + item["hh"] < y_max - item["hh"]


def sample_fitting(theme: Dict[str, Any], bounds: Tuple[float, float, float, float],
                   rng: np.random.Generator, redraws: int = 4) -> Dict[str, Any]:
    """Sample an obstacle small enough to fit the region, if the theme has one.

    On the small curriculum arenas the placeable strip is narrower than the
    widest kind of several themes, so a single draw loses the slot outright
    rather than placing something smaller. Redrawing spends the slot instead of
    silently thinning the layout.
    """
    item = sample_obstacle(theme, rng)
    for unused_redraw in range(redraws):
        if fits_region(item, bounds):
            break
        item = sample_obstacle(theme, rng)
    return item


def place_half(theme: Dict[str, Any], width: float, height: float, n_pairs: int,
               rng: np.random.Generator) -> List[Dict[str, Any]]:
    """Place obstacles in the left half, clear of both spawn strips and the top lane.

    Excluding the whole spawn strip costs a lot of room: on the default 30x20 m
    field the region a piece can live in shrinks from 197 to 125 m2, and random
    rejection placement saturates it at roughly eight pieces. Asking for more
    does not produce more, which is what COUNT_BAND is sized against.
    """
    bounds = placement_bounds(width, height)
    x_min, x_max, y_min, y_max = bounds
    placed: List[Dict[str, Any]] = []
    gap = config.PLAYER_RADIUS * 2.4

    # Each slot draws its kind once and then hunts for a spot, instead of
    # redrawing the kind on every attempt. Redrawing made placement order a size
    # filter: a wide container could only ever succeed while the half was still
    # empty, so the big pieces always landed in the roomy middle and the small
    # ones always in the leftovers around them. Drawing first decouples what a
    # layout contains from where there happened to be room when it was drawn.
    for unused_slot in range(n_pairs):
        item = sample_fitting(theme, bounds, rng)
        if not fits_region(item, bounds):
            continue
        for unused_attempt in range(40):
            item["cx"] = rng.uniform(x_min + item["hw"], x_max - item["hw"])
            item["cy"] = rng.uniform(y_min + item["hh"], y_max - item["hh"])
            if not any(overlaps(item, other, gap) for other in placed):
                placed.append(item)
                break
    return placed


def place_center(theme: Dict[str, Any], width: float, height: float, n_items: int,
                 rng: np.random.Generator, blocked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Place obstacles straddling the centre line; they are self-symmetric.

    `blocked` is everything already on the field, mirrors included. A centre
    piece is only kept out of the halves by CENTER_KEEPOUT, which is narrower
    than the widest kinds are, so without this test a container on the centre
    line grows straight through a crate placed a metre into the half.
    """
    x_min, unused_x_max, y_min, y_max = placement_bounds(width, height)
    # A centre piece grows both ways from the centre line, so its own keep-out is
    # the half bound mirrored: on a small curriculum arena the widest kinds do
    # reach a spawn strip from the middle of the field.
    bounds = (x_min, width - x_min, y_min, y_max)
    placed: List[Dict[str, Any]] = []
    gap = config.PLAYER_RADIUS * 2.4
    for unused_slot in range(n_items):
        item = sample_fitting(theme, bounds, rng)
        if not fits_region(item, bounds):
            continue
        for unused_attempt in range(30):
            item["cx"] = width * 0.5
            item["cy"] = rng.uniform(y_min + item["hh"], y_max - item["hh"])
            if not any(overlaps(item, other, gap) for other in placed + blocked):
                placed.append(item)
                break
    return placed


def mirror(item: Dict[str, Any], width: float) -> Dict[str, Any]:
    """Reflect an obstacle about the vertical centre line."""
    twin = dict(item)
    twin["cx"] = width - item["cx"]
    return twin


def build_occupancy(obstacles: List[Dict[str, Any]], width: float, height: float) -> np.ndarray:
    """Rasterize walkable space, inflating obstacles by the player radius.

    Inflation matters: a gap narrower than the player looks passable on a raw
    raster but traps the agent in the actual simulation.
    """
    nx = int(width / GRID_STEP)
    ny = int(height / GRID_STEP)
    xs = (np.arange(nx) + 0.5) * GRID_STEP
    ys = (np.arange(ny) + 0.5) * GRID_STEP
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    free = np.ones((nx, ny), dtype=bool)

    pad = config.PLAYER_RADIUS
    free &= (gx > pad) & (gx < width - pad) & (gy > pad) & (gy < height - pad)

    for item in obstacles:
        dx = np.abs(gx - item["cx"])
        dy = np.abs(gy - item["cy"])
        if item["shape"] == SHAPE_CIRCLE:
            free &= (dx * dx + dy * dy) > (item["hw"] + pad) ** 2
        else:
            free &= ~((dx < item["hw"] + pad) & (dy < item["hh"] + pad))
    return free


def flood_fill(free: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
    """Breadth-first reachability over the 4-connected walkable raster."""
    nx, ny = free.shape
    seen = np.zeros_like(free)
    if not free[start]:
        return seen
    stack = [start]
    seen[start] = True
    while stack:
        x, y = stack.pop()
        for ux, uy in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= ux < nx and 0 <= uy < ny and free[ux, uy] and not seen[ux, uy]:
                seen[ux, uy] = True
                stack.append((ux, uy))
    return seen


def spawn_box(team: int, width: float, height: float) -> Tuple[float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max) of a team's spawn strip."""
    depth = width * SPAWN_DEPTH_FRACTION
    if team == 0:
        return SPAWN_MARGIN, depth, SPAWN_MARGIN, height - SPAWN_MARGIN
    return width - depth, width - SPAWN_MARGIN, SPAWN_MARGIN, height - SPAWN_MARGIN


def spawn_cells_reachable(free: np.ndarray, reached: np.ndarray, team: int,
                          width: float, height: float) -> int:
    """Count reachable raster cells inside a team's spawn strip."""
    x_min, x_max, y_min, y_max = spawn_box(team, width, height)
    ix0, ix1 = int(x_min / GRID_STEP), int(x_max / GRID_STEP)
    iy0, iy1 = int(y_min / GRID_STEP), int(y_max / GRID_STEP)
    return int(reached[ix0:ix1, iy0:iy1].sum())


def segment_blocked(x0: float, y0: float, x1: float, y1: float,
                    obstacles: List[Dict[str, Any]]) -> bool:
    """Does any obstacle interrupt the straight line between two points?"""
    dx, dy = x1 - x0, y1 - y0
    for item in obstacles:
        if item["shape"] == SHAPE_CIRCLE:
            mx, my = x0 - item["cx"], y0 - item["cy"]
            a = dx * dx + dy * dy
            b = 2.0 * (mx * dx + my * dy)
            c = mx * mx + my * my - item["hw"] ** 2
            disc = b * b - 4.0 * a * c
            if disc <= 0.0:
                continue
            t = (-b - math.sqrt(disc)) / (2.0 * a)
            if 0.0 <= t <= 1.0:
                return True
        else:
            lo_x, hi_x = item["cx"] - item["hw"], item["cx"] + item["hw"]
            lo_y, hi_y = item["cy"] - item["hh"], item["cy"] + item["hh"]
            inv_x = 1.0 / dx if abs(dx) > 1e-12 else 1e12
            inv_y = 1.0 / dy if abs(dy) > 1e-12 else 1e12
            t1x, t2x = (lo_x - x0) * inv_x, (hi_x - x0) * inv_x
            t1y, t2y = (lo_y - y0) * inv_y, (hi_y - y0) * inv_y
            t_near = max(min(t1x, t2x), min(t1y, t2y))
            t_far = min(max(t1x, t2x), max(t1y, t2y))
            if t_near <= t_far and 0.0 <= t_near <= 1.0:
                return True
    return False


def spawn_sightline_share(obstacles: List[Dict[str, Any]], width: float, height: float,
                          samples: int = 7) -> float:
    """Fraction of spawn-to-spawn sightlines that are wide open.

    A layout where the two spawns can shoot straight at each other resolves in
    seconds and teaches nothing about using cover, so generation rejects it.
    """
    red = spawn_box(0, width, height)
    blue = spawn_box(1, width, height)
    red_x = (red[0] + red[1]) * 0.5
    blue_x = (blue[0] + blue[1]) * 0.5
    ys = np.linspace(red[2] + 1.0, red[3] - 1.0, samples)

    open_lines = 0
    for y0 in ys:
        for y1 in ys:
            if not segment_blocked(red_x, y0, blue_x, y1, obstacles):
                open_lines += 1
    return open_lines / float(samples * samples)


def obstacle_target(n_obstacles: int, rng: np.random.Generator) -> int:
    """Draw this layout's obstacle count from a band around the requested count."""
    # An explicit request for zero obstacles must yield a bare arena: the first
    # curriculum phase trains on open ground, so the band never applies to it.
    if n_obstacles <= 0:
        return 0
    low = max(1, int(round(n_obstacles * COUNT_BAND[0])))
    high = max(low, int(round(n_obstacles * COUNT_BAND[1])))
    return int(rng.integers(low, high + 1))


def generate_layout(width: float, height: float, n_obstacles: int, theme_key: Optional[str],
                    rng: np.random.Generator, max_tries: int = 30) -> Dict[str, Any]:
    """Generate one connected, mirror-symmetric layout."""
    for unused_try in range(max_tries):
        theme = themes.get_theme(theme_key) if theme_key else themes.THEMES[int(rng.integers(len(themes.THEMES)))]
        target = obstacle_target(n_obstacles, rng)
        n_center = int(rng.integers(0, 3)) if target >= 2 else 0
        n_pairs = max(0, (target - n_center) // 2)

        half = place_half(theme, width, height, n_pairs, rng)
        mirrored = half + [mirror(item, width) for item in half]
        center = place_center(theme, width, height, n_center, rng, mirrored)
        obstacles = mirrored + center

        free = build_occupancy(obstacles, width, height)
        x_min, unused_x_max, y_min, y_max = spawn_box(0, width, height)
        seed_x = int((x_min + 0.5) / GRID_STEP)
        seed_y = int((y_min + y_max) * 0.5 / GRID_STEP)
        if not free[seed_x, seed_y]:
            continue
        reached = flood_fill(free, (seed_x, seed_y))

        red_cells = spawn_cells_reachable(free, reached, 0, width, height)
        blue_cells = spawn_cells_reachable(free, reached, 1, width, height)
        if red_cells < 40 or blue_cells < 40:
            continue
        if reached.sum() < free.sum() * 0.85:
            continue
        if obstacles and spawn_sightline_share(obstacles, width, height) > 0.25:
            continue

        return {"theme": theme["key"], "obstacles": obstacles, "width": width, "height": height}

    logger.warning(f"Layout generation fell back to an empty arena after {max_tries} tries")
    theme = themes.get_theme(theme_key) if theme_key else themes.THEMES[0]
    return {"theme": theme["key"], "obstacles": [], "width": width, "height": height}


class ArenaPool:
    """A fixed set of layouts held as flat numpy arrays for batched simulation."""

    def __init__(self, layouts: List[Dict[str, Any]], width: float, height: float):
        self.width = width
        self.height = height
        self.layouts = layouts
        n = len(layouts)
        m = config.MAX_OBSTACLES

        self.shape  = np.zeros((n, m), dtype=np.int8)
        self.cx     = np.zeros((n, m), dtype=np.float32)
        self.cy     = np.zeros((n, m), dtype=np.float32)
        self.hw     = np.zeros((n, m), dtype=np.float32)
        self.hh     = np.zeros((n, m), dtype=np.float32)
        self.active = np.zeros((n, m), dtype=bool)
        self.theme_idx = np.zeros(n, dtype=np.int8)
        # Compiled kernels iterate obstacles by count instead of masking a padded
        # width, so an arena with 18 obstacles costs 18 tests, not MAX_OBSTACLES.
        self.count = np.zeros(n, dtype=np.int64)

        for i, layout in enumerate(layouts):
            self.theme_idx[i] = themes.THEME_KEYS.index(layout["theme"])
            self.count[i] = min(len(layout["obstacles"]), m)
            for j, item in enumerate(layout["obstacles"][:m]):
                self.shape[i, j]  = item["shape"]
                self.cx[i, j]     = item["cx"]
                self.cy[i, j]     = item["cy"]
                self.hw[i, j]     = item["hw"]
                self.hh[i, j]     = item["hh"]
                self.active[i, j] = True

        # Inactive slots are collapsed to a degenerate point outside the arena so
        # that vectorized intersection maths never has to branch on the mask.
        self.cx[~self.active] = -1000.0
        self.cy[~self.active] = -1000.0

    def __len__(self) -> int:
        return len(self.layouts)

    def describe(self, index: int) -> Dict[str, Any]:
        """Return a JSON-serialisable description of one layout, for whatever draws it."""
        layout = self.layouts[index]
        theme = themes.get_theme(layout["theme"])
        return {
            "theme": theme["key"],
            "theme_name": theme["name"],
            "palette": {
                "floor": theme["floor"], "floor_alt": theme["floor_alt"],
                "grid": theme["grid"], "fog": theme["fog"],
            },
            "width": self.width,
            "height": self.height,
            "obstacles": [
                {
                    "kind": item["kind"], "shape": item["shape"],
                    "cx": round(item["cx"], 3), "cy": round(item["cy"], 3),
                    "hw": round(item["hw"], 3), "hh": round(item["hh"], 3),
                    **kind_style(theme, item["kind"]),
                }
                for item in layout["obstacles"]
            ],
        }


def kind_style(theme: Dict[str, Any], kind_name: str) -> Dict[str, str]:
    """Look up renderer colours for an obstacle kind within a theme."""
    for kind in theme["kinds"]:
        if kind["kind"] == kind_name:
            style = {"color": kind["color"], "edge": kind["edge"]}
            if "glow" in kind:
                style["glow"] = kind["glow"]
            return style
    return {"color": "#888888", "edge": "#555555"}


def build_pool(n: int, width: float = config.ARENA_W, height: float = config.ARENA_H,
               n_obstacles: int = config.N_OBSTACLES, theme_key: Optional[str] = None,
               seed: int = 0) -> ArenaPool:
    """Generate a pool of layouts, cycling themes so all of them appear."""
    rng = np.random.default_rng(seed)
    layouts = []
    for i in range(n):
        key = theme_key if theme_key else themes.THEME_KEYS[i % len(themes.THEME_KEYS)]
        layouts.append(generate_layout(width, height, n_obstacles, key, rng))
    return ArenaPool(layouts, width, height)


def main():
    pass


if __name__ == "__main__":
    main()
