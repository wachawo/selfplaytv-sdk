#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Play demo matches between two teams and draw them in the terminal.

    python -m sdk.demo --bundle my-bundle "MY TEAM" WHITE
    python -m sdk.demo --bundle my-bundle --run runs/my "MY TEAM" WHITE --speed 4
    python -m sdk.demo WHITE BLACK --matches 5 --summary

Run from the repository root. A match is first to ROUNDS_TO_WIN rounds on the
full arena, whatever the curriculum would give a young team: a demo shows the
game the team is being trained for. The weights come from the bundle, or with
--run from RUN/checkpoints, which is the team as the training has it right now;
neither is ever written.

The field is seen from above and drawn with half-block characters, two pixels to
a character cell: the floor and the obstacles of the map, the two spawn strips,
every player as a block in the team's colour with a nose where it is facing,
the dead as a dim cross, and the paint in the air. Under it: the match score,
the round and its clock, who is still standing, the points of this match and
the average points a round.

NOTHING HERE STEPS THE GAME ITSELF. The matches are played by the trainer's own
rollout (trainer.selfplaytv.collect_rollout), the one sdk.train runs, with
nothing learning from it: the same observations, the same sampled actions, the
same order of slots. A second loop written here would be a second answer to
how a model is asked for a move, and the first difference between the two
would show a team playing worse than it is. The picture is drawn from the
trainer's step hook, and the points are the trainer's own tally, which is what
sdk.train counts, so the two cannot disagree about a round.

A terminal that speaks the kitty graphics protocol is sent the same picture as
real pixels instead (--graphics, auto by default: it asks the terminal, and
tmux, screen and a dumb terminal get half blocks without being asked).

With --summary, or whenever the output is not a terminal, nothing is drawn: the
matches are played side by side, an arena each, and the result is printed.
Drawn matches are played one after another in one arena, so a seed gives the
same matches in both ways only for a single match.
"""

import argparse
import logging
import logging.handlers
import os
import shutil
import signal
import sys
import textwrap
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Local imports
from sdk import graphics
from sdk import train
from sdk import watch

try:
    import termios
except ImportError:
    # Windows has none, and its console does not echo into the picture this way.
    termios = None

logger = logging.getLogger(__name__)

# One thread for numba and for torch. A demo is one arena, or a handful: there
# is nothing in it for a second thread to take, and it is watched while a
# training run holds the other cores.
THREADS = 1

# Arenas played side by side when nothing is drawn. A step costs about the same
# for one arena as for thirty, so matches played together take the time of the
# longest; past this, every arena more is a rollout buffer a model on the field
# is given by the trainer and that nothing here ever learns from.
SUMMARY_ARENAS = 32

# The lines of text under the field, and the smallest picture worth drawing: at
# 1.5 pixels to the metre a player's block of three pixels is already two
# metres across, and below it ten of them cannot be told apart.
TEXT_LINES = 5
MIN_SCALE = 1.5

# The static part of the picture is drawn this many times finer and averaged
# down, so a round obstacle a few pixels across is round and not a staircase.
SUPERSAMPLE = 3
# The floor is laid in squares of this many metres, the second colour of the
# theme washed over every other one.
TILE = 2.0
TILE_SHARE = 0.35
# How far into an obstacle its edge colour reaches, metres.
OBSTACLE_EDGE = 0.18
# A spawn strip is a whisper of its team's colour and an outline; the outline
# is what turns into the attackers' colour while they hold it.
STRIP_FILL = 0.08
STRIP_LINE = 0.3
# A player's block is all team colour where that colour stands this far off the
# floor, the contrast WCAG asks of a graphical object. Closer than that, the
# block is framed: see team_look.
BODY_CONTRAST = 3.0
# How much of the floor shows through the mark a dead player leaves.
DEAD_SHARE = 0.5

# Everything outside the field, and the text on it.
PANEL = "#0b0d10"
TEXT = "#c9d1d9"
QUIET = "#7d8590"

# Frames a second at most. Faster than this --speed only skips pictures: the
# game is ten steps a second, and a terminal fed two hundred frames a second
# spends its time parsing escapes nobody sees.
MAX_FPS = 30.0
# How long a finished round and a finished match stay on the screen, in seconds
# of the game, so --speed shortens them with everything else.
ROUND_PAUSE = 1.5
MATCH_PAUSE = 4.0

# The alternate screen, so the shell's own lines are back when the demo ends,
# and the two marks that let a terminal show a frame whole or not at all.
ENTER_SCREEN = "\x1b[?1049h"
LEAVE_SCREEN = "\x1b[?1049l"
FRAME_BEGIN = "\x1b[?2026h"
FRAME_END = "\x1b[?2026l"
HALF_BLOCK = "▀"

# How a round ended, in the order of train.REASON_KEYS.
REASON_WORDS = ("by capture", "by elimination", "on time")
# The summary's columns after the points, and the trainer.tally keys behind them.
PLAYER_COLUMNS = (("hits", "hits"), ("hit", "deaths"), ("shots", "shots"), ("captures", "captures"))
# The signals that end a drawn demo the way Ctrl-C does, where the platform has
# them: a kill, a closed terminal and Ctrl-\. Left to their defaults they end
# the process on the alternate screen with the cursor hidden and the echo off.
STOP_SIGNALS = ("SIGTERM", "SIGHUP", "SIGQUIT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m sdk.demo",
        description="Play demo matches between two teams, drawn in the terminal")
    parser.add_argument("teams", nargs=2, metavar="TEAM", help='the two teams, e.g. "MY TEAM" WHITE')
    parser.add_argument("--bundle", default=None,
                        help="bundle directory (default: the one --run was started from, else bundle)")
    parser.add_argument("--run", default=None,
                        help="take the weights from this run's checkpoints, as training has them now")
    parser.add_argument("--matches", type=int, default=1, help="how many matches to play (default 1)")
    parser.add_argument("--speed", type=float, default=1.0, help="times real time (default 1)")
    parser.add_argument("--theme", default=None, help="one map theme (default: all of them)")
    parser.add_argument("--seed", type=int, default=None,
                        help="play the same matches again (default: new ones every time)")
    parser.add_argument("--summary", action="store_true",
                        help="draw nothing: play as fast as possible and print the result")
    parser.add_argument("--graphics", choices=("auto", "kitty", "blocks"), default="auto",
                        help="kitty: the picture as pixels through the kitty graphics protocol; blocks: half "
                             "blocks; auto (default): kitty where the terminal answers that it can")
    return parser.parse_args()


def find_sources(args: argparse.Namespace) -> Tuple[str, str]:
    """The bundle that names the teams and the directory the weights are read from."""
    bundle = args.bundle
    weights = None
    if args.run is not None:
        run = os.path.realpath(args.run)
        weights = os.path.join(run, train.CHECKPOINTS_NAME)
        if not os.path.isfile(os.path.join(weights, train.MANIFEST_NAME)):
            train.refuse(f"{run} holds no {train.CHECKPOINTS_NAME}/{train.MANIFEST_NAME}; "
                         f"--run takes a directory sdk.train has written")
        marker = os.path.join(run, train.MARKER_NAME)
        if bundle is None and os.path.isfile(marker):
            bundle = train.read_json(marker).get("bundle")
    bundle = os.path.realpath(bundle or "bundle")
    for name in (train.TEAMS_NAME, train.MANIFEST_NAME):
        if not os.path.isfile(os.path.join(bundle, name)):
            train.refuse(f"{bundle} holds no {name}; a bundle is a directory with "
                         f"{train.TEAMS_NAME}, {train.MANIFEST_NAME} and one .pt per model")
    return bundle, weights or bundle


def check_weights(weights: str, teams: Dict[str, Any], playing: List[str]) -> None:
    """Every model on the field has a checkpoint; a missing one would play as a random network."""
    for team in playing:
        missing = [nick for nick in teams[team]["members"]
                   if not os.path.isfile(os.path.join(weights, f"{nick}.pt"))]
        if missing:
            train.refuse(f"{team}: no checkpoint for {', '.join(missing)} in {weights}")


def screen_room(field: Tuple[float, float]) -> Tuple[int, int]:
    """The smallest terminal, columns and lines, the field can be drawn in."""
    width, height = field
    return int(np.ceil(width * MIN_SCALE)), int(np.ceil(height * MIN_SCALE / 2.0)) + TEXT_LINES


def too_small(size: os.terminal_size, field: Tuple[float, float]) -> Optional[str]:
    columns, lines = screen_room(field)
    if size.columns >= columns and size.lines >= lines:
        return None
    return (f"The terminal is {size.columns}x{size.lines} and the field needs at least "
            f"{columns}x{lines}; make it larger, or use --summary")


def rgb(colour: str) -> np.ndarray:
    """An "#rrggbb" colour as three floats."""
    return np.array([int(colour[index:index + 2], 16) for index in (1, 3, 5)], dtype=np.float32)


def mix(under: np.ndarray, over: np.ndarray, share: float) -> np.ndarray:
    return under * (1.0 - share) + over * share


def pack(image: np.ndarray) -> np.ndarray:
    """(H, W, 3) colours as one integer a pixel, so a cell compares in one test."""
    whole = np.clip(np.rint(image), 0, 255).astype(np.uint32)
    return (whole[..., 0] << 16) | (whole[..., 1] << 8) | whole[..., 2]


def nearest_256(red: int, green: int, blue: int) -> int:
    """The xterm 256-colour index closest to a colour: the 6x6x6 cube or the grey ramp."""
    levels = (0, 95, 135, 175, 215, 255)
    cube = [min(range(6), key=lambda index: abs(levels[index] - value)) for value in (red, green, blue)]
    grey = min(23, max(0, round(((red + green + blue) / 3.0 - 8.0) / 10.0)))
    shade = 8 + 10 * grey

    def distance(first: int, second: int, third: int) -> int:
        return (first - red) ** 2 + (second - green) ** 2 + (third - blue) ** 2

    if distance(shade, shade, shade) < distance(*[levels[index] for index in cube]):
        return 232 + grey
    return 16 + 36 * cube[0] + 6 * cube[1] + cube[2]


def colour_code(view: Dict[str, Any], colour: int, layer: int) -> str:
    """The escape that sets one colour, foreground (layer 38) or background (48).

    24-bit where the terminal says it has it and the nearest of 256 where it
    does not. Kept by colour: a frame holds a few dozen colours and thousands
    of cells.
    """
    code = view["codes"].get((colour, layer))
    if code is None:
        red, green, blue = colour >> 16, (colour >> 8) & 255, colour & 255
        if view["truecolour"]:
            code = f"\x1b[{layer};2;{red};{green};{blue}m"
        else:
            code = f"\x1b[{layer};5;{nearest_256(red, green, blue)}m"
        view["codes"][(colour, layer)] = code
    return code


def text_code(view: Dict[str, Any], colour: str, layer: int) -> str:
    """colour_code for an "#rrggbb" colour."""
    return colour_code(view, int(pack(rgb(colour))), layer)


def disc(reach: float) -> np.ndarray:
    """The pixel offsets (dy, dx) within `reach` of a centre."""
    span = int(reach)
    down, across = np.mgrid[-span:span + 1, -span:span + 1]
    keep = down * down + across * across <= reach * reach
    return np.stack([down[keep], across[keep]], axis=1)


def cross(size: int) -> np.ndarray:
    """The pixel offsets of a diagonal cross reaching `size` pixels from its centre."""
    reach = np.arange(-size, size + 1)
    return np.concatenate([np.stack([reach, reach], axis=1), np.stack([reach, -reach], axis=1)])


def stamp(image: np.ndarray, offsets: np.ndarray, x: int, y: int, colour: np.ndarray) -> None:
    """Paint a shape of pixel offsets at (x, y), the part of it that is on the field."""
    down = offsets[:, 0] + y
    across = offsets[:, 1] + x
    keep = (down >= 0) & (down < image.shape[0]) & (across >= 0) & (across < image.shape[1])
    image[down[keep], across[keep]] = colour


def team_look(colour: str, floor: str) -> Dict[str, Any]:
    """The colours one team is drawn in on one floor.

    `body` is the team's colour, untouched: it is what the team is. `ink` is
    what lines, noses and paint are drawn in, a single pixel having nothing but
    its own colour to be seen by: the team's colour where that stands off the
    floor, and otherwise that colour moved away from the floor until it does.
    Where it had to be moved -- a black team on a night field, a white one on
    snow -- the block is `framed`: its outer pixels are drawn in the ink and
    the team's colour is the middle of it, so the two teams stay the same size
    and neither is a hole in the floor.
    """
    themes = train.themes
    framed = themes.contrast(colour, floor) < BODY_CONTRAST
    ink = rgb(themes.readable(colour, floor) if framed else colour)
    return {"body": rgb(colour), "ink": ink, "gone": mix(rgb(floor), ink, DEAD_SHARE), "framed": framed}


def new_view(args: argparse.Namespace, field: Tuple[float, float], colours: Dict[str, str]) -> Dict[str, Any]:
    """Everything the picture remembers between frames; `colours` is each team's own."""
    now = time.monotonic()
    return {
        "field": field,
        "colours": colours,
        # What each team's name and numbers are written in under the field:
        # BLACK TEAM in its own colour on a dark panel is a name nobody reads.
        "inks": {team: train.themes.readable(colour, PANEL) for team, colour in colours.items()},
        "truecolour": os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"),
        # Pixels through the kitty protocol: asked for in "mode", decided once
        # the screen is taken, and counted in "frames" for the two image ids.
        "mode": args.graphics,
        "graphics": False,
        "frames": 0,
        "codes": {},
        "speed": args.speed,
        "interval": train.config.CONTROL_DT / args.speed,
        "deadline": now,
        "drawn_at": 0.0,
        "size": None,
        "fits": False,
        # The round the static part was drawn for; a new episode is a new map
        # and may be a change of ends.
        "episode": -1,
        "top": None,
        "bottom": None,
        "text": [None] * TEXT_LINES,
    }


def fit_view(view: Dict[str, Any], size: os.terminal_size) -> str:
    """Lay the picture out for a terminal of this size; returns what clears the screen for it.

    A character cell is about twice as tall as it is wide, so half a cell is a
    square pixel and one scale serves both axes. The scale is the largest at
    which the whole field fits above the text, and the field is centred across.
    Every cell is painted once here, in the panel colour, rather than left to
    an erase: not every terminal erases in the colour that is set.

    With graphics the picture is (wide, high) real pixels over (span, rows)
    cells, sized by the cell's own pixels, and the old placement goes first: it
    belongs to the old grid, and erasing text does not remove a picture.
    """
    width, height = view["field"]
    view["size"] = size
    view["fits"] = too_small(size, view["field"]) is None
    if not view["graphics"]:
        scale = min(size.columns / width, 2.0 * (size.lines - TEXT_LINES) / height)
        view["wide"] = int(width * scale)
        view["high"] = int(height * scale) // 2 * 2
        view["left"] = (size.columns - view["wide"]) // 2
        view["span"], view["rows"] = view["wide"], view["high"] // 2
    elif view["fits"]:
        cell_wide, cell_high = graphics.cell_pixels()
        per_metre = min(size.columns * cell_wide / width, (size.lines - TEXT_LINES) * cell_high / height)
        span, rows = int(width * per_metre / cell_wide), int(height * per_metre / cell_high)
        # Past WIDE_MAX the terminal scales the picture up: a frame costs its pixels.
        shrink = min(1.0, graphics.WIDE_MAX / (span * cell_wide))
        view["wide"], view["high"] = int(span * cell_wide * shrink), int(rows * cell_high * shrink)
        view["span"], view["rows"], view["left"] = span, rows, (size.columns - span) // 2
    view["episode"] = -1
    view["top"] = view["bottom"] = None
    view["text"] = [None] * TEXT_LINES
    blank = " " * size.columns
    out = [graphics.forget()] if view["graphics"] else []
    out += [text_code(view, PANEL, 48), text_code(view, TEXT, 38)]
    out.extend(f"\x1b[{line};1H{blank}" for line in range(1, size.lines + 1))
    if not view["fits"]:
        wrapped = textwrap.wrap(too_small(size, view["field"]), size.columns)[:size.lines]
        out.extend(f"\x1b[{line};1H{text}" for line, text in enumerate(wrapped, start=1))
    return "".join(out)


def refit(view: Dict[str, Any]) -> str:
    """fit_view if the terminal has changed size since the last frame, and nothing if not.

    Asked before every frame: a terminal is resized while it is watched.
    """
    size = shutil.get_terminal_size()
    return fit_view(view, size) if size != view["size"] else ""


def draw_ground(view: Dict[str, Any], layout: Dict[str, Any], looks: List[Dict[str, Any]],
                boxes: np.ndarray) -> np.ndarray:
    """The part of the picture that holds still for a round: floor, strips, obstacles.

    `boxes` is the spawn strip of each end as x_min, x_max, y_min, y_max, and
    `looks` the colours of the team standing on that end this round.

    A terminal of 256 colours gets it without the half-tones -- no smoothed
    edges, no wash on the floor or in the strips. The nearest of 256 to two
    colours that are close can be two that are far apart, and a wash drawn
    that way is a chequerboard louder than anything standing on it. A picture
    sent as pixels is true colour whatever the terminal announces, and is drawn
    finer only while it is small: at three times, a 960x640 ground takes
    seconds once a round, and the picture freezes.
    """
    true = view["truecolour"] or view["graphics"]
    if view["graphics"]:
        fine = 2 if view["wide"] <= 480 else 1
    else:
        fine = SUPERSAMPLE if true else 1
    wide, high = view["wide"] * fine, view["high"] * fine
    x, y = np.meshgrid((np.arange(wide, dtype=np.float32) + 0.5) * (layout["width"] / wide),
                       (np.arange(high, dtype=np.float32) + 0.5) * (layout["height"] / high))
    floor = rgb(layout["palette"]["floor"])
    image = np.empty((high, wide, 3), dtype=np.float32)
    image[:] = floor
    if true:
        tiles = (np.floor(x / TILE) + np.floor(y / TILE)) % 2 == 1
        image[tiles] = mix(floor, rgb(layout["palette"]["floor_alt"]), TILE_SHARE)
        for box, look in zip(boxes, looks):
            inside = (x >= box[0]) & (x <= box[1]) & (y >= box[2]) & (y <= box[3])
            image[inside] = mix(image[inside], look["ink"], STRIP_FILL)

    for item in layout["obstacles"]:
        if item["shape"] == train.kernels.SHAPE_CIRCLE:
            depth = item["hw"] - np.hypot(x - item["cx"], y - item["cy"])
        else:
            depth = np.minimum(item["hw"] - np.abs(x - item["cx"]), item["hh"] - np.abs(y - item["cy"]))
        image[depth >= 0.0] = rgb(item["edge"])
        image[depth >= OBSTACLE_EDGE] = rgb(item["color"])
    return image.reshape(view["high"], fine, view["wide"], fine, 3).mean(axis=(1, 3))


def strip_outline(view: Dict[str, Any], box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The screen pixels on the border of one spawn strip, as (rows, columns), as thick as the pen."""
    width, height = view["field"]
    left, right = int(box[0] * view["wide"] / width), int(box[1] * view["wide"] / width)
    top, bottom = int(box[2] * view["high"] / height), int(box[3] * view["high"] / height)
    thick = 2 * int(view["pen"][:, 0].max()) + 1
    border = np.zeros((view["high"], view["wide"]), dtype=bool)
    border[top:bottom + 1, left:right + 1] = True
    border[top + thick:bottom + 1 - thick, left + thick:right + 1 - thick] = False
    return np.nonzero(border)


def thicken(offsets: np.ndarray, pen: np.ndarray) -> np.ndarray:
    """Every offset of a shape drawn with every offset of the pen.

    What is one pixel wide by construction -- a nose, paint and its tail, the
    cross of the dead -- is a hairline in a picture of real pixels. The pen of
    half blocks is the single offset, which gives back the shape itself.
    """
    return (offsets[:, None, :] + pen[None, :, :]).reshape(-1, 2)


def start_round(view: Dict[str, Any], state: Any, match: Dict[str, Any]) -> None:
    """Draw what holds still for the round that has just begun, and size the players for it."""
    layout = state.pool.describe(int(state.arena_idx[0]))
    floor = layout["palette"]["floor"]
    view["looks"] = [team_look(view["colours"][match["teams"][index]], floor) for index in match["on_side"]]
    # The strip of each end: home_box holds every slot's own, and the first
    # slot of each end speaks for it.
    boxes = state.home_box[[0, state.team_size]]
    pen = disc(0.09 * view["wide"] / layout["width"]) if view["graphics"] else np.zeros((1, 2), np.int64)
    view["pen"] = pen
    ground = draw_ground(view, layout, view["looks"], boxes)
    view["outlines"] = [strip_outline(view, box) for box in boxes]
    for outline, look in zip(view["outlines"], view["looks"]):
        ground[outline] = mix(ground[outline], look["ink"], STRIP_LINE)
    view["ground"] = ground
    view["theme"] = layout["theme_name"]
    # A player's block is its real size where the picture has the pixels for
    # it, and never less than three across: smaller, and ten players are ten
    # specks.
    size = max(1, int(train.config.PLAYER_RADIUS * view["wide"] / layout["width"] + 0.25))
    view["reach"] = size
    view["shapes"] = {"body": disc(size + 0.5), "core": disc(size - 0.5),
                      "cross": thicken(cross(size), view["pen"])}
    view["episode"] = int(state.episode[0])


def draw_field(view: Dict[str, Any], state: Any) -> np.ndarray:
    """One frame of the field as (H, W, 3): the ground with the strips, players and paint on it."""
    image = view["ground"].copy()
    width, height = view["field"]
    across, down = view["wide"] / width, view["high"] / height
    size = state.team_size
    progress = train.sim.capture_progress(state)[0]
    for end in (0, 1):
        # The strip of this end is taken by the players of the other one, and
        # its outline goes over to their colour as far as they have got.
        attackers = slice(size, 2 * size) if end == 0 else slice(0, size)
        held = float(progress[attackers].max())
        if held > 0.0:
            outline = view["outlines"][end]
            image[outline] = mix(image[outline], view["looks"][1 - end]["ink"],
                                 STRIP_LINE + (1.0 - STRIP_LINE) * held)

    spots = (state.pos[0] * (across, down)).astype(np.int64)
    alive = state.alive[0]
    shapes = view["shapes"]
    # The dead first, so nobody still playing is hidden under a mark.
    for slot in np.nonzero(~alive)[0].tolist():
        stamp(image, shapes["cross"], spots[slot, 0], spots[slot, 1], view["looks"][slot // size]["gone"])
    for slot in np.nonzero(alive)[0].tolist():
        look = view["looks"][slot // size]
        x, y = int(spots[slot, 0]), int(spots[slot, 1])
        stamp(image, shapes["body"], x, y, look["ink"] if look["framed"] else look["body"])
        if look["framed"]:
            stamp(image, shapes["core"], x, y, look["body"])
        # The nose: the pixels just past the block on the side it is facing,
        # as many as the block is pixels from its centre to its edge.
        heading = float(state.heading[0, slot])
        for step in range(view["reach"] + 1, 2 * view["reach"] + 1):
            nose = np.array([[round(np.sin(heading) * step), round(np.cos(heading) * step)]])
            stamp(image, thicken(nose, view["pen"]), x, y, look["ink"])

    for end in (0, 1):
        # A paintball is one pixel of its team's ink with a faded tail behind
        # it: it crosses two and a half metres between two frames, and a lone
        # pixel that far from where it last was is not seen to be the same one.
        slots = slice(end * size, (end + 1) * size)
        flying = state.ball_active[0, slots]
        spots = state.ball_pos[0, slots][flying] * (across, down)
        along = state.ball_vel[0, slots][flying]
        along = along / np.maximum(np.hypot(along[:, 0], along[:, 1]), 1e-6)[:, None]
        look = view["looks"][end]
        for back in range(view["reach"] + 1, -1, -1):
            paint = (spots - along * back).astype(np.int64)
            stamp(image, thicken(paint[:, ::-1], view["pen"]), 0, 0, look["gone"] if back else look["ink"])
    return image


def cells_out(view: Dict[str, Any], image: np.ndarray) -> str:
    """The escapes that turn the screen's field into this frame, changed cells only.

    Two rows of pixels to a row of cells: the upper half block in the upper
    pixel's colour on a background of the lower one's. Only the cells that
    differ from the last frame are written, each run of them behind one cursor
    move, because between two frames a handful of players and paintballs move
    and the rest of some thousands of cells does not.
    """
    packed = pack(image)
    top, bottom = packed[0::2], packed[1::2]
    if view["top"] is None:
        changed = np.ones(top.shape, dtype=bool)
    else:
        changed = (top != view["top"]) | (bottom != view["bottom"])
    view["top"], view["bottom"] = top, bottom
    parts = []
    fore = back = None
    for row in np.nonzero(changed.any(axis=1))[0].tolist():
        upper, lower = top[row].tolist(), bottom[row].tolist()
        at = -2
        for column in np.nonzero(changed[row])[0].tolist():
            if column != at + 1:
                parts.append(f"\x1b[{row + 1};{view['left'] + column + 1}H")
            if upper[column] != fore:
                fore = upper[column]
                parts.append(colour_code(view, fore, 38))
            if lower[column] != back:
                back = lower[column]
                parts.append(colour_code(view, back, 48))
            parts.append(HALF_BLOCK)
            at = column
    return "".join(parts)


def team_points(points: np.ndarray, game: Dict[str, Any]) -> List[int]:
    """The two teams' points out of a per-model tally."""
    return [int(points[game["members"][team]].sum()) for team in game["teams"]]


def rounds_text(count: int) -> str:
    return f"{count} {'round' if count == 1 else 'rounds'}"


def clock_text(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def pair(first: Any, second: Any, game: Dict[str, Any]) -> List[Tuple[str, str]]:
    """`first v second`, each in its team's colour."""
    return [(str(first), game["teams"][0]), (" v ", QUIET), (str(second), game["teams"][1])]


def text_lines(view: Dict[str, Any], game: Dict[str, Any], trainer: Any, match: Dict[str, Any],
               note: List[Tuple[str, str]], live: bool) -> List[Optional[List[Tuple[str, str]]]]:
    """The lines under the field, each a list of (text, colour or team name) pieces.

    `live` is false while a finished round is held on the screen: the arena has
    been reset under the picture by then, so the round's own line -- its clock,
    who is standing -- is left as the last frame wrote it (None here).
    """
    config = train.config
    first, second = game["teams"]
    lines = [[(first, first), (f"  {match['score'][0]} : {match['score'][1]}  ", TEXT), (second, second),
              (f"    match {len(game['results']) + 1} of {game['wanted']}", QUIET)]]
    if live:
        state = trainer.state
        size = state.team_size
        standing = [int(state.alive[0, :size].sum()), int(state.alive[0, size:].sum())]
        alive = [standing[match["on_side"].index(index)] for index in (0, 1)]
        # A round that has just ended has already been counted into the match.
        number = match["rounds"] + (0 if game["ended"][0] else 1)
        lines.append([(f"round {number}   {clock_text(float(state.tick[0]) * config.DT)} of "
                       f"{clock_text(game['phase']['match_seconds'])}   {view['theme']}   alive ", QUIET)]
                     + pair(alive[0], alive[1], game))
    else:
        lines.append(None)
    now = team_points(trainer.tally["points"], game)
    lines.append([("points this match   ", QUIET)]
                 + pair(now[0] - game["base"][0], now[1] - game["base"][1], game))
    averages = average_points(game)
    if averages is None:
        lines.append([("average a round     ", QUIET)] + pair("-", "-", game))
    else:
        rounds = f"   over {rounds_text(game['rounds'])}"
        lines.append([("average a round     ", QUIET)]
                     + pair(f"{averages[0]:.2f}", f"{averages[1]:.2f}", game) + [(rounds, QUIET)])
    lines.append(note)
    return lines


def text_out(view: Dict[str, Any], lines: List[Optional[List[Tuple[str, str]]]]) -> str:
    """The escapes that rewrite the lines of text that changed since they were last written.

    Written on the panel colour and padded to the right edge with spaces, for
    the reason fit_view paints every cell. A piece's colour is a colour, or a
    team's name for that team's ink, in bold.
    """
    panel = text_code(view, PANEL, 48)
    room = view["size"].columns - view["left"]
    parts = []
    for index, pieces in enumerate(lines):
        if pieces is None:
            continue
        used = 0
        line = [f"\x1b[{view['rows'] + 1 + index};{view['left'] + 1}H", panel]
        for text, colour in pieces:
            text = text[:room - used]
            used += len(text)
            weight = "\x1b[1m" if colour in view["inks"] else "\x1b[22m"
            line.append(f"{weight}{text_code(view, view['inks'].get(colour, colour), 38)}{text}")
        line.append("\x1b[22m" + " " * (room - used))
        written = "".join(line)
        if written != view["text"][index]:
            view["text"][index] = written
            parts.append(written)
    return "".join(parts)


def write_frame(parts: List[str]) -> None:
    sys.stdout.write(FRAME_BEGIN + "".join(parts) + FRAME_END)
    sys.stdout.flush()


def capture_note(view: Dict[str, Any], trainer: Any, match: Dict[str, Any]) -> List[Tuple[str, str]]:
    """The last line while a round is played: who is taking a strip, or how to leave."""
    state = trainer.state
    size = state.team_size
    progress = train.sim.capture_progress(state)[0]
    held = [float(progress[:size].max()), float(progress[size:].max())]
    end = int(np.argmax(held))
    if held[end] > 0.0:
        team = match["teams"][match["on_side"][end]]
        return [(team, team), (f" is taking the strip: {held[end] * state.capture_needed:.1f} of "
                               f"{state.capture_needed:.1f} s", TEXT)]
    speed = "" if view["speed"] == 1.0 else f"   speed {view['speed']:g}x"
    return [(f"Ctrl-C stops{speed}", QUIET)]


def keep_pace(view: Dict[str, Any]) -> None:
    """Wait for the moment the next step is due.

    Counted from the last step's own moment and not from now, so the time a
    frame took to draw comes out of the wait and the game runs at its speed. A
    step that is already late is not chased: the clock restarts from here,
    rather than playing a burst to catch up.
    """
    view["deadline"] += view["interval"]
    now = time.monotonic()
    if view["deadline"] > now:
        time.sleep(view["deadline"] - now)
    else:
        view["deadline"] = now


def hold(view: Dict[str, Any], seconds: float) -> None:
    time.sleep(seconds / view["speed"])
    view["deadline"] = time.monotonic()


def show_step(view: Dict[str, Any], game: Dict[str, Any], trainer: Any) -> None:
    """Draw the arena as the step just taken left it, and wait out the rest of the step."""
    state = trainer.state
    match = game["arenas"][0]
    parts = [refit(view)]
    now = time.monotonic()
    # The last frame of a round is always drawn: it is the one held on the
    # screen while the result is read.
    due = game["ended"][0] or now - view["drawn_at"] >= 1.0 / MAX_FPS
    if view["fits"] and due:
        if int(state.episode[0]) != view["episode"]:
            start_round(view, state, match)
        image = draw_field(view, state)
        if view["graphics"]:
            pixels = np.clip(np.rint(image), 0, 255).astype(np.uint8)
            parts.append(graphics.frame(pixels, view["left"] + 1, view["span"], view["rows"], view["frames"]))
            view["frames"] += 1
        else:
            parts.append(cells_out(view, image))
        parts.append(text_out(view, text_lines(view, game, trainer, match,
                                               capture_note(view, trainer, match), live=True)))
        view["drawn_at"] = now
    if any(parts):
        write_frame(parts)
    keep_pace(view)


def show_result(view: Dict[str, Any], game: Dict[str, Any], trainer: Any, match: Dict[str, Any],
                note: List[Tuple[str, str]], seconds: float) -> None:
    """Hold the last frame of a round under its result.

    Only the text is rewritten. The arena behind the picture has already been
    reset for the next round, and the field on the screen is still the frame
    the round ended on, which is the one to look at while the result is read.
    """
    parts = [refit(view)]
    if view["fits"]:
        parts.append(text_out(view, text_lines(view, game, trainer, match, note, live=False)))
    write_frame(parts)
    hold(view, seconds)


def new_game(args: argparse.Namespace, teams: List[str], members: Dict[str, np.ndarray],
             arenas: int, phase: Dict[str, Any]) -> Dict[str, Any]:
    """The matches of this demo: what is being played, and what has been.

    "arenas" holds each arena's match in progress, in the shape sdk.train keeps
    its own in (train.new_match), because train's lineup draw is handed this
    dict and reads that key.
    """
    return {
        "teams": teams,
        "members": members,
        "phase": phase,
        "wanted": args.matches,
        "started": 0,
        "arenas": [None] * arenas,
        # Per arena: its round has ended and the trainer has not yet paid it.
        "ended": [False] * arenas,
        # How the last round of each arena ended: the end that won it (or -1)
        # and the code of the reason.
        "ending": [None] * arenas,
        "results": [],
        # Rounds the tally has been paid for, and the tally as it stood at the
        # last of them: every average is these two and nothing else.
        "rounds": 0,
        "paid": None,
        # The two teams' points when the match on the screen began.
        "base": [0, 0],
    }


def average_points(game: Dict[str, Any]) -> Optional[List[float]]:
    """Each team's points a round over every round finished so far, or None before the first.

    Out of the tally as it stood when the last round was paid, over the rounds
    paid: the hits of a round still being played are in the live tally and in
    no round yet, and would be an average over one round too few.
    """
    if not game["rounds"]:
        return None
    return [points / game["rounds"] for points in team_points(game["paid"]["points"], game)]


def score_round(game: Dict[str, Any], trainer: Any, row: int, side: int) -> None:
    """Count a round that has just ended into its arena's match.

    As sdk.train's record_endings counts one, without the files: the reason is
    read before the trainer resets the arena, and the winner is the one the
    trainer pays out on. The simulation names the winner by END of the field,
    and on_side says which team stood there this round.
    """
    match = game["arenas"][row]
    code = int(train.sim.end_reasons(trainer.state, np.array([row]))[0])
    match["rounds"] += 1
    match[train.REASON_KEYS[code]] += 1
    if side >= 0:
        match["score"][match["on_side"][side]] += 1
    else:
        match["draws"] += 1
    game["ended"][row] = True
    game["ending"][row] = (side, code)


def round_note(game: Dict[str, Any], match: Dict[str, Any], row: int) -> List[Tuple[str, str]]:
    side, code = game["ending"][row]
    if side < 0:
        return [(f"round {match['rounds']} drawn {REASON_WORDS[code]}", TEXT)]
    team = match["teams"][match["on_side"][side]]
    return [(f"round {match['rounds']} to ", TEXT), (team, team), (f" {REASON_WORDS[code]}", TEXT)]


def match_note(match: Dict[str, Any], finished: bool) -> List[Tuple[str, str]]:
    if not finished:
        return [(f"match called off at {match['score'][0]}:{match['score'][1]} "
                 f"after {rounds_text(match['rounds'])}", TEXT)]
    team = match["teams"][int(np.argmax(match["score"]))]
    return [(team, team), (f" wins the match {max(match['score'])}:{min(match['score'])}", TEXT)]


def settle_round(game: Dict[str, Any], view: Optional[Dict[str, Any]], trainer: Any, row: int) -> None:
    """Close a round the trainer has now paid, and the match with it if that decided it.

    Called from the lineup draw, which the trainer makes right after it has
    paid a finished round into its tally: the first moment the points of that
    round are all there. The match ends as sdk.train ends one, on ROUNDS_TO_WIN
    or called off at the same cap of rounds.
    """
    config = train.config
    match = game["arenas"][row]
    game["ended"][row] = False
    game["rounds"] += 1
    game["paid"] = {key: values.copy() for key, values in trainer.tally.items()}
    finished = max(match["score"]) >= config.ROUNDS_TO_WIN
    decided = finished or match["rounds"] >= train.MATCH_ROUND_CAP_FACTOR * config.ROUNDS_TO_WIN
    if view is not None:
        show_result(view, game, trainer, match, round_note(game, match, row), ROUND_PAUSE)
        if decided:
            show_result(view, game, trainer, match, match_note(match, finished), MATCH_PAUSE)
    if decided:
        record = dict(match, finished=finished)
        game["results"].append(record)
        game["arenas"][row] = None
        if view is None:
            logger.info(train.describe_match(record))


def make_lineups(game: Dict[str, Any], view: Optional[Dict[str, Any]], draw_pairs: Callable) -> Callable:
    """The trainer's draw_lineups: sdk.train's own draw, a counted number of matches.

    Who stands where is train.make_assign_slots, untouched: a random end each
    round and a shuffle along it. What is added is the count. An arena whose
    match is over gets a new one while there are matches left to start, and is
    otherwise taken off the field: with `plays` false for the whole arena no
    model is asked for a move there, so nothing is fired, nobody is hit and no
    point is scored in it. Its clock still runs out, and the trainer still
    counts those empty rounds as drawn ones for the models last seen there,
    which is why rounds are counted here, from the matches, and never read
    from the tally.
    """
    def draw_lineups(trainer: Any, rows: np.ndarray) -> None:
        playing = []
        for row in rows.tolist():
            if game["ended"][row]:
                settle_round(game, view, trainer, row)
            if game["arenas"][row] is None:
                if game["started"] >= game["wanted"]:
                    trainer.plays[row] = False
                    continue
                game["started"] += 1
                game["base"] = team_points(trainer.tally["points"], game)
            playing.append(row)
        draw_pairs(trainer, np.array(playing, dtype=np.int64))

    return draw_lineups


def make_step_hook(game: Dict[str, Any], view: Optional[Dict[str, Any]]) -> Callable:
    """The trainer's on_step: score the rounds that ended on this step, then draw it."""
    def on_step(trainer: Any, done: np.ndarray, winner: np.ndarray) -> None:
        for row in np.nonzero(done)[0].tolist():
            if game["arenas"][row] is not None:
                score_round(game, trainer, row, int(winner[row]))
        if view is not None and game["arenas"][0] is not None:
            show_step(view, game, trainer)

    return on_step


def silence_keys() -> Optional[List[Any]]:
    """Stop the terminal echoing keys while the picture is up; returns what to restore.

    Only changed cells are redrawn, so a letter echoed onto the field would
    stay there until a player walked over it. Ctrl-C is not a key the terminal
    echoes its way past: it still interrupts.
    """
    if termios is None or not sys.stdin.isatty():
        return None
    saved = termios.tcgetattr(sys.stdin.fileno())
    quiet = termios.tcgetattr(sys.stdin.fileno())
    quiet[3] &= ~termios.ECHO
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, quiet)
    return saved


def restore_keys(saved: Optional[List[Any]]) -> None:
    """Put the terminal back, and drop what was typed at the picture so the shell does not run it."""
    if saved is not None:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)


def hold_log() -> Tuple[logging.Handler, List[logging.Handler]]:
    """Keep log lines off the picture: they are held, and written when the screen is back.

    A line written across the field would stay there for the reason an echoed
    key would, and a warning lost is worse than a warning late.
    """
    root = logging.getLogger()
    shown = list(root.handlers)
    # Never flushed by level or by filling up in any run that can happen: the
    # target is the terminal the picture is on.
    held = logging.handlers.MemoryHandler(100_000, flushLevel=logging.CRITICAL + 1,
                                          target=shown[0] if shown else None)
    for handler in shown:
        root.removeHandler(handler)
    root.addHandler(held)
    return held, shown


def release_log(held: logging.Handler, shown: List[logging.Handler]) -> None:
    root = logging.getLogger()
    root.removeHandler(held)
    for handler in shown:
        root.addHandler(handler)
    held.close()


def play(trainer: Any, game: Dict[str, Any]) -> None:
    """Rollout after rollout until every match is over.

    The matches end inside a rollout and the trainer finishes the rollout it
    began; by then every arena is off the field, so what is left of it is a
    hundred-odd steps in which nobody is asked for a move.
    """
    while len(game["results"]) < game["wanted"]:
        train.trainer_module.collect_rollout(trainer, game["phase"])


def play_drawn(trainer: Any, game: Dict[str, Any], view: Dict[str, Any]) -> None:
    """play(), on the alternate screen, with the terminal put back however it ends."""
    for name in STOP_SIGNALS:
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), watch.stop_on_signal)
    held, shown = hold_log()
    keys = silence_keys()
    sys.stdout.write(ENTER_SCREEN + watch.HIDE_CURSOR)
    sys.stdout.flush()
    try:
        # Asked on the alternate screen and before the first full repaint,
        # which covers whatever a terminal prints for an escape it does not know.
        view["graphics"] = graphics.detect(view["mode"])
        play(trainer, game)
    finally:
        # FRAME_END as well: an interrupt can land inside a frame's write, and
        # a terminal left waiting for the end of a frame shows nothing more.
        # In kitty mode it can land inside an image's escape, which would take
        # everything after it as image data: ESC \ closes that one first.
        forget = graphics.forget() if view["graphics"] else ""
        close = "\x1b\\" if view["graphics"] else ""
        sys.stdout.write(close + FRAME_END + forget + watch.RESET + watch.SHOW_CURSOR + LEAVE_SCREEN)
        sys.stdout.flush()
        restore_keys(keys)
        release_log(held, shown)


def table(rows: List[List[str]]) -> List[str]:
    """Rows of cells as lines, the first column to the left and the rest to the right."""
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return ["  ".join(f"{cell:<{width}}" if index == 0 else f"{cell:>{width}}"
                      for index, (cell, width) in enumerate(zip(row, widths))).rstrip()
            for row in rows]


def summary_lines(game: Dict[str, Any], teams: Dict[str, Any]) -> List[str]:
    """The result as text: every match, then the teams and the players over all of them.

    All of it out of the tally as it stood when the last round was paid, the
    same figures the lines under the picture were written from.
    """
    first, second = game["teams"]
    lines = [train.describe_match(record) for record in game["results"]]
    open_matches = [match for match in game["arenas"] if match is not None]
    lines.extend(f"Stopped at {match['teams'][0]} {match['score'][0]}:{match['score'][1]} "
                 f"{match['teams'][1]} after {rounds_text(match['rounds'])}" for match in open_matches)
    rounds = game["rounds"]
    if not rounds:
        return lines + ["No round was finished"]
    played = game["results"] + open_matches
    endings = ", ".join(f"{key} {sum(record[key] for record in played)}"
                        for key in ("time", "elimination", "capture", "draws"))
    lines.append(f"{rounds_text(rounds)}: {endings}")
    # The rounds each team won, over every match: the figure a team is judged
    # by before the matches, which are few and decided by a round or two.
    taken = [sum(record["score"][index] for record in played) for index in (0, 1)]
    share = f" ({100.0 * taken[0] / sum(taken):.1f}% of the decided)" if sum(taken) else ""
    lines.append(f"Rounds: {first} {taken[0]}:{taken[1]} {second}{share}")
    if len(game["results"]) > 1:
        won = [sum(1 for record in game["results"]
                   if record["finished"] and record["score"][index] > record["score"][1 - index])
               for index in (0, 1)]
        lines.append(f"Matches: {first} {won[0]}:{won[1]} {second}")

    paid = game["paid"]
    matches = len(game["results"])
    rows = [["team", "points", "a round", "a match"] + [name for name, unused in PLAYER_COLUMNS]]
    for team in game["teams"]:
        members = game["members"][team]
        points = int(paid["points"][members].sum())
        a_match = f"{points / matches:.2f}" if matches else "-"
        rows.append([team, str(points), f"{points / rounds:.2f}", a_match]
                    + [str(int(paid[key][members].sum())) for unused, key in PLAYER_COLUMNS])
    lines.extend([""] + table(rows))

    rows = [["player", "team", "points", "a round"] + [name for name, unused in PLAYER_COLUMNS]]
    for team in game["teams"]:
        for nick, model in zip(teams[team]["members"], game["members"][team].tolist()):
            points = int(paid["points"][model])
            rows.append([nick, team, str(points), f"{points / rounds:.2f}"]
                        + [str(int(paid[key][model])) for unused, key in PLAYER_COLUMNS])
    lines.extend([""] + table(rows))
    return lines


def read_teams(args: argparse.Namespace, drawn: bool) -> Tuple[Dict[str, Any], List[str], List[str], str]:
    """Everything that can be refused before the engine is imported.

    Returns the bundle's teams, the nick order of its manifest, the two teams
    that play and the directory their weights are read from.
    """
    if args.matches < 1:
        train.refuse("--matches must be at least 1")
    if not args.speed > 0.0:
        train.refuse("--speed must be positive")
    bundle, weights = find_sources(args)
    teams_doc, bundle_nicks = train.read_bundle(bundle)
    teams = teams_doc["teams"]
    train.check_bundle(teams, bundle_nicks)
    playing = train.resolve_teams(args.teams, teams)
    if len(playing) != 2:
        train.refuse(f"{playing[0]} cannot play itself: a model stands in one place; name two teams")
    check_weights(weights, teams, playing)
    if drawn:
        for team in playing:
            if watch.parse_colour(teams[team].get("colour")) is None:
                train.refuse(f"{team} has no #rrggbb colour in {train.TEAMS_NAME} to be drawn in")
    return teams, bundle_nicks, playing, weights


def main():
    args = parse_args()
    drawn = sys.stdout.isatty() and not args.summary
    if os.name == "nt":
        watch.enable_windows_ansi()
    teams, bundle_nicks, playing, weights = read_teams(args, drawn)

    train.set_environment(THREADS)
    train.import_engine()
    config, curriculum, torch = train.config, train.curriculum, train.torch
    torch.set_num_threads(THREADS)
    # torch checks every distribution it builds and every sample it scores, and
    # with ten models asked for a move one small batch at a time that is a
    # fifth of the time a match takes. What it would catch is a network putting
    # out NaN, which sampling from it refuses just the same.
    torch.distributions.Distribution.set_default_validate_args(False)
    if args.theme is not None and args.theme not in train.themes.THEME_KEYS:
        train.refuse(f"--theme {args.theme!r}: the themes are {', '.join(train.themes.THEME_KEYS)}")
    train.check_nicks(teams, playing)
    # Every match on the full arena, whatever the teams' own steps say.
    clock = train.full_arena_clock()
    phase = curriculum.phase_for(clock)
    field = (phase["width"], phase["height"])
    cramped = too_small(shutil.get_terminal_size(), field) if drawn else None
    if cramped:
        train.refuse(cramped)

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    # The trainer's generator decides the maps, the ends and the spawns; torch's
    # decides every action a model samples. Both, or a seed replays nothing.
    torch.manual_seed(seed)
    unused_team_of_model, members = train.seat_teams(bundle_nicks, teams, playing)
    train.kernels.warmup(train.TEAM_SIZE)
    # One arena when the match is drawn, because one can be watched at a time.
    arenas = 1 if drawn else min(args.matches, SUMMARY_ARENAS)
    device = torch.device("cpu")
    trainer = train.trainer_module.Trainer(arenas, device, seed=seed, team_size=train.TEAM_SIZE)
    manifest = train.persist.restore_run(weights, trainer.nets, trainer.optimizers, trainer.normalizers,
                                         list(config.MODEL_NICKS), trainer.obs_size, device)
    if not manifest or manifest["fresh"]:
        train.refuse(f"The checkpoints in {weights} could not be loaded; see the lines above")

    game = new_game(args, playing, members, arenas, phase)
    view = new_view(args, field, {team: teams[team]["colour"] for team in playing}) if drawn else None
    trainer.theme = args.theme
    trainer.read_clock = train.full_arena_clock
    trainer.draw_lineups = make_lineups(game, view, train.make_assign_slots(
        game, members, [playing[0]], {playing[0]: [playing[1]]}))
    trainer.on_step = make_step_hook(game, view)
    logger.info(f"{playing[0]} v {playing[1]}, {args.matches} "
                f"{'match' if args.matches == 1 else 'matches'}, weights from {weights}, seed {seed}")
    trainer.phase_idx = curriculum.phase_index(clock)
    train.trainer_module.apply_phase(trainer, phase)
    try:
        if drawn:
            play_drawn(trainer, game, view)
        else:
            play(trainer, game)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as exc:
        logger.error(f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}")
        sys.exit(1)
    sys.stdout.write("\n".join(summary_lines(game, teams)) + "\n")


if __name__ == "__main__":
    main()
