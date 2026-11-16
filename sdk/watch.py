#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Watch an SDK training run in a terminal: the score by matches and the stats by player.

    python -m sdk.watch runs/my
    python -m sdk.watch runs/my --once > frame.txt

Everything it shows comes out of RUN/status.json, which sdk/train.py rewrites
atomically about once a second. There is no picture of the arena: what a run is
judged by is how its matches end and what each player does in them, and both are
tables.

Two kinds of average are shown and they are not the same thing. POINTS are the
scoreboard's: counted by fixed rules, the same for every team, and the way to
compare one team with another. The line that starts `paid` is the REWARD: what
the learners' reward weights paid them, per player per round, over the last few
updates. Change a weight and the reward moves at once while the points move
only if the team comes to play differently.

Nothing here imports the engine, torch or numpy: the viewer has to start
on a laptop that only rsyncs a run directory back from the GPU box, and on a box
where the trainer holds every core, so it is the standard library and ANSI
escapes and nothing else -- not even curses, which is missing from the Windows
Python many people have.

The status file is read as untrusted, possibly half-written data. Every key may be
absent or of the wrong type, because the trainer that wrote it may be an older or
a newer SDK than this viewer, and a viewer that dies on a schema change is one
nobody leaves running. A missing piece is drawn as a dash or left out; a missing
file is "waiting for the trainer".
"""

import argparse
import json
import math
import os
import re
import shutil
import signal
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Escapes. A full clear on every frame is affordable here: at one frame every
# couple of seconds there is no blink to speak of, and a clear leaves no tail of
# a longer previous frame behind, which homing alone would.
CLEAR_SCREEN = "\x1b[H\x1b[2J"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
RESET = "\x1b[0m"

# The anti-camping number is the one this view exists for, so it is the one that
# is coloured regardless of team: amber reads as "watch this" on both a dark and
# a light terminal, where the team colours already mean a team.
HIGHLIGHT_RGB = (250, 200, 60)

# A status older than this is called out in the header. The trainer writes about
# once a second while it steps; a learn() pass on a slow CPU can hold it for
# several seconds, so the threshold is well past that and short of "the run has
# died".
STALE_SECONDS = 15.0

# Matches listed under RECENT MATCHES. The trainer keeps twenty; ten is what fits
# above the standings in a half-height tmux pane.
RECENT_MATCHES = 10

# engine/config.py's defaults for the training rewards, for a status file that
# does not carry its own `reward_defaults` (one written before the trainer
# recorded them). The viewer cannot import the engine (see the module
# docstring), so they are copied here, and they only decide which weights the
# header calls out: a run that changed nothing shows no rewards line, and a run
# that set timeout to -2 shows exactly that. A copy that drifts costs one extra
# or one missing word in the header of an old file, nothing more.
DEFAULT_REWARDS = {
    "REWARD_HIT": 1.0,
    "REWARD_HIT_TAKEN": -1.0,
    "REWARD_WIN": 2.0,
    "REWARD_LOSS": -1.0,
    "REWARD_CAPTURE": 3.0,
    "REWARD_CAPTURE_COVER": 2.0,
    "REWARD_DEFENCE": 1.0,
    "REWARD_DRAW": -1.0,
    "REWARD_TIMEOUT": 0.0,
}

# What every reward knob's name starts with; a weights file spells REWARD_TIMEOUT
# as `timeout`, and the header shows it the way it is written there.
REWARD_PREFIX = "REWARD_"

# Players with no team in status.json still get a row, under this heading.
NO_TEAM = "(no team)"

COLOUR_PATTERN = re.compile(r"#[0-9a-fA-F]{6}")
# A team's name is written in the team's colour only between these two relative
# luminances. The background of the terminal is not known here: a nearly black
# colour on a dark one is a blank where the name should be, and a nearly white
# one is the same on a light one. Outside them the name is bold in the
# terminal's own colour, which is readable on whatever it is.
INK_LUMINANCE = (0.05, 0.8)

# An average of points a round is shown once the row has played this many
# rounds for every arena of the run. Points are counted as they are scored and
# rounds as they end, so the rounds still being played, one in every arena, are
# points over no round at all; with only a round or two finished in each the
# figure reads up to twice what it settles at.
SETTLED_ROUNDS_PER_ARENA = 5

# The width the frame is laid out for. The environment rides at the end of the
# header's second line only in so far as that line stays inside it, so a narrow
# terminal loses it before anything else and a wide one never wraps it.
FRAME_COLUMNS = 100

CUT_NOTE = "cut to fit this terminal; all of it: python -m sdk.watch RUN --once | less -S"


# Defensive accessors. The status file is someone else's output (see the module
# docstring), so nothing below indexes it directly.
def get_dict(source: Any, key: str) -> Dict[str, Any]:
    value = source.get(key) if isinstance(source, dict) else None
    return value if isinstance(value, dict) else {}


def get_list(source: Any, key: str) -> List[Any]:
    value = source.get(key) if isinstance(source, dict) else None
    return value if isinstance(value, list) else []


def as_number(value: Any) -> Optional[float]:
    # bool is an int in Python, and a stray true must not read as 1 step.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def get_number(source: Any, key: str) -> Optional[float]:
    return as_number(source.get(key) if isinstance(source, dict) else None)


def get_text(source: Any, key: str, default: str = "-") -> str:
    value = source.get(key) if isinstance(source, dict) else None
    if value is None or isinstance(value, (dict, list)):
        return default
    return str(value)


def get_time(source: Any, key: str) -> Optional[float]:
    """A timestamp as epoch seconds, whether it was written as a number or ISO text."""
    value = source.get(key) if isinstance(source, dict) else None
    number = as_number(value)
    if number is not None:
        return number
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


def team_names(values: List[Any]) -> List[str]:
    return [str(value) for value in values if isinstance(value, str)]


def load_status(run_dir: str) -> Optional[Dict[str, Any]]:
    """RUN/status.json as a dict, or None while there is nothing readable yet.

    The trainer replaces the file with os.replace, so a torn read should not
    happen on POSIX; it still can on a network mount or when the run directory is
    being rsynced in, and that is one more "not yet" rather than a crash.
    """
    path = os.path.join(run_dir, "status.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            status = json.load(handle)
    except (OSError, ValueError):
        return None
    return status if isinstance(status, dict) else None


# Colour
def parse_colour(value: Any) -> Optional[Tuple[int, int, int]]:
    # The pattern and not int(x, 16) alone, which also reads "+1" and " 3".
    if not isinstance(value, str) or not COLOUR_PATTERN.fullmatch(value):
        return None
    return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)


def luminance(colour: Tuple[int, int, int]) -> float:
    """The relative luminance of an sRGB colour, 0 for black to 1 for white."""
    linear = [value / 3294.6 if value <= 10 else ((value / 255.0 + 0.055) / 1.055) ** 2.4
              for value in colour]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def paint(text: str, colour: bool, fg: Optional[Tuple[int, int, int]] = None,
          bold: bool = False, dim: bool = False) -> str:
    """Wrap `text` in SGR escapes, or return it untouched when colour is off.

    Callers pad BEFORE painting: the escapes take no columns on screen, and a
    width applied to the painted string would count them and misalign the row.
    """
    if not colour or (fg is None and not bold and not dim):
        return text
    codes = []
    if bold:
        codes.append("1")
    if dim:
        codes.append("2")
    if fg is not None:
        codes.append(f"38;2;{fg[0]};{fg[1]};{fg[2]}")
    return f"\x1b[{';'.join(codes)}m{text}{RESET}"


def team_colour(status: Dict[str, Any], team: str) -> Optional[Tuple[int, int, int]]:
    """The colour a team's name is written in, or None for the terminal's own; see INK_LUMINANCE."""
    colour = parse_colour(get_dict(get_dict(status, "teams"), team).get("colour"))
    if colour is None or not INK_LUMINANCE[0] <= luminance(colour) <= INK_LUMINANCE[1]:
        return None
    return colour


def paint_team(status: Dict[str, Any], team: str, width: int, colour: bool, align: str = "<") -> str:
    return paint(f"{team:{align}{width}}", colour, team_colour(status, team), bold=True)


# Formatting
def format_count(value: Optional[float]) -> str:
    return "-" if value is None else f"{int(value):,}"


def format_share(part: Optional[float], whole: Optional[float]) -> str:
    if part is None or not whole:
        return "-"
    return f"{100.0 * part / whole:.1f}%"


def format_ratio(top: Optional[float], bottom: Optional[float]) -> str:
    if top is None or bottom is None:
        return "-"
    if bottom == 0:
        # No deaths yet: K/D is conventionally the hits themselves, not infinity.
        return "-" if top == 0 else f"{top:.2f}"
    return f"{top / bottom:.2f}"


def format_plain(value: Optional[float]) -> str:
    """A count without thousands separators, for the W-L-U and W/D/L cells.

    Three counts share one cell there, and the separators alone would push a
    long run's row out of its column.
    """
    return "-" if value is None else str(int(value))


def format_average(top: Optional[float], bottom: Optional[float], digits: int = 2) -> str:
    if top is None or not bottom:
        return "-"
    return f"{top / bottom:,.{digits}f}"


def format_millions(value: float) -> str:
    return f"{value / 1e6:,.2f}M"


def format_points(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}" if value == int(value) else f"{value:,.1f}"


def format_age(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def name_width(status: Dict[str, Any], names: List[str]) -> int:
    """One width for every team-name column, so the tables line up with each other."""
    known = list(get_dict(status, "teams")) + names
    return max([len(name) for name in known] + [4])


# Header
def changed_rewards(status: Dict[str, Any]) -> List[str]:
    """`timeout -2 (examples/rewards.lua)` for every weight that is not its default."""
    rewards = get_dict(status, "rewards")
    defaults = get_dict(status, "reward_defaults") or DEFAULT_REWARDS
    sources = get_dict(status, "reward_sources")
    out = []
    for name in sorted(rewards):
        value = as_number(rewards[name])
        if value is None or value == as_number(defaults.get(name, 0.0)):
            continue
        key = name[len(REWARD_PREFIX):].lower() if name.startswith(REWARD_PREFIX) else name
        source = sources.get(name)
        out.append(f"{key} {value:g}" + (f" ({source})" if isinstance(source, str) else ""))
    return out


def steps_text(status: Dict[str, Any]) -> str:
    """The steps this run has trained, and how many of them since this start when it was resumed.

    `steps` alone is the counter the checkpoints carry, which a bundle that was
    trained before it was handed over brings with it; a status that says what
    the run itself has taken is shown by that.
    """
    total = get_number(status, "run_steps")
    if total is None:
        total = get_number(status, "steps")
    text = format_count(total)
    session = get_number(status, "session_steps")
    if total is not None and session is not None and session != total:
        text += f" (+{format_count(session)} this start)"
    return text


def learners_text(status: Dict[str, Any], learners: List[str]) -> str:
    """The learner teams, each with its own networks' steps when the status has them.

    Beside the run's steps on purpose: a copy of a veteran team has trained for
    thirty thousand steps in this run and for hundreds of millions in all, and
    it is the second figure the curriculum and the learning rate follow.
    """
    team_steps = get_dict(status, "team_steps")
    parts = []
    for team in learners:
        own = get_number(team_steps, team)
        parts.append(team if own is None else f"{team} ({format_millions(own)} steps of its own)")
    return ", ".join(parts) or "-"


def paid_line(status: Dict[str, Any], colour: bool) -> Optional[str]:
    """`paid  MY TEAM +1.84 per player per round  (updates 6-10)`, or None when nothing is recorded."""
    window = get_dict(status, "reward_window")
    parts = []
    for team, sums in get_dict(window, "teams").items():
        reward, rounds = get_number(sums, "reward"), get_number(sums, "player_rounds")
        if reward is None or not rounds:
            continue
        parts.append(f"{paint(team, colour, team_colour(status, team), bold=True)} {reward / rounds:+.2f}")
    if not parts:
        return None
    first, last = get_number(window, "from_update"), get_number(window, "to_update")
    span = "" if first is None or last is None else f"  (updates {int(first)}-{int(last)})"
    return f"paid     {', '.join(parts)} per player per round{span}"


def environment_parts(status: Dict[str, Any]) -> List[str]:
    """`torch 2.11.0`, `numba 0.66.0 tbb`, `py 3.12.3`, `sdk 2c71fbc`: libraries first, as far as known."""
    found = get_dict(status, "environment")
    parts = []
    for key, label in (("torch", "torch"), ("numba", "numba"), ("python", "py"), ("commit", "sdk")):
        value = get_text(found, key, "")
        if key == "numba" and value:
            value = f"{value} {get_text(found, 'threading_layer', '')}".rstrip()
        if value:
            parts.append(f"{label} {value}")
    return parts


def health_line(status: Dict[str, Any], colour: bool) -> Optional[str]:
    """`health   stalled: no hits since update 40`, or None while learning moves."""
    health = get_dict(status, "health")
    stalled = get_text(health, "stalled", "")
    if not stalled:
        return None
    since = get_number(health, "since_update")
    text = f"stalled: {stalled}" + ("" if since is None else f" since update {int(since)}")
    return "health   " + paint(text, colour, HIGHLIGHT_RGB, bold=True)


def render_header(status: Dict[str, Any], run_dir: str, colour: bool, now: float) -> List[str]:
    run_name = os.path.basename(os.path.normpath(run_dir)) or run_dir
    title = paint("SELFPLAYTV SDK", colour, bold=True)
    sps = get_number(status, "sps")
    sps_text = "-" if sps is None else f"{sps:,.0f}"
    first = (f"{title}  run {run_name}  steps {steps_text(status)}"
             f"  updates {format_count(get_number(status, 'updates'))}  sps {sps_text}")
    updated = get_time(status, "updated_at")
    started = get_time(status, "started_at")
    stale = updated is not None and now - updated > STALE_SECONDS
    # A stopped run's uptime is the time it ran, not the time since it started.
    uptime = "-"
    if started is not None:
        end = updated if stale else now
        uptime = format_age(max(0.0, end - started))
    second = (f"phase {get_text(status, 'phase')}  device {get_text(status, 'device')}"
              f"  arenas {format_count(get_number(status, 'arenas'))}  up {uptime}")
    for part in environment_parts(status):
        if len(second) + 2 + len(part) > FRAME_COLUMNS:
            break
        second += "  " + part
    if stale:
        # Frozen numbers look exactly like a run that has stopped improving, so a
        # stale file says so in words.
        second += "  " + paint(f"status is {format_age(now - updated)} old", colour, HIGHLIGHT_RGB, bold=True)
    learners = team_names(get_list(status, "learners"))
    opponents = team_names(get_list(status, "opponents"))
    # A --play-only run fields the same teams and trains nobody: a header that
    # called it learning would have its flat tables read as a stalled run, and
    # the weights it was started with pay no one, so they are not listed.
    measuring = status.get("play_only") is True
    third = (f"{'playing' if measuring else 'learning'} {learners_text(status, learners)}"
             f"  against {', '.join(opponents) or '-'}{'  (play only: nothing learns)' if measuring else ''}")
    lines = [first, second, third]
    rewards = [] if measuring else changed_rewards(status)
    if rewards:
        lines.append("rewards  " + paint("  ".join(rewards), colour, HIGHLIGHT_RGB))
    paid = paid_line(status, colour)
    if paid:
        lines.append(paid)
    health = None if measuring else health_line(status, colour)
    if health:
        lines.append(health)
    return lines


# Recent matches
def render_match(status: Dict[str, Any], match: Dict[str, Any], width: int, colour: bool) -> str:
    teams = team_names(get_list(match, "teams"))
    score = get_list(match, "score")
    if len(teams) < 2 or len(score) < 2:
        return "  -"
    left, right = as_number(score[0]), as_number(score[1])
    left_text = "-" if left is None else str(int(left))
    right_text = "-" if right is None else str(int(right))
    line = (f"  {paint_team(status, teams[0], width, colour, '>')}  {left_text:>2}:{right_text:<2}"
            f"  {paint_team(status, teams[1], width, colour)}"
            f"  {format_count(get_number(match, 'rounds')):>3} rounds"
            f"  time {format_count(get_number(match, 'time')):>2}"
            f"  elim {format_count(get_number(match, 'elimination')):>2}"
            f"  cap {format_count(get_number(match, 'capture')):>2}")
    draws = get_number(match, "draws")
    if draws:
        line += f"  draws {int(draws)}"
    if match.get("finished") is False:
        # The round cap reached with nobody on seven: the camping stalemate itself.
        line += "  " + paint("unfinished", colour, HIGHLIGHT_RGB, bold=True)
    return line


def render_matches(status: Dict[str, Any], width: int, colour: bool) -> List[str]:
    matches = get_dict(status, "matches")
    played = format_count(get_number(matches, "played"))
    lines = [paint(f"RECENT MATCHES  ({played} played)", colour, bold=True)]
    recent = [match for match in get_list(matches, "recent") if isinstance(match, dict)]
    if not recent:
        return lines + ["  (no match finished yet)"]
    for match in recent[:RECENT_MATCHES]:
        lines.append(render_match(status, match, width, colour))
    return lines


# Standings
def standing_order(status: Dict[str, Any], standings: Dict[str, Any]) -> List[str]:
    """Learners first, in the order they were named; the rest by matches won."""
    learners = [team for team in team_names(get_list(status, "learners")) if team in standings]
    others = [team for team in standings if team not in learners]
    others.sort(key=lambda team: (-(get_number(standings[team], "won") or 0), team))
    return learners + others


def settled_rounds(status: Dict[str, Any]) -> float:
    """The rounds a row needs before its points a round are shown; see SETTLED_ROUNDS_PER_ARENA."""
    return SETTLED_ROUNDS_PER_ARENA * (get_number(status, "arenas") or 0.0)


def team_points_per_round(status: Dict[str, Any], team: str) -> Optional[float]:
    """The scoreboard points a team's players make between them in one round, or None.

    Out of the PLAYERS rows and not the standings: points are counted per
    player, and a standings row only counts a round once its match is over,
    while a player's rounds are counted as each round ends. Every member plays
    every round of its team, so the rounds of the team are the mean of its
    members' and not their sum. None until the team has played settled_rounds.
    """
    rows = [row for row in get_dict(status, "players").values()
            if isinstance(row, dict) and row.get("team") == team]
    points = [get_number(row, "points") for row in rows]
    rounds = [get_number(row, "rounds") for row in rows]
    if not rows or None in points or None in rounds or not sum(rounds):
        return None
    if sum(rounds) / len(rows) < settled_rounds(status):
        return None
    return sum(points) * len(rows) / sum(rounds)


def render_standings(status: Dict[str, Any], width: int, colour: bool) -> List[str]:
    standings = get_dict(status, "standings")
    lines = [paint("STANDINGS", colour, bold=True)]
    if not standings:
        return lines + ["  (no rounds played yet)"]
    time_head = paint(f"{'time%':>6}", colour, HIGHLIGHT_RGB, bold=True)
    lines.append(f"  {'team':<{width}}  {'matches':>7}  {'W-L-U':>11}  {'rounds':>8}"
                 f"  {time_head}  {'elim%':>6}  {'cap%':>6}  {'draw%':>6}  {'pts/rnd':>7}  {'pts/match':>9}")
    for team in standing_order(status, standings):
        row = get_dict(standings, team)
        rounds, matches = get_number(row, "rounds"), get_number(row, "matches")
        record = "-".join(format_plain(get_number(row, key)) for key in ("won", "lost", "unfinished"))
        time_share = format_share(get_number(row, "time"), rounds)
        time_cell = paint(f"{time_share:>6}", colour, HIGHLIGHT_RGB, bold=True)
        shares = [format_share(get_number(row, key), rounds) for key in ("elimination", "capture", "draws")]
        # A match's points are the round's average times the rounds a match of
        # this team runs to: both halves are averages over whole finished
        # things, where points over matches would count the rounds of the
        # matches still being played against no match at all.
        per_round = team_points_per_round(status, team)
        per_match = None if per_round is None or rounds is None else per_round * rounds
        lines.append(f"  {paint_team(status, team, width, colour)}"
                     f"  {format_count(matches):>7}  {record:>11}"
                     f"  {format_count(rounds):>8}  {time_cell}"
                     f"  {shares[0]:>6}  {shares[1]:>6}  {shares[2]:>6}"
                     f"  {format_average(per_round, 1):>7}  {format_average(per_match, matches, 1):>9}")
    return lines


# Players
def player_groups(status: Dict[str, Any], players: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    """(team, nicks) in display order: learners first, each team in its roster order."""
    teams = get_dict(status, "teams")
    learners = team_names(get_list(status, "learners"))
    order = learners + [team for team in teams if team not in learners]
    by_team: Dict[str, List[str]] = {}
    for nick, row in players.items():
        by_team.setdefault(get_text(row, "team", NO_TEAM), []).append(nick)
    order += sorted(team for team in by_team if team not in order)
    groups = []
    for team in order:
        nicks = by_team.get(team)
        if not nicks:
            continue
        members = team_names(get_list(get_dict(teams, team), "members"))
        rank = {nick: index for index, nick in enumerate(members)}
        nicks.sort(key=lambda nick: (rank.get(nick, len(rank)), nick))
        groups.append((team, nicks))
    return groups


def render_player(nick: str, row: Dict[str, Any], width: int, settled: float) -> str:
    hits, deaths, shots = (get_number(row, key) for key in ("hits", "deaths", "shots"))
    rounds, points = get_number(row, "rounds"), get_number(row, "points")
    record = "/".join(format_plain(get_number(row, key)) for key in ("wins", "draws", "losses"))
    average = format_average(points, rounds) if rounds is not None and rounds >= settled else "-"
    return (f"    {nick:<{width}}  {format_count(rounds):>7}  {record:>15}"
            f"  {format_count(hits):>7}  {format_count(deaths):>7}  {format_ratio(hits, deaths):>5}"
            f"  {format_share(hits, shots):>6}  {format_count(get_number(row, 'captures')):>5}"
            f"  {format_points(points):>8}  {average:>7}")


def render_players(status: Dict[str, Any], colour: bool) -> List[str]:
    players = {nick: row for nick, row in get_dict(status, "players").items() if isinstance(row, dict)}
    lines = [paint("PLAYERS", colour, bold=True)]
    if not players:
        return lines + ["  (no player stats yet)"]
    width = max([len(nick) for nick in players] + [4])
    head = (f"    {'nick':<{width}}  {'rounds':>7}  {'W/D/L':>15}  {'hits':>7}  {'deaths':>7}"
            f"  {'K/D':>5}  {'acc':>6}  {'caps':>5}  {'points':>8}  {'pts/rnd':>7}")
    lines.append(paint(head, colour, dim=True))
    for team, nicks in player_groups(status, players):
        lines.append("  " + paint(team, colour, team_colour(status, team), bold=True))
        for nick in nicks:
            lines.append(render_player(nick, players[nick], width, settled_rounds(status)))
    return lines


def table_width(status: Dict[str, Any]) -> int:
    """The one width of every team-name column, over every team the tables below will name."""
    recent = [match for match in get_list(get_dict(status, "matches"), "recent") if isinstance(match, dict)]
    named = list(get_dict(status, "standings"))
    for match in recent[:RECENT_MATCHES]:
        named += team_names(get_list(match, "teams"))
    return name_width(status, named)


def render_frame(status: Optional[Dict[str, Any]], run_dir: str, colour: bool, now: float) -> List[str]:
    if status is None:
        run_name = os.path.basename(os.path.normpath(run_dir)) or run_dir
        # A mistyped path and a run that has not written yet are two different
        # things to wait for, and only one of them ever ends.
        waiting = ("waiting for the trainer (no readable status.json yet)" if os.path.isdir(run_dir)
                   else f"no such directory: {run_dir}")
        return [paint("SELFPLAYTV SDK", colour, bold=True) + f"  run {run_name}", waiting]
    width = table_width(status)
    return (render_header(status, run_dir, colour, now) + [""]
            + render_matches(status, width, colour) + [""]
            + render_standings(status, width, colour) + [""]
            + render_players(status, colour))


# Terminal loop
def clip_line(line: str, columns: int, skip: int = 0) -> str:
    """`line` cut to `columns` visible characters, its escapes kept and not counted.

    The tables are wider than a narrow pane, and a line that wraps pushes the
    rest of the frame down and the header off the top of the screen. `skip`
    drops that many visible characters first, for a view panned to the right;
    the escapes in the dropped part are kept, so the colour in force carries on.
    """
    out = []
    visible = 0
    index = 0
    while index < len(line):
        if line[index] == "\x1b":
            end = index + 1
            while end < len(line) and not line[end].isalpha():
                end += 1
            out.append(line[index:end + 1])
            index = end + 1
            continue
        if skip:
            skip -= 1
            index += 1
            continue
        if visible >= columns:
            out.append(RESET)
            break
        out.append(line[index])
        visible += 1
        index += 1
    return "".join(out)


def show_once(run_dir: str, colour: bool) -> None:
    sys.stdout.write("\n".join(render_frame(load_status(run_dir), run_dir, colour, time.time())) + "\n")


def stop_on_signal(signum: int, frame: Any) -> None:
    # SIGTERM (a closing tmux pane, `kill`) goes through the same cleanup as
    # Ctrl-C; otherwise the terminal is left with no cursor.
    raise KeyboardInterrupt


def watch(run_dir: str, refresh: float) -> None:
    signal.signal(signal.SIGTERM, stop_on_signal)
    sys.stdout.write(HIDE_CURSOR)
    try:
        while True:
            # Read on every frame, because a pane is resized while it is watched.
            # A frame taller than the terminal scrolls the header away, so the
            # bottom of the players table is what gets cut, never the top.
            size = shutil.get_terminal_size()
            frame = render_frame(load_status(run_dir), run_dir, True, time.time())
            lines = [clip_line(line, size.columns) for line in frame[:size.lines]]
            # Said on the last line that is shown: a table cut on the right or
            # at the bottom looks whole, and the columns that went are the
            # averages.
            if lines and (len(frame) > len(lines) or lines != frame[:len(lines)]):
                lines[-1] = clip_line(paint(CUT_NOTE, True, HIGHLIGHT_RGB, bold=True), size.columns)
            # No newline after the last line: a frame exactly the terminal's
            # height would otherwise scroll it by one.
            sys.stdout.write(CLEAR_SCREEN + "\n".join(lines))
            sys.stdout.flush()
            time.sleep(refresh)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(RESET + SHOW_CURSOR + "\n")
        sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m sdk.watch",
                                     description="Watch an SDK training run in the terminal.")
    parser.add_argument("run", help="the run directory given to sdk.train --run")
    parser.add_argument("--refresh", type=float, default=2.0,
                        help="seconds between redraws (default 2)")
    parser.add_argument("--once", action="store_true",
                        help="print one frame with no clearing and exit")
    return parser.parse_args()


def enable_windows_ansi() -> None:
    """Switch on escape processing in the classic Windows console.

    cmd.exe and a PowerShell 5 window print ANSI escapes as text unless the
    console is put in virtual-terminal mode, and every frame would then be
    garbage; Windows Terminal already has it on, and there this changes nothing.
    """
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def main():
    args = parse_args()
    if os.name == "nt":
        enable_windows_ansi()
    # Piped into a file or a CI log, escapes are noise and a redraw loop never
    # ends, so a stdout that is not a terminal gets one plain frame.
    tty = sys.stdout.isatty()
    if args.once or not tty:
        show_once(args.run, tty)
        return
    # Below a tenth of a second the loop only re-reads a file written once a second.
    watch(args.run, max(0.1, args.refresh))


if __name__ == "__main__":
    main()
