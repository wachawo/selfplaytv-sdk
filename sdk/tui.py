#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One terminal program for the whole journey: train, watch, set up, play a match, export.

    python -m sdk.tui [RUN] [--graphics auto|kitty|blocks]

Run from the repository root; every key is on the bottom row. IT TRAINS
NOTHING AND DRAWS NO MATCH ITSELF: the trainer is `python -m sdk.train`,
started detached with its log in RUN/train.log, the match is `python -m
sdk.demo` on a pseudo-terminal, and a team and an export are `python -m
sdk.new_team` and `python -m sdk.export_team`. What is true about a run is on
disk (the trainer's lock, status.json, the checkpoints, RUN/tui.json) and is
read again every half second, so quitting, a crash, a dropped ssh and a TUI
opened tomorrow are one case, and leaving never touches the trainer.

Raw termios and ANSI escapes, not curses: sdk.watch's tables and sdk.demo's
frames are strings of 24-bit escapes, kitty graphics is an escape protocol,
and curses passes none of them through.
"""

import argparse
import json
import logging
import os
import re
import select
import shlex
import signal
import struct
import subprocess
import sys
import textwrap
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import fcntl
    import pty
    import termios
    import tty
except ImportError:
    # Windows: refused in main() with a sentence rather than here with a traceback.
    fcntl = pty = termios = tty = None

# Local imports
from engine import config
from sdk import demo, graphics, new_team, rewards, train, watch

logger = logging.getLogger(__name__)

SETTINGS_NAME = "tui.json"
LOG_NAME = "train.log"
REWARDS_NAME = "rewards.lua"
# Renamed to train.log.1 at a start past this size: a running trainer writes to
# the descriptor it inherited, so a start is the only moment there is.
LOG_ROTATE_BYTES = 8 * 1024 * 1024
LOG_TAIL_BYTES = 16 * 1024
DEFAULT_BUNDLE = "my-bundle"
TICK = 0.5
# The state row, the lineup row and the key row, at the BOTTOM: the state sits
# right above the key that changes it, and the demo, which addresses its rows
# from 1, draws above them without knowing they are there.
CHROME = 3
MIN_COLUMNS, MIN_LINES = 60, 12
PAN = 8
MATCH_GRACE = 2.0
FLASH_SECONDS = 5.0

STARTING, TRAINING, STOPPING, STOPPED, FAILED, NO_RUN = (
    "STARTING", "TRAINING", "STOPPING", "STOPPED", "FAILED", "NO RUN")
ALIVE = (STARTING, TRAINING, STOPPING)
RED = (235, 90, 90)
WORD_RGB = {TRAINING: (90, 200, 120), STARTING: watch.HIGHLIGHT_RGB, STOPPING: watch.HIGHLIGHT_RGB,
            FAILED: RED}
BOLD, INVERSE, ERASE_LINE, CLEAR = "\x1b[1m", "\x1b[7m", "\x1b[K", "\x1b[2J"
OUTSIDE = "started outside sdk.tui: its log is in the terminal that started it"
NO_PID = "this trainer left no pid in {}/lock: stop it where it was started"
CUT_SHORT = "Stopping already. Cut this update short? It still saves.  Y at once   other key: wait"

# One read can carry several keys (a held arrow) and a late answer to the
# graphics question (ESC _ G ... ESC \, CSI ? ... c): a sequence is split off
# whole, and one that is not in KEY_NAMES is dropped whole, so its letters
# never reach the hotkeys.
SEQUENCE = re.compile(r"\x1b(?:_[^\x1b]*\x1b\\|\[[0-9;?]*[ -/]*[@-~]|O.)")
KEY_NAMES = {"\x1b[A": "up", "\x1b[B": "down", "\x1b[C": "right", "\x1b[D": "left", "\x1bOA": "up",
             "\x1bOB": "down", "\x1bOC": "right", "\x1bOD": "left", "\x1b[5~": "pgup", "\x1b[6~": "pgdn",
             "\x1b": "esc", "\r": "enter", "\n": "enter", "\x7f": "backspace", "\x08": "backspace",
             "\t": "tab", "\x03": "ctrl-c"}
ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
LOG_HEAD = re.compile(r"\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d)\.\d{3} \[(\w+)\]: \([^)]*\) ")

# The settings, by sdk.train's argparse dests: the flag is "--" and the dest
# with "-" for "_", and the value in force is status.args[dest].
FIELDS = (("arenas", "Arenas side by side", "count"), ("threads", "CPU threads", "count"),
          ("device", "Device", "text"), ("hours", "Stop after hours", "hours"),
          ("steps", "Stop after steps", "steps"), ("seed", "Seed", "seed"), ("theme", "Map theme", "text"),
          ("checkpoint_every", "Save every N updates", "count"), ("gpu_memory", "GPU memory share", "share"))
EMPTY_WORDS = {"device": "auto", "hours": "never", "steps": "never", "theme": "all", "gpu_memory": "all"}
REFUSALS = {"count": "a whole number, 1 or more", "seed": "a whole number, 0 or more",
            "hours": "a number such as 2 or 0.5, or empty for never",
            "steps": "a whole number: 50M, 2.5M, 500k, or empty for never",
            "share": "a number such as 0.5, or empty for all",
            "bundle": "no teams.json and manifest.json there"}
NUMBER = re.compile(r"[0-9]+(?:\.[0-9]*)?|\.[0-9]+")
ROLES = ("learns", "opponent", "out")
VIEWS = {"d": "dash", "m": "match", "s": "settings", "w": "weights", "l": "log", "n": "new", "x": "export"}
# The list views, and the lines above the first row of each, so the cursor's line is known.
LIST_HEADS = {"settings": 3, "weights": 3, "new": 2, "export": 2}
FORMS = {"new": (("name", "Name"), ("colour", "Colour"), ("players", "Players"), ("start", "Starts as")),
         "export": (("team", "Team"), ("name", "New name"), ("colour", "New colour"),
                    ("players", "New players"), ("into", "Into"))}
REQUIRED = {"new": ("name", "colour", "players"), "export": ("team", "into")}


def short_path(path: str) -> str:
    """A path as typed from the repository root, or whole when it is outside it."""
    return path if os.path.relpath(path).startswith("..") else os.path.relpath(path)


def same_path(first: Any, second: Any) -> bool:
    return (isinstance(first, str) and isinstance(second, str)
            and os.path.realpath(first) == os.path.realpath(second))


def amber(text: str) -> str:
    return watch.paint(text, True, watch.HIGHLIGHT_RGB)


def saved_age(ui: Dict[str, Any]) -> Optional[float]:
    """Seconds since the run last saved, which is the age of its checkpoint manifest, or None.

    None too for a manifest older than RUN/sdk-run.json: that is the bundle's,
    copied in with its own time by a start that has not saved yet, and its age
    would claim a save to fall back on that the run does not have.
    """
    path = os.path.join(ui["run"], train.CHECKPOINTS_NAME, train.MANIFEST_NAME)
    marker = os.path.join(ui["run"], train.MARKER_NAME)
    if not os.path.isfile(path) or os.path.isfile(marker) and os.path.getmtime(path) < os.path.getmtime(marker):
        return None
    return max(0.0, time.time() - os.path.getmtime(path))


# The run on disk
def read_lock(run: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """RUN/lock as its text and its record; (None, {}) when no trainer ever took it."""
    try:
        with open(os.path.join(run, train.LOCK_NAME)) as handle:
            text = handle.read()
    except OSError:
        return None, {}
    try:
        record = json.loads(text)
    except ValueError:
        record = {}
    return text, record if isinstance(record, dict) else {}


def lock_held(ui: Dict[str, Any], text: Optional[str]) -> bool:
    """Whether a trainer holds RUN/lock: the flock is the truth about being alive, the record the address.

    Shared and read only, so two TUIs do not see each other and the file is
    never created or emptied. A trainer asking in the very instant of a probe
    would be refused, so a record seen free once is not probed again: a trainer
    writes its record AFTER it has the lock.
    """
    if text is None or text == ui["free"]:
        return False
    try:
        handle = open(os.path.join(ui["run"], train.LOCK_NAME))
    except OSError:
        return False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError:
        return True
    finally:
        handle.close()
    ui["free"] = text
    return False


def look(ui: Dict[str, Any]) -> Dict[str, Any]:
    """The state of the run, out of this TUI's child, the lock, status.json and tui.json."""
    run, child, saved = ui["run"], ui["child"], ui["saved"]
    text, record = read_lock(run)
    # Never probed while this TUI's own child lives: the child IS the trainer.
    alive = child is not None or lock_held(ui, text)
    holder = child.pid if child is not None else record.get("pid") if alive else None
    status = watch.load_status(run)
    current = alive and holder is not None and status is not None and status.get("pid") == holder
    sent, started = watch.get_dict(saved, "stop_sent"), watch.get_dict(saved, "started")
    state = {"alive": alive, "holder": holder, "record": record, "status": status, "current": current,
             "before": False}
    # Started by hand, or by another program, when the lock's pid is not the one tui.json started.
    # A lock record older than that start is about an earlier trainer: the start
    # was refused before it took the lock, and its refusal is in RUN/train.log.
    state["outside"] = (child is None and record.get("pid") not in (None, started.get("pid"))
                        and (watch.get_number(record, "started_at") or 0.0)
                        >= (watch.get_number(started, "at") or 0.0))
    # The lock taken by a trainer that never wrote a status: it died loading.
    state["early"] = status is not None and record.get("pid") not in (None, status.get("pid"))
    if alive:
        # A stop note older than the lock record is about an earlier process that may have had this pid.
        asked = (holder is not None and sent.get("pid") == holder
                 and (watch.get_number(sent, "at") or 0.0) >= (watch.get_number(record, "started_at") or 0.0))
        stopping = asked or (current and "stopped_at" in status)
        state["word"] = STOPPING if stopping else TRAINING if current or holder is None else STARTING
    elif ui["ended"] is not None:
        state["before"] = ui["ended"]["code"] == -signal.SIGTERM and sent.get("pid") == ui["ended"]["pid"]
        state["word"] = STOPPED if ui["ended"]["code"] == 0 or state["before"] else FAILED
    elif status is not None and "pid" in status and ("stopped_at" not in status or state["early"]):
        state["word"] = FAILED
    else:
        state["word"] = NO_RUN if status is None and saved_age(ui) is None else STOPPED
    return state


def log_tail(run: str) -> List[str]:
    """The last LOG_TAIL_BYTES of RUN/train.log as lines, the cut first one dropped."""
    try:
        with open(os.path.join(run, LOG_NAME), "rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - LOG_TAIL_BYTES))
            lines = handle.read().decode("utf-8", "replace").replace("\r", "").split("\n")
    except OSError:
        return []
    return [line for line in lines[1 if size > LOG_TAIL_BYTES else 0:] if line.strip()]


def short_log(line: str) -> str:
    """`YYYY-MM-DD HH:MM:SS.mmm [INFO]: (sdk.train) text` as `HH:MM:SS INFO text`."""
    head = LOG_HEAD.match(line)
    return f"{head.group(1)} {head.group(2)} {line[head.end():]}" if head else line


def bundle_teams(bundle: str) -> Dict[str, Any]:
    """A bundle's teams by name, in its order and as it spells them; none when it cannot be read."""
    try:
        teams = train.read_json(os.path.join(bundle, train.TEAMS_NAME)).get("teams")
    except (OSError, ValueError, AttributeError):
        return {}
    if not isinstance(teams, dict):
        return {}
    return {name: entry for name, entry in teams.items() if isinstance(entry, dict)}


def clean_settings(found: Dict[str, Any]) -> Dict[str, Any]:
    """`found` over the defaults, the team names its bundle does not have dropped.

    The trainer's defaults come off its own parser: one rule set, in sdk.train.
    With no roles saved, or only roles of teams the bundle does not have, its
    last team learns (sdk.new_team appends a new one) and every other team is
    an opponent. Roles that leave nobody learning are kept: they were stepped
    there, and the start refuses them in words.
    """
    parser = train.build_parser()
    settings = {key: found.get(key, parser.get_default(key)) for key, unused, unused_kind in FIELDS}
    settings["bundle"] = (found["bundle"] if isinstance(found.get("bundle"), str) and found["bundle"]
                          else DEFAULT_BUNDLE if new_team.is_bundle(DEFAULT_BUNDLE)
                          else os.path.relpath(new_team.SHIPPED_BUNDLE))
    teams = list(bundle_teams(settings["bundle"]))
    learn = [team for team in found.get("learn") or [] if team in teams]
    against, named = found.get("against"), (found.get("learn") or []) + (found.get("against") or [])
    if not learn and (found.get("learn") is None or named and not [team for team in named if team in teams]):
        learn, against = teams[-1:], teams[:-1]
    elif against is None:
        against = [team for team in teams if team not in learn] + (learn if len(learn) > 1 else [])
    return dict(settings, learn=learn, against=[team for team in against if team in teams])


def with_bundle(settings: Dict[str, Any], bundle: str) -> Dict[str, Any]:
    """The settings on another bundle: it keeps what roles it can, and with nobody of it learning, the default."""
    moved = clean_settings(dict(settings, bundle=bundle))
    return moved if moved["learn"] else clean_settings(dict(moved, learn=None, against=None))


def first_settings(run: str) -> Dict[str, Any]:
    """The settings of a run with no tui.json: what it was last started with, by hand included."""
    status = watch.load_status(run) or {}
    if watch.get_dict(status, "args"):
        return clean_settings(dict(status["args"], learn=watch.team_names(watch.get_list(status, "learners")),
                                   against=watch.team_names(watch.get_list(status, "opponents"))))
    try:
        marker = train.read_json(os.path.join(run, train.MARKER_NAME))
    except (OSError, ValueError):
        return clean_settings({})
    bundle = marker.get("bundle")
    return clean_settings({"bundle": short_path(bundle) if isinstance(bundle, str) else None,
                           "learn": marker.get("learners")})


def load_saved(ui: Dict[str, Any]) -> None:
    """Read RUN/tui.json again; a file that cannot be read leaves the last copy in use."""
    try:
        doc = train.read_json(os.path.join(ui["run"], SETTINGS_NAME))
    except (OSError, ValueError):
        return
    if isinstance(doc, dict) and isinstance(doc.get("settings"), dict):
        ui["saved"] = dict(doc, settings=clean_settings(doc["settings"]))


def save_saved(ui: Dict[str, Any], **changes: Any) -> None:
    """Write tui.json whole with `changes` (None drops a key); the first write makes the run directory."""
    os.makedirs(ui["run"], exist_ok=True)
    doc = {key: value for key, value in dict(ui["saved"], **changes).items() if value is not None}
    # Staged under this process's own name: two TUIs on one run may save in the
    # same instant, and with one name the second rename finds nothing to move.
    path = os.path.join(ui["run"], SETTINGS_NAME)
    staging = f"{path}.{os.getpid()}.tmp"
    with open(staging, "w") as handle:
        json.dump(doc, handle)
    os.replace(staging, path)
    ui["saved"] = doc


def change_settings(ui: Dict[str, Any], **changes: Any) -> None:
    save_saved(ui, settings=dict(ui["saved"]["settings"], **changes))


def role_of(learn: List[str], against: List[str], team: str) -> str:
    return "learns" if team in learn else "opponent" if team in against else "out"


def in_force(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """What the trainer holding the lock was started with, from its own status.json; never remembered here."""
    status = state["status"]
    if not state["current"] or not watch.get_dict(status, "args"):
        return None
    return {"args": status["args"], "learn": watch.team_names(watch.get_list(status, "learners")),
            "against": watch.team_names(watch.get_list(status, "opponents")),
            "weights": watch.get_dict(status, "rewards")}


def compact(steps: int) -> str:
    if steps >= 1_000_000:
        return f"{steps / 1e6:g}M"
    return f"{steps / 1e3:g}k" if steps >= 1000 and steps % 100 == 0 else f"{steps:,}"


def show_value(key: str, value: Any) -> str:
    if value is None:
        return EMPTY_WORDS.get(key, "-")
    if key == "steps" and isinstance(value, int):
        return f"{compact(value)} ({value:,})"
    return f"{value:g}" if isinstance(value, float) else str(value)


def default_weights() -> Dict[str, float]:
    return {knob: float(getattr(config, knob)) for knob in rewards.knob_names(config)}


def next_weights(ui: Dict[str, Any]) -> Tuple[Dict[str, float], Optional[str]]:
    """The weights of the next start from RUN/rewards.lua, and the parser's refusal of it if any.

    With no file yet, the weights of the run's status.json: a run started by
    hand with a weights file must not fall back to the defaults here.
    """
    defaults = default_weights()
    path = os.path.join(ui["run"], REWARDS_NAME)
    if os.path.isfile(path):
        try:
            return dict(defaults, **rewards.read_rewards(path, list(defaults))), None
        except (OSError, ValueError) as exc:
            return defaults, str(exc)
    found = watch.get_dict(ui["state"] and ui["state"]["status"], "rewards")
    return dict(defaults, **{knob: float(found[knob]) for knob in defaults
                             if watch.as_number(found.get(knob)) is not None}), None


def list_rows(ui: Dict[str, Any], view: str, force: Optional[Dict[str, Any]]) -> List[Tuple]:
    """(kind, key, label, next start, default, in force now or None, differs) per row of S or W."""
    if view == "weights":
        weights, rows = next_weights(ui)[0], []
        for knob, default in default_weights().items():
            theirs = None if force is None else watch.as_number(force["weights"].get(knob))
            rows.append(("weight", knob, rewards.key_of(knob), f"{weights[knob]:g}", f"{default:g}",
                         None if force is None else "?" if theirs is None else f"{theirs:g}",
                         force is not None and theirs != weights[knob]))
        return rows
    settings, args = ui["saved"]["settings"], force["args"] if force else {}
    rows = [("bundle", "bundle", "Bundle", settings["bundle"], "", force and str(args.get("bundle")),
             force is not None and not same_path(settings["bundle"], args.get("bundle")))]
    for team in bundle_teams(settings["bundle"]):
        mine = role_of(settings["learn"], settings["against"], team)
        theirs = force and role_of(force["learn"], force["against"], team)
        rows.append(("role", team, team, mine, "", theirs, force is not None and mine != theirs))
    for key, label, kind in FIELDS:
        theirs = force and show_value(key, args.get(key))
        rows.append((kind, key, label, show_value(key, settings.get(key)), "", theirs,
                     force is not None and settings.get(key) != args.get(key)))
    return rows


def pending(ui: Dict[str, Any]) -> Optional[int]:
    """How many settings and weights differ from what is in force, or None when that is not known."""
    force = in_force(ui["state"])
    return None if force is None else sum(row[6] for view in ("settings", "weights")
                                          for row in list_rows(ui, view, force))


def lineup_text(learn: List[str], against: List[str], threads: Any, device: Any, arenas: Any,
                hours: Any, steps: Any) -> str:
    budget = ([f"{hours:g} hours"] if isinstance(hours, (int, float)) else []) + (
        [f"{compact(steps)} steps"] if isinstance(steps, int) else [])
    return (f"{', '.join(learn) or '-'} v {', '.join(against) or '-'}  {threads} threads  {device or 'auto'}"
            f"  {arenas} arenas  {' or '.join(budget) or 'until stopped'}")


def lineup_problem(settings: Dict[str, Any]) -> Optional[str]:
    """What would make the command line meaningless; everything else sdk.train refuses itself."""
    if not settings["learn"]:
        return "Nobody learns: give a team the role learns in S"
    alone = [team for team in settings["learn"] if not set(settings["against"]) - {team}]
    return f"{alone[0]} has nobody to play: make another team an opponent in S" if alone else None


# The trainer
def train_flags(ui: Dict[str, Any]) -> List[str]:
    """The flags of a start. Never --restart (it deletes a run), --reward or --play-only."""
    settings = ui["saved"]["settings"]
    flags = ["--bundle", settings["bundle"], "--run", ui["name"], "--learn", *settings["learn"],
             "--against", *settings["against"]]
    for key in ("arenas", "threads", "seed", "checkpoint_every", "hours", "steps", "device", "theme",
                "gpu_memory"):
        flags += ["--" + key.replace("_", "-"), str(settings[key])] if settings[key] is not None else []
    return flags + ["--rewards", os.path.join(ui["name"], REWARDS_NAME)]


def start_trainer(ui: Dict[str, Any]) -> None:
    """Start sdk.train detached, appending to RUN/train.log.

    start_new_session is the whole of the safety: with no controlling terminal
    a closed terminal sends the trainer no hangup and nothing typed here reaches
    it. A file and not a pipe: a pipe to a TUI that has gone is a broken pipe.
    """
    os.makedirs(ui["run"], exist_ok=True)
    load_saved(ui)
    ui["previous"] = {key: ui["saved"].get(key) for key in ("started", "stop_sent")}
    weights, log_path = os.path.join(ui["run"], REWARDS_NAME), os.path.join(ui["run"], LOG_NAME)
    if not os.path.isfile(weights):
        rewards.write_rewards(weights, next_weights(ui)[0])
    if os.path.isfile(log_path) and os.path.getsize(log_path) > LOG_ROTATE_BYTES:
        os.replace(log_path, log_path + ".1")
    argv = [sys.executable, "-m", "sdk.train"] + train_flags(ui)
    with open(log_path, "a") as log:
        # Reads like the trainer's own lines, and teaches the flags.
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}.000 [INFO]: (sdk.tui) python -m sdk.train "
                  f"{shlex.join(argv[3:])}\n")
        log.flush()
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 start_new_session=True)
    save_saved(ui, started={"pid": child.pid, "at": time.time()}, stop_sent=None)
    ui.update(child=child, ended=None, asked_stop=False)


def stop_trainer(ui: Dict[str, Any]) -> bool:
    """SIGTERM to this TUI's live child, or to the lock holder probed in this same call; nobody else.

    So a recycled pid is never hit. SIGTERM and not SIGINT: before the trainer
    installs its handler it ends the process quietly, where SIGINT is a
    traceback or, inherited as ignored, lost.
    """
    child = ui["child"]
    if child is not None and child.poll() is None:
        pid = child.pid
    else:
        text, record = read_lock(ui["run"])
        pid = record.get("pid") if lock_held(ui, text) else None
    if not isinstance(pid, int) or isinstance(pid, bool):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    save_saved(ui, stop_sent={"pid": pid, "at": time.time()})
    ui["asked_stop"] = True
    return True


def advance(ui: Dict[str, Any]) -> None:
    """Look at the run again: reap, ring on an end nobody here asked for, start a pending restart."""
    ui["looked"] = time.monotonic()
    if ui["child"] is not None and ui["child"].poll() is not None:
        ui["ended"] = {"pid": ui["child"].pid, "code": ui["child"].returncode, "at": time.time()}
        ui["child"] = None
    load_saved(ui)
    state = look(ui)
    if (ui["ended"] is not None and state["alive"] and state["holder"] != ui["ended"]["pid"]
            and watch.get_dict(ui["saved"], "started").get("pid") == ui["ended"]["pid"]):
        # This start lost the lock to one made in the same instant: tui.json names that one again.
        save_saved(ui, **ui["previous"])
        state = look(ui)
    ui["ended"] = None if state["alive"] else ui["ended"]
    if ui["state"] and ui["state"]["word"] in ALIVE and state["word"] in (STOPPED, FAILED):
        ui["bell"] = ui["bell"] or not ui["asked_stop"]
        problem = lineup_problem(ui["saved"]["settings"])
        if ui["restart"] and state["word"] == STOPPED and problem:
            flash(ui, problem)
        elif ui["restart"] and state["word"] == STOPPED:
            start_trainer(ui)
            state = look(ui)
        ui["restart"] = False
    ui["state"] = state


# The match
def pty_size(ui: Dict[str, Any]) -> bytes:
    """The demo's window: three rows shorter, the pixel height scaled with it (kitty mode sizes by it)."""
    try:
        wide, high = struct.unpack("HHHH", fcntl.ioctl(1, termios.TIOCGWINSZ, bytes(8)))[2:]
    except OSError:
        wide = high = 0
    rows = max(1, ui["size"].lines - CHROME)
    return struct.pack("HHHH", rows, ui["size"].columns, wide, high * rows // max(1, ui["size"].lines))


def match_teams(ui: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """The first learner, and the opponent Tab has stepped to among every other team of the bundle."""
    learn = ui["saved"]["settings"]["learn"]
    rivals = [team for team in bundle_teams(ui["saved"]["settings"]["bundle"]) if team not in learn[:1]]
    return (learn[0], rivals[ui["rival"] % len(rivals)]) if learn and rivals else None


def start_match(ui: Dict[str, Any]) -> None:
    """The unchanged sdk.demo on a pseudo-terminal, in a session of its own so it can be ended alone."""
    teams = match_teams(ui)
    if teams is None:
        flash(ui, "A match needs a team that learns and one other team: see S")
        return
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, pty_size(ui))
    argv = [sys.executable, "-m", "sdk.demo", "--bundle", ui["saved"]["settings"]["bundle"],
            "--graphics", ui["graphics"], *teams]
    argv += ["--run", ui["name"]] if saved_age(ui) is not None else []
    process = subprocess.Popen(argv, stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
    os.close(slave)
    ui["match"] = {"process": process, "master": master, "held": b"", "text": b"", "result": b"",
                   "begun": False, "left": False, "over": False, "at": time.monotonic()}


def pump_match(ui: Dict[str, Any]) -> bytes:
    """What the demo wrote since the last read, in whole frames only, without its screen switches.

    The demo redraws changed cells and sets its colours once a frame, so rows
    written here after half a frame would leave cells in colours never
    repainted: what follows the last frame's end waits for the next read. The
    text before the demo's screen (loading, a refusal) and after it (the
    result) is kept for the pane.
    """
    match = ui["match"]
    try:
        data = os.read(match["master"], 65536)
    except OSError:
        data = b""
    # An error on Linux and an empty read on macOS both mean the demo has ended.
    match["over"] = not data
    data, match["held"] = match["held"] + data, b""
    if not match["begun"]:
        match["text"], found, data = (match["text"] + data).partition(demo.ENTER_SCREEN.encode())
        if not found:
            return b""
        match["begun"] = True
    if match["left"]:
        match["result"] += data
        return b""
    shown, found, after = data.partition(demo.LEAVE_SCREEN.encode())
    if found:
        match["left"], match["result"] = True, match["result"] + after
    else:
        # Held for the next read, or dropped once there is none: a frame cut in
        # a kitty upload would swallow everything written after it.
        end = shown.rfind(demo.FRAME_END.encode())
        cut = end + len(demo.FRAME_END) if end >= 0 else 0
        shown, match["held"] = shown[:cut], b"" if match["over"] else shown[cut:]
    return shown.replace(watch.SHOW_CURSOR.encode(), b"")


def stop_match(ui: Dict[str, Any]) -> None:
    """End a match: SIGINT, two seconds to leave, then SIGKILL (it writes nothing, so it may be); reap it."""
    match = ui["match"]
    if match is None or match["master"] is None:
        return
    if not match["over"]:
        os.killpg(match["process"].pid, signal.SIGINT)
        end = time.monotonic() + MATCH_GRACE
        while not match["over"] and time.monotonic() < end:
            if select.select([match["master"]], [], [], 0.1)[0]:
                pump_match(ui)
        if match["process"].poll() is None:
            os.killpg(match["process"].pid, signal.SIGKILL)
    match["over"] = True
    match["process"].wait()
    os.close(match["master"])
    match["master"] = None
    # Erasing the text under a picture does not remove the picture.
    ui["out"] += graphics.forget() if ui["graphics"] == "kitty" else ""


def playing(ui: Dict[str, Any]) -> bool:
    """Whether the demo owns the rows above the chrome."""
    return ui["match"] is not None and ui["match"]["begun"] and not ui["match"]["over"]


# Forms
def new_team_out(ui: Dict[str, Any]) -> str:
    """The bundle a new team goes into: the one in the settings, never the shipped one."""
    bundle = ui["saved"]["settings"]["bundle"]
    return DEFAULT_BUNDLE if same_path(bundle, new_team.SHIPPED_BUNDLE) else bundle


def from_run(ui: Dict[str, Any]) -> bool:
    return saved_age(ui) is not None and os.path.isfile(os.path.join(ui["run"], train.MARKER_NAME))


def form_rows(ui: Dict[str, Any], view: str) -> List[Tuple[str, str, str, Optional[List[str]]]]:
    """(key, label, shown, choices or None) for each row of a form; a choice row holds its pick."""
    form, bundle = ui["forms"][view], ui["saved"]["settings"]["bundle"]
    if view == "new":
        out = new_team_out(ui)
        choices = list(bundle_teams(out if new_team.is_bundle(out) else bundle)) + [""]
        form["start"] = form["start"] if form["start"] in choices else choices[0]
        shown = {"start": f"a copy of {form['start']}" if form["start"] else "fresh networks"}
    else:
        teams = bundle_teams(bundle)
        choices = list(teams)
        if form["team"] not in teams:
            form["team"] = (ui["saved"]["settings"]["learn"] + choices + [""])[0]
        team = teams.get(form["team"], {})
        keep = {"name": form["team"], "colour": team.get("colour"),
                "players": " ".join(team.get("members") or [])}
        shown = {key: form[key] or f"(keep {value})" for key, value in keep.items()}
    return [(key, label, shown.get(key, form[key]), choices if key in ("start", "team") else None)
            for key, label in FORMS[view]]


def run_form(ui: Dict[str, Any]) -> None:
    """C: the form's command, output captured; its result, or its last ERROR line, goes under the form.

    Names, colours and nicknames are checked by the command and nowhere else,
    and sdk.new_team stages and commits, so a refusal leaves nothing behind.
    """
    view, form, bundle = ui["view"], ui["forms"][ui["view"]], ui["saved"]["settings"]["bundle"]
    form_rows(ui, view)
    missing = [label for key, label in FORMS[view] if key in REQUIRED[view] and not form[key].strip()]
    if missing:
        ui["results"][view] = [amber(f"  {missing[0]} is empty")]
        return
    if view == "new":
        argv = ["-m", "sdk.new_team", "--name", form["name"], "--colour", form["colour"], "--nicks",
                *form["players"].split(), *(["--from", form["start"]] if form["start"] else ["--fresh"]),
                "--out", new_team_out(ui)]
    else:
        argv = ["-m", "sdk.export_team", "--team", form["team"], "--bundle", bundle, "--out", form["into"],
                *(["--run", ui["name"]] if from_run(ui) else []),
                *(["--name", form["name"]] if form["name"] else []),
                *(["--colour", form["colour"]] if form["colour"] else []),
                *(["--nicks", *form["players"].split()] if form["players"] else [])]
    ui["prompt"] = {"kind": "working", "text": "working: python " + shlex.join(argv)}
    draw(ui)
    done = subprocess.run([sys.executable] + argv, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    ui["prompt"] = None
    if done.returncode != 0:
        errors = [line.split(") ", 1)[-1] for line in done.stderr.splitlines() if "[ERROR]" in line]
        last = (errors or done.stderr.strip().splitlines() or ["?"])[-1]
        ui["results"][view] = textwrap.wrap("ERROR " + last, max(20, ui["size"].columns - 4),
                                            initial_indent="  ", subsequent_indent="  ")
    elif view == "export":
        path = done.stdout.strip().splitlines()[-1]
        ui["results"][view] = [f"  Written: {short_path(path)}, {os.path.getsize(path) / 1e6:.1f} MB"]
    else:
        # The team's name as new_team wrote it (it cleans what was typed): the bundle's last team.
        out = new_team_out(ui)
        teams = list(bundle_teams(out))
        run = os.path.join("runs", new_team.run_name(teams[-1]))
        ui["results"][view] = [f"  Added {teams[-1]} to {out}."]
        settings = clean_settings({"bundle": out, "learn": teams[-1:], "against": teams[:-1]})
        ask(ui, f"Open {run} to train {teams[-1]}?  Enter open   other key: stay here",
            {"enter": lambda: open_run(ui, run, settings)})


# Drawing
def fit(line: str, columns: int, skip: int = 0) -> str:
    """A row cut to the terminal, ending in `>` where something was cut off on the right."""
    if len(ESCAPE.sub("", line)) - skip > columns:
        return watch.clip_line(line, columns - 1, skip) + watch.RESET + ">"
    return watch.clip_line(line, columns, skip)


def state_rows(ui: Dict[str, Any]) -> Tuple[str, str]:
    """Rows N-2 and N-1: the state as a word with its facts; the lineup, or the last log line."""
    state, now, saved = ui["state"], time.time(), saved_age(ui)
    word, status, settings = state["word"], state["status"] or {}, ui["saved"]["settings"]
    saved_text = "" if saved is None else f"saved {watch.format_age(saved)} ago"
    total = watch.get_number(status, "run_steps")
    total = watch.get_number(status, "steps") if total is None else total
    session = watch.get_number(status, "session_steps")
    steps = "" if total is None else f"{watch.format_millions(total)} steps" + (
        f" (+{watch.format_millions(session)})" if session is not None and session != total else "")
    if word == NO_RUN:
        facts = ["nothing has trained here yet"]
    elif word == STARTING:
        since = watch.get_number(watch.get_dict(ui["saved"], "started") if ui["child"] else state["record"],
                                 "at" if ui["child"] else "started_at") or now
        facts = [f"loading the engine, {max(0.0, now - since):.0f} s", steps, saved_text]
    elif word == TRAINING:
        sps, up = watch.get_number(status, "sps"), watch.get_time(status, "started_at")
        # First of the facts, where a cut row still shows it; which stall it is
        # and since when are on the dashboard and in the log.
        stalled = amber("stalled") if watch.get_text(watch.get_dict(status, "health"), "stalled", "") else ""
        facts = [stalled, steps, "" if sps is None else f"{sps:,.0f}/s",
                 "" if up is None else f"up {watch.format_age(now - up)}", saved_text]
    elif word == STOPPING:
        sent = watch.get_dict(ui["saved"], "stop_sent")
        asked = watch.get_number(sent, "at") if sent.get("pid") == state["holder"] else None
        facts = [("" if asked is None else f"asked {watch.format_age(now - asked)} ago: ")
                 + "finishing this update, then saving", steps]
    elif word == STOPPED:
        reason = watch.get_text(status, "stop_reason", "")
        facts = ["stopped before it began" if state["before"] else "" if reason == "stopped" else reason,
                 steps, saved_text]
    elif ui["ended"] is not None:
        ended = ui["ended"]
        after = ("" if saved is None
                 else f", {watch.format_age(max(0.0, saved - now + ended['at']))} after its last save")
        facts = [f"status {ended['code']} at {time.strftime('%H:%M:%S', time.localtime(ended['at']))}{after}",
                 steps]
    else:
        heard = watch.get_time(status, "updated_at")
        facts = ["ended before its first status: L shows why" if state["early"]
                 else "ended without a clean stop, last heard "
                 + ("-" if heard is None else time.strftime("%H:%M:%S", time.localtime(heard)))]
    colour = "\x1b[38;2;{};{};{}m".format(*WORD_RGB[word]) if word in WORD_RGB else ""
    first = (f"{BOLD}{INVERSE}{colour} {word} {watch.RESET} {ui['name']}  "
             + "  ".join(fact for fact in facts if fact))
    if word == TRAINING:
        force = in_force(state) or {"args": {}, "learn": watch.team_names(watch.get_list(status, "learners")),
                                    "against": watch.team_names(watch.get_list(status, "opponents"))}
        args, count = force["args"], pending(ui)
        second = (amber(f"{count} changed: R restarts") + " | " if count else "") + "now " + lineup_text(
            force["learn"], force["against"], args.get("threads", "?"), watch.get_text(status, "device"),
            watch.get_text(status, "arenas"), args.get("hours"), args.get("steps"))
        second += "  (play only: nothing learns)" if status.get("play_only") is True else ""
    elif word in (NO_RUN, STOPPED):
        second = "next start " + lineup_text(*[settings[key] for key in (
            "learn", "against", "threads", "device", "arenas", "hours", "steps")])
    else:
        lines = log_tail(ui["run"])
        second = ("then starting again | " if word == STOPPING and ui["restart"] else "") + (
            OUTSIDE if state["outside"] else short_log(lines[-1]) if lines else "")
    return first, " " + second


def bottom_row(ui: Dict[str, Any]) -> str:
    """Row N: the text being typed, the question, a one-line answer, or the keys."""
    prompt, view = ui["prompt"], ui["view"]
    if prompt is not None and prompt["kind"] == "text":
        tail = amber(prompt["error"]) if prompt["error"] else "Enter accept   Esc leave as it was"
        # Its end, where the typing is, when the whole does not fit.
        room, text = ui["size"].columns - len(prompt["label"]) - 4, prompt["text"]
        text = "<" + text[-(room - 1):] if len(text) > room else text
        return f"{prompt['label']}: {text}_   {tail}"
    if prompt is not None:
        return prompt["text"]
    if ui["flash"] and time.monotonic() - ui["flash_at"] < FLASH_SECONDS:
        return amber(ui["flash"])
    keys = [("T", "stop" if ui["state"]["word"] in ALIVE else "start"), ("D", "dash"), ("M", "match"),
            ("S", "settings"), ("W", "weights"), ("L", "log"), ("N", "team"), ("X", "export"), ("Q", "quit")]
    lead = ([("C", "create"), ("Esc", "leave")] if view in FORMS else [] if view != "match"
            else ([("Esc", "end"), ("Enter", "again")] if playing(ui) else [("Enter", "play")])
            + [("Tab", "opponent")])
    keys = [pair for pair in keys if pair[0] in "TDMSLQ" and pair[1] != view] if lead else keys
    return "  |  ".join("  ".join(f"{BOLD}{key}{watch.RESET} {label}" for key, label in group)
                        for group in (lead, keys) if group)


def dash_lines(ui: Dict[str, Any]) -> List[str]:
    """sdk.watch's own tables, the standings first so the averages are above the fold at 80x24."""
    status, settings, name = ui["state"]["status"], ui["saved"]["settings"], ui["name"]
    if status is None and ui["state"]["word"] in ALIVE:
        return ["SELFPLAYTV SDK", "", f"  waiting for the trainer's first status.json in {name}"]
    if status is None:
        shipped = same_path(settings["bundle"], new_team.SHIPPED_BUNDLE)
        learn = ", ".join(settings["learn"]) or "-"
        rivals = ", ".join(team for team in settings["against"] if team not in settings["learn"]) or "-"
        return ["SELFPLAYTV SDK", "",
                f"  Nothing has trained in {name} yet"
                + (", and you have no team of your own." if shipped else "."),
                "", "  N  make a team: a name, a colour, five nicknames",
                f"  T  train {learn} against {rivals}" + (" as they ship" if shipped else ""),
                "  M  watch the two shipped teams play a match" if shipped
                else f"  M  watch {learn} play a match",
                "  S  settings    W  reward weights    Q  quit"]
    width, learners = watch.table_width(status), watch.team_names(watch.get_list(status, "learners"))
    changed = [] if status.get("play_only") is True else watch.changed_rewards(status)
    return (watch.render_standings(status, width, True)
            + (["rewards  " + amber("  ".join(changed))] if changed else [])
            + [line for line in [watch.paid_line(status, True), watch.health_line(status, True)] if line]
            + [f"phase    {watch.get_text(status, 'phase')}  {watch.learners_text(status, learners)}", ""]
            + watch.render_matches(status, width, True) + [""] + watch.render_players(status, True))


def view_rows(ui: Dict[str, Any]) -> List[Tuple]:
    """The rows of the list view open: a form's, or those of S or W."""
    view = ui["view"]
    return form_rows(ui, view) if view in FORMS else list_rows(ui, view, in_force(ui["state"]))


def list_lines(ui: Dict[str, Any]) -> List[str]:
    """The settings, the weights or a form, with the cursor's row marked by `>`."""
    view, alive, rows = ui["view"], ui["state"]["word"] in ALIVE, view_rows(ui)
    ui["cursor"] = min(ui["cursor"], len(rows) - 1)
    marks = ["  > " if index == ui["cursor"] else "    " for index in range(len(rows))]
    if view in FORMS:
        source = (f"the run {ui['name']}, saved {watch.format_age(saved_age(ui))} ago." if from_run(ui)
                  else f"the bundle {ui['saved']['settings']['bundle']}.")
        head = f"NEW TEAM in {new_team_out(ui)}" if view == "new" else (
            f"EXPORT a team as one archive. Weights: {source}")
        notes = (["  A copy plays as well as its source from the first step; fresh networks",
                  "  lose every match for about two hours of training."] if view == "new" else
                 ["  Training may go on: the export takes one whole save, or says to try again."]
                 if from_run(ui) else [])
        return ([head, ""] + [f"{mark}{row[1]:<14}{row[2]}" for mark, row in zip(marks, rows)] + [""] + notes
                + ["", "  up/down choose   Enter change   C create   Esc leave", ""] + ui["results"][view])
    if view == "settings":
        lines = ["SETTINGS for the next start. Training is running: R restarts with them." if alive
                 else "SETTINGS for the next start: T starts with them", "",
                 f"{'':27}{'next start':<25}" + ("in force now" if alive else "")]
        for mark, (kind, key, label, mine, default, theirs, differs) in zip(marks, rows):
            line = f"{mark}{label:<23}{mine:<25}"
            lines.append(amber(f"{line}{theirs:<10}* restart") if differs
                         else line + ("?" if theirs is None else "=") if alive else line.rstrip())
        return lines + ["", "  up/down choose   Enter change (a team: its role)   Backspace default"]
    path, error = os.path.join(ui["name"], REWARDS_NAME), next_weights(ui)[1]
    lines = ["REWARD WEIGHTS: what a player of the learning team is paid. Points do not move.", ""]
    if error:
        return lines + [watch.paint(f"  {path}: {error}", True, RED), "",
                        "  Backspace writes the defaults back"]
    lines.append(f"    {'weight':<16}{'next start':>10}{'default':>10}"
                 + (f"{'in force now':>15}" if alive else ""))
    for mark, (kind, key, label, mine, default, theirs, differs) in zip(marks, rows):
        line = f"{mark}{label:<16}{mine:>10}{default:>10}" + (f"{theirs or '?':>15}" if alive else "")
        lines.append(amber(line + "   * restart") if differs else line)
    return lines + ["", f"  kept in {path} and given to every start from here",
                    "  what each weight pays: examples/rewards.lua",
                    "  up/down choose   Enter type a value   Backspace default"]


def body_lines(ui: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """(fixed head, the rest) of the view; only the log has a head, which never scrolls or pans."""
    view, state, match = ui["view"], ui["state"], ui["match"]
    if view == "log":
        pid = state["holder"] or state["record"].get("pid") or "-"
        head = [OUTSIDE] * state["outside"] + [
            f"LOG {os.path.join(ui['name'], LOG_NAME)}   trainer pid {pid}   up/down scroll   left/right pan"]
        return head, [watch.paint(short_log(line), True, RED) if "[ERROR]" in line else short_log(line)
                      for line in log_tail(ui["run"])]
    if view != "match":
        return [], dash_lines(ui) if view == "dash" else list_lines(ui)
    teams, saved = match_teams(ui), saved_age(ui)
    if teams is None:
        return [], ["MATCH: it needs a team that learns and one other team; see S"]
    # "The current weights" said honestly: the last save, with its age, never "now".
    lines = [f"MATCH: {teams[0]} v {teams[1]}, " + (
        f"with the weights saved {watch.format_age(saved)} ago." if saved is not None
        else f"with the weights of the bundle {ui['saved']['settings']['bundle']}.")]
    if match is None:
        return [], lines + ["Enter play, Tab other opponent"]
    text = [short_log(ESCAPE.sub("", line)) for line in (match["text"] + match["result"]).decode(
        "utf-8", "replace").replace("\r", "").split("\n") if line.strip()]
    if not match["over"]:
        return [], lines + ["", f"loading the engine... {time.monotonic() - match['at']:.0f} s"] + text
    return [], lines + ["Enter play again, Tab other opponent", ""] + text


def draw(ui: Dict[str, Any], forwarded: bytes = b"") -> None:
    """One frame, every row at an absolute position and erased to its end, in one write.

    While a match plays only the three rows are written, after the demo's frames.
    """
    size = os.get_terminal_size()
    out = [demo.FRAME_BEGIN, ui["out"]] + [CLEAR] * (size != ui["size"])
    ui["out"], ui["size"], columns, lines = "", size, size.columns, size.lines
    room = lines - CHROME
    if out[-1] == CLEAR and playing(ui):
        fcntl.ioctl(ui["match"]["master"], termios.TIOCSWINSZ, pty_size(ui))
    if columns < MIN_COLUMNS or lines < MIN_LINES:
        notice = f"make the terminal larger ({MIN_COLUMNS}x{MIN_LINES} at least)"
        rows = [(notice, 0)] + [("", 0)] * (lines - 1)
        forwarded, first = b"", 1
    elif playing(ui):
        rows, first = [], room + 1
    else:
        head, body = body_lines(ui)
        most = max(0, len(body) - room + len(head))
        if ui["view"] in LIST_HEADS:
            # On the last row the lines under it come into view too, the help included.
            last = ui["cursor"] >= len(view_rows(ui)) - 1
            focus = len(body) - 1 if last else ui["cursor"] + LIST_HEADS[ui["view"]]
            ui["scroll"] = min(max(ui["scroll"], focus - room + 2), ui["cursor"], most)
        ui["scroll"] = min(ui["scroll"], most)
        top = most - ui["scroll"] if ui["view"] == "log" else ui["scroll"]
        shown = body[top:top + room - len(head)]
        if top + len(shown) < len(body):
            shown[-1] = amber(f"... {len(body) - top - len(shown) + 1} more rows below: down, PgDn")
        pan = ui["pan"] if ui["view"] in ("dash", "log") else 0
        rows = [(line, 0) for line in head] + [(line, pan) for line in shown]
        rows, first = rows + [("", 0)] * (room - len(rows)), 1
    if columns >= MIN_COLUMNS and lines >= MIN_LINES:
        rows += [(row, 0) for row in state_rows(ui)] + [(bottom_row(ui), 0)]
    for number, (row, skip) in enumerate(rows, start=first):
        out.append(f"\x1b[{number};1H{watch.RESET}{fit(row, columns, skip)}{watch.RESET}{ERASE_LINE}")
    data = forwarded + "".join(out + [demo.FRAME_END, "\a" * ui["bell"]]).encode()
    ui["bell"] = False
    while data:
        data = data[os.write(1, data):]


# Keys
def read_keys() -> List[str]:
    """The keys of one read, by name; a read ending in a bare ESC waits 50 ms for the rest of it."""
    data = os.read(0, 1024)
    if data.endswith(b"\x1b") and select.select([0], [], [], 0.05)[0]:
        data += os.read(0, 1024)
    text, keys = data.decode("utf-8", "ignore"), []
    while text:
        found = SEQUENCE.match(text)
        piece = found.group() if found else text[0]
        keys += [name for name in [KEY_NAMES.get(piece, None if found else piece)] if name]
        text = text[len(piece):]
    return keys


def flash(ui: Dict[str, Any], text: str) -> None:
    ui["flash"], ui["flash_at"] = text, time.monotonic()


def ask(ui: Dict[str, Any], text: str, keys: Dict[str, Callable[[], Any]]) -> None:
    """A question on the bottom row; what was typed ahead of it is dropped, so it cannot answer it."""
    termios.tcflush(0, termios.TCIFLUSH)
    ui["prompt"] = {"kind": "ask", "text": text, "keys": keys}


def answer(ui: Dict[str, Any], key: str) -> None:
    """A key while a question or a text is open. Only the text is modal: letters type there."""
    prompt = ui["prompt"]
    if prompt["kind"] == "text":
        if key in ("esc", "ctrl-c"):
            ui["prompt"] = None
        elif key == "enter":
            prompt["error"] = prompt["accept"](prompt["text"]) or ""
            ui["prompt"] = prompt if prompt["error"] else None
        elif key == "backspace":
            prompt["text"], prompt["error"] = prompt["text"][:-1], ""
        elif len(key) == 1 and key.isprintable():
            prompt["text"] += key
        return
    ui["prompt"] = None
    action = prompt["keys"].get(key.lower() if len(key) == 1 else key)
    if action is not None:
        action()


def parse_value(kind: str, text: str) -> Tuple[Any, Optional[str]]:
    """A typed setting as its value, or the words it is refused with. Only the kind is checked here."""
    text = text.strip()
    if kind == "bundle":
        return (text, None) if text and new_team.is_bundle(text) else (None, REFUSALS[kind])
    if kind == "text" or (not text and kind in ("hours", "steps", "share")):
        return text or None, None
    scale = {"m": 1_000_000, "k": 1000}.get(text[-1:].lower(), 1) if kind == "steps" else 1
    text = text[:-1] if scale > 1 else text
    number = float(text) * scale if NUMBER.fullmatch(text) else -1.0
    if kind in ("count", "seed", "steps"):
        number = int(number) if number == int(number) else -1
    if number < (0 if kind == "seed" else 1 if kind == "count" else 1e-300):
        return None, REFUSALS[kind]
    return number, None


def accept_row(ui: Dict[str, Any], kind: str, key: str, typed: str) -> Optional[str]:
    """A typed value for a row of S or W, written at once, or the words it is refused with."""
    if kind == "weight":
        # Checked by the format's own parser, and refused in its words.
        try:
            value = rewards.parse_table("return { " + rewards.key_of(key) + " = " + typed + " }")[0][1]
        except ValueError as exc:
            return re.sub(r"^line \d+: ", "", str(exc))
        os.makedirs(ui["run"], exist_ok=True)
        rewards.write_rewards(os.path.join(ui["run"], REWARDS_NAME),
                              dict(next_weights(ui)[0], **{key: value}))
        return None
    value, error = parse_value(kind, typed)
    if not error:
        change_settings(ui, **(with_bundle(ui["saved"]["settings"], value) if kind == "bundle"
                               else {key: value}))
    return error


def edit_row(ui: Dict[str, Any], back: bool = False) -> None:
    """Enter: a text opens with the value, a team steps its role, a choice steps on; `back`: the default."""
    # Clamped here too: keys merged into one read move the cursor before a draw clamps it.
    view, rows = ui["view"], view_rows(ui)
    cursor = ui["cursor"] = min(ui["cursor"], len(rows) - 1)
    if view in FORMS:
        key, label, unused, choices = rows[cursor]
        form = ui["forms"][view]
        if choices:
            form[key] = choices[(choices.index(form[key]) + 1) % len(choices)]
        else:
            ui["prompt"] = {"kind": "text", "label": label, "text": form[key], "error": "",
                            "accept": lambda typed: form.update({key: typed.strip()})}
        return
    weights, error = next_weights(ui)
    kind, key, label, mine = rows[cursor][:4]
    settings = ui["saved"]["settings"]
    if view == "weights" and error and not back:
        flash(ui, "Backspace writes the defaults back")
    elif view == "weights" and back:
        # For a file the parser refuses, every weight's default.
        os.makedirs(ui["run"], exist_ok=True)
        rewards.write_rewards(os.path.join(ui["run"], REWARDS_NAME),
                              dict(default_weights() if error else weights, **{key: default_weights()[key]}))
    elif kind == "role":
        teams = list(bundle_teams(settings["bundle"]))
        roles = {team: role_of(settings["learn"], settings["against"], team) for team in teams}
        roles[key] = (("learns" if key == teams[-1] else "opponent") if back
                      else ROLES[(ROLES.index(roles[key]) + 1) % len(ROLES)])
        learn = [team for team in teams if roles[team] == "learns"]
        change_settings(ui, learn=learn, against=[team for team in teams if roles[team] == "opponent"]
                        + (learn if len(learn) > 1 else []))
    elif back:
        defaults = clean_settings({})
        change_settings(ui, **(with_bundle(settings, defaults["bundle"]) if kind == "bundle"
                               else {key: defaults[key]}))
    else:
        value = settings.get(key, weights.get(key))
        typed = "" if value is None else compact(value) if key == "steps" else mine
        ui["prompt"] = {"kind": "text", "label": label, "text": typed, "error": "",
                        "accept": lambda text: accept_row(ui, kind, key, text)}


def confirm_stop(ui: Dict[str, Any], asked: str) -> None:
    """The answer to a stop, the run looked at again first: a trainer that has gone is never signalled."""
    advance(ui)
    word = ui["state"]["word"]
    if word not in ALIVE:
        flash(ui, "It has stopped already: nothing was sent")
    elif word == STOPPING and asked != STOPPING:
        # Somebody else's stop came first, so a second signal is never an accident.
        ask(ui, CUT_SHORT, {"y": lambda: confirm_stop(ui, STOPPING)})
    elif stop_trainer(ui):
        # Looked at again at once, so the row says STOPPING from this frame on.
        advance(ui)
    else:
        flash(ui, "No trainer holds the lock any more: nothing was sent")


def toggle_training(ui: Dict[str, Any]) -> None:
    """T. Start is confirmed with Enter and stop with Y, so no key pressed twice can start and then stop."""
    word, problem = ui["state"]["word"], lineup_problem(ui["saved"]["settings"])
    questions = {
        TRAINING: "Stop training? It finishes this update, then saves.  Y stop   other key: keep on",
        STARTING: "Stop it before it has begun? Nothing has been trained yet.  Y stop   other key: no",
        STOPPING: CUT_SHORT}

    def confirm_start() -> None:
        advance(ui)
        if ui["state"]["word"] in ALIVE:
            flash(ui, "A trainer has started meanwhile: nothing was started")
        else:
            start_trainer(ui)
            advance(ui)

    if word in ALIVE and ui["state"]["holder"] is None:
        flash(ui, NO_PID.format(ui["name"]))
    elif word in ALIVE:
        ask(ui, questions[word], {"y": lambda: confirm_stop(ui, word)})
    elif problem:
        flash(ui, problem)
    else:
        ask(ui, "Start training with the line above?  Enter start   other key: not yet",
            {"enter": confirm_start})


def restart_training(ui: Dict[str, Any]) -> None:
    """R: stop, wait for the save, start with the settings as they are now."""
    word, count, problem = ui["state"]["word"], pending(ui), lineup_problem(ui["saved"]["settings"])
    if word in ALIVE and ui["state"]["holder"] is None:
        flash(ui, NO_PID.format(ui["name"]))
        return
    if word != TRAINING or problem or count == 0:
        flash(ui, "Nothing is training: T starts it" if word not in ALIVE
              else f"It is {word.lower()}: R restarts a trainer that is training" if word != TRAINING
              else problem or "Nothing differs from what is in force: nothing to restart for")
        return

    def confirm() -> None:
        advance(ui)
        if ui["state"]["word"] != TRAINING:
            flash(ui, "It is not training any more: nothing was done")
        elif stop_trainer(ui):
            ui["restart"] = True

    question = (f"Restart with the {count} change{'s' * (count > 1)}?" if count
                else "Restart with the settings in S?")
    ask(ui, f"{question} It saves, stops and starts again.  Y restart   other key: no", {"y": confirm})


def handle_key(ui: Dict[str, Any], key: str) -> None:
    ui["flash"] = None
    if ui["prompt"] is not None:
        answer(ui, key)
        return
    view, letter, word = ui["view"], key.lower() if len(key) == 1 else key, ui["state"]["word"]
    if key in ("ctrl-c", "esc") and playing(ui):
        stop_match(ui)
    elif key == "ctrl-c" and word in ALIVE:
        ask(ui, "Ctrl-C:  T stop the training   Q leave, training goes on   other key: neither",
            {"t": lambda: confirm_stop(ui, word), "q": lambda: ui.update(quit=True)})
    elif letter in ("ctrl-c", "q"):
        ui["quit"] = True
    elif letter in ("t", "r"):
        (toggle_training if letter == "t" else restart_training)(ui)
    elif letter == "c" and view in FORMS:
        run_form(ui)
    elif (letter in VIEWS or key == "esc") and VIEWS.get(letter, "dash") != view:
        stop_match(ui)
        ui.update(view=VIEWS.get(letter, "dash"), cursor=0, scroll=0, pan=0)
    elif key in ("left", "right"):
        ui["pan"] = max(0, ui["pan"] + (PAN if key == "right" else -PAN))
    elif key in ("up", "down", "pgup", "pgdn"):
        page = max(1, ui["size"].lines - CHROME - 1)
        step = {"up": -1, "down": 1, "pgup": -page, "pgdn": page}[key]
        if view in LIST_HEADS:
            ui["cursor"] = max(0, min(ui["cursor"] + step, len(view_rows(ui)) - 1))
        else:
            ui["scroll"] = max(0, ui["scroll"] + (-step if view == "log" else step))
    elif key in ("enter", "tab") and view == "match":
        again = key == "enter" or (ui["match"] is not None and not ui["match"]["over"])
        stop_match(ui)
        ui["rival"] += key == "tab"
        if again:
            start_match(ui)
    elif key == "enter" and view in LIST_HEADS or key == "backspace" and view in ("settings", "weights"):
        edit_row(ui, back=key == "backspace")


# The program
def enter_screen(ui: Dict[str, Any]) -> None:
    """Raw mode, the alternate screen, no cursor, the log held back; then the graphics question, once.

    Raw, so a key is read as it is pressed and Ctrl-C is a key that asks. The
    question is asked here and not by the demo: a child must not query a
    terminal somebody else is reading.
    """
    ui["tty"] = termios.tcgetattr(0)
    # Importing sdk.train put a log handler on stderr, which would write across the screen.
    ui["log"] = demo.hold_log()
    os.write(1, (demo.ENTER_SCREEN + watch.HIDE_CURSOR).encode())
    tty.setraw(0)
    ui["graphics"] = "kitty" if graphics.detect(ui["mode"]) else "blocks"


def leave_screen(ui: Dict[str, Any]) -> None:
    """Everything back as it was; on a terminal that has gone the writes fail, and that is all."""
    try:
        stop_match(ui)
        os.write(1, (demo.FRAME_END + ui["out"] + graphics.forget() * (ui["graphics"] == "kitty")
                     + watch.RESET + watch.SHOW_CURSOR + demo.LEAVE_SCREEN).encode())
        termios.tcsetattr(0, termios.TCSADRAIN, ui["tty"])
    except (OSError, termios.error):
        pass
    demo.release_log(*ui["log"])


def open_run(ui: Dict[str, Any], run: str, settings: Optional[Dict[str, Any]] = None) -> None:
    """Make `run` the one shown. A trainer of the run that was open goes on untouched."""
    stop_match(ui)
    ui.update(run=os.path.realpath(run), name=short_path(run), child=None, ended=None, restart=False, previous={},
              asked_stop=False, free=None, state=None, match=None, view="dash", cursor=0, scroll=0, pan=0,
              saved={"version": 1, "settings": settings or first_settings(run)})
    if settings is None:
        load_saved(ui)
    elif os.path.isdir(run):
        save_saved(ui)
    advance(ui)


def loop(ui: Dict[str, Any]) -> None:
    """Look at the run every half second, forward the match's frames, draw, wait for keys."""
    forwarded = b""
    while not ui["quit"]:
        if time.monotonic() - ui["looked"] >= TICK:
            advance(ui)
        draw(ui, forwarded)
        forwarded, match = b"", ui["match"]
        sources = [0] + ([match["master"]] if match is not None and not match["over"] else [])
        ready = select.select(sources, [], [], max(0.0, ui["looked"] + TICK - time.monotonic()))[0]
        if sources[-1] in ready and len(sources) > 1:
            forwarded = pump_match(ui)
            if match["over"]:
                stop_match(ui)
        for key in read_keys() if 0 in ready else []:
            asked = ui["prompt"]
            handle_key(ui, key)
            # The keys after the one that asked came in the same read: type-ahead too.
            if ui["prompt"] is not None and ui["prompt"] is not asked and ui["prompt"]["kind"] == "ask":
                break


def farewell(ui: Dict[str, Any]) -> str:
    """The last line on the normal screen: where the training is, and the way back to it."""
    advance(ui)
    holder, word = ui["state"]["holder"], ui["state"]["word"]
    # The run this TUI was opened on, when N moved it to another: its trainer goes on too.
    text, record = read_lock(ui["first"])
    also = ("" if ui["first"] == ui["run"] or not lock_held({"run": ui["first"], "free": None}, text)
            else f"Training also goes on in {short_path(ui['first'])}"
            + (f" (pid {record['pid']})" if record.get("pid") else "") + ".\n")
    pid = "" if holder is None else f" (pid {holder})"
    if word not in ALIVE:
        return also
    if ui["restart"]:
        return also + (f"Training is stopping{pid} and will NOT start again. Start it with: "
                       f"python -m sdk.train {shlex.join(train_flags(ui))}\n")
    log = (OUTSIDE.split(": ")[1].capitalize() if ui["state"]["outside"] or holder is None
           else f"Log: {os.path.join(ui['name'], LOG_NAME)}")
    return also + (f"Training goes on in the background{pid}. {log}. "
                   f"Back to it: python -m sdk.tui {ui['name']}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m sdk.tui", description="Train, watch, set up, play and export from one screen")
    parser.add_argument("run", nargs="?", default=None,
                        help="the run directory (default: the latest under runs/, else runs/<your team>)")
    parser.add_argument("--graphics", choices=("auto", "kitty", "blocks"), default="auto",
                        help="the match as kitty images where the terminal answers (auto, the default), "
                             "always, or as half blocks")
    return parser.parse_args()


def pick_run() -> str:
    """The latest run under runs/ that a TUI or sdk.train has used, else runs/ and the default learner."""
    found = [os.path.join("runs", name) for name in (os.listdir("runs") if os.path.isdir("runs") else [])]
    used = [path for path in found if any(os.path.isfile(os.path.join(path, name))
                                          for name in (SETTINGS_NAME, train.MARKER_NAME))]
    if used:
        return max(used, key=os.path.getmtime)
    return os.path.join("runs", new_team.run_name((clean_settings({})["learn"] or ["my"])[0]))


def refuse(message: str) -> None:
    """Stop before the screen is taken, saying why, under this module's own name. Exit status 2."""
    logger.error(message)
    sys.exit(2)


def main():
    args = parse_args()
    if termios is None or fcntl is None:
        refuse("sdk.tui needs a POSIX terminal; on Windows run it under WSL")
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        refuse("sdk.tui needs a terminal; the commands it drives do not: python -m sdk.train --help")
    run = args.run or pick_run()
    if os.path.exists(run) and not os.path.isdir(run):
        refuse(f"{run} is not a directory")
    # A kill, a closed terminal or Ctrl-\ leave through the same restore as Q.
    for name in demo.STOP_SIGNALS:
        signal.signal(getattr(signal, name), watch.stop_on_signal)
    # Handled rather than inherited as ignored: a child gets SIG_DFL across exec
    # only for a signal its parent handles, and a match is ended with SIGINT.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    ui = {"mode": args.graphics, "graphics": "blocks", "size": None, "quit": False, "prompt": None,
          "flash": None, "flash_at": 0.0, "bell": False, "out": "", "rival": 0, "looked": 0.0, "match": None,
          "results": {"new": [], "export": []}, "forms": {
              "new": {"name": "", "colour": "", "players": "", "start": None},
              "export": {"team": None, "name": "", "colour": "", "players": "", "into": "exports"}}}
    open_run(ui, run)
    ui["first"] = ui["run"]
    enter_screen(ui)
    failure = ""
    try:
        loop(ui)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        failure = f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}"
    finally:
        leave_screen(ui)
    if failure:
        logger.error(failure)
        sys.exit(1)
    # Unbuffered: after a hangup a buffered line would fail again at exit and turn the status into 120.
    try:
        os.write(1, farewell(ui).encode())
    except OSError:
        pass


if __name__ == "__main__":
    main()
