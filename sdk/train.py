#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train chosen teams against frozen checkpoints of the others.

    python -m sdk.train --bundle BUNDLE --run RUN --learn "MY TEAM" [--against ...]

Run from the repository root. It drives the rollout, the PPO update and the
checkpoint code of trainer/selfplaytv.py, and supplies the three things that
module asks its caller for:

- the lineups. Every arena is one learner TEAM against one other team, on a
  random end, for a whole match, because a team is what is being trained.
- the curriculum clock, which runs on the learners' own steps only, so frozen
  veterans do not set the pace for a team that is starting out.
- who learns: the models of the --learn teams. Everybody else plays with the
  weights it was given and never takes a gradient step.

A bundle is a directory holding teams.json, manifest.json and one <nick>.pt per
model, and it is only ever read. Everything the run writes lives in RUN:
checkpoints/ (a copy of the bundle's on the first start, the run's own progress
after that), status.json for `python -m sdk.watch RUN` (the match score, the
standings and every player's totals), matches.jsonl, one line per finished
match, and progress.jsonl, one line per checkpoint.

What the learners are paid for is a file of weights, --rewards FILE (see
sdk/rewards.py and examples/rewards.lua), with --reward NAME=VALUE on top of it
for one knob at a time. The weights belong to a start and not to the run: a
resume without them trains on the defaults.
"""

import argparse
import collections
import json
import logging
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Local imports
from sdk import rewards as reward_files

LOGGING = {
    "handlers": [logging.StreamHandler()],
    "format":   "%(asctime)s.%(msecs)03d [%(levelname)s]: (%(name)s) %(message)s",
    "level":    logging.INFO,
    "datefmt":  "%Y-%m-%d %H:%M:%S",
}
logging.basicConfig(**LOGGING)
logger = logging.getLogger(__name__)

TEAMS_NAME = "teams.json"
MANIFEST_NAME = "manifest.json"
STATUS_NAME = "status.json"
PROGRESS_NAME = "progress.jsonl"
MATCHES_NAME = "matches.jsonl"
CHECKPOINTS_NAME = "checkpoints"
# Written into RUN before anything is copied there, and required before a run's
# checkpoints are resumed into, saved over or deleted: a mistyped path is
# otherwise the only thing between --restart and a directory of checkpoints
# that some other training wrote.
MARKER_NAME = "sdk-run.json"
MARKER_VERSION = 1
# Held for the life of the process, so a second sdk.train on the same RUN is
# refused rather than left to interleave its saves with the first one's.
LOCK_NAME = "lock"

# The SDK plays 5v5 and nothing else: a bundle is teams of five, and the
# observation layout every checkpoint was trained on is sized for five.
TEAM_SIZE = 5
TEAMS_VERSION = 1
STATUS_VERSION = 1

# The end-reason codes of engine.env.sim.end_reasons, by name, in code order.
# Restated rather than imported because this module is loaded before the engine is.
REASON_KEYS = ("capture", "elimination", "time")
# status.json's per-player names and the trainer.tally keys they are read from.
# The tally calls a round a "match", from before the word meant first-to-seven;
# the file says what it counts.
PLAYER_KEYS = (("rounds", "matches"), ("wins", "wins"), ("losses", "losses"), ("draws", "draws"),
               ("hits", "hits"), ("deaths", "deaths"), ("shots", "shots"),
               ("captures", "captures"), ("points", "points"))
STANDING_KEYS = ("matches", "won", "lost", "unfinished", "rounds", "time", "elimination",
                 "capture", "draws")
RECENT_MATCHES = 20
# A match with no winner after this many times ROUNDS_TO_WIN rounds is called
# off as unfinished: drawn rounds score for nobody, so two teams that both wait
# out the clock could otherwise hold an arena -- and one pairing -- forever.
MATCH_ROUND_CAP_FACTOR = 4

# Learning is called stalled when one of the conditions below holds for this
# many updates in a row. One update is noise, and a veteran's figures wander
# from one to the next; twenty is a few hundred thousand steps at 16 arenas and
# a million and a half at 64, long enough that a stall is not a bad rollout and
# short enough to be said within minutes of a stall beginning.
STALL_UPDATES = 20
# The policy has stopped moving: an approx_kl that the log line prints as
# 0.0000, and not one sample clipped. A veteran trains at the floor learning
# rate (LEARNING_RATE_FLOOR, 6e-5) and still reads about 0.004 and 0.05, forty
# times above this, so only a rate annealed to zero or updates that never run
# come down here.
STALL_KL = 1e-4
# The entropy the log line prints is the heads' entropies weighted by
# ENTROPY_MOVE, ENTROPY_TURN and ENTROPY_SHOOT, 0.044 for a uniform
# policy. The shipped veterans, with the bonus annealed to its final fifth,
# read 0.006 to 0.02; under 0.001 every head is all but certain of one action.
STALL_ENTROPY = 1e-3
# Each condition's name in status.json and the log, and what to try.
STALL_ADVICE = {
    "policy frozen": "approx_kl under {kl:g} and no clipped sample: unset LEARNING_RATE_FLOOR and "
                     "ANNEAL_STEPS if they are set, and look above for 'update skipped' lines",
    "entropy collapsed": "the entropy is under {entropy:g}, so every player repeats one action: lower "
                         "the weights you raised, or start again from the bundle with --restart",
    "no hits": "the learners hit nobody: a team that hides waits for the clock, so make timeout "
               "more negative (--reward REWARD_TIMEOUT=-2), and check hit_taken is not far below -1",
}

STATUS_SECONDS = 1.0
# A failing telemetry write is logged at most this often: it is retried every
# second, and a full disk would otherwise bury the trainer's own lines.
TELEMETRY_WARN_SECONDS = 60.0
# The trainer's log line reads its tally, and the tally is emptied every this
# many updates so the line reads the last few of them. Never emptied, it would
# average the whole run and stop moving.
LOG_WINDOW_UPDATES = 5

# Filled by import_engine() once the environment is set. Module globals and not
# imports at the top for the reason in sdk/__init__.py: numba reads
# NUMBA_NUM_THREADS when it is first imported, so importing the engine here
# would fix the thread count before main() has read --threads.
config = None
curriculum = None
kernels = None
persist = None
policy = None
sim = None
themes = None
trainer_module = None
torch = None


def refuse(message: str) -> None:
    """Stop before anything starts, saying why. Exit status 2, as argparse uses."""
    logger.error(message)
    sys.exit(2)


def build_parser() -> argparse.ArgumentParser:
    """Every flag of sdk.train; sdk.tui reads the defaults off it, so there is one rule set."""
    parser = argparse.ArgumentParser(prog="python -m sdk.train",
                                     description="Train SELFPLAYTV teams against frozen checkpoints")
    parser.add_argument("--bundle", required=True,
                        help="bundle directory: teams.json, manifest.json and one .pt per model")
    parser.add_argument("--run", required=True, help="run directory, created if missing")
    parser.add_argument("--learn", nargs="+", required=True, help='teams that learn, e.g. "MY TEAM"')
    parser.add_argument("--against", nargs="+", default=None,
                        help="opponent teams (default: every other team in the bundle)")
    parser.add_argument("--arenas", type=int, default=64,
                        help="arenas simulated side by side (default 64)")
    parser.add_argument("--hours", type=float, default=None, help="stop after this many hours")
    parser.add_argument("--steps", type=int, default=None,
                        help="stop after this many agent-steps in this process")
    parser.add_argument("--device", default=None, help="cuda, cuda:1, cpu (default: cuda if available)")
    parser.add_argument("--gpu-memory", type=float, default=None,
                        help="fraction of the GPU's memory this process may take, 0..1")
    # Half and not all: a run that takes every core leaves the machine unusable
    # for everything else on it, and the speed lost to the other half is small.
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 1) // 2),
                        help="threads for the simulation and for torch (default: half the CPUs)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the maps, the lineups, the ends, the sampled actions and the "
                             "order the updates read them in (default 0)")
    parser.add_argument("--theme", default=None, help="one map theme (default: all of them)")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="updates between saves")
    # Internal, for sdk.tui: the pid of the app that started this training. The
    # app signals its trainer when it closes, but a crash or a SIGKILL runs no
    # code there, so the trainer checks for itself. Not in the help: a training
    # outlives the app only when its settings say so, and from nowhere else.
    parser.add_argument("--stop-with-parent", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rewards", default=None, metavar="FILE",
                        help="a Lua file of reward weights, e.g. examples/rewards.lua; "
                             "check one with python -m sdk.rewards FILE")
    parser.add_argument("--reward", action="append", default=[], metavar="NAME=VALUE",
                        help="override one weight, by its key in the weights file or its REWARD_* "
                             "name, e.g. timeout=-2; repeatable, and it wins over --rewards")
    parser.add_argument("--restart", action="store_true",
                        help="throw the run's checkpoints away and copy the bundle again")
    parser.add_argument("--play-only", action="store_true",
                        help="play the --learn teams' matches on the full arena without learning or "
                             "saving, to measure a bundle")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def parse_rewards(pairs: List[str]) -> Dict[str, float]:
    """The --reward overrides as {REWARD_NAME: value}; whether config has the name is checked later.

    A weight is named as the weights file names it, `timeout`, or as config
    does, `REWARD_TIMEOUT`: the file is where its owner learned the names, and
    a flag that took only the other spelling would refuse every one of them.
    """
    rewards = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        name = name.strip().upper()
        if not sep or not name:
            refuse(f"--reward {pair!r}: expected NAME=VALUE, e.g. timeout=-2")
        if not name.startswith(reward_files.KNOB_PREFIX):
            name = reward_files.KNOB_PREFIX + name
        # float() also reads nan, inf and 1_0. The first two are a run that
        # skips every update and saves all the same, and the third is ten where
        # its owner may have meant something else; the weights file refuses
        # all three, and so does this.
        try:
            number = float(value)
        except ValueError:
            number = math.nan
        if not math.isfinite(number) or "_" in value:
            refuse(f"--reward {pair!r}: {value!r} is not a number such as -2, 0.5 or 1e-3")
        rewards[name] = number
    return rewards


def check_args(args: argparse.Namespace) -> None:
    """Everything that can be refused before the engine is imported."""
    if args.arenas < 1:
        refuse("--arenas must be at least 1")
    if args.threads < 1:
        refuse("--threads must be at least 1")
    if args.checkpoint_every < 1:
        refuse("--checkpoint-every must be at least 1")
    if args.hours is not None and args.hours <= 0:
        refuse("--hours must be positive")
    if args.steps is not None and args.steps <= 0:
        refuse("--steps must be positive")
    if args.gpu_memory is not None and not 0.0 < args.gpu_memory <= 1.0:
        refuse("--gpu-memory is a fraction between 0 and 1")
    bundle = os.path.realpath(args.bundle)
    run = os.path.realpath(args.run)
    if not os.path.isdir(bundle):
        refuse(f"--bundle {bundle}: no such directory; a bundle of your own is created with "
               f"python -m sdk.new_team (see its --help)")
    for name in (TEAMS_NAME, MANIFEST_NAME):
        if not os.path.isfile(os.path.join(bundle, name)):
            refuse(f"{bundle} holds no {name}; a bundle is a directory with {TEAMS_NAME}, "
                   f"{MANIFEST_NAME} and one .pt per model")
    # save_run rewrites every roster model's .pt and the manifest beside them, so
    # a run inside its own bundle would overwrite the very files it was handed,
    # the frozen opponents' included. Inside ANY bundle, and not only the one
    # named here: a run pointed into the shipped bundle/ would leave its files
    # among the ones that are only ever read.
    holder = bundle_above(run)
    if holder:
        refuse(f"--run {run} is inside the bundle {holder}; put the run somewhere else")
    # And the other way round: a bundle at RUN/checkpoints IS the run's checkpoint
    # directory, so --restart would delete it and a resume would save over it.
    checkpoints = os.path.join(run, CHECKPOINTS_NAME)
    if bundle == checkpoints or bundle.startswith(checkpoints + os.sep):
        refuse(f"--bundle {bundle} is inside the run's {checkpoints}; put the bundle somewhere else")


def bundle_above(path: str) -> Optional[str]:
    """The bundle `path` is, or lies inside: the nearest directory upwards holding both bundle files."""
    while True:
        if all(os.path.isfile(os.path.join(path, name)) for name in (TEAMS_NAME, MANIFEST_NAME)):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent


def read_json(path: str) -> Dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write JSON so a reader never sees half a file: a tmp beside it, then a rename.

    One writer per run, so a fixed tmp name is enough; the viewer only ever
    reads the final name, and os.replace swaps it in one step.
    """
    staging = path + ".tmp"
    with open(staging, "w") as handle:
        json.dump(payload, handle)
    os.replace(staging, path)


def read_bundle(bundle: str) -> Tuple[Dict[str, Any], List[str]]:
    """teams.json and the manifest's nick order, refused if either is not what is expected."""
    teams_doc = read_json(os.path.join(bundle, TEAMS_NAME))
    if teams_doc.get("version") != TEAMS_VERSION or not isinstance(teams_doc.get("teams"), dict):
        refuse(f"{TEAMS_NAME} in {bundle} is not a version {TEAMS_VERSION} team list")
    nicks = read_json(os.path.join(bundle, MANIFEST_NAME)).get("nicks") or []
    if not nicks:
        refuse(f"{MANIFEST_NAME} in {bundle} names no models")
    return teams_doc, list(nicks)


def resolve_team(name: str, teams: Dict[str, Any]) -> str:
    """A team's name as the bundle spells it. Case does not matter, and WHITE means WHITE TEAM."""
    wanted = name.strip().upper()
    for team in teams:
        if team.upper() in (wanted, wanted + " TEAM"):
            return team
    refuse(f"No team {name!r} in the bundle; it holds {', '.join(sorted(teams))}")
    return ""


def resolve_teams(names: List[str], teams: Dict[str, Any]) -> List[str]:
    """Resolved names, in the order given, each once."""
    resolved = []
    for name in names:
        team = resolve_team(name, teams)
        if team not in resolved:
            resolved.append(team)
    return resolved


def check_bundle(teams: Dict[str, Any], nicks: List[str]) -> None:
    """Every team is exactly five models the manifest knows, and no nickname is used twice.

    Over the whole bundle and not only the teams that play, because a nickname
    is a checkpoint's FILE NAME: two models spelt alike are one file, whoever is
    on the field. And without regard to case, because that is how macOS and
    Windows compare file names - there hiroshi.pt and Hiroshi.pt are one file,
    and the second written is the end of the first.

    Exactly five and not at least five: a sixth member would be drawn into the
    lineup at random, five of six a round, and the team trained would be none
    of the fives its owner has in mind.

    A team's name is checked as well, by the rule sdk.new_team makes one under:
    bundles are passed between people, the name is written to the terminal by
    every command, and one carrying control characters would write escape
    sequences there.
    """
    owner = {}
    for team, entry in teams.items():
        if not team or not team.isprintable():
            refuse(f"{TEAMS_NAME} holds the team name {team!r}, which is not visible text on one line")
        members = entry.get("members") or []
        if len(members) != TEAM_SIZE:
            refuse(f"{team} has {len(members)} members and a team fields exactly {TEAM_SIZE}")
        for nick in members:
            if not isinstance(nick, str) or not nick.isprintable():
                refuse(f"{team}: the nickname {nick!r} is not visible text")
        unknown = [nick for nick in members if nick not in nicks]
        if unknown:
            refuse(f"{team}: {', '.join(unknown)} not in the bundle's {MANIFEST_NAME}")
        for nick in members:
            taken = owner.get(nick.casefold())
            if taken:
                refuse(f"{nick} of {team} and {taken[1]} of {taken[0]} are the same nickname (case "
                       f"does not count: a nickname is a file name); every model needs its own")
            owner[nick.casefold()] = (team, nick)
    folded = [nick.casefold() for nick in nicks]
    twice = sorted({nick for nick in nicks if folded.count(nick.casefold()) > 1})
    if twice:
        refuse(f"{MANIFEST_NAME} names {', '.join(twice)} more than once (case does not count: "
               f"a nickname is a file name)")


def check_rivals(learners: List[str], opponents: List[str], teams: Dict[str, Any]) -> None:
    """Every learner has somebody to play."""
    for team in learners:
        if [rival for rival in opponents if rival != team]:
            continue
        if len(teams) == 1:
            refuse(f"{team} is the only team in the bundle and has nobody to play; create a "
                   f"team first with python -m sdk.new_team (see its --help)")
        refuse(f"{team} has nobody to play: --against names no team but itself")


def check_nicks(teams: Dict[str, Any], playing: List[str]) -> None:
    """Refuse a nickname that cannot name a checkpoint, before a run directory exists.

    The engine refuses it too, but only at the restore, by which time the run
    has been created and its marker written for a run that can never start.
    """
    for team in playing:
        for nick in teams[team]["members"]:
            problem = config.nick_problem(nick)
            if problem:
                refuse(f"{team}: {nick!r} cannot name a model: {problem}")


def write_marker(run: str, bundle: str, learners: List[str]) -> None:
    """RUN/sdk-run.json: what the run was made from, and the step counter it was made at.

    The counter travels with the checkpoints - a bundle may have been trained
    before it was handed over - so the steps THIS run has taken are the counter
    less what it read when the bundle was copied in. Kept here, written once,
    and not in status.json, which is rewritten every second and lost with the
    process that dies before its last save.
    """
    steps = int(read_json(os.path.join(bundle, MANIFEST_NAME)).get("steps", 0))
    write_json(os.path.join(run, MARKER_NAME),
               {"version": MARKER_VERSION, "created_at": time.time(), "bundle": bundle,
                "learners": learners, "steps": steps})


def warn_changed_learners(run: str, learners: List[str]) -> None:
    """Say so when a resumed run is asked to train other teams than it was created for.

    Nothing stops it: the checkpoints load, the step counter and the tables
    carry on, and the dashboard then shows one team's matches on top of
    another's history. A run directory per team is the rule, and this is the
    only place a slip against it shows.
    """
    before = read_json(os.path.join(run, MARKER_NAME)).get("learners")
    if isinstance(before, list) and sorted(before) != sorted(learners):
        logger.warning(f"This run was created to train {', '.join(map(str, before))} and is now asked "
                       f"to train {', '.join(learners)}: its step counter and its tables carry on "
                       f"as one run. Give each team a --run of its own to keep them apart")


def run_base_steps(run: str) -> Optional[int]:
    """The step counter the run was created at, or None for a marker written without it."""
    steps = read_json(os.path.join(run, MARKER_NAME)).get("steps")
    return steps if isinstance(steps, int) else None


def check_marker(run: str, checkpoints: str) -> None:
    """A RUN/checkpoints with files in it and no marker beside is not this SDK's to touch.

    Refused for a resume and for --restart alike: a resume would save over
    whatever trained those files, and --restart would delete them.
    """
    if os.path.isfile(os.path.join(run, MARKER_NAME)):
        return
    if os.path.isdir(checkpoints) and os.listdir(checkpoints):
        refuse(f"{checkpoints} holds files and {run} has no {MARKER_NAME}: it is not an SDK run, "
               f"and nothing in it will be resumed, saved over or deleted; give a new --run")


def lock_run(run: str) -> Any:
    """Take RUN/lock for the life of the process, or refuse if another sdk.train holds it.

    Two processes on one RUN restore the same checkpoint, diverge, and save
    over each other every few updates, leaving a team on disk that is half of
    one training and half of the other. The handle is returned so the caller
    keeps it open: the lock goes when the process does, a crash included.
    """
    try:
        import fcntl
    except ImportError:
        logger.warning(f"No fcntl on this platform: nothing stops a second sdk.train on {run}")
        return None
    # "a" and not "w": "w" empties the file before the lock is asked for, so a
    # refused second start would wipe the record of the process that holds it.
    handle = open(os.path.join(run, LOCK_NAME), "a")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        refuse(f"Another sdk.train is using {run}; stop it first or give a different --run")
    # Who holds it, written under the lock: the flock says whether a trainer is
    # alive, and this says which process to signal (sdk.tui reads it).
    handle.truncate(0)
    json.dump({"pid": os.getpid(), "started_at": time.time()}, handle)
    handle.flush()
    return handle


def copy_checkpoint(source: str, target: str) -> None:
    """Copy one checkpoint so that it is there whole or not at all.

    Staged beside the target and renamed in: a copy cut short would otherwise
    leave half a network under the final name, and a later start, which never
    copies over a file the run already holds, would keep it.
    """
    shutil.copy2(source, target + ".tmp")
    os.replace(target + ".tmp", target)


def copy_checkpoints(bundle: str, checkpoints: str, missing_only: bool) -> List[str]:
    """Copy the bundle's .pt files into the run; with missing_only, never over one it holds."""
    copied = []
    for name in sorted(os.listdir(bundle)):
        target = os.path.join(checkpoints, name)
        if not name.endswith(".pt") or (missing_only and os.path.exists(target)):
            continue
        copy_checkpoint(os.path.join(bundle, name), target)
        copied.append(name[:-len(".pt")])
    return copied


def prepare_run(run: str, checkpoints: str, bundle: str, learners: List[str], restart: bool) -> bool:
    """The run's checkpoint directory, filled from the bundle on the first start.

    Copied rather than read in place because save_run rewrites every roster
    model's file, the frozen ones included -- harmless on a copy, and the end of
    the bundle on the original. On a later start the run's own files ARE the
    progress, so they are never copied over unless --restart says so.

    What a later start does copy is what the run does not hold at all: the
    models of a team added to the bundle since the run began. Without that a
    new opponent would be refused for having no checkpoint, and a new learner
    would start from random weights instead of the ones its bundle gives it.

    The manifest goes in last: a copy interrupted half way leaves no manifest,
    and the next start then sees an unstarted run and copies again, rather than
    resuming from a directory with half its models missing.

    Returns True when resuming. --restart clears the score files too: they
    describe networks it has just thrown away, and status.json would otherwise
    carry their tables on into the new run. The marker is written before the
    first file is copied, so a copy interrupted half way is still a run the next
    start may finish; --restart writes it again, because the bundle named on
    this command line, and its step counter, are what the run is now made from.
    """
    if restart:
        if os.path.isdir(checkpoints):
            shutil.rmtree(checkpoints)
        for name in (STATUS_NAME, MATCHES_NAME, PROGRESS_NAME):
            if os.path.isfile(os.path.join(run, name)):
                os.remove(os.path.join(run, name))
        logger.warning(f"--restart: {checkpoints} and the score files cleared, copying the bundle again")
    os.makedirs(checkpoints, exist_ok=True)
    if os.path.isfile(os.path.join(checkpoints, MANIFEST_NAME)):
        logger.info(f"Resuming the run in {checkpoints}")
        added = copy_checkpoints(bundle, checkpoints, missing_only=True)
        if added:
            logger.info(f"New in the bundle since this run started, copied in: {', '.join(added)}")
        return True
    if restart or not os.path.isfile(os.path.join(run, MARKER_NAME)):
        write_marker(run, bundle, learners)
    copied = copy_checkpoints(bundle, checkpoints, missing_only=False)
    shutil.copy2(os.path.join(bundle, MANIFEST_NAME), os.path.join(checkpoints, MANIFEST_NAME))
    logger.info(f"New run: {len(copied)} checkpoints copied from {bundle} to {checkpoints}")
    return False


def set_environment(threads: int) -> None:
    """The one thing the engine reads from the environment at import that a run decides.

    numba sizes its thread pool from NUMBA_NUM_THREADS once, when it is first
    imported, so this has to run before import_engine.
    """
    os.environ["NUMBA_NUM_THREADS"] = str(threads)


def import_engine() -> None:
    """Import the engine and the trainer, into this module's globals."""
    global config, curriculum, kernels, persist, policy, sim, themes, trainer_module, torch
    import torch as torch_module
    from engine import config as config_module
    from engine.env import kernels as kernels_module
    from engine.env import sim as sim_module
    from engine.env import themes as themes_module
    from engine.rl import curriculum as curriculum_module
    from engine.rl import persist as persist_module
    from engine.rl import policy as policy_module
    from trainer import selfplaytv
    torch = torch_module
    config = config_module
    kernels = kernels_module
    sim = sim_module
    themes = themes_module
    curriculum = curriculum_module
    persist = persist_module
    policy = policy_module
    trainer_module = selfplaytv


def apply_rewards(rewards: Dict[str, float]) -> None:
    """Set each override on config itself.

    The trainer reads every REWARD_* as config.NAME at call time, so the
    attribute reaches all of them, whichever way config came by its default.
    """
    known = reward_names()
    for name, value in rewards.items():
        if name not in known:
            refuse(f"--reward {name}: config has no such knob; it has {', '.join(known)}")
        setattr(config, name, value)


def reward_names() -> List[str]:
    return reward_files.knob_names(config)


def reward_weights() -> Dict[str, float]:
    """Every reward knob as config holds it now: the defaults until apply_rewards, what is in force after."""
    return {name: float(getattr(config, name)) for name in reward_names()}


def read_reward_file(path: str) -> Dict[str, float]:
    """The --rewards file as {REWARD_NAME: value}, refused with the line it fails on."""
    try:
        return reward_files.read_rewards(path, reward_names())
    except OSError as exc:
        refuse(f"--rewards {path}: {exc.strerror}")
    except ValueError as exc:
        refuse(f"--rewards {path}: {str(exc)}")
    return {}


def previous_rewards(path: str) -> Dict[str, float]:
    """The weights the run's last start wrote into status.json; nothing when it cannot be read."""
    try:
        recorded = read_json(path).get("rewards") or {}
        return {name: float(value) for name, value in recorded.items()}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def log_rewards(defaults: Dict[str, float], sources: Dict[str, str], resumed: bool,
                status_path: str) -> None:
    """Say which weights are in force and where the changed ones came from; on a resume, what moved.

    The weights belong to a start and not to the run: they are whatever this
    command line gave, so a resume without --rewards trains on the defaults.
    Said in so many words, and as a warning when the last start trained on
    something else, because nothing else would show it: the run carries on, its
    tables carry on, and the team is quietly being paid for a different game.
    Called before the first status write of this start, which replaces the
    file the last start's weights are read from.
    """
    weights = reward_weights()
    # Only a weight that is not its default is marked with where it came from:
    # a file that restates all nine would otherwise mark all nine, and the one
    # that matters would be as hard to find as with no marks at all.
    changed = [name for name, value in weights.items() if value != defaults[name]]
    listed = ", ".join(f"{reward_files.key_of(name)} {value:g}"
                       + (f" ({sources[name]})" if name in changed else "")
                       for name, value in weights.items())
    origin = "the unmarked at their defaults" if changed else "all at their defaults"
    logger.info(f"Reward weights, {origin}: {listed}")
    if not resumed:
        return
    if not sources:
        logger.info("Resumed without --rewards and without --reward: from here the run trains on the "
                    "default weights, whatever it was started with")
    before = previous_rewards(status_path)
    moved = [f"{reward_files.key_of(name)} was {before[name]:g} and is now {value:g}"
             for name, value in weights.items() if name in before and before[name] != value]
    if moved:
        logger.warning(f"The reward weights are not those of this run's last start: {', '.join(moved)}. "
                       f"A run does not remember its weights; pass the same --rewards file to keep them")


def warn_timeout_off() -> None:
    """One line when the learners go unpunished for running out the clock.

    The engine's default timeout weight is 0, and the drawn round's penalty
    rarely reaches a side that hides: most rounds that end on time are won on
    survivors. A run started without the example weights therefore trains a
    team that may learn to wait, and nothing else in the log would say so.
    """
    if config.REWARD_TIMEOUT == 0:
        logger.warning("The timeout penalty is off (REWARD_TIMEOUT 0), so waiting out the clock costs "
                       "nothing; pass --rewards examples/rewards.lua or --reward REWARD_TIMEOUT=-2")


def personality_scales(teams_doc: Dict[str, Any], nicks: List[str]) -> Dict[str, float]:
    """Each model's entropy scale, by nickname.

    The checkpoint does not store it: config.personality hands it out by roster
    INDEX, and an index here means nothing stable, because the roster is
    narrowed to the teams that play. So teams.json carries each model's scale by
    nickname, under "personality". A bundle without that table falls back to the
    order of its manifest.
    """
    stored = teams_doc.get("personality")
    if isinstance(stored, dict) and stored:
        scales = {}
        for nick, value in stored.items():
            scale = value.get("entropy_scale") if isinstance(value, dict) else value
            scales[nick] = float(scale)
        return scales
    logger.info(f"No personality in {TEAMS_NAME}; entropy scales follow the order of {MANIFEST_NAME}")
    return {nick: float(config.personality(index)["entropy_scale"]) for index, nick in enumerate(nicks)}


def install_personalities(scales: Dict[str, float]) -> None:
    """Wrap config.personality so a model keeps its own entropy scale, whatever its index here.

    The roster below is narrowed to the models on the field, which moves every
    index; the trainer asks config.personality(index) at the PPO update, so the
    answer is looked up by the nickname at that index instead. Everything else
    in a personality -- the hidden width the checkpoints are built for -- stays
    what the index gives, which is the same for every model.
    """
    by_index = config.personality

    def personality_by_nick(index: int) -> Dict[str, float]:
        base = by_index(index)
        scale = scales.get(config.MODEL_NICKS[index])
        if scale is None:
            return base
        return dict(base, entropy_scale=scale)

    config.personality = personality_by_nick


def field_roster(nicks: List[str], teams: Dict[str, Any], playing: List[str]) -> List[str]:
    """The models of the teams that play, in the manifest's order.

    Only those, and not the whole manifest: the trainer allocates a network, an
    optimiser and a rollout buffer on the device for EVERY roster model, fielded
    or not, so a bundle of many teams would spend most of that memory on models
    that never take a step.
    """
    wanted = set()
    for team in playing:
        wanted.update(teams[team]["members"])
    return [nick for nick in nicks if nick in wanted]


def check_opponent_files(checkpoints: str, teams: Dict[str, Any], learners: List[str],
                         opponents: List[str]) -> None:
    """A frozen opponent without weights would be a random network that never learns."""
    for team in opponents:
        if team in learners:
            continue
        missing = [nick for nick in teams[team]["members"]
                   if not os.path.isfile(os.path.join(checkpoints, f"{nick}.pt"))]
        if missing:
            refuse(f"{team} is a frozen opponent and {', '.join(missing)} "
                   f"{'has' if len(missing) == 1 else 'have'} no checkpoint in {checkpoints}")


def nick_index() -> Dict[str, int]:
    return {nick: index for index, nick in enumerate(config.MODEL_NICKS)}


def seat_teams(nicks: List[str], teams: Dict[str, Any],
               playing: List[str]) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Make the teams that play the roster, and say who is who in it.

    Returns each roster model's team, by roster index, and each team's members
    as roster indices in the order teams.json lists them. Everything built
    after this - the Trainer, its buffers, the lineups - is sized and indexed by
    the roster set here.
    """
    config.set_roster(field_roster(nicks, teams, playing))
    index_of = nick_index()
    team_of_model = [""] * config.N_MODELS
    for team in playing:
        for nick in teams[team]["members"]:
            team_of_model[index_of[nick]] = team
    members = {team: np.array([index_of[nick] for nick in teams[team]["members"]], dtype=np.int64)
               for team in playing}
    return team_of_model, members


def new_match(first: str, second: str) -> Dict[str, Any]:
    """One arena's match in progress, between two teams, at nil-nil.

    `on_side[s]` is the index into `teams` of the team standing on end s this
    round; the ends are redrawn every round, so a round's winner, which the
    simulation names by END, is turned into a team through it.
    """
    match = {"teams": [first, second], "on_side": [0, 1], "score": [0, 0], "rounds": 0, "draws": 0}
    match.update({key: 0 for key in REASON_KEYS})
    return match


def make_assign_slots(telemetry: Dict[str, Any], members: Dict[str, np.ndarray], learners: List[str],
                      rivals_of: Dict[str, List[str]]) -> Callable:
    """The trainer's draw_lineups: team against team, a match at a time.

    Per arena: a learner team at random and a rival from the opponent set at
    random, kept for the whole match -- first to ROUNDS_TO_WIN -- so the score
    is a match score and not a pile of unrelated rounds. Within the match each
    round puts the two teams on a random end and shuffles each team along it, so
    no model learns one spot on the line and no team learns one end of the
    field. A pair still in its match keeps it when apply_phase rebuilds every
    arena; a decided one (on_step sets the record to None) draws a new pair. The
    writes at the bottom are the three arrays trainer.assign_slots describes,
    which collect_rollout and learn read.
    """
    def assign_slots_by_team(trainer: Any, rows: np.ndarray) -> None:
        if rows.size == 0:
            return
        matches = telemetry["arenas"]
        for row in rows:
            match = matches[row]
            if match is None:
                learner = learners[int(trainer.rng.integers(len(learners)))]
                rivals = rivals_of[learner]
                match = new_match(learner, rivals[int(trainer.rng.integers(len(rivals)))])
                matches[row] = match
            match["on_side"] = [0, 1] if trainer.rng.random() < 0.5 else [1, 0]
            ends = [members[match["teams"][index]] for index in match["on_side"]]
            chosen = np.concatenate([trainer.rng.permutation(side)[:trainer.team_size] for side in ends])
            trainer.model_of_slot[row] = chosen
            trainer.plays[row] = False
            trainer.plays[row, chosen] = True
            trainer.slot_of_model[row] = 0
            trainer.slot_of_model[row, chosen] = np.arange(trainer.n_players)

    return assign_slots_by_team


def make_curriculum_clock(learners: np.ndarray) -> Callable:
    """The trainer's read_clock: the learners' own steps, summed.

    Not the run's step counter and not the whole field's steps: a frozen
    veteran's hundreds of millions would put a team starting from nothing on the
    full arena with its learning rate and entropy annealed to their floors on
    its first update. Frozen models never gain a step, so here they set no pace
    at all. Summed over the learners because the schedules are written against
    a team's total and not one model's.
    """
    def learners_clock(trainer: Any) -> int:
        return int(trainer.steps_by_model[learners].sum())

    return learners_clock


def warn_mixed_learners(trainer: Any, members: Dict[str, np.ndarray], learners: List[str]) -> None:
    """Say so when the learner teams stand in different phases of the curriculum.

    The learners share one clock, the sum of all their steps. A team of
    untrained networks beside a veteran is therefore put on the full arena, at
    the floor learning rate, from its first update: the start the curriculum
    exists to spare it. Each team's phase is read off its own steps here, which
    is what it would train in alone.
    """
    alone = {team: curriculum.phase_for(int(trainer.steps_by_model[members[team]].sum()))["name"]
             for team in learners}
    if len(set(alone.values())) > 1:
        logger.warning(f"The learners are at different stages of the curriculum on their own "
                       f"({', '.join(f'{team}: {phase}' for team, phase in alone.items())}) and share "
                       f"one clock here, so the younger ones skip theirs. Train them in separate runs")


def full_arena_clock(unused_trainer: Any = None) -> int:
    """A read_clock past every early phase: the full arena, whoever is on it.

    For play that MEASURES and does not train: --play-only here, and the
    matches of sdk.demo. The learners' own clock would put a young team on the
    small close-quarters arena with autofire on, and a result from there says
    nothing about the game the team is being trained for. The step the last
    timed phase ends at is the first step of the full arena.
    """
    return int(curriculum.PHASES[-2]["until"])


def new_telemetry(arenas: int, teams: List[str], team_of_model: List[str]) -> Dict[str, Any]:
    return {
        "team_of_model": team_of_model,
        # The match in progress in each arena, None between matches.
        "arenas": [None] * arenas,
        "played": 0,
        # Finished match records, newest first, as status.json carries them.
        "recent": collections.deque(maxlen=RECENT_MATCHES),
        "standings": {team: {key: 0 for key in STANDING_KEYS} for team in teams},
        # Per-model totals moved out of trainer.tally before each reset of it;
        # see bank_tally. The live tally is added on top when they are written.
        "banked": {name: np.zeros(len(team_of_model), dtype=np.int64) for name, unused in PLAYER_KEYS},
        # Filled by main once the teams are seated: each playing team's roster
        # indices, which of the teams learn, and the step counter the run was
        # created at. Left empty, status.json simply goes without what they feed.
        "members": {},
        "learners": [],
        "run_base": None,
        # What the learners were paid over the last closed tally window; see
        # note_reward_window.
        "reward_window": None,
        # Updates in a row each stall condition has held, and the stall in
        # force as status.json carries it, or None; see check_health.
        "streaks": {name: 0 for name in STALL_ADVICE},
        "health": None,
        "written_at": 0.0,
        "warned_at": None,
    }


def warn_telemetry(telemetry: Dict[str, Any], what: str, exc: OSError) -> None:
    """Log a failed status, match or progress write, throttled, and carry on training.

    The files are a courtesy to the viewer, and an OSError from one (a full
    disk, or Windows refusing os.replace while sdk.watch holds status.json
    open) would otherwise end the run without its exit save.
    """
    now = time.monotonic()
    if telemetry["warned_at"] is None or now - telemetry["warned_at"] >= TELEMETRY_WARN_SECONDS:
        logger.warning(f"{what} not written: {type(exc).__name__}: {str(exc)}")
        telemetry["warned_at"] = now


def append_line(run: str, name: str, line: Dict[str, Any], telemetry: Dict[str, Any]) -> None:
    try:
        with open(os.path.join(run, name), "a") as handle:
            handle.write(json.dumps(line) + "\n")
    except OSError as exc:
        warn_telemetry(telemetry, name, exc)


def describe_match(record: Dict[str, Any]) -> str:
    """`Match MY TEAM 7:4 WHITE TEAM (11 rounds: time 3, elimination 6, capture 2, draws 1)`."""
    first, second = record["teams"]
    score = f"{record['score'][0]}:{record['score'][1]}"
    called_off = "" if record["finished"] else " unfinished"
    endings = ", ".join(f"{key} {record[key]}" for key in ("time", "elimination", "capture", "draws"))
    return f"Match {first} {score} {second}{called_off} ({record['rounds']} rounds: {endings})"


def finish_match(run: str, telemetry: Dict[str, Any], trainer: Any, match: Dict[str, Any]) -> None:
    """Record a decided (or called-off) match: the log, matches.jsonl and both teams' standings."""
    finished = max(match["score"]) >= config.ROUNDS_TO_WIN
    record = {"at": time.time(), "steps": int(trainer.steps_done), "teams": list(match["teams"]),
              "score": list(match["score"]), "rounds": match["rounds"], "time": match["time"],
              "elimination": match["elimination"], "capture": match["capture"],
              "draws": match["draws"], "finished": finished}
    for index, team in enumerate(match["teams"]):
        standing = telemetry["standings"][team]
        standing["matches"] += 1
        if not finished:
            standing["unfinished"] += 1
        elif match["score"][index] >= config.ROUNDS_TO_WIN:
            standing["won"] += 1
        else:
            standing["lost"] += 1
        for key in ("rounds", "draws") + REASON_KEYS:
            standing[key] += match[key]
    telemetry["played"] += 1
    telemetry["recent"].appendleft(record)
    logger.info(describe_match(record))
    append_line(run, MATCHES_NAME, record, telemetry)


def record_endings(run: str, telemetry: Dict[str, Any], trainer: Any, done: np.ndarray,
                   winner: np.ndarray) -> None:
    """Score every finished round into its arena's match, and close the matches that are decided.

    Read here, before finish_episodes resets the arenas, because the reset
    clears the capture latch and revives the dead, and with them the reason.
    The winner is the one finish_episodes pays out on, so the match score and
    the players' wins and losses below can never disagree. A closed match
    leaves None behind, and the assign_slots that finish_episodes calls next
    draws that arena a new pair.
    """
    rows = np.nonzero(done)[0]
    if rows.size == 0:
        return
    cap = MATCH_ROUND_CAP_FACTOR * config.ROUNDS_TO_WIN
    reasons = sim.end_reasons(trainer.state, rows)
    for row, code in zip(rows.tolist(), reasons.tolist()):
        match = telemetry["arenas"][row]
        match["rounds"] += 1
        match[REASON_KEYS[code]] += 1
        side = int(winner[row])
        if side >= 0:
            match["score"][match["on_side"][side]] += 1
        else:
            match["draws"] += 1
        if max(match["score"]) >= config.ROUNDS_TO_WIN or match["rounds"] >= cap:
            finish_match(run, telemetry, trainer, match)
            telemetry["arenas"][row] = None


def bank_tally(telemetry: Dict[str, Any], tally: Dict[str, np.ndarray]) -> None:
    """Add a tally just taken off the trainer into the run's totals.

    The tally is the trainer's window: run_loop empties it every
    LOG_WINDOW_UPDATES so the trainer's log line keeps reading the last few
    updates. Those resets (drop_tally in run_loop) are the only ones there are,
    so banking each one makes the per-player totals run-long. The caller swaps
    the tally out first and banks the old one after, so no count can ever sit in
    the banked totals and the live tally at once, which status.json would add up
    twice.
    """
    for name, key in PLAYER_KEYS:
        telemetry["banked"][name] += tally[key]


def note_reward_window(telemetry: Dict[str, Any], tally: Dict[str, np.ndarray], first: int,
                       last: int) -> None:
    """Keep what each learner team was paid over a tally window just closed, and log it.

    The one figure that reads the reward weights back: the points columns are
    the scoreboard's and do not move when a weight does. Taken where run_loop
    takes the tally off the trainer, because only there is it whole: learn()
    credits a rollout's reward after the rollout, so a tally read mid-rollout
    holds rounds whose pay has not been added yet.

    Stored as sums, the reward of the team's five and the rounds they played
    counted per player, so reward over player_rounds is what one player was
    paid for one round - the unit the weights are written in. It is everything
    the buffers held, the fixed shaping terms included, and not the weights
    alone.
    """
    teams = {}
    for team in telemetry["learners"]:
        members = telemetry["members"][team]
        teams[team] = {"reward": round(float(tally["reward"][members].sum()), 3),
                       "player_rounds": int(tally["matches"][members].sum())}
    telemetry["reward_window"] = {"from_update": first, "to_update": last, "teams": teams}
    paid = ", ".join(f"{team} {sums['reward'] / sums['player_rounds']:+.2f}"
                     for team, sums in teams.items() if sums["player_rounds"])
    if paid:
        logger.info(f"Reward per player per round over updates {first}-{last}: {paid}")


def learner_hits(trainer: Any, telemetry: Dict[str, Any]) -> int:
    """The hits the learners' models have in the open tally window."""
    hits = trainer.tally["hits"]
    return int(sum(hits[telemetry["members"][team]].sum() for team in telemetry["learners"]))


def check_health(telemetry: Dict[str, Any], trainer: Any, stats: Dict[str, float], hits: int) -> None:
    """Warn once when learning has stalled, and once when it moves again; never stop it.

    Each condition counts the updates in a row it has held, and the first of
    them to reach STALL_UPDATES is the stall in force. Said when it begins and
    not on every update, because a stall goes on for as long as nobody acts on
    it, and a line repeated a thousand times is a line nobody reads.
    """
    holds = {"policy frozen": stats["approx_kl"] < STALL_KL and stats["clip_fraction"] == 0.0,
             "entropy collapsed": stats["entropy"] < STALL_ENTROPY,
             "no hits": hits == 0}
    for name, held in holds.items():
        telemetry["streaks"][name] = telemetry["streaks"][name] + 1 if held else 0
    stalled = [name for name, count in telemetry["streaks"].items() if count >= STALL_UPDATES]
    before = telemetry["health"]
    if not stalled:
        if before is not None:
            logger.info(f"Learning moves again at update {trainer.updates}: {before['stalled']} has cleared")
            telemetry["health"] = None
        return
    name = stalled[0]
    if before is not None and before["stalled"] == name:
        return
    since = trainer.updates - telemetry["streaks"][name] + 1
    telemetry["health"] = {"stalled": name, "since_update": since}
    advice = STALL_ADVICE[name].format(kl=STALL_KL, entropy=STALL_ENTROPY)
    logger.warning(f"Learning has stalled, {name} since update {since}: {advice}. Training goes on")


def steps_payload(telemetry: Dict[str, Any], trainer: Any) -> Dict[str, Any]:
    """The step counts status.json carries beside the counter itself.

    `steps` is the counter the checkpoints carry, which a bundle trained before
    it was handed over brings with it. These say what a reader actually asks:
    how much THIS run has trained (run_steps, over all its starts), how much of
    that since this start (session_steps: the two differ on a resumed run), and
    how far each team's own networks have come (team_steps: the learners' are
    the curriculum clock, and a frozen team's never move).
    """
    payload = {"session_steps": int(trainer.steps_done - trainer.steps_at_start),
               "team_steps": {team: int(trainer.steps_by_model[members].sum())
                              for team, members in telemetry["members"].items()}}
    if telemetry["run_base"] is not None:
        payload["run_steps"] = int(trainer.steps_done - telemetry["run_base"])
    return payload


def players_payload(telemetry: Dict[str, Any], trainer: Any) -> Dict[str, Dict[str, Any]]:
    """Every fielded model's run totals: what is banked plus the tally still open."""
    players = {}
    for model, team in enumerate(telemetry["team_of_model"]):
        row = {"team": team}
        for name, key in PLAYER_KEYS:
            row[name] = int(telemetry["banked"][name][model] + trainer.tally[key][model])
        players[config.MODEL_NICKS[model]] = row
    return players


def resume_telemetry(telemetry: Dict[str, Any], path: str) -> None:
    """Carry the tables of a resumed run on from its last status.json.

    So the match count, the standings and the player totals keep describing the
    RUN, as matches.jsonl does by being appended to, and not just this process.
    Only teams and models still on the field are carried; a matchup changed
    between starts simply starts the newcomers at zero. A file that cannot be
    read costs the tables their history, not the run.
    """
    if not os.path.isfile(path):
        return
    # Parsed into locals and committed only once the whole file has read: a bad
    # entry half way would otherwise leave the match count carried and the
    # player totals not, and the warning below would be telling a lie.
    index_of = nick_index()
    standings = {team: {key: 0 for key in STANDING_KEYS} for team in telemetry["standings"]}
    banked = {name: np.zeros_like(values) for name, values in telemetry["banked"].items()}
    try:
        previous = read_json(path)
        if previous.get("version") != STATUS_VERSION:
            raise ValueError(f"version {previous.get('version')!r}")
        matches = previous.get("matches") or {}
        played = int(matches.get("played", 0))
        recent = list(matches.get("recent") or [])
        for team, counts in (previous.get("standings") or {}).items():
            if team in standings:
                for key in STANDING_KEYS:
                    standings[team][key] += int(counts.get(key, 0))
        for nick, counts in (previous.get("players") or {}).items():
            if nick in index_of and telemetry["team_of_model"][index_of[nick]]:
                for name, unused in PLAYER_KEYS:
                    banked[name][index_of[nick]] += int(counts.get(name, 0))
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.warning(f"{path} not carried on, the tables start from zero: {type(exc).__name__}: {str(exc)}")
        return
    telemetry["played"] = played
    for record in reversed(recent):
        telemetry["recent"].appendleft(record)
    for team, counts in standings.items():
        for key in STANDING_KEYS:
            telemetry["standings"][team][key] += counts[key]
    for name, values in banked.items():
        telemetry["banked"][name] += values
    logger.info(f"Resuming the tables from {path}: {telemetry['played']} matches played")


def steps_per_second(trainer: Any) -> float:
    """This process's agent-steps a second, as the trainer's own log line computes it."""
    elapsed = time.time() - trainer.started
    return (trainer.steps_done - trainer.steps_at_start) / max(elapsed, 1e-6)


def current_phase(trainer: Any) -> str:
    return curriculum.phase_for(trainer_module.curriculum_clock(trainer))["name"]


def standings_payload(telemetry: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    return {team: dict(counts) for team, counts in telemetry["standings"].items()}


def write_status(run: str, trainer: Any, telemetry: Dict[str, Any], header: Dict[str, Any],
                 force: bool = False) -> None:
    """status.json for sdk.watch, at most once every STATUS_SECONDS unless forced."""
    now = time.monotonic()
    if not force and now - telemetry["written_at"] < STATUS_SECONDS:
        return
    # Stamped whether or not the write lands, so a failing one is retried at the
    # status pace and not on every step.
    telemetry["written_at"] = now
    status = dict(header)
    status.update({
        "updated_at": time.time(),
        "steps": int(trainer.steps_done),
        "updates": int(trainer.updates),
        "sps": round(steps_per_second(trainer), 1),
        "phase": current_phase(trainer),
        "matches": {"played": telemetry["played"], "recent": list(telemetry["recent"])},
        "standings": standings_payload(telemetry),
        "players": players_payload(telemetry, trainer),
    })
    status.update(steps_payload(telemetry, trainer))
    if telemetry["reward_window"]:
        status["reward_window"] = telemetry["reward_window"]
    if not header.get("play_only"):
        status["health"] = telemetry["health"]
    try:
        write_json(os.path.join(run, STATUS_NAME), status)
    except OSError as exc:
        warn_telemetry(telemetry, STATUS_NAME, exc)


def make_step_hook(run: str, telemetry: Dict[str, Any], header: Dict[str, Any]) -> Callable:
    """trainer.on_step: the match score on every step, status.json throttled.

    Throttled on the monotonic clock because it runs on every control step of
    every rollout, hundreds a second, and a viewer refreshing every two seconds
    needs a file no more often than once a second.
    """
    def on_step(trainer: Any, done: np.ndarray, winner: np.ndarray) -> None:
        record_endings(run, telemetry, trainer, done, winner)
        write_status(run, trainer, telemetry, header)

    return on_step


def save(run: str, checkpoints: str, trainer: Any, telemetry: Dict[str, Any]) -> None:
    """Checkpoints, then one progress.jsonl line, so a run can be judged after the fact.

    save_checkpoints refuses, with an error line of its own, to write over a
    manifest further along than this process; it returns nothing, so the
    manifest is read back, and a refused save logs no checkpoint and adds no
    progress point.
    """
    trainer_module.save_checkpoints(trainer, checkpoints)
    if trainer_module.read_saved_steps(checkpoints) != trainer.steps_done:
        return
    line = {"at": time.time(), "steps": int(trainer.steps_done), "updates": int(trainer.updates),
            "sps": round(steps_per_second(trainer), 1), "phase": current_phase(trainer),
            "standings": standings_payload(telemetry), "reward_window": telemetry["reward_window"]}
    append_line(run, PROGRESS_NAME, line, telemetry)
    logger.info(f"Checkpoint at update {trainer.updates}, {trainer.steps_done / 1e6:.2f}M steps")


def install_stop_signals(stop: Dict[str, Any]) -> None:
    """The first SIGINT, SIGTERM or SIGHUP stops after the update in progress; a second, at once.

    Between rollouts is the only clean place to stop: a rollout abandoned half
    way has been paid for and is lost, while one let finish is learned from and
    saved. The second signal is for whoever cannot wait for that; the loop
    still saves on its way out. It interrupts only while stop["abortable"]
    is set: never inside a save, which would leave some .pt files rewritten
    under the previous manifest, and never while a tally is being banked.

    SIGHUP is a closed terminal or a dropped ssh. Left to its default it ends a
    trainer started by hand on the spot, with everything since the last save
    lost; handled, the trainer saves and exits as it does on Ctrl-C. It never
    escalates: a closing terminal delivers it twice, once from the kernel and
    once from the shell, and the second must not cut the update short. Under
    nohup the hangup stays ignored, and the run goes on.
    """
    def request_stop(signum: int, unused_frame: Any) -> None:
        if stop["requested"]:
            if signum == getattr(signal, "SIGHUP", None):
                return
            if stop["abortable"]:
                raise KeyboardInterrupt
            logger.info(f"{signal.Signals(signum).name}: already stopping, and this part is not cut short")
            return
        stop["requested"] = True
        logger.info(f"{signal.Signals(signum).name}: stopping after this update and saving")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGHUP") and signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        signal.signal(signal.SIGHUP, request_stop)


def budget_spent(trainer: Any, args: argparse.Namespace, started: float) -> Optional[str]:
    """Why the run should stop now, or None while it has budget left."""
    if args.steps is not None and trainer.steps_done - trainer.steps_at_start >= args.steps:
        return f"--steps {args.steps} reached"
    if args.hours is not None and time.time() - started >= args.hours * 3600.0:
        return f"--hours {args.hours} reached"
    return None


def restore(trainer: Any, checkpoints: str, device: Any) -> None:
    """Load the run's checkpoints into the trainer, with the counters that go with them."""
    manifest = persist.restore_run(checkpoints, trainer.nets, trainer.optimizers,
                                   trainer.normalizers, list(config.MODEL_NICKS),
                                   trainer.obs_size, device)
    # Refused rather than started fresh: with nothing restored the step counter
    # is zero under a manifest that says otherwise, and save_checkpoints then
    # refuses every save for the rest of the run -- hours of training that can
    # never be written. The lines above say why the restore found nothing.
    if not manifest:
        refuse(f"Nothing in {checkpoints} could be restored; see the lines above")
    trainer.steps_done = int(manifest.get("steps", 0))
    trainer_module.adopt_model_steps(trainer, manifest)
    trainer.updates = int(manifest.get("updates", 0))
    generation = manifest.get("generation")
    trainer.generation = generation if isinstance(generation, int) else 0
    fresh = manifest.get("fresh") or []
    if fresh:
        logger.warning(f"Starting untrained: {', '.join(fresh)}")
    trainer.steps_at_start = trainer.steps_done


def skip_learning(trainer: Any) -> Dict[str, float]:
    """learn()'s place in a --play-only run: no gradient, the same counters.

    Not learn() with an empty learner list: the trainer reads an empty
    Trainer.learning as "every model learns". The update counter still moves,
    because the tally window and log_progress are paced by it.
    """
    trainer.updates += 1
    return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "approx_kl": 0.0, "clip_fraction": 0.0, "learning_rate": 0.0}


def run_loop(args: argparse.Namespace, run: str, checkpoints: str, trainer: Any,
             telemetry: Dict[str, Any], header: Dict[str, Any]) -> None:
    """Rollout, update, log, save, until the budget is spent or a signal says stop.

    With --play-only nothing learns and nothing is saved: the bundle's weights
    are what is being measured, and matches.jsonl and status.json are the result.
    """
    stop = {"requested": False, "abortable": True}
    install_stop_signals(stop)
    started = time.time()
    # The exit save is skipped when the last update already saved: a second
    # write of the same checkpoint would put a duplicate point in progress.jsonl.
    saved_at_update = trainer.updates
    # The update the open tally window began after: a resumed run's first window
    # is cut short by wherever the count stood, and is labelled as what it is.
    window_base = trainer.updates
    # The first window of a start is not kept as a reward figure. Every arena
    # begins a round at the start, so the window is paid for rounds that are
    # still being played when it closes and divides by the few that ended: it
    # reads several times too high. From the second window on, the rounds open
    # at one end stand against the rounds open at the other.
    settled = False
    # How the loop ended, for status.json: "stopped" by a signal unless the
    # budget says otherwise. An exception leaves without it, which is how a
    # reader tells a clean end from a crash.
    ended = "stopped"
    try:
        while not stop["requested"]:
            # Its parent, not any process with that pid: a child keeps its parent
            # across start_new_session, and is handed to init or a subreaper the
            # moment the parent dies, so a recycled pid is never mistaken for it.
            if args.stop_with_parent is not None and os.getppid() != args.stop_with_parent:
                logger.info("The app that started this training has closed; stopping")
                ended = "the app closed"
                break
            reason = budget_spent(trainer, args, started)
            if reason:
                logger.info(f"{reason}; stopping")
                ended = reason
                break
            clock = trainer_module.curriculum_clock(trainer)
            phase = curriculum.phase_for(clock)
            index = curriculum.phase_index(clock)
            if index != trainer.phase_idx:
                trainer.phase_idx = index
                trainer_module.apply_phase(trainer, phase)

            hits_before = learner_hits(trainer, telemetry)
            extra = trainer_module.collect_rollout(trainer, phase)
            stats = skip_learning(trainer) if args.play_only else trainer_module.learn(trainer)
            trainer_module.log_progress(trainer, stats, extra)
            # Before the tally window is dropped below, which would take the
            # hits of this update with it.
            if not args.play_only:
                check_health(telemetry, trainer, stats, learner_hits(trainer, telemetry) - hits_before)
            stop["abortable"] = False
            if trainer.updates % LOG_WINDOW_UPDATES == 0:
                tally = trainer.tally
                trainer_module.drop_tally(trainer)
                bank_tally(telemetry, tally)
                # Nothing is credited without learn(), so there is no figure to keep.
                if settled and not args.play_only:
                    note_reward_window(telemetry, tally, window_base + 1, trainer.updates)
                settled = True
                window_base = trainer.updates
            if not args.play_only and trainer.updates % args.checkpoint_every == 0:
                save(run, checkpoints, trainer, telemetry)
                saved_at_update = trainer.updates
            stop["abortable"] = True
            write_status(run, trainer, telemetry, header)
    except KeyboardInterrupt:
        logger.info("Interrupted again; saving now")
    stop["abortable"] = False
    if not args.play_only and trainer.updates != saved_at_update:
        save(run, checkpoints, trainer, telemetry)
    header["stopped_at"] = time.time()
    header["stop_reason"] = ended
    write_status(run, trainer, telemetry, header, force=True)
    logger.info(f"Stopped at {trainer.steps_done / 1e6:.2f}M steps, update {trainer.updates}")


def choose_device(args: argparse.Namespace) -> Any:
    device = policy.resolve_device(args.device or "auto")
    # A bare "cuda" carries no index, and set_per_process_memory_fraction refuses
    # a device without one (ValueError, before the Trainer is ever built). Pinned
    # here and returned, so the cap, the Trainer and status.json name one card.
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cuda" and args.gpu_memory is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory, device.index)
    elif args.gpu_memory is not None:
        logger.warning(f"--gpu-memory ignored on {device}")
    return device


def build_header(args: argparse.Namespace, device: Any, learners: List[str], opponents: List[str],
                 teams: Dict[str, Any], reward_defaults: Dict[str, float],
                 reward_sources: Dict[str, str]) -> Dict[str, Any]:
    """The part of status.json that does not move during a run.

    `rewards` is every weight in force. `reward_defaults` is what each would be
    with nothing given, and `reward_sources` names, for the ones this start was
    given, where from: the --rewards path as it was typed, or "--reward".
    `pid` and `args` say which process wrote the file and what it was started
    with, so sdk.tui can show what is in force without re-parsing a command line.
    """
    playing = learners + [team for team in opponents if team not in learners]
    return {
        "version": STATUS_VERSION,
        "started_at": time.time(),
        "device": str(device),
        "arenas": args.arenas,
        "learners": learners,
        "opponents": opponents,
        "teams": {team: {"colour": teams[team].get("colour"), "members": list(teams[team]["members"])}
                  for team in playing},
        "rewards": reward_weights(),
        "reward_defaults": reward_defaults,
        "reward_sources": reward_sources,
        "play_only": args.play_only,
        "pid": os.getpid(),
        "args": vars(args),
        "environment": environment_info(args.threads),
    }


def sdk_commit() -> Optional[str]:
    """The short commit of this checkout, or None when it is not a git checkout or git is missing.

    Asked of the SDK's own directory, and only when that directory is itself
    the repository: a copy unpacked inside some other repository would
    otherwise be stamped with that one's commit.
    """
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.exists(os.path.join(root, ".git")):
        return None
    try:
        done = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def threading_layer() -> Optional[str]:
    """The numba threading layer in use, or None before a parallel kernel has run.

    numba only picks its layer when the first parallel kernel is launched, so
    this is read after kernels.warmup; asked earlier it raises.
    """
    import numba
    try:
        return numba.threading_layer()
    except ValueError:
        return None


def environment_info(threads: int) -> Dict[str, Any]:
    """What a run was trained with, for a result that cannot be reproduced elsewhere.

    The same seed repeats a run only under the same libraries and thread
    count, and the threading layer decides much of its speed, so a status.json
    passed between people says what it was made under.
    """
    import numba
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "numba": numba.__version__, "threading_layer": threading_layer(), "threads": threads,
            "platform": platform.platform(), "commit": sdk_commit()}


def main():
    args = parse_args()
    check_args(args)
    rewards = parse_rewards(args.reward)
    bundle = os.path.realpath(args.bundle)
    run = os.path.realpath(args.run)
    teams_doc, bundle_nicks = read_bundle(bundle)
    teams = teams_doc["teams"]
    check_bundle(teams, bundle_nicks)
    learners = resolve_teams(args.learn, teams)
    # By default every other team, plus the learners themselves only when there
    # are several: one learner never plays itself, and naming it among its own
    # opponents would put "MY TEAM against MY TEAM, ..." in the log and the header.
    opponents = resolve_teams(args.against or [team for team in teams if team not in learners]
                              + (learners if len(learners) > 1 else []), teams)
    check_rivals(learners, opponents, teams)
    playing = learners + [team for team in opponents if team not in learners]

    checkpoints = os.path.join(run, CHECKPOINTS_NAME)
    set_environment(args.threads)
    import_engine()
    torch.set_num_threads(args.threads)
    # The Trainer seeds its own generator, which draws the maps, the lineups and
    # the ends; every action a model samples comes from torch's; and the PPO
    # update shuffles its minibatches with numpy's global one. All three, or
    # --seed replays nothing: two --play-only runs of the same weights would be
    # two different samples, and two trainings would part at the first update.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # The refusals that need the engine come before the run directory is made,
    # so a mistyped flag leaves nothing behind to be resumed by mistake.
    if args.theme is not None and args.theme not in themes.THEME_KEYS:
        refuse(f"--theme {args.theme!r}: the themes are {', '.join(themes.THEME_KEYS)}")
    check_nicks(teams, playing)
    # Read off config before anything is set on it. The file first and --reward
    # over it: the command line is where one knob is changed for one experiment
    # without editing a file that is kept with the team.
    reward_defaults = reward_weights()
    from_file = read_reward_file(args.rewards) if args.rewards else {}
    apply_rewards(dict(from_file, **rewards))
    reward_sources = dict({name: args.rewards for name in from_file},
                          **{name: "--reward" for name in rewards})
    # Asked before the run directory exists: a cuda device torch cannot see
    # would otherwise fail after the lock and the checkpoints were written, with
    # a traceback in place of a refusal.
    if (args.device or "").startswith("cuda") and not torch.cuda.is_available():
        refuse("--device cuda: torch sees no GPU here; use --device cpu or install a CUDA build of torch")
    check_marker(run, checkpoints)
    os.makedirs(run, exist_ok=True)
    # Kept in a local of main for the whole run: closing it would drop the lock.
    run_lock = lock_run(run)
    resumed = prepare_run(run, checkpoints, bundle, learners, args.restart)
    if resumed:
        warn_changed_learners(run, learners)
    check_opponent_files(checkpoints, teams, learners, opponents)
    # The scales off the bundle's full order BEFORE the roster is narrowed: the
    # index a model holds in the bundle's manifest is the one its character was
    # handed out by.
    install_personalities(personality_scales(teams_doc, bundle_nicks))
    team_of_model, members = seat_teams(bundle_nicks, teams, playing)
    index_of = nick_index()
    learner_indices = np.concatenate([members[team] for team in learners])

    telemetry = new_telemetry(args.arenas, playing, team_of_model)
    telemetry.update({"members": members, "learners": learners, "run_base": run_base_steps(run)})
    # Nothing learns in a --play-only run, so its weights pay nobody.
    if not args.play_only:
        log_rewards(reward_defaults, reward_sources, resumed, os.path.join(run, STATUS_NAME))
        warn_timeout_off()
    if resumed:
        resume_telemetry(telemetry, os.path.join(run, STATUS_NAME))

    device = choose_device(args)
    kernels.warmup(TEAM_SIZE)
    trainer = trainer_module.Trainer(args.arenas, device, seed=args.seed, team_size=TEAM_SIZE)
    # The three things the trainer asks its caller for, and the map theme.
    trainer.theme = args.theme
    trainer.learning = [nick for team in learners for nick in teams[team]["members"]]
    trainer.draw_lineups = make_assign_slots(
        telemetry, members, learners,
        {team: [rival for rival in opponents if rival != team] for team in learners})
    trainer.read_clock = full_arena_clock if args.play_only else make_curriculum_clock(learner_indices)
    logger.info(f"Device {device}, {trainer.n_models} models, {args.arenas} arenas, "
                f"{'playing' if args.play_only else 'learning'} {', '.join(learners)} "
                f"against {', '.join(opponents)}, "
                f"maps: {args.theme or 'all themes'}")
    for team in learners:
        scales = ", ".join(f"{nick} {config.personality(index_of[nick])['entropy_scale']:.2f}"
                           for nick in teams[team]["members"])
        logger.info(f"{team} entropy scales: {scales}")
    restore(trainer, checkpoints, device)
    if args.play_only:
        logger.info("--play-only: nothing learns, nothing is saved, and every match is on the full arena")
    else:
        logger.info(f"The curriculum follows the learners' own "
                    f"{trainer_module.curriculum_clock(trainer) / 1e6:.2f}M steps")
        warn_mixed_learners(trainer, members, learners)

    header = build_header(args, device, learners, opponents, teams, reward_defaults, reward_sources)
    trainer.on_step = make_step_hook(run, telemetry, header)
    try:
        run_loop(args, run, checkpoints, trainer, telemetry, header)
    except Exception as exc:
        logger.error(f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
