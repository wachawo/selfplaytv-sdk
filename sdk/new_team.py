#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create your own team of five models, in a bundle of your own.

    python -m sdk.new_team --name "MY TEAM" --colour "#12b5a6" --nicks Ada Bob Cy Dee Eve
                           [--from WHITE | --fresh] [--bundle bundle] [--out my-bundle]

Run from the repository root. The bundle the repository ships is only ever read.
The first call copies it to --out and adds the team there; every later call adds
another team to the same --out, which is how a team gets a sparring partner.

A team starts in one of two ways, and they are two different trainings:

- --from TEAM (the default, from WHITE) copies that team's five networks under
  the new nicknames, with everything that makes them veterans: the optimiser
  moments, the return normaliser and each model's own step count. The step
  count is the curriculum clock, so the copy trains on the full arena with
  small, careful updates. It is the short road: change what the team is paid
  for and train from there.
- --fresh gives five untrained networks with a step count of zero. They start
  on the small close-quarters arena, with large updates, and have the whole
  curriculum ahead of them.

Nothing existing is ever written over: a team name, a nickname or a checkpoint
file that is already in the bundle is refused. The new team's files are built in
a staging directory first and the team is named last, so a call that fails half
way leaves the bundle as it was.
"""

import argparse
import logging
import os
import re
import shlex
import shutil
import sys
import tempfile
import traceback
from typing import Any, Callable, Dict, List, Optional

# Local imports
from sdk import train

logger = logging.getLogger(__name__)

# The bundle this repository ships, found by where this file is and not by the
# working directory: it is the one directory this command must never write,
# whatever --out says and wherever the command is run from.
SHIPPED_BUNDLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bundle")
TEAM_NAME_MAX = 64
COLOUR_PATTERN = re.compile(r"#[0-9a-f]{6}")
# A run directory is suggested by the team's name, and everything in the name
# that a path is better off without becomes a hyphen.
UNSAFE_IN_RUN_NAME = re.compile(r"[^a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m sdk.new_team",
                                     description="Create your own SELFPLAYTV team of five models")
    parser.add_argument("--name", required=True, help='the team\'s name, e.g. "MY TEAM"')
    parser.add_argument("--colour", "--color", required=True,
                        help='the team\'s colour as "#rrggbb", in quotes: an unquoted # starts a '
                             'comment in the shell')
    parser.add_argument("--nicks", nargs="+", required=True, metavar="NICK",
                        help="five nicknames: letters, digits, hyphens and underscores")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--from", dest="source", default="WHITE", metavar="TEAM",
                       help="copy this team's networks (default: WHITE)")
    start.add_argument("--fresh", action="store_true", help="start from untrained networks instead")
    parser.add_argument("--bundle", default="bundle",
                        help="the bundle --out is copied from when it does not exist yet")
    parser.add_argument("--out", default="my-bundle",
                        help="your bundle: created on the first call, added to on later ones")
    return parser.parse_args()


def clean_team_name(value: str) -> str:
    """The name as it will be stored, refused if it could not be read back off a screen."""
    name = value.strip()
    # isprintable is false for control characters and for the zero-width ones,
    # so a name that would split a log line in two and a name that draws nothing
    # are both stopped here.
    if not name or not name.isprintable():
        train.refuse(f"--name {value!r}: a team name is visible text on one line")
    if len(name) > TEAM_NAME_MAX:
        train.refuse(f"--name: a team name is at most {TEAM_NAME_MAX} characters, this one is {len(name)}")
    # Every command takes a team's name as an argument, and one that starts
    # with a hyphen is read there as an option that does not exist.
    if name.startswith("-"):
        train.refuse(f"--name {value!r}: a team name cannot start with a hyphen")
    return name


def clean_colour(value: str) -> str:
    """The colour in lower case, refused unless it is exactly #rrggbb.

    Nothing looser: whatever shows the team paints the colour its own way, and
    "#abc" or "teal" means something to one reader and nothing to the next.
    """
    colour = value.strip().lower()
    if not COLOUR_PATTERN.fullmatch(colour):
        train.refuse(f"--colour {value!r}: expected #rrggbb, six hexadecimal digits after the #")
    return colour


def name_clash(name: str, teams: Dict[str, Any]) -> str:
    """The existing team that `name` would be mistaken for, or ''.

    Read the way train.resolve_team reads a name: without case, and with WHITE
    meaning WHITE TEAM. A team called "white" beside WHITE TEAM could never be
    named on a command line, because the older one would answer first.
    """
    wanted = name.upper()
    for team in teams:
        held = team.upper()
        if wanted in (held, held + " TEAM") or held == wanted + " TEAM":
            return team
    return ""


def check_new_nicks(nicks: List[str]) -> None:
    """Five nicknames the engine accepts, no two of them alike. Needs the engine imported."""
    if len(nicks) != train.TEAM_SIZE:
        train.refuse(f"--nicks names {len(nicks)} models and a team fields exactly {train.TEAM_SIZE}")
    seen = set()
    for nick in nicks:
        problem = train.config.nick_problem(nick)
        if problem:
            train.refuse(f"--nicks {nick!r}: {problem}")
        if nick.casefold() in seen:
            train.refuse(f"--nicks {nick!r} is given twice (case does not count: a nickname is a "
                         f"file name)")
        seen.add(nick.casefold())


def check_free(nicks: List[str], directory: str, teams: Dict[str, Any],
               bundle_nicks: List[str]) -> None:
    """Refuse a nickname the bundle already holds, as a team's member, a model or a file.

    Without regard to case, because the nickname is the checkpoint's file name
    and that is how macOS and Windows compare file names: there ada.pt written
    beside Ada.pt is the end of Ada.
    """
    owner = {nick.casefold(): team for team, entry in teams.items() for nick in entry["members"]}
    listed = {nick.casefold() for nick in bundle_nicks}
    files = {entry[:-len(".pt")].casefold(): entry for entry in os.listdir(directory)
             if entry.endswith(".pt")}
    for nick in nicks:
        folded = nick.casefold()
        if folded in owner:
            train.refuse(f"--nicks {nick!r}: {owner[folded]} already has a player of that name (case "
                         f"does not count: a nickname is a file name)")
        if folded in listed or folded in files:
            # In the manifest or on disk and in no team: what a call killed half
            # way leaves behind. Said in full, because "already taken" would send
            # its owner looking for a team that does not exist.
            train.refuse(f"--nicks {nick!r}: {directory} already holds a model of that name that no "
                         f"team owns ({files.get(folded, train.MANIFEST_NAME)}); choose another "
                         f"nickname")


def is_bundle(directory: str) -> bool:
    return all(os.path.isfile(os.path.join(directory, name))
               for name in (train.TEAMS_NAME, train.MANIFEST_NAME))


def check_out(out: str, bundle: str) -> None:
    """--out is somewhere this command may write: never a bundle it only reads, nor inside one.

    --bundle is only read while --out is still to be made from it. Once --out
    exists the call adds to it and --bundle is not looked at, so naming one's
    own bundle there, as every other command has it named, is no reason to
    refuse.
    """
    kept_bundles = [os.path.realpath(SHIPPED_BUNDLE)]
    if not os.path.exists(out):
        kept_bundles.append(bundle)
    for kept in kept_bundles:
        if out == kept or out.startswith(kept + os.sep):
            train.refuse(f"--out {out} is the bundle {kept} or is inside it, and that bundle is "
                         f"only read; name a directory of your own")
    if os.path.exists(out) and not is_bundle(out):
        train.refuse(f"--out {out} exists and is not a bundle (it holds no {train.TEAMS_NAME} and "
                     f"{train.MANIFEST_NAME}); remove it or give another --out")


def lock_bundle(directory: str) -> Any:
    """Wait for any other new_team on this bundle to finish, and hold it until this one has.

    Two calls at once would each read the manifest and the team list, add a
    team, and write them back; the second write would drop the first one's
    team and leave its five checkpoints named by nothing. Waiting and not
    refusing, so two calls started together both succeed, one after the other.
    The returned handle is kept open by the caller: the lock goes when the
    process does.
    """
    try:
        import fcntl
    except ImportError:
        return None
    handle = os.open(directory, os.O_RDONLY)
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def observation_size() -> int:
    # Imported here and not at the top for the reason in sdk/__init__.py: the
    # engine is imported only after train.set_environment has run.
    from engine.env import obs
    return obs.observation_size()


def read_checkpoint(source: Any) -> Dict[str, Any]:
    """One checkpoint, from a path or an open file, read the way a run reads it.

    weights_only, as everywhere in this package: a checkpoint is a pickle, and
    one from a stranger must not be able to run code. Onto the CPU, so what is
    written back loads on a machine with no GPU.
    """
    return train.torch.load(source, map_location="cpu", weights_only=True)


def check_checkpoint(blob: Dict[str, Any], nick: str) -> None:
    """Raise unless `blob` is a network this build can field, labelled `nick`.

    Loaded strictly into a network of the width the file itself names, which is
    what a run does with it: a file that fails here would fail every later
    restore, and there the error would be about the whole run.
    """
    if blob.get("nick") != nick:
        raise ValueError(f"the checkpoint written for {nick} is labelled {blob.get('nick')!r}")
    train.policy.ActorCritic(blob["obs_size"], blob["hidden"]).load_state_dict(blob["state_dict"])


def write_checkpoint(directory: str, nick: str, blob: Dict[str, Any]) -> None:
    train.persist.write_atomically(train.persist.checkpoint_path(directory, nick),
                                   lambda handle: train.torch.save(blob, handle))


def entropy_scales(teams_doc: Dict[str, Any], bundle_nicks: List[str]) -> Dict[str, float]:
    """Every model's entropy scale, by nickname: the bundle's table, with nobody left out.

    A model the table does not list gets the scale of its place in the
    manifest, which is what a bundle with no table at all goes by.
    """
    scales = train.personality_scales(teams_doc, bundle_nicks)
    for index, nick in enumerate(bundle_nicks):
        scales.setdefault(nick, float(train.config.personality(index)["entropy_scale"]))
    return scales


def stage_copies(staging: str, directory: str, members: List[str], nicks: List[str]) -> None:
    """Each source member's checkpoint, whole, under its new nickname.

    Whole: the optimiser moments, the normaliser and model_steps travel with
    the weights, so the copy carries on where the source stood and does not
    relearn its step sizes. Only the label inside the file changes, so that it
    names the model the file now belongs to.
    """
    for member, nick in zip(members, nicks):
        blob = read_checkpoint(train.persist.checkpoint_path(directory, member))
        blob["nick"] = nick
        write_checkpoint(staging, nick, blob)


def stage_fresh(staging: str, nicks: List[str], manifest: Dict[str, Any]) -> None:
    """Five untrained checkpoints, built the way a run builds a model it has no file for.

    The networks, optimisers and normalisers are taken from a Trainer, and
    written by the function a run saves with, so nothing about a new model is
    decided a second time here. The Trainer refuses a roster that cannot fill
    a field, so it is given two teams' worth of names and one team is kept.

    model_steps is written as 0 on purpose. A file without the figure is
    credited a share of the run's steps when it is restored, and a team that is
    credited steps it never took starts on the full arena with its learning
    rate already annealed to the floor.
    """
    count = len(nicks)
    train.config.set_roster(list(nicks) * 2)
    trainer = train.trainer_module.Trainer(1, train.torch.device("cpu"), team_size=count)
    train.persist.save_run(staging, trainer.nets[:count], trainer.optimizers[:count],
                           trainer.normalizers[:count], list(nicks), trainer.obs_size,
                           int(manifest.get("steps", 0)), int(manifest.get("updates", 0)),
                           trainer.generation, team_size=count, steps_by_model=[0] * count)


def commit_team(directory: str, staging: str, teams_doc: Dict[str, Any], manifest: Dict[str, Any],
                name: str, entry: Dict[str, Any], scales: Dict[str, float]) -> None:
    """Move the staged checkpoints in and name the team, or put the bundle back as it was.

    The files first, then the manifest, then teams.json: the team is named
    last, so until the final write the bundle is the one it was, with files in
    it that nothing reads yet. The nicknames are appended to the manifest, so no
    existing model's place in it moves.
    """
    nicks = entry["members"]
    listed = list(manifest["nicks"])
    manifest_file = os.path.join(directory, train.MANIFEST_NAME)
    moved = []
    try:
        for nick in nicks:
            target = train.persist.checkpoint_path(directory, nick)
            os.replace(train.persist.checkpoint_path(staging, nick), target)
            moved.append(target)
        train.persist.write_json(manifest_file, dict(manifest, nicks=listed + nicks))
        moved.append(manifest_file)
        teams_doc["teams"][name] = entry
        teams_doc["personality"] = scales
        train.persist.write_json(os.path.join(directory, train.TEAMS_NAME), teams_doc)
    except BaseException:
        if manifest_file in moved:
            moved.remove(manifest_file)
            train.persist.write_json(manifest_file, manifest)
        for target in moved:
            os.remove(target)
        raise


def add_team(directory: str, teams_doc: Dict[str, Any], bundle_nicks: List[str], name: str,
             colour: str, nicks: List[str], source: Optional[str]) -> List[float]:
    """Add the team to the bundle in `directory`; returns the entropy scales its five were given.

    `source` is the team whose networks are copied, or None for fresh ones. The
    new files are written and read back in a staging directory inside the
    bundle - the same file system, so moving them in is a rename - and only a
    team whose five files all passed is committed.
    """
    manifest = train.read_json(os.path.join(directory, train.MANIFEST_NAME))
    # Every existing model's scale is written down with the new ones. A bundle
    # with no table hands scales out by manifest order, and a table that listed
    # only the new team would send the others to a different fallback.
    scales = entropy_scales(teams_doc, bundle_nicks)
    if source:
        new_scales = [scales[member] for member in teams_doc["teams"][source]["members"]]
    else:
        # A fresh model has no character to inherit, so it gets the one of the
        # place it takes in the manifest, as a run would hand it out.
        new_scales = [float(train.config.personality(len(bundle_nicks) + offset)["entropy_scale"])
                      for offset in range(len(nicks))]
    scales.update(zip(nicks, new_scales))
    staging = tempfile.mkdtemp(dir=directory, suffix=".tmp")
    try:
        if source:
            stage_copies(staging, directory, teams_doc["teams"][source]["members"], nicks)
        else:
            stage_fresh(staging, nicks, manifest)
        for nick in nicks:
            check_checkpoint(read_checkpoint(train.persist.checkpoint_path(staging, nick)), nick)
        commit_team(directory, staging, teams_doc, manifest, name,
                    {"colour": colour, "members": list(nicks)}, scales)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return new_scales


def copy_bundle(bundle: str, target: str) -> None:
    """A working copy of `bundle` in the new directory `target`: both JSON files and every .pt."""
    os.makedirs(os.path.dirname(target), exist_ok=True)
    os.mkdir(target)
    train.copy_checkpoints(bundle, target, missing_only=False)
    for name in (train.MANIFEST_NAME, train.TEAMS_NAME):
        shutil.copy2(os.path.join(bundle, name), os.path.join(target, name))


def create_bundle(bundle: str, out: str, add: Callable[[str], List[float]]) -> List[float]:
    """Build --out beside where it will live, team included, and rename it into place.

    So --out either does not exist or is a whole bundle with the team in it: a
    copy cut short is never left under the name the next command would read.
    """
    building = f"{out}.{os.getpid()}.tmp"
    try:
        copy_bundle(bundle, building)
        new_scales = add(building)
        try:
            os.rename(building, out)
        except OSError:
            train.refuse(f"{out} appeared while it was being built; run the command again to add "
                         f"the team to it")
    finally:
        shutil.rmtree(building, ignore_errors=True)
    return new_scales


def check_team(args: argparse.Namespace, name: str, base: str, teams: Dict[str, Any],
               bundle_nicks: List[str]) -> Optional[str]:
    """Every refusal that needs the bundle read; returns the --from team, or None for --fresh.

    The nickname rules are the engine's, so this runs after import_engine.
    """
    clash = name_clash(name, teams)
    if clash:
        train.refuse(f"--name {name!r} would be taken for the existing team {clash!r} (case does not "
                     f"count, and a name is also read without its TEAM); choose another")
    check_new_nicks(args.nicks)
    check_free(args.nicks, base, teams, bundle_nicks)
    # A fresh network is built at this build's observation width, and one of
    # the wrong width would fail every later restore of the bundle.
    reason = train.persist.layout_mismatch(train.read_json(os.path.join(base, train.MANIFEST_NAME)),
                                           observation_size())
    if reason:
        train.refuse(f"{base} does not fit this build: {reason}")
    if args.fresh:
        return None
    source = train.resolve_team(args.source, teams)
    missing = [nick for nick in teams[source]["members"]
               if not os.path.isfile(os.path.join(base, f"{nick}.pt"))]
    if missing:
        train.refuse(f"--from {source}: {', '.join(missing)} "
                     f"{'has' if len(missing) == 1 else 'have'} no checkpoint in {base}")
    return source


def run_name(name: str) -> str:
    """A run directory's name for a team: MY TEAM is my, NEON FOXES is neon-foxes."""
    words = name.lower()
    if words.endswith(" team"):
        words = words[:-len(" team")]
    return UNSAFE_IN_RUN_NAME.sub("-", words).strip("-") or "team"


def train_hint(out: str, name: str, teams: Dict[str, Any]) -> str:
    """The command that trains the new team, quoted so that it can be pasted into a shell.

    A run directory named after the team, because a second team sent into the
    first one's run would carry on that run's counters and tables. And the
    opponents by name, every other team the bundle holds now: left to the
    default they would be joined by any team added later.
    """
    rivals = " ".join(shlex.quote(team) for team in teams if team != name)
    return (f"python -m sdk.train --bundle {shlex.quote(out)} --run runs/{run_name(name)} "
            f"--learn {shlex.quote(name)} --against {rivals}")


def main():
    args = parse_args()
    name = clean_team_name(args.name)
    colour = clean_colour(args.colour)
    out = os.path.realpath(args.out)
    bundle = os.path.realpath(args.bundle)
    check_out(out, bundle)
    extending = os.path.exists(out)
    base = out if extending else bundle
    if not is_bundle(base):
        train.refuse(f"{base} holds no {train.TEAMS_NAME} and {train.MANIFEST_NAME}; --bundle is the "
                     f"bundle a new --out is copied from")
    # Taken before the bundle is read, and kept in a local of main until the
    # process ends: what is read below is what this call adds to.
    bundle_lock = lock_bundle(out) if extending else None
    teams_doc, bundle_nicks = train.read_bundle(base)
    train.check_bundle(teams_doc["teams"], bundle_nicks)
    # The thread count is of no use here, nothing below runs the simulation;
    # set all the same, because the engine is imported only after it has been.
    train.set_environment(1)
    train.import_engine()
    source = check_team(args, name, base, teams_doc["teams"], bundle_nicks)

    def add(directory: str) -> List[float]:
        return add_team(directory, teams_doc, bundle_nicks, name, colour, list(args.nicks), source)

    try:
        new_scales = add(out) if extending else create_bundle(bundle, out, add)
    except Exception as exc:
        logger.error(f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}")
        sys.exit(1)
    if not extending:
        logger.info(f"New bundle {out}, a copy of {bundle}")
    logger.info(f"Added {name} ({colour}) to {out}: {', '.join(args.nicks)}, "
                f"{f'copies of {source}' if source else 'fresh networks'}")
    logger.info(f"{name} entropy scales: "
                f"{', '.join(f'{nick} {scale:.2f}' for nick, scale in zip(args.nicks, new_scales))}")
    logger.info(f"Train it with: {train_hint(args.out, name, teams_doc['teams'])}")


if __name__ == "__main__":
    main()
