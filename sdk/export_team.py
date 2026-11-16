#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export one team as a single archive, renamed on the way out if asked.

    python -m sdk.export_team --run runs/my --team "MY TEAM" [--name "NEW NAME"]
                              [--nicks A B C D E] [--colour "#rrggbb"] [--out exports]
    python -m sdk.export_team --bundle bundle --team WHITE

Run from the repository root. With --run the weights are the ones that run has
trained, as it last saved them; with --bundle alone they are the bundle's own.
Nothing in the run or the bundle is written: the archive is built in memory,
read back, and only then put in --out as <TEAM_NAME>_<YYYYMMDDHHMMSS>.tar.gz.

The archive is a gzipped tar with flat members, in the layout a SELFPLAYTV run
archive has: one <nick>.pt per player, manifest.json naming exactly those five,
stats.json with no counters in it, and teams.json describing the team - its
name, its colour, its players in order and the entropy scale each plays with.
Unpacked into an empty directory it is a bundle of one team.

Each .pt carries the whole model: the weights, the optimiser moments, the return
normaliser and the model's own step count. Renaming changes the file names, the
label inside each file, the manifest and teams.json together, so no part of the
archive still says the old name.
"""

import argparse
import io
import json
import logging
import os
import re
import sys
import tarfile
import time
import traceback
from typing import Any, Dict, List, Tuple

# Local imports
from sdk import new_team, train

logger = logging.getLogger(__name__)

ARCHIVE_SUFFIX = ".tar.gz"
# A team name is free text and the archive is named after it, so everything a
# file name cannot be trusted with becomes an underscore.
UNSAFE_IN_FILE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m sdk.export_team",
                                     description="Export a SELFPLAYTV team as one archive")
    parser.add_argument("--run", default=None, help="take the weights this run has trained")
    parser.add_argument("--bundle", default=None,
                        help="the bundle that describes the team (default with --run: the one the "
                             "run was started from); alone, also where the weights are taken from")
    parser.add_argument("--team", required=True, help='the team to export, e.g. "MY TEAM"')
    parser.add_argument("--name", default=None, help="export it under this team name")
    parser.add_argument("--nicks", nargs="+", default=None, metavar="NICK",
                        help="export its five players under these nicknames, in the team's order")
    parser.add_argument("--colour", "--color", default=None,
                        help='export it with this colour, "#rrggbb", in quotes')
    parser.add_argument("--out", default="exports", help="directory the archive is written to")
    return parser.parse_args()


def find_source(args: argparse.Namespace) -> Tuple[str, str]:
    """The directory the weights are read from, and the bundle that describes the team.

    A run does not keep its own teams.json: sdk.train reads the bundle's on
    every start. So the run's marker is asked which bundle that was, unless
    --bundle says the bundle has moved since.
    """
    if not args.run and not args.bundle:
        train.refuse("give --run RUN for the weights a run has trained, or --bundle BUNDLE for a "
                     "bundle's own")
    if not args.run:
        bundle = os.path.realpath(args.bundle)
        checkpoints = bundle
    else:
        run = os.path.realpath(args.run)
        checkpoints = os.path.join(run, train.CHECKPOINTS_NAME)
        marker = os.path.join(run, train.MARKER_NAME)
        saved = os.path.join(checkpoints, train.MANIFEST_NAME)
        if not os.path.isfile(marker) or not os.path.isfile(saved):
            train.refuse(f"{run} is not a run of sdk.train: it has no {train.MARKER_NAME} or no "
                         f"{train.CHECKPOINTS_NAME}/{train.MANIFEST_NAME}")
        bundle = os.path.realpath(args.bundle or str(train.read_json(marker).get("bundle")))
    if not new_team.is_bundle(bundle):
        train.refuse(f"{bundle} holds no {train.TEAMS_NAME} and {train.MANIFEST_NAME}; say where the "
                     f"bundle that describes the team is with --bundle")
    return checkpoints, bundle


def team_colour(args: argparse.Namespace, team: str, entry: Dict[str, Any]) -> str:
    """--colour if given, else the team's own, which has to be a real one to travel."""
    if args.colour is not None:
        return new_team.clean_colour(args.colour)
    colour = entry.get("colour")
    if not isinstance(colour, str) or not new_team.COLOUR_PATTERN.fullmatch(colour):
        train.refuse(f"{team} has the colour {colour!r} in {train.TEAMS_NAME}, which is not #rrggbb; "
                     f"give one with --colour")
    return colour


def check_trained(checkpoints: str, manifest: Dict[str, Any], team: str, members: List[str]) -> None:
    """Every member has a checkpoint in the source, and the source's manifest names it.

    A run's manifest names the models that played in it. A team that did not
    has only the files the run copied from the bundle on its first start, and
    exporting those as this run's work would be a lie.
    """
    absent = [nick for nick in members
              if nick not in (manifest.get("nicks") or [])
              or not os.path.isfile(os.path.join(checkpoints, f"{nick}.pt"))]
    if absent:
        train.refuse(f"{team}: {', '.join(absent)} {'is' if len(absent) == 1 else 'are'} not among "
                     f"the models saved in {checkpoints}; a team that did not play in a run is "
                     f"exported from its bundle, with --bundle and no --run")


def check_names(args: argparse.Namespace, team: str, name: str, nicks: List[str],
                teams: Dict[str, Any]) -> None:
    """Refuse a new name or nickname that belongs to another team of the bundle.

    The rule sdk.new_team makes a team under, on the way out: an archive
    called WHITE TEAM that does not hold WHITE TEAM, or one whose Hiroshi.pt
    is somebody else's network, would be taken for the real one wherever it
    is unpacked. The team's own name and its own players' nicknames stay free
    to keep or to hand round.
    """
    others = {other: entry for other, entry in teams.items() if other != team}
    clash = new_team.name_clash(name, others) if args.name is not None else ""
    if clash:
        train.refuse(f"--name {name!r} would be taken for the team {clash!r} of the same bundle (case "
                     f"does not count, and a name is also read without its TEAM); choose another")
    owner = {nick.casefold(): other for other, entry in others.items() for nick in entry["members"]}
    for nick in nicks if args.nicks else []:
        if nick.casefold() in owner:
            train.refuse(f"--nicks {nick!r}: {owner[nick.casefold()]} of the same bundle has a player "
                         f"of that name (case does not count: a nickname is a file name)")


def relabel(checkpoints: str, members: List[str], nicks: List[str]) -> Dict[str, Dict[str, Any]]:
    """Each member's checkpoint, whole, by the nickname it leaves under.

    Read and written again even when nothing is renamed: one path for every
    export, and what comes out is on the CPU whatever device saved it.
    """
    blobs = {}
    for member, nick in zip(members, nicks):
        blob = new_team.read_checkpoint(train.persist.checkpoint_path(checkpoints, member))
        blob["nick"] = nick
        blobs[nick] = blob
    return blobs


def read_save(args: argparse.Namespace, checkpoints: str, manifest: Dict[str, Any],
              members: List[str], nicks: List[str]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """The manifest and the team's checkpoints as ONE save of the run, or a refusal.

    A training run rewrites its .pt files one by one and its manifest last,
    and nothing locks an export out while it does. Read in that window, the
    files are of the new save and the manifest of the one before, and the
    archive would state a step count its checkpoints do not have. Every file a
    run saves carries the run's step counter, so they are compared; one more
    read, a moment later, is past any save that was in progress. Only for a
    run: a bundle is not being written, and its files may honestly come from
    different saves.
    """
    blobs = relabel(checkpoints, members, nicks)
    if args.run and any(blob.get("steps") != manifest.get("steps") for blob in blobs.values()):
        time.sleep(1.0)
        manifest = train.read_json(os.path.join(checkpoints, train.MANIFEST_NAME))
        blobs = relabel(checkpoints, members, nicks)
        if any(blob.get("steps") != manifest.get("steps") for blob in blobs.values()):
            train.refuse(f"The checkpoints in {checkpoints} are not all of the save its manifest "
                         f"describes. If the run is training, export again or stop it first; if it "
                         f"has never saved, export the team from its bundle, with --bundle and no --run")
    return manifest, blobs


def add_member(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    """One flat member, described by hand and not read off a file on disk.

    A header taken from a file carries its owner's user and group names, and
    an archive that is handed to somebody else has no business saying who is
    logged in on the machine that made it.
    """
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = int(time.time())
    archive.addfile(info, io.BytesIO(data))


def build_archive(blobs: Dict[str, Dict[str, Any]], manifest: Dict[str, Any],
                  teams_doc: Dict[str, Any]) -> bytes:
    """The archive's bytes: the checkpoints by file name, the manifest, the stats, the team.

    The first three in the order and the form a run archive has them;
    teams.json comes last.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in sorted(f"{nick}.pt" for nick in blobs):
            data = io.BytesIO()
            train.torch.save(blobs[name[:-len(".pt")]], data)
            add_member(archive, name, data.getvalue())
        add_member(archive, train.persist.MANIFEST_NAME, json.dumps(manifest, indent=2).encode())
        # No counters: wins and points belong to the matches they were scored
        # in, and whoever receives the team starts its record from nothing.
        stats = {"format": train.persist.FORMAT_VERSION, "exported_at": time.time(),
                 "manifest": manifest, "stats": {}}
        add_member(archive, train.persist.STATS_NAME, json.dumps(stats, indent=2).encode())
        add_member(archive, train.TEAMS_NAME, json.dumps(teams_doc, indent=2).encode())
    return buffer.getvalue()


def check_archive(data: bytes, blobs: Dict[str, Dict[str, Any]]) -> None:
    """Raise unless every checkpoint in the finished archive reads back as the one put in.

    Out of the archive's own bytes and not out of what was about to be packed:
    the archive is what travels. Read with weights_only, as whoever receives it
    should, so a file that only loads the permissive way is caught here.
    """
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for nick, blob in blobs.items():
            packed = new_team.read_checkpoint(io.BytesIO(archive.extractfile(f"{nick}.pt").read()))
            new_team.check_checkpoint(packed, nick)
            for key, tensor in blob["state_dict"].items():
                if not train.torch.equal(packed["state_dict"][key], tensor):
                    raise ValueError(f"{nick}.pt in the archive does not hold the weights it was "
                                     f"given ({key})")


def write_archive(out: str, name: str, data: bytes) -> str:
    """Put the archive in `out` under the team's name and the time, and return its path."""
    stem = UNSAFE_IN_FILE_NAME.sub("_", name).strip("_") or "team"
    path = os.path.join(out, f"{stem}_{time.strftime('%Y%m%d%H%M%S')}{ARCHIVE_SUFFIX}")
    # Two exports of one team inside the same second: the first one stays.
    if os.path.exists(path):
        train.refuse(f"{path} already exists; nothing is written over, run the export again")
    train.persist.write_atomically(path, lambda handle: handle.write(data))
    return path


def log_summary(args: argparse.Namespace, team: str, name: str, colour: str, members: List[str],
                blobs: Dict[str, Dict[str, Any]], manifest: Dict[str, Any], checkpoints: str,
                path: str) -> None:
    """Five lines: the team, the players, their steps, where it came from, the file."""
    nicks = list(blobs)
    logger.info(f"Team: {name} ({colour})" + (f", exported from {team}" if name != team else ""))
    logger.info("Players: " + ", ".join(nick if nick == member else f"{nick} (was {member})"
                                        for member, nick in zip(members, nicks)))
    logger.info("Steps per model: " + ", ".join(
        f"{nick} {(blobs[nick].get('model_steps') or 0) / 1e6:.2f}M" for nick in nicks))
    logger.info(f"Weights: {'the run in' if args.run else 'the bundle'} {checkpoints}, saved at "
                f"{int(manifest.get('steps', 0)) / 1e6:.2f}M steps, update {manifest.get('updates', 0)}")
    logger.info(f"Archive: {path}, {os.path.getsize(path) / 1e6:.1f} MB")


def main():
    args = parse_args()
    checkpoints, bundle = find_source(args)
    teams_doc, bundle_nicks = train.read_bundle(bundle)
    teams = teams_doc["teams"]
    train.check_bundle(teams, bundle_nicks)
    team = train.resolve_team(args.team, teams)
    members = list(teams[team]["members"])
    name = new_team.clean_team_name(args.name) if args.name is not None else team
    colour = team_colour(args, team, teams[team])
    nicks = list(args.nicks or members)

    # The thread count is of no use here; set because the engine is imported
    # only after it has been.
    train.set_environment(1)
    train.import_engine()
    new_team.check_new_nicks(nicks)
    check_names(args, team, name, nicks, teams)
    manifest = train.read_json(os.path.join(checkpoints, train.MANIFEST_NAME))
    check_trained(checkpoints, manifest, team, members)
    scales = new_team.entropy_scales(teams_doc, bundle_nicks)
    team_doc = {"version": train.TEAMS_VERSION,
                "teams": {name: {"colour": colour, "members": nicks}},
                "personality": {nick: scales[member] for member, nick in zip(members, nicks)}}
    try:
        manifest, blobs = read_save(args, checkpoints, manifest, members, nicks)
        manifest["nicks"] = nicks
        data = build_archive(blobs, manifest, team_doc)
        check_archive(data, blobs)
        path = write_archive(os.path.realpath(args.out), name, data)
    except Exception as exc:
        logger.error(f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}")
        sys.exit(1)
    log_summary(args, team, name, colour, members, blobs, manifest, checkpoints, path)
    # The path alone on standard output, the log being on standard error, so a
    # script can take it: ARCHIVE=$(python -m sdk.export_team ...).
    sys.stdout.write(path + "\n")


if __name__ == "__main__":
    main()
