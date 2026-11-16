#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Saving and restoring a training run.

A checkpoint holds everything needed to carry on rather than start again:
network weights, the optimiser moments, the return normaliser and the step
counters. One file per model, named by its nickname, and one manifest beside
them that says what the directory holds and which observation layout the
networks were built for.
"""

import json
import logging
import os
import tempfile
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

import torch

# Local imports
from engine import config

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
# The member a run archive keeps beside the manifest and the .pt files. Named
# here so whatever packs an archive and whatever unpacks one spell it the same
# way.
STATS_NAME = "stats.json"
FORMAT_VERSION = 1


def layout_mismatch(manifest: Dict[str, Any], obs_size: int) -> Optional[str]:
    """Why this manifest cannot be loaded by the current build, or None if it can.

    Compatibility is decided by the observation layout and not by the team
    size: the entity slots are always sized for MAX_TEAM_SIZE. A manifest that
    carries no max team size is judged on its observation width alone.
    """
    if manifest.get("obs_size") != obs_size:
        return (f"observation width {manifest.get('obs_size')} does not match this build's {obs_size}")
    saved_max = manifest.get("max_team_size")
    if saved_max is not None and int(saved_max) != config.MAX_TEAM_SIZE:
        return (f"entity slots are laid out for {saved_max} a side, this build uses "
                f"{config.MAX_TEAM_SIZE}")
    return None


def checkpoint_path(directory: str, nick: str) -> str:
    """Where one model's checkpoint lives. The nickname IS the file name.

    Which is why the name is checked here and not only where somebody types it.
    This is the one funnel every read and every write of a checkpoint passes
    through, and the names reaching it do not all come from a command line that
    checked them: a bundle from somebody else names its own models in its own
    teams.json. A nick of "../../something" would otherwise be joined straight
    onto the directory and written through.

    Raising rather than sanitising. A name quietly turned into a different name
    is a checkpoint saved where nothing will look for it, and the run goes on
    looking healthy until the next restart cannot find it.
    """
    problem = config.nick_problem(nick)
    if problem:
        raise ValueError(f"{nick!r} cannot name a checkpoint: {problem}")
    return os.path.join(directory, f"{nick}.pt")


def manifest_path(directory: str) -> str:
    return os.path.join(directory, MANIFEST_NAME)


def write_atomically(path: str, write: Callable[[Any], None]) -> None:
    """Stage `write` beside `path` under a private name, then move it into place.

    Every writer gets a name of its own. A staging name derived from the
    destination -- `path + ".tmp"` -- is the same string for every writer of the
    same file, and two processes writing one checkpoint then write one staging
    file between them and each move it into place. What that leaves is an
    unreadable checkpoint, a FileNotFoundError because one side's os.replace
    consumed the file the other was still writing, or, silently, the loser's
    bytes.

    The ".tmp" suffix is kept so nothing scanning this directory for ".pt" can
    pick up a half-written file.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("wb", dir=directory, delete=False, suffix=".tmp")
    try:
        with handle:
            write(handle)
        # NamedTemporaryFile creates at 0600 because it assumes a secret. A
        # checkpoint is not one, and whoever reads it need not share this
        # process's uid. Keep whatever mode the file already had, or the 0644 a
        # plain open() produces under the usual umask.
        mode = os.stat(path).st_mode & 0o777 if os.path.exists(path) else 0o644
        os.chmod(handle.name, mode)
        os.replace(handle.name, path)
    except BaseException:
        # A unique staging name is never reused, so one left behind by a failed
        # write would sit in the directory for good.
        if os.path.exists(handle.name):
            os.unlink(handle.name)
        raise


def write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write JSON atomically so a reader never sees a half-written file."""
    blob = json.dumps(payload, indent=2).encode()
    write_atomically(path, lambda handle: handle.write(blob))


def save_run(directory: str, nets: List[Any], optimizers: List[Any], normalizers: List[Any],
             nicks: List[str], obs_size: int, steps: int, updates: int, generation: int,
             team_size: Optional[int] = None, extra: Optional[Dict[str, Any]] = None,
             steps_by_model: Optional[List[int]] = None) -> None:
    """Persist every model plus the counters needed to resume the run.

    `steps_by_model` is each model's OWN learning steps, written into its own
    checkpoint. `steps` beside it is the whole run's, the same number in every
    file, and the two answer different questions: the run's length, and how much
    of it reached this model. They come apart the moment a model is frozen, which
    is the case the per-model figure exists for -- see the trainer's learn().

    Optional so a caller that has no per-model figures still writes a valid
    checkpoint; the field is then None, which restore_run reports as no count
    at all.
    """
    os.makedirs(directory, exist_ok=True)
    for index, nick in enumerate(nicks):
        personality = config.personality(index)
        blob = {
            "state_dict": nets[index].state_dict(),
            "optimizer": optimizers[index].state_dict(),
            "normalizer": {
                "mean": normalizers[index].mean,
                "var": normalizers[index].var,
                "count": normalizers[index].count,
            },
            "obs_size": obs_size,
            "hidden": personality["hidden"],
            "steps": steps,
            # This model's own, beside the run's. None where the caller has none
            # to give, and the trainer then seeds it from the run's share.
            "model_steps": (int(steps_by_model[index])
                            if steps_by_model is not None else None),
            "updates": updates,
            "nick": nick,
        }
        # The lambda is called before the loop advances, so it always saves the
        # blob built for this nick.
        write_atomically(checkpoint_path(directory, nick),
                         lambda handle: torch.save(blob, handle))

    manifest = {
        "format": FORMAT_VERSION,
        "nicks": list(nicks),
        "obs_size": obs_size,
        "max_team_size": config.MAX_TEAM_SIZE,
        # The team size this was last trained at, for the record only. It does
        # not gate loading: the layout above is what decides that.
        "team_size": int(team_size if team_size is not None else config.TEAM_SIZE),
        "steps": steps,
        "updates": updates,
        "generation": generation,
        "saved_at": time.time(),
    }
    manifest.update(extra or {})
    write_json(manifest_path(directory), manifest)


def read_manifest(directory: str) -> Optional[Dict[str, Any]]:
    """Manifest of the checkpoints on disk, or None if there are none."""
    path = manifest_path(directory)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (ValueError, OSError):
        return None


def restore_run(directory: str, nets: List[Any], optimizers: List[Any], normalizers: List[Any],
                nicks: List[str], obs_size: int,
                device: torch.device) -> Optional[Dict[str, Any]]:
    """Load a saved run in place. Returns the manifest, or None if unusable.

    Refuses to load anything shaped differently from the current build: an
    observation layout that no longer matches would load as noise and look like a
    training collapse rather than a bad restore.

    EVERY FILE IS OPENED WITH weights_only=True. Checkpoints pass from hand to
    hand, and a .pt is a pickle: loaded the permissive way, a file from a
    stranger runs whatever code it carries. Everything a checkpoint holds --
    tensors, numbers, strings, and dicts of them -- loads under the restriction,
    so nothing is given up for it.

    A DIFFERENT ROSTER IS NOT A REASON TO REFUSE. The loop below is keyed by
    FILE NAME, one file per nickname, and reads each into the roster position
    that name holds now, so the order of the manifest never matters. A name
    with no file is one model starting fresh among trained ones, and a file
    with no name in the roster is a model that is not playing, left on disk
    untouched.

    The returned manifest carries three keys of this function's own: `loaded`,
    the models whose weights came off disk, `fresh`, the ones the caller is now
    holding untrained networks for, and `model_steps`, each loaded model's own
    learning steps. A caller that needs to say which models are real has to be
    told; working it out again from the directory would be a second answer free
    to disagree with this one.
    """
    manifest = read_manifest(directory)
    if not manifest:
        return None
    reason = layout_mismatch(manifest, obs_size)
    if reason:
        logger.warning(f"Checkpoint layout is incompatible ({reason}); "
                       f"ignoring it and starting fresh")
        return None

    loaded: List[str] = []
    fresh: List[str] = []
    model_steps: Dict[str, int] = {}
    try:
        for index, nick in enumerate(nicks):
            path = checkpoint_path(directory, nick)
            if not os.path.exists(path):
                # A missing file is one model starting fresh, not a broken run.
                fresh.append(nick)
                continue
            blob = torch.load(path, map_location=device, weights_only=True)
            nets[index].load_state_dict(blob["state_dict"])
            if blob.get("optimizer"):
                optimizers[index].load_state_dict(blob["optimizer"])
            saved = blob.get("normalizer")
            if saved:
                normalizers[index].mean = saved["mean"]
                normalizers[index].var = saved["var"]
                normalizers[index].count = saved["count"]
            # This model's own learning steps, reported back so the trainer can
            # resume the per-model figures the same way it resumes the weights.
            # A checkpoint without the field reports none, and the trainer seeds
            # those from the run's share.
            own = blob.get("model_steps")
            if isinstance(own, (int, float)):
                model_steps[nick] = int(own)
            loaded.append(nick)
    except Exception as exc:
        logger.error(f"Could not restore checkpoints: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        return None

    # A restore that recognised nothing at all is a directory belonging to some
    # other roster, not a run to resume: carrying its step count forward would
    # print that run's steps over untrained networks, and the run on disk would
    # look like it had collapsed rather than like it had never been read. One
    # recognised model is enough to be the same run.
    if not loaded:
        logger.warning(f"No checkpoint in {directory} belongs to this roster "
                       f"({', '.join(nicks)}); starting fresh")
        return None

    report = dict(manifest)
    report["loaded"] = loaded
    report["fresh"] = fresh
    # Keyed by nickname rather than by index, because the caller's roster order
    # is its own, and a list could hand one model's history to another.
    report["model_steps"] = model_steps
    if fresh:
        logger.warning(f"No checkpoint for {', '.join(fresh)}; "
                       f"{'it starts' if len(fresh) == 1 else 'they start'} from a fresh network")
    orphans = [n for n in manifest.get("nicks", []) if n not in nicks]
    if orphans:
        # Named rather than deleted. The weights are still on disk and are still
        # worth something: a team sitting this run out plays again the moment it
        # is named.
        logger.info(f"Checkpoints for {', '.join(orphans)} are not in the roster and were left alone")
    logger.info(f"Checkpoints at {manifest.get('steps', 0) / 1e6:.2f}M steps "
                f"({manifest.get('updates', 0)} updates): "
                f"{len(loaded)} of {len(nicks)} models restored")
    return report


def main():
    pass


if __name__ == "__main__":
    main()
