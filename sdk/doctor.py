#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check the installation and say what to fix: the libraries, TBB, the GPU, the CPUs, the speed.

    python -m sdk.doctor
    python -m sdk.doctor --threads 4

Run from the repository root. Every line of the report starts OK, WARN or FIX.
WARN is usable as it is and slower or narrower than it could be; FIX blocks
training. The exit status is 0 when nothing says FIX and 1 when something does.

The last line is a measurement: a few seconds of WHITE TEAM against BLACK TEAM
from the shipped bundle, on the CPU, through the trainer's own rollout with
nothing learning, at two arena counts. It is the speed of play alone; training
also spends time in the update, and ran at about two fifths of it here.

The report is on standard output, and the engine's own warnings, if any, on
standard error.
"""

import argparse
import importlib
import logging
import os
import re
import sys
import time
from typing import Any, Optional, Tuple


# The oldest versions the SDK is written for: README.md and requirements.txt.
PYTHON_MIN = (3, 11)
LIBRARIES = {
    "torch": ((2, 4), "install torch first, from the index for your machine: see docs/sdk.md#install"),
    "numpy": ((2, 0), "pip install -r requirements.txt"),
    "numba": ((0, 60), "pip install -r requirements.txt"),
}
INSTALL_TBB = "install tbb (pip install -r requirements.txt, or libtbb12 off x86-64), see docs/sdk.md#install"
# The arena counts measured, and how long each is played for after one rollout
# of warm-up: 16 is the trial run of AGENTS.md, 64 the default of sdk.train.
# Two seconds is one or two rollouts at 64, enough for a rate to two figures.
ARENA_COUNTS = (16, 64)
MEASURE_SECONDS = 2.0
SHIPPED_TEAMS = ("WHITE TEAM", "BLACK TEAM")


def report(mark: str, name: str, text: str) -> bool:
    """Write one line of the report at once, so a slow measurement shows what came before it."""
    sys.stdout.write(f"{mark:<4}  {name:<8}  {text}\n")
    sys.stdout.flush()
    return mark != "FIX"


def version_tuple(text: str) -> Tuple[int, ...]:
    """`2.11.0+cu128` as (2, 11): the two numbers a minimum is written in."""
    return tuple(int(part) for part in re.findall(r"\d+", text)[:2])


def check_python() -> bool:
    found = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info[:2] < PYTHON_MIN:
        return report("FIX", "python", f"{found}: the SDK needs {'.'.join(map(str, PYTHON_MIN))} or newer")
    return report("OK", "python", found)


def check_library(name: str) -> Optional[Any]:
    """The module if it imports at a version the SDK takes, after its line of the report."""
    minimum, advice = LIBRARIES[name]
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        report("FIX", name, f"does not import ({type(exc).__name__}: {str(exc)}): {advice}")
        return None
    if version_tuple(module.__version__) < minimum:
        wanted = ".".join(map(str, minimum))
        report("FIX", name, f"{module.__version__}: the SDK needs {wanted} or newer; {advice}")
        return None
    report("OK", name, module.__version__)
    return module


def check_threads(train: Any, numba: Any) -> bool:
    """Whether TBB is there, and the threading layer numba chose, read after a parallel kernel has run."""
    try:
        importlib.import_module("numba.np.ufunc.tbbpool")
        tbb = True
        report("OK", "tbb", "present")
    except Exception:
        tbb = False
        report("WARN", "tbb", f"missing: {INSTALL_TBB}")
    try:
        train.kernels.warmup(train.TEAM_SIZE)
    except Exception as exc:
        return report("FIX", "numba", f"the simulation does not compile ({type(exc).__name__}: {str(exc)}): "
                                      f"reinstall numba with pip install -r requirements.txt")
    layer = train.threading_layer()
    threads = numba.get_num_threads()
    if layer == "tbb":
        return report("OK", "layer", f"numba runs on the tbb layer, {threads} threads")
    if tbb:
        return report("WARN", "layer", f"TBB is installed and numba chose the {layer} layer: unset "
                                       f"NUMBA_THREADING_LAYER")
    return report("WARN", "layer", f"numba is using the {layer} layer and not TBB: install tbb, "
                                   f"see the line above")


def check_devices(torch: Any) -> bool:
    """The GPU torch sees, if any, asked to compute once; on macOS, whether MPS exists."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        try:
            torch.ones(1, device="cuda").add_(1).cpu()
        except Exception as exc:
            # The wheel for a newer card installs on an older one and fails at
            # the first operation; sdk.train takes the GPU by default, so say it.
            return report("WARN", "gpu", f"{name} fails a first operation ({type(exc).__name__}): install "
                                         f"the torch for that card (docs/sdk.md#install), until then "
                                         f"--device cpu")
        report("OK", "gpu", f"{name}, CUDA {torch.version.cuda}: sdk.train uses it unless --device cpu")
    elif torch.version.cuda:
        report("OK", "gpu", f"none visible to this CUDA {torch.version.cuda} build of torch: training runs "
                            f"on the CPU, which is enough")
    else:
        report("OK", "gpu", "none, and this torch is a CPU build: training runs on the CPU, which is enough")
    if sys.platform == "darwin":
        mps = getattr(torch.backends, "mps", None)
        present = bool(mps and mps.is_available())
        report("OK", "mps", ("present; sdk.train uses it only with --device mps, which has not been run"
                             if present else "not available: training runs on the CPU"))
    return True


def check_cpus(threads: int) -> bool:
    count = os.cpu_count() or 1
    usable = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else count
    shown = f"{count}" if usable == count else f"{count}, {usable} of them usable by this process"
    return report("OK", "cpus", f"{shown}; this check plays on {threads} threads")


def measure(train: Any, arenas: int) -> float:
    """Agent-steps a second of WHITE TEAM against BLACK TEAM in `arenas` arenas, nobody learning.

    Fielded as sdk.train fields a --play-only run and sdk.demo a match: the
    lineup draw, the full-arena clock and the rollout are train's own.
    """
    bundle = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "bundle")
    teams_doc, nicks = train.read_bundle(bundle)
    teams = teams_doc["teams"]
    train.check_bundle(teams, nicks)
    first, second = train.resolve_teams(list(SHIPPED_TEAMS), teams)
    unused_team_of_model, members = train.seat_teams(nicks, teams, [first, second])
    device = train.torch.device("cpu")
    trainer = train.trainer_module.Trainer(arenas, device, seed=0, team_size=train.TEAM_SIZE)
    restored = train.persist.restore_run(bundle, trainer.nets, trainer.optimizers, trainer.normalizers,
                                         list(train.config.MODEL_NICKS), trainer.obs_size, device)
    if not restored or restored["fresh"]:
        raise RuntimeError(f"the checkpoints in {bundle} did not load")
    trainer.read_clock = train.full_arena_clock
    trainer.draw_lineups = train.make_assign_slots({"arenas": [None] * arenas}, members, [first],
                                                   {first: [second]})
    clock = train.full_arena_clock()
    phase = train.curriculum.phase_for(clock)
    trainer.phase_idx = train.curriculum.phase_index(clock)
    train.trainer_module.apply_phase(trainer, phase)
    # One rollout untimed: torch's first calls allocate and are not the rate.
    train.trainer_module.collect_rollout(trainer, phase)
    start, steps = time.perf_counter(), trainer.steps_done
    while time.perf_counter() - start < MEASURE_SECONDS:
        train.trainer_module.collect_rollout(trainer, phase)
    return (trainer.steps_done - steps) / (time.perf_counter() - start)


def check_speed(train: Any, threads: int) -> bool:
    rates = []
    for arenas in ARENA_COUNTS:
        try:
            rates.append(f"{arenas} arenas {measure(train, arenas):,.0f}/s")
        # SystemExit as well: the bundle checks of sdk.train refuse by exiting,
        # and here a refusal is a line of the report, after the one that says why.
        except (Exception, SystemExit) as exc:
            return report("FIX", "speed", f"the shipped teams do not play ({type(exc).__name__}: "
                                          f"{str(exc)}); bundle/ must be as it ships: git checkout bundle")
    return report("OK", "speed", f"{', '.join(rates)} agent-steps of play on {threads} threads")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m sdk.doctor",
                                     description="Check the installation and measure the speed of play")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 1) // 2),
                        help="threads for the measurement, as sdk.train --threads "
                             "(default: half the CPUs)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    return args


def main():
    args = parse_args()
    usable = check_python()
    numpy = check_library("numpy")
    if numpy is None:
        sys.stdout.write("Blocked: fix the lines marked FIX and run this again\n")
        sys.exit(1)
    # sdk.train needs numpy to import, and sets the numba thread count, which
    # numba reads once, when it is first imported: so it comes first.
    from sdk import train
    # The engine's info lines would interleave with the report; its warnings
    # and errors still show.
    logging.getLogger().setLevel(logging.WARNING)
    train.set_environment(args.threads)
    torch = check_library("torch")
    numba = check_library("numba")
    usable = usable and torch is not None and numba is not None
    if usable:
        train.import_engine()
        torch.set_num_threads(args.threads)
        usable = check_threads(train, numba)
        check_devices(torch)
        check_cpus(args.threads)
        usable = usable and check_speed(train, args.threads)
    sys.stdout.write("Usable: everything sdk.train needs is here\n" if usable
                     else "Blocked: fix the lines marked FIX and run this again\n")
    sys.exit(0 if usable else 1)


if __name__ == "__main__":
    main()
