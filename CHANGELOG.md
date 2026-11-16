# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- The game and the learning code as a standalone package: the paintball
  simulation (`engine/env/`), the network, PPO and checkpoints (`engine/rl/`)
  and the trainer (`trainer/selfplaytv.py`). The only requirements are torch,
  numpy, numba and tbb. The rules, the physics, the arena and the scoring are
  literals in `engine/config.py`: nothing in the shell's environment changes
  the game a team is trained for.
- A bundle of two trained teams, WHITE TEAM and BLACK TEAM, five models each,
  to copy from and to train against (`bundle/`).
- `python -m sdk.new_team`: create a team of your own, with a name, a colour
  and five nicknames, as a copy of a team in the bundle or from untrained
  networks. It lives in a bundle of your own; the shipped one is never
  written.
- Reward weights in a small Lua table (`examples/rewards.lua`), read as data
  without a Lua runtime, and `python -m sdk.rewards` to check a file and print
  its weights beside the defaults.
- `python -m sdk.train`: train one or more teams against frozen ones, whole
  matches at a time, with `--rewards` and `--reward` for the weights, a
  `--seed` that repeats a run, resume, `--restart`, and a clean stop on
  Ctrl-C, SIGTERM or a closed terminal (SIGHUP): it finishes its update and
  saves. Under `nohup` the hangup stays ignored and the run goes on. `--play-only` measures a bundle on the full arena without learning
  or saving. The run's lock file names the process that holds it, and
  `status.json` carries the trainer's `pid`, the `args` it was started with,
  and `stopped_at` and `stop_reason` once it has ended cleanly.
- `python -m sdk.watch`: a terminal dashboard of a run, from its
  `status.json`: recent matches, standings with how rounds end, every
  player's numbers, average points a round and a match, and the average
  reward the learners are paid.
- `python -m sdk.demo`: matches between two teams drawn in the terminal with
  half-block characters, with the score and the average points a round, or
  printed as a summary with the rounds each team won. `--graphics` sends the
  same picture as pixels to a terminal that speaks the kitty graphics
  protocol; by default it asks the terminal, and never inside tmux or screen.
- `python -m sdk.export_team`: one team as one archive, in the SELFPLAYTV run
  archive layout, with the team, its players and its colour renamed on the
  way out if asked, under the same naming rules a new team is made by.
- `python -m sdk.tui`: one full-screen program for the whole journey. Keys
  start and stop the training, change the settings and the reward weights of
  the next start, show the dashboard and the log, play a match of the saved
  weights in a pane, make a team and export one. The trainer runs detached,
  so leaving the screen or losing the terminal never stops it.
- Every checkpoint is loaded with `weights_only=True`.
- Documentation: `README.md`, the step-by-step guide `AGENTS.md`, the
  reference `docs/sdk.md`, and `examples/quickstart.sh`, the journey as one
  script.
- `LICENSE`: MIT.
- `python -m sdk.doctor`: checks the installation and says what to fix, a
  line each: the versions of Python, torch, numpy and numba, TBB and the
  threading layer numba uses, the GPU torch sees (MPS on macOS), the CPUs,
  and the speed of WHITE TEAM against BLACK TEAM at 16 and 64 arenas. It
  exits 1 when something blocks training.
- `sdk.train` warns once, at a learning start, when the timeout weight is 0
  and nothing charges a round that is waited out on the clock.
- `sdk.train` warns once when learning stalls for 20 updates in a row: a
  policy that no longer moves, an entropy that has collapsed, or learners
  that hit nobody. It says what to try and keeps training. `status.json`
  carries it as `health`, `sdk.watch` shows it in its header, and `sdk.tui`
  marks the training row `stalled`.
- `status.json` records the environment of each start under `environment`:
  the Python, torch, numpy and numba versions, numba's threading layer, the
  thread count, the platform and the SDK's git commit. `sdk.watch` shows the
  versions at the end of its header's second line where they fit. Both new
  keys are additions; the file's version stays 1.
- `start.sh`: one command from a fresh clone to the screen. The first run
  makes `.venv/`, installs torch from the index that fits the machine (CPU,
  cu128 when `nvidia-smi` works, PyPI on macOS) and the requirements, and
  says what to install, without sudo, when Python 3.11 or `python3-venv` is
  missing; later runs open the screen at once.
- `python -m sdk.tui` without a run is a guided path: a list of your own
  teams with NEW TEAM last, a form for a new team (a name, a colour from seven
  swatches, five players with free nicknames filled in), checked by
  `sdk.new_team` itself with each refusal under its field, and D to delete a
  team that is not training, after a question that lists what goes. A team
  opened or just made goes straight to training, against WHITE TEAM and BLACK
  TEAM by the example weights on all the CPUs but two, and to matches that
  play one after another, alternating the two opponents, each with the run's
  newest save. With a run named, the TUI is as it was.
- `sdk.new_team.remove_team`, which takes a team and its five networks out of
  a bundle of one's own in an order that leaves a readable bundle if it is cut
  short. The TUI's delete calls it; no command does.

### Changed

- The share of a hit paid to the shooter's and the victim's teammates is
  credited with numpy instead of a Python loop per hit and slot. The order of
  every sum is kept, and a seeded run writes byte-identical checkpoints.
