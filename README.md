# SELFPLAYTV SDK

SELFPLAYTV is team paintball played by neural networks that learned the game
by playing each other. This repository is the simulation and the trainer on
their own: you make a team of five networks, decide what it is paid for while
it learns, and train it on your own machine against the two teams that ship
here. You watch it in the terminal, as a dashboard of scores and as matches
drawn in pixels, and export it as one archive.

## Quick start

Five commands, on a CPU, from nothing to your own team training:

```bash
git clone https://github.com/wachawo/selfplaytv-sdk && cd selfplaytv-sdk
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
python -m sdk.new_team --name "MY TEAM" --colour "#12b5a6" --nicks Ada Bob Cy Dee Eve
python -m sdk.tui runs/my-team
```

In the terminal screen that opens, `T` starts the training and stops it, `S`
and `W` change its settings and reward weights, `M` plays a match of your team,
`D` is the dashboard, and `Q` leaves while the training goes on. The keys are
always listed at the bottom of the screen. With an NVIDIA GPU, install torch
from the index in [Requirements](#requirements) instead.

## Requirements

- Python 3.11 or newer.
- Linux or macOS. On Windows, use WSL.
- torch, installed first, from the index that matches your machine. Then the
  rest:

  ```bash
  python3 -m venv .venv && . .venv/bin/activate
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt
  ```

  | Machine | torch |
  |---|---|
  | CPU only | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
  | NVIDIA GPU | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |

  Older NVIDIA cards need an older index; see
  [docs/sdk.md](docs/sdk.md#install).

A GPU is optional. The networks are small and a run trains on a CPU, with
about 1 GB of memory.

## Step by step

The same journey without the terminal screen: six commands, from the
repository root, with the environment above active.

```bash
# 1. Your team: a name, a colour, five nicknames. It starts as a copy of WHITE TEAM.
#    Keep the colour in quotes: an unquoted # starts a comment in the shell.
python -m sdk.new_team --name "MY TEAM" --colour "#12b5a6" --nicks Ada Bob Cy Dee Eve

# 2. What it is paid for while it learns. This checks the example file.
python -m sdk.rewards examples/rewards.lua

# 3. Train it against the two shipped teams. Two million steps, about three minutes.
python -m sdk.train --bundle my-bundle --run runs/my --learn "MY TEAM" --against WHITE BLACK \
    --rewards examples/rewards.lua --arenas 16 --threads 4 --steps 2000000

# 4. The run as a dashboard: matches, standings, every player's numbers.
#    It redraws every two seconds until Ctrl-C, and wants about 100 columns by 46 lines.
python -m sdk.watch runs/my

# 5. One match, drawn in the terminal at real time: about five minutes.
#    --speed 4 plays it faster, Ctrl-C stops it.
python -m sdk.demo --run runs/my "MY TEAM" WHITE

# 6. The team as one archive, in exports/.
python -m sdk.export_team --run runs/my --team "MY TEAM"
```

`bash examples/quickstart.sh` runs the same journey as one script, with
smaller numbers, in three to four minutes.

Run command 3 again and it resumes. Give it `--hours 2` instead of `--steps`
for a real session, and stop it with Ctrl-C whenever you like: it saves first.
Command 4 belongs in a second terminal while command 3 is running; activate
the environment there too.

`No module named 'sdk'` means a command was not run from the repository root,
or a file was run directly: use `python -m sdk.<name>` from the root.

## One screen for all of it

```bash
python -m sdk.tui runs/my
```

The same journey from one terminal screen, by keys: `T` starts and stops the
training, `S` and `W` change its settings and reward weights for the next
start, `M` plays a match of the saved weights in the screen, `N` makes a team
and `X` exports one. The trainer runs as a process of its own, so `Q`, a
closed terminal or a dropped ssh leave it training, and the same command
brings the screen back. Here it is at 80 by 24, training NEON FOXES, a copy of
WHITE TEAM that is paid less for rounds that end on the clock:

```
STANDINGS
  team        matches        W-L-U    rounds   time%   elim%    cap%   draw%  p>
  NEON FOXES        4        2-2-0        44    0.0%   61.4%   38.6%    2.3%   >
  WHITE TEAM        4        2-2-0        44    0.0%   61.4%   38.6%    2.3%   >
  BLACK TEAM        0        0-0-0         0       -       -       -       -   >
rewards  timeout -2 (runs/neon-foxes/rewards.lua)
paid     NEON FOXES +0.47 per player per round  (updates 71-75)
phase    full arena  NEON FOXES (2,355.98M steps of its own)

RECENT MATCHES  (4 played)
  NEON FOXES   5:7   WHITE TEAM   12 rounds  time  0  elim  7  cap  5
  NEON FOXES   6:7   WHITE TEAM   13 rounds  time  0  elim  9  cap  4
  NEON FOXES   7:3   WHITE TEAM   11 rounds  time  0  elim  7  cap  4  draws 1
  NEON FOXES   7:1   WHITE TEAM    8 rounds  time  0  elim  4  cap  4

PLAYERS
    nick      rounds            W/D/L     hits   deaths    K/D     acc   caps  >
  NEON FOXES
    Kit          273        139/8/126      402      389   1.03   14.6%     23  >
    Vix          273        139/8/126      317      249   1.27   15.4%      0  >
... 15 more rows below: down, PgDn
 TRAINING  runs/neon-foxes  1.06M steps (+0.59M)  12,741/s  up 48 s  saved 9 s >
 now NEON FOXES v WHITE TEAM, BLACK TEAM  2 threads  cpu  16 arenas  until stop>
T stop  D dash  M match  S settings  W weights  L log  N team  X export  Q quit
```

The commands above stay as they are; the screen runs them for you. See
[docs/sdk.md](docs/sdk.md#sdktui-one-screen-for-all-of-it).

## What to expect

**Time.** On a desktop CPU with four threads and the default 64 arenas a run
does about 30,000 steps a second: a million steps in half a minute, a hundred
million in about an hour. Command 3 uses 16 arenas, which is slower, about
12,000 steps a second, but lets whole matches finish inside a short run: the
matches still being played when a run stops are not counted.

**Strength.** The two shipped teams are veterans, with over two billion steps
of training behind each.

- A team copied from one of them plays as well as its source from the first
  step. Against WHITE, a copy of WHITE wins about half its rounds.
- Training then changes it slowly, in the direction your reward weights pay
  for. A few million steps do not make it better: in the runs for this
  document, four million steps with a changed weight left a copy slightly
  behind where it started (46.8% of its rounds against WHITE, from 50%).
  Expect to train for tens of millions of steps, and measure.
- A team started from untrained networks (`--fresh`) loses every match at
  first: 353 of 353 in its first two million steps here. It needs about two
  hours just to reach the full-size field, and is then still far behind.

**Judging.** Results are noisy. The same team measured against an identical
copy of itself finished anywhere from 14-26 to 23-17 over about forty matches. Read
[how much is noise](docs/sdk.md#how-much-is-noise) before believing a
difference.

## Read more

- [AGENTS.md](AGENTS.md): the journey as step-by-step instructions, with how
  to choose the settings for your machine, how to judge a team and how to
  iterate on the reward weights. Written so that a coding agent can follow it.
- [docs/sdk.md](docs/sdk.md): the reference. Every command and flag, the file
  formats, the reward weights file, the curriculum, measuring, the archive.
- [CHANGELOG.md](CHANGELOG.md): what the package contains.
- [examples/rewards.lua](examples/rewards.lua): the example weights file.

Two things never change, whatever you train: the game itself (everything
under `engine/`) and the shipped teams in `bundle/`. A team trained on a
changed game is a team for a different game. The list is in
[docs/sdk.md](docs/sdk.md#what-must-not-change).

## License

MIT, see [LICENSE](LICENSE).
