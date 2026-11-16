# SELFPLAYTV SDK reference

This is the long document. The short one is [README.md](../README.md), and the
step-by-step working guide is [AGENTS.md](../AGENTS.md).

Every command here is run from the repository root, because the commands
import `engine/`, `trainer/` and `sdk/` from there and the default paths are
relative to it. Every command and every piece of output below was run on a
Linux desktop CPU with Python 3.12, under two sets of libraries: torch 2.11
with numpy 2.4 and numba 0.66, and torch 2.9 with numpy 2.2 and numba 0.62.
The pip lines under [Install](#install) are the exception: they were not run
for this document. Log lines are shown without their time stamp.

- [The game](#the-game)
- [Install](#install)
- [The commands at a glance](#the-commands-at-a-glance)
- [Bundles](#bundles)
- [sdk.new_team: create a team](#sdknew_team-create-a-team)
- [The reward weights file](#the-reward-weights-file)
- [sdk.train: train a team](#sdktrain-train-a-team)
- [The run directory](#the-run-directory)
- [The curriculum, and what to expect](#the-curriculum-and-what-to-expect)
- [sdk.watch: the dashboard](#sdkwatch-the-dashboard)
- [sdk.demo: matches in the terminal](#sdkdemo-matches-in-the-terminal)
- [Measuring a team](#measuring-a-team)
- [sdk.export_team: one archive](#sdkexport_team-one-archive)
- [sdk.tui: one screen for all of it](#sdktui-one-screen-for-all-of-it)
- [What must not change](#what-must-not-change)

## The game

Two teams of five play paintball on a field of 30 by 20 metres with 18
obstacles. A player who is hit is out for the rest of the round.

- **Round.** One fight of 60 seconds on the round clock. The clock stops
  while a strip is being taken, so a round can last up to 90 seconds in all.
  It ends in one of three ways:
  - **elimination**: one side has nobody left;
  - **capture**: a side stood in the enemy's spawn strip for five seconds
    without a break, which wins the round outright;
  - **time**: the clock ran out. The side with more survivors wins, and equal
    numbers are a draw.
- **Match.** First team to win seven rounds (`ROUNDS_TO_WIN`). A drawn round
  scores for nobody. A match still undecided after 28 rounds is recorded as
  unfinished.
- **Points.** The scoreboard, counted by fixed rules that nothing in this SDK
  changes:

  | Event | Points |
  |---|---|
  | a hit | 1 to the shooter |
  | a round won | 1 to each player of the winning side |
  | a capture | 3 to each player standing on the strip as it falls and 1 to each other player of that side, on top of the point for the round won |

- **Reward.** What a network is paid while it learns. This is the part you
  change, in [the reward weights file](#the-reward-weights-file). Points and
  reward are two different things: change a weight and the reward moves at
  once, while the points move only if the team comes to play differently.

Each model is one small network (two hidden layers of 256 units for the policy
and the same again for the value estimate). It sees 186 numbers about its own
surroundings and chooses a move, a turn and whether to shoot ten times a
second. A team is five such networks, each with its own weights.

## Install

Python 3.11 or newer. Install torch first, from the index that matches the
machine, then the rest:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

| Machine | torch |
|---|---|
| CPU only | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| NVIDIA, RTX 20 series or newer, T4 | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| NVIDIA, GTX 10 series, P40, P100 | `pip install "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126` |

The older cards need the older CUDA index: a newer wheel installs cleanly on
them and then fails at the first operation with "no kernel image is
available".

torch is not in `requirements.txt` on purpose. pip cannot choose an index per
package from a requirements file, and the default wheel is a large CUDA build
that a CPU-only machine cannot use.

`requirements.txt` installs numpy, numba and, on x86-64, `tbb`. Without TBB
numba falls back to a threading layer whose workers spin between calls, and
training runs at a fraction of its speed. On other machines install the system
package (`libtbb12` on Debian and Ubuntu) or keep `--threads` low.

A GPU is optional. The networks are small, and everything in this document was
run on a CPU.

The package is developed and run on Linux. macOS should behave the same.
On Windows use WSL: the code has fallbacks for a console without `fcntl` and
`termios`, but they have not been run.

## The commands at a glance

| Command | What it does | Writes |
|---|---|---|
| `python -m sdk.new_team` | creates your team in a bundle of your own | `my-bundle/` |
| `python -m sdk.rewards FILE` | checks a reward weights file | nothing |
| `python -m sdk.train` | trains teams against frozen ones; with `--play-only`, measures | the run directory |
| `python -m sdk.watch RUN` | shows a run in the terminal | nothing |
| `python -m sdk.demo TEAM TEAM` | plays matches and draws them in the terminal | nothing |
| `python -m sdk.export_team` | packs one team into one archive | `exports/` |
| `python -m sdk.tui [RUN]` | one screen that runs all of the above, for a person at a terminal | the run directory; through the commands it runs, `my-bundle/` and `exports/` |

Each of them prints its flags with `--help`. A command that refuses to start
says why in one log line and exits with status 2, before it has written
anything. An unexpected error is logged with its traceback and exits 1.

## Bundles

A bundle is a directory that holds teams and their weights:

| File | What it is |
|---|---|
| `teams.json` | the teams: each one's colour and its five members in order, and every model's entropy scale |
| `manifest.json` | the checkpoint manifest: the nicknames, the observation size, the counters |
| `<nick>.pt` | one checkpoint per model |

The repository ships one, `bundle/`, with two teams:

| Team | Colour | Members |
|---|---|---|
| WHITE TEAM | `#eef1f4` | Hiroshi, Kenji, Takeshi, Haruto, Yota |
| BLACK TEAM | `#1b1e23` | Ivan, Dmitriy, Maksim, Aleksey, Artyom |

Both are veterans: each of the ten models has between 440 and 506 million
learning steps of its own. They play differently. In five demo matches between
them, 48 of 54 rounds ended in an elimination and 3 in a capture. Against
WHITE alone about a third of the rounds end in a capture.

**`bundle/` is never written.** Your team lives in a copy, `my-bundle/` by
default, which `sdk.new_team` creates. `my-bundle/`, `runs/` and `exports/`
are ignored by git, and so is any other bundle whose name starts with
`my-bundle`. A bundle or a run kept anywhere else is not.

`teams.json`:

```json
{
  "version": 1,
  "teams": {
    "WHITE TEAM": {"colour": "#eef1f4", "members": ["Hiroshi", "Kenji", "Takeshi", "Haruto", "Yota"]},
    "BLACK TEAM": {"colour": "#1b1e23", "members": ["Ivan", "Dmitriy", "Maksim", "Aleksey", "Artyom"]}
  },
  "personality": {"Hiroshi": 1.1, "Kenji": 0.7, "Takeshi": 0.95, "Haruto": 1.2, "Yota": 1.05,
                  "Ivan": 0.9, "Dmitriy": 1.15, "Maksim": 0.75, "Aleksey": 1.0, "Artyom": 1.1}
}
```

- A team has exactly five members. A bundle with a team of any other size is
  refused, and so is one with a team name that is not visible text on one
  line: the name is written to the terminal by every command.
- A nickname is the checkpoint's file name. It holds letters, digits, hyphens
  and underscores, starts with a letter or a digit, and is at most 24
  characters. No two models in a bundle may share one, and case does not
  count: on a file system that ignores case `ada.pt` and `Ada.pt` are one
  file.
- `personality` is each model's entropy scale: how much it is rewarded for
  keeping its play varied while it learns. It travels by nickname because the
  checkpoint does not store it.

`manifest.json`:

```json
{
  "format": 1,
  "nicks": ["Hiroshi", "Kenji", "Takeshi", "Haruto", "Yota",
            "Ivan", "Dmitriy", "Maksim", "Aleksey", "Artyom"],
  "obs_size": 186, "max_team_size": 5, "team_size": 5,
  "steps": 0, "updates": 0, "generation": 0,
  "saved_at": 1794830470.99,
  "phase": "full arena (30x20m, 18 obstacles, 60s, hold 5.0s, autofire=off)"
}
```

`steps` and `updates` are the counters of the run the checkpoints were saved
by. The shipped bundle has them at zero, so a new run starts counting at zero.

A `<nick>.pt` is a dictionary saved by torch:

| Key | What it is |
|---|---|
| `state_dict` | the network's weights |
| `optimizer` | the Adam state, so training carries on where it stopped |
| `normalizer` | the running statistics of the returns: `mean`, `var`, `count` |
| `obs_size` | 186 |
| `hidden` | 256 |
| `steps`, `updates` | the counters of the run that saved the file |
| `model_steps` | this model's own learning steps; the curriculum clock reads it |
| `nick` | the nickname |

A shipped checkpoint is 2.8 MB. Every checkpoint is opened with
`torch.load(..., weights_only=True)`, everywhere in this package. A `.pt` is a
pickle, and one loaded the permissive way can run code. Load other people's
checkpoints the same way.

## sdk.new_team: create a team

```bash
python -m sdk.new_team --name "MY TEAM" --colour "#12b5a6" --nicks Ada Bob Cy Dee Eve
```

```
New bundle .../my-bundle, a copy of .../bundle
Added MY TEAM (#12b5a6) to .../my-bundle: Ada, Bob, Cy, Dee, Eve, copies of WHITE TEAM
MY TEAM entropy scales: Ada 1.10, Bob 0.70, Cy 0.95, Dee 1.20, Eve 1.05
Train it with: python -m sdk.train --bundle my-bundle --run runs/my --learn 'MY TEAM' --against 'WHITE TEAM' 'BLACK TEAM'
```

The last line is the command that trains the team, ready to paste. It names a
run directory after the team (`MY TEAM` is `runs/my`, `NEON FOXES` is
`runs/neon-foxes`) and every other team the bundle holds now as the opponents.

| Flag | Default | Meaning |
|---|---|---|
| `--name NAME` | required | the team's name |
| `--colour "#rrggbb"` | required | the team's colour: `#` and six hexadecimal digits, in quotes, because an unquoted `#` starts a comment in the shell; `--color` is the same flag |
| `--nicks A B C D E` | required | five nicknames |
| `--from TEAM` | `WHITE` | copy that team's five networks |
| `--fresh` | off | start from untrained networks instead |
| `--bundle DIR` | `bundle` | the bundle `--out` is copied from when it does not exist yet |
| `--out DIR` | `my-bundle` | your bundle: created on the first call, added to on later ones |

The first call copies `--bundle` to `--out` and adds the team there. Every
later call adds another team to the same `--out`, and `--bundle` is then
ignored, so `--bundle my-bundle` adds to `my-bundle` as well. `--out` is built
beside its final place and renamed in, and a team added to an existing bundle
is staged first and named last, so a call that fails half way leaves the
bundle as it was.

A team starts in one of two ways, and they are two different trainings:

- **`--from TEAM`** copies that team's networks under the new nicknames with
  everything that makes them veterans: the optimiser state, the return
  normaliser and each model's own step count. The copy plays exactly as well
  as its source from the first step, and it trains on the full arena with
  small updates. The new five take the source members' entropy scales in
  order. `--from` reads the bundle being extended, so a team of your own can
  be the source of another. It copies the weights the bundle holds, not the
  ones a run has trained since.
- **`--fresh`** gives five untrained networks with a step count of zero. They
  start on the small close-quarters arena with large updates and have the
  whole curriculum ahead of them. See
  [what to expect](#the-curriculum-and-what-to-expect).

```bash
python -m sdk.new_team --name "ROOKIES" --colour "#e07a1f" --nicks Rook1 Rook2 Rook3 Rook4 Rook5 --fresh
```

```
Added ROOKIES (#e07a1f) to .../my-bundle: Rook1, Rook2, Rook3, Rook4, Rook5, fresh networks
ROOKIES entropy scales: Rook1 0.75, Rook2 1.25, Rook3 0.85, Rook4 0.70, Rook5 0.95
```

Do not put a copied team and a fresh one in the same `--learn`. The learners
share one curriculum clock, and the veterans' steps would put the fresh team
on the full arena from its first update. `sdk.train` accepts it and warns:

```
[WARNING] The learners are at different stages of the curriculum on their own (MY TEAM: full arena, ROOKIES: close quarters) and share one clock here, so the younger ones skip theirs. Train them in separate runs
```

Refused, with the reason:

- a name that is blank, longer than 64 characters or starts with a hyphen, or
  that an existing team could be taken for. Case does not count, and a name is also read without a
  trailing `TEAM`, so `white` clashes with `WHITE TEAM`;
- a colour that is not `#rrggbb`. Upper-case digits are accepted and stored in
  lower case;
- anything but five nicknames, a nickname that breaks the rules above, or one
  that a model in the bundle already has;
- `--out` that is the shipped `bundle/` or inside it, a new `--out` that is
  the `--bundle` it would be copied from or inside it, or an `--out` that
  exists and is not a bundle.

## The reward weights file

What your team is paid for while it learns is a small file: one Lua table of
numbers. `examples/rewards.lua` is the example:

```lua
return {
  hit = 1.0,            -- the shooter, for each hit; the shooter's teammates get a share of it
  hit_taken = -1.0,     -- the player hit; that player's teammates take a share of it
  win = 2.0,            -- each player of the side that wins a round
  loss = -1.0,          -- each player of the side that loses it
  draw = -1.0,          -- each player of both sides when a round is drawn
  timeout = -2.0,       -- each player of both sides when a round ends on the clock, on top of those
  capture = 3.0,        -- each player standing on the enemy strip as it falls
  capture_cover = 2.0,  -- each player of the capturing side who was not on the strip
  defence = 1.0,        -- a hit on an intruder in your own spawn, in place of the ordinary hit
}
```

Keep your own copy beside your team, for example `my-bundle/rewards.lua`. It
is ignored by git there and travels with the bundle.

### The weights

Every weight is paid to one player. A weight left out of the file keeps its
default.

| Key | Default | Paid when |
|---|---|---|
| `hit` | 1.0 | to the shooter, for each hit; the shooter's teammates get a share of it |
| `hit_taken` | -1.0 | to the player hit; that player's teammates take a share of it |
| `win` | 2.0 | to each player of the side that wins a round |
| `loss` | -1.0 | to each player of the side that loses it |
| `draw` | -1.0 | to each player of both sides on a draw |
| `timeout` | 0.0 | to each player of both sides when the round ends on the clock, on top of the above |
| `capture` | 3.0 | to each player standing on the enemy strip as it falls |
| `capture_cover` | 2.0 | to each player of the capturing side who was not on the strip |
| `defence` | 1.0 | for a hit on an intruder in your own spawn, in place of the ordinary hit |

A key is a `REWARD_*` constant of `engine/config.py` in lower case without the
prefix: `timeout` is `REWARD_TIMEOUT`.

**`timeout`, and why the example switches it on.** A team can learn to hide
and wait for the clock. The draw penalty does not cure that: most rounds that
run out of time are settled on survivors and are not draws, so most of the
waiting is never charged. `timeout` charges every round that ends on the
clock, whoever wins it, and it is paid to the dead as well, so it cannot be
dodged by dying. At -2 a win on the clock pays 2 - 2 = 0 and a loss on the
clock pays -3, so finishing the round is the only way to earn the win.

The two shipped teams rarely wait: in the measurements for this document
between none and six in a hundred of their rounds ended on the clock.

Besides the weights, a network is also paid a few small fixed shaping terms
(for aiming, for approaching and holding the enemy strip, and a small cost for
a wasted shot). They are not in the file and cannot be changed.

### The grammar

The file is data. Nothing runs Lua: the reader takes the one shape below and
refuses everything else with the line it stopped on. A weights file is passed
between people along with a team, and one that could execute code would have
to be audited before every use.

Accepted:

- line comments `--`, block comments `--[[ ]]` and `--[==[ ]==]`;
- an optional leading `return` or `rewards =`;
- one table of `name = number` pairs, separated by `,` or `;`, with an
  optional separator after the last pair;
- decimal numbers with an optional minus sign, a fraction and an exponent:
  `-2`, `0.5`, `.5`, `3.`, `1e-3`;
- an empty table, which changes nothing.

Refused: an unknown key, a key given twice, an upper-case key, a nested table,
a string, a function call, arithmetic, hexadecimal numbers, `local`, anything
after the closing brace, and `+1.0`. A file that starts with `return` or
`rewards =` is also valid Lua, and Lua has no unary plus. The bare table,
without either word, is read here although Lua itself would not take it.

### Checking a file

```bash
python -m sdk.rewards examples/rewards.lua
```

```
examples/rewards.lua: 9 of 9 weights set, 1 different from the default

  weight         knob                   default      file  in force
  capture        REWARD_CAPTURE               3         3         3
  capture_cover  REWARD_CAPTURE_COVER         2         2         2
  defence        REWARD_DEFENCE               1         1         1
  draw           REWARD_DRAW                 -1        -1        -1
  hit            REWARD_HIT                   1         1         1
  hit_taken      REWARD_HIT_TAKEN            -1        -1        -1
  loss           REWARD_LOSS                 -1        -1        -1
  timeout        REWARD_TIMEOUT               0        -2        -2  changed
  win            REWARD_WIN                   2         2         2
```

A mistake is found in a second and not after the engine has started:

```
bad.lua: line 3: unknown weight 'camping'; expected one of capture, capture_cover, defence, draw, hit, hit_taken, loss, timeout, win
bad.lua: line 2: expected a number for hit, found '+' (Lua has no unary plus, and arithmetic is not read: write the number itself)
bad.lua: line 2: expected ',' or '}' after the value of hit, found '*' (arithmetic: write the result as one number)
```

### The weights belong to a start, not to the run

A run does not remember its weights. They are whatever the command line of
this start gave: `--rewards FILE`, and `--reward NAME=VALUE` on top of it for
one weight at a time. A resume without either trains on the defaults, and says
so:

```
Resumed without --rewards and without --reward: from here the run trains on the default weights, whatever it was started with
[WARNING] The reward weights are not those of this run's last start: timeout was -2 and is now 0. A run does not remember its weights; pass the same --rewards file to keep them
```

Pass the same file on every start of a run.

## sdk.train: train a team

```bash
python -m sdk.train --bundle my-bundle --run runs/my --learn "MY TEAM" --against WHITE BLACK \
    --rewards my-bundle/rewards.lua --arenas 16 --threads 4 --steps 2000000
```

```
New run: 15 checkpoints copied from .../my-bundle to .../runs/my/checkpoints
Reward weights, the unmarked at their defaults: capture 3, capture_cover 2, defence 1, draw -1, hit 1, hit_taken -1, loss -1, timeout -2 (my-bundle/rewards.lua), win 2
Device cpu, 15 models, 16 arenas, learning MY TEAM against WHITE TEAM, BLACK TEAM, maps: all themes
MY TEAM entropy scales: Ada 1.10, Bob 0.70, Cy 0.95, Dee 1.20, Eve 1.05
Checkpoints at 0.00M steps (0 updates): 15 of 15 models restored
The curriculum follows the learners' own 2355.45M steps
Phase: full arena (30x20m, 18 obstacles, 60s, hold 5.0s, autofire=off), mode: 5v5, maps: all themes
upd     1 | steps    0.02M |    2,708/s | avg 35.00 | hr 1.00 | dpm 35.00 | acc  8.4% | matches       0 | hits      35 | cap     0 | kl 0.0044 | clip 0.05 | vloss 0.226 | ent 0.006
...
Match MY TEAM 4:7 WHITE TEAM (11 rounds: time 0, elimination 8, capture 3, draws 0)
...
Reward per player per round over updates 91-95: MY TEAM +0.87
--steps 2000000 reached; stopping
Checkpoint at update 98, 2.01M steps
Stopped at 2.01M steps, update 98
```

That run took just under three minutes. It uses 16 arenas and not the default
64 because it is short: see [how many arenas](#how-many-arenas-for-how-many-steps).

The `--learn` teams learn. Every other team on the field plays with the
weights it was given and never takes a gradient step.

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--bundle DIR` | required | the bundle; only read |
| `--run DIR` | required | the run directory; created if missing, and never a bundle or inside one |
| `--learn TEAM [TEAM ...]` | required | the teams that learn; any case, and `WHITE` means `WHITE TEAM` |
| `--against TEAM [...]` | every other team in the bundle | the opponents |
| `--arenas N` | 64 | arenas simulated side by side |
| `--hours H` | none | stop after this long |
| `--steps N` | none | stop after this many agent-steps in this process; counted afresh on every start |
| `--device` | `cuda` if torch sees one, else `cpu` | the torch device |
| `--gpu-memory FRACTION` | none | cap this process's share of the GPU's memory, for a shared card |
| `--threads N` | the CPUs, 16 at most | threads for the simulation and for torch |
| `--seed N` | 0 | seeds the maps, the lineups, the ends, the sampled actions and the order each update reads its rollout in; the same seed, weights, flags and thread count repeat a run |
| `--theme KEY` | all | train on one map theme: `warehouse`, `forest`, `canyon`, `snowbase`, `neon` |
| `--checkpoint-every N` | 10 | save every N updates, and always on exit |
| `--rewards FILE` | none | a reward weights file |
| `--reward NAME=VALUE` | none | override one weight, named as in the file (`timeout=-2`) or as the constant (`REWARD_TIMEOUT=-2`); repeatable, and it wins over the file |
| `--restart` | off | throw the run's checkpoints and score files away and copy the bundle again |
| `--play-only` | off | play without learning or saving; see [Measuring a team](#measuring-a-team) |

With `--hours` and `--steps` both given, whichever comes first stops the run.
With neither, it runs until it is stopped.

`--against` matters as soon as your bundle holds more than the three teams.
The default is every other team, so a fresh team you added for another
experiment becomes an opponent of every run that does not name its opponents:

```bash
python -m sdk.train --bundle my-bundle --run runs/win3 --learn "MY TEAM" --against WHITE BLACK \
    --rewards my-bundle/rewards.lua --reward REWARD_WIN=3 --threads 4 --steps 200000
```

```
Reward weights, the unmarked at their defaults: capture 3, capture_cover 2, defence 1, draw -1, hit 1, hit_taken -1, loss -1, timeout -2 (my-bundle/rewards.lua), win 3 (--reward)
```

### How many arenas for how many steps

A match takes about 35,000 agent-steps of its arena, and every start begins
all its matches afresh: the ones still being played when it stops are not
counted. A start that gives each arena fewer than about 100,000 steps
therefore records only its shortest matches, which are the one-sided ones,
and its STANDINGS favour whoever wins fast. Two million steps at 64 arenas
are 32,000 for each: in such a run a team that draws level with BLACK over
whole matches showed 0-12 against it.

So for a start of a few million steps use `--arenas 16`, and for a real
session, at the default 64 or more, give every start at least 6.4 million
steps, about four minutes. The per-player `W/D/L` in PLAYERS counts every
round as it ends and does not have this bias.

### How the matches are drawn

Each arena plays whole matches. For each match it draws a learner team at
random and an opponent from `--against` other than the learner itself, and
keeps that pair until the match is decided. Every round the two teams take a
random end of the field, and each team's five are shuffled along it. A model
plays only with its own team.

Several teams may learn at once (`--learn A B`). They then also meet each
other, and they share one curriculum clock.

### Stopping and resuming

Stop a run with Ctrl-C or SIGTERM. It finishes the update in progress, saves
and exits 0. A second Ctrl-C saves at once. A closed terminal or a dropped
ssh (SIGHUP) is a stop like the others, never cut short: the run finishes its
update, saves and exits instead of dying with everything since its last save.
Under `nohup` the hangup stays ignored, and the run goes on.

```
SIGINT: stopping after this update and saving
Checkpoint at update 11, 0.11M steps
Stopped at 0.11M steps, update 11
```

Run the same command again to resume:

```
Resuming the run in .../runs/my/checkpoints
Resuming the tables from .../runs/my/status.json: 45 matches played
Checkpoints at 2.01M steps (98 updates): 15 of 15 models restored
```

- A resume carries on from the step count, the curriculum phase and the
  optimiser state the run last saved, and the dashboard's tables carry on with
  it. The matches that were being played when the run stopped are not carried:
  every arena starts a new match.
- The tables carry the teams that are on the field in this start. A team left
  out of a start, by a narrower `--against`, loses its rows for good, while
  `matches.jsonl` keeps its matches.
- A run belongs to the teams it was created to train. A resume with another
  `--learn` is accepted with a warning, and that run's counters and tables
  then describe two teams as one. Give each team a run directory of its own.
- The bundle is not copied over the run's own progress. What a resume does
  copy is what the run does not hold at all: the checkpoints of a team added
  to the bundle since the run began.
- `--restart` deletes the run's checkpoints, `status.json`, `matches.jsonl`
  and `progress.jsonl` and starts again from the bundle.
- While it runs, `sdk.train` holds a lock on the run directory. A second one
  on the same run is refused, and the lock file keeps saying who holds it:
  `{"pid": 4193378, "started_at": 1794843479.8}`.
- A directory with checkpoints in it and no `sdk-run.json` beside them is not
  a run of this SDK. It is refused, for a resume and for `--restart` alike.

### The log line of an update

```
upd    98 | steps    2.01M |   12,418/s | avg  1.33 | hr 1.00 | dpm 0.68 | acc 10.5% | matches     190 | hits     130 | cap     4 | kl 0.0047 | clip 0.05 | vloss 0.216 | ent 0.015
```

It covers every model on the field, both sides together, over the last few
updates, so it says what the play looks like and not who is winning.

| Column | Meaning |
|---|---|
| `upd`, `steps`, `/s` | the update number, the run's agent-steps, agent-steps a second since this start |
| `avg` | points one player scores in one round |
| `hr` | hits per death; always 1.00, because every hit is somebody's death |
| `dpm` | deaths per player per round |
| `acc` | hits per shot |
| `matches` | rounds played, counted once per player |
| `hits`, `cap` | hits and capture credits in the window |
| `kl`, `clip` | how far the update moved the policy; both near zero for many updates means nothing is being learned |
| `vloss`, `ent` | the value loss and the policy entropy |

In the first lines of a start, before a round has ended, `avg` and `dpm` are
totals and not averages: `avg 35.00` above is 35 points over no finished
round.

An agent-step is one decision of one player. An update is 128 decisions of
each of the ten players in every arena: 81,920 agent-steps at 64 arenas.

Who is winning is in the `Match` lines, and in the dashboard.

## The run directory

| File | What it is |
|---|---|
| `checkpoints/` | the run's checkpoints and manifest: a copy of the bundle's on the first start, the run's own after that |
| `status.json` | the live state for the dashboard, rewritten about once a second |
| `matches.jsonl` | one line per finished match |
| `progress.jsonl` | one line per checkpoint save |
| `sdk-run.json` | the marker that says this is an SDK run, with the bundle it was started from |
| `lock` | held while `sdk.train` runs; holds the pid of the process that holds it and when it took it |
| `tui.json`, `rewards.lua`, `train.log` | written by [sdk.tui](#sdktui-one-screen-for-all-of-it) when it is used on the run |

`sdk-run.json` stores the absolute path of the bundle. `sdk.demo` and
`sdk.export_team` use it to find the team's description. If the bundle has
moved, give them `--bundle`.

Every save rewrites every fielded model's `.pt` in `checkpoints/`, the frozen
ones included. Their weights do not change. The manifest there names only the
models that play in the run.

### status.json

```json
{
  "version": 1, "started_at": 0.0, "updated_at": 0.0,
  "device": "cpu", "arenas": 64, "play_only": false,
  "learners": ["MY TEAM"], "opponents": ["WHITE TEAM", "BLACK TEAM"],
  "teams": {"MY TEAM": {"colour": "#12b5a6", "members": ["Ada", "Bob", "Cy", "Dee", "Eve"]}},
  "rewards": {"REWARD_TIMEOUT": -2.0, "REWARD_WIN": 2.0},
  "reward_defaults": {"REWARD_TIMEOUT": 0.0, "REWARD_WIN": 2.0},
  "reward_sources": {"REWARD_TIMEOUT": "my-bundle/rewards.lua"},
  "steps": 4096000, "updates": 50, "sps": 31467.9, "phase": "full arena",
  "session_steps": 2048000, "run_steps": 4096000,
  "team_steps": {"MY TEAM": 2357499495, "WHITE TEAM": 2355451495, "BLACK TEAM": 2449122392},
  "reward_window": {"from_update": 46, "to_update": 50,
                    "teams": {"MY TEAM": {"reward": 253.331, "player_rounds": 640}}},
  "matches": {"played": 44, "recent": []},
  "standings": {"MY TEAM": {"matches": 44, "won": 25, "lost": 19, "unfinished": 0, "rounds": 451,
                            "time": 0, "elimination": 318, "capture": 133, "draws": 5}},
  "players": {"Ada": {"team": "MY TEAM", "rounds": 1241, "wins": 609, "losses": 611, "draws": 21,
                      "hits": 1229, "deaths": 1076, "shots": 7862, "captures": 102, "points": 2179}}
}
```

The example is a real file after a resume, shortened: `teams`, `standings`
and `players` hold every team and model on the field, the three reward tables
hold every weight, and `matches.recent` is not empty.

- `started_at` and `updated_at` are epoch seconds.
- `rewards` is every weight in force, `reward_defaults` what each would be
  with nothing given, and `reward_sources` where each given one came from: the
  `--rewards` path as typed, or `--reward`.
- `steps` is the counter the checkpoints carry. `run_steps` is what this run
  has trained over all its starts, and `session_steps` what it has trained
  since this start.
- `team_steps` is the sum of each team's members' own learning steps. The
  learners' sum is the curriculum clock, and a frozen team's never moves.
- `reward_window` is what the learners were paid over the last five updates:
  `reward / player_rounds` is the reward of one player for one round. It
  includes the fixed shaping terms. The first window of every start is not
  kept: it is paid for rounds that are still being played when it closes, and
  reads several times too high. So the key is absent until the second
  window of a start closes, ten updates into a new run, and in a `--play-only`
  run.
- `matches.recent` holds the last 20 finished matches, newest first, each in
  the shape of a `matches.jsonl` line.
- `standings` counts finished matches only: a team's `rounds`, `time`,
  `elimination`, `capture` and `draws` are those of its finished matches, and
  `matches` is `won + lost + unfinished`.
- `players` counts every round as it ends, so it runs ahead of `standings`.
  `rounds` is the number of rounds the model played, and `deaths` the number
  of times it was hit.

- `pid` is the process that wrote the file, and `args` every flag it was
  started with, by argparse name (`checkpoint_every`, `gpu_memory`), as it
  was given: `bundle` and `run` as typed, `learn` as the names were typed,
  while `learners` and `opponents` are the teams they resolved to.
- `stopped_at` and `stop_reason` are written once, at a clean end: the reason
  is `stopped` for a signal, or the budget, `--steps 30000 reached` or
  `--hours 2.0 reached`. A file with a `pid` and no `stopped_at` belongs to a
  trainer that is still running, or that was killed or crashed.

A reader must expect keys it does not know and treat a missing one as absent:
fields are only ever added.

### matches.jsonl and progress.jsonl

A `matches.jsonl` line:

```json
{"at": 1794836142.65, "steps": 4090880, "teams": ["MY TEAM", "WHITE TEAM"], "score": [7, 5],
 "rounds": 12, "time": 0, "elimination": 6, "capture": 6, "draws": 0, "finished": true}
```

`score` follows the order of `teams`. `time + elimination + capture` equals
`rounds`, and `draws` is counted among them. `finished` is false for a match
called off at 28 rounds.

A `progress.jsonl` line is written at every save: `at`, `steps`, `updates`,
`sps`, `phase`, the `standings` at that moment and the `reward_window`.

## The curriculum, and what to expect

A team does not start on the full field. The arena grows with the learners'
own experience:

| Phase | Until (learners' steps) | Arena | Round | Capture hold | Notes |
|---|---|---|---|---|---|
| close quarters | 10M | 16 x 11 m, 4 obstacles | 20 s | 1.5 s | shots are fired for the player |
| trigger handover | 30M | 20 x 14 m, 8 obstacles | 30 s | 2.5 s | |
| open field | 80M | 26 x 17 m, 14 obstacles | 45 s | 3.5 s | |
| full arena | from 80M on | 30 x 20 m, 18 obstacles | 60 s | 5.0 s | the real game |

The clock is the sum of the learners' own steps (`team_steps` in
`status.json`). Frozen opponents set no pace. The learning rate and the
entropy bonus are annealed on the same clock, down to a floor of a tenth of
the starting rate.

The learners take half of a run's agent-steps when one team learns, so the
full arena is about 160 million run steps away for a fresh team.

### A copied team

A team made with `--from WHITE` has 2,355 million steps on its clock. It
trains on the full arena, at the floor learning rate, and it starts exactly as
strong as WHITE: against WHITE it wins about half of its rounds.

What it becomes depends on what you pay it for, and that takes time. Updates
are small on purpose. Do not expect the first millions of steps to improve a
veteran. In the runs for this document a copy trained for 4.1 million steps on
the default weights stayed where it was, at 49.7% of its decided rounds
against WHITE. Trained as long with the example weights it won 46.8% (see
[how much is noise](#how-much-is-noise)): a team that is paid for something
new first gets worse at what it knew. Think in tens of millions of steps and
in hours, and measure.

### A fresh team

```bash
python -m sdk.train --bundle my-bundle --run runs/rookies --learn ROOKIES --against WHITE BLACK \
    --threads 4 --steps 2000000
```

```
The curriculum follows the learners' own 0.00M steps
Phase: close quarters (16x11m, 4 obstacles, 20s, hold 1.5s, autofire=on), mode: 5v5, maps: all themes
Match ROOKIES 0:7 WHITE TEAM (7 rounds: time 0, elimination 7, capture 0, draws 0)
```

In its first two million steps a fresh team lost all 353 of its matches and
scored 1.35 points a round to the veterans' 9.96. That is what the start looks
like, not a fault. It reaches the full arena after about 160 million run
steps. That is about two hours if the later phases keep the 26,000 steps a
second measured in the first one, and the team is then still far behind
opponents with over two billion steps of their own. A fresh team is the
long road. Copying a shipped team and changing what it is paid for is the
short one.

### Speed

Measured on one desktop CPU, one team learning against the two shipped teams,
on the full arena:

| `--arenas` | `--threads` | agent-steps a second |
|---|---|---|
| 8 | 1 | 6,400 |
| 8 | 4 | 8,400 |
| 16 | 4 | 12,000 |
| 32 | 4 | 24,000 |
| 64 | 1 | 19,000 |
| 64 | 2 | 26,000 |
| 64 | 4 | 33,000 |
| 128 | 4 | 45,000 |
| 256 | 4 | 46,000 |

- More arenas cost little up to about 128, because much of an update's time is
  spent asking each network for its move, and that is done once per step
  however many arenas there are.
- Threads help less than arenas: four threads are 1.7 times one thread at 64
  arenas. More than four were not measured.
- At 64 arenas and 4 threads, one million steps take about half a minute and
  one hundred million about an hour.
- More arenas also mean more rounds behind every update. Few arenas give
  noisy updates: the reward figure of a five-update window moved between -0.26
  and +1.04 at 16 arenas.
- Peak memory was about 1.0 GB up to 64 arenas, 1.3 GB at 128 and 1.7 GB at
  256.

No GPU was used for any of this. `--device cuda` is there, and the default
picks a GPU when torch sees one.

## sdk.watch: the dashboard

```bash
python -m sdk.watch runs/my
```

It redraws every two seconds until Ctrl-C. `--refresh SECONDS` changes the
pace, and `--once` prints one frame and exits. When the output is not a
terminal, one plain frame is printed without colour.

```bash
python -m sdk.watch runs/my --once
```

```
SELFPLAYTV SDK  run my  steps 2,007,040  updates 98  sps 12,414
phase full arena  device cpu  arenas 16  up 3 min
learning MY TEAM (2,356.46M steps of its own)  against WHITE TEAM, BLACK TEAM
rewards  timeout -2 (my-bundle/rewards.lua)
paid     MY TEAM +0.87 per player per round  (updates 91-95)

RECENT MATCHES  (45 played)
     MY TEAM   7:3   WHITE TEAM   10 rounds  time  0  elim  6  cap  4
     MY TEAM   7:4   BLACK TEAM   12 rounds  time  2  elim 10  cap  0  draws 1
     ...

STANDINGS
  team        matches        W-L-U    rounds   time%   elim%    cap%   draw%  pts/rnd  pts/match
  MY TEAM          45      22-23-0       497    3.4%   78.3%   18.3%    2.2%     6.37       70.3
  WHITE TEAM       23      12-11-0       260    0.8%   68.1%   31.2%    1.9%     6.83       77.2
  BLACK TEAM       22      11-11-0       237    6.3%   89.5%    4.2%    2.5%     6.10       65.7

PLAYERS
    nick      rounds            W/D/L     hits   deaths    K/D     acc   caps    points  pts/rnd
  MY TEAM
    Ada          577       280/11/286      507      506   1.00   16.0%     35       913     1.58
    Bob          577       280/11/286      605      413   1.46   17.9%      5       951     1.65
    Cy           577       280/11/286      281      356   0.79    9.9%      1       619     1.07
    Dee          577       280/11/286      237      351   0.68    4.9%     18       609     1.06
    Eve          577       280/11/286      245      354   0.69    3.4%      0       581     1.01
  WHITE TEAM
    Hiroshi      297        146/5/146      271      237   1.14   12.5%     40       543     1.83
    ...
```

It reads `RUN/status.json` and nothing else, and needs only the standard
library: no torch, no numpy, no engine. It can run on another machine against
a run directory copied or synced from the one that trains.

The frame is 96 columns wide and up to 46 lines high. In a smaller terminal
the right-hand columns and the bottom rows are cut, and the last line shown
says so. `python -m sdk.watch runs/my --once | less -S` shows all of it. A run
directory that does not exist is said to be missing, and one that has no
`status.json` yet is waited for.

A team's name is written in the team's colour, unless that colour is nearly
black or nearly white: the terminal's background is not known, so such a name
is written bold in the terminal's own colour.

**The header.**

- `steps` is what this run has trained. After a resume it also shows what this
  start has added: `steps 4,096,000 (+2,048,000 this start)`.
- `learning MY TEAM (2,356.48M steps of its own)` is the curriculum clock of
  the learners. A `--play-only` run says `playing ... (play only: nothing
  learns)`.
- `rewards` lists only the weights that differ from the defaults, with where
  each came from. A run on the defaults has no such line.
- `paid` is the reward: what one player of the learner team was paid for one
  round, averaged over the last five updates. It appears when the second
  window of a start closes, ten updates into a new run: the first window is
  not kept, because it reads several times too high.
- A status file that has not been rewritten for 15 seconds is called out:
  `status is 59 s old`. A run that has stopped shows it from then on.

**RECENT MATCHES.** The last ten finished matches, newest first: the score,
the number of rounds and how they ended.

**STANDINGS.** One row per team, over its finished matches. The matches still
being played when a start ends are not counted, so a start shorter than about
100,000 steps for every arena records its shortest matches and little else,
and `W-L` favours whoever wins fast: see
[how many arenas](#how-many-arenas-for-how-many-steps). In such a run read
`W/D/L` in PLAYERS, or measure with `sdk.demo --summary`.

| Column | Meaning |
|---|---|
| `matches`, `W-L-U` | matches finished: won, lost, unfinished |
| `rounds` | rounds of those matches |
| `time%` | the share of them that ended on the clock; highlighted, because it is the waiting measure |
| `elim%`, `cap%` | the share that ended in an elimination, in a capture |
| `draw%` | the share that scored for nobody |
| `pts/rnd` | points the team's five players scored together, per round the team played |
| `pts/match` | `pts/rnd` times the team's average rounds per finished match |

`pts/rnd` is counted from every round played, not only those of finished
matches. It is shown, here and in PLAYERS, once the row has played five rounds
for every arena of the run, and is a dash until then: points are counted as
they are scored and rounds as they end, so with a round or two finished in
each arena the figure reads up to twice what it settles at. `pts/match` is an
estimate, because points are not recorded per match.

**PLAYERS.** One row per model, learners first.

| Column | Meaning |
|---|---|
| `rounds`, `W/D/L` | rounds played, and the team's result in them |
| `hits`, `deaths`, `K/D` | hits scored, times hit, and hits per death |
| `acc` | hits per shot |
| `caps` | capture credits: one to each player standing on the enemy strip as it falls, so a capture by three players is three |
| `points`, `pts/rnd` | scoreboard points, and points per round for this one player |

## sdk.demo: matches in the terminal

```bash
python -m sdk.demo --run runs/my "MY TEAM" WHITE
```

One match, first to seven, drawn in the terminal from above at real time. A
match takes about five minutes. The field is drawn with half-block characters,
two pixels to a character cell: the floor and the obstacles of the map, the
two spawn strips, every player as a block in the team's colour with a nose
where it faces, the dead as a dim cross, and the paint in the air. Five lines
of text sit under the field:

```
MY TEAM  3 : 1  WHITE TEAM    match 1 of 1
round 5   0:12 of 1:00   Forest   alive 3 v 4
points this match   40 v 20
average a round     9.75 v 4.50   over 4 rounds
WHITE TEAM is taking the strip: 1.1 of 5.0 s
```

| Flag | Default | Meaning |
|---|---|---|
| `TEAM TEAM` | required | two different teams; their order is the order of every score |
| `--bundle DIR` | the bundle `--run` was started from, else `bundle` | names the teams |
| `--run DIR` | none | take the weights from this run's checkpoints, as training has them now |
| `--matches N` | 1 | how many matches to play |
| `--speed X` | 1 | times real time |
| `--theme KEY` | all | one map theme; by default a new map every round |
| `--seed N` | random, and logged | play the same matches again |
| `--summary` | off | draw nothing: play as fast as possible and print the result |
| `--graphics MODE` | `auto` | `kitty`, `blocks` or `auto`; see [Graphics](#graphics) |

- Without `--run` the weights are the bundle's:
  `python -m sdk.demo WHITE BLACK --matches 5 --summary` plays the two shipped
  teams.
- A match is always on the full arena, whatever the curriculum would give a
  young team: a demo shows the game the team is being trained for.
- Nothing is written, to the run or to the bundle. It is safe to run while
  the same run is training.
- It always plays on the CPU, on one thread.
- Ctrl-C stops it, restores the terminal and prints the summary of what was
  played.
- The terminal must be at least 45 columns by 20 lines. A terminal that does
  not announce 24-bit colour (`COLORTERM=truecolor`) gets the picture in 256
  colours.
- The points line counts the hits of the round in progress, and the average
  only finished rounds. They agree at the end of every round.

With `--summary`, or whenever the output is not a terminal, nothing is drawn.
The matches are played side by side, up to 32 at a time, and the result is
printed. Twenty matches take about half a minute:

```bash
python -m sdk.demo --run runs/my "MY TEAM" WHITE --matches 20 --summary --seed 7
```

```
Match MY TEAM 7:4 WHITE TEAM (11 rounds: time 0, elimination 6, capture 5, draws 0)
...
229 rounds: time 1, elimination 152, capture 76, draws 3
Rounds: MY TEAM 112:114 WHITE TEAM (49.6% of the decided)
Matches: MY TEAM 12:8 WHITE TEAM

team        points  a round  a match  hits  hit  shots  captures
MY TEAM       1527     6.67    76.35   706  703   7200        38
WHITE TEAM    1554     6.79    77.70   703  706   7514        43

player         team  points  a round  hits  hit  shots  captures
Ada         MY TEAM     443     1.93   232  177   1568        31
Bob         MY TEAM     327     1.43   174  151    786         2
...
```

- The `rounds:` line counts both teams together: 76 of the 229 rounds ended
  in a capture, by either side.
- `Rounds:` is the rounds each team won, and the first team's share of the
  rounds that had a winner. It is the round win rate a team is judged by.
- `a round` is points divided by the rounds played, `a match` points divided
  by the matches, and `hit` the number of times the player was hit.
- `captures` is capture credits: one to each player standing on the enemy
  strip as it falls, so a capture by three players is three. It is the figure
  that says whether a team captures more than it did.

The log goes to standard error and the result to standard output. The log
repeats each `Match` line as the match ends, so the two read together show
every match twice.

A seed replays the same matches exactly, as long as the weights and the number
of matches are the same. Drawn matches are played one after another and
summary matches side by side, so a seed gives the same matches in both ways
only for `--matches 1`.

### Graphics

A terminal that speaks the kitty graphics protocol can be sent the picture as
real pixels instead of half blocks: the same field, players and paint, sized
to the same rectangle of cells, with the five lines of text under it as
before.

- `--graphics blocks` draws half blocks and never sends a graphics escape.
- `--graphics kitty` sends pixels without asking. Inside tmux or screen it
  shows no picture and leaves the escapes in the status line: do not force it
  there.
- `--graphics auto`, the default, asks the terminal once, after taking the
  screen: kitty if it answers that it can, half blocks otherwise. It does not
  ask, and draws half blocks, when `TMUX` or `STY` is set, when `TERM` starts
  with `tmux` or `screen`, and when `TERM` is unset, `dumb` or `linux`. A
  terminal that does not answer within half a second gets half blocks.

The picture is sent at most 960 pixels wide, and the terminal scales it to
the cells. A frame costs far more than in half blocks: about 30 kB at 600 by
400 pixels against about 1 kB, so over a slow link use `--graphics blocks`.
This was checked in kitty 0.32.2 on Linux, and in a stream decoded back into
pictures; other terminals that implement the protocol were not tried.

## Measuring a team

Training results are noisy, and the training tables mix every stage of a team
that was changing while they were filled. To judge a team, play it without
learning.

**Quick: `sdk.demo --summary`.** It reads the weights of a run or of a bundle
and writes nothing:

```bash
python -m sdk.demo --run runs/my "MY TEAM" WHITE --matches 20 --summary --seed 7
python -m sdk.demo --bundle my-bundle "MY TEAM" WHITE --matches 20 --summary --seed 7
```

The first line plays the team as the run has trained it, the second as the
bundle holds it, which is how it started. Twenty matches are about 230 rounds.
`--matches 50` is about 550, the size the noise figures below are given for,
and takes about a minute.

**Larger: `sdk.train --play-only`.** It plays exactly the matches a training
run would, on the full arena, with no update and no save. `status.json` and
`matches.jsonl` are the result, and `sdk.watch` shows them:

```bash
python -m sdk.train --bundle my-bundle --run runs/measure-before --learn "MY TEAM" --against WHITE \
    --play-only --arenas 16 --threads 4 --steps 1500000 --seed 11
python -m sdk.watch runs/measure-before --once
```

```
SELFPLAYTV SDK  run measure-before  steps 1,515,520  updates 74  sps 27,664
phase full arena  device cpu  arenas 16  up 55 s  status is 59 s old
playing MY TEAM (2,355.45M steps of its own)  against WHITE TEAM  (play only: nothing learns)

STANDINGS
  team        matches        W-L-U    rounds   time%   elim%    cap%   draw%  pts/rnd  pts/match
  MY TEAM          42      22-20-0       459    0.4%   65.4%   34.2%    2.6%     6.99       76.4
  WHITE TEAM       42      20-22-0       459    0.4%   65.4%   34.2%    2.6%     6.59       72.0
```

That is a minute for about 40 matches. A match needs about 35,000 agent-steps
of its arena, so give `--steps` about 100,000 for every arena. At 19,000 for
every arena not one match had finished.

- Give a measurement **a run directory of its own**. `--play-only` on a
  training run's directory would add its matches to that run's tables.
- A new run directory starts from the weights **of the bundle**. To measure
  what a run has trained, make a bundle that holds it: copy your bundle and
  put the trained team's five files over the copy.

  ```bash
  cp -r my-bundle my-bundle-trained
  cp runs/my/checkpoints/{Ada,Bob,Cy,Dee,Eve}.pt my-bundle-trained/
  python -m sdk.train --bundle my-bundle-trained --run runs/measure-after --learn "MY TEAM" --against WHITE \
      --play-only --arenas 16 --threads 4 --steps 1500000 --seed 11
  ```

  Copy the `.pt` files only. The run's `manifest.json` names just the models
  that played in it.
- The same `--seed`, `--arenas` and weights replay exactly the same matches.
  The seed fixes the maps and the first draws. Once two teams act differently
  the matches drift apart, so a seed does not pair a "before" with an "after"
  round by round.

### How much is noise

A team was measured against an identical copy of itself, about 40 matches and
530 rounds at a time, ten times:

| Measurement | Matches | Share of decided rounds | Points a round |
|---|---|---|---|
| 1 | 14-26 | 45.1% | 6.40 v 7.19 |
| 2 | 21-20 | 49.3% | 6.87 v 6.85 |
| 3 | 21-18 | 51.8% | 6.99 v 6.61 |
| 4 | 20-18 | 50.1% | 6.91 v 6.74 |
| 5 | 23-19 | 49.5% | 6.71 v 6.98 |
| 6 | 23-17 | 50.6% | 6.81 v 6.80 |
| 7 | 23-18 | 50.8% | 6.85 v 6.65 |
| 8 | 22-20 | 51.7% | 6.99 v 6.59 |
| 9 | 23-19 | 52.8% | 7.06 v 6.64 |
| 10 | 21-22 | 47.6% | 6.57 v 7.08 |

Two equal teams can finish 14-26 over forty matches. So:

- **Matches won is the noisiest figure.** Read the share of rounds and the
  points a round first.
- Over about 500 rounds, a difference of less than 5 points of round share, or
  of less than about 0.8 points a round, is not evidence of anything.
- To see a small change, play more: 10 million `--play-only` steps are about
  3,500 rounds and take six minutes at 16 arenas. Or repeat the measurement on
  three or four seeds and add them up.

The same team after 4.1 million training steps, measured four times the same
way:

| Trained with | Matches | Share of decided rounds | Together |
|---|---|---|---|
| the default weights | 25-22, 23-19, 21-24, 26-18 | 50.5%, 50.3%, 47.6%, 50.6% | 1,090 of 2,192 rounds, 49.7% |
| the example weights (`timeout = -2`) | 11-31, 17-22, 18-20, 15-25 | 43.1%, 47.3%, 50.4%, 46.4% | 979 of 2,094 rounds, 46.8% |

No single measurement says much. Four together do: the default weights left
the team where it was, and the changed weight cost it about three points in
its first four million steps.

## sdk.export_team: one archive

```bash
python -m sdk.export_team --run runs/my --team "MY TEAM"
```

```
Team: MY TEAM (#12b5a6)
Players: Ada, Bob, Cy, Dee, Eve
Steps per model: Ada 440.00M, Bob 445.98M, Cy 491.85M, Dee 487.51M, Eve 492.16M
Weights: the run in .../runs/my/checkpoints, saved at 4.10M steps, update 50
Archive: .../exports/MY_TEAM_20261116133545.tar.gz, 12.5 MB
```

Those five lines go to standard error. Standard output carries one line, the
absolute path of the archive, so a script can take it:

```bash
ARCHIVE="$(python -m sdk.export_team --run runs/my --team "MY TEAM")"
```

The team can be renamed on the way out:

```bash
python -m sdk.export_team --run runs/my --team "MY TEAM" --name "TEAL TEAM" \
    --nicks Uno Dos Tres Cuatro Cinco --colour "#0fa3b1"
```

```
Team: TEAL TEAM (#0fa3b1), exported from MY TEAM
Players: Uno (was Ada), Dos (was Bob), Tres (was Cy), Cuatro (was Dee), Cinco (was Eve)
```

| Flag | Default | Meaning |
|---|---|---|
| `--team TEAM` | required | the team to export |
| `--run DIR` | none | take the weights this run has trained, as it last saved them |
| `--bundle DIR` | with `--run`: the bundle the run was started from | the bundle that describes the team; alone, also where the weights come from |
| `--name NAME` | the team's own | export it under this team name |
| `--nicks A B C D E` | the team's own | export its players under these nicknames, in the team's order |
| `--colour "#rrggbb"` | the team's own | export it with this colour; `--color` is the same flag |
| `--out DIR` | `exports` | the directory the archive is written to |

- At least one of `--run` and `--bundle` is needed. `--bundle` alone exports a
  team with the weights its bundle holds:
  `python -m sdk.export_team --bundle bundle --team WHITE`.
- A team that did not play in a run cannot be exported from it. Export it from
  its bundle.
- The names follow the same rules as in `sdk.new_team`, and a new name or
  nickname may not be one that another team of the bundle has: an archive
  called `WHITE TEAM`, or holding a `Hiroshi.pt`, is the shipped team's alone
  to be. The team's own nicknames may be kept or handed round.
- Nothing in the run or the bundle is written. The archive is built in memory,
  every checkpoint in it is read back with `weights_only=True`, and only then
  is the file written. An existing file is never written over.
- Exporting while the same run is training reads its last save. An export
  that catches the run in the middle of a save reads once more, and refuses
  rather than pack checkpoints of two different saves under one manifest.

### The archive

`<TEAM_NAME>_<YYYYMMDDHHMMSS>.tar.gz`, local time. In the file name every run
of characters outside `A-Z a-z 0-9 _ -` in the team's name becomes `_`. It is
a gzipped tar with flat members, in the layout of a SELFPLAYTV run archive:

| Member | What it is |
|---|---|
| `<nick>.pt`, five of them, in name order | each player's whole checkpoint: weights, optimiser state, return normaliser, `model_steps`; `nick` is the exported nickname |
| `manifest.json` | the source's manifest with `nicks` narrowed to the five, in the team's order |
| `stats.json` | `{"format": 1, "exported_at": ..., "manifest": {...}, "stats": {}}`: no counters travel with a team |
| `teams.json` | the team alone: its name, colour, members in order and their entropy scales |

A rename changes the file names, the `nick` inside each checkpoint, the
manifest and `teams.json` together. The members carry no user or group name of
the machine that made them.

`steps` and `updates` in the manifest are those of the run the weights were
saved by: an archive from a run says how far that run went, and one from the
shipped bundle says 0. Each model's whole experience is its `model_steps`.

Unpacked into an empty directory, an archive is a bundle of one team. There is
no command that merges it into another bundle.

## sdk.tui: one screen for all of it

```bash
python -m sdk.tui runs/my
```

A full-screen program for a person at a terminal. Keys start and stop the
training, change the settings and the reward weights of the next start, show
the dashboard and the log, play a match of the saved weights, make a team and
export one. It trains nothing and draws nothing itself: the trainer is
`python -m sdk.train`, started as a process of its own; the match is
`python -m sdk.demo` on a pseudo-terminal; a team and an export are
`python -m sdk.new_team` and `python -m sdk.export_team`. Everything it shows
is read again from the run directory every half second, so leaving it, losing
the terminal or opening it again the next day never touches the trainer.

| Argument | Default | Meaning |
|---|---|---|
| `RUN` | the most recently changed directory under `runs/` that holds `tui.json` or `sdk-run.json`; with none, `runs/` and the name of the team that learns by default (`runs/black` with only the shipped bundle) | the run. Opening it creates nothing: the directory is made by the first thing written into it |
| `--graphics MODE` | `auto` | how a match is drawn, as for [sdk.demo](#graphics); the TUI asks the terminal once and tells the demo |

It needs a POSIX terminal (Linux, macOS, WSL) of at least 60 columns by 12
lines, and refuses with status 2 when stdin or stdout is not a terminal. The
view fills the screen above three rows: the state of the training, the
lineup (what trains now, or what the next start will be, or the last line of
the log), and the keys, a question or the text being typed. At 100 by 30,
while training:

```
STANDINGS
  team        matches        W-L-U    rounds   time%   elim%    cap%   draw%  pts/rnd  pts/match
  NEON FOXES        4        2-2-0        44    0.0%   61.4%   38.6%    2.3%     7.73       85.0
  WHITE TEAM        4        2-2-0        44    0.0%   61.4%   38.6%    2.3%     7.65       84.2
  BLACK TEAM        0        0-0-0         0       -       -       -       -     6.83          -
rewards  timeout -2 (runs/neon-foxes/rewards.lua)
paid     NEON FOXES +0.64 per player per round  (updates 66-70)
phase    full arena  NEON FOXES (2,355.97M steps of its own)

RECENT MATCHES  (4 played)
  NEON FOXES   5:7   WHITE TEAM   12 rounds  time  0  elim  7  cap  5
  NEON FOXES   6:7   WHITE TEAM   13 rounds  time  0  elim  9  cap  4
  NEON FOXES   7:3   WHITE TEAM   11 rounds  time  0  elim  7  cap  4  draws 1
  NEON FOXES   7:1   WHITE TEAM    8 rounds  time  0  elim  4  cap  4

PLAYERS
    nick      rounds            W/D/L     hits   deaths    K/D     acc   caps    points  pts/rnd
  NEON FOXES
    Kit          268        137/8/123      398      387   1.03   14.6%     23       615     2.29
    Vix          268        137/8/123      315      247   1.28   15.4%      0       486     1.81
    Rey          268        137/8/123      160      174   0.92    7.9%      0       331     1.24
    Tod          268        137/8/123      151      200   0.76    4.6%     13       348     1.30
    Fen          268        137/8/123      121      163   0.74    2.4%      0       292     1.09
  WHITE TEAM
    Hiroshi      147          70/5/72      215      174   1.24   12.6%     17       342     2.33
    Kenji        147          70/5/72      161      117   1.38   15.2%      1       256     1.74
... 9 more rows below: down, PgDn
 TRAINING  runs/neon-foxes  1.05M steps (+0.58M)  12,793/s  up 47 s  saved 8 s ago
 now NEON FOXES v WHITE TEAM, BLACK TEAM  2 threads  cpu  16 arenas  until stopped
T stop  D dash  M match  S settings  W weights  L log  N team  X export  Q quit
```

### Keys

| Key | Does |
|---|---|
| `T` | start training, confirmed with Enter; or stop it, confirmed with Y. While it stops, Y cuts the update short; it still saves |
| `R` | restart with the settings as they are now: stop, save, start again (Y). Only while training, and only when something differs from what is in force |
| `D` | the dashboard: the tables of `sdk.watch`, standings first |
| `M` | a match of the first learning team against another team of the bundle: Enter plays, Tab steps to the next opponent, Esc ends it |
| `S`, `W` | the settings and the reward weights of the next start. Enter changes the row under the cursor (a text opens on the bottom row; a team steps through learns, opponent and out), Backspace puts its default back. A weights file that does not parse is shown with its line, and only Backspace writes the defaults over it |
| `L` | the end of `RUN/train.log` |
| `N` | make a team: name, colour, five nicknames, a copy of a team or fresh networks; C runs `sdk.new_team`, and on success offers to open a run for the new team |
| `X` | export a team, renamed if asked; C runs `sdk.export_team` |
| `Q` | leave. The training goes on, and the last line printed says how to come back |
| Ctrl-C | leave when nothing trains; otherwise it asks whether to stop the training or leave; during a match it ends the match |
| Esc | close a question or a text, end a match, or go back to the dashboard |
| up, down, PgUp, PgDn | scroll, or move the cursor in a list |
| left, right | pan the dashboard and the log by eight columns, which is how the points columns are reached at 80 columns |

Start is confirmed with Enter and stop with Y, so a key pressed twice can
never start and then stop. Keys typed ahead of a question are dropped. A
question answered after the run has changed does nothing and says so: a stop
answered after the trainer has gone sends no signal.

### States

| State | Means |
|---|---|
| `NO RUN` | nothing has trained in this directory yet |
| `STARTING` | a trainer holds the run's lock and has not written its first status yet: importing, copying the bundle, compiling; a few seconds, longer on the first start on a machine |
| `TRAINING` | it trains; the row shows its steps, speed, uptime and the age of its last save, which a new run has only after its first |
| `STOPPING` | a stop was sent; it finishes the update in progress, then saves |
| `STOPPED` | it ended cleanly. The reason is shown when it was the budget (`--steps 30000 reached`), and `stopped before it began` for a stop sent while it was starting |
| `FAILED` | it ended any other way: with its exit status when this TUI started it, `ended without a clean stop` when only the files say so, and `ended before its first status` when it died while loading. The last line of its log is under it |

A bell rings when the training ends and this TUI did not ask it to.

Nothing reaches a running trainer: its threads, its weights and its teams are
fixed when it starts. A setting or a weight changed while it trains is for the
next start. S and W then show two columns, `next start` and `in force now`, a
row that differs is marked `* restart`, and the lineup row starts with
`N changed: R restarts`. What is in force is read from the trainer's own
`status.json` (`args`, `learners`, `opponents`, `rewards`), so it is right for
a trainer started by hand as well.

### Files

| File | Holds |
|---|---|
| `RUN/tui.json` | every setting of the next start, rewritten on each change, with the pid of the last trainer this run's TUI started and of the last stop it sent |
| `RUN/rewards.lua` | the nine reward weights, as an ordinary weights file: `python -m sdk.rewards RUN/rewards.lua` checks it. Every start from the TUI passes it as `--rewards`. A change in W rewrites it whole, without comments. Before the file exists W shows the weights of the run's `status.json` |
| `RUN/train.log` | the output of every trainer the TUI starts, each start after one line with its whole command; renamed to `train.log.1` at a start once it passes 8 MB |

A start from the TUI is exactly `python -m sdk.train` with `--bundle`,
`--run`, `--learn`, `--against`, `--arenas`, `--threads`, `--seed`,
`--checkpoint-every` and `--rewards RUN/rewards.lua`, and `--hours`,
`--steps`, `--device`, `--theme` and `--gpu-memory` when they are set: the
line in `train.log` is the command to run it by hand. It never passes
`--restart`, `--play-only` or `--reward`, which stay on the command line. A
run with no `tui.json` yet takes its first settings from the `args` of its
`status.json`, else from its `sdk-run.json`, else the defaults, which are
`sdk.train`'s own.

A trainer started by hand on the same run is shown, stopped and restarted
like one the TUI started; its log is in the terminal that started it, and the
TUI says so. Two TUIs may be open on the same run: they read the same files.

### Leaving, and losing the terminal

The trainer runs in a session of its own, with no terminal and its output in
a file. Q, a crash of the TUI, a closed terminal and a dropped ssh all leave
it training, and `python -m sdk.tui RUN` shows it again. Q names the run it
leaves training, and the one the TUI was opened on too when N moved it to
another. Q with a restart
pending leaves the trainer to stop and does not start it again; the line it
prints gives the command that does. SIGTERM and SIGHUP restore the terminal
and leave with status 0. A TUI killed with SIGKILL cannot restore anything
and leaves the terminal raw on the alternate screen: type `reset` and Enter,
although the letters do not show.

A system where systemd's `KillUserProcesses=yes` is set ends every process of
a login when it logs out, a trainer started from the TUI included.

## What must not change

Checkpoints from this package stay interchangeable with every other
SELFPLAYTV checkpoint only while the following stay as they are. Change any of
them and the team you train is a team for a different game, whose archive
loads nowhere else or loads and plays as noise.

- **The observation layout**: 186 numbers, built for teams of five
  (`engine/env/obs.py`, and the feature counts in `engine/config.py`).
- **The network**: `engine/rl/policy.py`, hidden width 256.
- **The simulation**: everything in `engine/env/`, and the physics, arena and
  match constants in `engine/config.py`.
- **The scoring**: the `POINTS_*` constants and `ROUNDS_TO_WIN`.
- **The checkpoint and manifest format**: `engine/rl/persist.py`.
- **The nickname rules**: `NICK_PATTERN` and `NICK_MAX_LENGTH`.
- **The shipped bundle**: `bundle/` is read, never written.

Nothing that decides the game is read from the environment: the rules, the
physics, the arena and the scoring are literals in `engine/config.py`, and so
are the reward weights, whose only ways in are the weights file and
`--reward`, both recorded in `status.json`.

Five knobs of the training alone can be set from the environment before the
first import: `ROLLOUT_STEPS`, `ANNEAL_STEPS`, `LEARNING_RATE_FLOOR`,
`AIM_SHAPING_FLOOR` and `TORCH_DEVICE`. They change how a team learns, never
what it plays, and a checkpoint trained under them stays compatible. A run
does not record them, so leave them unset unless the change is the experiment.

What is yours to change freely: the reward weights, every command-line flag,
and your own teams, bundles, runs and exports.
