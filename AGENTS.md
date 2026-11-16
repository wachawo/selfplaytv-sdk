# Working guide: making a team with this repository

This file is for whoever does the work when a person says "make my own team":
a coding agent, or the person. It is the journey as numbered steps, with the
exact commands and a way to tell that each step worked. The reference for
every flag and file format is [docs/sdk.md](docs/sdk.md).

## What this repository is

SELFPLAYTV is team paintball played by neural networks that learned the game
by playing each other. This repository is the simulation and the trainer on
their own. With it a person makes a team of five networks, decides what the
team is paid for while it learns, trains it on their own machine against two
teams that ship here, watches it play in the terminal and exports it as one
archive.

```
bundle/                 the two shipped teams, WHITE TEAM and BLACK TEAM. Read, never written.
examples/rewards.lua    the example reward weights file
examples/quickstart.sh  the whole journey as one script, with small numbers
sdk/doctor.py           check the installation
sdk/new_team.py         create a team
sdk/rewards.py          check a reward weights file
sdk/train.py            train, and measure with --play-only
sdk/watch.py            the terminal dashboard
sdk/demo.py             matches drawn in the terminal
sdk/graphics.py         the kitty graphics protocol, for sdk/demo.py
sdk/tui.py              one terminal screen that drives all of the above, for a person
sdk/export_team.py      pack a team into one archive
trainer/selfplaytv.py   the PPO trainer that sdk/train.py drives
engine/                 the game and the learning code. Do not change it.
docs/sdk.md             the reference
```

Three directories appear as you work, all ignored by git: `my-bundle/` (the
person's teams), `runs/` (training runs and measurements) and `exports/`
(archives).

`python -m sdk.tui` is a full-screen program for a person at a terminal: keys
start and stop the training, change its settings and weights, show the
dashboard, play a match, make a team and export one. It runs the same
commands as this guide, as separate processes. An agent keeps using the
commands below: they print, exit and can be checked, and the screen cannot.
A person may have the TUI open on the same run at the same time; it shows
whatever an agent starts or stops there.

A person can also skip every command: `bash start.sh` installs into `.venv/`
on its first run and opens the TUI without a run, which is a list of their
teams and a form for a new one. A team made there is a copy of WHITE in
`my-bundle/`, trained in `runs/<its run name>` against WHITE and BLACK with
the weights of `examples/rewards.lua`, and it starts training as soon as the
form is sent. An agent keeps using the commands: `start.sh` ends in a screen
it cannot read. A run the person made that way is an ordinary run, and the
commands below work on it, with `--rewards RUN/rewards.lua` as rule 6 says.

## Rules

1. Run every command from the repository root. `python` below is the
   interpreter of the environment torch was installed into, for example
   `.venv/bin/python`: a shell that has not activated that environment may
   have another `python`, or none.
2. Never write to `bundle/`. The person's team lives in `my-bundle/`.
3. Change only what is free to change. See
   [What you may change](#what-you-may-change-and-what-you-must-not).
4. Give every experiment and every measurement a run directory of its own,
   and every team its own.
5. Stop a training run with Ctrl-C or `kill` (SIGTERM). It then finishes its
   update and saves; a closed terminal (SIGHUP) does the same, unless the run
   was started under `nohup`, which keeps it training. Never `kill -9` it.
6. Pass the same `--rewards` file on every start of a run. A run does not
   remember its weights. A start from `sdk.tui` always passes the run's own
   `RUN/rewards.lua`; when the person also uses the TUI on a run, start that
   run with `--rewards RUN/rewards.lua` too.
7. Do not believe a small difference. Two measurements of the very same team
   differ by several points. See [Judging a team](#judging-a-team).
8. Keep runs under `runs/`, archives under `exports/`, and name bundles
   `my-bundle` or `my-bundle-something`. Only these are ignored by git: a
   bundle or a run anywhere else shows up in `git status`. Never `git add`
   any of them.

## Ask first

Before step 1, get these from the person:

- the team's name, its colour as `#rrggbb`, and five nicknames (letters,
  digits, hyphens and underscores, 24 characters at most);
- whether to start from a copy of WHITE or BLACK (the short road) or from
  untrained networks (the long road, see step 1);
- how long the machine may train, and how many CPU threads it may use;
- whether a GPU may be used. Without `--device`, `sdk.train` takes the GPU
  when torch sees one. If it may not, or if in doubt, add `--device cpu` to
  every `sdk.train` command: every time in this guide is a CPU time;
- what they want the team to do differently. That decides which weight to
  change in step 2.

## The journey

A person who wants to look before reading gets there with the five commands
of the README's [By hand](README.md#by-hand), which end in the terminal
screen. The steps below are the same journey by single commands,
which is how an agent works through it.

### 0. Check the installation

```bash
python -m sdk.doctor --threads 4
python -m sdk.demo WHITE BLACK --matches 5 --summary
```

**It worked if** the first command ends with `Usable: everything sdk.train
needs is here` and exits 0, after about fifteen seconds, and the second, after
about half a minute, prints five `Match WHITE TEAM ...` lines and two tables.
Every line of the first starts `OK`, `WARN` or `FIX`: a `FIX` line says what
to install, and a `WARN` line is worth telling the person (no TBB, a GPU torch
cannot use) but does not stop the journey. Its `speed` line is play alone;
training runs at about two fifths of it.

The demo's result is on standard output and the log on standard error, and the log
repeats each `Match` line as it is played: read the two together and there
are ten. If an import fails, install as [docs/sdk.md](docs/sdk.md#install) says: torch
first, then `pip install -r requirements.txt`.

### 1. Create the team

```bash
python -m sdk.new_team --name "MY TEAM" --colour "#12b5a6" --nicks Ada Bob Cy Dee Eve
```

This copies `bundle/` to `my-bundle/` and adds the team as a copy of WHITE.
Use `--from BLACK` to copy the other team, or `--fresh` for untrained
networks.

**It worked if** the last log line reads
`Train it with: python -m sdk.train --bundle my-bundle --run runs/my --learn 'MY TEAM' --against 'WHITE TEAM' 'BLACK TEAM'`
and `my-bundle/` holds `teams.json`, `manifest.json` and fifteen `.pt` files.
The line names a run directory after the team. Keep to it: a second team
trained in the first one's run directory carries on that run's counters and
tables, with only a warning.

Choose with the person:

- **A copy** plays exactly as well as its source from the first step. Training
  changes it slowly. This is the road for "a team that plays my way".
- **A fresh team** starts on a small arena and loses every match to the
  shipped teams for a long time: 353 of 353 in its first two million steps
  here. It needs about 160 million steps, about two hours on a desktop CPU,
  just to reach the full field. Take this road only if the person wants a
  team trained from nothing and has the hours.

Never put a copied team and a fresh one in the same `--learn`. It is accepted
with a warning, and the fresh team then skips its curriculum.

### 2. Write the reward weights

Copy the example beside the team and edit the copy:

```bash
cp examples/rewards.lua my-bundle/rewards.lua
python -m sdk.rewards my-bundle/rewards.lua
```

The file is one table of nine numbers: what one player is paid for a hit, for
being hit, for a round won, lost or drawn, for a round that ends on the clock,
for a capture, for covering a capture, and for a hit on an intruder at home.
The table with every weight and its default is in
[docs/sdk.md](docs/sdk.md#the-weights).

**It worked if** the check prints a table whose `changed` marks are exactly
the weights you meant to change. A mistake is refused with its line number.

The weights change what the team learns. They do not change the scoreboard:
points are counted by fixed rules.

### 3. A trial run

```bash
python -m sdk.train --bundle my-bundle --run runs/my --learn "MY TEAM" --against WHITE BLACK \
    --rewards my-bundle/rewards.lua --arenas 16 --threads 4 --steps 2000000
```

About three minutes on a desktop CPU. The 16 arenas are for the short run: a
match takes about 35,000 steps of its arena, the matches still being played
when a start ends are not counted, and at the default 64 arenas two million
steps would leave time for the shortest matches only.

**It worked if** the log shows

- `Reward weights, ...` with your changed weights marked with the file's name;
- `Phase: full arena ...` for a copied team, `Phase: close quarters ...` for a
  fresh one;
- `Match MY TEAM 7:5 WHITE TEAM (...)` lines;
- `Checkpoint at update ...` and, at the end, `Stopped at 2.01M steps, update 98`.

Always name the opponents with `--against`. The default is every other team
in the bundle, which silently includes any team added later.

### 4. Look at the run

```bash
python -m sdk.watch runs/my --once
```

**It worked if** it prints a header, RECENT MATCHES, STANDINGS and PLAYERS.
Without `--once`, in a terminal, it redraws every two seconds until Ctrl-C.
Show it to the person in a second terminal while training runs. The frame is
about 100 columns by 46 lines; in a smaller terminal it is cut, and its last
line says so.

### 5. Train for real

Run the same command with a time budget instead of `--steps`. It resumes the
trial run. A long run is best started in the background with its log in a
file:

```bash
nohup python -m sdk.train --bundle my-bundle --run runs/my --learn "MY TEAM" --against WHITE BLACK \
    --rewards my-bundle/rewards.lua --arenas 64 --threads 4 --hours 2 > runs/my.log 2>&1 &
echo $! > runs/my.pid
```

Check on it:

```bash
tail -n 3 runs/my.log
python -m sdk.watch runs/my --once
```

Stop it early, and let it save:

```bash
kill "$(cat runs/my.pid)"
```

**It worked if** the log ends with `Checkpoint at update ...` and
`Stopped at ...`. Start the same command again to carry on from there.

### 6. Watch a match

```bash
python -m sdk.demo --run runs/my "MY TEAM" WHITE
```

This draws one match in the terminal with the weights the run has saved, at
real time, about five minutes. It needs a real terminal of at least 45 by 20
characters, so this one is for the person to run. `--speed 4` plays four
times faster. It writes nothing and is safe while training runs.

Where there is no terminal, add `--summary` and read the result instead.

### 7. Measure

```bash
python -m sdk.demo --run runs/my "MY TEAM" WHITE --matches 50 --summary --seed 7
python -m sdk.demo --bundle my-bundle "MY TEAM" WHITE --matches 50 --summary --seed 7
```

The first line plays the team as trained, the second as it started. Each takes
about a minute and plays about 550 rounds, which is the size the noise figures
below are given for. The result is on standard output; the log on standard
error repeats every `Match` line, so count them in the result alone.

**It worked if** each prints fifty `Match` lines, a line such as
`Rounds: MY TEAM 246:302 WHITE TEAM (44.9% of the decided)`, a `Matches:` line
and two tables. Compare the `Rounds:` share and `a round` (average points a
round). Read [Judging a team](#judging-a-team) before drawing a conclusion.

### 8. Export

```bash
python -m sdk.export_team --run runs/my --team "MY TEAM"
```

**It worked if** it logs five lines (`Team:`, `Players:`, `Steps per model:`,
`Weights:`, `Archive:`) and prints the path of
`exports/MY_TEAM_<date and time>.tar.gz`, about 12.5 MB.

To rename the team or its players on the way out:

```bash
python -m sdk.export_team --run runs/my --team "MY TEAM" --name "TEAL TEAM" \
    --nicks Uno Dos Tres Cuatro Cinco --colour "#0fa3b1"
```

The archive is the deliverable. It holds the five checkpoints, a manifest and
the team's description, and nothing about the machine it was made on.

## Choosing arenas, threads and hours

Measured on one desktop CPU, one team learning against the two shipped teams:

| `--arenas` | `--threads` | agent-steps a second |
|---|---|---|
| 8 | 4 | 8,400 |
| 16 | 4 | 12,000 |
| 32 | 4 | 24,000 |
| 64 | 1 | 19,000 |
| 64 | 2 | 26,000 |
| 64 | 4 | 33,000 |
| 128 | 4 | 45,000 |
| 256 | 4 | 46,000 |

Peak memory was about 1.0 GB up to 64 arenas, 1.3 GB at 128 and 1.7 GB at 256.

- **Arenas.** For a real session start with the default, 64. Go to 128 if the
  machine has the memory. Below 32 a run is slow and its updates are noisy.
  Above 128 nothing was gained. Every start should give each arena at least
  100,000 steps, or its tables hold only the shortest matches: that is 6.4
  million steps at 64 arenas, and the reason the trial run in step 3 uses 16.
- **Threads.** Start with 4. Threads help less than arenas: four are 1.7 times
  one. Leave cores free for the dashboard, the demo and the person's own work.
  More than four were not measured. Without `--threads` a run takes every CPU,
  up to 16.
- **Hours.** At 64 arenas and 4 threads a million steps take about half a
  minute, and a hundred million about an hour. Read the real rate off the
  first lines of the log (`28,247/s`) and work out the hours from that.
- **A GPU** is not needed. None was used for these numbers. With a GPU and a
  matching torch, the default device is the GPU.
- On a small machine lower the threads first, then the arenas.

How long is long enough is not something this guide can promise. A copied
team is a veteran with over two billion steps of its own, and it learns at a
small learning rate. Do not expect the first millions of steps to improve it.
In the runs for this guide a copy trained for 4.1 million steps on the default
weights stayed at 50% of its decided rounds against WHITE, and one trained as
long with the example weights fell to 46.8%. A team that is paid for something
new first gets worse at what it knew. Plan for tens of millions of steps per
experiment, which is under an hour, and measure after each.

## Judging a team

Measure with step 7. The training tables are not a measurement: they mix
every stage of a team that was changing while they were filled. They also
count finished matches only, and the matches still being played when a start
ends are dropped. A start shorter than about 100,000 steps for every arena
therefore records its shortest matches and little else, and `W-L` favours
whoever wins fast. In such a run read `W/D/L` in PLAYERS, which counts every
round, and measure.

Read these, in this order:

1. **Time share** (`time%` in STANDINGS, the `time` count in the demo's
   `rounds:` line): the
   share of rounds that ran out the clock. It is the waiting measure. The
   shipped teams were between 0% and 6% in every measurement here; a team
   that has learned to hide shows a large number.
2. **Round win rate**: rounds won out of rounds decided (the `Rounds:` line
   of the demo summary, `W/D/L` in PLAYERS). An equal opponent gives 50%.
3. **Average points a round** (`pts/rnd`, `a round`): the team's five players
   together. WHITE and its own copy score between 6.4 and 7.2 against each
   other. Points
   reward hits, round wins and captures, so this is the one figure that sums
   up how the team plays.
4. **Hits per death** (`K/D`): above 1.00 the player hits more often than it
   is hit. Look at the five together: a team can win by captures with a K/D
   below 1.
5. **How rounds end** (`elim%`, `cap%`): the team's style. WHITE against its
   own copy ends about a third of the rounds with a capture; against BLACK
   almost every round ends in an elimination. These shares, and the `capture`
   count in the demo's `rounds:` line, are of both sides together. Whether
   YOUR team captures more is its own `captures` in the demo's team table,
   before and after: one credit for each of its players standing on the strip
   as it falls, so a capture by three players is three.

The `paid` line of the dashboard is the reward, not a result. It moves when
you change a weight even if the play is the same. Compare it only between
runs with the same weights file.

**Noise.** The same team against an identical copy of itself, ten times,
about 40 matches and 530 rounds each: match records from 14-26 to 23-17, round
win rates from 45% to 53%, points a round from 6.4 to 7.1. So over about 500
rounds:

- a difference under 5 points of round win rate, or under 0.8 points a round,
  is nothing;
- matches won is the noisiest figure of all.

For a firm answer, play more rounds or repeat on several seeds. A larger
measurement, `sdk.train --play-only`, is described in
[docs/sdk.md](docs/sdk.md#measuring-a-team).

## Iterating on the weights

1. **Measure the starting point once** (step 7, second command) and keep the
   output.
2. **Change one weight.** Edit one number in `my-bundle/rewards.lua` and check
   the file with `python -m sdk.rewards`. Two changes at once cannot be told
   apart afterwards.
3. **Train it in a run directory named after the change**, with the same
   `--arenas`, `--threads` and `--seed` as the run you compare it with:

   ```bash
   python -m sdk.train --bundle my-bundle --run runs/win3 --learn "MY TEAM" --against WHITE BLACK \
       --rewards my-bundle/rewards.lua --reward REWARD_WIN=3 --threads 4 --steps 200000
   ```

   `--reward` sets one weight on top of the file for one experiment, without
   editing the file. The steps here are small to show the command. A real
   experiment needs tens of millions.
4. **Measure it** with the same demo command, the same `--seed` and the same
   `--matches` as the starting point, with `--run` pointing at the new run.
5. **Keep or discard.** Keep the change only if the difference is larger than
   the noise above. Write down each experiment: the weights, the steps, the
   measured figures.

What each weight pays is in the table in
[docs/sdk.md](docs/sdk.md#the-weights). Some directions to try, none of them
a promise:

- more captures: raise `capture` and `capture_cover`;
- less waiting: make `timeout` more negative;
- fewer careless deaths: make `hit_taken` more negative;
- more decided rounds: make `draw` more negative.

A weight pushed far from its default can make a team worse at everything
else. Move in small steps.

## What you may change, and what you must not

**Free to change:**

- the reward weights file, and `--reward` overrides;
- every command-line flag: arenas, threads, hours, seed, theme, opponents;
- the person's own teams, bundles, runs and exports.

**Never change:**

- anything under `engine/`: the simulation, the observation (186 numbers), the
  network (hidden width 256), the checkpoint format, the scoring constants
  (`POINTS_*`, `ROUNDS_TO_WIN`), the nickname rules;
- the collect and learn path of `trainer/selfplaytv.py`;
- the files in `bundle/`;
- the environment of the training shell. The game cannot be changed from
  there: its constants are literals in `engine/config.py`. Five training
  knobs can (`ROLLOUT_STEPS`, `ANNEAL_STEPS`, `LEARNING_RATE_FLOOR`,
  `AIM_SHAPING_FLOOR`, `TORCH_DEVICE`), and a run does not record them: leave
  them unset.

The reason is one rule: a checkpoint trained here must load and play
everywhere else exactly as it played here. A team trained on a changed game,
a changed network or a changed observation is a team for a different game.
Its archive is refused elsewhere, or worse, loads and plays as noise.

If the person asks for something that needs one of these changed, say that it
breaks compatibility and stop.

## When a command refuses

A refusal is one `[ERROR]` line and exit status 2, before anything is
written. The common ones:

| The line says | Do this |
|---|---|
| `No module named 'sdk'` (from Python itself, not a log line) | the command was not run from the repository root, or a file was run directly; use `python -m sdk.<name>` from the root |
| `--bundle ...: no such directory` | the path is wrong, or step 1 has not been done yet |
| `No team 'RED' in the bundle; it holds ...` | use one of the names it lists |
| `... is the only team in the bundle and has nobody to play` | the bundle is a one-team archive; create the team in a copy of `bundle/` with `sdk.new_team` |
| `--name 'white' would be taken for the existing team 'WHITE TEAM'` | choose another team name |
| `--nicks 'ada': MY TEAM already has a player of that name` | choose another nickname; case does not count |
| `line 3: unknown weight 'camping'; expected one of ...` | fix that line of the weights file |
| `Another sdk.train is using runs/my` | a run is still going there; stop it or use another `--run` |
| `--run ... is inside the bundle ...` | a run never lives in a bundle; put it under `runs/` |
| `This run was created to train ... and is now asked to train ...` (a warning) | that run directory belongs to another team; stop and give this team its own `--run` |
| `The learners are at different stages of the curriculum` (a warning) | a copied team and a fresh one share a `--learn`; stop and train them in separate runs |
| `... holds files and ... has no sdk-run.json` | that directory is not a run of this SDK; give a new `--run` |
| `The reward weights are not those of this run's last start` (a warning) | you resumed without the same `--rewards`; stop and start again with it |
| `The timeout penalty is off` (a warning) | no weights file was given, or it keeps `timeout` at 0; pass `--rewards`, unless the person chose that |
| `Learning has stalled, ...` (a warning) | the run goes on; tell the person, and try what the line says. The causes are in [docs/sdk.md](docs/sdk.md#the-log-line-of-an-update) |
