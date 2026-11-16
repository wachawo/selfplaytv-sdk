#!/usr/bin/env bash
# The whole journey with small numbers, as one script: make a team, check what
# it is paid for, train it, look at the run, play demo matches, export it.
#
#   bash examples/quickstart.sh
#
# Three to four minutes on a desktop CPU. The numbers are small so that every
# step is seen to work, not so that the team improves: 1,600,000 steps is under
# a thousandth of what a shipped team has behind it. They are not smaller still
# because a match takes about 35,000 steps of its arena, and a start that gives
# each arena less than about 100,000 records only its shortest matches. Run it
# a second time and the training resumes where it stopped, and a second archive
# is written.
#
# Every variable below can be set from the environment, for example
#   STEPS=20000000 ARENAS=64 bash examples/quickstart.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"
TEAM="${TEAM:-MY TEAM}"
COLOUR="${COLOUR:-#12b5a6}"
NICKS="${NICKS:-Ada Bob Cy Dee Eve}"
SOURCE="${SOURCE:-WHITE}"
BUNDLE="${BUNDLE:-my-bundle}"
RUN="${RUN:-runs/quickstart}"
REWARDS="${REWARDS:-examples/rewards.lua}"
EXPORTS="${EXPORTS:-exports}"
ARENAS="${ARENAS:-16}"
THREADS="${THREADS:-4}"
STEPS="${STEPS:-1600000}"
# A GPU is not needed at this size. Set DEVICE=cuda to use one.
DEVICE="${DEVICE:-cpu}"
MATCHES="${MATCHES:-3}"

# Everything is run from the repository root: the commands import engine/,
# trainer/ and sdk/ from there, and the default paths are relative to it.
cd "$(dirname "$0")/.."

# 1. Your team, in a bundle of your own. The shipped bundle/ is only read.
#    Skipped when the bundle is already there, so the script can be run again.
if [ ! -d "$BUNDLE" ]; then
    # NICKS is split on spaces on purpose: five words, five nicknames.
    # shellcheck disable=SC2086
    "$PYTHON" -m sdk.new_team --name "$TEAM" --colour "$COLOUR" --nicks $NICKS \
        --from "$SOURCE" --out "$BUNDLE"
fi

# 2. What the team is paid for while it learns. This only checks the file.
"$PYTHON" -m sdk.rewards "$REWARDS"

# 3. Train against the two shipped teams, which stay frozen. They are named,
#    because the default is every other team in the bundle, and the bundle may
#    hold more by the time this is run again. Stops by itself after STEPS;
#    Ctrl-C stops it earlier and saves.
"$PYTHON" -m sdk.train --bundle "$BUNDLE" --run "$RUN" --learn "$TEAM" \
    --against WHITE BLACK --rewards "$REWARDS" --arenas "$ARENAS" \
    --threads "$THREADS" --device "$DEVICE" --steps "$STEPS"

# 4. The run as the dashboard shows it. Without --once, and in a second
#    terminal while step 3 is running, it redraws every two seconds.
"$PYTHON" -m sdk.watch "$RUN" --once

# 5. Demo matches with the weights the run has now. In a terminal, leave out
#    --summary to watch one drawn.
"$PYTHON" -m sdk.demo --run "$RUN" "$TEAM" WHITE --matches "$MATCHES" --summary

# 6. One archive with the trained team. Its path is the only line the command
#    writes to standard output.
ARCHIVE="$("$PYTHON" -m sdk.export_team --run "$RUN" --team "$TEAM" --out "$EXPORTS")"
echo "Exported: $ARCHIVE"
