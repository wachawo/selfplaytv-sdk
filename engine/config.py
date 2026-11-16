#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Central configuration: every tunable constant of the simulation and the training.

Everything that decides the GAME is a plain literal, on purpose: the rules, the
physics, the arena, the observation layout and the scoring. The checkpoints are
built for them, and a value picked up from the shell would train a team for a
different game without a word in the run's files. The rewards are literals too,
and a run changes them through its weights file.

Five knobs of the TRAINING alone are read with os.getenv and can be set from
the environment before the first import: AIM_SHAPING_FLOOR, LEARNING_RATE_FLOOR,
ROLLOUT_STEPS, ANNEAL_STEPS and TORCH_DEVICE. They change how a team learns and
never what it plays.
"""

import os
import re
from typing import Any, Dict, List, Optional

# Roster
#
# The models this process works with, by nickname, in the order everything
# indexed per model follows. Empty until set_roster is called: the caller names
# the models of the teams that play, and the networks, optimisers and rollout
# buffers are built one per name after that.
MODEL_NICKS: List[str] = []
N_MODELS = 0

# What a nickname may be, said once, because a nickname is the WHOLE of a
# checkpoint's file name, and that is what the character set is for.
# persist.checkpoint_path is a bare f"{nick}.pt", so a name carrying a separator
# or a pair of dots would write outside the checkpoint directory; requiring the
# first character to be alphanumeric and the rest alphanumeric, hyphen or
# underscore makes "..", "../../etc/passwd", ".hidden" and "-rf" all unspellable
# rather than caught by a check somewhere downstream that the next caller can
# forget.
#
# 24 characters, because a nickname is also drawn over a player's head and
# printed in narrow table columns, with nowhere wider to spread into.
NICK_MAX_LENGTH = 24
NICK_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def nick_problem(nick: Any) -> Optional[str]:
    """Which rule this nickname breaks, or None when it breaks none.

    Answers WHICH rule and not yes-or-no, because somebody typed this name:
    "invalid name" tells them nothing about what to type instead, and the rule
    they broke is the only useful half of a refusal.

    A name already taken is NOT decided here: that needs the other names, which
    the caller has and this module does not.
    """
    if not isinstance(nick, str):
        return "a nickname has to be text"
    if nick != nick.strip():
        return "a nickname cannot begin or end with a space"
    if not nick:
        return "a nickname cannot be empty"
    if len(nick) > NICK_MAX_LENGTH:
        return f"a nickname can be at most {NICK_MAX_LENGTH} characters, this one is {len(nick)}"
    if not NICK_PATTERN.match(nick):
        return ("a nickname can hold letters, digits, hyphens and underscores, and has to "
                "start with a letter or a digit")
    return None


def set_roster(nicks: List[str]) -> None:
    """Adopt the roster.

    Everything reads config.MODEL_NICKS through this module at call time, so
    rebinding here reaches all of it - except what was already built from it.
    The trainer makes one network, one optimiser and one rollout buffer per
    model when it is constructed, so the roster is set before the Trainer is.
    """
    global MODEL_NICKS, N_MODELS
    MODEL_NICKS = list(nicks)
    N_MODELS = len(MODEL_NICKS)


# Match rules
# Five a side. The simulation takes the team size as a parameter, but every
# checkpoint this package ships or exchanges was trained five against five.
TEAM_SIZE      = 5
# How many players a side the observation layout has entity slots for. A literal
# and not a knob: changing it changes the observation width, and every saved
# network's first layer is built for the old width and will not load against
# the new one.
MAX_TEAM_SIZE  = 5
SHOT_COOLDOWN  = 1.0
MATCH_SECONDS  = 60.0
# A match is a series of rounds, and the first team to win this many takes it.
# A drawn round scores for nobody.
ROUNDS_TO_WIN  = 7
POINTS_PER_HIT = 1
POINTS_PER_WIN = 1
# A capture pays the whole side, and pays the players who were standing on the
# strip more. One point for being on the team that took it, two more for having
# been the one holding it: 3 in total for a holder, 1 for everybody else.
#
# Both are stated as TOTALS rather than as a base and a bonus, which is the
# convention every price here follows -- see POINTS_PER_DEFENCE below for the
# reasoning. So the split is expressed by the pair of numbers and the code pays
# one or the other, never both.
#
# Scoreboard counterparts of REWARD_CAPTURE and REWARD_CAPTURE_COVER, and kept
# as their own knobs on purpose, the same way the hit and win points are: this
# decides what the table credits, the REWARD_ constants decide what the models
# learn from, and the two are free to disagree. They do disagree: the cover
# reward is 2 and the scoreboard pays 1.
POINTS_PER_CAPTURE = 3
POINTS_PER_CAPTURE_COVER = 1
# What a hit on an intruder in a team's own spawn strip is worth IN TOTAL, replacing
# POINTS_PER_HIT rather than adding to it. Stated as a total so the rule stays
# whatever it is set to, whatever the ordinary hit is later changed to; as an
# increment it would quietly become "the hit price plus one".
#
# Level with the ordinary hit by default. It used to carry a premium, which made
# the easiest shot in the game -- one at a stationary intruder on open ground,
# since obstacles are kept off the strip -- the best paid in it, and so quietly
# financed staying home. The simulation still marks the shot, so raising this is
# a one-line change if defending the strip should ever be worth more again.
POINTS_PER_DEFENCE = 1
FRIENDLY_FIRE  = False

# Capture. Standing inside the opposing spawn strip for this long, without a
# break, wins the match outright.
#
# The rule exists to give the match an objective other than killing. With
# elimination as the only way to win, a team that is behind has nothing to do
# but hide and run the clock down, and the team that is ahead has nothing to do
# but wait. A strip that has to be entered and held gives the losing side
# somewhere to go and forces the winning side to come out and defend it.
CAPTURE_SECONDS = 5.0
# How many attackers on the strip still speed the hold up. At 1 there is no
# speed-up: the strip takes CAPTURE_SECONDS whether one player is standing on it
# or five, and bringing more of them buys holding the ground against the defence
# rather than a shorter clock.
#
# It was 3 for a while, and that acceleration is what first made captures happen
# at all -- they sat near one percent without it, because the strip is the one
# place on the map with no cover and five seconds alone under fire is close to
# impossible. Raising this brings the speed-up back; the hold then fills at one
# second per attacker, capped here.
CAPTURE_RATE_CAP = 1
# Share of *training* episodes that begin with everyone spread over the whole
# map instead of in their own strips. Measured time on the enemy strip is about
# a fifth of a second per match, so from ordinary spawns the objective is a
# state the critic never sees and therefore cannot learn to value, however much
# it pays. This is an exploring start, not a hint: no action is prescribed, only
# where the episode begins. Off, and a literal like the rest of the rules: it
# also reaches play that only measures a team, and a scattered start is not a
# round of the real game.
EXPLORING_START_FRACTION = 0.0

# Simulation timing
PHYSICS_HZ    = 20
DT            = 1.0 / PHYSICS_HZ
ACTION_REPEAT = 2
CONTROL_DT    = DT * ACTION_REPEAT
COOLDOWN_TICKS = max(1, round(SHOT_COOLDOWN / DT))
MATCH_TICKS    = round(MATCH_SECONDS / DT)

# Hard ceiling on the real elapsed ticks of one episode.
#
# A capture in progress stops the match clock, so the clock alone can no longer
# bound an episode: an attacker that steps in and out of the strip freezes the
# clock every time it is inside and gives the ground back every time it leaves,
# and the match runs forever. Forever is not merely untidy - a rollout buffer is
# a fixed window of steps that only pays out its terminal reward when an episode
# ends, so one arena stuck in an endless match contributes nothing but
# bootstrapped guesses for the rest of the run.
#
# Real ticks are counted separately from the match clock and capped here. One
# and a half times the match length is a budget rather than a derived bound: the
# worst case has no bound, because an attacker can spend an arbitrary share of
# its time inside the strip. Half as long again is chosen because a fight over
# the strip is the part of a match most worth playing out, one uncontested hold
# only ever costs CAPTURE_SECONDS (a twelfth of a match), and the factor keeps
# the worst-case cost of a single episode predictable and small.
# Never below 1.0: a ceiling under the match length would end every match on
# the real-tick limit and quietly make the match clock dead code.
MATCH_OVERTIME  = 1.5
MATCH_TICKS_MAX = round(MATCH_TICKS * MATCH_OVERTIME)

# Arena
ARENA_W      = 30.0
ARENA_H      = 20.0
N_OBSTACLES  = 18

# Depth of the lane along the top edge (small y) that every layout leaves
# completely open, full width, both halves.
#
# It exists so there is always one route from one side of the field to the
# other that generation cannot take away. Without it the only guarantee is the
# flood fill, which asks whether a path exists at all, not whether it is a path
# anyone can fight along -- a layout that squeezes the two halves together
# through one diagonal slot passes that check and still plays as a corridor.
#
# 2.4 m is a bit under three player diameters (radius 0.45 m). Two players can
# pass abreast and either can still sidestep a stream of paint without clipping
# the wall, which one diameter of clearance does not buy. It costs 12% of the
# 20 m height, and the lane is symmetric about the centre line, so it does not
# favour either spawn.
ARENA_TOP_LANE = 2.4
MAX_OBSTACLES = 32  # fixed tensor width across the batch; unused slots are masked

# Player kinematics
PLAYER_RADIUS   = 0.45
SPEED_FORWARD   = 4.5
SPEED_LATERAL   = 3.0
ACCELERATION    = 12.0
TURN_RATE       = 2.618  # 150 deg/s in radians
# How much of that -- and of the two speeds above -- a player is actually dealt,
# drawn per player per ROUND, so no two players of a round move quite alike.
PLAYER_SCALE_RANGE = (0.85, 1.15)
TURN_BINS       = (-1.0, -0.4, -0.15, 0.0, 0.15, 0.4, 1.0)
N_MOVE_ACTIONS  = 9
N_TURN_ACTIONS  = len(TURN_BINS)
N_SHOOT_ACTIONS = 2

# Paintball projectile
BALL_SPEED    = 25.0
BALL_RADIUS   = 0.10
# Range is SPEED * LIFETIME, and it has to stay well under the width of the
# field. At 1.2 s a marker carried 30 m across a 30 m arena, so a player
# standing on the lip of his own spawn already covered the enemy strip 24.6 m
# away with five metres to spare. Advancing bought no coverage it did not
# already have and cost the cover it was standing behind, which makes holding
# the doorway the correct play rather than a timid one. At 0.6 s the reach is
# 15 m, half the field, and the enemy has to be approached to be shot at.
BALL_LIFETIME = 0.6
BALL_TICKS    = round(BALL_LIFETIME * PHYSICS_HZ)
BALL_SPREAD   = 0.0209  # 1.2 deg in radians
MAX_BALLS_PER_PLAYER = 2  # >= lifetime / cooldown, keeps the tensor fixed

# Observation layout
N_RAYS       = 12
RAY_FEATURES = 4
ENTITY_FEATURES = 10
SELF_FEATURES   = 10
GLOBAL_FEATURES = 6
BALL_FEATURES   = 4
MAX_SIGHT       = 40.0
MEMORY_SECONDS  = 3.0

# Gunshots. Firing gives the shooter away: every enemy hears the shot and keeps
# the bearing to it for a few seconds, which is the only cue available through a
# wall. Range is the distance at which the report fades to nothing.
SOUND_FEATURES       = 9
# Enemy paint in flight nearby: bearing, distance and how much of it. See
# obs.tracer_block for why this is not the same thing as the incoming block.
TRACER_FEATURES      = 4
SOUND_MEMORY_SECONDS = 3.0
SOUND_RANGE          = 40.0

# Capture awareness: own progress, best progress on my team, best progress among
# the enemies, whether I am standing in the enemy strip right now, the sine and
# cosine of the bearing to the nearest enemy standing in my own strip, and
# whether the capture this player cares about is currently contested.
# Without the enemy figure a defender has no cue that the strip is being taken,
# and the rule would only ever be exploited, never contested; without the
# bearing it knows the strip is being taken but not where to go. The contested
# flag is what closes the loop: walking into your own strip is what pauses an
# enemy countdown, and an agent that cannot observe the pause has no way to
# discover that standing there is the thing that stops it.
CAPTURE_FEATURES = 11

# Rewards
#
# Plain literals, every one of them, and not read from the environment. A run
# changes what it pays through its weights file (sdk/rewards.py) or --reward,
# and both are written into the run's status.json beside these defaults. A
# value picked up from the shell would be a default nobody can see: the run
# would report it as untouched while training on something else.
REWARD_HIT        = 1.0
REWARD_HIT_TAKEN  = -1.0
REWARD_WIN        = 2.0
# A loss and a draw cost the same by default, so a side that is behind gains
# nothing by settling for the draw and has no reason not to go for the win.
# Tuned together with REWARD_DRAW: the gap between them is the whole rule, and
# moving one without looking at the other is how it gets set by accident.
REWARD_LOSS       = -1.0
# Two bonuses that exist because taking ground otherwise pays strictly less than
# shooting. A firefight pays REWARD_HIT for every elimination on the way to the
# same win; a capture pays nothing but the win, so a policy weighing the two
# correctly should always fight. These make the objective worth the trip, and
# they are deliberately NOT potential-based -- unlike the shaping terms they are
# meant to change what the optimal play is.
#
# The capture bonus is paid per player standing on the strip as it falls, not
# per member of the winning team: it buys the act of being there, which is the
# part that costs something. The defence price, REWARD_DEFENCE below, is what a
# hit on an intruder in your own spawn pays in place of REWARD_HIT.
REWARD_CAPTURE    = 3.0
# What covering a capture is worth, paid to every member of the capturing side
# who was NOT on the strip -- the holders are already paid REWARD_CAPTURE and do
# not draw this as well. Without it a capture is worth something only to the one
# player standing there, and nobody else has a reason to trade a position, or a
# life, for a hold they are not carrying. Taking a base cannot be done alone, so
# the payout for it cannot be private; it is worth less than being there, because
# being there is the part that costs something.
REWARD_CAPTURE_COVER = 2.0
# The learning counterpart of POINTS_PER_DEFENCE, and a total in the same way:
# it replaces REWARD_HIT for that shot rather than adding to it. Level with the
# ordinary hit by default, and a weight of its own for the same reason the
# points have a price of their own: the scoreboard price and the gradient are
# free to disagree, and raising one without the other is usually a mistake
# worth being able to make deliberately.
REWARD_DEFENCE    = 1.0
# A goalless draw is the outcome both sides get by refusing to commit, and at
# zero it was free. Dead players collect the terminal reward too, so a penalty
# here cannot be dodged by dying -- unlike a per-tick clock penalty, which is
# why that one is still absent. See REWARD_LOSS: what matters is the gap.
REWARD_DRAW       = -1.0
# Off by default. Paid to EVERY player of BOTH sides, the dead ones too, exactly
# as REWARD_DRAW is, when a round ends on the clock rather than on a capture or
# a wipe -- and on top of the win, loss or draw payout, not instead of it. The
# draw penalty alone cannot cure a side that waits in a corner for the clock:
# most rounds that run out of time are still settled on survivors and are not
# draws, so most of the waiting is never charged.
REWARD_TIMEOUT    = 0.0
SHOT_PENALTY      = -0.01
SHOT_PENALTY_FROM = 20_000_000
AIM_SHAPING_K     = 0.05
# Where the aim shaping stops, and deliberately not zero, for the same reason
# the learning rate has a floor. The term is scaffolding for an aim the sparse
# hit reward is supposed to take over from -- but a hit arrives only about once
# every few hundred player-steps, and with the scaffolding gone the only dense
# signal left points at the enemy strip. A fifth of the starting weight is
# enough to keep aiming on the gradient without it steering the policy: the
# term is potential-based, so whatever the value it cannot change the optimal
# play.
# One of the five training knobs the environment can set; see the module
# docstring.
AIM_SHAPING_FLOOR = float(os.getenv("AIM_SHAPING_FLOOR", "0.01"))
# Capture shaping. Holding the enemy strip for five seconds wins the match, but
# a policy that has never done it by accident will never be told that it is
# worth doing: the terminal reward is far too rare to discover. Rewarding
# progress makes the rule learnable. Potential-based, so it changes how fast the
# rule is learned and not what the optimal play is. Symmetric, because letting
# an enemy stand in your own strip has to cost the same as gaining ground.
CAPTURE_SHAPING_K = 0.5
# Weight of the approach term inside the capture potential, as a share of the
# progress term. Deliberately small: it exists to give the objective a gradient
# that reaches across the field, not to make walking at the enemy spawn pay
# better than fighting. Potential-based, so it cannot change the optimal policy
# whatever the value -- only how quickly the rule is found.
CAPTURE_APPROACH_K = 0.35
AIM_ANNEAL_FROM   = 20_000_000
AIM_ANNEAL_TO     = 60_000_000
AUTOFIRE_CONE     = 0.0873  # 5 deg in radians

# PPO
#
# Gamma is tied to how long a match can run. The win bonus only arrives at the
# very end, so it has to survive the discounting on the way back. A one-minute
# match is 600 control steps and 0.999 leaves 0.55 of the bonus there; the
# capture rule can freeze the match clock, so the real ceiling is
# MATCH_TICKS_MAX, which at an overtime of 1.5 is 900 control steps and still
# leaves 0.41. Both are comfortable. Raise MATCH_SECONDS or MATCH_OVERTIME and
# recompute this: at 1200 steps it is 0.30, and it falls off a cliff after that.
GAMMA          = 0.999
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
LEARNING_RATE  = 6e-4
# Where the annealing stops, and deliberately not zero. A rate of zero is a
# model that cannot learn anything at all: a run annealed that far spends its
# updates at kl 0.0000 and clip 0.00, still burning a GPU while nothing moves.
# A run with no fixed end must never anneal itself into that state.
LEARNING_RATE_FLOOR = float(os.getenv("LEARNING_RATE_FLOOR", "6e-5"))
ADAM_EPS       = 1e-5
VALUE_COEF     = 0.5
GRAD_CLIP      = 0.5
PPO_EPOCHS     = 4
N_MINIBATCHES  = 8
ENTROPY_MOVE   = 0.01
ENTROPY_TURN   = 0.01
ENTROPY_SHOOT  = 0.003
ENTROPY_FINAL_SCALE = 0.2

ROLLOUT_STEPS = int(os.getenv("ROLLOUT_STEPS", "128"))
# The horizon the learning rate and the entropy bonus are annealed over. It is
# not the same thing as the length of the run: with no end there is nothing to
# anneal towards, and the schedule still has to finish somewhere or the models
# would keep taking beginner-sized steps for ever.
ANNEAL_STEPS  = int(os.getenv("ANNEAL_STEPS", "400000000"))
TORCH_DEVICE  = os.getenv("TORCH_DEVICE", "auto")

# Per-model personality: entropy scale and team blend.
#
# A spread of entropy from 0.5x to 2.0x gives the models visible character, and
# the ones at the top of that range never converge an aim: they fire millions of
# rounds at a 2% hit rate. The spread here is narrow enough to differentiate
# without preventing convergence, and every network is the same width, so one
# model's weights can stand in any other model's place.
MODEL_HIDDEN = 256
MODEL_PERSONALITIES = [
    {"entropy_scale": 0.7, "team_blend": 0.30, "hidden": MODEL_HIDDEN},
    {"entropy_scale": 0.9, "team_blend": 0.40, "hidden": MODEL_HIDDEN},
    {"entropy_scale": 1.1, "team_blend": 0.30, "hidden": MODEL_HIDDEN},
    {"entropy_scale": 1.3, "team_blend": 0.45, "hidden": MODEL_HIDDEN},
    {"entropy_scale": 0.8, "team_blend": 0.35, "hidden": MODEL_HIDDEN},
    {"entropy_scale": 1.2, "team_blend": 0.40, "hidden": MODEL_HIDDEN},
    {"entropy_scale": 1.0, "team_blend": 0.35, "hidden": MODEL_HIDDEN},
    {"entropy_scale": 0.75, "team_blend": 0.45, "hidden": MODEL_HIDDEN},
    {"entropy_scale": 1.25, "team_blend": 0.30, "hidden": MODEL_HIDDEN},
    # One entry per model of a ten-model roster, so the tenth gets a character
    # of its own rather than wrapping around onto the first.
    {"entropy_scale": 0.85, "team_blend": 0.35, "hidden": MODEL_HIDDEN},
]

# The spread the ten above were chosen out of, stated so a personality handed to
# an eleventh model stays inside the range that was measured rather than inventing
# one outside it. Thirteen scales from 0.70 to 1.30 in steps of 0.05: below that
# the models stop differing, above it they never converge an aim -- see the note
# over the list. Four blends, which is the set the ten already use.
ENTROPY_SCALE_MIN   = 0.70
ENTROPY_SCALE_STEP  = 0.05
ENTROPY_SCALE_COUNT = 13
TEAM_BLENDS = (0.30, 0.35, 0.40, 0.45)


def personality_grid() -> List[Dict[str, float]]:
    """Every character in the measured spread that the ten above do not already use.

    A roster is not a fixed ten, so there has to be an answer for the eleventh
    model. `MODEL_PERSONALITIES[index % len(...)]` would give it entry 0 and have
    it play with the first model's character -- exactly what the tenth entry
    above was written to stop, one model further along.

    Thirteen scales by four blends is 52 pairs; ten are spoken for, and these are
    the other 42. Stepping the scale by 5 of 13 and the blend by 1 of 4 visits all
    52 exactly once before repeating, because 5 is coprime with 13 and 13 is
    coprime with 4 -- so this both covers the grid and spreads consecutive models
    apart instead of handing three neighbours 0.70, 0.75 and 0.80.
    """
    written = {(p["entropy_scale"], p["team_blend"]) for p in MODEL_PERSONALITIES}
    grid = []
    for step in range(ENTROPY_SCALE_COUNT * len(TEAM_BLENDS)):
        scale = round(ENTROPY_SCALE_MIN + ENTROPY_SCALE_STEP * ((step * 5) % ENTROPY_SCALE_COUNT), 2)
        blend = TEAM_BLENDS[step % len(TEAM_BLENDS)]
        if (scale, blend) in written:
            continue
        grid.append({"entropy_scale": scale, "team_blend": blend, "hidden": MODEL_HIDDEN})
    return grid


EXTRA_PERSONALITIES = personality_grid()


def personality(index: int) -> Dict[str, float]:
    """The character model `index` plays with, for a roster of any size.

    The ten written ones first, then the 42 the grid has left. Past 52 models it
    wraps, and that wrap is deliberate: the spread is finite, every character
    inside it has been handed out once, and a 53rd model can only be given one
    somebody already has. It is a different model either way -- a personality is
    two scalars, and the weights behind them are its own -- so the cost of the
    repeat is that two models play with the same temperament.
    """
    if index < len(MODEL_PERSONALITIES):
        return MODEL_PERSONALITIES[index]
    return EXTRA_PERSONALITIES[(index - len(MODEL_PERSONALITIES)) % len(EXTRA_PERSONALITIES)]


def main():
    pass


if __name__ == "__main__":
    main()
