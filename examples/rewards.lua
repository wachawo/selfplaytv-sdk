-- What your team is paid for while it learns. The scoreboard does not move:
-- points are counted by fixed rules, and these weights only shape the training.
--
--   python -m sdk.rewards examples/rewards.lua                 check this file
--   python -m sdk.train ... --rewards examples/rewards.lua     train with it
--
-- Every weight is paid to one player. A weight left out of the file keeps its
-- default, and the numbers below are the defaults for all of them but timeout.
--
-- timeout is 0 by default and is switched on here, against camping. Most rounds
-- that run out of time are settled on survivors and are not draws, so the draw
-- penalty rarely reaches a side that hides and waits for the clock. At -2 a win
-- on the clock pays 2 - 2 = 0 and a loss on the clock -3: finishing the round is
-- the only way to earn the win.
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
