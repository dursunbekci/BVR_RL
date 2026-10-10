# Flying the player's F-16C with stick commands

Branch `feature/dcs-f16c-stick`, from `f3a8fb0` (SIM_REV 15, the head of
`feature/envelope-target-alt` on 10 October).

## Why

BLUE-1 is an AI unit, flown by the bridge through a route. The DCS AI follows
a route at 45° of bank (1.6°/s), climbs at about Mach 0.85 and does not light
its afterburner for a speed. Most of the work on `feature/envelope-target-alt`
(the F-16C-DCSAI platforms, auto-defence, hot turns) is a way round that.
A player-flown F-16C module gets axis commands instead, so the policy's
heading, altitude and speed can be flown at the airframe's full agility, and
the simulator's own F-16 (`F-16C-DCS`, no handicap) is then the right model.

## What this branch keeps and drops

- **Keeps:** everything that is not about the DCS AI's flying: SIM_REV 13 (the
  energy weight 0.07, the red-bearing fix, ADAPTIVE at DCS speeds), the
  altitude-aware envelopes, the DCS-fitted AIM-120, the 14 headings and 5
  speeds (SIM_REV 14), the reward balance (SIM_REV 15), the `dcs_live.py`
  recording and results machinery.
- **Removed:** the DCS-AI auto-defence and hot-turn work (bridge versions 4
  to 6: `OPT autodefend`, `OPT hot`, `OPT hotturn`; the simulator's
  `AUTO_DEFEND`, `HOT_TURN_BANK`, the break-turn mode; the `F-16C-DCSAI-AD`
  platform; the fire/heading/altitude/speed masks while defending; the
  tests and docs). They stay on `feature/envelope-target-alt`. With a
  player-flown F-16C the policy flies its own defence, so the **defence
  reward term stays** (`-0.8 * (1 - tanh(time-to-go / 25 s))`: it was never
  changed for auto-defence) and the policy is trained to defend itself.
- **Kept from that work:** `RTB_ON_BINGO` off for both aircraft (the bridge
  is version 7 here), and the turn test's end-of-test behaviour.
- **Still there, off:** the route-mode platforms (`F-16C-DCSAI`:
  `CLIMB_MACH`, `CLIMB_THROTTLE`, `TURN_BANK_MAX`). They model the DCS AI on a
  route, which this branch does not use; say if they should go too.
- **Model to train:** `F-16C-DCS` against `F-16C-DCS` (not dcs_v10/v11, which
  were trained on the handicapped platform and expect its auto-defence).

## Plan

1. **Telemetry.** Export.lua (`Saved Games\DCS\Scripts\Export.lua`) runs in a
   different Lua environment from the mission-scripting bridge. It can read the
   player's aircraft and send the same `bvr_rl.dcs.v1` lines over UDP, so
   `dcs_world.py` and `dcs_live.py` work unchanged. Check the update rate.
2. **Control.** `LoSetCommand(command, value)` from Export.lua for pitch, roll,
   rudder and throttle. Command numbers (2001 to 2004 as far as I know) to be
   verified in `Scripts\command_defs.lua` of the install. The F-16's
   fly-by-wire turns stick position into g and roll rate, so an inner loop at
   20 Hz or more mirrors `f16_sim.py`'s autopilot: heading to bank to g,
   altitude to flight-path angle, speed to throttle. Its gains are fitted in
   DCS with step and turn tests like `dcs/turn_test.py`.
3. **Radar lock and launch.** Through the same command interface, to verify.
   The fire mask and the support rule stay in `dcs_live.py`.
4. **Red and events.** The mission bridge keeps flying red and reporting shots,
   hits and kills, and can keep removing unsupported missiles; only BLUE-1's
   control moves to Export.lua.
5. **Policy.** Train on `F-16C-DCS` with SIM_REV 15, then fly it.

## Open questions

- Export.lua may be disabled by servers with integrity checks; fine in
  single-player.
- The transport, the ports and the exact command numbers are not tested yet.
- A controller that is slower or looser than the simulator's autopilot is a
  new gap between training and DCS; measure it first (plan item 2).
