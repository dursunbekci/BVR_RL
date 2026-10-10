# Flying the player's F-16C with stick commands

Branch `feature/dcs-f16c-stick`, from `f3a8fb0` (SIM_REV 15, the head of
`feature/envelope-target-alt` on 10 October).

## Why

BLUE-1 is an AI unit, flown by the bridge through a route. The DCS AI follows
a route at 45° of bank (1.6°/s), climbs at about Mach 0.85 and does not light
its afterburner for a speed. Most of the work on `feature/envelope-target-alt`
(the F-16C-DCSAI platforms, auto-defence, hot turns; all removed here) is a way round that.
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
- **Also removed:** the route-mode platform `F-16C-DCSAI` and its parameters
  (`CLIMB_MACH`, `CLIMB_THROTTLE`, `TURN_BANK_MAX`, the autopilot's
  `climbMach`/`bankMax` modes), and the self-play grouping of platforms that
  differed only in how they climb. Self-play needs the same platform again.
- **Model to train:** `F-16C-DCS` against `F-16C-DCS` (not dcs_v10/v11, which
  were trained on the handicapped platform and expect its auto-defence).

## Step 1 and 2: the telemetry and control test

Written and tested against a mock of the Export API and a synthetic aircraft;
**not yet run in DCS**. The first run in DCS is the test of the test: the API
names, the command numbers and the signs below are from memory.

| File | What it is |
|---|---|
| `bvr_stick_export.lua` | The Export.lua module: a JSON line per frame to UDP 15401, `AXES` / `RELEASE` / `PING` / `CMD` lines in on 15402, a watchdog that zeroes pitch, roll and rudder 0.5 s after the last command. Every DCS call is inside `pcall`; missing functions are reported. |
| `install_export.py` | Copies the module to `Saved Games\DCS\Scripts\bvr_rl\` and adds a guarded block to `Export.lua` (backup, `--check`, `--undo`; no administrator rights). |
| `stick_test.py` | `info`, `rate`, `pulses`, `throttle`, `hold`, `all`, `release`, `raw`. Stops and releases at 85° of bank, 50° of pitch, below 1500 m above ground, below 110 m/s, outside -3..8.5 g, or when telemetry stops. |
| `fake_export.py` | A synthetic aircraft behind the same interface: `stick_test.py all --fake`. |
| `test_stick.py` | Offline tests (needs `pip install lupa`): the Lua module in Lua 5.1, the UDP link, `stick_test.py` end to end with the roll and pitch signs both ways, the installer. |
| `../make_mission.py --stick-test` | A mission with only you, in the F-16C Viper, in the air at 7000 m, 280 m/s over the sea. No other aircraft, no bridge. |

Run it (Windows, from the repository, in the Python environment):

```
python dcs\stick\install_export.py                 # once; again after editing the .lua
python dcs\make_mission.py --stick-test             # needs pydcs; copy dcs\bvr_rl_sticktest.miz to Saved Games\DCS\Missions
```

Start the mission. Level out at about 300 m/s (Mach 0.9), trim, then take
your hands **off the stick and throttle**; if the physical axes stay assigned
and noisy they can fight the commands (unassign them for the test if so).
Do not pause or accelerate time. Then, in a second window:

```
python dcs\stick\stick_test.py info        # codes found, API functions missing
python dcs\stick\stick_test.py rate        # 10 s: lines per second, gaps, fields, units
python dcs\stick\stick_test.py pulses      # 0.1 stick pulses: signs, g and roll rate per unit, lag
python dcs\stick\stick_test.py throttle    # code against RPM and acceleration
python dcs\stick\stick_test.py hold        # +60 deg heading, back, +300 m, +40 m/s, closed loop
```

`all` runs the four in order and takes about four minutes. `hold` is the
prototype of plan item 2: the simulator's autopilot (heading to bank to roll
rate, altitude to flight-path angle to g, speed to throttle) on stick axes, 
with the gains `pulses` and `throttle` measured. Its turn rate at 60° of bank
is the first number to look at; once it works, try `--bank 75 --nz 6 --turn 90`
for the 8°/s the simulator's F-16C-DCS turns at. If anything goes wrong,
`stick_test.py release` (or move the stick) hands control back; the F-16's
flight control then holds 1 g and the bank, so recover by hand.

**Send back** everything in `dcs_runs\` named `stick_*` (frames as `.jsonl`,
`stick_gains.json`, `stick_summary_*.json`) and the last lines of
`Saved Games\DCS\Logs\dcs.log` mentioning `bvr_stick`.

What the first run decides:

- **`rate`:** whether a line per frame is fast enough (needs 20/s or more),
  and whether `source = "event"` (see the comment `install_export.py` puts in
  `Export.lua`) does better. At 60 fps an event at frame boundaries gives 30/s.
- **`info`:** whether the command numbers 2001 to 2004 are DCS's
  (`command_defs.lua` is read when it can be) and which `LoGet*` functions the
  Viper lacks. Radar lock and launch are not tested yet: `LoGetTargetInformation`
  and `LoGetLockedTargetInformation` are only recorded (`radar` in the frames).
- **`pulses`:** the sign of each axis (the tool does not assume them), the
  roll rate and g per unit of stick, the lag from command to response.
- **`throttle`:** whether code 2004 is the throttle and its range (-1 as idle
  is assumed), RPM and acceleration against the code, and the code that holds speed.
  The throttle stays at the last code after `release`: set it by hand afterwards.
- **`hold`:** how hard and how fast the airframe can be flown from 20 to 60 Hz
  commands, and the command-to-telemetry delay. That is the gap between the
  simulator's autopilot and DCS, measured before any policy is trained on it.

## Plan

1. **Telemetry.** (Test written, see above.) Export.lua (`Saved Games\DCS\Scripts\Export.lua`) runs in a
   different Lua environment from the mission-scripting bridge. It can read the
   player's aircraft and send the same `bvr_rl.dcs.v1` lines over UDP, so
   `dcs_world.py` and `dcs_live.py` work unchanged. Check the update rate.
2. **Control.** (Test written, see above.) `LoSetCommand(command, value)` from Export.lua for pitch, roll,
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
