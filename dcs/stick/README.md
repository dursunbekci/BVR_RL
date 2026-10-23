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
python dcs\stick\stick_test.py curve       # pulses up to full stick: response against command (the axes are not linear)
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
3. **Radar lock and launch.** Done (13 October): TWS through the target management switch, launch with the
   weapon release held 1.5 s (results log above). The fire mask and the support rule stay in `dcs_live.py`.
4. **Red and events.** Written (`dcs_live.py --stick`, "The live runner" below; tested on the fakes, not
   yet in DCS): the mission bridge keeps flying red and reporting shots, hits and kills, and keeps removing
   unsupported missiles; only BLUE-1's control moves to Export.lua.
5. **Policy.** Train on `F-16C-DCS` with SIM_REV 15, then fly it.

## Open questions

- Export.lua may be disabled by servers with integrity checks; fine in
  single-player.
- The transport, the ports and the exact command numbers are not tested yet.
- A controller that is slower or looser than the simulator's autopilot is a
  new gap between training and DCS; measure it first (plan item 2).

## First results in DCS (13 October)

Export.lua telemetry works: 78 lines a second at 125 fps, all fields present, units as
expected, model time 1.00 x real time. Axis codes 2001 to 2003 are right (every axis
responded, symmetric both ways; `command_defs.lua` could not be found, so they are the
defaults). Lag from command to response: 105 ms in roll, 228 ms in pitch.

`pulses` at 0.1 gave only 0.94 g and 61 deg/s per unit of stick, a fraction of what a
linear Viper stick would: the exported axes are **strongly non-linear** (or scaled). A
linear gain would saturate far below the airframe's limits, so `curve` measures the
response up to full stick and `hold` flies through the inverse of the measured curve.

`curve` (13 October): roll is smooth, 61 deg/s per unit up to 0.2 and 186 deg/s at full stick.
Pitch is flat near the centre (0.15 g at 0.1, 0.63 at 0.2, 1.66 at 0.35, 2.70 at 0.5) and the
0.75 and 1.0 points were cut off by the 3.5 g early stop (the pulses reached 3.9 and 4.1 g).
The stop is now 5.5 g up and -2 g down; a pitch pulse stopped at the limit is marked `*` and
left out of the stored curve (it is only a lower bound). Re-measure only pitch with
`stick_test.py curve --axes pitch` (`--max-nz` changes the limit).

`curve --axes pitch` with the 5.5 g stop (13 October): pull side 0.13 g at 0.1, 0.64 at 0.2, 1.68 at
0.35, 2.7 at 0.5, 4.3 at 0.75, 4.8 at 1.0 (1 s pulses, so about 5.8 g in all at Mach 0.9, 6.8 km: enough
for a level turn at about 80 deg of bank). The push side stops at about -2.7 g of change whatever the
command. Pull and push are now stored separately (`pitch_curve`, `pitch_push_curve`) and `hold` uses
each for its own direction; averaging them under-read the pull.

`throttle` (13 October, started by accident at Mach 1.04): the raw frames showed three things.
1. **The code is inverted**: -1 is the afterburner (fuel flow 6.8, RPM 102), +0.5 and +1 are lower and
   idle (fuel 0.54 and 0.13, RPM 87 and 68), 0 is about mid (fuel 1.0), and the afterburner lights only
   near -1. `stick_test.py` now finds the sign from RPM and fuel flow and stores `thr_sign`.
2. **The RPM printed earlier was half the real one**: the Viper has one engine, DCS sends `right = 0`, and
   the average halved it. The largest of left and right is used now (and `fuel_flow` is read too).
3. **The stick loop held the aircraft well** while the speed went from Mach 1.07 to 0.90: altitude within
   13 m, bank 0.2 deg, load factor 0.9 to 1.27 g.
The acceleration column was meaningless (the sweep ran through the transonic drag rise); the speed law is
now a PI on speed (position u = trim + 0.04 e + integral, code = sign * u), found by `hold`'s speed step.

`throttle` after the fix (13 October, started at Mach 1.13): the sign was found (inverted), RPM 70, 86, 94,
101, 102 and fuel flow 0.13, 0.48, 0.83, 1.19, 6.0 for u = -1 to +1 (afterburner only at the top), acceleration
-4.1, -1.5, 0.0, +0.9, +4.2 m/s2 at about 300 m/s: monotonic, thrust-neutral at u = +0.02 (Mach 0.94).
`hold --speed 280` first captures a speed with the PI loop (heading and altitude held on the stick), then runs
the steps from there.

`hold --speed 280` (13 October): the speed capture worked (316 to 283 m/s in 19 s on the PI loop, heading and
altitude held), then telemetry stopped for 3 s as the plan began. The abort message now says whether DCS is
still running frames with the model time stopped (a pause: the module sends an `idle` line once a second
while paused) or nothing at all arrives, and gives the last frame.

**First closed-loop `hold --speed 280` (13 October)**: it works. Speed captured 307 to 283 m/s in 16 s (PI on
the throttle); held 8 m altitude, 0.0 deg heading and 2 m/s speed in the first 8 s; altitude step +300 m: 90% in
14 s, overshoot 10 m; speed +40 m/s: 35 m/s in 24 s (afterburner is slow); command to the frame that shows it
9 ms p50, 14 ms p95 (DCS ran 125 fps). The turns, with a 60 deg bank limit, 3.8 and 3.4 deg/s (60 deg of bank
sustains 3.6 deg/s at 283 m/s, so that is the limit, not the airframe), peak 3.2 and 3.7 g. But the bank went to
76 and 80 deg against the 60 deg limit: the Viper rolls about 140 deg/s for a command of 0.5 to 0.8, follows
within 50 to 100 ms, and needs about 0.4 s to stop, so a bank loop without lead overshoots by 15 to 20 deg.
The roll loop now aims at the bank the roll rate would reach in 0.25 s (`Hold.T_LEAD`). The aircraft also
unloads to about 0.4 g for 0.3 s in a fast roll (the pitch axis lags 228 ms); altitude moved by metres.
Next: `hold --speed 280 --bank 70 --nz 5` (5 deg/s) and then `--bank 80 --nz 6` (the simulator's 8 deg/s).

**`hold --speed 280 --bank 70 --nz 5` with the roll lead (13 October): the inner loop is done.**
Peak bank 70.06 and 70.07 deg against the 70 deg limit (the 15 to 20 deg overshoot is gone). Turns of 60 deg:
half way in 4.4 s, mean 8.3 and 7.9 deg/s while more than 30 deg were to go, peak 8.7 and 9.5 deg/s, peak
4.0 and 4.3 g. The simulator's autopilot (F-16C-DCS, 80 deg bank limit, `dcs/turn_test.py --sim --speed 283`)
gives 6.2 to 9.3 deg/s mean and 9.8 to 11.3 peak: the player-flown Viper, driven through Export.lua, is as
agile as the simulator's model (route mode gave 1.6 deg/s). Altitude hold 7 m, speed hold 2 m/s, altitude step
90% in 14 s, command to the frame that shows it 8 ms p50 and 16 ms p95. Slow: the approach to the last 10 deg
(the proportional heading law, K = 2, as in the simulator) and the speed step (the afterburner spools slowly).
Steps 1 and 2 of the plan are done. Open: radar lock and launch (step 3), the policy loop on this link, a
speed range test (`hold --speed 400`), a model trained on F-16C-DCS at SIM_REV 15.

## Step 3: radar lock and launch (13 October; written from the module's own files, not yet run in DCS)

The F-16C's controls are (device, command, value) triples. `extract_f16_defs.py` ran the module's own
`command_defs.lua` and `devices.lua` and read `Input\F-16C\keyboard\default.lua` for the value a key sends on press
and release: `f16c_defs.json` (HOTAS is device 16, the target management switch up is 3010, the weapon release
button 3022, the master arm switch device 19 command 3002, the ICP A-A button device 17 command 3018, the FCR
power switch device 31 command 3001, MFD buttons devices 24 and 25 commands 3001 to 3020). The Export module takes
`CLICK <device> <command> <value>` and calls `GetDevice(0):performClickableAction`, the same call the key bindings
make. It always refuses the electrical, fuel, hydraulic, engine, gear and oxygen devices, the canopy and ejection
(CPT_MECH), the flight control interface, the countermeasures dispenser, and the stores jettison and alternate
release commands.

`python dcs\make_mission.py --stick-test --target` adds an unarmed AI F-16C (TGT-1) 60 km ahead, flying the same way.
Commands, in the order to try them (`cockpit.py` describes each):

```
python dcs\stick\stick_test.py click ufc.AA                 press a control by name (see f16c_defs.json)
python dcs\stick\stick_test.py radar --secs 40              what the radar and the world objects report, once a second;
                                                            lock the target yourself while it runs
python dcs\stick\stick_test.py lock                         A-A mode, target management up; --scan also slews the cursor
python dcs\stick\stick_test.py fire                         master arm, weapon release, watch the payload; arm back OFF
```
`lock` and `fire` keep heading, altitude and speed on the stick (they need the gains from `pulses`, `curve`, `throttle`).
What the radar needs to lock (the page, the cursor, the mode) is learned by trying: each step prints what the radar
reports afterwards, and `radar` saves the raw tables (`stick_radar_sample_*.json`) for the next step.

**First `radar` run (13 October)**: the target showed in the world objects all the way (56 km down to 10 km, 0.1 to 2 deg
off the nose), but `LoGetTargetInformation` and `LoGetLockedTargetInformation` stayed empty: the F-16C module may not export
its radar through those calls (they come from the Flaming Cliffs aircraft). The target was 2.2 to 3.7 km above the Viper
(2 to 4 deg at 50 km), at the edge of the radar's default vertical scan. Changes: `radar`, `lock` and `fire` take
`--match-alt` (fly to the target's altitude) and `--alt M`; `radar` also flies heading, altitude and speed now; `ind` prints
and saves the text of the cockpit displays (`list_indication`, read-only), to be run with and without a lock to see how the
lock shows on the HUD and the MFDs; `fire` presses A-A itself, says when it is a maddog shot (no lock) and prints which
stations changed.

**Target behaviour (13 October)**: in the first radar runs the closure rate was +270 to +780 m/s, not the +40 of a target
flying away at 240 m/s: the target (a CAP flight) turned to intercept the Viper, even unarmed, and closed to 10 km in
90 s. `make_mission.py --stick-test --target` now gives it task Nothing, ROE weapon hold and no reaction to threats: it
flies its straight route. A manual lock in the first screenshots showed the HUD target designator box and steering
circle, `ARM` and `4 MRM`, while the Export radar tables stayed empty: the lock has to be read from the display text (`ind`).

**`ind` on the Viper (13 October)**: `list_indication(id)` returns the drawn elements of a display by name (HUD_glass,
FCR_Antenna_Azimuth_Tic_0, SMS_AA_Weap1, ...), with a value on the lines that carry text; ids seen: 1 HUD, 4 an MFD showing
the radar page (FCR_Antenna tics), 5 an MFD showing the SMS A-A page, 6 UHF/VHF, 13 the EHSI, 16 countermeasures, 17 the
RWR. The first dump was cut at 2500 characters; the module now sends each display whole in its own message (up to 30000
characters) and `ind` ... `inddiff FILE_A FILE_B` lists the lines present in one save and not the other, to see what a
designation or a lock adds to the HUD and the radar page.

**What the displays say about the radar (13 October, `inddiff` of three saves: not designated, designated, second press)**:
the first target management up adds `HUD_AA_TD_box` to the HUD and, on the radar page (the MFD that draws
`FCR_Antenna_Azimuth_Tics_Root`), `FCR_TrackedTgtPlaceholder`, `FCR_trackedTargetSymbolPlaceholder`, `FCR_Symbol_Unknown`,
`FCR_SAM_Target*` and `Azimuth_Scan_Limits_*`; the second drops `Cursor_Root`, the `FCR_SAM_*` pointers, the scan limits and
`Max/Min_Search_Altitude` from the radar page (a single-target track), and the HUD swaps the box for `HUD_TLL` (target locator
line, drawn when the target is out of view) and `HUD_TargetAngleReadout`; the RWR gets a threat symbol from the first press on
(the target's radar). The module reads these markers every 20 frames (`rd` in the telemetry: fcr, tracked, cursor, sam, limits,
td, tll, threat) and `cockpit.lock_state` turns them into search / designated / stt; `radar`, `lock` and `fire` use it.
Marker names are in `MARK` in the module (override with `BVR_STICK_CFG = {markers = {...}}`).

**`radar --match-alt` (13 October)**: the reading follows the hand presses: `search`, then `designated` (10 s), then `stt`
(11 s, stable for the 50 s after), with the target 55 to 59 km ahead, 0.1 deg off the nose, within 15 m of the Viper's altitude
(`--match-alt`), closing at 60 m/s on its passive route. The ICP's A-A button toggles (pressed in A-A mode it goes back), so
`lock` and `fire` press it only with `--aa`; otherwise A-A is assumed selected.

**`lock` after a hand-dropped lock (13 October)**: `click hotas.STICK_TGT_MANAGE_DOWN` took the radar back to `search`;
`lock` then sent target management up (the module answered ok) and the radar stayed in `search`. Either the radar cursor was
no longer over the target (dropping a lock probably re-centres it; the target was at 57 km, 31 NM) or the 0.2 s press was too
short. `cursor` slews the radar cursor out and back in range and prints, after each move, which lines of the radar page
changed: the lines that follow the cursor are what an automatic lock can steer by.

**HUD markers for the master mode (13 October, `inddiff` of a NAV save and an A-A save)**: in NAV the HUD draws the text `NAV`;
in A-A it draws `AA_Gun_Root`, `4 MRM`, `HUD_AA_AIM120_Selected`, `HUD_AA_AIM120_Reticle` and `HUD_AA_AIM120_Diamond`. The module
reports `aa` and `msl120` in `rd`; `lock` and `fire` press the ICP A-A button (a toggle) only when the HUD is not in A-A
(`--aa` always, `--no-aa` never) and `fire` warns if the AIM-120 is not the selected weapon. The radar page was the same in
both saves apart from the OSB labels and two numbers beside the cursor line (34 and 10, then 35 and 11), which look like the
search altitude band at the cursor's range. In the latest screenshot the cursor mark sat nearer than the contact: target
management up designates only a contact near the cursor, which fits a press that does nothing.

**`cursor` probe and `lock` (13 October)**: slewing the radar cursor out and back in range 8 times each way changed none of the radar page's
lines (the altitude band beside the cursor, 34 and 11, did not move; only a slowly drifting `174 24x` did), and `lock` (A-A mode seen
by the HUD, so not toggled) sent target management up with no effect, as did the same key by hand. The earlier hand run that
worked had the radar page as the sensor of interest; A-A mode puts the weapons page on the right MFD (seen in the screenshot) and
the cursor slew and target management act on the sensor of interest. `lock` now first sends display management left (`--soi
left`, `right` or `none`): the keymap sends -1.0 for left and down, +1.0 for right and up. `click` takes `--release` to set an
axis back after `--hold`.

**Lock works from a script (13 October, corrects the paragraph above)**: the cursor and target management did nothing because the
HOTAS controls were delivered the wrong way, not because of the radar's state. `GetDevice(0):performClickableAction` (a mouse
click on the element) reaches the MFD and the drop-lock switch but does not move the radar cursor; `GetDevice(dev):SetCommand`
(how the module's key bindings deliver a command) does, and so does the cursor axis through it (`cursortry`: A to F and H, the
click route and the global command queue, did nothing; G and I, SetCommand, moved it). `CLICK` takes a fifth word `c`, `s` or
`l` for the route; the HOTAS controls (device 16) now go through `s` by default (`click --mode` overrides). The cursor lines I
read as unchanged were slewed by the old route, so they proved nothing: with the working route, the radar page's
`174 xxx` / two numbers lines follow the cursor in range, and while a contact is under the cursor the page lists
`FCR_TUC_contactData`, `FCR_TUC_contact_Altitude` and the contact's altitude in thousands of feet (it shows for one cursor step as the
cursor crosses the contact). `lock --scan` slews the cursor in range until that appears and presses target management up
only then: designated after 7 steps, a second press gave the single-target track (radar `stt`), 59 km ahead at the target's altitude.
Display management is no longer pressed by default (`--soi none`): in one run it left the left MFD on its menu page.

**Launch works from a script (13 October)**: with the radar in single-target track (`lock --scan`) and `ARM` on the HUD,
`fire` launched an AIM-120C at 58 km: the HUD's count went from `4 MRM` to `3 MRM` and an `AIM_120C` (level 4 4) joined the world
objects within a second. What made it work was the press length: weapon release has to stay pressed until the missile leaves the
rail. Held for 0.4 s it launched nothing by any of four routes (device SetCommand, clickable element, global command queue,
DCS's generic pickle 350/351); held for 1.5 s (`--release-hold`, the default now) the SetCommand route fired. The missile showed in
the default world objects, not in the `ballistic` list, and the payload table stays `None` for this module, so `fire` goes by
the HUD's `N MRM` count and new world objects. Master arm: device 19 command 3002, value 1 (clickable route) shows `ARM` on the HUD.

**Aimed lock, `lock --aim` (13 October; written from the `cursormap` pages, tested on the fake cockpit only)**: the range-only
scan finds a target only dead ahead and level. `cursormap` showed what to steer by instead: the radar page's `FCR_Bearing_range`
line is the cursor's bearing and range from the bullseye (whole magnetic degrees and NM: at the bottom edge, where the cursor is
on our own aircraft, it read 174/255 with the HUD heading 354 on a true-north course, so the bearings are magnetic, 6 deg east
here); left and right do move the cursor (the bearing rose to 177 with left, fell to 170 with right, at the edge within a second);
`Cursor_Root` lists `Max_Search_Altitude` / `Min_Search_Altitude`, the height coverage at the cursor's range in thousands of feet;
the page also shows RWS, A6 (60 deg either side), 4 B (four bars), the range scale and `SOI`. `make_mission.py` now puts the
bullseye near the fight (stick test: halfway to the target; 1v1: the middle), and the module sends the magnetic heading
(`myaw`, `LoGetMagneticYaw`) for the variation. `lock --aim`: (1) with the cursor at the bottom edge, its readout places the
bullseye (or `--bull X Z`); (2) each step converts the readout to the cursor's range and azimuth from us and moves it toward
the target's (from the world objects) with a pulse sized by the slew rate it has seen so far, range or azimuth, whichever is
further off; a tolerance widens when its axis overshoots; (3) near the target's range, if its altitude is outside the coverage,
antenna elevation up or down; (4) at the point, if nothing is under the cursor, a small search around it (1-3 NM, 2.5-5 deg);
(5) target under the cursor: target management up, and again for the single-target track. A target more than 58 deg off the
nose is refused (turn first). On the fake (my guess at the slew rates and a four-bar coverage of 3.4 deg either side) it locks
targets at 0, +30 deg (+2 km), +10 deg (+6 km, antenna up) and -45 deg (-3 km) in 7 to 11 steps.

**Aimed lock, first DCS runs (13 October)**: the mission was the old one, with DCS's default bullseye 262 NM away, so a whole
degree of the readout was 4.6 NM (8 deg at the target's range). Dead ahead: the cursor reached the target's range in 8 steps,
then the azimuth hunted (small left pulses the readout could not see, about 80 of them) and the point search (2.5 deg) was
smaller than the readout's error: not locked. 30 deg right: locked after 72 steps. The third run found run 2's lock still on
(`--relock` now drops it first). `fire` launched at 55 km on a target flying away at 240 m/s: after 60 s the missile was still
36 km short (closing at 0.13 km/s), a shot out of range, not a fault. Changes: the readout's uncertainty is computed from the
bullseye's distance (2 x (half a degree at that distance + half a mile)) and widens the azimuth tolerance; the search around the
target's place is rows swept left to right in 2-deg steps timed by the learnt slew rate (the readout cannot see steps that
small when the bullseye is far), at the target's range and then 1, 2, ... NM either side; slew rates are learnt only from
moves bigger than the readout's error. On the fake, with the bullseye 261 NM away, it locks at 0, +30, -20 (+3 km) and
+45 deg (-3 km); with it 15 NM away, in 7 to 11 steps without a sweep.

**Aimed lock, second DCS runs (13 October)**: the cockpit still read from a bullseye 141-265 NM away (the mission's bullseye did
not reach the Viper, or the old mission was flown), and finding it at the bottom edge made the radar step its range scale down
(the cursor then moved under 1 NM per 0.6 s pulse) and back up near the top (the cursor jumped from about 30 NM back to 12, again
and again: 150 steps in the 30 deg run). The `--offset` setup aimed at a heading fixed at the start, so the angle drifted past
30 to 39-41 deg; the climb for `--dalt` was held to 30 m/s. Changes: (1) the readout is read from our own aircraft: with
BULLSEYE off in the DED the F-16's radar page gives the cursor's bearing and range from ownship; `lock --aim` sees a bullseye
readout (range beyond the scale or bearing outside 60 deg) and switches it with the ICP (LIST, 0 MISC, 8 BULL, 0 M-SEL, RTN),
checks the readout again, and only falls back to the bottom-edge bullseye if the switch did not take (`--ref own|bull` forces
one); no visit to an edge, 1 deg and 1 NM anywhere. (2) If the target is beyond 85% of the range scale the scale is stepped up
with OSB 19 first. (3) Missions put the bullseye 450 km away on purpose, so a bullseye readout cannot look like an ownship one.
(4) `--offset` follows the target's bearing (the heading is recomputed from every world-objects frame) with up to 60 deg of
bank; `--dalt` climbs at up to 80 m/s; setup is done within 3 deg and 300 m. On the fake: locks in 2-6 steps after the switch.

**Aimed lock, third DCS run, and the TWS lock (13 October)**: the ICP keys (LIST, 0, 8, M-SEL, RTN) did not switch the readout
("still 174/227"), so the bottom-edge bullseye search ran again; the cursor kept jumping from about 30 NM back to 12 after small
forward pulses near the target's range (more than 15 times; not yet explained), the sweeps' 0.05 s pulses were too short to move
it, and after 200 steps the target was only designated. The screenshot showed the cursor on the contact and the page in A 3:
the scan covers 30 deg either side of the cursor, not of the nose. Changes: `lock --tws` locks without the cursor: target
management right puts the radar in TWS, where it makes track files itself; target management right again bugs a track
(designated: an AIM-120 can be fired at it) and target management up gives the single-target track (`--no-stt` stops at the
bug). The page's mode (RWS/TWS), azimuth (A1/A3/A6) and bars are read and printed. `lock --aim` prints the DED (display 6) after
the ICP keys, to see where the key presses went, and its sweep pulses are at least 0.12 s.

**TWS lock works (13 October)**: a short target management right did nothing in RWS; held 1.5 s it put the radar in TWS (the
page then draws `TWS` and `AA_RadarTracksKeeper`). Short presses 1.5 s apart bugged the target on the fifth, about 14 s after
entering TWS (A6, 4 bars: a scan or two to build the track file); with the bug the page draws the AIM-120 launch zone
(`DLZ_AIM120_root`, `DLZ_Rmax1`, `DLZ_Rmin1`, ...) and the scan limits, and the display markers read designated. Target
management up after that stayed designated (no single-target track), which is fine: an AIM-120 is fired at a bugged TWS
track. `lock --tws` now holds the switch straight away and then presses it about once a second until the launch zone is drawn
(or the markers say designated), and stops at the bug (`--stt` asks for the single-target track as well).

**TWS timing (13 October)**: from RWS, the bug came on the 7th press, 20 s after entering TWS (presses 2 s apart after 4 s of
scanning): the track file is what takes the time. Run again with the radar already in TWS (the track file kept), the first
press bugged it, 5 s after the start (4 of them the initial scan). In TWS the page reads A2 (25 deg either side of the cursor).
For the policy this means: put the radar in TWS once, early in the fight, and leave it there; FIRE is then a press of target
management right and the held weapon release, a few seconds.

**Launch at a TWS bug (13 October)**: `lock --tws` from RWS bugged the target 13 s after entering TWS; `fire` launched at the bugged
TWS track (HUD 4 -> 3 MRM, `AIM_120C` in the world objects, closing on the target). The two offset runs after it set up on the
missile, not the target (it was the nearest world object, lofting at 10 km, 17 km away): `nearest()` now takes aircraft only
(level1 1), so a weapon in flight is never the target.

**TWS off the nose (13 October)**: with the target 28 deg off the nose, 13 presses over 36 s bugged nothing: in TWS the page reads
A2, a scan 25 deg either side of the cursor, and the cursor was in the middle. `lock --tws` now moves the cursor to the
target's bearing first, without reading it back: held toward the target's side for 1.5 s it stops at the display's edge
(60 deg), and a timed move back of (60 - |bearing|) / 100 deg/s leaves it near the target (100 deg/s from an aim run: 39 deg in
0.37 s); 10 deg of error is still inside the scan. Once a track is bugged the scan follows the bug. (The fake now slews at 100
deg/s and its target-under-cursor window is 5 deg either side.)

**TWS off the nose works (13 October)**: with the cursor moved to the target's bearing first (edge, then a timed move back at 100
deg/s), the target was bugged at +28 deg (2 presses, 13 s from RWS), -38 deg (1 press, 7 s, the radar already in TWS) and -23 deg
with us 3 km above it (3 presses, 13 s). The 100 deg/s estimate is good enough for a 25 deg scan.

## The live runner: the policy flies your Viper (`dcs_live.py --stick`)

The mission's bridge (`bvr_bridge.lua`) still reports both aircraft, shots, hits and kills, flies red and removes unsupported
missiles; it never takes BLUE-1, because it only does on a `CMD` line and `--stick` sends none. The policy's observations are
built from the bridge's reports exactly as before. Its 1 Hz commands go to a `StickPilot` (`pilot.py`), a thread on the Export
link that:

* flies heading, altitude and speed at 50 Hz with `stick_test.Hold` (the inner loop measured above) with the simulator's limits:
  80 deg of bank (the simulator's `PHI_MAX`), up to 8 g, climbs and descents up to 80 m/s;
* runs cockpit jobs between frames, the aircraft flown all the while: **radar** (A-A, TWS with target management right held,
  the cursor to the target's bearing; repeated every 15 s while no track is bugged, the cursor only moved when the bearing has
  changed by 10 deg) and **fire** (bug the TWS track, master arm on, weapon release held 1.5 s, launch confirmed by the HUD's
  `N MRM` or a new AIM-120 in the world objects); a fire job's result ("launched" or "timeout" and why) goes into the fire
  bookkeeping (`fire_pending`, `fire_log`, the "fire requests" in `results.csv`);
* on a pause, a frozen DCS or a guard (below 300 m, under 80 m/s, beyond -3.5..9.5 g) lets go of the stick and tries again;
  at the end of an episode it turns master arm off and releases the axes.

To fly it:

1. `python dcs/stick/install_export.py` (once) and `python dcs/stick/stick_test.py all` (once: writes `dcs_runs/stick_gains.json`).
2. `python dcs/make_mission.py --blue-player` -> `dcs/bvr_rl_1v1_stick.miz` (you in BLUE-1, the F-16C; RED-1 the DCS AI).
3. `python dcs_live.py <checkpoint.zip> --stick --platform F-16C-DCS`
4. Start the mission, take BLUE-1, trim for level flight and take your hands off the stick and throttle. Ctrl+C in the
   dcs_live window gives you the aircraft back.

Tested on the fakes (`test_sim.py`, "DCS live stick"): the policy's commands reach the pilot and no `CMD` reaches the bridge; a
FIRE runs A-A, TWS, the bug, master arm and the held release on the fake cockpit, and its launch comes back as "fire launched".

## Making the simulator fly like the DCS Viper (`stick_test.py manoeuvre`, `manoeuvre.py`)

The policy only gives heading, altitude and speed targets; how fast the aircraft gets there is the airframe and its
autopilot. For the policy to meet in DCS the aircraft it trained on, the simulator's F-16C must fly those targets as the
stick pilot flies the DCS Viper. The stick pilot now uses the simulator's altitude law when the command carries the
policy's flight-path limits (`Hold.axes(..., dive, climb)`: 0.0003 rad of flight path per metre of altitude error, limited
to the platform's climb angle and 25 deg down, and 7 g per rad of flight-path error), as the simulator's autopilot does;
`dcs_live.py --stick` passes `altFPA` and `climbFPA` on. Heading to bank (2.0, 80 deg) and speed to throttle (PI) were
already alike. What is left to differ is the aircraft:

1. `python dcs/stick/stick_test.py manoeuvre` (in DCS, stick-test mission, trimmed level at 7 km about Mach 0.9, hands off):
   about 7 minutes of targets relative to the start: hold 10 s, turn right 90, turn left 180, back, climb 2000 m, descend
   2000 m, accelerate 100 m/s (at most 460), decelerate to 80 below the start (at least 220), turn 170 at that speed.
   Saved every 0.1 s: `dcs_runs/manoeuvre_<time>.json`.
2. `python dcs/stick/manoeuvre.py dcs_runs/manoeuvre_<time>.json` (no DCS): the simulator's F-16C-DCS flies the same targets
   from the same start; per segment turn rate, time to 90%, peak bank and g, speed lost in turns, vertical speed, overshoot,
   acceleration, final Mach, DCS against the simulator; a plot of both (`_compare.png`) and what the differences point at.
3. `... --fit`: a Nelder-Mead over factors on the airframe's thrust (T_MIL_SL and T_AB_SL together), CD0, K_IND, CL_MAX and
   DRAG_RISE_PEAK, minimising speed RMS / 5 m/s + heading RMS / 5 deg + altitude RMS / 50 m over the whole program (about
   1.4 s a simulation, 5 minutes); the result is compared again (`_fit.png`) and saved to the library as a new airframe and
   platform, `F-16C-DCSFIT` (`--save-as`), to train with. Missile envelopes are keyed to the missile, so none need
   recalculating. Checked on a record the simulator made with thrust x0.85, CD0 x1.3, CL_MAX x0.9: the fit took the
   mismatch from 2.19 to 0.17 and found CL_MAX x0.90 exactly, thrust x0.885 and CD0 x1.20 (thrust and drag trade against
   each other in one record; the flight is what has to match).

**First DCS comparison (14 October, `manoeuvre_20261014_084543`, 6.3 km, 304 m/s, fuel 0.86)**: climbs and descents matched
(vertical speed 81 against 79 and -114 against -103 m/s, the same times to 90%), and so did the bank (80 deg both). The turns
did not, and the cause was the DCS pilot, not the airframe: it pulled 7.0 g at 80 deg of bank where holding altitude takes
1/cos 80 = 5.76 g (the simulator's autopilot flies exactly that), so DCS's turns climbed 80-340 m, turned faster (12-13
against 8.5-10 deg/s) and bled more speed (49 against 15 m/s in 90 deg, 121 against 41 in 180). The pitch stick came from
the curve measured at 243 m/s, open loop; at 304 m/s the same stick gives more g. The fit then blamed drag (K_IND x1.82, CD0
x2.0 at its bound), which the turn rates did not support: that F-16C-DCSFIT is not to be used. Real differences: DCS
accelerated faster (2.84 against 2.19 m/s2) and also slowed faster (-2.97 against -2.20), more thrust and more drag (four
AIM-120s on pylons). Fix: `Hold` closes the loop on the measured g (an integrator, 1.5 g per g-second, up to 3 g, not while
the stick is at its stop); on the fake with a stick 40% stronger than its curve it holds 5.6 g at 80 deg and the altitude
within 6 m (63 m without the loop). The fit's bounds were widened (CD0 up to x3.5, K_IND to x2.5).

**Second try (14 October)**: with the g loop the Viper held the g the altitude law asked for, 5.76 g at 80 deg of bank,
which at 7 km it cannot sustain: from 312 m/s (Mach 1.0) it bled to 110 m/s in the 180 deg turn and the speed guard
stopped the run. The simulator never gets there: its autopilot limits g to what the wing gives (dynamic pressure x S_REF x
CL_MAX / weight, at most NZ_POS_MAX) and the bank to what leaves BANK_NZ_MARGIN (0.85) of that to hold the flight path. The
stick pilot now does the same, from the policy's platform's airframe and DCS's fuel (`Hold(..., airframe=)`; `pilot.py`,
`dcs_live.py --stick` and `stick_test.py manoeuvre --platform` pass it): at 7 km, 86% fuel, F-16C-DCS, 80 deg down to
about 250 m/s, 73 at 200, 60 at 150, 38 at 120.

**Third try (14 October)**: stopped again at 110 m/s, but the record showed why: a pitch oscillation. Rolling out of the 80 deg
turn the g took about a second to unload (6 g still at 50 deg of bank: climbing 24 m/s); the altitude law then asked for a
push and the fast g correction (1.5 g per g-second, added) drove the stick from stop to stop: +3.4 to -1.5 g every 4 s, the
heading swinging back and forth; in the 180 deg turn the stick stayed at its stop for 8 s at 20-23 deg of angle of attack and
the speed fell from 290 to 180 m/s. Changes to `Hold`: (1) the g correction is a factor on the pull (the stick's g per unit
grows with dynamic pressure; an added g learnt at 6 g was wrong at 1 g), learnt slowly (0.3 per second on the relative
error, between x0.5 and x1.6) and only on a steady pull (above 2 g, not rolling faster than 20 deg/s, the demand steady,
demand and measured g through the same 0.5 s filter, not at the wing's limit or the stick's stop); (2) the g that holds the
flight path is for the bank 0.4 s ahead when rolling out (`NZ_LEAD`), so the g comes off as the wings come level. The fake
now has DCS's lag and a g rate limit (0.25 s, 6 g/s): with a stick 40% stronger than its curve, the factor learnt is 0.71
(true 0.71) and the altitude stays within 21 m through turn and roll-out (65 m without the correction).

**Fourth try (14 October, `manoeuvre_20261014_093818`)**: the whole program flew, the roll-outs smooth (no oscillation), and
climbs, descents and speed changes compare as before (vertical speed 79 against 77 m/s; DCS accelerates 3.08 against 2.17
m/s2 and slows -2.91 against -2.20: more thrust and more drag than the simulator's F-16C-DCS). The turns still pulled 6.7-6.9 g
for 5.76 asked (turn rate 12 against 8.8 deg/s, 75-145 m of climb): the stick factor did not learn. Likely why: at 80 deg of
bank the altitude law's demand swings with every half degree of flight path (0.35 g), and the steadiness gate (0.3 g) shut
the learning out. The gate is now 1 g and the learning 0.5 per second; the manoeuvre record keeps the pilot's own state (g
asked, stick factor, learning or not, stick position) and the comparison prints it for the hard part of each turn.

**Fifth try (14 October, `manoeuvre_20261014_095036`)**: the pilot's own state answered it. Above 70 deg of bank it asked
5.71 / 5.78 / 5.32 g and flew 5.31 / 4.99 / 5.35, with the stick at its stop 65 / 50 / 58% of the time (so the factor could
not learn: never against the stop): the DCS Viper does not give the g the simulator's wing does. Only the roll-in, at full
speed, overshot (6.6-6.7 g). In lift coefficient: 6.7 g at 300 m/s is CL about 1.0, 5 g at 250 m/s about 1.1, where the
simulator's F-16C has CL_MAX 1.45 (DCS's flight control limits the angle of attack; four AIM-120 on the rails). That is the
aircraft, and it is measured directly, not fitted: `manoeuvre.py` takes every sample with the stick at its stop above 2 g,
CL = g x weight / (dynamic pressure x S_REF), and prints the 90th percentile; `--fit` sets CL_MAX to it and fits thrust and
drag only. The simulator and the stick pilot both take their g and bank limits from that CL_MAX (the pilot from the
platform's airframe), so both cut the bank where the wing runs out.

**The fit on the fifth record (14 October)**: the measured wing limit came out higher than the rough estimate: CL 1.28 (90th
percentile of 102 samples at the stick's stop), median 1.15, against the simulator's 1.45; `--fit` fixed CL_MAX to 1.28 (x0.881)
and fitted thrust x1.147, CD0 x1.867, K_IND x2.196, DRAG_RISE_PEAK x0.795: mismatch 6.34 -> 4.10. The energy now matches:
acceleration 2.53 against DCS's 2.81 m/s2 (was 2.23), deceleration -3.05 against -2.74 (was -2.13), final speeds within 3 m/s,
speed lost in the 90 and 180 deg turns 56 / 108 against 50 / 116 m/s (was 21 / 48); climbs and descents as before. Drag this
high is this loadout's (four AIM-120 on pylons), not a clean F-16. What is left is the turns' first seconds: DCS turns 12-12.5
deg/s against the simulator's 9.2 while the heading error is above 30 deg, and climbs 90-145 m, because the roll-in at full
speed still pulls 6.6-6.7 g for a moment (the simulator's level turn at 80 deg is 5.76 g). That is the pilot's roll-in, not
the airframe, and the fit cannot (and should not) absorb it. Next: fly the program with the fitted limits
(`stick_test.py manoeuvre --platform F-16C-DCSFIT`) and compare against F-16C-DCSFIT.

**Sixth try, with F-16C-DCSFIT's limits (14 October, `manoeuvre_20261014_101605`)**: speed, climbs and descents matched
again (speed lost in the 180 deg turn 108 against 112 m/s, acceleration 2.52 against 2.80, deceleration -3.05 against
-2.74; the wing limit measured again 1.26 for the 1.28 set), mismatch 3.76. The turns did not: DCS's heading ran ahead
(12-12.5 against 9.2 deg/s), then stopped for 8 s, 20 deg short, while the aircraft climbed 110-150 m. The record (this
time uploaded with the comparison) showed two causes, neither of them thrust or drag:

1. **The heading was the nose's.** `LoGetSelfData().Heading` is where the nose points; the simulator's heading, and the
   policy's (`dcs_world.py` takes it from the velocity), is the course. In a 6 g turn the nose is inside the flight path by
   the angle of attack: 15-22 deg in the raw frames of the third try (nose minus course against AoA: 15.2 at 15.5, 21.1
   at 21.4 deg). The pilot steered by the nose, so it rolled out about 20 deg early; the "heading" then fell back to the
   course as the g came off, which is the 8 s stop. Now `Frame.crs` is the course from `LoGetVectorVelocity` (x north,
   z east; the nose when slower than 30 m/s or without it), `Hold` and `pilot.py` steer by it, and the manoeuvre record
   keeps the course as `hdg` and the nose as `nose`. `manoeuvre.py` says when a record is from before.
2. **The g lags.** The pilot's demand turns into DCS's g as a first-order lag of 0.9 s building up and 2.0 s coming off
   (`g_lag`: fitted over the record wherever the stick was off its stop; the error is 0.42 g RMS with the lag, 1.18 g
   without). The roll-in sank 30 m before the g came, the altitude law then asked for 6.6 g at 80 deg of bank, and on
   the roll-out the g stayed: 4.5 g at 53 deg of bank, 35 m/s of climb. The simulator's g was instant. `f16_sim` now has
   `NZ_TAU_ON` and `NZ_TAU_OFF` (airframe parameters, 0 by default: every existing platform and checkpoint flies exactly
   as before; 52/52 of `test_sim.py`); `manoeuvre.py` prints the measured lag and `--fit` writes it into the fitted
   airframe with the measured CL_MAX, and fits thrust and drag on top (`--no-lag` leaves it out). The fit now always starts
   from F-16C-DCS (`--fit-from`), also for a record flown with an earlier fit's limits, so factors do not compound.

Next: fly the program again (the pilot now steers by the course), then `--fit` on that record.

**Seventh try, steering by the course (14 October, `manoeuvre_20261014_104111`)**: the turns now line up. Before the fit,
against the F-16C-DCSFIT of the fifth record: turn rate 9.3 / 9.9 / 8.3 against 9.2, speed lost in 180 deg 109 against
107 m/s, mismatch 3.30 (3.76 before). The first `--fit` with the lag (1.40 / 1.85 s) took it to 2.40, and with it the
simulator's turns climbed and pulled as DCS's (peak g 6.49-6.56 against 6.53-6.64, altitude 130 / 54 against 144 / 57 m).
One thing was still the pilot's: rolling out of the 180 deg turn, the demand came down while DCS's g lagged, the lag read
as a stick too strong, and the stick factor fell from 0.96 to 0.54 in 4 s; the next turn asked 7 g with the stick at 0.5-0.7
(flew 4.9), which also made the g look 1.4 s slow to build. Two changes: `Hold` now passes its demand through DCS's lag (the
airframe's NZ_TAU_ON/OFF, else 0.9 / 2.0 s) and learns the factor only from what that does not explain, and only once the
lag has caught up; `g_lag` uses only samples with the factor within 10% of 1. Both records then give the same lag: 0.93 /
1.97 s (fifth) and 1.00 / 1.93 s (seventh).

**Check flight, the fixed pilot against the corrected F-16C-DCSFIT (14 October, `manoeuvre_20261014_112025`)**: passes,
with no fitting on this record: mismatch 2.73; turn rates 9.3 / 10.0 / 9.4 against 9.5 / 9.8 / 9.4 deg/s, peak g 6.57-6.72
against 6.61-6.65, the climb in each turn 150 / 59 / 131 against 140 / 54 / 138 m; the return turn now matches (speed lost
63 against 68 m/s; it was 50 against 66 with the wrongly learnt factor); acceleration 2.84 against 2.63, final speeds within 3
m/s. Measured again: CL 1.27 (1.28 set), g lag 0.85 / 2.00 s (1.00 / 1.93 set). Left as it is: the simulator slows at idle
about 20% faster (-3.30 against -2.75 m/s2). **F-16C-DCSFIT is the platform to train on.** The stick factor held 1.00 in the
turns but learnt 0.69 in the pull-up into the climb at 300 m/s, and that one factor then under-pulled the last turn at 176
m/s (4.05 g asked, 3.24 flown): it is now learnt per speed (150-450 m/s in 50 m/s bands, interpolated), so what is learnt at
one speed stays there.

## F-16C-DCSFIT in the library, and the launch delay (14 October)

`library/user/airframe/F-16C-DCSFIT.json` (the `--fit` of `manoeuvre_20261014_104111`, with the sources of the fitted and
measured values) and `library/user/platform/F-16C-DCSFIT.json` (F-16C-DCS with that airframe) are now in git. The missile is
AIM-120C-DCS, whose envelope (`library/envelopes/AIM-120C-DCS-b8a3b10542.npz`) is already there: nothing to recalibrate.

**Launch delay.** In DCS the fire decision is followed by cockpit work: the cursor to the target's bearing, target management
right until the TWS track is bugged, master arm, the weapon release held 1.5 s. Seconds pass, the aircraft keeps flying the
policy's commands, and if the track is lost meanwhile nothing launches. The simulator launched in the decision's own step.
Platforms now have `LAUNCH_DELAY_MIN` / `LAUNCH_DELAY_MAX` (s, 0 by default: every existing platform fires as before): the
launch goes a uniformly drawn time after the decision, the fire mask is shut meanwhile (as `dcs_live.py` shuts it while a
request is pending), and at launch time it goes only if a missile is left and the target is still tracked within the 60 deg
fire cone, else it is dropped (`fire_dropped` in the episode info), as a DCS request times out. The range is not checked again:
DCS launches at whatever range the bugged track is then. It applies to the agent and to self-play snapshots (1v1; 2v1 fires at
once). `dcs_live.py` sets it to 0 for itself, since DCS does the work. F-16C-DCSFIT has 3-7 s, an estimate from the pilot's
fire job; `dcs_live.py --stick` now writes each request-to-launch time (`launch_delays_s`) and why each timed-out request did
(`fire_timeout_reasons`) into `results.csv`, to replace it.

**Measuring the launch delay (`fire_delay.py`).** `python dcs/stick/fire_delay.py measure [--trials 10] [--fire 1]` (DCS,
stick-test mission, the target ahead, hands off) runs the live runner's own pilot (`StickPilot`): it flies so the target is
0, +20, -20, +35, -35 deg off the nose in turn, and each time drops the bug (target management down) and times a new one
exactly as a fire job does it (`StickPilot.time_bug`), so no missile is spent; `--fire N` ends with N real launches, which
also time master arm, the held release and the launch seen (each fire job now records `bug_s` and `total_s`). Saved as
`dcs_runs/fire_delay_<time>.json`. `python dcs/stick/fire_delay.py set F-16C-DCSFIT <files>` takes those and any
`results.csv` from `dcs_live.py --stick` (its `launch_delays_s`, request to launch end to end, including a wait for a radar
job: the best evidence, and the one used once there are five) and writes the 10th and 90th percentiles, to 0.5 s, as the
platform's LAUNCH_DELAY_MIN / MAX (`--dry-run` only shows them). On the fake: bugs 3.2-3.7 s, a launch 3.0 s after its bug.

**The launch delay, split (14 October).** The measured 7-14 s (`fire_delay_20261014_141533`, every trial with the bug
dropped first) cost the warm start dcs_v8 most of its fights: 60 fights against ADAPTIVE on F-16C-DCSFIT, 33% won with no
delay, 38% with 2.5-4 s, 13% with 7-14 s (shots launched 1.77 -> 1.32 a fight, 0.20 dropped); a training run with it sat at
0-10% for 500k steps. Most of it is the bug, and a pilot bugs the target before deciding to shoot. So: the stick pilot's
radar job (every 15 s while idle in `dcs_live.py --stick`) now bugs a held track ahead of a shot (up to 6 s of presses, the
target within 50 deg) and sets master arm; a fire request then waits only for the release (the fake: 3.8 s from request to
launch). The simulator follows it: LAUNCH_DELAY_MIN/MAX is the release, and LAUNCH_BUG_MIN/MAX is added when the track has
been held less than LAUNCH_BUG_READY_S (25 s; the track adapter's consecutive updates). F-16C-DCSFIT: release 2.5-4 s, bug
4.5-10.5 s more, 25 s (the 7-14 s less an estimated 2.5 s release; `fire_delay.py set F-16C-DCSFIT <file>` now writes the two
parts from the record). Same 60 fights: 35% won, 1.58 launched, 0.12 dropped. Scripted opponents (ADAPTIVE and the rest)
launch at once, as the DCS AI does; self-play snapshots fly the same platform and wait as the agent does.

## dcsfit_v2 in DCS, and dcsfit_v3 (15 October)

**First DCS fight with the policy on the stick (`live_20261015_084910`)**: the plumbing worked (109 s flown, the radar track
held, no aborts, no wing rocking: heading choices 30 then 50 deg off), and the fight was lost: the policy never asked to
fire. Replayed on its recorded inputs, its probability of firing stayed below 0.004 from 13 to 80 s although the shot was
open (0.6-0.9 of R-max, the target 34-58 deg off the nose, four missiles). The DCS AI fired at 44 s from 64 km; we held 50
deg off, climbed to 12.3 km at Mach 1.25 and never broke away; the missile hit at 110 s. No input was out of training's
range (each one swapped for its value at the policy's simulator shots changed nothing): it waits, by choice, for a shot
near its no-escape range (median 0.44 of R-max, 0.96 of R-nez at its simulator shots) and cranks while closing (before
anyone fires: 0 deg 3% of the time, 15 deg 43%, 30-50 deg 47%; first shot at 122 s median, none in 15 of 60 fights).
Against ADAPTIVE's DCS-like episodes (shoot at 0.95-1.0 of R-max, press) red fires first in 94% and the policy loses 49%.
The envelope term of the shaping (our band minus red's) pays for a crank before any shot: crossing shrinks red's R-max on
us while ours on red stays. Its self-play copies hold fire too, and DCS-like red was 1 fight in 9.

**dcsfit_v3**: `train_bvr.py --adaptive-dcs-frac` (GUI: ADAPTIVE DCS-LIKE %, default 33) sets the share of ADAPTIVE episodes
flown like the DCS AI, also as self-play's scripted opponent; unset, the draws are the old ones exactly. For v3: 67%, and
the self-play scripted share 50%. Red then shoots first from about 60 km in most scripted fights: waiting costs the fight,
so the policy has to shoot first or defend. The reward is unchanged (the shaping is potential-based); if v3 still cranks
before shooting, the envelope term's threat half is the next thing to look at. Also: the F-16C-DCSFIT airframe in git now
has the number formatting of the copy dcsfit_v2 trained with (15240 against 15240.0 changed its fingerprint and a warm
start reported a changed airframe; the values were the same).

**SIM_REV 16 (15 October), before dcsfit_v3.** Checking another session's reward review against the code found a bigger
problem: running out of time was returned as `truncated`, so stable-baselines3 added gamma x V(last state) to R_TIMEOUT as
if the fight went on. The critic's value of late states fed on itself towards R_TIMEOUT / (1 - gamma): on dcsfit_v2 along
three fights that timed out, V was -0.3 to -0.8 at the start, -1.5 to -8.5 at step 300 and -7.9 to -12.0 at step 400, where
no fight is worth below about -1.4. A long fight looked worse than being shot down (-1), which fits a policy that did not
defend in DCS. Now out of time ends the episode (`terminated`, 1v1 and 2v1). And in 1v1 the end state's potential is 0, so
the shaping sums to -phi(start) whatever happens (Ng et al.); it was the end state's own potential, +0.25 to +0.38 after a
kill or a loss (mostly the track term). Kept as they are, by choice: mutual kill -0.4 above timeout -0.8, gamma 0.997, and
2v1's end-state potential (that env's own comment argues for it). dcsfit_v3 warm-starts from dcsfit_v2 with a critic that
has to re-learn its late values.

**SIM_REV 17: the missile's endgame (15 October).** Is defending easier in the simulator than in DCS? Two checks.
(1) The flight: the simulator's AIM-120C-DCS flown from the DCS fight's launch (64.7 km) at our recorded flight path tracks
the DCS missile within 14 m/s and is 1.4 km behind, closing at 540 m/s, when the DCS one hit: the same crank would have
been shot down here too. (2) The hit: with dcsfit_v2 against DCS-like ADAPTIVE (60 fights), 11 of our 12 endgame misses
were at 10.0-10.5 m and 13 of red's 23 at 10-31 m, against a 10 m fuze. Guided at 50 Hz the missile's own lag left 2-10 m
against a target flying straight (0.03-0.7 m at 500 Hz). Now the last 2.5 km of an intercept is flown in 10 sub-steps
(the target moved along its velocity within the frame) and the closest approach is taken between samples: ours 27 hits /
12 misses -> 34 / 3 (hits at about 1 m), red's misses now 11-490 m (real evasion). R-max is unchanged at 8 envelope
points, so no recalibration. Still to look at: red's missiles that die because red's radar loses us (14-24 of 102 in
those fights; none in the DCS fight, where the DCS AI kept us in its radar).

**A launch that was not ours (15 October, dcsfit_v3_selfplay, live_20261015_152411)**: the first fire request printed
`fire launched at 68.7 s: HUD None -> None MRM`, yet the episode ended with 0 shots: DCS reported no missile from BLUE-1.
The HUD count could not be read (None: the HUD did not draw `N MRM`), so the fire job went by new world objects alone, and
it took any new weapon object (level 4) for our launch: the bandit's missile or a countermeasure would do. A launch now has to
be a missile (level 4 4) of our own coalition within 5 km of the jet, and the launched line names it (`AIM_120C off the
rail`). A request that times out now says whether the HUD was in A-A and showed the AIM-120 selected, and the radar's lock
state, so the next run tells why no missile left the rail.

**Why nothing left the rail (same run, its cockpit log `stick_20261015_152411_live.jsonl`)**: the HUD never drew the A-A
master mode: `rd.aa` and `rd.msl120` were false in all 453 display reads from 0 to 109 s, while the HUD itself was read (its
target locator line `tll` came on at 35 s, the bug) and the payload showed four AIM-120C on stations 1, 2, 8 and 9 with
`CurrentStation` 0 (no weapon selected). The radar job pressed the ICP A-A button every 15 s (it does so whenever the HUD reads
not-A-A) and the HUD stayed in NAV: the ICP press does not work from this route in a live run. The radar ran in TWS and bugged
the target in NAV all the same, so only the release failed. The `AIM_120C` at 68.8 s was RED's (coalition 1, 42 km away)
and at 108.5 s RED's second. The radar job and each fire job now put the throttle's DOGFIGHT switch to MISSILE OVERRIDE
(HOTAS device 16 command 3030, value -1: the HOTAS route that target management and weapon release take; a switch position,
not a toggle) and press the ICP A-A button once only if the HUD still reads not-A-A; the console says which worked
(`radar: A-A master mode by the missile override`) or that the HUD stayed out of A-A. The switch goes back to centre at the
end of the episode. To check by hand in a mission: `python dcs\stick\stick_test.py click hotas.THROTTLE_DOG_FIGHT --value -1`
(the HUD should show `4 MRM`), `--value 0` back to centre.

## Five live fights with dcsfit_v3_selfplay, SIM_REV 18 and chaff (15 October)

**The fights** (seeded missions, `dcs_live.py --stick`, missiles leaving the rail by missile override): one mutual kill,
four shot down. In all five the DCS AI fired first, at 42.7-49.5 km (0.73-1.02 of its R-max against us, median 0.77), and
its first missile hit every time. The policy shot twice (42.7 km, the same second as red; 31.8 km with red's missile in the
air) and under fire never defended: it held a +30 to +50 deg crank, climbed, and in the last 10 s before impact turned 90 deg
or more at no decision (P(break) 0.2-4%).

**Why it did not defend.** Replayed through the policy, the DCS inputs gave the same RWR picture as the simulator (launch
warning, time to impact, seeker active, the last on earlier in DCS). In the simulator dcsfit_v3 broke with under 10 s to
impact at 3% of decisions while it held a radar track, 4% while coasting, 89% with none; with the track inputs of the DCS
frames set to "no track" P(break) rose from about 0.005 to 0.1-0.95. It had learned to break when the track dropped. The
simulated radar's azimuth limit was drawn from 51-69 deg per episode, so a 50 deg crank lost the track in many episodes
just as red's missile arrived; the F-16's radar in DCS held it at 56 deg off the nose to impact. And cranking alone was
enough in the simulator: of red's long shots (beyond 38 km) 5 of 36 hit; 10 were removed by the 3-second support rule
(red turned away and its radar lost us), 9 ran out of energy and 12 missed in the endgame by 50-460 m against a target only
cranking (red fired at 0.95-1.0 of R-max, 64.7 km median, and its missiles arrived slow). In DCS, 5 of 5 from 0.75.

**SIM_REV 18** (1v1; `bvr_library._SIM_REV_NOTES[18]`):

- No missile has the 3-second support rule (`BvrEnv.SUPPORT_RULE = False`, `AIM120.support_rule`), ours, a scripted
  red's and a self-play snapshot's alike: without its shooter's guidance a missile flies on to the extrapolated last
  estimate and turns its seeker on as usual, as a DCS missile does when its shooter loses the lock. Support still pays (a
  missile on fresh guidance aims where the target is, an unsupported one where it was going); it is no longer required to
  keep the missile. `dcs_live.py` now leaves every missile to DCS by default (`--support-rule dcs`); `--support-rule
  training` is the old rule, both sides, for checkpoints trained before SIM_REV 18. (A first version of this change kept the
  rule for our missiles and, in `dcs_live.py --support-rule training`, still removed red's DCS missiles when red's
  simulated radar lost us, the opposite of training: now one rule for both, in training and in DCS.)
- The radar's azimuth limit is drawn from 60 deg x (1 +- 0.05), 57-63 deg (`RadarSim.FOV_AZ_SPREAD`; was 0.15), ours and
  red's; range and noise randomisation unchanged.
- The DCS-like red (ADAPTIVE LONG_SHOT) fires at 0.72-0.90 of R-max, a fifth of episodes at 0.95-1.0 (was 0.95-1.0).

Measured, dcsfit_v3_selfplay against the DCS-like red, the same 60 starts:

| | SIM_REV 17 | SIM_REV 18, rule off for red only (first version) | SIM_REV 18 |
|---|---|---|---|
| red's missiles | 110 | 75 | 72 |
| hit | 27 (25%) | 22 (29%) | 19 (26%) |
| removed by the support rule | 33 | 0 | 0 |
| ran out of energy | 16 | 29 (22 after our jet turned cold, guidance long lost) | 32 |
| missed in the endgame | 19 (8 within 50 m) | 16 (10 within 50 m) | 14 (11 within 50 m) |
| our missiles: hit / ran out of energy / endgame miss | | | 41 / 31 / 5 of 82; 17 ended without our guidance, 6 of them hits (the rule would have removed all 17) |
| score / win / loss | 0.60 / 47% / 27% | 0.55 / 47% / 37% | 0.56 / 47% / 35% |
| we fired first | 3% | 17% | 17% |

Turning away from red no longer kills red's missile; only defeating the missile does (running it out of energy cold,
or beating it in the endgame), and the track no longer drops at a 50 deg crank, so the inputs left to defend on are the
missile ones (RWR warning, time to impact, seeker active), which look the same in DCS. Still easier than DCS (5 of 5);
the next model has to learn the defence there. Fuze radius left at 10 m: at SIM_REV 17 a 15 m fuze would have turned 1 of
red's 19 endgame misses into a hit and none of ours.

**Chaff (our jet, DCS only; the simulator has no countermeasures).** The radar job sets the countermeasures panel once:
MODE to MAN, PROGRAM to 1, CH (chaff) ON. While a missile is inbound with its seeker active or under 20 s from impact
(`dcs_live.chaff_wanted`, from the policy's own RWR inputs), the pilot presses CMS forward for 0.3 s every 1.5 s (each press
dispenses manual program 1), from the frame callback, so also during a radar or fire job. The console says when the panel is
set (`chaff: countermeasures panel MAN, program 1, CH on (N chaff, M flares)`), when the count goes down
(`chaff: dispensing (60 -> 59)`) or that four presses changed nothing; `results.csv` gains `chaff_presses` and
`chaff_used`; `--no-chaff` turns it off. It needs **bvr_stick_export.lua version 2**: version 1 refused the whole
countermeasures panel; version 2 lets its MODE (32 3010), PROGRAM (32 3009) and CH (32 3007) controls through (JETT and the
rest stay refused) and sends the chaff and flare counts (`LoGetSnares`, `cm` in the telemetry). Re-install it:
`python dcs\stick\install_export.py`. Whether chaff decoys red's AIM-120 in DCS, and how often, is not modelled in training.

**Going direct, and the envelope term (SIM_REV 18, 15 October).** dcsfit_v3 cranks 30-50 deg before its first shot. Forced
to fly at red until either side fired and to shoot inside a set fraction of R-max (then flown as trained), against the
DCS-like red at SIM_REV 18, the same 60 starts: as trained 0.56 (win 47%, loss 35%); hot, shooting inside 0.85 of R-max,
0.53 (win 28%: the missiles arrived slow); inside 0.70, **0.68** (win 52%, loss 15%); inside 0.55, 0.64. Going direct pays
in the simulator now; shooting at the edge does not. No reward for either was added: the shaping is potential-based
(gamma*phi' - phi, phi 0 at the end), so it cannot change which tactic wins, and a bonus would also pay where it is wrong.

The envelope term (our R-max band minus the threat's) had a slant. The threat's R-max is computed for our aspect, so it
shrinks as we turn away, but ours was the table's, launched at the target, whatever our heading, so cranking paid at once
and pointing never did. Ours is now scaled by the R-max an off-axis launch keeps (`BvrEnv._offaxis_factor`, shaping only;
the inputs and the fire mask keep the table's), measured with sweep_envelope's idealised shot, AIM-120C-DCS against a
target flying at us:

| off the lead bearing | 15 | 30 | 45 | 60 | 75 | 90 deg |
|---|---|---|---|---|---|---|
| launched at 5 km | 0.99 | 0.95 | 0.88 | 0.69 | 0.46 | 0.19 |
| launched at 9 km | 0.99 | 0.98 | 0.96 | 0.92 | 0.87 | 0.80 |

The angle is to the lead (collision) bearing, so cranking toward a crossing target's lead point costs nothing. The effect
is small: at 9 km, red at Mach 0.9 flying at us, the term's gain from a 50 deg crank falls from +0.23 to +0.19 at 70 km
(+0.21 to +0.19 at 45 km). Most of the crank's pay is real: it cuts red's R-max against us from 98 to 58 km, ours by 6%.
What decides the tactic is the fight's outcome, which SIM_REV 18 changed; check v4's heading before its first shot.

## Fight records: the same log for the simulator and DCS (20 October)

Every 1v1 fight now ends with a record (`bvr_report.py`, `info["report"]` from `BvrEnv`), written one JSON line per fight
by WATCH (`watch_runs/watch_<time>_<checkpoint>_<opponent>.jsonl`) and by `dcs_live.py` (`dcs_runs/episodes.jsonl`), so a
fight in the simulator and one in DCS read the same way:

- **red**: the bandit's style. In the simulator ADAPTIVE is `DCS-like` (shoots first from long range and presses, the
  style the DCS AI fights in), `runner`, or `defence shallow / deep_beam / deep_cold`; a self-play snapshot; in DCS
  `DCS AI`. WATCH's default ADAPTIVE mix has about a third DCS-like.
- **missiles**, ours and red's: at launch the range, the shooter's r/Rmax (red's from the truth, plus our estimate of it),
  how far off the shooter's nose the target was, both aircraft's speed and altitude; at the end HIT, KINEMATIC (out of
  energy), ENDGAME with the miss distance, or still flying; the time of flight and the longest it flew without its
  shooter's guidance.
- **defence**, each red missile at us: the RWR warning and the time to impact then, our heading choices while it flew,
  the first turn of 90 deg or more (time to impact, our radar track then), the chaff presses while it flew (DCS), how
  it ended. This is the check on the DCS failure of 15 October: no break while the track was held.
- **first**: who fired first, and by how many seconds.
- **phases**: heading choices (|offset|) before our first shot, while our missile flies, under fire, and after.
- **track_losses**: each loss of the radar track, how long, and its likely cause (outside the azimuth limit, beyond
  the radar's range, low closure, other).
- **timeout**, for a fight out of time: range, both aspects, closure, missiles left on each side, fuel, a cause (red
  running and we follow / turned away, both turned away, red cranking or beaming, both pointing, both out of missiles),
  our last 60 heading choices.

WATCH STATISTICS gains BY RED STYLE, FIRST SHOT, DEFENCE, OUR MISSILES, HEADING BY PHASE, OUT OF TIME and RADAR TRACK
LOSSES. `dcs_live.py`'s `results.csv` gains `red_style`, `first_shot`, `first_gap_s`, `red_missiles`, `red_hits`,
`broke`, `broke_with_track`, `tti_at_break_s`, `chaff_per_defence`, `timeout_cause`, `track_losses`, and how well the
stick pilot flew the commands, a second after each (`hdg_err_med_deg`, `hdg_err_p90_deg`, and `alt_as_asked_pct`: the share of decisions whose climb, descent or level was flown as asked; the altitude command is relative, the present altitude plus the action's step, so its distance from the altitude flown is about the step by design); the console
prints each red missile's defence and who fired first.

## dcsfit_v4 in DCS, and SIM_REV 19 (20 October)

**Five live fights** (seeded missions, `dcs_live.py models_bvr\dcsfit_v4.zip --stick`): five losses, but not the way
v3 lost. v4 fired first in two, defeated 4 of red's missiles (v3: none) and broke from 6 of 7 (3 with the track
held); red's next missile then hit, after breaks 13, 9 and 3 s before impact, twice after v4 had dived to the 1 km
floor and stayed low and slow. All ten fire requests were made with red 43-58 deg off the nose (v4 cranks before it
shoots): four never got a TWS bug ("no TWS track bugged in 10 s", the target at the edge of the scan) and the five
launched all ran out of energy (78-100 s of flight) against a DCS AI turning cold. The simulator says the same: of
v4's shots beyond 45 deg off the nose 4% hit (balanced; aggressive 0%), within 45 deg 71%.

**One crank side.** Before its first shot 99% of v4's off-target heading choices put red on its left (a + offset),
whether red was on its right or left and whichever way red crossed; mirrored inputs flipped the choice in 2% of
decisions. Nothing in the simulator prefers a side (red's crank and notch sides are random, starts are mirrored); the
policy cannot see which way red crosses its line of sight (trk_aspect is unsigned, 0-180 deg, and so is los_rate),
so it settled on one side. On the same 60 starts v4 scored 0.83 and its mirror image 0.76 (noise), but the better
side per start 0.97 (an upper bound): the side matters per geometry.

**SIM_REV 19** (`bvr_library._SIM_REV_NOTES[19]`):

- The policy fires only within 40 deg of the nose (`BvrEnv.MAX_OFF_BORESIGHT_DEG`, was 60; scripted red keeps 60),
  and only inside the R-max a launch along its present heading reaches (`OFFAXIS_IN_MASK`, the off-axis factor).
- Three inputs at the end of each frame (`GEO_OBS_LABELS`): `trk_aspect_side` (which way red flies across our line of
  sight, + to our right), `los_rate_signed` (+ clockwise) and `r_over_rmax_axis` (range over the off-axis R-max).
  `bvr_compat` widens older checkpoints with zero weights for them: widened dcsfit_v4's action probabilities do not
  change at all with them, so a run resumed from v4 starts as v4 flies.
- `dcs_live.py --point-before-fire DEG` (a DCS test, not trained for): a fire request first turns the nose to within
  DEG of the target (up to 10 s), holds it through the bug and the release; the console says how far and how long
  (`point: nose to the target before the release: -52 -> -18 deg in 4.5 s`).
- `dcs_live.py --fire-cone 60` flies a checkpoint trained before SIM_REV 19 under its own fire rule (60 deg, the
  table's R-max): under the 40 deg rule dcsfit_v4, which cranks 30-50 deg before shooting, fired 2 shots in 60
  simulated fights against the DCS-like red (14 with its actions sampled, half of them hits). To test pointed shots
  with v4 in DCS: `--fire-cone 60 --point-before-fire 20`.

## dcsfit_v5 in DCS, and SIM_REV 20 (20 October)

**dcsfit_v5** (3M steps from v4, SIM_REV 19) against the DCS-like red in the simulator, the same 60 starts: balanced
0.71, aggressive 0.85 (v4 under its own 60 deg rule 0.83); 1% of its shots beyond 40 deg off the nose (v4 44%), median
21-22 deg; red's missiles hit it 14-19% (v4 25%), and under fire it turns away at 52-59% of decisions (v4 27%). It
still cranked to the + side in 99% of its choices, whatever the geometry (mirrored inputs flipped 3%): the new
crossing inputs went unused in 3M steps.

**In DCS (live_20261020_165232)** red's first missile was defeated (v5 broke 29 s before impact, the track held), but
DCS kept it flying until 301 s (249 s of flight), and the RWR built from DCS's missile list warned for 146 s: v5 ran
from it, climbing to 12.9 km, until red fired again (250 s, 313 s) and hit. In the simulator a defeated missile also
kept the warning on, until it was gone: a median 18 s after its closest approach (up to 94 s). All three of v5's fire
requests went unbugged ("no TWS track bugged in 10 s"): one at 83 km, 9 deg off the nose; two at 69 and 54 km, 32-36
deg off, but the policy kept cranking (+40, +50) through the 10 s the bug may take and the target drifted to 45-47 deg.

**SIM_REV 20** (`bvr_library._SIM_REV_NOTES[20]`):

- A missile is a threat only while it closes on its target (closing faster than 20 m/s), and 3 s after
  (`missile_sim.is_threat`, `THREAT_CLOSING_MPS`, `THREAT_HOLD_S`): the RWR warning, the missile list's `threat` flag,
  the threat missile in the observation and the defence term, scripted red's defence and the fight records'
  "under fire" go by it, in `sim_world.py` and `dcs_world.py` alike. A missile that has overshot or run out of
  energy drops off in 3 s, as a real RWR stops warning once the seeker loses the aircraft.
- The stick pilot holds the nose where it was at the request through the bug and the release (unless
  `--point-before-fire` points it first), so the target stays in the TWS scan while the radar bugs it.
- `train_bvr.py --symmetric-policy` (GUI: SYMMETRIC POLICY, on by default): each choice is the average of the
  policy's logits for the fight as seen and, mirrored back, for its mirror image (`bvr_policy.AsymmetricMaskablePolicy`,
  `MIRROR_LABELS`). In a mirror-symmetric world a policy left to itself settles on one side; averaged, a choice not
  tied to the geometry comes out even, so the side has to come from the inputs that tell left from right. Exactly
  mirror-equivariant (a test checks it, after training too); stored in the checkpoint's policy_kwargs, so a resumed
  checkpoint is switched over and every checkpoint the run saves, self-play snapshots included, keeps it. Off for
  older checkpoints, which fly as before.

## dcsfit_v6 in DCS, and SIM_REV 21 (21 October)

**dcsfit_v6** (3M steps, SIM_REV 20, symmetric policy) in five live fights: one kill, four losses. Two faults, both
in code the simulator and DCS share:

- **It did not come back after running (live_20261021_093711).** Red started 171 deg off the nose, outside the
  radar's 60 deg, so v6 never had a track. Without a track the heading choices are offsets from a held reference, and
  that reference was set in `reset()` from a state not received yet: 0, north. v6 flew 330-345 deg (offsets -30/-15
  from north) for 240 s with red 100-180 deg off the nose, until red shot it from 7 km. In the simulator against the
  DCS-like red, the same: 23 of v6's 60 fights never found the bandit (all timeouts), 55% of its decisions had no
  track, and the reference was a median 164 deg off red's true bearing.
- **It rocked its wings with a missile behind (093321 from 136 s, 090924 from 186 s).** With red behind and no track,
  v6 alternated -15/+15 (-30/+30) every second around the held bearing to red, which was behind it: commands 161 and
  191 deg while flying 0 deg. Both the stick pilot and the simulator's autopilot took the shorter way round, which
  flips at 180 deg, so the bank reversed every second (+22/-22 deg) for 70 s and the jet never turned. The policy
  wanted to come back; the autopilot could not. The simulator does the same (3 deg turned in 40 s, 39 reversals),
  but v6 rarely alternated there.

**SIM_REV 21** (`bvr_library._SIM_REV_NOTES[21]`):

- The fight starts with a picture call (`BvrEnv.GCI_START`, `_start_ref`): until the radar has a track, the
  heading choices are offsets from the bandit's bearing at the start, then from the last track's bearing, held, as
  before. Self-play's red gets the same. In DCS the bearing comes from the start positions (`initial_ic`).
- Turn hold (`f16_sim.turn_hold`, used by the simulator's autopilot and the stick pilot's `Hold`): a heading error
  beyond 150 deg against the bank the aircraft is already in (over 10 deg) is flown the long way, so it keeps
  turning. Alternating 161/191 deg off the tail now turns round in 12 s with no reversal.

v6 itself, unchanged, on the SIM_REV 21 simulator against the DCS-like red (the same 60 starts, balanced): 67% kills,
4 timeouts (was 33% and 33); decisions without a track 18% (was 55%), the reference a median 4 deg from red's true
bearing. v5 on SIM_REV 20 for comparison: 40% kills, 7 timeouts. Both fixes apply to any checkpoint in DCS: no
retraining is needed to fly v6 with them, though training on SIM_REV 21 lets the policy learn from starts it used
to waste.

**The fire timeouts (stick logs of 090242 and 090924).** Two different causes, neither the nose hold:

- **090924: the HUD was in NAV for the first 125 s.** The display markers show no A-A symbology and no AIM-120
  selected from the start until 125.4 s, so the radar never left RWS for TWS and all four fire requests (81, 66,
  52, 42 km, red nose-on 1-17 deg off our nose) timed out as "no TWS track bugged in 10 s". The pilot's missile
  override click did nothing and its one ICP A-A press did not help either; the old pilot on a fake cockpit with
  the switch already aft and the HUD in NAV fails the same way, with the same message. The pilot now cycles the
  switch through centre to override when the HUD stays in NAV, presses ICP A-A whenever the HUD has just said NAV,
  tries again at every radar or fire job, and a timeout in NAV says so ("the HUD was not in A-A (NAV)").
- **090242: the requests came before DCS's radar had a track to bug.** The HUD was in A-A from 1.2 s. The request
  at 84 km (red 9-10 deg off, nose-on) timed out; the one at 74 km bugged at 67 km, after 10 s of target
  management, and launched. The simulator's APG-68 holds a track on an F-16 from 77-104 km (90 km +-15%), so
  the policy's inputs said TRACK where DCS had nothing yet. One fight is not enough to recalibrate on.

dcs_live.py now keeps what the console said in `<episode>_console.txt` (the pilot's A-A, TWS, bug and fire events
were only on screen), and results.csv records `hud_aa_pct` (the share of the fight with the HUD in A-A) and
`first_bug_km` / `first_bug_t` (the range and time of DCS's first bugged track): over a few fights, the second
says how far DCS's radar really tracks an F-16, for `library/builtin/radar/APG-68.json`.

**live_20261021_142940 (dcsfit_v6 with SIM_REV 21 and the A-A recovery).** HUD in A-A the whole fight; wing rocking
gone (4 bank reversals in 234 s, was 40 a minute); the track was lost cold and regained twice. Two DCS-side findings:

- **Unlimited Weapons was on.** BLUE-1's AIM-120 count went 4 -> 2 and back to 4 three times, and the policy fired
  eight missiles from four. The policy's count is now loaded minus fired, as in training (`DcsReplayWorld._wpn`), the
  console warns once per aircraft, and results.csv counts `ammo_refills`. Turn the option off for comparable fights.
- **The pilot flew the wreck.** After the hit (234.4 s) the episode waited 17 s for our last missile, a mutual kill
  still being possible, and the pilot released the axes at every frame (-7 g). It now stops when BLUE-1 is down.

The first bugged track came at 46.9 km, though the request at 71.5 km had red nose-on, 11 deg off and co-altitude;
the same mission bugged at 67 km in 090242 and launched at 71.6 km in 140225. So the bugging, not DCS's detection
range, is what varies; the fight's stick log (`stick_*_live.jsonl`) shows the radar page markers through it.

**Why the first bug came at 46.9 km (stick log of 142940).** The radar page's scan-limit marks (drawn in TWS's narrow
scan) were there from 28.2 to 44.7 s, then gone until 73.5 s, when the target bugged at once: the radar was in RWS
for 29 s with red nose-on, 12-18 deg off the nose, 65 to 47 km. Target management right held toggles RWS <-> TWS,
and the pilot held it whenever its read of the page did not say TWS, an empty read (a lost reply: mode None)
included; on the fake cockpit, one empty read takes the old pilot from TWS to RWS. It now holds it only when the page
says RWS, reads an empty page once more, and otherwise leaves the radar as it is ("the radar page did not answer");
the console names each switch ("radar RWS -> TWS"). The fake cockpit now tells a long press (toggle) from a short one
(bug), as DCS does.

**Bugging is now what keeps v6 from shooting (22 October, three fights, all lost).** First bugs at 71.6, 49.6 and 35.5
km; 5 of 9 fire requests timed out; two shots in three fights. The scan-limit marks are not a TWS sign (the paragraph
above reads them as one): in 095954 the radar went to TWS before 10 s and the marks came only at 22.3 s, after the cursor
was moved. Twice the target became buggable only as it drifted 5-7 deg further out than the cursor was put (095954: put
at +32, bugged at +37-39; 142940: put near +12, bugged at +18), which fits a cursor left short of where it was sent
(a timed move back from the display's edge: 0.28 and 0.48 s), but 090654 bugged at once 7 deg off. The cursor's
position is not read back, so the pilot now logs each move (`radar: cursor to +32 deg (edge, then 0.28 s back)`, and
`ev: cursor` in the stick log) and, at each bug timeout, the radar page's mode, azimuth, bars and range scale in the
reason, and the whole page in the stick log (`ev: fcr_page`).

**The radar showed nothing (100922).** The bug timeout's page: TWS A6 4B, 40 NM scale, the target +28 deg off the
nose (the cursor at +22: inside the 60 deg scan), 38 km, the height coverage at the cursor's range 16-39 thousand feet
(the target at 27.6), and both `AA_RadarTracksKeeper` (contacts) and `System_AirTracksKeeper` (TWS tracks) empty. So
there was nothing to bug: the late first bugs (35-72 km) are DCS's radar not seeing the target yet, not the cursor. The
pilot now counts both lists at every bug attempt (`ev: bug_try` in the stick log, and in the timeout reason), which
dates the first contact in each fight. To check in the mission editor: does RED-1 carry an ECM pod, with ECM allowed?

**102153: DCS's radar needs 15-20 s with the target in the scan; our chaff was taken for the target.** From the same
start as 095954 (red behind, 52.5 km): red entered the scan about 22 s (TWS A6 4B), nose-on (aspect 3-25 deg, closure
510 m/s: no notch), 1-2.5 deg above us; the policy asked to fire at 26 s (45 km, its simulated radar's track); every
bug attempt to 41.8 s read 0 contacts; the bug came at 42.3 s (37 km) and the missile launched at 44.2 s. The earlier
`stage_twslock` measurement (about 14 s after entering TWS with A6 4B) agrees: DCS builds a TWS track much later than
the simulator's radar, so requests made as soon as the policy has a track time out. The contact counter is not yet
proven (it read 0 even 0.5 s before a bug): the page right after a bug is now noted too (`ev: bug_ok`).

From 102.5 s our own chaff and flares (world objects level1 1, level2 3, no name, no coalition, 0-0.6 km away) were
taken for the nearest aircraft: the cursor and the target bearing went to our tail (-177 deg) while red was at -17.
`cockpit.nearest` now skips unnamed objects and any but aircraft (level2 1) and helicopters (2).

`dcs_live.py --tws-scan A3,4B` (a test, off by default): the pilot sets a narrower TWS scan once TWS is up, stepping
the radar page's OSB 18 (azimuth) and 17 (bars) and reading the page back after each press; a press that changes
nothing ends it ("OSB 18 did not change the scan's az"). A3 4B scans half the azimuth of A6 4B, so each frame takes
about half as long, with the same height coverage.

**103934 (--tws-scan A3,4B): a launch 4 s after red's; red hidden by our chaff; red's chaff.** The scan change did not
take ("OSB 18 did not change the scan's az (6)"), so the fight ran on A6 4B. Even so the pre-bug got red at 26.1 s
(45.5 km, +44 deg) and the missile left at 27.9 s, 4 s after red's: the track time varies. From 107 s red was absent
from the pilot's world objects: the export sends the 12 nearest, and all 12 were our own chaff and flares, so the
fire job had no target ("target ? deg"). bvr_stick_export.lua version 3 leaves unnamed objects out (reinstall it:
`python dcs\stick\install_export.py`); the pilot says once if the list holds only chaff. Red dispenses chaff too:
unnamed objects within 3 km of red, far from us, in 142940 (from 129 s) and 103934 (from 72.7 s), both while our
missiles flew, so our chaff stays on by default (`--no-chaff` turns it off). test_chaff waited for the setup flag,
set before the clicks: it now waits for the panel itself.

**111114 (episode 2): the new pilot read the old mission first.** The stick log opens with the previous mission's last
frames (t 159 s, our wreck at -17 to +14 g), still queued on the stick link when the next episode's pilot started; it
read them one by one, released the stick at each and slept 0.5 s, so it flew nothing for the new fight's first 26 s
(the console's 50 "load factor" lines). `UdpLink.flush()` now drops the telemetry already waiting (events kept), and
the pilot calls it as it starts. The screenshot taken during the fight agrees with the counters: TWS, 40 NM, the cursor
right of centre with its height readout, and no contact symbol on the radar page.

**SIM_REV 22 (22 October): the simulator fitted to what DCS did.** From dcs/missile_replay.py and the stick logs:
an active seeker inside 10 km is decoyed by a target that beams it (0.30 per second, 0.01 otherwise; both sides,
chaff assumed automatic as the pilot's), a pressing red runs a fleeing target down at its dash speed, and the platform
F-16C-DCSFIT-R carries APG-68-DCS (55 km detection, 15 s to build a new track). The stick pilot drops chaff every
0.5 s against an active seeker (was 1.5). Train the agent on F-16C-DCSFIT-R against F-16C-DCSFIT. dcsfit_v6 on it
(24 fights against the DCS-like red, untrained for it): 1.4 shots a fight (3.2 before), its far shots decoyed or
short, red's hitting 20%.

**The human red (SIM_REV 22).** A human in a DCS F-16 runs the same player radar as our jet, the DCS AI does not (it
tracks at once, at full range). For fights against a human, red flies F-16C-DCSFIT-H: radar APG-68-DCS-HUMAN, the same
55 km, but 4 s to a track (a narrowed scan or a single-target lock; an estimate, not measured). Platforms differing only
in radar now count as the same aircraft, so self-play runs on F-16C-DCSFIT-R v F-16C-DCSFIT-H, its snapshots flying red
with the human's radar.

**SIM_REV 23 (23 October): dcsfit_v7 on SIM_REV 22 kept away.** Resumed from dcsfit_v6 (16M) and left on the ADAPTIVE
stage against a human-radar red, it timed out in every fight after about 10,000 episodes (0% wins, a track 0% of the
time, 0.02 shots a fight): a timeout's -0.8 comes 400 s in (-0.24 seen from the start) where a loss at 100 s is -0.74,
and against SIM_REV 22's scripted red (beaming our missiles away at 0.30/s, chasing runners) dcsfit_v6 won one fight
in 24. Every decision now costs 0.002 (keeping away to a timeout -0.71, lost at 100 s -0.91, won at 100 s +0.57), and
the beaming decoy rate is 0.15/s. Train on self-play, both sides F-16C-DCSFIT-R.
The first SIM_REV 23 run (self-play, both sides F-16C-DCSFIT-R) found the next way out: at 4M steps a track 0.4% of the
time, 0.04 shots a fight, and win, loss and timeout together about 40% of fights; the rest ESCAPE (range beyond 150 km),
worth -0.6 and reached in about two minutes when both copies turn away (-0.62 discounted, against -0.71 for a
timeout). An escape is now a loss (-1.0), and the dashboard's outcome chart draws escapes.

**The radar procedure (23 October), to try in DCS: `--tws-scan`, `--ant-elev`, `--stt`.** APG-68-DCS's 15 s and 55 km
are what this pilot got out of DCS's radar, not what the radar can do: it searched in DCS's widest TWS scan, A6 4B,
whose frame takes about 8 s (two frames before a track: the 15-20 s measured), and it never moved the antenna
elevation. Three options, off by default, try what a skilled pilot does:

- `--tws-scan A3,2B`: a narrower scan, about 2 s a frame. The scan buttons (OSB 18 azimuth, 17 bars) are now
  pressed by a mouse click on the left MFD, then by SetCommand, then on the right MFD, until a press changes the
  page (in 103934 the first did nothing); the console says which worked (`TWS scan A3 2B (OSBs on the left MFD)`)
  and `results.csv` has the scan in `scan`.
- `--ant-elev`: before the bug the cursor goes to the target's range and the antenna (ANT ELEV up/down) until the
  target's altitude is inside the height coverage the page gives at the cursor (`Max/Min_Search_Altitude`), 20% in
  from its edges: a pilot does this with the bandit's altitude from AWACS, the pilot here from the world objects. The
  cursor's range comes from its bearing-range readout, once a fight referred to our own position by holding the cursor
  to the bottom edge (a bullseye 100+ NM away makes it coarse: about 2 NM). A 2-bar scan covers half the height of a
  4-bar one, so `A3,2B` needs this. `results.csv`'s `ant_elev` is "radar jobs with the target inside / radar jobs";
  the stick log has `ev: elev` for each (the coverage and the rates learnt).
- `--stt`: a fire job asks for a single-target track (target management up) between the bug and the release, so
  the missile is supported by a lock that follows the target, not a TWS track file (dcsfit_v6's lost theirs 6-33 s
  after launch). On 13 October target management up after a TWS bug stayed designated; with the cursor on the target
  (`--ant-elev` puts it there) it may not. The console says `single-target track` or `firing on the TWS track`;
  `results.csv` counts `stt_shots`. The target's RWR sees the lock, a DCS AI or a human reacts to it.

On the fake cockpit (a track after two scan frames with the target inside the scan, 2 km below us): A6 4B bugged it
16-18 s after it could be seen, A3 2B alone never, A3 2B with the antenna aimed 5-6 s. To measure in DCS:

```
python dcs_live.py models_bvr\dcsfit_v7.zip --stick --episodes 5 --tws-scan A3,2B --ant-elev --stt
```

and compare `first_bug_km` / `first_bug_t` in `results.csv` with the earlier runs (first bugs 35-72 km). If the track
then takes about 5 s and comes at 60 km or more, APG-68-DCS is set from these numbers (SIM_REV 24).

**174626 (23 October, dcsfit_v6 at 3M, `--tws-scan A3,2B --ant-elev --stt`): the scan and the antenna work, STT does
not.** The scan went to A3 2B by SetCommand on the left MFD (the mouse click route is what did nothing in 103934).
The antenna aiming found the readout's origin (a bullseye 196 NM away) and had red, 0.5 km above, inside the coverage
(16-47 kft) at 76 km, 28 s in; red was inside the scan's azimuth from 12 s (86 km). Target management right every
0.5 s from 29 s bugged nothing until 47.5 s, at 63.2 km: so with the target in a 2 s scan, the bug waits on DCS's
detection range (about 63-65 km here, head-on), not on building the track. A later re-bug (28 km) took 0.5 s.
Four launches, no fire request timed out (launch delays 17.7, 13.0 and 4.9 s); all four missed, and red's third
missile hit. Target management up after the bug left the radar designated at all four launches, the cursor on the
target: `--stt` gets no single-target track from a TWS bug in DCS. The contact and track counters read 0 at both
bugs: they are not where DCS lists tracks. The antenna's third job ran out of steps chasing the cursor's range with
the bullseye's 2 NM readout; the range is now good within 2.5 NM.

**175236 (episode 2 of the same run): first bug at 52.8 km.** Red 0.8 km below, inside the scan's azimuth and the
coverage (11-41 kft, red at 24) from 19 s at 63 km; target management right every 0.5 s bugged it at 36.7 s, 52.8
km. Again the detection range sets it, here 10 km shorter than 174626 with red slightly below. A fire job at 140 s
spent 7 s aiming the antenna while red drifted from -27 to -55 deg and out of the scan (bug timeout at -90): a fire
job now leaves the antenna alone if a radar job aimed it in the last 20 s. At 166 s the page read A6 2B: DCS had
put the azimuth back. Each radar job now checks the scan and sets it again (`scan` in results.csv says how often).
