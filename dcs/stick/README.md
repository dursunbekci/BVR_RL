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
