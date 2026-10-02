# Flying a trained 1v1 policy in DCS World

This folder connects a BVR_RL policy to DCS World. DCS flies the aircraft
and the missiles; the policy decides, once a second, where to point the
aircraft and when to shoot.

```
 DCS World (Windows)                                 your Python environment
┌──────────────────────────────┐   UDP 15301   ┌────────────────────────────────┐
│ mission bvr_rl_1v1.miz       │ ────────────▶ │ dcs_live.py                    │
│   BLUE-1  (DCS AI airframe)  │  state 10 Hz  │   radar + track model (ours)   │
│   RED-1   (DCS AI opponent)  │               │   observation, as in training  │
│   bvr_bridge.lua             │ ◀──────────── │   policy (your .zip)           │
│     steers BLUE-1, fires     │   UDP 15302   │   records everything           │
└──────────────────────────────┘  command 1 Hz └────────────────────────────────┘
```

| File | What it is |
|---|---|
| `bvr_bridge.lua` | Mission script. Streams the fight to Python, flies BLUE-1 on its commands |
| `make_mission.py` | Builds the mission (`bvr_rl_1v1.miz`) with the bridge inside |
| `setup_dcs.py` | One-time patch that lets mission scripts open a network socket |
| `../dcs_live.py` | Runs the policy against DCS (or only watches, `--shadow`) |
| `fake_dcs.py` | Pretends to be DCS using BVR_RL's simulator, to test without DCS |
| `test_bridge.py` | Runs the bridge in Lua 5.1 against a mock of DCS's API |
| `bvr_logger.lua` | Older step: records a mission to a file, no Python needed (see the end) |

Everything below assumes Windows and no DCS on the computer yet. The same
steps, with more explanation, are in the PDF guide
`docs/BVR_RL_DCS_Guide.pdf`.

---

## Step 1. Install DCS World (free)

1. Make an account at **digitalcombatsimulator.com** (top right, *Sign up*).
2. On the same site open *Downloads → DCS World* and download the installer
   (the "standalone" version). DCS is also on Steam as *DCS World Steam
   Edition*; either works with this project, but the steps below use the
   standalone paths.
3. Run the installer. Keep the default folder,
   `C:\Program Files\Eagle Dynamics\DCS World`, unless that disk is short
   of space. The download is large (well over 100 GB); put it on an SSD.
   Check the current system requirements on the DCS website first; in
   practice 16 GB of RAM is the minimum and 32 GB is better.
4. Start DCS from the desktop icon and log in with your account. The first
   start takes a while (it prepares shaders).
5. Try it once: *Instant Action* → any free aircraft → *Fly*. Esc → *Quit*
   to come back.

**What is free:** the game, the Caucasus map, and two flyable aircraft
(Su-25T, TF-51D). What matters here: **every aircraft type can be placed as
an AI unit without buying it**, including the F-16C. The policy's aircraft
and its opponent are both AI units, so nothing has to be bought. You only
need a module (e.g. the F-16C) if you want to fly the red aircraft yourself.

## Step 2. Get this branch and check the Python side, without DCS

Use the same Python environment you train with.

```
cd BVR_RL
git fetch
git checkout feature/dcs-1v1
pip install pydcs
```

Now run the whole chain against the fake DCS, which uses our own simulator.
Open two terminals in the BVR_RL folder:

```
terminal 1:   python dcs_live.py models_bvr\latest.zip
terminal 2:   python dcs\fake_dcs.py --speed 4
```

Use any **1v1** checkpoint you have (2v1 checkpoints are refused). Terminal 1
prints the episode as the policy flies it, in lines like these (the numbers
are only an example; yours depend on the model):

```
episode 1: BLUE-1 v RED-1, start range 91.6 km
  t=   10s  range  87.3 km  track TRACK  heading  +30  alt    +0  speed 340
  t=   41s  range  38.0 km  track TRACK  heading  +30  alt    +0  speed 340  FIRE
episode 1: KILL after 88 s, 1 shots (0 fire requests not answered by a launch)
```

If Windows asks whether Python may use the network, allow it (*private
networks* is enough; everything stays on this computer). If this step
works, the Python side is fine and anything that fails later is on the DCS
side.

## Step 3. Let DCS missions talk to Python (once, and after each DCS update)

DCS mission scripts can't open network connections. `setup_dcs.py` adds a
few lines to DCS's `Scripts\MissionScripting.lua` that give scripts one
UDP/TCP socket library (LuaSocket, which ships with DCS), and nothing else.

1. Close DCS.
2. Open a terminal **as administrator**: Start menu, type `cmd`, right
   click *Command Prompt* → *Run as administrator*. (DCS sits in
   `Program Files`, which needs it.) Activate your Python environment in it
   if you use one, and `cd` to the BVR_RL folder.
3. Run:

```
python dcs\setup_dcs.py
```

It finds DCS, makes a backup (`MissionScripting.lua.bvr_rl_backup`) and
prints:

```
DCS World: C:\Program Files\Eagle Dynamics\DCS World
LuaSocket: found
MissionScripting.lua: not patched
backup: ...\Scripts\MissionScripting.lua.bvr_rl_backup
patched. Restart DCS if it is running. Run this again after every DCS update.
```

If DCS is elsewhere: `python dcs\setup_dcs.py --dcs "D:\Games\DCS World"`.
`--check` only reports, `--undo` removes the lines again.

Be aware:
- With the patch, **any** mission you open can make network connections.
  Run `--undo` when you stop using BVR_RL, and don't join multiplayer
  servers with the patch in place (some reject a modified file).
- Every DCS update puts the original file back: run `setup_dcs.py` again.

## Step 4. Make the mission

```
python dcs\make_mission.py
```

writes `dcs\bvr_rl_1v1.miz`: two F-16Cs, 90 km apart, head-on over the
Black Sea, each with four AIM-120C (as the training platform), and a
trigger that starts the bridge. Copy it to

```
C:\Users\<you>\Saved Games\DCS\Missions\
```

Options: `--range-km 70`, `--bearing 45` (direction blue → red),
`--alt-blue 8000`, `--alt-red 10000`, `--speed 250`, `--red-type Su-27`.
Keep `--blue-type F-16C` for a policy trained as the F-16C.

`pydcs` prints *Couldn't detect any installed DCS World version* if it can't
find DCS; the mission is still written.

<details>
<summary>Building it by hand in the Mission Editor instead</summary>

1. DCS main menu → *Mission Editor* → *New*, map **Caucasus**.
2. Place a blue airplane over the sea: *Add airplane group*, country CJTF
   Blue (or USA), type **F-16C bl.50**, task CAP, skill Excellent, altitude
   9000 m, speed 1000 km/h. Unit name **BLUE-1**. Payload: four AIM-120C.
   Give it a second waypoint towards the red aircraft.
3. The same for red (CJTF Red / Russia), unit name **RED-1**, 70–110 km away,
   flying towards blue.
4. *Triggers* (left toolbar, "SET RULES FOR TRIGGER"): *New*, type
   **4 MISSION START**; under ACTIONS *New* → **DO SCRIPT FILE** → open
   `dcs\bvr_bridge.lua`.
5. Save into `Saved Games\DCS\Missions\`.

The bridge flies the first blue airplane and fights the first red one; for
other names set `CFG.agent` / `CFG.red` at the top of `bvr_bridge.lua`
(and re-select the file in the trigger, since the mission keeps a copy).
</details>

## Step 5. First run: shadow mode (the policy only watches)

In shadow mode DCS's own AI flies BLUE-1 and fights; the policy sees
everything and decides what it would do, but nothing is sent to DCS. It
checks the link and shows how the policy reads a DCS fight.

1. Terminal (normal, not administrator):

   ```
   python dcs_live.py models_bvr\latest.zip --shadow
   ```

   It waits: `waiting for DCS (start the mission that runs bvr_bridge.lua) ...`
2. DCS: *Mission* → open `bvr_rl_1v1` → *Fly*. Top right of the screen:
   `bvr_bridge: BLUE-1 v RED-1, sending to 127.0.0.1:15301, listening on 15302`.
   The mission has no player aircraft, so DCS opens the map (F10). **F2**
   steps through external views of the aircraft (F2 again for the next one),
   **F10** is the map. If the mission starts paused, press **Pause**.
3. The terminal prints the episode every 10 seconds, with the heading,
   altitude and speed choices the policy would make and `FIRE` where it
   would shoot.
4. Afterwards compare the inputs the policy saw with the simulator's:

   ```
   python dcs_obs_check.py dcs_runs\live_<date>_ep1_shadow.jsonl --platform F-16C --opp-platform F-16C
   ```

   (see *Checking the inputs* below).

## Step 6. Let the policy fly

1. Terminal:

   ```
   python dcs_live.py models_bvr\latest.zip
   ```
2. Start the mission in DCS as in step 5.

As soon as the first command arrives the bridge takes BLUE-1 over: the DCS
AI's own evasion is switched off and its weapons are held, so the policy
decides when to turn, defend and shoot. Each second the policy's heading,
altitude and speed become a route for the AI to fly. On `FIRE` the bridge
lets the AI shoot one AIM-120 at RED-1, then takes control back. RED-1 is
ordered to attack BLUE-1 and fights as the DCS AI does.

The terminal shows each shot and how the episode ends (example numbers):

```
  t=   52s  range  41.2 km  track TRACK  heading  +30  alt    +0  speed 340  FIRE
episode 1: KILL after 97 s, 1 shots (0 fire requests not answered by a launch)
  written: dcs_runs\live_20261002_181500_ep1.jsonl, _steps.csv
```

When the episode ends (a kill, a loss or the step limit) BLUE-1 is given
back to the DCS AI, which carries on with its own mission.

**More episodes:** `--episodes 5`. After each one, restart the mission (Esc
→ *Quit* → in the debriefing *FLY AGAIN*), and the next episode starts
by itself.

**Faster:** DCS time acceleration works: **Ctrl+Z** faster, **Alt+Z** back
to normal. Python handles about 8 mission seconds per real second on an
ordinary computer; if it falls behind it prints
`this computer is ... s of mission time behind DCS`. Slow down then.

Other options: `--doctrine CONSERVATIVE`, `--max-steps 400` (if you trained
with 400), `--opp-platform` (the library platform whose envelopes the policy
should assume for red, if red is not an F-16C), `--out` for the output
folder.

## Step 7. Look at the results

In `dcs_runs\`:

| File | Contents |
|---|---|
| `results.csv` | One row per episode: outcome, flight time, start range, shots, fire requests that never became a launch |
| `live_..._ep1.jsonl` | Everything DCS sent, in `bvr_logger.lua`'s format |
| `live_..._ep1_steps.csv` | Each second: the action, the command sent, the track state, true range and all policy inputs |

### Checking the inputs

```
python dcs_obs_check.py dcs_runs\live_<date>_ep1.jsonl --platform F-16C --opp-platform F-16C --max-steps 300
```

It replays the recording through the observation code and compares every
policy input with simulator episodes. `OUTSIDE` means DCS pushed that input
where the policy hardly saw it in training; `CLIP` means DCS pinned it at
its limit more often than the simulator did. Each flag is either a bridge
mistake (a unit, sign or axis error, to be fixed) or a real difference
between DCS and the simulator (speeds, turn rates, missile ranges) to
decide on. Several recordings can be given at once.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| No `bvr_bridge: ...` message when the mission starts | The trigger is missing or the script failed. Open `Saved Games\DCS\Logs\dcs.log` and search for `bvr_bridge` |
| `bvr_bridge: LuaSocket is not available` | Run `setup_dcs.py` (as administrator), restart DCS. After a DCS update, run it again |
| `bvr_bridge: cannot listen on port 15302` | Another program uses the port. Change `CFG.port_in` in `bvr_bridge.lua` and use `--port-out` with the same number |
| Python keeps printing `waiting for DCS` | DCS isn't sending: check the message above, and that the mission isn't paused. A firewall blocking Python on *private* networks can also do it |
| `fire timeout` in the terminal | The DCS AI didn't launch within 8 s: its radar may not have the target, or the target is outside the AI's own launch range. If it never fires, set `fire_weapon = nil` in `bvr_bridge.lua` (lets the AI pick the missile) and rebuild the mission |
| BLUE-1 turns and climbs more slowly than in the simulator | The DCS AI flies the route its own way. This is part of the difference between the two worlds; see below |
| `... behind DCS` | Lower time acceleration (Alt+Z) |
| `this checkpoint was trained for 2v1` | The DCS link is 1v1 only for now |

Errors inside the bridge are written to `dcs.log` as `bvr_bridge ... error`.

## How DCS differs from training

The policy was trained in BVR_RL's simulator. In DCS, expect it to do
worse at first, for reasons that are useful to know:

- **Flying.** BLUE-1 is flown by the DCS AI along the commanded route,
  with DCS's F-16 flight model. Turn rates, climbs and speed changes are
  not the training autopilot's.
- **Shooting.** "Fire" is a request. The DCS AI launches when its own
  logic agrees (it is set to shoot at maximum range), usually within a
  second or two, and the policy is not allowed to request another shot
  until the launch is seen. Unanswered requests are counted.
- **Missile guidance after the launch.** The policy keeps flying after a
  shot. Whether DCS keeps the missile guided while the AI isn't in an
  attack task is one of the first things to check in the recordings
  (hits and misses in `results.csv` and the `.jsonl`).
- **Radar.** The track the policy sees comes from BVR_RL's own radar model
  running on DCS's true positions, exactly as in training. DCS's radar is
  not used.
- **The opponent** is the DCS AI, which none of the training opponents
  imitate exactly.
- **Missiles.** DCS's AIM-120 has different ranges from the envelope
  tables the policy was trained with.

`dcs_obs_check.py` shows the first two kinds of difference in numbers.

## Testing without DCS

```
python test_sim.py                # includes "DCS live link": dcs_live.py v fake_dcs.py over UDP
pip install lupa
python dcs\test_bridge.py         # bvr_bridge.lua and the setup patch, in Lua 5.1
```

`test_bridge.py` runs the bridge against `mock/dcs_api_mock.lua`, a small
stand-in for the DCS scripting API. It checks the bridge's logic; it cannot
show where the real API behaves differently. The bridge has not yet been run
inside DCS itself.

## The protocol

DCS → Python, UDP 15301, one JSON object per datagram, the same lines
`bvr_logger.lua` writes (see its header), plus:

```
{"format":"bvr_rl.dcs.v1", ..., "bridge":1, "agent":"BLUE-1", "red":"RED-1"}   every 2 s
{"ev":"fire", "t":..., "status":"requested|launched|timeout|refused", "seq":...}
{"ev":"bridge", "t":..., "status":"control|released", "agent":...}
{"ev":"mission_end", "t":...}
```

Python → DCS, UDP 15302, plain text:

```
CMD <seq> <heading deg, map north, clockwise> <altitude m> <speed m/s> <fire 0|1>
STOP
```

Commands are re-sent every second; the bridge re-issues the AI's route only
when the command changes (2° of heading, 50 m, 1 m/s) or every 10 s.

## Recording only: bvr_logger.lua

Before the live link, `bvr_logger.lua` recorded a mission to
`Saved Games\DCS\Logs\bvr_rl_<n>.jsonl` with no Python running (load it
with a MISSION START → DO SCRIPT FILE trigger). It needs `io` and `lfs`,
which `setup_dcs.py` does **not** restore: to use it, comment out
`sanitizeModule('io')` and `sanitizeModule('lfs')` in
`Scripts\MissionScripting.lua` by hand, and put them back afterwards (with
them removed, any mission can read and write your files). The live link
records the same lines in `dcs_runs\` without that.
