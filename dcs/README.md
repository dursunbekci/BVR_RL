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
`--alt-blue 8000`, `--alt-red 10000`, `--speed 250`, `--speed-red 250`, `--red-type Su-27`,
`--defender` (you in the free Su-25T, see below).
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

## Fly against the policy yourself (free Su-25T)

You fly RED-1 in the free **Su-25T** and defend: it has no air-to-air radar
and only two short-range R-73s, so you can't shoot back beyond visual
range. The question: can the policy shoot down a human who evades?

```
python dcs\make_mission.py --defender          # -> dcs\bvr_rl_defender.miz, copy to Saved Games\DCS\Missions\
python dcs_live.py models_bvr\latest_selfplay_v6.zip --episodes 5
```

Then *Mission* → `bvr_rl_defender` → *Fly*: you start in the cockpit, in
the air, at 6000 m and 200 m/s, 90 km from BLUE-1. Keep time acceleration
at 1×; restart the mission after each fight (Esc → Quit → FLY AGAIN).

Defending: watch the radar warning receiver (it may warn of a missile only
late, when the missile's radar switches on); beam the missile (put it at
your 3 or 9 o'clock), release chaff, descend; or turn away early so the
missile runs out of energy. Within a few kilometres your R-73s can hit
BLUE-1.

| Outcome | Means |
|---|---|
| KILL | the policy shot you down |
| TIMEOUT | you survived the time limit (300 s by default) |
| ESCAPE | you got more than 150 km away |
| SHOT_DOWN | you shot BLUE-1 down with an R-73 |
| BANDIT_CRASH | you crashed or ejected |

The policy only knows the F-16C, so it assumes you carry AIM-120s and
defends against shots you can't make; a slow Su-25T at low altitude is
also outside its training. Read these fights as a check of how it
tracks, shoots and follows up, not as a fair score.

## Fitting the missile model to DCS

Every recording holds each DCS missile's flight ten times a second.
`missile_fit.py` fits a library missile to them: the drag curve comes
straight from the coast (after burn-out), then thrust, burn time and a
loft (climb angle, and how far below the target must be before the dive)
are searched until the model, fired from each recorded launch at the
recorded target, matches DCS's speed and altitude over time.

```
python dcs\missile_fit.py dcs_runs\live_*.jsonl                                   # compare only
python dcs\missile_fit.py dcs_runs\live_*.jsonl --save AIM-120C-DCS --platform-id F-16C-DCS
python sweep_envelope.py --missile AIM-120C-DCS                                     # its launch ranges
```

`AIM-120C-DCS` (and the platform `F-16C-DCS`, an F-16C carrying it) is in
the library, fitted to three recorded fights (six missile flights) with its
envelope calibrated: about twice the head-on reach of the original AIM-120.
To fly an existing policy in DCS with those ranges (no retraining):

```
python dcs_live.py models_bvr\latest_selfplay_v6.zip --platform F-16C-DCS
```

and to train with it, warm-started from a model trained on the F-16C:

```
python train_bvr.py --platform F-16C-DCS --opponent-platform F-16C-DCS --resume models_bvr\latest_selfplay_v6.zip
```

## Measuring the DCS AI's turns (turn_test.py)

The DCS AI flies BLUE-1 along the bridge's route, and it turns gently: in the
v8 fights about 45° of bank and 1.5 g, **about 2°/s**, where the simulator's
F-16 turns at about 8°/s. `turn_test.py` measures this with no policy: BLUE-1
flies 90° right, 90° back and 170° right, for each of four ways of building
the route, while RED-1 holds its fire:

| Variant | Route |
|---|---|
| `base` | first point 3 km ahead (as `dcs_live.py` flies) |
| `near1000` | first point 1 km ahead |
| `far_only` | no first point, only the far one (60 km) |
| `flyover1000` | first point 1 km ahead, both "Fly Over Point" |

```
python dcs\make_mission.py              # once: the turn test needs bridge version 3
python dcs\turn_test.py                 # start the mission; all four variants, about 15 min
python dcs\turn_test.py --variants base far_only --speed 280 --alt 6000
python dcs\turn_test.py --sim           # the same turns in BVR_RL's simulator, for comparison
```

For each turn it prints the turn rate while more than 30° is still to go,
the time to half and to 90% of the turn, the bank and the g, and appends a
row per turn to `dcs_runs\turn_test.csv` (the raw fight goes to
`dcs_runs\turn_<time>.jsonl`). If one variant turns much harder, the bridge
can use it; if none does, the simulator has to fly like the DCS AI instead.

**Result (8 October):** every variant turned the same way: 45° of bank,
1.41 g, 1.5-1.6°/s at 340 m/s and 9 km (a 90° turn half done in 30 s, 90%
in 52 s; 170° in 56 s and 97 s). That is the most bank the DCS AI uses on a
route; the waypoint distance makes no difference. So the platform
F-16C-DCSAI turns at no more than 45° of bank (`TURN_BANK_MAX`): 1.6°/s in
the simulator, against 7-9°/s for F-16C-DCS. The fly-over variant was not
measured: after about 11 minutes BLUE-1 reached bingo fuel (16%) and the
DCS AI took it home on its own, ignoring the route. Since bridge version 4
neither aircraft goes home at bingo fuel (`RTB_ON_BINGO` off).

**The attack-task turn (`hot`, bridge 4).** An AI that *attacks* is not
flying a route, and may turn as hard as it can. The `hot` variant tests
that: on the route it puts RED-1 90° off one wing, then gives BLUE-1 an
attack task on RED-1 with guns only (`OPT hot 1`) and measures the turn to
it; then the same off the other wing. It stops before RED-1 comes within
25 km, so the attack never gets near gun range, and RED-1 holds its fire.

```
python dcs\make_mission.py              # once: the hot variant needs bridge version 4
python dcs\turn_test.py --variants hot
```

In the simulator (`--sim`) the attack turn is flown at the airframe's full
agility (about 9°/s with 80° of bank): what the policy would get. If DCS
turns anywhere near that, the bridge can attack-steer the policy's big
heading changes; if not, the stick-and-throttle link (`Export.lua`, for a
player-flown F-16C) is the way.

## Letting the DCS AI defend BLUE-1 (--auto-defend)

The DCS AI on a route does not defend BLUE-1 hard: it flies the policy's
heading at 45° of bank while a missile comes in. In combat the DCS AI
turns at 6-12°/s, beams, dives and drops chaff. With `--auto-defend`
(bridge version 4) the bridge hands BLUE-1 to the DCS AI's own missile
defence (`REACTION_ON_THREAT EVADE_FIRE`) as soon as a missile is in
flight at it, and gives it back to the policy when the missile is gone.
While it defends, the policy's commands are kept but not flown and it may
not fire.

The simulator does the same for a platform with **`AUTO_DEFEND` 1**:
while an enemy missile is in flight at it, the aircraft beams the missile
on the side nearer its heading, dives 3 km (not below 1.5 km), flies its
fastest speed at the airframe's full agility, and holds fire. The platform
**F-16C-DCSAI-AD** is F-16C-DCSAI with `AUTO_DEFEND` 1 (and, like every
F-16 platform since SIM_REV 14, speed choices up to 460 m/s: BLUE-1 does
fly supersonic when level). Train on it, and `dcs_live.py` turns the bridge's auto-defence on
by itself for a checkpoint trained on it (or use `--auto-defend` with any
checkpoint):

```
python train_bvr.py --platform F-16C-DCSAI-AD --opponent-platform F-16C-DCS ...
python dcs\make_mission.py              # once: bridge version 4
python dcs_live.py models_bvr\<model>.zip --platform F-16C-DCSAI-AD --opp-platform F-16C-DCS --speed-boost
```

The terminal says when the DCS AI takes over and hands back, and
`results.csv` gets `auto_defend`, `defences` (how many times it took
over) and `defend_s` (seconds in all). F-16C-DCSAI-AD can meet F-16C-DCS in
self-play: the two see and choose the same things.

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

To see what DCS reported in a run (every shot, hit, kill and death, each
aircraft's altitude range, and how each death was counted):

```
python dcs\inspect_run.py dcs_runs\live_<date>_ep1.jsonl
```

A missile hit counts as a kill from the moment of the hit, as in the
simulator: DCS often reports the death only when the wreck reaches the
ground, which used to read as a CRASH.

## How DCS differs from training

The policy was trained in BVR_RL's simulator. In DCS, expect it to do
worse at first, for reasons that are useful to know:

- **Flying.** BLUE-1 is flown by the DCS AI along the commanded route,
  with DCS's F-16 flight model. Turn rates, climbs and speed changes are
  not the training autopilot's. A DCS AI on a route does not light its
  afterburner for the speed it is asked for: BLUE-1 stayed at full dry
  thrust (99% RPM, never above 280 m/s) while RED-1, in combat, reached
  490. **Fly with `--speed-boost`**: while BLUE-1 is more than 20 m/s
  slow, DCS is asked for 550 m/s, and it then does use the afterburner
  (the flame is visible). It spends the thrust mostly on climbing: in the
  same fight BLUE-1 ended 2.3 km higher, with 2.8 km more energy height,
  close to the simulator's F-16 with afterburner (13.8 km against 14.0,
  but 268 m/s against 317). So train as F-16C-DCS (with afterburner) and
  fly with `--speed-boost`. After each episode `dcs_live.py` prints how
  often BLUE-1 flew more than 30 m/s below the commanded speed
  (`speed_short_pct` in `results.csv`). `--eta-lock` (locked arrival
  times on the route) made no difference and is not needed; the platform
  F-16C-DCS-MIL (no afterburner) models BLUE-1 without the boost.
  **Climbs** are different again: on a route the DCS AI climbs at about
  Mach 0.85 whatever speed it is asked for (220, 400 or 550 m/s), at about
  0.02 m/s per metre still to climb (22-33 m/s with 1.2 km to go, 50-60
  with 3 km at 10 km, slower higher up), where the simulator's F-16 climbs
  with afterburner and accelerates past Mach 1. The platform
  **F-16C-DCSAI** (F-16C-DCS plus `CLIMB_MACH` 0.85, `CLIMB_THROTTLE` 0.94,
  fitted to the recorded climbs, and `TURN_BANK_MAX` 45° from
  `turn_test.py`) flies its climbs and turns that way. Train the agent
  on it with red on F-16C-DCS: self-play still works, since the two differ
  only in how they climb and turn.
  RED-1, the DCS AI in combat, is not limited this way: in five recorded
  fights it climbed at 60-138 m/s while holding about Mach 1.3, and went on
  to Mach 1.5 (446-518 m/s) once level. Since SIM_REV 13 the scripted
  ADAPTIVE red flies at those speeds (hot and cranking at 380-450 m/s,
  running at 420-500).
- **Shooting.** "Fire" is a request. The DCS AI launches when its own
  logic agrees (it is set to shoot at maximum range), usually within a
  second or two, and the policy is not allowed to request another shot
  until the launch is seen. Unanswered requests are counted.
- **Missile support.** In training a missile misses after 3 s without
  guidance from its shooter's radar track, until its own seeker takes
  over. Since SIM_REV 9 this holds for both sides: scripted opponents fly
  with their own copy of the radar model, aim their missiles at their own
  track and shoot only when the policy would be allowed to. DCS is more
  forgiving (a missile that loses its shooter's lock usually flies on and
  goes active). By default `dcs_live.py` applies the training rule to both
  aircraft: blue's support comes from the policy's track, red's from the
  same radar model run from red's side on DCS's true positions (DCS's own
  radar state is not readable). A missile that loses support is removed by
  the bridge (`DESTROY`); this counts as a SUPPORT_LOST miss, is shown in
  the terminal and in `results.csv` (`missiles_removed`). So a DCS AI that
  fires and turns away before its missile goes active loses that missile,
  as a scripted opponent would in training. `--support-rule dcs` leaves
  every missile to DCS.
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
{"format":"bvr_rl.dcs.v1", ..., "bridge":3, "agent":"BLUE-1", "red":"RED-1"}   every 2 s
{"ev":"fire", "t":..., "status":"requested|launched|timeout|refused", "seq":...}
{"ev":"bridge", "t":..., "status":"control|released", "agent":...}
{"ev":"mission_end", "t":...}
```

Python → DCS, UDP 15302, plain text:

```
CMD <seq> <heading deg, map north, clockwise> <altitude m> <speed m/s> <fire 0|1> <mission time>
                          (the time is dcs_live.py's; the bridge ignores it)
STOP
DESTROY <missile id>      (the support rule; answered with {"ev":"support_lost", "id", "status"})
OPT eta <off|mission|abs> (--eta-lock: locked arrival times on the route's points)
OPT near <m>              (turn_test.py: first route point this far ahead, 0 = none)
OPT wpt <turn|flyover>    (turn_test.py: the route points' type)
OPT redhold <0|1>         (turn_test.py: an AI red holds its fire)
OPT autodefend <0|1>      (--auto-defend: the DCS AI defends BLUE-1 while a missile is inbound)
OPT hot <0|1>             (turn_test.py: an attack task on red, guns only)
                          (each answered with {"ev":"bridge", "status":"opt", "near_m", "wpt",
                           "redhold", "autodefend", "hot"})
```

While it defends BLUE-1 the bridge sends
`{"ev":"bridge", "status":"defend", "on":true}` (and `false` when it hands
back), and refuses a fire request with reason `defending` (`hot` while an
attack task flies).

The header's `bridge` is the version: 2 understands OPT eta, 3 OPT near,
wpt and redhold, 4 OPT autodefend and hot (and no aircraft goes home at
bingo fuel).

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
