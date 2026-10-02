# DCS World bridge, step 1: record and check

Goal: before a trained policy flies in DCS World, check that the inputs it
would receive there look like the ones it saw in training.

1. Record an engagement in DCS with `bvr_logger.lua`.
2. Replay the recording through BVR_RL's observation code with `dcs_obs_check.py`,
   and compare every policy input with the simulator.

Nothing is sent to DCS yet; the policy does not fly. Later steps (shadow mode,
then flying the aircraft) build on the same `DcsReplayWorld` interface.

## 1. Let mission scripts write files

The logger needs Lua's `io` and `lfs` modules, which DCS removes from mission
scripts. In `<DCS install folder>\Scripts\MissionScripting.lua`, comment out:

```lua
--sanitizeModule('io')
--sanitizeModule('lfs')
```

**Security:** while these lines are commented out, any mission you load,
including ones downloaded or joined online, can read and write files on your
computer. Restore the two lines when you are done recording. DCS updates also
overwrite this file, so the edit has to be redone after an update.

## 2. Build the mission

- Place one blue and one red airplane (any types). Only the first blue and
  first red aircraft are replayed unless you name others with `--blue` / `--red`,
  but every airplane in the mission is recorded.
- Add a trigger: **MISSION START**, no condition, action **DO SCRIPT FILE** →
  `bvr_logger.lua`.
- Give them a fight, for example two AI flights with an intercept task and
  AIM-120s, starting 60–110 km apart like the training starts.

When the mission starts, DCS shows "bvr_logger: recording to …". The file is
`Saved Games\DCS\Logs\bvr_rl_<numbers>.jsonl`, written ten times a second,
and flushed every second.

## 3. Check the inputs

```
python dcs_obs_check.py "C:\Users\<you>\Saved Games\DCS\Logs\bvr_rl_43200_804789.jsonl" --platform F-16C --opp-platform F-16C --max-steps 400 --out dcs_check
```

- `--platform` / `--opp-platform`: the library platforms whose missile envelope,
  radar and speed choices the policy was trained with.
- `--max-steps`: the episode length the policy was trained with. The policy
  sees elapsed time as a fraction of it.
- Several recordings can be given at once.

The checker also flies 20 simulator episodes against SHOOTER (`--ref-episodes`,
`--ref-opponent`) as the reference. It prints a table and writes three files
to `dcs_check/`:

| File | Contents |
|---|---|
| `report.txt` | The table: each input's 1st, 50th and 99th percentile in DCS and in the simulator |
| `summary.csv` | The same, one row per input, with the flags |
| `dcs_inputs.csv` | Every replayed decision: time, the action inferred from what DCS flew, all inputs |

Flags:

- **OUTSIDE**: more than 20% of the DCS steps are outside the range the
  simulator produced. The policy has rarely or never seen those values.
- **CLIP**: DCS pushes the input beyond its normalisation limits at least 5
  percentage points more often than the simulator does, so the policy sees it
  pinned at the limit.

A flag means one of two things. It may be a bridge mistake (a unit, sign or
axis error), which should be fixed. Or it may be a real difference between DCS
and the simulator (fuel load, speeds, missile ranges), which has to be decided
on before the policy flies.

## What is estimated rather than recorded

DCS mission scripts don't expose everything the simulator reports:

| Input | How it is estimated |
|---|---|
| Angle of attack | Pitch minus flight-path angle |
| Body rates, load factor | Finite differences of the recorded attitude and velocity. Rolls faster than about 3 rad/s are smoothed at 10 samples per second |
| Missile seeker active | Inside the platform missile's mean hand-off range of its target |
| Missile time-to-go | Range to target ÷ closing speed, from true positions |
| Weapons remaining | Radar-guided air-to-air missiles in the aircraft's last ammo report |

Radar tracks are produced by BVR_RL's own radar and track model, running on
DCS's true positions, exactly as in training. DCS's own radar is not used yet.

## Status

- Tested by writing simulator episodes in the logger's format and replaying
  them (`test_sim.py`: DCS round trip). The replayed inputs match the
  simulator's, apart from the estimated ones above and the first half-second
  while the track forms.
- The logger was run in Lua 5.1 against a mock of the DCS scripting API. It
  has **not yet been run inside DCS itself**. If a DCS call behaves
  differently from the mock, the logger reports it in `dcs.log` as
  `bvr_logger … error`.
- 1v1 only. 2v1 replay, with the wingman's inputs, comes later.
