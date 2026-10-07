# Parameter library

Every number the simulation uses for an aircraft, its radar and its missile.

```
builtin/<kind>/<id>.json   shipped with the code, read-only
user/<kind>/<id>.json      your copies (create them in the GUI: LIBRARY → Copy)
envelopes/<missile>-<fingerprint>.npz   calibrated launch envelopes
```

Kinds: `airframe`, `radar`, `missile`, and `platform` (an airframe + radar +
missile, with weapon count, radar cross-section and the agent's command set).

Each item holds `params` (the values) and `sources` (where each value comes
from). What each parameter means, its unit and its valid range are defined
once, in `bvr_library.py` (`SCHEMA`).

- Built-ins are never modified. Copy one and edit the copy.
- `python bvr_perf.py <platform>` (or the GUI's *Performance card*) flies the
  flight model with an item's parameters and reports top speed, turn, climb,
  ceiling and autopilot behaviour, with warnings.
- A missile needs a calibrated envelope before it can be trained with:
  `python sweep_envelope.py --missile <id>` (or the GUI's *Calibrate*). The
  file name carries a fingerprint of the missile's parameters, so editing the
  missile invalidates the old table and training refuses to start until it
  is recalibrated.
- Every checkpoint stores the platforms and the full items it was trained
  with, so later edits here never change what an old checkpoint means.

The F-16C, AIM-120 and APG-68 items reproduce the model classes in
`f16_sim.py`, `missile_sim.py` and `bvr_radar_sim.py` exactly. The generic
UCAV items are illustrative, not data for any real aircraft.
