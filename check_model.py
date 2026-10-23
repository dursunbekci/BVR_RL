"""
check_model.py  —  a quick look at a checkpoint before (or after) training from it
==================================================================================

Flies the model a few fights against a copy of itself (self-play) and against
the scripted ADAPTIVE shooter, with the symmetric policy on as training applies
it, and prints the outcomes, the share of decisions with a firm radar track
and the shots per fight. Escapes and a track near 0% mean the model keeps away
from the fight; a healthy one holds a track a third of the time or more.

    python check_model.py models_bvr/dcsfit_v6.zip
    python check_model.py models_bvr/dcsfit_v6_16M.zip --fights 12 --platform F-16C-DCSFIT-R

Run on the training machine it also checks that machine's code: the same
model should give about the same numbers here and there.
"""

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def fights(model_path, platform, opponent_platform, opponent, n, max_steps=400):
    import torch
    torch.set_num_threads(1)
    from bvr_env import BvrEnv
    from bvr_opponents import BvrOpponentType as T
    from bvr_selfplay import FrameStacker, N_STACK, load_policy
    model, priv = load_policy(model_path)
    model.policy.set_symmetric(True)
    with tempfile.TemporaryDirectory() as pool:
        shutil.copy(model_path, os.path.join(pool, "sp_0000_0000000000.zip"))
        env = BvrEnv(opponent_type=T.SELF_PLAY, privileged_critic=priv, doctrine="BALANCED", seed=3,
                     selfplay_pool=pool, platform=platform, opponent_platform=opponent_platform)
        env.MAX_STEPS = max_steps
        env._sp_scripted_frac = 0.0 if opponent == "self" else 1.0
        out = []
        for s in range(n):
            obs, _ = env.reset(seed=100 + s)
            st = FrameStacker(N_STACK)
            so = st.reset(obs)
            trk = steps = 0
            while True:
                a, _ = model.predict(so, deterministic=True, action_masks=env.action_masks())
                obs, r, done, trunc, info = env.step(a)
                so = st.update(obs)
                steps += 1
                trk += env._trk_state() == 2
                if done or trunc:
                    break
            out.append((info.get("terminal_outcome", "?"), trk / steps, info.get("shots_fired", 0)))
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--platform", default="F-16C-DCSFIT-R")
    ap.add_argument("--opponent-platform", default=None, help="default: the same as --platform")
    ap.add_argument("--fights", type=int, default=8, help="per opponent (default 8)")
    args = ap.parse_args(argv)
    from bvr_library import SIM_REV
    opp_plat = args.opponent_platform or args.platform
    print(f"{args.model} on SIM_REV {SIM_REV}, {args.platform} v {opp_plat}, {args.fights} fights each")
    for opponent in ("self", "scripted"):
        res = fights(args.model, args.platform, opp_plat, opponent, args.fights)
        oc = [o for o, _, _ in res]
        print(f"  v {opponent:8s}: " + ", ".join(f"{k} {oc.count(k)}" for k in sorted(set(oc)))
              + f" | track {np.mean([t for _, t, _ in res]):.0%} of decisions, "
                f"{np.mean([s for *_, s in res]):.1f} shots a fight")


if __name__ == "__main__":
    main()
