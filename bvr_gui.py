"""
bvr_gui.py  —  BVR training and visualisation GUI server
=========================================================

    python bvr_gui.py [--port 5006] [--ws-port 5007] [--model-dir models_bvr]

Open http://localhost:5006 in a browser.

Two concurrent services:
  HTTP :5006  — serves bvr_gui.html (the UI)
  WS   :5007  — real-time push channel to the browser

Training mode:
  Spawns train_bvr.py as a subprocess. The training callback writes
  bvr_metrics.json every 10 episodes; the server polls it every second
  and pushes updates over WebSocket. This keeps training and GUI fully
  decoupled — the GUI never slows down training.

Eval mode:
  Loads a checkpoint with MaskablePPO.load() and runs episodes inside
  THIS process in a background thread. Each sim frame is pushed over
  WebSocket at the speed the user selects (0.5×–100× realtime).

Episode recording:
  The last RECORD_BUFFER_SIZE frames of the most recent eval episode
  are kept in memory. The browser can request a full replay of that
  episode at any speed.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
METRICS_F  = HERE / "bvr_metrics.json"
GUI_HTML   = HERE / "bvr_gui.html"
MODEL_DIR  = HERE / "models_bvr"   # overwritten from --model-dir in main()

RECORD_BUFFER_SIZE = 3000   # frames per recorded episode

# ─────────────────────────────────────────────────────────────────────
# shared state (written by threads, read by the async broadcast loop)
# ─────────────────────────────────────────────────────────────────────
class Hub:
    """Thread-safe message hub. All threads push here; WS loop drains."""
    def __init__(self):
        self._lock   = threading.Lock()
        self._latest_telemetry = None   # most recent eval frame
        self._latest_metrics   = None   # most recent training metrics
        self._status  = {"state": "idle", "msg": "Ready"}
        self._clients = set()
        self._queue   = []              # outbound message queue
        self._replay_frames = []        # last episode recording
        self._replay_meta   = {}

    def push(self, msg: dict):
        with self._lock:
            self._queue.append(msg)

    def drain(self) -> list:
        with self._lock:
            out, self._queue = self._queue, []
        return out

    def set_status(self, state: str, msg: str = ""):
        self.push({"type": "status", "state": state, "msg": msg})

    def set_replay(self, frames: list, meta: dict):
        with self._lock:
            self._replay_frames = frames
            self._replay_meta   = meta


HUB = Hub()


# ─────────────────────────────────────────────────────────────────────
# HTTP handler — serves GUI_HTML and bvr_metrics.json
# ─────────────────────────────────────────────────────────────────────
class GUIHandler(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            if not GUI_HTML.exists():
                self.send_error(404, "bvr_gui.html not found next to bvr_gui.py")
                return
            data = GUI_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/metrics":
            if METRICS_F.exists():
                data = METRICS_F.read_bytes()
            else:
                data = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/replay":
            with HUB._lock:
                payload = json.dumps({"frames": HUB._replay_frames,
                                      "meta": HUB._replay_meta}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path == "/models":
            data = json.dumps(_list_models()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)


def _list_models() -> dict:
    """
    Scan MODEL_DIR (recursively — checkpoints live both at the top level,
    e.g. "latest.zip", and under "archive/") for saved SB3 checkpoints.
    SB3's save() only appends ".zip" when the given path has no suffix at
    all, so a name like "best_STRAIGHT_0.720" is written WITHOUT a .zip
    extension — filtering by extension would silently drop it. Filtering by
    zipfile.is_zipfile() catches every real checkpoint regardless of name.
    """
    items = []
    root = Path(MODEL_DIR)
    if root.is_dir():
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            try:
                if not zipfile.is_zipfile(p):
                    continue
                st = p.stat()
            except OSError:
                continue
            items.append({"path": p.relative_to(root).as_posix(),
                          "mtime": st.st_mtime, "size": st.st_size})
    items.sort(key=lambda d: d["mtime"], reverse=True)
    return {"models": items, "model_dir": str(root)}


# ─────────────────────────────────────────────────────────────────────
# Training subprocess manager
# ─────────────────────────────────────────────────────────────────────
class TrainingManager:
    def __init__(self, model_dir="models_bvr"):
        self._proc      = None
        self._thread    = None
        self._model_dir = model_dir
        self._stop_evt  = threading.Event()

    def start(self, config: dict):
        if self._proc and self._proc.poll() is None:
            return   # already running
        self._stop_evt.clear()
        cmd = [
            sys.executable, str(HERE / "train_bvr.py"),
            "--save-dir",   self._model_dir,
            "--steps",      str(config.get("steps", 5_000_000)),
            "--n-envs",     str(config.get("n_envs", 1)),
            "--opponent",   config.get("opponent", "straight"),
            "--seed",       str(config.get("seed", 42)),
            "--gamma",      str(config.get("gamma", 0.997)),
            "--lr",         str(config.get("lr", 2.5e-4)),
            "--batch-size", str(config.get("batch_size", 256)),
        ]
        if config.get("resume"):
            # The GUI's dropdown sends a path relative to the model dir (as
            # listed by /models, e.g. "latest.zip" or "archive/best_..."),
            # not relative to this process's cwd — join it with the same
            # model_dir this manager was constructed with.
            cmd += ["--resume", os.path.join(self._model_dir, config["resume"])]
        if config.get("run_name"):
            cmd += ["--run-name", str(config["run_name"]).strip()]
        if not config.get("curriculum", True):
            cmd += ["--no-curriculum"]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True)
        self._thread = threading.Thread(target=self._tail, daemon=True)
        self._thread.start()
        HUB.set_status("training", f"Training: {config.get('opponent','straight')}")

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._stop_evt.set()
        HUB.set_status("idle", "Training stopped")

    def _tail(self):
        """Read subprocess stdout line by line, push to hub."""
        for line in self._proc.stdout:
            line = line.rstrip()
            if line:
                HUB.push({"type": "log", "msg": line})
        self._proc.wait()
        HUB.set_status("idle", f"Training ended (exit {self._proc.returncode})")

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


# ─────────────────────────────────────────────────────────────────────
# Eval / visualisation runner
# ─────────────────────────────────────────────────────────────────────
# Every model this GUI can load was trained by train_bvr.py, which wraps the
# vec env in VecFrameStack(n_stack=N_STACK) — the saved policy's actual input
# is N_STACK stacked frames per key, not one. N_STACK is a module constant in
# train_bvr.py (not a CLI flag), so it can't drift between a checkpoint and
# this file without a code change on both sides.
N_STACK = 8


class FrameStacker:
    """
    Reproduces stable_baselines3.common.vec_env.stacked_observations.
    StackedObservations for a single, non-vectorized env — verified
    byte-for-byte identical to it (same seed, same action sequence, both
    with and without a reset mid-sequence) before this was wired in.

    EvalRunner drives BvrEnv directly, one frame at a time, so it can read
    env._state for the telemetry websocket and control exactly when an
    episode ends. Wrapping it in the real DummyVecEnv+VecFrameStack instead
    would fight that: VecEnv auto-resets the underlying env the instant a
    step returns done=True, so env._state would already belong to the NEXT
    episode by the time this loop got to build the terminal frame or decide
    to break. This class gets the same stacked observation without taking
    stepping control away from the loop below.
    """
    def __init__(self, n_stack: int):
        self.n_stack = n_stack
        self._buf: dict = {}

    def _ensure(self, key, dim, dtype):
        cur = self._buf.get(key)
        if cur is None or cur.shape[0] != dim * self.n_stack:
            self._buf[key] = np.zeros(dim * self.n_stack, dtype=dtype)

    def reset(self, obs):
        if isinstance(obs, dict):
            return {k: self._reset_one(k, v) for k, v in obs.items()}
        return self._reset_one("__box__", obs)

    def _reset_one(self, key, arr):
        arr = np.asarray(arr)
        self._ensure(key, arr.shape[0], arr.dtype)
        buf = self._buf[key]
        buf[:] = 0
        buf[-arr.shape[0]:] = arr
        return buf.copy()

    def update(self, obs):
        if isinstance(obs, dict):
            return {k: self._update_one(k, v) for k, v in obs.items()}
        return self._update_one("__box__", obs)

    def _update_one(self, key, arr):
        arr = np.asarray(arr)
        dim = arr.shape[0]
        buf = self._buf[key]
        buf[:] = np.roll(buf, -dim)
        buf[-dim:] = arr
        return buf.copy()


class EvalRunner:
    def __init__(self, model_dir="models_bvr"):
        self._model_dir = model_dir
        self._thread    = None
        self._stop_evt  = threading.Event()
        self._speed     = 5.0    # simulated seconds per real second
        self._speed_lock = threading.Lock()

    def set_speed(self, s: float):
        with self._speed_lock: self._speed = float(np.clip(s, 0.1, 200.0))

    def start(self, config: dict):
        if self._thread and self._thread.is_alive():
            self._stop_evt.set()
            self._thread.join(timeout=3)
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, args=(config,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        HUB.set_status("idle", "Eval stopped")

    def _run(self, config: dict):
        try:
            from bvr_env import BvrEnv
            from bvr_opponents import BvrOpponentType
            opponent_name = config.get("opponent", "STRAIGHT").upper()
            opponent = BvrOpponentType[opponent_name]

            # Load the checkpoint FIRST — env construction below needs to know
            # whether this model was trained with the privileged (Dict, "obs"
            # + "priv") observation space or not, and building the env with
            # the wrong one is exactly what used to blow up inside
            # model.predict() with an opaque numpy IndexError: the policy's
            # own observation_space is a Dict, obs[key] on a plain Box array
            # is not valid indexing, and that's the whole error.
            model = None
            stacker = None
            privileged = False
            ckpt = config.get("checkpoint")
            # The GUI's dropdown sends a path relative to the model dir (as
            # listed by /models, e.g. "latest.zip" or "archive/best_..."),
            # not relative to the process cwd — resolve it against the same
            # model_dir this runner was constructed with.
            ckpt_path = Path(self._model_dir) / ckpt if ckpt else None
            if ckpt_path and ckpt_path.exists():
                try:
                    from sb3_contrib import MaskablePPO
                    from gymnasium import spaces
                    model = MaskablePPO.load(str(ckpt_path), env=None)
                    privileged = isinstance(model.observation_space, spaces.Dict)
                    stacker = FrameStacker(N_STACK)
                    HUB.push({"type":"log","msg":f"Loaded checkpoint: {ckpt_path}"})
                except Exception as e:
                    HUB.push({"type":"log","msg":f"Could not load checkpoint: {e}; using random policy"})
            elif ckpt:
                HUB.push({"type":"log","msg":f"Checkpoint not found: {ckpt_path}; using random policy"})

            # Use the same calibrated Rmax/Rnez table training used (see
            # sweep_envelope.py) — evaluating against a different envelope
            # than the one trained on silently invalidates every launch
            # decision in this GUI's HUD.
            envelope_table = "envelope.npz" if os.path.exists("envelope.npz") else None
            env = BvrEnv(opponent_type=opponent, seed=config.get("seed", 0),
                         privileged_critic=privileged, gamma_discount=0.997,
                         envelope_table=envelope_table)

            HUB.set_status("eval", f"Eval vs {opponent_name}")
            episode = 0
            while not self._stop_evt.is_set():
                obs, info = env.reset()
                sobs = stacker.reset(obs) if stacker else None
                frames = []
                ep_r = 0.0
                while not self._stop_evt.is_set():
                    # action
                    if model is not None:
                        mask = env.action_masks()
                        action, _ = model.predict(sobs, deterministic=True,
                                                  action_masks=mask)
                    else:
                        action = env.action_space.sample()
                        action[-1] = 0   # don't fire randomly

                    _t0 = time.perf_counter()
                    obs, r, done, trunc, step_info = env.step(action)
                    sobs = stacker.update(obs) if stacker else None
                    _step_ms = time.perf_counter() - _t0
                    ep_r += r
                    s = env._state
                    frame = _build_frame(s, env, step_info)
                    frame["ep"] = episode          # lets the view reset trails on a new episode
                    frames.append(frame)
                    if len(frames) > RECORD_BUFFER_SIZE:
                        frames.pop(0)
                    HUB.push({"type": "telemetry", "data": frame})

                    # Pacing: env.step() = 1 sim-second (DECISION_HZ=1).
                    # Sleep so the wall clock matches the requested playback speed.
                    # At 1x: sleep ~977ms (sim took ~23ms).  At 50x: sleep 0ms.
                    with self._speed_lock: spd = self._speed
                    sleep_needed = (1.0 / spd) - _step_ms
                    if sleep_needed > 0.002:
                        time.sleep(sleep_needed)

                    if done or trunc: break

                # store replay
                meta = {"episode": episode, "outcome": step_info.get("terminal_outcome","?"),
                        "reward": round(ep_r, 3),
                        "shots": step_info.get("shots_fired", 0),
                        "support_losses": step_info.get("support_losses", 0)}
                HUB.set_replay(list(frames), meta)
                HUB.push({"type":"episode_end","meta":meta})
                episode += 1
            env.close()
        except Exception as e:
            import traceback
            HUB.push({"type":"log","msg":f"Eval error: {e}\n{traceback.format_exc()}"})
            HUB.set_status("idle", f"Eval error: {e}")


def _build_frame(s: dict, env, info: dict) -> dict:
    """Compact frame for the WebSocket stream — only what the display needs."""
    def _pos(la, lo, alt):
        import math
        R=6_371_000.0; D=math.pi/180
        x=(lo-35.0)*D*R*math.cos(39.0*D)
        y=(la-39.0)*D*R
        return [round(x,1), round(y,1), round(alt,1)]

    own = _pos(s.get("lat",39),s.get("lon",35),s.get("alt",9000))
    tgt = _pos(s.get("lat_t",39),s.get("lon_t",35),s.get("alt_t",9000))

    # missiles: compact
    msls = []
    for m in s.get("missiles",[]) or []:
        if m.get("state") in ("HIT","MISS"): continue
        la,lo,al = m.get("lat",39),m.get("lon",35),m.get("alt",9000)
        msls.append({
            "id": m["id"], "owner": m["owner"],
            "pos": _pos(la,lo,al),
            "vel": m.get("vel",[0.0,0.0,0.0]),   # ENU — display points the body along this
            "state": m["state"],
            "needs_support": m.get("needs_support",0),
            "seeker_active":  m.get("seeker_active",0),
            "tgo":            round(m.get("tgo_est",0),1),
            "t_stale":        round(m.get("t_since_update",0),2),
            "handoff_range":  m.get("handoff_range",10000),
            "mach":           m.get("mach",0),
            "rng_to_target":  m.get("rng_to_target",0),
        })

    est = env._track.estimate()
    rmax,rnez = env._own_envelope(est)
    rmxt,rnzt = env._threat_envelope(est)
    rwr = s.get("rwr",{}) or {}

    return {
        "t":      round(s.get("t_sim",0),2),
        "own":    own,
        "tgt":    tgt,
        # attitude — psi (heading), theta (pitch), phi (bank); the 3D view banks
        # and pitches the jets with these, so climbs/dives/turns are visible
        "psi":    round(s.get("psi",0),4),
        "theta":  round(s.get("theta",0),4),
        "phi":    round(s.get("phi",0),4),
        "psi_t":  round(s.get("psi_t",0),4),
        "theta_t":round(s.get("theta_t",0),4),
        "phi_t":  round(s.get("phi_t",0),4),
        "gamma":  round(s.get("gamma_fpa",0),4),
        "gamma_t":round(s.get("gamma_fpa_t",0),4),
        "nz":     round(s.get("nz",1),2),
        "nz_t":   round(s.get("nz_t",1),2),
        "speed":  round(s.get("speed",280),1),
        "speed_t":round(s.get("speed_t",290),1),
        "mach":   round(s.get("mach",0.9),3),
        "mach_t": round(s.get("mach_t",0.9),3),
        "alt":    round(s.get("alt",9000),1),
        "alt_t":  round(s.get("alt_t",9000),1),
        "rng":    round(s.get("range",50000),1),
        "clos":   round(s.get("closure",0),1),
        "aa":     round(s.get("aa_deg",0),1),
        "aa_t":   round(s.get("aa_deg_t",0),1),
        "fuel":   round(s.get("fuel_frac",1),3),
        "wpn":    int(s.get("wpn_remaining",4)),
        "wpn_t":  int(s.get("wpn_remaining_t",4)),
        "track":  env._track.state,
        "pos_sig":round(est["pos_sigma"],0) if est["valid"] else 9999,
        "trk_age":round(est["age"],1),
        "rmax":   round(rmax,0),"rnez": round(rnez,0),
        "rmax_t": round(rmxt,0),"rnez_t":round(rnzt,0),
        "missiles": msls,
        "rwr_warn": int(rwr.get("launch_warn",0)),
        "rwr_sa":   int(rwr.get("seeker_active",0)),
        "rwr_bear": round(rwr.get("launch_bearing",0),3),
        "rwr_n":    int(rwr.get("n_threats",0)),
        "step":   info.get("step",0),
        "outcome": info.get("terminal_outcome"),
        "shots":   info.get("shots_fired",0),
    }


# ─────────────────────────────────────────────────────────────────────
# Metrics poller — reads bvr_metrics.json written by train_bvr.py
# ─────────────────────────────────────────────────────────────────────
async def metrics_poll_loop():
    last_mtime = 0.0
    while True:
        await asyncio.sleep(1.0)
        try:
            if not METRICS_F.exists(): continue
            mt = METRICS_F.stat().st_mtime
            if mt <= last_mtime: continue
            last_mtime = mt
            data = json.loads(METRICS_F.read_text())
            HUB.push({"type": "metrics", "data": data})
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
# WebSocket server
# ─────────────────────────────────────────────────────────────────────
TRAINING_MGR = None
EVAL_RUNNER  = None


async def handle_client(ws):
    HUB._clients.add(ws)
    await ws.send(json.dumps(HUB._status))
    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.05)
                msg = json.loads(raw)
                await _handle_cmd(msg)
            except asyncio.TimeoutError:
                pass
            except Exception:
                break   # client disconnected — normal on tab refresh
            for m in HUB.drain():
                try:
                    await ws.send(json.dumps(m))
                except Exception:
                    break
    except Exception:
        pass   # suppress WinError 10054 and ConnectionClosedOK noise
    finally:
        HUB._clients.discard(ws)


async def _handle_cmd(msg: dict):
    cmd = msg.get("cmd")
    if cmd == "start_training":
        TRAINING_MGR.start(msg.get("config", {}))
    elif cmd == "stop_training":
        TRAINING_MGR.stop()
    elif cmd == "start_eval":
        EVAL_RUNNER.start(msg.get("config", {}))
    elif cmd == "stop_eval":
        EVAL_RUNNER.stop()
    elif cmd == "set_eval_speed":
        EVAL_RUNNER.set_speed(float(msg.get("speed", 5.0)))
    elif cmd == "ping":
        HUB.push({"type": "pong"})


# ─────────────────────────────────────────────────────────────────────
def run_http(port: int):
    os.chdir(HERE)
    srv = HTTPServer(("0.0.0.0", port), GUIHandler)
    srv.serve_forever()


async def run_ws(port: int):
    import websockets
    async with websockets.serve(handle_client, "0.0.0.0", port,
                                 max_size=4*1024*1024):
        await metrics_poll_loop()   # runs forever alongside WS server


def main():
    global TRAINING_MGR, EVAL_RUNNER, MODEL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",      type=int, default=5006)
    ap.add_argument("--ws-port",   type=int, default=5010)
    ap.add_argument("--model-dir", type=str, default="models_bvr")
    args = ap.parse_args()

    MODEL_DIR    = HERE / args.model_dir
    TRAINING_MGR = TrainingManager(args.model_dir)
    EVAL_RUNNER  = EvalRunner(args.model_dir)

    # HTTP in a daemon thread
    t = threading.Thread(target=run_http, args=(args.port,), daemon=True)
    t.start()
    print(f"[bvr_gui]  http://localhost:{args.port}")
    print(f"[bvr_gui]  ws://localhost:{args.ws_port}")
    print(f"[bvr_gui]  model dir: {args.model_dir}")
    print(f"[bvr_gui]  Press Ctrl+C to exit")

    try:
        asyncio.run(run_ws(args.ws_port))
    except KeyboardInterrupt:
        TRAINING_MGR.stop()
        EVAL_RUNNER.stop()


if __name__ == "__main__":
    main()
