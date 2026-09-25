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
import re
import signal
import subprocess
import sys
import threading
import time
import zipfile
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
METRICS_F  = HERE / "bvr_metrics.json"
GUI_HTML   = HERE / "bvr_gui.html"
MODEL_DIR  = HERE / "models_bvr"   # overwritten from --model-dir in main()
XPLAY_DIR  = HERE / "crossplay_results"

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
        # Training and eval are independent: training is a subprocess, eval is
        # a thread in this process, and running both at once is supported. A
        # single "state" string cannot represent that, so each has its own flag.
        self._status  = {"type": "status", "training": False,
                         "eval": False, "crossplay": False, "msg": "Ready"}
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

    def set_status(self, *, training: bool = None, evaluating: bool = None,
                   crossplay: bool = None, msg: str = None):
        """Update only the fields given, then broadcast the whole snapshot."""
        with self._lock:
            if training is not None:   self._status["training"] = bool(training)
            if evaluating is not None: self._status["eval"]     = bool(evaluating)
            if crossplay is not None:  self._status["crossplay"] = bool(crossplay)
            if msg is not None:        self._status["msg"]      = msg
            snapshot = dict(self._status)
        self.push(snapshot)

    def status_snapshot(self) -> dict:
        with self._lock:
            return dict(self._status)

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

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        """Library edits. Built-ins are protected in bvr_library itself."""
        import bvr_library as L
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            path = urlparse(self.path).path
            if path == "/library/copy":
                it = L.copy_item(body["kind"], body["id"], body["new_id"], body.get("new_name"))
                self._json({"ok": True, "item": it})
            elif path == "/library/save":
                L.save_item(body["item"])
                self._json({"ok": True})
            elif path == "/library/delete":
                L.delete_item(body["kind"], body["id"])
                self._json({"ok": True})
            else:
                self.send_error(404)
        except (L.LibraryError, KeyError, ValueError) as e:
            self._json({"ok": False, "error": str(e)}, 400)

    def do_GET(self):
        if self.path.startswith("/library"):
            return self._library_get()
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
        elif self.path == "/crossplay":
            data = json.dumps(_crossplay_summary()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/models":
            data = json.dumps(_list_models()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)


def _library_summary() -> dict:
    import bvr_library as L
    items = L.list_items()
    calibrated = {}
    for it in items:
        if it["kind"] == "missile":
            try:
                calibrated[it["id"]] = L.envelope_path(L.missile_config(it["id"])).exists()
            except Exception:
                calibrated[it["id"]] = False
    return {"items": items, "schema": L.schema_json(), "calibrated": calibrated}


def _library_item(kind, item_id) -> dict:
    import bvr_library as L
    it = L.get_item(kind, item_id)
    out = {"item": it, "fingerprint": L.fingerprint(it), "errors": L.validate(it)}
    if kind == "missile":
        out["calibrated"] = L.envelope_path(L.missile_config(item_id)).exists()
    out["used_by"] = [p["id"] for p in L.list_items("platform")
                      if kind != "platform"
                      and L.get_item("platform", p["id"])["params"].get(kind) == item_id]
    return out


def _handler_library_get(self):
    import bvr_library as L
    u = urlparse(self.path)
    q = {k: v[0] for k, v in parse_qs(u.query).items()}
    try:
        if u.path == "/library":
            self._json(_library_summary())
        elif u.path == "/library/item":
            self._json(_library_item(q["kind"], q["id"]))
        elif u.path == "/library/perf":
            from bvr_perf import card_for
            self._json(card_for(q["kind"], q["id"]))
        else:
            self.send_error(404)
    except (L.LibraryError, KeyError, ValueError) as e:
        self._json({"error": str(e)}, 400)


GUIHandler._library_get = _handler_library_get


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


def _crossplay_summary() -> dict:
    """The last cross-play run, without the per-game list the page doesn't use."""
    f = XPLAY_DIR / "crossplay.json"
    if not f.exists():
        return {}
    try:
        d = json.loads(f.read_text())
    except (OSError, ValueError):
        return {}
    d.pop("games", None)
    return d


# ─────────────────────────────────────────────────────────────────────
# Cross-play evaluation subprocess
# ─────────────────────────────────────────────────────────────────────
class CrossplayManager:
    """
    Runs crossplay.py as a subprocess, like training, so a long round-robin
    never blocks the server and survives the browser closing. Progress lines
    become progress messages; everything else goes to the log.
    """
    _PROGRESS = re.compile(r"\[crossplay\] (\d+)/(\d+) games")

    def __init__(self, model_dir="models_bvr"):
        self._proc = None
        self._model_dir = model_dir

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, config: dict):
        if self.running:
            return
        models = [os.path.join(self._model_dir, m) for m in config.get("models", [])]
        if not models:
            HUB.push({"type": "log", "msg": "[crossplay] no checkpoints selected"})
            return
        cmd = [sys.executable, str(HERE / "crossplay.py"), *models,
               "--episodes", str(int(config.get("episodes", 40))),
               "--workers",  str(int(config.get("workers", 8))),
               "--doctrine", str(config.get("doctrine", "balanced")).lower(),
               "--out",      str(XPLAY_DIR)]
        scripted = [str(x).lower() for x in config.get("scripted", [])]
        if scripted:
            cmd += ["--scripted", *scripted]
        if config.get("stochastic"):
            cmd += ["--stochastic"]
        if config.get("scripted_platform"):
            cmd += ["--scripted-platform", str(config["scripted_platform"])]
        # Own process group, so stop() can interrupt the run and its worker
        # processes together without signalling this server.
        kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
              else {"start_new_session": True})
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      bufsize=1, text=True, cwd=str(HERE), **kw)
        threading.Thread(target=self._tail, daemon=True).start()
        HUB.set_status(crossplay=True, msg=f"Cross-play: {len(models)} checkpoints")

    def _tail(self):
        proc = self._proc
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            m = self._PROGRESS.search(line)
            if m:
                HUB.push({"type": "crossplay_progress", "done": int(m.group(1)),
                          "total": int(m.group(2)), "msg": line})
            else:
                HUB.push({"type": "log", "msg": line})
        proc.wait()
        ok = proc.returncode == 0
        HUB.push({"type": "crossplay_done", "ok": ok})
        HUB.set_status(crossplay=False, msg="Cross-play finished" if ok
                       else f"Cross-play stopped (exit {proc.returncode})")

    def stop(self):
        if not self.running:
            return
        threading.Thread(target=self._stop, daemon=True).start()

    def _stop(self):
        """Interrupt the whole group (the run and its workers), then force it."""
        proc = self._proc
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=15)
            return
        except Exception:
            pass
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                               capture_output=True)
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()


class CalibrationManager:
    """Runs sweep_envelope.py for one missile; progress lines go to the page."""
    _PROGRESS = re.compile(r"\[(\d+)/(\d+)\]")

    def __init__(self):
        self._proc = None
        self._missile = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, missile_id: str):
        if self.running:
            HUB.push({"type": "log", "msg": f"[calibrate] already calibrating {self._missile}"})
            return
        self._missile = missile_id
        self._proc = subprocess.Popen(
            [sys.executable, str(HERE / "sweep_envelope.py"), "--missile", missile_id],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True, cwd=str(HERE))
        threading.Thread(target=self._tail, daemon=True).start()
        HUB.push({"type": "log", "msg": f"[calibrate] {missile_id}: flying the missile over the "
                                        f"launch grid (a few minutes)"})

    def _tail(self):
        proc, mid = self._proc, self._missile
        for line in proc.stdout:
            line = line.rstrip()
            m = self._PROGRESS.search(line)
            if m:
                HUB.push({"type": "calibrate_progress", "id": mid, "done": int(m.group(1)),
                          "total": int(m.group(2)), "msg": line.strip()})
            elif line:
                HUB.push({"type": "log", "msg": f"[calibrate] {line}"})
        proc.wait()
        HUB.push({"type": "calibrate_done", "id": mid, "ok": proc.returncode == 0})


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
            "--doctrine",   str(config.get("doctrine", "mixed")).lower(),
            "--platform",   str(config.get("platform") or "F-16C"),
            "--opponent-platform", str(config.get("opponent_platform") or "F-16C"),
            "--format",     str(config.get("format") or "1v1"),
        ]
        if config.get("format") == "2v1" and config.get("wingman_platform"):
            cmd += ["--wingman-platform", str(config["wingman_platform"])]
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
        # CREATE_NEW_PROCESS_GROUP (Windows only) is what makes it possible to
        # send CTRL_BREAK to the trainer alone in stop() without also signalling
        # this server. On POSIX the default group is already fine.
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True, creationflags=flags)
        self._thread = threading.Thread(target=self._tail, daemon=True)
        self._thread.start()
        HUB.set_status(training=True,
                       msg=f"Training: {config.get('opponent','straight')}")

    # Seconds to let the trainer write its checkpoint before escalating.
    GRACE_TIMEOUT = 60.0

    def stop(self):
        """
        Ask the trainer to shut down CLEANLY so it saves its model.

        train_bvr.py saves in a `finally:` after catching KeyboardInterrupt.
        SIGTERM (what terminate() sends) does NOT run `finally` — Python exits
        immediately — so terminating here silently threw away every step since
        the last archived checkpoint. An interrupt raises KeyboardInterrupt in
        the child instead, which reaches that `finally` and writes the model.
        Escalation to terminate/kill still exists, but only as a last resort
        after the trainer has been given time to save.
        """
        if not (self._proc and self._proc.poll() is None):
            self._stop_evt.set()
            HUB.set_status(training=False, msg="Training stopped")
            return
        self._stop_evt.set()
        # Waiting must not block the asyncio loop that dispatches commands.
        threading.Thread(target=self._graceful_stop, daemon=True).start()

    def stop_blocking(self):
        """stop(), but wait for the trainer to finish saving before returning."""
        if not (self._proc and self._proc.poll() is None):
            return
        self._stop_evt.set()
        self._graceful_stop()

    def _graceful_stop(self):
        proc = self._proc
        HUB.set_status(training=True, msg="Stopping — saving checkpoint…")
        HUB.push({"type": "log", "msg": "[bvr_gui] interrupting trainer; "
                                        "waiting for it to save its model…"})
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
        except Exception as e:
            HUB.push({"type": "log", "msg": f"[bvr_gui] interrupt failed ({e}); terminating"})
            proc.terminate()

        try:
            proc.wait(timeout=self.GRACE_TIMEOUT)
            HUB.push({"type": "log", "msg": "[bvr_gui] trainer exited cleanly — model saved."})
            return
        except subprocess.TimeoutExpired:
            pass

        HUB.push({"type": "log",
                  "msg": f"[bvr_gui] no clean exit after {self.GRACE_TIMEOUT:.0f}s — "
                         f"terminating (checkpoint may be lost)."})
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            HUB.push({"type": "log", "msg": "[bvr_gui] still alive — killing."})
            proc.kill()

    def _tail(self):
        """Read subprocess stdout line by line, push to hub."""
        for line in self._proc.stdout:
            line = line.rstrip()
            if line:
                HUB.push({"type": "log", "msg": line})
        self._proc.wait()
        HUB.set_status(training=False,
                       msg=f"Training ended (exit {self._proc.returncode})")

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
from bvr_selfplay import FrameStacker, N_STACK


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
        HUB.set_status(evaluating=False, msg="Eval stopped")

    def _run_team(self, config, model, privileged, opponent, scen, red_platform):
        from bvr_team import TeamBvrEnv
        from bvr_opponents import BvrOpponentType
        if opponent == BvrOpponentType.SELF_PLAY:
            raise RuntimeError("2v1 has no self-play opponent yet: choose a scripted one")
        blue = (scen["platform"], scen.get("wingman_platform") or scen["platform"])
        env = TeamBvrEnv(opponent_type=opponent, seed=config.get("seed", 0),
                         privileged_critic=privileged, gamma_discount=0.997,
                         doctrine=config.get("doctrine", "BALANCED"),
                         blue_platforms=blue, red_platform=red_platform)
        HUB.push({"type": "log", "msg": f"Eval 2v1: {blue[0]} + {blue[1]} v {red_platform}"
                                        f" (one policy flies both blue aircraft)"})
        HUB.set_status(evaluating=True,
                       msg=f"Eval 2v1 vs {opponent.name}, {env._doctrine_cfg} doctrine")
        _eval_team_loop(self, env, model, privileged, self._stop_evt)

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
                    from gymnasium import spaces
                    from bvr_compat import load_model
                    # Widens checkpoints saved before the latest observation
                    # inputs (e.g. the doctrine input) were added.
                    model = load_model(str(ckpt_path), env=None,
                                       log=lambda m: HUB.push({"type":"log","msg":m}))
                    privileged = isinstance(model.observation_space, spaces.Dict)
                    stacker = FrameStacker(N_STACK)
                    HUB.push({"type":"log","msg":f"Loaded checkpoint: {ckpt_path}"})
                except Exception as e:
                    HUB.push({"type":"log","msg":f"Could not load checkpoint: {e}; using random policy"})
            elif ckpt:
                HUB.push({"type":"log","msg":f"Checkpoint not found: {ckpt_path}; using random policy"})

            # The checkpoint flies the platform it was trained on; the opponent
            # flies the one chosen here, or the checkpoint's training opponent.
            from bvr_library import scenario_of, DEFAULT_PLATFORM
            scen = scenario_of(model) if model is not None else {
                "platform": config.get("platform") or DEFAULT_PLATFORM,
                "opponent_platform": DEFAULT_PLATFORM}
            platform = scen["platform"]
            opp_platform = config.get("opponent_platform") or scen["opponent_platform"]
            fmt = config.get("format") or scen.get("format", "1v1")
            if fmt == "2v1":
                self._run_team(config, model, privileged, opponent, scen, opp_platform)
                return
            if opponent_name == "SELF_PLAY" and platform != opp_platform:
                raise RuntimeError(f"SELF_PLAY needs both sides on one platform; this checkpoint "
                                   f"flies {platform} and the opponent {opp_platform}")
            env = BvrEnv(opponent_type=opponent, seed=config.get("seed", 0),
                         privileged_critic=privileged, gamma_discount=0.997,
                         selfplay_pool=str(Path(self._model_dir) / "selfplay_pool"),
                         doctrine=config.get("doctrine", "BALANCED"),
                         platform=platform, opponent_platform=opp_platform)
            HUB.push({"type": "log", "msg": f"Eval platforms: {platform} v {opp_platform}"})

            HUB.set_status(evaluating=True,
                           msg=f"Eval vs {opponent_name}, {env._doctrine_cfg} doctrine")
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
                        "support_losses": step_info.get("support_losses", 0),
                        "doctrine": step_info.get("doctrine", ""),
                        "bank_rev_per_min": round(60.0 * step_info.get("bank_reversals", 0)
                                                  / max(step_info.get("flight_time", 0.0), 1.0), 1)}
                HUB.set_replay(list(frames), meta)
                HUB.push({"type":"episode_end","meta":meta})
                episode += 1
            env.close()
        except Exception as e:
            import traceback
            HUB.push({"type":"log","msg":f"Eval error: {e}\n{traceback.format_exc()}"})
            HUB.set_status(evaluating=False, msg=f"Eval error: {e}")


def _eval_team_loop(runner, env, model, privileged, stop_evt):
    """Evaluation episodes in 2v1: the checkpoint flies both blue aircraft."""
    episode = 0
    while not stop_evt.is_set():
        obs, info = env.reset()
        stackers = [FrameStacker(N_STACK) for _ in obs] if model is not None else None
        sobs = [st.reset(o) for st, o in zip(stackers, obs)] if stackers else None
        frames, ep_r = [], 0.0
        infos = [{}]
        while not stop_evt.is_set():
            masks = env.action_masks()
            if model is not None:
                acts = [model.predict(sobs[k], deterministic=True, action_masks=masks[k])[0]
                        for k in range(env.n_agents)]
            else:
                acts = []
                for k in range(env.n_agents):
                    a = env.action_space.sample(); a[-1] = 0
                    acts.append(a)
            t0 = time.perf_counter()
            obs, r, term, trunc, infos = env.step(acts)
            if stackers:
                sobs = [st.update(o) for st, o in zip(stackers, obs)]
            step_s = time.perf_counter() - t0
            ep_r += float(np.mean(r))
            frame = _build_team_frame(env, infos)
            frame["ep"] = episode
            frames.append(frame)
            if len(frames) > RECORD_BUFFER_SIZE:
                frames.pop(0)
            HUB.push({"type": "telemetry", "data": frame})
            with runner._speed_lock: spd = runner._speed
            sleep_needed = (1.0 / spd) - step_s
            if sleep_needed > 0.002:
                time.sleep(sleep_needed)
            if term or trunc:
                break
        fin = infos[0]
        meta = {"episode": episode, "outcome": fin.get("terminal_outcome", "?"),
                "reward": round(ep_r, 3), "shots": fin.get("shots_fired", 0),
                "support_losses": fin.get("support_losses", 0),
                "doctrine": fin.get("doctrine", ""), "blue_losses": fin.get("blue_losses", 0),
                "bank_rev_per_min": round(60.0 * fin.get("bank_reversals", 0)
                                          / max(fin.get("flight_time", 0.0), 1.0), 1)}
        HUB.set_replay(list(frames), meta)
        HUB.push({"type": "episode_end", "meta": meta})
        episode += 1
    env.close()


def _pos(la, lo, alt):
    import math
    R=6_371_000.0; D=math.pi/180
    x=(lo-35.0)*D*R*math.cos(39.0*D)
    y=(la-39.0)*D*R
    return [round(x,1), round(y,1), round(alt,1)]


def _build_team_frame(env, infos: list) -> dict:
    """
    2v1 frame: the lead's 1v1 frame (built from fresh truth, so it stays
    right after the lead is lost) plus the wingman and whom red engages.
    """
    w = env._world
    lead, wing = env._obs
    f = _build_frame(w.telemetry(1, env.RED), lead, infos[0])
    sw = w.telemetry(2, env.RED)
    f["fmt"] = "2v1"
    f["own_alive"] = bool(lead._alive)
    f["dl"] = int(lead._alive and lead._dl_used())
    f["red_tgt"] = int(env._red_tgt)
    f["blue_losses"] = len(env._lost)
    f["wing"] = {
        "pos":   _pos(sw.get("lat", 39), sw.get("lon", 35), sw.get("alt", 9000)),
        "psi":   round(sw.get("psi", 0), 4), "theta": round(sw.get("theta", 0), 4),
        "phi":   round(sw.get("phi", 0), 4), "alt": round(sw.get("alt", 9000), 1),
        "speed": round(sw.get("speed", 0), 1), "mach": round(sw.get("mach", 0), 3),
        "nz":    round(sw.get("nz", 1), 2), "wpn": int(sw.get("wpn_remaining", 0)),
        "rng":   round(sw.get("range", 0), 1),
        "alive": bool(wing._alive),
        "track": int(wing._trk_state()) if wing._alive else 0,
        "dl":    int(wing._alive and wing._dl_used()),
    }
    return f


def _build_frame(s: dict, env, info: dict) -> dict:
    """Compact frame for the WebSocket stream — only what the display needs."""

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

    est = env._est()          # the track the aircraft fights on (its own, or its wingman's)
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
        "track":  env._trk_state(),
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
XPLAY_MGR    = None
CALIB_MGR    = None


async def handle_client(ws):
    HUB._clients.add(ws)
    await ws.send(json.dumps(HUB.status_snapshot()))
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
    elif cmd == "start_crossplay":
        XPLAY_MGR.start(msg.get("config", {}))
    elif cmd == "stop_crossplay":
        XPLAY_MGR.stop()
    elif cmd == "calibrate_missile":
        CALIB_MGR.start(str(msg.get("id", "")))
    elif cmd == "set_eval_speed":
        EVAL_RUNNER.set_speed(float(msg.get("speed", 5.0)))
    elif cmd == "ping":
        HUB.push({"type": "pong"})


# ─────────────────────────────────────────────────────────────────────
def run_http(port: int):
    os.chdir(HERE)
    # Threaded: a performance card takes seconds and must not stall metrics polling.
    srv = ThreadingHTTPServer(("0.0.0.0", port), GUIHandler)
    srv.serve_forever()


async def run_ws(port: int):
    import websockets
    async with websockets.serve(handle_client, "0.0.0.0", port,
                                 max_size=4*1024*1024):
        await metrics_poll_loop()   # runs forever alongside WS server


def main():
    global TRAINING_MGR, EVAL_RUNNER, XPLAY_MGR, CALIB_MGR, MODEL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",      type=int, default=5006)
    ap.add_argument("--ws-port",   type=int, default=5010)
    ap.add_argument("--model-dir", type=str, default="models_bvr")
    args = ap.parse_args()

    MODEL_DIR    = HERE / args.model_dir
    TRAINING_MGR = TrainingManager(args.model_dir)
    EVAL_RUNNER  = EvalRunner(args.model_dir)
    XPLAY_MGR    = CrossplayManager(args.model_dir)
    CALIB_MGR    = CalibrationManager()

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
        EVAL_RUNNER.stop()
        if XPLAY_MGR.running:
            XPLAY_MGR._stop()
        # Block here: stop() hands the wait to a daemon thread, and daemon
        # threads die the moment main() returns — which would cut the trainer
        # off mid-save, the exact data loss this path exists to prevent.
        if TRAINING_MGR.running:
            print("[bvr_gui]  stopping trainer — waiting for it to save…")
            TRAINING_MGR.stop_blocking()
        print("[bvr_gui]  bye")


if __name__ == "__main__":
    main()
