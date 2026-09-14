"""WhisperLive benchmark: a faster_whisper server fed by streaming clients.

    python ao_bench.py                     # 8 clips through one server, small model
    python ao_bench.py --units 4 --model tiny

Starts `run_server.py --backend faster_whisper` in this repo as a subprocess,
then for each unit opens a websocket session exactly as whisper_live.client
does (same handshake JSON, float32 PCM at 16 kHz in 4096-sample packets,
`END_OF_AUDIO` at the end) but WITHOUT pacing the packets to real time: the
server's buffering, chunking and transcription loop is what is measured, from
the first packet to the transcript that covers the clip. The units are
speed-perturbed copies of assets/jfk.flac (11 s), so no two are identical.
The loop over units is the unit of work. Writes out/summary.json with each
unit's wall time and transcript.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import soundfile as sf
import websocket
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET = os.path.join(HERE, "assets", "jfk.flac")
RATE = 16000
CHUNK = 4096
SPEEDS = [1.0, 0.9, 1.1, 0.95, 1.05, 0.85, 1.15, 0.8]


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != RATE:
        audio = resample_poly(audio, RATE, sr).astype(np.float32)
    return audio


def make_units(audio, n):
    out = []
    for i in range(n):
        speed = SPEEDS[i % len(SPEEDS)]
        p, q = int(round(100 / speed)), 100
        out.append(resample_poly(audio, p, q).astype(np.float32) if speed != 1.0 else audio.copy())
    return out


def wait_port(port, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            s.settimeout(1.0)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


class Session:
    """One websocket session, driven the way whisper_live.client.Client drives it."""

    def __init__(self, port, model, language):
        self.uid = str(uuid.uuid4())
        self.model, self.language = model, language
        self.ready = threading.Event()
        self.closed = threading.Event()
        self.segments = {}
        self.last_msg = 0.0
        self.error = None
        self.ws = websocket.WebSocketApp(
            f"ws://127.0.0.1:{port}", on_open=self._on_open, on_message=self._on_message,
            on_error=self._on_error, on_close=lambda *_: self.closed.set())
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def _on_open(self, ws):
        ws.send(json.dumps({
            "uid": self.uid, "language": self.language, "task": "transcribe", "model": self.model,
            "use_vad": False, "send_last_n_segments": 10, "no_speech_thresh": 0.45,
            "clip_audio": False, "same_output_threshold": 10, "enable_translation": False,
            "target_language": None, "hotwords": None, "enable_diarization": False,
            "max_speakers": None, "known_speakers": [], "word_timestamps": False,
            "initial_prompt": None, "vad_parameters": None}))

    def _on_message(self, ws, message):
        m = json.loads(message)
        if m.get("uid") != self.uid:
            return
        self.last_msg = time.time()
        if m.get("status") == "ERROR":
            self.error = m.get("message")
        if m.get("message") == "SERVER_READY":
            self.ready.set()
        for seg in m.get("segments") or []:
            self.segments[(round(float(seg["start"]), 2))] = seg

    def _on_error(self, ws, err):
        self.error = str(err)
        self.ready.set()

    def covered(self):
        return max((float(s["end"]) for s in self.segments.values()), default=0.0)

    def transcribe(self, audio, timeout=120.0):
        if not self.ready.wait(60) or self.error:
            raise RuntimeError(f"server not ready: {self.error}")
        duration = len(audio) / RATE
        t0 = time.perf_counter()
        for i in range(0, len(audio), CHUNK):
            self.ws.send(audio[i:i + CHUNK].tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
        self.ws.send(b"END_OF_AUDIO", opcode=websocket.ABNF.OPCODE_BINARY)
        # Done when the transcript reaches the end of the clip, or when the
        # server has gone quiet for 3 s after saying anything at all.
        while time.perf_counter() - t0 < timeout:
            if self.error:
                raise RuntimeError(self.error)
            if self.covered() >= duration - 0.5:
                break
            if self.segments and time.time() - self.last_msg > 3.0:
                break
            time.sleep(0.02)
        dt = time.perf_counter() - t0
        text = " ".join(s["text"].strip() for _, s in sorted(self.segments.items()))
        try:
            self.ws.close()
        except Exception:                                         # noqa: BLE001
            pass
        return dt, text, self.covered(), duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=int, default=8)
    ap.add_argument("--model", default="small")
    ap.add_argument("--language", default="en")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    audio = load_audio(ASSET)
    units = make_units(audio, args.units)
    print(f"audio: {ASSET} {len(audio) / RATE:.1f}s -> {len(units)} units "
          f"({', '.join(f'{len(u) / RATE:.1f}s' for u in units)})", flush=True)

    log = open(os.path.join(args.out, "server.log"), "w")
    server = subprocess.Popen(
        [sys.executable, "run_server.py", "--port", str(args.port), "--backend", "faster_whisper",
         "--max_clients", "4", "--max_connection_time", "600"],
        cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_port(args.port, 300):
            raise RuntimeError("server did not open its port; see out/server.log")
        # The first session downloads/loads the model: warm it outside the loop.
        t0 = time.perf_counter()
        Session(args.port, args.model, args.language).transcribe(units[0][:RATE * 2])
        print(f"server: up, model {args.model} warm ({time.perf_counter() - t0:.1f}s)", flush=True)

        rows, walls = [], []
        for i, unit in enumerate(units):
            dt, text, covered, duration = Session(args.port, args.model, args.language).transcribe(unit)
            walls.append(dt)
            rows.append({"unit": i, "s": round(dt, 4), "audio_s": round(duration, 2),
                         "covered_s": round(covered, 2), "text": text})
            print(f"  unit {i} {dt:.3f}s for {duration:.1f}s of audio (x{duration / dt:.1f} realtime): "
                  f"{text[:70]}", flush=True)
    finally:
        server.terminate()
        try:
            server.wait(20)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"units": rows, "total_s": sum(walls), "median_s": float(np.median(walls)),
                   "model": args.model}, fh, indent=1)
    print(f"done: {len(walls)} units, median {np.median(walls):.3f}s, total {sum(walls):.2f}s")


if __name__ == "__main__":
    main()
