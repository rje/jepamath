"""Global GPU render-rate cap shared by every worker process, plus a thermal pause.

The cap is on TOTAL renders per second across all processes that point at the same
state file. Each render call takes one slot; slots are spaced 1/max_fps apart.
"""

from __future__ import annotations

import fcntl
import os
import struct
import subprocess
import sys
import time


class RenderLimiter:
    def __init__(self, state_path: str, max_fps: float, temp_ceiling_c: float = 83.0,
                 temp_resume_drop_c: float = 10.0, temp_check_s: float = 30.0, log=None):
        if max_fps <= 0:
            raise ValueError("max_fps must be > 0")
        self.path = state_path
        self.interval = 1.0 / max_fps
        self.temp_ceiling = temp_ceiling_c
        self.temp_resume = temp_ceiling_c - temp_resume_drop_c
        self.temp_check_s = temp_check_s
        self._last_temp_check = 0.0
        self.log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
        os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
        fd = os.open(state_path, os.O_RDWR | os.O_CREAT, 0o644)
        os.close(fd)

    def acquire(self) -> None:
        self._thermal_gate()
        with open(self.path, "r+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                raw = f.read(8)
                next_slot = struct.unpack("d", raw)[0] if len(raw) == 8 else 0.0
                now = time.monotonic()
                # monotonic clocks are system-wide on Linux, so slots are comparable across processes
                if next_slot > now + 60:  # stale file from before a reboot
                    next_slot = now
                wait = next_slot - now
                if wait > 0:
                    time.sleep(wait)
                    now = next_slot
                f.seek(0)
                f.write(struct.pack("d", max(now, next_slot) + self.interval))
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _thermal_gate(self) -> None:
        now = time.monotonic()
        if now - self._last_temp_check < self.temp_check_s:
            return
        self._last_temp_check = now
        temp = gpu_temperature_c()
        if temp < self.temp_ceiling:
            return
        self.log(f"[limiter] GPU at {temp:.0f}C >= {self.temp_ceiling:.0f}C; pausing until < {self.temp_resume:.0f}C")
        while temp >= self.temp_resume:
            time.sleep(5)
            temp = gpu_temperature_c()
        self.log(f"[limiter] GPU at {temp:.0f}C; resuming")


def gpu_temperature_c() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                         check=True, capture_output=True, text=True).stdout
    return max(float(x) for x in out.split())


def gpu_status() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used",
                           "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
