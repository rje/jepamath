"""GPU compute guard for model workloads (inference, later training).

The GPU on this machine has been unstable, so compute never runs flat out:
  * duty cycle: after each guarded step, sleep so that GPU-busy time is at most `duty` of wall time;
  * thermal pause: above `temp_ceiling_c`, stop until the GPU cools by `temp_resume_drop_c`;
  * power ceiling: if board power exceeds `power_ceiling_w` at a check, raise the sleep factor.
"""

from __future__ import annotations

import subprocess
import sys
import time

import torch

from jepagen.limiter import gpu_temperature_c


def gpu_power_w() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                         check=True, capture_output=True, text=True).stdout
    return max(float(x) for x in out.split())


class ComputeGuard:
    def __init__(self, duty: float = 0.5, temp_ceiling_c: float = 83.0, temp_resume_drop_c: float = 10.0,
                 power_ceiling_w: float = 200.0, check_s: float = 10.0, log=None):
        if not 0 < duty <= 1:
            raise ValueError("duty must be in (0, 1]")
        self.duty = duty
        self.temp_ceiling = temp_ceiling_c
        self.temp_resume = temp_ceiling_c - temp_resume_drop_c
        self.power_ceiling = power_ceiling_w
        self.check_s = check_s
        self._last_check = 0.0
        self._t0 = 0.0
        self.busy_s = 0.0
        self.sleep_s = 0.0
        self.max_temp = 0.0
        self.max_power = 0.0
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))

    def __enter__(self):
        self._check()
        torch.cuda.synchronize()
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        busy = time.monotonic() - self._t0
        self.busy_s += busy
        rest = busy * (1.0 / self.duty - 1.0)
        if rest > 0:
            time.sleep(rest)
            self.sleep_s += rest
        return False

    def _check(self):
        now = time.monotonic()
        if now - self._last_check < self.check_s:
            return
        self._last_check = now
        temp, power = gpu_temperature_c(), gpu_power_w()
        self.max_temp, self.max_power = max(self.max_temp, temp), max(self.max_power, power)
        if power > self.power_ceiling:
            self.duty = max(0.1, self.duty * 0.8)
            self.log(f"[guard] {power:.0f} W > {self.power_ceiling:.0f} W; duty lowered to {self.duty:.2f}")
        if temp >= self.temp_ceiling:
            self.log(f"[guard] GPU {temp:.0f}C >= {self.temp_ceiling:.0f}C; pausing until < {self.temp_resume:.0f}C")
            while temp >= self.temp_resume:
                time.sleep(5)
                temp = gpu_temperature_c()
            self.log(f"[guard] GPU {temp:.0f}C; resuming")

    def summary(self) -> dict:
        return {"duty": self.duty, "busy_s": round(self.busy_s, 1), "sleep_s": round(self.sleep_s, 1),
                "max_temp_c": self.max_temp, "max_power_w": self.max_power}
