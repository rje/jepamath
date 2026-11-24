"""Vast.ai API: search and rank offers, rent, list, destroy (verified).

API quirks learned the hard way in microlab:
  * instance LIST is v1 (/api/v0/instances/ answers 410), but DESTROY is still v0: a v1 DELETE answers 404
    while the instance keeps billing. So destroy() confirms against the list instead of trusting the response.
  * sending `price` is what makes a contract interruptible; omitting it rents on-demand.
  * a preempted instance stays listed with actual_status "exited" and still bills for disk; Vast may resume it
    by itself later. It has to be destroyed, not ignored.
  * the `verified` server-side filter is real even though offers read back verified=None.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from .config import Machine

API_V0 = "https://console.vast.ai/api/v0"
API_V1 = "https://console.vast.ai/api/v1"
# container states that can still be making progress; anything else (notably "exited") is a dead box
RUNNING_STATES = frozenset({"running", "loading", "created", "starting"})


class VastError(RuntimeError):
    pass


class Vast:
    def __init__(self, key: str):
        self._key = key

    def __repr__(self):
        return "Vast(<key>)"

    def call(self, method: str, path: str, body: dict | None = None, base: str = API_V0):
        req = urllib.request.Request(
            f"{base}{path}", method=method, data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode()
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            raise VastError(f"{method} {path} -> {e.code}: {e.read().decode()[:400]}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            # network failure (DNS, timeout, reset): a VastError, so the supervisor's poll loop retries it and its
            # cleanup still runs (24 Nov: an unwrapped URLError killed the supervisor and left a box running 12 h)
            raise VastError(f"{method} {path} -> network error: {e}") from e

    def search(self, m: Machine, gpu: str) -> list[dict]:
        q = {"gpu_name": {"eq": gpu}, "num_gpus": {"eq": m.num_gpus}, "rentable": {"eq": True},
             **({"verified": {"eq": True}} if m.verified_only else {}),
             "order": [["dph_total", "asc"]], "limit": 300}
        return self.call("GET", f"/bundles/?q={urllib.parse.quote(json.dumps(q))}").get("offers", [])

    def search_all(self, m: Machine) -> list[dict]:
        out = []
        for g in m.gpus:
            out += self.search(m, g)
        return out

    def rent(self, offer_id: int, body: dict) -> int:
        r = self.call("PUT", f"/asks/{offer_id}/", body)
        inst = r.get("new_contract")
        if not inst:
            raise VastError(f"create failed: {r}")
        return int(inst)

    def instances(self) -> list[dict]:
        r = self.call("GET", "/instances/", base=API_V1)
        return r.get("instances", []) if isinstance(r, dict) else list(r or [])

    def status(self, inst: int) -> str | None:
        """actual_status of the instance, or None if it is not listed."""
        return self.poll(inst)[0]

    def poll(self, inst: int) -> tuple[str | None, float | None, float]:
        """(actual_status, billed $/h, bandwidth $ so far) of the instance; (None, None, 0) if not listed.
        `dph_total` is the GPU price plus disk; bandwidth is billed separately per GB moved."""
        for i in self.instances():
            if i.get("id") == inst:
                dph = i.get("dph_total")
                return i.get("actual_status") or "unknown", float(dph) if dph is not None else None, bandwidth_usd(i)
        return None, None, 0.0

    def destroy(self, inst: int, wait_s: float = 60) -> None:
        """Destroy, then confirm it is gone from the list. Raises if it is still listed: it is billing."""
        try:
            self.call("DELETE", f"/instances/{inst}/", base=API_V0)
        except VastError as e:
            if "no_such_instance" not in str(e):     # already gone is the goal state
                raise
        t0 = time.time()
        while time.time() - t0 < wait_s:
            time.sleep(5)
            if self.status(inst) is None:
                return
        raise VastError(f"instance {inst} STILL LISTED after destroy: it is billing. "
                        f"Kill it at https://cloud.vast.ai/instances/ now")


def eligible(offers: list[dict], m: Machine) -> list[dict]:
    """Hardware filters applied locally (field names as Vast returns them; RAM fields are MB)."""
    ok = []
    for o in offers:
        if o.get("num_gpus") != m.num_gpus or o.get("gpu_name") not in m.gpus:
            continue
        if (o.get("reliability2") or 0) < m.min_reliability:
            continue
        if (o.get("disk_space") or 0) < m.min_disk_gb:
            continue
        if (o.get("gpu_ram") or 0) / 1024 < m.min_gpu_ram_gb:
            continue
        if (o.get("cpu_cores_effective") or 0) < m.min_cpu_cores:
            continue
        if (o.get("cpu_ram") or 0) / 1024 < m.min_ram_gb:
            continue
        if (o.get("inet_down") or 0) < m.min_inet_down_mbps:
            continue
        if (o.get("cuda_max_good") or 0) < m.min_cuda:
            continue
        ok.append(o)
    return ok


def planned_rate(o: dict, m: Machine) -> tuple[float, bool] | None:
    """(the $/h we would actually be charged, on_demand) for one offer, or None if over the cap.

    Interruptible: bid = floor * bid_multiplier, capped at the ceiling. On-demand is taken instead when it
    costs at most 10% more than that bid, because a preemption costs a re-setup plus lost steps."""
    cap = m.max_price_per_gpu * m.num_gpus
    od = o.get("dph_total")
    if not m.interruptible:
        return (od, True) if od and od <= cap else None
    floor = o.get("min_bid")
    if not floor or floor > cap:
        return (od, True) if od and od <= cap else None
    bid = round(min(floor * m.bid_multiplier, cap), 4)
    if od and od <= cap and od <= bid * 1.10:
        return od, True
    return bid, False


def bandwidth_usd(inst: dict) -> float:
    """Bandwidth billed so far on an instance. `inet_*_billed` count in units of 1e-6 GB, judged by the
    account credit matching ~28 GB pulled per box at the listed $/GB."""
    return sum((inst.get(f"inet_{d}_billed") or 0) / 1e6 * (inst.get(f"inet_{d}_cost") or 0) for d in ("down", "up"))


def episode_cost(rate: float, o: dict, m: Machine, down_gb: float) -> float:
    """Expected $ for one episode on this offer: GPU hours plus data in and checkpoints out."""
    return (rate * m.episode_hours + down_gb * (o.get("inet_down_cost") or 0)
            + m.upload_gb * (o.get("inet_up_cost") or 0))


def rank(offers: list[dict], m: Machine, down_gb: float = 0.0) -> list[tuple[float, bool, dict]]:
    """Eligible offers as (rate, on_demand, offer), cheapest expected episode first (see episode_cost)."""
    out = []
    for o in eligible(offers, m):
        r = planned_rate(o, m)
        if r is not None:
            out.append((r[0], r[1], o))
    out.sort(key=lambda t: episode_cost(t[0], t[2], m, down_gb))
    return out


def fmt_offers(ranked: list[tuple[float, bool, dict]], n: int = 12, m: Machine | None = None,
               down_gb: float = 0.0) -> str:
    lines = [f"{'offer':>10} {'gpu':<14} {'rate $/h':>8} {'kind':<6} {'$/GB':>6} {'ep $':>6} {'floor':>6} {'od':>6} "
             f"{'rel':>5} {'cpu':>4} {'ramG':>5} {'down':>6} {'cuda':>5}  location"]
    for rate, od, o in ranked[:n]:
        ep = episode_cost(rate, o, m, down_gb) if m else float("nan")
        lines.append(f"{o['id']:>10} {o.get('gpu_name', '?')[:14]:<14} {rate:>8.3f} {'od' if od else 'bid':<6} "
                     f"{o.get('inet_down_cost') or 0:>6.4f} {ep:>6.2f} "
                     f"{o.get('min_bid') or 0:>6.3f} {o.get('dph_total') or 0:>6.3f} {o.get('reliability2', 0):>5.3f} "
                     f"{o.get('cpu_cores_effective', 0):>4.0f} {(o.get('cpu_ram') or 0) / 1024:>5.0f} "
                     f"{o.get('inet_down', 0):>6.0f} {o.get('cuda_max_good', 0):>5} "
                     f" {str(o.get('geolocation'))[:24]}")
    return "\n".join(lines)
