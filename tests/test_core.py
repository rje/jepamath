import os
import time

import numpy as np
import pytest

from jepagen.limiter import RenderLimiter
from jepagen.run import run_episode
from jepagen.sampler import Manifest, Template, make_twin, sample_episode
from jepagen.spec import Episode, Violation
from jepagen.validate import validate_clip, validate_twins


class NoLimit:
    def acquire(self):
        pass


def test_sampler_deterministic():
    t, m = Template(), Manifest()
    assert sample_episode(t, m, "train", 42) == sample_episode(t, m, "train", 42)
    assert sample_episode(t, m, "train", 42) != sample_episode(t, m, "train", 43)


def test_spec_roundtrip():
    ep = sample_episode(Template(), Manifest(), "train", 3)
    assert Episode.from_json(ep.to_json()) == ep


def test_splits_respect_manifest():
    t, m = Template(count_range=(0, 15), final_range=(0, 15)), Manifest()
    for s in range(200):
        ep = sample_episode(t, m, "train", s)
        mt = ep.meta
        assert mt["n0"] not in m.heldout_counts and mt["final"] not in m.heldout_counts
        assert (mt["n0"], mt["op"], mt["k"]) not in m.heldout_combos
    for s in range(50):
        mt = sample_episode(t, m, "test_counts", s).meta
        assert mt["n0"] in m.heldout_counts or mt["final"] in m.heldout_counts


def test_violation_must_be_hidden():
    ep = sample_episode(Template(), Manifest(), "train", 5)
    bad = ep.model_dump()
    bad["violation"] = Violation(kind="delete", t=0.01, object_id=None, pos=None).model_dump()
    with pytest.raises(ValueError, match="visible"):
        Episode.model_validate(bad)


def test_twin_pair_identical_until_violation():
    ep = sample_episode(Template(), Manifest(), "train", 2)
    tw = make_twin(ep, 2)
    a, da = run_episode(ep, "mujoco", NoLimit())
    b, db = run_episode(tw, "mujoco", NoLimit())
    assert validate_clip(a, da).ok, validate_clip(a, da).reasons
    assert validate_clip(b, db).ok, validate_clip(b, db).reasons
    assert validate_twins(a, b).ok
    assert a.states[-1].count_per_region["box"] != b.states[-1].count_per_region["box"]
    assert not np.array_equal(a.frames[-1], b.frames[-1])


def test_backend_deterministic():
    ep = sample_episode(Template(), Manifest(), "train", 9)
    a, _ = run_episode(ep, "mujoco", NoLimit())
    b, _ = run_episode(ep, "mujoco", NoLimit())
    assert np.array_equal(a.frames, b.frames)


def test_limiter_caps_rate(tmp_path):
    lim = RenderLimiter(str(tmp_path / "lim"), max_fps=50, temp_check_s=1e9)
    t0 = time.monotonic()
    for _ in range(26):
        lim.acquire()
    assert time.monotonic() - t0 >= 0.48
