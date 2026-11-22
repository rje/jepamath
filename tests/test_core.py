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


def test_batched_render_matches_unbatched(tmp_path):
    """A clip rendered inside a batch must be pixel-identical to the same clip rendered alone."""
    from jepagen.run import run_episodes_batched

    lim = RenderLimiter(str(tmp_path / "lim"), max_fps=1000)
    t, m = Template(), Manifest()
    eps = [sample_episode(t, m, "train", s) for s in range(3)]
    eps += [make_twin(e, s) for e, s in zip(eps, range(3))]
    batched = run_episodes_batched(eps, lim)
    for ep, (bclip, bdiag) in zip(eps, batched):
        clip, diag = run_episode(ep, "rust", lim)
        assert np.array_equal(clip.frames, bclip.frames), ep.episode_id
        assert np.array_equal(clip.seg, bclip.seg), ep.episode_id
        assert diag["max_penetration"] == bdiag["max_penetration"]
        assert clip.states == bclip.states


def test_control_twins_keep_count(tmp_path):
    """move/swap twins: identical before the change, same final count, a visible difference at the end."""
    from jepagen.sampler import can_control

    lim = RenderLimiter(str(tmp_path / "lim"), max_fps=1000)
    t, m = Template(), Manifest()
    seen = set()
    for seed in range(40):
        ep = sample_episode(t, m, "train", seed)
        if not can_control(ep):
            continue
        tw = make_twin(ep, seed, "control")
        if tw.violation.kind in seen:
            continue
        seen.add(tw.violation.kind)
        assert tw.twin_kind == "control" and tw.meta["shown_final"] == ep.meta["final"]
        a, da = run_episode(ep, "rust", lim)
        b, db = run_episode(tw, "rust", lim)
        va, vb, vt = validate_clip(a, da), validate_clip(b, db), validate_twins(a, b)
        assert va.ok and vb.ok and vt.ok, (seed, tw.violation.kind, va.reasons, vb.reasons, vt.reasons)
        assert b.states[-1].count_per_region["box"] == a.states[-1].count_per_region["box"]
        if seen == {"move", "swap"}:
            return
    raise AssertionError(f"only saw {seen} in 40 seeds")


def test_trackb_masks_and_prefix_encoding():
    import random
    import torch
    from jepatrain.model import Encoder, sincos_3d
    from jepatrain.train import sample_masks
    rng = random.Random(0)
    for _ in range(50):
        kind, ctx, tgt = sample_masks(rng, 8, 16, 0.5)
        assert len(ctx) > 0 and len(tgt) > 0
        assert set(ctx.tolist()).isdisjoint(tgt.tolist()) and len(ctx) + len(tgt) == 8 * 256
        if kind == "prefix":
            assert ctx.max() < tgt.min()
    assert sincos_3d(8, 16, 16, 384).shape == (2048, 384)
    enc = Encoder(dim=64, depth=1, heads=2)
    x = torch.randn(1, 3, 32, 128, 128)
    with torch.no_grad():
        a = enc(x[:, :, :8])
        b = enc(x, keep=torch.arange(512)[None])
    assert torch.allclose(a, b, atol=1e-5)


def test_default_template_reproduces_v2():
    # v2's seed 0 (from its episode metadata): the occluded-only default draws nothing new
    ep = sample_episode(Template(), Manifest(), "train", 0)
    assert ep.meta["visibility"] == "occluded"
    assert ep.meta["scene"]["cam_distance"] == pytest.approx(1.0277857194974376, abs=1e-12)


def test_visibility_modes():
    from jepagen.spec import ScreenMove, Transfer
    m = Manifest()
    seen = set()
    for vis in ("visible", "mixed"):
        t = Template(visibility_modes=("occluded", vis), visibility_weights=(0.2, 0.8))
        for seed in range(40):
            ep = sample_episode(t, m, "train", seed)
            v = ep.meta["visibility"]
            seen.add(v)
            acts = sorted(ep.actions, key=lambda a: a.t_start)
            screen = [a.kind for a in acts if isinstance(a, ScreenMove)]
            transfers = [a for a in acts if isinstance(a, Transfer)]
            if v == "visible":
                assert screen == []
                assert not any(ep.box_occluded_at(a.t_end) for a in transfers)
                with pytest.raises(ValueError, match="only of occluded"):
                    make_twin(ep, seed)
            elif v == "mixed":
                assert screen == ["screen_raise", "screen_lower"]
                assert not ep.box_occluded_at(transfers[0].t_end)
                assert all(ep.box_occluded_at(a.t_end) for a in transfers[1:])
            else:
                assert screen == ["screen_raise", "screen_lower"]
    assert {"visible", "mixed", "occluded"} <= seen


def test_mixed_alone_rejected():
    with pytest.raises(ValueError, match="mixed alone"):
        Template(visibility_modes=("mixed",), visibility_weights=(1.0,))


def test_packed_triple_roundtrip():
    from jepatrain.data import PackedTriple, heldout_res, pack
    f = np.random.default_rng(0).integers(0, 255, (4, 16, 16, 3), dtype=np.uint8)
    t = PackedTriple({"possible": pack(f), "impossible": pack(f[::-1]), "control": pack(f), "seed_id": "x"})
    assert np.array_equal(t["possible"], f) and np.array_equal(t["impossible"], f[::-1])
    assert t["seed_id"] == "x" and heldout_res([t]) == 16


def test_encoder_256_tokens():
    import torch
    from jepatrain.model import Encoder
    e = Encoder(img=256, patch=8, frames=8, tubelet=4, dim=64, depth=1, heads=2)
    assert e.tokens_per_slot == 32 * 32
    assert e(torch.zeros(1, 3, 8, 256, 256)).shape == (1, 2 * 1024, 64)


def test_change_mask():
    import torch
    from jepatrain.train import change_mask
    clips = torch.zeros(1, 8, 32, 32, 3, dtype=torch.uint8)
    clips[0, 6:, 8:16, 16:24] = 200          # a patch-sized object appears in frame 6 at patch (1, 2)
    m = change_mask(clips, ctx_frames=4, tubelet=4, patch=8, thresh=10)
    assert m.shape == (1, 16)                # one target slot x 4x4 patches
    assert m[0].nonzero().flatten().tolist() == [1 * 4 + 2]
    with pytest.raises(ValueError):
        change_mask(clips, ctx_frames=3, tubelet=4, patch=8, thresh=10)
