"""PossibleClips: per-sample mixing of a second dataset, and no change to the unmixed stream."""
import io
import itertools
import os

import numpy as np
import pytest
import webdataset as wds

from jepatrain.data import PossibleClips


def _shards(root, tag, n_shards, per, value):
    d = os.path.join(root, "train", "possible")
    os.makedirs(d, exist_ok=True)
    for k in range(n_shards):
        with wds.TarWriter(os.path.join(d, f"{tag}-{k:03d}.tar")) as w:
            for i in range(per):
                buf = io.BytesIO()
                np.savez_compressed(buf, frames=np.full((4, 8, 8, 3), value, np.uint8))
                w.write({"__key__": f"{tag}-{k}-{i}", "clip.npz": buf.getvalue()})
    return root


def test_mix_fraction(tmp_path):
    main = _shards(str(tmp_path / "main"), "m", 2, 50, 10)
    mix = _shards(str(tmp_path / "mix"), "x", 1, 20, 200)
    ds = PossibleClips(main, set(), 8, buffer=16, mix_data=mix, mix_frac=0.5)
    vals = [int(c[0, 0, 0, 0]) for c in itertools.islice(iter(ds), 2000)]
    assert set(vals) == {10, 200}
    frac = vals.count(200) / len(vals)
    assert 0.45 < frac < 0.55


def test_unmixed_stream_unchanged(tmp_path):
    main = _shards(str(tmp_path / "main"), "m", 3, 30, 7)
    a = [c.sum().item() for c in itertools.islice(iter(PossibleClips(main, set(), 8, buffer=8)), 200)]
    assert len(a) == 200 and set(a) == {7 * 4 * 8 * 8 * 3}


def test_mix_args_validated(tmp_path):
    main = _shards(str(tmp_path / "main"), "m", 1, 5, 1)
    with pytest.raises(ValueError):
        PossibleClips(main, set(), 8, mix_frac=0.3)
    with pytest.raises(ValueError):
        PossibleClips(main, set(), 8, mix_data=main, mix_frac=0.0)
    with pytest.raises(FileNotFoundError):
        PossibleClips(main, set(), 8, mix_data=str(tmp_path / "nope"), mix_frac=0.5)
