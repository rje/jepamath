# JEPA Number Sense — Data Generator Design

Date: 2026-11-19
Source: handoff doc "JEPA Number Sense — Project Handoff" (claude.ai/artifact/CnN5PDVCEeY1CQDWBANxhd), plus decisions in this session.

## Intent

The project tests whether a JEPA world model trained only on procedurally generated video acquires number sense and arithmetic without labels. Every result depends on the data: if clips leak the count through shortcuts, or physics glitches look like violations, the experiment says nothing. This design covers the **data generator** — the tool that turns a declarative episode spec into validated, leak-audited video shards with full per-frame ground truth.

The generator has two simulation backends behind one interface:

- **MuJoCo (Python)** — fast to stand up; the reference backend and v1 data source.
- **Own engine (Rust)** — decided by the user: real physics, and a structure where new systems/mechanics are easy to add as we want more variance (rolling, stacking, containment, tools, later glyphs).

## Goals

1. One episode spec format; both backends consume it and produce the same output format.
2. First episode type: add/remove k objects into a box with a visible claw, 128px flat shading, per-frame state logging.
3. Wynn-style possible/impossible pairs from shared seeds (IntPhys design: identical up to the hidden violation).
4. Validator that rejects bad clips; automated leak audit over a dataset sample.
5. Deterministic: (spec, seed, backend, version) reproduces any clip bit-for-bit within a backend.
6. Rust engine where a mechanic is a self-contained module added without touching the core.
7. GPU rendering throughput is capped (user requirement: GPU has been unstable).

## Non-goals (this design)

- JEPA training, Track A smoke test, Garrido reproduction (handoff tasks 7–8) — separate design once data exists.
- Track A 256px rich render pass (textures, PBR, random backgrounds) — the render-pass interface must allow it, but it is not built now.
- Numeral glyphs (phase 4).
- Matching physics trajectories exactly across backends. Backends must agree on *semantics* (counts per region over time, action outcomes), not positions.

## Anti-goals (would make this a failure even if it runs)

- Objects spawning or teleporting into view. All objects enter and leave via the visible claw/chute or are present from frame 0. (Exception: the impossible twin's hidden add/delete while occluded — that is the point.)
- Impossible clips reaching any training shard.
- Lossy video compression (H.264 etc.) — blurs small objects together.
- Contact jitter, interpenetration or escapes in accepted clips (they pollute surprise).
- Action timing, occluder timing or any rendering artifact correlating with the answer.
- Uncapped GPU usage.
- Silent fallbacks: if a backend, renderer, or spec field fails, raise. No defaulting to another backend or skipping a field.

## Constraints

- Machine: RTX 6000 Ada (48 GB), 48 CPU cores, 91 GB RAM, headless Vulkan 1.4 available, Rust 1.99, conda.
- Disk: 435 GB free on /home. 1M clips × 16 × 128² × 3 B ≈ 790 GB raw — does not fit. Default target is **100k clips (~79 GB raw, less as PNG)**; the shard writer checks free space before writing and refuses if the run would leave < 50 GB free.
- Python env: a new conda env `jepamath` (Python 3.11), leaving `base` untouched. PyTorch is not needed by the generator; it is installed later for training.

## Approaches considered (Rust engine)

| Approach | Trade-off | Decision |
|---|---|---|
| **A. Small engine on crates: rapier3d physics + wgpu offscreen rendering, mechanics as modules** | We write rendering and glue (~2–3k lines to start); full control of determinism, twins, frame pacing | **Chosen** by the user |
| B. Bevy, headless | Built-in systems model and plugins, but heavy, and headless deterministic frame-stepping fights its main loop | Rejected |
| C. Own physics from scratch | Total control, but stable contacts take months; jitter would pollute surprise | Rejected |

MuJoCo stays as the reference backend used to cross-check the Rust engine.

## Architecture

Six stages, one spec (from the handoff):

```
Episode spec ─► Sampler ─► Simulator ─► Renderer ─► Validator ─► Shard writer
 (template)     (seeded     (MuJoCo |    (flat 128px  (per-clip     (WebDataset tar:
                concrete    Rust)        now; rich    checks;       PNG frames +
                episode)                 later)       dataset leak  JSON state)
                                                      audit)
```

### Repository layout

```
jepamath/
  spec/            JSON Schema for the concrete episode (generated from the Python models)
  gen/             Python package `jepagen`
    spec.py        Pydantic models: template + concrete episode
    sampler.py     template + seed + split manifest -> concrete episode
    backends/      Backend protocol; mujoco_backend.py; rust_backend.py (thin wrapper over the PyO3 module)
    render/        MuJoCo renderer (EGL); frame limiter
    validate.py    per-clip checks
    audit.py       leak audit
    shards.py      WebDataset writer + index
    cli.py         `jepagen generate|validate|audit|preview|crosscheck`
  engine/          Rust workspace
    core/          world, rapier wrapper, Mechanic trait, state snapshot/restore
    mechanics/     claw, chute, screen, container (one module each)
    render/        wgpu offscreen renderer + frame limiter
    py/            PyO3 bindings, built with maturin
  tests/           pytest (Python) ; cargo test (Rust)
  docs/
```

### Episode spec

Two levels:

- **Template** (YAML, human-authored): distributions — count range, k range, object type pool, size/color ranges, camera jitter, action timing jitter, control mode (area-matched, size-matched, density-matched…).
- **Concrete episode** (JSON, produced by the sampler, consumed by backends): every value fixed. Fields:
  - `seed`, `episode_type`, `split`, `spec_version`
  - `scene`: table, containers (shape, pose, size), screen (pose, size), camera (pose, fov), lights
  - `objects`: id, shape (sphere, cube, cylinder, capsule to start), size, color, initial pose / initial location (on table, in container, in claw hopper)
  - `script`: timed actions — `screen_down(t)`, `screen_up(t)`, `add(k, t)`, `remove(k, t)` — times in seconds
  - `violation` (impossible twins only): `{kind: add|delete, object_id, at_t}`, `at_t` required to fall while the screen fully occludes the container
  - `clip`: `n_frames` (default 16), `clip_fps` (default 4 → 4 s clip), `resolution` (128), `render_pass` ("flat")
  - `physics`: timestep (default 1/240 s), solver iterations

`clip_fps` is the event rate in the video. It is **unrelated** to the GPU throughput cap below.

### Backend interface (both backends implement it)

```
load(episode) -> handle
step_to(t)                   # advance physics + mechanics
snapshot() -> State ; restore(State)
apply_violation(v)           # add/delete an object in sim state
frame_state() -> FrameState  # per-frame ground truth
render(pass) -> uint8[H,W,3], segmentation uint16[H,W], depth float32[H,W]
```

Per-frame state (sidecar JSON): `t`, `action` active at t, `count_per_region` (container, table, claw, screen-hidden), object `id`, `position`, `quaternion`, `velocity`, `visible` (from segmentation: ≥ N visible pixels), `region`, `seed`.

Counterfactual branching: `snapshot()` at a branch point, run K different scripts from it. Impossible twin = run possible clip, restore the snapshot taken just after the screen fully covers the container, `apply_violation`, continue. Both clips share every frame before the violation (verified by the validator: identical pixels before `at_t`).

### Rust engine

- **Physics:** rapier3d, `enhanced-determinism` feature, fixed timestep, single-threaded per episode (parallelism comes from running many episodes across processes).
- **Mechanics:** a `Mechanic` trait with hooks (`init(world, spec)`, `pre_step(world, t)`, `post_step(world, t)`, `state(world) -> json`). Claw, chute, screen, container are each a module. The claw moves via kinematic bodies and grasps with a fixed joint or a cradle — it physically carries objects; no teleport. A new mechanic = new module + new spec field + registration; the core does not change.
- **Rendering:** wgpu over Vulkan, offscreen render targets, flat shading + one shadow-casting light, instanced primitive meshes, segmentation and depth outputs. Render pass is a trait so the Track A rich pass can be added later.
- **Bindings:** PyO3 module `jepamath_engine` built with maturin into the `jepamath` env; exposes the backend interface above; errors become Python exceptions.

### MuJoCo backend

- MJCF built per episode from the concrete spec; claw as a mocap/actuated body; screen as a kinematic slab.
- Rendering via MuJoCo's renderer on EGL (GPU, headless), same resolution and camera conventions as Rust.
- Same mechanics semantics as Rust; the `crosscheck` command runs one spec on both and compares `count_per_region` timelines and action outcomes, failing on any mismatch.

### GPU frame cap

User requirement. One limiter caps **total rendered frames per second across all worker processes** (not per process), shared through a file-backed token bucket under the run's output directory. Default `--max-render-fps 30`; configurable. Both renderers acquire a token before each render call. Physics runs on CPU and is not capped. GPU temperature and power are logged every 30 s via `nvidia-smi`; if temperature exceeds a configurable ceiling (default 83 °C) generation pauses until it drops 10 °C, and the pause is logged. At 30 fps, 100k × 16 frames ≈ 15 hours; raising the cap is a flag.

### Validator (per clip, all must pass)

- Logged `count_per_region` matches counts recomputed from object positions and region geometry.
- Final counts match the script's arithmetic (possible clips) or differ by exactly the violation (impossible).
- No interpenetration beyond tolerance; no object outside the scene bounds; no object below the table.
- Resting objects' speeds below a jitter threshold.
- Every object that becomes visible, or stops being visible, does so via a mechanic or occlusion — not by appearing/disappearing (segmentation-based check), except the violation in impossible twins, which must occur while hidden.
- Twins: pixels identical before `violation.at_t`.
- Rejected clips are logged with the reason; rejection rate per reason reported. No silent skips.

### Controls and splits

- Sampler implements matched sets per the handoff (decorrelate count from total area, object size, density, convex hull, brightness — Stoianov & Zorzi style), randomized shape/color/texture per episode, mixed types, randomized camera/lighting/background/container, randomized action timing.
- Count distribution: **uniform by default**; Zipf as a flag for the ablation.
- Seed manifest written at generation time defines held-out counts (10–15), held-out a + b combinations, held-out object types, longer action sequences. Training splits contain only possible clips.

### Leak audit

Over a sample (default 10k clips): compute cheap features (total object pixels, mean brightness, convex hull area, number of connected components, reveal frame index) and fit logistic regression and gradient-boosted classifiers to predict the count. Report accuracy vs chance with confidence intervals; exit non-zero if any classifier beats chance by more than a threshold. Training-scale generation refuses to start without a passing audit for the template.

### Shards

WebDataset tar shards (~1 GB each): per clip `NNNNNN.frames.npz` (uint8 [T,H,W,3], losslessly compressed) or per-frame PNG (flag), `NNNNNN.state.json`, `NNNNNN.episode.json`. An index (parquet) maps clip → shard, split, counts, action, backend, twin id.

### Preview

`jepagen preview` renders ~20 clips to GIFs (possible and impossible pairs side by side) for visual review before scaling.

## Order of work

1. Conda env + project skeleton (Python package, Rust workspace, CI-style test commands).
2. Spec models + sampler + manifest.
3. MuJoCo backend + renderer + frame limiter → first claw episode.
4. Validator, shard writer, preview GIFs.
5. Twins.
6. Leak audit on 10k clips.
7. Rust engine: core + claw/screen/container mechanics + wgpu renderer + PyO3 → `crosscheck` against MuJoCo.

Rust work can run in parallel with steps 3–6 once the spec (step 2) is fixed.

## Decided by the user

- Rust engine built alongside MuJoCo, with physics, for flexibility in adding systems/mechanics.
- Approach A (rapier3d + wgpu + mechanics as modules).
- Python work in a conda environment.
- GPU work runs with capped frame rates.

## Builder's calls (made in this design, open to correction)

- New `jepamath` conda env instead of `base`.
- Total render cap 30 fps default, global across workers, plus a temperature pause at 83 °C.
- 100k-clip default target given disk space.
- Uniform count distribution as default.
- Clip defaults: 16 frames at 4 fps, 128px, physics at 240 Hz.

## Left to the builder

Crate and package versions; exact MJCF construction; claw grasp mechanism details; jitter/interpenetration thresholds (tune on preview clips); shard size; feature list details in the audit beyond those named.
