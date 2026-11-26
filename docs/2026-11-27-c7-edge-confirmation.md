# C7: confirmation of C6's graded count-ordered signal (pre-registration, 26–27 Nov)

## Why

C6's pre-registered "is the count 2" endpoint was at chance for every model. Exploratory analysis of the frozen C6 matrices (scripts/c6_graded.py) showed something else for objlabel-100k. Its graded preference for the 3-object over the 1-object ending follows all three history inputs, with the correct signs:

| Input | Coefficient [95% CI] |
|---|---|
| Starting count N | +0.98 [0.89, 1.08] |
| Add | +0.60 [0.52, 0.67] |
| Remove | −0.44 [−0.52, −0.36] |

The ordering is not area: the slope is +0.67 [0.62, 0.73] when the endings have equal area.

Astra (research-partner consult, 26 Nov) proposed confirming this on fresh items with an edge endpoint, which cancels donor bias. It also found that on the frozen C6 matrices the conflict edge (N1A, final 2, vs N2R, final 1) is only 53%. Every number in this section is exploratory. C7 is the confirmation.

## Items (fresh, frozen before any checkpoint sees them)

- **Generator.** `scripts/c6_items.py`, unchanged from C6 v2 (commit aecea48 + 111e01f). This keeps the retimed reveal, simulator-verified transfers, masked visibility, the src/dst cue checks, per-count donor area calibration and an independent source scale.
- **Seeds.** Sources [2,000,000, 2,250,000), 440 accepted. Donors [2,300,000, 2,700,000), 880 accepted. The top-up window [2,700,000, 2,800,000) is used only if a slot order has fewer than 200 pairs, and only before the freeze. None of these seeds overlap training (0–99,999), C1–C6 or the pilots.
- **Pairing.** `scripts/c6_manifest.py --per-order 200` gives exactly 200 blocks per slot order, 400 in all. Each block keeps the lowest source seeds. Pairing uses unique donors, IoU ≥ 0.6 and a common-patch area ratio ≤ 1.20. The manifest records sha256 hashes for every file.

## Endpoints (scripts/c7_score.py)

The graded preference is p(h) = d(h, 1-object ending) − d(h, 3-object ending), from the C4/C6 model matrix with the hopper masked. An edge is a pair of histories that differ in one input. It is correct when p moves the way the count does; a tie within 1e-12 gets half credit. Edges are averaged equally within an input, then over blocks.

| Endpoint | Edges |
|---|---|
| **N** | (N1-, N2-), (N1A, N2A), (N1AR, N2AR) |
| **add** | (N1-, N1A), (N2-, N2A), (N2R, N2AR) |
| **remove** | (N1A, N1AR), (N2-, N2R), (N2A, N2AR) |
| **conflict** | (N2R → N1A): the initial-count cue points the wrong way |

A reader that lacks an input ties on that input's edges, so it scores exactly 0.50.

The bootstrap uses 10,000 whole-block resamples, stratified by slot order. Intervals are two-sided 95%.

## Validity (must hold, or the item set is invalid)

- The state counter, computed from the simulated finals, scores 1.0 on all four endpoints.
- Every symbolic reader that lacks an input scores exactly 0.50 on that input's edges. For example, the N-only reader scores 0.50 on add and remove, and the net-change reader scores 0.50 on N.
- The **nearest-match** shortcut readers `meta_area:pos`, `pixel_last_ctx:pos` and `pixel_first:pos` must each have an upper bound **< 0.55 on their weakest input** (min_3). `meta_volume:pos` is reported only.
- **Why only nearest-match is gated.** A "reversed" reader maps a larger source quantity to a *smaller* donor quantity. Under area matching, donor volume falls with count, so reversed volume (`meta_volume:neg`, 0.83–0.85 on the pilot) is a count-equivalent rather than a shortcut, like C6's count_equiv. The nearest-volume reader scores 0.15–0.17, the opposite of the C6 behaviour.
- The reversed readers are reported. For edges, any reader that integrates a *carried quantity* passes by construction. C7 tests integration of the history, not which ending property is read out.

## Hypotheses (objlabel-100k = cloud-v4b-128-mask05-100k weights_100000)

**H-C7 (integration). All of the following must hold:**

- lower bound > 0.55 on each of N, add and remove;
- lower bound > 0 on the paired advantage over **untrained**, on each of N, add and remove;
- lower bound > 0 on the paired advantage over each gated nearest-match shortcut reader, on mean_3.

**H-C7c (conflict).** Lower bound > 0.55 on the conflict edge. This is the stronger claim: the inputs are combined, not merely additive cues that the initial state can dominate.

**Reported.**

- plain-100k, objlabel-30k, objlabel-30k-s1, plain-30k and track-30k, with paired differences against objlabel-100k;
- the C6 endpoints (is-2, exact, 1v3) on the fresh items;
- per-slot-order results.

**What a pass shows.** The model's predicted hidden ending is ordered by an integrated quantity that responds to the starting count, adds and removes, on fresh scenes. Under area matching, that quantity is not area. It may be number, or a count-correlated ending property such as spread; the ending-side readout is open and is the next test. Additive appearance and route cues remain viable unless H-C7c passes.

## Spend tied to C7

| Outcome | Spend |
|---|---|
| H-C7 passes | Launch the second objlabel-100k seed (job trackb-v4b-128-mask05-100k-seed1, same recipe, --seed 1, cap $7) and a matched plain-100k seed 1 (cap $7). Together they test that the signal replicates across training seeds and that object labels contribute beyond longer training. |
| H-C7 fails | Neither launches. Redirect with Astra. |
