# C9: source-side number versus carried area or volume (pre-registration, 27 Nov)

**Design.** Astra's design (docs/2026-11-27-c9-number-vs-area-draft.md), implemented in `scripts/c9_items.py` and `scripts/c9_score.py`.

**Why.** C8 passed, but only as narrowed by its amendment 1. Within a C6–C8 source scene, carried area always moves with count, so a source that tracks area passes every edge. C9 builds histories where they move in opposite directions.

## Items (fresh; frozen before any checkpoint sees them)

### Sources

Seeds are [4,300,000, 4,600,000), with 330 accepted. Each block is one shape (sphere, box or cylinder, by seed mod 3) with 12 clips: 3 families × 4 histories.

| History | Start: box, then hopper | Transfers | Final box | Final count |
|---|---|---|---|---|
| H | a, s; s_in | BB, HB | a, s, s_in | 3 |
| L-N | a; b_low | BB, HB | a, b_low | 2 |
| L-add | a, b_low; s | BB, BB | a, b_low | 2 |
| L-remove | a, s; b_low | BH, HB | a, b_low | 2 |

Base sizes are a = 0.035, s = 0.037 and b = 0.062.

| Family | b_low | s_in | What it does |
|---|---|---|---|
| orig | b | s | H gains an object but has 28% less final area and about 49% less volume than L |
| area | √2·s | √(b² − s²) | swaps the orig final summed areas exactly |
| volume | ∛2·s | ∛(b³ − s³) | swaps the orig final summed volumes exactly |

**Checks on every source clip.** A scene is accepted only if all of these hold:

- the simulator-verified grasped object IDs are as designed;
- actual sources and destinations, releases inside context, masked visibility, and source/destination cues pass;
- the scene shows the same hopper-masked frames before transfer 1 in L-remove and H for each family;
- there is one ctx and one first post-reveal slot per block.

### Donors

| Layout | Seeds | Accepted | Two-object ending | Three-object ending |
|---|---|---|---|---|
| **rev** | [4,600,000, 4,900,000) | 660 | radius 0.061, placed wide | radius 0.037, placed compact |
| **fwd** | [4,900,000, 5,200,000) | 660 | radius 0.037, compact | radius 0.061, wide |

In the pilot, pixel area and spread were reversed in 12 of 12 donors of each layout.

### Pairing and freeze

`c9_score.py --freeze --per-shape 80` builds **240 blocks**, 80 per shape. Each source gets one rev and one fwd donor, by max-IoU assignment per layout:

- unique donors;
- identical ctx, first slot and length;
- IoU ≥ 0.6.

Every file is hashed. If a shape is short of 80, more seeds are rendered from [5,200,000, 5,300,000) for that role **before** the freeze.

## Endpoints

The graded preference is p = d(2-object ending) − d(3-object ending). An edge L → H is correct when p(H) > p(L). Ties within 1e-12 get half credit, and each edge is averaged over the rev and fwd donors.

- **Edges.** `conflict:<fam>` is L-remove → H, `LN:<fam>` is L-N → H, and `Ladd:<fam>` is L-add → H.
- **Paired endpoints.** `pair_area:<edge>` is the mean of orig and area for that edge; `pair_volume:<edge>` is the mean of orig and volume.
- **Bootstrap.** 10,000 resamples of whole blocks, stratified by shape. Intervals are two-sided 95%.

## Validity (from the simulator records)

- **State counter.** `count:nearest` scores 1.0 on every endpoint.
- **Area and volume readers.** `area:nearest/farthest` and `volume:nearest/farthest` are exactly 0.50 on their respective paired endpoints (pair_area and pair_volume).
- **Pixel readers.** `pixel_first` and `pixel_last_ctx` each have an upper bound < 0.55 on their weakest registered endpoint.

## H-C9 (objlabel-100k)

All of the following must hold:

- lower bound > 0.55 on conflict:orig, conflict:area and conflict:volume;
- lower bound > 0.55 on pair_area and pair_volume for each of conflict, LN and Ladd;
- lower bound > 0 on the paired advantage over **untrained**, on each family's conflict edge;
- lower bound > 0 on the paired advantage over `radius_sum:nearest` and `radius_sum:farthest`, on pair_area:conflict and pair_volume:conflict. Summed radius is not swapped by design: 2s > b, so it rises with count on the conflict edge.

**What a pass shows.** The predicted ending tracks the hidden **count** rather than the source's carried summed area, its summed volume, or its summed radius. Each scalar alone is excluded; a joint size-distribution reader is not.

**What a failure shows.** The C8 response is consistent with source-side area or volume tracking.

**Reported.**

- plain-100k, untrained and the 30k models;
- the trained seed-1 100k checkpoints, if they have finished;
- per-shape and per-layout results.

## Deviations from the draft

- **Source quantities.** Source quantities are taken from the simulator's final box contents (metadata). The pilot did not measure the carried-object pixel area at 128 px. The size ratios involved (b/s = 1.7 in radius) are large, but that check is deferred and reported.
- **Spend.** None is tied to C9.
