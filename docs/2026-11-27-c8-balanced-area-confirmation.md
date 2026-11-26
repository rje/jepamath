# C8: confirmation of the graded count-ordered signal, with donor area asymmetry balanced (pre-registration, 27 Nov)

## Why C8

C7 (docs/2026-11-27-c7-edge-confirmation.md) was invalid under its own gate. The C6 v2 donor calibration equalised rendered pixel area, but that left the 3-object ending about 5% larger in metadata area in 99% of donors. A metadata-area reader therefore scored 0.76 on every edge.

The model's C7 numbers were strong, and the area check points the other way:

| | N | add | remove |
|---|---|---|---|
| objlabel-100k, all blocks | 0.76 | 0.68 | 0.64 |
| objlabel-100k, blocks where *visible* area favoured the 1-object ending | 0.75 | 0.65 | 0.61 |

All of these are exploratory, and C7 confirms nothing. C8 fixes the item flaw and runs the same pre-registered test on fresh items.

## Items (fresh, frozen before any checkpoint sees them)

- **Generator.** `scripts/c6_items.py` as of this commit. Sources are unchanged from C6 v2 and C7.
- **Donors.** Donors use `--area-jitter 0.15`:
  - the radius calibration is set halfway between pixel-matched and metadata-matched;
  - each donor draws a log metadata-area offset δ ~ U(−0.15, 0.15) between its 3- and 1-object endings, split symmetrically.
- **Donor pilot.** A renderer-only pilot (96 donors, seeds 1.90–1.96M) gave the 3-object ending the larger area in 61% of donors on metadata and 40% on pixels. Pixel and metadata area agreed in sign in 70% of donors.
- **Seeds.** Sources come from [3,000,000, 3,300,000), 600 accepted. Donors come from [3,400,000, 3,900,000), 1,200 accepted. These ranges are disjoint from training, C1–C7 and the pilots. The top-up window is [3,900,000, 4,000,000), used only if balancing fails, and only before the freeze.
- **Pairing.** `scripts/c6_manifest.py --per-order 200 --balance-area`. Pairing works as in C7: unique donors, IoU ≥ 0.6, common-patch area ratio ≤ 1.20. Within each slot order:
  - exactly half the blocks have the 3-object ending larger in **metadata** area;
  - exactly half have it larger in **pixel** area on the pair's common patches;
  - the two concordant cells hold equal numbers of blocks, and so do the two discordant cells.
- **What balancing does to area readers.** Any count-monotone source matched to either donor area then scores **exactly 0.50** on every edge, by construction.

## Endpoints, validity and hypotheses

These are C7's, unchanged (`scripts/c7_score.py`):

- **Edges:** N, add, remove and conflict. The graded preference is p = d(1-object) − d(3-object). Ties within 1e-12 earn half credit.
- **Bootstrap:** 10,000 resamples of whole blocks, stratified by slot order.
- **Validity:** the item set is invalid unless all of these hold:
  - the state counter scores 1;
  - every omitted-input symbolic reader scores exactly 0.50 on that input's edges;
  - `meta_area:pos`, `pixel_area:pos` (new: the visible pixel-area reader), `pixel_last_ctx:pos` and `pixel_first:pos` each have an upper bound **< 0.55** on their weakest input.
- **Reported, not gated:** `meta_volume:±` and the reversed readers.
- **H-C8, for objlabel-100k.** All of the following must hold:
  - the lower bound is above 0.55 on each of N, add and remove;
  - the paired advantage over untrained has a lower bound above 0 on each input;
  - the paired advantage over each gated shortcut reader has a lower bound above 0 on mean_3.
- **H-C8c:** the lower bound on the conflict edge is above 0.55.
- **Also reported:** the other checkpoints, paired differences and per-cell results. The per-cell results include the cell where both area signs favour the 1-object ending.

## What a pass shows

The model's predicted hidden ending is ordered by a quantity that integrates the starting count, adds and removes, on fresh scenes. That quantity is neither visible area nor metadata area.

**What stays open:**

- whether the ending-side readout is number, or a count-correlated ending property such as spread or blob count;
- whether additive cues remain viable, unless H-C8c passes.

## Spend tied to C8 (tonight's approval: up to $30)

| Outcome | Spend |
|---|---|
| H-C8 passes | Launch `trackb-v4b-128-mask05-100k-seed1` and `trackb-v4b-128-cw50-100k-seed1`, each capped at $7. They test replication across training seeds and the contribution of object labels. |
| H-C8 fails, or the item set is invalid | Nothing launches. |
