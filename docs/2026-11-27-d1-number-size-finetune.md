# D1: does breaking the count–size link in training make the model's readout follow number? (pre-registration, 27 Nov)

## Why

On both seeds, a linear probe reads the final count from objlabel-100k's predicted hidden ending at about 0.99 (P1). But the model's own ending match (C9 L1 readout) follows carried size, scoring 0.51–0.55 on the conflict edges.

In v4b training every object in a clip shares one base radius (±12%), so within a clip an add always increases summed area and a remove always decreases it. D1 asks whether training on clips where count and carried size disagree moves the native readout toward number.

The design follows Astra's review (27 Nov, scratchpad astra_d1.md):

- fine-tune two matched continuations of one checkpoint rather than train from scratch;
- keep counts at 0–3;
- use a treatment that differs from the control only in how size is assigned;
- confirm on fresh C9 blocks.

## Training clips (`scripts/d1_items.py`)

A v4b base scene (2 transfers, any visibility, stock timing; one shape per scene) becomes a matched pair of histories:

| Direction | Start | "two" history | "one" history |
|---|---|---|---|
| add | box {a} or empty (n0 ∈ {0,1}); hopper always 2 objects | HB, HB: both hopper objects X carried → n0+2 | HB + BB (order random; HB first at n0=0): Y carried, filler f stays → n0+1 |
| remove | box {a, s, b} or {s, b} (n0 ∈ {3,2}); hopper {f} | BH, BH → n0−2 | BH + BB (order random) → n0−1 |

**Size assignment (add scenes).**

- **concordant:** X = two big, Y = small. More objects means more area.
- **discordant:** X = two small, Y = big. More objects means **less** summed area and volume; this is checked per pair.

Bands: small U(0.035, 0.040), big U(0.058, 0.065), a U(0.035, 0.065). The claw takes the tallest object in the source tray, so the removal scenes can only be concordant, and they are shared by both arms.

**Checks.** validate_clip, two grasps and two releases, and the final box contents by size class (by radius for removals), all from the simulated last state. The test render of 12 scenes had no rejects, and the frames were inspected.

**Sets.** Seeds are disjoint from training (0–99,999) and from C1–C9 (< 5.3M).

| Set | Seeds | Scenes | Clips |
|---|---|---|---|
| A (add, rendered both ways) | [6.0M, 6.3M) | 1000 | 2000 disc + 2000 conc |
| B (add, concordant only) | [6.3M, 6.6M) | 1000 | 2000 conc |
| R (remove) | [6.6M, 6.9M) | 1000 | 2000 |

**Arms** (6,000 new clips each, the same marginal radii and the same action schedules):

- **treatment** = A_disc + A_conc + R (add clips 50% discordant);
- **control** = A_conc + B_conc + R (add clips 100% concordant).

## Fine-tune

Both arms start from objlabel-100k `ckpt_100000.pt` (encoder, predictor, target and mask head; fresh AdamW). Recipe:

- 10,000 steps, lr 1e-4, warmup 500, cosine to 1e-6;
- every other flag as the parent (l2, pool 1.0, change 50, p-prefix 0.8, mask-weight 0.5), seed 0;
- each sample drawn from the arm's new clips with probability 0.5 (`--mix-data --mix-frac 0.5`), otherwise from v4b_128s.

The two arms differ only in `--mix-data`. Spend is a $3 cap per arm.

## Test items

- **C9b (confirmatory).** Fresh C9 blocks rendered with the unchanged `scripts/c9_items.py`: sources [7.0M, 7.3M), rev donors [7.3M, 7.6M), fwd donors [7.6M, 7.9M), top-up [7.9M, 8.0M) before the freeze. `c9_score.py --freeze --per-shape 80` gives 240 blocks. The histories, sizes and donors are C9's, which differ from the D1 training scripts.
- **C9 v1 frozen blocks.** Reported only, since they informed this design.
- **C8 frozen manifest.** Used for non-inferiority.

## Hypotheses (weights_010000 of each arm; parent = objlabel-100k)

**H-D1 (the lever works).** On C9b, the treatment arm has:

- a lower bound > 0.55 on both `pair_area:conflict` and `pair_volume:conflict`;
- a paired treatment − control lower bound > 0 on both.

**H-D1-NI (nothing broken).** On C8, the treatment − parent paired lower bound is > −0.05 on each of N, add and remove.

**Practically ineffective.** Both treatment − control intervals on `pair_area:conflict` and `pair_volume:conflict` lie wholly inside ±0.03. Wider intervals that fail H-D1 are inconclusive.

**Reported.**

- C9b `conflict:orig/area/volume` for both arms and the parent;
- each donor layout (rev, fwd) separately;
- C9 v1 for the same models;
- the P1 probe and the P2 steering test on both arms.

**Next steps by outcome.**

- **Pass:** replicate the contrast from objlabel-100k-s1, then run the plain-parent comparison.
- **Ineffective:** the size bias is not a training-statistics problem at this dose. Redirect to the readout (P2/P3).
