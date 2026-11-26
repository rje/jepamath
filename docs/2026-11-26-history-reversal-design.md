# History-reversal test (section C of the 26 Nov handoff): design v2

v1 was reviewed by Astra 6.2 (26 Nov). Its flaws:
- the visible hopper let a box+hopper total-count scorer pass without the action;
- a direction-only rule passed every registered reversal;
- the targets depended on the history;
- the validation cutoff was wrong;
- the move control was uncalibrated;
- the metadata and drop hints were stale;
- the stock loader could not hold the items;
- the decision rules overlapped.

v2 replaces hidden violations with natural endings scored across histories.

**Question.** Does a trained JEPA predictor use the visible action history to predict what the hidden box holds at
the reveal? The narrow claim is history use for box contents. An arithmetic claim needs the later held-out
combinations.

## Stimuli: four histories per scene, all natural (no violations)

Base: v4b occluded scenes (`configs/v4b_small.yaml` ranges). The test seed range is disjoint from v4b train,
v4b_probe and the heldout triples. Every history has exactly **two transfers with the same timeline**:
- screen raise;
- transfer 1 and transfer 2, with identical t_start and t_end;
- screen lower.

Only the source and destination of each transfer differ. Transfers pick the topmost object of the source tray
(the engine rule). The initial state is identical, so the picks are deterministic.

| History | Transfer 1 | Transfer 2 | Final box count |
|---|---|---|---|
| H++ | hopper→box | hopper→box | n0+2 |
| H+ | hopper→box | box→box | n0+1 |
| H0 | box→box | box→box | n0 |
| H− | box→hopper | box→box | n0−1 |
| H−− | box→hopper | box→hopper | n0−2 |

- n0 = 1 uses {H++, H+, H0, H−} (finals 3, 2, 1, 0).
- n0 = 2 uses {H+, H0, H−, H−−} (finals 3, 2, 1, 0).

All finals lie in the trained range 0–3, and each scene has four histories with four distinct finals. The
hopper starts with ≥ 2 objects. Each history gets correct meta (op, k, shuffles, final). Drop hints are drawn
once per transfer in destination-relative coordinates for both trays, and the one for the actual destination is
used. Size-control mode is recorded and stratified (area matching uses one reference count).

**Hopper masking.** The hopper's state at the end reveals the history (it is visible throughout). So in every
clip of a scene, a fixed image region is painted with the mean table colour in all frames:
- the hopper tray polygon, projected from the camera;
- plus the pixels of hopper objects and their shadows (from the seg map), dilated;
- the union is taken over the scene's four histories.

The carry over the table, between hopper and box, stays visible.

## Scoring: history-independent targets, one prediction per history

For each scene, the four histories h, and their natural clips X_h:
- **P_h** = predictor output for the post-reveal target tokens, from X_h's own context (frames before the screen
  starts lowering, rounded down to a tubelet). Computed once.
- **T_e** = target-encoder features of X_e's post-reveal tokens. X_e is the natural clip of history e, encoded
  once. T_e depends on e only, so the target does not change with the scored history.
- **S[h, e]** = mean L1 between P_h and T_e over **box-region tokens**: tokens whose patch overlaps the box tray
  interior polygon, in post-reveal slots. Post-reveal slots are those whose **first** frame is at or after the
  screen is fully down. The diagonal is legal.

Every candidate is a natural, physically consistent clip of the same scene. There is no teleport or seam, and the
hopper is masked.

**Primary endpoint.** 4-way identification accuracy: for each (scene, h), is argmin_e S[h, e] = h? Chance is 25%.
Ties count as wrong.

**Secondary endpoints:**
- **Adjacent reversal rate.** For adjacent pairs (h, e) with |final_h − final_e| = 1, the share where both
  S[h, e] > S[h, h] and S[e, h] > S[e, e]. A random scorer scores 25%. Any history-blind scorer scores 0%,
  because it ranks the endings the same way for every h.
- **Magnitude cells.** Identification restricted to the same-direction pairs (H+ vs H++, and H− vs H−−). These
  defeat a direction-only rule.
- **Error profile.** Mean S[h, e] − S[h, h] as a function of final_e − final_h. A graded code would increase
  with |Δ|.

**Cheap strategies** (same items, same endpoint):
- an amount prior (fewest box-region object pixels in the ending, or most);
- last visible count (always picks H0);
- a pixel persistence baseline: L1 between X_h's last context frame and X_e's post-reveal frames, on the same box
  tokens;
- a direction-only oracle (it knows the sign of the change, not the size);
- an oracle state counter (always right; checks the items).

Each history-blind strategy should score ≤ 25%, and its adjacent reversal rate should be 0%. That is checked as
an acceptance test of the item set before any model is scored.

**Validation of every clip:**
- `validate_clip`, including the RGB leak check;
- the X_h are pixel-identical up to transfer 1's t_start;
- each transfer's carried object is visible (outside the box and hopper masks) in some context frame;
- the masks cover the hopper in every frame.

The rejection reasons and the retained scene count per n0 are recorded.

**Statistics.** Bootstrap over scenes, resampling all of a scene's cells together, stratified by n0. Equal n0
weights. 95% CIs.

**Models:**
- untrained (step 0);
- plain cw50-only 30k;
- object-label mask05b 30k and mask05-100k;
- track-v4b 30k ×2;
- motion-label v5 100k (off-distribution, flagged).

No training spend. Size: 200 accepted scenes per n0 × 4 clips = 1,600 renders.

## Pre-registration (v3, 26 Nov; fixed before any trained model is scored)

v3 changes after Astra's second review:
- **Target is post-reveal box tokens only** (target encoder with `keep` indices). The target therefore sees no
  prefix, hopper or claw route. The hopper mask is dropped.
- **n0 = 1 H− runs the shuffle first.** The other single-op histories put the shuffle in slot 1 or 2 by a
  per-scene coin.
- **Per-transfer visibility.** Each carried object must be visible while held, in the model's context frames.
- **Scored slots.** Only tubelets whose first frame is at or after the reveal end are scored.

The pilot (20 scenes, seeds 900000+) gave this acceptance ladder. An oracle reads one piece of the history and
breaks ties uniformly.

| Strategy | Identification | Adjacent reversal | Magnitude |
|---|---|---|---|
| History-blind, last-visible-count, pixel persistence, untrained model | 0.25 | 0.00 | 0.50 |
| First transfer only | 0.55 | 0.40 | 0.68 |
| Last transfer only | 0.70 | 0.60 | 0.83 |
| Sign of change only | 0.75 | 0.67 | 0.50 |
| Count oracle | 1.00 | 1.00 | 1.00 |

**Item set.** Seeds 905000 and up (disjoint from the pilot), 200 accepted scenes per n0. The size is fixed, and no
scenes are added after scoring. Scene-cluster bootstrap stratified by n0, equal n0 weights, 95% CIs.

**Primary model:** object-label mask05-100k. Every other model is exploratory.

- **H-C1 (history used):** identification lower bound > 0.30 and reversal lower bound > 0.05. History-blind
  scorers get 0.25 and 0.
- **H-C1′ (not used):** identification upper bound < 0.30. This rules out an effect of that size on this readout
  only.
- **H-C2 (more than one transfer's worth):** identification lower bound > 0.75, which beats every partial-history
  oracle.
- **Inconclusive:** neither H-C1 nor H-C1′ holds.
- **Magnitude** is reported against both 0.50 and the last-transfer oracle's 0.83. Only > 0.83 rules out a
  last-transfer reader.
- **Exploratory:** the full-clip-target score, which leaks the prefix and is reported only for comparison.

## Known weaknesses

- **Identity rather than number.** The legal ending also matches the carried object's identity (colour and shape)
  and the drop spot. Winning shows the model tracks which objects went where. It does not show abstract number.
  H-C3, and later held-out (n0, action) combinations, address that.
- **The hopper mask is a distribution shift** the encoder never saw. It is the same for all candidates, so it
  cannot favour one ending. It may lower overall accuracy.
- **Box-region tokens are selected by a projected polygon.** The target encoder still attends over the whole
  masked clip.
