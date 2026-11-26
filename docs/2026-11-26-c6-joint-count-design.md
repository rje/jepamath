# C6 draft — Does the predicted ending depend jointly on the initial count and both transfers?

**Design decision.** Keep three natural donor endings, with final counts 1, 2 and 3. Change the confirmatory endpoint. Use a crossed set of seven source histories to test whether the predictor distinguishes **two objects from one or three**, then require a separate test of **one versus three**. All scoring uses frozen checkpoints, local renders and CPU inference.

### 1. Why the proposed conjunction fails

Under the proposed foil rules, an exact `n0`-matching oracle scores **0** and an exact sequence-matching oracle scores **0**. A reader that excludes the matching foil and guesses between the survivors scores **1/2**. A reader with access to both donor history labels identifies the unmatched candidate with score **1**, without computing a final count. Donor prefixes are hidden from the JEPA target, but donor appearance can retain proxies for those labels. First transfer, last transfer, sign and net change have **no fixed expected score** under “different where possible.” Their scores depend on the realized candidate sets.

There is also a limit that better foil sampling cannot fix. Write \(F=N+D\), where \(N\) is initial count and \(D\) is net transfer. On a fixed slate with one ending per positive count, making the *best* \(N\)-only and \(D\)-only readers both chance would require \(F\) to be uniform conditional on either cue. That would give \(\operatorname{Cov}(F,N)=\operatorname{Cov}(F,D)=0\), yet their sum is \(\operatorname{Var}(F)>0\). Increasing the initial-count range cannot solve this. C6 therefore keeps exact three-way identification as a reported endpoint and uses a different endpoint for the chance guarantee.

### 2. Source histories and donor endings

One source scene supplies the following seven clips. `A` switches its slot from box→box (`BB`) to hopper→box (`HB`). `R` switches the other slot from `BB` to box→hopper (`BH`). Assign the add-capable slot first in half the scenes and second in half. Every clip has two physical transfers on the same timeline.

| Initial box \(N\) | Neither active | A only | R only | Both active |
|---|---:|---:|---:|---:|
| 1 | 1 | 2 | omit | 1 |
| 2 | 2 | 3 | 1 | 2 |

Entries are final box counts. The omitted cell ends empty. The retained cells contain three complete four-history comparisons:

| Comparison | Vary | Hold fixed | What a pass requires |
|---|---|---|---|
| I | Initial count and add slot | Remove slot is `BB` | Sensitivity to initial state and an incoming transfer |
| II | Initial count and remove slot | Add slot is `HB` | Sensitivity to initial state and an outgoing transfer |
| III | Add slot and remove slot | Initial count is 2 | Sensitivity to both transfer slots |

In **each** comparison, “final is 2” is true in two cells and false in two. Holding either varying factor fixed leaves one true and one false. This is the chance guarantee. It holds within a scene with the *same actual donor pixels* for all seven queries.

**Generator changes.**

- Replace `base_ok` with an occluded, no-glide, no-violation, `n0=2`, `op=add`, `k=1`, `shuffles=1`, `n_hopper=1` source template. Require exactly two `Transfer`s and the raise/lower timeline. Derive the `n0=1` start by moving one randomly selected initial box object to a legal reserve pose before simulation. Keep camera, physics, other object identities, screen, times and per-slot/per-destination drop hints fixed. Use an independently drawn source size scale. The stock area mode uses the **sampled base final**, which is stale after route substitution.
- Replace `histories()` and the `[0,1,2,3]` assertion with the table above. One initial hopper object suffices in either order. With `BH` first it can temporarily hold two. In the `n0=1` round trip, a removal can empty the box, but the following `HB` picks from the hopper. Every retained `BB` has a box object available. The stock `k_range=[1,1]` supplies the **base timeline**. It does not generate the mixed round trip. Store `n_in`, `n_out`, ordered route and actual `BB` count explicitly. For `HB`+`BH`, use net `op=none,k=0` for the existing hopper validator and separately check both actual region transitions. Current `make_histories` would label that route `add,1` and fail validation.
- Run `validate_clip` on every clip. Run the prefix-identity part of `item_checks` **within each initial state**. Check both grasps and releases, initial box visibility after preprocessing, and visibility of each held object **after hopper masking and downsampling**, before the rounded `ctx` cutoff. Audit resting hopper pixels and shadows outside the mask, box leaks during occlusion, and the fill ring. Record actual carry and release positions from simulation. The engine chooses a low free drop site near the hint, so a common hint does not imply a common landing point.
- Use a **shared pool of independent donor scenes**, each with one fixed slate. Within a donor, render three natural `BB,BB` clips starting at 1, 2 and 3, using the same donor geometry, timing and nuisance draws. Construct area-matched object sets for each count using the v4b radius range and one reference area. As a proposed acceptance tolerance, require largest/smallest measured post-reveal box-object area ≤1.20 on the scored region. Record height, hull, size and area separately. Freeze the tolerance on a disjoint renderer-only pilot. If yield is poor, revise the draft before scoring.
- Assign one donor to each source **without replacement**, using geometry and timing only. Require identical `ctx`, `first_post_slot` and clip length, box-token IoU ≥0.60, and a nonempty common token mask. Use that same intersection for every row and column. Select whole blocks, never a different nearest donor for each candidate. Freeze the manifest, candidate permutation, seed ranges and rejection counts before checkpoint inference.

**Budget.** 400 source scenes × 7 clips plus 400 independent donor scenes × 3 clips = **4,000 accepted renders**, 2.5 times C5. This gives 400 independent source–donor blocks. The local C5 builder log spans about 92 seconds for 1,600 retained clips and records 58 rejected scenes. Linear scaling gives about **4 minutes of rendering at C5 yield**. Allow roughly **5–15 minutes** if matching and area rejection are moderate. C5 checkpoint outputs were written about 2.3 minutes apart. Budget **4–6 minutes per C6 checkpoint** with source predictions and donor targets cached per block. These are planning estimates. Measure yield and throughput on the pilot. The batch limiter operates per GPU submission, so dividing all rendered frames by `--max-render-fps` gives the wrong estimate.

### 3. Scoring, oracle ladder and registration

Use the C4 scoring path. Mask the hopper in every source context frame. Query all future tokens, then select post-reveal box tokens. Give the target encoder only the donor’s selected post-reveal tokens. Compute mean tokenwise L1 to each of the three endings. Replace the hard-coded four-row loader, diagonal `ident` and `ident_pos` slicing with an explicit seven-row truth vector.

Let \(\hat F\) be the final label of the minimum-distance donor. For comparisons I–III score whether \([\hat F=2]=[F=2]\). Use equal cell weights within each comparison. For the one-versus-three test, compare only those two donor distances on the actual extreme rows, weighting true 1 and true 3 equally. Use uniform expected credit among tied minimum-distance **columns** for these endpoints, including models, and report tie frequency. Also report strict-tie exact three-way accuracy, the confusion matrix and errors by route order. Uniform guessing gives **1/3** exact accuracy. Always choosing 2 gives **1/2** on each four-row comparison, including its raw exact accuracy.

| Reader | Expected result or ceiling |
|---|---|
| Ending-only, empty/nonempty, fixed candidate position, amount prior, donor pile height/area alone | **1/2 on I, II and III**, regardless of which donor ending it prefers. |
| Initial count only, last-visible-count rule, initial pile area/size alone | **1/2 on all three**. A source prefix shared across routes gives exactly the same decision for opposite labels. |
| First transfer, last transfer, either single symbolic slot | **1/2 on all three**. Mirror slot order and report both orders. |
| Full ordered sequence, sign, net change, inbound/outbound counts, `k` of crossing moves, number of shuffles, any route-only lookup | **1/2 on I and II**, even with optimal decoding. On III a full-route, net-change or suitable move-count reader can score **1** because initial count is fixed. |
| Initial count plus one symbolic slot | Can score **1** on the corresponding initial-count comparison. Scores **1/2** on III and on the initial-count comparison where its slot is fixed. |
| Initial hopper occupancy, shared timing, nominal drop hints | **1/2** when constant within the block. With fixed initial hopper occupancy, its later symbolic occupancy is route information. An unmasked reader combining visible total and ending hopper can recover \(F\) and score **1**. Report that positive control and its masked counterpart. |
| Actual drop location/height, claw trajectory, visible object identity or repeat grasp, carried size, shadows, pile geometry, box-region RGB change | **Empirical.** Topmost selection and collision-aware placement can encode hidden occupancy. Score metadata and visible-pixel versions. Include last-context and initial-to-ending pixel L1, projected area/height matching, and initial area plus signed carried area or volume matched to donor endings. Test both shortcut directions. |
| State counter | **1** everywhere, including exact three-way. Any miss invalidates the manifest or scorer. |
| Parity or finite lookup | An initial-parity bit toggled on each tray crossing scores **1 on I–III**. Parity alone gives **1/2 on one versus three**. A lookup over the seven joint states can score **1 everywhere**. A perfect C6 result cannot identify the internal algorithm. |

**Inference.** Register `objlabel-100k` as primary. Report untrained and `plain-100k` on the identical blocks, with paired differences. Bootstrap 10,000 times over whole source–donor blocks, stratified by slot order, retaining every repeated cell, endpoint and model in the same draw. C5 used **231 unique donors for 398 sources**, with one donor used seven times. Its source-only interval is an optimistic planning reference. If unique matching fails, generate more candidate scenes before the freeze. Reusing donors after the freeze would require a new clustering plan.

**Proposed decision thresholds**, using lower and upper ends of two-sided 95% intervals:

- **Item validity.** Require the exact symbolic chance identities, a perfect state counter, and an upper bound below **0.55** on the mean of I and II for every registered plausible visual shortcut in its better direction. Freeze shortcut formulas on the pilot. Report per-comparison scores as well. A visually encoded shortcut that clears the tolerance invalidates the clean interpretation even if the JEPA beats it.
- **H-C6, joint use.** Require lower bounds above **0.55** on each of I and II, above **0.50** on III and on one versus three, and above zero for the paired advantage over each registered visual shortcut on the mean of I and II. All gates must pass.
- **H-C6+, stronger integration evidence.** Additionally require the lower bound of the equal-weight mean of I–III to exceed **2/3**. A fixed reader accessing any two of the three symbolic factors can win one comparison perfectly and scores chance on the other two. Its ceiling is \((1+.5+.5)/3=2/3\). The bound also covers input-independent mixtures of those readers. H-C6 alone permits such mixtures.
- **Power.** C5’s positive-only interval implies SE ≈0.010 and an optimistic 80%-power detectable lift of **2.8 percentage points** above a single cutoff. For 400 independent bounded block scores, the conservative normal approximation gives SE ≤0.025 and a detectable lift of **7 points**. Thus a true score around **0.58–0.62** is a planning range for clearing a 0.55 gate. Joint power is lower. C5’s **0.467** measures a different endpoint and supplies no expected C6 effect. If “400 scenes” means 200 sources plus 200 fresh donors, the conservative lift rises to **10 points**.

### 4. Interpretation and spend decision

| Outcome | Conclusion and next step |
|---|---|
| Validation, visibility, matching or shortcut gate fails | The item set admits a shortcut or obscures a required input. Repair that mechanism on fresh seeds and preregister again. No paid run. |
| Valid set, and an upper bound on I or II is ≤0.55 | This checkpoint falls short of the registered joint-use effect on that comparison. Use the failed cells to distinguish incoming-transfer, outgoing-transfer and initial-state errors. Retain C5’s event-reader interpretation. |
| I and II pass but III or one-versus-three fails | Report the specific capability demonstrated. A middle-versus-extremes result alone remains compatible with parity. Inspect round trips, slot order and extreme-count confusions locally. |
| H-C6 passes, H-C6+ does not | The checkpoint uses information from initial state and transfers across comparisons. Partial-cue mixtures remain viable. Prioritize a free held-out-route or appearance-transfer test. |
| H-C6+ passes on valid items | Evidence supports joint, count-sensitive prediction across independent scenes beyond the registered shortcut and mixture ceilings. **This is the trigger for a paid second 100k seed.** Freeze the same scoring plan for replication. A claim specifically about the object-grouping loss also needs the paired plain-model comparison and training-seed replication. |
| Intervals straddle a required cutoff | Inconclusive at the registered size. Report the interval and the limiting comparison. No post-score scene top-up. |

The strongest defensible claim is behavioral. Object-file tracking, a seven-state lookup and some continuous-quantity computations can produce the same answers as a hidden-number tracker. Generalization to unseen combinations and changed object-size statistics is what would distinguish those explanations.
---

## Pre-registration v1 (26 Nov). Freezes the draft above. Where this section and the draft differ, this section governs.

**Status.** Written before any trained checkpoint was run on a C6 item. The pilot used only the renderer, the scorer's metadata readers and an *untrained* network.

### Pilot findings (renderer-only, seeds 970000–977000)

- **Base filter.** Every accepted scene passed its checks: 11 of 11 sources and 11 of 11 donors, with no validation, prefix or masked-visibility rejects. The base filter is rare, at 0.55% of seeds, so the seed ranges below are wide.
- **Donor area match.** The largest/smallest post-reveal box-object pixel area is 1.02–1.20 (median about 1.10). The tolerance is frozen at **1.20**.
- **Pairing.** From an equal-sized donor pool, 7 of 11 sources matched. The full run uses a donor pool twice the size of the source set.
- **Scorer sanity.**
  - The state counter scores 1.0 on every endpoint.
  - Every single-feature symbolic reader scores exactly 0.50 on I and II.
  - Route, net, sign and in/out readers score 1.0 on III, as the draft predicts.
  - The untrained network scores 0.500 on I–III, with no ties.
  - The metadata area reader scores 0.46 on I–III, and the pixel readers score 0.50.
- **Design correction: 1-versus-3 is not chance-guarded.** With the extreme rows N1-, N1AR, N2R (final 1) and N2A (final 3), weighted equally by final, an **initial-count-only reader scores 0.83**. The same is true of a first-transfer or last-transfer reader, and route/net readers score 1.0. 1-versus-3 is therefore **reported, not a gate**, next to its reader ceilings. This change is to the draft's H-C6.

### Items (frozen)

| Role | Seeds | Accepted scenes |
|---|---|---|
| Sources | [1,000,000, 1,250,000) | 400 |
| Donors | [1,300,000, 1,700,000) | 800 |

- Training data uses seeds 0–99,999, and all earlier test sets used seeds below 1,000,000.
- Scenes use `configs/v4b_small.yaml`, train-split sampling rules and `scripts/c6_items.py` at the registered commit.
- **Pairing** uses `scripts/c6_manifest.py`:
  - box-token IoU ≥ 0.60;
  - identical ctx, first post-reveal slot and clip length;
  - donors with area ratio > 1.20 are dropped first;
  - one donor per source, without replacement, maximising total IoU.
- **Shortfall rule.** If fewer than 360 sources are matched, more donors are rendered from [1,700,000, 1,800,000) **before** the manifest is written. Nothing is topped up after the freeze.
- **Freeze.** The manifest records sha256 hashes for every scene file. The scorer refuses any file that changed.

### Models

| Role | Checkpoints |
|---|---|
| Primary | `objlabel-100k` (cloud-v4b-128-mask05-100k, weights_100000) |
| Registered comparisons | `plain-100k` (cw50-100k, weights_100000); `untrained` (the same architecture at random init, seed 0) |
| Exploratory | objlabel-30k, objlabel-30k-s1, plain-30k, track-30k |

### Endpoints and gates (scripts/c6_score.py; 10,000 block bootstraps stratified by slot order; two-sided 95%)

- **Validity (must hold, or the item set is invalid):**
  - The state counter scores 1.0 on I, II, III and exact.
  - Every single-feature symbolic reader ceiling is exactly 0.50 on I and II.
  - Each registered shortcut reader (`meta_area`, `pixel_last_ctx`, `pixel_first`) has an upper bound **< 0.55** on mean(I, II).
- **H-C6 (joint use) for objlabel-100k:**
  - lower bound > 0.55 on I, and on II;
  - lower bound > 0.50 on III;
  - lower bound > 0 on the paired difference over each registered shortcut on mean(I, II).
- **H-C6+:** H-C6, plus lower bound > 2/3 on mean(I, II, III).
- **Reported:** 1-versus-3 with its reader ceilings, exact 3-way (chance 1/3), confusion, tie rate, per-slot-order results, and objlabel-100k − plain-100k and − untrained paired differences.
- **Decisions:** as in the draft's interpretation table. A paid second 100k seed is triggered only by **H-C6+** on valid items.

## Pre-registration v2 (26 Nov, after Astra's code review; still before any trained checkpoint saw a C6 item)

This version supersedes v1 where they differ.

1. **Timeline.**
   - **Problem:** with the stock timeline, the second release fell *after* the last context frame in 42 of 49 pilot histories. The model often could not see where the second object went.
   - **Fix:** `retime()` moves the reveal so both transfers, releases included, end at least 1.5 frames inside the context. Every source now has ctx = 24 and its first post-reveal slot at 1.
   - **Caveat:** the C1–C5 reversal items used the stock timeline. Their second-transfer destination was often only partly visible in context, which limits what those results can show about the second transfer.
2. **Source sizes.** Each source scene draws its object scale independently, U(0.035, 0.065) with ±10% jitter, instead of inheriting the stock area-mode scale.
3. **Checks from the simulator, at model resolution.** Generator rejection totals are recorded in the manifest.
   - **Transfers:** each transfer's actual pre-grasp and settled post-release region must equal its schedule.
   - **Releases:** both releases must fall inside the context.
   - **Visibility:** each carried object must be visible outside the hopper mask at 128 px in context, first seen nearer its source tray and last seen nearer its destination tray (when they differ).
   - **State counter:** it is scored from the *simulated* finals, and the scorer refuses any block whose simulated and scheduled finals differ.
4. **Donor area.**
   - **Per-count calibration:** radius factors of 0.984, 1.0055 and 1.011 for 1, 2 and 3 objects, from a 64-donor renderer-only pilot.
   - **Result on fresh seeds (1.80–1.85M, 64 donors):** relative area 0.995, 0.993 and 1.012, each with SE about 0.009.
   - **Pairing gate:** area matching (≤ 1.20) is enforced on each pair's **common patches**, from post-reveal coverage at 128 px.
5. **Registered shortcut readers.** Each is scored under three frozen rules: nearest, farthest, and nearest with the binary answer inverted.
   - The readers are `meta_area`, `meta_volume` (signed volume, from the simulated transfers), `pixel_last_ctx` and `pixel_first`. The pixel readers use L1 to the donor's temporal-mean post-reveal image.
   - The validity gate (upper bound < 0.55 on mean(I, II)) and the paired-advantage gate (lower bound > 0) apply to **every reader × rule**.
6. **Count-equivalent positive control.** Astra's size-normalised quantity reader divides source area by the carried object's area, and reads donor count as (R1 / largest radius)². It scores 1.0, like the state counter.
   - Under area matching, "count" and "size-normalised quantity" are the same thing. **The claim is narrowed to tracking of the hidden count, or an equivalent size-normalised quantity.**
   - Because parity passes H-C6+, **H-C6+ supports replicating joint middle-versus-extremes sensitivity. It does not identify the algorithm.**
7. **Ties.**
   - One tie predicate (|d − min| ≤ 1e-12) for every endpoint.
   - Confusion matrices split tied credit.
   - Results are reported per slot order.
   - `add_first = seed mod 2`, and the bootstrap is stratified by it.
   - Column order is fixed at 1, 2, 3, since the target encoder never sees column position.
8. **Freeze.**
   - `items.jsonl` files are hashed alongside the NPZs.
   - Donor pools: registered [1.3M, 1.7M), 800 accepted; top-up [1.7M, 1.8M) only if fewer than 360 pairs.
   - `c6_manifest.py` refuses to freeze below 360 pairs. Between 360 and 399 pairs, all matches are used.
