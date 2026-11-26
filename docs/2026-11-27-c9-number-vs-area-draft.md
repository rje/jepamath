# C9 draft: source-side number versus carried area or volume (Astra design, 27 Nov; NOT yet registered)

> Status: draft from Astra's design brainstorm. A renderer-only pilot (32 blocks) and a pre-registration will come before any scoring.

**C9 should use a 2-versus-3 edge with the same starting box, plus two companion edges that swap the source quantities.** The conflict edge alone is insufficient. A reader that maps *less* source area to *more* donor spread can pass it. Swapping the actual area values between the two counts gives that reader an exact chance ceiling, even for a nonlinear donor readout.

### Histories

Use one shape per source block, balanced across spheres, boxes and cylinders. Let `a=.035` be an anchor, `s=.037` a small object and `b=.062` a big object. Keep the remove slot first. `BB` is a box shuffle, `BH` a removal and `HB` an addition.

| History | Initial box | Initial hopper | Slots | Final box |
|---|---|---|---|---|
| H, shared high endpoint | `a,s` | `s` | BB, HB | `a,s,s` |
| L-N | `a` | `b` | BB, HB | `a,b` |
| L-add | `a,b` | `s` | BB, BB | `a,b` |
| L-remove | `a,s` | `b` | BH, HB | `a,b` |

**L-remove → H is the decisive edge.** Give both rows identical initial box objects and poses. L-remove takes `s` out, then brings `b` in. H shuffles `s`, then brings another `s` in. Thus H gains one object but less carried area and volume. The two-object final has **28% more area and 95% more volume** than the three-object final. L-N and L-add supply the other omitted-input checks.

In `make_sources`, replace the shared-scale draw with these role-specific sizes and recompute mass. For L-N move the non-anchor box object to reserve. Use flat, separated placements and shared drop hints. On L-remove, `s` stands above `a` in the box, then `b` stands above the returned `s` in the hopper. The hopper peaks at two. Keep `retime()` and reject wrong **grasp IDs**, stacks, route errors or late releases. Require identical **hopper-masked** pre-transfer pixels on L-remove/H, where the visible prefix is intended to match. [Generator](/home/rje/src/python/jepamath/scripts/c6_items.py) · [Claw selection](/home/rje/src/python/jepamath/src/jepagen/backends/base.py:133)

### Exact chance controls

Render the same four histories twice more. Leave `a` and the initial `s` fixed. Change the lone `b` in each low final and the **incoming** `s` in H.

| Companion | Lone object in low final | Incoming object in H | Quantity exchanged exactly |
|---|---:|---:|---|
| Area | `√2 s = .05233` | `√(b²−s²) = .04975` | Sum of `πr²` |
| Volume | `∛2 s = .04662` | `∛(b³−s³) = .05725` | True shape volume |

Generate from the formulas at full precision. Both replacement hopper objects in L-remove still exceed `.037`.

Use the **same donor endings and common patches** for the original and both companions. Score each L→H edge by the rise in `p = d(2) − d(3)`, with half credit for ties. For each family register three endpoints: conflict alone, mean(conflict, area companion), and mean(conflict, volume companion).

- A count tracker scores **1** throughout. For any reader `p=f(final source area, fixed donor)`, the area-paired edge differences are exact negatives, giving **0.50**. The volume pair gives the identical guarantee for volume. This covers either polarity, saturation, and an arbitrary donor readout including spread or radius.
- N-only ties on add and remove. Add-only and remove-only each tie on the other input. A complete route-only reader ties on N. The identical masked prefix makes a first-frame-only reader tie on the decisive conflict edge.
- The proof covers each **scalar** separately. Joint area-plus-volume, size-distribution readers and arbitrary pixel histories need separate treatment. In particular, source summed radii rise with count on the conflict edge because `2s>b`. A pass distinguishes number from either registered source total, without uniquely identifying counting.

### Donors, pilot and decision

Use **two donor pairs per block**. In one, give the two-object ending `.061` objects placed wide and the three-object ending `.037` objects placed compactly. In the other, exchange the size and layout assignments. This reverses area, volume, individual radius, summed radii and intended spread; even `2×.061 > 3×.037`. Use natural BB/BB endings. Replace C8’s ≤1.20 area-ratio filter with **opposite measured signs** for metadata area, common-patch pixel area, volume, mean/max/summed radius and projected RMS spread. Verify after both shuffles. Score every source against both pairs equally. The scalar chance proof survives even if a donor retains a perfect count cue.

- **Pilot:** 32 complete blocks, renderer and CPU readers only. Check actual IDs, masked carry visibility at 128 px, common-patch IoU ≥.60, donor signs and scalar swaps. Measure source pixel area from the visible initial box plus signed carried-object mask areas at a frozen carry phase. Require the predicted conflict direction with ≥10% pixel margin in ≥80% of pilot blocks. Freeze the phase, seed windows and rejection rules before confirmation.
- **Confirmation:** 240 fresh independent blocks, 80 per shape, each with 12 source clips and four donor clips. Hash before checkpoint scoring. Bootstrap whole blocks 10,000 times within shape. Require simulated count = 1.00 and symbolic scalar controls = .50 on their respective paired endpoints. Gate measured pixel-area × donor-cue readers in both polarities, plus first/last-frame readers: each must have a 95% upper bound <.55 on its weakest registered endpoint.
- **Pass:** frozen objlabel-100k has a 95% lower bound >.55 on all three conflict edges and on both paired endpoints for each family, with paired advantage over untrained >0 on each family’s conflict edge. Report both donor polarities separately. Local render and CPU inference only.

**Size verdict:** the trained range gives a substantial physical reversal. The uncertainty is whether the masked 128 px carry preserves it. If the pilot fails there, drop the anchor and run L-remove/H as **one big versus two small**, with the same scalar swaps. The area margin rises to **40%** and volume to **135%**. That fallback tests the source quantity cleanly, but singleton centroid spread is structurally zero, so drop the donor-spread reversal claim. These are calculated margins; no render or checkpoint result is assumed.