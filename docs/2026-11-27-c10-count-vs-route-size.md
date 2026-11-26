# C10: is P1's count really the count? Route, start and size controls on the frozen probe (pre-registration, 27 Nov)

## Why

P1/P2 show that a ridge probe, fit on C8 and frozen, orders 3 vs 2 objects from objlabel-100k's predicted hidden ending at 0.99 on C9. Astra's review (27 Nov, scratchpad astra_r2.md) names three cheaper explanations for that number:

1. **Route recognition.** In C9 the 3-object history always begins with a box shuffle, and the 2-object conflict history begins with a removal. The probe could be reading the move pattern.
2. **A residual size cue.** On C9's original conflict edge, summed radius still rises with count.
3. **Unfair encoder pooling.** That is a P2 matter and is not tested here.

C10 tests the first two on fresh scenes with the probe frozen. No training is involved.

## Items (`scripts/c10_items.py`)

One base scene uses the C6/C9 machinery: occluded, two transfers, retimed so both releases fall inside the context, and the hopper masked for scoring. Each scene gets **17 histories**, crossing the starting count n0 with the two-move route:

| Net change | Routes | n0 | Final |
|---|---|---|---|
| 0 | BB-BB, HB-BH, BH-HB | 1, 2, 3 | n0 |
| +1 | BB-HB, HB-BB | 1, 2 | n0 + 1 |
| −1 | BH-BB, BB-BH | 2, 3 | n0 − 1 |

**Objects.** There are three box objects (n0 uses the first n0, the rest go to reserve) at fixed slots, one hopper object, and one reserve. One shape per scene (seed mod 3), with 80 scenes per shape and 240 in all.

**Sizes (amendment 1).** One size class per history, small U(0.033, 0.038) or big U(0.058, 0.062), each with probability 1/2, independent of n0 and route (one shared pool made E4 impossible). So summed radius, area and volume often disagree with the count. The claw takes the tallest object, so the box contents are read from the simulation (`final_objects`), not from the schedule.

**Checks.** As C9 sources: validate_clip; simulator transfer records (actual src/dst, release inside the context, carried object visible outside the hopper mask, src/dst cues); one ctx and first post slot per scene.

**Seeds.** [8,000,000, 8,500,000), disjoint from training and from C1–C9, C9b and D1.

## Probe (frozen)

As P1, for each model: a ridge on mean-pooled predicted post-reveal box tokens, fit on all C8 sources (alpha by 5-fold grouped CV), then applied unchanged to C10. C10 never enters fitting.

## Endpoints

An edge is a within-scene pair (a, b) with final(a) > final(b). It is correct when the probe count c(a) > c(b), with half credit for a tie within 1e-12. Edges are averaged within a scene, then over scenes. The bootstrap uses 10,000 scene resamples stratified by shape.

| Endpoint | Edges | Which reader is defeated |
|---|---|---|
| **E1 same-route** | same route, different n0 | route / net-change reader = 0.50 (ties) |
| **E2 route-opposed** | net(a) < net(b) | net-change reader = 0 |
| **E3 start-opposed** | n0(a) < n0(b) | start-count reader = 0 |
| **E4 size-opposed** | summed radius, area and volume of the final box contents all larger for b | every size reader = 0 |

**Validity (or the items are invalid).**

- The state counter (simulated final count) scores 1.0 on E1–E4.
- The net reader scores 0.50 on E1 and 0 on E2.
- The start reader scores 0 on E3.
- The size readers score 0 on E4.
- E4 has ≥ 1 edge in at least 150 of 240 scenes.

**Reported.**

- **E5 equal-count spread:** the mean |c(a) − c(b)| over same-final, different-route pairs, divided by the mean c(a) − c(b) over final-differs-by-1 pairs. 0 means the route does not move the count.
- Per n0 and per route means of c.

## Hypotheses

Models are objlabel-100k and objlabel-100k-s1, both weights_100000.

**H-C10.** For both seeds, all of the following hold:

- the lower bound is > 0.60 on each of E1, E2, E3 and E4;
- the paired lower bound over **untrained** is > 0 on each.

**Reported.** plain-100k, untrained, and the D1 arms (treat, ctrl), which doubles as D1's registered P1 report.

**Reading.**

- **Pass:** P1's count survives route, start and size shortcuts. The claim becomes "the predicted ending linearly encodes the post-transfer count, beyond route, start count and carried size".
- **E1 passes, E2/E3 fail:** the probe tracks one input, not the integrated count.
- **E4 fails alone:** a size cue remains in the readout.
