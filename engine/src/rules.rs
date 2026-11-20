//! Shared mechanics rules. Must match `jepagen/backends/base.py` exactly: these define what
//! an episode *means* (which object the claw picks, where it drops, when the screen hides
//! the box, which region an object is in). Physics may differ between backends; these may not.

use crate::spec::{Action, Episode, Shape, Tray, TrayName, V2, V3, V4};

pub const CLAW_PAD_HALF: V3 = [0.022, 0.022, 0.006];
pub const CLAW_ROD_RADIUS: f64 = 0.006;
pub const CLAW_ROD_LEN: f64 = 0.6;
pub const RESERVE_Z: f64 = -3.0;
pub const CLAW_RETREAT_S: f64 = 0.6;
pub const RELEASE_GAP: f64 = 0.008;
pub const DROP_GRID: usize = 9;
pub const SCREEN_DOWN_TOP_Z: f64 = -0.06;

pub const SEG_BG: u8 = 0;
pub const SEG_CLAW: u8 = 254;
pub const SEG_SCREEN: u8 = 253;

pub const SEGMENTS: [(&str, f64); 8] = [
    ("approach", 0.20),
    ("descend", 0.17),
    ("grasp", 0.05),
    ("lift", 0.14),
    ("carry", 0.20),
    ("lower", 0.16),
    ("settle", 0.04),
    ("release", 0.04),
];
pub const VIOLATION_ADD_GAP: f64 = 0.002;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    Box,
    Hopper,
    Reserve,
    Claw,
    Table,
}

impl Region {
    pub fn name(&self) -> &'static str {
        match self {
            Region::Box => "box",
            Region::Hopper => "hopper",
            Region::Reserve => "reserve",
            Region::Claw => "claw",
            Region::Table => "table",
        }
    }
    pub const ALL: [Region; 5] = [Region::Box, Region::Hopper, Region::Reserve, Region::Claw, Region::Table];
}

pub type PoseQ = (V3, V4); // position, quaternion (w, x, y, z)

pub fn smoothstep(u: f64) -> f64 {
    let u = u.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

pub fn lerp3(a: V3, b: V3, s: f64) -> V3 {
    [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s]
}

pub fn screen_top_z(ep: &Episode, t: f64) -> f64 {
    let (down, up) = (SCREEN_DOWN_TOP_Z, ep.scene.screen.height);
    let mut z = down;
    let mut moves: Vec<&Action> =
        ep.actions.iter().filter(|a| matches!(a, Action::ScreenRaise { .. } | Action::ScreenLower { .. })).collect();
    moves.sort_by(|a, b| a.t_start().partial_cmp(&b.t_start()).unwrap());
    for a in moves {
        if t < a.t_start() {
            break;
        }
        let s = smoothstep((t - a.t_start()) / (a.t_end() - a.t_start()));
        z = match a {
            Action::ScreenRaise { .. } => down + (up - down) * s,
            _ => up + (down - up) * s,
        };
    }
    z
}

pub fn in_tray(tray: &Tray, p: V3) -> bool {
    let (hx, hy) = (tray.inner_size[0] / 2.0, tray.inner_size[1] / 2.0);
    (p[0] - tray.center[0]).abs() <= hx
        && (p[1] - tray.center[1]).abs() <= hy
        && p[2] >= -0.005
        && p[2] <= tray.wall_height + 0.25
}

pub fn classify_region(ep: &Episode, p: V3, held: bool) -> Region {
    if held {
        Region::Claw
    } else if p[2] < RESERVE_Z + 1.0 {
        Region::Reserve
    } else if in_tray(&ep.scene.box_, p) {
        Region::Box
    } else if in_tray(&ep.scene.hopper, p) {
        Region::Hopper
    } else {
        Region::Table
    }
}

fn tray_region(t: TrayName) -> Region {
    match t {
        TrayName::Box => Region::Box,
        TrayName::Hopper => Region::Hopper,
    }
}

pub fn z_extent(shape: Shape, size: &[f64], q: V4) -> f64 {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let r20 = 2.0 * (x * z - w * y);
    let r21 = 2.0 * (y * z + w * x);
    let r22 = 1.0 - 2.0 * (x * x + y * y);
    match shape {
        Shape::Sphere => size[0],
        Shape::Box => r20.abs() * size[0] + r21.abs() * size[1] + r22.abs() * size[2],
        Shape::Cylinder => r22.abs() * size[1] + size[0] * (1.0 - r22 * r22).max(0.0).sqrt(),
    }
}

pub fn horiz_radius(shape: Shape, size: &[f64]) -> f64 {
    match shape {
        Shape::Box => size[0].hypot(size[1]),
        _ => size[0],
    }
}

/// The object in `region` whose top is highest; ties -> lowest id.
pub fn pick_topmost(ep: &Episode, region: Region, poses: &[PoseQ], held: &[usize]) -> Result<usize, String> {
    let mut best: Option<(f64, usize)> = None;
    for (oid, (pos, quat)) in poses.iter().enumerate() {
        if held.contains(&oid) || classify_region(ep, *pos, false) != region {
            continue;
        }
        let o = &ep.objects[oid];
        let top = pos[2] + z_extent(o.shape, &o.size, *quat);
        match best {
            Some((bt, _)) if !(top > bt + 1e-9) => {}
            _ => best = Some((top, oid)),
        }
    }
    best.map(|b| b.1).ok_or_else(|| format!("no object in region {} to pick", region.name()))
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Event {
    pub t: f64,
    pub event: String,
    pub object_id: usize,
}

pub enum ClawCmd {
    Grasp(usize),
    Release(usize),
}

/// Shared drop rule (claw releases and hidden 'add' violations); see base.py `choose_drop`.
pub fn choose_drop(ep: &Episode, dst_name: TrayName, oid: usize, drop_xy: V2, poses: &[PoseQ]) -> Result<V3, String> {
    let dst = ep.tray(dst_name);
    let o = &ep.objects[oid];
    let rh = horiz_radius(o.shape, &o.size);
    let mut others = vec![];
    for (other, (p, q)) in poses.iter().enumerate() {
        if other == oid || classify_region(ep, *p, false) != tray_region(dst_name) {
            continue;
        }
        let ot = &ep.objects[other];
        others.push((*p, horiz_radius(ot.shape, &ot.size), p[2] + z_extent(ot.shape, &ot.size, *q)));
    }
    let hx = dst.inner_size[0] / 2.0 - rh - 0.005;
    let hy = dst.inner_size[1] / 2.0 - rh - 0.005;
    if hx <= 0.0 || hy <= 0.0 {
        return Err("object too large for destination tray".into());
    }
    let mut best: Option<((i64, f64, usize), V3)> = None;
    let g = DROP_GRID;
    for i in 0..g {
        for j in 0..g {
            let x = dst.center[0] - hx + 2.0 * hx * i as f64 / (g - 1) as f64;
            let y = dst.center[1] - hy + 2.0 * hy * j as f64 / (g - 1) as f64;
            let mut surf = dst.wall_thickness;
            for (p, r, top) in &others {
                if (p[0] - x).hypot(p[1] - y) < rh + r {
                    surf = surf.max(*top);
                }
            }
            let key = (round_half_even(surf / 0.001), (x - drop_xy[0]).hypot(y - drop_xy[1]), i * g + j);
            let better = match &best {
                None => true,
                Some((bk, _)) => key.0 < bk.0 || (key.0 == bk.0 && (key.1 < bk.1 || (key.1 == bk.1 && key.2 < bk.2))),
            };
            if better {
                best = Some((key, [x, y, surf]));
            }
        }
    }
    Ok(best.unwrap().1)
}

/// Where a hidden 'add' places its object: the drop rule over the box, resting just above the surface.
pub fn violation_add_pos(ep: &Episode, oid: usize, hint: V3, poses: &[PoseQ]) -> Result<V3, String> {
    let o = &ep.objects[oid];
    let d = choose_drop(ep, TrayName::Box, oid, [hint[0], hint[1]], poses)?;
    Ok([d[0], d[1], d[2] + z_extent(o.shape, &o.size, [1.0, 0.0, 0.0, 0.0]) + VIOLATION_ADD_GAP])
}

/// Where a hidden 'move' relocates the topmost box object `oid`: the drop rule aimed at the box
/// quadrant centre farthest from the object's current position (ties -> lowest quadrant index).
pub fn violation_move_pos(ep: &Episode, oid: usize, poses: &[PoseQ]) -> Result<V3, String> {
    let b = &ep.scene.box_;
    let (qx, qy) = (b.inner_size[0] / 4.0, b.inner_size[1] / 4.0);
    let p = poses[oid].0;
    let mut best: Option<(f64, V2)> = None;
    for (sx, sy) in [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
        let c = [b.center[0] + sx * qx, b.center[1] + sy * qy];
        let d = (c[0] - p[0]).hypot(c[1] - p[1]);
        if best.map(|(bd, _)| d > bd + 1e-12).unwrap_or(true) {
            best = Some((d, c));
        }
    }
    let hint = best.unwrap().1;
    violation_add_pos(ep, oid, [hint[0], hint[1], 0.0], poses)
}

pub struct ClawController {
    home: V3,
    travel_z: f64,
    transfers: Vec<(TrayName, TrayName, f64, f64, V2)>,
    pub pos: V3,
    seg_key: Option<(i64, i64)>,
    seg_from: V3,
    seg_to: V3,
    target_oid: Option<usize>,
    pub held: Option<usize>,
    drop: Option<V3>,
    pub events: Vec<Event>,
}

impl ClawController {
    pub fn new(ep: &Episode) -> Self {
        let mut transfers: Vec<_> = ep
            .actions
            .iter()
            .filter_map(|a| match a {
                Action::Transfer { src, dst, t_start, t_end, drop_xy } => Some((*src, *dst, *t_start, *t_end, *drop_xy)),
                _ => None,
            })
            .collect();
        transfers.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        let home = ep.physics.claw_home;
        ClawController {
            home,
            travel_z: ep.physics.claw_travel_z,
            transfers,
            pos: home,
            seg_key: None,
            seg_from: home,
            seg_to: home,
            target_oid: None,
            held: None,
            drop: None,
            events: vec![],
        }
    }

    fn resolve_target(&mut self, ep: &Episode, name: &str, tr: usize, poses: &[PoseQ]) -> Result<V3, String> {
        let (src, dst, _, _, drop_xy) = self.transfers[tr];
        Ok(match name {
            "approach" => {
                let oid = pick_topmost(ep, tray_region(src), poses, &[])?;
                self.target_oid = Some(oid);
                let p = poses[oid].0;
                [p[0], p[1], self.travel_z]
            }
            "descend" => {
                let oid = self.target_oid.unwrap();
                let (p, q) = poses[oid];
                let o = &ep.objects[oid];
                [p[0], p[1], p[2] + z_extent(o.shape, &o.size, q) + CLAW_PAD_HALF[2] + 0.001]
            }
            "grasp" | "settle" | "release" => self.pos,
            "lift" => [self.pos[0], self.pos[1], self.travel_z],
            "carry" => {
                let d = choose_drop(ep, dst, self.held.ok_or("carry without a held object")?, drop_xy, poses)?;
                self.drop = Some(d);
                [d[0], d[1], self.travel_z]
            }
            "lower" => {
                let held = self.held.unwrap();
                let o = &ep.objects[held];
                let (_, q) = poses[held];
                let ext = z_extent(o.shape, &o.size, q);
                let claw_offset = ext + CLAW_PAD_HALF[2] + 0.001; // nominal, as set up at grasp
                let d = self.drop.unwrap();
                [d[0], d[1], d[2] + RELEASE_GAP + ext + claw_offset]
            }
            _ => unreachable!(),
        })
    }

    /// Advance to time t. `poses` is called lazily for live object poses. Returns the claw
    /// position and any grasp/release commands to apply before stepping physics.
    pub fn update(&mut self, ep: &Episode, t: f64, poses: &dyn Fn() -> Vec<PoseQ>) -> Result<(V3, Vec<ClawCmd>), String> {
        let mut cmds = vec![];
        for i in 0..self.transfers.len() {
            let (_, _, ts, te, _) = self.transfers[i];
            if ts <= t && t < te {
                let u = (t - ts) / (te - ts);
                let mut acc = 0.0;
                for (j, (name, frac)) in SEGMENTS.iter().enumerate() {
                    if u < acc + frac || j == SEGMENTS.len() - 1 {
                        let key = (i as i64, j as i64);
                        if self.seg_key != Some(key) {
                            self.seg_key = Some(key);
                            self.seg_from = self.pos;
                            if *name == "grasp" {
                                let oid = self.target_oid.unwrap();
                                self.held = Some(oid);
                                cmds.push(ClawCmd::Grasp(oid));
                                self.events.push(Event { t, event: "grasp".into(), object_id: oid });
                            }
                            if *name == "release" {
                                let oid = self.held.take().unwrap();
                                cmds.push(ClawCmd::Release(oid));
                                self.events.push(Event { t, event: "release".into(), object_id: oid });
                            }
                            let p = poses();
                            self.seg_to = self.resolve_target(ep, name, i, &p)?;
                        }
                        let s = smoothstep((u - acc) / frac);
                        self.pos = lerp3(self.seg_from, self.seg_to, s);
                        return Ok((self.pos, cmds));
                    }
                    acc += frac;
                }
            }
        }
        let done: Vec<_> = self.transfers.iter().filter(|tr| tr.3 <= t).collect();
        if self.held.is_some() && !done.is_empty() {
            return Err("transfer ended with an object still held".into());
        }
        if let Some(last) = done.last() {
            let key = (-1, done.len() as i64);
            if self.seg_key != Some(key) {
                self.seg_key = Some(key);
                self.seg_from = self.pos;
            }
            let s = smoothstep((t - last.3) / CLAW_RETREAT_S);
            self.pos = lerp3(self.seg_from, self.home, s);
        }
        Ok((self.pos, cmds))
    }
}

fn round_half_even(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        let f = x.floor();
        if (f as i64) % 2 == 0 { f as i64 } else { f as i64 + 1 }
    } else {
        r as i64
    }
}
