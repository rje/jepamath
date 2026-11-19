//! Physics world: rapier3d bodies for the table, trays and objects; kinematic claw and screen.
//! Mechanics are modules implementing `Mechanic`; the world calls them every physics step.

use rapier3d::glamx::glam::{Quat, Vec3};
use rapier3d::prelude::*;
use serde::Serialize;

use crate::rules::{self, ClawCmd, ClawController, Event, PoseQ, Region};
use crate::spec::{Episode, Shape, Tray, V3};

/// A mechanic drives kinematic parts of the scene and may issue commands to the world.
/// New mechanics (chute, conveyor, stacking, ...) implement this and get registered in `World::new`.
pub trait Mechanic {
    fn name(&self) -> &'static str;
    /// Called before each physics step with the time at the END of that step.
    fn pre_step(&mut self, ep: &Episode, t: f64, world: &mut Physics) -> Result<(), String>;
    fn events(&self) -> Vec<Event> {
        vec![]
    }
}

pub struct Physics {
    pub pw: PhysicsWorld,
    pub objects: Vec<RigidBodyHandle>,
    pub claw: RigidBodyHandle,
    pub screen: RigidBodyHandle,
    /// held object id and its fixed offset from the claw
    pub grip: Option<(usize, Vec3)>,
}

impl Physics {
    pub fn poses(&self) -> Vec<PoseQ> {
        self.objects
            .iter()
            .map(|h| {
                let b = &self.pw.bodies[*h];
                let t = b.translation();
                let r = b.rotation();
                ([t.x as f64, t.y as f64, t.z as f64], [r.w as f64, r.x as f64, r.y as f64, r.z as f64])
            })
            .collect()
    }

    pub fn held(&self) -> Vec<usize> {
        self.grip.iter().map(|g| g.0).collect()
    }

    /// Grasp: the object becomes kinematic and follows the claw rigidly at its current offset
    /// (no joint compliance, so it cannot sag into the pile before release).
    fn grasp(&mut self, oid: usize) -> Result<(), String> {
        if self.grip.is_some() {
            return Err("grasp while already holding".into());
        }
        let claw_t = self.pw.bodies[self.claw].translation();
        let b = &mut self.pw.bodies[self.objects[oid]];
        let offset = b.translation() - claw_t;
        b.set_body_type(RigidBodyType::KinematicPositionBased, true);
        self.grip = Some((oid, offset));
        Ok(())
    }

    fn release(&mut self, oid: usize) -> Result<(), String> {
        match self.grip.take() {
            Some((g, _)) if g == oid => {
                let b = &mut self.pw.bodies[self.objects[oid]];
                b.set_body_type(RigidBodyType::Dynamic, true);
                b.set_linvel(Vec3::ZERO, true);
                b.set_angvel(Vec3::ZERO, true);
                Ok(())
            }
            other => Err(format!("release of {oid} but holding {:?}", other.map(|g| g.0))),
        }
    }

    pub fn place(&mut self, oid: usize, p: V3) {
        let b = &mut self.pw.bodies[self.objects[oid]];
        b.set_translation(Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32), true);
        b.set_rotation(Quat::IDENTITY, true);
        b.set_linvel(Vec3::ZERO, true);
        b.set_angvel(Vec3::ZERO, true);
    }
}

pub struct ClawMechanic {
    pub ctl: ClawController,
}

impl Mechanic for ClawMechanic {
    fn name(&self) -> &'static str {
        "claw"
    }
    fn pre_step(&mut self, ep: &Episode, t: f64, w: &mut Physics) -> Result<(), String> {
        let poses_fn = || w.poses();
        let (pos, cmds) = self.ctl.update(ep, t, &poses_fn)?;
        for c in cmds {
            match c {
                ClawCmd::Grasp(o) => w.grasp(o)?,
                ClawCmd::Release(o) => w.release(o)?,
            }
        }
        let target = Vec3::new(pos[0] as f32, pos[1] as f32, pos[2] as f32);
        w.pw.bodies[w.claw].set_next_kinematic_translation(target);
        if let Some((oid, offset)) = w.grip {
            w.pw.bodies[w.objects[oid]].set_next_kinematic_translation(target + offset);
        }
        Ok(())
    }
    fn events(&self) -> Vec<Event> {
        self.ctl.events.clone()
    }
}

pub struct ScreenMechanic;

impl Mechanic for ScreenMechanic {
    fn name(&self) -> &'static str {
        "screen"
    }
    fn pre_step(&mut self, ep: &Episode, t: f64, w: &mut Physics) -> Result<(), String> {
        let s = &ep.scene.screen;
        let z = rules::screen_top_z(ep, t) - s.height / 2.0;
        w.pw.bodies[w.screen].set_next_kinematic_translation(Vec3::new(s.center_x as f32, s.y as f32, z as f32));
        Ok(())
    }
}

pub const FRICTION: f32 = 0.9;
/// rapier has no rolling friction; angular damping stands in for it so spheres come to rest.
pub const ANGULAR_DAMPING: f32 = 2.0;
pub const LINEAR_DAMPING: f32 = 0.05;

fn fixed_box(pw: &mut PhysicsWorld, center: V3, half: V3) {
    let c = ColliderBuilder::cuboid(half[0] as f32, half[1] as f32, half[2] as f32)
        .translation(Vec3::new(center[0] as f32, center[1] as f32, center[2] as f32))
        .friction(FRICTION)
        .restitution(0.0);
    pw.insert_collider(c, None);
}

/// Tray geometry shared with the renderer: (center, half extents) for floor + 4 walls.
pub fn tray_boxes(tr: &Tray) -> Vec<(V3, V3)> {
    let (cx, cy) = (tr.center[0], tr.center[1]);
    let (hx, hy) = (tr.inner_size[0] / 2.0, tr.inner_size[1] / 2.0);
    let (t, h) = (tr.wall_thickness, tr.wall_height);
    vec![
        ([cx, cy, t / 2.0], [hx + t, hy + t, t / 2.0]),
        ([cx, cy + hy + t / 2.0, h / 2.0], [hx + t, t / 2.0, h / 2.0]),
        ([cx, cy - hy - t / 2.0, h / 2.0], [hx + t, t / 2.0, h / 2.0]),
        ([cx + hx + t / 2.0, cy, h / 2.0], [t / 2.0, hy, h / 2.0]),
        ([cx - hx - t / 2.0, cy, h / 2.0], [t / 2.0, hy, h / 2.0]),
    ]
}

pub struct World {
    pub ep: Episode,
    pub phys: Physics,
    pub mechanics: Vec<Box<dyn Mechanic>>,
    pub t: f64,
    pub events: Vec<Event>,
}

#[derive(Serialize)]
pub struct ObjState {
    pub id: usize,
    pub pos: V3,
    pub quat: [f64; 4],
    pub lin_vel: V3,
    pub ang_vel: V3,
    pub region: &'static str,
}

#[derive(Serialize)]
pub struct FrameState {
    pub t: f64,
    pub count_per_region: serde_json::Map<String, serde_json::Value>,
    pub objects: Vec<ObjState>,
    pub claw_pos: V3,
    pub screen_top_z: f64,
    pub active_action: Option<&'static str>,
}

impl World {
    pub fn new(ep: Episode) -> Result<Self, String> {
        let mut pw = PhysicsWorld::new();
        pw.gravity = Vec3::new(0.0, 0.0, -9.81);
        pw.integration_parameters.dt = ep.physics.timestep as f32;

        let sc = &ep.scene;
        fixed_box(&mut pw, [0.0, 0.0, -0.025], [sc.table_size[0], sc.table_size[1], 0.025]);
        for tr in [&sc.box_, &sc.hopper] {
            for (c, h) in tray_boxes(tr) {
                fixed_box(&mut pw, c, h);
            }
        }
        let n = ep.objects.len() as f64;
        fixed_box(&mut pw, [(n - 1.0) * 0.15, 0.0, rules::RESERVE_Z - 0.05], [n * 0.15 + 0.3, 0.3, 0.05]);

        let mut objects = vec![];
        for o in &ep.objects {
            let q = o.start_quat;
            let rb = RigidBodyBuilder::dynamic()
                .translation(Vec3::new(o.start_pos[0] as f32, o.start_pos[1] as f32, o.start_pos[2] as f32))
                .rotation(Quat::from_xyzw(q[1] as f32, q[2] as f32, q[3] as f32, q[0] as f32).to_scaled_axis())
                .angular_damping(ANGULAR_DAMPING)
                .linear_damping(LINEAR_DAMPING);
            let h = pw.insert_body(rb);
            let s = &o.size;
            let col = match o.shape {
                Shape::Sphere => ColliderBuilder::ball(s[0] as f32),
                Shape::Box => ColliderBuilder::cuboid(s[0] as f32, s[1] as f32, s[2] as f32),
                // parry cylinders are y-aligned; ours are z-aligned
                Shape::Cylinder => ColliderBuilder::cylinder(s[1] as f32, s[0] as f32)
                    .rotation(Vec3::new(std::f32::consts::FRAC_PI_2, 0.0, 0.0)),
            }
            .mass(o.mass as f32)
            .friction(FRICTION)
            .restitution(0.0);
            pw.insert_collider(col, Some(h));
            objects.push(h);
        }
        let home = ep.physics.claw_home;
        let claw = pw.insert_body(
            RigidBodyBuilder::kinematic_position_based().translation(Vec3::new(home[0] as f32, home[1] as f32, home[2] as f32)),
        );
        let s = &sc.screen;
        let screen = pw.insert_body(RigidBodyBuilder::kinematic_position_based().translation(Vec3::new(
            s.center_x as f32,
            s.y as f32,
            (rules::SCREEN_DOWN_TOP_Z - s.height / 2.0) as f32,
        )));

        let phys = Physics { pw, objects, claw, screen, grip: None };
        let mechanics: Vec<Box<dyn Mechanic>> =
            vec![Box::new(ScreenMechanic), Box::new(ClawMechanic { ctl: ClawController::new(&ep) })];
        let mut w = World { ep, phys, mechanics, t: 0.0, events: vec![] };

        // preroll: settle with everything parked, unrendered
        let n_pre = (w.ep.physics.preroll / w.ep.physics.timestep).round() as usize;
        for _ in 0..n_pre {
            w.phys.pw.step();
        }
        w.t = 0.0;
        w.run_mechanics(0.0)?;
        Ok(w)
    }

    fn run_mechanics(&mut self, t: f64) -> Result<(), String> {
        for m in self.mechanics.iter_mut() {
            m.pre_step(&self.ep, t, &mut self.phys).map_err(|e| format!("{}: {e}", m.name()))?;
        }
        Ok(())
    }

    pub fn step_to(&mut self, t: f64) -> Result<(), String> {
        let dt = self.ep.physics.timestep;
        while self.t + dt / 2.0 < t {
            self.run_mechanics(self.t + dt)?;
            self.phys.pw.step();
            self.t += dt;
        }
        Ok(())
    }

    pub fn apply_violation(&mut self) -> Result<(), String> {
        let v = self.ep.violation.clone().ok_or("no violation in this episode")?;
        let oid = match v.kind.as_str() {
            "add" => {
                let oid = v.object_id.ok_or("add violation without object_id")?;
                let hint = v.pos.ok_or("add violation without pos")?;
                let p = rules::violation_add_pos(&self.ep, oid, hint, &self.phys.poses())?;
                self.phys.place(oid, p);
                oid
            }
            "delete" => {
                let oid = rules::pick_topmost(&self.ep, Region::Box, &self.phys.poses(), &self.phys.held())?;
                self.phys.place(oid, [oid as f64 * 0.3, 0.0, rules::RESERVE_Z + 0.1]);
                oid
            }
            k => return Err(format!("unknown violation kind {k}")),
        };
        self.events.push(Event { t: self.t, event: format!("violation_{}", v.kind), object_id: oid });
        Ok(())
    }

    pub fn all_events(&self) -> Vec<Event> {
        let mut ev: Vec<Event> = self.mechanics.iter().flat_map(|m| m.events()).collect();
        ev.extend(self.events.iter().cloned());
        ev.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
        ev
    }

    pub fn claw_pos(&self) -> V3 {
        let t = self.phys.pw.bodies[self.phys.claw].translation();
        [t.x as f64, t.y as f64, t.z as f64]
    }

    pub fn frame_state(&self) -> FrameState {
        let held = self.phys.held();
        let mut counts = [0usize; 5];
        let mut objs = vec![];
        for (oid, (pos, quat)) in self.phys.poses().into_iter().enumerate() {
            let region = rules::classify_region(&self.ep, pos, held.contains(&oid));
            counts[Region::ALL.iter().position(|r| *r == region).unwrap()] += 1;
            let b = &self.phys.pw.bodies[self.phys.objects[oid]];
            let (lv, av) = (b.linvel(), b.angvel());
            objs.push(ObjState {
                id: oid,
                pos,
                quat,
                lin_vel: [lv.x as f64, lv.y as f64, lv.z as f64],
                ang_vel: [av.x as f64, av.y as f64, av.z as f64],
                region: region.name(),
            });
        }
        let mut cpr = serde_json::Map::new();
        for (i, r) in Region::ALL.iter().enumerate() {
            cpr.insert(r.name().into(), counts[i].into());
        }
        let active = self.ep.actions.iter().filter(|a| a.t_start() <= self.t && self.t < a.t_end()).last().map(|a| a.kind());
        FrameState {
            t: self.t,
            count_per_region: cpr,
            objects: objs,
            claw_pos: self.claw_pos(),
            screen_top_z: rules::screen_top_z(&self.ep, self.t),
            active_action: active,
        }
    }

    /// Deepest current penetration in metres (>= 0), recomputed exactly at the current poses for
    /// every pair the narrow phase tracks (manifold distances can be stale due to contact recycling).
    pub fn penetration(&self) -> Result<f64, String> {
        let mut worst = 0.0f32;
        let cs = &self.phys.pw.colliders;
        for pair in self.phys.pw.narrow_phase.contact_pairs() {
            let (c1, c2) = (&cs[pair.collider1], &cs[pair.collider2]);
            let c = rapier3d::parry::query::contact(c1.position(), c1.shape(), c2.position(), c2.shape(), 0.0)
                .map_err(|_| "unsupported shape pair in penetration query".to_string())?;
            if let Some(c) = c {
                worst = worst.max(-c.dist);
            }
        }
        Ok(worst as f64)
    }
}
