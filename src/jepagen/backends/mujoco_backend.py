"""MuJoCo reference backend. Rendering via EGL (headless GPU)."""

from __future__ import annotations

import math

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ..spec import Episode, Tray  # noqa: E402
from .base import (  # noqa: E402
    CLAW_PAD_HALF, CLAW_ROD_LEN, CLAW_ROD_RADIUS, MAX_OBJECTS, RESERVE_Z, SCREEN_DOWN_TOP_Z, SEG_BG, SEG_CLAW, SEG_SCREEN,
    ClawController, FrameState, classify_region, violation_add_pos, pick_topmost, screen_top_z,
)


def _f(v) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def _tray_xml(tr: Tray) -> str:
    cx, cy = tr.center
    hx, hy = tr.inner_size[0] / 2, tr.inner_size[1] / 2
    t, h = tr.wall_thickness, tr.wall_height
    rgba = _f(tr.color)
    parts = [f'<geom name="{tr.name}_floor" type="box" pos="{cx} {cy} {t / 2}" size="{hx + t} {hy + t} {t / 2}" rgba="{rgba}" material="flat"/>']
    walls = [
        (cx, cy + hy + t / 2, hx + t, t / 2), (cx, cy - hy - t / 2, hx + t, t / 2),
        (cx + hx + t / 2, cy, t / 2, hy), (cx - hx - t / 2, cy, t / 2, hy),
    ]
    for i, (x, y, sx, sy) in enumerate(walls):
        parts.append(f'<geom name="{tr.name}_wall{i}" type="box" pos="{x} {y} {h / 2}" size="{sx} {sy} {h / 2}" rgba="{rgba}" material="flat"/>')
    return "\n".join(parts)


def build_mjcf(ep: Episode) -> str:
    sc = ep.scene
    lights = "\n".join(
        f'<light directional="true" pos="{_f(l.pos)}" dir="{_f(l.dir)}" diffuse="{_f(l.diffuse)}" '
        f'specular="0 0 0" castshadow="{"true" if l.castshadow else "false"}"/>' for l in sc.lights)
    objs = []
    for o in ep.objects:
        if o.shape == "sphere":
            g = f'type="sphere" size="{o.size[0]}"'
        elif o.shape == "box":
            g = f'type="box" size="{_f(o.size)}"'
        else:
            g = f'type="cylinder" size="{_f(o.size)}"'
        pos = o.start_pos
        objs.append(f'<body name="obj{o.id}" pos="{_f(pos)}" quat="{_f(o.start_quat)}"><freejoint name="obj{o.id}"/>'
                    f'<geom name="obj{o.id}" {g} rgba="{_f(o.color)}" mass="{o.mass:.6g}" material="flat"/></body>')
    welds = "\n".join(f'<weld name="grip{o.id}" body1="claw" body2="obj{o.id}" active="false" solref="0.004 1"/>'
                      for o in ep.objects)
    s = sc.screen
    tw, th = sc.table_size
    c = sc.camera
    hx, hy, hz = CLAW_PAD_HALF
    n_res = len(ep.objects)
    return f"""<mujoco model="{ep.episode_id}">
  <option timestep="{ep.physics.timestep}" integrator="implicitfast" cone="elliptic"/>
  <visual>
    <global offwidth="{max(ep.clip.width, 64)}" offheight="{max(ep.clip.height, 64)}"/>
    <headlight ambient="{_f(sc.ambient)}" diffuse="0 0 0" specular="0 0 0"/>
    <quality shadowsize="2048"/>
    <map znear="0.01"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="{_f(sc.background)}" rgb2="{_f(sc.background)}" width="16" height="16"/>
    <material name="flat" specular="0" shininess="0" reflectance="0"/>
  </asset>
  <default>
    <geom friction="0.9 0.02 0.01" condim="6" solref="0.004 1"/>
  </default>
  <worldbody>
    {lights}
    <camera name="cam" mode="targetbody" target="lookat" pos="{_f(c.pos)}" fovy="{c.fovy}"/>
    <body name="lookat" pos="{_f(c.lookat)}"/>
    <geom name="table" type="box" pos="0 0 -0.025" size="{tw} {th} 0.025" rgba="{_f(sc.table_color)}" material="flat"/>
    {_tray_xml(sc.box)}
    {_tray_xml(sc.hopper)}
    <geom name="reserve_platform" type="box" pos="{(n_res - 1) * 0.15} 0 {RESERVE_Z - 0.05}" size="{n_res * 0.15 + 0.3} 0.3 0.05" rgba="0 0 0 1"/>
    <body name="screen" mocap="true" pos="{s.center_x} {s.y} {SCREEN_DOWN_TOP_Z - s.height / 2}" quat="{math.cos(s.yaw / 2)} 0 0 {math.sin(s.yaw / 2)}">
      <geom name="screen" type="box" size="{s.width / 2} {s.thickness / 2} {s.height / 2}" rgba="{_f(s.color)}" material="flat" contype="0" conaffinity="0"/>
    </body>
    <body name="claw" mocap="true" pos="{_f(ep.physics.claw_home)}">
      <geom name="claw_pad" type="box" size="{hx} {hy} {hz}" rgba="0.15 0.15 0.17 1" material="flat" contype="0" conaffinity="0"/>
      <geom name="claw_rod" type="cylinder" pos="0 0 {hz + CLAW_ROD_LEN / 2}" size="{CLAW_ROD_RADIUS} {CLAW_ROD_LEN / 2}" rgba="0.15 0.15 0.17 1" material="flat" contype="0" conaffinity="0"/>
    </body>
    {"".join(objs)}
  </worldbody>
  <equality>
    {welds}
  </equality>
</mujoco>"""


class MujocoBackend:
    name = "mujoco"

    def __init__(self, episode: Episode, limiter):
        if len(episode.objects) > MAX_OBJECTS:
            raise ValueError("too many objects")
        self.ep = episode
        self.limiter = limiter
        self.m = mujoco.MjModel.from_xml_string(build_mjcf(episode))
        self.d = mujoco.MjData(self.m)
        self.obj_body = {o.id: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"obj{o.id}") for o in episode.objects}
        self.obj_joint_qadr = {o.id: self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"obj{o.id}")]
                               for o in episode.objects}
        self.obj_joint_vadr = {o.id: self.m.jnt_dofadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"obj{o.id}")]
                               for o in episode.objects}
        self.weld = {o.id: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, f"grip{o.id}") for o in episode.objects}
        self.claw_mocap = self.m.body_mocapid[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "claw")]
        self.screen_mocap = self.m.body_mocapid[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "screen")]
        self.cam = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "cam")

        # geom id -> seg label
        lut = np.full(self.m.ngeom + 1, SEG_BG, dtype=np.uint8)
        for o in episode.objects:
            lut[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"obj{o.id}")] = o.id + 1
        for gname in ("claw_pad", "claw_rod"):
            lut[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, gname)] = SEG_CLAW
        lut[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "screen")] = SEG_SCREEN
        self.seg_lut = lut

        self.claw = ClawController(episode, self._poses, self._grasp, self._release)
        self.held: set[int] = set()
        self.events = self.claw.events
        self.m.vis.quality.offsamples = 4
        self.renderer = mujoco.Renderer(self.m, episode.clip.height, episode.clip.width)
        # segmentation ids are encoded as colours: multisampling blends them at edges into wrong ids
        self.m.vis.quality.offsamples = 0
        self.seg_renderer = mujoco.Renderer(self.m, episode.clip.height, episode.clip.width)
        self.seg_renderer.enable_segmentation_rendering()

        # preroll: settle objects with everything parked, unrendered
        mujoco.mj_forward(self.m, self.d)
        n = int(round(episode.physics.preroll / episode.physics.timestep))
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
        self.d.time = 0.0
        self.t = 0.0
        self._apply_mechanics(0.0)
        mujoco.mj_forward(self.m, self.d)

    # --- mechanics hooks ---
    def _poses(self):
        return {oid: (tuple(self.d.xpos[b]), tuple(self.d.xquat[b])) for oid, b in self.obj_body.items()}

    def _grasp(self, oid: int):
        b, e = self.obj_body[oid], self.weld[oid]
        claw_pos = self.d.mocap_pos[self.claw_mocap]
        # weld relpose = object pose in claw frame (claw never rotates)
        self.m.eq_data[e, 0:3] = 0.0
        self.m.eq_data[e, 3:6] = self.d.xpos[b] - claw_pos
        self.m.eq_data[e, 6:10] = self.d.xquat[b]
        self.d.eq_active[e] = 1
        self.held.add(oid)

    def _release(self, oid: int):
        self.d.eq_active[self.weld[oid]] = 0
        self.held.discard(oid)

    def _apply_mechanics(self, t: float):
        self.d.mocap_pos[self.claw_mocap] = self.claw.update(t)
        s = self.ep.scene.screen
        self.d.mocap_pos[self.screen_mocap] = (s.center_x, s.y, screen_top_z(self.ep, t) - s.height / 2)

    def step_to(self, t: float) -> None:
        dt = self.ep.physics.timestep
        while self.t + dt / 2 < t:
            self._apply_mechanics(self.t + dt)
            mujoco.mj_step(self.m, self.d)
            self.t += dt
        mujoco.mj_forward(self.m, self.d)

    def apply_violation(self) -> None:
        v = self.ep.violation
        if v is None:
            raise RuntimeError("no violation in this episode")
        if v.kind == "add":
            oid = v.object_id
            pos = violation_add_pos(self.ep, oid, v.pos, self._poses())
            q = self.obj_joint_qadr[oid]
            self.d.qpos[q:q + 3] = pos
            self.d.qpos[q + 3:q + 7] = (1, 0, 0, 0)
        elif v.kind == "delete":
            oid = pick_topmost(self.ep, "box", self._poses(), self.held)
            q = self.obj_joint_qadr[oid]
            self.d.qpos[q:q + 3] = (oid * 0.3, 0.0, RESERVE_Z + 0.1)
        else:
            raise NotImplementedError(f"{v.kind} violations are Rust-engine only (MuJoCo is the cross-check "
                                      "for the original claw/screen mechanics)")
        va = self.obj_joint_vadr[oid]
        self.d.qvel[va:va + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.events.append({"t": self.t, "event": f"violation_{v.kind}", "object_id": int(oid)})

    def frame_state(self) -> FrameState:
        objs, counts = [], {"box": 0, "hopper": 0, "reserve": 0, "claw": 0, "table": 0}
        for o in self.ep.objects:
            b = self.obj_body[o.id]
            va = self.obj_joint_vadr[o.id]
            pos = tuple(float(x) for x in self.d.xpos[b])
            region = classify_region(self.ep, pos, o.id in self.held)
            counts[region] += 1
            objs.append({"id": o.id, "pos": pos, "quat": tuple(float(x) for x in self.d.xquat[b]),
                         "lin_vel": tuple(float(x) for x in self.d.qvel[va:va + 3]),
                         "ang_vel": tuple(float(x) for x in self.d.qvel[va + 3:va + 6]), "region": region})
        active = None
        for a in self.ep.actions:
            if a.t_start <= self.t < a.t_end:
                active = a.kind
        return FrameState(t=float(self.t), count_per_region=counts, objects=objs,
                          claw_pos=tuple(float(x) for x in self.d.mocap_pos[self.claw_mocap]),
                          screen_top_z=float(screen_top_z(self.ep, self.t)), active_action=active)

    def penetration(self) -> float:
        """Deepest contact penetration (m) between any two geoms, >= 0."""
        if self.d.ncon == 0:
            return 0.0
        return float(max(0.0, -min(self.d.contact[i].dist for i in range(self.d.ncon))))

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        self.limiter.acquire()
        self.renderer.update_scene(self.d, camera=self.cam)
        rgb = self.renderer.render().copy()
        self.limiter.acquire()
        self.seg_renderer.update_scene(self.d, camera=self.cam)
        raw = self.seg_renderer.render()
        ids, types = raw[..., 0], raw[..., 1]
        geom = np.where(types == int(mujoco.mjtObj.mjOBJ_GEOM), ids, self.m.ngeom)
        geom = np.where(geom < 0, self.m.ngeom, geom)
        return rgb, self.seg_lut[geom]

    def close(self) -> None:
        self.renderer.close()
        self.seg_renderer.close()
