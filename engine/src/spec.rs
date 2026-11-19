//! Concrete episode spec. Mirrors `jepagen/spec.py`; the Python side is the source of truth.

use serde::Deserialize;

pub type V2 = [f64; 2];
pub type V3 = [f64; 3];
pub type V4 = [f64; 4];

#[derive(Debug, Clone, Deserialize)]
pub struct Tray {
    pub name: String,
    pub center: V2,
    pub inner_size: V2,
    pub wall_height: f64,
    pub wall_thickness: f64,
    pub color: V4,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Screen {
    pub center_x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub thickness: f64,
    pub color: V4,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Camera {
    pub pos: V3,
    pub lookat: V3,
    pub fovy: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Light {
    pub pos: V3,
    pub dir: V3,
    pub diffuse: V3,
    pub castshadow: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Scene {
    pub table_size: V2,
    pub table_color: V4,
    pub background: V3,
    pub ambient: V3,
    #[serde(rename = "box")]
    pub box_: Tray,
    pub hopper: Tray,
    pub screen: Screen,
    pub camera: Camera,
    pub lights: Vec<Light>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Shape {
    Sphere,
    Box,
    Cylinder,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ObjectSpec {
    pub id: usize,
    pub shape: Shape,
    pub size: Vec<f64>,
    pub color: V4,
    pub mass: f64,
    pub start_region: String,
    pub start_pos: V3,
    pub start_quat: V4, // w, x, y, z
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrayName {
    Box,
    Hopper,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind")]
pub enum Action {
    #[serde(rename = "screen_raise")]
    ScreenRaise { t_start: f64, t_end: f64 },
    #[serde(rename = "screen_lower")]
    ScreenLower { t_start: f64, t_end: f64 },
    #[serde(rename = "transfer")]
    Transfer { src: TrayName, dst: TrayName, t_start: f64, t_end: f64, drop_xy: V2 },
}

impl Action {
    pub fn t_start(&self) -> f64 {
        match self {
            Action::ScreenRaise { t_start, .. } | Action::ScreenLower { t_start, .. } | Action::Transfer { t_start, .. } => *t_start,
        }
    }
    pub fn t_end(&self) -> f64 {
        match self {
            Action::ScreenRaise { t_end, .. } | Action::ScreenLower { t_end, .. } | Action::Transfer { t_end, .. } => *t_end,
        }
    }
    pub fn kind(&self) -> &'static str {
        match self {
            Action::ScreenRaise { .. } => "screen_raise",
            Action::ScreenLower { .. } => "screen_lower",
            Action::Transfer { .. } => "transfer",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Violation {
    pub kind: String, // "add" | "delete"
    pub t: f64,
    pub object_id: Option<usize>,
    pub pos: Option<V3>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Clip {
    pub n_frames: usize,
    pub fps: f64,
    pub width: u32,
    pub height: u32,
    pub render_pass: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Physics {
    pub timestep: f64,
    pub preroll: f64,
    pub claw_home: V3,
    pub claw_travel_z: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Episode {
    pub spec_version: String,
    pub episode_id: String,
    pub seed: u64,
    pub episode_type: String,
    pub split: String,
    pub scene: Scene,
    pub objects: Vec<ObjectSpec>,
    pub actions: Vec<Action>,
    pub violation: Option<Violation>,
    pub twin_of: Option<String>,
    pub clip: Clip,
    pub physics: Physics,
    pub meta: serde_json::Value,
}

pub const SUPPORTED_SPEC_VERSION: &str = "0.1";

impl Episode {
    pub fn from_json(s: &str) -> Result<Self, String> {
        let ep: Episode = serde_json::from_str(s).map_err(|e| format!("episode json: {e}"))?;
        if ep.spec_version != SUPPORTED_SPEC_VERSION {
            return Err(format!("spec_version {} not supported (engine supports {})", ep.spec_version, SUPPORTED_SPEC_VERSION));
        }
        if ep.clip.render_pass != "flat" {
            return Err(format!("render_pass {} not supported", ep.clip.render_pass));
        }
        for (i, o) in ep.objects.iter().enumerate() {
            if o.id != i {
                return Err("object ids must be 0..n in order".into());
            }
            let n = match o.shape {
                Shape::Sphere => 1,
                Shape::Box => 3,
                Shape::Cylinder => 2,
            };
            if o.size.len() != n {
                return Err(format!("object {} size has {} values, expected {}", o.id, o.size.len(), n));
            }
        }
        Ok(ep)
    }

    pub fn tray(&self, t: TrayName) -> &Tray {
        match t {
            TrayName::Box => &self.scene.box_,
            TrayName::Hopper => &self.scene.hopper,
        }
    }
}
