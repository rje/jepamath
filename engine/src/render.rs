//! Offscreen wgpu (Vulkan) renderer: flat Lambert shading, one shadow-casting directional
//! light, ambient term. Produces an RGB image (4x MSAA, resolved) and a segmentation image
//! (no MSAA, uint8 labels) in one submission.
#![allow(deprecated)] // glam's Mat4::look_at_rh etc. are deprecated in favour of camera:: fns

use bytemuck::{Pod, Zeroable};
use rapier3d::glamx::glam::{Mat3, Mat4, Quat, Vec3};

use crate::rules::{self, SEG_BG, SEG_CLAW, SEG_SCREEN};
use crate::spec::{Episode, Shape, V3};
use crate::world::{tray_boxes, World};

const MSAA: u32 = 4;
const SHADOW_SIZE: u32 = 2048;
const COLOR_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SEG_FMT: wgpu::TextureFormat = wgpu::TextureFormat::R8Uint;
const DEPTH_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Instance {
    model: [[f32; 4]; 4],
    normal: [[f32; 4]; 3],
    color: [f32; 4],
    label: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    light_view_proj: [[f32; 4]; 4],
    light_dir: [f32; 4], // direction light travels, normalized
    light_diffuse: [f32; 4],
    ambient: [f32; 4],
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum MeshKind {
    Cube,
    Sphere,
    Cylinder,
}

struct Mesh {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    n_idx: u32,
}

const SHADER: &str = r#"
struct Globals {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    light_diffuse: vec4<f32>,
    ambient: vec4<f32>,
};
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var shadow_tex: texture_depth_2d;
@group(0) @binding(2) var shadow_samp: sampler_comparison;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) m0: vec4<f32>, @location(3) m1: vec4<f32>, @location(4) m2: vec4<f32>, @location(5) m3: vec4<f32>,
    @location(6) n0: vec4<f32>, @location(7) n1: vec4<f32>, @location(8) n2: vec4<f32>,
    @location(9) color: vec4<f32>,
    @location(10) label: u32,
};
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) @interpolate(flat) label: u32,
};

fn model(v: VIn) -> mat4x4<f32> { return mat4x4<f32>(v.m0, v.m1, v.m2, v.m3); }

@vertex fn vs_main(v: VIn) -> VOut {
    var o: VOut;
    let w = model(v) * vec4<f32>(v.pos, 1.0);
    o.clip = g.view_proj * w;
    o.world = w.xyz;
    let nm = mat3x3<f32>(v.n0.xyz, v.n1.xyz, v.n2.xyz);
    o.normal = normalize(nm * v.normal);
    o.color = v.color;
    o.label = v.label;
    return o;
}

@vertex fn vs_shadow(v: VIn) -> @builtin(position) vec4<f32> {
    return g.light_view_proj * model(v) * vec4<f32>(v.pos, 1.0);
}

fn shadow_factor(world: vec3<f32>, n: vec3<f32>) -> f32 {
    let p = g.light_view_proj * vec4<f32>(world, 1.0);
    let ndc = p.xyz / p.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0) { return 1.0; }
    let bias = 0.0015;
    var s = 0.0;
    let texel = 1.0 / f32(textureDimensions(shadow_tex).x);
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            s += textureSampleCompareLevel(shadow_tex, shadow_samp, uv + vec2<f32>(f32(dx), f32(dy)) * texel, ndc.z - bias);
        }
    }
    return s / 9.0;
}

@fragment fn fs_color(i: VOut) -> @location(0) vec4<f32> {
    let n = normalize(i.normal);
    let l = -g.light_dir.xyz;
    let diff = max(dot(n, l), 0.0) * shadow_factor(i.world, n);
    let rgb = i.color.rgb * (g.ambient.rgb + g.light_diffuse.rgb * diff);
    return vec4<f32>(min(rgb, vec3<f32>(1.0)), 1.0);
}

@fragment fn fs_seg(i: VOut) -> @location(0) u32 {
    return i.label;
}
"#;

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    width: u32,
    height: u32,
    meshes: Vec<(MeshKind, Mesh)>,
    color_pipe: wgpu::RenderPipeline,
    seg_pipe: wgpu::RenderPipeline,
    shadow_pipe: wgpu::RenderPipeline,
    globals: wgpu::Buffer,
    bind: wgpu::BindGroup,
    shadow_bind: wgpu::BindGroup,
    shadow_view: wgpu::TextureView,
    color_msaa: wgpu::TextureView,
    color_resolve: wgpu::Texture,
    depth_msaa: wgpu::TextureView,
    seg_tex: wgpu::Texture,
    depth_seg: wgpu::TextureView,
    readback_color: wgpu::Buffer,
    readback_seg: wgpu::Buffer,
    color_row: u32,
    seg_row: u32,
    adapter_info: String,
}

fn align256(x: u32) -> u32 {
    x.div_ceil(256) * 256
}

fn cube_mesh() -> (Vec<Vertex>, Vec<u32>) {
    let mut v = vec![];
    let mut idx = vec![];
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1., 0., 0.], [0., 1., 0.], [0., 0., 1.]),
        ([-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]),
        ([0., 1., 0.], [0., 0., 1.], [1., 0., 0.]),
        ([0., -1., 0.], [1., 0., 0.], [0., 0., 1.]),
        ([0., 0., 1.], [1., 0., 0.], [0., 1., 0.]),
        ([0., 0., -1.], [0., 1., 0.], [1., 0., 0.]),
    ];
    for (n, a, b) in faces {
        let base = v.len() as u32;
        for (sa, sb) in [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)] {
            let p = [n[0] + a[0] * sa + b[0] * sb, n[1] + a[1] * sa + b[1] * sb, n[2] + a[2] * sa + b[2] * sb];
            v.push(Vertex { pos: p, normal: n });
        }
        idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (v, idx)
}

fn sphere_mesh(seg: u32, rings: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut v = vec![];
    let mut idx = vec![];
    for r in 0..=rings {
        let th = std::f32::consts::PI * r as f32 / rings as f32;
        for s in 0..=seg {
            let ph = 2.0 * std::f32::consts::PI * s as f32 / seg as f32;
            let p = [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()];
            v.push(Vertex { pos: p, normal: p });
        }
    }
    for r in 0..rings {
        for s in 0..seg {
            let a = r * (seg + 1) + s;
            let b = a + seg + 1;
            idx.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    (v, idx)
}

/// Unit cylinder: radius 1, half-height 1, axis z.
fn cylinder_mesh(seg: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut v = vec![];
    let mut idx = vec![];
    for s in 0..=seg {
        let ph = 2.0 * std::f32::consts::PI * s as f32 / seg as f32;
        let (c, sn) = (ph.cos(), ph.sin());
        v.push(Vertex { pos: [c, sn, -1.0], normal: [c, sn, 0.0] });
        v.push(Vertex { pos: [c, sn, 1.0], normal: [c, sn, 0.0] });
    }
    for s in 0..seg {
        let a = 2 * s;
        idx.extend_from_slice(&[a, a + 2, a + 1, a + 1, a + 2, a + 3]);
    }
    for (z, nz) in [(1.0f32, 1.0f32), (-1.0, -1.0)] {
        let center = v.len() as u32;
        v.push(Vertex { pos: [0.0, 0.0, z], normal: [0.0, 0.0, nz] });
        for s in 0..=seg {
            let ph = 2.0 * std::f32::consts::PI * s as f32 / seg as f32;
            v.push(Vertex { pos: [ph.cos(), ph.sin(), z], normal: [0.0, 0.0, nz] });
        }
        for s in 0..seg {
            if nz > 0.0 {
                idx.extend_from_slice(&[center, center + 1 + s, center + 2 + s]);
            } else {
                idx.extend_from_slice(&[center, center + 2 + s, center + 1 + s]);
            }
        }
    }
    (v, idx)
}

fn v3(a: V3) -> Vec3 {
    Vec3::new(a[0] as f32, a[1] as f32, a[2] as f32)
}

fn instance(model: Mat4, color: [f64; 4], label: u8) -> Instance {
    let nm = Mat3::from_mat4(model).inverse().transpose();
    let c = |v: Vec3| [v.x, v.y, v.z, 0.0];
    Instance {
        model: model.to_cols_array_2d(),
        normal: [c(nm.x_axis), c(nm.y_axis), c(nm.z_axis)],
        color: [color[0] as f32, color[1] as f32, color[2] as f32, color[3] as f32],
        label: label as u32,
        _pad: [0; 3],
    }
}

impl Renderer {
    pub fn new(width: u32, height: u32) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|e| format!("no Vulkan adapter: {e}"))?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("jepamath"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("request_device: {e}"))?;

        use wgpu::util::DeviceExt;
        let mut meshes = vec![];
        for (kind, (v, i)) in
            [(MeshKind::Cube, cube_mesh()), (MeshKind::Sphere, sphere_mesh(24, 16)), (MeshKind::Cylinder, cylinder_mesh(24))]
        {
            let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&v),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&i),
                usage: wgpu::BufferUsages::INDEX,
            });
            meshes.push((kind, Mesh { vbuf, ibuf, n_idx: i.len() as u32 }));
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tex = |label, w, h, samples, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let att = wgpu::TextureUsages::RENDER_ATTACHMENT;
        let shadow = tex("shadow", SHADOW_SIZE, SHADOW_SIZE, 1, DEPTH_FMT, att | wgpu::TextureUsages::TEXTURE_BINDING);
        let shadow_view = shadow.create_view(&Default::default());
        let color_msaa = tex("color_msaa", width, height, MSAA, COLOR_FMT, att).create_view(&Default::default());
        let color_resolve = tex("color", width, height, 1, COLOR_FMT, att | wgpu::TextureUsages::COPY_SRC);
        let depth_msaa = tex("depth_msaa", width, height, MSAA, DEPTH_FMT, att).create_view(&Default::default());
        let seg_tex = tex("seg", width, height, 1, SEG_FMT, att | wgpu::TextureUsages::COPY_SRC);
        let depth_seg = tex("depth_seg", width, height, 1, DEPTH_FMT, att).create_view(&Default::default());
        let color_row = align256(width * 4);
        let seg_row = align256(width);
        let rb = |size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let readback_color = rb((color_row * height) as u64);
        let readback_seg = rb((seg_row * height) as u64);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        // The shadow pass must not bind the shadow texture it writes: it gets its own layout below.
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&shadow_bgl)],
            immediate_size: 0,
        });

        let vattrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
        let iattrs = wgpu::vertex_attr_array![
            2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4,
            6 => Float32x4, 7 => Float32x4, 8 => Float32x4, 9 => Float32x4, 10 => Uint32
        ];
        let buffers = [
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &vattrs,
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Instance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &iattrs,
            }),
        ];
        let prim = wgpu::PrimitiveState { cull_mode: None, ..Default::default() };
        let depth = |bias: wgpu::DepthBiasState| {
            Some(wgpu::DepthStencilState {
                format: DEPTH_FMT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias,
            })
        };
        let mk = |label, layout: &wgpu::PipelineLayout, vs, fs: Option<(&'static str, wgpu::TextureFormat)>, samples, bias| {
            let targets = fs.map(|(_, f)| [Some(wgpu::ColorTargetState { format: f, blend: None, write_mask: wgpu::ColorWrites::ALL })]);
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &buffers,
                },
                primitive: prim,
                depth_stencil: depth(bias),
                multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
                fragment: fs.map(|(entry, _)| wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: targets.as_ref().unwrap(),
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let color_pipe = mk("color", &layout, "vs_main", Some(("fs_color", COLOR_FMT)), MSAA, Default::default());
        let seg_pipe = mk("seg", &layout, "vs_main", Some(("fs_seg", SEG_FMT)), 1, Default::default());
        let shadow_pipe = mk(
            "shadow",
            &shadow_layout,
            "vs_shadow",
            None,
            1,
            wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
        );
        let shadow_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &shadow_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() }],
        });
        let r = Renderer {
            device,
            queue,
            width,
            height,
            meshes,
            color_pipe,
            seg_pipe,
            shadow_pipe,
            globals,
            bind,
            shadow_bind,
            shadow_view,
            color_msaa,
            color_resolve,
            depth_msaa,
            seg_tex,
            depth_seg,
            readback_color,
            readback_seg,
            color_row,
            seg_row,
            adapter_info: format!("{} ({:?})", info.name, info.backend),
        };
        Ok(r)
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn adapter_info(&self) -> &str {
        &self.adapter_info
    }

    fn scene_instances(ep: &Episode, w: &World) -> Vec<(MeshKind, Instance)> {
        let sc = &ep.scene;
        let mut out = vec![];
        let cube = |c: V3, h: V3, color, label| {
            (MeshKind::Cube, instance(Mat4::from_scale_rotation_translation(v3(h), Quat::IDENTITY, v3(c)), color, label))
        };
        out.push(cube([0.0, 0.0, -0.025], [sc.table_size[0], sc.table_size[1], 0.025], sc.table_color, SEG_BG));
        for tr in [&sc.box_, &sc.hopper] {
            for (c, h) in tray_boxes(tr) {
                out.push(cube(c, h, tr.color, SEG_BG));
            }
        }
        let s = &sc.screen;
        let sz = rules::screen_top_z(ep, w.t) - s.height / 2.0;
        out.push(cube([s.center_x, s.y, sz], [s.width / 2.0, s.thickness / 2.0, s.height / 2.0], s.color, SEG_SCREEN));
        let cp = w.claw_pos();
        let ph = rules::CLAW_PAD_HALF;
        let dark = [0.15, 0.15, 0.17, 1.0];
        out.push(cube(cp, ph, dark, SEG_CLAW));
        let rod_c = [cp[0], cp[1], cp[2] + ph[2] + rules::CLAW_ROD_LEN / 2.0];
        out.push((
            MeshKind::Cylinder,
            instance(
                Mat4::from_scale_rotation_translation(
                    Vec3::new(rules::CLAW_ROD_RADIUS as f32, rules::CLAW_ROD_RADIUS as f32, (rules::CLAW_ROD_LEN / 2.0) as f32),
                    Quat::IDENTITY,
                    v3(rod_c),
                ),
                dark,
                SEG_CLAW,
            ),
        ));
        for (oid, (p, q)) in w.phys.poses().into_iter().enumerate() {
            let o = &ep.objects[oid];
            let rot = Quat::from_xyzw(q[1] as f32, q[2] as f32, q[3] as f32, q[0] as f32);
            let (kind, scale) = match o.shape {
                Shape::Sphere => (MeshKind::Sphere, Vec3::splat(o.size[0] as f32)),
                Shape::Box => (MeshKind::Cube, Vec3::new(o.size[0] as f32, o.size[1] as f32, o.size[2] as f32)),
                Shape::Cylinder => (MeshKind::Cylinder, Vec3::new(o.size[0] as f32, o.size[0] as f32, o.size[1] as f32)),
            };
            out.push((kind, instance(Mat4::from_scale_rotation_translation(scale, rot, v3(p)), o.color, (oid + 1) as u8)));
        }
        out
    }

    /// Render the world. Returns (rgb HxWx3, seg HxW).
    pub fn render(&mut self, w: &World) -> Result<(Vec<u8>, Vec<u8>), String> {
        let ep = &w.ep;
        if (ep.clip.width, ep.clip.height) != (self.width, self.height) {
            return Err("renderer size does not match episode clip size".into());
        }
        let sc = &ep.scene;
        let cam = &sc.camera;
        let aspect = self.width as f32 / self.height as f32;
        let view = Mat4::look_at_rh(v3(cam.pos), v3(cam.lookat), Vec3::Z);
        let proj = Mat4::perspective_rh((cam.fovy as f32).to_radians(), aspect, 0.01, 20.0);
        let light = sc.lights.first().ok_or("scene has no light")?;
        let ldir = v3(light.dir).normalize();
        let center = Vec3::new(-0.05, 0.0, 0.1);
        let lup = if ldir.cross(Vec3::Z).length() < 1e-3 { Vec3::Y } else { Vec3::Z };
        let lview = Mat4::look_at_rh(center - ldir * 3.0, center, lup);
        let lproj = Mat4::orthographic_rh(-0.9, 0.9, -0.9, 0.9, 0.1, 6.0);
        let shadows = light.castshadow;
        let g = Globals {
            view_proj: (proj * view).to_cols_array_2d(),
            light_view_proj: (lproj * lview).to_cols_array_2d(),
            light_dir: [ldir.x, ldir.y, ldir.z, 0.0],
            light_diffuse: [light.diffuse[0] as f32, light.diffuse[1] as f32, light.diffuse[2] as f32, 0.0],
            ambient: [sc.ambient[0] as f32, sc.ambient[1] as f32, sc.ambient[2] as f32, 0.0],
        };
        self.queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(&g));

        let mut inst = Self::scene_instances(ep, w);
        inst.sort_by_key(|(k, _)| *k);
        let data: Vec<Instance> = inst.iter().map(|(_, i)| *i).collect();
        use wgpu::util::DeviceExt;
        let ibuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instances"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let mut ranges: Vec<(MeshKind, u32, u32)> = vec![];
        for (i, (k, _)) in inst.iter().enumerate() {
            match ranges.last_mut() {
                Some((lk, _, end)) if lk == k => *end = i as u32 + 1,
                _ => ranges.push((*k, i as u32, i as u32 + 1)),
            }
        }
        let draw = |pass: &mut wgpu::RenderPass<'_>| {
            pass.set_vertex_buffer(1, ibuf.slice(..));
            for (k, a, b) in &ranges {
                let m = &self.meshes.iter().find(|(mk, _)| mk == k).unwrap().1;
                pass.set_vertex_buffer(0, m.vbuf.slice(..));
                pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..m.n_idx, 0, *a..*b);
            }
        };

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if shadows {
                pass.set_pipeline(&self.shadow_pipe);
                pass.set_bind_group(0, &self.shadow_bind, &[]);
                draw(&mut pass);
            }
        }
        let bg = sc.background;
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("color"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_msaa,
                    depth_slice: None,
                    resolve_target: Some(&self.color_resolve.create_view(&Default::default())),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: bg[0], g: bg[1], b: bg[2], a: 1.0 }),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_msaa,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.color_pipe);
            pass.set_bind_group(0, &self.bind, &[]);
            draw(&mut pass);
        }
        {
            let seg_view = self.seg_tex.create_view(&Default::default());
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("seg"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &seg_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_seg,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.seg_pipe);
            pass.set_bind_group(0, &self.bind, &[]);
            draw(&mut pass);
        }
        let copy = |enc: &mut wgpu::CommandEncoder, tex: &wgpu::Texture, buf: &wgpu::Buffer, row: u32| {
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo {
                    buffer: buf,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(self.height) },
                },
                wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
            );
        };
        copy(&mut enc, &self.color_resolve, &self.readback_color, self.color_row);
        copy(&mut enc, &self.seg_tex, &self.readback_seg, self.seg_row);
        self.queue.submit([enc.finish()]);

        let read = |buf: &wgpu::Buffer, row: u32, px_bytes: u32, keep: usize| -> Result<Vec<u8>, String> {
            let slice = buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
            self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("poll: {e}"))?;
            let data = slice.get_mapped_range().map_err(|e| format!("map range: {e}"))?;
            let mut out = Vec::with_capacity((self.width * self.height) as usize * keep);
            for y in 0..self.height {
                let rowd = &data[(y * row) as usize..(y * row + self.width * px_bytes) as usize];
                for x in 0..self.width as usize {
                    out.extend_from_slice(&rowd[x * px_bytes as usize..x * px_bytes as usize + keep]);
                }
            }
            drop(data);
            buf.unmap();
            Ok(out)
        };
        let rgb = read(&self.readback_color, self.color_row, 4, 3)?;
        let seg = read(&self.readback_seg, self.seg_row, 1, 1)?;
        Ok((rgb, seg))
    }
}
