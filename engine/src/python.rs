//! PyO3 bindings: `jepamath_engine.Engine(episode_json)`.

use std::cell::RefCell;

use numpy::{PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::render::Renderer;
use crate::spec::Episode;
use crate::world::World;

thread_local! {
    // Device creation is slow; one renderer per thread, rebuilt only if the clip size changes.
    static RENDERER: RefCell<Option<Renderer>> = const { RefCell::new(None) };
}

fn err(e: String) -> PyErr {
    PyRuntimeError::new_err(e)
}

#[pyclass(unsendable)]
struct Engine {
    world: World,
}

#[pymethods]
impl Engine {
    #[new]
    fn new(episode_json: &str) -> PyResult<Self> {
        let ep = Episode::from_json(episode_json).map_err(err)?;
        Ok(Engine { world: World::new(ep).map_err(err)? })
    }

    fn step_to(&mut self, t: f64) -> PyResult<()> {
        self.world.step_to(t).map_err(err)
    }

    fn apply_violation(&mut self) -> PyResult<()> {
        self.world.apply_violation().map_err(err)
    }

    fn frame_state(&self) -> PyResult<String> {
        serde_json::to_string(&self.world.frame_state()).map_err(|e| err(e.to_string()))
    }

    fn events(&self) -> PyResult<String> {
        serde_json::to_string(&self.world.all_events()).map_err(|e| err(e.to_string()))
    }

    fn penetration(&self) -> PyResult<f64> {
        self.world.penetration().map_err(err)
    }

    fn time(&self) -> f64 {
        self.world.t
    }

    /// Returns (rgb uint8[H, W, 3], seg uint8[H, W]).
    fn render<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray3<u8>>, Bound<'py, PyArray2<u8>>)> {
        let (w, h) = (self.world.ep.clip.width, self.world.ep.clip.height);
        let (rgb, seg) = RENDERER.with(|cell| -> Result<(Vec<u8>, Vec<u8>), String> {
            let mut r = cell.borrow_mut();
            if r.as_ref().map(|r| r.size() != (w, h)).unwrap_or(true) {
                *r = Some(Renderer::new(w, h)?);
            }
            r.as_mut().unwrap().render(&self.world)
        }).map_err(err)?;
        let (w, h) = (w as usize, h as usize);
        let rgb = PyArray1::from_vec(py, rgb).reshape([h, w, 3])?;
        let seg = PyArray1::from_vec(py, seg).reshape([h, w])?;
        Ok((rgb, seg))
    }
}

/// Render several engines' current state in one GPU submission.
/// Returns (rgb uint8[N, H, W, 3], seg uint8[N, H, W]); all engines must share the clip size.
#[pyfunction]
fn render_batch<'py>(
    py: Python<'py>,
    engines: Vec<PyRef<'py, Engine>>,
) -> PyResult<(Bound<'py, PyArray4<u8>>, Bound<'py, PyArray3<u8>>)> {
    let first = engines.first().ok_or_else(|| err("render_batch needs at least one engine".into()))?;
    let (w, h) = (first.world.ep.clip.width, first.world.ep.clip.height);
    let worlds: Vec<&World> = engines.iter().map(|e| &e.world).collect();
    let out = RENDERER.with(|cell| -> Result<Vec<(Vec<u8>, Vec<u8>)>, String> {
        let mut r = cell.borrow_mut();
        if r.as_ref().map(|r| r.size() != (w, h)).unwrap_or(true) {
            *r = Some(Renderer::new(w, h)?);
        }
        r.as_mut().unwrap().render_batch(&worlds)
    })
    .map_err(err)?;
    let n = out.len();
    let (w, h) = (w as usize, h as usize);
    let mut rgb = Vec::with_capacity(n * h * w * 3);
    let mut seg = Vec::with_capacity(n * h * w);
    for (c, s) in out {
        rgb.extend_from_slice(&c);
        seg.extend_from_slice(&s);
    }
    let rgb = PyArray1::from_vec(py, rgb).reshape([n, h, w, 3])?;
    let seg = PyArray1::from_vec(py, seg).reshape([n, h, w])?;
    Ok((rgb, seg))
}

#[pyfunction]
fn max_batch() -> usize {
    crate::render::MAX_BATCH
}

#[pyfunction]
fn adapter_info() -> PyResult<String> {
    RENDERER.with(|cell| {
        let mut r = cell.borrow_mut();
        if r.is_none() {
            *r = Some(Renderer::new(64, 64).map_err(err)?);
        }
        Ok(r.as_ref().unwrap().adapter_info().to_string())
    })
}

#[pymodule]
fn jepamath_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_function(wrap_pyfunction!(adapter_info, m)?)?;
    m.add_function(wrap_pyfunction!(render_batch, m)?)?;
    m.add_function(wrap_pyfunction!(max_batch, m)?)?;
    Ok(())
}
