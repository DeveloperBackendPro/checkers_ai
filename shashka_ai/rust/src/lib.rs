//! PyO3 bog'lash: Python uchun `shashka_engine` moduli.

mod engine;
mod mcts;
use engine::Pos;
use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
#[pyclass(name = "GameState")]
pub struct PyGameState {inner: Pos,}

#[pymethods]
impl PyGameState {
    #[new]
    fn new() -> Self {PyGameState { inner: Pos::initial() }}
    #[staticmethod]
    fn from_board(pieces: Vec<i8>, player: i8) -> PyResult<Self> {
        if pieces.len() != 32 {return Err(pyo3::exceptions::PyValueError::new_err("pieces 32 ta bo'lishi kerak"));}
        Ok(PyGameState { inner: Pos::from_board(&pieces, player) })
    }
    fn clone_state(&self) -> Self {
        PyGameState { inner: self.inner.clone() }
    }
    fn legal_actions(&self) -> Vec<u16> {
        engine::legal_moves(&self.inner).into_iter().map(|(f, t)| f as u16 * 32 + t as u16).collect()
    }
    fn apply(&mut self, action: u16) -> PyResult<()> {
        if action >= 1024 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("action chegaradan tashqari: {} (0..1024)", action)));
        }
        engine::apply(&mut self.inner, action);
        Ok(())
    }
    fn status(&self) -> (Option<i8>, Vec<u16>) {
        let (w, m) = engine::status(&self.inner);
        (w, m.into_iter().map(|(f, t)| f as u16 * 32 + t as u16).collect())
    }
    fn winner(&self) -> Option<i8> {
        engine::status(&self.inner).0
    }
    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let mut buf = vec![0f32; engine::INPUT_SIZE];
        engine::encode(&self.inner, &mut buf);
        buf.into_pyarray_bound(py)
    }
    fn canon_action(&self, a: u16) -> u16 {
        engine::canon_action(&self.inner, a)
    }
    fn board_list(&self) -> Vec<i8> {
        self.inner.board_list()
    }
    #[getter]
    fn player(&self) -> i8 {
        self.inner.player
    }
    #[getter]
    fn ply(&self) -> u16 {
        self.inner.ply
    }
    #[getter]
    fn halfmove(&self) -> u16 {
        self.inner.halfmove
    }
    #[getter]
    fn chain_sq(&self) -> Option<u8> {
        if self.inner.chain_sq >= 0 {
            Some(self.inner.chain_sq as u8)
        } else {
            None
        }
    }
}

#[pyclass(name = "MctsForest")]
pub struct PyForest {
    inner: mcts::Forest,
}

#[pymethods]
impl PyForest {
    #[new]
    fn new(n_games: usize, c_puct: f64, dirichlet_alpha: f64, dirichlet_eps: f64, seed: u64) -> Self {
        PyForest { inner: mcts::Forest::new(n_games, c_puct, dirichlet_alpha, dirichlet_eps, seed) }
    }
    fn set_state(&mut self, g: usize, st: &PyGameState) {
        self.inner.reset(g, st.inner.clone());
    }
    fn set_active(&mut self, g: usize, active: bool) {
        self.inner.trees[g].active = active;
    }
    fn collect_roots<'py>(&mut self, py: Python<'py>) -> (Bound<'py, PyArray1<f32>>, usize) {
        let mut buf: Vec<f32> = Vec::new();
        self.inner.collect_roots(&mut buf);
        let k = buf.len() / engine::INPUT_SIZE;
        (buf.into_pyarray_bound(py), k)
    }
    #[pyo3(signature = (leaves, max_pending=16384))]
    fn collect<'py>(&mut self, py: Python<'py>, leaves: usize, max_pending: usize) -> (Bound<'py, PyArray1<f32>>, usize) {
        let mut buf: Vec<f32> = Vec::new();
        self.inner.collect(leaves, &mut buf, max_pending);
        let k = buf.len() / engine::INPUT_SIZE;
        (buf.into_pyarray_bound(py), k)
    }
    fn collect_roots_subset<'py>(&mut self, py: Python<'py>, subset: Vec<usize>) -> (Bound<'py, PyArray1<f32>>, usize) {
        let mut buf: Vec<f32> = Vec::new();
        self.inner.collect_roots_subset(&subset, &mut buf);
        let k = buf.len() / engine::INPUT_SIZE;
        (buf.into_pyarray_bound(py), k)
    }
    #[pyo3(signature = (subset, leaves, max_pending=16384))]
    fn collect_subset<'py>(&mut self, py: Python<'py>, subset: Vec<usize>, leaves: usize, max_pending: usize) -> (Bound<'py, PyArray1<f32>>, usize) {
        let mut buf: Vec<f32> = Vec::new();
        self.inner.collect_subset(&subset, leaves, &mut buf, max_pending);
        let k = buf.len() / engine::INPUT_SIZE;
        (buf.into_pyarray_bound(py), k)
    }
    fn best_action(&self, g: usize) -> u16 {
        let (acts, vis) = self.inner.visits(g);
        let mut bi = 0usize;
        let mut bv = 0u32;
        let mut found = false;
        for (i, &v) in vis.iter().enumerate() {
            if !found || v > bv {
                bv = v;
                bi = i;
                found = true;
            }
        }
        if found { acts[bi] } else { u16::MAX }
    }
    fn apply_evals(&mut self, logits: PyReadonlyArray2<f32>, values: PyReadonlyArray1<f32>) -> PyResult<()> {
        let l = logits.as_slice()?;
        let v = values.as_slice()?;
        self.inner.apply_evals(l, v);
        Ok(())
    }
    fn add_noise(&mut self, g: usize) {
        self.inner.add_noise(g);
    }
    fn visits(&self, g: usize) -> (Vec<u16>, Vec<u32>) {
        self.inner.visits(g)
    }
    fn advance(&mut self, g: usize, action: u16) {
        self.inner.advance(g, action);
    }
    fn winner(&self, g: usize) -> Option<i8> {
        self.inner.winner(g)
    }
    fn root_player(&self, g: usize) -> i8 {
        self.inner.trees[g].state.player
    }
    fn root_ply(&self, g: usize) -> u16 {
        self.inner.trees[g].state.ply
    }
    fn root_q(&self, g: usize) -> f64 {
        let t = &self.inner.trees[g];
        let r = &t.nodes[t.root as usize];
        if r.n > 0 {
            r.w / r.n as f64
        } else {
            0.0
        }
    }
    fn root_encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let mut buf = vec![0f32; engine::INPUT_SIZE];
        engine::encode(&self.inner.trees[0].state, &mut buf);
        buf.into_pyarray_bound(py)
    }
    fn root_encode_g<'py>(&self, py: Python<'py>, g: usize) -> Bound<'py, PyArray1<f32>> {
        let mut buf = vec![0f32; engine::INPUT_SIZE];
        engine::encode(&self.inner.trees[g].state, &mut buf);
        buf.into_pyarray_bound(py)
    }
}

#[pymodule]
fn shashka_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameState>()?;
    m.add_class::<PyForest>()?;
    m.add("INPUT_SIZE", engine::INPUT_SIZE)?;
    m.add("ACTION_SIZE", 1024usize)?;
    Ok(())
}