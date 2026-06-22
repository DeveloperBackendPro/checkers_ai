//! O'yin vaqti AI: bitta daraxtli PUCT MCTS + ONNX baholash (`ort`).
//! Semantika trening (`mcts.py` / Rust forest) bilan bir xil:
//!   - PUCT tanlash, c_puct = 1.6
//!   - tugun qiymati o'sha tugunda yuruvchi o'yinchi nuqtai nazaridan
//!   - value = softmax(wdl)[win] - softmax(wdl)[loss]
//!   - policy: kanonik harakatlar ustida softmax
//! O'yinda Dirichlet shovqin QO'YILMAYDI (eng kuchli, deterministik o'yin)
//! va harakat = eng ko'p tashrif buyurilgan bola (temperature = 0).

use std::path::Path;
use std::sync::Mutex;
use ort::value::Tensor;
use crate::engine::{self, Pos, INPUT_SIZE};
use ort::session::{builder::GraphOptimizationLevel, Session};

const ACTION_SIZE: usize = 1024;
const C_PUCT: f64 = 1.6;

#[derive(Clone)]
struct Node {
    prior: f32,
    player: i8,
    n: u32,
    w: f64,
    expanded: bool,
    terminal: Option<f32>,
    children: Vec<(u16, usize)>,
}

impl Node {
    fn new(prior: f32) -> Node {
        Node {
            prior,
            player: 0,
            n: 0,
            w: 0.0,
            expanded: false,
            terminal: None,
            children: Vec::new(),
        }
    }
}

pub struct Ai {
    session: Mutex<Session>,
}

struct Eval {
    policy: Vec<f32>,
    value: f32,
}

impl Ai {
    pub fn load(model_path: &Path, threads: usize) -> Result<Ai, String> {
        let session = Session::builder()
            .map_err(|e| format!("ort builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("ort opt level: {e}"))?
            .with_intra_threads(threads.max(1))
            .map_err(|e| format!("ort threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| format!("ONNX yuklashda xato: {e}"))?;
        Ok(Ai { session: Mutex::new(session), })
    }
    fn eval(&self, pos: &Pos) -> Result<Eval, String> {
        let mut input = vec![0f32; INPUT_SIZE];
        engine::encode(pos, &mut input);
        let tensor = Tensor::from_array(([1usize, INPUT_SIZE], input)).map_err(|e| format!("tensor: {e}"))?;
        let mut sess = self.session.lock().map_err(|_| "sessiya lock".to_string())?;
        let model_inputs = ort::inputs!["x" => tensor];
        let outputs = sess.run(model_inputs).map_err(|e| format!("ONNX run: {e}"))?;
        let (_p_shape, policy) = outputs["policy_logits"].try_extract_tensor::<f32>().map_err(|e| format!("policy extract: {e}"))?;
        let (_w_shape, wdl) = outputs["wdl_logits"].try_extract_tensor::<f32>().map_err(|e| format!("wdl extract: {e}"))?;
        let mx = wdl.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e0 = (wdl[0] - mx).exp();
        let e1 = (wdl[1] - mx).exp();
        let e2 = (wdl[2] - mx).exp();
        let s = e0 + e1 + e2;
        let value = (e0 - e2) / s;
        Ok(Eval {
            policy: policy.to_vec(),
            value,
        })
    }
    pub fn search(&self, pos: &Pos, sims: u32, time_limit_ms: u64, ) -> Result<SearchResult, String> {
        if sims == 0 {
            let (winner, moves) = engine::status(pos);
            if winner.is_some() || moves.is_empty() {
                return Err("Legal harakat yo'q".to_string());
            }
            let ev = self.eval(pos)?;
            let canon: Vec<u16> = moves.iter().map(|&(f, t)| engine::canon_action(pos, f as u16 * 32 + t as u16)).collect();
            let mut mx = f64::NEG_INFINITY;
            for &c in &canon {
                mx = mx.max(ev.policy[c as usize] as f64);
            }
            let exps: Vec<f64> = canon.iter().map(|&c| ((ev.policy[c as usize] as f64) - mx).exp()).collect();
            let sum: f64 = exps.iter().sum();
            let (mut best_i, mut best_p) = (0usize, f64::NEG_INFINITY);
            for (i, &p) in exps.iter().enumerate() {
                if p > best_p {
                    best_p = p;
                    best_i = i;
                }
            }
            let (f, t) = moves[best_i];
            let confidence = if sum > 0.0 { (best_p / sum) as f32 } else { 0.0 };
            return Ok(SearchResult {
                action: f as u16 * 32 + t as u16,
                confidence,
                value: ev.value,
                sims_done: 0,
            });
        }
        let mut nodes: Vec<Node> = vec![Node::new(1.0)];
        let root = 0usize;
        self.expand(&mut nodes, root, pos)?;
        let start = std::time::Instant::now();
        let mut done = 0u32;
        for i in 0..sims {
            self.simulate(&mut nodes, root, pos)?;
            done = i + 1;
            if time_limit_ms > 0 && (i & 31) == 0 {
                if start.elapsed().as_millis() as u64 >= time_limit_ms { break; }
            }
        }
        let root_node = &nodes[root];
        if root_node.children.is_empty() {
            return Err("Legal harakat yo'q".to_string());
        }
        let (mut best_a, mut best_n) = (0u16, 0u32);
        let mut total_n = 0u32;
        for &(a, ci) in &root_node.children {
            let n = nodes[ci].n;
            total_n += n;
            if n > best_n {
                best_n = n;
                best_a = a;
            }
        }
        let confidence = if total_n > 0 {
            best_n as f32 / total_n as f32
        } else {
            0.0
        };
        let value = if root_node.n > 0 {
            (root_node.w / root_node.n as f64) as f32
        } else {
            0.0
        };
        Ok(SearchResult {
            action: best_a,
            confidence,
            value,
            sims_done: done,
        })
    }
    fn simulate(&self, nodes: &mut Vec<Node>, root: usize, root_pos: &Pos) -> Result<(), String> {
        let mut st = root_pos.clone();
        let mut idx = root;
        let mut path = vec![idx];
        loop {
            let nd = &nodes[idx];
            if !nd.expanded || nd.terminal.is_some() {
                break;
            }
            let (a, ci) = best_child(nodes, idx);
            engine::apply(&mut st, a);
            idx = ci;
            path.push(idx);
        }
        let leaf_player = st.player;
        let value: f64;
        if let Some(tv) = nodes[idx].terminal {
            value = tv as f64;
        } else {
            let (winner, moves) = engine::status(&st);
            if let Some(wn) = winner {
                let tv = if wn == 0 {
                    0.0
                } else if wn == leaf_player {
                    1.0
                } else {
                    -1.0
                };
                nodes[idx].player = leaf_player;
                nodes[idx].terminal = Some(tv as f32);
                value = tv;
            } else {
                let ev = self.eval(&st)?;
                expand_with_eval(nodes, idx, &st, &moves, &ev);
                value = ev.value as f64;
            }
        }
        for &i in &path {
            let nd = &mut nodes[i];
            nd.n += 1;
            nd.w += if nd.player == leaf_player { value } else { -value };
        }
        Ok(())
    }
    fn expand(&self, nodes: &mut Vec<Node>, idx: usize, pos: &Pos) -> Result<(), String> {
        if nodes[idx].expanded { return Ok(()); }
        let (winner, moves) = engine::status(pos);
        let leaf_player = pos.player;
        if let Some(wn) = winner {
            let tv = if wn == 0 {
                0.0
            } else if wn == leaf_player {
                1.0
            } else {
                -1.0
            };
            nodes[idx].player = leaf_player;
            nodes[idx].terminal = Some(tv);
            return Ok(());
        }
        let ev = self.eval(pos)?;
        expand_with_eval(nodes, idx, pos, &moves, &ev);
        nodes[idx].n += 1;
        nodes[idx].w += ev.value as f64;
        Ok(())
    }
}

pub struct SearchResult {
    pub action: u16,
    pub confidence: f32,
    pub value: f32,
    pub sims_done: u32,
}

fn best_child(nodes: &[Node], idx: usize) -> (u16, usize) {
    let node = &nodes[idx];
    let sqrt_n = ((node.n + 1) as f64).sqrt();
    let (mut ba, mut bc, mut bs) = (0u16, 0usize, f64::NEG_INFINITY);
    for &(a, ci) in &node.children {
        let ch = &nodes[ci];
        let nv = ch.n;
        let q = if nv > 0 {
            let w_signed = if ch.player == node.player { ch.w } else { -ch.w };
            w_signed / nv as f64
        } else {
            0.0
        };
        let s = q + C_PUCT * ch.prior as f64 * sqrt_n / (1.0 + nv as f64);
        if s > bs {
            ba = a;
            bc = ci;
            bs = s;
        }
    }
    (ba, bc)
}

fn expand_with_eval(nodes: &mut Vec<Node>, idx: usize, st: &Pos, moves: &[(u8, u8)], ev: &Eval, ) {
    let env: Vec<u16> = moves.iter().map(|&(f, t)| f as u16 * 32 + t as u16).collect();
    let canon: Vec<u16> = env.iter().map(|&a| engine::canon_action(st, a)).collect();
    let mut mx = f64::NEG_INFINITY;
    for &ci in &canon {
        mx = mx.max(ev.policy[ci as usize] as f64);
    }
    let mut pr: Vec<f64> = canon.iter().map(|&ci| ((ev.policy[ci as usize] as f64) - mx).exp()).collect();
    let s: f64 = pr.iter().sum();
    if s > 0.0 {
        for v in pr.iter_mut() { *v /= s; }
    }
    let leaf_player = st.player;
    let mut kids = Vec::with_capacity(env.len());
    for (&a, &q) in env.iter().zip(pr.iter()) {
        let ni = nodes.len();
        nodes.push(Node::new(q as f32));
        kids.push((a, ni));
    }
    let nd = &mut nodes[idx];
    nd.children = kids;
    nd.expanded = true;
    nd.player = leaf_player;
    let _ = ACTION_SIZE;
}