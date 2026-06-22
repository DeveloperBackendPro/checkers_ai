//! Lockstep-batch MCTS o'rmoni (forest) — N ta o'yin daraxtini parallel
//! yuritadi. NN baholash Python tomonda (PyTorch/GPU); Rust faqat tezkor
//! qismni qiladi: tanlash, kengaytirish, backprop, holat klonlash.
//! Semantika Python `mcts.py` bilan bir xil: PUCT, virtual loss,
//! tugun qiymati o'sha tugunda yuruvchi o'yinchi nuqtai nazaridan.

use crate::engine::{self, Pos};
pub struct Rng {
    s: u64,
    cached_normal: Option<f64>,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Rng {
            s: s ^ 0x9E37_79B9_7F4A_7C15,
            cached_normal: None,
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
    pub fn normal(&mut self) -> f64 {
        if let Some(v) = self.cached_normal.take() {return v;}
        let (mut u1, u2) = (self.uniform(), self.uniform());
        if u1 < 1e-300 {u1 = 1e-300;}
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f64::consts::PI * u2;
        self.cached_normal = Some(r * th.sin());
        r * th.cos()
    }
    pub fn gamma(&mut self, alpha: f64) -> f64 {
        if alpha < 1.0 {
            let g = self.gamma(alpha + 1.0);
            let u = self.uniform().max(1e-300);
            return g * u.powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {continue;}
            let v3 = v * v * v;
            let u = self.uniform().max(1e-300);
            if u.ln() < 0.5 * x * x + d - d * v3 + d * v3.ln() {return d * v3;}
        }
    }
    pub fn dirichlet(&mut self, n: usize, alpha: f64) -> Vec<f64> {
        let mut g: Vec<f64> = (0..n).map(|_| self.gamma(alpha)).collect();
        let s: f64 = g.iter().sum();
        if s > 0.0 {
            for v in g.iter_mut() {*v /= s;}
        }
        g
    }
}

#[derive(Clone)]
pub struct Node {
    pub prior: f32,
    pub player: i8,
    pub n: u32,
    pub w: f64,
    pub vloss: u32,
    pub expanded: bool,
    pub terminal: Option<f32>,
    pub pending: bool,
    pub children: Vec<(u16, u32)>,
}

impl Node {
    pub fn new(prior: f32) -> Node {
        Node {
            prior,
            player: 0,
            n: 0,
            w: 0.0,
            vloss: 0,
            expanded: false,
            terminal: None,
            pending: false,
            children: Vec::new(),
        }
    }
}

pub struct Tree {
    pub nodes: Vec<Node>,
    pub root: u32,
    pub state: Pos,
    pub active: bool,
}

impl Tree {
    pub fn new(state: Pos) -> Tree {
        Tree {
            nodes: vec![Node::new(1.0)],
            root: 0,
            state,
            active: true,
        }
    }
}

struct Pending {
    game: usize,
    node: u32,
    path: Vec<u32>,
    env: Vec<u16>,
    canon: Vec<u16>,
    leaf_player: i8,
}

pub struct Forest {
    pub trees: Vec<Tree>,
    pending: Vec<Pending>,
    c_puct: f64,
    alpha: f64,
    eps: f64,
    rng: Rng,
}

fn backprop(nodes: &mut [Node], path: &[u32], leaf_player: i8, v: f64, revert: bool) {
    for &i in path {
        let nd = &mut nodes[i as usize];
        if revert {nd.vloss -= 1;}
        nd.n += 1;
        nd.w += if nd.player == leaf_player { v } else { -v };
    }
}

impl Forest {
    pub fn new(n_games: usize, c_puct: f64, alpha: f64, eps: f64, seed: u64) -> Forest {
        Forest {
            trees: (0..n_games).map(|_| Tree::new(Pos::initial())).collect(),
            pending: Vec::new(),
            c_puct,
            alpha,
            eps,
            rng: Rng::new(seed),
        }
    }
    fn best_child(&self, g: usize, idx: u32) -> (u16, u32) {
        let tree = &self.trees[g];
        let node = &tree.nodes[idx as usize];
        let sqrt_n = ((node.n + node.vloss + 1) as f64).sqrt();
        let (mut ba, mut bc, mut bs) = (0u16, 0u32, -1e18f64);
        for &(a, ci) in &node.children {
            let ch = &tree.nodes[ci as usize];
            let nv = ch.n + ch.vloss;
            let q = if nv > 0 {
                let w_signed = if ch.player == node.player { ch.w } else { -ch.w };
                (w_signed - ch.vloss as f64) / nv as f64
            } else {
                0.0
            };
            let s = q + self.c_puct * ch.prior as f64 * sqrt_n / (1.0 + nv as f64);
            if s > bs {
                ba = a;
                bc = ci;
                bs = s;
            }
        }
        (ba, bc)
    }
    fn collect_one(&mut self, g: usize, buf: &mut Vec<f32>) {
        let mut st = self.trees[g].state.clone();
        let mut idx = self.trees[g].root;
        let mut path = vec![idx];
        self.trees[g].nodes[idx as usize].vloss += 1;
        loop {
            let nd = &self.trees[g].nodes[idx as usize];
            if !nd.expanded || nd.terminal.is_some() {
                break;
            }
            let (a, ci) = self.best_child(g, idx);
            engine::apply(&mut st, a);
            idx = ci;
            self.trees[g].nodes[idx as usize].vloss += 1;
            path.push(idx);
        }
        let nd = &self.trees[g].nodes[idx as usize];
        if let Some(tv) = nd.terminal {
            let lp = nd.player;
            backprop(&mut self.trees[g].nodes, &path, lp, tv as f64, true);
            return;
        }
        if nd.pending {
            for &i in &path {self.trees[g].nodes[i as usize].vloss -= 1;}
            return;
        }
        let (winner, moves) = engine::status(&st);
        let leaf_player = st.player;
        if let Some(wn) = winner {
            let tv = if wn == 0 {
                0.0
            } else if wn == leaf_player {
                1.0
            } else {
                -1.0
            };
            let nd = &mut self.trees[g].nodes[idx as usize];
            nd.player = leaf_player;
            nd.terminal = Some(tv);
            backprop(&mut self.trees[g].nodes, &path, leaf_player, tv as f64, true);
            return;
        }
        let env: Vec<u16> = moves.iter().map(|&(f, t)| f as u16 * 32 + t as u16).collect();
        let canon: Vec<u16> = env.iter().map(|&a| engine::canon_action(&st, a)).collect();
        {
            let nd = &mut self.trees[g].nodes[idx as usize];
            nd.player = leaf_player;
            nd.pending = true;
        }
        let start = buf.len();
        buf.resize(start + engine::INPUT_SIZE, 0.0);
        engine::encode(&st, &mut buf[start..start + engine::INPUT_SIZE]);
        self.pending.push(Pending {
            game: g,
            node: idx,
            path,
            env,
            canon,
            leaf_player,
        });
    }
    pub fn collect_roots(&mut self, buf: &mut Vec<f32>) {
        for g in 0..self.trees.len() {
            if !self.trees[g].active {continue;}
            let r = self.trees[g].root as usize;
            if !self.trees[g].nodes[r].expanded && self.trees[g].nodes[r].terminal.is_none() {
                self.collect_one(g, buf);
            }
        }
    }
    pub fn collect(&mut self, leaves: usize, buf: &mut Vec<f32>, max_pending: usize) {
        'outer: for g in 0..self.trees.len() {
            if !self.trees[g].active {continue;}
            for _ in 0..leaves {
                self.collect_one(g, buf);
                if self.pending.len() >= max_pending {break 'outer;}
            }
        }
    }
    pub fn collect_subset(&mut self, subset: &[usize], leaves: usize, buf: &mut Vec<f32>, max_pending: usize) {
        'outer: for &g in subset {
            if !self.trees[g].active {continue;}
            for _ in 0..leaves {
                self.collect_one(g, buf);
                if self.pending.len() >= max_pending {break 'outer;}
            }
        }
    }
    pub fn collect_roots_subset(&mut self, subset: &[usize], buf: &mut Vec<f32>) {
        for &g in subset {
            if !self.trees[g].active {continue;}
            let r = self.trees[g].root as usize;
            if !self.trees[g].nodes[r].expanded && self.trees[g].nodes[r].terminal.is_none() {self.collect_one(g, buf);}
        }
    }
    pub fn apply_evals(&mut self, logits: &[f32], values: &[f32]) {
        let pend = std::mem::take(&mut self.pending);
        for (k, p) in pend.into_iter().enumerate() {
            let row = &logits[k * 1024..(k + 1) * 1024];
            let mut mx = f64::NEG_INFINITY;
            for &ci in &p.canon {mx = mx.max(row[ci as usize] as f64);}
            let mut pr: Vec<f64> = p.canon.iter().map(|&ci| ((row[ci as usize] as f64) - mx).exp()).collect();
            let s: f64 = pr.iter().sum();
            for v in pr.iter_mut() {*v /= s;}
            let tree = &mut self.trees[p.game];
            let mut kids = Vec::with_capacity(p.env.len());
            for (&a, &q) in p.env.iter().zip(pr.iter()) {
                let ni = tree.nodes.len() as u32;
                tree.nodes.push(Node::new(q as f32));
                kids.push((a, ni));
            }
            {
                let nd = &mut tree.nodes[p.node as usize];
                nd.children = kids;
                nd.expanded = true;
                nd.pending = false;
            }
            backprop(&mut tree.nodes, &p.path, p.leaf_player, values[k] as f64, true);
        }
    }

    pub fn add_noise(&mut self, g: usize) {
        let n = self.trees[g].nodes[self.trees[g].root as usize].children.len();
        if n == 0 {return;}
        let noise = self.rng.dirichlet(n, self.alpha);
        let root = self.trees[g].root as usize;
        let kids: Vec<u32> = self.trees[g].nodes[root].children.iter().map(|&(_, c)| c).collect();
        for (ci, nz) in kids.into_iter().zip(noise) {
            let ch = &mut self.trees[g].nodes[ci as usize];
            ch.prior = ((1.0 - self.eps) * ch.prior as f64 + self.eps * nz) as f32;
        }
    }
    pub fn visits(&self, g: usize) -> (Vec<u16>, Vec<u32>) {
        let tree = &self.trees[g];
        let root = &tree.nodes[tree.root as usize];
        let mut acts = Vec::with_capacity(root.children.len());
        let mut vis = Vec::with_capacity(root.children.len());
        for &(a, ci) in &root.children {
            acts.push(a);
            vis.push(tree.nodes[ci as usize].n);
        }
        (acts, vis)
    }
    pub fn advance(&mut self, g: usize, action: u16) {
        engine::apply(&mut self.trees[g].state, action);
        let tree = &mut self.trees[g];
        let child = tree.nodes[tree.root as usize].children.iter().find(|&&(a, _)| a == action).map(|&(_, c)| c);
        match child {
            Some(ci) => {
                let mut new_nodes: Vec<Node> = Vec::new();
                copy_subtree(&tree.nodes, ci, &mut new_nodes);
                tree.nodes = new_nodes;
                tree.root = 0;
            }
            None => {
                tree.nodes = vec![Node::new(1.0)];
                tree.root = 0;
            }
        }
    }
    pub fn winner(&self, g: usize) -> Option<i8> {
        engine::status(&self.trees[g].state).0
    }
    pub fn reset(&mut self, g: usize, state: Pos) {
        self.trees[g] = Tree::new(state);
    }
}

fn copy_subtree(old: &[Node], oi: u32, new: &mut Vec<Node>) -> u32 {
    let ni = new.len() as u32;
    let mut nd = old[oi as usize].clone();
    let kids = std::mem::take(&mut nd.children);
    new.push(nd);
    let mut new_kids = Vec::with_capacity(kids.len());
    for (a, c) in kids {
        let nci = copy_subtree(old, c, new);
        new_kids.push((a, nci));
    }
    new[ni as usize].children = new_kids;
    ni
}