//! Rus shashkasi dvijogi — bitboard (u32, 32 qora katak).
//! MUHIM: bu modul Python `checkers_engine.py` semantikasini AYNAN
//! takrorlaydi (yurish tartibi, qoidalar, kodlash). Ikkalasi differential
//! fuzzing bilan solishtiriladi (`test_diff.py`) — bittagina farq = bug.

use once_cell::sync::Lazy;
pub const NO_PROGRESS_LIMIT: u16 = 60;
pub const MAX_PLIES: u16 = 300;
pub const INPUT_SIZE: usize = 194;

pub static SQ_RC: Lazy<Vec<(i8, i8)>> = Lazy::new(|| {
    let mut v = Vec::with_capacity(32);
    for r in 0..8i8 { for c in 0..8i8 { if (r + c) % 2 == 1 { v.push((r, c)); } } }
    v
});

pub static RC_SQ: Lazy<[[i8; 8]; 8]> = Lazy::new(|| {
    let mut m = [[-1i8; 8]; 8];
    for (i, &(r, c)) in SQ_RC.iter().enumerate() {
        m[r as usize][c as usize] = i as i8;
    }
    m
});

const DIRS: [(i8, i8); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];

pub static RAYS: Lazy<Vec<[Vec<u8>; 4]>> = Lazy::new(|| {
    let mut all = Vec::with_capacity(32);
    for sq in 0..32usize {
        let (r0, c0) = SQ_RC[sq];
        let mut dirs: [Vec<u8>; 4] = Default::default();
        for (d, &(dr, dc)) in DIRS.iter().enumerate() {
            let (mut r, mut c) = (r0 + dr, c0 + dc);
            while (0..8).contains(&r) && (0..8).contains(&c) {
                dirs[d].push(RC_SQ[r as usize][c as usize] as u8);
                r += dr;
                c += dc;
            }
        }
        all.push(dirs);
    }
    all
});

#[inline]
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

pub static ZOBRIST: Lazy<([[u64; 32]; 4], u64)> = Lazy::new(|| {
    let mut seed: u64 = 0x5A5A_2026_0610;
    let mut t = [[0u64; 32]; 4];
    for c in 0..4 {
        for s in 0..32 {
            t[c][s] = splitmix64(&mut seed);
        }
    }
    let side = splitmix64(&mut seed);
    (t, side)
});

#[inline]
pub fn bit(sq: u8) -> u32 {
    1u32 << sq
}

#[derive(Clone)]
pub struct Pos {
    pub white: u32,
    pub black: u32,
    pub kings: u32,
    pub player: i8,
    pub chain_sq: i8,
    pub captured: u32,
    pub halfmove: u16,
    pub ply: u16,
    pub window: Vec<u64>,
}

pub fn hash(p: &Pos) -> u64 {
    let (tbl, side) = &*ZOBRIST;
    let mut h = 0u64;
    let mut occ = p.white | p.black;
    while occ != 0 {
        let sq = occ.trailing_zeros() as usize;
        occ &= occ - 1;
        let white = p.white & bit(sq as u8) != 0;
        let king = p.kings & bit(sq as u8) != 0;
        let cls = match (white, king) {
            (true, false) => 0,
            (true, true) => 1,
            (false, false) => 2,
            (false, true) => 3,
        };
        h ^= tbl[cls][sq];
    }
    if p.player == -1 {
        h ^= side;
    }
    h
}

impl Pos {
    pub fn initial() -> Pos {
        let (mut w, mut b) = (0u32, 0u32);
        for (sq, &(r, _)) in SQ_RC.iter().enumerate() {
            if r < 3 {
                b |= bit(sq as u8);
            } else if r > 4 {
                w |= bit(sq as u8);
            }
        }
        let mut p = Pos {
            white: w,
            black: b,
            kings: 0,
            player: 1,
            chain_sq: -1,
            captured: 0,
            halfmove: 0,
            ply: 0,
            window: Vec::with_capacity(64),
        };
        let h = hash(&p);
        p.window.push(h);
        p
    }
    pub fn from_board(pieces: &[i8], player: i8) -> Pos {
        let (mut w, mut b, mut k) = (0u32, 0u32, 0u32);
        for (sq, &pc) in pieces.iter().enumerate().take(32) {
            match pc {
                1 => w |= bit(sq as u8),
                2 => {
                    w |= bit(sq as u8);
                    k |= bit(sq as u8);
                }
                -1 => b |= bit(sq as u8),
                -2 => {
                    b |= bit(sq as u8);
                    k |= bit(sq as u8);
                }
                _ => {}
            }
        }
        let mut p = Pos {
            white: w,
            black: b,
            kings: k,
            player,
            chain_sq: -1,
            captured: 0,
            halfmove: 0,
            ply: 0,
            window: Vec::with_capacity(64),
        };
        let h = hash(&p);
        p.window.push(h);
        p
    }
    pub fn board_list(&self) -> Vec<i8> {
        (0..32u8)
            .map(|sq| {
                if self.white & bit(sq) != 0 {
                    if self.kings & bit(sq) != 0 {
                        2
                    } else {
                        1
                    }
                } else if self.black & bit(sq) != 0 {
                    if self.kings & bit(sq) != 0 {
                        -2
                    } else {
                        -1
                    }
                } else {
                    0
                }
            })
            .collect()
    }
}

pub fn raw_captures(white: u32, black: u32, sq: u8, my_white: bool, king: bool, captured: u32, out: &mut Vec<(u8, u8)>,) {
    let occ = white | black;
    let enemy = if my_white { black } else { white };
    let rays = &RAYS[sq as usize];
    if !king {
        for d in 0..4 {
            let ray = &rays[d];
            if ray.len() >= 2 {
                let (m, t) = (ray[0], ray[1]);
                if occ & bit(t) == 0 && enemy & bit(m) != 0 && captured & bit(m) == 0 { out.push((t, m)); }
            }
        }
    } else {
        for d in 0..4 {
            let ray = &rays[d];
            let mut i = 0;
            while i < ray.len() && occ & bit(ray[i]) == 0 { i += 1; }
            if i >= ray.len() { continue; }
            let m = ray[i];
            if enemy & bit(m) == 0 || captured & bit(m) != 0 { continue; }
            let mut j = i + 1;
            while j < ray.len() && occ & bit(ray[j]) == 0 {
                out.push((ray[j], m));
                j += 1;
            }
        }
    }
}

pub fn has_capture(white: u32, black: u32, sq: u8, my_white: bool, king: bool, captured: u32) -> bool {
    let occ = white | black;
    let enemy = if my_white { black } else { white };
    let rays = &RAYS[sq as usize];
    if !king {
        for d in 0..4 {
            let ray = &rays[d];
            if ray.len() >= 2 {
                let (m, t) = (ray[0], ray[1]);
                if occ & bit(t) == 0 && enemy & bit(m) != 0 && captured & bit(m) == 0 { return true; }
            }
        }
    } else {
        for d in 0..4 {
            let ray = &rays[d];
            let mut i = 0;
            while i < ray.len() && occ & bit(ray[i]) == 0 { i += 1; }
            if i >= ray.len() { continue; }
            let m = ray[i];
            if enemy & bit(m) != 0 && captured & bit(m) == 0 && i + 1 < ray.len() && occ & bit(ray[i + 1]) == 0 { return true; }
        }
    }
    false
}

pub fn captures_from(p: &Pos, sq: u8, king: bool, my_white: bool, captured: u32) -> Vec<(u8, u8)> {
    let mut segs = Vec::new();
    raw_captures(p.white, p.black, sq, my_white, king, captured, &mut segs);
    if segs.is_empty() || !king { return segs; }
    let mut caps_order: Vec<u8> = Vec::new();
    for &(_, c) in &segs { if !caps_order.contains(&c) { caps_order.push(c); } }
    let (mut w0, mut b0) = (p.white, p.black);
    if my_white {
        w0 &= !bit(sq);
    } else {
        b0 &= !bit(sq);
    }
    let mut out = Vec::new();
    for &cap in &caps_order {
        let lands: Vec<u8> = segs.iter().filter(|s| s.1 == cap).map(|s| s.0).collect();
        let new_cap = captured | bit(cap);
        let mut cont: Vec<u8> = Vec::new();
        for &t in &lands {
            let (mut w2, mut b2) = (w0, b0);
            if my_white {
                w2 |= bit(t);
            } else {
                b2 |= bit(t);
            }
            if has_capture(w2, b2, t, my_white, true, new_cap) { cont.push(t); }
        }
        let chosen = if cont.is_empty() { &lands } else { &cont };
        for &t in chosen { out.push((t, cap)); }
    }
    out
}

pub fn legal_moves(p: &Pos) -> Vec<(u8, u8)> {
    let my_white = p.player == 1;
    let mine = if my_white { p.white } else { p.black };
    if p.chain_sq >= 0 {
        let sq = p.chain_sq as u8;
        let king = p.kings & bit(sq) != 0;
        return captures_from(p, sq, king, my_white, p.captured).into_iter().map(|(t, _)| (sq, t)).collect();
    }
    let mut caps = Vec::new();
    let mut bbs = mine;
    while bbs != 0 {
        let sq = bbs.trailing_zeros() as u8;
        bbs &= bbs - 1;
        let king = p.kings & bit(sq) != 0;
        for (t, _) in captures_from(p, sq, king, my_white, 0) { caps.push((sq, t)); }
    }
    if !caps.is_empty() { return caps; }
    let occ = p.white | p.black;
    let mut quiets = Vec::new();
    let mut bbs = mine;
    while bbs != 0 {
        let sq = bbs.trailing_zeros() as u8;
        bbs &= bbs - 1;
        let king = p.kings & bit(sq) != 0;
        let rays = &RAYS[sq as usize];
        if !king {
            let dirs: [usize; 2] = if my_white { [0, 1] } else { [2, 3] };
            for d in dirs {
                if let Some(&t) = rays[d].first() { if occ & bit(t) == 0 { quiets.push((sq, t)); } }
            }
        } else {
            for d in 0..4 {
                for &t in &rays[d] {
                    if occ & bit(t) != 0 { break; }
                    quiets.push((sq, t));
                }
            }
        }
    }
    quiets
}

pub fn apply(p: &mut Pos, action: u16) {
    assert!(action < 1024, "action chegaradan tashqari: {} (0..1024)", action);
    let f = (action / 32) as u8;
    let t = (action % 32) as u8;
    let (fr, fc) = SQ_RC[f as usize];
    let (tr, tc) = SQ_RC[t as usize];
    let my_white = p.player == 1;
    let mut king = p.kings & bit(f) != 0;
    let was_man = !king;
    let dr = (tr - fr).signum();
    let dc = (tc - fc).signum();
    let d = DIRS.iter().position(|&(a, b)| a == dr && b == dc).unwrap();
    let mut cap: i8 = -1;
    for &s in &RAYS[f as usize][d] {
        if s == t { break; }
        if (p.white | p.black) & bit(s) != 0 { cap = s as i8; }
    }

    if my_white {
        p.white &= !bit(f);
        p.white |= bit(t);
    } else {
        p.black &= !bit(f);
        p.black |= bit(t);
    }
    if king {
        p.kings &= !bit(f);
    }
    if was_man && ((my_white && tr == 0) || (!my_white && tr == 7)) {
        king = true;
    }
    if king {
        p.kings |= bit(t);
    }
    p.ply += 1;
    if cap >= 0 {
        let new_cap = p.captured | bit(cap as u8);
        if has_capture(p.white, p.black, t, my_white, king, new_cap) {
            p.chain_sq = t as i8;
            p.captured = new_cap;
            return;
        }
        p.white &= !new_cap;
        p.black &= !new_cap;
        p.kings &= !new_cap;
        p.captured = 0;
        p.chain_sq = -1;
        p.halfmove = 0;
        p.player = -p.player;
        p.window.clear();
        let h = hash(p);
        p.window.push(h);
    } else {
        if was_man {
            p.halfmove = 0;
            p.player = -p.player;
            p.window.clear();
        } else {
            p.halfmove += 1;
            p.player = -p.player;
        }
        let h = hash(p);
        p.window.push(h);
    }
}

pub fn status(p: &Pos) -> (Option<i8>, Vec<(u8, u8)>) {
    let moves = legal_moves(p);
    if moves.is_empty() { return (Some(-p.player), moves); }
    if p.chain_sq < 0 {
        let h = hash(p);
        if p.window.iter().filter(|&&x| x == h).count() >= 3 { return (Some(0), moves); }
        if p.halfmove >= NO_PROGRESS_LIMIT || p.ply >= MAX_PLIES { return (Some(0), moves); }
    }
    (None, moves)
}

pub fn encode(p: &Pos, out: &mut [f32]) {
    for v in out.iter_mut() { *v = 0.0; }
    let rot = p.player == -1;
    let mut occ = p.white | p.black;
    while occ != 0 {
        let sq = occ.trailing_zeros() as u8;
        occ &= occ - 1;
        let csq = (if rot { 31 - sq } else { sq }) as usize;
        let white = p.white & bit(sq) != 0;
        let king = p.kings & bit(sq) != 0;
        let mine = white == (p.player == 1);
        let cls = match (mine, king) {
            (true, false) => 0,
            (true, true) => 1,
            (false, false) => 2,
            (false, true) => 3,
        };
        out[csq * 6 + cls] = 1.0;
        if p.captured & bit(sq) != 0 {
            out[csq * 6 + 5] = 1.0;
        }
    }
    if p.chain_sq >= 0 {
        let cs = (if rot {
            31 - p.chain_sq as u8
        } else {
            p.chain_sq as u8
        }) as usize;
        out[cs * 6 + 4] = 1.0;
    }
    out[192] = ((p.halfmove.min(NO_PROGRESS_LIMIT) as f64) / (NO_PROGRESS_LIMIT as f64)) as f32;
    out[193] = if p.chain_sq >= 0 { 1.0 } else { 0.0 };
}

#[inline]
pub fn canon_action(p: &Pos, a: u16) -> u16 {
    if p.player == 1 {
        a
    } else {
        let f = a / 32;
        let t = a % 32;
        (31 - f) * 32 + (31 - t)
    }
}