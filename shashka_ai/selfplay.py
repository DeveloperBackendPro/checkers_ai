""" Parallel self-play — lockstep-batch MCTS bilan.
Bir vaqtning o'zida N ta o'yin yuritiladi; har bir MCTS simulyatsiya
to'lqinida barcha o'yinlarning barglari BITTA batch qilib GPU ga yuboriladi.
Bu GPU dan samarali foydalanishning kaliti.
Qo'shimchalar:
  - Curriculum Learning: o'yinlarning bir qismi tasodifiy kam-donali
    (endshpil) pozitsiyalardan boshlanadi — model endshpilni chuqur o'rganadi.
  - Opening Diversification: birinchi bir necha yarim-yurish tasodifiy
    tanlanadi (qidiruvsiz, treningga yozilmaydi) — debyut xilma-xilligi.
  - Temperature scheduling: dastlabki yurishlarda tau=1, keyin argmax.
Trening yozuvi: (x[194], canon_action_idx[], pi[], player, -> z_wdl_class)
z_wdl_class: 0 = yutdi, 1 = durang, 2 = yutqazdi (o'sha pozitsiya
o'yinchisi nuqtai nazaridan). """

from __future__ import annotations
import numpy as np
from collections import Counter
from mcts import MCTS, Node, EvalFn
from dataclasses import dataclass, field
from typing import Counter as CounterT, Dict, List, Optional, Tuple
from checkers_engine import (BK, BM, SQ_TO_RC, WHITE, BLACK, WK, WM, GameState)
Position = Tuple[np.ndarray, np.ndarray, np.ndarray, int]

def random_endgame_pieces(rng: np.random.Generator, max_per_side: int = 3) -> Tuple[List[int], int]:
    pieces = [0] * 32
    n_w = int(rng.integers(1, max_per_side + 1))
    n_b = int(rng.integers(1, max_per_side + 1))
    squares = rng.choice(32, size=n_w + n_b, replace=False)
    for i, sq in enumerate(squares):
        white = i < n_w
        r, _ = SQ_TO_RC[int(sq)]
        king = bool(rng.random() < 0.5)
        if white:
            pieces[int(sq)] = 2 if (king or r == 0) else 1
        else:
            pieces[int(sq)] = -2 if (king or r == 7) else -1
    player = WHITE if rng.random() < 0.5 else BLACK
    return pieces, player

def random_endgame_state(rng: np.random.Generator, max_per_side: int = 3, max_attempts: int = 200) -> GameState:
    for _ in range(max_attempts):
        pieces, player = random_endgame_pieces(rng, max_per_side)
        st = GameState(empty=True)
        for sq, pc in enumerate(pieces):
            if pc:
                r, c = SQ_TO_RC[sq]
                st.board[r, c] = pc
        st.player = player
        st.start_from_setup()
        winner, _ = st.status()
        if winner is None:
            return st
    return GameState()

@dataclass
class _Slot:
    state: GameState
    root: Node
    records: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = field(default_factory=list)
    is_curriculum: bool = False

@dataclass
class SelfPlayStats:
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    games: int = 0
    total_plies: int = 0

    @property
    def avg_length(self) -> float:
        return self.total_plies / max(self.games, 1)

    @property
    def draw_rate(self) -> float:
        return self.draws / max(self.games, 1)

class ParallelSelfPlay:
    def __init__(self, eval_fn: EvalFn, n_parallel: int = 64, sims: int = 160,
        temp_moves: int = 24, temp_schedule: str = "step",
        opening_random_plies: int = 4,
        curriculum_frac: float = 0.25, curriculum_max_pieces: int = 4,
        leaves_per_wave: int = 4, c_puct: float = 1.6,
        dirichlet_alpha: float = 1.0, dirichlet_eps: float = 0.25,
        seed: Optional[int] = None) -> None:
        self.eval_fn = eval_fn
        self.n_parallel = n_parallel
        self.sims = sims
        self.temp_moves = temp_moves
        self.temp_schedule = temp_schedule
        self.opening_random_plies = opening_random_plies
        self.curriculum_frac = curriculum_frac
        self.curriculum_max_pieces = curriculum_max_pieces
        self.leaves_per_wave = max(1, leaves_per_wave)
        self.rng = np.random.default_rng(seed)
        self.mcts = MCTS(eval_fn, c_puct=c_puct, dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps, seed=seed)

    def _temperature(self, ply: int) -> float:
        if self.temp_schedule == "linear":
            return max(0.0, 1.0 - ply / max(2 * self.temp_moves, 1))
        return 1.0 if ply < self.temp_moves else 0.0

    def _new_slot(self) -> _Slot:
        if self.rng.random() < self.curriculum_frac:
            return _Slot(random_endgame_state(self.rng, max_per_side=self.curriculum_max_pieces), Node(1.0), is_curriculum=True)
        st = GameState()
        for _ in range(self.opening_random_plies):
            winner, moves = st.status()
            if winner is not None:
                return self._new_slot()
            f, t = moves[int(self.rng.integers(0, len(moves)))]
            st.apply(f * 32 + t)
        winner, _ = st.status()
        if winner is not None:
            return self._new_slot()
        return _Slot(st, Node(1.0))

    def _expand_roots(self, slots: List[_Slot]) -> None:
        jobs = []
        for s in slots:
            if not s.root.is_expanded and s.root.terminal_value is None:
                st = s.state.clone()
                x = self.mcts.prepare_leaf(s.root, st)
                if x is not None:
                    jobs.append((s.root, st, x))
        if jobs:
            X = np.stack([j[2] for j in jobs])
            logits, vals = self.eval_fn(X)
            for (root, st, _), lg, v in zip(jobs, logits, vals):
                self.mcts.finish_leaf(root, st, lg)
                self.mcts.backprop([root], root.player, float(v))

    def _simulate_wave(self, slots: List[_Slot]) -> None:
        jobs = []
        for s in slots:
            for _ in range(self.leaves_per_wave):
                path, leaf, st = self.mcts.collect(s.root, s.state)
                if leaf.terminal_value is not None:
                    self.mcts.backprop(path, leaf.player, leaf.terminal_value, revert_vloss=True)
                    continue
                if leaf.pending is not None:
                    self.mcts.revert_virtual(path)
                    continue
                x = self.mcts.prepare_leaf(leaf, st)
                if x is None:
                    self.mcts.backprop(path, leaf.player, leaf.terminal_value, revert_vloss=True)
                else:
                    jobs.append((path, leaf, st, x))
        if jobs:
            X = np.stack([j[3] for j in jobs])
            logits, vals = self.eval_fn(X)
            for (path, leaf, st, _), lg, v in zip(jobs, logits, vals):
                self.mcts.finish_leaf(leaf, st, lg)
                self.mcts.backprop(path, leaf.player, float(v), revert_vloss=True)

    def play_games(self, n_games: int, progress_cb=None) -> Tuple[List[Position], SelfPlayStats]:
        results: List[Position] = []
        stats = SelfPlayStats()
        remaining = n_games
        slots: List[_Slot] = []
        while remaining > 0 and len(slots) < self.n_parallel:
            slots.append(self._new_slot())
            remaining -= 1
        while slots:
            self._expand_roots(slots)
            for s in slots:
                self.mcts.add_noise(s.root)
            n_waves = max(1, (self.sims + self.leaves_per_wave - 1) // self.leaves_per_wave)
            for _ in range(n_waves):
                self._simulate_wave(slots)
            finished: List[_Slot] = []
            for s in slots:
                acts = list(s.root.children.keys())
                visits = np.array([s.root.children[a].n for a in acts], dtype=np.float64)
                if visits.sum() <= 0:
                    visits += 1.0
                pi = (visits / visits.sum()).astype(np.float32)
                canon = np.array([s.state.canon_action(a) for a in acts], dtype=np.int64)
                s.records.append((s.state.encode(), canon, pi, s.state.player))
                tau = self._temperature(s.state.ply)
                if tau <= 1e-3:
                    a = acts[int(np.argmax(visits))]
                else:
                    w = visits ** (1.0 / tau)
                    w /= w.sum()
                    a = acts[int(self.rng.choice(len(acts), p=w))]
                s.state.apply(a)
                s.root = s.root.children.get(a) or Node(1.0)
                winner, _ = s.state.status()
                if winner is not None:
                    for (x, idx, p, pl) in s.records:
                        if winner == 0:
                            z = 1
                        elif winner == pl:
                            z = 0
                        else:
                            z = 2
                        results.append((x, idx, p, z))
                    stats.games += 1
                    stats.total_plies += s.state.ply
                    if winner == WHITE:
                        stats.white_wins += 1
                    elif winner == BLACK:
                        stats.black_wins += 1
                    else:
                        stats.draws += 1
                    finished.append(s)
            for s in finished:
                slots.remove(s)
                if remaining > 0:
                    slots.append(self._new_slot())
                    remaining -= 1
            if progress_cb is not None and finished:
                progress_cb(stats)
        return results, stats

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: List[Position] = []
        self.ptr = 0

    def extend(self, items: List[Position]) -> None:
        for it in items:
            if len(self.data) < self.capacity:
                self.data.append(it)
            else:
                self.data[self.ptr] = it
                self.ptr = (self.ptr + 1) % self.capacity

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from checkers_engine import ACTION_SIZE
        idxs = rng.integers(0, len(self.data), size=batch_size)
        xs = np.stack([self.data[i][0] for i in idxs])
        Pi = np.zeros((batch_size, ACTION_SIZE), dtype=np.float32)
        zs = np.empty(batch_size, dtype=np.int64)
        for j, i in enumerate(idxs):
            _, ci, p, z = self.data[i]
            Pi[j, ci] = p
            zs[j] = z
        return xs, Pi, zs

    def state(self) -> dict:
        return {"data": self.data, "ptr": self.ptr, "capacity": self.capacity}

    def load_state(self, st: dict) -> None:
        data = st.get("data", [])
        if len(data) > self.capacity:
            data = data[-self.capacity:]
        self.data = list(data)
        self.ptr = st.get("ptr", 0) % max(1, self.capacity)
        if self.ptr >= len(self.data):
            self.ptr = 0