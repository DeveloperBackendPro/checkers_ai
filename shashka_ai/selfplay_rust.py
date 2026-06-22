""" Rust dvijok + Rust MCTS asosidagi tezkor parallel self-play.
`selfplay.ParallelSelfPlay` bilan BIR XIL interfeys va semantika:
curriculum, debyut diversifikatsiyasi, temperatura jadvali, Dirichlet noise,
virtual loss, daraxt qayta ishlatish. Faqat og'ir qism (tanlash, klonlash,
kengaytirish, backprop, kodlash) Rust'da bajariladi — NN baholash Python'da
(PyTorch/GPU) qoladi.
Dvijok to'g'riligi `test_diff.py` bilan Python etaloniga bog'langan
(bit-darajada moslik). """

from __future__ import annotations
import numpy as np
from mcts import EvalFn
import shashka_engine as se
from typing import List, Optional, Tuple
from selfplay import Position, SelfPlayStats, random_endgame_pieces

class RustParallelSelfPlay:
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
        self.forest = se.MctsForest(
        n_parallel, c_puct, dirichlet_alpha,
        dirichlet_eps, int(seed if seed is not None else self.rng.integers(1, 2 ** 62)))

    def _temperature(self, ply: int) -> float:
        if self.temp_schedule == "linear":
            return max(0.0, 1.0 - ply / max(2 * self.temp_moves, 1))
        return 1.0 if ply < self.temp_moves else 0.0

    def _new_state(self) -> se.GameState:
        if self.rng.random() < self.curriculum_frac:
            for _ in range(200):
                pieces, player = random_endgame_pieces(
                    self.rng, self.curriculum_max_pieces)
                st = se.GameState.from_board(pieces, player)
                if st.winner() is None:
                    return st
            return se.GameState()
        st = se.GameState()
        for _ in range(self.opening_random_plies):
            winner, acts = st.status()
            if winner is not None:
                return self._new_state()
            st.apply(int(acts[int(self.rng.integers(0, len(acts)))]))
        if st.winner() is not None:
            return self._new_state()
        return st

    def _evaluate_pending(self, enc: np.ndarray, k: int) -> None:
        if k == 0:
            return
        X = np.asarray(enc, dtype=np.float32).reshape(k, -1)
        logits, vals = self.eval_fn(X)
        self.forest.apply_evals(
            np.ascontiguousarray(logits, dtype=np.float32),
            np.ascontiguousarray(vals, dtype=np.float32))

    @staticmethod
    def _canon(a: int, player: int) -> int:
        if player == 1:
            return a
        f, t = divmod(a, 32)
        return (31 - f) * 32 + (31 - t)

    def play_games(self, n_games: int, progress_cb=None) -> Tuple[List[Position], SelfPlayStats]:
        results: List[Position] = []
        stats = SelfPlayStats()
        forest = self.forest
        remaining = n_games
        n_slots = min(self.n_parallel, n_games)
        records: List[List] = [[] for _ in range(self.n_parallel)]
        for g in range(self.n_parallel):
            if g < n_slots:
                forest.set_state(g, self._new_state())
                forest.set_active(g, True)
                remaining -= 1
            else:
                forest.set_active(g, False)
        active = set(range(n_slots))
        n_waves = max(1, (self.sims + self.leaves_per_wave - 1) // self.leaves_per_wave)
        while active:
            enc, k = forest.collect_roots()
            self._evaluate_pending(enc, k)
            for g in active:
                forest.add_noise(g)
            for _ in range(n_waves):
                enc, k = forest.collect(self.leaves_per_wave)
                self._evaluate_pending(enc, k)
            finished: List[int] = []
            for g in list(active):
                acts, vis = forest.visits(g)
                acts = [int(a) for a in acts]
                visits = np.array(vis, dtype=np.float64)
                if visits.sum() <= 0:
                    visits += 1.0
                pi = (visits / visits.sum()).astype(np.float32)
                player = int(forest.root_player(g))
                canon = np.array([self._canon(a, player) for a in acts], dtype=np.int64)
                x = np.array(forest.root_encode_g(g), dtype=np.float32)
                records[g].append((x, canon, pi, player))
                tau = self._temperature(int(forest.root_ply(g)))
                if tau <= 1e-3:
                    a = acts[int(np.argmax(visits))]
                else:
                    w = visits ** (1.0 / tau)
                    w /= w.sum()
                    a = acts[int(self.rng.choice(len(acts), p=w))]
                forest.advance(g, a)
                winner = forest.winner(g)
                if winner is not None:
                    winner = int(winner)
                    for (x, idx, p, pl) in records[g]:
                        z = 1 if winner == 0 else (0 if winner == pl else 2)
                        results.append((x, idx, p, z))
                    stats.games += 1
                    stats.total_plies += int(forest.root_ply(g))
                    if winner == 1:
                        stats.white_wins += 1
                    elif winner == -1:
                        stats.black_wins += 1
                    else:
                        stats.draws += 1
                    records[g] = []
                    finished.append(g)
            for g in finished:
                if remaining > 0:
                    forest.set_state(g, self._new_state())
                    remaining -= 1
                else:
                    forest.set_active(g, False)
                    active.discard(g)
            if progress_cb is not None and finished:
                progress_cb(stats)
        return results, stats