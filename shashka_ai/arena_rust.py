""" Rust forest asosidagi tezkor parallel arena.
Python `train.play_arena` ning o'rnini bosadi, lekin:
  - barcha arena o'yinlari BIR VAQTDA (parallel) yuritiladi,
  - har MCTS to'lqinida shu navbatda yuruvchi modelga tegishli o'yinlarning
    barglari BITTA batch qilib GPU ga yuboriladi (GPU to'liq ishlatiladi),
  - Rust dvijok (tezlik).
Ya'ni Python arena (bitta o'yin, batch'siz, ~minutlar) o'rniga ~sekundlar.
Har o'yinning yarmi "yangi" oq bilan, yarmi "best" oq bilan boshlanadi
(rang balansi). Har yurishda root navbatchisi (root_player) qaysi modelга
tegishli ekanini aniqlaydi. """

from __future__ import annotations
import numpy as np
from mcts import EvalFn
import shashka_engine as se
from typing import List, Optional

def play_arena_rust(eval_new: EvalFn, eval_best: EvalFn, n_games: int,
    sims: int, seed: int, leaves_per_wave: int = 8,
    opening_temp_plies: int = 6, c_puct: float = 1.6,
    guard=None) -> float:
    rng = np.random.default_rng(seed)
    forest = se.MctsForest(n_games, c_puct, 1.0, 0.0, int(seed))
    new_is_white = np.zeros(n_games, dtype=bool)
    for g in range(n_games):
        new_is_white[g] = (g % 2 == 0)
        forest.set_state(g, se.GameState())
        forest.set_active(g, True)
    active = set(range(n_games))
    n_waves = max(1, (sims + leaves_per_wave - 1) // leaves_per_wave)
    scores = np.zeros(n_games, dtype=np.float64)
    done = np.zeros(n_games, dtype=bool)
    def evaluate(enc, k, ev) -> None:
        if k == 0:
            return
        X = np.asarray(enc, dtype=np.float32).reshape(k, -1)
        logits, vals = ev(X)
        forest.apply_evals(np.ascontiguousarray(logits, dtype=np.float32), np.ascontiguousarray(vals, dtype=np.float32))
    while active:
        new_turn: List[int] = []
        best_turn: List[int] = []
        for g in active:
            p = int(forest.root_player(g))
            new_to_move = (p == 1) == bool(new_is_white[g])
            (new_turn if new_to_move else best_turn).append(g)
        if new_turn:
            enc, k = forest.collect_roots_subset(new_turn)
            evaluate(enc, k, eval_new)
        if best_turn:
            enc, k = forest.collect_roots_subset(best_turn)
            evaluate(enc, k, eval_best)
        for _ in range(n_waves):
            if new_turn:
                enc, k = forest.collect_subset(new_turn, leaves_per_wave)
                evaluate(enc, k, eval_new)
            if best_turn:
                enc, k = forest.collect_subset(best_turn, leaves_per_wave)
                evaluate(enc, k, eval_best)
        finished: List[int] = []
        for g in list(active):
            ply = int(forest.root_ply(g))
            if ply < opening_temp_plies:
                acts, vis = forest.visits(g)
                visits = np.array(vis, dtype=np.float64)
                if visits.sum() <= 0:
                    a = int(acts[0])
                else:
                    a = int(acts[int(rng.choice(len(acts), p=visits / visits.sum()))])
            else:
                a = int(forest.best_action(g))
                if a == 0xFFFF:
                    raise RuntimeError(f"arena: o'yin {g} da yurish topilmadi (kutilmagan  holat — faol o'yinda doim yurish bo'lishi kerak)")
            forest.advance(g, a)
            w = forest.winner(g)
            if w is not None:
                w = int(w)
                if w == 0:
                    scores[g] = 0.5
                elif (w == 1) == bool(new_is_white[g]):
                    scores[g] = 1.0
                else:
                    scores[g] = 0.0
                done[g] = True
                finished.append(g)
        for g in finished:
            forest.set_active(g, False)
            active.discard(g)
        if guard is not None:
            guard.periodic_check()
    return float(scores.sum() / n_games)