"""Benchmark: Python vs Rust — dvijok va self-play tezligi.
    python bench.py
"""

from __future__ import annotations
import time
import numpy as np
from checkers_engine import ACTION_SIZE, GameState as PyState, action_encode

def uniform_eval(x: np.ndarray):
    return (np.zeros((x.shape[0], ACTION_SIZE), np.float32), np.zeros(x.shape[0], np.float32))

def bench_engine_python(n_games: int = 300) -> float:
    rng = np.random.default_rng(0)
    t0 = time.time()
    plies = 0
    for _ in range(n_games):
        st = PyState()
        while True:
            winner, moves = st.status()
            if winner is not None:
                break
            f, t = moves[int(rng.integers(0, len(moves)))]
            st.apply(action_encode(f, t))
            plies += 1
    return plies / (time.time() - t0)

def bench_engine_rust(n_games: int = 300) -> float:
    import shashka_engine as se
    rng = np.random.default_rng(0)
    t0 = time.time()
    plies = 0
    for _ in range(n_games):
        st = se.GameState()
        while True:
            winner, acts = st.status()
            if winner is not None:
                break
            st.apply(int(acts[int(rng.integers(0, len(acts)))]))
            plies += 1
    return plies / (time.time() - t0)

def bench_selfplay(cls, n_games: int = 4, sims: int = 96, parallel: int = 4, leaves: int = 4) -> float:
    sp = cls(uniform_eval,
    n_parallel=parallel, sims=sims, temp_moves=10,
    opening_random_plies=2, curriculum_frac=0.0,
    leaves_per_wave=leaves, seed=0)
    t0 = time.time()
    positions, stats = sp.play_games(n_games)
    dt = time.time() - t0
    total_sims = len(positions) * sims
    return total_sims / dt

def main() -> None:
    print("=== Dvijok (tasodifiy o'yinlar, ply/sekund) ===")
    py = bench_engine_python()
    print(f"  Python: {py:,.0f} ply/s")
    try:
        rs = bench_engine_rust()
        print(f"  Rust:   {rs:,.0f} ply/s   ({rs / py:.0f}x tezroq)")
    except ImportError:
        print("  Rust:   o'rnatilmagan")
        return
    print("\n=== Self-play MCTS (simulyatsiya/sekund, uniform eval) ===")
    from selfplay import ParallelSelfPlay
    sp_py = bench_selfplay(ParallelSelfPlay)
    print(f"  Python: {sp_py:,.0f} sim/s")
    from selfplay_rust import RustParallelSelfPlay
    sp_rs = bench_selfplay(RustParallelSelfPlay)
    print(f"  Rust:   {sp_rs:,.0f} sim/s   ({sp_rs / sp_py:.0f}x tezroq)")
    print("\n=== Arena (20 o'yin, sims=40, sekund) ===")
    import time as _t
    from network import ShashkaNet
    from inference import TorchEvaluator
    from arena_rust import play_arena_rust
    from train import play_arena
    net = ShashkaNet(d_model=64, n_layers=2, n_heads=4, policy_dim=32)
    ev = TorchEvaluator(net, "cpu")
    t0 = _t.time()
    play_arena_rust(ev, ev, n_games=20, sims=40, seed=1)
    t_rs = _t.time() - t0
    t0 = _t.time()
    play_arena(ev, ev, n_games=20, sims=40, seed=1)
    t_py = _t.time() - t0
    print(f"  Python: {t_py:.1f}s")
    print(f"  Rust:   {t_rs:.1f}s   ({t_py / t_rs:.0f}x tezroq)")
    print("\nEslatma: CPU'da arena farqi ~5x; GPU'da batch afzalligi tufayli")
    print("~20-40x bo'ladi (Python arena GPU ga bittadan pozitsiya yuboradi,")
    print("Rust arena esa parallel o'yinlarni bitta batch qiladi).")

if __name__ == "__main__":
    main()
