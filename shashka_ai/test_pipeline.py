""" MCTS + self-play pipeline testi (torch talab qilmaydi).
Mock eval_fn tasodifiy logits qaytaradi — bu MCTS daraxti, zanjirli
urishlardagi backprop ishorasi, curriculum va trening yozuvlarining
to'g'riligini tekshiradi. """

from __future__ import annotations
import numpy as np
from checkers_engine import ACTION_SIZE, INPUT_SIZE, GameState
from mcts import MCTS, Node, select_action, visit_distribution
from selfplay import ParallelSelfPlay, ReplayBuffer, random_endgame_state

def mock_eval(x: np.ndarray):
    rng = np.random.default_rng(int(abs(x.sum() * 1000)) % (2**31))
    logits = rng.normal(0, 0.1, size=(x.shape[0], ACTION_SIZE)).astype(np.float32)
    values = np.zeros(x.shape[0], dtype=np.float32)
    return logits, values

def test_mcts_single() -> None:
    st = GameState()
    m = MCTS(mock_eval, seed=1)
    root = m.run(st, sims=64, add_noise=True)
    assert root.n >= 64
    acts, pi = visit_distribution(root)
    assert abs(pi.sum() - 1.0) < 1e-5
    assert set(acts) == set(st.legal_actions())
    a = select_action(root, temperature=0.0)
    assert a in acts

def test_mcts_tree_reuse() -> None:
    st = GameState()
    m = MCTS(mock_eval, seed=2)
    root = m.run(st, sims=32)
    a = select_action(root, 0.0)
    st.apply(a)
    child = root.children[a]
    n_before = child.n
    root2 = m.run(st, root=child, sims=32)
    assert root2.n >= n_before + 32

def test_selfplay_records() -> None:
    sp = ParallelSelfPlay(mock_eval, n_parallel=3, sims=12, temp_moves=10, opening_random_plies=2, curriculum_frac=0.5, seed=3)
    positions, stats = sp.play_games(3)
    assert stats.games == 3
    assert len(positions) > 0
    for x, idx, pi, z in positions[:50]:
        assert x.shape == (INPUT_SIZE,)
        assert abs(pi.sum() - 1.0) < 1e-4
        assert z in (0, 1, 2)
        assert idx.min() >= 0 and idx.max() < ACTION_SIZE
    print(f"  selfplay: {stats.games} o'yin, {len(positions)} pozitsiya, "
    f"durang={stats.draws}, o'rtacha={stats.avg_length:.0f} ply")

def test_replay_buffer() -> None:
    sp = ParallelSelfPlay(mock_eval, n_parallel=2, sims=8, seed=4)
    positions, _ = sp.play_games(2)
    buf = ReplayBuffer(capacity=100)
    buf.extend(positions)
    rng = np.random.default_rng(0)
    xs, Pi, zs = buf.sample(16, rng)
    assert xs.shape == (16, INPUT_SIZE)
    assert Pi.shape == (16, ACTION_SIZE)
    assert np.allclose(Pi.sum(axis=1), 1.0, atol=1e-4)
    assert zs.shape == (16,) and set(np.unique(zs)) <= {0, 1, 2}

def test_curriculum_states_legal() -> None:
    rng = np.random.default_rng(5)
    for _ in range(50):
        st = random_endgame_state(rng)
        winner, moves = st.status()
        assert winner is None and len(moves) > 0

if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"OK   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} test o'tdi")
    sys.exit(1 if failed else 0)