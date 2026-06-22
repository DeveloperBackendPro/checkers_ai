""" O'yin dvijogi unit-testlari: python test_engine.py  (yoki pytest). """

from __future__ import annotations
import numpy as np
from checkers_engine import (BK, BLACK, BM, EMPTY, RC_TO_SQ, SQ_TO_RC, WHITE, WK, WM, GameState, action_encode)

def sq(r: int, c: int) -> int:
    s = int(RC_TO_SQ[r, c])
    assert s >= 0, f"({r},{c}) qora katak emas"
    return s

def empty_state() -> GameState:
    return GameState(empty=True)

def test_initial_position() -> None:
    st = GameState()
    whites = (st.board > 0).sum()
    blacks = (st.board < 0).sum()
    assert whites == 12 and blacks == 12
    moves = st.legal_moves()
    assert len(moves) == 7, f"boshlang'ich yurishlar 7 bo'lishi kerak, {len(moves)}"
    assert st.player == WHITE

def test_mandatory_capture() -> None:
    st = empty_state()
    st.board[4, 3] = WM
    st.board[3, 2] = BM
    st.board[6, 1] = WM
    st.start_from_setup()
    moves = st.legal_moves()
    assert moves == [(sq(4, 3), sq(2, 1))], f"faqat urish legal: {moves}"

def test_man_captures_backward() -> None:
    st = empty_state()
    st.board[3, 2] = BM
    st.board[2, 1] = WM
    st.player = BLACK
    st.start_from_setup()
    moves = st.legal_moves()
    assert (sq(3, 2), sq(1, 0)) in moves, f"orqaga urish yo'q: {moves}"

def test_chain_capture() -> None:
    st = empty_state()
    st.board[5, 2] = WM
    st.board[4, 1] = BM
    st.board[2, 1] = BM
    st.start_from_setup()
    moves = st.legal_moves()
    assert moves == [(sq(5, 2), sq(3, 0))]
    st.apply(action_encode(sq(5, 2), sq(3, 0)))
    assert st.player == WHITE
    assert st.chain_sq == sq(3, 0)
    assert int(st.board[4, 1]) == BM, "urilgan shashka zanjir tugaguncha doskada"
    moves = st.legal_moves()
    assert moves == [(sq(3, 0), sq(1, 2))]
    st.apply(action_encode(sq(3, 0), sq(1, 2)))
    assert st.player == BLACK
    assert st.chain_sq is None
    assert int(st.board[4, 1]) == EMPTY and int(st.board[2, 1]) == EMPTY
    assert int(st.board[1, 2]) == WM

def test_flying_king_moves() -> None:
    st = empty_state()
    st.board[7, 0] = WK
    st.board[0, 7] = BM
    st.start_from_setup()
    moves = st.legal_moves()
    assert len(moves) == 6, f"damka yurishlari: {moves}"

def test_flying_king_capture_with_landing_restriction() -> None:
    st = empty_state()
    st.board[7, 0] = WK
    st.board[4, 3] = BM
    st.board[1, 4] = BM
    st.start_from_setup()
    moves = st.legal_moves()
    assert moves == [(sq(7, 0), sq(2, 5))], f"tushish cheklovi buzildi: {moves}"
    st.apply(action_encode(sq(7, 0), sq(2, 5)))
    assert st.chain_sq == sq(2, 5) and st.player == WHITE
    moves = st.legal_moves()
    assert moves == [(sq(2, 5), sq(0, 3))]
    st.apply(action_encode(sq(2, 5), sq(0, 3)))
    assert st.player == BLACK
    assert (st.board < 0).sum() == 0, "ikkala qora ham olingan"

def test_turkish_strike_blocking() -> None:
    st = empty_state()
    st.board[4, 3] = WK
    st.board[3, 2] = BM
    st.start_from_setup()
    segs = st._captures_from(sq(4, 3), piece=WK, captured=frozenset({sq(3, 2)}))
    assert segs == [], "urilgan shashka qayta urilmasligi kerak"

def test_promotion_mid_chain_continues_as_king() -> None:
    st = empty_state()
    st.board[2, 1] = WM
    st.board[1, 2] = BM
    st.board[2, 5] = BM
    st.start_from_setup()
    moves = st.legal_moves()
    assert moves == [(sq(2, 1), sq(0, 3))]
    st.apply(action_encode(sq(2, 1), sq(0, 3)))
    assert int(st.board[0, 3]) == WK
    assert st.chain_sq == sq(0, 3) and st.player == WHITE
    moves = st.legal_moves()
    assert set(moves) == {(sq(0, 3), sq(3, 6)), (sq(0, 3), sq(4, 7))}, moves
    st.apply(action_encode(sq(0, 3), sq(3, 6)))
    assert st.player == BLACK
    assert (st.board < 0).sum() == 0
    assert int(st.board[3, 6]) == WK

def test_quiet_promotion() -> None:
    st = empty_state()
    st.board[1, 2] = WM
    st.board[7, 6] = BK
    st.start_from_setup()
    st.apply(action_encode(sq(1, 2), sq(0, 1)))
    assert int(st.board[0, 1]) == WK

def test_no_moves_is_loss() -> None:
    st = empty_state()
    st.board[1, 0] = WM
    st.board[0, 1] = BK
    st.player = WHITE
    st.start_from_setup()
    assert st.winner() == BLACK

def test_threefold_repetition_draw() -> None:
    st = empty_state()
    st.board[7, 0] = WK
    st.board[0, 7] = BK
    st.start_from_setup()
    seq = [(sq(7, 0), sq(6, 1)), (sq(0, 7), sq(1, 6)),
           (sq(6, 1), sq(7, 0)), (sq(1, 6), sq(0, 7))]
    for _ in range(2):
        for f, t in seq:
            assert st.winner() is None
            st.apply(action_encode(f, t))
    assert st.winner() == 0, "3 marta takror — durang"

def test_no_progress_draw_counter() -> None:
    st = empty_state()
    st.board[7, 0] = WK
    st.board[4, 7] = BK
    st.start_from_setup()
    st.apply(action_encode(sq(7, 0), sq(6, 1)))
    assert st.halfmove == 1, "damka tinch yurishi hisoblagichni oshiradi"
    st.apply(action_encode(sq(4, 7), sq(5, 6)))
    assert st.halfmove == 2

def test_encode_and_canon() -> None:
    st = GameState()
    x = st.encode()
    assert x.shape == (194,) and x.dtype == np.float32
    assert x[:192].reshape(32, 6)[:, 0].sum() == 12
    st.player = BLACK
    for a in (0, 5 * 32 + 7, 1023):
        assert st.canon_action(st.canon_action(a)) == a
    st2 = GameState()
    st2.apply(st2.legal_actions()[0])
    x2 = st2.encode()
    assert x2.reshape(-1)[:192].reshape(32, 6)[:, 0].sum() == 12

def test_clone_independence() -> None:
    st = GameState()
    cl = st.clone()
    cl.apply(cl.legal_actions()[0])
    assert st.ply == 0 and cl.ply == 1
    assert not np.array_equal(st.board, cl.board)

def is_capture_move(st: GameState, f: int, t: int) -> bool:
    fr, fc = SQ_TO_RC[f]
    tr, tc = SQ_TO_RC[t]
    dr = (tr > fr) - (tr < fr)
    dc = (tc > fc) - (tc < fc)
    rr, cc = fr + dr, fc + dc
    cnt = 0
    while (rr, cc) != (tr, tc):
        if st.board[rr, cc] != EMPTY:
            cnt += 1
        rr += dr
        cc += dc
    return cnt > 0

def test_king_cannot_jump_two_adjacent_pieces() -> None:
    st = empty_state()
    st.board[7, 0] = WK
    st.board[5, 2] = BM
    st.board[4, 3] = BM
    st.start_from_setup()
    caps = st._captures_from(sq(7, 0))
    assert caps == [], f"ikki yonma-yon shashka ustidan sakrash taqiqlanadi: {caps}"
    moves = st.legal_moves()
    targets = {t for f, t in moves if f == sq(7, 0)}
    assert sq(4, 3) not in targets and sq(3, 4) not in targets

def test_nonconsecutive_repetition_draw() -> None:
    st = empty_state()
    st.board[7, 0] = WK
    st.board[4, 7] = BK
    st.start_from_setup()
    seq = [(sq(7, 0), sq(6, 1)), (sq(4, 7), sq(5, 6)),
           (sq(6, 1), sq(7, 0)), (sq(5, 6), sq(4, 7)),
           (sq(7, 0), sq(6, 1)), (sq(4, 7), sq(3, 6)),
           (sq(6, 1), sq(7, 0)), (sq(3, 6), sq(4, 7))]
    for f, t in seq:
        assert st.winner() is None
        st.apply(action_encode(f, t))
    assert st.winner() == 0, "ketma-ket bo'lmagan 3-takror ham durang"

def test_mcts_virtual_loss_balance() -> None:
    from mcts import MCTS, Node
    def uniform(x):
        from checkers_engine import ACTION_SIZE
        return (np.zeros((x.shape[0], ACTION_SIZE), np.float32), np.zeros(x.shape[0], np.float32))
    st = GameState()
    m = MCTS(uniform, seed=0)
    root = m.run(st, sims=80, add_noise=True)
    def check(node) -> None:
        assert node.vloss == 0, f"vloss != 0: {node.vloss}"
        for ch in node.children.values():
            check(ch)
    check(root)
    assert root.n >= 80

def test_random_playout_terminates_and_stays_legal() -> None:
    import os
    n_games = int(os.environ.get("SHASHKA_FUZZ", "500"))
    rng = np.random.default_rng(0)
    results = {WHITE: 0, BLACK: 0, 0: 0}
    for _ in range(n_games):
        st = GameState()
        prev_total = 24
        while True:
            winner, moves = st.status()
            if winner is not None:
                results[winner] += 1
                break
            cap_flags = [is_capture_move(st, f, t) for f, t in moves]
            assert all(cap_flags) or not any(cap_flags), \
            "urish va tinch yurishlar aralashgan (majburiylik buzildi)"
            if st.chain_sq is not None:
                assert all(f == st.chain_sq for f, _ in moves)
                assert all(cap_flags), "zanjirda faqat urish bo'ladi"
            for f, t in moves:
                fr, fc = SQ_TO_RC[f]
                tr, tc = SQ_TO_RC[t]
                dr = (tr > fr) - (tr < fr)
                dc = (tc > fc) - (tc < fc)
                rr, cc = fr + dr, fc + dc
                between = 0
                while (rr, cc) != (tr, tc):
                    if st.board[rr, cc] != EMPTY:
                        between += 1
                    rr += dr
                    cc += dc
                assert between <= 1, "bir segmentda 2+ shashka urilmoqda!"
            f, t = moves[int(rng.integers(0, len(moves)))]
            st.apply(action_encode(f, t))
            total = int((st.board != 0).sum())
            assert total <= prev_total, "shashka soni oshib ketdi!"
            if st.chain_sq is None:
                prev_total = total
        assert st.chain_sq is None or winner is not None
    assert sum(results.values()) == n_games
    print(f"  fuzzing ({n_games} o'yin): oq={results[WHITE]} "
    f"qora={results[BLACK]} durang={results[0]} — invariantlar buzilmadi")

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