"""Differential fuzzing: Rust dvijok vs Python etalon.
Ikkala dvijokda BIR XIL tasodifiy o'yinlar o'ynatiladi va HAR YURISHDA
solishtiriladi: legal yurishlar to'plami, o'yinchi, zanjir katagi,
halfmove, doska, g'olib va NN-kodlash (bit-darajada).
Bittagina farq = bug. Bu Rust dvijokning to'g'riligini Python'ning
10,000-o'yinlik fuzzing'dan o'tgan etaloniga bog'laydi.
Ishlatish:
    python test_diff.py                    # 1000 o'yin
    SHASHKA_DIFF=5000 python test_diff.py  # chuqurroq """

from __future__ import annotations
import os
import sys
import numpy as np
from checkers_engine import GameState as PyState, action_encode
try:
    import shashka_engine as se
except ImportError:
    print("XATO: shashka_engine o'rnatilmagan. README'dagi Rust qurish bo'limiga qarang.")
    sys.exit(1)

def compare(py: PyState, rs, ply_info: str) -> None:
    pw, pm = py.status()
    rw, rm = rs.status()
    py_acts = sorted(action_encode(f, t) for f, t in pm)
    rs_acts = sorted(rm)
    assert py_acts == rs_acts, f"{ply_info}: legal farq\n  py={py_acts}\n  rs={rs_acts}"
    assert pw == rw, f"{ply_info}: winner farq py={pw} rs={rw}"
    assert py.player == rs.player, f"{ply_info}: player farq"
    assert (py.chain_sq if py.chain_sq is not None else None) == rs.chain_sq, \
    f"{ply_info}: chain farq py={py.chain_sq} rs={rs.chain_sq}"
    assert py.halfmove == rs.halfmove, f"{ply_info}: halfmove farq"
    py_board = [0] * 32
    from checkers_engine import SQ_TO_RC
    for sq, (r, c) in enumerate(SQ_TO_RC):
        py_board[sq] = int(py.board[r, c])
    assert py_board == rs.board_list(), f"{ply_info}: doska farq"
    ex, rx = py.encode(), np.asarray(rs.encode())
    assert ex.shape == rx.shape and np.array_equal(ex, rx), \
    f"{ply_info}: encode farq, max={np.abs(ex - rx).max()}"
    for a in (py_acts[:3] if py_acts else []):
        assert py.canon_action(a) == rs.canon_action(a), f"{ply_info}: canon farq"

def main() -> None:
    n_games = int(os.environ.get("SHASHKA_DIFF", "1000"))
    rng = np.random.default_rng(2026)
    total_plies = 0
    for game in range(n_games):
        py = PyState()
        rs = se.GameState()
        ply = 0
        while True:
            compare(py, rs, f"o'yin {game} ply {ply}")
            winner, moves = py.status()
            if winner is not None:
                break
            f, t = moves[int(rng.integers(0, len(moves)))]
            a = action_encode(f, t)
            py.apply(a)
            rs.apply(a)
            ply += 1
            total_plies += 1
        if (game + 1) % 200 == 0:
            print(f"  {game + 1}/{n_games} o'yin tekshirildi "
            f"({total_plies:,} ply)...")
    print(f"\nDIFFERENTIAL FUZZING O'TDI: {n_games} o'yin, {total_plies:,} ply — "
          f"Rust va Python dvijoklari har yurishda BIT-DARAJADA mos.")

if __name__ == "__main__":
    main()