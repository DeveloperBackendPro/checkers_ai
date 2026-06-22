""" Rus shashkasi (Russian Checkers / Shashki) o'yin dvijogi.
Qoidalar (rasmiy):
  - 8x8 doska, faqat qora kataklarda o'yin (32 katak), har tomonda 12 ta shashka.
  - Oq birinchi yuradi. Oddiy shashka faqat oldinga yuradi, lekin TO'RT
    diagonal yo'nalishda (oldinga ham, orqaga ham) URADI.
  - Urish MAJBURIY. Boshlangan urish zanjiri oxirigacha davom ettirilishi shart.
  - Damka (flying king): istalgan masofaga yuradi va uradi; urgandan keyin
    dushman ortidagi istalgan bo'sh katakka tushadi, AMMO agar biror tushish
    katagidan urishni davom ettirish mumkin bo'lsa, faqat shunday kataklarga
    tushish majburiy.
  - "Turk zarbasi" qoidasi: zanjir davomida urilgan shashkalar doskada qoladi
    (to'siq bo'lib) va ularni ikkinchi marta urib bo'lmaydi; ular faqat yurish
    tugagach olinadi.
  - Oddiy shashka zanjir davomida oxirgi qatorga yetib borsa, DARHOL damkaga
    aylanadi va zanjirni damka sifatida davom ettiradi.
  - Durang: pozitsiya 3 marta takrorlansa, yoki 30 yurish (60 yarim-yurish)
    davomida urish ham, oddiy shashka yurishi ham bo'lmasa, yoki o'yin
    MAX_PLIES dan oshsa.
Action kodlash: bitta action = bitta segment (oddiy yurish yoki bitta sakrash).
  action = from_sq * 32 + to_sq   (0..1023)
Zanjirli urishda navbat o'sha o'yinchida qoladi (chain_sq orqali). """

from __future__ import annotations
import numpy as np
from typing import Dict, FrozenSet, List, Optional, Tuple

WHITE: int = 1
BLACK: int = -1
EMPTY: int = 0
WM, WK, BM, BK = 1, 2, -1, -2
NUM_SQUARES: int = 32
ACTION_SIZE: int = NUM_SQUARES * NUM_SQUARES
INPUT_SIZE: int = NUM_SQUARES * 6 + 2
NO_PROGRESS_LIMIT: int = 60
MAX_PLIES: int = 300
SQ_TO_RC: List[Tuple[int, int]] = []
RC_TO_SQ: np.ndarray = -np.ones((8, 8), dtype=np.int32)
for _r in range(8):
    for _c in range(8):
        if (_r + _c) % 2 == 1:
            RC_TO_SQ[_r, _c] = len(SQ_TO_RC)
            SQ_TO_RC.append((_r, _c))
DIRS: Tuple[Tuple[int, int], ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_PIECE_IDX: Dict[int, int] = {WM: 0, WK: 1, BM: 2, BK: 3}
_zrng = np.random.default_rng(20260610)
_ZOBRIST = _zrng.integers(1, 2 ** 62, size=(4, 32), dtype=np.int64)
_ZOBRIST_SIDE = int(_zrng.integers(1, 2 ** 62, dtype=np.int64))

def action_encode(f: int, t: int) -> int:
    return f * 32 + t

def action_decode(a: int) -> Tuple[int, int]:
    return divmod(a, 32)

def sq_name(sq: int) -> str:
    r, c = SQ_TO_RC[sq]
    return "abcdefgh"[c] + str(8 - r)

def action_name(a: int) -> str:
    f, t = action_decode(a)
    return f"{sq_name(f)}-{sq_name(t)}"

class GameState:
    __slots__ = ("board", "player", "chain_sq", "captured", "halfmove", "ply", "history")
    def __init__(self, empty: bool = False) -> None:
        self.board: np.ndarray = np.zeros((8, 8), dtype=np.int8)
        self.player: int = WHITE
        self.chain_sq: Optional[int] = None
        self.captured: FrozenSet[int] = frozenset()
        self.halfmove: int = 0
        self.ply: int = 0
        self.history: Dict[int, int] = {}
        if not empty:
            for sq, (r, c) in enumerate(SQ_TO_RC):
                if r < 3:
                    self.board[r, c] = BM
                elif r > 4:
                    self.board[r, c] = WM
            self._record()

    def clone(self) -> "GameState":
        st = GameState.__new__(GameState)
        st.board = self.board.copy()
        st.player = self.player
        st.chain_sq = self.chain_sq
        st.captured = self.captured
        st.halfmove = self.halfmove
        st.ply = self.ply
        st.history = dict(self.history)
        return st

    def pos_hash(self) -> int:
        h = 0
        for sq, (r, c) in enumerate(SQ_TO_RC):
            p = int(self.board[r, c])
            if p != 0:
                h ^= int(_ZOBRIST[_PIECE_IDX[p], sq])
        if self.player == BLACK:
            h ^= _ZOBRIST_SIDE
        return h

    def _record(self) -> None:
        h = self.pos_hash()
        self.history[h] = self.history.get(h, 0) + 1

    def start_from_setup(self) -> None:
        self.history = {}
        self._record()

    def _raw_captures(self, sq: int, piece: int, captured: FrozenSet[int]) -> List[Tuple[int, int]]:
        r, c = SQ_TO_RC[sq]
        b = self.board
        sign = 1 if piece > 0 else -1
        out: List[Tuple[int, int]] = []
        if abs(piece) == 1:
            for dr, dc in DIRS:
                tr, tc = r + 2 * dr, c + 2 * dc
                if 0 <= tr < 8 and 0 <= tc < 8 and b[tr, tc] == EMPTY:
                    mr, mc = r + dr, c + dc
                    mid = int(b[mr, mc])
                    if mid != 0 and mid * sign < 0:
                        msq = int(RC_TO_SQ[mr, mc])
                        if msq not in captured:
                            out.append((int(RC_TO_SQ[tr, tc]), msq))
        else:
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                while 0 <= rr < 8 and 0 <= cc < 8 and b[rr, cc] == EMPTY:
                    rr += dr
                    cc += dc
                if not (0 <= rr < 8 and 0 <= cc < 8):
                    continue
                mid = int(b[rr, cc])
                msq = int(RC_TO_SQ[rr, cc])
                if mid * sign >= 0 or msq in captured:
                    continue
                lr, lc = rr + dr, cc + dc
                while 0 <= lr < 8 and 0 <= lc < 8 and b[lr, lc] == EMPTY:
                    out.append((int(RC_TO_SQ[lr, lc]), msq))
                    lr += dr
                    lc += dc
        return out

    def _captures_from(self, sq: int, piece: Optional[int] = None, captured: Optional[FrozenSet[int]] = None) -> List[Tuple[int, int]]:
        r, c = SQ_TO_RC[sq]
        if piece is None:
            piece = int(self.board[r, c])
        if captured is None:
            captured = self.captured
        segs = self._raw_captures(sq, piece, captured)
        if not segs or abs(piece) == 1:
            return segs
        by_cap: Dict[int, List[int]] = {}
        for to_sq, cap_sq in segs:
            by_cap.setdefault(cap_sq, []).append(to_sq)
        b = self.board
        saved = int(b[r, c])
        b[r, c] = EMPTY
        out: List[Tuple[int, int]] = []
        for cap_sq, lands in by_cap.items():
            new_cap = captured | {cap_sq}
            cont: List[int] = []
            for to_sq in lands:
                tr, tc = SQ_TO_RC[to_sq]
                b[tr, tc] = piece
                if self._raw_captures(to_sq, piece, new_cap):
                    cont.append(to_sq)
                b[tr, tc] = EMPTY
            out.extend((t, cap_sq) for t in (cont if cont else lands))
        b[r, c] = saved
        return out

    def legal_moves(self) -> List[Tuple[int, int]]:
        b = self.board
        if self.chain_sq is not None:
            return [(self.chain_sq, t) for t, _ in self._captures_from(self.chain_sq)]
        mine = [sq for sq, (r, c) in enumerate(SQ_TO_RC) if b[r, c] != 0 and (b[r, c] > 0) == (self.player > 0)]
        caps: List[Tuple[int, int]] = []
        for sq in mine:
            caps.extend((sq, t) for t, _ in self._captures_from(sq))
        if caps:
            return caps
        quiets: List[Tuple[int, int]] = []
        fwd = -1 if self.player == WHITE else 1
        for sq in mine:
            r, c = SQ_TO_RC[sq]
            p = int(b[r, c])
            if abs(p) == 1:
                for dc in (-1, 1):
                    tr, tc = r + fwd, c + dc
                    if 0 <= tr < 8 and 0 <= tc < 8 and b[tr, tc] == EMPTY:
                        quiets.append((sq, int(RC_TO_SQ[tr, tc])))
            else:
                for dr, dc in DIRS:
                    tr, tc = r + dr, c + dc
                    while 0 <= tr < 8 and 0 <= tc < 8 and b[tr, tc] == EMPTY:
                        quiets.append((sq, int(RC_TO_SQ[tr, tc])))
                        tr += dr
                        tc += dc
        return quiets

    def legal_actions(self) -> List[int]:
        return [action_encode(f, t) for f, t in self.legal_moves()]

    def apply(self, action: int) -> None:
        f, t = action_decode(action)
        fr, fc = SQ_TO_RC[f]
        tr, tc = SQ_TO_RC[t]
        piece = int(self.board[fr, fc])
        assert piece != 0 and (piece > 0) == (self.player > 0), f"Noto'g'ri yurish: {action_name(action)}"
        was_man = abs(piece) == 1
        dr = (tr > fr) - (tr < fr)
        dc = (tc > fc) - (tc < fc)
        cap_sq: Optional[int] = None
        rr, cc = fr + dr, fc + dc
        while (rr, cc) != (tr, tc):
            if self.board[rr, cc] != EMPTY:
                cap_sq = int(RC_TO_SQ[rr, cc])
            rr += dr
            cc += dc
        self.board[fr, fc] = EMPTY
        if was_man and ((piece > 0 and tr == 0) or (piece < 0 and tr == 7)):
            piece = WK if piece > 0 else BK
        self.board[tr, tc] = piece
        self.ply += 1
        if cap_sq is not None:
            new_cap = self.captured | {cap_sq}
            if self._captures_from(t, piece=piece, captured=new_cap):
                self.chain_sq = t
                self.captured = new_cap
                return
            for csq in new_cap:
                cr, cc2 = SQ_TO_RC[csq]
                self.board[cr, cc2] = EMPTY
            self.captured = frozenset()
            self.chain_sq = None
            self.halfmove = 0
        else:
            self.halfmove = 0 if was_man else self.halfmove + 1
        self.player = -self.player
        self._record()

    def status(self) -> Tuple[Optional[int], List[Tuple[int, int]]]:
        moves = self.legal_moves()
        if not moves:
            return -self.player, moves
        if self.chain_sq is None:
            if self.history.get(self.pos_hash(), 0) >= 3:
                return 0, moves
            if self.halfmove >= NO_PROGRESS_LIMIT or self.ply >= MAX_PLIES:
                return 0, moves
        return None, moves

    def winner(self) -> Optional[int]:
        return self.status()[0]

    def encode(self) -> np.ndarray:
        rot = self.player == BLACK
        feat = np.zeros((NUM_SQUARES, 6), dtype=np.float32)
        for sq, (r, c) in enumerate(SQ_TO_RC):
            p = int(self.board[r, c])
            if p == 0:
                continue
            csq = 31 - sq if rot else sq
            feat[csq, _PIECE_IDX[p * self.player]] = 1.0
            if sq in self.captured:
                feat[csq, 5] = 1.0
        if self.chain_sq is not None:
            cs = 31 - self.chain_sq if rot else self.chain_sq
            feat[cs, 4] = 1.0
        glob = np.array([
            min(self.halfmove, NO_PROGRESS_LIMIT) / NO_PROGRESS_LIMIT,
            1.0 if self.chain_sq is not None else 0.0],
            dtype=np.float32)
        return np.concatenate([feat.reshape(-1), glob])

    def canon_action(self, a: int) -> int:
        if self.player == WHITE:
            return a
        f, t = action_decode(a)
        return action_encode(31 - f, 31 - t)

    def __str__(self) -> str:
        sym = {WM: "w", WK: "W", BM: "b", BK: "B", EMPTY: "."}
        lines = []
        for r in range(8):
            row = " ".join(sym[int(self.board[r, c])] if (r + c) % 2 == 1 else " " for c in range(8))
            lines.append(f"{8 - r} {row}")
        lines.append("  a b c d e f g h")
        side = "OQ" if self.player == WHITE else "QORA"
        lines.append(f"Navbat: {side}  ply={self.ply}  halfmove={self.halfmove}"
        + (f"  zanjir={sq_name(self.chain_sq)}" if self.chain_sq is not None else ""))
        return "\n".join(lines)