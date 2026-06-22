""" play.py va game.py uchun umumiy pygame chizish kodi. """

from __future__ import annotations
from typing import Iterable, Optional, Sequence, Tuple
from checkers_engine import (RC_TO_SQ, SQ_TO_RC, WHITE, GameState, action_decode, sq_name)
CELL = 80
BOARD_PX = CELL * 8
SIDE_PX = 260
WIN_W, WIN_H = BOARD_PX + SIDE_PX, BOARD_PX
C_LIGHT = (238, 217, 181)
C_DARK = (181, 136, 99)
C_SEL = (246, 246, 105)
C_TARGET = (106, 190, 89)
C_LAST = (96, 130, 182)
C_BG = (32, 32, 36)
C_TEXT = (230, 230, 230)
C_WHITE_P = (245, 245, 245)
C_BLACK_P = (40, 40, 40)
C_CROWN = (212, 175, 55)

def rc_to_xy(r: int, c: int, flip: bool) -> Tuple[int, int]:
    if flip:
        r, c = 7 - r, 7 - c
    return c * CELL, r * CELL

def xy_to_sq(x: int, y: int, flip: bool) -> Optional[int]:
    if not (0 <= x < BOARD_PX and 0 <= y < BOARD_PX):
        return None
    c, r = x // CELL, y // CELL
    if flip:
        r, c = 7 - r, 7 - c
    sq = int(RC_TO_SQ[r, c])
    return sq if sq >= 0 else None

def draw_state(screen, font, small, state: GameState,
    selected: Optional[int] = None,
    targets: Iterable[int] = (),
    last_move: Optional[Tuple[int, int]] = None,
    flip: bool = False,
    eval_white: Optional[float] = None,
    info_lines: Sequence[str] = ()) -> None:
    import pygame
    screen.fill(C_BG)
    targets = set(targets)
    for r in range(8):
        for c in range(8):
            x, y = rc_to_xy(r, c, flip)
            col = C_DARK if (r + c) % 2 == 1 else C_LIGHT
            pygame.draw.rect(screen, col, (x, y, CELL, CELL))
    if last_move is not None:
        for sq in last_move:
            r, c = SQ_TO_RC[sq]
            x, y = rc_to_xy(r, c, flip)
            pygame.draw.rect(screen, C_LAST, (x, y, CELL, CELL), 5)
    if selected is not None:
        r, c = SQ_TO_RC[selected]
        x, y = rc_to_xy(r, c, flip)
        pygame.draw.rect(screen, C_SEL, (x, y, CELL, CELL), 5)
    for sq in targets:
        r, c = SQ_TO_RC[sq]
        x, y = rc_to_xy(r, c, flip)
        pygame.draw.circle(screen, C_TARGET, (x + CELL // 2, y + CELL // 2), CELL // 7)
    for sq, (r, c) in enumerate(SQ_TO_RC):
        p = int(state.board[r, c])
        if p == 0:
            continue
        x, y = rc_to_xy(r, c, flip)
        cx, cy = x + CELL // 2, y + CELL // 2
        col = C_WHITE_P if p > 0 else C_BLACK_P
        edge = (90, 90, 90) if p > 0 else (200, 200, 200)
        pygame.draw.circle(screen, col, (cx, cy), CELL // 2 - 8)
        pygame.draw.circle(screen, edge, (cx, cy), CELL // 2 - 8, 3)
        if abs(p) == 2:
            pygame.draw.circle(screen, C_CROWN, (cx, cy), CELL // 5, 4)
        if sq in state.captured:
            pygame.draw.line(screen, (200, 60, 60), (cx - 14, cy - 14), (cx + 14, cy + 14), 4)
            pygame.draw.line(screen, (200, 60, 60), (cx - 14, cy + 14), (cx + 14, cy - 14), 4)
    px = BOARD_PX + 16
    y = 16
    if eval_white is not None:
        bar_h = 160
        pygame.draw.rect(screen, (15, 15, 15), (px, y, 26, bar_h))
        frac = (eval_white + 1.0) / 2.0
        wh = int(bar_h * frac)
        pygame.draw.rect(screen, (235, 235, 235), (px, y + bar_h - wh, 26, wh))
        screen.blit(small.render(f"{eval_white:+.2f}", True, C_TEXT), (px + 34, y + bar_h // 2 - 8))
        y += bar_h + 14
    for line in info_lines:
        screen.blit(small.render(line, True, C_TEXT), (px, y))
        y += 24

def move_text(action: int) -> str:
    f, t = action_decode(action)
    return f"{sq_name(f)}-{sq_name(t)}"