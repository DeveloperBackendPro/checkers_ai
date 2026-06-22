""" game.py — Train qilingan MODEL BILAN O'YNASH (inson vs AI, pygame GUI).
Foydalanish:
    python game.py --ckpt checkpoints/best.pt --color white --sims 400
    python game.py --onnx checkpoints/best.onnx --color black --time-limit 3
Boshqaruv:
    Sichqoncha — shashkani tanlash va yurish (legal kataklar yashil nuqta)
    Urish majburiy: faqat urish yurishlari ko'rsatiladi.
    Zanjirli urishda o'sha shashka bilan davom etish shart (avtomatik cheklanadi).
    R — yangi o'yin, Q/ESC — chiqish
Kuch sozlamalari:
    --sims        har yurishdagi MCTS simulyatsiyalari (qidiruv chuqurligi)
    --time-limit  yurish uchun maksimal soniya (ixtiyoriy) """

from __future__ import annotations
import sys
import time
import argparse
from inference import build_eval_fn
from typing import Dict, List, Optional, Set, Tuple
from mcts import MCTS, Node, root_value_white, select_action
from checkers_engine import (GameState, WHITE, BLACK, action_decode, action_encode)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inson vs train qilingan model")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--onnx", default=None)
    p.add_argument("--random", action="store_true", help="MODEL'SIZ raqib (sof MCTS) — qoidalarni o'ynab tekshirish uchun, torch kerak emas")
    p.add_argument("--color", choices=["white", "black"], default="white", help="sizning rangingiz")
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--time-limit", type=float, default=None, help="AI yurishi uchun maksimal soniya")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()

def result_text(winner: int, human: int) -> str:
    if winner == 0:
        return "DURANG"
    return "SIZ YUTDINGIZ!" if winner == human else "Model yutdi"

def main() -> None:
    args = parse_args()
    if args.random:
        from play import uniform_eval_fn
        eval_fn = uniform_eval_fn
        print("MODEL'SIZ raqib: sof MCTS + qoidalar")
    elif args.ckpt or args.onnx:
        eval_fn = build_eval_fn(ckpt=args.ckpt, onnx=args.onnx, device="cpu")
    else:
        print("Xato: --ckpt, --onnx yoki --random bering")
        sys.exit(1)
    import pygame
    from gui_common import WIN_W, WIN_H, draw_state, move_text, xy_to_sq
    mcts = MCTS(eval_fn, seed=args.seed)
    human = WHITE if args.color == "white" else BLACK
    flip = human == BLACK
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Rus shashkasi — siz vs model")
    font = pygame.font.SysFont("dejavusans", 22)
    small = pygame.font.SysFont("dejavusans", 18)
    clock = pygame.time.Clock()
    st = GameState()
    ai_root: Optional[Node] = None
    selected: Optional[int] = None
    last_move: Optional[Tuple[int, int]] = None
    last_eval = 0.0
    winner: Optional[int] = None
    move_log: List[str] = []
    quit_flag = False

    def legal_map() -> Dict[int, Set[int]]:
        m: Dict[int, Set[int]] = {}
        for f, t in st.status()[1]:
            m.setdefault(f, set()).add(t)
        return m

    def advance_root(action: int) -> None:
        nonlocal ai_root
        if ai_root is not None:
            ai_root = ai_root.children.get(action)

    def pump() -> None:
        nonlocal quit_flag
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                quit_flag = True

    def redraw(thinking: bool = False) -> None:
        lm = legal_map() if (winner is None and st.player == human) else {}
        targets = lm.get(selected, set()) if selected is not None else set()
        info = [
            f"Siz: {'OQ' if human == WHITE else 'QORA'}",
            f"Ply: {st.ply}",
            f"Sims: {args.sims}" + (f" / {args.time_limit}s" if args.time_limit else ""),
            "Model o'ylayapti..." if thinking else
            ("Sizning navbatingiz" if (winner is None and st.player == human) else ""),
        ]
        if st.chain_sq is not None and st.player == human:
            info.append("Zanjir: urishni davom ettiring!")
        if winner is not None:
            info.append(f"NATIJA: {result_text(winner, human)}")
            info.append("R — yangi o'yin")
        info.append("")
        info.extend(move_log[-8:])
        draw_state(screen, font, small, st, selected=selected, targets=targets, last_move=last_move, flip=flip, eval_white=last_eval, info_lines=info)
        pygame.display.flip()

    def do_move(action: int, by_ai: bool) -> None:
        nonlocal last_move, winner
        last_move = action_decode(action)
        who = "AI" if by_ai else "Siz"
        move_log.append(f"{st.ply + 1}. {who}: {move_text(action)}")
        st.apply(action)
        advance_root(action)
        winner = st.status()[0]
    redraw()
    while not quit_flag:
        if winner is None and st.player != human:
            redraw(thinking=True)
            ai_root = mcts.run(st, root=ai_root, sims=args.sims, time_limit=args.time_limit, pump=pump)
            if quit_flag:
                break
            a = select_action(ai_root, temperature=0.0)
            last_eval = root_value_white(ai_root, st)
            do_move(a, by_ai=True)
            selected = st.chain_sq if st.player == human else None
            redraw()
            continue
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                quit_flag = True
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    quit_flag = True
                elif ev.key == pygame.K_r:
                    st = GameState()
                    ai_root = None
                    selected = None
                    last_move = None
                    winner = None
                    move_log.clear()
            elif (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                  and winner is None and st.player == human):
                sq = xy_to_sq(*ev.pos, flip)
                if sq is None:
                    continue
                lm = legal_map()
                if st.chain_sq is not None:
                    selected = st.chain_sq
                if selected is not None and sq in lm.get(selected, set()):
                    do_move(action_encode(selected, sq), by_ai=False)
                    selected = st.chain_sq if (winner is None and st.player == human) else None
                elif sq in lm:
                    selected = sq
                else:
                    if st.chain_sq is None:
                        selected = None
        redraw()
        clock.tick(30)
    pygame.quit()

if __name__ == "__main__":
    main()