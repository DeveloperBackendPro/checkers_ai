""" play.py — Train qilingan modelning O'ZI BILAN o'ynashini kuzatish (CPU, test).
Maqsad: trening sifatini vizual baholash — model qanday harakat qilyapti,
qanday kombinatsiyalar topyapti, endshpilni qanday o'ynayapti.
Grafik rejim (pygame):
    python play.py --ckpt checkpoints/best.pt --sims 200
    python play.py --onnx checkpoints/best.onnx --sims 200
Klavishlar:
    SPACE  — pauza / davom
    RIGHT  — pauzada bitta yurish
    + / -  — tezlikni o'zgartirish
    N      — yangi o'yin
    Q/ESC  — chiqish
Matnli rejim (GUI siz server/Colab uchun):
    python play.py --ckpt checkpoints/best.pt --ascii --games 3 """

from __future__ import annotations
import sys
import time
import argparse
import numpy as np
from inference import build_eval_fn
from checkers_engine import GameState, WHITE, action_decode
from mcts import MCTS, Node, root_value_white, select_action

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model self-play kuzatuvi (CPU test)")
    p.add_argument("--ckpt", default=None, help="PyTorch checkpoint (latest.pt/best.pt)")
    p.add_argument("--onnx", default=None, help="ONNX model")
    p.add_argument("--sims", type=int, default=200, help="har yurishdagi MCTS simulyatsiyalari")
    p.add_argument("--temp-moves", type=int, default=8, help="dastlabki shu yurishlarda tau=1 (xilma-xillik)")
    p.add_argument("--noise", action="store_true", help="root Dirichlet noise (xilma-xillik)")
    p.add_argument("--delay", type=float, default=0.6, help="yurishlar orasidagi soniya")
    p.add_argument("--games", type=int, default=0, help="ascii rejimda o'yinlar soni (0=cheksiz)")
    p.add_argument("--ascii", action="store_true", help="grafiksiz, terminalda")
    p.add_argument("--random", action="store_true", help="MODEL'SIZ rejim: bir xil prior + nol baho bilan sof MCTS. Torch/checkpoint kerak emas — qoidalarni vizual tekshirish uchun")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()

def uniform_eval_fn(x):
    import numpy as _np
    from checkers_engine import ACTION_SIZE as _A
    return (_np.zeros((x.shape[0], _A), dtype=_np.float32),
    _np.zeros(x.shape[0], dtype=_np.float32))

def result_text(winner: int) -> str:
    return {1: "OQ yutdi", -1: "QORA yutdi", 0: "DURANG"}[winner]

def run_ascii(args, eval_fn) -> None:
    mcts = MCTS(eval_fn, seed=args.seed)
    game_no = 0
    while args.games == 0 or game_no < args.games:
        game_no += 1
        st = GameState()
        root = None
        print(f"\n===== O'yin {game_no} =====")
        while True:
            winner, _ = st.status()
            if winner is not None:
                print(st)
                print(f">>> Natija: {result_text(winner)} ({st.ply} ply)")
                break
            root = mcts.run(st, root=root, sims=args.sims, add_noise=args.noise)
            tau = 1.0 if st.ply < args.temp_moves else 0.0
            a = select_action(root, tau)
            v = root_value_white(root, st)
            print(st)
            from gui_common import move_text
            print(f"Yurish: {move_text(a)}   baho(oq)={v:+.2f}   "
            f"tashriflar={root.n}")
            st.apply(a)
            root = root.children.get(a)
            time.sleep(args.delay)

def run_gui(args, eval_fn) -> None:
    import pygame
    from gui_common import (WIN_W, WIN_H, draw_state, move_text)
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Rus shashkasi — model self-play (test)")
    font = pygame.font.SysFont("dejavusans", 22)
    small = pygame.font.SysFont("dejavusans", 18)
    clock = pygame.time.Clock()
    mcts = MCTS(eval_fn, seed=args.seed)
    st = GameState()
    root = None
    last_move = None
    last_eval = 0.0
    paused = False
    step_once = False
    delay = args.delay
    winner = None
    move_log: list[str] = []
    quit_flag = False

    def pump() -> None:
        nonlocal quit_flag
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                quit_flag = True
            elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE):
                quit_flag = True

    def redraw(thinking: bool = False) -> None:
        info = [
            f"Ply: {st.ply}",
            f"Navbat: {'OQ' if st.player == WHITE else 'QORA'}",
            f"Sims: {args.sims}",
            f"Tezlik: {delay:.1f}s",
            "PAUZA" if paused else ("o'ylayapti..." if thinking else ""),
        ]
        if winner is not None:
            info.append(f"NATIJA: {result_text(winner)}")
            info.append("N — yangi o'yin")
        info.append("")
        info.extend(move_log[-8:])
        draw_state(screen, font, small, st, last_move=last_move, eval_white=last_eval, info_lines=info)
        pygame.display.flip()
    redraw()
    last_step = time.time()
    while not quit_flag:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                quit_flag = True
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    quit_flag = True
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_RIGHT:
                    step_once = True
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    delay = max(0.0, delay - 0.2)
                elif ev.key == pygame.K_MINUS:
                    delay = min(5.0, delay + 0.2)
                elif ev.key == pygame.K_n:
                    st = GameState()
                    root = None
                    last_move = None
                    winner = None
                    move_log.clear()
        do_step = (winner is None and ((not paused and time.time() - last_step >= delay) or step_once))
        if do_step:
            step_once = False
            winner, _ = st.status()
            if winner is None:
                redraw(thinking=True)
                root = mcts.run(st, root=root, sims=args.sims, add_noise=args.noise, pump=pump)
                if quit_flag:
                    break
                tau = 1.0 if st.ply < args.temp_moves else 0.0
                a = select_action(root, tau)
                last_eval = root_value_white(root, st)
                last_move = action_decode(a)
                move_log.append(f"{st.ply + 1}. {move_text(a)}  ({last_eval:+.2f})")
                st.apply(a)
                root = root.children.get(a)
                winner, _ = st.status()
            last_step = time.time()
            redraw()
        clock.tick(30)
        redraw()
    pygame.quit()

def main() -> None:
    args = parse_args()
    if args.random:
        eval_fn = uniform_eval_fn
        print("MODEL'SIZ rejim: sof MCTS + qoidalar (torch kerak emas)")
    elif args.ckpt or args.onnx:
        eval_fn = build_eval_fn(ckpt=args.ckpt, onnx=args.onnx, device="cpu")
    else:
        print("Xato: --ckpt, --onnx yoki --random bering\nqoidalarni model'siz ko'rish: python play.py --random")
        sys.exit(1)
    if args.ascii:
        run_ascii(args, eval_fn)
    else:
        run_gui(args, eval_fn)

if __name__ == "__main__":
    main()
