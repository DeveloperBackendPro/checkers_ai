""" analyze.py — Model NIMANI o'rganganini ko'rish (tadqiqot vositasi).
Rejimlar:
  --mode attention   Transformer attention xaritalarini PNG ga chiqaradi:
                     har qatlam uchun 33x33 (CLS + 32 katak) o'rtacha xarita
                     va eng faol kataklarning doskadagi issiqlik xaritasi.
  --mode values      Bitta self-play o'yin davomida WDL bahosi qanday
                     o'zgarishini grafikda ko'rsatadi (model qachon ustunlikni
                     "ko'rishini" tahlil qilish uchun).
Foydalanish:
  python analyze.py --ckpt checkpoints/best.pt --mode attention --plies 20
  python analyze.py --ckpt checkpoints/best.pt --mode values --sims 100 """

from __future__ import annotations
import os
import argparse
import numpy as np
from checkers_engine import SQ_TO_RC, GameState, sq_name

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model tahlili")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mode", choices=["attention", "values"], default="attention")
    p.add_argument("--plies", type=int, default=16, help="attention: shuncha yurishdan keyingi pozitsiya olinadi")
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--out-dir", default="analysis")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def _play_to_position(eval_fn, plies: int, sims: int, seed: int) -> GameState:
    from mcts import MCTS, select_action
    st = GameState()
    m = MCTS(eval_fn, seed=seed)
    root = None
    for _ in range(plies):
        winner, _ = st.status()
        if winner is not None:
            break
        root = m.run(st, root=root, sims=sims, add_noise=True)
        a = select_action(root, temperature=1.0)
        st.apply(a)
        root = root.children.get(a)
    return st

def mode_attention(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from inference import TorchEvaluator, build_network_from_ckpt
    net, _ = build_network_from_ckpt(args.ckpt)
    eval_fn = TorchEvaluator(net, "cpu")
    st = _play_to_position(eval_fn, args.plies, args.sims, args.seed)
    print("Tahlil qilinayotgan pozitsiya:")
    print(st)
    for blk in net.blocks:
        blk.attn.store_attn = True
    with torch.no_grad():
        net(torch.from_numpy(st.encode()[None]))
    for blk in net.blocks:
        blk.attn.store_attn = False
    labels = ["CLS"] + [sq_name(s) for s in range(32)]
    os.makedirs(args.out_dir, exist_ok=True)
    for li, blk in enumerate(net.blocks):
        att = blk.attn.last_attn[0].mean(dim=0).numpy()
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ax = axes[0]
        im = ax.imshow(att, cmap="viridis")
        ax.set_title(f"Qatlam {li}: attention (boshlar o'rtachasi)")
        ax.set_xticks(range(33), labels, rotation=90, fontsize=6)
        ax.set_yticks(range(33), labels, fontsize=6)
        fig.colorbar(im, ax=ax)
        cls_att = att[0, 1:]
        heat = np.zeros((8, 8))
        for s, (r, c) in enumerate(SQ_TO_RC):
            heat[r, c] = cls_att[s]
        ax = axes[1]
        im = ax.imshow(heat, cmap="hot")
        ax.set_title("CLS -> kataklar (strategik diqqat)")
        ax.set_xticks(range(8), list("abcdefgh"))
        ax.set_yticks(range(8), [str(8 - r) for r in range(8)])
        fig.colorbar(im, ax=ax)
        path = os.path.join(args.out_dir, f"attention_layer{li}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"  saqlandi: {path}")

def mode_values(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from inference import build_eval_fn
    from mcts import MCTS, root_value_white, select_action
    eval_fn = build_eval_fn(ckpt=args.ckpt, device="cpu")
    st = GameState()
    m = MCTS(eval_fn, seed=args.seed)
    root = None
    vals, plies = [], []
    while True:
        winner, _ = st.status()
        if winner is not None:
            break
        root = m.run(st, root=root, sims=args.sims, add_noise=True)
        vals.append(root_value_white(root, st))
        plies.append(st.ply)
        a = select_action(root, 1.0 if st.ply < 16 else 0.0)
        st.apply(a)
        root = root.children.get(a)
    res = {1: "OQ yutdi", -1: "QORA yutdi", 0: "DURANG"}[winner]
    os.makedirs(args.out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(plies, vals, lw=2)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("ply")
    ax.set_ylabel("baho (oq nuqtai nazaridan)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"O'yin davomida model bahosi — natija: {res} ({st.ply} ply)")
    path = os.path.join(args.out_dir, "game_values.png")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  saqlandi: {path}  (natija: {res})")

def main() -> None:
    args = parse_args()
    if args.mode == "attention":
        mode_attention(args)
    else:
        mode_values(args)

if __name__ == "__main__":
    main()
