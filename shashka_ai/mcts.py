""" PUCT asosidagi Monte-Carlo Tree Search (MCTS).
Xususiyatlar:
  - PUCT formulasi (AlphaZero)
  - Dirichlet noise (root exploration)
  - Temperature bilan yurish tanlash
  - Daraxtni qayta ishlatish (tree reuse)
  - Vaqt limiti (dynamic search budget)
  - Lockstep-batch self-play uchun collect/prepare/finish API
Qiymat konventsiyasi: har bir tugun qiymati O'SHA TUGUNDA YURADIGAN o'yinchi
nuqtai nazaridan saqlanadi. Zanjirli urishlarda navbat o'zgarmasligi to'g'ri
hisobga olinadi (backprop'da player solishtiriladi). """

from __future__ import annotations
import math
import time
import numpy as np
from checkers_engine import GameState, action_encode
from typing import Callable, Dict, List, Optional, Tuple
EvalFn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]

class Node:
    __slots__ = ("prior", "player", "n", "w", "vloss", "children", "is_expanded", "terminal_value", "pending")
    def __init__(self, prior: float = 1.0) -> None:
        self.prior: float = prior
        self.player: int = 0
        self.n: int = 0
        self.w: float = 0.0
        self.vloss: int = 0
        self.children: Dict[int, "Node"] = {}
        self.is_expanded: bool = False
        self.terminal_value: Optional[float] = None
        self.pending: Optional[List[int]] = None

    @property
    def q(self) -> float:
        return self.w / self.n if self.n > 0 else 0.0

class MCTS:
    def __init__(self, eval_fn: EvalFn, c_puct: float = 1.6,
        dirichlet_alpha: float = 1.0, dirichlet_eps: float = 0.25,
        seed: Optional[int] = None) -> None:
        self.eval_fn = eval_fn
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = np.random.default_rng(seed)

    def _best_child(self, node: Node) -> Tuple[int, Node]:
        sqrt_n = math.sqrt(node.n + node.vloss + 1)
        best_a, best_c, best_s = -1, None, -1e18
        for a, ch in node.children.items():
            nv = ch.n + ch.vloss
            if nv > 0:
                w_signed = ch.w if ch.player == node.player else -ch.w
                q = (w_signed - ch.vloss) / nv
            else:
                q = 0.0
            s = q + self.c_puct * ch.prior * sqrt_n / (1 + nv)
            if s > best_s:
                best_a, best_c, best_s = a, ch, s
        assert best_c is not None
        return best_a, best_c

    def collect(self, root: Node, root_state: GameState) -> Tuple[List[Node], Node, GameState]:
        st = root_state.clone()
        node, path = root, [root]
        node.vloss += 1
        while node.is_expanded and node.terminal_value is None:
            a, child = self._best_child(node)
            st.apply(a)
            node = child
            node.vloss += 1
            path.append(node)
        return path, node, st

    def prepare_leaf(self, leaf: Node, st: GameState) -> Optional[np.ndarray]:
        winner, moves = st.status()
        leaf.player = st.player
        if winner is not None:
            leaf.terminal_value = (0.0 if winner == 0 else (1.0 if winner == st.player else -1.0))
            return None
        leaf.pending = [action_encode(f, t) for f, t in moves]
        return st.encode()

    def finish_leaf(self, leaf: Node, st: GameState, logits: np.ndarray) -> None:
        acts = leaf.pending or []
        leaf.pending = None
        canon = np.fromiter((st.canon_action(a) for a in acts), dtype=np.int64, count=len(acts))
        l = logits[canon].astype(np.float64)
        l -= l.max()
        p = np.exp(l)
        p /= p.sum()
        for a, pr in zip(acts, p):
            leaf.children[a] = Node(float(pr))
        leaf.is_expanded = True

    def backprop(self, path: List[Node], leaf_player: int, value: float, revert_vloss: bool = False) -> None:
        for node in path:
            if revert_vloss:
                node.vloss -= 1
            node.n += 1
            node.w += value if node.player == leaf_player else -value

    def revert_virtual(self, path: List[Node]) -> None:
        for node in path:
            node.vloss -= 1

    def add_noise(self, root: Node) -> None:
        n = len(root.children)
        if n == 0:
            return
        noise = self.rng.dirichlet([self.dirichlet_alpha] * n)
        for ch, x in zip(root.children.values(), noise):
            ch.prior = (1 - self.dirichlet_eps) * ch.prior + self.dirichlet_eps * float(x)

    def run(self, state: GameState, root: Optional[Node] = None, sims: int = 200,
        add_noise: bool = False, time_limit: Optional[float] = None,
        pump: Optional[Callable[[], None]] = None) -> Node:
        t0 = time.time()
        if root is None:
            root = Node(1.0)
        if not root.is_expanded and root.terminal_value is None:
            x = self.prepare_leaf(root, state.clone())
            if x is not None:
                logits, vals = self.eval_fn(x[None])
                self.finish_leaf(root, state, logits[0])
                self.backprop([root], root.player, float(vals[0]))
        if add_noise:
            self.add_noise(root)
        for i in range(sims):
            path, leaf, st = self.collect(root, state)
            if leaf.terminal_value is not None:
                self.backprop(path, leaf.player, leaf.terminal_value, revert_vloss=True)
            else:
                x = self.prepare_leaf(leaf, st)
                if x is None:
                    self.backprop(path, leaf.player, leaf.terminal_value, revert_vloss=True)
                else:
                    logits, vals = self.eval_fn(x[None])
                    self.finish_leaf(leaf, st, logits[0])
                    self.backprop(path, leaf.player, float(vals[0]), revert_vloss=True)
            if pump is not None and (i & 15) == 0:
                pump()
            if time_limit is not None and time.time() - t0 > time_limit:
                break
        return root

def visit_distribution(root: Node) -> Tuple[List[int], np.ndarray]:
    acts = list(root.children.keys())
    v = np.array([root.children[a].n for a in acts], dtype=np.float32)
    s = float(v.sum())
    if s > 0:
        v /= s
    else:
        v[:] = 1.0 / max(len(acts), 1)
    return acts, v

def select_action(root: Node, temperature: float,
    rng: Optional[np.random.Generator] = None) -> int:
    acts = list(root.children.keys())
    visits = np.array([root.children[a].n for a in acts], dtype=np.float64)
    if visits.sum() <= 0:
        visits += 1.0
    if temperature <= 1e-3:
        return acts[int(np.argmax(visits))]
    rng = rng or np.random.default_rng()
    w = visits ** (1.0 / temperature)
    w /= w.sum()
    return acts[int(rng.choice(len(acts), p=w))]

def root_value_white(root: Node, state: GameState) -> float:
    v = root.q
    return v if state.player == 1 else -v