""" train.py — Rus shashkasi AI ni NOLDAN, faqat self-play orqali o'qitish.
Hech qanday tayyor model, inson partiyalari, debyut kitobi yoki qo'lda
yozilgan baholash funksiyasi ishlatilmaydi. Model barcha strategiyalarni
o'zi kashf qiladi.
Tsikl (har iteratsiya):
  1) Self-play: joriy model o'zi bilan N ta o'yin o'ynaydi (parallel,
     lockstep-batch MCTS, GPU da). Curriculum (endshpil-start) va debyut
     diversifikatsiyasi qo'llanadi.
  2) Trening: replay bufferdan namunalar bilan policy CE + WDL CE loss,
     AMP (mixed precision), gradient accumulation.
  3) Checkpoint: latest.pt har iteratsiyada saqlanadi (resume uchun).
  4) Arena gating (har --arena-every iteratsiyada): yangi model "best" ga
     qarshi o'ynaydi; >= --gate (default 0.55) ball olsa yangi "best"
     bo'ladi, ELO yangilanadi va best.onnx eksport qilinadi.
  5) Xavfsizlik: GPU/CPU harorati chegaradan oshsa avtomatik pauza.
Foydalanish (Google Colab T4):
    python train.py --device cuda --iterations 300
Tezkor sinov (CPU):
    python train.py --quick
Faqat ONNX eksport:
    python train.py --export-onnx checkpoints/best.pt --onnx-out model.onnx """

from __future__ import annotations
import os
import time
import copy
import math
import torch
import argparse
import numpy as np
from typing import Optional
import torch.nn.functional as F
from safety import ThermalGuard
from mcts import MCTS, select_action
from network import ShashkaNet, export_onnx
from checkers_engine import GameState, WHITE
from inference import TorchEvaluator, load_ckpt, wdl_to_value
from selfplay import ParallelSelfPlay, ReplayBuffer

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rus shashkasi AI — self-play trening")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--engine", choices=["auto", "rust", "python"], default="auto", help="self-play dvijogi: rust (~60x tez), python (etalon), auto (rust bor bo'lsa rust)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--log-dir", default="runs")
    p.add_argument("--resume", default=None, help="checkpoint yo'li (default: auto latest.pt)")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--policy-dim", type=int, default=64)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--max-hours", type=float, default=None, help="vaqt budjeti (soat): tugashidan oldin chiroyli to'xtab, checkpoint+ONNX saqlaydi (Colab sessiyasi uchun, masalan: --max-hours 9.5)")
    p.add_argument("--games-per-iter", type=int, default=64)
    p.add_argument("--parallel", type=int, default=64)
    p.add_argument("--sims", type=int, default=120, help="boshlang'ich simulyatsiyalar")
    p.add_argument("--sims-final", type=int, default=240, help="oxirgi iteratsiyadagi simulyatsiyalar (dynamic budget)")
    p.add_argument("--temp-moves", type=int, default=24)
    p.add_argument("--temp-schedule", choices=["step", "linear"], default="step", help="temperatura jadvali: step (AlphaZero) yoki linear (silliq)")
    p.add_argument("--opening-random-plies", type=int, default=4)
    p.add_argument("--curriculum-frac", type=float, default=0.25)
    p.add_argument("--curriculum-max-pieces", type=int, default=4, help="curriculum endshpilida har tomonda maks. dona (1-6)")
    p.add_argument("--leaves-per-wave", type=int, default=4, help="har daraxtdan bir to'lqinda yig'iladigan barglar (virtual loss bilan; GPU batch samaradorligi)")
    p.add_argument("--c-puct", type=float, default=1.6)
    p.add_argument("--dirichlet-alpha", type=float, default=1.0)
    p.add_argument("--dirichlet-eps", type=float, default=0.25)
    p.add_argument("--buffer-size", type=int, default=300_000)
    p.add_argument("--min-buffer", type=int, default=8_000)
    p.add_argument("--train-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--accum", type=int, default=1, help="gradient accumulation qadamlar")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--arena-every", type=int, default=5)
    p.add_argument("--arena-games", type=int, default=20)
    p.add_argument("--arena-sims", type=int, default=120)
    p.add_argument("--arena-temp-plies", type=int, default=6, help="arena o'yinlarida dastlabki shu yurishlar tashriflarga proporsional tanlanadi (xilma-xillik); keyin argmax")
    p.add_argument("--gate", type=float, default=0.55)
    p.add_argument("--gpu-max-temp", type=int, default=79)
    p.add_argument("--gpu-resume-temp", type=int, default=68)
    p.add_argument("--cpu-max-temp", type=int, default=90)
    p.add_argument("--cpu-resume-temp", type=int, default=78)
    p.add_argument("--export-onnx", default=None, metavar="CKPT", help="faqat eksport rejimi: checkpointdan ONNX yasash")
    p.add_argument("--onnx-out", default="model.onnx")
    p.add_argument("--quick", action="store_true", help="kichik smoke-test rejimi")
    args = p.parse_args()
    if args.quick:
        args.iterations = 2
        args.games_per_iter = 4
        args.parallel = 4
        args.sims = 16
        args.sims_final = 16
        args.train_steps = 10
        args.batch_size = 64
        args.min_buffer = 50
        args.arena_every = 2
        args.arena_games = 2
        args.arena_sims = 16
        args.d_model = 64
        args.n_layers = 2
        args.n_heads = 4
        args.policy_dim = 32
        args.curriculum_frac = 0.0
    return args

def play_arena(eval_new, eval_best, n_games: int, sims: int, seed: int, guard: Optional[ThermalGuard] = None) -> float:
    score = 0.0
    for g in range(n_games):
        new_is_white = (g % 2 == 0)
        st = GameState()
        m_new = MCTS(eval_new, dirichlet_eps=0.10, seed=seed + g)
        m_best = MCTS(eval_best, dirichlet_eps=0.10, seed=seed + 10_000 + g)
        roots = {"new": None, "best": None}
        while True:
            winner, _ = st.status()
            if winner is not None:
                break
            actor = "new" if ((st.player == WHITE) == new_is_white) else "best"
            m = m_new if actor == "new" else m_best
            root = m.run(st, root=roots[actor], sims=sims, add_noise=(st.ply < 8))
            a = select_action(root, temperature=1.0 if st.ply < 6 else 0.0)
            st.apply(a)
            for k in roots:
                r = roots[k]
                roots[k] = r.children.get(a) if (r is not None and a in r.children) else None
            roots[actor] = root.children.get(a)
            if guard is not None:
                guard.periodic_check()
        if winner == 0:
            score += 0.5
        elif (winner == WHITE) == new_is_white:
            score += 1.0
    return score / n_games

def elo_delta(score: float, n_games: int) -> float:
    lo = 1.0 / (2.0 * max(n_games, 1))
    s = min(max(score, lo), 1.0 - lo)
    return 400.0 * math.log10(s / (1.0 - s))

def train_phase(model, optimizer, scaler, buffer: ReplayBuffer, args,
    rng: np.random.Generator, guard: ThermalGuard,
    global_step: int, device: str):
    model.train()
    use_amp = (not args.no_amp) and device.startswith("cuda")
    pol_losses, wdl_losses = [], []
    pol_correct = wdl_correct = seen = 0
    for step in range(args.train_steps):
        guard.periodic_check()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            xs, Pi, zs = buffer.sample(args.batch_size, rng)
            x = torch.from_numpy(xs).to(device)
            pi = torch.from_numpy(Pi).to(device)
            z = torch.from_numpy(zs).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits, wdl = model(x)
                logp = F.log_softmax(logits.float(), dim=-1)
                policy_loss = -(pi * logp).sum(dim=1).mean()
                wdl_loss = F.cross_entropy(wdl.float(), z)
                loss = (policy_loss + wdl_loss) / args.accum
            scaler.scale(loss).backward()
            pol_losses.append(float(policy_loss.detach()))
            wdl_losses.append(float(wdl_loss.detach()))
            with torch.no_grad():
                pol_correct += int((logits.argmax(1) == pi.argmax(1)).sum())
                wdl_correct += int((wdl.argmax(1) == z).sum())
                seen += x.shape[0]
        global_step += 1
        lr = args.lr * min(1.0, global_step / max(args.warmup_steps, 1))
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if (step + 1) % 50 == 0 or step + 1 == args.train_steps:
            print(f"\r        trening {step + 1}/{args.train_steps} "
            f"({(step + 1) / args.train_steps:.0%}): "
            f"policy={np.mean(pol_losses[-50:]):.3f} "
            f"wdl={np.mean(wdl_losses[-50:]):.3f} "
            f"yurish-aniqligi={pol_correct / max(seen, 1):.1%} "
            f"natija-aniqligi={wdl_correct / max(seen, 1):.1%}", end="", flush=True)
    print()
    model.eval()
    return (float(np.mean(pol_losses)), float(np.mean(wdl_losses)), pol_correct / max(seen, 1), wdl_correct / max(seen, 1), global_step)

def save_ckpt(path: str, model, optimizer, iteration: int, elo: float, global_step: int, args) -> None:
    import tempfile
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "elo": elo,
        "global_step": global_step,
        "hparams": model.hparams,
        "args": vars(args),
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".pt.tmp", dir=directory)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

def save_buffer(path: str, buffer) -> None:
    import pickle
    import tempfile
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".pkl.tmp", dir=directory)
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            pickle.dump(buffer.state(), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

def load_buffer(path: str, buffer) -> bool:
    import pickle
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            st = pickle.load(f)
        buffer.load_state(st)
        return True
    except Exception as e:
        print(f"[!] Buffer o'qishda xato ({e}). Buffer 0 dan to'ladi.")
        return False

def main() -> None:
    args = parse_args()
    if args.export_onnx:
        ckpt = load_ckpt(args.export_onnx)
        net = ShashkaNet(**ckpt["hparams"])
        net.load_state_dict(ckpt["model"])
        export_onnx(net, args.onnx_out)
        print(f"ONNX eksport qilindi: {args.onnx_out}")
        return
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = args.device
    print(f"Qurilma: {device}")
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.log_dir)
    guard = ThermalGuard(
    gpu_max=args.gpu_max_temp, gpu_resume=args.gpu_resume_temp,
    cpu_max=args.cpu_max_temp, cpu_resume=args.cpu_resume_temp)
    model = ShashkaNet(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, policy_dim=args.policy_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(not args.no_amp) and device.startswith("cuda"))
    start_iter, elo, global_step = 1, 1000.0, 0
    resume_path = args.resume or os.path.join(args.ckpt_dir, "latest.pt")
    if os.path.exists(resume_path):
        try:
            ckpt = load_ckpt(resume_path)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_iter = ckpt["iteration"] + 1
            elo = ckpt.get("elo", 1000.0)
            global_step = ckpt.get("global_step", 0)
            print(f"Resume: {resume_path} (iteratsiya {start_iter}, ELO {elo:.0f})")
        except Exception as e:
            print(f"[!] Checkpoint o'qishda xato ({e}). Noldan boshlanadi. "
            f"Eski faylni saqlab qo'yish uchun nusxalab oling.")
    print(f"Model parametrlari: {model.num_params():,}")
    SPClass = ParallelSelfPlay
    engine_name = "python"
    if args.engine in ("auto", "rust"):
        try:
            from selfplay_rust import RustParallelSelfPlay
            SPClass = RustParallelSelfPlay
            engine_name = "rust"
        except ImportError:
            if args.engine == "rust":
                raise SystemExit(
                "Rust dvijok topilmadi. Qurish: pip install maturin && "
                "cd rust && maturin build --release -o dist && "
                "pip install dist/*.whl")
            print("[!] Rust dvijok topilmadi — Python dvijokda davom etamiz "
            "(~60x sekinroq). Qurish yo'riqnomasi README'da.")
    print(f"Self-play dvijogi: {engine_name.upper()}")
    best_model = copy.deepcopy(model)
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    if os.path.exists(best_path):
        bckpt = load_ckpt(best_path)
        best_model.load_state_dict(bckpt["model"])
    buffer = ReplayBuffer(args.buffer_size)
    buffer_path = os.path.join(args.ckpt_dir, "buffer.pkl")
    if os.path.exists(resume_path):
        if load_buffer(buffer_path, buffer):
            print(f"Buffer yuklandi: {len(buffer):,}/{args.buffer_size:,} "
            f"({len(buffer) / args.buffer_size:.0%})")
        else:
            print("Buffer topilmadi — 0 dan to'ladi (birinchi marta normal).")
    model.eval()
    t_start = time.time()
    iter_times: list = []
    print("\nKo'rsatkichlar izohi:")
    print("  policy_loss      pasaysa — model MCTS qidiruvi tanlagan yurishlarni o'rganyapti")
    print("  wdl_loss         pasaysa — model o'yin natijasini (W/D/L) aniqroq bashorat qilyapti")
    print("  yurish-aniqligi  modelning 1-tanlovi MCTS tanlovi bilan mos kelgan foiz (oshishi kerak)")
    print("  natija-aniqligi  W/D/L bashorati to'g'ri chiqqan foiz (oshishi kerak)")
    print("  durang foizi     o'sib borishi normal — kuchli o'yinchilar ko'p durang qiladi")
    print("  ELO              faqat arena gating'dan o'tganda oshadi — isbotlangan kuchayish\n")
    for it in range(start_iter, args.iterations + 1):
        elapsed_h = (time.time() - t_start) / 3600.0
        if args.max_hours is not None and elapsed_h >= args.max_hours:
            save_buffer(os.path.join(args.ckpt_dir, "buffer.pkl"), buffer)
            print(f"\n[VAQT] {elapsed_h:.1f} soat o'tdi (budjet "
            f"{args.max_hours}h) — trening chiroyli to'xtatilmoqda. "
            f"Keyingi sessiyada xuddi shu buyruq bilan davom etadi.")
            break
        done_frac = (it - start_iter) / max(args.iterations - start_iter + 1, 1)
        if iter_times:
            avg_it = float(np.mean(iter_times[-5:]))
            remain_iters = args.iterations - it + 1
            eta_h = avg_it * remain_iters / 3600.0
            if args.max_hours is not None:
                eta_h = min(eta_h, max(args.max_hours - elapsed_h, 0.0))
            eta_txt = f"ETA ~{eta_h:.1f} soat"
        else:
            eta_txt = "ETA hisoblanmoqda..."
        print(f"=== Iteratsiya {it}/{args.iterations} ({done_frac:.0%}) | "
        f"o'tdi {elapsed_h:.1f}h | {eta_txt} | ELO {elo:.0f} | "
        f"buffer {len(buffer):,}/{args.buffer_size:,} "
        f"({len(buffer) / args.buffer_size:.0%}) ===")
        t_iter = time.time()
        guard.wait_if_hot()
        frac = (it - 1) / max(args.iterations - 1, 1)
        sims = int(round(args.sims + frac * (args.sims_final - args.sims)))
        eval_fn = TorchEvaluator(model, device=device, use_amp=not args.no_amp)
        sp = SPClass(
            eval_fn, n_parallel=args.parallel, sims=sims,
            temp_moves=args.temp_moves, temp_schedule=args.temp_schedule,
            opening_random_plies=args.opening_random_plies,
            curriculum_frac=args.curriculum_frac,
            curriculum_max_pieces=args.curriculum_max_pieces,
            leaves_per_wave=args.leaves_per_wave, c_puct=args.c_puct,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_eps=args.dirichlet_eps, seed=args.seed + it)
        t0 = time.time()
        def _sp_progress(s) -> None:
            el = time.time() - t0
            per_game = el / max(s.games, 1)
            left = per_game * (args.games_per_iter - s.games)
            print(f"\r        self-play: {s.games}/{args.games_per_iter} o'yin "
            f"({s.games / args.games_per_iter:.0%}) | durang {s.draws} | "
            f"~{left:.0f}s qoldi   ", end="", flush=True)
        positions, stats = sp.play_games(args.games_per_iter, progress_cb=_sp_progress)
        print()
        sp_time = time.time() - t0
        buffer.extend(positions)
        print(f"        self-play yakuni: {len(positions)} pozitsiya | "
        f"oq {stats.white_wins} / durang {stats.draws} / "
        f"qora {stats.black_wins} (durang {stats.draw_rate:.0%}) | "
        f"o'rtacha {stats.avg_length:.0f} ply | sims={sims} | "
        f"{sp_time:.0f}s")
        writer.add_scalar("selfplay/draw_rate", stats.draw_rate, it)
        writer.add_scalar("selfplay/avg_length", stats.avg_length, it)
        writer.add_scalar("selfplay/positions", len(positions), it)
        writer.add_scalar("selfplay/sims", sims, it)
        g_t, c_t = guard.temps()
        if g_t is not None:
            writer.add_scalar("safety/gpu_temp", g_t, it)
        if c_t is not None:
            writer.add_scalar("safety/cpu_temp", c_t, it)
        if len(buffer) >= args.min_buffer:
            guard.wait_if_hot()
            pol, wdl, pol_acc, wdl_acc, global_step = train_phase(model, optimizer, scaler, buffer, args, rng, guard, global_step, device)
            writer.add_scalar("loss/policy", pol, it)
            writer.add_scalar("loss/wdl", wdl, it)
            writer.add_scalar("acc/policy", pol_acc, it)
            writer.add_scalar("acc/wdl", wdl_acc, it)
        else:
            print(f"        buffer hali kichik ({len(buffer)}<{args.min_buffer}), "
            f"trening o'tkazib yuborildi")
        if it % args.arena_every == 0 and len(buffer) >= args.min_buffer:
            guard.wait_if_hot()
            eval_new = TorchEvaluator(model, device=device, use_amp=not args.no_amp)
            eval_best = TorchEvaluator(best_model.to(device), device=device, use_amp=not args.no_amp)
            t0 = time.time()
            if engine_name == "rust":
                from arena_rust import play_arena_rust
                score = play_arena_rust(
                eval_new, eval_best, args.arena_games, args.arena_sims,
                seed=args.seed * 7 + it,
                leaves_per_wave=args.leaves_per_wave,
                opening_temp_plies=args.arena_temp_plies,
                c_puct=args.c_puct, guard=guard)
            else:
                score = play_arena(eval_new, eval_best, args.arena_games,
                args.arena_sims, seed=args.seed * 7 + it,
                guard=guard)
            print(f"        arena ({engine_name}, {args.arena_games} o'yin): "
            f"yangi vs best = {score:.2f} "
            f"({time.time() - t0:.0f}s)", end="")
            writer.add_scalar("arena/score", score, it)
            if score >= args.gate:
                elo += elo_delta(score, args.arena_games)
                best_model.load_state_dict(model.state_dict())
                save_ckpt(best_path, model, optimizer, it, elo, global_step, args)
                onnx_path = os.path.join(args.ckpt_dir, "best.onnx")
                try:
                    export_onnx(copy.deepcopy(model), onnx_path)
                    print(f" -> YANGI BEST! ELO={elo:.0f}, {onnx_path} yangilandi")
                except Exception as e:
                    print(f" -> YANGI BEST! ELO={elo:.0f} (ONNX xato: {e})")
                model.to(device)
            else:
                print(" -> rad etildi (gate dan past)")
            writer.add_scalar("arena/elo", elo, it)
        save_ckpt(os.path.join(args.ckpt_dir, "latest.pt"), model, optimizer, it, elo, global_step, args)
        if it % 5 == 0:
            save_buffer(os.path.join(args.ckpt_dir, "buffer.pkl"), buffer)
        iter_times.append(time.time() - t_iter)
        writer.add_scalar("time/iteration_s", time.time() - t_iter, it)
        writer.flush()
    final_onnx = os.path.join(args.ckpt_dir, "final.onnx")
    export_onnx(copy.deepcopy(model).cpu(), final_onnx)
    print(f"\nTrening tugadi. Yakuniy ONNX: {final_onnx}")
    print(f"Eng kuchli model: {best_path} (ELO {elo:.0f})")
    writer.close()

if __name__ == "__main__":
    main()