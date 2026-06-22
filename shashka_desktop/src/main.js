const { invoke } = window.__TAURI__.core;
const SQ_RC = [];
for (let r = 0; r < 8; r++)
  for (let c = 0; c < 8; c++)
    if ((r + c) % 2 === 1) SQ_RC.push([r, c]);
const RC_SQ = Array.from({ length: 8 }, () => Array(8).fill(-1));
SQ_RC.forEach(([r, c], i) => (RC_SQ[r][c] = i));
let state = null;
let selected = null;
let legalMap = new Map();
let flipped = false;
let sims = 0;
let busy = false;
let lastMove = null;
const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const veilEl = document.getElementById("veil");
const veilText = document.getElementById("veil-text");
const evalFill = document.getElementById("evalfill");
const evalVal = document.getElementById("evalval");
const confVal = document.getElementById("confval");
const simsVal = document.getElementById("simsval");
function renderCoords() {
  const files = ["A", "B", "C", "D", "E", "F", "G", "H"];
  const ranks = ["8", "7", "6", "5", "4", "3", "2", "1"];
  const f = flipped ? [...files].reverse() : files;
  const rk = flipped ? [...ranks].reverse() : ranks;
  const fill = (id, arr) => {
    document.getElementById(id).innerHTML = arr.map((x) => `<span>${x}</span>`).join("");
  };
  fill("coords-top", f);
  fill("coords-bottom", f);
  fill("coords-left", rk);
  fill("coords-right", rk);
}
function dispToEngine(dr, dc) {
  return flipped ? [7 - dr, 7 - dc] : [dr, dc];
}
function buildBoard() {
  boardEl.innerHTML = "";
  for (let dr = 0; dr < 8; dr++) {
    for (let dc = 0; dc < 8; dc++) {
      const cell = document.createElement("div");
      const [er, ec] = dispToEngine(dr, dc);
      const isDark = (er + ec) % 2 === 1;
      cell.className = "sq " + (isDark ? "dark" : "light");
      if (isDark) {
        const sq = RC_SQ[er][ec];
        cell.dataset.sq = sq;
        cell.classList.add("playable");
        const hint = document.createElement("div");
        hint.className = "hint";
        cell.appendChild(hint);
        cell.addEventListener("click", () => onSquareClick(sq));
      }
      boardEl.appendChild(cell);
    }
  }
}
function cellBySq(sq) {
  return boardEl.querySelector(`.sq[data-sq="${sq}"]`);
}
function syncPieces() {
  for (let sq = 0; sq < 32; sq++) {
    const cell = cellBySq(sq);
    if (!cell) continue;
    const v = state.board[sq];
    let piece = cell.querySelector(".piece");
    if (v === 0) {
      if (piece) piece.remove();
      continue;
    }
    const white = v > 0;
    const king = Math.abs(v) === 2;
    const cls = "piece " + (white ? "white" : "black") + (king ? " king" : "");
    if (!piece) {
      piece = document.createElement("div");
      cell.appendChild(piece);
    }
    if (piece.className !== cls) piece.className = cls;
    if (piece.style.transform) piece.style.transform = "";
    if (piece.style.transition) piece.style.transition = "";
    if (piece.style.zIndex) piece.style.zIndex = "";
    if (piece.style.opacity) piece.style.opacity = "";
  }
}
function render() {
  if (!state) return;
  boardEl.querySelectorAll(".sq").forEach((c) => {
    c.classList.remove("selected", "show-hint", "capture-hint", "movable");
  });
  syncPieces();
  if (!busy && state.winner === null && state.player === state.human) {
    state.board.forEach((v, sq) => {
      if (v === 0) return;
      const white = v > 0;
      const isHumanPiece = white === (state.human === 1);
      if (isHumanPiece && legalMap.has(sq)) {
        const cell = cellBySq(sq);
        if (cell) cell.classList.add("movable");
      }
    });
  }
  if (selected !== null) {
    const c = cellBySq(selected);
    if (c) c.classList.add("selected");
    const dests = legalMap.get(selected) || [];
    dests.forEach((to) => {
      const tc = cellBySq(to);
      if (tc) tc.classList.add("show-hint");
    });
  }
  updateStatus();
}
function updateStatus() {
  const dot = statusEl.querySelector(".turn-dot");
  const txt = statusEl.querySelector(".turn-text");
  statusEl.className = "turn";
  const setDot = (cls) => {
    if (dot) dot.className = "turn-dot " + cls;
  };
  const set = (t) => {
    if (txt) txt.textContent = t;
    else statusEl.textContent = t;
  };
  if (state.winner !== null) {
    if (state.winner === 0) {
      set("Durang");
      statusEl.classList.add("draw");
      setDot("draw");
    } else if (state.winner === state.human) {
      set("Вы выиграли !");
      statusEl.classList.add("win");
      setDot("win");
    } else {
      set("ИИ выиграл !");
      statusEl.classList.add("lose");
      setDot("lose");
    }
    return;
  }
  if (busy) {
    set("ИИ думает ...");
    setDot("ai");
  } else if (state.player === state.human) {
    const side = state.human === 1 ? "Белые" : "Чёрные";
    set(state.chain_sq >= 0 ? "Бейте дальше" : `Ваш ход (${side})`);
    setDot(state.human === 1 ? "white" : "black");
  } else {
    set("Ход ИИ");
    setDot("ai");
  }
}
function rebuildLegal() {
  legalMap = new Map();
  for (const [f, t] of state.legal) {
    if (!legalMap.has(f)) legalMap.set(f, []);
    legalMap.get(f).push(t);
  }
}
async function onSquareClick(sq) {
  if (busy || !state || state.winner !== null) return;
  if (state.player !== state.human) return;
  if (state.chain_sq >= 0) selected = state.chain_sq;
  if (selected !== null) {
    const dests = legalMap.get(selected) || [];
    if (dests.includes(sq)) {
      await doHumanMove(selected, sq);
      return;
    }
  }
  if (legalMap.has(sq)) {
    selected = sq;
    render();
  } else {
    selected = null;
    render();
  }
}
async function doHumanMove(from, to) {
  try {
    busy = true;
    const prevBoard = state.board.slice();
    const newState = await invoke("human_move", { from, to });
    if (!reducedMotion) await animateMove(prevBoard, from, to, newState.board);
    state = newState;
    lastMove = [from, to];
    rebuildLegal();
    if (state.winner === null && state.player === state.human && state.chain_sq >= 0) {
      selected = state.chain_sq;
      busy = false;
      render();
      return;
    }
    selected = null;
    busy = false;
    render();
    if (state.winner === null && state.player !== state.human) {
      await runAi();
    }
  } catch (e) {
    busy = false;
    console.error(e);
    statusEl.textContent = "Ошибка: " + e;
  }
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
async function animateMove(prevBoard, from, to, newBoard) {
  const fromCell = cellBySq(from);
  const toCell = cellBySq(to);
  if (!fromCell || !toCell) return;
  const piece = fromCell.querySelector(".piece");
  if (!piece) return;
  const fr = fromCell.getBoundingClientRect();
  const tr = toCell.getBoundingClientRect();
  const dx = tr.left - fr.left;
  const dy = tr.top - fr.top;
  const captured = [];
  for (let sq = 0; sq < 32; sq++) {
    if (prevBoard[sq] !== 0 && newBoard[sq] === 0 && sq !== from && sq !== to) {
      captured.push(sq);
    }
  }
  piece.style.zIndex = "6";
  piece.style.transition = "transform 0.40s cubic-bezier(.45,.05,.25,1)";
  piece.style.transform = `translate(${dx}px, ${dy}px)`;
  await sleep(170);
  captured.forEach((sq) => {
    const c = cellBySq(sq);
    const p = c && c.querySelector(".piece");
    if (p) {
      p.style.transition = "opacity 0.30s ease, transform 0.30s ease";
      p.style.opacity = "0";
      p.style.transform = "scale(0.25)";
    }
  });
  await sleep(260);
  piece.style.transition = "";
  piece.style.transform = "";
  piece.style.zIndex = "";
  toCell.appendChild(piece);
  captured.forEach((sq) => {
    const c = cellBySq(sq);
    const p = c && c.querySelector(".piece");
    if (p) p.remove();
  });
}
async function runAi() {
  busy = true;
  selected = null;
  render();
  showVeil(true);
  try {
    const res = await invoke("ai_move", { sims, timeLimitMs: 0 });
    showVeil(false);
    const steps = res.steps || [];
    let prevBoard = state.board.slice();
    for (let i = 0; i < steps.length; i++) {
      const s = steps[i];
      if (!reducedMotion) await animateMove(prevBoard, s.from, s.to, s.state.board);
      state = s.state;
      lastMove = [s.from, s.to];
      rebuildLegal();
      render();
      prevBoard = s.state.board.slice();
      if (!reducedMotion && steps.length > 1 && i < steps.length - 1) {
        await sleep(160);
      }
    }
    state = res.state;
    rebuildLegal();
    updateReadout(res.value, res.confidence, res.sims);
  } catch (e) {
    console.error(e);
    statusEl.textContent = "Ошибка ИИ: " + e;
  } finally {
    busy = false;
    showVeil(false);
    render();
  }
}
function showVeil(on) {
  veilEl.classList.toggle("show", on);
}
function updateReadout(value, confidence, simsDone) {
  const shown = -value;
  const pct = Math.max(-1, Math.min(1, shown));
  evalFill.style.left = `${(pct + 1) * 50}%`;
  evalVal.textContent = (shown >= 0 ? "+" : "") + shown.toFixed(2);
  confVal.textContent = Math.round(confidence * 100) + "%";
  simsVal.textContent = simsDone.toLocaleString("en-US");
}
async function newGame(humanColor) {
  flipped = humanColor === "black";
  renderCoords();
  buildBoard();
  state = await invoke("new_game", { humanColor });
  rebuildLegal();
  selected = null;
  lastMove = null;
  busy = false;
  evalFill.style.left = "50%";
  evalVal.textContent = "—";
  confVal.textContent = "—";
  simsVal.textContent = "—";
  render();
  if (state.winner === null && state.player !== state.human) {
    await runAi();
  }
}
document.getElementById("btn-new-white").addEventListener("click", () => newGame("white"));
document.getElementById("btn-new-black").addEventListener("click", () => newGame("black"));
document.querySelectorAll(".seg-btn").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    sims = parseInt(b.dataset.sims, 10);
  });
});
async function init() {
  renderCoords();
  buildBoard();
  try {
    const ready = await invoke("model_ready");
    if (!ready) {
      statusEl.textContent = "best.onnx не найден. Разместите модель";
    }
  } catch (_) {}
  await newGame("white");
}
init();
