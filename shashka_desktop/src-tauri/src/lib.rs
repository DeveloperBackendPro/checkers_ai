//! Shashka Desktop — Tauri backend.
//!
//! O'yin holati (Pos) va ONNX AI (Ai) shu yerda boshqariladi. Frontend
//! (HTML/JS) bilan buyruqlar orqali muloqot qiladi. Barcha qoidalar
//! `engine.rs` da (trening bilan bir xil, fuzzing-sinangan).

mod ai;
mod engine;
use ai::Ai;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::{Manager, State};
use engine::{self as eng, Pos};

struct AppState {
    pos: Arc<Mutex<Pos>>,
    ai: Arc<Mutex<Option<Ai>>>,
    human: Arc<Mutex<i8>>,
}

#[derive(Serialize, Clone)]
struct BoardState {
    board: Vec<i8>,
    player: i8,
    chain_sq: i8,
    winner: Option<i8>,
    legal: Vec<[u8; 2]>,
    human: i8,
    ply: u16,
}

#[derive(Serialize, Clone)]
struct AiStep {
    from: u8,
    to: u8,
    state: BoardState,
}

#[derive(Serialize, Clone)]
struct AiResult {
    steps: Vec<AiStep>,
    confidence: f32,
    value: f32,
    sims: u32,
    state: BoardState,
}

fn make_state(pos: &Pos, human: i8) -> BoardState {
    let (winner, moves) = eng::status(pos);
    let legal: Vec<[u8; 2]> = moves.iter().map(|&(f, t)| [f, t]).collect();
    BoardState {
        board: pos.board_list(),
        player: pos.player,
        chain_sq: pos.chain_sq,
        winner,
        legal,
        human,
        ply: pos.ply,
    }
}

#[tauri::command]
fn new_game(state: State<AppState>, human_color: String) -> BoardState {
    let human = if human_color == "black" { -1 } else { 1 };
    let mut pos = state.pos.lock().unwrap();
    *pos = Pos::initial();
    *state.human.lock().unwrap() = human;
    make_state(&pos, human)
}

#[tauri::command]
fn get_state(state: State<AppState>) -> BoardState {
    let pos = state.pos.lock().unwrap();
    let human = *state.human.lock().unwrap();
    make_state(&pos, human)
}

#[tauri::command]
fn human_move(state: State<AppState>, from: u8, to: u8) -> Result<BoardState, String> {
    let mut pos = state.pos.lock().unwrap();
    let (_, moves) = eng::status(&pos);
    if !moves.iter().any(|&(f, t)| f == from && t == to) {
        return Err("Noqonuniy yurish".to_string());
    }
    let action = from as u16 * 32 + to as u16;
    eng::apply(&mut pos, action);
    let human = *state.human.lock().unwrap();
    Ok(make_state(&pos, human))
}

#[tauri::command]
async fn ai_move(
    state: State<'_, AppState>,
    sims: u32,
    time_limit_ms: u64,
) -> Result<AiResult, String> {
    let ai_arc = state.ai.clone();
    let pos_arc = state.pos.clone();
    let human_arc = state.human.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<AiResult, String> {
        let ai_guard = ai_arc.lock().unwrap();
        let ai = ai_guard.as_ref().ok_or_else(|| "Model yuklanmagan. best.onnx faylini tekshiring.".to_string())?;
        let mut pos = pos_arc.lock().unwrap();
        let human = *human_arc.lock().unwrap();
        if eng::status(&pos).0.is_some() {
            return Err("O'yin tugagan".to_string());
        }
        if pos.player == human {
            return Err("Hozir inson navbati".to_string());
        }
        let mut steps = Vec::new();
        let mut last_conf = 0.0f32;
        let mut last_val = 0.0f32;
        let mut total_sims = 0u32;
        for _ in 0..20 {
            if pos.player == human || eng::status(&pos).0.is_some() {
                break;
            }
            let res = ai.search(&pos, sims, time_limit_ms)?;
            let from = (res.action / 32) as u8;
            let to = (res.action % 32) as u8;
            last_conf = res.confidence;
            last_val = res.value;
            total_sims += res.sims_done;
            eng::apply(&mut pos, res.action);
            steps.push(AiStep {
                from,
                to,
                state: make_state(&pos, human),
            });
        }
        Ok(AiResult {
            steps,
            confidence: last_conf,
            value: last_val,
            sims: total_sims,
            state: make_state(&pos, human),
        })
    })
    .await
    .map_err(|e| format!("fon oqim xatosi: {e}"))?
}

#[tauri::command]
fn model_ready(state: State<AppState>) -> bool {
    state.ai.lock().unwrap().is_some()
}
fn find_model(app: &tauri::App) -> Option<PathBuf> {
    if let Ok(dir) = app.path().resource_dir() {
        let p = dir.join("models").join("best.onnx");
        if p.exists() {
            return Some(p);
        }
    }
    let dev = PathBuf::from("models/best.onnx");
    if dev.exists() {
        return Some(dev);
    }
    let dev2 = PathBuf::from("src-tauri/models/best.onnx");
    if dev2.exists() {
        return Some(dev2);
    }
    None
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState {
            pos: Arc::new(Mutex::new(Pos::initial())),
            ai: Arc::new(Mutex::new(None)),
            human: Arc::new(Mutex::new(1)),
        })
        .setup(|app| {
            let threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            if let Some(model_path) = find_model(app) {
                match Ai::load(&model_path, threads) {
                    Ok(ai) => {
                        let state: State<AppState> = app.state();
                        *state.ai.lock().unwrap() = Some(ai);
                        println!("Model yuklandi: {}", model_path.display());
                    }
                    Err(e) => eprintln!("Model yuklashda xato: {e}"),
                }
            } else {
                eprintln!(
                    "OGOHLANTIRISH: best.onnx topilmadi. \
                     src-tauri/models/best.onnx ga model faylini joylang."
                );
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            new_game,
            get_state,
            human_move,
            ai_move,
            model_ready,
        ])
        .run(tauri::generate_context!())
        .expect("Tauri ishga tushmadi");
}