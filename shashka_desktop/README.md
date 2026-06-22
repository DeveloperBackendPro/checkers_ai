<h1 align="center">Checkers with AI</h1>

<p align="center">
  <b>Rus shashkasi (Шашки) uchun ish stoli ilovasi</b><br>
  <i>Inson AI raqibga qarshi o'ynaydi. AI = o'qitilgan model (best.onnx) + MCTS.</i>
</p>

<p align="center">
  <img alt="tauri" src="https://img.shields.io/badge/Tauri-2-FFC131">
  <img alt="rust" src="https://img.shields.io/badge/Rust-backend-DEA584?logo=rust&logoColor=black">
  <img alt="onnx" src="https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white">
  <img alt="license" src="https://img.shields.io/badge/License-MIT-lightgrey">
</p>

---

## Loyiha haqida

Bu — Tauri + Rust + ONNX asosida qurilgan tezkor ish stoli ilovasi. Inson
o'qitilgan AI modelга qarshi shashka o'ynaydi.

| Qism | Tafsilot |
|---|---|
| **Backend** | Rust — o'yin dvigateli + MCTS + ONNX (`ort` crate). CPU uchun optimal, tez |
| **Frontend** | HTML / CSS / JS — chiroyli yog'och taxta, silliq animatsiya |
| **Hajmi** | Kichik (~10–15 MB), tez ishlaydi |

Asosiy g'oya: ilova tayyor, siz faqat o'z modelingizni (`best.onnx`) joylaysiz.

---

## 1. Modelni joylash

Yagona kerakli qadam. Trening tugagach, `best.onnx` faylini quyidagi papkaga
ko'chiring:

```
src-tauri/models/best.onnx
```

Agar model ikki fayldan iborat bo'lsa (external data), `best.onnx.data` ni
ham shu papkaga qo'ying. Ilova ishga tushganda modelni avtomatik yuklaydi.

> **Maslahat:** modelni bitta faylga eksport qilish `.data` muammosining
> oldini oladi:
> ```python
> import onnx
> m = onnx.load("best.onnx", load_external_data=True)
> onnx.save_model(m, "best.onnx", save_as_external_data=False)
> ```

---

## 2. Talablar (bir marta o'rnatiladi)

- **Rust:** https://rustup.rs
- **Tauri tizim kutubxonalari:** https://tauri.app/start/prerequisites/
  - **Windows:** WebView2 (odatda o'rnatilgan) + Visual Studio Build Tools
  - **macOS:** Xcode Command Line Tools
  - **Linux:** `webkit2gtk`, `libayatana-appindicator` va h.k.
- **Tauri CLI:**

```bash
cargo install tauri-cli --version "^2"
```

> `ort` (ONNX Runtime) kutubxonasi `download-binaries` xususiyati bilan
> avtomatik yuklanadi — alohida o'rnatish shart emas.

---

## 3. Ishga tushirish (dev rejimi)

Loyiha ildizidan:

```bash
cargo tauri dev
```

Birinchi build uzoqroq davom etadi (Rust va ONNX Runtime yuklanadi),
keyingilari tez bo'ladi.

---

## 4. Tayyor ilova qurish (release)

```bash
cargo tauri build
```

Natijada quyidagilar yaratiladi:

| Platforma | O'rnatkich |
|---|---|
| **Windows** | `.msi` va `.exe` (NSIS) |
| **macOS** | `.dmg` |
| **Linux** | `.deb`, `.rpm`, `.AppImage` |

O'rnatkichlar `src-tauri/target/release/bundle/` papkasida bo'ladi. Release
build maksimal optimallashtirilgan (LTO, opt-level 3, strip).

> Faqat bitta format kerak bo'lsa, masalan `.deb`:
> ```bash
> cargo tauri build --bundles deb
> ```

> **Diqqat:** har bir format o'sha operatsion tizimda quriladi (`.exe` →
> Windows, `.deb` → Linux). Bir nechta platforma uchun GitHub Actions
> ishlatish qulay.

---

## 5. O'ynash

- **Oq** yoki **Qora** tugmasini bosib yangi o'yin boshlang
- Toshni bosing → mumkin yurishlar tilla nuqta bilan ko'rsatiladi → manzilni bosing
- Zanjirli urishda davom etish majburiy

**Daraja tanlash:**

| Daraja | Sims | Tavsif |
|---|---|---|
| **Oddiy** | 0 | MCTS'siz, xom model tanlovi — oniy, eng tez |
| **Tez** | 250 | MCTS bilan |
| **O'rta** | 500 | MCTS bilan, kuchliroq |
| **Kuchli** | 700 | MCTS bilan, eng kuchli |

Ko'proq sims = kuchliroq o'yin, lekin sekinroq.

**Ko'rsatkichlar:** *Baho* (pozitsiya, sizning foydangizga), *Ishonch* (AI
tanlovining aniqligi), *Sims* (bajarilgan simulyatsiyalar soni).

---

## Loyiha tuzilishi

```
shashka_desktop/
├─ src/                    # frontend (statik, bundler kerak emas)
│  ├─ index.html
│  ├─ styles.css
│  └─ main.js
└─ src-tauri/
   ├─ src/
   │  ├─ engine.rs         # o'yin qoidalari (trening bilan bir xil, fuzzing-sinangan)
   │  ├─ ai.rs             # MCTS + ONNX baholash (ort)
   │  ├─ lib.rs            # Tauri buyruqlari + holat
   │  └─ main.rs           # kirish nuqtasi
   ├─ models/
   │  └─ best.onnx         # ← SIZ shu yerga modelni qo'yasiz
   ├─ icons/
   ├─ capabilities/
   ├─ Cargo.toml
   ├─ build.rs
   └─ tauri.conf.json
```

---

## Tezlik

- MCTS Rust'da, ko'p yadroli (`with_intra_threads`) — UI muzlamasligi uchun
  hisob fon oqimda bajariladi
- ONNX Runtime CPU'da tez ishlaydi
- Har yurish vaqti darajaga (sims) bog'liq — taxminiy:

| Daraja | Taxminiy vaqt |
|---|---|
| Oddiy (0) | oniy (~millisekund) |
| Tez (250) | ~0.3–1 s |
| O'rta (500) | ~0.6–1.5 s |
| Kuchli (700) | ~1–2 s |

Aniq tezlik protsessoringizga bog'liq.

---

## Texnik eslatma

`engine.rs` trening kodidagi qoidalar bilan **aynan bir xil** (bitboard,
kodlash, kanonik harakat). Model ONNX interfeysi:

```
Kirish:    x [194]
Chiqishlar: policy_logits [1024],  wdl_logits [3]
```

Bu interfeys trening bilan mos — shu sabab model to'g'ri ishlaydi.
Modelni almashtirishingiz mumkin (har qanday qatlam soni yoki parametr):
ichki arxitektura ONNX faylda saqlanadi, ilova kodi o'zgarmaydi. Faqat
kirish/chiqish shakli (194 / 1024 / 3) bir xil qolishi shart.

> `ort` versiyasi `Cargo.toml` da belgilangan. Boshqa versiya ishlatsangiz,
> `ai.rs` dagi sessiya API'si (`run`, `try_extract_tensor`) ozgina farq
> qilishi mumkin.

---

## Litsenziya

MIT — ta'lim va shaxsiy foydalanish uchun erkin.
