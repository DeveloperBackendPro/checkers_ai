<h1 align="center">Checkers with AI</h1>

<p align="center">
  <b>Rus shashkasi (Шашки) uchun self-play sun'iy intellekt</b><br>
  <i>Noldan, faqat o'zi bilan o'ynab o'rganadi — inson bilimisiz.</i>
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white">
  <img alt="rust" src="https://img.shields.io/badge/Rust-engine-DEA584?logo=rust&logoColor=black">
  <img alt="pytorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="onnx" src="https://img.shields.io/badge/ONNX-export-005CED?logo=onnx&logoColor=white">
  <img alt="license" src="https://img.shields.io/badge/License-MIT-lightgrey">
</p>

<p align="center">
  AlphaZero uslubi: <b>Transformer</b> neyron tarmoq + <b>Monte-Carlo daraxt qidiruvi (MCTS)</b>.<br>
  Tayyor model, inson partiyalari yoki debyut kitobi <b>ishlatilmaydi</b> — model barcha strategiyalarni o'zi kashf qiladi.
</p>

---

## 📚 Loyiha haqida

Bu loyiha **ta'lim va tadqiqot maqsadida** yaratilgan. U AlphaZero g'oyasini
— sof self-play orqali o'rganishni — kichik, tushunarli va to'liq ochiq
kodli misolda ko'rsatadi.
Quyidagilarni o'rganmoqchi bo'lganlar uchun foydali:

- AlphaZero arxitekturasi qanday ishlaydi — tarmoq, MCTS va arena gating birgalikda
- Transformer'ni o'yin pozitsiyalarini baholash uchun qo'llash
- Rust va Python'ni birga ishlatish (tezlik uchun) hamda differential testing bilan tekshirish
- Modelni ONNX'ga eksport qilib, mustaqil ilovada ishlatish

Bu tijoriy mahsulot emas — o'rganish uchun ochiq qo'llanma.

---

## ⚙️ Qanday ishlaydi

AlphaZero sikli uch bosqichdan iborat va doimiy takrorlanadi:

1. **Self-play** — model MCTS yordamida o'zi bilan ko'plab o'yin o'ynaydi.
   Har bir pozitsiya, tanlangan yurish va o'yin natijasi xotiraga (replay
   buffer) yoziladi.
2. **Trening** — model shu xotiradagi pozitsiyalardan o'rganadi: qaysi
   yurish yaxshi (policy) va pozitsiya kim foydasiga (value).
3. **Arena** — yangi model eski eng yaxshi model bilan teng sharoitda
   o'ynaydi. Faqat haqiqatan kuchliroq bo'lsagina yangi "eng yaxshi" bo'ladi
   va ELO oshadi.

Shu sikl orqali model bosqichma-bosqich, inson yordamisiz kuchayadi.

---

## 📐 Arxitektura

| Komponent | Tafsilot |
|---|---|
| **Tarmoq** | Transformer: 32 katak = 32 token + CLS · RoPE · Multi-Head Attention · Pre-LN · GELU · residual |
| **Policy head** | Bilinear (from × to) → 1024 ta harakat logit'i |
| **Value head** | WDL (Win / Draw / Loss, 3 sinf) — durangga moyil shashka uchun aniq kalibrlangan |
| **Qidiruv** | PUCT MCTS: Dirichlet shovqin · temperatura jadvali · daraxtni qayta ishlatish · dinamik byudjet |
| **Dynamics** | Aniq o'yin dvigateli (qoidalar to'liq ma'lum — MuZero'ning o'rganilgan dynamics'i ataylab ishlatilmadi) |
| **Self-play** | Parallel lockstep-batch (GPU samaradorligi) · curriculum (endshpildan boshlash) · debyut diversifikatsiyasi |
| **Gating** | Yangi model "best"ni belgilangan foizdan ko'p yengsagina qabul qilinadi; ELO yuritiladi |
| **Xavfsizlik** | GPU/CPU termal pauza · atomic checkpoint/ONNX yozish (uzilishda buzilmaydi) · Rust action-range himoyasi |
| **Tezlik** | Rust dvigatel + Rust MCTS (PyO3): self-play ~60×, arena ~20–40× tez. Python etalon bo'lib qoladi |
| **Eksport** | ONNX (dinamik batch) — istalgan dasturlash tilida ishlatish mumkin |

---

## 📦 Loyiha tuzilishi

| Fayl | Vazifa |
|---|---|
| `checkers_engine.py` | Rus shashkasi qoidalari (urish, zanjir, damka, durang) |
| `network.py` | Transformer + WDL head, ONNX eksport |
| `mcts.py` | PUCT MCTS qidiruvi |
| `selfplay.py` | Parallel self-play, curriculum, replay buffer |
| `inference.py` | PyTorch / ONNX yagona evaluator |
| `safety.py` | Termal qo'riqchi (avtomatik pauza) |
| `train.py` | Trening (GPU) |
| `play.py` | Modelni kuzatish — o'zi bilan o'ynaydi |
| `game.py` | Model bilan o'ynash (GUI) |
| `analyze.py` | Attention xaritalari va baho grafigi (tadqiqot) |
| `selfplay_rust.py` · `arena_rust.py` | Rust forest asosidagi tezkor self-play va arena |
| `rust/` | Rust dvigatel + MCTS (PyO3) va tayyor wheel |
| `test_engine.py` | Qoidalar testlari (18 ta) + invariant fuzzing |
| `test_diff.py` | Rust vs Python: bit-darajada moslik testi |
| `test_pipeline.py` · `bench.py` | MCTS/self-play testlari va tezlik benchmarki |

---

## 🔧 O'rnatish

Kutubxonalar va Rust dvigatelni o'rnatish:

```bash
pip install -r requirements.txt
pip install rust/dist/shashka_engine-*.whl
```

Testlar bilan tekshirish:

```bash
python test_engine.py      # qoidalar testlari (18/18 o'tishi kerak)
python test_pipeline.py    # MCTS/self-play testlari (5/5)
python test_diff.py        # Rust vs Python: bit-darajada moslik
python bench.py            # tezlik benchmarki (~60x)
```

Tayyor wheel Colab / Linux x86_64 da ishlaydi (Python 3.9+). Wheel
o'rnatilmasa ham hammasi ishlaydi — `train.py` avtomatik Python dvigatelga
qaytadi (sekinroq). Boshqa platformada qurish:

```bash
pip install maturin
cd rust && maturin build --release -o dist && pip install dist/*.whl
```
---

## 🎮 O'ynash va kuzatish

Modelni `.onnx` yoki `.pt` (PyTorch checkpoint) bilan ishlatish mumkin.

**Modelni kuzatish** — model o'zi bilan o'ynaydi (GUI):

```bash
python play.py --ckpt checkpoints/best.pt --sims 200      # PyTorch checkpoint
python play.py --onnx checkpoints/best.onnx --sims 200    # ONNX
```

**Terminalda kuzatish** — GUI'siz, Colab yoki server uchun:

```bash
python play.py --ckpt checkpoints/best.pt --ascii --games 3
python play.py --onnx checkpoints/best.onnx --ascii --games 3
```

GUI boshqaruvi: `SPACE` — pauza, `RIGHT` — bitta yurish, `+` / `-` — tezlik,
`N` — yangi o'yin, `Q` — chiqish. Eval-bar oq nuqtai nazaridan bahoni
ko'rsatadi.

**O'zingiz model bilan o'ynash** (GUI):

```bash
python game.py --ckpt checkpoints/best.pt --color white --sims 400         # oq bilan
python game.py --onnx checkpoints/best.onnx --color black --sims 400       # qora bilan
python game.py --onnx checkpoints/best.onnx --color white --time-limit 3   # vaqt chegarasi (sekund)
```

Sichqoncha bilan yuriladi; legal kataklar yashil nuqta bilan ko'rsatiladi.

**Qoidalarni modelsiz ko'rish** — random rejim (yurishlar kuchsiz, lekin
barcha qoidalar to'liq amal qiladi; torch ham kerak emas):

```bash
python play.py --random --sims 100         # GUI: o'zi bilan o'ynaydi
python play.py --random --ascii --games 3  # terminalda
python game.py --random --color white      # siz qoidalarga qarshi o'ynaysiz
```

Urish majburiy; zanjirli urishda davom etish avtomatik talab qilinadi.
`--sims` qidiruv chuqurligini (ko'proq = kuchliroq, lekin sekinroq),
`--time-limit` esa yurish vaqti chegarasini (sekund) belgilaydi.

---

## 📊 Tahlil

Model nimani o'rganganini ko'rish (tadqiqot uchun):

```bash
python analyze.py --ckpt checkpoints/best.pt --mode attention --plies 20
python analyze.py --ckpt checkpoints/best.pt --mode values --sims 200
```

`analysis/` papkasiga har qatlam attention xaritasi (CLS qaysi kataklarga
"qaraydi") va o'yin davomidagi baho grafigi saqlanadi.

---

## 📕 ONNX spetsifikatsiyasi

Model ONNX'ga eksport qilinadi va istalgan tilda (Rust, C++, JS...)
ishlatilishi mumkin.

**Kirish** — `x`: `float32 [B, 194]`

| Bo'lim | Mazmun |
|---|---|
| `x[0:192]` | 32 katak × 6 belgi: `[mening shashkam, mening damkam, raqib shashkasi, raqib damkasi, zanjir shashkasi, urilgan belgisi]` |
| `x[192]` | halfmove / 60 |
| `x[193]` | zanjir bormi (0 / 1) |

Pozitsiya **kanonik**: navbatdagi o'yinchi doim "oq". Qora yursa doska
180° buriladi (`sq → 31 - sq`).

**Chiqishlar**

| Chiqish | Shakl | Izoh |
|---|---|---|
| `policy_logits` | `[B, 1024]` | harakat = `from × 32 + to` (kanonik koordinatada) |
| `wdl_logits` | `[B, 3]` | `value = softmax(wdl)[0] − softmax(wdl)[2]` |

MCTS'ni boshqa tilda yozsangiz, `mcts.py` dagi PUCT mantig'ini takrorlang.
Model fayli (ONNX) o'zgarmaydi.

---

## 📌 Qoidalar

To'liq Rus shashkasi qoidalari qo'llab-quvvatlanadi (`test_engine.py` da
isbotlangan):

- Majburiy urish · zanjirli urish · oddiy shashka 4 yo'nalishda uradi
- Damka (flying king) va tushish-cheklovi (davom bor katakka tushish majburiy)
- Turk zarbasi — urilganlar zanjir tugaguncha doskada qoladi, qayta urilmaydi
- Zanjir ichida damkaga aylanish va damka sifatida davom etish
- 3-takror durang · 30-yurish qoidasi · yura olmagan yutqazadi

---

## 🧮 To'g'rilik kafolati

`test_diff.py` Rust va Python dvigatellarini minglab tasodifiy o'yinlarda
har yurishda solishtiradi: legal yurishlar, doska holati, g'olib va
NN-kodlash bit-darajada bir xil bo'lishi tekshiriladi.

Python dvigatel 10 000 o'yinlik invariant-fuzzing'dan o'tgan etalon; Rust
unga matematik bog'langan. Bu — differential testing, professional
loyihalardagi standart yondashuv.

---

## 🔴 Trening Google Colab da ishga tushirish bo‘yicha bosqichma bosqich ishga tushirish
```
!nvidia-smi

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

!mkdir -p /content/drive/MyDrive/shashka_ckpt

from google.colab import files
files.upload()

!unzip -q shashka_ai.zip

%cd shashka_ai

!pip install -r requirements.txt
!pip install rust/dist/shashka_engine-*.whl

!python test_engine.py
!python test_pipeline.py
!python test_diff.py
!python -c "import shashka_engine; print('Rust dvijok OK')"

%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/shashka_ckpt/runs

!python train.py --device cuda --engine rust \
--d-model 192 --n-layers 6 --n-heads 6 --policy-dim 96 \
--iterations 900 --max-hours 15.5 \
--games-per-iter 160 --parallel 160 \
--sims 300 --sims-final 600 \
--buffer-size 700000 --min-buffer 60000 \
--train-steps 150 --batch-size 768 --lr 1.5e-3 \
--arena-every 10 --arena-games 100 --arena-sims 300 --gate 0.53 \
--gpu-max-temp 84 --gpu-resume-temp 70 \
--ckpt-dir /content/drive/MyDrive/shashka_ckpt \
--log-dir /content/drive/MyDrive/shashka_ckpt/runs 
```
<img width="981" height="600" alt="Image" src="https://github.com/user-attachments/assets/747b6b87-79f1-445f-88b1-3dab6e1ac107" />
