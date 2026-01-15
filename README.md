# CYK Parser - Bahasa Bali Prepositional Phrase Validator

> **Aplikasi Validasi Kalimat Bahasa Bali Berpredikat Frasa Preposisi**  
> Implementasi Algoritma Cocke-Younger-Kasami (CYK) untuk parsing dan analisis struktur gramatikal kalimat Bahasa Bali menggunakan Context-Free Grammar dalam Chomsky Normal Form.

---

## Daftar Isi

- [Tentang Aplikasi](#tentang-aplikasi)
- [Fitur Utama](#fitur-utama)
- [Teknologi](#teknologi)
- [Requirements](#requirements)
- [Instalasi](#instalasi)
- [Cara Menjalankan](#cara-menjalankan)
- [Struktur Project](#struktur-project)
- [Dataset & Kamus Kata](#dataset--kamus-kata)
- [Dataset Evaluasi](#dataset--evaluasi)
- [Cara Kerja Algoritma](#cara-kerja-algoritma)
- [Contoh Penggunaan](#contoh-penggunaan)
- [Troubleshooting](#troubleshooting)

---

## Tentang Aplikasi

Aplikasi ini adalah implementasi parser bahasa formal untuk menganalisis struktur kalimat Bahasa Bali, khususnya kalimat yang berpredikat frasa preposisi. Menggunakan **Algoritma CYK (Cocke-Younger-Kasami)**, aplikasi dapat:

- Memvalidasi apakah kalimat sesuai dengan grammar yang ditentukan
- Menganalisis struktur kalimat
- Menampilkan parse tree (pohon penurunan)
- Mengidentifikasi komponen kalimat (Subjek, Predikat, Pelengkap, Keterangan)
- Visualisasi proses parsing dalam triangular table

### Latar Belakang

Proyek ini dikembangkan sebagai bagian dari mata kuliah **Teori Bahasa dan Otomata**, dengan tujuan:
1. Memahami konsep Context-Free Grammar (CFG)
2. Mengimplementasikan Chomsky Normal Form (CNF)
3. Menerapkan algoritma parsing CYK
4. Menganalisis struktur bahasa alami (Bahasa Bali)

---

## Fitur Utama

### 1. **Validasi Kalimat**
- Input kalimat Bahasa Bali
- Validasi otomatis terhadap kamus kata
- Deteksi kata yang tidak dikenali

### 2. **Analisis Gramatikal Lengkap**
- Parse tree visualization (pohon penurunan)
- Pola kalimat (sentence pattern)
- Langkah-langkah derivasi
- Komponen kalimat (frasa preposisional, subjek, pelengkap, keterangan)

### 3. **CYK Triangular Table**
- Visualisasi proses parsing bottom-up
- Tampilan tabel interaktif
- Trace lengkap dari kata ke kalimat

### 4. **User-Friendly Interface**
- Built with Streamlit
- Responsive design
- Error handling yang jelas

---

## Teknologi

| Teknologi | Versi | Kegunaan |
|-----------|-------|----------|
| **Python** | 3.8+ | Bahasa pemrograman utama |
| **Streamlit** | 1.28+ | Web framework untuk UI |
| **Pandas** | 1.5+ | Data manipulation & display |
| **NumPy** | 1.24+ | Numerical operations |

---

## Requirements

### Sistem Requirements

- **OS:** Windows, macOS, atau Linux
- **Python:** Versi 3.8 atau lebih baru
- **RAM:** Minimum 2GB (Disarankan)

### Python Dependencies

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
```

File lengkap ada di `requirements.txt`

---

## Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/juliarta99/cyk-parser-4a.git
cd cyk-parser-bahasa-bali
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Atau install manual:**
```bash
pip install streamlit pandas numpy
```

### 3. Verifikasi Instalasi

```bash
streamlit --version
python --version
```

---

## Cara Menjalankan

### Metode 1: Menggunakan Streamlit

```bash
streamlit run main.py
```

### Metode 2: Menggunakan Python Module

```bash
python -m streamlit run main.py
```

### Akses Aplikasi

Setelah menjalankan perintah di atas, aplikasi akan terbuka otomatis di browser:
```
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Jika tidak terbuka otomatis, buka browser dan akses `http://localhost:8501`

---

## Struktur Project

```
cyk-parser-bahasa-bali/
│
├── 📄 main.py                       # Main application (Streamlit UI)
├── 📄 cyk_process.py                # CYK Algorithm implementation
├── 📄 grammar.py                    # Grammar rules (CNF)
├── 📄 general.py                    # Lexicon loader & validator
├── 📄 evaluation.py                 # Modul evaluasi sistem
│
├── 📂 alphabets/                    # Dataset kamus kata
│   ├── noun.txt                     # Kata benda
│   ├── verb.txt                     # Kata kerja
│   ├── adj.txt                      # Kata sifat
│   ├── prep.txt                     # Kata preposisi
│   ├── pronoun.txt                  # Kata ganti
│   ├── propnoun.txt                 # Nama diri
│   ├── det.txt                      # Determiner
│   ├── num.txt                      # Numeralia
│   ├── adv.txt                      # Adverbia
│   └── nountime.txt                 # Kata benda waktu
│
├── 📂 evaluation_dataset/           # Dataset untuk testing
│   └── evaluation_dataset.txt       # Test cases (LABEL|KALIMAT)
│
├── 📄 evaluation_report.json        # Hasil evaluasi (generated)
├── 📄 requirements.txt              # Python dependencies
├── 📄 README.md                     # Dokumentasi ini
└── 📄 .gitignore                    # Git ignore rules
```

---

## Penjelasan File

### **1. `main.py` - Main Application**

File utama aplikasi Streamlit yang menangani:
- User interface dan input handling
- Workflow parsing (validasi → CYK → analisis)
- Visualisasi hasil (tabel, parse tree, komponen)
- Error handling dan feedback

**Fungsi Utama:**
- Input kalimat dari user
- Tokenisasi dan validasi kata
- Trigger CYK parsing
- Display hasil analisis

**Cara Menjalankan:**
```bash
streamlit run main.py
```

---

### **2. `cyk_process.py` - CYK Algorithm**

Implementasi algoritma Cocke-Younger-Kasami untuk parsing.

**Fungsi Utama:**

| Fungsi | Deskripsi |
|--------|-----------|
| `cyk_parse(words)` | Algoritma CYK utama, return table & backpointer |
| `create_table(n)` | Membuat tabel n×n kosong |
| `get_combinations(set_a, set_b)` | Gabungkan 2 set untuk aturan A→BC |
| `is_valid_sentence()` | Cek apakah kalimat valid |
| `build_parse_tree()` | Rekonstruksi parse tree dari backpointer |
| `get_sentence_pattern()` | Analisis pola kalimat |
| `format_parse_tree()` | Format tree untuk display |

**Cara Kerja:**
1. **Bottom-up parsing**: Mulai dari kata (terminal) → frasa → kalimat
2. **Dynamic programming**: Simpan hasil substring dalam tabel
3. **Try all splits**: Untuk setiap substring, coba semua cara split jadi 2 bagian

---

### **3. `grammar.py` - Grammar Rules**

Definisi tata bahasa dalam bentuk Chomsky Normal Form (CNF).

**Komponen:**

```python
# Non-terminals
variable = ["K", "P", "S", "Pel", "Ket", "NP_S", ...]

# Production rules (CNF)
production = {
    "K": ["PS", "X1Pel", "X2S", ...],  # K → P S, K → X1 Pel, ...
    "P": ["PrepNP_Loc"],                 # P → Prep NP_Loc
    "S": kata_benda + proper_noun + ... # S → 'kata'
}

# Start symbol
start_symbol = ["K"]
```

**Fungsi Utama:**
- `check_production(array)`: Cari non-terminal yang bisa menghasilkan kombinasi
- `check_symbol(array)`: Cek apakah ada start symbol

**CNF Rules:**
1. **A → BC** (branching): Non-terminal → 2 non-terminal
2. **A → terminal** (lexical): Non-terminal → 1 kata

---

### **4. `general.py` - Lexicon & Validation**

Manajemen kamus kata dan validasi input.

**Fungsi Utama:**

| Fungsi | Deskripsi |
|--------|-----------|
| `load_words(filename)` | Baca file txt dari folder alphabets |
| `check_alphabet(input_array)` | Validasi apakah kata ada dalam kamus |

**Kamus Kata:**
```python
kata_benda = ["paon", "jumah", ...]
kata_preposisi = ["ring", "ka", ...]
kata_benda_waktu = ["dibi", "tuni", ...]
# ... dll

alphabet = kata_benda + kata_preposisi + ... # Master dictionary
```

**Auto-load saat import:**
Semua file di folder `alphabets/` otomatis dimuat saat `import general`

---

### **5. `evaluation.py` - Modul Evaluasi**

Sistem evaluasi otomatis untuk testing parser dengan metrics lengkap.

**Fitur:**
- Load dataset dari file txt
- Testing otomatis semua test cases
- Hitung metrics: Accuracy, Precision, Recall, F1 Score
- Generate confusion matrix
- Export hasil ke JSON
- Verbose output (print setiap test case)

**Cara Menjalankan:**
```bash
# Dengan dataset default
python evaluation.py

# Dengan custom dataset
python evaluation.py path/to/dataset.txt
```

**Output:**
- Console: Summary lengkap dengan metrics
- File: `evaluation_report.json`

**Metrics Yang Dihitung:**

| Metric | Formula | Interpretasi |
|--------|---------|--------------|
| **Accuracy** | (TP+TN)/Total × 100% | Persentase prediksi benar |
| **Precision** | TP/(TP+FP) × 100% | Ketepatan prediksi VALID |
| **Recall** | TP/(TP+FN) × 100% | Kelengkapan deteksi VALID |
| **F1 Score** | 2×(P×R)/(P+R) | Balance precision & recall |

---

### **6. `evaluation_dataset/` - Folder Dataset Testing**

Berisi file dataset untuk evaluasi sistem.

**Struktur:**
```
evaluation_dataset/
└── evaluation_dataset.txt    # Test cases
```

**Format File: `evaluation_dataset.txt`**

```
VALID|KALIMAT|KOMPONEN
INVALID|KALIMAT
```

Detail lebih lanjut dapat dilihat pada [Dataset Evaluasi](#dataset--evaluasi)

---

### **7. `evaluation_report.json` - Hasil Evaluasi**

File JSON yang di-generate otomatis saat menjalankan `evaluation.py`.

**Struktur:**
```json
{
  "timestamp": "2026-01-14T...",
  "summary": {
    "total_tests": 90,
    "passed": 85,
    "failed": 5,
    "accuracy": 94.44,
    "precision": 96.30,
    "recall": 92.86,
    "f1_score": 94.55,
    "avg_parse_time": 0.0023
  },
  "confusion_matrix": {
    "true_positive": 52,
    "true_negative": 33,
    "false_positive": 2,
    "false_negative": 3
  },
  "category_stats": {
    "POLA DASAR": {
      "total": 8,
      "passed": 8,
      "failed": 0
    },
    ...
  },
  "test_cases": [
    {
      "sentence": "ring sekolah murid",
      "expected": true,
      "actual": true,
      "correct": true,
      "parse_time": 0.0015,
      "category": "POLA DASAR",
      "pattern": "K → P S",
      "error": null
    },
    ...
  ]
}
```

**Kegunaan:**
- Analisis mendalam hasil evaluasi
- Tracking performa dari waktu ke waktu
- Input untuk visualisasi/grafik
- Dokumentasi untuk laporan

---

**File yang di-ignore:**
- Python cache (`__pycache__/`)
- Virtual environment (`venv/`)
- IDE config (`.vscode/`, `.idea/`)
- OS files (`.DS_Store`)
- Temporary files (`*.swp`, `*.log`)

**File yang di-track:**
- Source code (`.py`)
- Dataset (`alphabets/`, `evaluation_dataset/`)
- Documentation (`README.md`)
- Dependencies (`requirements.txt`)

---

## Dataset & Kamus Kata

### Lokasi: `alphabets/`

Dataset berisi kamus kata Bahasa Bali yang dikategorikan berdasarkan **Part of Speech (POS)**.

### Struktur Dataset

| File | Kategori | Contoh Kata | Jumlah |
|------|----------|-------------|--------|
| **noun.txt** | Kata Benda | paon, jumah, carik | 100+ |
| **verb.txt** | Kata Kerja | malajah, majalan | 50+ |
| **adj.txt** | Kata Sifat | jegeg, gede, cenik | 50+ |
| **prep.txt** | Preposisi | ring, ka | 10+ |
| **pronoun.txt** | Kata Ganti | ia, tiang, ipun | 20+ |
| **propnoun.txt** | Nama Diri | Bali, Jakarta, Made | 30+ |
| **det.txt** | Determiner | ento, ne | 10+ |
| **num.txt** | Numeralia | lelima, dadua, adiri | 20+ |
| **adv.txt** | Adverbia | sesai, cepok | 20+ |
| **nountime.txt** | Kata Waktu | dibi, tuni, semeng | 30+ |

### Format File

Setiap file berisi satu kata per baris:

```txt
paon
jumah
carik
banjar
```

**Catatan:**
- Semua kata dalam **huruf kecil**
- Tidak ada spasi di awal/akhir
- Tidak ada baris kosong
- Encoding: **UTF-8**

### Menambah Kata Baru

1. Buka file yang sesuai (misal: `noun.txt` untuk kata benda)
2. Tambahkan kata baru di baris baru
3. Save dengan encoding UTF-8
4. Restart aplikasi

**Contoh:**
```txt
# alphabets/noun.txt
paon
jumah
carik
banjar
kandang  ← kata baru
```

---

## Dataset Evaluasi

### Format Dataset: `evaluation_dataset.txt`

```
VALID|KALIMAT|KOMPONEN
INVALID|KALIMAT
```

**Contoh:**
```txt
# PP [P S] - Preposisi + Subjek
VALID|Ring sekolah murid|P S
VALID|Di Paon I Meme|P S

# PP [X1 Pel] - (P S) + Pelengkap
VALID|Di carik sampi ne dadua|X1 Pel

# INVALID - Struktur salah
INVALID|murid buku sekolah
```

### 5 Komponen Pola Grammar

| Komponen | Struktur | Contoh Kalimat |
|----------|----------|----------------|
| **P S** | Preposisi + Subjek | Ring sekolah murid |
| **X1 Pel** | (P S) + Pelengkap | Di carik sampi ne dadua |
| **X2 S** | (P Pel) + Subjek | Uli Bandung mobil I Made |
| **X1 X3** | (P S) + (Pel Ket) | Ring pura pragina ne dasa dibi |
| **X2 X4** | (P Pel) + (S Ket) | Di jumah laptop wayan jani |

---

## Cara Kerja Algoritma

### CYK Algorithm - Bottom-Up Dynamic Programming

#### **Konsep Dasar:**

```
BOTTOM-UP: Kata → Frasa → Kalimat

Terminal (kata individual)
    ↓ gabung
Frasa 2 kata
    ↓ gabung
Frasa 3 kata
    ↓ gabung
Kalimat lengkap ✓
```

#### **Langkah Algoritma:**

1. **Inisialisasi:** Buat tabel n×n (n = jumlah kata)

2. **Tahap 1 - Baris Bawah (Length = 1):**
   ```python
   Untuk setiap kata:
       Cari POS tag dari grammar
       Simpan di table[0][i]
   ```

3. **Tahap 2 - Isi Ke Atas (Length 2 sampai n):**
   ```python
   Untuk setiap length:
       Untuk setiap posisi awal:
           Untuk setiap split point:
               LEFT = substring kiri
               RIGHT = substring kanan
               Kombinasi = LEFT + RIGHT
               Cari parent di grammar
               Simpan di table
   ```

4. **Validasi:**
   ```python
   Cek: apakah table[n-1][0] mengandung start symbol "K"?
   ```

---

## Contoh Penggunaan

### Contoh 1: Kalimat Valid

**Input:**
```
ring sekolah murid ento
```

**Proses:**
1. Tokenisasi: `["ring", "sekolah", "murid", "ento"]`
2. Validasi: Semua kata ada dalam kamus
3. CYK Parsing:
   ```
   row 3: [K, X1]
   row 2: [P, PP]      [NP_S]      [...]
   row 1: [Prep]       [Noun, ...]  [Noun, ...]  [Det]
          ring         sekolah      murid        ento
   ```
4. Hasil: **VALID** - Kalimat sesuai grammar

**Parse Tree:**
```
K
├─ P
│  ├─ Prep → 'ring'
│  └─ NP_Loc → 'sekolah'
└─ S
   ├─ NP_S
   │  └─ Noun → 'murid'
   └─ Det → 'ento'
```

**Pola:** K → P S (Frasa Preposisional + Subjek)

---

### Contoh 2: Kalimat Tidak Valid

**Input:**
```
rumah makan enak
```

**Proses:**
1. Tokenisasi: `["rumah", "makan", "enak"]`
2. Validasi: Semua kata ada
3. CYK Parsing:
   ```
   row 2: [  ]  ← Kosong!
   row 1: [...]  [...]
   row 0: [Noun] [Verb] [Adj]
   ```
4. Hasil: **TIDAK VALID** - Tidak sesuai grammar

**Penjelasan:** Struktur "Noun Verb Adj" tidak cocok dengan pola grammar yang didefinisikan.

---

### Contoh 3: Kata Tidak Dikenali

**Input:**
```
ring sekolah xyz
```

**Hasil:**
```
Kata tidak dikenali dalam kamus: xyz
```

**Solusi:** Tambahkan "xyz" ke file yang sesuai di folder `alphabets/`

---

## Troubleshooting

### Problem 1: `ModuleNotFoundError: No module named 'streamlit'`

**Solusi:**
```bash
pip install streamlit
```

---

### Problem 2: `FileNotFoundError: [Errno 2] No such file or directory: 'alphabets/noun.txt'`

**Penyebab:** Folder `alphabets/` tidak ada atau file hilang

**Solusi:**
1. Pastikan folder `alphabets/` ada di root project
2. Pastikan semua file .txt ada di dalamnya
3. Jalankan aplikasi dari root directory:
   ```bash
   cd path/to/cyk-parser-bahasa-bali
   streamlit run dev.py
   ```

---

### Problem 3: Aplikasi tidak membuka di browser

**Solusi:**
1. Cek port 8501 tidak digunakan aplikasi lain
2. Buka manual: `http://localhost:8501`
3. Gunakan port lain:
   ```bash
   streamlit run dev.py --server.port 8502
   ```

---

### Problem 4: Kata valid tapi ditolak

**Penyebab:** Kata ada dalam kamus tapi tidak ada aturan grammar yang sesuai

**Solusi:**
1. Periksa apakah struktur kalimat sesuai pola grammar
2. Grammar hanya menerima pola tertentu (P S, P Pel, dll.)
3. Modifikasi `grammar.py` jika perlu menambah pola baru

---

## Tim Pengembang

**Kelompok 4A - Teori Bahasa dan Otomata**

- [Nayla Lareina]
- [Juliarta]
- [Yudhiastara]
- [Titania]
- [Wahyu Saputra]

---

## Kontak

Untuk pertanyaan, saran, atau bug report:

- **Email:** adij4255@gmail.com
- **GitHub Issues:** [Create Issue](https://github.com/juliarta99/cyk-parser-4a/issues)
- **Discussion:** [GitHub Discussions](https://github.com/juliarta99/cyk-parser-4a/discussions)

**Last Updated:** Januari 2026

---

Made with ❤️ by Kelompok 4A

[Back to Top](#cyk-parser---bahasa-bali-prepositional-phrase-validator)