# PRD — AI Capability Checker

**Version:** 1.0  
**Status:** Draft  
**Target Platform:** Windows & Linux  
**Deployment:** Portable USB / Flash Drive  
**Primary Benchmark:** 4B LLM  
**Benchmark Engine:** llama.cpp  
**Database:** SQLite

---

## 1. Product Overview

**AI Capability Checker** adalah aplikasi portable untuk menguji kemampuan sebuah PC dalam menjalankan Local LLM/AI.

User cukup:

```text
Colok Flash Drive
       ↓
Jalankan AI Checker
       ↓
Detect Hardware
       ↓
Benchmark 4B LLM
       ↓
Analisis hasil terhadap Benchmark Dataset
       ↓
Recommendation
```

Tidak membutuhkan instalasi aplikasi, Python, package manager, atau dependency tambahan pada PC target.

Output utama:

```text
AI COMPUTE
~20 TOPS

LLM BENCHMARK
4B Q4_K_M
38.7 tok/s

RECOMMENDATION
4B     : Excellent
8B     : Good
12B    : Possible
14B+   : Not Recommended

Confidence: High
Based on: 127 comparable benchmarks
```

---

## 2. Problem

Informasi seperti:

> "Laptop ini 20 TOPS, berarti bisa menjalankan LLM 8B."

tidak cukup akurat.

TOPS hanya menggambarkan kemampuan komputasi AI secara teoritis. Performa LLM juga dipengaruhi oleh:

- GPU architecture
- CPU
- Memory bandwidth
- RAM
- VRAM
- GPU/NPU
- Backend
- Driver
- Quantization
- LLM architecture
- Context size
- Runtime

Karena itu aplikasi harus melakukan **actual LLM benchmark**, bukan hanya hardware detection.

---

## 3. Goals

### Primary Goals

1. Portable tanpa instalasi.
2. Bisa berjalan langsung dari USB.
3. Support Windows dan Linux.
4. Mendeteksi hardware AI yang tersedia.
5. Mengukur TOPS/AI compute jika datanya tersedia.
6. Menjalankan benchmark LLM nyata.
7. Menggunakan standardized 4B model.
8. Menghasilkan actual tokens/sec.
9. Membandingkan hasil dengan benchmark dataset.
10. Memberikan rekomendasi ukuran LLM berdasarkan data benchmark.
11. Bisa bekerja offline.
12. Hasil benchmark dapat disimpan/export.

---

## 4. Non-Goals

Versi pertama **tidak bertujuan**:

- Training LLM.
- Fine-tuning.
- Menjadi chatbot.
- Menginstall GPU driver.
- Menginstall CUDA/ROCm/Vulkan.
- Benchmark gaming.
- Benchmark CPU umum.
- Benchmark GPU gaming.
- Menentukan performa LLM hanya berdasarkan TOPS.

---

## 5. Target User

### IT / Technician

Untuk mengecek kemampuan PC sebelum deployment AI.

### Developer

Untuk mengetahui apakah sebuah PC cocok menjalankan Local LLM.

### AI Infrastructure Team

Untuk inventory dan klasifikasi hardware.

### End User

Untuk mengetahui:

> "PC saya cocok menjalankan LLM berapa B?"

---

## 6. Deployment Model

Aplikasi berada seluruhnya di USB.

```text
AI-CHECKER/
│
├── windows/
│   └── ai-checker.exe
│
├── linux/
│   └── ai-checker
│
├── engine/
│   └── llama.cpp/
│
├── models/
│   └── benchmark-4b-q4_k_m.gguf
│
├── dataset/
│   └── benchmark.db
│
├── config/
│   └── benchmark.json
│
└── results/
```

User tidak perlu:

```text
pip install
npm install
apt install
docker
registry
```

---

## 7. Runtime Strategy

Model dapat dijalankan langsung dari USB.

```text
USB
 │
 ├── Application
 ├── llama.cpp
 ├── Dataset
 └── 4B Model
          │
          ▼
       Target PC
          │
          ▼
      GPU/CPU/NPU
```

### Optional Cache

Untuk PC yang ingin benchmark berulang:

```text
USB
 ↓
Check model cache
 ↓
Already exists?
 ├── YES → gunakan cache
 └── NO  → copy model ke TEMP
```

Cache dapat menggunakan:

```text
model hash
+
model version
```

sehingga model tidak perlu dicopy ulang.

---

## 8. Hardware Detection

Aplikasi harus mendeteksi:

### CPU

- Vendor
- Model
- Core
- Thread
- Frequency
- Architecture

### GPU

- Vendor
- Model
- VRAM
- Driver
- Compute capability jika tersedia

### NPU

Jika tersedia:

- Vendor
- Model
- AI accelerator
- TOPS jika exposed oleh OS/API

### Memory

- Total RAM
- Available RAM

### Storage

- USB type jika dapat dideteksi
- Available space
- Temporary directory

---

## 9. AI Compute / TOPS

Aplikasi menampilkan:

```text
AI COMPUTE

CPU AI Compute:  X TOPS
GPU AI Compute:  X TOPS
NPU AI Compute:  X TOPS

Available AI Compute:
X TOPS
```

Namun TOPS **tidak digunakan sebagai direct predictor tokens/sec**.

Contoh:

```text
20 TOPS
```

tidak otomatis berarti:

```text
8B = 30 tok/s
```

TOPS hanya menjadi **hardware capability information**.

Jika TOPS tidak dapat diperoleh secara reliable dari hardware/OS, aplikasi harus menampilkan:

```text
AI Compute: Unknown
```

dan tidak membuat angka TOPS palsu.

---

## 10. LLM Benchmark

### Standard Model

Versi pertama menggunakan:

```text
Parameter: ~4B
Format: GGUF
Quantization: Q4_K_M
```

Model harus dipilih berdasarkan:

- License yang memungkinkan redistribution.
- Stabil.
- Banyak benchmark reference.
- Cocok dengan llama.cpp.
- Representative untuk local LLM.

Model identity/version harus disimpan di configuration.

---

## 11. Standard Workload

Semua PC harus menggunakan workload yang sama.

Contoh:

```text
Model:
4B Q4_K_M

Context:
4096 tokens

Prompt:
Fixed benchmark prompt

Generation:
128 / 256 tokens

Runs:
3–5

Warmup:
1 run
```

Primary metric:

```text
Generation tokens/sec
```

Secondary metrics:

```text
Prompt tokens/sec
TTFT
Total latency
Memory usage
Load time
```

### Important

Waktu membaca model dari USB **tidak boleh dianggap sebagai LLM performance utama**.

Model load time harus dicatat sebagai metric terpisah.

---

## 12. Benchmark Result

Contoh:

```text
MODEL
4B Q4_K_M

GENERATION
38.72 tok/s

PROMPT PROCESSING
112.45 tok/s

TTFT
0.41 sec

RAM
8.2 GB

VRAM
4.1 GB

BACKEND
CUDA

ENGINE
llama.cpp
```

Benchmark harus menggunakan beberapa run.

Contoh:

```text
Run 1: 38.21 tok/s
Run 2: 39.01 tok/s
Run 3: 38.94 tok/s

Average:
38.72 tok/s

Std deviation:
0.42
```

---

## 13. Benchmark Dataset

Dataset merupakan komponen penting sistem.

Dataset berisi **actual measured benchmark**, bukan theoretical estimates.

Contoh record:

```json
{
  "hardware": {
    "vendor": "AMD",
    "model": "Radeon 780M",
    "ram_gb": 32,
    "vram_gb": 0
  },
  "model": {
    "name": "Example 4B",
    "parameters_b": 4,
    "quantization": "Q4_K_M"
  },
  "runtime": {
    "engine": "llama.cpp",
    "backend": "Vulkan"
  },
  "workload": {
    "context": 4096,
    "prompt_tokens": 512,
    "generation_tokens": 128
  },
  "result": {
    "generation_tps": 38.7,
    "prompt_tps": 112.4
  },
  "source": {
    "type": "measured",
    "url": "...",
    "date": "..."
  }
}
```

---

## 14. Dataset Sources

Dataset dapat menggabungkan beberapa sumber benchmark seperti:

- Token Atlas
- Local LLM Advisor
- Local AI Hardware Dataset
- Internal benchmark results
- llama.cpp benchmark results

Setiap record harus memiliki:

```text
source
date
hardware
model
quantization
backend
runtime
workload
result
```

Dataset **tidak boleh mencampurkan estimated benchmark dengan measured benchmark tanpa penanda yang jelas.**

---

## 15. Recommendation Engine

Ini merupakan bagian inti aplikasi.

Input:

```text
Hardware
+
4B actual benchmark
+
TOPS
+
RAM
+
VRAM
+
Backend
+
Benchmark Dataset
```

Output:

```text
Recommended Model Size
```

Contoh:

```text
4B  → Excellent
7B  → Excellent
8B  → Good
12B → Acceptable
14B → Poor
```

---

## 16. Recommendation Method

Tidak menggunakan rule sederhana seperti:

```text
70 tok/s = 12B
```

Sebaliknya:

```text
Actual benchmark
       ↓
Find comparable benchmark records
       ↓
Normalize benchmark conditions
       ↓
Find hardware with similar performance
       ↓
Analyze larger-model performance
       ↓
Generate recommendation
```

### Comparable Factors

Prioritas:

1. Same model family.
2. Same quantization.
3. Same backend.
4. Similar hardware.
5. Similar RAM/VRAM.
6. Similar context.
7. Similar benchmark workload.

Recommendation harus mempertimbangkan bahwa 4B tok/s tidak dapat dipetakan secara linear ke semua model yang lebih besar.

---

## 17. Confidence Score

Recommendation harus memiliki confidence.

```text
HIGH
MEDIUM
LOW
```

Contoh:

```text
Recommended: 4B–8B

Confidence: HIGH

Comparable benchmarks:
127

Same backend:
92

Same quantization:
87
```

Jika dataset kurang:

```text
Recommended: 4B–8B

Confidence: LOW

Comparable benchmarks:
3
```

---

## 18. Example Final Result

```text
╔════════════════════════════════════════╗
║        AI CAPABILITY CHECKER          ║
╠════════════════════════════════════════╣
║ CPU                                    ║
║ Ryzen 7 7840HS                         ║
║                                        ║
║ GPU                                    ║
║ Radeon 780M                            ║
║                                        ║
║ RAM                                    ║
║ 32 GB                                  ║
╠════════════════════════════════════════╣
║ AI COMPUTE                             ║
║ ~20 TOPS                               ║
╠════════════════════════════════════════╣
║ LLM BENCHMARK                          ║
║                                        ║
║ Model       4B Q4_K_M                  ║
║ Generation  38.7 tok/s                 ║
║ Prompt      112.4 tok/s                ║
║ Backend     Vulkan                     ║
╠════════════════════════════════════════╣
║ RECOMMENDATION                         ║
║                                        ║
║ 4B       Excellent                     ║
║ 7B       Excellent                     ║
║ 8B       Good                          ║
║ 12B      Acceptable                    ║
║ 14B+     Not Recommended               ║
║                                        ║
║ Confidence: HIGH                       ║
║ Based on 127 benchmarks                ║
╚════════════════════════════════════════╝
```

---

## 19. UI

Versi pertama tidak perlu UI kompleks.

### Main Screen

```text
AI Capability Checker

[ Detect Hardware ]

CPU
GPU
NPU
RAM
VRAM

AI Compute
XX TOPS

Benchmark

Model: 4B Q4_K_M

[ Start Benchmark ]

Progress:
████████████████░░░░ 80%

Result:
38.7 tok/s

[ View Recommendation ]
[ Export Report ]
```

---

## 20. CLI

Aplikasi juga harus menyediakan CLI.

```bash
ai-checker detect
```

```bash
ai-checker benchmark
```

```bash
ai-checker benchmark --model 4b
```

```bash
ai-checker report
```

```bash
ai-checker export --format json
```

Contoh:

```bash
ai-checker benchmark
```

Output:

```text
Detecting hardware...
GPU: NVIDIA RTX 3050
VRAM: 6 GB
RAM: 16 GB
Backend: CUDA

Loading benchmark model...
Running warmup...

Run 1: 31.2 tok/s
Run 2: 32.1 tok/s
Run 3: 31.8 tok/s

Average: 31.7 tok/s

Analyzing benchmark dataset...

Recommendation:
4B - Excellent
8B - Good
12B - Possible

Confidence: HIGH
```

---

## 21. Export

Supported:

```text
JSON
CSV
HTML
TXT
```

Example JSON:

```json
{
  "hardware": {},
  "ai_compute": {},
  "benchmark": {},
  "recommendation": {},
  "confidence": "high"
}
```

HTML report harus dapat dibuka tanpa internet.

---

## 22. Offline Mode

Core functionality harus bekerja tanpa internet.

USB membawa:

```text
Application
+
Model
+
Benchmark Database
+
Configuration
```

Internet hanya diperlukan untuk:

```text
Optional Dataset Update
Optional Software Update
```

---

## 23. Dataset Update

Optional:

```text
[ Update Benchmark Database ]
```

Flow:

```text
Check internet
      ↓
Download latest dataset
      ↓
Validate checksum
      ↓
Replace/update SQLite
      ↓
Done
```

Version:

```text
Dataset v1.0.0
Dataset v1.1.0
Dataset v1.2.0
```

---

## 24. Security

Application harus:

- Tidak membutuhkan Administrator/root untuk benchmark normal.
- Tidak mengubah system configuration.
- Tidak install driver.
- Tidak modify registry.
- Tidak install services.
- Tidak melakukan persistence.
- Tidak mengirim hardware information tanpa explicit permission.

---

## 25. Error Handling

### GPU backend tidak tersedia

```text
CUDA backend unavailable.

Reason:
NVIDIA driver not detected.

Fallback:
CPU benchmark
```

### RAM tidak cukup

```text
Unable to load 4B model.

Required memory: ~X GB
Available: ~X GB
```

### Backend failure

```text
GPU benchmark failed.

Attempting CPU fallback...
```

### Unsupported platform

```text
Platform not supported.

Supported:
Windows x64
Linux x64
```

---

## 26. Performance Requirements

Application startup:

```text
< 3 seconds
```

Hardware detection:

```text
< 5 seconds
```

Benchmark:

Tidak ada fixed maximum karena tergantung hardware.

Target:

```text
Typical PC:
< 2 minutes
```

Model loading time harus ditampilkan terpisah dari benchmark performance.

---

## 27. Technology Stack

### Recommended

```text
Language:
Rust

LLM Runtime:
llama.cpp

Database:
SQLite

UI:
Lightweight native UI / local HTML UI

Configuration:
JSON/TOML

Packaging:
Native executable
```

### Rationale

Rust dipilih karena:

- Single native binary.
- Tidak membutuhkan runtime tambahan.
- Memory efficient.
- Cocok untuk USB application.
- Support Windows + Linux.
- Mudah mengintegrasikan atau menjalankan benchmark engine.

---

## 28. Project Structure

```text
ai-capability-checker/
│
├── src/
│   ├── hardware/
│   │   ├── cpu
│   │   ├── gpu
│   │   ├── npu
│   │   └── memory
│   │
│   ├── benchmark/
│   │   ├── runner
│   │   ├── parser
│   │   └── metrics
│   │
│   ├── recommendation/
│   │   ├── matcher
│   │   ├── analyzer
│   │   └── confidence
│   │
│   ├── dataset/
│   │   ├── loader
│   │   └── updater
│   │
│   ├── report/
│   │   ├── json
│   │   ├── html
│   │   └── text
│   │
│   └── main.rs
│
├── engine/
│   └── llama.cpp
│
├── models/
│   └── benchmark-4b-q4_k_m.gguf
│
├── dataset/
│   └── benchmark.db
│
├── config/
│   └── benchmark.json
│
└── results/
```

---

## 29. MVP

### Phase 1 — Hardware Detection

- CPU
- GPU
- RAM
- VRAM
- Backend
- NPU jika tersedia

### Phase 2 — LLM Benchmark

- llama.cpp
- standardized 4B GGUF
- fixed workload
- multiple runs
- tok/s

### Phase 3 — Dataset

- SQLite
- benchmark records
- provenance
- normalization

### Phase 4 — Recommendation

```text
Actual benchmark
+
Dataset
→
Model recommendation
```

### Phase 5 — Portable Packaging

```text
Windows
Linux
USB
```

### Phase 6 — Reporting

- CLI
- JSON
- HTML
- TXT

---

## 30. Success Criteria

- [ ] Bisa dijalankan langsung dari USB.
- [ ] Tidak membutuhkan instalasi.
- [ ] Windows dapat menjalankan benchmark.
- [ ] Linux dapat menjalankan benchmark.
- [ ] Hardware berhasil dideteksi.
- [ ] GPU/NPU backend berhasil digunakan jika tersedia.
- [ ] 4B LLM benar-benar dijalankan.
- [ ] Menghasilkan actual tok/s.
- [ ] Benchmark repeatable.
- [ ] Dataset berisi measured benchmarks.
- [ ] Recommendation tidak berdasarkan hardcoded TOPS threshold.
- [ ] Recommendation memiliki confidence.
- [ ] Bisa bekerja offline.
- [ ] Bisa menghasilkan report.

---

## 31. Core Product Principle

> **TOPS tells us what the hardware claims it can compute.**  
> **LLM benchmark tells us what the PC actually does.**  
> **Benchmark dataset tells us what that result means.**

Produk ini bukan sekadar **TOPS Checker**, tetapi **AI Capability Checker**:

```text
DETECT
  ↓
ACTUAL BENCHMARK
  ↓
COMPARE
  ↓
ANALYZE
  ↓
RECOMMEND
```

Tujuan akhirnya adalah memberikan jawaban praktis:

> **"PC ini secara nyata mampu menjalankan Local LLM sampai ukuran/model apa dengan performa yang masuk akal?"**
