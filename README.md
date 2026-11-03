# AI Capability Checker

Aplikasi Rust offline Windows x64 / Linux x64: hardware → benchmark nyata Qwen3-4B → pembanding SQLite → report. Tanpa Node/Python di PC target. UI HTML tertanam dalam binary; jalankan aplikasi tanpa argumen lalu buka alamat loopback yang dicetak.

## Persiapan pada mesin pengembangan

Butuh Rust/Cargo, C/C++ compiler, CMake, Git, curl, internet, dan ruang untuk model/build. Linux: `bash scripts/prepare.sh cpu` (atau `cuda` jika SDK tersedia; membangun CPU + CUDA). Windows: `scripts/prepare.ps1 -Package` default CUDA + CPU; `-Backend cpu -Package` untuk CPU-only. Build Windows membutuhkan Visual Studio 2022 C++ Build Tools dengan MSVC v142 (14.29); CUDA juga membutuhkan toolkit 12.6.3, nvcc dan CUDA_PATH. Rust/compiler/toolkit hanya dibutuhkan di mesin build, bukan PC target.

Script membangun llama.cpp tag **b7750**, commit `6e7fc8a146556fe20be36906e1b1b7e03bc36d44`, mengunduh model 2.50 GB ke `.part`, memverifikasi SHA-256, lalu menamai file final. Aplikasi tidak mengunduh apa pun otomatis.

Model resmi: [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF), `Qwen3-4B-Q4_K_M.gguf`, **Apache-2.0**. Revision dan checksum ada di `config/benchmark.json`. Salinan lisensi disertakan oleh script. Jangan mengganti model tanpa mengganti identitas/hash dan memisahkan dataset.

## Pemakaian

- `linux/ai-checker` — UI di `http://127.0.0.1:8787`; browser dibuka manual.
- `linux/ai-checker detect` — hardware JSON.
- `linux/ai-checker benchmark --model 4b` — benchmark; `--cpu` memaksa CPU.
- `linux/ai-checker report` — report terakhir.
- `linux/ai-checker export --format html` — report ke stdout; format JSON/CSV/HTML/TXT.
- `linux/ai-checker dataset-import measured.json` — import atomik, hanya measured records.
- `linux/ai-checker --root /path/AI-CHECKER ui --port 8788` — root/port eksplisit.

Windows memakai `windows/ai-checker.exe` dengan argumen sama. Tidak perlu root/admin. UI hanya listen loopback, memvalidasi Host/Origin, dan membutuhkan header khusus untuk POST. Tidak ada telemetri.

## USB portable

### Launcher Linux

Sertakan `START-LINUX.sh` di folder utama flashdisk. Klik-ganda lalu pilih **Run / Run in Terminal** bila file manager menawarkan pilihan tersebut. Launcher membuka browser lokal pada port 8798 dan menjalankan aplikasi tanpa mengetik argumen. Jika browser terbuka sebelum server siap, refresh halaman. Tutup dengan Ctrl+C pada terminal aplikasi.

Script prepare memberi izin executable pada launcher. Jika izin hilang setelah penyalinan, aktifkan **Properties → Permissions → Allow executing file as program**. Sebagian file manager membuka `.sh` sebagai teks atau tidak mendukung menjalankan script lewat klik-ganda; gunakan menu **Run as a Program** jika tersedia. Pembatasan mount `noexec` tidak dapat dilewati launcher—salin seluruh paket ke penyimpanan lokal atau gunakan mount yang mengizinkan executable. Tidak ada autorun USB atau perubahan konfigurasi keamanan otomatis.

### Paket Windows klik-ganda

Di mesin Windows build, jalankan `scripts/prepare.ps1 -Backend cuda -Package`. Output `dist/AI-CHECKER-Windows-cuda.zip` berisi executable Windows x64, engine terpisah `engine/llama.cpp/windows/cpu/` dan `cuda/`, DLL engine masing-masing, CUDA cuBLAS/cuBLASLt/cudart + app-local MSVC CRT, CUDA EULA, model Qwen3-4B, SQLite baru tanpa report pribadi, JSON Token Atlas, konfigurasi, lisensi, dan `START-WINDOWS.cmd`. `-Backend cpu` membuat ZIP CPU-only tanpa toolkit. SHA-256 ZIP disimpan di file pendamping. Packaging menolak menimpa staging/ZIP lama. ZIP64 dipakai untuk model >2 GB; Windows 10/11 x64 disarankan.

Alternatif tanpa compiler lokal: **Actions → Windows portable USB → Run workflow**, pilih `accelerator=cuda` (default) atau `cpu` eksplisit. Unduh artifact `AI-CHECKER-Windows-cuda` setelah berhasil, ekstrak ZIP ke flashdisk, klik `START-WINDOWS.cmd` (port 8798; refresh jika browser terlalu cepat). Workflow membangun source llama.cpp terpin; toolkit resmi NVIDIA 12.6.3 dipasang di runner melalui [Jimver/cuda-toolkit v0.2.24](https://github.com/Jimver/cuda-toolkit/tree/c35baa1a18fd1fc9dcf47c5bd839bf30559c0bc3), dipin ke commit yang diverifikasi melalui GitHub API. Tidak memakai binary CUDA pihak ketiga tanpa checksum. MSVC v142 dipilih karena CUDA 12.6 tidak mendukung MSVC 19.4x terbaru. Build menargetkan SASS SM60/61/70/75/80/86/89/90 + PTX SM90; GPU generasi lain tetap harus diuji. Review NVIDIA EULA dan izin redistribusi CRT sebelum distribusi.

Saat runtime, `nvidia-smi` adalah probe best-effort. CUDA dipilih hanya jika probe berhasil, engine CUDA dibundel, dan GPU layers bukan nol. Driver/VRAM/arsitektur yang tidak kompatibel dapat membuat engine gagal; aplikasi menjalankan ulang **executable CPU baseline berbeda**, bukan engine CUDA yang sama dengan `-ngl 0`. `--cpu` selalu memilih baseline. PC target tidak membutuhkan toolkit/internet, tetapi CUDA membutuhkan driver NVIDIA kompatibel yang sudah terpasang; aplikasi tidak memasang driver. Tidak ada dukungan NPU oleh runner llama.cpp ini (termasuk NPU yang terdeteksi).

UI menampilkan engine CUDA tersedia/hilang, pilihan otomatis sementara, backend report aktual, dan peringatan. `NVIDIA CUDA` hanya dilaporkan jika stderr verbose membuktikan inisialisasi CUDA dan offload layer positif; nol offload dilabel CPU, bukti tidak lengkap dilabel **Unverified (CUDA requested)** dengan peringatan. Path engine, jumlah layer, dan lokasi log tersimpan di report. `-ngl` atau daftar backend kompilasi saja tidak membuktikan GPU. Linux bundle CPU lama tetap didukung; preparation baru memisahkan `linux/cpu/` dan `linux/cuda/`. Linux CUDA masih membutuhkan runtime toolkit lokal; portable CUDA DLL packaging hanya untuk Windows.

Workflow harus berhasil di Windows sebelum paket dapat dinyatakan tersedia. Pemeriksaan di Linux tidak membuktikan build MSVC/CUDA, kelengkapan DLL pada Windows bersih, atau offload GPU nyata. Uji startup CPU/CUDA tanpa toolkit/PATH build, fallback driver tidak kompatibel, dan benchmark di PC NVIDIA target tetap diperlukan; runner cloud tidak melakukan benchmark 4B penuh.

Dataset menyertakan sumber dengan izin redistribusi belum jelas; audit `dataset/token-atlas/sources.json` sebelum membagikan paket kepada pihak lain. Workflow tidak menjalankan benchmark penuh pada runner cloud, hanya deteksi hardware, engine startup, test, dan import; uji benchmark di PC Windows target tetap diperlukan.

Salin folder `linux/` dan/atau `windows/`, `engine/`, `models/` beserta lisensi, `config/`, `dataset/`, dan `results/` (boleh kosong) ke USB. Gunakan exFAT atau filesystem yang mendukung file model. Linux membutuhkan izin executable dan mount **tanpa noexec**, glibc kompatibel, serta library sistem yang sesuai build. Driver GPU harus sudah tersedia; aplikasi tidak menginstalnya. Audit `ldd`/DLL dan uji pada PC bersih sebelum menyatakan paket tanpa dependency tambahan. Build CUDA/Vulkan dapat membutuhkan library vendor yang harus disertakan sesuai lisensi. Binary Windows dan Linux harus dibangun serta diuji masing-masing.

## Metodologi yang benar-benar diimplementasikan

Model disalin sekali ke cache OS TEMP `ai-checker-model-cache-v1` (Windows: `%TEMP%\ai-checker-model-cache-v1`) dengan nama SHA-256 model. Benchmark ulang memverifikasi hash dari disk lokal lalu melewati penyalinan dari flashdisk. Cache rusak disalin ulang; salinan parsial tidak dipakai. UI menampilkan progres penyalinan dan tahap persiapan/engine/analisis/report. Cache memakai sekitar 2,5 GB per versi model dan tetap ada setelah aplikasi ditutup; hapus folder cache ketika aplikasi tidak berjalan untuk membersihkan ruang. OS dapat membersihkan TEMP sehingga penyalinan perlu diulang. `model_preparation_seconds` terpisah dari waktu engine dan tok/s. Cache mempercepat persiapan, bukan menjamin komputasi benchmark lebih cepat. Engine belum mengekspos tahap internal terstruktur; tidak ada persentase benchmark palsu.

- llama-bench synthetic token workload: prompt 512, generation 128, prefilled KV depth 4096, 3 pengulangan, warmup bawaan engine aktif. Ini **bukan prompt teks chat** dan depth bukan batas context chat.
- Tok/s dan sampel berasal dari JSON engine, identitas commit dan workload divalidasi. Hash model dicek setiap benchmark. Waktu hashing/loading tidak masuk tok/s.
- Total proses dicatat; **load time, TTFT, peak RAM/VRAM belum diukur** dan tetap `null`/Unknown. Backend yang dikompilasi tidak otomatis membuktikan GPU offload; label dan engine log menjelaskan ketidakpastian.
- Kegagalan CUDA dicoba ulang dengan executable CPU baseline dan `-ngl 0`. Engine yang berhasil tetapi tanpa bukti offload tidak diklaim GPU. Tidak semua hang driver bisa dipulihkan; belum ada batas waktu engine. Jangan hapus lock saat benchmark berjalan; setelah crash, hapus `results/benchmark.lock` manual.
- TOPS tidak ditebak atau dijumlahkan. GPU PCI Linux yang tidak bisa diberi nama tetap tampil sebagai ID. NPU inventory Windows/Linux bersifat best-effort; NPU **tidak didukung** oleh runner ini.
- SQLite menyertakan referensi Token Atlas setelah script prepare/import dijalankan. Tabel pembanding strict tetap kosong sampai tersedia data kompatibel; report lokal tersimpan terpisah dan tidak otomatis menjadi benchmark referensi.
- Pembanding memerlukan exact family/quant/backend/engine/workload; 4B anchor exact model hash, throughput ±20%, RAM ±25%; ukuran lain harus berasal dari `machine_id` yang sama. Tidak ada scaling linier. Pengulangan pada satu mesin dihitung satu mesin independen. Kesesuaian hardware/VRAM dan memory-fit target belum dimodelkan sehingga rekomendasi ukuran besar bersifat bukti pembanding, **bukan jaminan dapat dimuat**.
- Rating kecepatan: ≥30 Excellent, ≥15 Good, ≥5 Possible, selainnya Not recommended. Ini rating **kecepatan terukur**, bukan threshold TOPS atau pemetaan 4B ke parameter besar. Confidence minimum semua target: ≥20 mesin high, ≥5 medium, selainnya low. Kekurangan data → Insufficient data.

## Token Atlas JSON offline

Snapshot resmi 41.103 measured records di `dataset/token-atlas/records.json`, dipin ke revision upstream. Script prepare mengimport ke SQLite otomatis; import manual menggunakan `ai-checker dataset-import dataset/token-atlas/records.json`. Registry atribusi/lisensi disertakan di `sources.json`; baca `dataset/token-atlas/README.md` sebelum redistribusi flashdisk karena beberapa sumber tidak menyatakan lisensi. Tidak mengunduh estimates/cluster.

Data Token Atlas dipakai dalam report baru sebagai referensi eksternal keluarga/quant yang sama, **bukan pasangan strict** jika hash/commit/workload/identitas mesin tidak lengkap. Karena itu dataset besar tidak otomatis berarti confidence tinggi. Tabel `atlas` mempertahankan record asli dan dipisahkan dari `measured`; tidak mengubah report lama. Belum ada normalisasi lintas engine atau tebakan metadata hilang.

## Format import dataset

File JSON berisi array object dengan seluruh field berikut (bukan contoh angka benchmark): `id`, `machine_id`, `hardware` (string), `ram_bytes` (integer), `model_family`, `model_sha256`, `parameters_b`, `quantization` (`Q4_K_M`), `backend`, `engine_commit`, `context_depth`, `prompt_tokens`, `generation_tokens`, `generation_tps`, `source` (URL atau rujukan internal), `date` (tanggal pengukuran), `kind` (`measured`). Simpan model hash aktual setiap ukuran; commit/backend/workload harus identik dengan report untuk dianggap comparable. Jangan mengimpor angka estimated sebagai measured. Import tidak memverifikasi kejujuran sumber eksternal: kurasi dataset tetap wajib.

`cargo test` menjalankan pemeriksaan parser dan rekomendasi; `cargo clippy --all-targets -- -D warnings` memeriksa kode. Lisensi Rust dependencies dan llama.cpp perlu ikut diperiksa saat redistribusi.

## Batas versi ini

Belum ada dataset eksternal terkurasi, auto-update, pembatalan benchmark dari UI, pengukuran TTFT/load/peak memory, backend NPU, atau sertifikasi USB Windows/Linux bersih. Cache model TEMP tersedia. Core berjalan offline setelah provisioning. Target startup/deteksi dan dua menit benchmark belum dijamin pada setiap hardware.