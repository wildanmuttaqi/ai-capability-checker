# AI Capability Checker

Aplikasi Rust offline Windows x64 / Linux x64: hardware → benchmark nyata Qwen3-4B → pembanding SQLite → report. Tanpa Node/Python di PC target. UI HTML tertanam dalam binary; jalankan aplikasi tanpa argumen lalu buka alamat loopback yang dicetak.

## Persiapan pada mesin pengembangan

Butuh Rust/Cargo, C/C++ compiler, CMake, Git, curl, internet, dan sekitar 8 GB ruang kosong. Linux: jalankan `bash scripts/prepare.sh cpu` (atau `vulkan`/`cuda` jika SDK build tersedia). Windows: jalankan `powershell -File scripts/prepare.ps1 -Backend cpu` dari Developer PowerShell dengan Visual Studio C++ Build Tools. Rust dipakai saat build saja, bukan di PC target.

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

### Paket Windows klik-ganda

Di mesin Windows build, jalankan `scripts/prepare.ps1 -Backend cpu -Package` dari PowerShell. Output `dist/AI-CHECKER-Windows-cpu.zip` berisi executable Windows x64, engine CPU dan DLL, model Qwen3-4B, SQLite baru tanpa report pribadi, JSON Token Atlas, konfigurasi, lisensi sumber yang tersedia, serta `START-WINDOWS.cmd`. SHA-256 ZIP disimpan di file pendamping. Packaging menolak menimpa staging/ZIP lama. ZIP64 dipakai untuk model >2 GB; Windows 10/11 x64 disarankan.

Alternatif tanpa compiler lokal: push proyek ke GitHub, buka **Actions → Windows portable USB → Run workflow**. Setelah build berhasil, unduh artifact `AI-CHECKER-Windows-CPU`, ekstrak ZIP paket ke flashdisk, lalu klik `START-WINDOWS.cmd`. Model sudah ikut di ZIP sehingga PC target tidak memerlukan internet/Rust/compiler. Jika browser terbuka sebelum server siap, refresh halaman. Port lokal 8798 digunakan. Paket default CPU; GPU memerlukan build backend dan driver yang sesuai. Workflow harus berhasil di Windows sebelum executable/paket dapat dinyatakan tersedia; menulis workflow saja bukan verifikasi Windows.

Dataset menyertakan sumber dengan izin redistribusi belum jelas; audit `dataset/token-atlas/sources.json` sebelum membagikan paket kepada pihak lain. Workflow tidak menjalankan benchmark penuh pada runner cloud, hanya deteksi hardware, engine startup, test, dan import; uji benchmark di PC Windows target tetap diperlukan.

Salin folder `linux/` dan/atau `windows/`, `engine/`, `models/` beserta lisensi, `config/`, `dataset/`, dan `results/` (boleh kosong) ke USB. Gunakan exFAT atau filesystem yang mendukung file model. Linux membutuhkan izin executable dan mount **tanpa noexec**, glibc kompatibel, serta library sistem yang sesuai build. Driver GPU harus sudah tersedia; aplikasi tidak menginstalnya. Audit `ldd`/DLL dan uji pada PC bersih sebelum menyatakan paket tanpa dependency tambahan. Build CUDA/Vulkan dapat membutuhkan library vendor yang harus disertakan sesuai lisensi. Binary Windows dan Linux harus dibangun serta diuji masing-masing.

## Metodologi yang benar-benar diimplementasikan

- llama-bench synthetic token workload: prompt 512, generation 128, prefilled KV depth 4096, 3 pengulangan, warmup bawaan engine aktif. Ini **bukan prompt teks chat** dan depth bukan batas context chat.
- Tok/s dan sampel berasal dari JSON engine, identitas commit dan workload divalidasi. Hash model dicek setiap benchmark. Waktu hashing/loading tidak masuk tok/s.
- Total proses dicatat; **load time, TTFT, peak RAM/VRAM belum diukur** dan tetap `null`/Unknown. Backend yang dikompilasi tidak otomatis membuktikan GPU offload; label dan engine log menjelaskan ketidakpastian.
- Kegagalan GPU dicoba ulang dengan `-ngl 0`. Tidak semua driver crash bisa dipulihkan. Benchmark lama tidak memiliki batas waktu; hentikan CLI jika perlu. Jangan hapus lock saat benchmark berjalan; setelah crash, hapus `results/benchmark.lock` manual.
- TOPS tidak ditebak atau dijumlahkan. GPU PCI Linux yang tidak bisa diberi nama tetap tampil sebagai ID; NPU discovery belum membuktikan dukungan runtime. Windows NPU belum dideteksi.
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

Belum ada dataset eksternal terkurasi, auto-update, cache model TEMP, pembatalan benchmark dari UI, pengukuran TTFT/load/peak memory, Windows NPU inventory, atau sertifikasi USB Windows/Linux bersih. Core berjalan offline setelah provisioning. Target startup/deteksi dan dua menit benchmark belum dijamin pada setiap hardware. Tambahkan fitur tersebut setelah baseline dan data referensi tervalidasi.