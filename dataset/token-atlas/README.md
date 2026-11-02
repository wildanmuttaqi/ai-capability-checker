# Token Atlas offline snapshot

Sumber: https://tokatlas.github.io/data/ (Token Atlas, tokatlas).
Revision: `59caf5d5a81d12b7182d04ff5b4df2a83a455ffb`.
Files: `records.json`, `sources.json`, downloaded 2026-11-02.
Snapshot records: 41,103. Retrieval date upstream: 2026-10-22.

SHA-256 `records.json`: `cbdd98e4e83b97b0e11d20ede12d639619bea2f6b4adc847d0b959c484d50cfb`.
SHA-256 `sources.json`: `705e6d4aa1a218dbc7cd95c61c9399d66fe4eb2ec7d1e3afd86b4b1155829e16`.

`records.json` hanya community/sourced measured records. File estimates dan cluster tidak diambil. Tiap record mempertahankan source URL, source name, quote, tanggal, flags, dan notes persis dari upstream. Bukan hasil ukur lokal dan bukan jaminan dapat direproduksi.

Lisensi **tidak seragam**: lihat `sources.json`. LLMCheck dan LLM Configurator menyatakan CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Sumber lain memiliki lisensi tidak disebutkan atau hanya atribusi per record; akses publik bukan izin redistribusi blanket. Sebelum mendistribusikan flashdisk ke pihak lain, periksa izin per sumber atau batasi snapshot ke sumber berizin. Jangan menghapus atribusi ini atau field provenance.

Import: `ai-checker dataset-import dataset/token-atlas/records.json`. Script prepare mengimport otomatis. Data disimpan utuh pada tabel SQLite `atlas`, terpisah dari tabel pembanding ketat `measured`. Import ulang identik aman; ID yang berubah ditolak atomik (snapshot baru harus direview sebelum mengganti database).

Recommendation memakai JSON ini sebagai **referensi eksternal** keluarga/quant yang sama, maksimal 5 contoh tanpa flags. Hash model, identitas mesin unik, commit, dan workload lengkap umumnya tidak tersedia, sehingga data ini tidak dinaikkan menjadi pasangan benchmark strict dan tidak menaikkan confidence secara palsu. Nama hardware sama tidak membuktikan mesin yang sama. llama-bench tetap menghasilkan pengukuran lokal aplikasi.