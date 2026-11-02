#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
app=./linux/ai-checker
engine=./engine/llama.cpp/linux/llama-bench
if [ ! -f "$app" ] || [ ! -f "$engine" ]; then
    printf '%s\n' 'Paket belum lengkap. Binary Linux dan engine harus berada di folder paket.' >&2
    exit 1
fi
if [ ! -x "$app" ] || [ ! -x "$engine" ]; then
    chmod +x "$app" "$engine" || {
        printf '%s\n' 'Tidak dapat mengatur izin executable. Salin seluruh paket ke penyimpanan lokal.' >&2
        exit 1
    }
fi
printf '%s\n' 'Buka http://127.0.0.1:8798. Jika browser belum terhubung, refresh halaman.' 
printf '%s\n' 'Biarkan terminal terbuka. Ctrl+C untuk berhenti. Jika Permission denied, periksa mount noexec flashdisk.'
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open http://127.0.0.1:8798 >/dev/null 2>&1 &
fi
exec "$app" --root "$PWD" ui --port 8798