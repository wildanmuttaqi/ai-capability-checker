@echo off
setlocal
cd /d "%~dp0"
if not exist "windows\ai-checker.exe" (
  echo File windows\ai-checker.exe tidak ditemukan. Ekstrak seluruh paket terlebih dahulu.
  pause
  exit /b 1
)
echo AI Capability Checker - portable offline
echo Buka http://127.0.0.1:8798 di browser.
echo Tutup jendela ini atau tekan Ctrl+C untuk menghentikan aplikasi.
start "" "http://127.0.0.1:8798"
"windows\ai-checker.exe" --root "%CD%" ui --port 8798
if errorlevel 1 (
  echo Aplikasi berhenti dengan error. Port 8798 mungkin sudah digunakan.
  pause
  exit /b 1
)