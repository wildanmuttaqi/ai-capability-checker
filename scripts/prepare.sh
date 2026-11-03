#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
backend="${1:-cpu}"
case "$backend" in cpu|cuda) ;; *) echo 'Usage: prepare.sh [cpu|cuda]' >&2; exit 1;; esac
for tool in cargo cmake git curl sha256sum; do command -v "$tool" >/dev/null || { echo "Build dependency missing: $tool" >&2; exit 1; }; done
mkdir -p engine/llama.cpp/linux models linux results dataset
src=target/llama.cpp
commit=6e7fc8a146556fe20be36906e1b1b7e03bc36d44
if [[ ! -d "$src/.git" ]]; then git clone --depth 1 --branch b7750 https://github.com/ggml-org/llama.cpp.git "$src"; fi
[[ "$(git -C "$src" rev-parse HEAD)" == "$commit" ]] || { echo 'Engine commit mismatch' >&2; exit 1; }
variants=(cpu)
if [[ "$backend" == cuda ]]; then variants+=(cuda); fi
for variant in "${variants[@]}"; do
  flags=(-DGGML_NATIVE=OFF -DGGML_VULKAN=OFF -DGGML_CUDA=OFF)
  if [[ "$variant" == cuda ]]; then flags+=(-DGGML_CUDA=ON); fi
  cmake -S "$src" -B "$src/build-$variant" -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH='$ORIGIN' -DLLAMA_CURL=OFF -DGGML_BACKEND_DL=OFF "${flags[@]}"
  cmake --build "$src/build-$variant" --config Release --target llama-bench -j "${JOBS:-2}"
  destination="engine/llama.cpp/linux/$variant"
  mkdir -p "$destination"
  cp "$src/build-$variant/bin/llama-bench" "$destination/"
  find "$src/build-$variant" -name '*.so*' \( -type f -o -type l \) -exec cp -L '{}' "$destination/" \;
done
cp "$src/LICENSE" engine/llama.cpp/LICENSE
cargo build --release --locked
cp target/release/ai-checker linux/ai-checker
chmod +x START-LINUX.sh linux/ai-checker engine/llama.cpp/linux/cpu/llama-bench
linux/ai-checker dataset-import dataset/token-atlas/records.json
file=models/Qwen3-4B-Q4_K_M.gguf
hash=7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5
url=https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/bc640142c66e1fdd12af0bd68f40445458f3869b
if [[ ! -f "$file" ]]; then
  curl --fail --location --retry 5 --retry-all-errors --continue-at - "$url/Qwen3-4B-Q4_K_M.gguf" -o "$file.part"
  printf '%s  %s\n' "$hash" "$file.part" | sha256sum -c -
  mv "$file.part" "$file"
fi
printf '%s  %s\n' "$hash" "$file" | sha256sum -c -
curl --fail --location "$url/LICENSE" -o models/Qwen3-LICENSE
echo 'Ready: linux/ai-checker (offline UI), or linux/ai-checker detect'