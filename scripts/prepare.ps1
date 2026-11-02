param([ValidateSet('cpu','vulkan','cuda')][string]$Backend = 'cpu', [switch]$Package)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
function Check { if ($LASTEXITCODE -ne 0) { throw "Native command failed: $LASTEXITCODE" } }
foreach ($tool in @('cargo','cmake','git','curl.exe')) { Get-Command $tool -ErrorAction Stop | Out-Null }
New-Item -ItemType Directory -Force engine/llama.cpp/windows,models,windows,results,dataset | Out-Null
$src = 'target/llama.cpp'
if (!(Test-Path "$src/.git")) { git clone --depth 1 --branch b7750 https://github.com/ggml-org/llama.cpp.git $src; Check }
if ((git -C $src rev-parse HEAD) -ne '6e7fc8a146556fe20be36906e1b1b7e03bc36d44') { throw 'Engine commit mismatch' }
$flags = @('-DGGML_NATIVE=OFF','-DLLAMA_CURL=OFF','-DGGML_BACKEND_DL=OFF','-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded','-DGGML_OPENMP=OFF')
if ($Backend -eq 'cuda') { $flags += '-DGGML_CUDA=ON' }
if ($Backend -eq 'vulkan') { $flags += '-DGGML_VULKAN=ON' }
cmake -S $src -B "$src/build-$Backend" -A x64 @flags; Check
cmake --build "$src/build-$Backend" --config Release --target llama-bench -j 2; Check
Get-ChildItem "$src/build-$Backend" -Recurse -File | Where-Object { $_.Name -eq 'llama-bench.exe' -or $_.Extension -eq '.dll' } | Copy-Item -Destination engine/llama.cpp/windows
Copy-Item "$src/LICENSE" engine/llama.cpp/LICENSE
$env:RUSTFLAGS = "$env:RUSTFLAGS -C target-feature=+crt-static"
cargo build --release --locked; Check
Copy-Item target/release/ai-checker.exe windows/
& windows/ai-checker.exe detect; Check
& engine/llama.cpp/windows/llama-bench.exe --help | Out-Null; Check
& windows/ai-checker.exe dataset-import dataset/token-atlas/records.json; Check
$file = 'models/Qwen3-4B-Q4_K_M.gguf'
$hash = '7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5'
$url = 'https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/bc640142c66e1fdd12af0bd68f40445458f3869b'
if (!(Test-Path $file)) {
    curl.exe --fail --location --retry 5 --retry-all-errors --continue-at - "$url/Qwen3-4B-Q4_K_M.gguf" -o "$file.part"; Check
    if ((Get-FileHash "$file.part" -Algorithm SHA256).Hash.ToLower() -ne $hash) { throw 'Model checksum mismatch' }
    Move-Item "$file.part" $file
}
if ((Get-FileHash $file -Algorithm SHA256).Hash.ToLower() -ne $hash) { throw 'Model checksum mismatch' }
curl.exe --fail --location "$url/LICENSE" -o models/Qwen3-LICENSE; Check
if ($Package) {
    Get-Command tar.exe -ErrorAction Stop | Out-Null
    $stage = 'dist/AI-CHECKER-Windows'
    if (Test-Path $stage) { throw "Staging directory already exists: $stage. Move or remove it before packaging again." }
    $archive = "dist/AI-CHECKER-Windows-$Backend.zip"
    if (Test-Path $archive) { throw "Archive already exists: $archive" }
    New-Item -ItemType Directory -Force "$stage/windows","$stage/engine/llama.cpp/windows","$stage/models","$stage/dataset/token-atlas","$stage/results" | Out-Null
    Copy-Item windows/ai-checker.exe "$stage/windows/"
    Copy-Item engine/llama.cpp/windows/* "$stage/engine/llama.cpp/windows/"
    Copy-Item engine/llama.cpp/LICENSE "$stage/engine/llama.cpp/"
    Copy-Item $file,models/Qwen3-LICENSE "$stage/models/"
    Copy-Item dataset/token-atlas/* "$stage/dataset/token-atlas/"
    Copy-Item config $stage -Recurse
    Copy-Item README.md $stage
    Copy-Item scripts/start-windows.cmd "$stage/START-WINDOWS.cmd"
    # Fresh DB excludes private hardware reports from the development machine.
    & windows/ai-checker.exe --root (Resolve-Path $stage).Path dataset-import dataset/token-atlas/records.json; Check
    # tar supports ZIP64; Compress-Archive cannot handle the >2 GB GGUF file.
    tar.exe -a -c -f $archive -C dist AI-CHECKER-Windows; Check
    $digest = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLower()
    Set-Content "$archive.sha256" "$digest  $(Split-Path $archive -Leaf)" -Encoding ASCII
    Write-Output "Portable archive: $archive"
}
Write-Output 'Ready: windows/ai-checker.exe. Audit bundled engine DLL dependencies before distributing.'