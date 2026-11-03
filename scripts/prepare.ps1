param([ValidateSet('cpu','cuda')][string]$Backend = 'cuda', [switch]$Package)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
function Check { if ($LASTEXITCODE -ne 0) { throw "Native command failed: $LASTEXITCODE" } }
foreach ($tool in @('cargo','cmake','git','curl.exe')) { Get-Command $tool -ErrorAction Stop | Out-Null }
$vswhere = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vs) { throw 'Visual Studio C++ build tools missing' }
# CUDA 12.6 rejects newer MSVC 19.4x. windows-2022 includes v142 (19.29).
if (!(Test-Path "$vs/VC/Tools/MSVC/14.29*")) { throw 'Install MSVC v142 x64/x86 build tools (14.29) for CUDA 12.6 compatibility' }
New-Item -ItemType Directory -Force engine/llama.cpp/windows,models,windows,results,dataset | Out-Null
$src = 'target/llama.cpp'
if (!(Test-Path "$src/.git")) { git clone --depth 1 --branch b7750 https://github.com/ggml-org/llama.cpp.git $src; Check }
if ((git -C $src rev-parse HEAD) -ne '6e7fc8a146556fe20be36906e1b1b7e03bc36d44') { throw 'Engine commit mismatch' }
$variants = @('cpu')
if ($Backend -eq 'cuda') {
    Get-Command nvcc -ErrorAction Stop | Out-Null
    if (!(Test-Path "$env:CUDA_PATH/bin/nvcc.exe")) { throw 'CUDA_PATH must point to the installed CUDA toolkit (build host only)' }
    $variants += 'cuda'
}
foreach ($variant in $variants) {
    $destination = "engine/llama.cpp/windows/$variant"
    # Never mix DLLs from an older build or a different accelerator.
    if (Test-Path $destination) { Remove-Item $destination -Recurse -Force }
    New-Item -ItemType Directory -Force $destination | Out-Null
    $flags = @('-DGGML_NATIVE=OFF','-DLLAMA_CURL=OFF','-DGGML_BACKEND_DL=OFF','-DBUILD_SHARED_LIBS=ON','-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded','-DGGML_OPENMP=OFF','-DGGML_VULKAN=OFF','-DGGML_CUDA=OFF')
    if ($variant -eq 'cuda') {
        # CUDA 12.6: Pascal through Hopper SASS + forward-compatible PTX, no runner GPU required.
        # Avoid importing nvcuda.dll at process load: --help must also work on GPU-less CI.
        $flags += @('-DGGML_CUDA=ON','-DGGML_CUDA_NO_VMM=ON','-DCMAKE_CUDA_ARCHITECTURES=60-real;61-real;70-real;75-real;80-real;86-real;89-real;90','-DCMAKE_CUDA_RUNTIME_LIBRARY=Static')
    }
    cmake -S $src -B "$src/build-windows-$variant" -G 'Visual Studio 17 2022' -A x64 -T v142 @flags; Check
    cmake --build "$src/build-windows-$variant" --config Release --target llama-bench -j 2; Check
    Get-ChildItem "$src/build-windows-$variant/bin/Release" -File | Where-Object { $_.Name -eq 'llama-bench.exe' -or $_.Extension -eq '.dll' } | Copy-Item -Destination $destination
    if ($variant -eq 'cuda') {
        # cuBLAS is dynamically linked by ggml-cuda; include its complete runtime pair.
        foreach ($pattern in @('cublas64_*.dll','cublasLt64_*.dll','cudart64_*.dll')) {
            $runtime = @(Get-ChildItem "$env:CUDA_PATH/bin/$pattern" -File)
            if ($runtime.Count -eq 0) { throw "Missing CUDA runtime: $pattern" }
            $runtime | Copy-Item -Destination $destination
        }
        $license = Join-Path $env:CUDA_PATH 'EULA.txt'
        if (!(Test-Path $license)) { throw 'CUDA redistribution license missing; review toolkit redistribution terms' }
        Copy-Item $license "$destination/CUDA-EULA.txt"
        # NVIDIA DLLs may depend on the dynamic MSVC CRT even though our own code uses /MT.
        $crt = Get-ChildItem "$vs/VC/Redist/MSVC/*/x64/Microsoft.VC*.CRT" -Directory | Sort-Object FullName -Descending | Select-Object -First 1
        if (!$crt) { throw 'MSVC app-local redistributable CRT missing' }
        Copy-Item "$($crt.FullName)/*.dll" $destination
    }
    # Prove app-local startup without CUDA/Visual Studio directories masking missing DLLs.
    $savedPath = $env:PATH
    try {
        $env:PATH = "$env:SystemRoot/System32;$env:SystemRoot"
        & "$destination/llama-bench.exe" --help | Out-Null; Check
    } finally { $env:PATH = $savedPath }
}
Copy-Item "$src/LICENSE" engine/llama.cpp/LICENSE
$env:RUSTFLAGS = "$env:RUSTFLAGS -C target-feature=+crt-static"
cargo build --release --locked; Check
Copy-Item target/release/ai-checker.exe windows/
& windows/ai-checker.exe detect; Check
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
    foreach ($variant in $variants) { Copy-Item "engine/llama.cpp/windows/$variant" "$stage/engine/llama.cpp/windows/" -Recurse }
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
Write-Output "Ready: windows/ai-checker.exe; variants: $variants. Target PC needs a compatible NVIDIA driver for CUDA, not a toolkit. No driver is installed."