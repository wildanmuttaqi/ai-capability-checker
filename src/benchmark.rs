use crate::{dataset, detect, stamp, Config};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::Path,
    process::{Command, Stdio},
    time::Instant,
};

struct Lock(std::path::PathBuf);
impl Drop for Lock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[derive(serde::Serialize)]
pub struct EngineSelection {
    pub path: std::path::PathBuf,
    pub cuda: bool,
    pub warnings: Vec<String>,
}

pub fn select_engine(root: &Path, c: &Config, hardware: &Value, cpu: bool) -> EngineSelection {
    let eligible = hardware["gpu"]
        .as_array()
        .is_some_and(|gpus| gpus.iter().any(|gpu| gpu["cuda_probe"] == true));
    let bundled = c.cuda_engine(root).is_file();
    let cuda = !cpu && c.gpu_layers != 0 && eligible && bundled;
    let mut warnings = vec![];
    if !cpu && c.gpu_layers != 0 {
        if !bundled {
            warnings.push("CUDA engine missing from bundle; using CPU baseline. Prepare a CUDA package to enable NVIDIA acceleration.".into());
        }
        if !eligible {
            warnings.push("NVIDIA CUDA probe unavailable: nvidia-smi did not report a device. Using CPU baseline; inventory alone does not prove CUDA eligibility. No driver is installed by this app.".into());
        }
    }
    EngineSelection {
        path: if cuda {
            c.cuda_engine(root)
        } else {
            c.engine(root)
        },
        cuda,
        warnings,
    }
}

// Compiled backend names and -ngl are not proof of execution on a GPU.
fn offload_evidence(log: &str) -> Option<u32> {
    log.lines()
        .filter_map(|line| {
            let (_, tail) = line.split_once(": offloaded ")?;
            let (counts, _) = tail.split_once(" layers to GPU")?;
            let (used, total) = counts.split_once('/')?;
            let (used, total) = (used.parse::<u32>().ok()?, total.parse::<u32>().ok()?);
            (total > 0 && used <= total).then_some(used)
        })
        .min()
}

fn provenance(measured: &mut Value, cuda: bool, log: &str) -> Option<String> {
    let offloaded = offload_evidence(log);
    let no_offload = measured["raw"].as_array().is_some_and(|rows| {
        !rows.is_empty()
            && rows
                .iter()
                .all(|row| row["n_gpu_layers"].as_i64() == Some(0))
    });
    let (backend, warning) = if !cuda || no_offload || offloaded == Some(0) {
        (
            "CPU",
            cuda.then(|| {
                "CUDA engine ran without GPU layer offload; result is CPU, not CUDA acceleration."
                    .to_string()
            }),
        )
    } else if offloaded.is_some_and(|n| n > 0) && log.contains("ggml_cuda_init: found ") {
        ("NVIDIA CUDA", None)
    } else {
        ("Unverified (CUDA requested)", Some("CUDA layer offload could not be verified from engine stderr; GPU throughput must not be assumed. See results/engine-primary.log.".into()))
    };
    measured["backend"] = json!(backend);
    measured["offloaded_layers"] = json!(offloaded);
    warning
}

fn with_cpu_fallback(
    primary: &EngineSelection,
    baseline: &EngineSelection,
    warnings: &mut Vec<String>,
    invoke: impl Fn(&EngineSelection, &str) -> Result<Value>,
) -> Result<Value> {
    match invoke(primary, "primary") {
        Ok(v) => Ok(v),
        Err(e) if primary.cuda => {
            warnings.push(format!(
                "CUDA attempt failed: {e}. Retried using distinct CPU baseline {}.",
                baseline.path.display()
            ));
            invoke(baseline, "cpu-fallback")
        }
        Err(e) => Err(e),
    }
}

pub fn parse(bytes: &[u8], c: &Config, cpu: bool) -> Result<Value> {
    let rows: Vec<Value> =
        serde_json::from_slice(bytes).context("llama-bench did not return a JSON array")?;
    let generation = rows
        .iter()
        .find(|r| {
            r["n_gen"].as_u64() == Some(c.generation_tokens as u64)
                && r["n_prompt"].as_u64() == Some(0)
        })
        .context("Missing generation result")?;
    let prompt = rows
        .iter()
        .find(|r| {
            r["n_prompt"].as_u64() == Some(c.prompt_tokens as u64) && r["n_gen"].as_u64() == Some(0)
        })
        .context("Missing prompt-processing result")?;
    for r in [generation, prompt] {
        if r["n_depth"].as_u64() != Some(c.context_depth as u64) {
            bail!("Engine returned a different context depth");
        }
        let rate = r["avg_ts"].as_f64().context("Missing token rate")?;
        if !rate.is_finite() || rate <= 0.0 {
            bail!("Invalid token rate");
        }
        let samples = r["samples_ts"]
            .as_array()
            .context("Missing per-run samples")?;
        if samples.len() != c.runs as usize
            || samples
                .iter()
                .any(|s| !s.as_f64().is_some_and(|v| v.is_finite() && v > 0.0))
        {
            bail!("Invalid benchmark samples");
        }
        if r["build_commit"].as_str().unwrap_or("").is_empty() {
            bail!("Missing engine identity");
        }
        if !r["stddev_ts"]
            .as_f64()
            .is_some_and(|n| n.is_finite() && n >= 0.0)
        {
            bail!("Invalid standard deviation");
        }
    }
    if generation["build_commit"] != prompt["build_commit"] {
        bail!("Mixed engine versions in benchmark results");
    }
    for row in [generation, prompt] {
        if row["n_gpu_layers"].as_i64().is_none() {
            bail!("Missing offload configuration");
        }
        if cpu && row["n_gpu_layers"].as_i64() != Some(0) {
            bail!("CPU baseline returned nonzero GPU layer configuration");
        }
    }
    let backend = if cpu {
        "CPU"
    } else {
        "Unverified (CUDA requested)"
    };
    Ok(
        json!({"generation_tps":generation["avg_ts"],"prompt_tps":prompt["avg_ts"],
        "stddev_tps":generation["stddev_ts"],"samples_tps":generation["samples_ts"],"backend":backend,
        "engine_commit":generation["build_commit"],"load_seconds":null,"ttft_seconds":null,"raw":rows}),
    )
}

pub fn run(root: &Path, cpu: bool) -> Result<Value> {
    run_with_progress(root, cpu, &|message| eprintln!("{message}"))
}

struct TempModel(std::path::PathBuf);
impl Drop for TempModel {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn copy_verified(
    source: &Path,
    target: &Path,
    expected: &str,
    progress: &dyn Fn(String),
) -> Result<()> {
    let mut input = File::open(source).context("4B GGUF missing. Run preparation script first.")?;
    let total = input.metadata()?.len();
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(target)
        .context("Cannot create TEMP model; check free space and permissions")?;
    let mut hash = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    let mut copied = 0;
    let mut previous = u64::MAX;
    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            break;
        }
        output
            .write_all(&buf[..n])
            .context("Copy to TEMP failed; check free storage")?;
        hash.update(&buf[..n]);
        copied += n as u64;
        let percent = copied.saturating_mul(100) / total.max(1);
        if percent != previous {
            progress(format!(
                "Menyalin model ke TEMP + menghitung SHA-256: {percent}% ({} / {} MiB)",
                copied / 1048576,
                total / 1048576
            ));
            previous = percent;
        }
    }
    progress("Menyelesaikan penulisan model TEMP dan memeriksa checksum…".into());
    output.sync_all()?;
    if copied != total || format!("{:x}", hash.finalize()) != expected.to_lowercase() {
        bail!("Model checksum mismatch; benchmark refused");
    }
    Ok(())
}

fn cache_model(
    source: &Path,
    cache: &Path,
    expected: &str,
    progress: &dyn Fn(String),
) -> Result<std::path::PathBuf> {
    fs::create_dir_all(cache)?;
    let target = cache.join(format!("{}.gguf", expected.to_lowercase()));
    if target.is_file() {
        progress("Cache lokal ditemukan. Memverifikasi SHA-256 dari disk lokal; tidak menyalin ulang dari flashdisk…".into());
        let mut file = File::open(&target)?;
        let mut hash = Sha256::new();
        let mut buf = vec![0u8; 1024 * 1024];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hash.update(&buf[..n]);
        }
        if format!("{:x}", hash.finalize()) == expected.to_lowercase() {
            progress("Cache lokal valid. Melewati penyalinan model dari flashdisk.".into());
            return Ok(target);
        }
        drop(file);
        progress("Checksum cache tidak cocok; menyiapkan salinan baru dari flashdisk…".into());
        fs::remove_file(&target)?;
    }
    let staging = cache.join(format!("copy-{}-{}", std::process::id(), stamp()));
    fs::create_dir(&staging)?;
    let _guard = TempModel(staging.clone());
    let partial = staging.join("model.part");
    copy_verified(source, &partial, expected, progress)?;
    // Another process may have published the same content-addressed model during copying.
    if !target.exists() {
        fs::rename(&partial, &target)?;
    }
    Ok(target)
}

pub fn run_with_progress(root: &Path, cpu: bool, progress: &dyn Fn(String)) -> Result<Value> {
    fs::create_dir_all(root.join("results"))?;
    let lock_path = root.join("results/benchmark.lock");
    let mut lock = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
        .context(
            "Benchmark already running (after a crash, remove results/benchmark.lock manually)",
        )?;
    let _guard = Lock(lock_path);
    writeln!(lock, "{}", std::process::id())?;
    let c = Config::load(root)?;
    let baseline = select_engine(root, &c, &Value::Null, true);
    // Legacy Linux layout is accepted only for CPU bundles, never as a CUDA fallback.
    if baseline.path.parent().is_some_and(|parent| {
        fs::read_dir(parent)
            .into_iter()
            .flatten()
            .flatten()
            .any(|entry| {
                let name = entry.file_name().to_string_lossy().to_lowercase();
                name.contains("ggml-cuda")
            })
    }) {
        bail!("CPU baseline directory contains CUDA libraries; rebuild with scripts/prepare.sh cpu or scripts/prepare.ps1 to isolate the engines");
    }
    if !baseline.path.is_file() {
        bail!(
            "llama-bench missing: {}. Run preparation script on a development machine.",
            baseline.path.display()
        );
    }
    let source = root.join(&c.model_path);
    progress("Memeriksa hardware dan RAM sebelum menyalin model…".into());
    let hardware = detect();
    let selected = select_engine(root, &c, &hardware, cpu);
    for warning in &selected.warnings {
        progress(warning.clone());
    }
    let model_bytes = fs::metadata(&source).context("4B GGUF missing")?.len();
    let required = model_bytes + 1024 * 1024 * 1024;
    let available = hardware["ram_available_bytes"].as_u64().unwrap_or(0);
    if available < required {
        bail!(
            "Insufficient available RAM: {available} bytes; conservative minimum {required} bytes"
        );
    }
    let temp_dir = std::env::temp_dir().join("ai-checker-model-cache-v1");
    let preparation = Instant::now();
    progress(format!("Menyiapkan model lokal di {}", temp_dir.display()));
    let model = cache_model(&source, &temp_dir, &c.model_sha256, progress)?;
    let preparation_seconds = preparation.elapsed().as_secs_f64();
    let threads = if c.threads == 0 {
        hardware["cores"].as_u64().unwrap_or(1).max(1) as usize
    } else {
        c.threads
    };
    let started = Instant::now();
    let invoke = |selection: &EngineSelection, suffix: &str| -> Result<Value> {
        let engine = &selection.path;
        let use_cpu = !selection.cuda;
        let log_path = root.join(format!("results/engine-{suffix}.log"));
        let log = File::create(&log_path)?;
        progress(format!("Engine aktif ({}): memuat model dari TEMP → warmup → prompt {} token → generation {} token; {} pengulangan, prefill {} token. Tahap internal tidak dilaporkan engine; tunggu hasil, bukan persentase perkiraan.",if use_cpu{"CPU"}else{"backend otomatis"},c.prompt_tokens,c.generation_tokens,c.runs,c.context_depth));
        let output = Command::new(engine)
            .current_dir(engine.parent().context("Invalid engine path")?)
            .arg("-m")
            .arg(&model)
            .args([
                "-o",
                "json",
                "--verbose",
                "-p",
                &c.prompt_tokens.to_string(),
                "-n",
                &c.generation_tokens.to_string(),
                "-d",
                &c.context_depth.to_string(),
                "-r",
                &c.runs.to_string(),
                "-t",
                &threads.to_string(),
                "-ngl",
                &if use_cpu {
                    "0".to_string()
                } else {
                    c.gpu_layers.to_string()
                },
            ])
            .stdin(Stdio::null())
            .stderr(Stdio::from(log))
            .output()
            .context("Cannot launch bundled llama-bench")?;
        if !output.status.success() {
            bail!(
                "llama-bench failed: {} (see results/engine-{suffix}.log)",
                output.status
            );
        }
        let mut measured = parse(&output.stdout, &c, use_cpu)?;
        let log_text =
            fs::read_to_string(&log_path).context("Cannot read engine offload evidence")?;
        measured["backend_warning"] = json!(provenance(&mut measured, selection.cuda, &log_text));
        measured["engine_path"] = json!(engine);
        measured["engine_log"] = json!(format!("results/engine-{suffix}.log"));
        measured["requested_backend"] = json!(if selection.cuda { "NVIDIA CUDA" } else { "CPU" });
        Ok(measured)
    };
    let mut warnings = vec!["Synthetic llama-bench token workload; tokenization/sampling excluded. Context depth is prefilled KV, not a chat context limit.".to_string(),
        "Load time, TTFT and peak RAM/VRAM are not measured by this runner; reported as unknown, never inferred from throughput.".to_string()];
    warnings.extend(selected.warnings.clone());
    warnings.push(
        "NPU not supported by this llama.cpp runner; detected NPU hardware is not benchmarked."
            .into(),
    );
    let mut measured = with_cpu_fallback(&selected, &baseline, &mut warnings, invoke)?;
    if let Some(warning) = measured["backend_warning"].as_str() {
        warnings.push(warning.to_string());
    }
    measured["total_seconds"] = json!(started.elapsed().as_secs_f64());
    measured["model_preparation_seconds"] = json!(preparation_seconds);
    measured["model_storage"] = json!("OS TEMP/ai-checker-model-cache-v1; content-addressed cache, SHA-256 verified on every run; retained for reuse");
    progress(
        "Benchmark selesai. Mencocokkan hasil 4B dengan dataset dan menganalisis hardware…".into(),
    );
    let db = dataset::open(root)?;
    let recommendation = dataset::recommend(&db, &c, &hardware, &measured)?;
    let report = json!({"id":stamp().to_string(),"hardware":hardware,"model":c,"benchmark":measured,
        "recommendation":recommendation,"warnings":warnings});
    let text = serde_json::to_string_pretty(&report)?;
    progress("Menyimpan report ke SQLite dan folder results…".into());
    // Immutable reports in SQLite are the source of truth; exports never replace existing reports.
    db.execute("INSERT INTO reports(json) VALUES (?1)", [&text])?;
    let path = root.join(format!(
        "results/report-{}.json",
        report["id"].as_str().unwrap()
    ));
    let mut out = OpenOptions::new().create_new(true).write(true).open(path)?;
    out.write_all(text.as_bytes())?;
    out.sync_all()?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selects_distinct_engines_and_falls_back_to_baseline() {
        let dir = std::env::temp_dir().join(format!(
            "engine-selection-{}-{}",
            std::process::id(),
            stamp()
        ));
        fs::create_dir(&dir).unwrap();
        let _guard = TempModel(dir.clone());
        let c: Config = serde_json::from_str(include_str!("../config/benchmark.json")).unwrap();
        let nvidia = json!({"gpu":[{"name":"NVIDIA test", "cuda_probe":true}]});
        let missing = select_engine(&dir, &c, &nvidia, false);
        assert!(!missing.cuda);
        assert!(missing.warnings.iter().any(|w| w.contains("missing")));
        let cuda = c.cuda_engine(&dir);
        fs::create_dir_all(cuda.parent().unwrap()).unwrap();
        fs::write(&cuda, []).unwrap();
        let baseline = select_engine(&dir, &c, &nvidia, true);
        let primary = select_engine(&dir, &c, &nvidia, false);
        assert!(primary.cuda);
        assert_ne!(primary.path, baseline.path);
        assert!(!baseline.cuda);
        assert!(!select_engine(&dir, &c, &json!({"gpu":[]}), false).cuda);
        assert!(
            !select_engine(
                &dir,
                &c,
                &json!({"gpu":[{"name":"NVIDIA inventory only"}]}),
                false
            )
            .cuda
        );
        let mut disabled = c.clone();
        disabled.gpu_layers = 0;
        assert!(!select_engine(&dir, &disabled, &nvidia, false).cuda);
        let calls = std::cell::RefCell::new(vec![]);
        let mut warnings = vec![];
        let result = with_cpu_fallback(&primary, &baseline, &mut warnings, |selection, suffix| {
            calls
                .borrow_mut()
                .push((selection.path.clone(), suffix.to_string()));
            if selection.cuda {
                bail!("simulated missing CUDA DLL")
            }
            Ok(json!({"backend":"CPU", "engine_path":selection.path}))
        })
        .unwrap();
        assert_eq!(result["backend"], "CPU");
        assert_eq!(result["engine_path"], json!(baseline.path));
        assert_eq!(
            calls.borrow().as_slice(),
            &[
                (cuda, "primary".into()),
                (baseline.path.clone(), "cpu-fallback".into())
            ]
        );
        assert!(warnings[0].contains("distinct CPU baseline"));
        assert!(
            with_cpu_fallback(&baseline, &baseline, &mut vec![], |_, _| bail!(
                "CPU failure"
            ))
            .is_err()
        );
    }

    #[test]
    fn cuda_label_requires_actual_offload_evidence() {
        let log = "ggml_cuda_init: found 1 CUDA devices:\nllama_model_load: offloaded 37/37 layers to GPU\n";
        let mut measured = json!({"raw":[{"n_gpu_layers":99,"backends":"CUDA"}]});
        assert!(provenance(&mut measured, true, log).is_none());
        assert_eq!(measured["backend"], "NVIDIA CUDA");
        assert_eq!(measured["offloaded_layers"], 37);
        assert!(provenance(&mut measured, true, "").is_some());
        assert_eq!(measured["backend"], "Unverified (CUDA requested)");
        assert!(provenance(
            &mut measured,
            true,
            "llama_model_load: offloaded 0/37 layers to GPU"
        )
        .is_some());
        assert_eq!(measured["backend"], "CPU");
        measured["raw"][0]["n_gpu_layers"] = json!(0);
        assert!(provenance(&mut measured, true, "").is_some());
        assert_eq!(measured["backend"], "CPU");
        assert!(provenance(&mut measured, false, log).is_none());
        assert_eq!(measured["backend"], "CPU");
        assert_eq!(offload_evidence("offloaded bananas/37 layers to GPU"), None);
    }

    #[test]
    fn cache_reuses_valid_copy_and_repairs_corruption() {
        let dir =
            std::env::temp_dir().join(format!("cache-test-{}-{}", std::process::id(), stamp()));
        fs::create_dir(&dir).unwrap();
        let _guard = TempModel(dir.clone());
        let source = dir.join("source");
        fs::write(&source, b"model").unwrap();
        let hash = format!("{:x}", Sha256::digest(b"model"));
        let cache = dir.join("cache");
        let model = cache_model(&source, &cache, &hash, &|_| {}).unwrap();
        fs::remove_file(&source).unwrap();
        assert_eq!(cache_model(&source, &cache, &hash, &|_| {}).unwrap(), model);
        fs::write(&model, b"corrupt").unwrap();
        fs::write(&source, b"model").unwrap();
        cache_model(&source, &cache, &hash, &|_| {}).unwrap();
        assert_eq!(fs::read(model).unwrap(), b"model");
    }
    #[test]
    fn temp_copy_verifies_bytes_and_cleans_up() {
        let dir =
            std::env::temp_dir().join(format!("ai-copy-test-{}-{}", std::process::id(), stamp()));
        fs::create_dir(&dir).unwrap();
        {
            let _guard = TempModel(dir.clone());
            let source = dir.join("source");
            fs::write(&source, b"verified model").unwrap();
            let hash = format!("{:x}", Sha256::digest(b"verified model"));
            let target = dir.join("target");
            copy_verified(&source, &target, &hash, &|_| {}).unwrap();
            assert_eq!(fs::read(&target).unwrap(), b"verified model");
            assert!(copy_verified(&source, &target, &hash, &|_| {}).is_err());
            assert!(copy_verified(&source, &dir.join("bad"), &"0".repeat(64), &|_| {}).is_err());
        }
        assert!(!dir.exists());
    }
    #[test]
    fn rejects_incomplete_or_wrong_workload() {
        let c: Config = serde_json::from_str(include_str!("../config/benchmark.json")).unwrap();
        assert!(parse(b"[]", &c, true).is_err());
        assert!(parse(b"not json", &c, true).is_err());
        let mut rows = vec![];
        for (p, n) in [(512, 0), (0, 128)] {
            rows.push(
                json!({"n_prompt":p,"n_gen":n,"n_depth":4096,"avg_ts":30.0,"stddev_ts":1.0,
                "samples_ts":[29.0,30.0,31.0],"build_commit":"test","n_gpu_layers":0}),
            );
        }
        assert_eq!(
            parse(&serde_json::to_vec(&rows).unwrap(), &c, true).unwrap()["backend"],
            "CPU"
        );
        rows[0]["n_depth"] = json!(0);
        assert!(parse(&serde_json::to_vec(&rows).unwrap(), &c, true).is_err());
    }
}
