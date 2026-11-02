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
    let layers = generation["n_gpu_layers"]
        .as_i64()
        .context("Missing offload configuration")?;
    // ponytail: compiled backends do not prove offload; retain raw GPU metadata and label requested offload explicitly.
    let backend = if cpu || layers == 0 || generation["backends"].as_str() == Some("CPU") {
        "CPU".to_string()
    } else {
        format!(
            "{} (offload requested; verify engine log)",
            generation["backends"].as_str().unwrap_or("Unknown")
        )
    };
    Ok(
        json!({"generation_tps":generation["avg_ts"],"prompt_tps":prompt["avg_ts"],
        "stddev_tps":generation["stddev_ts"],"samples_tps":generation["samples_ts"],"backend":backend,
        "engine_commit":generation["build_commit"],"load_seconds":null,"ttft_seconds":null,"raw":rows}),
    )
}

pub fn run(root: &Path, cpu: bool) -> Result<Value> {
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
    let engine = c.engine(root);
    if !engine.is_file() {
        bail!(
            "llama-bench missing: {}. Run preparation script on a development machine.",
            engine.display()
        );
    }
    let model = root.join(&c.model_path);
    let mut f = File::open(&model).context("4B GGUF missing. Run preparation script first.")?;
    eprintln!("Verifying model SHA-256...");
    let mut hash = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    if format!("{:x}", hash.finalize()) != c.model_sha256.to_lowercase() {
        bail!("Model checksum mismatch; benchmark refused");
    }
    let hardware = detect();
    let required = fs::metadata(&model)?.len() + 1024 * 1024 * 1024;
    let available = hardware["ram_available_bytes"].as_u64().unwrap_or(0);
    if available < required {
        bail!(
            "Insufficient available RAM: {available} bytes; conservative minimum {required} bytes"
        );
    }
    let threads = if c.threads == 0 {
        hardware["cores"].as_u64().unwrap_or(1).max(1) as usize
    } else {
        c.threads
    };
    let started = Instant::now();
    let invoke = |use_cpu: bool, suffix: &str| -> Result<Value> {
        let log = File::create(root.join(format!("results/engine-{suffix}.log")))?;
        eprintln!(
            "Loading model, warmup, then {} measured runs ({}).",
            c.runs,
            if use_cpu { "CPU" } else { "automatic GPU" }
        );
        let output = Command::new(&engine)
            .current_dir(engine.parent().context("Invalid engine path")?)
            .arg("-m")
            .arg(&model)
            .args([
                "-o",
                "json",
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
        parse(&output.stdout, &c, use_cpu)
    };
    let mut warnings = vec!["Synthetic llama-bench token workload; tokenization/sampling excluded. Context depth is prefilled KV, not a chat context limit.".to_string(),
        "Load time, TTFT and peak RAM/VRAM are not measured by this runner; reported as unknown, never inferred from throughput.".to_string()];
    let mut measured = match invoke(cpu, "primary") {
        Ok(v) => v,
        Err(e) if !cpu && c.gpu_layers != 0 => {
            warnings.push(format!("GPU attempt failed: {e}. Retried on CPU."));
            invoke(true, "cpu-fallback")?
        }
        Err(e) => return Err(e),
    };
    measured["total_seconds"] = json!(started.elapsed().as_secs_f64());
    let db = dataset::open(root)?;
    let recommendation = dataset::recommend(&db, &c, &hardware, &measured)?;
    let report = json!({"id":stamp().to_string(),"hardware":hardware,"model":c,"benchmark":measured,
        "recommendation":recommendation,"warnings":warnings});
    let text = serde_json::to_string_pretty(&report)?;
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
