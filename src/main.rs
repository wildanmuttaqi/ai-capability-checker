mod benchmark;
mod dataset;
mod web;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Parser)]
#[command(
    version,
    about = "Portable, offline, measured Local LLM capability checker"
)]
struct Cli {
    #[arg(long, global = true)]
    root: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Detect,
    Benchmark {
        #[arg(long, default_value = "4b")]
        model: String,
        #[arg(long)]
        cpu: bool,
    },
    Report,
    /// Recompute recommendations using the latest report's original hardware and 4B measurement.
    Recommend,
    Export {
        #[arg(long, default_value = "json", value_parser = ["json", "csv", "html", "txt"])]
        format: String,
    },
    DatasetImport {
        file: PathBuf,
    },
    Ui {
        #[arg(long, default_value_t = 8787)]
        port: u16,
    },
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Config {
    pub model_name: String,
    pub model_family: String,
    pub parameters_b: f64,
    pub quantization: String,
    pub model_path: String,
    pub model_sha256: String,
    pub model_revision: String,
    pub model_url: String,
    pub license: String,
    pub engine_path: String,
    pub prompt_tokens: u32,
    pub generation_tokens: u32,
    pub context_depth: u32,
    pub runs: u32,
    pub threads: usize,
    pub gpu_layers: i32,
}

impl Config {
    pub fn load(root: &Path) -> Result<Self> {
        let c: Self = serde_json::from_slice(&fs::read(root.join("config/benchmark.json"))?)?;
        if !(3..=5).contains(&c.runs)
            || c.prompt_tokens == 0
            || c.prompt_tokens > 32768
            || c.generation_tokens == 0
            || c.generation_tokens > 4096
            || c.context_depth > 32768
            || c.model_sha256.len() != 64
            || !c.model_sha256.bytes().all(|b| b.is_ascii_hexdigit())
            || c.parameters_b != 4.0
            || c.quantization != "Q4_K_M"
            || c.gpu_layers < -1
        {
            bail!("Invalid standardized benchmark configuration");
        }
        Ok(c)
    }
    pub fn engine(&self, root: &Path) -> PathBuf {
        let legacy = root.join(
            self.engine_path
                .replace(
                    "{platform}",
                    if cfg!(windows) { "windows" } else { "linux" },
                )
                .replace("{exe}", if cfg!(windows) { ".exe" } else { "" }),
        );
        let baseline = legacy
            .parent()
            .unwrap()
            .join("cpu")
            .join(legacy.file_name().unwrap());
        // Existing Linux CPU bundles remain usable; new Windows bundles require a distinct baseline.
        if cfg!(windows) || baseline.is_file() {
            baseline
        } else {
            legacy
        }
    }

    pub fn cuda_engine(&self, root: &Path) -> PathBuf {
        root.join("engine/llama.cpp")
            .join(if cfg!(windows) { "windows" } else { "linux" })
            .join("cuda")
            .join(if cfg!(windows) {
                "llama-bench.exe"
            } else {
                "llama-bench"
            })
    }
}

pub fn detect() -> Value {
    use sysinfo::{Disks, System};
    let mut s = System::new();
    s.refresh_cpu_all();
    s.refresh_memory();
    let cpu = s.cpus().first();
    let mut gpu = Vec::new();
    if let Ok(o) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        if o.status.success() {
            for line in String::from_utf8_lossy(&o.stdout).lines() {
                let p: Vec<_> = line.split(',').map(str::trim).collect();
                if p.len() == 3 {
                    gpu.push(json!({"name":p[0],"vram_bytes":p[1].parse::<u64>().ok().and_then(|n|n.checked_mul(1024*1024)),"driver":p[2],"cuda_probe":true}));
                }
            }
        }
    }
    #[cfg(target_os = "linux")]
    if let Ok(entries) = fs::read_dir("/sys/class/drm") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with("card") || name.contains('-') {
                continue;
            }
            let d = entry.path().join("device");
            let vendor = fs::read_to_string(d.join("vendor")).unwrap_or_default();
            if vendor.trim() == "0x10de" && !gpu.is_empty() {
                continue;
            }
            if vendor.is_empty() {
                continue;
            }
            let device = fs::read_to_string(d.join("device")).unwrap_or_default();
            let driver = fs::read_link(d.join("driver"))
                .ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));
            let vram = fs::read_to_string(d.join("mem_info_vram_total"))
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok());
            gpu.push(json!({"name":format!("PCI {}:{}",vendor.trim(),device.trim()),"vram_bytes":vram,"driver":driver}));
        }
    }
    #[cfg(windows)]
    let devices = windows_devices();
    #[cfg(windows)]
    if let Some(rows) = devices["gpu"].as_array() {
        for row in rows {
            if !gpu.iter().any(|g| g["name"] == row["name"]) {
                gpu.push(row.clone());
            }
        }
    }
    #[cfg(target_os = "linux")]
    let npu: Vec<String> = fs::read_dir("/sys/class/accel")
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| {
            let driver = fs::read_link(entry.path().join("device/driver"))
                .ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));
            format!(
                "{}{}",
                entry.file_name().to_string_lossy(),
                driver.map(|d| format!(" ({d})")).unwrap_or_default()
            )
        })
        .collect();
    #[cfg(windows)]
    let npu: Vec<String> = devices["npu"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|r| r.as_str().map(str::to_owned))
        .collect();
    let temp = std::env::temp_dir();
    let disks = Disks::new_with_refreshed_list();
    let free = disks
        .iter()
        .filter(|d| temp.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space());
    json!({"cpu":cpu.map(|c|c.brand()),"vendor":cpu.map(|c|c.vendor_id()),"cores":s.physical_core_count(),
        "threads":s.cpus().len(),"frequency_mhz":cpu.map(|c|c.frequency()),"architecture":std::env::consts::ARCH,
        "os":System::long_os_version(),"ram_total_bytes":s.total_memory(),"ram_available_bytes":s.available_memory(),
        "gpu":gpu,"npu":npu,"npu_status":"Not supported by this llama.cpp runner (even if detected); no NPU benchmark is available", "tops":null,"temp_dir":temp,"free_storage_bytes":free,
        "notes":["TOPS unknown: no reliable OS measurement. NPU discovery does not imply llama.cpp support.","GPU enumeration is best-effort; backend availability is verified by running the engine."]})
}

#[cfg(windows)]
fn windows_devices() -> Value {
    // ponytail: driver registry memory is informational, not a GPU allocation budget; use DXGI for budget/usage telemetry.
    let script = r#"
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$cards = @(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Video\*\0000' -ErrorAction SilentlyContinue)
$gpu = @(Get-CimInstance Win32_VideoController | ForEach-Object {
    $controller = $_
    $card = $cards | Where-Object { $_.DriverDesc -eq $controller.Name -and $_.'HardwareInformation.qwMemorySize' -is [ValueType] } | Select-Object -First 1
    $memory = $null
    $source = 'Unavailable; integrated GPUs may use shared RAM'
    if ($null -ne $card -and [UInt64]$card.'HardwareInformation.qwMemorySize' -gt 0) {
        $memory = [UInt64]$card.'HardwareInformation.qwMemorySize'
        $source = 'Driver registry qwMemorySize; not available VRAM or shared RAM budget'
    }
    # Win32_VideoController.AdapterRAM is UInt32 and can truncate at 4 GiB; do not use it as a fallback.
    @{name=$controller.Name;driver=$controller.DriverVersion;vram_bytes=$memory;vram_source=$source}
})
$npu = @(Get-CimInstance Win32_PnPEntity | Where-Object {
    $_.Present -ne $false -and $_.ConfigManagerErrorCode -eq 0 -and
    ($_.PNPClass -eq 'NeuralProcessor' -or $_.Name -match '(?i)\bNPU\b|Neural Processing|Neural Processor|Intel\(R\) AI Boost|AMD.*IPU|Qualcomm.*Hexagon')
} | ForEach-Object { $_.Name } | Sort-Object -Unique)
@{gpu=$gpu;npu=$npu} | ConvertTo-Json -Depth 4 -Compress
"#;
    std::process::Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", script])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| serde_json::from_slice(&o.stdout).ok())
        .unwrap_or(Value::Null)
}

pub fn latest(root: &Path) -> Result<Value> {
    let db = dataset::open(root)?;
    let text: String = db
        .query_row(
            "SELECT json FROM reports ORDER BY rowid DESC LIMIT 1",
            [],
            |r| r.get(0),
        )
        .context("No benchmark report yet. Run benchmark first.")?;
    Ok(serde_json::from_str(&text)?)
}

pub fn stamp() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

pub fn reanalyze(root: &Path) -> Result<Value> {
    let mut report = latest(root)?;
    let c: Config = serde_json::from_value(report["model"].clone())?;
    if c.parameters_b != 4.0 || c.quantization != "Q4_K_M" {
        bail!("Reanalysis requires a recorded 4B Q4_K_M benchmark");
    }
    let db = dataset::open(root)?;
    report["recommendation"] =
        dataset::recommend(&db, &c, &report["hardware"], &report["benchmark"])?;
    report["derived_from_report"] = report["id"].clone();
    report["id"] = json!(stamp().to_string());
    report["report_type"] =
        json!("recommendation refresh; original benchmark and hardware snapshot retained");
    let text = serde_json::to_string_pretty(&report)?;
    // SQLite remains the source of truth; original reports are never replaced.
    db.execute("INSERT INTO reports(json) VALUES (?1)", [&text])?;
    Ok(report)
}

pub fn export(report: &Value, format: &str) -> Result<(String, &'static str)> {
    let text = serde_json::to_string_pretty(report)?;
    Ok(match format {
        "json" => (text, "application/json"),
        "txt" => (text, "text/plain; charset=utf-8"),
        "html" => (format!("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>AI Capability Report</title><style>body{{background:#0c1220;color:#e2e8f0;font:16px system-ui;margin:3rem}}pre{{white-space:pre-wrap}}</style><h1>AI Capability Checker</h1><pre>{}</pre></html>",text.replace('&',"&amp;").replace('<',"&lt;").replace('>',"&gt;")), "text/html; charset=utf-8"),
        "csv" => {
            let b = &report["benchmark"];
            (format!("generation_tps,prompt_tps,stddev_tps,total_seconds\r\n{},{},{},{}\r\n",b["generation_tps"],b["prompt_tps"],b["stddev_tps"],b["total_seconds"]), "text/csv; charset=utf-8")
        },
        _ => bail!("Unsupported export format"),
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if !cfg!(any(target_os = "windows", target_os = "linux")) || std::env::consts::ARCH != "x86_64"
    {
        bail!("Supported platforms: Windows x64 and Linux x64");
    }
    let root = match cli.root {
        Some(p) => p.canonicalize()?,
        None => {
            let exe = std::env::current_exe()?;
            exe.ancestors()
                .skip(1)
                .find(|p| p.join("config/benchmark.json").is_file())
                .map(Path::to_path_buf)
                .unwrap_or(std::env::current_dir()?)
        }
    };
    match cli.command.unwrap_or(Commands::Ui { port: 8787 }) {
        Commands::Detect => println!("{}", serde_json::to_string_pretty(&detect())?),
        Commands::Benchmark { model, cpu } => {
            if model != "4b" {
                bail!("Only the standardized 4b model is supported");
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&benchmark::run(&root, cpu)?)?
            );
        }
        Commands::Report => println!("{}", serde_json::to_string_pretty(&latest(&root)?)?),
        Commands::Recommend => println!("{}", serde_json::to_string_pretty(&reanalyze(&root)?)?),
        Commands::Export { format } => print!("{}", export(&latest(&root)?, &format)?.0),
        Commands::DatasetImport { file } => println!(
            "Imported {} measured records",
            dataset::import(&root, &file)?
        ),
        Commands::Ui { port } => web::serve(root, port)?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_html_escapes_report_content() {
        let report = json!({"hardware":{"cpu":"<script>alert(1)</script> &"}});
        let (html, mime) = export(&report, "html").unwrap();
        assert_eq!(mime, "text/html; charset=utf-8");
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;"));
        assert!(export(&report, "unsupported").is_err());
    }
}
