use crate::Config;
use anyhow::{bail, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::Path,
};

pub fn open(root: &Path) -> Result<Connection> {
    fs::create_dir_all(root.join("dataset"))?;
    let db = Connection::open(root.join("dataset/benchmark.db"))?;
    db.busy_timeout(std::time::Duration::from_secs(5))?;
    db.execute_batch(
        "PRAGMA user_version=1;
        CREATE TABLE IF NOT EXISTS reports (json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS measured (id TEXT PRIMARY KEY, json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS atlas (id TEXT PRIMARY KEY, json TEXT NOT NULL);",
    )?;
    Ok(db)
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct Record {
    id: String,
    machine_id: String,
    hardware: String,
    ram_bytes: u64,
    model_family: String,
    model_sha256: String,
    parameters_b: f64,
    quantization: String,
    backend: String,
    engine_commit: String,
    context_depth: u32,
    prompt_tokens: u32,
    generation_tokens: u32,
    generation_tps: f64,
    source: String,
    date: String,
    kind: String,
}

pub fn import(root: &Path, path: &Path) -> Result<usize> {
    if fs::metadata(path)?.len() > 64 * 1024 * 1024 {
        bail!("Dataset import exceeds 64 MiB limit");
    }
    let input: Value = serde_json::from_slice(&fs::read(path)?)?;
    if input.is_object() {
        return import_atlas(root, &input);
    }
    let records: Vec<Record> = serde_json::from_value(input)?;
    let mut ids = HashSet::new();
    for r in &records {
        if r.kind != "measured"
            || r.id.trim().is_empty()
            || r.machine_id.trim().is_empty()
            || r.hardware.trim().is_empty()
            || r.source.trim().is_empty()
            || r.date.trim().is_empty()
            || r.backend.trim().is_empty()
            || r.engine_commit.trim().is_empty()
            || r.model_family.trim().is_empty()
            || r.model_sha256.len() != 64
            || !r.model_sha256.bytes().all(|b| b.is_ascii_hexdigit())
            || r.quantization != "Q4_K_M"
            || r.ram_bytes == 0
            || r.prompt_tokens == 0
            || r.generation_tokens == 0
            || !r.generation_tps.is_finite()
            || r.generation_tps <= 0.0
            || !r.parameters_b.is_finite()
            || !(0.1..=1000.0).contains(&r.parameters_b)
            || !ids.insert(&r.id)
        {
            bail!(
                "Invalid, estimated, or duplicate benchmark record: {}",
                r.id
            );
        }
    }
    let mut db = open(root)?;
    let tx = db.transaction()?;
    for r in &records {
        let text = serde_json::to_string(r)?;
        let existing: Option<String> = tx
            .query_row("SELECT json FROM measured WHERE id=?1", [&r.id], |r| {
                r.get(0)
            })
            .optional()?;
        if let Some(old) = existing {
            if old != text {
                bail!(
                    "Record {} already exists with different content; import rolled back",
                    r.id
                );
            }
        } else {
            tx.execute(
                "INSERT INTO measured(id,json) VALUES (?1,?2)",
                params![r.id, text],
            )?;
        }
    }
    tx.commit()?;
    Ok(records.len())
}

fn atlas_number(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_str()?.parse().ok())
        .filter(|n| n.is_finite())
}

fn import_atlas(root: &Path, input: &Value) -> Result<usize> {
    let rows = input["records"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Missing Token Atlas records"))?;
    if input["count"].as_u64() != Some(rows.len() as u64) {
        bail!("Token Atlas count mismatch");
    }
    let mut ids = HashSet::new();
    let mut db = open(root)?;
    let tx = db.transaction()?;
    for r in rows {
        let nonempty = |k: &str| r[k].as_str().is_some_and(|s| !s.trim().is_empty());
        let positive = |k: &str| atlas_number(&r[k]).is_some_and(|v| v > 0.0);
        let id = r["id"].as_str().unwrap_or("");
        let url = r["source_url"].as_str().unwrap_or("");
        if ![
            "id",
            "model",
            "hardware",
            "backend",
            "source_url",
            "source_name",
            "retrieved",
            "quote",
        ]
        .iter()
        .all(|k| nonempty(k))
            || !matches!(r["provenance"].as_str(), Some("community" | "sourced"))
            || !r["scope"].as_str().unwrap_or("").is_empty()
            || !(url.starts_with("https://") || url.starts_with("http://"))
            || (!positive("tps") && !positive("pp_tps"))
            || !ids.insert(id)
        {
            bail!("Invalid, estimated, cluster, or duplicate Token Atlas record: {id}");
        }
        let text = serde_json::to_string(r)?;
        let old: Option<String> = tx
            .query_row("SELECT json FROM atlas WHERE id=?1", [id], |r| r.get(0))
            .optional()?;
        if let Some(old) = old {
            if old != text {
                bail!("Conflicting Token Atlas record {id}; transaction rolled back");
            }
        } else {
            tx.execute(
                "INSERT INTO atlas(id,json) VALUES (?1,?2)",
                params![id, text],
            )?;
        }
    }
    tx.commit()?;
    Ok(rows.len())
}

fn qwen_size(model: &str) -> Option<u32> {
    let name = model.to_ascii_lowercase();
    let rest = name
        .strip_prefix("qwen3-")
        .or_else(|| name.strip_prefix("qwen3 "))?;
    let (size, suffix) = rest.split_once('b')?;
    if !suffix.is_empty() && !suffix.starts_with(['-', ' ']) {
        return None;
    }
    if suffix.contains("-a") {
        return None;
    }
    size.parse().ok()
}

fn atlas_reference(db: &Connection, c: &Config) -> Result<Value> {
    let total: u64 = db.query_row("SELECT COUNT(*) FROM atlas", [], |r| r.get(0))?;
    let mut stmt=db.prepare("SELECT json FROM atlas WHERE lower(json_extract(json,'$.model')) LIKE ?1 AND json_extract(json,'$.quant')=?2 ORDER BY id")?;
    let candidates = stmt.query_map(
        params![
            format!("{}%", c.model_family.to_lowercase()),
            c.quantization
        ],
        |r| r.get::<_, String>(0),
    )?;
    let mut examples = Vec::new();
    let mut matching = 0;
    for row in candidates {
        let r: Value = serde_json::from_str(&row?)?;
        if qwen_size(r["model"].as_str().unwrap_or("")) != Some(4)
            || !r["flags"].as_array().is_some_and(|f| f.is_empty())
            || !atlas_number(&r["tps"]).is_some_and(|n| n > 0.0)
        {
            continue;
        }
        matching += 1;
        if examples.len() < 5 {
            examples.push(r);
        }
    }
    Ok(
        json!({"total_records":total,"same_family_quant_records":matching,"examples":examples,
        "usage":"External measured reference only, not comparable local predictions. Hardware, build, model hash and workload may differ; missing metadata remains unknown."}),
    )
}

fn atlas_cpu(r: &Value) -> bool {
    matches!(r["backend"].as_str(), Some("CPU" | "llama.cpp (CPU)"))
}

fn atlas_workload(r: &Value, c: &Config) -> bool {
    [
        ("ctx", c.context_depth),
        ("pp_tokens", c.prompt_tokens),
        ("tg_tokens", c.generation_tokens),
    ]
    .iter()
    .all(|(key, expected)| r[*key].is_null() || atlas_number(&r[*key]) == Some(*expected as f64))
        && (r["batch"].is_null() || atlas_number(&r["batch"]) == Some(1.0))
}

fn atlas_pairs(a: &Value, b: &Value) -> bool {
    ["source_url", "hardware", "backend", "quant"]
        .iter()
        .all(|k| a[*k] == b[*k])
        && [
            "ctx",
            "pp_tokens",
            "tg_tokens",
            "batch",
            "threads",
            "engine_commit",
            "settings",
        ]
        .iter()
        .all(|k| a[*k] == b[*k])
}

fn memory_candidate(size: u32, c: &Config, hardware: &Value, backend: &str) -> Value {
    // ponytail: dense Q4_K_M planning envelope, not model-specific KV measurement; inspect GGUF architecture for tighter estimates.
    let gib = 1_073_741_824.0;
    let weights = size as f64 * 0.70;
    let kv = 4.0 * (c.context_depth + c.prompt_tokens + c.generation_tokens) as f64 / 4096.0;
    let required = ((weights + kv + 2.0) * gib).ceil() as u64;
    let available = hardware["ram_available_bytes"].as_u64();
    let status = if backend != "CPU" {
        "unknown"
    } else {
        match available {
            Some(n) if n >= required => "candidate",
            Some(_) => "memory pressure",
            None => "unknown",
        }
    };
    json!({"status":status,"estimated_required_bytes":required,"available_bytes":available,
        "basis":"Estimated dense Q4_K_M weights (0.70 GiB/B) + KV reserve (4 GiB per 4096 tokens) + 2 GiB overhead. Available RAM snapshot only; not guaranteed fit. GPU/shared memory is not added; non-CPU placement unknown."})
}

fn rating(tps: f64) -> &'static str {
    if tps >= 30.0 {
        "Excellent"
    } else if tps >= 15.0 {
        "Good"
    } else if tps >= 5.0 {
        "Possible"
    } else {
        "Not recommended"
    }
}

fn hardware_assessment(h: &Value, b: &Value) -> Value {
    let backend = b["backend"].as_str().unwrap_or("Unknown");
    let cpu_active = backend == "CPU";
    let gpu = h["gpu"].as_array().cloned().unwrap_or_default();
    let npu = h["npu"].as_array().cloned().unwrap_or_default();
    json!({
        "cpu":{"name":h["cpu"],"cores":h["cores"],"threads":h["threads"],
            "role":if cpu_active{"Benchmark 4B mengukur jalur CPU dan RAM secara nyata"}else{"CPU terdeteksi; kontribusi terpisah belum diukur"}},
        "gpu":{"devices":gpu,"role":if cpu_active{"GPU tidak digunakan pada benchmark ini; hasil CPU tidak memprediksi performa GPU"}else if backend == "NVIDIA CUDA" {"NVIDIA CUDA layer offload verified from engine stderr; CPU still participates"}else{"Offload unverified; GPU execution must not be assumed"}},
        "vram":{"role":"Kapasitas driver bukan VRAM bebas. Full GPU offload memerlukan bobot + KV cache + buffer muat pada satu GPU; tidak menjumlahkan VRAM antar-GPU atau shared RAM. Kelayakan GPU belum terverifikasi tanpa pengukuran offload."},
        "npu":{"devices":npu,"used_in_benchmark":false,
            "role":"NPU tidak digunakan runner llama.cpp ini; keberadaan NPU/TOPS tidak menambah skor rekomendasi CPU/GPU"},
        "ram":{"available_bytes":h["ram_available_bytes"],"role":"Batas kapasitas host, bukan pengukur kecepatan"},
        "backend":backend,"generation_tps":b["generation_tps"],
        "conclusion":"Kecepatan mengikuti hasil 4B pada backend aktif dan bukti pembanding yang cocok. Identitas CPU/GPU, jalur backend, VRAM dan RAM menentukan relevansi/kapasitas; perangkat yang hanya terdeteksi tidak dianggap sudah diuji."
    })
}

pub fn recommend(
    db: &Connection,
    c: &Config,
    hardware: &Value,
    benchmark: &Value,
) -> Result<Value> {
    let mut stmt = db.prepare("SELECT json FROM measured")?;
    let records: Vec<Record> = stmt
        .query_map([], |r| r.get::<_, String>(0))?
        .map(|r| Ok(serde_json::from_str(&r?)?))
        .collect::<Result<_>>()?;
    let backend = benchmark["backend"].as_str().unwrap_or("");
    let commit = benchmark["engine_commit"].as_str().unwrap_or("");
    let tps = benchmark["generation_tps"].as_f64().unwrap_or(0.0);
    let ram = hardware["ram_total_bytes"].as_u64().unwrap_or(0) as f64;
    let same_workload = |r: &Record| {
        r.model_family == c.model_family
            && r.quantization == c.quantization
            && r.backend == backend
            && r.engine_commit == commit
            && r.context_depth == c.context_depth
            && r.prompt_tokens == c.prompt_tokens
            && r.generation_tokens == c.generation_tokens
    };
    let anchors: Vec<_> = records
        .iter()
        .filter(|r| {
            same_workload(r)
                && r.model_sha256 == c.model_sha256
                && (r.parameters_b - 4.0).abs() < 0.01
                && (r.generation_tps / tps - 1.0).abs() <= 0.2
                && (r.ram_bytes as f64 / ram - 1.0).abs() <= 0.25
        })
        .collect();
    let machines: HashSet<_> = anchors.iter().map(|r| r.machine_id.as_str()).collect();
    let mut groups: BTreeMap<u32, BTreeMap<&str, Vec<f64>>> = BTreeMap::new();
    for r in records.iter().filter(|r| {
        same_workload(r) && r.parameters_b > 4.0 && machines.contains(r.machine_id.as_str())
    }) {
        groups
            .entry((r.parameters_b * 10.0).round() as u32)
            .or_default()
            .entry(&r.machine_id)
            .or_default()
            .push(r.generation_tps);
    }
    let mut atlas_stmt =
        db.prepare("SELECT json FROM atlas WHERE json_extract(json,'$.quant')='Q4_K_M'")?;
    let atlas: Vec<Value> = atlas_stmt
        .query_map([], |r| r.get::<_, String>(0))?
        .map(|r| Ok(serde_json::from_str(&r?)?))
        .collect::<Result<_>>()?;
    let eligible = |r: &&Value| {
        backend == "CPU"
            && atlas_cpu(r)
            && atlas_workload(r, c)
            && r["flags"].as_array().is_some_and(|f| f.is_empty())
            && atlas_number(&r["tps"]).is_some_and(|n| n > 0.0)
    };
    let atlas: Vec<_> = atlas.iter().filter(eligible).collect();
    let atlas_anchors: Vec<_> = atlas
        .iter()
        .filter(|r| {
            qwen_size(r["model"].as_str().unwrap_or("")) == Some(4)
                && atlas_number(&r["tps"])
                    .is_some_and(|n| tps > 0.0 && (n / tps - 1.0).abs() <= 0.20)
        })
        .collect();
    let mut models = vec![
        json!({"parameters_b":4,"rating":rating(tps),"measured_tps":tps,"samples":c.runs,"confidence":"high","evidence":"direct local measurement","memory":memory_candidate(4,c,hardware,backend)}),
    ];
    let mut min_support = usize::MAX;
    for size in [7u32, 8, 12, 14] {
        let group = groups.get(&(size * 10));
        let mut values: Vec<f64> = group
            .map(|g| {
                g.values()
                    .map(|v| v.iter().sum::<f64>() / v.len() as f64)
                    .collect()
            })
            .unwrap_or_default();
        values.sort_by(f64::total_cmp);
        let n = values.len();
        min_support = min_support.min(n);
        let median = if n == 0 {
            None
        } else if n.is_multiple_of(2) {
            Some((values[n / 2 - 1] + values[n / 2]) / 2.0)
        } else {
            Some(values[n / 2])
        };
        let references: Vec<_> = atlas.iter().filter_map(|r| {
            if qwen_size(r["model"].as_str().unwrap_or("")) != Some(size) { return None; }
            let anchor = atlas_anchors.iter().find(|a| atlas_pairs(a, r));
            let direct = r["hardware"].as_str().is_some_and(|h| hardware["cpu"].as_str().is_some_and(|cpu| h.eq_ignore_ascii_case(cpu)));
            if anchor.is_none() && !direct { return None; }
            Some(json!({"record_id":r["id"],"source_url":r["source_url"],"hardware":r["hardware"],
                "model":r["model"],"tps":atlas_number(&r["tps"]),"anchor_id":anchor.map(|a| &a["id"]),
                "tier":if anchor.is_some(){"source/hardware pair"}else{"exact CPU label"},
                "limitations":["Hardware label is not a unique machine identity; source records may share a machine.",
                    "Context field semantics, threads, model hashes, engine builds and other settings may be unknown. External reference, not predicted local speed."]}))
        }).collect();
        let mut speeds: Vec<f64> = references
            .iter()
            .filter_map(|r| r["tps"].as_f64())
            .collect();
        speeds.sort_by(f64::total_cmp);
        let range = speeds
            .first()
            .zip(speeds.last())
            .map(|(min, max)| json!({"min":min,"max":max}));
        models.push(json!({"parameters_b":size,"rating":median.map(rating).unwrap_or("Insufficient data"),
            "measured_tps":median,"samples":n,"evidence_count":n+references.len(),
            "confidence":if n>=20 {"high"}else if n>=5 {"medium"}else{"low"},
            "reference_tps_range":range,"references":references,"memory":memory_candidate(size,c,hardware,backend),
            "evidence":if median.is_some(){"strict paired measurements; not local"}else if speeds.is_empty(){"memory only; insufficient comparable speed evidence"}else{"Atlas measured reference; local speed unknown"}}));
    }
    // ponytail: exact workload/engine matching only; add validated normalization when a curated dataset supports it.
    let confidence = if min_support >= 20 {
        "high"
    } else if min_support >= 5 {
        "medium"
    } else {
        "low"
    };
    let external_reference = atlas_reference(db, c)?;
    Ok(
        json!({"confidence":confidence,"comparable_benchmarks":machines.len(),"models":models,
        "hardware_assessment":hardware_assessment(hardware,benchmark),
        "external_reference":external_reference,
        "notes":["No TOPS-based or linear parameter extrapolation. No comparable measurements means insufficient data, not a negative recommendation.",
        "Strict evidence: exact 4B hash, family/quant/backend/build/workload; anchor throughput ±20%, RAM ±25%. Atlas CPU evidence requires explicit CPU backend, Q4_K_M, Qwen3 names, compatible known workload; paired source/hardware/backend/settings or exact CPU label. Unknown metadata caps Atlas confidence at low. No GPU/Apple throughput substitution for CPU.",
        "Memory candidates are planning estimates, not speed ratings or guaranteed fit. Insufficient speed evidence is not a negative recommendation. Memory uses available RAM in the recorded hardware snapshot, never RAM + shared VRAM.",
        "Confidence uses independent machine counts for every target size. Ratings describe measured speed (30/15/5 tok/s), not guaranteed memory fit or model quality."]}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detected_accelerators_do_not_inflate_cpu_recommendation() {
        let h = json!({"cpu":"test CPU","gpu":[{"name":"test GPU","vram_bytes":24u64*1_073_741_824}],"npu":["test NPU"]});
        let a = hardware_assessment(&h, &json!({"backend":"CPU","generation_tps":3.0}));
        assert_eq!(a["generation_tps"], 3.0);
        assert_eq!(a["npu"]["used_in_benchmark"], false);
        assert!(a["gpu"]["role"]
            .as_str()
            .unwrap()
            .contains("tidak digunakan"));
        assert_eq!(a["gpu"]["devices"][0]["vram_bytes"], 24u64 * 1_073_741_824);
    }

    #[test]
    fn memory_and_atlas_filters_do_not_invent_speed() {
        let c: Config = serde_json::from_str(include_str!("../config/benchmark.json")).unwrap();
        let h = json!({"ram_available_bytes":14u64*1_073_741_824});
        assert_eq!(memory_candidate(7, &c, &h, "CPU")["status"], "candidate");
        assert_eq!(
            memory_candidate(14, &c, &h, "CPU")["status"],
            "memory pressure"
        );
        assert_eq!(memory_candidate(7, &c, &h, "CUDA")["status"], "unknown");
        assert_eq!(qwen_size("Qwen3-8B"), Some(8));
        assert_eq!(qwen_size("Qwen3.8-Flash-Next"), None);
        let a = json!({"source_url":"https://example.org/run","hardware":"CPU A","backend":"llama.cpp (CPU)","quant":"Q4_K_M"});
        assert!(atlas_cpu(&a));
        assert!(atlas_workload(&a, &c));
        assert!(atlas_pairs(&a, &a));
        let mut b = a.clone();
        b["backend"] = json!("CUDA");
        assert!(!atlas_cpu(&b));
        assert!(!atlas_pairs(&a, &b));
        b = a.clone();
        b["ctx"] = json!(512);
        assert!(!atlas_workload(&b, &c));
    }

    #[test]
    fn atlas_import_preserves_provenance_and_rejects_estimates() {
        let root = std::env::temp_dir().join(format!(
            "atlas-test-{}-{}",
            std::process::id(),
            crate::stamp()
        ));
        let record = json!({"id":"test","model":"Qwen3-4B","hardware":"test CPU","backend":"llama.cpp",
            "source_url":"https://example.org/test","source_name":"unit test","retrieved":"2026-11-02",
            "quote":"10.5","provenance":"community","tps":"10.5","quant":"Q4_K_M","flags":[]});
        let mut input = json!({"count":1,"records":[record]});
        assert_eq!(import_atlas(&root, &input).unwrap(), 1);
        assert_eq!(import_atlas(&root, &input).unwrap(), 1);
        let c: Config = serde_json::from_str(include_str!("../config/benchmark.json")).unwrap();
        let db = open(&root).unwrap();
        let reference = atlas_reference(&db, &c).unwrap();
        assert_eq!(reference["same_family_quant_records"], 1);
        assert_eq!(reference["examples"][0]["tps"], "10.5");
        input["records"][0]["provenance"] = json!("estimated");
        assert!(import_atlas(&root, &input).is_err());
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM measured", [], |r| r.get::<_, u32>(0))
                .unwrap(),
            0
        );
        drop(db);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn import_is_idempotent_and_rejects_conflicting_records_atomically() {
        let root = std::env::temp_dir().join(format!(
            "ai-checker-test-{}-{}",
            std::process::id(),
            crate::stamp()
        ));
        fs::create_dir_all(&root).unwrap();
        let file = root.join("import.json");
        let record = json!({"id":"original","machine_id":"test","hardware":"test",
            "ram_bytes":16_000_000_000u64,"model_family":"qwen3","model_sha256":"a".repeat(64),
            "parameters_b":4.0,"quantization":"Q4_K_M","backend":"CPU","engine_commit":"test",
            "context_depth":4096,"prompt_tokens":512,"generation_tokens":128,"generation_tps":10.0,
            "source":"unit-test-only","date":"2026-11-01","kind":"measured"});
        fs::write(&file, serde_json::to_vec(&vec![&record]).unwrap()).unwrap();
        assert_eq!(import(&root, &file).unwrap(), 1);
        assert_eq!(import(&root, &file).unwrap(), 1);
        let mut added = record.clone();
        added["id"] = json!("new");
        let mut conflicting = record.clone();
        conflicting["generation_tps"] = json!(99.0);
        fs::write(
            &file,
            serde_json::to_vec(&vec![added, conflicting]).unwrap(),
        )
        .unwrap();
        assert!(import(&root, &file).is_err());
        let db = open(&root).unwrap();
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM measured", [], |r| r.get::<_, u32>(0))
                .unwrap(),
            1
        );
        drop(db);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn empty_dataset_never_invents_larger_model_performance() {
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch(
            "CREATE TABLE measured(id TEXT,json TEXT); CREATE TABLE atlas(id TEXT,json TEXT)",
        )
        .unwrap();
        let c: Config = serde_json::from_str(include_str!("../config/benchmark.json")).unwrap();
        let r = recommend(
            &db,
            &c,
            &json!({"ram_total_bytes":16_000_000_000u64}),
            &json!({"backend":"CPU","generation_tps":31.0,"engine_commit":"abc"}),
        )
        .unwrap();
        assert_eq!(r["confidence"], "low");
        assert_eq!(r["models"][0]["rating"], "Excellent");
        assert_eq!(r["models"][1]["rating"], "Insufficient data");
        assert!(r["models"][1]["measured_tps"].is_null());
    }

    #[test]
    fn paired_records_count_machines_not_repeated_runs() {
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch(
            "CREATE TABLE measured(id TEXT,json TEXT); CREATE TABLE atlas(id TEXT,json TEXT)",
        )
        .unwrap();
        let c: Config = serde_json::from_str(include_str!("../config/benchmark.json")).unwrap();
        for (id, size, speed) in [
            ("anchor", 4.0, 31.0),
            ("larger", 8.0, 16.0),
            ("repeat", 8.0, 18.0),
        ] {
            let r = Record {
                id: id.into(),
                machine_id: "same-machine".into(),
                hardware: "test".into(),
                ram_bytes: 16_000_000_000,
                model_family: c.model_family.clone(),
                model_sha256: c.model_sha256.clone(),
                parameters_b: size,
                quantization: c.quantization.clone(),
                backend: "CPU".into(),
                engine_commit: "abc".into(),
                context_depth: c.context_depth,
                prompt_tokens: c.prompt_tokens,
                generation_tokens: c.generation_tokens,
                generation_tps: speed,
                source: "unit-test-only".into(),
                date: "2026-11-01".into(),
                kind: "measured".into(),
            };
            db.execute(
                "INSERT INTO measured VALUES (?1,?2)",
                params![id, serde_json::to_string(&r).unwrap()],
            )
            .unwrap();
        }
        let r = recommend(
            &db,
            &c,
            &json!({"ram_total_bytes":16_000_000_000u64}),
            &json!({"backend":"CPU","generation_tps":31.0,"engine_commit":"abc"}),
        )
        .unwrap();
        assert_eq!(r["comparable_benchmarks"], 1);
        assert_eq!(r["models"][2]["samples"], 1);
        assert_eq!(r["models"][2]["measured_tps"], 17.0);
        assert_eq!(r["confidence"], "low");
        let mismatch = recommend(
            &db,
            &c,
            &json!({"ram_total_bytes":16_000_000_000u64}),
            &json!({"backend":"CPU","generation_tps":31.0,"engine_commit":"different"}),
        )
        .unwrap();
        assert_eq!(mismatch["comparable_benchmarks"], 0);
    }
}
