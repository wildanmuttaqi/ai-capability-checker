use crate::{benchmark, dataset, detect, export, latest, Config};
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tiny_http::{Header, Method, Response, Server, StatusCode};

pub fn serve(root: PathBuf, port: u16) -> Result<()> {
    let config = Config::load(&root)?;
    let server = Server::http(("127.0.0.1", port)).map_err(|e| anyhow!("{e}"))?;
    let address = server
        .server_addr()
        .to_ip()
        .ok_or_else(|| anyhow!("No listener address"))?;
    let host = address.to_string();
    let job = Arc::new(Mutex::new(json!({"state":"idle","message":"Ready"})));
    let mut hardware = Value::Null;
    println!("AI Capability Checker: http://{host}\nOffline, loopback only. Ctrl+C to stop.");
    for request in server.incoming_requests() {
        let request_host = request
            .headers()
            .iter()
            .find(|h| h.field.equiv("Host"))
            .map(|h| h.value.as_str());
        let origin = request
            .headers()
            .iter()
            .find(|h| h.field.equiv("Origin"))
            .map(|h| h.value.as_str());
        let expected_origin = format!("http://{host}");
        if request_host != Some(host.as_str()) || origin.is_some_and(|o| o != expected_origin) {
            let _ = request
                .respond(Response::from_string("Forbidden origin/host").with_status_code(403));
            continue;
        }
        if request.method() == &Method::Post
            && !request
                .headers()
                .iter()
                .any(|h| h.field.equiv("X-AI-Checker") && h.value.as_str() == "1")
        {
            let _ = request
                .respond(Response::from_string("Missing request header").with_status_code(403));
            continue;
        }
        let url = request.url().to_string();
        let result: Result<(String, &str)> = (|| match (request.method(), url.as_str()) {
            (&Method::Get, "/") => Ok((
                include_str!("../ui/index.html").to_string(),
                "text/html; charset=utf-8",
            )),
            (&Method::Get, "/api/status") => {
                let db = dataset::open(&root)?;
                let count: u64 = db.query_row(
                    "SELECT (SELECT COUNT(*) FROM measured)+(SELECT COUNT(*) FROM atlas)",
                    [],
                    |r| r.get(0),
                )?;
                Ok((json!({"hardware":hardware,"config":config,
                        "readiness":{"model":root.join(&config.model_path).is_file(),"engine":config.engine(&root).is_file(),"dataset_records":count},
                        "job":job.lock().map_err(|_|anyhow!("Job lock poisoned"))?.clone(),"report":latest(&root).ok()}).to_string(),"application/json"))
            }
            (&Method::Post, "/api/detect") => {
                hardware = detect();
                Ok((hardware.to_string(), "application/json"))
            }
            (&Method::Post, "/api/benchmark") => {
                let mut state = job.lock().map_err(|_| anyhow!("Job lock poisoned"))?;
                if state["state"] == "running" {
                    return Err(anyhow!("Benchmark already running"));
                }
                *state = json!({"state":"running","message":"Verifying model, loading, warming up and measuring. This can take several minutes."});
                drop(state);
                let job = Arc::clone(&job);
                let root = root.clone();
                std::thread::spawn(move || {
                    let result = std::panic::catch_unwind(|| benchmark::run(&root, false));
                    if let Ok(mut state) = job.lock() {
                        *state = match result {
                            Ok(Ok(_)) => json!({"state":"done","message":"Benchmark complete"}),
                            Ok(Err(e)) => json!({"state":"error","message":format!("{e:#}")}),
                            Err(_) => {
                                json!({"state":"error","message":"Benchmark worker stopped unexpectedly"})
                            }
                        };
                    }
                });
                Ok((json!({"started":true}).to_string(), "application/json"))
            }
            (&Method::Get, u) if u.starts_with("/api/export?format=") => {
                export(&latest(&root)?, u.trim_start_matches("/api/export?format="))
            }
            _ => Err(anyhow!("Unknown route")),
        })();
        let (body, mime, code) = match result {
            Ok((b, m)) => (b, m, 200),
            Err(e) => (
                json!({"error":format!("{e:#}")}).to_string(),
                "application/json",
                400,
            ),
        };
        let mut response = Response::from_string(body).with_status_code(StatusCode(code));
        for (name,value) in [("Content-Type",mime),("Cache-Control","no-store"),("X-Content-Type-Options","nosniff"),
            ("Content-Security-Policy","default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")] {
            if let Ok(h)=Header::from_bytes(name,value) {response.add_header(h);}
        }
        let _ = request.respond(response);
    }
    Ok(())
}
