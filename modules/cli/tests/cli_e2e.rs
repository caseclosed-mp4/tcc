use std::process::Command;

fn tcc_bin() -> String {
    std::env::var("CARGO_BIN_EXE_tcc")
        .unwrap_or_else(|_| env!("CARGO_BIN_EXE_tcc").to_string())
}

#[test]
fn help_lists_commands() {
    let out = Command::new(tcc_bin()).arg("help").output().expect("run tcc help");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "help failed: {}", stdout);
    assert!(stdout.contains("demo"));
    assert!(stdout.contains("publish"));
    assert!(stdout.contains("query"));
}

#[test]
fn demo_runs_full_loop() {
    let out = Command::new(tcc_bin()).arg("demo").output().expect("run tcc demo");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "demo failed: {}", stdout);
    assert!(stdout.contains("Publishing hypothesis"));
    assert!(stdout.contains("randomized trials"));
    assert!(stdout.contains("gossip network"));
    assert!(stdout.contains("causal question"));
    assert!(stdout.contains("differential privacy"));
    assert!(stdout.contains("every peer now holds the full causal DAG"));
}

#[test]
fn publish_prints_node_id() {
    let published = Command::new(tcc_bin())
        .args([
            "publish",
            "--treatment",
            "evening_walk",
            "--outcome",
            "sleep_quality",
            "--positive",
        ])
        .output()
        .expect("publish");
    let stdout = String::from_utf8_lossy(&published.stdout);
    assert!(published.status.success(), "publish failed: {}", stdout);
    assert!(stdout.contains("published claim evening_walk -> sleep_quality"));
    assert!(stdout.contains("node:"));
}

#[test]
fn list_contains_seed_claims() {
    let list = Command::new(tcc_bin()).arg("list").output().expect("list");
    let stdout = String::from_utf8_lossy(&list.stdout);
    assert!(list.status.success());
    assert!(stdout.contains("screen_time_after_9pm"));
    assert!(stdout.contains("total claims"));
}

#[test]
fn query_returns_answer() {
    let query = Command::new(tcc_bin())
        .args(["query", "screen_time_after_9pm", "sleep_onset_latency"])
        .output()
        .expect("query");
    let stdout = String::from_utf8_lossy(&query.stdout);
    assert!(query.status.success(), "query failed: {}", stdout);
    assert!(stdout.contains("Expected effect"));
    assert!(stdout.contains("participants"));
}
