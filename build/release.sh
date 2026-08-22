#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v cargo >/dev/null 2>&1; then
    CARGO=cargo
elif [ -x "$HOME/.cargo/bin/cargo" ]; then
    CARGO="$HOME/.cargo/bin/cargo"
else
    echo "error: cargo not found. Install Rust from https://rustup.rs" >&2
    exit 1
fi

echo "building release workspace..."
"$CARGO" build --release --workspace

BIN="$ROOT/target/release/tcc"
if [ -f "$BIN" ]; then
    echo ""
    echo "built $BIN"
    "$BIN" help || true
fi
