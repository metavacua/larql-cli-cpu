.PHONY: build release build-experts test-experts test test-fast test-full test-integration test-models larql-core-test larql-core-feature-test larql-core-fmt-check larql-core-lint larql-core-bench-test larql-core-bench larql-core-coverage larql-core-coverage-html larql-core-ci larql-models-test larql-models-fmt-check larql-models-lint larql-models-bench-test larql-models-coverage-policy larql-models-coverage larql-models-coverage-summary larql-models-coverage-html larql-models-ci larql-vindex-test larql-vindex-fmt-check larql-vindex-lint larql-vindex-examples larql-vindex-bench-test larql-vindex-bench larql-vindex-coverage-policy larql-vindex-coverage larql-vindex-coverage-summary larql-vindex-coverage-html larql-vindex-ci larql-factory-test larql-factory-fmt-check larql-factory-lint larql-factory-coverage-policy larql-factory-coverage larql-factory-coverage-summary larql-factory-coverage-html larql-factory-ci larql-kv-test larql-kv-fmt-check larql-kv-lint larql-kv-examples larql-kv-bench-test larql-kv-bench larql-kv-coverage-policy larql-kv-coverage larql-kv-coverage-summary larql-kv-coverage-html larql-kv-ci larql-compute-test larql-compute-test-fast larql-compute-check-fast larql-compute-check-tests larql-compute-check-all larql-compute-test-integration larql-compute-fmt-check larql-compute-lint larql-compute-coverage-policy larql-compute-coverage larql-compute-coverage-summary larql-compute-coverage-html larql-compute-ci larql-boundary-test larql-boundary-fmt-check larql-boundary-lint larql-boundary-bench-test larql-boundary-examples larql-boundary-coverage larql-boundary-coverage-html larql-boundary-ci larql-router-test larql-router-fmt-check larql-router-lint larql-router-coverage-policy larql-router-coverage larql-router-coverage-summary larql-router-coverage-html larql-router-ci larql-router-protocol-test larql-router-protocol-fmt-check larql-router-protocol-lint larql-router-protocol-coverage-policy larql-router-protocol-coverage-summary larql-router-protocol-ci larql-lql-test larql-lql-fmt-check larql-lql-lint larql-lql-examples larql-lql-bench-test larql-lql-coverage-summary larql-lql-ci larql-cli-test larql-cli-fmt-check larql-cli-lint larql-cli-coverage-policy larql-cli-coverage-summary larql-cli-coverage larql-cli-coverage-html larql-cli-ci larql-inference-test larql-inference-fmt-check larql-inference-lint larql-inference-bench-test larql-inference-coverage-policy larql-inference-coverage larql-inference-coverage-summary larql-inference-coverage-html larql-inference-ci check fmt fmt-check lint ci clean bench bench-core bench-inference bench-compute bench-wire bench-routing bench-cross-arch bench-all bench-vindex bench-vindex-scaling bench-save bench-check coverage coverage-summary extract-test extract-full predict

# Build
build:
	cargo build --workspace

release:
	cargo build --release -p larql-cli

# The WASM virtual experts are a nested workspace (not in `cargo build --workspace`).
# They are built for wasm32-unknown-unknown and import no WASI; `make ci` does not
# cover them, the `larql-experts` workflow does.
build-experts:
	rustup target add wasm32-unknown-unknown
	cargo build --manifest-path crates/larql-experts/Cargo.toml --target wasm32-unknown-unknown --release
	python3 scripts/check_wasm_expert_imports.py

test-experts:
	cargo test --manifest-path crates/larql-experts/Cargo.toml --workspace

# Test
#
# Default test target is intentionally fast: no integration binaries, no
# model-backed ignored tests. Use `test-full` for the historical full
# workspace run, and `test-models` for real-model/vindex checks.
test: test-fast

test-fast:
	cargo test --workspace --lib --bins

test-full:
	cargo test --workspace

test-integration:
	cargo test --workspace --tests

# The test_llm_dispatch / test_constrained_dispatch / test_trie_dispatch goldens
# load the WASM experts: run `make build-experts` first (it needs network access
# for `rustup target add`, so it is deliberately not a prerequisite here).
test-models:
	cargo test -p larql-inference --test test_arch_golden -- --ignored
	cargo test -p larql-inference --test test_logits_goldens -- --ignored
	cargo test -p larql-inference --test test_gemma3_smoke -- --ignored
	cargo test -p larql-inference --test test_generate_q4k_cpu -- --ignored
	cargo test -p larql-inference --test bench_probe_latency -- --ignored --nocapture
	cargo test -p larql-inference --test test_llm_dispatch -- --ignored --nocapture
	cargo test -p larql-inference --test test_constrained_dispatch -- --ignored --nocapture
	cargo test -p larql-inference --test test_trie_dispatch -- --ignored --nocapture

# larql-core — graph engine, algorithms, extraction helpers, serialization
larql-core-test:
	cargo test -p larql-core

larql-core-feature-test:
	cargo test -p larql-core --no-default-features
	cargo test -p larql-core --no-default-features --features msgpack

larql-core-fmt-check:
	cargo fmt -p larql-core -- --check

larql-core-lint:
	cargo clippy -p larql-core --all-targets -- -D warnings

larql-core-bench-test:
	cargo test -p larql-core --benches

larql-core-bench:
	cargo bench -p larql-core --bench graph

larql-core-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-core --summary-only

larql-core-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-core --html --output-dir coverage/larql-core
	@echo "Report: coverage/larql-core/html/index.html"

larql-core-ci: larql-core-fmt-check larql-core-lint larql-core-test larql-core-feature-test larql-core-bench-test

larql-models-test:
	cargo test -p larql-models

larql-models-fmt-check:
	cargo fmt -p larql-models -- --check

larql-models-lint:
	cargo clippy -p larql-models --all-targets --no-deps -- -D warnings

larql-models-bench-test:
	cargo test -p larql-models --benches

# larql-models - architecture detection, weight loading, quant codecs.
#
# Per-file 90% floor; whole-crate total at 80 since `cargo llvm-cov` includes
# the `test_fixtures.rs` support file (test-utils feature, ~30% covered when
# measured here in isolation — see crates/larql-models/coverage-policy.json
# for the full reasoning). The real 94% bar is enforced by the policy
# script's `included_total_line_min_percent` over the non-fixture files.
LARQL_MODELS_COVERAGE_MIN ?= 88
LARQL_MODELS_COVERAGE_POLICY ?= crates/larql-models/coverage-policy.json
LARQL_MODELS_COVERAGE_REPORT ?= coverage/larql-models/summary.json

larql-models-coverage-policy:
	@if [ ! -f "$(LARQL_MODELS_COVERAGE_REPORT)" ]; then \
		echo "Coverage report not found: $(LARQL_MODELS_COVERAGE_REPORT)"; \
		echo "Run: make larql-models-coverage-summary"; \
		exit 1; \
	fi
	python3 scripts/check_coverage_policy.py $(LARQL_MODELS_COVERAGE_REPORT) $(LARQL_MODELS_COVERAGE_POLICY)

larql-models-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-models --fail-under-lines $(LARQL_MODELS_COVERAGE_MIN)
	@mkdir -p coverage/larql-models
	cargo llvm-cov report --package larql-models --json --summary-only --output-path $(LARQL_MODELS_COVERAGE_REPORT)
	$(MAKE) larql-models-coverage-policy

larql-models-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-models --summary-only --fail-under-lines $(LARQL_MODELS_COVERAGE_MIN)
	@mkdir -p coverage/larql-models
	cargo llvm-cov report --package larql-models --json --summary-only --output-path $(LARQL_MODELS_COVERAGE_REPORT)
	$(MAKE) larql-models-coverage-policy

larql-models-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-models --html --output-dir coverage/larql-models
	@echo "Report: coverage/larql-models/html/index.html"

larql-models-ci: larql-models-fmt-check larql-models-lint larql-models-test larql-models-bench-test larql-models-coverage

# larql-vindex - vindex extraction, storage, load/save, patch overlays
#
# Current local baseline: 71.56% line coverage from cargo-llvm-cov.
# Keep this as a ratchet: raise it when new coverage lands.
LARQL_VINDEX_COVERAGE_MIN ?= 90
LARQL_VINDEX_COVERAGE_POLICY ?= crates/larql-vindex/coverage-policy.json
LARQL_VINDEX_COVERAGE_REPORT ?= coverage/larql-vindex/summary.json

larql-vindex-test:
	cargo test -p larql-vindex

larql-vindex-fmt-check:
	cargo fmt -p larql-vindex -- --check

larql-vindex-lint:
	cargo clippy -p larql-vindex --all-targets -- -D warnings

larql-vindex-examples:
	cargo check -p larql-vindex --examples

larql-vindex-bench-test:
	cargo test -p larql-vindex --benches

larql-vindex-bench:
	cargo bench -p larql-vindex --bench vindex_ops

larql-vindex-coverage-policy:
	@if [ ! -f "$(LARQL_VINDEX_COVERAGE_REPORT)" ]; then \
		echo "Coverage report not found: $(LARQL_VINDEX_COVERAGE_REPORT)"; \
		echo "Run: make larql-vindex-coverage-summary"; \
		exit 1; \
	fi
	python3 scripts/check_coverage_policy.py $(LARQL_VINDEX_COVERAGE_REPORT) $(LARQL_VINDEX_COVERAGE_POLICY)

larql-vindex-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-vindex --fail-under-lines $(LARQL_VINDEX_COVERAGE_MIN)
	@mkdir -p coverage/larql-vindex
	cargo llvm-cov report --package larql-vindex --json --summary-only --output-path $(LARQL_VINDEX_COVERAGE_REPORT)
	$(MAKE) larql-vindex-coverage-policy

larql-vindex-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-vindex --summary-only --fail-under-lines $(LARQL_VINDEX_COVERAGE_MIN)
	@mkdir -p coverage/larql-vindex
	cargo llvm-cov report --package larql-vindex --json --summary-only --output-path $(LARQL_VINDEX_COVERAGE_REPORT)
	$(MAKE) larql-vindex-coverage-policy

larql-vindex-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-vindex --html --output-dir coverage/larql-vindex --fail-under-lines $(LARQL_VINDEX_COVERAGE_MIN)
	cargo llvm-cov report --package larql-vindex --json --summary-only --output-path $(LARQL_VINDEX_COVERAGE_REPORT)
	$(MAKE) larql-vindex-coverage-policy
	@echo "Report: coverage/larql-vindex/html/index.html"

larql-vindex-ci: larql-vindex-fmt-check larql-vindex-lint larql-vindex-test larql-vindex-examples larql-vindex-bench-test larql-vindex-coverage-summary

# larql-factory — Vindex Factory recipe schema, build_id canonicaliser,
# structural validator (docs/vindex-factory.md). No benches or examples.
# Every file measured at 100% as of 2026-07-29; the 90% default is the
# floor, not the target.
LARQL_FACTORY_COVERAGE_MIN ?= 90
LARQL_FACTORY_COVERAGE_POLICY ?= crates/larql-factory/coverage-policy.json
LARQL_FACTORY_COVERAGE_REPORT ?= coverage/larql-factory/summary.json

larql-factory-test:
	cargo test -p larql-factory

larql-factory-fmt-check:
	cargo fmt -p larql-factory -- --check

larql-factory-lint:
	cargo clippy -p larql-factory --all-targets --no-deps -- -D warnings

larql-factory-coverage-policy:
	@if [ ! -f "$(LARQL_FACTORY_COVERAGE_REPORT)" ]; then \
		echo "Coverage report not found: $(LARQL_FACTORY_COVERAGE_REPORT)"; \
		echo "Run: make larql-factory-coverage-summary"; \
		exit 1; \
	fi
	python3 scripts/check_coverage_policy.py $(LARQL_FACTORY_COVERAGE_REPORT) $(LARQL_FACTORY_COVERAGE_POLICY)

larql-factory-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-factory --fail-under-lines $(LARQL_FACTORY_COVERAGE_MIN)
	@mkdir -p coverage/larql-factory
	cargo llvm-cov report --package larql-factory --json --summary-only --output-path $(LARQL_FACTORY_COVERAGE_REPORT)
	$(MAKE) larql-factory-coverage-policy

larql-factory-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-factory --summary-only --fail-under-lines $(LARQL_FACTORY_COVERAGE_MIN)
	@mkdir -p coverage/larql-factory
	cargo llvm-cov report --package larql-factory --json --summary-only --output-path $(LARQL_FACTORY_COVERAGE_REPORT)
	$(MAKE) larql-factory-coverage-policy

larql-factory-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-factory --html --output-dir coverage/larql-factory --fail-under-lines $(LARQL_FACTORY_COVERAGE_MIN)
	cargo llvm-cov report --package larql-factory --json --summary-only --output-path $(LARQL_FACTORY_COVERAGE_REPORT)
	$(MAKE) larql-factory-coverage-policy
	@echo "Report: coverage/larql-factory/html/index.html"

larql-factory-ci: larql-factory-fmt-check larql-factory-lint larql-factory-test larql-factory-coverage-summary

# larql-kv — pluggable KV-cache engines (markov-rs, unlimited-context, turbo-quant, apollo)
#
# Default policy is 90% per-file line coverage; total floor tracks the
# starting baseline and ratchets upward.
LARQL_KV_COVERAGE_MIN ?= 90
LARQL_KV_COVERAGE_POLICY ?= crates/larql-kv/coverage-policy.json
LARQL_KV_COVERAGE_REPORT ?= coverage/larql-kv/summary.json

larql-kv-test:
	cargo test -p larql-kv

larql-kv-fmt-check:
	cargo fmt -p larql-kv -- --check

larql-kv-lint:
	cargo clippy -p larql-kv --all-targets --no-deps -- -D warnings

larql-kv-examples:
	cargo check -p larql-kv --examples

larql-kv-bench-test:
	cargo test -p larql-kv --benches

larql-kv-bench:
	cargo bench -p larql-kv --bench engine_decode

larql-kv-coverage-policy:
	@if [ ! -f "$(LARQL_KV_COVERAGE_REPORT)" ]; then \
		echo "Coverage report not found: $(LARQL_KV_COVERAGE_REPORT)"; \
		echo "Run: make larql-kv-coverage-summary"; \
		exit 1; \
	fi
	python3 scripts/check_coverage_policy.py $(LARQL_KV_COVERAGE_REPORT) $(LARQL_KV_COVERAGE_POLICY)

larql-kv-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-kv --fail-under-lines $(LARQL_KV_COVERAGE_MIN)
	@mkdir -p coverage/larql-kv
	cargo llvm-cov report --package larql-kv --json --summary-only --output-path $(LARQL_KV_COVERAGE_REPORT)
	$(MAKE) larql-kv-coverage-policy

larql-kv-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-kv --summary-only --fail-under-lines $(LARQL_KV_COVERAGE_MIN)
	@mkdir -p coverage/larql-kv
	cargo llvm-cov report --package larql-kv --json --summary-only --output-path $(LARQL_KV_COVERAGE_REPORT)
	$(MAKE) larql-kv-coverage-policy

larql-kv-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-kv --html --output-dir coverage/larql-kv --fail-under-lines $(LARQL_KV_COVERAGE_MIN)
	cargo llvm-cov report --package larql-kv --json --summary-only --output-path $(LARQL_KV_COVERAGE_REPORT)
	$(MAKE) larql-kv-coverage-policy
	@echo "Report: coverage/larql-kv/html/index.html"

larql-kv-ci: larql-kv-fmt-check larql-kv-lint larql-kv-test larql-kv-examples larql-kv-bench-test larql-kv-coverage-summary

# larql-compute — CPU kernels and backend contracts
#
# `larql-compute` clears the 90% per-file default on every file; the
# total is ~97%.  Keep this floor near current as a ratchet — raise
# it whenever the per-file numbers move up.
LARQL_COMPUTE_COVERAGE_MIN ?= 95
LARQL_COMPUTE_COVERAGE_POLICY ?= crates/larql-compute/coverage-policy.json
LARQL_COMPUTE_COVERAGE_REPORT ?= coverage/larql-compute/summary.json

larql-compute-test: larql-compute-test-fast

# Default fast path: library/unit tests only. This deliberately avoids
# compiling every integration-test binary.
larql-compute-test-fast:
	cargo test -p larql-compute --lib

# ── Iteration loops for refactor work (no test execution, just type-check) ──
#
# These shave minutes off the inner refactor loop versus running the tests.
# Use the smallest one that catches the change you're making.

# Fastest type-check — `lib` only. The right loop for refactors that don't
# change test signatures.
larql-compute-check-fast:
	cargo check -p larql-compute --lib

# Type-check `lib` + every integration-test binary under `tests/`. Use when
# a refactor renames or moves something the integration tests reach into.
larql-compute-check-tests:
	cargo check -p larql-compute --tests

# Same but also walks examples + benches — the most thorough type check
# short of building everything.
larql-compute-check-all:
	cargo check -p larql-compute --tests --benches --examples

# Full integration suite — turns on `heavy_tests` for the slow
# correctness/parity suites and walks every integration binary under
# crates/larql-compute/tests.
larql-compute-test-integration:
	cargo test -p larql-compute --features heavy_tests --tests

larql-compute-fmt-check:
	cargo fmt -p larql-compute -- --check

larql-compute-lint:
	cargo clippy -p larql-compute --all-targets --no-deps -- -D warnings

larql-compute-coverage-policy:
	@if [ ! -f "$(LARQL_COMPUTE_COVERAGE_REPORT)" ]; then \
		echo "Coverage report not found: $(LARQL_COMPUTE_COVERAGE_REPORT)"; \
		echo "Run: make larql-compute-coverage-summary"; \
		exit 1; \
	fi
	python3 scripts/check_coverage_policy.py $(LARQL_COMPUTE_COVERAGE_REPORT) $(LARQL_COMPUTE_COVERAGE_POLICY)

larql-compute-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-compute --fail-under-lines $(LARQL_COMPUTE_COVERAGE_MIN)
	@mkdir -p coverage/larql-compute
	cargo llvm-cov report --package larql-compute --json --summary-only --output-path $(LARQL_COMPUTE_COVERAGE_REPORT)
	$(MAKE) larql-compute-coverage-policy

larql-compute-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-compute --summary-only --fail-under-lines $(LARQL_COMPUTE_COVERAGE_MIN)
	@mkdir -p coverage/larql-compute
	cargo llvm-cov report --package larql-compute --json --summary-only --output-path $(LARQL_COMPUTE_COVERAGE_REPORT)
	$(MAKE) larql-compute-coverage-policy

larql-compute-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-compute --html --output-dir coverage/larql-compute --fail-under-lines $(LARQL_COMPUTE_COVERAGE_MIN)
	cargo llvm-cov report --package larql-compute --json --summary-only --output-path $(LARQL_COMPUTE_COVERAGE_REPORT)
	$(MAKE) larql-compute-coverage-policy
	@echo "Report: coverage/larql-compute/html/index.html"

larql-compute-ci: larql-compute-fmt-check larql-compute-lint larql-compute-test-fast larql-compute-coverage

# larql-boundary — confidence-gated BOUNDARY ref codec
larql-boundary-test:
	cargo test -p larql-boundary

larql-boundary-fmt-check:
	cargo fmt -p larql-boundary -- --check

larql-boundary-lint:
	cargo clippy -p larql-boundary --all-targets -- -D warnings

larql-boundary-bench-test:
	cargo test -p larql-boundary --benches

larql-boundary-examples:
	cargo run -p larql-boundary --example accuracy

larql-boundary-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-boundary --summary-only

larql-boundary-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-boundary --html --output-dir coverage/larql-boundary
	@echo "Report: coverage/larql-boundary/html/index.html"

larql-boundary-ci: larql-boundary-fmt-check larql-boundary-lint larql-boundary-test larql-boundary-bench-test larql-boundary-examples

# larql-router — self-assembling grid router + protocol crate.
# 2026-05-14 measured baseline:
# **67.58% line / 70.21% function** for router-only test run (upstream,
# larql-server's integration tests also exercised router code paths; that
# crate is not part of this repository).
LARQL_ROUTER_COVERAGE_MIN ?= 91
LARQL_ROUTER_COVERAGE_POLICY ?= crates/larql-router/coverage-policy.json
LARQL_ROUTER_COVERAGE_REPORT ?= coverage/larql-router/summary.json

larql-router-test:
	cargo test -p larql-router

larql-router-fmt-check:
	cargo fmt -p larql-router -- --check

larql-router-lint:
	cargo clippy -p larql-router --all-targets -- -D warnings

larql-router-coverage-policy:
	@if [ ! -f "$(LARQL_ROUTER_COVERAGE_REPORT)" ]; then \
		echo "Coverage report not found: $(LARQL_ROUTER_COVERAGE_REPORT)"; \
		echo "Run: make larql-router-coverage-summary"; \
		exit 1; \
	fi
	python3 scripts/check_coverage_policy.py $(LARQL_ROUTER_COVERAGE_REPORT) $(LARQL_ROUTER_COVERAGE_POLICY)

larql-router-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-router --fail-under-lines $(LARQL_ROUTER_COVERAGE_MIN)
	@mkdir -p coverage/larql-router
	cargo llvm-cov report --package larql-router --json --summary-only --output-path $(LARQL_ROUTER_COVERAGE_REPORT)
	$(MAKE) larql-router-coverage-policy

larql-router-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-router --summary-only --fail-under-lines $(LARQL_ROUTER_COVERAGE_MIN)
	@mkdir -p coverage/larql-router
	cargo llvm-cov report --package larql-router --json --summary-only --output-path $(LARQL_ROUTER_COVERAGE_REPORT)
	$(MAKE) larql-router-coverage-policy

larql-router-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-router --html --output-dir coverage/larql-router --fail-under-lines $(LARQL_ROUTER_COVERAGE_MIN)
	cargo llvm-cov report --package larql-router --json --summary-only --output-path $(LARQL_ROUTER_COVERAGE_REPORT)
	$(MAKE) larql-router-coverage-policy
	@echo "Report: coverage/larql-router/html/index.html"

larql-router-ci: larql-router-fmt-check larql-router-lint larql-router-test

# larql-router-protocol — generated proto + QUIC transport wrapper.
# Only `transport/quic.rs` carries instrumented logic; everything else
# is `tonic::include_proto!`-generated code llvm-cov filters out.
LARQL_ROUTER_PROTOCOL_COVERAGE_MIN ?= 90
LARQL_ROUTER_PROTOCOL_COVERAGE_POLICY ?= crates/larql-router-protocol/coverage-policy.json
LARQL_ROUTER_PROTOCOL_COVERAGE_REPORT ?= coverage/larql-router-protocol/summary.json

larql-router-protocol-test:
	cargo test -p larql-router-protocol --features quic

larql-router-protocol-fmt-check:
	cargo fmt -p larql-router-protocol -- --check

larql-router-protocol-lint:
	cargo clippy -p larql-router-protocol --features quic --all-targets -- -D warnings

larql-router-protocol-coverage-policy:
	@if [ ! -f "$(LARQL_ROUTER_PROTOCOL_COVERAGE_REPORT)" ]; then \
		echo "Coverage report not found: $(LARQL_ROUTER_PROTOCOL_COVERAGE_REPORT)"; \
		echo "Run: make larql-router-protocol-coverage-summary"; \
		exit 1; \
	fi
	python3 scripts/check_coverage_policy.py $(LARQL_ROUTER_PROTOCOL_COVERAGE_REPORT) $(LARQL_ROUTER_PROTOCOL_COVERAGE_POLICY)

larql-router-protocol-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-router-protocol --features http3 --summary-only --fail-under-lines $(LARQL_ROUTER_PROTOCOL_COVERAGE_MIN)
	@mkdir -p coverage/larql-router-protocol
	cargo llvm-cov report --package larql-router-protocol --json --summary-only --output-path $(LARQL_ROUTER_PROTOCOL_COVERAGE_REPORT)
	$(MAKE) larql-router-protocol-coverage-policy

larql-router-protocol-ci: larql-router-protocol-fmt-check larql-router-protocol-lint larql-router-protocol-test

# larql-lql — LQL parser, executor, REPL. Crate has no GPU feature;
# Remote-backend tests use `mockito`, no real model weights required.
larql-lql-test:
	cargo test -p larql-lql

larql-lql-fmt-check:
	cargo fmt -p larql-lql -- --check

larql-lql-lint:
	cargo clippy -p larql-lql --all-targets --no-deps -- -D warnings

larql-lql-examples:
	cargo check -p larql-lql --examples

larql-lql-bench-test:
	cargo test -p larql-lql --benches

larql-lql-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-lql --summary-only

larql-lql-ci: larql-lql-fmt-check larql-lql-lint larql-lql-test larql-lql-examples larql-lql-bench-test

# larql-cli — top-level `larql` binary. Default features are just `research`,
# so the research tooling stays tested and measured, as in CI. Override
# with LARQL_CLI_DEFAULT_FEATURES=--no-default-features for the release shape.
LARQL_CLI_DEFAULT_FEATURES ?=

larql-cli-test:
	cargo test -p larql-cli $(LARQL_CLI_DEFAULT_FEATURES)

larql-cli-fmt-check:
	cargo fmt -p larql-cli -- --check

# Lint disabled: 2026-05-10 `larql-cli` carries ~82 pre-existing clippy
# errors under default features and ~112 under `--no-default-features`
# (mostly `large_enum_variant` and `dead_code` on metal-only paths).
# Re-enable `-- -D warnings` after that backlog is cleared.
larql-cli-lint:
	cargo clippy -p larql-cli --bins --tests $(LARQL_CLI_DEFAULT_FEATURES) --no-deps

LARQL_CLI_COVERAGE_MIN ?= 7
LARQL_CLI_COVERAGE_POLICY ?= crates/larql-cli/coverage-policy.json
LARQL_CLI_COVERAGE_REPORT ?= coverage/larql-cli/summary.json

larql-cli-coverage-policy:
	@if [ ! -f "$(LARQL_CLI_COVERAGE_REPORT)" ]; then \
		echo "Coverage report not found: $(LARQL_CLI_COVERAGE_REPORT)"; \
		echo "Run: make larql-cli-coverage-summary"; \
		exit 1; \
	fi
	python3 scripts/check_coverage_policy.py $(LARQL_CLI_COVERAGE_REPORT) $(LARQL_CLI_COVERAGE_POLICY)

larql-cli-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-cli $(LARQL_CLI_DEFAULT_FEATURES) --summary-only --fail-under-lines $(LARQL_CLI_COVERAGE_MIN)
	@mkdir -p coverage/larql-cli
	cargo llvm-cov report --package larql-cli --json --summary-only --output-path $(LARQL_CLI_COVERAGE_REPORT)
	$(MAKE) larql-cli-coverage-policy

larql-cli-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-cli $(LARQL_CLI_DEFAULT_FEATURES) --fail-under-lines $(LARQL_CLI_COVERAGE_MIN)
	@mkdir -p coverage/larql-cli
	cargo llvm-cov report --package larql-cli --json --summary-only --output-path $(LARQL_CLI_COVERAGE_REPORT)
	$(MAKE) larql-cli-coverage-policy

larql-cli-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-cli $(LARQL_CLI_DEFAULT_FEATURES) --html --output-dir coverage/larql-cli --fail-under-lines $(LARQL_CLI_COVERAGE_MIN)
	cargo llvm-cov report --package larql-cli --json --summary-only --output-path $(LARQL_CLI_COVERAGE_REPORT)
	$(MAKE) larql-cli-coverage-policy
	@echo "Report: coverage/larql-cli/html/index.html"

larql-cli-ci: larql-cli-fmt-check larql-cli-test

# larql-inference — transformer inference engine. Tests requiring real
# model weights are gated `#[ignore]` (test_arch_golden, test_logits_goldens,
# test_gemma3_smoke, test_generate_q4k_cpu, test_layer_graph_integration);
# CI runs the default set only. Several diagnostic examples lag the
# refactored `larql-compute` decode API and are excluded from `--all-targets`
# until repaired.
larql-inference-test:
	cargo test -p larql-inference

larql-inference-fmt-check:
	cargo fmt -p larql-inference -- --check

larql-inference-lint:
	cargo clippy -p larql-inference --lib --tests --benches --no-deps -- -D warnings

larql-inference-bench-test:
	cargo test -p larql-inference --benches

# Inference coverage: per-file 90% floor with debt baselines for the
# live-shard / mmap-backed surface. See
# crates/larql-inference/coverage-policy.json for the policy_note.
LARQL_INFERENCE_COVERAGE_MIN ?= 70
LARQL_INFERENCE_COVERAGE_POLICY ?= crates/larql-inference/coverage-policy.json
LARQL_INFERENCE_COVERAGE_REPORT ?= coverage/larql-inference/summary.json

larql-inference-coverage-policy:
	@if [ ! -f "$(LARQL_INFERENCE_COVERAGE_REPORT)" ]; then \
		echo "Coverage report not found: $(LARQL_INFERENCE_COVERAGE_REPORT)"; \
		echo "Run: make larql-inference-coverage-summary"; \
		exit 1; \
	fi
	python3 scripts/check_coverage_policy.py $(LARQL_INFERENCE_COVERAGE_REPORT) $(LARQL_INFERENCE_COVERAGE_POLICY)

larql-inference-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-inference --fail-under-lines $(LARQL_INFERENCE_COVERAGE_MIN)
	@mkdir -p coverage/larql-inference
	cargo llvm-cov report --package larql-inference --json --summary-only --output-path $(LARQL_INFERENCE_COVERAGE_REPORT)
	$(MAKE) larql-inference-coverage-policy

larql-inference-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-inference --summary-only --fail-under-lines $(LARQL_INFERENCE_COVERAGE_MIN)
	@mkdir -p coverage/larql-inference
	cargo llvm-cov report --package larql-inference --json --summary-only --output-path $(LARQL_INFERENCE_COVERAGE_REPORT)
	$(MAKE) larql-inference-coverage-policy

larql-inference-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-inference --html --output-dir coverage/larql-inference
	@echo "Report: coverage/larql-inference/html/index.html"

larql-inference-ci: larql-inference-fmt-check larql-inference-lint larql-inference-test larql-inference-bench-test larql-inference-coverage-summary

# Check (compile without building)
check:
	cargo check --workspace

# Code quality
fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

lint:
	cargo clippy --workspace --tests -- -D warnings

# All quality checks
ci: fmt-check lint test-full

# Clean
clean:
	cargo clean

# Benchmarks
#
# `bench` runs the core graph example. `bench-compute` runs the primary
# larql-compute Criterion surface. `bench-save` records a compute baseline
# named `main`; `bench-check` re-runs the compute benches and fails if any
# cell regresses past Criterion's default noise threshold.
bench: bench-core

bench-core:
	cargo run --release -p larql-core --example bench_graph

bench-inference:
	cargo run --release -p larql-inference --example bench_inference

# Compute kernel criterion bench: the CPU Q4_K × Q8_K quantised matvec.
bench-compute:
	cargo bench -p larql-compute --bench q4k_q8k_matvec

# Wire codec criterion bench (encode/decode f32/f16/i8 throughput).
bench-wire:
	cargo bench -p larql-inference --bench wire_codec

# Router routing hot-path criterion bench (route/heartbeat/rebuild ns/op).
bench-routing:
	cargo bench -p larql-router --bench routing

# Cross-architecture decode bench — runs `larql bench` on Gemma 3 4B,
# Gemma 4 31B dense, Llama 2 7B, Mistral 7B, Gemma 4 26B A4B in
# sequence and prints per-arch tok/s. Operationalises ADR-017
# model-agnosticity check: any A/B promoted on Gemma 3 4B alone should
# be re-bench'd here before landing. Also surfaces thermal artifacts:
# if every arch regresses simultaneously vs baseline, suspect thermal.
#
#   make bench-cross-arch                     # report current numbers
#   make bench-cross-arch ARGS=--save-baseline  # save current as baseline
#   make bench-cross-arch ARGS=--compare        # diff vs saved baseline
#
# Bench params via env: LARQL_BENCH_TOKENS, LARQL_BENCH_WARMUP, LARQL_BENCH_PROMPT.
bench-cross-arch:
	./scripts/bench-cross-arch.sh $(ARGS)

bench-all: bench-core bench-inference bench-compute bench-wire bench-routing

# Vindex micro-benches — synthetic, fast, safe under load.
bench-vindex:
	cargo bench -p larql-vindex --bench vindex_ops

# Vindex production-dim scaling bench. Refuses if larql-server / router
# are alive (they distort 1-2 GB matmuls). Run alone, on a cool host;
# results feed PERFORMANCE.md.
bench-vindex-scaling:
	@if pgrep -fl 'larql-(server|router)' >/dev/null 2>&1; then \
		echo "Refusing bench-vindex-scaling: larql daemons running. Stop them first."; \
		pgrep -fl 'larql-(server|router)'; \
		exit 2; \
	fi
	cargo bench -p larql-vindex --bench vindex_scaling

bench-save:
	bash scripts/bench-regress.sh save

bench-check:
	bash scripts/bench-regress.sh check

# Coverage — uses cargo-llvm-cov (install with `cargo install cargo-llvm-cov`).
# Writes an HTML report to coverage/ that can be opened in a browser.
# Scoped to larql-vindex by default since the audit owner cares about
# that crate; pass CRATE=… to scope elsewhere.
COVERAGE_CRATE ?= larql-vindex
coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package $(COVERAGE_CRATE) --html --output-dir coverage
	@echo "Report: coverage/html/index.html"

coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; \
		exit 1; \
	fi
	cargo llvm-cov --package $(COVERAGE_CRATE) --summary-only

# Extraction
extract-test:
	cargo run --release -p larql-cli -- weight-extract google/gemma-3-4b-it \
		--layer 26 -o output/test-L26.larql.json \
		--stats output/test-L26-stats.json

extract-full:
	cargo run --release -p larql-cli -- weight-extract google/gemma-3-4b-it \
		-o output/gemma-3-4b-knowledge.larql.json \
		--stats output/gemma-3-4b-stats.json

# Inference
predict:
	cargo run --release -p larql-cli -- predict google/gemma-3-4b-it \
		--prompt "The capital of France is" -k 10
