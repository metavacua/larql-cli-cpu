# Workspace crate facts

**Class: CURRENT — generated.**

Run `python3 scripts/workspace_facts.py --write`; CI runs `--check`.
These are manifest declarations, not a resolved platform/feature build graph.
[JSON](workspace-facts.json) retains normal, build and dev dependencies,
target conditions, default-feature choices and explicit example targets.

[Architecture guide](../architecture-stack.md) explains responsibility and capability.

## `Cargo.toml`

| Package | Version | Normal local dependencies | Default features |
|---|---|---|---|
| [larql-boundary](../../crates/larql-boundary/Cargo.toml) | `0.2.0` | None | None |
| [larql-cli](../../crates/larql-cli/Cargo.toml) | `0.2.0` | `larql-compute`, `larql-core`, `larql-factory`, `larql-inference`, `larql-kv`, `larql-lql`, `larql-models`, `larql-router`, `larql-vindex`, `larql-vindex-spec` | `research` |
| [larql-compute](../../crates/larql-compute/Cargo.toml) | `0.2.0` | `larql-execution`, `larql-models` | None |
| [larql-continuation-fixture](../../crates/larql-continuation-fixture/Cargo.toml) | `0.2.0` | `larql-vindex` | None |
| [larql-core](../../crates/larql-core/Cargo.toml) | `0.2.0` | None | `http`, `msgpack`, `io`, `engine` |
| [larql-execution](../../crates/larql-execution/Cargo.toml) | `0.2.0` | None | None |
| [larql-factory](../../crates/larql-factory/Cargo.toml) | `0.2.0` | `larql-models`, `larql-vindex-spec` | None |
| [larql-inference](../../crates/larql-inference/Cargo.toml) | `0.2.0` | `larql-compute`, `larql-execution`, `larql-models`, `larql-router-protocol`, `larql-vindex` | None |
| [larql-kv](../../crates/larql-kv/Cargo.toml) | `0.2.0` | `larql-boundary`, `larql-compute`, `larql-execution`, `larql-inference`, `larql-vindex` | None |
| [larql-lql](../../crates/larql-lql/Cargo.toml) | `0.2.0` | `larql-compute`, `larql-inference`, `larql-kv`, `larql-models`, `larql-vindex` | None |
| [larql-models](../../crates/larql-models/Cargo.toml) | `0.2.0` | `larql-vindex-spec` | None |
| [larql-router](../../crates/larql-router/Cargo.toml) | `0.2.0` | `larql-router-protocol` | None |
| [larql-router-protocol](../../crates/larql-router-protocol/Cargo.toml) | `0.2.0` | None | None |
| [larql-vindex](../../crates/larql-vindex/Cargo.toml) | `0.2.0` | `larql-compute`, `larql-core`, `larql-execution`, `larql-models`, `larql-vindex-spec` | None |
| [larql-vindex-spec](../../crates/larql-vindex-spec/Cargo.toml) | `0.2.0` | None | None |

## `crates/larql-experts/Cargo.toml`

| Package | Version | Normal local dependencies | Default features |
|---|---|---|---|
| [expert-interface](../../crates/larql-experts/expert-interface/Cargo.toml) | `0.1.0` | None | None |
| [larql-expert-arithmetic](../../crates/larql-experts/experts/arithmetic/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-conway](../../crates/larql-experts/experts/conway/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-date](../../crates/larql-experts/experts/date/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-dijkstra](../../crates/larql-experts/experts/dijkstra/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-element](../../crates/larql-experts/experts/element/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-finance](../../crates/larql-experts/experts/finance/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-geometry](../../crates/larql-experts/experts/geometry/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-graph](../../crates/larql-experts/experts/graph/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-hash](../../crates/larql-experts/experts/hash/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-http_status](../../crates/larql-experts/experts/http_status/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-isbn](../../crates/larql-experts/experts/isbn/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-logic](../../crates/larql-experts/experts/logic/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-luhn](../../crates/larql-experts/experts/luhn/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-markov](../../crates/larql-experts/experts/markov/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-sql](../../crates/larql-experts/experts/sql/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-statistics](../../crates/larql-experts/experts/statistics/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-string_ops](../../crates/larql-experts/experts/string_ops/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-trig](../../crates/larql-experts/experts/trig/Cargo.toml) | `0.1.0` | `expert-interface` | None |
| [larql-expert-unit](../../crates/larql-experts/experts/unit/Cargo.toml) | `0.1.0` | `expert-interface` | None |

The experts workspace is separate: root workspace test/build/coverage sweeps do
not include it. A package appearing here does not claim a published release or
a supported VINDEX3 operator. See each crate README and its tests.
