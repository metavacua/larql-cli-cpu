# chuk-larql-rs

Knowledge graphs extracted from neural network weights. One library, one binary, one graph format.

LARQL probes a language model with structured prompts, collects next-token predictions, and assembles them into a directed labeled knowledge graph. The graph can be queried, walked, merged, and serialized to JSON or MessagePack.

## Quick start

```bash
# Build
cargo build --release -p larql-cli

# Extract a graph using a mock provider (no model needed)
larql bfs \
    --seeds "France,Germany,Japan" \
    --templates examples/templates.json \
    --mock --mock-knowledge examples/mock_knowledge.json \
    --output knowledge.larql.json

# Query it
larql query --graph knowledge.larql.json France capital-of
#   France --capital-of--> Paris  (0.89)

larql describe --graph knowledge.larql.json France
# France
#   Outgoing:
#     --capital-of--> Paris  (0.89)
#     --language-of--> French  (0.84)
#     --continent--> Europe  (0.95)
#     --currency--> Euro  (0.91)
#   Incoming:
#     Paris --located-in-->  (0.98)

larql stats knowledge.larql.json
# Graph: knowledge.larql.json
#   Entities:    13
#   Edges:       15
#   Relations:   5

larql validate knowledge.larql.json
# Validated: knowledge.larql.json
#   13 entities, 15 edges, 5 relations
#   OK — no issues found
```

## Extract from a real model

Point LARQL at any OpenAI-compatible completions endpoint — Ollama, vLLM, llama.cpp, LM Studio:

```bash
# Start Ollama with a model
ollama run gemma3:4b-it

# Extract
larql bfs \
    --seeds "France,Germany,Japan,Mozart,Einstein" \
    --templates examples/templates.json \
    --endpoint http://localhost:11434/v1 \
    --model gemma3:4b-it \
    --max-entities 1000 \
    --max-depth 3 \
    --output knowledge.larql.json
```

## Workspace structure

```
chuk-larql-rs/
├── crates/
│   ├── larql-core/     Library crate — graph engine, providers, extraction, I/O
│   └── larql-cli/      Binary crate — thin CLI over larql-core
├── examples/
│   ├── templates.json            Example relation templates
│   ├── mock_knowledge.json       Example mock provider knowledge
│   └── gemma_4b_knowledge.json   Example graph (Python-compatible)
└── README.md
```

- **larql-core** is the library. Other Rust projects depend on it.
- **larql-cli** is the binary. Users download and run it.
- The `.larql.json` / `.larql.bin` file is the interface between Rust and Python.

## Concepts

### Everything is config, nothing is hardcoded

The engine is domain-agnostic. There are no built-in relation types, node types, or prompt templates. All domain knowledge lives in config files loaded at runtime:

- **Templates** define how to probe the model for each relation type
- **Schema** defines relation metadata and type inference rules
- **Mock knowledge** (for testing) maps prompts to expected answers

This means you can use LARQL for any domain — geography, music, biology, software — just by writing different templates.

### Edges are facts

Every fact is a directed labeled edge: `subject --relation--> object` with a confidence score and source type. Edges are the only stored data; nodes are derived from edges on demand.

### BFS extraction

Starting from seed entities, LARQL:

1. Probes every template for the current entity
2. Chains multiple forward passes for multi-token answers
3. Adds high-confidence results as edges
4. Queues discovered entities for further exploration
5. Checkpoints after each entity for crash recovery

## Configuration

### Templates

Templates define how to probe a model for each relation. They are loaded from a JSON file passed via `--templates`.

```json
[
  {
    "relation": "capital-of",
    "template": "The capital of {subject} is",
    "multi_token": true,
    "stop_tokens": [".", "\n", ",", ";"]
  },
  {
    "relation": "birth-year",
    "template": "{subject} was born in the year",
    "multi_token": false,
    "stop_tokens": [".", "\n", ",", ";"]
  }
]
```

| Field | Type | Description |
|---|---|---|
| `relation` | string | Name of the relation this template probes |
| `template` | string | Prompt text. `{subject}` is replaced with the entity name |
| `multi_token` | bool | If true, chain multiple forward passes to build the answer |
| `reverse_template` | string? | Optional reverse probe (`{object}` placeholder) |
| `stop_tokens` | char[] | Characters that terminate multi-token chaining |

### Schema

The schema is embedded in `.larql.json` graph files. It defines relation metadata and optional type inference rules.

```json
{
  "schema": {
    "relations": [
      {
        "name": "capital-of",
        "subject_types": ["country"],
        "object_types": ["city"],
        "reversible": true
      }
    ],
    "type_rules": [
      {
        "node_type": "country",
        "outgoing": ["capital-of", "language-of", "currency", "continent"],
        "incoming": []
      },
      {
        "node_type": "city",
        "outgoing": [],
        "incoming": ["capital-of"]
      }
    ]
  }
}
```

Type rules are optional. If present, they assign a type string to nodes based on which relations they participate in. If no rule matches, the node type is `None`.

### Mock knowledge

For testing without a model, provide a JSON file mapping prompts to answers:

```json
[
  {"prompt": "The capital of France is", "answer": "Paris", "probability": 0.89},
  {"prompt": "The capital of Germany is", "answer": "Berlin", "probability": 0.81}
]
```

## Serialization formats

LARQL supports two serialization formats, auto-detected from the file extension:

| Extension | Format | Use case |
|---|---|---|
| `.larql.json` / `.json` | JSON (pretty-printed) | Human-readable, Python interop |
| `.larql.bin` / `.bin` | MessagePack | Compact storage, faster I/O |

Both formats are semantically identical — a graph saved as JSON can be loaded and re-saved as MessagePack, and vice versa.

**Performance at 100k edges:**

| | JSON | MessagePack |
|---|---|---|
| Serialize | 136ms | 126ms |
| Deserialize | 316ms | 285ms |
| File size | 9.9 MB | 4.7 MB (53% smaller) |

Just change the output extension:

```bash
larql bfs --output graph.larql.json ...   # JSON
larql bfs --output graph.larql.bin  ...   # MessagePack
```

All CLI commands read either format transparently.

### JSON format (Python interop)

The `.larql.json` format is the contract between Rust and Python. Structure:

```json
{
  "larql_version": "0.1.0",
  "metadata": {
    "model": "google/gemma-3-4b-it",
    "extraction_method": "bfs"
  },
  "schema": { ... },
  "edges": [
    {"s": "France", "r": "capital-of", "o": "Paris", "c": 0.89, "src": "parametric"}
  ]
}
```

Compact edge keys: `s` (subject), `r` (relation), `o` (object), `c` (confidence), `src` (source type), `meta` (metadata), `inj` (injection).

## CLI reference

### `larql bfs`

BFS extraction from a model endpoint.

```
larql bfs --seeds <SEEDS> --templates <TEMPLATES> --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `-s, --seeds` | Comma-separated seed entities |
| `-t, --templates` | Path to templates JSON file |
| `-o, --output` | Output file (.larql.json or .larql.bin) |
| `-e, --endpoint` | Model endpoint URL (default: `http://localhost:11434/v1`) |
| `-m, --model` | Model name for the endpoint |
| `--mock` | Use mock provider instead of HTTP |
| `--mock-knowledge` | Path to mock knowledge JSON |
| `--max-depth` | Maximum BFS depth (default: 3) |
| `--max-entities` | Maximum entities to probe (default: 1000) |
| `--min-confidence` | Minimum edge confidence (default: 0.3) |
| `--resume` | Resume from a checkpoint file |

### `larql stats`

```
larql stats <GRAPH>
```

Show entity count, edge count, relation count, connected components, average degree, average confidence, and source distribution.

### `larql query`

```
larql query --graph <GRAPH> <SUBJECT> [RELATION]
```

Select edges from a subject, optionally filtered by relation.

### `larql describe`

```
larql describe --graph <GRAPH> <ENTITY>
```

Show all outgoing and incoming edges for an entity.

### `larql validate`

```
larql validate <GRAPH>
```

Check a graph file for issues: zero-confidence edges, self-loops, empty subjects/objects.

## Library usage

Add to your `Cargo.toml`:

```toml
[dependencies]
larql-core = { git = "https://github.com/chrishayuk/chuk-larql-rs" }
```

```rust
use larql_core::*;

// Build a graph
let mut graph = Graph::new();
graph.add_edge(
    Edge::new("France", "capital-of", "Paris")
        .with_confidence(0.89)
        .with_source(SourceType::Parametric)
);

// Query
let edges = graph.select("France", Some("capital-of"));
let result = graph.describe("France");
let (dest, path) = graph.walk("France", &["capital-of", "located-in"]).unwrap();

// Search
let results = graph.search("France capital", 10);

// Subgraph
let neighborhood = graph.subgraph("France", 2);

// Stats
let stats = graph.stats();

// Save / load (format from extension)
save(&graph, "graph.larql.json")?;
save(&graph, "graph.larql.bin")?;
let g = load("graph.larql.json")?;

// Explicit format
let bytes = to_bytes(&graph, Format::MessagePack)?;
let g = from_bytes(&bytes, Format::Json)?;

// Merge graphs
let added = merge_graphs(&mut graph, &other_graph);

// Shortest path (weight = 1 - confidence)
if let Some((cost, path)) = shortest_path(&graph, "France", "Salzburg") {
    println!("Cost: {cost}, hops: {}", path.len());
}
```

### Feature flags

| Feature | Default | Description |
|---|---|---|
| `http` | yes | HTTP model provider (adds reqwest) |
| `msgpack` | yes | MessagePack serialization (adds rmp-serde) |

Disable for minimal dependency footprint:

```toml
larql-core = { git = "...", default-features = false }
```

## Status

This project is in early development. Here's where things stand:

### What's working

- **Core graph engine** — full implementation. Insert, remove, deduplicate, select, reverse select, describe, walk, keyword search, subgraph extraction, count, node lookup, stats, connected components. All indexed, lazy node computation.
- **BFS extraction** — complete pipeline. Template-based probing, multi-token chaining, confidence thresholds, entity queuing, checkpoint callbacks.
- **Model providers** — HTTP provider (OpenAI-compatible: Ollama, vLLM, llama.cpp, LM Studio) and configurable mock provider for testing.
- **Templates and schema** — fully config-driven. No hardcoded relation types, node types, or domain knowledge. Everything loaded from JSON at runtime.
- **Serialization** — JSON and MessagePack with format auto-detection from file extension.
- **Checkpoint / resume** — append-only edge log for crash recovery.
- **CLI** — all five commands implemented and working: `bfs`, `stats`, `query`, `describe`, `validate`.
- **Algorithms** — shortest path (Dijkstra), graph merge. Walk and components are implemented on `Graph` directly.
- **Benchmarks** — `bench_graph` example exercises insert, query, search, stats, subgraph, and serialization at 100k edges.
- **Test suite** — 102 tests across 10 test files, all passing. Coverage:

| Test file | Tests | Covers |
|---|---|---|
| `test_graph` | 28 | add, remove, dedup, select, reverse, describe, exists, walk, search, subgraph, count, node, stats, components |
| `test_roundtrip` | 14 | JSON and MessagePack roundtrip (value, bytes, file), cross-format, confidence/source/metadata preservation, format detection |
| `test_edge` | 11 | defaults, builder, clamping, equality/hash, triple, compact roundtrip, JSON serialization |
| `test_bfs_mock` | 10 | extraction, depth following, multiple seeds, max_entities, min_confidence, empty provider, dedup, source/metadata |
| `test_algo` | 8 | shortest path (direct, multi-hop, preference, no route, same node), graph merge |
| `test_python_compat` | 8 | load Python-produced graph, verify edges/confidence/source/schema/type_rules/stats, roundtrip both formats |
| `test_schema` | 7 | empty schema, add/get, type inference rules, JSON roundtrip |
| `test_templates` | 6 | registry, format, JSON roundtrip, load from example file |
| `test_chain` | 5 | single token, confidence stop, empty response, probability stats |
| `test_checkpoint` | 4 | write/replay, append across sessions, empty file, metadata preservation |

### What's missing

- **Real-model validation** — HTTP provider is implemented but not yet tested against a live endpoint.
- **CLI formatting** — `formatting.rs` is a stub. No table or colour output yet.
- **CI / GitHub Actions** — no pipeline configured.
- **Crate publishing** — not yet published to crates.io.
- **PyO3 binding** — Python API parity is ready (`remove_edge`, `count`, `node` added), binding not yet written.

### Next up

1. Test against a live Ollama endpoint
2. CI pipeline
3. PyO3 binding for Python interop

## License

Apache-2.0
