pub mod algo;
pub mod core;
#[cfg(feature = "engine")]
pub mod engine;
#[cfg(feature = "io")]
pub mod io;

// Re-export the essential types at crate root.
pub use core::edge::Edge;
pub use core::enums::{MergeStrategy, SourceType};
pub use core::graph::{EdgeInsertResult, Graph};
pub use core::schema::Schema;

#[cfg(feature = "engine")]
pub use engine::bfs::{extract_bfs, BfsCallbacks, BfsConfig, BfsResult};
#[cfg(feature = "engine")]
pub use engine::chain::{chain_tokens, ChainResult};
#[cfg(feature = "engine")]
pub use engine::provider::{ModelProvider, PredictionResult, TokenPrediction};
#[cfg(feature = "engine")]
pub use engine::templates::TemplateRegistry;

#[cfg(feature = "io")]
pub use io::checkpoint::CheckpointLog;
#[cfg(feature = "io")]
pub use io::format::Format;
#[cfg(feature = "io")]
pub use io::json::{load_json, save_json};
#[cfg(feature = "io")]
pub use io::{from_bytes, load, load_with_format, save, save_with_format, to_bytes};

pub use algo::components::{are_connected, connected_components};
pub use algo::diff::{diff, ChangedEdge, GraphDiff};
pub use algo::filter::{filter_graph, FilterConfig, MetadataCompare, MetadataPredicate};
pub use algo::merge::{
    default_source_priority, merge_graphs, merge_graphs_with_source_priority,
    merge_graphs_with_strategy,
};
pub use algo::pagerank::{pagerank, PageRankResult};
pub use algo::shortest_path::{astar, shortest_path, shortest_path_with_weight, PathResult};
pub use algo::traversal::{bfs as bfs_traversal, dfs, TraversalResult};
pub use algo::walk::{walk_all_paths, WalkResult};
#[cfg(feature = "io")]
pub use io::csv::{load_csv, save_csv};
#[cfg(feature = "io")]
pub use io::packed::{from_packed_bytes, load_packed, save_packed, to_packed_bytes};
