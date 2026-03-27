use larql_core::*;
use std::time::Instant;

fn main() {
    let n = 100_000usize;
    let mut graph = Graph::new();

    let start = Instant::now();
    for i in 0..n {
        let edge = Edge::new(
            format!("Entity_{}", i / 10),
            format!("rel_{}", i % 10),
            format!("Target_{}", i),
        )
        .with_confidence(0.5 + (i as f64 % 50.0) / 100.0);
        graph.add_edge(edge);
    }
    let insert_time = start.elapsed();

    println!(
        "Inserted {} unique edges in {:.1}ms",
        graph.edge_count(),
        insert_time.as_secs_f64() * 1000.0
    );

    // Query
    let start = Instant::now();
    let iterations = 100_000;
    for i in 0..iterations {
        let _ = graph.select(&format!("Entity_{}", i % 10_000), None);
    }
    let query_time = start.elapsed();
    println!(
        "{} select() calls in {:.1}ms ({:.0} ns/call)",
        iterations,
        query_time.as_secs_f64() * 1000.0,
        query_time.as_nanos() as f64 / iterations as f64
    );

    // Search
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = graph.search("Entity 42", 10);
    }
    let search_time = start.elapsed();
    println!(
        "1000 search() calls in {:.1}ms ({:.1} us/call)",
        search_time.as_secs_f64() * 1000.0,
        search_time.as_nanos() as f64 / 1000.0 / 1000.0
    );

    // Stats (includes node computation + components)
    let start = Instant::now();
    let stats = graph.stats();
    let stats_time = start.elapsed();
    println!(
        "stats() in {:.0}ms — {} entities, {} edges, {} components",
        stats_time.as_secs_f64() * 1000.0,
        stats.entities,
        stats.edges,
        stats.connected_components
    );

    // Subgraph extraction
    let start = Instant::now();
    let sub = graph.subgraph("Entity_0", 2);
    let subgraph_time = start.elapsed();
    println!(
        "subgraph(depth=2) in {:.1}ms — {} edges",
        subgraph_time.as_secs_f64() * 1000.0,
        sub.edge_count()
    );

    // JSON roundtrip
    let start = Instant::now();
    let json_bytes = to_bytes(&graph, Format::Json).unwrap();
    let json_ser_time = start.elapsed();
    println!(
        "JSON: serialized {} edges to {:.1} MB in {:.0}ms",
        graph.edge_count(),
        json_bytes.len() as f64 / 1024.0 / 1024.0,
        json_ser_time.as_secs_f64() * 1000.0
    );

    let start = Instant::now();
    let _ = from_bytes(&json_bytes, Format::Json).unwrap();
    let json_deser_time = start.elapsed();
    println!(
        "JSON: deserialized in {:.0}ms",
        json_deser_time.as_secs_f64() * 1000.0
    );

    // MessagePack roundtrip
    let start = Instant::now();
    let msgpack_bytes = to_bytes(&graph, Format::MessagePack).unwrap();
    let msgpack_ser_time = start.elapsed();
    println!(
        "MsgPack: serialized {} edges to {:.1} MB in {:.0}ms",
        graph.edge_count(),
        msgpack_bytes.len() as f64 / 1024.0 / 1024.0,
        msgpack_ser_time.as_secs_f64() * 1000.0
    );

    let start = Instant::now();
    let _ = from_bytes(&msgpack_bytes, Format::MessagePack).unwrap();
    let msgpack_deser_time = start.elapsed();
    println!(
        "MsgPack: deserialized in {:.0}ms",
        msgpack_deser_time.as_secs_f64() * 1000.0
    );

    println!(
        "\nSize: JSON {:.1} MB vs MsgPack {:.1} MB ({:.0}% smaller)",
        json_bytes.len() as f64 / 1024.0 / 1024.0,
        msgpack_bytes.len() as f64 / 1024.0 / 1024.0,
        (1.0 - msgpack_bytes.len() as f64 / json_bytes.len() as f64) * 100.0
    );
}
