use clap::{Parser, Subcommand};

mod bfs_cmd;
mod describe_cmd;
mod formatting;
mod query_cmd;
mod stats_cmd;
mod validate_cmd;

#[derive(Parser)]
#[command(
    name = "larql",
    version,
    about = "LARQL knowledge graph extraction and querying"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// BFS extraction from a model endpoint.
    Bfs(bfs_cmd::BfsArgs),

    /// Show graph statistics.
    Stats(stats_cmd::StatsArgs),

    /// Validate a .larql.json file.
    Validate(validate_cmd::ValidateArgs),

    /// Query a graph for facts.
    Query(query_cmd::QueryArgs),

    /// Describe an entity.
    Describe(describe_cmd::DescribeArgs),
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Bfs(args) => bfs_cmd::run(args),
        Commands::Stats(args) => stats_cmd::run(args),
        Commands::Validate(args) => validate_cmd::run(args),
        Commands::Query(args) => query_cmd::run(args),
        Commands::Describe(args) => describe_cmd::run(args),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
