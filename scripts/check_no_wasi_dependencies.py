#!/usr/bin/env python3
"""Keep wasmtime-wasi (and the socket stack it drags in) out of the local graph.

The WASM experts are built for wasm32-unknown-unknown and import no WASI, so the
host instantiates them with a plain wasmtime `Linker`. wasmtime-wasi compiles in
socket code and pulls cap-net-ext; neither may exist here. `tokio` and `url` stay
legitimately (tonic, hyper-util, reqwest), so only the WASI crates are banned.

Run without `--locked` in the lockfile-refresh job (it re-resolves first); the
manifest scan needs no cargo at all.
"""

from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parent.parent
BANNED = {"wasmtime-wasi", "wasmtime-wasi-io", "wasi-common", "cap-net-ext"}
DEP_TABLES = {"dependencies", "dev-dependencies", "build-dependencies"}


def manifest_errors(path: Path) -> list[str]:
    """Check every dependency table, including target-specific and workspace ones."""
    errors = []

    def visit(table):
        for key, value in table.items():
            if not isinstance(value, dict):
                continue
            if key in DEP_TABLES or key == "workspace.dependencies":
                for alias, dep in value.items():
                    package = dep.get("package", alias) if isinstance(dep, dict) else alias
                    if package in BANNED:
                        errors.append(f"{path}: depends on {package}")
            else:
                visit(value)

    visit(tomllib.loads(path.read_text()))
    return errors


def banned_packages(tree: str) -> set[str]:
    return {line.split()[0] for line in tree.splitlines() if line.split()} & BANNED


def main() -> None:
    manifests = [ROOT / "Cargo.toml", *(ROOT / "crates").glob("*/Cargo.toml")]
    errors = [e for path in manifests for e in manifest_errors(path)]
    if errors:
        raise SystemExit("\n".join(errors))
    tree = subprocess.run(
        ["cargo", "tree", "--workspace", "--all-features", "--target", "all",
         "-e", "normal,build,dev", "--prefix", "none", "--format", "{p}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if tree.returncode:
        raise SystemExit(tree.stderr.strip() or f"cargo tree failed with status {tree.returncode}")
    found = banned_packages(tree.stdout)
    if found:
        raise SystemExit(f"WASI/socket crates in the dependency graph: {', '.join(sorted(found))}")
    print("no wasmtime-wasi or cap-net-ext in the workspace dependency graph (all targets)")


if __name__ == "__main__":
    main()
