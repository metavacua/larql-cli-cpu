"""Path normalisation shared by every reader.

Cells run on Linux, macOS and Windows runners, so the same source file
arrives as `/home/runner/work/r/r/crates/x/src/a.rs`,
`/Users/runner/work/r/r/crates/x/src/a.rs` or `D:\\a\\r\\r\\crates\\x\\src\\a.rs`.
Everything is keyed by the repo-relative POSIX path from `crates/` on.
"""

from __future__ import annotations

CRATES_DIR = "crates/"


def repo_relative(path: str) -> str | None:
    """`crates/...` for a workspace source file, `None` for anything else
    (registry sources, the standard library, generated OUT_DIR files)."""
    posix = path.replace("\\", "/")
    if posix.startswith(CRATES_DIR):
        return posix
    marker = "/" + CRATES_DIR
    index = posix.rfind(marker)
    if index < 0:
        return None
    return posix[index + 1 :]


def in_crate(path: str | None, crate_dir: str) -> bool:
    """Whether a normalised path is a source file of `crate_dir`
    (e.g. `crates/larql-cli`)."""
    return path is not None and path.startswith(crate_dir.rstrip("/") + "/")
