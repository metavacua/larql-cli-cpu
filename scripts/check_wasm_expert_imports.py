#!/usr/bin/env python3
"""Assert every built WASM expert is a freestanding module: zero imports, full ABI exports.

The experts are built for wasm32-unknown-unknown and the host (larql-inference)
instantiates them with a plain wasmtime `Linker`, so a module that imports anything
(WASI or an unresolved `env` symbol that wasm-ld let through) cannot be loaded.
That failure shows up at load time, not build time, so CI checks the artefacts here.

Usage: check_wasm_expert_imports.py [WASM_DIR]
The expected module count is the number of expert crates under
crates/larql-experts/experts/, so adding an expert needs no edit here.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERTS_WORKSPACE = ROOT / "crates" / "larql-experts"
DEFAULT_WASM_DIR = EXPERTS_WORKSPACE / "target" / "wasm32-unknown-unknown" / "release"

WASM_MAGIC = b"\0asm"
WASM_VERSION = b"\x01\0\0\0"
HEADER_LEN = 8
SECTION_IMPORT = 2
SECTION_EXPORT = 7
# The host ABI (larql-inference/src/experts/caller.rs).
REQUIRED_EXPORTS = {"memory", "larql_alloc", "larql_dealloc", "larql_call", "larql_metadata"}


class WasmError(Exception):
    pass


def read_u32(data: bytes, pos: int) -> tuple[int, int]:
    """Unsigned LEB128, at most 5 bytes."""
    result = 0
    for shift in range(0, 35, 7):
        if pos >= len(data):
            raise WasmError("truncated LEB128")
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
    raise WasmError("LEB128 longer than 5 bytes")


def read_name(data: bytes, pos: int) -> tuple[str, int]:
    length, pos = read_u32(data, pos)
    end = pos + length
    if end > len(data):
        raise WasmError("truncated name")
    return data[pos:end].decode("utf-8", errors="replace"), end


def skip_import_desc(data: bytes, pos: int) -> int:
    # Only needed to walk past an import entry; the module/field names are all we report.
    if pos >= len(data):
        raise WasmError("truncated import descriptor")
    kind = data[pos]
    pos += 1
    if kind == 0:  # func: type index
        _, pos = read_u32(data, pos)
    elif kind == 1:  # table: reftype, limits
        pos += 1
        pos = skip_limits(data, pos)
    elif kind == 2:  # memory: limits
        pos = skip_limits(data, pos)
    elif kind == 3:  # global: valtype, mutability
        pos += 2
    elif kind == 4:  # tag: attribute, type index
        pos += 1
        _, pos = read_u32(data, pos)
    else:
        raise WasmError(f"unknown import kind {kind}")
    return pos


def skip_limits(data: bytes, pos: int) -> int:
    flags, pos = read_u32(data, pos)
    _, pos = read_u32(data, pos)  # min
    if flags & 1:
        _, pos = read_u32(data, pos)  # max
    return pos


def inspect(data: bytes) -> tuple[list[str], set[str]]:
    """Return (imports as `module::name`, exported names)."""
    if data[:4] != WASM_MAGIC or data[4:HEADER_LEN] != WASM_VERSION:
        raise WasmError("not a wasm binary")
    imports: list[str] = []
    exports: set[str] = set()
    pos = HEADER_LEN
    while pos < len(data):
        section_id = data[pos]
        size, body = read_u32(data, pos + 1)
        end = body + size
        if end > len(data):
            raise WasmError("section overruns file")
        if section_id == SECTION_IMPORT:
            count, p = read_u32(data, body)
            for _ in range(count):
                module, p = read_name(data, p)
                name, p = read_name(data, p)
                p = skip_import_desc(data, p)
                imports.append(f"{module}::{name}")
        elif section_id == SECTION_EXPORT:
            count, p = read_u32(data, body)
            for _ in range(count):
                name, p = read_name(data, p)
                p += 1  # export kind
                _, p = read_u32(data, p)  # index
                exports.add(name)
        pos = end
    return imports, exports


def expected_count() -> int:
    experts = EXPERTS_WORKSPACE / "experts"
    return sum(1 for p in experts.iterdir() if (p / "Cargo.toml").is_file())


def main() -> None:
    wasm_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WASM_DIR
    modules = sorted(wasm_dir.glob("*.wasm"))
    errors: list[str] = []
    want = expected_count()
    if want == 0:
        raise SystemExit(f"no expert crates found under {EXPERTS_WORKSPACE / 'experts'}")
    if len(modules) != want:
        errors.append(f"{wasm_dir}: found {len(modules)} .wasm files, expected {want}")
    for module in modules:
        try:
            imports, exports = inspect(module.read_bytes())
        except WasmError as err:
            errors.append(f"{module.name}: {err}")
            continue
        if imports:
            errors.append(f"{module.name}: imports {', '.join(imports)} (must import nothing)")
        missing = REQUIRED_EXPORTS - exports
        if missing:
            errors.append(f"{module.name}: missing exports {', '.join(sorted(missing))}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"{len(modules)} expert modules in {wasm_dir}: no imports, ABI exports present")


if __name__ == "__main__":
    main()
