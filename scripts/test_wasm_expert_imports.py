import unittest

from check_wasm_expert_imports import WasmError, inspect, REQUIRED_EXPORTS

HEADER = b"\0asm\x01\0\0\0"


def name(text: str) -> bytes:
    raw = text.encode()
    return bytes([len(raw)]) + raw


def section(section_id: int, body: bytes) -> bytes:
    return bytes([section_id, len(body)]) + body


def export_section(names) -> bytes:
    body = bytes([len(names)]) + b"".join(name(n) + b"\x02\x00" for n in names)
    return section(7, body)


class InspectTests(unittest.TestCase):
    def test_empty_module_has_nothing(self):
        self.assertEqual(inspect(HEADER), ([], set()))

    def test_wasi_import_is_reported(self):
        # type section (4 x i32 -> i32), then one func import.
        types = section(1, bytes([1, 0x60, 4, 0x7F, 0x7F, 0x7F, 0x7F, 1, 0x7F]))
        imports = section(2, bytes([1]) + name("wasi_snapshot_preview1") + name("fd_write") + b"\x00\x00")
        found, _ = inspect(HEADER + types + imports)
        self.assertEqual(found, ["wasi_snapshot_preview1::fd_write"])

    def test_exports_are_collected(self):
        _, exports = inspect(HEADER + export_section(sorted(REQUIRED_EXPORTS)))
        self.assertEqual(exports, REQUIRED_EXPORTS)

    def test_rejects_non_wasm(self):
        with self.assertRaises(WasmError):
            inspect(b"not wasm at all")

    def test_rejects_truncated_section(self):
        with self.assertRaises(WasmError):
            inspect(HEADER + bytes([2, 50, 1]))


if __name__ == "__main__":
    unittest.main()
