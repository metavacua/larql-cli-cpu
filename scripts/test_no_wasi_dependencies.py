import tempfile
from pathlib import Path
import unittest

from check_no_wasi_dependencies import banned_packages, manifest_errors


class NoWasiDependencyTests(unittest.TestCase):
    def check(self, manifest):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Cargo.toml"
            path.write_text(manifest)
            return manifest_errors(path)

    def test_plain_wasmtime_is_allowed(self):
        self.assertEqual(self.check('[dependencies]\nwasmtime = "36"'), [])

    def test_wasmtime_wasi_is_caught(self):
        self.assertTrue(self.check('[dependencies]\nwasmtime-wasi = "36"'))

    def test_renamed_and_target_specific_edges_are_caught(self):
        self.assertTrue(self.check('[dependencies]\nw = { package = "wasmtime-wasi", version = "36" }'))
        self.assertTrue(self.check('[target.\'cfg(unix)\'.dev-dependencies]\ncap-net-ext = "3"'))

    def test_tree_scan(self):
        tree = "larql-inference v0.1.0\nwasmtime v36.0.16\ncap-net-ext v3.4.0\n"
        self.assertEqual(banned_packages(tree), {"cap-net-ext"})
        self.assertEqual(banned_packages("tokio v1\nurl v2\n"), set())


if __name__ == "__main__":
    unittest.main()
