"""Unit tests for lqlfacts.py's IR extractor (no compiler needed)."""
import os, subprocess, sys, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "lqlfacts.py")

def facts(d, name):
    with open(os.path.join(d, name + ".facts")) as f:
        return {tuple(l.rstrip("\n").split("\t")) for l in f if l.strip()}

class IrExtract(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp()
        subprocess.run([sys.executable, TOOL, "ir", "--demangler", "cat",
                        "--crate", "larql_lql", os.path.join(HERE, "fixture.ll"), self.out], check=True)

    def test_direct_and_invoke_edges(self):
        e = facts(self.out, "calls")
        self.assertIn(("exec_hidden", "peek"), e)
        self.assertIn(("exec_walk", "require_vindex"), e)
        self.assertIn(("exec_show_models", "list_dir"), e)

    def test_foreign_callees_are_dropped(self):
        self.assertNotIn(("exec_hidden", "format"), facts(self.out, "calls"))

    def test_closure_attributed_to_enclosing_fn(self):
        self.assertIn(("exec_stats", "1"), facts(self.out, "indirect"))

    def test_trait_impl_method_named_by_method(self):
        self.assertIn(("fmt", "exec_show_models"), facts(self.out, "calls"))

if __name__ == "__main__":
    unittest.main()
