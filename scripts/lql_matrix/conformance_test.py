import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import conformance as C


def make_leg(name="lg", cells=None, descriptor=None, produce=None, meta=None):
    return C.Leg(name=name, meta=meta or {"level": name},
                 cells=cells or {}, descriptor=descriptor or {},
                 produce=produce or {})


def test_load_reads_meta_cells_descriptor_produce(tmp_path):
    d = tmp_path / "results-lgA"
    d.mkdir()
    (d / "results-lgA.jsonl").write_text(
        json.dumps({"type": "meta", "level": "lgA"}) + "\n" +
        json.dumps({"level": "lgA", "id": "stats", "bucket": "ok",
                    "exit_code": 0, "stdout_head": "(24 layers, 5K features, m)"}) + "\n",
        encoding="utf-8")
    (d / "descriptor-lgA.json").write_text(json.dumps({"name": "lgA", "family": "qwen2"}), encoding="utf-8")
    (d / "produce-lgA.json").write_text(json.dumps({"name": "lgA", "op": "extract", "bucket": "ok"}), encoding="utf-8")
    legs = C.load(str(tmp_path / "results-*/results-*.jsonl"))
    assert set(legs) == {"lgA"}
    assert legs["lgA"].cells["stats"]["bucket"] == "ok"
    assert legs["lgA"].descriptor["family"] == "qwen2"
    assert legs["lgA"].produce["op"] == "extract"


def test_run_with_empty_registry_is_green(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "INVARIANTS", [])
    legs = {"lg": make_leg()}
    monkeypatch.setattr(C, "load", lambda g: legs)
    code = C.run("x", str(tmp_path / "c.md"), str(tmp_path / "c.json"), strict=True)
    assert code == 0
    assert json.loads((tmp_path / "c.json").read_text())["violations"] == []


def _cell(cid, stdout="", bucket="ok", exit_code=0, err_signal=0, err_line="", stderr=""):
    return {"id": cid, "bucket": bucket, "exit_code": exit_code, "err_signal": err_signal,
            "err_line": err_line, "stdout_head": stdout, "stderr_head": stderr}


def test_feature_count_parses_banner_with_suffix():
    lg = make_leg(cells={"stats": _cell("stats", "Using: x (24 layers, 116.7K features, m)")})
    assert C.feature_count(lg) == 116700
    lg0 = make_leg(cells={"stats": _cell("stats", "Using: x (24 layers, 0 features, m)")})
    assert C.feature_count(lg0) == 0


def test_completeness_flags_hollow_vindex():
    hollow = make_leg("granite", cells={"stats": _cell("stats", "(24 layers, 0 features, m)")})
    healthy = make_leg("qwen", cells={"stats": _cell("stats", "(24 layers, 5K features, m)")})
    vs = C.inv_completeness({"granite": hollow, "qwen": healthy})
    assert [v.leg for v in vs] == ["granite"]
    assert vs[0].invariant == "completeness"


def test_produce_failure_is_distinct_produce_violation_not_hollow():
    # fp4-like: produce FAILED (exit 1, err bucket), descriptor knows produced=False,
    # and every corpus cell errored so there is no feature banner. This must be a
    # definitive `produce` violation, NOT a hedged completeness/hollow one.
    fp4 = make_leg("qwen15.xform.fp4",
                   cells={"stats": _cell("stats", "Error: vindex not found", err_signal=1)},
                   descriptor={"produced": False, "quant_match": False},
                   produce={"op": "quantize-fp4", "bucket": "err", "exit_code": 1})
    # a genuinely produced-but-hollow leg (granite-like) must STAY a completeness/hollow
    hollow = make_leg("granite",
                      cells={"stats": _cell("stats", "(24 layers, 0 features, m)")},
                      descriptor={"produced": True},
                      produce={"op": "extract", "bucket": "ok", "exit_code": 0})
    legs = {"qwen15.xform.fp4": fp4, "granite": hollow}

    prod = C.inv_produce(legs)
    assert [v.leg for v in prod] == ["qwen15.xform.fp4"]
    assert prod[0].invariant == "produce"
    assert "1" in prod[0].detail  # the exit code is surfaced

    comp = C.inv_completeness(legs)
    # the produce-failed leg is NOT reported as hollow; only the real hollow is
    assert [v.leg for v in comp] == ["granite"]


def test_produce_crash_left_to_no_crash_not_double_reported():
    # a produce CRASH (137) is already handled by inv_no_crash; inv_produce must
    # not also report it (avoid double-counting the same failure).
    crash = make_leg("boom", descriptor={"produced": False},
                     produce={"op": "extract", "bucket": "crash", "exit_code": 137})
    legs = {"boom": crash}
    assert any(v.invariant == "no-crash" and v.cell == "produce" for v in C.inv_no_crash(legs))
    assert C.inv_produce(legs) == []


def test_feature_count_tolerates_malformed_banner():
    lg = make_leg(cells={"stats": _cell("stats", "(24 layers, . features, m)")})
    assert C.feature_count(lg) is None
    lg2 = make_leg(cells={"stats": _cell("stats", "(24 layers, 1.2.3 features, m)")})
    assert C.feature_count(lg2) is None


def test_no_crash_flags_panic_and_crash_not_graceful():
    legs = {
        "panic": make_leg("panic", cells={"infer.top5": _cell("infer.top5", bucket="err", exit_code=101)}),
        "crash": make_leg("crash", cells={"x": _cell("x", bucket="crash", exit_code=137)}),
        "graceful": make_leg("graceful", cells={"infer.top5": _cell("infer.top5", bucket="ok", exit_code=0, err_signal=1, err_line="Error: requires model weights")}),
        "cleanerr": make_leg("cleanerr", cells={"y": _cell("y", bucket="err", exit_code=3)}),
        "produce_panic": make_leg("pp", produce={"bucket": "crash", "exit_code": 139}),
    }
    vs = C.inv_no_crash(legs)
    flagged = sorted({v.leg for v in vs})
    assert flagged == ["crash", "panic", "produce_panic"]
    assert all(v.invariant == "no-crash" for v in vs)


def test_descriptor_match_flags_quant_mismatch_and_generic_fallback():
    legs = {
        "ok": make_leg("ok", descriptor={"name": "ok", "quant_match": True, "family": "qwen2"}),
        "dequant": make_leg("dequant", descriptor={"name": "dequant", "quant_match": False,
                            "observed_quant": "none", "expect_quant": "q4k", "family": "qwen2"}),
        "generic": make_leg("generic", descriptor={"name": "generic", "quant_match": True, "family": "generic"}),
        "nodesc": make_leg("nodesc", descriptor={}),
    }
    vs = C.inv_descriptor_match(legs)
    assert sorted({v.leg for v in vs}) == ["dequant", "generic"]
    assert all(v.invariant == "descriptor-match" for v in vs)


SHOW_LAYERS_HOLLOW = ("Layer      Features  With Meta       Top Token\n"
                      "------------------------------------------------\n"
                      "L0                0          0\n"
                      "L1                0          0\n")
SHOW_LAYERS_OK = ("Layer      Features  With Meta       Top Token\n"
                  "------------------------------------------------\n"
                  "L0             2560          0\n"
                  "L1             2560          0\n")


def test_cross_check_flags_show_layers_zero_vs_stats_nonzero():
    bad = make_leg("bad", cells={
        "stats": _cell("stats", "(24 layers, 5K features, m)"),
        "show.layers": _cell("show.layers", SHOW_LAYERS_HOLLOW)})
    good = make_leg("good", cells={
        "stats": _cell("stats", "(24 layers, 5K features, m)"),
        "show.layers": _cell("show.layers", SHOW_LAYERS_OK)})
    vs = C.inv_cross_check({"bad": bad, "good": good})
    assert [v.leg for v in vs] == ["bad"]
    assert vs[0].invariant == "cross-check"


SHOW_LAYERS_MIXED = ("Layer      Features  With Meta       Top Token\n"
                     "------------------------------------------------\n"
                     "L0                0          0\n"
                     "L1             2.6K          0\n")


def test_cross_check_no_false_positive_on_ksuffix_rows():
    lg = make_leg("mixed", cells={
        "stats": _cell("stats", "(24 layers, 5K features, m)"),
        "show.layers": _cell("show.layers", SHOW_LAYERS_MIXED)})
    assert C.show_layers_total(lg) == 2600
    assert C.inv_cross_check({"mixed": lg}) == []


def test_diagnostic_flag_honored_and_message_attribution():
    silent_override = make_leg("q4kbrowse",
        produce={"op": "extract", "level": "browse", "flags": "--quant q4k",
                 "stdout_head": "Extracting…", "stderr_head": ""})
    honored = make_leg("q4kall",
        produce={"op": "extract", "level": "all", "flags": "--quant q4k",
                 "stdout_head": "Extracting…", "stderr_head": ""})
    misattrib = make_leg("mis", produce={"op": "extract", "level": "all", "flags": ""},
        cells={"infer.top5": _cell("infer.top5", bucket="err", exit_code=101,
               err_line="panicked: FFN weight tensor missing — this is a `--compact` vindex")})
    vs = C.inv_diagnostic({"q4kbrowse": silent_override, "q4kall": honored, "mis": misattrib})
    invs = sorted((v.leg, v.invariant) for v in vs)
    assert ("mis", "diagnostic") in invs
    assert ("q4kbrowse", "diagnostic") in invs
    assert not any(v.leg == "q4kall" for v in vs)


def test_load_folds_produce_err_into_stderr_head(tmp_path):
    d = tmp_path / "results-lg1"
    d.mkdir()
    (d / "results-lg1.jsonl").write_text(
        json.dumps({"type": "meta", "level": "lg1"}) + "\n", encoding="utf-8")
    (d / "produce-lg1.json").write_text(
        json.dumps({"name": "lg1", "op": "extract", "level": "browse", "flags": "--quant q4k"}),
        encoding="utf-8")
    (d / "produce-lg1.err").write_text("extract stderr, no override warning here", encoding="utf-8")
    legs = C.load(str(tmp_path / "results-*/results-*.jsonl"))
    assert "no override warning" in legs["lg1"].produce["stderr_head"]


def test_render_has_no_static_consumer_prose():
    legs = {"qwen.q4k": make_leg("qwen.q4k",
                                 produce={"op": "extract", "flags": "--quant q4k"},
                                 descriptor={"family": "qwen2", "observed_quant": "q4k"})}
    md = C.render(legs, [])
    assert not hasattr(C, "transformation_report")
    for banned in ("Consumer reachability", "Ollama", "Transformation & consumability"):
        assert banned not in md


PANIC = ("panic/crash (exit 101): thread 'larql-main' ({tid}) panicked at "
         "crates/larql-compute/src/sparse_compute.rs:30:9")


def _panic_violations(legs=("legA", "legB"), ncells=17, tid_base=3000):
    out = []
    for li, leg in enumerate(legs):
        for ci in range(ncells):
            out.append(C.Violation("no-crash", leg, f"cell{ci}",
                                   PANIC.format(tid=tid_base + li * 100 + ci)))
    return out


def test_root_causes_collapse_panic_family_into_one_row():
    vs = _panic_violations()
    assert len(vs) == 34
    rcs = C.root_causes(vs)
    assert len(rcs) == 1
    rc = rcs[0]
    assert rc["invariant"] == "no-crash"
    assert rc["violations"] == 34
    assert rc["legs"] == 2
    assert rc["cells"] == 17
    assert rc["leg_names"] == ["legA", "legB"]
    assert "sparse_compute.rs:30:9" in rc["example"]
    assert "exit 101" in rc["detail"]  # exit codes are kept in the key


def test_root_causes_keep_distinct_invariants_and_details_distinct():
    vs = [C.Violation("no-crash", "a", "c1", "panic/crash (exit 101): boom"),
          C.Violation("no-crash", "a", "c2", "panic/crash (exit 137): boom"),
          C.Violation("produce", "a", "produce", "panic/crash (exit 101): boom"),
          C.Violation("no-crash", "b", "c1", "panic/crash (exit 101): boom")]
    rcs = C.root_causes(vs)
    assert len(rcs) == 3
    top = rcs[0]  # sorted by count desc
    assert (top["invariant"], top["violations"], top["legs"]) == ("no-crash", 2, 2)
    assert {r["invariant"] for r in rcs} == {"no-crash", "produce"}


def test_normalise_detail_strips_thread_ids_only():
    n = C.normalise_detail
    assert n("thread 'larql-main' (3135) panicked at x.rs:1:2") == n(
        "thread 'larql-main' (9) panicked at x.rs:1:2")
    assert "3135" not in n("thread 'larql-main' (3135) panicked")
    assert "larql-main" in n("thread 'larql-main' (3135) panicked")
    # line numbers, exit codes survive
    assert "x.rs:1:2" in n("thread 'm' (7) panicked at x.rs:1:2")
    assert "(exit 101)" in n("panic/crash (exit 101): y")
    # a bare, id-less thread banner groups with the id-bearing one
    assert n("thread 'm' panicked at z") == n("thread 'm' (4) panicked at z")


def _repr_detail(tid, leg, tmp, head_extra=""):
    # what inv_no_crash/inv_produce build: repr() of stderr_head holding BOTH quote
    # kinds, so repr delimits with ' and escapes the inner ' as \\'
    head = (f"thread 'larql-main' ({tid}) panicked at crates/x/src/y.rs:5:7:\n"
            f"cannot open \"out/{leg}.vindex\" under {tmp}{head_extra}")
    return f"produce crashed (exit 101): {head!r}"


def test_normalise_thread_id_inside_repr_with_escaped_quotes():
    d1 = _repr_detail(3135, "legA", "/tmp/tmp.AbC123")
    d2 = _repr_detail(9, "legA", "/tmp/tmp.AbC123")
    assert "\\'" in d1                       # the escaped-quote shape is really exercised
    assert C.normalise_detail(d1) == C.normalise_detail(d2)
    assert "3135" not in C.normalise_detail(d1)


def test_normalise_strips_leg_vindex_path_tmproot_and_leg_name():
    a = _repr_detail(11, "legA", "/tmp/tmp.AbC123")
    b = _repr_detail(22, "legB", "/tmp/tmp.Zz9Y8x")
    assert C.normalise_detail(a, leg="legA") == C.normalise_detail(b, leg="legB")
    n = C.normalise_detail(a, leg="legA")
    assert "legA" not in n and "AbC123" not in n
    assert "y.rs:5:7" in n                      # source location survives
    # leg-name stripping must not eat substrings of unrelated words
    assert C.normalise_detail("panicked in alpha", leg="a") == "panicked in alpha"
    # a leg that is a dotted prefix of another leg is not clobbered inside the longer one
    assert "smol135.v3.hollow" in C.normalise_detail("x smol135.v3.hollow", leg="smol135.v3")


def test_root_causes_one_failure_across_legs_and_thread_ids_is_one_row():
    vs = [C.Violation("no-crash", leg, "cellX",
                      _repr_detail(100 + i, leg, f"/tmp/tmp.R{i}rand"))
          for i, leg in enumerate(["legA", "legB", "legC"])]
    rcs = C.root_causes(vs)
    assert len(rcs) == 1
    assert rcs[0]["violations"] == 3 and rcs[0]["legs"] == 3


def test_root_causes_stub_crashes_split_by_cell():
    vs = [C.Violation("no-crash", "legA", "show.layers", "panic/crash (exit 139)"),
          C.Violation("no-crash", "legB", "show.layers", "panic/crash (exit 139)"),
          C.Violation("no-crash", "legA", "infer.top5", "panic/crash (exit 139)")]
    rcs = C.root_causes(vs)
    assert len(rcs) == 2
    top = rcs[0]
    assert (top["violations"], top["legs"], top["cells"]) == (2, 2, 1)
    assert "show.layers" in top["detail"]
    assert "infer.top5" in rcs[1]["detail"]


def test_root_causes_stub_with_location_or_message_still_collapses():
    vs = [C.Violation("no-crash", "l1", "c1", "panic/crash (exit 101): boom"),
          C.Violation("no-crash", "l1", "c2", "panic/crash (exit 101): boom")]
    assert len(C.root_causes(vs)) == 1


def test_root_causes_produce_crash_stub_collapses_across_legs():
    vs = [C.Violation("no-crash", leg, "produce", "produce crashed (exit 137)")
          for leg in ("legA", "legB")]
    rcs = C.root_causes(vs)
    assert len(rcs) == 1 and rcs[0]["legs"] == 2


def test_root_cause_example_is_clipped():
    vs = [C.Violation("no-crash", "a", "c", "x" * 1000)]
    assert len(C.root_causes(vs)[0]["example"]) <= 240


def test_root_cause_leg_names_capped():
    vs = [C.Violation("completeness", f"leg{i}", "", "hollow") for i in range(9)]
    rc = C.root_causes(vs)[0]
    assert rc["legs"] == 9
    assert len(rc["leg_names"]) == C.ROOT_CAUSE_LEG_NAMES
    assert rc["cells"] == 0  # leg-level violations carry no cell


def test_render_contains_root_causes_section_and_full_violations():
    vs = _panic_violations()
    md = C.render({"legA": make_leg("legA"), "legB": make_leg("legB")}, vs)
    assert "## Root causes" in md
    assert "Violations: 34" in md
    assert md.count("sparse_compute.rs:30:9") >= 35  # 34 rows + 1 root-cause row
    rc_section = md.split("## Root causes", 1)[1]
    assert "| 34 | 2 | 17 |" in rc_section


def test_render_no_violations_has_no_root_cause_rows():
    md = C.render({"lg": make_leg()}, [])
    assert "No invariant violations." in md
    assert "No root causes." in md


def test_json_has_violations_and_root_causes(tmp_path, monkeypatch):
    vs = _panic_violations()
    monkeypatch.setattr(C, "INVARIANTS", [lambda legs: vs])
    monkeypatch.setattr(C, "load", lambda g: {"legA": make_leg("legA")})
    C.run("x", str(tmp_path / "c.md"), str(tmp_path / "c.json"), strict=False)
    data = json.loads((tmp_path / "c.json").read_text())
    assert list(data) == ["violations", "root_causes"]
    assert data["violations"] == [v.__dict__ for v in vs]  # unchanged shape
    assert len(data["root_causes"]) == 1
    assert data["root_causes"][0]["violations"] == 34


def test_strict_exit_unchanged_by_grouping(tmp_path, monkeypatch):
    vs = _panic_violations()
    monkeypatch.setattr(C, "INVARIANTS", [lambda legs: vs])
    monkeypatch.setattr(C, "load", lambda g: {"legA": make_leg("legA")})
    assert C.run("x", str(tmp_path / "c.md"), str(tmp_path / "c.json"), strict=True) == 1
    assert C.run("x", str(tmp_path / "c.md"), str(tmp_path / "c.json"), strict=False) == 0
    monkeypatch.setattr(C, "INVARIANTS", [lambda legs: []])
    assert C.run("x", str(tmp_path / "c.md"), str(tmp_path / "c.json"), strict=True) == 0


def test_run_strict_fails_on_violation(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "INVARIANTS", [lambda legs: [C.Violation("x", "lg", "", "boom")]])
    monkeypatch.setattr(C, "load", lambda g: {"lg": make_leg()})
    assert C.run("x", str(tmp_path/"c.md"), str(tmp_path/"c.json"), strict=True) == 1
    assert C.run("x", str(tmp_path/"c.md"), str(tmp_path/"c.json"), strict=False) == 0


def test_completeness_v3_uses_descriptor_not_stats_banner():
    # a v3 leg has no STATS "N features" banner (no cells at all here) — the v2
    # code path would call it "unknown"/produce-may-have-failed; v3 must instead
    # read has_model_weights straight off the descriptor.
    healthy_v3 = make_leg("smol135.v3",
        descriptor={"generation": "v3", "has_model_weights": True, "num_representations": 42})
    hollow_v3 = make_leg("smol135.v3.hollow",
        descriptor={"generation": "v3", "has_model_weights": False, "num_representations": 0})
    vs = C.inv_completeness({"smol135.v3": healthy_v3, "smol135.v3.hollow": hollow_v3})
    assert [v.leg for v in vs] == ["smol135.v3.hollow"]
    assert "v3" in vs[0].detail


def test_non_dict_sidecar_does_not_crash(tmp_path):
    d = tmp_path / "results-lg"
    d.mkdir()
    (d / "results-lg.jsonl").write_text(
        json.dumps({"type": "meta", "level": "lg"}) + "\n", encoding="utf-8")
    (d / "produce-lg.json").write_text("[1, 2, 3]", encoding="utf-8")   # valid JSON, not a dict
    (d / "descriptor-lg.json").write_text("null", encoding="utf-8")
    legs = C.load(str(tmp_path / "results-*/results-*.jsonl"))
    assert legs["lg"].produce == {}
    assert legs["lg"].descriptor == {}
    # running the full oracle over this leg must not raise
    C.run(str(tmp_path / "results-*/results-*.jsonl"),
          str(tmp_path / "m.md"), str(tmp_path / "j.json"), strict=False)
