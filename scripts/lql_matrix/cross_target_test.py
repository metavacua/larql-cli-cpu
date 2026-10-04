import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import cross_target as X

STATS = "(24 layers, 5K features, m)"


def write_instance(root, tag, leg="smol", cells=None, descriptor=None, produce=None):
    """Lay out one artifact directory exactly as CI names it: results-<TAG>-<leg>."""
    d = root / f"results-{tag}-{leg}"
    d.mkdir(parents=True)
    rows = [{"type": "meta", "level": leg}]
    for c in (cells if cells is not None else [_cell("stats", stdout=STATS), _cell("show.layers")]):
        rows.append({"level": leg, **c})
    (d / f"results-{leg}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    desc = {"name": leg, "family": "llama", "observed_quant": "f16", "generation": "v2",
            "produced": True, "has_model_weights": True}
    desc.update(descriptor or {})
    (d / f"descriptor-{leg}.json").write_text(json.dumps(desc), encoding="utf-8")
    prod = {"name": leg, "op": "extract", "exit_code": 0, "bucket": "ok"}
    prod.update(produce or {})
    (d / f"produce-{leg}.json").write_text(json.dumps(prod), encoding="utf-8")


def _cell(cid, stdout="", bucket="ok", exit_code=0, duration_ms=10, stderr="", cat="browse",
          err_signal=0, err_line=""):
    return {"id": cid, "cat": cat, "bucket": bucket, "exit_code": exit_code,
            "duration_ms": duration_ms, "stdout_head": stdout, "stderr_head": stderr,
            "err_signal": err_signal, "err_line": err_line}


A, B = "x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"
Q = "riscv64gc-unknown-linux-gnu"
GLOB = "results-*/results-*.jsonl"


def compare(tmp_path, slow=()):
    return X.compare(X.load_instances(str(tmp_path / GLOB)), slow_tags=slow)


def facts(report, status):
    return {(f["leg"], f["fact"]) for f in report["findings"] if f["status"] == status}


def test_identical_instances_agree(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B)
    r = compare(tmp_path)
    assert r["instances"] == sorted([A, B])
    assert r["disagreements"] == 0 and r["inconclusive"] == 0


def test_differing_exit_code_is_a_disagreement(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, cells=[_cell("stats", stdout=STATS), _cell("show.layers", exit_code=1)])
    assert ("smol", "cell show.layers exit_code") in facts(compare(tmp_path), "disagree")


def test_differing_bucket_is_a_disagreement(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, cells=[_cell("stats", stdout=STATS), _cell("show.layers", bucket="err")])
    assert ("smol", "cell show.layers bucket") in facts(compare(tmp_path), "disagree")


def test_differing_feature_count_is_a_disagreement(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, cells=[_cell("stats", stdout="(24 layers, 6K features, m)"),
                                       _cell("show.layers")])
    assert ("smol", "feature_count") in facts(compare(tmp_path), "disagree")


def test_differing_descriptor_field_is_a_disagreement(tmp_path):
    for field, value in [("family", "qwen2"), ("observed_quant", "q4k"), ("generation", "v3"),
                         ("has_model_weights", False)]:
        root = tmp_path / field
        root.mkdir()
        write_instance(root, A)
        write_instance(root, B, descriptor={field: value})
        r = X.compare(X.load_instances(str(root / GLOB)))
        assert ("smol", f"descriptor {field}") in facts(r, "disagree"), field


def test_differing_produced_and_produce_outcome_are_disagreements(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, descriptor={"produced": False}, produce={"exit_code": 1, "bucket": "err"})
    f = facts(compare(tmp_path), "disagree")
    assert ("smol", "descriptor produced") in f
    assert ("smol", "produce outcome") in f


def test_differing_leg_set_is_a_disagreement(tmp_path):
    write_instance(tmp_path, A, leg="smol")
    write_instance(tmp_path, A, leg="smol-instruct")
    write_instance(tmp_path, B, leg="smol")
    r = compare(tmp_path)
    assert ("smol-instruct", "leg present") in facts(r, "disagree")


def test_differing_cell_set_is_a_disagreement(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, cells=[_cell("stats", stdout=STATS)])
    assert ("smol", "cell show.layers present") in facts(compare(tmp_path), "disagree")


def test_stdout_stderr_and_duration_are_not_compared(tmp_path):
    write_instance(tmp_path, A, cells=[_cell("stats", stdout=STATS, duration_ms=5, stderr="a"),
                                       _cell("show.layers", stdout="the moon is cheese")])
    write_instance(tmp_path, B, cells=[_cell("stats", stdout=STATS, duration_ms=90000, stderr="b"),
                                       _cell("show.layers", stdout="gibberish 0.1234")])
    r = compare(tmp_path)
    assert r["disagreements"] == 0


def test_timeout_makes_a_cell_inconclusive_not_a_disagreement(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, cells=[_cell("stats", stdout=STATS),
                                       _cell("show.layers", bucket="timeout", exit_code=124)])
    r = compare(tmp_path, slow=[B])
    assert r["disagreements"] == 0
    assert ("smol", "cell show.layers") in facts(r, "inconclusive")


def test_timeout_on_the_first_instance_is_also_inconclusive(tmp_path):
    write_instance(tmp_path, A, cells=[_cell("stats", stdout=STATS),
                                       _cell("show.layers", bucket="timeout", exit_code=124)])
    write_instance(tmp_path, B)
    r = compare(tmp_path, slow=[A])
    assert r["disagreements"] == 0 and r["inconclusive"] >= 1


def test_produce_timeout_makes_the_whole_leg_inconclusive(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, descriptor={"produced": False, "family": ""},
                   produce={"exit_code": 124, "bucket": "timeout"},
                   cells=[_cell("stats", bucket="err", exit_code=1)])
    r = compare(tmp_path, slow=[B])
    assert r["disagreements"] == 0
    assert ("smol", "produce outcome") in facts(r, "inconclusive")


def test_one_instance_is_nothing_to_compare_and_exit_zero_under_strict(tmp_path):
    write_instance(tmp_path, A)
    md, js = tmp_path / "o.md", tmp_path / "o.json"
    code = X.main([str(tmp_path / GLOB), "--strict", "--out-md", str(md), "--out-json", str(js)])
    assert code == 0
    assert "nothing to compare" in md.read_text().lower()
    assert json.loads(js.read_text())["disagreements"] == 0


def test_no_artifacts_is_nothing_to_compare(tmp_path):
    assert X.main([str(tmp_path / GLOB), "--strict"]) == 0


def test_strict_exits_one_iff_a_conclusive_disagreement(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, descriptor={"family": "qwen2"})
    md, js = tmp_path / "o.md", tmp_path / "o.json"
    args = [str(tmp_path / GLOB), "--out-md", str(md), "--out-json", str(js)]
    assert X.main(args) == 0                      # discovery preserved without --strict
    assert X.main(args + ["--strict"]) == 1
    assert "descriptor family" in md.read_text()
    assert json.loads(js.read_text())["disagreements"] == 1


def test_strict_is_green_when_only_inconclusive(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, cells=[_cell("stats", stdout=STATS),
                                       _cell("show.layers", bucket="timeout", exit_code=124)])
    assert X.main([str(tmp_path / GLOB), "--strict", "--slow-tag", B]) == 0


def test_tag_parsing_with_hyphenated_targets_and_dotted_legs():
    assert X.parse_tag("results-riscv64gc-unknown-linux-gnu-smol-135m.q4k", ["smol-135m.q4k"]) \
        == "riscv64gc-unknown-linux-gnu"
    assert X.parse_tag("results-x86_64-unknown-linux-gnu-a", ["a"]) == "x86_64-unknown-linux-gnu"
    # a leg that is a suffix of another leg must not be confused with it
    assert X.parse_tag("results-t-1-smol-instruct", ["instruct", "smol-instruct"]) == "t-1"


def test_load_instances_groups_legs_by_tag_with_dotted_legs(tmp_path):
    write_instance(tmp_path, "riscv64gc-unknown-linux-gnu", leg="smol-135m.q4k")
    write_instance(tmp_path, "riscv64gc-unknown-linux-gnu", leg="smol-135m")
    write_instance(tmp_path, A, leg="smol-135m")
    inst = X.load_instances(str(tmp_path / GLOB))
    assert set(inst) == {"riscv64gc-unknown-linux-gnu", A}
    assert set(inst["riscv64gc-unknown-linux-gnu"]) == {"smol-135m.q4k", "smol-135m"}


# ---- stream B review fixes -------------------------------------------------------------

import pytest  # noqa: E402

KILLED = dict(bucket="crash", exit_code=137)
TIMED = dict(bucket="timeout", exit_code=124)
RV = "riscv64gc-unknown-linux-gnu"


def test_hard_killed_cell_is_inconclusive_on_a_slow_tag_not_a_crash_disagreement(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, cells=[_cell("stats", stdout=STATS), _cell("show.layers", **KILLED)])
    r = compare(tmp_path, slow=[B])
    assert r["disagreements"] == 0
    assert ("smol", "cell show.layers") in facts(r, "inconclusive")


def test_exit_124_or_137_is_speed_dependent_whatever_the_bucket(tmp_path):
    for code in (124, 137):
        root = tmp_path / str(code)
        root.mkdir()
        write_instance(root, A)
        write_instance(root, B, cells=[_cell("stats", stdout=STATS),
                                       _cell("show.layers", bucket="err", exit_code=code)])
        r = compare(root, slow=[B])
        assert r["disagreements"] == 0, code
        assert ("smol", "cell show.layers") in facts(r, "inconclusive"), code


def test_other_crash_codes_are_still_conclusive(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, cells=[_cell("stats", stdout=STATS),
                                       _cell("show.layers", bucket="crash", exit_code=139)])
    assert ("smol", "cell show.layers exit_code") in facts(compare(tmp_path, slow=[B]), "disagree")


def test_hard_killed_produce_is_inconclusive_on_a_slow_tag(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, descriptor={"produced": False, "family": ""},
                   produce={"exit_code": 137, "bucket": "crash"},
                   cells=[_cell("stats", bucket="err", exit_code=1)])
    r = compare(tmp_path, slow=[B])
    assert r["disagreements"] == 0
    assert ("smol", "produce outcome") in facts(r, "inconclusive")


def test_timeout_on_a_non_slow_tag_where_a_peer_finished_is_a_disagreement(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, cells=[_cell("stats", stdout=STATS), _cell("show.layers", **TIMED)])
    r = compare(tmp_path)                       # no slow tags: a hang here is target-specific
    assert ("smol", "cell show.layers") in facts(r, "disagree")


def test_hard_kill_on_a_non_slow_tag_where_a_peer_finished_is_a_disagreement(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, cells=[_cell("stats", stdout=STATS), _cell("show.layers", **KILLED)])
    assert ("smol", "cell show.layers") in facts(compare(tmp_path, slow=[RV]), "disagree")


def test_a_slow_tag_does_not_excuse_a_different_non_slow_tag(tmp_path):
    write_instance(tmp_path, A, cells=[_cell("stats", stdout=STATS), _cell("show.layers", **TIMED)])
    write_instance(tmp_path, B)
    write_instance(tmp_path, RV)
    r = compare(tmp_path, slow=[RV])
    assert ("smol", "cell show.layers") in facts(r, "disagree")


def test_every_instance_timing_out_on_a_cell_agree(tmp_path):
    cells = [_cell("stats", stdout=STATS), _cell("show.layers", **TIMED)]
    write_instance(tmp_path, A, cells=cells)
    write_instance(tmp_path, B, cells=cells)
    r = compare(tmp_path)
    assert r["disagreements"] == 0
    assert ("smol", "cell show.layers") in facts(r, "agree")


def test_produce_hang_on_a_non_slow_tag_where_a_peer_finished_is_a_disagreement(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, descriptor={"produced": False}, produce=TIMED,
                   cells=[_cell("stats", bucket="err", exit_code=1)])
    assert ("smol", "produce outcome") in facts(compare(tmp_path), "disagree")


def test_leg_missing_from_a_non_slow_tag_is_a_disagreement_even_if_a_slow_tag_exists(tmp_path):
    write_instance(tmp_path, A, leg="smol")
    write_instance(tmp_path, A, leg="smol-instruct")
    write_instance(tmp_path, B, leg="smol")
    write_instance(tmp_path, RV, leg="smol")
    write_instance(tmp_path, RV, leg="smol-instruct")
    r = compare(tmp_path, slow=[RV])
    assert ("smol-instruct", "leg present") in facts(r, "disagree")


def test_leg_missing_only_from_a_slow_tag_is_inconclusive(tmp_path):
    write_instance(tmp_path, A, leg="smol")
    write_instance(tmp_path, A, leg="smol-instruct")
    write_instance(tmp_path, B, leg="smol")
    write_instance(tmp_path, B, leg="smol-instruct")
    write_instance(tmp_path, RV, leg="smol")
    r = compare(tmp_path, slow=[RV])
    assert r["disagreements"] == 0
    assert ("smol-instruct", "leg present") in facts(r, "inconclusive")


def test_cell_missing_only_from_a_slow_tag_is_inconclusive(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, cells=[_cell("stats", stdout=STATS)])
    r = compare(tmp_path, slow=[B])
    assert r["disagreements"] == 0
    assert ("smol", "cell show.layers present") in facts(r, "inconclusive")


def test_slow_tag_option_is_repeatable_and_reported(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, cells=[_cell("stats", stdout=STATS), _cell("show.layers", **TIMED)])
    write_instance(tmp_path, RV, cells=[_cell("stats", stdout=STATS), _cell("show.layers", **TIMED)])
    js = tmp_path / "o.json"
    args = [str(tmp_path / GLOB), "--strict", "--out-json", str(js)]
    assert X.main(args + ["--slow-tag", B]) == 1          # RV timed out, not excused
    assert X.main(args + ["--slow-tag", B, "--slow-tag", RV]) == 0
    assert sorted(json.loads(js.read_text())["slow_tags"]) == sorted([B, RV])


def test_err_signal_is_compared_for_a_non_fp_category(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, cells=[_cell("stats", stdout=STATS),
                                       _cell("show.layers", err_signal=1, err_line="Error: x")])
    assert ("smol", "cell show.layers err_signal") in facts(compare(tmp_path), "disagree")


def test_err_signal_is_not_compared_for_fp_dependent_categories(tmp_path):
    for cat in sorted(X.FP_DEPENDENT_CATS):
        root = tmp_path / cat
        root.mkdir()
        write_instance(root, A, cells=[_cell("stats", stdout=STATS), _cell("c", cat=cat)])
        write_instance(root, B, cells=[_cell("stats", stdout=STATS),
                                       _cell("c", cat=cat, err_signal=1, err_line="Error: x")])
        assert compare(root)["disagreements"] == 0, cat


def test_err_line_text_is_never_compared(tmp_path):
    write_instance(tmp_path, A, cells=[_cell("stats", stdout=STATS),
                                       _cell("show.layers", err_signal=1, err_line="Error: aaa 0.1")])
    write_instance(tmp_path, B, cells=[_cell("stats", stdout=STATS),
                                       _cell("show.layers", err_signal=1, err_line="Error: bbb 0.2")])
    assert compare(tmp_path)["disagreements"] == 0


def test_fp_dependent_categories_are_real_corpus_categories():
    cats = {json.loads(line)["cat"] for line in
            open(os.path.join(os.path.dirname(__file__), "commands.jsonl")) if line.strip()}
    assert {"inference", "roundtrip"} <= X.FP_DEPENDENT_CATS <= cats
    # the deterministic categories stay compared
    assert not ({"browse", "lifecycle", "negative"} & X.FP_DEPENDENT_CATS)


def test_feature_count_is_inconclusive_when_any_cell_of_the_leg_timed_out(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, cells=[_cell("stats", stdout="(24 layers, 6K features, m)"),
                                       _cell("show.layers", **TIMED)])
    r = compare(tmp_path, slow=[B])
    assert r["disagreements"] == 0
    assert ("smol", "feature_count") in facts(r, "inconclusive")


def test_feature_count_is_inconclusive_when_the_timeout_is_on_the_first_instance(tmp_path):
    write_instance(tmp_path, A, cells=[_cell("stats", stdout="(24 layers, 6K features, m)"),
                                       _cell("show.layers", **KILLED)])
    write_instance(tmp_path, B)
    r = compare(tmp_path, slow=[A])
    assert r["disagreements"] == 0
    assert ("smol", "feature_count") in facts(r, "inconclusive")


def test_a_timeout_elsewhere_does_not_hide_a_feature_count_disagreement_of_other_pairs(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, cells=[_cell("stats", stdout="(24 layers, 6K features, m)"),
                                       _cell("show.layers")])
    write_instance(tmp_path, RV, cells=[_cell("stats", stdout=STATS), _cell("show.layers", **TIMED)])
    assert ("smol", "feature_count") in facts(compare(tmp_path, slow=[RV]), "disagree")


def test_descriptor_comparison_skips_an_instance_whose_produce_timed_out(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B, descriptor={"family": "", "produced": False}, produce=TIMED,
                   cells=[_cell("stats", bucket="err", exit_code=1)])
    r = compare(tmp_path, slow=[B])
    assert r["disagreements"] == 0
    assert ("smol", "descriptor family") not in facts(r, "disagree")


def test_parse_tag_fails_loudly_when_the_directory_does_not_end_with_the_leg():
    with pytest.raises(X.TagError) as e:
        X.parse_tag("results-weird", ["nope"])
    assert "results-weird" in str(e.value)
    with pytest.raises(X.TagError):
        X.parse_tag("results-smol", ["smol"])             # no tag left
    with pytest.raises(X.TagError):
        X.parse_tag("artifact-x-smol", ["smol"])          # not results-<TAG>-<leg>


def test_load_instances_raises_naming_the_bad_directory(tmp_path):
    write_instance(tmp_path, A)
    bad = tmp_path / "results-orphan-other"
    bad.mkdir()
    (bad / "results-smol.jsonl").write_text(
        json.dumps({"type": "meta", "level": "smol"}) + "\n", encoding="utf-8")
    with pytest.raises(X.TagError) as e:
        X.load_instances(str(tmp_path / GLOB))
    assert "results-orphan-other" in str(e.value)


def test_main_exits_two_on_a_bad_directory_name_and_names_it(tmp_path, capsys):
    write_instance(tmp_path, A)
    bad = tmp_path / "results-orphan-other"
    bad.mkdir()
    (bad / "results-smol.jsonl").write_text(
        json.dumps({"type": "meta", "level": "smol"}) + "\n", encoding="utf-8")
    assert X.main([str(tmp_path / GLOB), "--strict"]) == 2
    assert "results-orphan-other" in capsys.readouterr().err


def test_duplicate_tag_and_leg_is_an_error_not_an_overwrite(tmp_path, capsys):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B)
    d = tmp_path / f"results-{A}-smol"
    (d / "results-smol-again.jsonl").write_text(
        (d / "results-smol.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(X.DuplicateError):
        X.load_instances(str(tmp_path / GLOB))
    assert X.main([str(tmp_path / GLOB), "--strict"]) == 2
    assert f"results-{A}-smol" in capsys.readouterr().err


def test_every_corpus_category_is_classified_deterministic_or_fp_dependent():
    # FP_DEPENDENT_CATS is a deny-list for err_signal. A category added to commands.jsonl would
    # otherwise be silently treated as deterministic; this forces a deliberate decision.
    cats = {json.loads(line)["cat"] for line in
            open(os.path.join(os.path.dirname(__file__), "commands.jsonl")) if line.strip()}
    assert not (X.FP_DEPENDENT_CATS & X.DETERMINISTIC_CATS)
    assert cats == X.FP_DEPENDENT_CATS | X.DETERMINISTIC_CATS


def test_an_expected_native_instance_with_no_artifacts_is_a_disagreement(tmp_path):
    # aarch64's whole lql run uploaded nothing: only A is loaded, B never appears.
    write_instance(tmp_path, A)
    r = X.compare(X.load_instances(str(tmp_path / GLOB)), expect_tags=[A, B])
    assert ("(instance)", "instance present") in facts(r, "disagree")
    assert r["disagreements"] == 1


def test_an_expected_slow_instance_with_no_artifacts_is_inconclusive(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B)
    r = X.compare(X.load_instances(str(tmp_path / GLOB)), slow_tags=[Q], expect_tags=[A, B, Q])
    assert ("(instance)", "instance present") in facts(r, "inconclusive")
    assert r["disagreements"] == 0


def test_expected_instances_all_present_adds_nothing(tmp_path):
    write_instance(tmp_path, A)
    write_instance(tmp_path, B)
    r = X.compare(X.load_instances(str(tmp_path / GLOB)), expect_tags=[A, B])
    assert r["disagreements"] == 0 and r["inconclusive"] == 0
    assert ("(instance)", "instance present") not in facts(r, "agree")


def test_strict_exits_1_when_an_expected_native_instance_is_missing(tmp_path):
    write_instance(tmp_path, A)
    argv = [str(tmp_path / GLOB), "--strict", "--expect-tag", A, "--expect-tag", B]
    assert X.main(argv) == 1
    assert X.main(argv[:2] + ["--expect-tag", A]) == 0  # nothing expected is missing


def test_render_shows_a_missing_expected_instance_even_with_one_instance_left(tmp_path):
    write_instance(tmp_path, A)
    r = X.compare(X.load_instances(str(tmp_path / GLOB)), expect_tags=[A, B])
    md = X.render(r)
    assert "Nothing to compare" not in md
    assert "instance present" in md and f"`{B}`" in md  # the missing tag has its own column
    assert "Disagreements" in md
