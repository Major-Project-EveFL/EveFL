"""
`evefl/selfcheck.py`: the live system check shown on the dashboard's home page.

Every check calls real library code (controller, BB84 simulator, link protection, secure-aggregation primitives, the recorded
validation files, a saved checkpoint). A check that fails or raises is reported as a failure, never hidden and never fatal to the
others; a check that cannot run (no checkpoint on this machine) is reported as skipped with the reason.
"""

import re

import torch
from flwr.common import ndarrays_to_parameters

from evefl import selfcheck as sc
from evefl.fl.checkpoint import CheckpointSaver
from evefl.fl.client import get_model_parameters
from evefl.fl.model import build_resnet18

AUC = "macro_auc_roc_excl_thin"


def _checkpoint_dir(tmp_path):
    torch.manual_seed(0)
    params = ndarrays_to_parameters(get_model_parameters(build_resnet18(pretrained=False)))
    saver = CheckpointSaver(tmp_path, name="run", metadata={"config_hash": "abc123", "experiment": "run"})
    saver.observe(7, params, (0.2, {AUC: 0.7}))
    saver.finalize(params, final_round=9)
    return tmp_path


def _by_name(results):
    return {r.name: r for r in results}


def test_every_library_check_passes_and_reports_what_it_ran_and_how_long(tmp_path):
    results = sc.run_checks(checkpoint_dir=tmp_path / "no_checkpoint_here")
    assert len(results) >= 6
    assert all(r.status in ("pass", "skip") for r in results), [(r.name, r.status, r.detail) for r in results]
    for r in results:
        assert r.title and r.what and r.detail and r.seconds >= 0.0
        assert r.kind in ("real code", "simulated", "recorded")
    names = {r.name for r in results}
    assert {"controller", "bb84", "link_protection", "secure_aggregation", "partition_audit", "recorded_results"} <= names


def test_the_controller_check_covers_the_boundaries_and_the_guard_on_the_lockdown_threshold():
    r = _by_name(sc.run_checks(only=["controller"]))["controller"]
    assert r.status == "pass"
    assert "0.11" in r.detail and "refused" in r.detail.lower()


def test_the_checkpoint_check_is_skipped_with_a_reason_when_there_is_no_checkpoint(tmp_path):
    r = _by_name(sc.run_checks(only=["model"], checkpoint_dir=tmp_path / "missing"))["model"]
    assert r.status == "skip" and "no checkpoint" in r.detail.lower()


def test_the_checkpoint_check_loads_a_model_and_scores_a_picture_when_one_exists(tmp_path):
    r = _by_name(sc.run_checks(only=["model"], checkpoint_dir=_checkpoint_dir(tmp_path)))["model"]
    assert r.status == "pass", r.detail
    assert "14" in r.detail and "round 7" in r.detail


def test_a_check_that_raises_is_a_failure_and_the_others_still_run():
    def boom():
        raise RuntimeError("deliberate failure for the test")

    extra = [sc.Check(name="boom", title="Deliberate failure", what="raises", kind="real code", run=boom)]
    results = sc.run_checks(only=["controller"], extra=extra)
    by = _by_name(results)
    assert by["controller"].status == "pass"
    assert by["boom"].status == "fail" and "deliberate failure for the test" in by["boom"].detail


def test_a_check_that_returns_the_wrong_answer_is_a_failure_not_a_pass():
    def wrong():
        assert 1 + 1 == 3, "arithmetic broke"
        return "unreachable"

    extra = [sc.Check(name="wrong", title="Wrong answer", what="asserts", kind="real code", run=wrong)]
    r = _by_name(sc.run_checks(only=[], extra=extra))["wrong"]
    assert r.status == "fail" and "arithmetic broke" in r.detail


def test_the_summary_counts_and_only_calls_the_run_healthy_when_nothing_failed(tmp_path):
    ok = sc.run_checks(only=["controller", "bb84"])
    summary = sc.summarise(ok)
    assert summary == {"total": 2, "passed": 2, "failed": 0, "skipped": 0, "healthy": True, "seconds": summary["seconds"]}
    bad = ok + sc.run_checks(only=[], extra=[sc.Check(name="x", title="x", what="x", kind="real code", run=lambda: 1 / 0)])
    assert sc.summarise(bad)["failed"] == 1 and sc.summarise(bad)["healthy"] is False


def test_details_never_contain_key_material_or_long_hex_strings(tmp_path):
    for r in sc.run_checks(checkpoint_dir=tmp_path / "none"):
        assert not re.search(r"[0-9a-f]{32,}", r.detail), (r.name, r.detail)


def test_the_bb84_check_uses_a_fixed_seed_so_two_runs_agree():
    a = _by_name(sc.run_checks(only=["bb84"]))["bb84"].detail
    b = _by_name(sc.run_checks(only=["bb84"]))["bb84"].detail
    assert a == b
