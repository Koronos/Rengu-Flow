"""The contract suite's own helpers: a missing extra skips, a missing rengu_flow module fails."""

from __future__ import annotations

import pytest

from contract_helpers import import_or_skip

pytestmark = pytest.mark.no_ui_db


def _module(tmp_path, monkeypatch, name: str, body: str) -> str:
    (tmp_path / f"{name}.py").write_text(body, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


def test_import_or_skip_skips_only_for_a_missing_external_dependency(tmp_path, monkeypatch):
    name = _module(tmp_path, monkeypatch, "needs_an_extra", "import some_extra_that_is_not_installed_xyz\n")
    with pytest.raises(pytest.skip.Exception, match="some_extra_that_is_not_installed_xyz"):
        import_or_skip(name)


@pytest.mark.parametrize("missing", ["rengu_flow.model.no_such_module", "rengu_flow.model.no_such_pkg.inner"])
def test_import_or_skip_reraises_a_missing_rengu_flow_module(tmp_path, monkeypatch, missing):
    """A broken internal import (a renamed / deleted rengu_flow module) must fail the suite, not
    skip the whole contract for that model."""
    name = _module(tmp_path, monkeypatch, "broken_internal", f"import {missing}\n")
    try:
        import_or_skip(name)
    except ModuleNotFoundError as exc:
        assert exc.name == missing.rsplit(".", 1)[0] or exc.name == missing
        return
    except pytest.skip.Exception as skip:  # the bug this guards against: a silent skip
        pytest.fail(f"a missing rengu_flow module was skipped: {skip}")
    pytest.fail("the broken import did not raise")


def test_import_or_skip_returns_the_module():
    assert import_or_skip("rengu_flow.model.testkit_common").ALL_CAPABILITIES
