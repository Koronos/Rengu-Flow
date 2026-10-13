"""Web UI help for nanosaur2's preview guidance mode.

The guidance-mode select (options, default, shown only on nanosaur2), the sampler defaults and the
DiT preview knobs are checked for every model by tests/contract/test_spec_contract.py.
"""

import pytest

pytestmark = pytest.mark.no_ui_db


def test_field_help_documents_every_mode():
    from rengu_flow_ui.config_field_help import FIELD_HELP

    text = FIELD_HELP["preview.guidance_mode"]["detail"]
    assert all(mode in text for mode in ("cfg", "path_drop", "alternate"))
