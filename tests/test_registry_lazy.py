"""RF-04: importing the registry / resolving SDXL must not eager-import the Cosmos pipeline.

Run in a subprocess so other tests' sys.modules can't mask the assertion.
"""

import subprocess
import sys
import textwrap


def test_registry_does_not_eager_import_cosmos():
    script = textwrap.dedent(
        """
        import sys
        import rengu_flow.registry.models as m
        assert "rengu_flow.model.cosmos_predict2" not in sys.modules, "registry import pulled in cosmos"

        from rengu_flow.registry.model_capabilities import get_canonical_model_types
        assert set(get_canonical_model_types()) >= {"sdxl", "cosmos_predict2"}, get_canonical_model_types()

        m._ensure_model_imported("sdxl")
        assert "rengu_flow.model.sdxl" in sys.modules
        assert "rengu_flow.model.cosmos_predict2" not in sys.modules, "sdxl resolution pulled in cosmos"
        print("OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout!r}\nstderr={r.stderr!r}"
    assert "OK" in r.stdout
