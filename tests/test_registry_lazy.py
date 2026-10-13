"""RF-04: importing the registry / resolving SDXL must not eager-import the Cosmos pipeline, and
the model specs (``registry.model_capabilities``) must stay torch-free.

Run in a subprocess so other tests' sys.modules can't mask the assertion. The cosmos *package* is
imported for its torch-free ``spec`` module (lazy ``__init__``); the pipeline module must not be.
"""

import subprocess
import sys
import textwrap


def test_registry_does_not_eager_import_cosmos():
    script = textwrap.dedent(
        """
        import sys
        import rengu_flow.registry.models as m
        assert "rengu_flow.model.cosmos_predict2.pipeline" not in sys.modules, "registry import pulled in cosmos"

        from rengu_flow.registry.model_capabilities import get_canonical_model_types
        assert set(get_canonical_model_types()) >= {"sdxl", "cosmos_predict2"}, get_canonical_model_types()

        m._ensure_model_imported("sdxl")
        assert "rengu_flow.model.sdxl" in sys.modules
        assert "rengu_flow.model.cosmos_predict2.pipeline" not in sys.modules, "sdxl resolution pulled in cosmos"
        print("OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout!r}\nstderr={r.stderr!r}"
    assert "OK" in r.stdout


def test_model_specs_import_without_torch():
    """Specs + capability lookups (every type and alias) load no torch/diffusers/transformers."""
    script = textwrap.dedent(
        """
        import sys
        from rengu_flow.registry.model_capabilities import get_canonical_model_types, get_capability
        from rengu_flow.registry.model_capabilities import model_capability_registry

        types = get_canonical_model_types()
        assert set(types) >= {"sdxl", "cosmos_predict2", "krea2", "qwen_image21", "nanosaur2"}, types
        for type_id in [*types, *(a for c in model_capability_registry.values() for a in c.aliases)]:
            assert get_capability(type_id) is not None, type_id
        heavy = sorted(m for m in ("torch", "diffusers", "transformers") if m in sys.modules)
        assert not heavy, f"spec import pulled in {heavy}"
        print("OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout!r}\nstderr={r.stderr!r}"
    assert "OK" in r.stdout
