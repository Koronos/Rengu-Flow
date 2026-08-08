"""End-to-end: a resume checkpoint reproduces training bit-for-bit with a real lookahead
optimizer (Nekaon), exercising the actual TorchEngine save_checkpoint/load_checkpoint plus the
eval-mode save the Saver applies (_persist_at_true_iterate).

Lookahead/MSAM optimizers keep the live weights displaced between steps; writing the checkpoint
in eval mode stores the TRUE iterate so resume lands on-point. This run-through proves a step
taken after resume equals the step the uninterrupted run would have taken.

Skipped when the external kaon (K-Optimizers) package is not installed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
kaon = pytest.importorskip("kaon")

from rengu_flow.engine.single_device import SequentialPipe, TorchEngine


def _build_engine():
    layers = [torch.nn.Linear(8, 16), torch.nn.Linear(16, 4)]
    module = SequentialPipe(layers, loss_fn=lambda out, label: out)
    params = [p for layer in layers for p in layer.parameters()]
    return TorchEngine(
        module,
        lambda ps: kaon.Nekaon(ps, lr=1e-2),
        params,
        {"gradient_accumulation_steps": 1, "gradient_clipping": 0.0},
    )


def _manual_step(engine, seed):
    """One optimizer step with deterministic synthetic gradients (no data pipeline needed)."""
    g = torch.Generator().manual_seed(seed)
    engine.optimizer.zero_grad()
    for p in engine._trainable:
        p.grad = torch.randn(p.shape, generator=g).to(p.device)
    engine.optimizer.step()


def _weights(engine):
    return {n: p.detach().clone() for n, p in engine.module.named_parameters()}


def test_nekaon_resume_is_bit_exact(tmp_path):
    torch.manual_seed(0)
    eng = _build_engine()
    for k in range(6):  # populate optimizer state (4bit momentum + msam displacement)
        _manual_step(eng, seed=100 + k)

    # Save the resume checkpoint the way the Saver does: bracket it in optimizer eval/train so
    # the stored weights are the optimizer's true iterate, not the displaced live weights.
    eng.optimizer.eval()
    eng.save_checkpoint(str(tmp_path), client_state={"step": 6})
    eng.optimizer.train()

    # Uninterrupted continuation.
    _manual_step(eng, seed=999)
    direct = _weights(eng)

    # Fresh engine (different random init) that resumes from the checkpoint, then the same step.
    resumed_eng = _build_engine()
    load_path, client = resumed_eng.load_checkpoint(str(tmp_path))
    assert load_path is not None and client["step"] == 6
    _manual_step(resumed_eng, seed=999)
    resumed = _weights(resumed_eng)

    for name in direct:
        assert torch.equal(direct[name], resumed[name]), (
            f"{name}: resume diverged "
            f"(max|d|={(direct[name] - resumed[name]).abs().max().item():.3e})"
        )


def test_nekaon_resume_train_mode_save_is_rejected(tmp_path):
    """Guard the reason for the eval-mode save: a checkpoint written in train mode (displaced
    live weights) used to resume off-point; since kaon 0.7.9 it is rejected outright on load.
    Either way this proves the eval bracket above is load-bearing."""
    torch.manual_seed(0)
    eng = _build_engine()
    for k in range(6):
        _manual_step(eng, seed=100 + k)

    eng.save_checkpoint(str(tmp_path), client_state={"step": 6})  # NO eval bracket

    resumed_eng = _build_engine()
    with pytest.raises(ValueError, match="train mode"):
        resumed_eng.load_checkpoint(str(tmp_path))
