"""TorchEngine (engine='accelerate') unit check: grad accumulation, step, checkpoint roundtrip.

No DeepSpeed, no GPU required (runs on CPU). Verifies the single-GPU engine satisfies the
slice of the training-loop surface it owns. The full SDXL-LoRA behaviour is covered by the
manual GPU smoke; this guards the engine's own logic.
"""

import torch
import torch.nn as nn
from unittest.mock import MagicMock

from rengu_flow.engine import SequentialPipe, TorchEngine, resolve_backend, select_backend


class _TupleLinear(nn.Module):
    """Pipeline-convention layer: takes (x,), returns (Wx,)."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        (feat,) = x
        return (self.lin(feat),)


def _loss_fn(output, label):
    (pred,) = output
    (target,) = label
    return torch.nn.functional.mse_loss(pred, target)


def _make_engine(gas=2):
    pipe = SequentialPipe([_TupleLinear()], _loss_fn)
    ds_config = {"gradient_accumulation_steps": gas, "gradient_clipping": 1.0}
    return TorchEngine(pipe, lambda p: torch.optim.SGD(p, lr=0.1), list(pipe.parameters()), ds_config)


def _micro_batches(engine, gas, target):
    feat = torch.randn(2, 4, device=engine.device)
    target = target.to(engine.device)
    return iter([((feat,), (target,)) for _ in range(gas)])


def test_resolve_backend_default_is_accelerate(monkeypatch):
    # accelerate is the default on every platform (faster single-GPU path); deepspeed is opt-in.
    # Clear RENGU_ENGINE so this is not polluted by another test in the same xdist worker (the
    # --engine flag sets os.environ['RENGU_ENGINE'] directly).
    monkeypatch.delenv("RENGU_ENGINE", raising=False)
    assert resolve_backend({}) == "accelerate"
    assert resolve_backend({"engine": "deepspeed"}) == "deepspeed"
    monkeypatch.setenv("RENGU_ENGINE", "deepspeed")
    assert resolve_backend({}) == "deepspeed"  # env overrides the default


def test_build_pipe_accelerate_is_sequential_no_deepspeed():
    pipe = select_backend({"engine": "accelerate"}).build_pipe(
        layers=[_TupleLinear()], num_stages=1, partition_method="parameters",
        manual_partition_split=None, loss_fn=_loss_fn, extra_kw={},
    )
    assert isinstance(pipe, SequentialPipe)


def test_train_batch_steps_and_loss_decreases():
    gas = 2
    engine = _make_engine(gas)
    target = torch.zeros(2, 4)
    first = engine.train_batch(_micro_batches(engine, gas, target)).item()
    for _ in range(20):
        last = engine.train_batch(_micro_batches(engine, gas, target)).item()
    assert last < first, f"loss did not decrease ({first} -> {last})"
    assert engine.get_global_grad_norm() is not None  # grads flowed


def test_activation_checkpoint_interval_trains():
    from functools import partial
    import torch.utils.checkpoint as ckpt

    pipe = select_backend({"engine": "accelerate"}).build_pipe(
        layers=[_TupleLinear(), _TupleLinear()], num_stages=1,
        partition_method="parameters", manual_partition_split=None, loss_fn=_loss_fn,
        extra_kw={
            "activation_checkpoint_interval": 1,
            "checkpointable_layers": ["_TupleLinear"],
            "activation_checkpoint_func": partial(ckpt.checkpoint, use_reentrant=False),
        },
    )
    assert pipe._ac_interval == 1 and pipe._ac_func is not None
    ds_config = {"gradient_accumulation_steps": 1, "gradient_clipping": 1.0}
    engine = TorchEngine(pipe, lambda p: torch.optim.SGD(p, lr=0.1), list(pipe.parameters()), ds_config)
    target = torch.zeros(2, 4, device=engine.device)
    first = engine.train_batch(_micro_batches(engine, 1, target)).item()
    for _ in range(25):
        last = engine.train_batch(_micro_batches(engine, 1, target)).item()
    assert last < first, f"checkpointed training did not reduce loss ({first} -> {last})"
    assert engine.get_global_grad_norm() and engine.get_global_grad_norm() > 0


def test_eval_batch_no_grad_finite():
    engine = _make_engine()
    loss = engine.eval_batch(_micro_batches(engine, 2, torch.zeros(2, 4)))
    assert torch.isfinite(loss)


def test_module_mode_tree_is_only_walked_on_real_transitions():
    engine = _make_engine(gas=1)
    original_train = engine.module.train
    engine.module.train = MagicMock(wraps=original_train)
    target = torch.zeros(2, 4)

    engine.train_batch(_micro_batches(engine, 1, target))
    engine.train_batch(_micro_batches(engine, 1, target))
    engine.module.train.assert_not_called()

    engine.eval_batch(_micro_batches(engine, 1, target))
    engine.eval_batch(_micro_batches(engine, 1, target))
    engine.module.train.assert_called_once_with(False)

    engine.train_batch(_micro_batches(engine, 1, target))
    assert engine.module.train.call_args_list[-1].args == ()
    assert engine.module.train.call_count == 2


def test_single_device_engine_streams_micro_batches():
    assert _make_engine().preload_micro_batches is False


def test_checkpoint_roundtrip(tmp_path):
    engine = _make_engine()
    engine.train_batch(_micro_batches(engine, 2, torch.zeros(2, 4)))
    before = [p.detach().clone() for p in engine.module.parameters()]
    engine.save_checkpoint(str(tmp_path), client_state={"step": 5, "examples": 10, "custom_loader": {}})

    # Perturb, then load must restore.
    with torch.no_grad():
        for p in engine.module.parameters():
            p.add_(1.0)
    load_path, client_state = engine.load_checkpoint(str(tmp_path))
    assert load_path is not None
    assert client_state["step"] == 5 and client_state["examples"] == 10
    for p, b in zip(engine.module.parameters(), before):
        assert torch.allclose(p, b, atol=1e-6)


if __name__ == "__main__":
    test_resolve_backend_default_is_accelerate()
    test_build_pipe_accelerate_is_sequential_no_deepspeed()
    test_activation_checkpoint_interval_trains()
    test_train_batch_steps_and_loss_decreases()
    test_eval_batch_no_grad_finite()
    import tempfile, pathlib  # noqa: E401
    test_checkpoint_roundtrip(pathlib.Path(tempfile.mkdtemp()))
    print("ALL ENGINE CHECKS PASSED")
