import torch

# Simple wrapper for use with gradient release. Grad hooks do the optimizer steps, so this no-ops
# the step() and zero_grad() methods. It also handles state_dict.
class GradientReleaseOptimizerWrapper(torch.optim.Optimizer):
    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        ret = []
        for opt in self.optimizers:
            ret.extend(opt.param_groups)
        return ret

    def state_dict(self):
        return {i: opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict):
        for i, sd in state_dict.items():
            self.optimizers[i].load_state_dict(sd)

    def step(self):
        pass

    def zero_grad(self, set_to_none=True):
        pass

    def eval(self):  # noqa: A003 - mirrors lookahead optimizer API (not nn.Module.eval)
        """Restore the true iterate on inner optimizers that support eval/train.

        ``gradient_release`` builds one optimizer per trainable parameter; lookahead-style
        optimizers (MSAM/Nekaon, ScheduleFree, Lookahead) keep live weights displaced in
        train mode. ``Saver._persist_at_true_iterate`` calls ``eval()`` before checkpoint /
        export reads. Forward to every inner optimizer that exposes ``eval``; no-op when none
        do (plain AdamW/SGD), which preserves the previous behaviour for non-lookahead types.
        Returns ``self`` for chaining (kaon optimizers return ``self``).
        """
        for opt in self.optimizers:
            fn = getattr(opt, "eval", None)
            if callable(fn):
                fn()
        return self

    def train(self):
        """Re-apply train-mode displacement on inner optimizers that support eval/train."""
        for opt in self.optimizers:
            fn = getattr(opt, "train", None)
            if callable(fn):
                fn()
        return self