"""Unified ``renga`` command-line interface."""

__all__ = ["main"]


def main(argv=None):
    from renga_flow.cli.main import main as _main

    return _main(argv)
