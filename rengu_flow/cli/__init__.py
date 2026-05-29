"""Unified ``rengu`` command-line interface."""

__all__ = ["main"]


def main(argv=None):
    from rengu_flow.cli.main import main as _main

    return _main(argv)
