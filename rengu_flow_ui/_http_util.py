"""Shared HTTP-error translation helpers for FastAPI route handlers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from fastapi import HTTPException

from rengu_flow_ui.gpu_lease import LeaseBusyError


@contextmanager
def http_errors(not_found_msg: str = "Not found") -> Generator[None, None, None]:
    """Translate common errors to HTTP status codes.

    KeyError  -> 404 (not_found_msg)
    ValueError -> 400 (str(e))
    gpu_lease.LeaseBusyError -> 409 (str(e)): an explicit start found the GPU held
    """
    try:
        yield
    except KeyError:
        raise HTTPException(404, not_found_msg)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except LeaseBusyError as e:
        raise HTTPException(409, str(e))
