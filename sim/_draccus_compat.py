"""Python 3.14 + draccus argparse compatibility shim.

CPython 3.14 made argparse strict about ``type=`` being callable, but draccus
(≤ 0.11.x) still passes the raw ``X | None`` ``UnionType`` for Optional fields
(e.g. ``PI0Config.device``), which crashes parsing with
``TypeError: str | None is not callable``. We collapse union types to their
first non-None member here before argparse sees them. Patching argparse's
``_ActionsContainer.add_argument`` (the common base) catches both direct parser
calls and argument-group calls — draccus uses both.

Import this module for its side effect *before* any draccus-driven CLI parse.
Every ``sim`` policy module (``safe_pi0_policy``, ``safe_diffusion_policy``)
imports it, so it is active anywhere a policy is loaded — training and
``PolicyRollout`` alike. Applying it twice is a no-op.
"""

from __future__ import annotations

import sys

_APPLIED = False


def apply() -> None:
    """Idempotently patch argparse to tolerate draccus' ``X | None`` types."""
    global _APPLIED
    if _APPLIED or sys.version_info < (3, 14):
        return

    import argparse as _argparse
    import typing as _typing

    _orig_add_argument = _argparse._ActionsContainer.add_argument

    def _add_argument_union_safe(self, *args, **kwargs):
        t = kwargs.get("type")
        if t is not None and not callable(t):
            members = [a for a in _typing.get_args(t) if a is not type(None)]
            kwargs["type"] = members[0] if members else str
        return _orig_add_argument(self, *args, **kwargs)

    _argparse._ActionsContainer.add_argument = _add_argument_union_safe
    _APPLIED = True


apply()
