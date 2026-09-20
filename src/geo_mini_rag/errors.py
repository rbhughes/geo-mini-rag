"""Errors that are the user's problem, not a bug.

`main()` prints these as a one-line message and exits 1. Anything else keeps
its traceback, because a traceback is what a real bug deserves.
"""

from __future__ import annotations


class UserError(RuntimeError):
    """A problem the user can fix: no index, wrong config, missing key or library."""
