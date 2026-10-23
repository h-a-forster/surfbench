"""Surface models. Each model fits a filtered snapshot and returns a fitted surface that
evaluates total variance and its first two k-derivatives on every listed expiry.

Specification: ``docs/DESIGN.md`` section 4.2 and 4.3.
"""

from __future__ import annotations
