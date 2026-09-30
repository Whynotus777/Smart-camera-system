"""Dataset converters to the canonical format. `python -m eval.converters <id> [...]`.

Importing this package registers every converter in `CONVERTERS`.
"""

from eval.converters import meva, own, poselift, retails, smartspaces  # noqa: F401  (registration)
from eval.converters.base import APPROVAL, CONVERTERS, LICENSES, LicenseError, check_license

__all__ = ["APPROVAL", "CONVERTERS", "LICENSES", "LicenseError", "check_license"]
