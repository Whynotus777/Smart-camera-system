"""Suite registry. Add a suite = add a module here with a `@register_suite` class (eval/README.md)."""

from eval.suites.base import SUITES, RunContext, Suite, SuiteResult, discover, register_suite

__all__ = ["SUITES", "RunContext", "Suite", "SuiteResult", "discover", "register_suite"]
