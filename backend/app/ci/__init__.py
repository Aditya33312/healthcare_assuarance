"""CI release-gate clients."""

from backend.app.ci.github import GateConfig, GateOutcome, run_release_gate

__all__ = ["GateConfig", "GateOutcome", "run_release_gate"]
