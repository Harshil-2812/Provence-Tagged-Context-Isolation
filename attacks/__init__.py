"""Attacks package init."""
from .indirect_injection import AttackCase, load_attack_cases, get_attack_by_id, get_attacks_by_category
from .tool_output_attacks import build_malicious_tool_output, build_benign_tool_output, is_attack_successful

__all__ = [
    "AttackCase", "load_attack_cases", "get_attack_by_id", "get_attacks_by_category",
    "build_malicious_tool_output", "build_benign_tool_output", "is_attack_successful"
]
