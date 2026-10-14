"""
Attack case loader and helper functions.
"""

import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class AttackCase:
    attack_id: str
    category: str
    task: str
    tool: str
    tool_query: str
    benign_context: str
    malicious_payload: str
    expected_safe_behavior: str
    expected_attack_behavior: str
    attack_category: str
    severity: str
    
    def to_dict(self) -> dict:
        return self.__dict__.copy()


def load_attack_cases(filepath: str = None) -> List[AttackCase]:
    """Load attack cases from JSON file."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "attack_cases.json")
    
    with open(filepath, "r") as f:
        data = json.load(f)
    
    return [AttackCase(**case) for case in data]


def get_attack_by_id(attack_id: str, cases: List[AttackCase] = None) -> Optional[AttackCase]:
    if cases is None:
        cases = load_attack_cases()
    for case in cases:
        if case.attack_id == attack_id:
            return case
    return None


def get_attacks_by_category(category: str, cases: List[AttackCase] = None) -> List[AttackCase]:
    if cases is None:
        cases = load_attack_cases()
    return [c for c in cases if c.attack_category == category]
