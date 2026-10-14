"""
Baseline 1: No Defense
Tool output is directly inserted into the agent context with no special handling.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import Agent, LLMClient


def create_no_defense_agent(llm_client: LLMClient) -> Agent:
    """Create an agent with no defense (baseline 1)."""
    return Agent(llm_client=llm_client, defense_mode="no_defense")


def run_no_defense(user_task: str, tool_outputs: list, provider: str = "mock"):
    """Run agent with no defense baseline."""
    llm = LLMClient(provider=provider)
    agent = create_no_defense_agent(llm)
    return agent.run(user_task=user_task, tool_outputs=tool_outputs)
