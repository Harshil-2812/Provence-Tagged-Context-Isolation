"""
Baseline 2: Simple Delimiter Defense
Tool output is wrapped in <UNTRUSTED_DATA> tags. 
This is a widely-discussed simple mitigation that provides some visual separation
but does NOT enforce trust boundaries through provenance metadata.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import Agent, LLMClient


def create_delimiter_defense_agent(llm_client: LLMClient) -> Agent:
    """Create an agent with delimiter-based defense (baseline 2)."""
    return Agent(llm_client=llm_client, defense_mode="delimiter_defense")


def run_delimiter_defense(user_task: str, tool_outputs: list, provider: str = "mock"):
    """Run agent with delimiter defense baseline."""
    llm = LLMClient(provider=provider)
    agent = create_delimiter_defense_agent(llm)
    return agent.run(user_task=user_task, tool_outputs=tool_outputs)
