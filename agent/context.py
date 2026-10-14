"""
Context management for the agent — builds and maintains ordered context elements.
"""

from typing import List, Optional, Dict
from .provenance import (
    ContextElement, ProvenanceLevel, ProvenanceTag,
    make_system_element, make_user_element, make_tool_element, make_trusted_data_element
)


class AgentContext:
    """
    Manages the ordered list of context elements for an agent interaction.
    Provides convenient methods for adding elements with correct provenance.
    """
    
    def __init__(self, system_instructions: str = ""):
        self._elements: List[ContextElement] = []
        if system_instructions:
            self.add_system(system_instructions)
    
    def add_system(self, content: str, source: str = "system_config") -> None:
        self._elements.append(make_system_element(content, source))
    
    def add_user(self, content: str, source: str = "user_input") -> None:
        self._elements.append(make_user_element(content, source))
    
    def add_tool_output(self, content: str, tool_name: str) -> None:
        """Add tool output — always tagged TOOL_UNTRUSTED regardless of content."""
        self._elements.append(make_tool_element(content, tool_name))
    
    def add_trusted_data(self, content: str, source: str) -> None:
        self._elements.append(make_trusted_data_element(content, source))
    
    @property
    def elements(self) -> List[ContextElement]:
        return list(self._elements)
    
    @property
    def trusted_elements(self) -> List[ContextElement]:
        return [e for e in self._elements if e.is_trusted]
    
    @property
    def untrusted_elements(self) -> List[ContextElement]:
        return [e for e in self._elements 
                if e.provenance.level in {ProvenanceLevel.TOOL_UNTRUSTED, ProvenanceLevel.UNKNOWN}]
    
    def get_user_task(self) -> str:
        """Extract the primary user task from context."""
        user_elems = [e for e in self._elements if e.provenance.level == ProvenanceLevel.USER]
        return user_elems[0].content if user_elems else ""
    
    def get_system_instructions(self) -> str:
        """Extract system instructions from context."""
        sys_elems = [e for e in self._elements if e.provenance.level == ProvenanceLevel.SYSTEM]
        return "\n\n".join(e.content for e in sys_elems)
    
    def clear(self) -> None:
        self._elements.clear()
    
    def summary(self) -> Dict:
        counts = {}
        for e in self._elements:
            lvl = e.provenance.level.value
            counts[lvl] = counts.get(lvl, 0) + 1
        return {"element_count": len(self._elements), "by_provenance": counts}
    
    def __len__(self) -> int:
        return len(self._elements)
    
    def __repr__(self) -> str:
        return f"AgentContext({self.summary()})"
