"""
Provenance model for the context isolation defense system.

Each context element carries provenance metadata that tracks its origin and trust level.
Provenance cannot be modified by content of any context element (textual claims in tool
output cannot elevate trust level).
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
import time
import uuid


class ProvenanceLevel(str, Enum):
    """Trust levels for context elements, ordered from most to least trusted."""
    SYSTEM = "SYSTEM"                    # System prompt / developer instructions
    USER = "USER"                        # Direct user input
    TRUSTED_DATA = "TRUSTED_DATA"        # Verified/trusted data sources
    TOOL_UNTRUSTED = "TOOL_UNTRUSTED"    # Output from external tools (untrusted by default)
    UNKNOWN = "UNKNOWN"                  # Unknown provenance (treated as untrusted)


# Trust mapping: which levels are considered trusted for instruction authority
TRUSTED_LEVELS = {ProvenanceLevel.SYSTEM, ProvenanceLevel.USER, ProvenanceLevel.TRUSTED_DATA}
UNTRUSTED_LEVELS = {ProvenanceLevel.TOOL_UNTRUSTED, ProvenanceLevel.UNKNOWN}


@dataclass
class ProvenanceTag:
    """
    Provenance metadata attached to a context element.
    
    Critical invariant: The provenance of an element is determined by its origin,
    NOT by any claim within the element's content. A tool output claiming to be
    a system message remains TOOL_UNTRUSTED.
    """
    level: ProvenanceLevel
    source: str                          # e.g., "web_search", "user_input", "system_config"
    timestamp: float = field(default_factory=time.time)
    element_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: dict = field(default_factory=dict)
    
    @property
    def is_trusted(self) -> bool:
        """Returns True if this element has instruction authority."""
        return self.level in TRUSTED_LEVELS
    
    @property
    def has_instruction_authority(self) -> bool:
        """
        Returns True if this element's content can legitimately instruct the agent.
        Only SYSTEM and USER level have instruction authority.
        """
        return self.level in {ProvenanceLevel.SYSTEM, ProvenanceLevel.USER}
    
    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "source": self.source,
            "timestamp": self.timestamp,
            "element_id": self.element_id,
            "is_trusted": self.is_trusted,
            "has_instruction_authority": self.has_instruction_authority,
            "metadata": self.metadata
        }


@dataclass
class ContextElement:
    """
    A single element in the agent's context, tagged with provenance.
    
    The content can be any text, but its authority is determined solely
    by the provenance tag, not by claims within the content itself.
    """
    content: str
    provenance: ProvenanceTag
    role: str = "user"  # for LLM API: "system", "user", "assistant"
    
    @property
    def is_trusted(self) -> bool:
        return self.provenance.is_trusted
    
    @property
    def has_instruction_authority(self) -> bool:
        return self.provenance.has_instruction_authority
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "provenance": self.provenance.to_dict(),
            "role": self.role,
            "trusted": self.is_trusted,
            "source": self.provenance.source
        }
    
    def __repr__(self) -> str:
        return (f"ContextElement(provenance={self.provenance.level.value}, "
                f"source={self.provenance.source!r}, "
                f"content={self.content[:50]!r}...)")


def make_system_element(content: str, source: str = "system_config") -> ContextElement:
    """Create a trusted SYSTEM-level context element."""
    return ContextElement(
        content=content,
        provenance=ProvenanceTag(level=ProvenanceLevel.SYSTEM, source=source),
        role="system"
    )


def make_user_element(content: str, source: str = "user_input") -> ContextElement:
    """Create a USER-level context element."""
    return ContextElement(
        content=content,
        provenance=ProvenanceTag(level=ProvenanceLevel.USER, source=source),
        role="user"
    )


def make_tool_element(content: str, tool_name: str) -> ContextElement:
    """
    Create an UNTRUSTED tool output context element.
    
    Critical: No matter what the content claims, tool outputs are ALWAYS
    tagged as TOOL_UNTRUSTED. Textual claims cannot elevate provenance.
    """
    return ContextElement(
        content=content,
        provenance=ProvenanceTag(
            level=ProvenanceLevel.TOOL_UNTRUSTED,
            source=tool_name,
            metadata={"tool_name": tool_name}
        ),
        role="user"
    )


def make_trusted_data_element(content: str, source: str) -> ContextElement:
    """Create a TRUSTED_DATA context element (verified data sources)."""
    return ContextElement(
        content=content,
        provenance=ProvenanceTag(level=ProvenanceLevel.TRUSTED_DATA, source=source),
        role="user"
    )
