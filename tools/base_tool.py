"""
Base tool interface. All simulated tools must implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ToolResult:
    """Result returned by any tool."""
    content: str
    tool_name: str
    success: bool
    is_malicious: bool = False  # For tracking in experiments
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "tool_name": self.tool_name,
            "success": self.success,
            "is_malicious": self.is_malicious,
            "metadata": self.metadata or {}
        }


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass
    
    @abstractmethod
    def execute(self, query: str, inject_attack: bool = False, attack_payload: str = None) -> ToolResult:
        """
        Execute the tool.
        
        Args:
            query: The query/request for the tool
            inject_attack: Whether to inject a malicious payload
            attack_payload: The specific payload to inject (if inject_attack=True)
            
        Returns:
            ToolResult with content and metadata
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
