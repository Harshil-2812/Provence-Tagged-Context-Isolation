"""Tools package init."""
from .base_tool import BaseTool, ToolResult
from .mock_search import MockWebSearch
from .mock_database import MockDatabase
from .mock_web import MockWebRetriever

TOOL_REGISTRY = {
    "web_search": MockWebSearch(),
    "database": MockDatabase(),
    "web_retriever": MockWebRetriever(),
}

def get_tool(name: str) -> BaseTool:
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}. Available: {list(TOOL_REGISTRY.keys())}")
    return TOOL_REGISTRY[name]

__all__ = ["BaseTool", "ToolResult", "MockWebSearch", "MockDatabase", "MockWebRetriever", 
           "TOOL_REGISTRY", "get_tool"]
