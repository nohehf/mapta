"""
Playwright MCP integration using OpenAI Agents SDK built-in support.

This uses the official MCPServerStdio class from the OpenAI Agents SDK
for proper MCP integration with automatic tool discovery and caching.
"""

from agents.mcp import MCPServerStdio


def create_playwright_mcp_server():
    """
    Create a Playwright MCP server using the OpenAI Agents SDK.

    This uses the official MCPServerStdio class which provides:
    - Automatic tool discovery via list_tools()
    - Built-in caching support
    - Proper async handling
    - Tool filtering capabilities
    """
    return MCPServerStdio(
        params={
            "command": "docker",
            "args": ["run", "-i", "--rm", "--init", "mcr.microsoft.com/playwright/mcp"]
        },
        # Enable caching since tool list won't change
        cache_tools_list=True
    )
