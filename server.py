#!/usr/bin/env python3
"""
Main entry point for the Long-Term Memory MCP Server.

This is a refactored version that uses a modular architecture with separate concerns:
- config.py: Configuration constants
- models.py: Data models (dataclasses)
- memory_system.py: Core RobustMemorySystem class
- mcp_tools.py: MCP tool handler registration
"""

import asyncio
import atexit
import argparse
import sys

# Third-party imports
try:
    from fastmcp import FastMCP
except ImportError as e:
    print("Missing required package. Install with: pip install fastmcp")
    print(f"Error: {e}")
    sys.exit(1)

# Local imports
from memory_mcp import RobustMemorySystem, register_tools


def main():
    """Main entry point for the MCP server"""

    # Initialize the memory system
    memory_system = RobustMemorySystem()

    # Setup FastMCP
    mcp = FastMCP("RobustMemory")

    # Register all MCP tools
    register_tools(mcp, memory_system)

    # Cleanup on exit
    atexit.register(memory_system.close)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Long-Term Memory MCP Server")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for HTTP transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/mcp/",
        help="URL path for HTTP transport (default: /mcp/)",
    )

    args = parser.parse_args()

    try:
        if args.transport == "http":
            print(
                f"Starting Long-Term Memory MCP Server on http://{args.host}:{args.port}{args.path}"
            )
            asyncio.run(
                mcp.run(
                    transport="http", host=args.host, port=args.port, path=args.path
                )
            )
        else:
            # Default: stdio transport
            asyncio.run(mcp.run_stdio_async(show_banner=False))
    except KeyboardInterrupt:
        print("\nShutting down memory system...")
        memory_system.close()
    except Exception as e:
        print(f"Error running MCP server: {e}")
        memory_system.close()


if __name__ == "__main__":
    main()
