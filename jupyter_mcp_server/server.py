###############################################
# Copyright (c) 2023-2025 Datalayer, Inc.      #
#                                             #
# BSD 3‑Clause License                         #
###############################################
"""
Extended MCP server for Jupyter that allows dynamic
kernel discovery and selection from the client side.

New MCP tools
-------------
* list_kernels()             → Return a list of available kernel names.
* switch_kernel(kernel_name) → Shut down the current kernel (if any) and
                               start a new one with the specified name.

The existing tools (`add_markdown_cell`, `add_execute_code_cell`) now share
**one** globally‑managed `KernelClient` instance that can be replaced at
runtime when the user calls `switch_kernel()`.

Notes
-----
* The implementation queries ``/api/kernelspecs`` to discover kernels; this
  is more reliable than shelling out to `jupyter kernelspec list` because it
  respects the running server’s environment.
* A minimal amount of error handling is included so that attempts to switch
  to an unknown kernel produce a helpful message instead of crashing.
* All network I/O that might block is kept synchronous; these functions are
  intended for human‑scale interaction and the overhead is negligible.
* Requires the additional PyPI dependency ``requests``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import List

import requests
from jupyter_kernel_client import KernelClient
from jupyter_nbmodel_client import (
    NbModelClient,
    get_jupyter_notebook_websocket_url,
)
from mcp.server.fastmcp import FastMCP

# ──────────────── Configuration ─────────────────

NOTEBOOK_PATH = os.getenv("NOTEBOOK_PATH", "notebook.ipynb")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8888")
TOKEN = os.getenv("TOKEN", "MY_TOKEN")
DEFAULT_KERNEL_NAME = os.getenv("KERNEL_NAME", None)  # "python3", "julia‑1.10", etc.

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ──────────────── MCP initialisation ────────────

mcp = FastMCP("jupyter")

# We keep a module‑level reference so every tool sees the same client.
_kernel_client: KernelClient | None = None


def _start_kernel(kernel_name: str | None = None) -> KernelClient:  # noqa: D401
    """(Re)start the global kernel client.

    If a kernel is already running we attempt to shut it down first.
    The new client is stored in the module level variable ``_kernel_client``.
    """

    global _kernel_client

    if _kernel_client is not None:
        try:
            logger.info("Shutting down existing kernel …")
            _kernel_client.shutdown()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Ignoring error while shutting down kernel: %s", exc)

    logger.info("Starting kernel %s …", kernel_name or "(server default)")
    _kernel_client = KernelClient(
        server_url=SERVER_URL,
        token=TOKEN,
        kernel_name=kernel_name,
    )
    _kernel_client.start()
    return _kernel_client


# Start immediately so the first request can execute code straight away.
_start_kernel(DEFAULT_KERNEL_NAME)

# ──────────────── Helper utilities ──────────────


def _extract_output(output: dict) -> str:
    """Extract readable text from a Jupyter cell output dict."""

    output_type = output.get("output_type")

    if output_type == "stream":
        return output.get("text", "")

    if output_type in {"display_data", "execute_result"}:
        data = output.get("data", {})
        if "text/plain" in data:
            return str(data["text/plain"])
        if "text/html" in data:
            return "[HTML Output]"
        if "image/png" in data:
            return "[Image Output (PNG)]"
        return f"[{output_type} Data: keys={list(data.keys())}]"

    if output_type == "error":
        # Jupyter already gives us a list of ANSI‑escaped traceback lines.
        return "\n".join(output.get("traceback", []))

    return f"[Unknown output type: {output_type}]"


# ──────────────── MCP tools ─────────────────────


@mcp.tool()
async def list_kernels() -> List[str]:
    """Return a list of *kernel_name*s available on the server."""

    try:
        url = f"{SERVER_URL}/api/kernelspecs?token={TOKEN}"
        logger.debug("GET %s", url)
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        specs = response.json().get("kernelspecs", {})
        return sorted(specs.keys())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Failed to fetch kernel specs: %s", exc)
        return []


@mcp.tool()
async def switch_kernel(kernel_name: str) -> str:
    """Switch to *kernel_name*.

    Shuts down the current kernel (if any) and starts a new one. Returns a
    confirmation message or raises a ValueError if the kernel does not exist.
    """

    available = await list_kernels()
    if kernel_name not in available:
        raise ValueError(
            f"Kernel '{kernel_name}' not found. Available kernels: {', '.join(available) if available else '(none found)'}"
        )

    _start_kernel(kernel_name)
    return f"Switched to kernel '{kernel_name}'."


@mcp.tool()
async def add_markdown_cell(cell_content: str) -> str:  # noqa: D401
    """Add a markdown cell to the notebook and return a success message."""

    notebook = NbModelClient(
        get_jupyter_notebook_websocket_url(
            server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH
        )
    )
    await notebook.start()

    notebook.add_markdown_cell(cell_content)

    await notebook.stop()
    return "Jupyter markdown cell added."


@mcp.tool()
async def add_execute_code_cell(cell_content: str) -> List[str]:  # noqa: D401
    """Add **and execute** a code cell, returning its textual outputs."""

    if _kernel_client is None:
        raise RuntimeError("Kernel client is not initialised. Call switch_kernel() first?")

    notebook = NbModelClient(
        get_jupyter_notebook_websocket_url(
            server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH
        )
    )
    await notebook.start()

    cell_index = notebook.add_code_cell(cell_content)
    notebook.execute_cell(cell_index, _kernel_client)

    # Extract outputs as strings so we can return them over MCP.
    ydoc = notebook._doc  # pylint: disable=protected-access
    outputs = ydoc._ycells[cell_index]["outputs"]  # type: ignore[attr-defined]
    str_outputs = [_extract_output(output) for output in outputs]

    await notebook.stop()
    return str_outputs


# ──────────────── CLI entry point ───────────────

if __name__ == "__main__":
    # Allow running as ``python jupyter_mcp_server_with_kernel.py`` when debugging.
    from mcp.server.fastmcp import main as fastmcp_main  # pylint: disable=import‑error

    sys.exit(fastmcp_main())

