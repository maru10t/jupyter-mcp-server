# Copyright (c) 2023-2024 Datalayer, Inc.
#
# BSD 3-Clause License
import logging
import os
from typing import Optional
from jupyter_kernel_client import KernelClient
from jupyter_nbmodel_client import (
    NbModelClient,
    get_jupyter_notebook_websocket_url,
)
import jupyter_client
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("jupyter")

# 環境変数の設定
NOTEBOOK_PATH = os.getenv("NOTEBOOK_PATH", "notebook.ipynb")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8888")
TOKEN = os.getenv("TOKEN", "MY_TOKEN")
DEFAULT_KERNEL_NAME = os.getenv("KERNEL_NAME", "python3")

logger = logging.getLogger(__name__)

# グローバルなカーネルクライアント
kernel: Optional[KernelClient] = None
current_kernel_name: str = DEFAULT_KERNEL_NAME

def initialize_kernel(kernel_name: str = DEFAULT_KERNEL_NAME) -> KernelClient:
    """Initialize kernel client with specified kernel name."""
    global kernel, current_kernel_name
    
    if kernel is not None:
        try:
            kernel.stop()
        except Exception as e:
            logger.warning(f"Error stopping previous kernel: {e}")
    
    kernel = KernelClient(server_url=SERVER_URL, token=TOKEN, kernel_name=kernel_name)
    kernel.start()
    current_kernel_name = kernel_name
    logger.info(f"Kernel initialized: {kernel_name}")
    return kernel

# 初期カーネルの起動
initialize_kernel(DEFAULT_KERNEL_NAME)

def extract_output(output: dict) -> str:
    """
    Extracts readable output from a Jupyter cell output dictionary.
    Args:
        output (dict): The output dictionary from a Jupyter cell.
    Returns:
        str: A string representation of the output.
    """
    output_type = output.get("output_type")
    if output_type == "stream":
        return output.get("text", "")
    elif output_type in ["display_data", "execute_result"]:
        data = output.get("data", {})
        if "text/plain" in data:
            return data["text/plain"]
        elif "text/html" in data:
            return "[HTML Output]"
        elif "image/png" in data:
            return "[Image Output (PNG)]"
        elif "image/jpeg" in data:
            return "[Image Output (JPEG)]"
        elif "image/svg+xml" in data:
            return "[SVG Output]"
        else:
            return f"[{output_type} Data: keys={list(data.keys())}]"
    elif output_type == "error":
        traceback = output.get("traceback", [])
        if isinstance(traceback, list):
            return "\n".join(traceback)
        return str(traceback)
    else:
        return f"[Unknown output type: {output_type}]"

@mcp.tool()
async def list_available_kernels() -> dict:
    """List all available Jupyter kernels.
    Returns:
        dict: Dictionary containing available kernels and current kernel info
    """
    try:
        # 利用可能なカーネルスペックを取得
        kernel_specs = jupyter_client.kernelspec.find_kernel_specs()
        kernel_manager = jupyter_client.kernelspec.KernelSpecManager()
        
        kernels_info = {}
        for name, path in kernel_specs.items():
            try:
                spec = kernel_manager.get_kernel_spec(name)
                kernels_info[name] = {
                    "display_name": spec.display_name,
                    "language": spec.language,
                    "path": path
                }
            except Exception as e:
                kernels_info[name] = {
                    "display_name": name,
                    "language": "unknown",
                    "path": path,
                    "error": str(e)
                }
        
        return {
            "available_kernels": kernels_info,
            "current_kernel": current_kernel_name,
            "total_kernels": len(kernels_info)
        }
    except Exception as e:
        logger.error(f"Error listing kernels: {e}")
        return {
            "error": f"Failed to list kernels: {str(e)}",
            "current_kernel": current_kernel_name
        }

@mcp.tool()
async def get_current_kernel() -> dict:
    """Get information about the currently active kernel.
    Returns:
        dict: Information about the current kernel
    """
    global kernel, current_kernel_name
    
    kernel_info = {
        "kernel_name": current_kernel_name,
        "kernel_active": kernel is not None
    }
    
    if kernel is not None:
        try:
            # カーネルの状態を確認
            kernel_info["kernel_ready"] = True
        except Exception as e:
            kernel_info["kernel_ready"] = False
            kernel_info["error"] = str(e)
    
    return kernel_info

@mcp.tool()
async def change_kernel(kernel_name: str) -> str:
    """Change the active kernel to the specified kernel.
    Args:
        kernel_name: Name of the kernel to switch to (e.g., 'python3', 'julia-1.10', 'ir')
    Returns:
        str: Success or error message
    """
    try:
        # 利用可能なカーネルを確認
        available_kernels = jupyter_client.kernelspec.find_kernel_specs()
        
        if kernel_name not in available_kernels:
            available_names = list(available_kernels.keys())
            return f"Kernel '{kernel_name}' not found. Available kernels: {', '.join(available_names)}"
        
        # 新しいカーネルを初期化
        initialize_kernel(kernel_name)
        
        return f"Successfully changed kernel to '{kernel_name}'"
        
    except Exception as e:
        logger.error(f"Error changing kernel to {kernel_name}: {e}")
        return f"Failed to change kernel to '{kernel_name}': {str(e)}"

@mcp.tool()
async def restart_current_kernel() -> str:
    """Restart the current kernel.
    Returns:
        str: Success or error message
    """
    global kernel, current_kernel_name
    
    try:
        if kernel is not None:
            kernel.stop()
        
        # 同じカーネルで再起動
        initialize_kernel(current_kernel_name)
        
        return f"Successfully restarted kernel '{current_kernel_name}'"
        
    except Exception as e:
        logger.error(f"Error restarting kernel: {e}")
        return f"Failed to restart kernel: {str(e)}"

@mcp.tool()
async def add_markdown_cell(cell_content: str) -> str:
    """Add a markdown cell in a Jupyter notebook.
    Args:
        cell_content: Markdown content
    Returns:
        str: Success message
    """
    try:
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        await notebook.start()
        notebook.add_markdown_cell(cell_content)
        await notebook.stop()
        return "Jupyter Markdown cell added successfully."
    except Exception as e:
        logger.error(f"Error adding markdown cell: {e}")
        return f"Failed to add markdown cell: {str(e)}"

@mcp.tool()
async def add_execute_code_cell(cell_content: str) -> dict:
    """Add and execute a code cell in a Jupyter notebook.
    Args:
        cell_content: Code content to execute
    Returns:
        dict: Execution results including outputs and kernel info
    """
    global kernel
    
    if kernel is None:
        return {
            "error": "No active kernel. Please select a kernel first.",
            "outputs": [],
            "kernel_name": current_kernel_name
        }
    
    try:
        notebook = NbModelClient(
            get_jupyter_notebook_websocket_url(server_url=SERVER_URL, token=TOKEN, path=NOTEBOOK_PATH)
        )
        await notebook.start()
        
        # コードセルを追加
        cell_index = notebook.add_code_cell(cell_content)
        
        # セルを実行
        notebook.execute_cell(cell_index, kernel)
        
        # 出力を取得
        ydoc = notebook._doc
        outputs = ydoc._ycells[cell_index]["outputs"]
        str_outputs = [extract_output(output) for output in outputs]
        
        await notebook.stop()
        
        return {
            "outputs": str_outputs,
            "kernel_name": current_kernel_name,
            "cell_index": cell_index,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error executing code cell: {e}")
        return {
            "error": f"Failed to execute code cell: {str(e)}",
            "outputs": [],
            "kernel_name": current_kernel_name,
            "success": False
        }

@mcp.tool()
async def execute_code_direct(code: str, kernel_name: str = None) -> dict:
    """Execute code directly with optional kernel specification.
    Args:
        code: Code to execute
        kernel_name: Optional kernel name to use (if different from current)
    Returns:
        dict: Execution results
    """
    global kernel, current_kernel_name
    
    # 指定されたカーネルが現在のものと異なる場合は切り替え
    if kernel_name and kernel_name != current_kernel_name:
        change_result = await change_kernel(kernel_name)
        if "Failed" in change_result:
            return {
                "error": change_result,
                "outputs": [],
                "kernel_name": current_kernel_name
            }
    
    # コードセルとして実行
    return await add_execute_code_cell(code)

if __name__ == "__main__":
    try:
        logger.info(f"Starting Jupyter MCP Server with kernel: {current_kernel_name}")
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if kernel is not None:
            kernel.stop()
    except Exception as e:
        logger.error(f"Server error: {e}")
        if kernel is not None:
            kernel.stop()
        raise
