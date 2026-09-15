"""
Tools package for the AI Research Agent.
Exposes modular tools and a safe dispatcher.
"""
from typing import Any, Dict, List
from src.tools.calculator import calculate
from src.tools.datetime_tool import get_current_datetime
from src.tools.document_search import search_documents
from src.tools.web_search import search_web
from src.utils.validators import validate_tool_query, sanitize_math_expression

AVAILABLE_TOOLS = {
    "web_search": {
        "name": "web_search",
        "description": "Searches the web for up-to-date information, news, facts, and citations.",
        "parameters": {"query": "string", "max_results": "integer"}
    },
    "calculator": {
        "name": "calculator",
        "description": "Performs exact mathematical and scientific calculations safely.",
        "parameters": {"expression": "string"}
    },
    "document_search": {
        "name": "document_search",
        "description": "Searches curated local knowledge documents and technical documentation.",
        "parameters": {"query": "string", "top_k": "integer"}
    },
    "datetime": {
        "name": "datetime",
        "description": "Returns current UTC and local date, time, and temporal reference.",
        "parameters": {}
    }
}

def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely dispatches and executes a tool call with parameter validation.
    
    Args:
        tool_name: One of 'web_search', 'calculator', 'document_search', 'datetime'
        arguments: Key-value parameters for the tool
        
    Returns:
        Structured result dict containing 'status', 'output', and metadata.
    """
    name = tool_name.lower().strip()

    if name == "web_search":
        raw_query = arguments.get("query") or arguments.get("q") or ""
        query = validate_tool_query(str(raw_query))
        max_results = int(arguments.get("max_results", 4))
        results = search_web(query, max_results=max_results)
        return {
            "status": "success",
            "tool": "web_search",
            "query": query,
            "results": results,
            "summary": f"Found {len(results)} web sources for query: '{query}'"
        }

    elif name == "calculator":
        raw_expr = arguments.get("expression") or arguments.get("expr") or ""
        try:
            expr = sanitize_math_expression(str(raw_expr))
            res = calculate(expr)
            return {
                "status": res["status"],
                "tool": "calculator",
                "expression": expr,
                "result": res["result"],
                "error": res["error"],
                "summary": f"Result of '{expr}' = {res['result']}" if res["status"] == "success" else f"Calculation failed: {res['error']}"
            }
        except Exception as e:
            return {
                "status": "error",
                "tool": "calculator",
                "expression": str(raw_expr),
                "result": None,
                "error": str(e),
                "summary": f"Calculation error: {e}"
            }

    elif name == "document_search":
        raw_query = arguments.get("query") or arguments.get("q") or ""
        query = validate_tool_query(str(raw_query))
        top_k = int(arguments.get("top_k", 3))
        results = search_documents(query, top_k=top_k)
        return {
            "status": "success",
            "tool": "document_search",
            "query": query,
            "results": results,
            "summary": f"Found {len(results)} document passages matching: '{query}'"
        }

    elif name in ("datetime", "datetime_tool", "date_time"):
        info = get_current_datetime()
        return {
            "status": "success",
            "tool": "datetime",
            "datetime_info": info,
            "summary": info["summary"]
        }

    else:
        return {
            "status": "error",
            "tool": tool_name,
            "error": f"Unknown tool '{tool_name}'. Available: {list(AVAILABLE_TOOLS.keys())}",
            "summary": f"Failed: unknown tool '{tool_name}'"
        }

__all__ = [
    "AVAILABLE_TOOLS",
    "calculate",
    "get_current_datetime",
    "search_documents",
    "search_web",
    "execute_tool"
]
