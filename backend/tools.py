"""Tool definitions for Gemini function calling.

Each tool has:
- A schema (for the model)
- An execute function (runs locally)
"""

import ast
import math
import operator
from typing import Any

from google.genai import types

from backend.rag import query_collection

# ---------------------------------------------------------------------------
# 1) Calculator tool
# ---------------------------------------------------------------------------

CALC_DECLARATION = types.FunctionDeclaration(
    name="calc",
    description="Evaluate a mathematical expression and return the numeric result. "
    "Supports +, -, *, /, **, %, sqrt, abs, sin, cos, tan, log, pi, e.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": 'Math expression, e.g. "2 + 3 * 4" or "sqrt(16)"',
            }
        },
        "required": ["expression"],
    },
)

# Safe AST-based math evaluator (no exec/eval of arbitrary code)
_ALLOWED_NAMES: dict[str, Any] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "pi": math.pi,
    "e": math.e,
    "pow": pow,
    "round": round,
}

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node with only safe math ops."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _ALLOWED_NAMES:
        val = _ALLOWED_NAMES[node.id]
        if callable(val):
            raise ValueError(f"'{node.id}' is a function, use {node.id}(...)")
        return float(val)
    if isinstance(node, ast.BinOp):
        op_fn = _ALLOWED_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _ALLOWED_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op_fn(_safe_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_NAMES:
            fn = _ALLOWED_NAMES[node.func.id]
            if not callable(fn):
                raise ValueError(f"'{node.func.id}' is not callable")
            args = [_safe_eval_node(a) for a in node.args]
            return float(fn(*args))
        raise ValueError(f"Function not allowed: {ast.dump(node.func)}")
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def execute_calc(expression: str) -> dict:
    """Safely evaluate a math expression. Returns {"result": number}."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval_node(tree)
        return {"result": result}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# 2) Search docs tool (uses RAG retrieval)
# ---------------------------------------------------------------------------

SEARCH_DOCS_DECLARATION = types.FunctionDeclaration(
    name="search_docs",
    description="Search the uploaded document collection and return the most relevant chunks. "
    "Use this when you need factual information from the user's documents.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant document chunks.",
            }
        },
        "required": ["query"],
    },
)


def execute_search_docs(query: str, api_key: str, top_k: int = 5) -> dict:
    """Search the vector DB and return top chunks."""
    try:
        results = query_collection(query, api_key, top_k=top_k)
        return {"chunks": results}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# 3) Web search tool (via Google Search grounding in Gemini)
# ---------------------------------------------------------------------------

WEB_SEARCH_DECLARATION = types.FunctionDeclaration(
    name="web_search",
    description="Search the web for current information. Use when the user asks about "
    "recent events, live data, or anything not in the uploaded documents.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The web search query.",
            }
        },
        "required": ["query"],
    },
)


def execute_web_search(query: str, api_key: str) -> dict:
    """Use Gemini with Google Search grounding to answer a web query."""
    try:
        from google import genai
        from google.genai import types as gtypes

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=query,
            config=gtypes.GenerateContentConfig(
                tools=[gtypes.Tool(google_search=gtypes.GoogleSearch())],
            ),
        )
        text = ""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    text += part.text
        grounding = None
        if (
            response.candidates
            and response.candidates[0].grounding_metadata
            and response.candidates[0].grounding_metadata.grounding_chunks
        ):
            grounding = [
                {"title": c.web.title, "uri": c.web.uri}
                for c in response.candidates[0].grounding_metadata.grounding_chunks
                if c.web
            ]
        return {"result": text, "sources": grounding or []}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [CALC_DECLARATION, SEARCH_DOCS_DECLARATION, WEB_SEARCH_DECLARATION]

ALL_TOOLS = types.Tool(function_declarations=TOOL_SCHEMAS)


def execute_tool(name: str, args: dict, api_key: str = "") -> dict:
    """Dispatch a tool call to the right executor."""
    if name == "calc":
        return execute_calc(args.get("expression", ""))
    if name == "search_docs":
        return execute_search_docs(args.get("query", ""), api_key)
    if name == "web_search":
        return execute_web_search(args.get("query", ""), api_key)
    return {"error": f"Unknown tool: {name}"}
