"""
Safe AST-based Calculator Tool.
Evaluates mathematical expressions without using arbitrary or unsafe eval().
Supports arithmetic operations, powers, and standard mathematical functions.
"""
import ast
import math
import operator
from typing import Any, Dict, Union

# Allowed binary and unary operators
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed safe math functions
SAFE_FUNCTIONS: Dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "pi": math.pi,
    "e": math.e,
}

class SafeEvalVisitor(ast.NodeVisitor):
    """AST visitor that only evaluates safe mathematical operations."""

    def visit(self, node: ast.AST) -> Union[int, float]:
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def visit_Expression(self, node: ast.Expression) -> Union[int, float]:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Union[int, float]:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    def visit_BinOp(self, node: ast.BinOp) -> Union[int, float]:
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        # Guard against excessively large exponentiation
        if op_type is ast.Pow and (abs(right) > 1000 or (abs(left) > 10 and right > 100)):
            raise ValueError("Exponentiation computation exceeds safe limits")
            
        return SAFE_OPERATORS[op_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Union[int, float]:
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = self.visit(node.operand)
        return SAFE_OPERATORS[op_type](operand)

    def visit_Call(self, node: ast.Call) -> Union[int, float]:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only standard named math functions are permitted")
        func_name = node.func.id
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' is not allowed in safe calculator")
        args = [self.visit(arg) for arg in node.args]
        func = SAFE_FUNCTIONS[func_name]
        return func(*args)

    def visit_Name(self, node: ast.Name) -> Union[int, float]:
        if node.id in SAFE_FUNCTIONS and isinstance(SAFE_FUNCTIONS[node.id], (int, float)):
            return SAFE_FUNCTIONS[node.id]
        raise ValueError(f"Unrecognized variable or identifier: {node.id}")

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def calculate(expression: str) -> Dict[str, Any]:
    """
    Safely calculates mathematical expression.
    
    Args:
        expression: Mathematical string, e.g. "sqrt(144) + 25 * 4"
        
    Returns:
        dict with keys: 'result', 'expression', 'status', 'error'
    """
    cleaned = expression.strip()
    if not cleaned:
        return {"result": None, "expression": expression, "status": "error", "error": "Empty expression"}

    # Replace common formatting characters
    cleaned = cleaned.replace("^", "**").replace("×", "*").replace("÷", "/")

    try:
        tree = ast.parse(cleaned, mode="eval")
        visitor = SafeEvalVisitor()
        value = visitor.visit(tree)
        return {
            "result": value,
            "expression": expression,
            "status": "success",
            "error": None
        }
    except ZeroDivisionError:
        return {
            "result": None,
            "expression": expression,
            "status": "error",
            "error": "Division by zero"
        }
    except Exception as exc:
        return {
            "result": None,
            "expression": expression,
            "status": "error",
            "error": f"Evaluation error: {str(exc)}"
        }
