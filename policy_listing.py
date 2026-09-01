"""
policy_listing.py
-----------------
Render a policy tree in the multi-line listing style the paper uses for its
normative policies (Algorithms Cat I / II-a / II-b / II-c), e.g.

    IF COUNT(ready, ALL) > 3 OR MAX(truck, status, ALL) = 1
        THEN FIRE(transporting)
    ELSE POSTPONE

`repr` on a tree gives one fully-parenthesised line, which is fine for a log
but unreadable in a results table, so this collapses the else-chain into
IF / ELSE IF / ELSE and drops the parentheses that operator precedence already
implies.
"""

from __future__ import annotations

from policy_grammar import (
    Count, Sum, Min, Max, Mean, Clock, Number,
    Add, Sub, Mul, Div, Compare, And, Or, Not,
    Fire, Postpone, IfThenElse,
)

# Binding power, low to high; used to decide when a sub-expression needs
# bracketing.  OR binds loosest, then AND, NOT, comparison, +/-, and finally
# */( -- the same hierarchy the grammar's production rules enforce.
_PREC = {Or: 1, And: 2, Not: 3, Compare: 4,
         Add: 5, Sub: 5, Mul: 6, Div: 6}

_AGG_NAME = {Count: 'COUNT', Sum: 'SUM', Min: 'MIN', Max: 'MAX', Mean: 'MEAN'}
_BINOP    = {Add: '+', Sub: '-', Mul: '*', Div: '/', And: 'AND', Or: 'OR'}


def _prec(node) -> int:
    return _PREC.get(type(node), 99)


def _fmt_number(value) -> str:
    """Integers without a trailing .0; floats without trailing zeros."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def expr(node, parent_prec: int = 0) -> str:
    """Render an expression / condition node, bracketing only where needed."""
    t = type(node)

    if t is Count:
        return f"COUNT({node.place}, {node.selector})"
    if t in (Sum, Min, Max, Mean):
        return f"{_AGG_NAME[t]}({node.place}, {node.attribute}, {node.selector})"
    if t is Clock:
        return "CLOCK"
    if t is Number:
        return _fmt_number(node.value)

    if t is Not:
        return f"NOT {expr(node.operand, _prec(node))}"

    if t is Compare:
        # Comparison operands bind tighter than the comparison itself, so they
        # never need brackets of their own here.
        body = f"{expr(node.left, _prec(node))} {node.op} {expr(node.right, _prec(node))}"
        return f"({body})" if parent_prec > _prec(node) else body

    if t in _BINOP:
        p = _prec(node)
        # Left-associative: the right operand needs brackets at equal precedence
        # (a - (b - c) is not a - b - c), the left operand does not.
        body = f"{expr(node.left, p)} {_BINOP[t]} {expr(node.right, p + 1)}"
        return f"({body})" if parent_prec > p else body

    return repr(node)


def action(node) -> str:
    if isinstance(node, Fire):
        return f"FIRE({node.transition})"
    if isinstance(node, Postpone):
        return "POSTPONE"
    return repr(node)


def _wrap(text: str, width: int, indent: str) -> list[str]:
    """Break a long condition at OR / AND boundaries, not mid-expression."""
    if len(text) <= width:
        return [text]
    out, line = [], ""
    for token in text.split(" "):
        if line and len(line) + 1 + len(token) > width:
            out.append(line)
            line = indent + token
        else:
            line = token if not line else f"{line} {token}"
    if line:
        out.append(line)
    return out


def listing(tree, width: int = 78) -> str:
    """The paper-style listing for *tree*."""
    lines: list[str] = []
    node = tree
    first = True

    while isinstance(node, IfThenElse):
        keyword = "IF" if first else "ELSE IF"
        cond = _wrap(f"{keyword} {expr(node.condition)}", width, " " * (len(keyword) + 4))
        lines.extend(cond)
        lines.append(f"    THEN {action(node.then)}")
        node, first = node.else_, False

    # The chain always bottoms out in an action.
    lines.append(("ELSE " if not first else "") + action(node))
    return "\n".join(lines)
