from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol, List


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # Create a mutable list from vals
    val_list = list(vals)
    original_value = val_list[arg]

    # Forward value
    val_list[arg] = original_value + epsilon
    f_plus = f(*val_list)

    # Backward value
    val_list[arg] = original_value - epsilon
    f_minus = f(*val_list)

    # Restore original value
    val_list[arg] = original_value

    # Central difference formula
    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    """Protocol for variables in the computational graph."""

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative value."""
        ...

    @property
    def unique_id(self) -> int:
        """Return the unique ID of the variable."""
        ...

    def is_leaf(self) -> bool:
        """Check if the variable is a leaf node in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """Check if the variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return the parent variables of the current variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Perform the chain rule for backpropagation."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # visited = set()  # To keep track of visited nodes
    # order = []  # To store the topological order

    # def visit(v: Variable) -> None:
    #     """Visit the node and its parents."""
    #     if v not in visited:
    #         visited.add(v)
    #         for parent in v.parents:
    #             visit(parent)
    #         order.append(v)

    # visit(variable)  # Start the visit from the right-most variable
    # return reversed(order)  # Return in topological order

    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
        variable: The right-most variable in the computation graph.
        deriv: The derivative of the output with respect to this variable that we want to propagate backward to the leaves.

    """
    # # Accumulate the derivative only for leaf nodes
    # if variable.is_leaf():
    #     variable.accumulate_derivative(deriv)
    # else:
    #     # If it's not a leaf, propagate the derivative using the chain rule
    #     for parent, grad in variable.chain_rule(deriv):
    #         backpropagate(parent, grad)

    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved tensors."""
        return self.saved_values
