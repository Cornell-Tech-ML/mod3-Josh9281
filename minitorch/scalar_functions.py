from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Apply the backward function using the stored context and output derivative."""
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Apply the forward function using the provided inputs and context."""
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function and return the result as a Scalar.

        Args:
        ----
            vals: Input values for the scalar function.

        Returns:
        -------
            A new Scalar object created from the function's result.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform forward pass for addition."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Perform backward pass for addition."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform forward pass for log function."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform backward pass for log function."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform forward pass for multiplication."""
        ctx.save_for_backward(a, b)
        return float(a * b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Perform backward pass for multiplication."""
        (a, b) = ctx.saved_values
        return float(d_output * b), float(d_output * a)


class Inv(ScalarFunction):
    """Inverse function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform forward pass for inverse function."""
        ctx.save_for_backward(a)
        return float(1 / a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform backward pass for inverse function."""
        (a,) = ctx.saved_values
        return float(-d_output / (a**2))


class Neg(ScalarFunction):
    """Negation function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform forward pass for negation."""
        ctx.save_for_backward(a)
        return float(-a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform backward pass for negation."""
        return float(-d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform forward pass for sigmoid function."""
        # ctx.save_for_backward(a)
        # return float(1 / (1 + minitorch.operators.exp(-a)))
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform backward pass for sigmoid function."""
        # (a,) = ctx.saved_values
        # sig = 1 / (1 + minitorch.operators.exp(-a))
        # return float(d_output * sig * (1 - sig))
        sigma: float = ctx.saved_values[0]
        return sigma * (1.0 - sigma) * d_output


class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform forward pass for ReLU function."""
        ctx.save_for_backward(a)
        return float(max(0, a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform backward pass for ReLU function."""
        (a,) = ctx.saved_values
        return float(d_output * (1 if a > 0 else 0))


class Exp(ScalarFunction):
    """Exponential function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform forward pass for exponential function."""
        # ctx.save_for_backward(a)
        # return float(operators.exp(a))
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform backward pass for exponential function."""
        # (a,) = ctx.saved_values
        # return float(d_output * operators.exp(a))
        out: float = ctx.saved_values[0]
        return d_output * out


class LT(ScalarFunction):
    """Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform forward pass for less-than function."""
        return float(1.0 if a < b else 0.0)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Perform backward pass for less-than function."""
        return (
            float(0.0),
            float(0.0),
        )  # The derivative of less-than is zero everywhere.


class EQ(ScalarFunction):
    """Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform forward pass for equal function."""
        ctx.save_for_backward(a, b)
        return float(1.0 if a == b else 0.0)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Perform backward pass for equal function."""
        (a, b) = ctx.saved_values
        return (
            float(d_output * (1.0 if a == b else 0.0)),
            float(d_output * (1.0 if a == b else 0.0)),
        )
