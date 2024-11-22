"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple, Optional

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        """Backward pass for the function.

        Args:
        ----
            ctx (Context): The context containing saved values for backpropagation.
            grad_out (Tensor): Gradient of the output with respect to some loss.

        Returns:
        -------
            Tuple[Tensor, ...]: Gradients with respect to the inputs.

        """
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        """Forward pass for the function.

        Args:
        ----
            ctx (Context): The context to save necessary values for backpropagation.
            inps (Tensor): Input tensors for the forward pass.

        Returns:
        -------
            Tensor: Result of the forward operation.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Negate the elements of the input tensor.

        Args:
        ----
            ctx (Context): Context for saving values for backward pass.
            t1 (Tensor): Input tensor to negate.

        Returns:
        -------
            Tensor: Negated tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for negation.

        Args:
        ----
            ctx (Context): Context with saved values from the forward pass.
            grad_output (Tensor): Gradient of the output with respect to some loss.

        Returns:
        -------
            Tensor: Gradient of the input with respect to the loss.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for the Inv function."""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for the Inv function."""
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the forward pass for the Add function."""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass for the Add function."""
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


# TODO: Implement for Task 2.3.
class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the forward pass for the Mul function."""
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass for the Mul function."""
        t1, t2 = ctx.saved_tensors
        return grad_output * t2, grad_output * t1


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for the Sigmoid function."""
        ctx.save_for_backward(t1)
        return t1.f.sigmoid_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for the Sigmoid function."""
        (t1,) = ctx.saved_tensors
        sig = t1.f.sigmoid_map(t1)
        one_minus_sigmoid = t1.f.add_zip(
            t1.f.id_map(sig) / sig, t1.f.neg_map(sig)
        )  # (1 - sigmoid(x))
        sigmoid_derivative = t1.f.mul_zip(
            sig, one_minus_sigmoid
        )  # sigmoid(x) * (1 - sigmoid(x))
        return t1.f.mul_zip(
            sigmoid_derivative, grad_output
        )  # Multiply the derivative by the incoming gradient (grad_output)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for the ReLU function."""
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for the ReLU function."""
        (t1,) = ctx.saved_tensors
        relu_grad = grad_output.f.relu_back_zip(t1, grad_output)
        return relu_grad


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for the Log function."""
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for the Log function."""
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for the Exp function."""
        ctx.save_for_backward(t1)
        return t1.f.exp_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for the Exp function."""
        (t1,) = ctx.saved_tensors
        exp_t1 = t1.f.exp_map(t1)  # exp(t1)
        return t1.f.mul_zip(exp_t1, grad_output)  # grad_output * exp(t1)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Compute the forward pass for the Sum function."""
        ctx.save_for_backward(t1)
        return t1.f.add_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the backward pass for the Sum function."""
        (t1,) = ctx.saved_values
        return grad_output.f.id_map(grad_output, t1), t1


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the forward pass for the LT function."""
        ctx.save_for_backward(t1, t2)
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass for the LT function."""
        # Create tensors filled with zeros having the same shape as t1 and t2
        (t1, t2) = ctx.saved_values
        zero_grad_t1 = grad_output.f.id_map(tensor(0), t1)
        zero_grad_t2 = grad_output.f.id_map(tensor(0), t2)

        return zero_grad_t1, zero_grad_t2


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for the Equal (EQ) function.

        Args:
        ----
            ctx (Context): The context to save necessary values for backpropagation.
            t1 (Tensor): The first input tensor to compare.
            t2 (Tensor): The second input tensor to compare.

        Returns:
        -------
            Tensor: A tensor indicating the result of the comparison (t1 == t2).

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for the Equal (EQ) function.

        Args:
        ----
            ctx (Context): Context with saved values from the forward pass.
            grad_output (Tensor): Gradient of the output with respect to some loss.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the inputs t1 and t2 with respect to the loss.

        """
        # Create tensors filled with zeros having the same shape as t1 and t2
        (t1, t2) = ctx.saved_values
        zero_grad_t1 = grad_output.f.id_map(tensor(0), t1)
        zero_grad_t2 = grad_output.f.id_map(tensor(0), t2)

        return zero_grad_t1, zero_grad_t2


class GT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for the Greater Than (GT) function.

        Args:
        ----
            ctx (Context): The context to save necessary values for backpropagation.
            t1 (Tensor): The first input tensor to compare.
            t2 (Tensor): The second input tensor to compare.

        Returns:
        -------
            Tensor: A tensor indicating the result of the comparison (t1 > t2).

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.lt_zip(t2, t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for the Less Than (LT) function.

        Args:
        ----
            ctx (Context): Context with saved values from the forward pass.
            grad_output (Tensor): Gradient of the output with respect to some loss.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the inputs t1 and t2 with respect to the loss.

        """
        # Create tensors filled with zeros having the same shape as t1 and t2
        (t1, t2) = ctx.saved_values
        zero_grad_t1 = grad_output.f.id_map(tensor(0), t1)
        zero_grad_t2 = grad_output.f.id_map(tensor(0), t2)

        return zero_grad_t1, zero_grad_t2


class IsClose(Function):
    @staticmethod
    def forward(
        ctx: Context, t1: Tensor, t2: Tensor, atol: float = 1e-8, rtol: float = 1e-5
    ) -> Tensor:
        """Forward pass for the IsClose function.

        Args:
        ----
            ctx (Context): The context to save necessary values for backpropagation.
            t1 (Tensor): The first input tensor to compare.
            t2 (Tensor): The second input tensor to compare.
            atol (float, optional): Absolute tolerance. Default is 1e-8.
            rtol (float, optional): Relative tolerance. Default is 1e-5.

        Returns:
        -------
            Tensor: A tensor indicating whether the elements of t1 and t2 are close.

        """
        return t1.f.is_close_zip(t1, t2)

    # No backward since it's specified not to require backward


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tensor) -> Tensor:
        """Forward pass for the Permute function.

        Args:
        ----
            ctx (Context): The context to save necessary values for backpropagation.
            t1 (Tensor): The input tensor to permute.
            order (Tensor): The order in which to permute the dimensions.

        Returns:
        -------
            Tensor: A new tensor with dimensions permuted according to the specified order.

        """
        # ctx.save_for_backward(order)
        # # order_list = order.to_numpy().astype(int).tolist()
        # # order_list = order.cpu().numpy().astype(int).tolist() # Convert order tensor to a Python list
        # # order_list = order._tensor.tolist()
        # order_list = np.array(order._tensor).astype(int).tolist()
        # return t1._new(t1._tensor.permute(*order_list))

        _order = [int(order[i]) for i in range(order.size)]
        ctx.save_for_backward(_order)
        return t1._new(t1._tensor.permute(*_order))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the Permute function.

        Args:
        ----
            ctx (Context): Context with saved values from the forward pass.
            grad_output (Tensor): Gradient of the output with respect to some loss.

        Returns:
        -------
            Tuple[Tensor, float]: Gradient of the input tensor and a placeholder float.

        """
        (order,) = ctx.saved_values
        reverse_order, _ = zip(*sorted(enumerate(order), key=lambda pair: pair[1]))
        return grad_output._new(grad_output._tensor.permute(*reverse_order)), 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Reshape the input tensor to the specified shape.

        Args:
        ----
            ctx (Context): The context to save necessary values for backpropagation.
            a (Tensor): The input tensor to reshape.
            shape (Tensor): The desired shape for the output tensor.

        Returns:
        -------
            Tensor: A new tensor with the specified shape.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Calculate the gradient using central difference approximation.

    Args:
    ----
        f: The function for which the gradient is being calculated.
        vals: The input tensors to the function.
        arg: The index of the argument to differentiate.
        epsilon: The small value used for the central difference.
        ind: The index of the element to perturb.

    Returns:
    -------
        The estimated gradient at the specified index.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
