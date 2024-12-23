"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    GT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Sets whether gradients should be tracked for this tensor.

        Args:
        ----
            x (bool): If True, operations on this tensor will be tracked for backpropagation.

        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Returns whether gradients are being tracked for this tensor.

        Returns
        -------
            bool: True if gradients are being tracked, False otherwise.

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Creates a tensor filled with zeros of the specified shape.

        Args:
        ----
            shape (Optional[UserShape]): The shape of the tensor.

        Returns:
        -------
            Tensor: A tensor filled with zeros.

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Checks if the tensor is constant (i.e., does not require gradients).

        Returns
        -------
            bool: True if the tensor is constant, False otherwise.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the parent variables of the current tensor in the computational graph.

        Returns
        -------
            Iterable[Variable]: Parent variables.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Implements the chain rule for backpropagation through the computational graph.

        Args:
        ----
            d_output (Any): The derivative with respect to the output.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: The chain of derivatives for backpropagation.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Performs backpropagation to compute the gradient of the loss with respect to this tensor.

        Args:
        ----
            grad_output (Optional[Tensor]): The gradient of the output tensor, or None if the tensor is a scalar.

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor."""
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Returns the number of dimensions (rank) of the tensor."""
        return len(self._tensor.shape)

    # Functions
    # TODO: Implement for Task 2.3.
    def __add__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, Neg.apply(self._ensure_tensor(b)))

    def __mul__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, self._ensure_tensor(b))

    def __lt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:
        return EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        return GT.apply(self, self._ensure_tensor(b))

    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    def __hash__(self):
        return hash((self.unique_id, self.shape))

    def __radd__(self, b: TensorLike) -> Tensor:
        return Add.apply(self._ensure_tensor(b), self)

    def __rmul__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), self)

    def all(self, dim: Optional[Union[int, Tensor]] = None) -> Tensor:
        """Checks if all elements along the specified dimension are True.

        Args:
        ----
            dim (Optional[Union[int, Tensor]]): The dimension to check.

        Returns:
        -------
            Tensor: A tensor indicating if all elements are True.

        """
        if dim is not None:
            return All.apply(self, self._ensure_tensor(dim))
        else:
            return All.apply(self)

    def is_close(self, b: TensorLike) -> Tensor:
        """Checks if this tensor is close to another tensor element-wise.

        Args:
        ----
            b (TensorLike): The tensor to compare.

        Returns:
        -------
            Tensor: A tensor of boolean values indicating closeness.

        """
        return IsClose.apply(self, self._ensure_tensor(b))

    def sigmoid(self) -> Tensor:
        """Applies the sigmoid function element-wise.

        Returns
        -------
            Tensor: A tensor with the sigmoid of each element.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Applies the ReLU (Rectified Linear Unit) function element-wise.

        Returns
        -------
            Tensor: A tensor with ReLU applied to each element.

        """
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Computes the logarithm of each element in the tensor.

        Returns
        -------
            Tensor: A tensor with the logarithm of each element.

        """
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Computes the exponential of each element in the tensor.

        Returns
        -------
            Tensor: A tensor with the exponential of each element.

        """
        return Exp.apply(self)

    def sum(self, dim: Optional[Union[int, Tensor]] = None) -> Tensor:
        """Computes the sum of elements along the specified dimension.

        Args:
        ----
            dim (Optional[Union[int, Tensor]]): The dimension to sum over.

        Returns:
        -------
            Tensor: The sum of the elements along the specified dimension.

        """
        if dim is None:
            return Sum.apply(
                self.contiguous().view(int(operators.prod(self.shape))),
                self._ensure_tensor(0),
            )
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Computes the mean of elements along the specified dimension.

        Args:
        ----
            dim (Optional[int]): The dimension to compute the mean over.

        Returns:
        -------
            Tensor: The mean of the elements along the specified dimension.

        """
        if dim is None:
            # If no dimension is specified, we compute the mean over all elements in the tensor
            total_elements = operators.prod(self.shape)
            sum_tensor = self.sum()
            return sum_tensor / Tensor.make(
                [total_elements], (1,), backend=self.backend
            )
        else:
            # If a dimension is specified, we compute the mean along that dimension
            sum_tensor = self.sum(dim)
            dim_size = self.shape[dim]
            return sum_tensor / Tensor.make([dim_size], (1,), backend=self.backend)

    def permute(self, *order: int) -> Tensor:
        """Rearranges the dimensions of the tensor according to the specified order.

        Args:
        ----
            order (int): The desired order of dimensions.

        Returns:
        -------
            Tensor: The permuted tensor.

        """
        order_tensor = Tensor.make(list(order), (len(order),), backend=self.backend)
        return Permute.apply(self, order_tensor)

    def view(self, *shape: int) -> Tensor:
        """Reshapes the tensor to the specified shape.

        Args:
        ----
            shape (int): The desired shape.

        Returns:
        -------
            Tensor: The reshaped tensor.

        """
        order_tensor = Tensor.make(list(shape), (len(shape),), backend=self.backend)
        return View.apply(self, order_tensor)

    def zero_grad_(self) -> None:
        """Zeros the gradient of the tensor."""
        self.grad = None
