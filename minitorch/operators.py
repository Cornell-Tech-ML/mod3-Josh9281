"""Collection of the core mathematical operators used throughout the code base."""


# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

# TODO: Implement for Task 0.1.
import math
from typing import Union, Callable, Iterable

Number = Union[int, float]


def mul(x: Number, y: Number) -> Number:
    """Multiply two numbers."""
    return x * y


def id(x: Number) -> Number:
    """Return the input number (identity function)."""
    return x


def add(x: Number, y: Number) -> Number:
    """Add two numbers."""
    return x + y


def neg(x: Number) -> Number:
    """Negate a number."""
    return -x


def lt(x: Number, y: Number) -> bool:
    """Check if x is less than y."""
    return x < y


def eq(x: Number, y: Number) -> bool:
    """Check if x is equal to y."""
    return x == y


def max(x: Number, y: Number) -> Number:
    """Return the maximum of two numbers."""
    return x if x > y else y


def is_close(x: Number, y: Number) -> bool:
    """Check if two numbers are close (within 1e-2)."""
    return abs(x - y) < 1e-2


def sigmoid(x: Number) -> float:
    """Compute the sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: Number) -> Number:
    """Compute the ReLU (Rectified Linear Unit) function."""
    return 0 if x < 0 else x


def log(x: Number) -> float:
    """Compute the natural logarithm of a number."""
    return math.log(x)


def exp(x: Number) -> float:
    """Compute the exponential of a number."""
    return math.exp(x)


def log_back(x: Number, d: Number) -> float:
    """Compute the gradient of the natural logarithm."""
    return d / x


def inv(x: Number) -> float:
    """Compute the inverse (reciprocal) of a number."""
    return 1.0 / x


def inv_back(x: Number, d: Number) -> float:
    """Compute the gradient of the inverse function."""
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Compute the gradient of the ReLU function."""
    return d if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Apply a function to each element in a list and return the results."""

    def _map(lst: Iterable[float]) -> Iterable[float]:
        res = []
        for x in lst:
            res.append(fn(x))
        return res

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Apply a function to pairs of elements from two lists and return the results."""

    def _zipWith(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
        res = []
        for x, y in zip(lst1, lst2):
            res.append(fn(x, y))
        return res

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduce a list to a single value by applying a function cumulatively."""

    def _reduce(lst: Iterable[float]) -> float:
        val = start  # Explicit type hint for val
        for item in lst:
            val = fn(val, item)
        return val

    return _reduce


# Use these to implement
def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negate all numbers in the given list."""
    return map(neg)(lst)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements of two lists."""
    return zipWith(add)(lst1, lst2)


def sum(lst: Iterable[float]) -> float:
    """Calculate the sum of all numbers in the given list."""
    return reduce(add, 0.0)(lst)


def prod(lst: Iterable[float]) -> float:
    """Calculate the product of all numbers in the given list."""
    return reduce(mul, 1.0)(lst)


# TODO: Implement for Task 0.3.
