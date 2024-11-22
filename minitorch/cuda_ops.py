# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
    """JIT compile the given function for device execution with optional keyword arguments.

    Args:
    ----
        fn (Callable[..., Any]): The function to be JIT compiled.
        **kwargs (Any): Additional keyword arguments for the JIT compilation.

    Returns:
    -------
        Callable[..., Any]: The JIT compiled function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Callable[..., Any], **kwargs: Any) -> FakeCUDAKernel:
    """JIT compile the given function with optional keyword arguments.

    Args:
    ----
        fn (Callable[..., Any]): The function to be JIT compiled.
        **kwargs (Dict[str, Any]): Additional keyword arguments for the JIT compilation.

    Returns:
    -------
        FakeCUDAKernel: The JIT compiled function.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Zip two tensors using the provided function.

        Args:
        ----
            fn (Callable[[float, float], float]): The function to apply to pairs of elements.

        Returns:
        -------
            Callable[[Tensor, Tensor], Tensor]: A callable that zips two tensors.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduce the tensor using the provided function.

        Args:
        ----
            fn (Callable[[float, float], float]): The function to apply for reduction.
            start (float, optional): The initial value for the reduction. Defaults to 0.0.

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: A callable that reduces the tensor.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Multiply two tensors.

        Args:
        ----
            a (Tensor): The first tensor to multiply.
            b (Tensor): The second tensor to multiply.

        Returns:
        -------
            Tensor: The result of the matrix multiplication.

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        while i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            pos = int(index_to_position(in_index, in_strides))
            out[i] = fn(in_storage[pos])
            i += cuda.blockDim.x * cuda.gridDim.x

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        while i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            out[i] = fn(a_storage[a_pos], b_storage[b_pos])
            i += cuda.blockDim.x * cuda.gridDim.x

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """Perform a practice sum kernel to prepare for reduction.

    Given an array of length `n` and an output array of size `n // blockDIM`,
    this function sums up each `blockDIM` values into a single output cell.

    For example, given `[a_1, a_2, ..., a_{100}]`, the output array will contain
    the sum of each block of values.

    Note:
    ----
        Each block must perform the summation using shared memory.

    Args:
    ----
        out (Storage): Storage for the output tensor.
        a (Storage): Storage for the input tensor.
        size (int): The length of the input tensor.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    # fetch data from global memory to shared memory
    cache[pos] = a[i] if i < size else 0.0
    cuda.syncthreads()

    if i < size:
        offset = BLOCK_DIM // 2
        while offset != 0:
            if pos < offset:
                cache[pos] += cache[pos + offset]
            cuda.syncthreads()
            offset //= 2

        if pos == 0:
            out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Sum the elements of the given tensor.

    Args:
    ----
        a (Tensor): The input tensor to sum.

    Returns:
    -------
        TensorData: The result of summing the elements of the tensor.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        if out_pos < out_size:
            to_index(out_pos, out_shape, out_index)
            out_index[reduce_dim] = pos
            cache[pos] = (
                a_storage[index_to_position(out_index, a_strides)]
                if pos < a_shape[reduce_dim]
                else reduce_value
            )
            cuda.syncthreads()
            if pos < a_shape[reduce_dim]:
                offset = BLOCK_DIM // 2
                while offset != 0:
                    if pos < offset:
                        cache[pos] = fn(cache[pos], cache[pos + offset])
                    cuda.syncthreads()
                    offset //= 2

                if pos == 0:
                    out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Perform a practice square matrix multiplication kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:
    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute:
    ```
    for i:
        for j:
            for k:
                out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    # # Define the shared memory
    # shared_a = cuda.shared.array((32, 32), dtype=numba.float64)
    # shared_b = cuda.shared.array((32, 32), dtype=numba.float64)

    # # Determine thread row and column within the block
    # tx = cuda.threadIdx.x
    # ty = cuda.threadIdx.y

    # # Determine global row and column indices
    # row = tx
    # col = ty

    # # Initialize the output value for this thread
    # result = 0.0

    # # Loop over tiles of the input matrices
    # for tile in range((size + 31) // 32):  # Tile index (accounting for boundary)
    #     # Load elements into shared memory
    #     if row < size and (tile * 32 + col) < size:
    #         shared_a[row, col] = a[row * size + (tile * 32 + col)]
    #     else:
    #         shared_a[row, col] = 0.0  # Boundary padding

    #     if (tile * 32 + row) < size and col < size:
    #         shared_b[row, col] = b[(tile * 32 + row) * size + col]
    #     else:
    #         shared_b[row, col] = 0.0  # Boundary padding

    #     # Synchronize threads to ensure shared memory is fully loaded
    #     cuda.syncthreads()

    #     # Compute the partial dot product for the tile
    #     for k in range(32):  # Loop over shared memory dimension
    #         result += shared_a[row, k] * shared_b[k, col]

    #     # Synchronize threads before loading the next tile
    #     cuda.syncthreads()

    # # Write the final result to global memory
    # if row < size and col < size:
    #     out[row * size + col] = result

    BLOCK_DIM = 32  # Maximum block size, matching the constraint of size < 32
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    # Shared memory for tiles of A and B
    shared_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), dtype=numba.float32)
    shared_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), dtype=numba.float32)

    # Calculate row and column indices
    row = by * cuda.blockDim.y + ty
    col = bx * cuda.blockDim.x + tx

    # Initialize the output value for this thread
    temp = 0.0

    # Loop over all tiles of the input matrices
    for t in range((size + BLOCK_DIM - 1) // BLOCK_DIM):
        # Load a tile of A and B into shared memory
        if row < size and t * BLOCK_DIM + tx < size:
            shared_a[ty, tx] = a[row * size + t * BLOCK_DIM + tx]
        else:
            shared_a[ty, tx] = 0.0

        if t * BLOCK_DIM + ty < size and col < size:
            shared_b[ty, tx] = b[(t * BLOCK_DIM + ty) * size + col]
        else:
            shared_b[ty, tx] = 0.0

        # Synchronize to ensure all threads have loaded their tiles
        cuda.syncthreads()

        # Perform the computation for this tile
        for k in range(BLOCK_DIM):
            temp += shared_a[ty, k] * shared_b[k, tx]

        # Synchronize again to prevent race conditions on shared memory
        cuda.syncthreads()

    # Write the result to the output matrix
    if row < size and col < size:
        out[row * size + col] = temp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Perform matrix multiplication on two tensors.

    Args:
    ----
        a (Tensor): The first tensor.
        b (Tensor): The second tensor.

    Returns:
    -------
        TensorData: The result of the matrix multiplication.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.

    # Initialize the output value for this thread
    sum_out = 0.0

    # Loop over tiles of the matrices
    for t in range((a_shape[-1] + BLOCK_DIM - 1) // BLOCK_DIM):
        # Load a tile of a into shared memory
        if i < a_shape[-2] and t * BLOCK_DIM + pj < a_shape[-1]:
            a_index = (
                batch * a_batch_stride
                + i * a_strides[-2]
                + (t * BLOCK_DIM + pj) * a_strides[-1]
            )
            a_shared[pi, pj] = a_storage[a_index]
        else:
            a_shared[pi, pj] = 0.0

        # Load a tile of b into shared memory
        if t * BLOCK_DIM + pi < b_shape[-2] and j < b_shape[-1]:
            b_index = (
                batch * b_batch_stride
                + (t * BLOCK_DIM + pi) * b_strides[-2]
                + j * b_strides[-1]
            )
            b_shared[pi, pj] = b_storage[b_index]
        else:
            b_shared[pi, pj] = 0.0

        # Synchronize to ensure all threads have loaded their tiles
        cuda.syncthreads()

        # Compute partial dot product for the current tile
        for k in range(BLOCK_DIM):
            sum_out += a_shared[pi, k] * b_shared[k, pj]

        # Synchronize before loading the next tile
        cuda.syncthreads()

    # Write the final result to global memory
    if i < out_shape[-2] and j < out_shape[-1]:
        out_index = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
        out[out_index] = sum_out


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
