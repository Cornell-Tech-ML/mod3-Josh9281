import minitorch
import time
import numpy as np
from typing import Dict

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend: minitorch.TensorBackend, size: int = 16) -> None:
    """Perform matrix multiplication using the specified backend.

    Args:
    ----
        backend (minitorch.TensorBackend): The backend to use for matrix multiplication.
        size (int, optional): The size of the square matrices. Defaults to 16.

    """
    batch_size = 2
    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    _ = x @ y  # Explicitly discard result to avoid linting errors


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    times: Dict[int, Dict[str, float]] = {}

    for size in [64, 128, 256, 512, 1024]:
        print(f"Running size {size}")
        times[size] = {}
        fast_times = []
        gpu_times = []
        for _ in range(ntrials):
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        times[size]["fast"] = float(np.mean(fast_times))
        times[size]["gpu"] = float(np.mean(gpu_times))
        print(times[size])

    print("\nTiming summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for backend, timing in stimes.items():
            print(f"    {backend}: {timing:.5f}")
