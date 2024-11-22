import random

import numba

import minitorch

import time

from typing import Any, Callable

datasets = minitorch.datasets
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def default_log_fn(epoch: int, total_loss: float, correct: int, losses: list) -> None:
    """Log the training progress.

    Args:
        epoch (int): The current epoch number.
        total_loss (float): The total loss for the epoch.
        correct (int): The number of correct predictions.
        losses (list): A list of loss values for the epoch.

    Returns:
        None

    """
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


def RParam(*shape: int, backend: type) -> minitorch.Parameter:
    """Create a random parameter tensor.

    Args:
        *shape (int): The dimensions of the parameter tensor.
        backend (type): The backend to use for tensor operations.

    Returns:
        minitorch.Parameter: A parameter initialized with random values.

    """
    r = minitorch.rand(shape, backend=backend) - 0.5
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden: int, backend: type):
        """Initialize the Network model.

        Args:
            hidden (int): The number of hidden units in the network.
            backend (type): The backend to use for computations.

        """
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden, backend)
        self.layer2 = Linear(hidden, hidden, backend)
        self.layer3 = Linear(hidden, 1, backend)

    def forward(self, x: Any) -> Any:
        """Perform a forward pass through the model.

        Args:
            x (Any): The input data for the forward pass.

        Returns:
            Any: The output from the forward pass.

        """
        # TODO: Implement for Task 3.5.
        x = self.layer1.forward(
            x
        ).relu()  # Pass through the first layer and apply ReLU activation
        x = self.layer2.forward(
            x
        ).relu()  # Pass through the second layer and apply ReLU activation
        return (
            self.layer3.forward(x).sigmoid()
        )  # Pass through the third layer and apply Sigmoid activation for output


class Linear(minitorch.Module):
    def __init__(self, in_size: int, out_size: int, backend: type):
        """Initialize the Linear layer.

        Args:
            in_size (int): The size of the input features.
            out_size (int): The size of the output features.
            backend (type): The backend to use for computations.

        """
        super().__init__()
        self.weights = RParam(in_size, out_size, backend=backend)
        s = minitorch.zeros((out_size,), backend=backend)
        s = s + 0.1
        self.bias = minitorch.Parameter(s)
        self.out_size = out_size

    def forward(self, x: Any) -> Any:
        """Perform a forward pass through the model.

        Args:
            x (Any): The input data for the forward pass.

        Returns:
            Any: The output from the forward pass.

        """
        # TODO: Implement for Task 3.5.
        # inputs: (BATCH, FEATURES)
        # weights: (FEATURES, HIDDEN)
        # bias: (HIDDEN)

        # Step 1: Reshape x to add an extra dimension for broadcasting
        # inputs: (BATCH, FEATURES, 1)
        x = x.view(x.shape[0], x.shape[1], 1)

        # Step 2: Element-wise multiplication
        # Multiply inputs (BATCH, FEATURES, 1) with weights (FEATURES, HIDDEN)
        # Result: (BATCH, FEATURES, HIDDEN)
        weighted_x = x * self.weights.value

        # Step 3: Sum over the FEATURES dimension to simulate dot product
        # Result: (BATCH, HIDDEN)
        summed_x = weighted_x.sum(dim=1)

        # Step 4: Add the bias (HIDDEN) using broadcasting
        # bias: (HIDDEN) is broadcasted to (BATCH, HIDDEN)
        output = summed_x + self.bias.value

        # Return the final result
        return output.view(output.shape[0], output.shape[2])


class FastTrain:
    def __init__(self, hidden_layers: int, backend: type = FastTensorBackend):
        """Initialize the FastTrain model.

        Args:
            hidden_layers (int): The number of hidden layers in the model.
            backend (type, optional): The backend to use for training. Defaults to FastTensorBackend.

        """
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers, backend)
        self.backend = backend

    def run_one(self, x: Any) -> Any:
        """Run the model on a single input.

        Args:
            x (Any): The input data to be processed by the model.

        Returns:
            Any: The output from the model after processing the input data.

        """
        return self.model.forward(minitorch.tensor([x], backend=self.backend))

    def run_many(self, X: Any) -> Any:
        """Run the model on multiple inputs.

        Args:
            X (Any): The input data to be processed by the model.

        Returns:
            Any: The output from the model after processing the input data.

        """
        return self.model.forward(minitorch.tensor(X, backend=self.backend))


    def train(self, data: Any, learning_rate: float, max_epochs: int = 500, log_fn: Callable = default_log_fn) -> None:
        """Train the model with the given data.

        Args:
            data (Any): The training data.
            learning_rate (float): The learning rate for the optimizer.
            max_epochs (int, optional): The maximum number of training epochs. Defaults to 500.
            log_fn (Callable, optional): The logging function to use. Defaults to `default_log_fn`.

        """
        self.model = Network(self.hidden_layers, self.backend)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        BATCH = 10
        losses = []
        epoch_times = []  # List to store epoch durations

        for epoch in range(max_epochs):
            # Start time
            start_time = time.time()

            total_loss = 0.0
            c = list(zip(data.X, data.y))
            random.shuffle(c)
            X_shuf, y_shuf = zip(*c)

            for i in range(0, len(X_shuf), BATCH):
                optim.zero_grad()
                X = minitorch.tensor(X_shuf[i : i + BATCH], backend=self.backend)
                y = minitorch.tensor(y_shuf[i : i + BATCH], backend=self.backend)

                # Forward
                out = self.model.forward(X).view(y.shape[0])
                prob = (out * y) + (out - 1.0) * (y - 1.0)
                loss = -prob.log()
                (loss / y.shape[0]).sum().view(1).backward()

                total_loss = loss.sum().view(1)[0]

                # Update
                optim.step()

            losses.append(total_loss)

            # Stop timing the epoch
            end_time = time.time()
            epoch_duration = end_time - start_time
            epoch_times.append(epoch_duration)

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                X = minitorch.tensor(data.X, backend=self.backend)
                y = minitorch.tensor(data.y, backend=self.backend)
                out = self.model.forward(X).view(y.shape[0])
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)
                # Log epoch duration
                print(f"Epoch {epoch}: Time = {epoch_duration:.2f} seconds")

        # Print average epoch time
        print(f"Average epoch time: {sum(epoch_times) / len(epoch_times):.2f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--PTS", type=int, default=50, help="number of points")
    parser.add_argument("--HIDDEN", type=int, default=10, help="number of hiddens")
    parser.add_argument("--RATE", type=float, default=0.05, help="learning rate")
    parser.add_argument("--BACKEND", default="cpu", help="backend mode")
    parser.add_argument("--DATASET", default="simple", help="dataset")
    parser.add_argument("--PLOT", default=False, help="dataset")

    args = parser.parse_args()

    PTS = args.PTS

    if args.DATASET == "xor":
        data = minitorch.datasets["Xor"](PTS)
    elif args.DATASET == "simple":
        # data = minitorch.datasets["Simple"].simple(PTS)
        data = minitorch.datasets["Simple"](PTS)
    elif args.DATASET == "split":
        data = minitorch.datasets["Split"](PTS)

    HIDDEN = int(args.HIDDEN)
    RATE = args.RATE

    FastTrain(
        HIDDEN, backend=FastTensorBackend if args.BACKEND != "gpu" else GPUBackend
    ).train(data, RATE)
