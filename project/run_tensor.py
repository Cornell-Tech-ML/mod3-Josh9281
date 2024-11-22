"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch


# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


# TODO: Implement for Task 2.5.
# Define the Linear class (handles a single linear transformation)
class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)

    def forward(self, inputs):
        # return inputs @ self.weights.value + self.bias.value

        # inputs: (BATCH, FEATURES)
        # weights: (FEATURES, HIDDEN)
        # bias: (HIDDEN)

        # Step 1: Reshape inputs to add an extra dimension for broadcasting
        # inputs: (BATCH, FEATURES, 1)
        inputs = inputs.view(inputs.shape[0], inputs.shape[1], 1)

        # Step 2: Element-wise multiplication
        # Multiply inputs (BATCH, FEATURES, 1) with weights (FEATURES, HIDDEN)
        # Result: (BATCH, FEATURES, HIDDEN)
        weighted_inputs = inputs * self.weights.value

        # Step 3: Sum over the FEATURES dimension to simulate dot product
        # Result: (BATCH, HIDDEN)
        summed_inputs = weighted_inputs.sum(dim=1)

        # Step 4: Add the bias (HIDDEN) using broadcasting
        # bias: (HIDDEN) is broadcasted to (BATCH, HIDDEN)
        output = summed_inputs + self.bias.value

        # Return the final result
        return output.view(output.shape[0], output.shape[2])


# Define the Network class (multi-layer perceptron with three layers)
class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)  # Input layer to hidden layer
        self.layer2 = Linear(
            hidden_layers, hidden_layers
        )  # Hidden layer to hidden layer
        self.layer3 = Linear(hidden_layers, 1)  # Hidden layer to output layer

    def forward(self, x):
        x = self.layer1.forward(
            x
        ).relu()  # Pass through the first layer and apply ReLU activation
        x = self.layer2.forward(
            x
        ).relu()  # Pass through the second layer and apply ReLU activation
        return (
            self.layer3.forward(x).sigmoid()
        )  # Pass through the third layer and apply Sigmoid activation for output


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
