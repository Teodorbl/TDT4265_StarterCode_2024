import numpy as np
import utils
import typing

np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784, f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)

    u = 33.55274553571429
    s = 78.87550070784701

    print("Using values from train set:")
    print("Mean: ", u)
    print("Standard deviation: ", s, "\n")

    # Input normalization
    X = (X - u) / s    

    # Bias trick: Add  1 to the end of each input vector
    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)    

    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    
    N = targets.shape[0]
    C_vec = np.sum( targets * np.log(outputs), axis=1 )
    C = -1 / N * np.sum(C_vec)

    return C


class SoftmaxModel:

    def __init__(
        self,
        # Number of neurons per layer
        neurons_per_layer: typing.List[int],
        use_improved_sigmoid: bool,  # Task 3b hyperparameter
        use_improved_weight_init: bool,  # Task 3a hyperparameter
        use_relu: bool,  # Task 3c hyperparameter
    ):
        np.random.seed(
            1
        )  # Always reset random seed before weight init to get comparable results.
        # Define number of input nodes
        self.I = 784 # Edited
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_relu = use_relu
        self.use_improved_weight_init = use_improved_weight_init

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev + 1, size) # Edited for bias trick
            print("Initializing weight to shape:", w_shape)
            w = np.random.uniform(-1, 1, w_shape) # Edited for weight initialization
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    # Added:
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Args:
            z: input of shape [num_outputs, batch size]
        Returns:
            sigmoid activation of input
        """
        return 1 / (1 + np.exp(-z))
    
    # Added:
    def sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Args:
            z: input of shape [num_outputs, batch size]
        Returns:
            derivative of sigmoid activation of input
        """
        sigmoid_z = self.sigmoid(z)
        return sigmoid_z * (1 - sigmoid_z)
    
    # Added:
    def softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Args:
            z: input of shape [num_outputs, batch size]
        Returns:
            softmax activation of input
        """
        # Subtract max for numerical stability
        z -= np.max(z, axis=0, keepdims=True)
        exps = np.exp(z)
        return exps / np.sum(exps, axis=0, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...
        
        # Transpose matrices to follow standard convention for matrix dimensions in neural networks
        # Its slow, I know
        x = X.T                 # [I+1, N]
        W_h = self.ws[0].T      # [J, I+1]
        W_o = self.ws[1].T      # [K, J+1]

        # Pre- and post-activations
        self.z_h = W_h @ x                  # [J, N]
        self.a_h = self.sigmoid(self.z_h)   # [J, N]
        self.a_h = np.append(self.a_h, np.ones((1, self.a_h.shape[1])), axis=0)  # [J+1, N] Bias trick
        z_o = W_o @ self.a_h                # [K, N]
        y_hat = self.softmax(z_o)           # [K, N]

        return y_hat.T

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert (
            targets.shape == outputs.shape
        ), f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer

        N = X.shape[0]
        x = X.T                 # [I+1, N]
        W_o = self.ws[1].T      # [K, J+1]
        W_o_tilde = W_o[:, :-1] # [K, J]
        y_hat = outputs.T       # [K, N]
        y = targets.T           # [K, N]

        Delta_o = y_hat - y     # [K, N]
        Delta_h = (W_o_tilde.T @ Delta_o) * self.sigmoid_derivative(self.z_h)   # [J, N]

        grad_o = 1/N * Delta_o @ self.a_h.T     # [K, J+1]
        grad_h = 1/N * Delta_h @ x.T            # [J, I+1]

        self.grads[0] = grad_h.T                # [I+1, J]
        self.grads[1] = grad_o.T                # [J+1, K]

        for grad, w in zip(self.grads, self.ws):
            assert (
                grad.shape == w.shape
            ), f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # Create a [Num examples, num classes] matrix with zeros
    n_examples = Y.size
    Y_one_hot = np.zeros((n_examples, num_classes))

    for i in range(n_examples):
        val = Y[i]
        Y_one_hot[i][ val ] = 1

    return Y_one_hot


def gradient_approximation_test(model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
    Numerical approximation for gradients. Should not be edited.
    Details about this test is given in the appendix in the assignment.
    """

    assert isinstance(X, np.ndarray) and isinstance(
        Y, np.ndarray
    ), f"X and Y should be of type np.ndarray!, got {type(X), type(Y)}"

    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**1, (
                    f"Calculated gradient is incorrect. "
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n"
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n"
                    f"If this test fails there could be errors in your cross entropy loss function, "
                    f"forward function or backward function"
                )


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert (
        Y[0, 3] == 1 and Y.sum() == 1
    ), f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert (
        X_train.shape[1] == 785
    ), f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_relu = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    main()
