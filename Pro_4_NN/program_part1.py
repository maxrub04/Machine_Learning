import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class NeuralNetworkRegression:
    def __init__(self, input_size, hidden_size, output_size,
                 activation='tanh', init='xavier', seed=42):
        """
        Initialization of weights and biases, and the history of MSE and R² values
        """

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_name = activation
        self.init_name = init

        rng = np.random.default_rng(seed)

        # Initializing weights for the input-hidden and hidden-output layers
        if init == 'xavier':
            limit_ih = np.sqrt(6.0 / (input_size + hidden_size))
            limit_ho = np.sqrt(6.0 / (hidden_size + output_size))
            self.weights_input_hidden = rng.uniform(-limit_ih, limit_ih, (input_size, hidden_size))
            self.weights_hidden_output = rng.uniform(-limit_ho, limit_ho, (hidden_size, output_size))
        elif init == 'he':
            self.weights_input_hidden = rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)
            self.weights_hidden_output = rng.standard_normal((hidden_size, output_size)) * np.sqrt(2.0 / hidden_size)
        elif init == 'small_normal':
            self.weights_input_hidden = rng.standard_normal((input_size, hidden_size)) * 0.3
            self.weights_hidden_output = rng.standard_normal((hidden_size, output_size)) * 0.3
        else:
            raise ValueError(f"Unknown init strategy: {init}")

        # Initializing biases for the hidden and output layers
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)
        # Initializing the history of MSE and R^2 values
        self.history_mse = []
        self.history_r2 = []

    def relu(self, x):
        if self.activation_name == 'tanh':
            return np.tanh(x)
        if self.activation_name == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-x))
        if self.activation_name == 'swish':
            return x * (1.0 / (1.0 + np.exp(-x)))
        if self.activation_name == 'elu':
            return np.where(x > 0.0, x, np.exp(x) - 1.0)
        raise ValueError(self.activation_name)

    def relu_derivative(self, x):
        if self.activation_name == 'tanh':
            return 1.0 - np.tanh(x) ** 2
        if self.activation_name == 'sigmoid':
            s = 1.0 / (1.0 + np.exp(-x))
            return s * (1.0 - s)
        if self.activation_name == 'swish':
            s = 1.0 / (1.0 + np.exp(-x))
            return s + x * s * (1.0 - s)
        if self.activation_name == 'elu':
            return np.where(x > 0.0, 1.0, np.exp(x))
        raise ValueError(self.activation_name)

    def forward(self, X):
        # Forward pass through the network
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.relu(hidden_input)
        return np.dot(hidden_output, self.weights_hidden_output) + self.bias_output

    def backward(self, X, y, output, learning_rate, reg_lambda):
        # Backward propagation of error through the network
        output_error = y - output
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.relu(hidden_input)
        gradient_hidden_output = np.dot(hidden_output.T, output_error)
        hidden_error = np.dot(output_error, self.weights_hidden_output.T)
        hidden_error *= self.relu_derivative(hidden_input)
        gradient_input_hidden = np.dot(X.T, hidden_error)
        # Updating weights and biases
        self.weights_hidden_output += (gradient_hidden_output - reg_lambda * self.weights_hidden_output) * learning_rate
        self.weights_input_hidden += (gradient_input_hidden - reg_lambda * self.weights_input_hidden) * learning_rate
        self.bias_output += np.sum(output_error, axis=0) * learning_rate
        self.bias_hidden += np.sum(hidden_error, axis=0) * learning_rate

    def train(self, X, y, epochs, learning_rate, reg_lambda):
        # Network training
        print("Start training neural networks")
        for epoch in range(1, epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate, reg_lambda)
            if epoch % (epochs // 10) == 0:
                print(f"[%] {epoch / epochs}")  # Intermediate progress display
            mse = mean_squared_error(y, output)
            r2 = r2_score(y, output)
            self.history_mse.append(mse)
            self.history_r2.append(r2)
            if r2 >= 0.95:  # Stop training if R^2 reaches 0.95
                print("Training stopped, R^2 score reached 0.95")
                break


# Per-dataset configuration. Activation + init + lr + lambda tuned to the
# visible shape of each file; ReLU is deliberately avoided.
DATASETS = [
    # dane1  – parabola in [-2, 2]
    dict(file=1,  hidden=10, activation='tanh',    init='xavier',       lr=0.005, reg=1e-4, epochs=15000, seed=42),
    # dane2  – cubic in [-3, 3]
    dict(file=2,  hidden=12, activation='tanh',    init='xavier',       lr=0.005, reg=1e-5, epochs=20000, seed=42),
    # dane3  – noisy double-hump in [-3, 3]
    dict(file=3,  hidden=15, activation='tanh',    init='xavier',       lr=0.012, reg=1e-4, epochs=40000, seed=42),
    # dane6  – quartic-like, sharp tails in [-4, 4]
    dict(file=6,  hidden=25, activation='tanh',    init='xavier',       lr=0.012, reg=1e-5, epochs=40000, seed=42),
    # dane8  – smooth bell-shaped curve in [-3, 3]
    dict(file=8,  hidden=12, activation='tanh',    init='xavier',       lr=0.005, reg=1e-4, epochs=15000, seed=42),
    # dane10 – nearly linear trend in [-2, 5]; tiny sigmoid net is enough
    dict(file=10, hidden=4,  activation='sigmoid', init='small_normal', lr=0.02,  reg=0.0,  epochs=10000, seed=42),
    # dane12 – exponential growth in [-2, 3]; swish + He init handle it well
    dict(file=12, hidden=14, activation='swish',   init='he',           lr=0.003, reg=1e-4, epochs=15000, seed=42),
    # dane14 – noisy negative cubic in [-3, 3]
    dict(file=14, hidden=20, activation='tanh',    init='xavier',       lr=0.01,  reg=1e-5, epochs=25000, seed=3),
]


for cfg in DATASETS:
    XX = cfg['file']
    hidden_size = cfg['hidden']
    learning_rate = cfg['lr']
    reg_lambda = cfg['reg']
    epochs = cfg['epochs']

    # Data read
    with open(f"./dane{XX}.txt", "r") as file:
        data = file.readlines()
    x_data, y_data = zip(*[map(float, line.split()) for line in data])

    # Data normalization and split
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_data_normalized = scaler_x.fit_transform(np.array(x_data).reshape(-1, 1))
    y_data_normalized = scaler_y.fit_transform(np.array(y_data).reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X_data_normalized, y_data_normalized, test_size=0.2,
                                                        random_state=42)

    # Initializing and training a neural network
    input_size = 1
    output_size = 1
    nn = NeuralNetworkRegression(input_size, hidden_size, output_size,
                                 activation=cfg['activation'], init=cfg['init'], seed=cfg['seed'])
    nn.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate, reg_lambda=reg_lambda)

    # Predictions for training and test data
    output_train = nn.forward(X_train)
    output_test = nn.forward(X_test)

    # Last MSE and R^2 values
    final_mse = nn.history_mse[-1]
    final_r2 = nn.history_r2[-1]
    n_params = (nn.weights_input_hidden.size + nn.bias_hidden.size
                + nn.weights_hidden_output.size + nn.bias_output.size)

    # Charts
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.scatter(range(len(nn.history_mse)), nn.history_mse, marker='.')
    plt.xlabel('')
    plt.ylabel('MSE')
    plt.title(f'MSE vs. Epochs, Final MSE:{round(final_mse, 4)}')

    plt.subplot(2, 2, 2)
    plt.scatter(range(len(nn.history_r2)), nn.history_r2, marker='.')
    plt.xlabel('Epochs')
    plt.ylabel('R^2')
    plt.title(f'R^2 vs. Epochs, Final R^2:{round(final_r2, 2)}')
    plt.ylim(min(-0.1, min(nn.history_r2)), 1.05)
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    plt.scatter(X_train, y_train, color='blue', label='Training Data')
    plt.scatter(X_test, y_test, color='red', label='Test Data')
    plt.scatter(X_train, output_train, color='green', label='Predictions', marker='s')
    plt.scatter(X_test, output_test, color='green', label='Predictions', marker='x')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"Neurons={hidden_size}, Params={n_params}, Epochs={len(nn.history_r2)}, "
              f"learn_r={learning_rate}, lambda={reg_lambda}")
    plt.legend()

    plt.suptitle(f"dane{XX}.txt — activation={cfg['activation']}, init={cfg['init']}", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    plt.show()
