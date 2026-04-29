### Template courtesy of Dr. Michał Majewski, translated from Polish
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class NeuralNetworkRegression:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialization of weights and biases, and the history of MSE and R² values
        """

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Initializing weights for the input-hidden and hidden-output layers
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        # Initializing biases for the hidden and output layers
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)
        # Initializing the history of MSE and R² values
        self.history_mse = []
        self.history_r2 = []

    def relu(self, x):
        # ReLU activation function
        return np.maximum(0, x)

    def relu_derivative(self, x):
        # Derivative of the ReLU activation function
        return np.where(x > 0, 1, np.where(x < 0, 0, 0.5))

    def forward(self, X):
        # Forward pass through the network
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.relu(hidden_input)
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        return self.relu(output_input)

    def backward(self, X, y, output, learning_rate, reg_lambda):
        # Backward propagation of error through the network
        output_error = y - output
        hidden_output = self.relu(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        gradient_hidden_output = np.dot(hidden_output.T, output_error)
        hidden_error = np.dot(output_error, self.weights_hidden_output.T)
        hidden_error *= self.relu_derivative(hidden_output)
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
            if r2 >= 0.95:  # Stop training if R² reaches 0.95
                print("Training stopped, R^2 score reached 0.95")
                break


XX = 3

# Data read
with open(f"./dane{XX}.txt", "r") as file:
    data = file.readlines()
x_data, y_data = zip(*[map(float, line.split()) for line in data])

# Data normalization and split
scaler = MinMaxScaler()
X_data_normalized = scaler.fit_transform(np.array(x_data).reshape(-1, 1))
y_data_normalized = scaler.fit_transform(np.array(y_data).reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X_data_normalized, y_data_normalized, test_size=0.2,
                                                    random_state=42)

# Initializing and training a neural network
input_size = 1
hidden_size = 1000
output_size = 1
epochs = 30000
learning_rate = 0.0001
reg_lambda = 0.001
nn = NeuralNetworkRegression(input_size, hidden_size, output_size)
nn.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate, reg_lambda=reg_lambda)

# Predictions for training and test data
output_train = nn.forward(X_train)
output_test = nn.forward(X_test)

# Last MSE and R² values
final_mse = nn.history_mse[-1]
final_r2 = nn.history_r2[-1]

# Charts
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.scatter(range(len(nn.history_mse)), nn.history_mse, marker='.')
plt.xlabel('')
plt.ylabel('MSE')
plt.title(f'MSE vs. Epochs, Final MSE:{round(final_mse, 3)}')
plt.ylim(0, 0.2)

plt.subplot(2, 2, 2)
plt.scatter(range(len(nn.history_r2)), nn.history_r2, marker='.')
plt.xlabel('Epochs')
plt.ylabel('R^2')
plt.title(f'R^2 vs. Epochs, Final R^2:{round(final_r2, 2)}')
plt.ylim(0, 1)
plt.tight_layout()

plt.subplot(2, 2, 3)
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Test Data')
plt.scatter(X_train, output_train, color='green', label='Predictions', marker='s')
plt.scatter(X_test, output_test, color='green', label='Predictions', marker='x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Neurons={hidden_size}, Epochs={len(nn.history_r2)}, learn_r={learning_rate}, lambda={reg_lambda}')
plt.legend()

# Saving the chart
now = datetime.now()  # current date and time
nazwa_pliku = now.strftime(f'DataSet_{XX}_%Y%m%d_%H%M%S_neu_{hidden_size}_epch_{len(nn.history_r2)}.png')
plt.savefig(nazwa_pliku)
plt.show()