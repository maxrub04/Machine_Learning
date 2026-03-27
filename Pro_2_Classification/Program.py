import numpy as np
import matplotlib.pyplot as plt
from typing import *
from sklearn.model_selection import train_test_split


points_per_class = 50
centers = np.array([[1, 1], [5, 5], [1, 5], [5, 1]])
x = []
y = []

for i, center in enumerate(centers):
    stddev = 1.1
    points = np.random.randn(points_per_class, 2) * stddev + center
    x.append(points)
    y.extend([i] * points_per_class)

x = np.vstack(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

_, ax = plt.subplots()
for i in np.unique(y):
    ax.scatter(x[y == i, 0], x[y == i, 1], label=f'Class {i}')
ax.legend()
ax.set_title("Generated Data")
plt.show()


class Perceptron:
    def __init__(self, size: int):
        self.weights = np.random.rand(size) * 2 - 1
        self.bias = np.random.rand()
        self.lr = 0.1
        self.epochs = 10
        self.activate = lambda x: 1 if x > 0 else 0

    def _forward(self, xs: np.ndarray) -> Literal[0, 1]:
        state = np.dot(self.weights, xs) + self.bias
        return self.activate(state)

    def _backward(self, xs: np.ndarray, error: Literal[-1, 0, 1]) -> None:
        self.weights += self.lr * error * xs
        self.bias += self.lr * error

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        for _ in range(self.epochs):
            for x, y in zip(x_train, y_train):
                prediction = self._forward(x)
                error = y - prediction
                self._backward(x, error)

    def eval(self, x_test: np.ndarray) -> List[int]:
        return [self._forward(x) for x in x_test]


class OneVsOne:
    def __init__(self, input_dim: int, num_classes: int):
        self.models = {}
        self.input_dim = input_dim
        self.num_classes = num_classes

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                x_pair = x_train[(y_train == i) | (y_train == j)]
                y_pair = y_train[(y_train == i) | (y_train == j)]
                y_pair = np.where(y_pair == i, 0, 1)

                model = Perceptron(size=self.input_dim)
                model.train(x_pair, y_pair)
                self.models[(i, j)] = model

    def predict(self, x: np.ndarray) -> int:
        votes = np.zeros(self.num_classes)
        for (i, j), model in self.models.items():
            prediction = model._forward(x)
            if prediction == 0:
                votes[i] += 1
            else:
                votes[j] += 1
        return np.argmax(votes)

    def eval(self, x_test: np.ndarray) -> np.ndarray:
        return np.array([self.predict(x) for x in x_test])

class OneVsRest:
    def __init__(self, input_dim: int, num_classes: int):
        self.models = [Perceptron(size=input_dim) for _ in range(num_classes)]

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        for i, model in enumerate(self.models):
            binary_y = (y_train == i).astype(int)
            model.train(x_train, binary_y)

    def predict(self, x: np.ndarray) -> int:
        scores = [model._forward(x) for model in self.models]
        return np.argmax(scores)

    def eval(self, x_test: np.ndarray) -> np.ndarray:
        return np.array([self.predict(x) for x in x_test])



class LogisticRegression:
    def __init__(self, input_dim, num_classes):
        self.weights = np.random.randn(num_classes, input_dim) * 0.01
        self.biases = np.zeros(num_classes)
        self.lr = 0.1

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / exp_z.sum(axis=0)

    def predict_probs(self, x):
        z = np.dot(self.weights, x) + self.biases
        return self.softmax(z)

    def predict(self, x):
        probs = self.predict_probs(x)
        return np.argmax(probs, axis=0)

    def train(self, x_train, y_train, epochs=100):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                y_one_hot = np.zeros(self.weights.shape[0])
                y_one_hot[y] = 1

                z = np.dot(self.weights, x) + self.biases
                probs = self.softmax(z)

                grad_loss = probs - y_one_hot

                self.weights -= self.lr * np.outer(grad_loss, x)
                self.biases -= self.lr * grad_loss

    def eval(self, x_test):
        predictions = []
        for x in x_test:
            predictions.append(self.predict(x))
        return np.array(predictions)


num_classes = len(np.unique(y_train))
input_dim = x_train.shape[1]

# One-vs-One
ovo_classifier = OneVsOne(input_dim=input_dim, num_classes=num_classes)
ovo_classifier.train(x_train, y_train)
y_pred_ovo = ovo_classifier.eval(x_test)
accuracy_ovo = np.mean(y_pred_ovo == y_test)
print(f"One-vs-One Accuracy: {accuracy_ovo:.2f}")

# One-vs-Rest
ovr_classifier = OneVsRest(input_dim=input_dim, num_classes=num_classes)
ovr_classifier.train(x_train, y_train)
y_pred_ovr = ovr_classifier.eval(x_test)
accuracy_ovr = np.mean(y_pred_ovr == y_test)
print(f"One-vs-Rest Accuracy: {accuracy_ovr:.2f}")

# Logistic Regression
logreg = LogisticRegression(input_dim=input_dim, num_classes=num_classes)
logreg.train(x_train, y_train, epochs=100)
y_pred_logreg = logreg.eval(x_test)
accuracy_logreg = np.mean(y_pred_logreg == y_test)
print(f"Logistic Regression Accuracy: {accuracy_logreg:.2f}")


_, ax = plt.subplots(1, 3, figsize=(18, 6))

# One-vs-One Results
ax[0].set_title(f"One-vs-One Accuracy: {accuracy_ovo:.2f}")
for i in np.unique(y_test):
    ax[0].scatter(x_test[y_test == i, 0], x_test[y_test == i, 1], label=f'Class {i}')
ax[0].legend()

ax[1].set_title(f"One-vs-Rest Accuracy: {accuracy_ovr:.2f}")
for i in np.unique(y_test):
    ax[1].scatter(x_test[y_test == i, 0], x_test[y_test == i, 1], label=f'Class {i}')
ax[1].legend()

ax[2].set_title(f"Logistic Regression Accuracy: {accuracy_logreg:.2f}")
for i in np.unique(y_test):
    ax[2].scatter(x_test[y_test == i, 0], x_test[y_test == i, 1], label=f'Class {i}')
ax[2].legend()

plt.tight_layout()
plt.show()
