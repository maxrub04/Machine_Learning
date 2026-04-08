import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def load_data(filename):
    data = np.loadtxt(filename)
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    return train_test_split(X, y, test_size=0.2, random_state=42)

X1_train, X1_test, y1_train, y1_test = load_data('dane1.txt')
X2_train, X2_test, y2_train, y2_test = load_data('dane2.txt')
X3_train, X3_test, y3_train, y3_test = load_data('dane3.txt')


def train_models(X_train, y_train, degree):
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    poly_model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    poly_model.fit(X_train, y_train)
    return lin_model, poly_model

lin_model1, poly_model1 = train_models(X1_train, y1_train, degree=2)
lin_model2, poly_model2 = train_models(X2_train, y2_train, degree=3)
lin_model3, poly_model3 = train_models(X3_train, y3_train, degree=8)


def calculate_r2_scores(lin_model, poly_model, X_test, y_test):
    r2_lin = r2_score(y_test, lin_model.predict(X_test))
    r2_poly = r2_score(y_test, poly_model.predict(X_test))
    return r2_lin, r2_poly

r2_lin1, r2_poly1 = calculate_r2_scores(lin_model1, poly_model1, X1_test, y1_test)
r2_lin2, r2_poly2 = calculate_r2_scores(lin_model2, poly_model2, X2_test, y2_test)
r2_lin3, r2_poly3 = calculate_r2_scores(lin_model3, poly_model3, X3_test, y3_test)


def plot_data(ax, X, y, lin_model, poly_model, r2_lin, r2_poly, title):
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    ax.scatter(X, y, s=10, label='Data')
    ax.plot(X_plot, lin_model.predict(X_plot), label=f'Linear R²={r2_lin:.3f}')
    ax.plot(X_plot, poly_model.predict(X_plot), label=f'Poly R²={r2_poly:.3f}')
    ax.legend()
    ax.set_title(title)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plot_data(axes[0], X1_train, y1_train, lin_model1, poly_model1, r2_lin1, r2_poly1, 'dane1')
plot_data(axes[1], X2_train, y2_train, lin_model2, poly_model2, r2_lin2, r2_poly2, 'dane2')
plot_data(axes[2], X3_train, y3_train, lin_model3, poly_model3, r2_lin3, r2_poly3, 'dane3')

plt.tight_layout()
plt.show()

"""
A simple straight line is too stiff to accurately track data that naturally curves and waves.
However, a polynomial model can easily bend to follow those changing shapes, making it the much better choice here.
"""