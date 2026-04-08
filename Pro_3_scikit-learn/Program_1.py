import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


X, y = make_moons(n_samples=10000, noise=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression()
svm = SVC()
rf = RandomForestClassifier()

voting = VotingClassifier(estimators=[
    ('lr', log_reg),
    ('svm', svm),
    ('rf', rf)
])

for model in [log_reg, svm, rf, voting]:
    model.fit(X_train,y_train)

for name, model in [('LogReg', log_reg), ('SVM', svm), ('RF', rf), ('Voting', voting)]:
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"{name}: train={train_acc:.2f}, test={test_acc:.2f}")

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

models = [
    ('LogReg', log_reg),
    ('SVM', svm),
    ('RandomForest', rf),
    ('Voting', voting)
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, (name, model) in zip(axes, models):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=5)

    test_acc = model.score(X_test, y_test)
    ax.set_title(f"{name}\ntest={test_acc:.2f}")

plt.tight_layout()
plt.show()

"""
RandomForest shows clear overfitting (train=1.00, test=0.83), 
while LogisticRegression underfits due to its linear decision boundary. 
The VotingClassifier achieves the best test accuracy (0.84) by combining all three models, 
compensating for individual weaknesses.
"""