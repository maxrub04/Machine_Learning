import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Input, Model, layers, callbacks, optimizers


tf.keras.utils.set_random_seed(42)


X, y = make_moons(n_samples=2000, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                  random_state=42, stratify=y_train)

inp = Input(shape=(2,), name='features')
h = layers.Dense(16, activation='relu', name='hidden_1')(inp)
h = layers.Dense(8, activation='relu', name='hidden_2')(h)
out = layers.Dense(1, activation='sigmoid', name='proba')(h)
model = Model(inputs=inp, outputs=out, name='moons_classifier')
model.compile(optimizer=optimizers.Adam(learning_rate=0.01),
              loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early = callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                restore_best_weights=True)
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=300, batch_size=32,
                    verbose=0, callbacks=[early])

epochs_run = len(history.history['loss'])
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
test_err = 1.0 - test_acc

print(f"Epochs run: {epochs_run}")
print(f"Test loss (BCE): {test_loss:.4f}")
print(f"Test accuracy : {test_acc:.4f}")
print(f"Test error    : {test_err:.4f}")

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
proba = model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0).reshape(xx.shape)

plt.rcParams.update({'font.size': 14})
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

axes[0].plot(history.history['loss'], label='train loss')
axes[0].plot(history.history['val_loss'], label='val loss')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Binary cross-entropy')
axes[0].set_title(f'Loss vs. Epochs, Final test BCE: {round(test_loss, 4)}')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='train acc')
axes[1].plot(history.history['val_accuracy'], label='val acc')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].set_title(f'Accuracy vs. Epochs, Test acc: {round(test_acc, 3)}')
axes[1].legend()

axes[2].contourf(xx, yy, proba, levels=20, cmap='RdBu', alpha=0.6)
axes[2].contour(xx, yy, proba, levels=[0.5], colors='black', linewidths=1.5)
axes[2].scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                cmap='RdBu_r', edgecolors='k', s=25)
axes[2].set_xlabel('x1')
axes[2].set_ylabel('x2')
axes[2].set_title(f'Decision boundary on test data\n'
                  f'Epochs={epochs_run}, test acc={round(test_acc, 3)}, '
                  f'test err={round(test_err, 3)}')

plt.suptitle('make_moons classifier — Keras Functional API '
             '(2 -> 16 relu -> 8 relu -> 1 sigmoid)', fontsize=15)
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()
