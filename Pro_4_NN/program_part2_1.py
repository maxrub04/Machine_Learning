import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import Input, Model, layers, callbacks, optimizers


DATASETS = [
    # dane1  - parabola in [-2, 2]
    dict(file=1,  hidden1=12, hidden2=6,  lr=0.01,  epochs=1500, batch=8,  patience=150, seed=42),
    # dane2  - cubic in [-3, 3]
    dict(file=2,  hidden1=16, hidden2=8,  lr=0.01,  epochs=1500, batch=8,  patience=150, seed=42),
    # dane3  - noisy double-hump in [-3, 3]
    dict(file=3,  hidden1=24, hidden2=12, lr=0.01,  epochs=2000, batch=8,  patience=200, seed=13),
    # dane6  - quartic-like with sharp tails in [-4, 4]
    dict(file=6,  hidden1=24, hidden2=12, lr=0.01,  epochs=2000, batch=16, patience=200, seed=42),
    # dane8  - smooth bell-shaped curve in [-3, 3]
    dict(file=8,  hidden1=16, hidden2=8,  lr=0.01,  epochs=1500, batch=8,  patience=150, seed=42),
    # dane10 - nearly linear trend in [-2, 5]
    dict(file=10, hidden1=8,  hidden2=4,  lr=0.01,  epochs=1500, batch=8,  patience=150, seed=42),
    # dane12 - exponential growth in [-2, 3]
    dict(file=12, hidden1=16, hidden2=8,  lr=0.005, epochs=2000, batch=8,  patience=200, seed=42),
    # dane14 - noisy negative cubic in [-3, 3]
    dict(file=14, hidden1=20, hidden2=10, lr=0.01,  epochs=2000, batch=16, patience=200, seed=42),
]


def build_model(hidden1, hidden2, lr):
    inp = Input(shape=(1,), name='x')
    h = layers.Dense(hidden1, activation='tanh', name='hidden_1')(inp)
    h = layers.Dense(hidden2, activation='tanh', name='hidden_2')(h)
    out = layers.Dense(1, activation='linear', name='y')(h)
    model = Model(inputs=inp, outputs=out, name=f'reg_{hidden1}_{hidden2}')
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='mse', metrics=['mae'])
    return model


for cfg in DATASETS:
    XX = cfg['file']

    with open(f"./dane{XX}.txt", "r") as file:
        data = file.readlines()
    x_data, y_data = zip(*[map(float, line.split()) for line in data])

    # Data normalization and split
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_data_normalized = scaler_x.fit_transform(np.array(x_data).reshape(-1, 1))
    y_data_normalized = scaler_y.fit_transform(np.array(y_data).reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(
        X_data_normalized, y_data_normalized, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # Build, train
    tf.keras.utils.set_random_seed(cfg['seed'])
    model = build_model(cfg['hidden1'], cfg['hidden2'], cfg['lr'])
    n_params = model.count_params()

    early = callbacks.EarlyStopping(monitor='val_loss', patience=cfg['patience'],
                                    min_delta=1e-5, restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=cfg['epochs'], batch_size=cfg['batch'],
                        verbose=0, callbacks=[early])

    # Predictions for training and test data
    output_train = model.predict(X_train, verbose=0)
    output_test = model.predict(X_test, verbose=0)

    # Final loss / R^2 on test data
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    test_mse = mean_squared_error(y_test, output_test)
    test_r2 = r2_score(y_test, output_test)
    epochs_run = len(history.history['loss'])

    print(f"dane{XX:>2}.txt | params={n_params:>3} | epochs={epochs_run:>3} "
          f"| train MSE={final_loss:.4f} | val MSE={final_val_loss:.4f} "
          f"| test MSE={test_mse:.4f} | test R^2={test_r2:.3f}")

    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.title(f'Loss vs. Epochs, Final train MSE: {round(final_loss, 4)}')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history.history['mae'], label='train MAE')
    plt.plot(history.history['val_mae'], label='val MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title(f'MAE vs. Epochs, Test R^2: {round(test_r2, 3)}')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.scatter(X_train, y_train, color='blue', label='Training Data', s=20)
    plt.scatter(X_test, y_test, color='red', label='Test Data', s=20)
    plt.scatter(X_train, output_train, color='green', label='Predictions (train)',
                marker='s', s=20)
    plt.scatter(X_test, output_test, color='lime', label='Predictions (test)',
                marker='x', s=40)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"Params={n_params}, Epochs={epochs_run}, "
              f"learn_r={cfg['lr']}, batch={cfg['batch']}")
    plt.legend(fontsize=10)

    plt.subplot(2, 2, 4)
    order = np.argsort(X_test.ravel())
    plt.plot(X_test.ravel()[order], y_test.ravel()[order],
             'o-', color='red', label='Test target', markersize=4)
    plt.plot(X_test.ravel()[order], output_test.ravel()[order],
             's-', color='green', label='Test prediction', markersize=4)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Test fit, MSE={round(test_mse, 4)}')
    plt.legend()

    plt.suptitle(f"dane{XX}.txt — hidden=({cfg['hidden1']}, {cfg['hidden2']}), "
                 f"activation=tanh", fontsize=15)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
