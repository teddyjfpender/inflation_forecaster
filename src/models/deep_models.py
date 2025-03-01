import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

def create_lstm_model(input_shape):
    """
    Create and compile an LSTM model.

    :param input_shape: Tuple (timesteps, features)
    :return: Compiled Keras model.
    """
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_lstm_data(df, target_column='inflation_yoy', timesteps=12):
    """
    Prepare 3D data (samples, timesteps, features) for LSTM training.

    :param df: DataFrame containing features and target.
    :param target_column: Name of the target column.
    :param timesteps: Number of past time steps to use.
    :return: (X, y) arrays for LSTM training.
    """
    X, y = [], []
    data = df.copy()
    target = data[target_column].values
    features = data.drop(columns=[target_column]).values
    
    for i in range(timesteps, len(data)):
        X.append(features[i-timesteps:i])
        y.append(target[i])
    return np.array(X), np.array(y)

def train_lstm_model(X_train, y_train, X_val, y_val, timesteps=12, epochs=50, batch_size=16):
    """
    Train the LSTM model.

    :param X_train: Training features (3D array).
    :param y_train: Training targets.
    :param X_val: Validation features.
    :param y_val: Validation targets.
    :param timesteps: Number of time steps.
    :param epochs: Training epochs.
    :param batch_size: Batch size.
    :return: (trained model, training history)
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        callbacks=[early_stop])
    return model, history

def evaluate_lstm(model, X_test, y_test):
    """
    Evaluate the LSTM model using RMSE.

    :param model: Trained LSTM model.
    :param X_test: Test features.
    :param y_test: Test targets.
    :return: (rmse, predictions)
    """
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse, preds
