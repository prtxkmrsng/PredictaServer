import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def generate_server_metrics_data():
    """Generates continuous time-series representing noisy server utilization spikes."""
    np.random.seed(42)
    timesteps = 2000
    time_axis = np.linspace(0, 50, timesteps)
    # Sine wave base layout with added random spikes representing peak usage hours
    base_cpu = np.abs(np.sin(time_axis) * 60 + 20 + np.random.normal(0, 5, timesteps))
    base_cpu = np.clip(base_cpu, 10, 98) / 100.0 # Normalize scaling bound
    return base_cpu.reshape(-1, 1)

def build_sequences(data, window_size=10):
    """Slices structural data into rolling feature windows and target labels."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_network():
    print("[Deep Learning] Synthesizing sequential server log metrics...")
    raw_data = generate_server_metrics_data()
    
    # Slice arrays into a 10-step rolling lookback matrix window
    WINDOW_SIZE = 10
    X, y = build_sequences(raw_data, window_size=WINDOW_SIZE)
    
    # Train-test partition split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Build Keras LSTM Topology Sequential Stack
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
        Dropout(0.2), # Dropout regularization layers mitigate network co-adaptation
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1) # Linear output predicting next step utilization percentage
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("[Deep Learning] Training LSTM Network on native CPU/GPU hardware...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=32,
        verbose=1
    )
    
    # Serialize complete model structure directly to local disk
    model.save("server_lstm_model.keras")
    print("[Deep Learning] Successfully saved network checkpoint to 'server_lstm_model.keras'")

if __name__ == "__main__":
    train_network()