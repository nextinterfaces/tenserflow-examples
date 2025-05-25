import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def generate_time_series(samples=1000):
    """Generate a synthetic time series with multiple seasonal patterns."""
    time = np.arange(samples)
    # Generate a time series with multiple components
    series = (0.5 * np.sin(time/10.0) +  # Slow wave
              0.2 * np.sin(time/2.0) +    # Fast wave
              0.1 * np.random.randn(samples))  # Noise
    return series

def prepare_time_series_data(data, lookback=60, forecast=1, split_ratio=0.8):
    """Prepare time series data for training and testing."""
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - lookback - forecast + 1):
        X.append(scaled_data[i:(i + lookback)])
        y.append(scaled_data[i + lookback:i + lookback + forecast])
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test sets
    train_size = int(len(X) * split_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test, scaler

def create_lstm_model(lookback):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(30, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    return model

def plot_predictions(actual, predicted, title="Time Series Prediction"):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predicted, label='Predicted', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Parameters
    LOOKBACK = 60  # Number of time steps to look back
    FORECAST = 1   # Number of time steps to predict
    
    # Generate synthetic time series
    series = generate_time_series()
    
    # Prepare data
    X_train, y_train, X_test, y_test, scaler = prepare_time_series_data(
        series, LOOKBACK, FORECAST
    )
    
    # Create and compile model
    model = create_lstm_model(LOOKBACK)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Display model summary
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform predictions
    train_predict = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform(y_train)
    test_predict = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform(y_test)
    
    # Calculate RMSE
    train_rmse = np.sqrt(np.mean((y_train_inv - train_predict) ** 2))
    test_rmse = np.sqrt(np.mean((y_test_inv - test_predict) ** 2))
    print(f'\nTrain RMSE: {train_rmse:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')
    
    # Plot training predictions
    plot_predictions(
        y_train_inv.flatten(),
        train_predict.flatten(),
        "Training Data: Actual vs Predicted"
    )
    
    # Plot test predictions
    plot_predictions(
        y_test_inv.flatten(),
        test_predict.flatten(),
        "Test Data: Actual vs Predicted"
    )

if __name__ == "__main__":
    main() 