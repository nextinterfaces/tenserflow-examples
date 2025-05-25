import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    return (x_train, y_train), (x_test, y_test)

# Create a simple feedforward neural network
def create_model():
    model = tf.keras.Sequential([
        # Flatten the 28x28 images into a 784-dimensional vector
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        
        # First hidden layer with 128 neurons and ReLU activation
        tf.keras.layers.Dense(128, activation='relu'),
        
        # Dropout layer to prevent overfitting
        tf.keras.layers.Dropout(0.2),
        
        # Second hidden layer
        tf.keras.layers.Dense(64, activation='relu'),
        
        # Output layer with 10 neurons (one for each digit)
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Create and compile the model
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    model.summary()
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Make some predictions
    predictions = model.predict(x_test[:5])
    print("\nPredictions for first 5 test images:")
    for i, pred in enumerate(predictions):
        print(f"Image {i+1}: Predicted digit {np.argmax(pred)} (Actual: {y_test[i]})")

if __name__ == "__main__":
    main() 