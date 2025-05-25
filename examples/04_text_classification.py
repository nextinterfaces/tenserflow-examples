import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

def load_imdb_data():
    """Load IMDB movie reviews dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=10000
    )
    
    # Get the word index dictionary
    word_index = tf.keras.datasets.imdb.get_word_index()
    
    # Create a reverse word index
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()]
    )
    
    # Decode the reviews
    decode = lambda review: ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in review]
    )
    
    return (x_train, y_train), (x_test, y_test), decode

def prepare_data(x_train, x_test, maxlen=200):
    """Pad sequences to the same length."""
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    return x_train, x_test

def create_model(vocab_size, maxlen):
    """Create a text classification model with embedding and LSTM layers."""
    model = tf.keras.Sequential([
        # Embedding layer
        tf.keras.layers.Embedding(vocab_size, 100, input_length=maxlen),
        
        # Bidirectional LSTM layers
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32)
        ),
        
        # Dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def plot_training_history(history):
    """Plot training history."""
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

def predict_sentiment(model, text, tokenizer, maxlen=200):
    """Predict sentiment for a given text."""
    # Tokenize and pad the text
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=maxlen)
    
    # Make prediction
    prediction = model.predict(padded)[0][0]
    return "Positive" if prediction > 0.5 else "Negative", prediction

def main():
    # Parameters
    VOCAB_SIZE = 10000
    MAXLEN = 200
    
    # Load and prepare data
    (x_train, y_train), (x_test, y_test), decode = load_imdb_data()
    x_train, x_test = prepare_data(x_train, x_test, MAXLEN)
    
    # Create and compile model
    model = create_model(VOCAB_SIZE, MAXLEN)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
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
    
    # Example predictions
    sample_reviews = [
        "This movie was fantastic! The acting was great and the plot was engaging.",
        "I really didn't like this film. The story was boring and predictable.",
        "An average movie, neither good nor bad. Some parts were interesting."
    ]
    
    # Create a new tokenizer for the sample reviews
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(sample_reviews)
    
    print("\nExample predictions:")
    for review in sample_reviews:
        sentiment, confidence = predict_sentiment(model, review, tokenizer, MAXLEN)
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    main() 