import tensorflow as tf
from model import ImageCaptioningModel
from data_loader import DataLoader
from tensorflow.keras.utils import to_categorical
import os
import pickle

def train_model(features_path, captions_path, epochs=10, batch_size=64):
    # Initialize data loader
    data_loader = DataLoader(features_path, captions_path)
    
    # Load data
    features = data_loader.load_features()
    descriptions = data_loader.load_descriptions()
    
    # Create tokenizer and get vocabulary size
    tokenizer = data_loader.create_tokenizer()
    vocab_size = data_loader.vocab_size
    
    # Save tokenizer for later use
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Get maximum sequence length
    max_length = data_loader.get_max_length()
    
    # Create sequences
    X1, X2, y = data_loader.create_sequences()
    
    # Convert y to one-hot encoding
    y_one_hot = to_categorical(y, num_classes=vocab_size)
    
    # Create and compile model
    model_builder = ImageCaptioningModel(vocab_size, max_length)
    model = model_builder.build_model()
    
    # Train model without checkpointing
    history = model.fit(
        [X1, X2],
        y_one_hot,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2
    )
    
    # Save model directly
    model.save('image_captioning_model')
    
    return model, history

if __name__ == "__main__":
    train_model('features.pkl', 'captions.txt') 