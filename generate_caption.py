import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
import matplotlib.pyplot as plt

class CaptionGenerator:
    def __init__(self, model_path, tokenizer_path, max_length):
        self.model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.max_length = max_length
        self.word_to_index = self.tokenizer.word_index
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        
    def extract_features(self, image_path):
        # Load InceptionV3 model
        model = InceptionV3(weights='imagenet')
        # Remove the last layer
        model = tf.keras.Model(model.input, model.layers[-2].output)
        
        # Load and preprocess image
        image = load_img(image_path, target_size=(299, 299))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        
        # Extract features
        features = model.predict(image, verbose=0)
        return features
        
    def generate_caption(self, image_path):
        # Extract features
        features = self.extract_features(image_path)
        
        # Initialize caption
        caption = 'startseq'
        
        # Generate caption
        for i in range(self.max_length):
            # Convert caption to sequence
            sequence = self.tokenizer.texts_to_sequences([caption])[0]
            # Pad sequence
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            # Predict next word
            yhat = self.model.predict([features, sequence], verbose=0)
            # Get word with highest probability
            yhat = np.argmax(yhat)
            # Convert index to word
            word = self.index_to_word.get(yhat, '')
            # Break if endseq is predicted
            if word == 'endseq':
                break
            # Add word to caption
            caption += ' ' + word
            
        # Remove startseq
        caption = caption.replace('startseq ', '')
        return caption

def display_image_with_caption(image_path, caption):
    # Load and display image
    img = load_img(image_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption)
    plt.show()

def main():
    # Initialize caption generator
    generator = CaptionGenerator(
        model_path='image_captioning_model',  # Changed path
        tokenizer_path='tokenizer.pkl',
        max_length=39
    )
    
    # Generate caption for test image
    image_path = 'Test_image.png'
    caption = generator.generate_caption(image_path)
    print(f"Generated caption: {caption}")
    
    # Display image with caption
    display_image_with_caption(image_path, caption)

if __name__ == "__main__":
    main() 