import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add

class ImageCaptioningModel:
    def __init__(self, vocab_size, max_length, embedding_dim=256, lstm_units=256):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        
    def build_model(self):
        # Image feature input
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(self.embedding_dim, activation='relu')(fe1)
        
        # Sequence input
        inputs2 = Input(shape=(self.max_length,))
        se1 = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        
        # LSTM layer
        se3 = LSTM(self.lstm_units)(se2)
        
        # Combine features
        decoder1 = Add()([fe2, se3])
        decoder2 = Dense(self.embedding_dim, activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)
        
        # Create model
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        return model 