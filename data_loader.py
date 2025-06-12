import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataLoader:
    def __init__(self, features_path, captions_path):
        self.features_path = features_path
        self.captions_path = captions_path
        self.features = None
        self.descriptions = None
        self.tokenizer = None
        self.max_length = None
        self.vocab_size = None
        
    def load_features(self):
        with open(self.features_path, 'rb') as f:
            self.features = pickle.load(f)
        return self.features
    
    def load_descriptions(self):
        descriptions = {}
        with open(self.captions_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.lower().startswith("image"):
                    continue
                try:
                    image_id, image_desc = line.split(maxsplit=1)
                    image_id = image_id.split('.')[0]
                    caption = f"startseq {image_desc} endseq"
                    descriptions.setdefault(image_id, []).append(caption)
                except ValueError:
                    continue
        self.descriptions = descriptions
        return descriptions
    
    def create_tokenizer(self):
        all_captions = [caption for captions_list in self.descriptions.values() 
                       for caption in captions_list]
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(all_captions)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        return self.tokenizer
    
    def get_max_length(self):
        all_captions = [caption for captions_list in self.descriptions.values() 
                       for caption in captions_list]
        self.max_length = max(len(caption.split()) for caption in all_captions)
        return self.max_length
    
    def create_sequences(self):
        X1, X2, y = [], [], []
        
        for img_id, caption_list in self.descriptions.items():
            img_id_with_extension = f"{img_id}.jpg"
            
            if img_id_with_extension not in self.features:
                continue
                
            feature = self.features[img_id_with_extension]
            
            for desc in caption_list:
                seq = self.tokenizer.texts_to_sequences([desc])[0]
                
                if len(seq) == 0:
                    continue
                    
                if len(seq) == 1:
                    X1.append(feature)
                    X2.append(pad_sequences([[seq[0]]], maxlen=self.max_length)[0])
                    y.append(seq[0])
                else:
                    for i in range(1, len(seq)):
                        in_seq, out_seq = seq[:i], seq[i]
                        in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                        X1.append(feature)
                        X2.append(in_seq)
                        y.append(out_seq)
                        
        return np.array(X1), np.array(X2), np.array(y) 