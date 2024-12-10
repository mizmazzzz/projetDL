import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import cv2
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
import tensorflow as tf
import json

# Load the model and required components
# Reference the existing model loading code from the notebook
# Custom object scope to handle the old optimizer configuration
with tf.keras.utils.custom_object_scope({'Adam': tf.keras.optimizers.legacy.Adam}):
    model = load_model('./model_weights/model_9.h5', compile=False)
# Recompile the model with the current optimizer version
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Load word mappings
# Load the word-to-index and index-to-word mappings
with open('./model_weights/word_to_idx.json', 'r') as f:
    word_to_idx = json.load(f)
with open('./model_weights/idx_to_word.json', 'r') as f:
    idx_to_word = json.load(f)

# Debug print
print("Vocabulary size:", len(word_to_idx))
print("Sample words:", list(word_to_idx.keys())[:10])
print("Sample indices:", list(idx_to_word.keys())[:10])

# Initialize ResNet50 with global average pooling to get a 2048-dimensional vector
model_new = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def encode_image(img_path):
    img = preprocess_img(img_path)
    feature_vector = model_new.predict(img)
    feature_vector = feature_vector.reshape((2048,))
    return feature_vector

def predict_caption_for_image(image_path):
    try:
        # Encode the image
        photo = encode_image(image_path)
        photo = photo.reshape((1, 2048))
        
        # Generate caption
        in_text = "startseq"
        print("\nStarting caption generation...")
        print(f"Initial text: {in_text}")
        
        for i in range(35):  # max_len from training
            sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
            print(f"\nStep {i}:")
            print(f"Current sequence: {sequence}")
            
            sequence = pad_sequences([sequence], maxlen=35, padding='post')
            
            ypred = model.predict([photo, sequence])
            ypred = ypred.argmax()
            print(f"Raw prediction: {ypred}")
            
            word = idx_to_word.get(str(ypred), '')
            print(f"Mapped word: {word}")
            
            if word == "endseq" or word == '':
                print(f"Stopping on word: {word}")
                break
                
            in_text += ' ' + word
            print(f"Current caption: {in_text}")
        
        # Clean up the caption
        final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
        print(f"\nFinal caption: {final_caption}")
        
        if not final_caption:
            return "Unable to generate caption"
        return final_caption
    except Exception as e:
        print(f"Error in caption generation: {str(e)}")
        return f"Error generating caption: {str(e)}"