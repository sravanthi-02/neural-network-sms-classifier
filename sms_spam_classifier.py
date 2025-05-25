
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# Load data (assuming train_data and test_data are provided in the environment)
# For example purposes; in actual notebook, this might be pre-loaded.
# train_data = pd.read_csv('train.csv')
# test_data = pd.read_csv('test.csv')

# Encode labels
le = LabelEncoder()
train_labels = le.fit_transform(train_data['label'])
test_labels = le.transform(test_data['label'])

# Tokenize text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['message'])

train_sequences = tokenizer.texts_to_sequences(train_data['message'])
test_sequences = tokenizer.texts_to_sequences(test_data['message'])

# Pad sequences
max_length = 100
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# Build the model
model = Sequential([
    Embedding(10000, 16, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_padded, train_labels, epochs=10, validation_data=(test_padded, test_labels))

# Define the predict_message function
def predict_message(msg):
    seq = tokenizer.texts_to_sequences([msg])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    pred = model.predict(padded)[0][0]
    label = "spam" if pred > 0.5 else "ham"
    return [float(pred), label]
