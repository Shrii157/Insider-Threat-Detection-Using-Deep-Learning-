import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
#from keras.layers import Embedding, LSTM, Dense

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Embedding, Conv1D, MaxPool1D, Dropout, SimpleRNN, LSTM
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing import sequence
import pickle


# Step 1: Read the CSV file
dfd = pd.read_csv('final.csv')

# Step 3: Prepare features and target
X = dfd.drop(columns=['label'])
y = dfd['label']

# Step 4: Concatenate text columns
text_data = X["id"] + " " + X["datel"] + " " + X["user"] + " " + X["pc"] + " " + X["activity_x"] + " " + X["filename"] + " " + X["dated"] + " " + X["activity_y"]

# Step 5: Tokenization and Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
encoded_text = tokenizer.texts_to_sequences(text_data)
max_length = max([len(seq) for seq in encoded_text])
padded_text = pad_sequences(encoded_text, maxlen=max_length, padding='post')
print(padded_text)

# Step 6: Encoding labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y)
# Reshape the labels
encoded_labels = encoded_labels.reshape(-1, 1)
print(encoded_labels)

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_text, encoded_labels, test_size=0.2, random_state=0)

# Step 8: Define LSTM model
vocab_size = len(tokenizer.word_index)
embedding_dim = 100

model = Sequential()
model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_length))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

# Step 9: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Step 10: Train the model
lstmhistory = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, verbose=2)

# Step 11: Save the model
model.save("train.h5")
print("lstm train completed")
f = open('lstmhistory.pckl', 'wb')
pickle.dump(lstmhistory.history, f)
f.close()



