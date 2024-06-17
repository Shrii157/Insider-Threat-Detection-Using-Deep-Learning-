# Load necessary libraries
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import pickle

# Step 1: Read the new data CSV file
X_new = pd.read_csv('TH1.csv')  # Change 'new_data.csv' to the path of your new data file

# Step 2: Prepare features
# Assuming 'new_data' has the same structure as the training data
#X_new = new_data.drop(columns=['label'])
text_data_new = X_new["id"] + " " + X_new["datel"] + " " + X_new["user"] + " " + X_new["pc"] + " " + X_new["activity_x"] + " " + X_new["filename"] + " " + X_new["dated"] + " " + X_new["activity_y"]

# Step 3: Load the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data_new)

# Step 4: Tokenization and Padding
encoded_text_new = tokenizer.texts_to_sequences(text_data_new)
max_length = max([len(seq) for seq in encoded_text_new])
padded_text_new = pad_sequences(encoded_text_new, maxlen=22, padding='post')

# Step 5: Load the trained model
model = load_model("train.h5")  # Change "train.h5" to the path of your trained model file

# Step 6: Make predictions
predictions = model.predict(padded_text_new)

threshold = 0.7
predicted_labels = (predictions >= threshold).astype(int)

# Step 8: Output predictions
print(predictions)
print(predicted_labels)

f = open('lstmhistory.pckl', 'rb')
history = pickle.load(f)
f.close()
print(history)
plt.figure()
plt.plot(history['accuracy'],label='ACCURACY')
plt.title('ACCURACY')
plt.xlabel('LABELS')
plt.ylabel('ACCURACY')
plt.legend()
plt.show()




