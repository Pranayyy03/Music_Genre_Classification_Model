from google.colab import drive
drive.mount('/content/drive')
import os

# Define the dataset ZIP path
zip_path = "/content/drive/MyDrive/Music_Genre_Classification/Data/archive.zip"

# Check if the file exists
if os.path.exists(zip_path):
    print("✅ File exists! Ready to extract.")
else:
    print("❌ File NOT found! Check your Google Drive path.")
import zipfile

# Define extraction path
extract_to = "/content/drive/MyDrive/Music_Genre_Classification/Data/"

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("✅ Dataset extracted successfully!")
data_folder = "/content/drive/MyDrive/Music_Genre_Classification/Data"
print(os.listdir(data_folder))
import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from sklearn import metrics
import time
# Define dataset paths
audio_dataset_path = "/content/drive/MyDrive/Music_Genre_Classification/Data/Data/genres_original"
metadata_path = "/content/drive/MyDrive/Music_Genre_Classification/Data/Data/features_30_sec.csv"

# Load the CSV metadata file
metadata = pd.read_csv(metadata_path)

# Display metadata preview
print(metadata.head())
# Define dataset paths
audio_dataset_path = "/content/drive/MyDrive/Music_Genre_Classification/Data/Data/genres_original"
metadata_path = "/content/drive/MyDrive/Music_Genre_Classification/Data/Data/features_30_sec.csv"

# Load the CSV metadata file
metadata = pd.read_csv(metadata_path)

# Display metadata preview
print(metadata.head())
# Feature Extraction Function
def features_extractor(file):
    try:
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')  # Load audio file
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)  # Extract MFCC
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)  # Compute mean across time axis
        return mfccs_scaled_features
    except Exception as e:
        print(f"Error extracting features from {file}: {e}")
        return None
extracted_features = []

for index_num, row in tqdm(metadata.iterrows()):
    try:
        final_class_label = row["label"]
        file_name = os.path.join(os.path.abspath(audio_dataset_path), final_class_label + '/', str(row["filename"]))
        data = features_extractor(file_name)
        if data is not None:
            extracted_features.append([data, final_class_label])
    except Exception as e:
        print(f"Error: {e}")
        continue
# Convert extracted features list into a Pandas DataFrame
extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])

# Display first few rows
print(extracted_features_df.head())
# Convert feature column to NumPy array
X = np.array(extracted_features_df['feature'].tolist())  # Independent variables (Features)

# Convert class labels to categorical
y = np.array(extracted_features_df['class'].tolist())  # Dependent variable (Genre labels)

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
y = to_categorical(label_encoder.fit_transform(y))  # One-hot encode labels

# Display dataset shape
print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
# Split dataset into training and testing sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Display dataset split info
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Testing set: {X_test.shape}, {y_test.shape}")
# Number of output labels (genres)
num_labels = y.shape[1]

# Define a deeper Neural Network model
model = Sequential()

# Input layer
model.add(Dense(1024, input_shape=(40,), activation="relu"))
model.add(Dropout(0.3))

# Hidden layers (more layers added for better performance)
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.3))

# Output layer (softmax for classification)
model.add(Dense(num_labels, activation="softmax"))

# Display model summary
model.summary()
# Compile model with categorical crossentropy loss
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))

# Create a directory for saving models if it doesn't exist
save_dir = "saved_models"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Generate a timestamp for unique filenames
current_time = time.strftime("%H-%M-%S", time.localtime())

# Define checkpoint path (fixing the extension issue)
checkpoint_path = os.path.join(save_dir, f"audio_classification_{current_time}.keras")

checkpointer = ModelCheckpoint(filepath=checkpoint_path,
                               verbose=1, save_best_only=True)

# Train the model
start = datetime.now()

history = model.fit(X_train, y_train,
                    batch_size=num_batch_size,
                    epochs=num_epochs,
                    validation_data=(X_test, y_test),
                    callbacks=[checkpointer])

# Calculate training duration
duration = datetime.now() - start
print("Training completed in time: ", duration)
# Evaluate model on test data
evaluation = model.evaluate(X_test, y_test, verbose=1)

print("Test Loss: ", evaluation[0])
print("Test Accuracy: ", evaluation[1])
# Predict class labels for test data
y_pred = model.predict(X_test)

# Convert predictions to class labels
predicted_classes = np.argmax(y_pred, axis=1)
actual_classes = np.argmax(y_test, axis=1)

print("Predicted Classes: ", predicted_classes)
print("Actual Classes: ", actual_classes)
# Load an audio file (replace with actual file path)
filename = "/content/drive/MyDrive/Music_Genre_Classification/Data/Data/generes_original/metal/metal.00006.wav"

audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

# Reshape for prediction
mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

# Predict genre
predicted_label = np.argmax(model.predict(mfccs_scaled_features), axis=1)
prediction_class = label_encoder.inverse_transform(predicted_label)

print("Predicted Genre: ", prediction_class)
