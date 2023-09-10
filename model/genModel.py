import os
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

dataset_dir = './data/processed_dataset'
num_classes = 2

def preprocess_frame(frame):
    frame = frame.astype('float32') / 255.0

    return frame

def loadframes(dataset_dir):

    temp_data = []
    temp_labels = []

    action_labels = os.listdir(dataset_dir)

    for label in action_labels:
        label_dir = os.path.join(dataset_dir, label)

        for video_file in os.listdir(label_dir):
            video_path = os.path.join(label_dir, video_file)

            cap = cv2.VideoCapture(video_path)

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.resize(frame, (224, 224)) #possibly redundant?

                frame = preprocess_frame(frame)

                temp_data.append(frame)
                temp_labels.append(label)
            
            cap.release()
            
            print(f"Video {video_path} processed...")
            
    return np.array(temp_data), np.array(temp_labels)

data, labels = loadframes(dataset_dir)
print(data.size)
print(labels.size)
print("Frame loading complete...")

#one-hot encoding
label_mapping = {label: i for i, label in enumerate(np.unique(labels))} #map label to integer
labels = [label_mapping[label] for label in labels] #replace label with its integer
labels = to_categorical(labels, num_classes=num_classes) #one hot conversion
print("One-hot encoding complete...")

frame_train, frame_val, label_train, label_val = train_test_split(data, labels, test_size=0.2, random_state=42)
frame_val, frame_test, label_val, label_test = train_test_split(frame_val, label_val, test_size=0.5, random_state=42)
print("Data splitting complete...")

#convert frame data and label sets into numpy arrays
frame_train = np.array(frame_train)
frame_val = np.array(frame_val)
label_train = np.array(label_train)
label_val = np.array(label_val)

print("Training data shape:", frame_train.shape)
print("Training data size (MB):", frame_train.nbytes / (1024 * 1024))
print("Validation data shape:", frame_val.shape)
print("Validation data size (MB):", frame_val.nbytes / (1024 * 1024))
print("Training labels shape:", label_train.shape)
print("Validation labels shape:", label_val.shape)

model = Sequential([Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)), MaxPooling2D((2, 2)), Conv2D(64, (3, 3), activation='relu'),  MaxPooling2D((2, 2)), Flatten(), Dense(64, activation='relu'), Dense(num_classes, activation='softmax')])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compilation complete...")

history = model.fit(frame_train, label_train, validation_data=(frame_val, label_val), epochs=10, batch_size=32, verbose=1)
print("Model training complete...")

test_loss, test_accuracy = model.evaluate(frame_test, label_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")