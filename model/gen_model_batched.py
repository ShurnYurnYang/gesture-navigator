import time
import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

dataset_dir = './data/processed_dataset' #Directory to dataset | root = /gesture-navigator 
frames_list_dir = './model/data/frames'
labels_list_dir = './model/data/labels'
num_classes = 2 #Number of action classes | NOTE: will increase as new data is added

# everything here should probably be a separate script
train_video_path_list = []
train_label_list = []
val_video_path_list = []
val_label_list = []
test_video_path_list = []
test_label_list = []

num_train_frames = 0
num_val_frames = 0
num_test_frames = 0

for label in dataset_dir:
        label_dir = os.path.join(dataset_dir, label)

        for video_file in os.listdir(label_dir): #os.listdir() here will return an empty list of list_dir points to a file instead of to the subdirectory
            video_path = os.path.join(label_dir, video_file)

            rand = np.random.randint(11)
            if rand <= 8: #train
                train_video_path_list.append(video_path)
                train_label_list.append(label)

                cap = cv2.VideoCapture(video_path)

                num_train_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            elif rand == 9: #val
                val_video_path_list.append(video_path)
                val_label_list.append(label)

                cap = cv2.VideoCapture(video_path)

                num_val_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else: #test
                test_video_path_list.append(video_path)
                test_label_list.append(label)

                cap = cv2.VideoCapture(video_path)

                num_test_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# -------------------------------------------------------------

video_pointer = 0
label_pointer = 0

lb = LabelBinarizer()

def preprocess_frame(frame): #Normalize pixel values (8-bit colour)
    frame = frame.astype('float32') / 255.0

    return frame

def frame_generator(type, batch_size, binarizer):
    while True:
        images = []
        labels = []
        while len(images) < batch_size: #batch size refers to the number of videos NOT frames
            match type:
                case 'train': 
                    cap = cv2.VideoCapture(train_video_path_list[video_pointer])
                case 'val':
                    cap = cv2.VideoCapture(val_video_path_list[video_pointer])
                case 'test':
                    cap = cv2.VideoCapture(test_video_path_list[video_pointer])

            while True:
                ret, frame = cap.read()

                if not ret:
                    label_pointer += 1
                    video_pointer += 1
                    break

                frame = cv2.resize(frame, (128, 128)) #Scaled down from 512x512 because of memory concerns

                frame = preprocess_frame(frame) #Calls to normalization

                images.append(frame)

                match type:
                    case 'train': 
                        labels.append(train_label_list[label_pointer]) #For each frame, appends the label to that frame's respective index in the labels array
                    case 'val':
                        labels.append(val_label_list[label_pointer])
                    case 'test':
                        labels.append(test_label_list[label_pointer])
            cap.release()
        labels = binarizer.transform(np.array(labels))

        yield (np.array(images), labels)

train_gen = frame_generator(type='train', batch_size=8, binarizer=lb)

test_gen = frame_generator(type='test', batch_size=8, binarizer=lb)

model = Sequential([Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)), MaxPooling2D((2, 2)), Conv2D(64, (3, 3), activation='relu'),  MaxPooling2D((2, 2)), Flatten(), Dense(64, activation='relu'), Dense(num_classes, activation='softmax')])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_compile_time = time.time()

history = model.fit(x=train_gen, steps_per_epoch=(), validation_data=(frame_val, label_val), epochs=10, batch_size=32, verbose=1)
train_time = time.time()