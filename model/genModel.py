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

#Notes: preoptizimation system RAM usage ~ 6GB

start_time = time.time()

dataset_dir = './data/processed_dataset' #Directory to dataset | root = /gesture-navigator 
frames_list_dir = './model/data/frames'
labels_list_dir = './model/data/labels'
num_classes = 2 #Number of action classes | NOTE: will increase as new data is added

def preprocess_frame(frame): #Normalize pixel values (8-bit colour)
    frame = frame.astype('float32') / 255.0

    return frame

"""def loadframes(dataset_dir): #load frames into dataset
    
    temp_data = []
    temp_labels = []

    action_labels = os.listdir(dataset_dir) #lists all folders and files in dataset_dir

    for label in action_labels:
        label_dir = os.path.join(dataset_dir, label)

        for video_file in os.listdir(label_dir): #os.listdir() here will return an empty list of list_dir points to a file instead of to the subdirectory
            video_path = os.path.join(label_dir, video_file)

            cap = cv2.VideoCapture(video_path)

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.resize(frame, (128, 128)) #Scaled down from 512x512 because of memory concerns

                frame = preprocess_frame(frame) #Calls to normalization

                temp_data.append(frame)
                temp_labels.append(label) #For each frame, appends the label to that frame's respective index in the labels array
            
            cap.release()
            
            print(f"Video {video_path} processed...")
            
    return np.array(temp_data), np.array(temp_labels)

data, labels = loadframes(dataset_dir) #Loads temp_data and temp_labels into the data, labels numpy arrays

print("Data shape:", data.shape)
print("Data size (MB):", data.nbytes / (1024 * 1024))
print("Labels shape:", labels.shape)
print("Labels size (MB):", labels.nbytes / (1024 * 1024))
"""

# REWRITE THIS BY CONSTANTLY WRITING TO ONE FILE INSTEAD OF COMBINING FILES LATER
def load_frame_batched(dataset_dir, batched): #a better way of doing this would probably be to use numpy memmap but both work
    action_labels = os.listdir(dataset_dir)

    #global temp_data 
    temp_data = []
    #global temp_labels 
    temp_labels = []

    counter = 0
    with open('./model/data/labels/master.pkl', 'ab') as label_file:
        with open('./model/data/frames/master.pkl', 'ab') as frame_file:
            for label in action_labels:
                label_dir = os.path.join(dataset_dir, label)

                for video_file in os.listdir(label_dir): #os.listdir() here will return an empty list of list_dir points to a file instead of to the subdirectory

                    counter += 1

                    video_path = os.path.join(label_dir, video_file)

                    cap = cv2.VideoCapture(video_path)

                    while True:
                        ret, frame = cap.read()

                        if not ret:
                            break

                        frame = cv2.resize(frame, (128, 128)) #Scaled down from 512x512 because of memory concerns

                        frame = preprocess_frame(frame) #Calls to normalization

                        temp_data.append(frame)
                        temp_labels.append(label) #For each frame, appends the label to that frame's respective index in the labels array
                    
                    cap.release()
                    
                    print(f"Video {video_path} processed | counter is at {counter}...")

                    if counter % batched == 0:
                        pickle.dump(temp_data, frame_file)
                        pickle.dump(temp_labels, label_file)

                        del temp_data, temp_labels
                        #global temp_data 
                        temp_data = []
                        #global temp_labels 
                        temp_labels = []
            if counter % 40 != 0:
                pickle.dump(temp_data, frame_file)
                pickle.dump(temp_labels, label_file)

            del temp_data, temp_labels
            print("Frames written to file")

                
def concat_lists(list_dir):
    for item in os.listdir(list_dir):
        source_file_path = os.path.join(list_dir, item)
        destination_file_path = os.path.join(list_dir, 'master.pkl') #Fill this

        with open(source_file_path, 'rb') as source_file:
            with open(destination_file_path, 'ab') as destination_file:
                while True:
                    try:
                        data = pickle.load(source_file)
                        pickle.dump(data, destination_file)
                    except EOFError:
                        break

def read_into_numpy(master_dir):
    with open(master_dir, 'rb') as file:
        #master_np = np.empty((0, 128, 128, 3))
        master_list = []
        while True:
            try:
                #np_temp = np.array(pickle.load(file))
                #master_np = np.concatenate((master_np, np_temp), axis=0)
                #del np_temp
                master_list = master_list + pickle.load(file)
            except EOFError:
                break
        #loaded_array = pickle.load(file)
        #loaded_array_np = np.array(loaded_array)
        master_np = np.array(master_list)
        print(master_np.shape)
        return master_np


load_frame_batched(dataset_dir, 40)

#concat_lists('./model/data/frames')

#concat_lists('./model/data/labels')

data = read_into_numpy('./model/data/frames/master.pkl')

labels = read_into_numpy('./model/data/labels/master.pkl')

print("Data shape:", data.shape)
print("Data size (MB):", data.nbytes / (1024 * 1024))
print("Labels shape:", labels.shape)
print("Labels size (MB):", labels.nbytes / (1024 * 1024))

frame_loading_time = time.time()
print(f"Frame loading completed in {(frame_loading_time - start_time):.3f} seconds...")


#one-hot encoding
label_mapping = {label: i for i, label in enumerate(np.unique(labels))} #map each label to an int | label : int, label : int, label : int, etc.
labels = [label_mapping[label] for label in labels] #replace label with its integer | {0, 1, 0}
labels = to_categorical(labels, num_classes=num_classes) #one hot conversion since no ordinal relationship between classes
one_hot_time = time.time()
print(f"One-hot encoding completed in {(one_hot_time - frame_loading_time):.3f} seconds...")

frame_train, frame_val, label_train, label_val = train_test_split(data, labels, test_size=0.2, random_state=42) #0.8 train, 0.2 val split
frame_val, frame_test, label_val, label_test = train_test_split(frame_val, label_val, test_size=0.5, random_state=42) #0.5 val, 0.5 test split
data_split_time = time.time()
print(f"Data splitting completed in {(data_split_time - one_hot_time):.3f} seconds...")

#convert frame data and label sets into numpy arrays
frame_train = np.array(frame_train)
frame_val = np.array(frame_val)
label_train = np.array(label_train)
label_val = np.array(label_val)

data_to_numpy_time = time.time()
print(f"Data to numpy array completed in {(data_to_numpy_time - data_split_time):.3f} seconds...")
print("Training data shape:", frame_train.shape)
print("Training data size (MB):", frame_train.nbytes / (1024 * 1024))
print("Validation data shape:", frame_val.shape)
print("Validation data size (MB):", frame_val.nbytes / (1024 * 1024))
print("Training labels shape:", label_train.shape)
print("Validation labels shape:", label_val.shape)

#       Model type: Sequential 
#       Note: Baseline CNN with 2 convlutional layers, maxpool layer, flatten layer, 2 connected layers, softmax acvitaiton function
model = Sequential([Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)), MaxPooling2D((2, 2)), Conv2D(64, (3, 3), activation='relu'),  MaxPooling2D((2, 2)), Flatten(), Dense(64, activation='relu'), Dense(num_classes, activation='softmax')])

#             Optimizer: ADAM   Loss: Categorical_crossentropy   Metrics: accuracy
#             Try: RMSprop | Lookahead | Yogi
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_compile_time = time.time()
print(f"Model compilation completed in {(model_compile_time - data_to_numpy_time):.3f} seconds...")

# 10 Epoch (experimental), 32 batch size, verbose 1: 0(silent), 1(loading bar), 2(epoch no.)
history = model.fit(frame_train, label_train, validation_data=(frame_val, label_val), epochs=10, batch_size=32, verbose=1)
train_time = time.time()
print(f"Model training completed in {(train_time - model_compile_time):.3f} seconds...")

#Evaluate
test_loss, test_accuracy = model.evaluate(frame_test, label_test)
evaluate_time = time.time()
print(f"Evaluation completed in {(evaluate_time - train_time):.3f} seconds...")
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

#Save entire model (and weights)
model.save('./model/model.keras')
print("Model saved...")
end_time = time.time()
print(f"Execution completed in {(end_time - start_time):.3f} seconds...")