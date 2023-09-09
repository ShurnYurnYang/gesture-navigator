import os
import cv2
import numpy as np

dataset_dir = './data/processed_dataset'

def preprocess_frame(frame):
    frame = frame.astype('float32') / 255.0

    return frame

def loadframes(dataset_dir):
    data = []
    labels = []

    action_labels = os.listdir(dataset_dir)

    for label in action_labels:
        label_dir = os.path.join(dataset_dir, label)

        for video_file in os.listdir(label_dir):
            video_path = os.path.join(dataset_dir, video_file)

            cap = cv2.VideoCapture(video_path)

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.resize(frame, (512, 512)) #possibly redundant?

                frame = preprocess_frame(frame)

                data.append(frame)
                labels.append(label)
            
            cap.release()
    return np.array(data), np.array(labels)



