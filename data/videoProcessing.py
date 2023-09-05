import os
import cv2
import imutils

# basic idea: 1. gamma correction (light conditions) 2. blur (out of focus) 3. rotation (camera rotation) 4. shearing (camera placement) 5. noise 6. backgrounds

folder_path = './videos'

list_paths = os.listdir(folder_path)

for video_name in list_paths:
    video_path = os.path.join(folder_path, video_name)

    cap = cv2.VideoCapture(video_path)

    print(f"Reading {video_name}...\n")

    while True:
        ret, frame = cap.read()

        if ret:
            cv2.imshow('Video Frame', frame)
        else:
            print(f"No more files")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()