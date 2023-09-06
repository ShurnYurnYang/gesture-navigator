import os
import cv2
import imutils
import numpy as np

# basic idea: 1. gamma correction (light conditions) 2. blur (out of focus) 3. rotation (camera rotation) 4. shearing (camera placement) 5. noise 6. backgrounds

folder_path = './videos'

list_paths = os.listdir(folder_path)

def writerStart(filename, operation):
    _, frame = cap.read()
    scaled = imutils.resize(frame, 512, 512, inter=cv2.INTER_NEAREST)
    (h, w) = scaled.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'./videos/{filename}_{operation}.mp4', fourcc, 30.0, (w, h), True)
    return out

def gamma_correct(frame, gamma):
    
    frame = frame / 255.0

    corrected_frame = np.power(frame, 1.0 / gamma)

    corrected_frame = (corrected_frame * 255).astype(np.uint8)

    return corrected_frame

def blur_correct(frame, kernal):

    blurred_frame = cv2.GaussianBlur(frame, (kernal, kernal), sigmaX=0)

    return blurred_frame

for video_name in list_paths:
    video_path = os.path.join(folder_path, video_name)

    cap = cv2.VideoCapture(video_path)

    gamma_bright_out = writerStart(video_name, "gamma_bright")
    gamma_dark_out = writerStart(video_name, "gamma_dark")

    gaussian_blur_out = writerStart(video_name, "gaussian_blur")

    print(f"Reading {video_name}...\n")

    reading = True

    while reading:
        ret, frame = cap.read()

        if ret:
            gamma_bright_out.write(gamma_correct(frame, 2.0))
            gamma_dark_out.write(gamma_correct(frame, 0.5))
            gaussian_blur_out.write(blur_correct(frame, 7))
        else:
            reading = False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

    gamma_bright_out.release()
    gamma_dark_out.release()
    gaussian_blur_out.release()

print("Video processing complete...")