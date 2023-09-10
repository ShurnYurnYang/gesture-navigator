import cv2
import imutils

cap = cv2.VideoCapture(0)

count = 45

recording = False

def writerStart():
    _, frame = cap.read()
    scaled = imutils.resize(frame, 512, 512, inter=cv2.INTER_NEAREST)
    (h, w) = scaled.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    global out 
    out = cv2.VideoWriter('./videos/capture_%03i.mp4' % count, fourcc, 30.0, (w, h), True)

if cap.isOpened():
   writerStart()
else:
    print("Camera is not opened")

while cap.isOpened():
    _, frame = cap.read()

    scaled = imutils.resize(frame, 512, 512, inter=cv2.INTER_NEAREST)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        if recording:
            recording = False
            count+=1
            out.release()
        else:
            recording = True
            writerStart()

    if recording:
        out.write(scaled)

    cv2.imshow("Output", scaled)
    
    if key == ord("q"):
        break


cv2.destroyAllWindows()
cap.release()
out.release()