import cv2
import numpy as np

def main():
    # Load the Haar Cascades for face and eye detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

            for (ex, ey, ew, eh) in eyes:
                eye = roi_gray[ey:ey+eh, ex:ex+ew]
                threshold_eye = process_eye(eye)
                gaze_direction = estimate_gaze(threshold_eye)
                cv2.putText(frame, gaze_direction, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

        cv2.imshow('Gaze Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_eye(eye):
    _, threshold = cv2.threshold(eye, 70, 255, cv2.THRESH_BINARY_INV)
    return threshold

def estimate_gaze(threshold_eye):
    height, width = threshold_eye.shape
    left_eye = threshold_eye[0:height, 0:int(width/2)]
    right_eye = threshold_eye[0:height, int(width/2):width]

    left_white = cv2.countNonZero(left_eye)
    right_white = cv2.countNonZero(right_eye)

    if left_white == 0:
        left_white = 1
    if right_white == 0:
        right_white = 1

    gaze_ratio = left_white / right_white
    if gaze_ratio > 1.5:
        return "Looking left"
    elif gaze_ratio < 0.5:
        return "Looking right"
    else:
        return "Looking forward"

if __name__ == "__main__":
    main()
