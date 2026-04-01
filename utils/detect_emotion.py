import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("emotion_model.hdf5")

emotion_labels = ['angry', 'happy', 'sad']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_emotion():
    cap = cv2.VideoCapture(0)
    detected_emotion = "neutral"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.reshape(face, (1, 48, 48, 1))

            prediction = model.predict(face)
            detected_emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return detected_emotion
