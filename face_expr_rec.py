from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier(
    r'D:\Desktop\ml-keras\haarcascade_frontalface_default.xml')

model = load_model(r'D:\Desktop\ml-keras\Emotion_little_vgg.h5')

class_labels = ['Angry', 'Happy', 'Sad', 'Neutral', 'Surprise']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_posistion = (x, y)
            cv2.putText(frame, label, label_posistion,
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No face Found', (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imshow('Emotion detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
