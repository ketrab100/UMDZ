from time import sleep

import cv2
import numpy as np

from data import emotion_labels
import tensorflow as tf

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(1)

model = tf.keras.models.load_model("model_five.keras")

while 1:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = img[y : y + h, x : x + w]

    if faces is not None and len(faces) > 0:
        x, y, w, h = faces[0]

        crop_img = img[y : y + h, x : x + w]

        gray_crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_crop_img, (48, 48))
        resized_image = np.expand_dims(resized_image, axis=-1)

        img_array = np.array([resized_image])

        predictions = model.predict(img_array)

        predicted_class = np.argmax(predictions, axis=1)
        print(emotion_labels[predicted_class[0]])

        resized_image = cv2.resize(resized_image, (480, 480))

        crop_img = cv2.resize(crop_img, (480, 480))

        cv2.imshow("img", crop_img)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
    sleep(0.2)

cap.release()

cv2.destroyAllWindows()
