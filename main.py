import cv2
import numpy as np
import tensorflow as tf 

model = tf.keras.models.load_model('keras_model.h5')
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    img = cv2.resize(frame,(224,224))
    testimg = np.array(img,dtype=np.float32)
    testimg=np.expand_dims(testimg,axis=0)
    normalisedimage=testimg/255.0
    prediction = model.predict(normalisedimage)
    print("prediction: ",prediction)
    cv2.imshow("result",frame)
    key = cv2.waitKey(1)
    
    if key == 32:
        break
video.release()   