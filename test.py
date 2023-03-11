import os
import cv2
import numpy
import tensorflow
from keras.models import model_from_json
from keras.preprocessing import image

Mod=model_from_json(open("MODEL.json","r").read())

Mod.load_weights('MODEL.h5')

F_haar_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture=cv2.VideoCapture(0)

while True:
    return_,testimage=capture.read()
    if not return_:
        continue
    grayimage=cv2.cvtColor(testimage, cv2.COLOR_BGR2GRAY)

    facedetected=F_haar_cascade.detectMultiScale(grayimage,1.32,5)


    for (a,b,c,d) in facedetected:
         cv2.rectangle(testimage,(a,b),(a+c,b+d),(255,0,0),thickness=7)
         Roigray=grayimage[b:b+c,a:a+d]
         Roigray=cv2.resize(Roigray,(48,48))
         imagepixels= tensorflow.keras.utils.img_to_array(Roigray)
         imagepixels=numpy.expand_dims(imagepixels, axis=0)
         imagepixels /=255

         predict= Mod.predict(imagepixels)


         maxindex= numpy.argmax(predict[0])

         Emotions=('angry','disgust','fear','happy','sad','neutral','surprise')
         predict_Emotions= Emotions[maxindex]

         cv2.putText(testimage,predict_Emotions,(int(a), int(b)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


    resizedimage = cv2.resize(testimage,(1000,700))     
    cv2.imshow("Facial Recognition",resizedimage)


    if cv2.waitKey(10) == ord('g'):
        break


capture.release()
cv2.destroyAllWindows
