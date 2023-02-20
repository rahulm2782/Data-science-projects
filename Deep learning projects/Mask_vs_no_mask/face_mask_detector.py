import cv2
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array,load_img,array_to_img

model = load_model('/home/rahul/Documents/Data science files/deep learning projects/Mask_and_No_mask/face_mask.h5')

face_cascade = cv2.CascadeClassifier('/home/rahul/Documents/Data science files/cascade files/haarcascade_frontalface_default.xml')

labels = {0:'mask_weared_incorrect', 1:'with_mask', 2:'without_mask'}


def detect(gray,frame):
    faces = face_cascade.detectMultiScale(gray,1.1,2)
    
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,predictions_,(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
    return frame

def prediction(frame):
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img = cv2.imwrite("image_path.png", frame)
    img = load_img("image_path.png",target_size=(224,224,3))
    img = img_to_array(img)
    img = img/255.
    img = np.expand_dims(img,[0])
    pred = model.predict(img)
    pred = np.argmax(pred)
    pred = labels[pred]
    return str(pred)


cap = cv2.VideoCapture(0)

count = 0

while(True):

    ret, frame = cap.read()
    count += 1
    if count %3 == 0:
        continue
    else:

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        predictions_ = prediction(frame)
        frame = detect(gray,frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()