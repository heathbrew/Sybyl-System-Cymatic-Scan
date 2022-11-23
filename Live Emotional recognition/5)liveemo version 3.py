"""
stability -  unstable 
This version of live emo is to track face via a rectangle
"""
import cv2
import face_recognition
from deepface import DeepFace
print("Package Imported")
cap = cv2.VideoCapture(0)
while True:
    success,imgimported = cap.read()
    cv2.imshow("Output",imgimported)
    #resize the image to 1/4th size to increase processing speed
    imgS = cv2.resize(imgimported,(0,0),None,0.25,0.25)
    img = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    try:
        faceLoc = face_recognition.face_locations(imgS)
        y1,x2,y2,x1 = faceLoc
        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        predictions = DeepFace.analyze(img)
        print(predictions['dominant_emotion'])   
    except ValueError as e:
        continue
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
