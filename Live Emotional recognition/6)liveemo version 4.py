"""
stability -  Stable
execution ends after tracking printing dominant emotion once
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
        facesCurFrame = face_recognition.face_locations(imgS)
        #encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)    
        predictions = DeepFace.analyze(img)
        print(predictions['dominant_emotion'])   
        break
    except ValueError as e:
        continue
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
