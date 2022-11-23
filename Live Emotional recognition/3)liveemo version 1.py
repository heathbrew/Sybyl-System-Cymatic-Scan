import cv2
from deepface import DeepFace
print("Package Imported")
cap = cv2.VideoCapture(0)
while True:
    success,imgimported = cap.read()
    cv2.imshow("Output",imgimported)
    img = cv2.cvtColor(imgimported,cv2.COLOR_BGR2RGB)
    predictions = DeepFace.analyze(img)
    print(predictions['dominant_emotion'])   
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
