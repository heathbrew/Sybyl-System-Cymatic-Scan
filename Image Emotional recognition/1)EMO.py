import cv2
import face_recognition
import matplotlib.pyplot as plt
# img = face_recognition.load_image_file('Emotional recognition\\ayushmanpranav.jpeg')
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# cv2.imshow('ayushman pranav',img)
# cv2.waitKey(0)
img = plt.imread('Emotional recognition\happyface.jpg')
if(img is not None):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)