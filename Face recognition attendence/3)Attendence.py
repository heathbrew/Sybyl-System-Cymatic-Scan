import cv2 
import numpy as np
import face_recognition
import os
import sys
from datetime import datetime
from datetime import date
import geocoder
import folium
path = 'Face recognition attendence\CriminalImages'
#create a list of images
images =[]
classNames = []
mylist = os.listdir(path)
#lists the filenames
#print(mylist)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
#print(classNames)
#print(len(images))
def findEncodings(images):
    encodeList = []
    for img in images :
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # encode = face_recognition.face_encodings(img)[0]
        # encodeList.append(encode)
        # return encodeList
        try:            
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
            
        except IndexError as e:
            continue
    return encodeList  
encodeListKnown = findEncodings(images)
#print(len(encodeListKnown))
print("Encoding complete ......")
g = geocoder.ip('182.79.102.194')
def markAttendence(name):
    with open('Face recognition attendence\Attendence.csv', 'r+') as f:
        """if somebody has arrived we don't want to repeat it'"""
        myDataList = f.readlines()
        #print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            today = date.today()
            d2 = today.strftime("%B %d, %Y")
            tstring = now.strftime('%H:%M:%S')
            ip='182.79.102.194'
            g = geocoder.ip('182.79.102.194')
            f.writelines(f'\n{name},{d2},{tstring},{ip},{g.lat},{g.lng}')
            print("csv updated")
markAttendence('Carl Eugene Watts')
#time for real time image comparison
#let's open the camera for the same
cap = cv2.VideoCapture(0)
while True:
    success,img = cap.read()
    #resize the image to 1/4th size to increase processing speed
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    #there can be multiple faces in a frame 
    #therefor we need to find face location
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    
    """ we will now compare all the faces found in current frame 
    with the faces present in the directory"""
    
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        """ it will now give as number of elem in face dis
            as no. of faces in directory
            
            we have to select the best match"""
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            """print(name)"""
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            #here (0,255,0) denotes green color
            #here 2 denotes the thickness of border
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            #border for text
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv2.imshow('webcam',img)
    cv2.waitKey(1)
