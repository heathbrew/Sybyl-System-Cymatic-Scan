import cv2 
import numpy as np
import face_recognition
import os
import sys
from datetime import datetime
from datetime import date
import geocoder
import folium
from deepface import DeepFace
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

"""this version is for clean code"""
import csv
file = 'Face recognition attendence\\Datasets\\fetcher.csv' # Target CSV file path
with open(file, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
datalist = list(data)
namelist=[]
CrimeCofficient=[]
for i in range(1,len(datalist)):
    namelist.append(datalist[i][1])
    CrimeCofficient.append(float(datalist[i][8]))
fetcher = {}
for i in range(len(namelist)):
    fetcher[namelist[i]] = CrimeCofficient[i]
print("CrimeCofficients imported")
#print(fetcher['Carl Eugene Watts']+1.00)
g = geocoder.ip('182.79.102.194')
def markAttendence(name):
    with open('newspotted.csv', 'r+') as f:
        """if somebody has arrived we don't want to repeat it'"""
        myDataList = f.readlines()
        #print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',').
            nameList.append(entry[0])
        if name not in nameList:
            print(fetcher[name])
            now = datetime.now()
            today = date.today()
            d2 = today.strftime("%B %d, %Y")
            tstring = now.strftime('%H:%M:%S')
            ip='182.79.102.194'
            g = geocoder.ip(ip)
            f.writelines(f'{name},{d2},{tstring},{g.lat},{g.lng},{fetcher[name]}')
            print("csv updated")
markAttendence('Carl Eugene Watts')
markAttendence('Philipp Tyurin')

def crime(x,y,z):
    if x == ("angry" or "fear"):
        return 3.00 + y
    elif x == ("disgust" or "surprise"):
        return 2.00 + y
    else :
        return y
    
    
#time for real time image comparison
#let's open the camera for the same
cap = cv2.VideoCapture(0)
while True:
    success,imgimported = cap.read()
    #resize the image to 1/4th size to increase processing speed
    imgS = cv2.resize(imgimported,(0,0),None,0.25,0.25)
    img = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    imgsize = cv2.resize(imgimported,(0,0),None,0.20,0.20)
    imgnew = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    #there can be multiple faces in a frame 
    #therefor we need to find face location
    try:
        facesCurFrame = face_recognition.face_locations(img)
        encodesCurFrame = face_recognition.face_encodings(img,facesCurFrame)    
        
    
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
                name = classNames[matchIndex]
                CrimeCoff = fetcher[name]
                name = name.upper()
                """print(name)"""
                y1,x2,y2,x1 = faceLoc
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(imgimported,(x1,y1),(x2,y2),(0,255,0),2)
            #here (0,255,0) denotes green color
            #here 2 denotes the thickness of border
                cv2.rectangle(imgimported,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            #border for text
                predictions = DeepFace.analyze(imgnew)
                dom =predictions['dominant_emotion']
                print(dom)
                coff = crime(dom,CrimeCoff)
                print(coff)
                cv2.putText(imgimported,str(coff)+" "+str(name),(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    except ValueError as e:
        continue
    cv2.imshow('webcam',imgimported)
    cv2.waitKey(1)
