import os
import cv2
import face_recognition
import pickle

folderPath = r'C:\Users\SHIVAM\Desktop\GazeTracking\Face_rec\Images'

pathlist = os.listdir(folderPath)

print(pathlist)

imagelist = []
studentid = []

for path in pathlist:
    imagelist.append(cv2.imread(os.path.join(folderPath, path)))
    studentid.append(os.path.splitext(path)[0])

def findEncoding(imagelist):
    encodeList = []
    for img in imagelist:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
        
encodelist = findEncoding(imagelist)

encodelist_wids = [encodelist, studentid]


file = open("encodelist.p", "wb")
pickle.dump(encodelist_wids, file)
file.close()

# print(encodelist)