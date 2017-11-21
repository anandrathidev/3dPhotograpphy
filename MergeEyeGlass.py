# -*- coding: utf-8 -*-
"""
Created on 

@author: Anand Rathi
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


imgPath = "C:/Users/rb117/Documents/personal/Anand/ImgProcessing/"
fnList = []
fnList.append(cv2.imread(imgPath + "anand_img_1.jpg"))
fnList.append(cv2.imread(imgPath + "hrithik_roshan_1.jpg"))
fnList.append(cv2.imread(imgPath + "Kangana_Ranaut_1.jpg"))

sgList = []
sgList.append(cv2.imread(imgPath + "sg1.jpg"))
sgList.append(cv2.imread(imgPath + "sg2.jpg"))
sgList.append(cv2.imread(imgPath + "sg3.jpg"))
sgList.append(cv2.imread(imgPath + "sg4.jpg"))
sgList.append(cv2.imread(imgPath + "sg5.jpg"))
sgList.append(cv2.imread(imgPath + "sg6.jpg"))



def extractObject(img):
  plt.imshow(img),plt.show()
  newmask = img
  mask = np.zeros(img.shape[:2],np.uint8)
  mask[newmask == 0] = 0
  mask[newmask == 255] = 1
  mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
  mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
  img = img*mask[:,:,np.newaxis]
  plt.imshow(img),plt.colorbar(),plt.show()
  plt.imshow(img),plt.colorbar(),plt.show()

extractObject(sgList[4])

face_cascade = cv2.CascadeClassifier(imgPath + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(imgPath +  'haarcascade_eye.xml')
img = cv2.imread(fn1)

sg1img = cv2.imread(sg1)
sg2img = cv2.imread(sg2)
sg3img = cv2.imread(sg3)
sg4img = cv2.imread(sg4)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

for fimg in fnList:
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (x,y,w,h) in faces:
    #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
      cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
      small=cv2.resize(small, (x, y))  
plt.imshow(img)

plt.imshow(sg1img)


cv2.waitKey(0)
cv2.destroyAllWindows()