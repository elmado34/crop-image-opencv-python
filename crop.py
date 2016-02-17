#! /usr/bin/python



import sys, os
import cv2
import matplotlib.pyplot as plt
import numpy as np


  
def detecte_visages(image,show = False):
  
    img = cv2.imread(image)
    
    face_model = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml") 
  
    rects = face_model.detectMultiScale(img)
    [l,c,d]=img.shape
    # on place un cadre autour des visages
    print ("nombre de visages", len(rects), "dimension de l'image", img.shape,"image",image)
    i=0
    for x1,x2,y1,y2 in rects:
        cv2.rectangle(img, (x1,x2), (x1 + y1, x2 + y2), (255, 0, 0), 3)
        face=img[x2:x2+y2,x1:x1+y1]
   
        face=cv2.resize(face,(92,112))
	face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(( "visage_" + str(i)+ file ),face)
        i=i+1
  
   
    
   
if __name__ == "__main__":
    # applique
    for file in os.listdir(".") :
        if file.startswith("visage") : continue 
        if os.path.splitext(file)[-1].lower() in [".jpg", ".jpeg", ".png" ] :
            detecte_visages (file)
import sys
import os
import numpy as np
import cv2
 
def greyscale(img):
  '''Convert an image to greyscale.
  image  - a numpy array of shape (rows, columns, 3).
  output - a grey scale image which is a numpy array of shape (rows, columns) 
           containing the average of image's 3 channels. 
  '''
  image = np.uint16(img)
  avg = np.zeros((image.shape[0], image.shape[1],image.shape[2]))
  avg[:, :,0] = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2])/3
  output = avg.astype(np.uint8)
  return output[:,:,0]
 
def main():
  '''Convert images to greyscale.
 
  searches for images in directory images/knowapa, and applies the grey scale 
  functiony to each image in the same directory and saves greyscaleimage for 
  each file with the word grey appended to the image
  '''
 
  imagesfolder = os.path.join('images', 'knowpapa')
  exts = ['.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpeg', '.jpg', 
    '.jpe', '.jp2', '.tiff', '.tif', '.png']
  for dirname, dirnames, filenames in os.walk(imagesfolder):
    for filename in filenames:
      name, ext = os.path.splitext(filename)
      if ext in exts and 'grey' not in name:
        img = cv2.imread(os.path.join(dirname, filename))
        greyimage = greyscale(img)
        cv2.imwrite(os.path.join(dirname, name+"grey"+ext), greyimage)
 
 
 
if __name__ == "__main__":
  main()
