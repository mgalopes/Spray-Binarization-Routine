# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:36:14 2022

@author: Peixe
"""
import cv2
import math
import numpy as np

path = 'etanol_conv_25C_50bar.png'
img = cv2.imread(path)
pointsList = []
# convert the input image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# apply thresholding to convert grayscale to binary image
ret,thresh = cv2.threshold(gray,30,255,cv2.THRESH_BINARY)
median = np.median(img)
lower = int(max(0, (1.0 - 0.33) * median))
upper = int(min(255, (1.0 + 0.33) * median))
#lower sigma-->tighter threshold(default value of sigma can be 0.33)
edge_image= cv2.Canny(img, lower, upper)
cv2.imshow("edgeDetection", img)
cv2.waitKey(0)
cv2.imshow("edgeDetection2", thresh) 
cv2.waitKey(0)

# find the contours
contours,hierarchy = cv2.findContours(thresh,
cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
print("Number of contours detected:", len(contours))

# loop over all the contours
for cnt in contours:
   # extreme points
   leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
   rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
   topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
   bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
   points = [leftmost, rightmost, topmost, bottommost]
   print("Extreme points",points)
   cv2.drawContours(img, contours, -1, (0,255,0), 3)
   pointsList.append(contours) # Lista de pontos das regiões de contorno
   contourarea=cv2.contourArea(cnt)
   if contourarea > 1:
      print("Area em pixels",contourarea)
   else:
      print('Area infinitesimal')
      
   #print("Countour",contours) >> Permite extrair os pontos das regiões de contorno
   
   #Extract area inside the countour
   
   # draw the points on th image
   for point in points:
      cv2.circle(img, point, 4, (0, 0, 255), -1)
      
# display the image with drawn extreme points
while True:
   cv2.imshow("Extreme Points", img)
   if cv2.waitKey(1) & 0xFF == 27:
      break
cv2.destroyAllWindows()
