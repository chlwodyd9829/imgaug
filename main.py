import cv2
import imgaug as ia

img = cv2.imread('/Users/choijaeyong/Desktop/긴코센정120mg.jpg')
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('img',gray_img)
cv2.waitKey(0)