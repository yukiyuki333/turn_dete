import cv2
import numpy as np

img = cv2.imread('123.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# 寻找二值化图中的轮廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt=contours[0]
rect = cv2.minAreaRect(cnt)
box=np.int32(cv2.boxPoints(rect))  #左下、左上、右上、右下
print(box)
if abs(box[1][1]-box[2][1])<50:
    print("Straight")
elif box[1][1]<box[2][1]:
    print("Turn right")
elif box[1][1]>box[2][1]:
  print("Turn Left")
cv2.drawContours(img, [box], 0, (0,0,255), 2, 8)
cv2.imshow("test",img)
cv2.waitKey(0)
