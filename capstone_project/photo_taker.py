import cv2
cap = cv2.VideoCapture(0)

'''
while 1:
    ret, frame = cap.read()
    cv2.imshow("video", frame)
    cv2.waitKey(33)
'''
while 1:
    a = input()
    ret, frame = cap.read()
    cv2.imwrite("./final_real/"+a+".jpg", frame)