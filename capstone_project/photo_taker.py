import cv2
import matplotlib.pyplot as plt
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
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

    cv2.imwrite("./final_real/"+a+".jpg", frame)