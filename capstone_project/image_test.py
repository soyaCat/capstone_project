import image_function as IMG_F
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while 1:
    # img = cv2.imread("./final_real/"+"test"+".jpg")
    ret, img = cap.read()
    img = cv2.resize(img, (410, 300), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result, seta = IMG_F.image_process(img)
    print(seta)