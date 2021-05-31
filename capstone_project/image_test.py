import image_function as IMG_F
import cv2
import matplotlib.pyplot as plt

for i in range(1,4):
    img = cv2.imread("./real2_img/"+str(i)+".jpg")
    img = cv2.resize(img, (440, 290), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result, seta = IMG_F.image_process(img)
    plt.imshow(result)
    plt.show()
    print(seta)