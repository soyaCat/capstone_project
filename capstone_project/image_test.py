import image_function as IMG_F
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./test_img/92_main.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
result, seta = IMG_F.image_process(img)
plt.imshow(result)
plt.show()
print(seta)