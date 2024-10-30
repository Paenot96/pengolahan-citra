import numpy as np 
import imageio as img
import matplotlib.pyplot as plt

image = img.imread("D:\canon\gol.jpg")

red = image[:, :, 0]
green = image[:, :, 1]
blue = image[:, :, 2]

imgRed = np.zeros_like(image)
imgGreen = np.zeros_like(image)
imgBlue = np.zeros_like(image)

imgRed[:, :, 0] = red
imgGreen[:, :, 1] = green
imgBlue[:, :, 2] = blue

plt.figure(figsize=(10, 10))

#citra asli
plt.subplot(4, 1, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(4, 1, 2)
plt.imshow(imgRed)
plt.title("Red Image")

plt.subplot(4, 1, 3)
plt.imshow(imgGreen)
plt.title("Green Chanel")

plt.subplot(4, 1, 4)
plt.imshow(imgBlue)
plt.title("Blue Chanel")

plt.tight_layout()
plt.show()