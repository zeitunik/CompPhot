import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('yosemite.png', 0)
cv2.imshow('img', img)
cv2.waitKey(0)

######################
## 	gaussian noise 	##
######################

x,y = img.shape
mean = 0
var = 256
sigma = var**0.5

noise_gauss = np.random.normal(mean, sigma, (x,y))
img_gauss = np.clip(img + noise_gauss, a_min=0, a_max=255)
cv2.imshow('gaussian noise', img_gauss.astype(np.uint8))
cv2.waitKey(0)

######################
##	poisson noise 	##
######################

mean = 10

noise_poisson = np.random.poisson(mean, (x,y))
img_poisson = np.clip(img + noise_poisson, a_min=0, a_max=255)
cv2.imshow('poisson noise', img_poisson.astype(np.uint8))
cv2.waitKey(0)

# h,w = I.shape
# mean = 0
# var = 256
# sigma = var**0.5
# gauss = np.random.normal(mean, sigma, (h,w))

# I_gauss = np.clip(I + gauss, 0, 255)

fig = plt.figure(figsize=(16,16))
fig.add_subplot(211).imshow(img_gauss, cmap='gray')
fig.add_subplot(212).imshow(img_poisson, cmap='gray')
plt.show()