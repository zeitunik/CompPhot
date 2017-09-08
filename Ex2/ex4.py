import cv2
import numpy as np
import matplotlib.pyplot as plt

parallel = cv2.imread('parallel.png')
perp = cv2.imread('perp.png')

# a)
diff = parallel - perp
cv2.imshow('difference', diff)
cv2.waitKey(0)

fig = plt.figure(figsize=(16,8))
plt.imshow(diff)
plt.show()

# b)

diffs = []
for i in range(100):
    # lam = 100
    # noise_poisson = np.random.poisson(lam, parallel.shape)
    # parallel_p = np.clip(parallel + noise_poisson, 0, 255)
    # perp_p = np.clip(perp + noise_poisson, 0, 255)
    parallel_p = np.random.poisson(parallel)
    perp_p = np.random.poisson(perp)
    diff = parallel_p - perp_p
    diffs.append(diff)

# c)
mean_d = np.mean(diffs, axis=0) /255.
var_d = np.var(diffs, axis=0) /255.

fig = plt.figure(figsize=(15, 10))
fig.add_subplot(121).imshow(mean_d)
fig.add_subplot(122).imshow(var_d)
plt.show()

# d)

eta = 1.1

mean_1 = mean_d/(2.*eta) + var_d/(2.*eta**2)
mean_2 = var_d/(eta**2) - mean_1


fig = plt.figure(figsize=(15, 10))
fig.add_subplot(121).imshow(mean_1)
fig.add_subplot(122).imshow(mean_2)
plt.show()