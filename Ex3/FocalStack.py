import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal



# define kernel
kernel = None
# ---
# your code ...
## Laplacian filter
kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
# ---



# load image stack
def LoadImages():
    imageStack = None
    # ---
    # your code ...
    ## read the files
    images = [mpimg.imread("kaktus1_{}.JPG".format(format(i, "04"))) for i in range(1,24)]
    imageStack = np.stack(images, axis=0)
    # ---
    assert (23, 683, 1024, 3) == imageStack.shape
    return imageStack



# compute the depth map
def Depth(imageStack):
    depth = None
    # ---
    # your code ...
    contrast = np.empty(imageStack.shape[0:3])
    for img in range(imageStack.shape[0]):
        ## convolve each image channel with the kernel
        channels = np.array([scipy.signal.convolve(imageStack[img,:,:,ch], kernel, mode='same') for ch in range(3)])
        ## add the absolute values of the channels
        contrast[img] = np.sum(np.abs(channels), axis=0)         

    ## choose the index of the image which is maximal (sharpest) 
    depth = np.argmax(contrast, axis=0)
    # ---
    print("Depth.type", depth.dtype)
    return depth



# compute the combined image:
def Combine(imageStack, depth):
    combined = np.empty(imageStack.shape[1:4], dtype='uint8')
    # ---
    # your code ...
    ## for each pixel choose the image given by the depth map 
    for y in range(imageStack.shape[1]):
        for x in range(imageStack.shape[2]):
            combined[y,x,:] = imageStack[depth[y,x],y,x,:]
    # ---
    return combined


# compute the denoised depth map
def DenoisedDepth(imageStack):
    denoisedDepth = None
    # ---
    # your code ...
    depth = Depth(imageStack)
    ## apply a median filter with size 5 to the depth map
    denoisedDepth = scipy.signal.medfilt(depth, 5).astype('int64')
    # ---
    return denoisedDepth


# The main programm:

imageStack = LoadImages()
depth = Depth(imageStack)
print("depth: ", depth.shape)
combined = Combine(imageStack, depth)
print("combined: ", combined.shape)
mpimg.imsave("combined.jpg", combined)


# display them:
plt.figure("Combined")
plt.imshow(combined)

plt.figure("Depth")
plt.imshow(depth)
plt.colorbar()


# enhanced combined image:
denoisedDepth = DenoisedDepth(imageStack)
enhanced = Combine(imageStack, denoisedDepth)
mpimg.imsave("enhanced.jpg", enhanced)

plt.figure("Denoise Depth")
plt.imshow(denoisedDepth)
plt.colorbar()

plt.figure("enhanced")
plt.imshow(enhanced)

plt.show()