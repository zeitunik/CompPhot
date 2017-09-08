import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.sparse as ss
import scipy.signal


# some helper functions (internally, images are represented as floats to avoid errors)
def LoadImage(filename):
	return mpimg.imread(filename).astype(np.float)

def SaveImage(filename, img):
	plt.imsave(filename, np.clip(img, 0, 255).astype(np.uint8))

def ShowImage(title, img):
	plt.figure(title)
	assert img.dtype == np.float
	plt.imshow(np.clip(img, 0, 255).astype(np.uint8))


def ComputeConvolutionMatrix(K, h, w):
	
	values = None # stores matrix values with coordinates
	# ---------
	# your code
	# ---------
	## assuming the kernel is of shape K = [[0,x0,0],[x1,x2,x3],[0,x4,0]]
	values = np.array([K[0,1], K[1,0], K[1,1], K[1,2], K[2,1]])


	# create sparse matrix from value lists
	# ---------
	# your code
	# ---------

	## first create the block that is on the diagonal 
	A = ss.diags([values[4], values[2], values[0]], [-1,0,1], shape=(h,h))
	
	## create the sparse matrix that has A on the diagonal
	## M.shape = (hw,hw)
	size = h*w
	M = ss.block_diag((A,)*w)
	assert M.shape == (size, size)

	## create another sparse matrix with the rest of the coefficients on the 
	## -h and h off-diagonals
	B = ss.diags([values[3], values[1]],[-h, h], shape=(size, size))

	## final matrix
	M = M + B

	return M.tocsr()


def TransformImage(M, img):
	# return np.moveaxis([
	# 	M.dot(img[:,:,i].flatten()).reshape(img.shape[:2])
	# 	for i in range(3)], 0, 2)
	return np.stack([M.dot(img[:,:,i].flatten('F')).reshape(img.shape[:2], order='F') 
		for i in range(3)], axis = 2)

def test(K, img):
	M = ComputeConvolutionMatrix(K, img.shape[0], img.shape[1])
	T = TransformImage(M, img)
	C = np.stack([scipy.signal.convolve2d(img[:,:,ch], K, 'same') for ch in range(3)], axis=2)
	return (T, C) 

def BlendImage(img1, img2, mask):
	mask /= np.max(mask)
	inv = 1 - mask
	return img1*mask + img2*inv
	

# Test area
K = np.array([[0,1,0],[2,3,4],[0,5,0]])

img1 = np.zeros((4,4,3))
img1[1,1,:] = 1

img2 = np.zeros((4,4,3))
img2[2,3,:] = 1

img3 = np.zeros((4,5,3))
img3[1,1,:] = 1
img3[2,4,:] = 1

T1, C1 = test(K, img1)
T2, C2 = test(K, img2)
T3, C3 = test(K, img3)

print("Kernel:\n", K)
print("Test matrix 1:\n", T1[:,:,0])
print("Test matrix 2:\n", T2[:,:,0])
print("Test matrix 3:\n", T3[:,:,0])

assert np.all(T1 == C1)
assert np.all(T2 == C2)
assert np.all(T3 == C3)

# Main Programm
img = LoadImage("diver.jpg")
h = img.shape[0]
w = img.shape[1]
ShowImage("Original", img)

# define kernels:
K_blur = None
K_laplace = None
# ---------
# your code
# ---------

K_blur = 0.25*np.array([[0,1,0],[1,0,1],[0,1,0]])
K_laplace = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

# blur the image using convolution:
M = ComputeConvolutionMatrix(K_blur, h, w)
ShowImage("blurred", TransformImage(M, img))


# naive blending:
shark = LoadImage("shark.jpg")
mask = LoadImage("shark-mask4.jpg")
ShowImage("naive blend", BlendImage(shark, img, mask))


# improved blending
L = ComputeConvolutionMatrix(K_laplace, h, w)

diverL = TransformImage(L, img)
ShowImage("Laplace", diverL+128)

sharkL = TransformImage(L, shark)

lblend = BlendImage(sharkL, diverL, mask)
ShowImage("LBlend", lblend+128)


reconstructed = None
# ---------
# your code
# ---------
## solve L x = lblend
reconstructed = np.stack([
	ss.linalg.spsolve(L, lblend[:,:,ch].flatten('F')).reshape(lblend.shape[:2], order='F') 
	for ch in range(3)], axis=2)

ShowImage("Combined", reconstructed)



# now add constraints:

constrainedReconstruction = None
# ---------
# your code
# ---------

ShowImage("Constrained", constrainedReconstruction)


print("done")