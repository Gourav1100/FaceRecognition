import matplotlib
import numpy as np
from skimage.io import imread
data = np.load('./dataset/face_images.npz')['face_images']
data = data.transpose((2,0,1))
data.shape
for i in range(len(data)):
    print("saving face_"+str(i))
    matplotlib.image.imsave("./dataset/jpegs/face_"+str(i)+".jpeg",data[i])

