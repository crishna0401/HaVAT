import cv2
import scipy
import numpy as np
import pandas as pd
from utils import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity



############################# Generate synthetic data (S1 - 3 circles data) #################################

# Number of data points in each circle
num_points = 300

# Radii of the circles
radii = [1, 3, 5]

# Generate data for each circle
data = []
labels = []
for i, radius in enumerate(radii):
    theta = np.linspace(0, 2*np.pi, num_points)
    r = np.random.normal(radius, 0.1, num_points)  # Add some noise to the radius
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    data.append(np.column_stack((x, y)))
    labels.extend([i] * num_points)

# Combine the data from all circles
data = np.concatenate(data)
labels = np.array(labels)

# save plot in pdf format
plt.figure(figsize=(8,8))
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.axis('off')
plt.savefig('../Experimental-datasets/S1/scatter-plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()


rs = euclidean_distances(data, data)
res = iVAT(rs)
a = res[0]
rescaled = (255.0 / a.max() * (a - a.min())).astype(np.uint8)
cut = res[-2]
reodered_ind = res[-1]
plt.imshow(rescaled,cmap='gray')
plt.show()

np.save('../Experimental-datasets/S1/cut_values.npy',cut)
np.save('../Experimental-datasets/S1/gt.npy',labels)
np.save('../Experimental-datasets/S1/reorderedindices.npy',reodered_ind)
np.save('../Experimental-datasets/S1/data.npy',data)
np.save('../Experimental-datasets/S1/ivat.npy',a)
plt.imsave('../Experimental-datasets/S1/ivat.png',rescaled,cmap='gray')

################################# Automatic Assessment ###########################################
data_name = 'S2'
ivat_img = cv2.imread('../Experimental-datasets/'+data_name+'/ivat.png')
cut = np.load('../Experimental-datasets/'+data_name+'/cut_values.npy')
data = np.load('../Experimental-datasets/'+data_name+'/data.npy')
gt = np.load('../Experimental-datasets/'+data_name+'/gt.npy')
reodered_ind = np.load('../Experimental-datasets/'+data_name+'/reorderedindices.npy')

part_img, _, indices, _, pred, act = automated_evaluation(ivat_img.copy(),gt,reodered_ind,data,cut,low_th = 10, high_th= 200, min_cluster_size= 0.1,flag=False)
