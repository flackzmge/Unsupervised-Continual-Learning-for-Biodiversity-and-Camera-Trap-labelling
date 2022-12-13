# The k-means implementation was adapted from Ayoosh Kathuria's article here:
# https://blog.paperspace.com/speed-up-kmeans-numpy-vectorization-broadcasting-profiling/

import numpy as np 
import matplotlib.pyplot as plt 
import random
import time 
import os
from PIL import Image
from torchvision import transforms
import torch

# === Helper functions ===================

# Rescale image
def rescale(an_img, fac):
    a_size = an_img.size;
    new_size = (int(np.round(a_size[0]*fac)), int(np.round(a_size[1]*fac)))
    new_image = an_img.resize(new_size)
    return new_image

# Re-orient to horizontal, if vertical
def reorient(img):
    (x_leng, y_leng) = img.size
    if y_leng > x_leng:
        # rotate
        img = img.transpose(Image.ROTATE_90)
    
    return img

# Function for loading the data from a single folder
def load_data_dir(source_path, scale_fac):

    # Source directory contents
    dir_contents = os.listdir(source_path) # returns list
    dir_contents.sort()

    num_pics = len(dir_contents)

    # Load first image to get dimensions, etc.
    img_1 = Image.open(source_path + dir_contents[0])
    img1_rs = rescale(img_1, scale_fac)
    img1_rs = reorient(img1_rs)
    (xleng, yleng) = img1_rs.size
    num_chan = 3 # we assume that images are RGB
    all_dim = xleng*yleng*num_chan 
    data_dim = (yleng, xleng, num_chan)
    # Initialize data 
    data = np.zeros((num_pics, all_dim))
    
    for i in range(num_pics-1):
        # Load and prepare images
        print('Image name: ', dir_contents[i]);
        img_1 = Image.open(source_path + dir_contents[i])
        img1_rs = rescale(img_1, scale_fac)
        img1_rs = reorient(img1_rs) # re-orient the image, if necessary
        # img1_rs.show()
        img_1_arr = np.asarray(img1_rs)
        data[i,:] = img_1_arr.flatten()

    return data, data_dim

# Function for displaying a centroid
def disp_centroid(a_centroid, data_dim): 
    (y_leng, x_leng, num_chan) = data_dim
    # Reshape vector into image
    img_arr = a_centroid.reshape((y_leng, x_leng, num_chan))
    img_arr = img_arr.astype('uint8')
    img = Image.fromarray(img_arr)
    img.show()
        

# === Main =========================================== 

# Data parameters
source_path = '/home/tomasmaul/Downloads/AI4E-Code/Data-c/Tree-Ant-Data-1-c/SourcePicsOlympus-c/'
target_path = '/home/tomasmaul/Downloads/AI4E-Code/Data-c/Tree-Ant-Data-1-c/TargetPics'
boundaries_path_file = '/home/tomasmaul/Downloads/AI4E-Code/Data-c/Tree-Ant-Data-1-c/Boundaries.txt'
scale_fac = 0.1 # given my laptop constraints 0.2 is probably a safe upper limit.

# Load data
data, data_dim = load_data_dir(source_path, scale_fac)

# Prepara data structures  

# Size of dataset to be generated. The final size is 4 * data_size
data_size = data.shape[0]
num_iters = 50
num_clusters = 4

# Shuffle the data
np.random.shuffle(data)

# Set random seed for reproducibility 
random.seed(0)

# Initialise centroids
centroids = data[random.sample(range(data.shape[0]), num_clusters)]

# Create a list to store which centroid is assigned to each dataset
assigned_centroids = np.zeros(len(data), dtype = np.int32)

def compute_l2_distance(x, centroid):
    # Compute the difference, following by raising to power 2 and summing
    dist = ((x - centroid) ** 2).sum(axis = x.ndim - 1)
    
    return dist

def get_closest_centroid(x, centroids):
    
    # Loop over each centroid and compute the distance from data point.
    dist = compute_l2_distance(x, centroids)

    # Get the index of the centroid with the smallest distance to the data point 
    closest_centroid_index =  np.argmin(dist, axis = 1)
    
    return closest_centroid_index

# DO --> debug/test this one
def compute_sse(data, centroids, assigned_centroids):
    # Initialise SSE 
    sse = 0

    # Compute SSE
    sse = compute_l2_distance(data, centroids[assigned_centroids]).sum() / len(data)
    
    return sse

# Number of dimensions in centroid
num_centroid_dims = data.shape[1]

# List to store SSE for each iteration 
sse_list = []

# Start time
tic = time.time()

# Main Loop
for n in range(num_iters):

    # Display iteration
    print('Iteration {0}'.format(n))

    # Get closest centroids to each data point
    assigned_centroids = get_closest_centroid(data[:, None, :], centroids[None,:, :])    
    
    # Compute new centroids
    for c in range(centroids.shape[0]):
        # Get data points belonging to each cluster 
        cluster_members = data[assigned_centroids == c]
        
        # Compute the mean of the clusters
        cluster_members = cluster_members.mean(axis = 0)
        
        # Update the centroids
        centroids[c] = cluster_members
    
    # Compute SSE
    sse = compute_sse(data.squeeze(), centroids.squeeze(), assigned_centroids)
    sse_list.append(sse)

# End time
toc = time.time()

print(round(toc - tic, 4)/50)
print(centroids)
print(sse_list)

# Display centroids
for c in range(centroids.shape[0]):
    disp_centroid(centroids[c,:], data_dim)


    
    
