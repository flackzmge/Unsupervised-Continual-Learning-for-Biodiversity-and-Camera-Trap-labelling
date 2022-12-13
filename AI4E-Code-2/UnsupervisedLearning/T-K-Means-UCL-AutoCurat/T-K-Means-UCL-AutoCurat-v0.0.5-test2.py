# The k-means implementation was adapted from Ayoosh Kathuria's article here:
# https://blog.paperspace.com/speed-up-kmeans-numpy-vectorization-broadcasting-profiling/

# Version history
# v0.0.1 - Adapted Kathuria's code to auto-curation data. Simple k-means; no adaptive thresholds.
# v0.0.2 - Add code to visualize centroids.
# v0.0.3 - Add simple performance metrics.
# v0.0.4 - Improve initialization
# v0.0.5-test2 - Use Dunn index to check clustering quality

import numpy as np 
import matplotlib.pyplot as plt 
import random
import time 
import os
import sys
from PIL import Image
from torchvision import transforms
import pandas as pd
from sklearn.metrics import davies_bouldin_score

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
    
    for i in range(num_pics):
        # Load and prepare images
        print('Image name: ', dir_contents[i]);
        img_1 = Image.open(source_path + dir_contents[i])
        img1_rs = rescale(img_1, scale_fac)
        img1_rs = reorient(img1_rs) # re-orient the image, if necessary
        # img1_rs.show()
        img_1_arr = np.asarray(img1_rs)
        data[i,:] = img_1_arr.flatten()

    return data, data_dim, num_pics

# Function for displaying a centroid
def disp_centroid(a_centroid, data_dim): 
    (y_leng, x_leng, num_chan) = data_dim
    # Reshape vector into image
    img_arr = a_centroid.reshape((y_leng, x_leng, num_chan))
    img_arr = img_arr.astype('uint8')
    img = Image.fromarray(img_arr)
    img.show()

# Counvert boundary lists into binary vectors
def binarize_boundaries(boundaries, num_elem):
    bin_vec = np.zeros(num_elem)
    num_boundaries = len(boundaries)
    for bi in range(num_boundaries):
        bin_vec[boundaries[bi]-1] = 1
    return bin_vec

# Compute performance metrics    
def compPerform1(pred_boundaries, actual_boundaries):
    # Compute largest boundary index
    num_pred_bound = len(pred_boundaries)
    if num_pred_bound > 0:
        max_bound = max(max(pred_boundaries),max(actual_boundaries))
    else:
        max_bound = max(actual_boundaries)
    # Counvert boundary lists into binary vectors
    bin_pred_bound = binarize_boundaries(pred_boundaries,max_bound)
    bin_actual_bound = binarize_boundaries(actual_boundaries,max_bound)
    # Scan boundary vectors
    tp = 0; fp = 0; tn = 0; fn = 0; # initializations
    # Compute TP, FP, TN, FN
    for bi in range(max_bound):
        # If actual==1 and pred==1, increment true positives
        if (bin_actual_bound[bi] == 1) and (bin_pred_bound[bi] == 1):
            tp += 1
        # If actual==1 and pred==0, increment false negatives
        if (bin_actual_bound[bi] == 1) and (bin_pred_bound[bi] == 0):
            fn += 1
        # If actual==0 and pred==1, increment false positives
        if (bin_actual_bound[bi] == 0) and (bin_pred_bound[bi] == 1):
            fp += 1
        # If actual==0 and pred==0, increment true negatives
        if (bin_actual_bound[bi] == 0) and (bin_pred_bound[bi] == 0):
            tp += 0
    # Display tp, fp, tn, fn
    #print('True positives: ', tp)
    #print('False positives: ', fp)
    #print('True negatives: ', tn)
    #print('False negatives: ', fn)
    # Compute precision and recall
    denom = (tp + fp)
    if denom > 0:
        precision = tp / denom
    else:
        precision = 0
    denom = (tp + fn)
    if denom > 0:
        recall = tp / denom
    else:
        recall = 0
    # Compute F1 score
    denom = (precision+recall)
    if denom > 0:
        f1 = 2 * ((precision*recall)/denom)
    else:
        f1 = 0
    # Return all metrics
    res = {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    return res
        
# === Initialization functions ======================

def init1(data, num_clusters):
    centroids = data[random.sample(range(data.shape[0]), num_clusters)]
    return centroids

def distance(p1, p2):
    return np.sum((p1 - p2)**2)

# Adapted from https://www.geeksforgeeks.org/ml-k-means-algorithm/
def init2(data, k):
    '''
    initialized the centroids for K-means++
    inputs:
        data - numpy array of data points
        k - number of clusters
    '''
    ## initialize the centroids list and add
    ## a randomly selected data point to the list
    centroids = []
    centroids.append(data[np.random.randint(
            data.shape[0]), :])
    #plot(data, np.array(centroids))
  
    ## compute remaining k - 1 centroids
    for c_id in range(k - 1):
         
        ## initialize a list to store distances of data
        ## points from nearest centroid
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize
             
            ## compute distance of 'point' from each of the previously
            ## selected centroid and store the minimum distance
            for j in range(len(centroids)):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)
             
        ## select data point with maximum distance as our next centroid
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []
        #plot(data, np.array(centroids))
    
    centroid_array = np.array(centroids)
    return centroid_array

# Compute the simplest version of the Dunn index based on simpler inter and intra cluster distances
# As of 23/04/22 this is a naive, preliminary, and inneficient implementation.
# Based on: https://python-bloggers.com/2022/03/dunn-index-for-k-means-clustering-evaluation/#:~:text=The%20Dunn%20index%20is%20calculated,further%20away%20from%20each%20other.
#def comp_dunn_index_1(data, centroids):
    # Compute "centroid linkage distance" for inter-cluster distances
    # "The centroid linkage distance is the distance between the centroids of two clusters"
    # Compute min inter-cluster distance
    # Compute "centroid diameter distance" for intra-cluster distances
    # "The centroid diameter distance measures the double average distance between a centroid of a 
    # cluster and all observations from the same cluster"
    # Compute max intra-cluster distance
    # Compute dunn index

# === K-means support functions ======================

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


# === Main =========================================== 

# Data parameters
source_path = '/home/tomasmaul/Downloads/AI4E-Wrap-Up/AI4E-Code/Data-c/Tree-Ant-Data-1-c/SourcePicsOlympus-c/'
#source_path = '/home/tomasmaul/Downloads/AI4E-Wrap-Up/AI4E-Code/Data-c/TestData-5clusters/Source/'
target_path = '/home/tomasmaul/Downloads/AI4E-Wrap-Up/AI4E-Code/Data-c/Tree-Ant-Data-1-c/TargetPics'
#target_path = '/home/tomasmaul/Downloads/AI4E-Wrap-Up/AI4E-Code/Data-c/TestData-5clusters/target/'
boundaries_path_file = '/home/tomasmaul/Downloads/AI4E-Wrap-Up/AI4E-Code/Data-c/Tree-Ant-Data-1-c/Boundaries.txt'
#boundaries_path_file = '/home/tomasmaul/Downloads/AI4E-Wrap-Up/AI4E-Code/Data-c/TestData-5clusters/Boundaries.txt'
scale_fac = 0.05 # 0.1 # given my laptop constraints 0.2 is probably a safe upper limit.

# Load data
data, data_dim, num_pics = load_data_dir(source_path, scale_fac)

# Prepara data structures  

# Size of dataset to be generated. The final size is 4 * data_size
data_size = data.shape[0]
num_iters = 50
num_ks = [5, 10] # [5,10,20,30] # [5,10,15,20]
init_type = 2 # 1 = random; 2 = k-means++

# Set random seed for reproducibility 
random.seed(0)
db_score_list = []
centroid_list = []

for num_clusters in num_ks:

    # Initialise centroids
    print('Initializing centroids for k = {} ...'.format(num_clusters))
    if init_type == 1:
        centroids = init1(data, num_clusters)
    elif init_type == 2:
        centroids = init2(data, num_clusters)

    # Create a list to store which centroid is assigned to each dataset
    assigned_centroids = np.zeros(len(data), dtype = np.int32)

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

    # Compute clustering quality
    #dunn_index = comp_dunn_index_1(data, centroids)
    #print('Dunn index for k = {0}: {1}'.format(num_clusters, dunn_index))

    db_score = davies_bouldin_score(data, assigned_centroids)
    print('Davies Bouldin score for k = {0}: {1}'.format(num_clusters, db_score))
    db_score_list.append(db_score)
    centroid_list.append(centroids)

# Final results

# Compute clustering scores
print('List of ks: {0}'.format(num_ks))
print('List of Davies Bouldin scores:')
print(db_score_list)

# Select best k
best_k_i = np.argmin(db_score_list)
best_k = num_ks[best_k_i]
print('Selected k: {0}'.format(best_k))

# Select centroids for best k
centroids = centroid_list[best_k_i]

# Compute a final assigned_centroids for observation segmentations
# Key idea: adjacent cluster changes correspond to observation changes.
assigned_centroids = get_closest_centroid(data[:, None, :], centroids[None,:, :]) 
print('=== Centroid assignments ======== ')
print(assigned_centroids)
print('============================ ')

# Compute differences with adjacent centroid assignements
print('=== Observation segmentations ======== ')
diff_assign = assigned_centroids[0:-1]-assigned_centroids[1:]
new_observat = (diff_assign != 0) 
print('diff_assign: ', diff_assign)
print('new_observat: ', new_observat)
print('============================ ')

# Compute predicted boundaries
indices = list(range(1,num_pics))
pred_boundaries = [x for i, x in enumerate(indices) if new_observat[i]]
print('pred_boundaries: ', pred_boundaries)

# Load true boundaries
boundaries_df = pd.read_csv(boundaries_path_file)
bound_strings = boundaries_df.columns.tolist()
num_bound = len(bound_strings)
true_boundaries = []
for i in range(num_bound):
    true_boundaries.append(int(bound_strings[i]))
print('true_boundaries: ', true_boundaries)

# Compute performance metrics
res_k_means = compPerform1(pred_boundaries, true_boundaries)
print('Performance for k-means:')
print(res_k_means)

# Display centroids
#for c in range(centroids.shape[0]):
#    disp_centroid(centroids[c,:], data_dim)


    
    
