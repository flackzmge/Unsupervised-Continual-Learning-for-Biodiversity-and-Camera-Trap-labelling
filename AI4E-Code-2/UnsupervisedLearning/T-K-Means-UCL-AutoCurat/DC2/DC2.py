# The k-means implementation was adapted from Ayoosh Kathuria's article here:
# https://blog.paperspace.com/speed-up-kmeans-numpy-vectorization-broadcasting-profiling/

# Version history
# v0.0.1 - Adapted Kathuria's code to auto-curation data. Simple k-means; no adaptive thresholds.
# v0.0.2 - Add code to visualize centroids.
# v0.0.3 - Add simple performance metrics.
# v0.0.4 - Improve initialization; bug fixes; tweaks.
# v0.0.5-test2 - Built from v0.0.4. Test different k in terms of clustering quality.
# v0.0.6 - Built from v0.0.5-test2. Added oher clustering quality measures.
# v0.0.7 - Based on v0.0.6. Use a pre-trained network for feature extraction.
# v0.0.8 - Fixed memory leak.
# v0.0.9 - Batch clustering version.
# v0.0.10 - Built from v0.0.9. Added Distance Metrics - Chebysev chosen
# v0.0.11 - Built from v0.0.10. Added Clustering Estimator - Davies Bouldin score chosen
from __future__ import division, print_function, absolute_import

import math

import numpy
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
from sklearn.utils import check_X_y
from sklearn.preprocessing import LabelEncoder
import torch
import torchvision
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter
import gc
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cityblock
from scipy.cluster.vq import kmeans, vq
from scipy.spatial import distance as spdist
from numpy.linalg import norm


# === 

# os.environ['LRU_CACHE_CAPACITY'] = '1'
# os.environ['LD_PRELOAD'] = './libjemalloc.so.1'

# === Prepare pre-trained network 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16_model = models.vgg16(pretrained=True)
vgg16_model.eval()
torch.no_grad()
for param in vgg16_model.parameters():
    param.requires_grad = False
#print(vgg16_model)

# --- Intermediate layers
#return_layers_1 = {'30': 'out_layer30'}
#mult_layer_1 = IntermediateLayerGetter(vgg16_model.features, return_layers=return_layers_1)


# Fully connected layers
# Chopping off final layers; using a fully-connect layer for feature extraction.
# vgg16_model.classifier = vgg16_model.classifier[:-1] # first layer before output layer
vgg16_model.classifier = vgg16_model.classifier[:-2] # second layer before output layer
num_feat = 4096
#print(vgg16_model)
# Testing ---
#x = torch.rand(5,3,512,512)
#y = vgg16_model(x)
#print('Que?')
# ------------

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

# # Function for loading the data from a single folder
# def load_data_dir(source_path, scale_fac):

#     # Source directory contents
#     dir_contents = os.listdir(source_path) # returns list
#     dir_contents.sort()

#     num_pics = len(dir_contents)

#     # Load first image to get dimensions, etc.
#     img_1 = Image.open(source_path + dir_contents[0])
#     img1_rs = rescale(img_1, scale_fac)
#     img1_rs = reorient(img1_rs)
#     (xleng, yleng) = img1_rs.size
#     num_chan = 3 # we assume that images are RGB
#     all_dim = xleng*yleng*num_chan 
#     data_dim = (yleng, xleng, num_chan)
#     # Initialize data 
#     #imgs_array = np.zeros((num_pics, all_dim))
#     imgs = []
    
#     for i in range(num_pics):
#         # Load and prepare images
#         print('Image name: ', dir_contents[i]);
#         img_1 = Image.open(source_path + dir_contents[i])
#         # img_1 = reorient(img_1)
#         imgs.append(img_1)
#         #img1_rs = rescale(img_1, scale_fac)
#         #img1_rs = reorient(img1_rs) # re-orient the image, if necessary
#         # img1_rs.show()
#         #img_1_arr = np.asarray(img1_rs)
#         #imgs_array[i,:] = img_1_arr.flatten()

#     return imgs, data_dim, num_pics

# Function for loading the data from a single folder
def load_data_dir(source_path, scale_fac):

    # Source directory contents
    dir_contents = os.listdir(source_path) # returns list
    dir_contents.sort()

    num_pics = len(dir_contents)

    # Load first image to get dimensions, etc.
    img_1 = Image.open(source_path + dir_contents[0])
    #img1_rs = rescale(img_1, scale_fac)
    img1_rs = img_1.resize((224, 224))
    img1_rs = reorient(img1_rs)
    (xleng, yleng) = img1_rs.size
    num_chan = 3 # we assume that images are RGB
    data_dim = (yleng, xleng, num_chan)
    # Initialize data 
    #imgs_array = np.zeros((num_pics, all_dim))
    imgs = torch.zeros((num_pics, num_chan, yleng, xleng))
    
    fromPILtoTensor = transforms.ToTensor()

    for i in range(num_pics):
        # Load and prepare images
        print('Image name: ', dir_contents[i]);
        img = Image.open(source_path + dir_contents[i])
        #img1_rs = rescale(img1, scale_fac)
        img_rs = img.resize((224, 224))
        img_rs = reorient(img_rs)
        img_tensor = fromPILtoTensor(img_rs)
        imgs[i,:,:,:] = img_tensor

    return imgs, data_dim, num_pics

# Design choice 7 (DC7). Different pre-trained networks for feature extraction (e.g. ImageNet vs. iNaturalist).
def extract_features(index, images, preprocess, vgg16_model, data):
    input_tensor = preprocess(images[index,:,:,:])
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model    
    output = vgg16_model(input_batch)
    # Store in data
    # Design choice 8 (DC8). Different feature extraction approaches (e.g. one or more layers; which layers; etc.).
    data[index,:] = output.detach().numpy()
    return data

# Convert images to features
def convert_imgs_to_feat(images, num_pics, num_features):

    # Initialize data tensor
    data = np.zeros((num_pics, num_features))
    # Initialize preprocessing
    preprocess = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        #transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Scan through images
    for img_i in range(num_pics):
        print('Converting image {0} to features ...'.format(img_i+1))
        # Convert to features
        data = extract_features(img_i, images, preprocess, vgg16_model, data)
        #torch.cuda.empty_cache()
        gc.collect() 
    
    return data

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
            tn += 1
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

def forgy_initialize(X, k):
    '''Return Randomly sampled points from the data'''
    return X[np.random.choice(range(X.shape[0]), replace = False,     size = k), :]







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
    #dist = ((x - centroid) ** 2)

    
    return dist

# Design choice 4 (DC4). Different distance metrics to be used within the chosen clustering algorithm.




def chebyshev(x, y):
    
    dist = (abs(x - y)).max(axis = x.ndim - 1 )

    return dist





def get_closest_centroid(x, centroids):


    # Loop over each centroid and compute the distance from data point.
    dist = chebyshev(x, centroids)


    # Get the index of the centroid with the smallest distance to the data point 
    closest_centroid_index =  np.argmin(dist, axis= 1)
    
    return closest_centroid_index

# DO --> debug/test this one
def compute_sse(data, centroids, assigned_centroids):
    # Initialise SSE 
    sse = 0

    # Compute SSE
    sse = chebyshev(data, centroids[assigned_centroids]).sum() / len(data)

    return sse

def check_number_of_labels(n_labels, n_samples):
    """Check that number of labels are valid.
    Parameters
    ----------
    n_labels : int
        Number of labels.
    n_samples : int
        Number of samples.
    """
    if not 1 < n_labels < n_samples:
        raise ValueError(
            "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)"
            % n_labels
        )

def calinski_harabasz_score(X, labels):
    """Compute the Calinski and Harabasz score.
    It is also known as the Variance Ratio Criterion.
    The score is defined as ratio between the within-cluster dispersion and
    the between-cluster dispersion.
    Read more in the :ref:`User Guide <calinski_harabasz_index>`.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    Returns
    -------
    score : float
        The resulting Calinski-Harabasz score.
    References
    ----------
    .. [1] `T. Calinski and J. Harabasz, 1974. "A dendrite method for cluster
       analysis". Communications in Statistics
       <https://www.tandfonline.com/doi/abs/10.1080/03610927408827101>`_
    """
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    check_number_of_labels(n_labels, n_samples)

    extra_disp, intra_disp = 0.0, 0.0
    mean = np.mean(X, axis=0)
    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        intra_disp += np.sum((cluster_k - mean_k) ** 2)

    return (
        1.0
        if intra_disp == 0.0
        else extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))
    )

# === Compute batch ==================================

# In "version 0", centroids is not being used as an argument 
def comp_batch(source_path, boundaries_path_file, scale_fac, centroids):
    
    # Load data
    print('Loading images ...')
    images, data_dim, num_pics = load_data_dir(source_path, scale_fac)
    # Design choice 1 (DC1) - specimen detection
    # Specimen detection (optional)
    # Feature extraction
    print('Converting images to features ...')
    data = convert_imgs_to_feat(images,num_pics,num_feat)

    # Prepara data structures  

    # Size of dataset to be generated. The final size is 4 * data_size
    data_size = data.shape[0]
    #num_iters = 50
    num_iters = 100
    #num_ks = [5, 10, 20, 30] # [5,10,20,30] # [5,10,15,20]
    #num_ks = [27, 28, 29, 30]
    num_ks = [5, 10, 10, 15]
    init_type = 2 # 1 = random; 2 = k-means++; 3 = forgy (random)

    # Set random seed for reproducibility 
    random.seed(0)
    db_score_list = []
    ch_score_list = []
    centroid_list = []

    # Design choice (DC2) - type of clustering algorithm
    for num_clusters in num_ks: # Design choice 6 (DC6). Different search procedures for optimal k (e.g. grid search is a baseline).
    
        # Initialise centroids
        print('Initializing centroids for k = {} ...'.format(num_clusters))
        # Design choice 3 (DC3). Different cluster initializations.
        if init_type == 1:
            centroids = init1(data, num_clusters)
        elif init_type == 2:
            centroids = init2(data, num_clusters)
        elif init_type == 3:
            centroids = forgy_initialize(data, num_clusters)




        # Create a list to store which centroid is assigned to each dataset
        assigned_centroids = np.zeros(len(data), dtype = np.int32)

        # Number of dimensions in centroid
        num_centroid_dims = data.shape[1]

        # List to store SSE for each iteration 
        sse_list = []

        # Start time
        tic = time.time()

        # Main Loop
        print('Computing {0} Iterations ...'.format(num_iters))
        for n in range(num_iters):

            # Display iteration
            #print('Iteration {0}'.format(n))

            # Get closest centroids to each data point
            assigned_centroids = get_closest_centroid(data[:, None, :], centroids[None,:, :])

            # Compute new centroids
            for c  in range(centroids.shape[0]):
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

        # Design choice 5 (DC5). Different clustering quality estimators (i.e. how to select the best k).
        db_score = davies_bouldin_score(data, assigned_centroids)
        ch_score = calinski_harabasz_score(data, assigned_centroids) # need to update scikit learn
        print('Davies Bouldin score for k = {0}: {1}'.format(num_clusters, db_score))
        print('Calinski Harabasz score for k = {0}: {1}'.format(num_clusters, ch_score))

        db_score_list.append(db_score)
        ch_score_list.append(ch_score)
        centroid_list.append(centroids)

    # Final results

    # Compute clustering scores
    print('List of ks: {0}'.format(num_ks))
    print('List of Davies Bouldin scores:')
    print(db_score_list)
    print('List of Calinski Harabasz scores:')
    print(ch_score_list)

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

    return centroids, assigned_centroids, pred_boundaries, true_boundaries, res_k_means


# === Main =========================================== 


# Data parameters
source_paths = []
#source_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-1-c/SourcePicsOlympus-c/')
#source_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-2-c/SourcePicsOlympus-c/')
#source_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-3-c/SourcePicsOlympus-c/')
#source_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-4-c/SourcePicsOlympus-c/')
#source_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-5-c/SourcePicsOlympus-c/')
#source_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-6-c/SourcePicsOlympus-c/')
#source_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-7-c/SourcePicsOlympus-c/')
#source_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-8-c/SourcePicsOlympus-c/')
#source_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-9-c/SourcePicsOlympus-c/')
#source_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-10-c/SourcePicsOlympus-c/')
# create a loop to append all the paths to the source_paths list from 1 to number of source paths
for i in range(1,11):
    source_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-{0}-c/SourcePicsOlympus-c/'.format(i))

#source_path = '/home/tomasmaul/Downloads/AI4E-Wrap-Up/AI4E-Code/Data-c/TestData-5clusters/Source/'
target_paths = []
#target_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-1-c/TargetPics')
#target_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-2-c/TargetPics')
#target_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-3-c/TargetPics')
#target_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-4-c/TargetPics')
#target_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-5-c/TargetPics')
#target_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-6-c/TargetPics')
#target_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-7-c/TargetPics')
#target_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-8-c/TargetPics')
#target_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-9-c/TargetPics')
#target_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-10-c/TargetPics')
# create a loop to append all the paths to the target_paths list from 1 to number of target paths
for i in range(1,11):
    target_paths.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-{0}-c/TargetPics'.format(i))
#target_path = '/home/tomasmaul/Downloads/AI4E-Wrap-Up/AI4E-Code/Data-c/TestData-5clusters/target/'
boundaries_path_files = []
#boundaries_path_files.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-1-c/Boundaries.txt')
#boundaries_path_files.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-2-c/Boundaries.txt')
#boundaries_path_files.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-3-c/Boundaries.txt')
#boundaries_path_files.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-4-c/Boundaries.txt')
#boundaries_path_files.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-5-c/Boundaries.txt')
#boundaries_path_files.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-6-c/Boundaries.txt')
#boundaries_path_files.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-7-c/Boundaries.txt')
#boundaries_path_files.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-8-c/Boundaries.txt')
#boundaries_path_files.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-9-c/Boundaries.txt')
#boundaries_path_files.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-10-c/Boundaries.txt')
# create a loop to append all the paths to the boundaries_path_files list from 1 to number of boundaries paths
for i in range(1,11):
    boundaries_path_files.append('/Users/nathangilbert/Documents/Computer Science MSc/Dissertation/AI4E-Code-2/Data-c/Tree-Ant-Data-{0}-c/Boundaries.txt'.format(i))

scale_fac = 0.02 # 0.05 # 0.1 # given my laptop constraints 0.2 is probably a safe upper limit.

# Processing bataches
num_batches = len(source_paths)
# Scan through batches
centroids = None
all_centroids = []
all_pred_boundaries = []
all_true_boundaries = []
all_res_k_means = []
all_assigned_centroids = []
for b_i in range(num_batches):
    # Compute batch
    source_path = source_paths[b_i]
    boundaries_path = boundaries_path_files[b_i] 
    # Design choice 9 (DC9). Different batch-clustering/UCL logic.
    centroids, assigned_centroids, pred_boundaries, true_boundaries, res_k_means = comp_batch(source_path, boundaries_path, scale_fac, centroids)
    all_centroids.append(centroids)
    all_pred_boundaries.append(pred_boundaries)
    all_true_boundaries.append(true_boundaries)
    all_res_k_means.append(res_k_means)
    all_assigned_centroids.append(assigned_centroids)

# Display all results
print('=== Final compiled results ==========')
for bi in range(num_batches):
    print('--- Results for batch {0} ---'.format(bi+1))
    a_res_k_means = all_res_k_means[bi]
    a_pred_boundaries = all_pred_boundaries[bi]
    assigned_centroids = all_assigned_centroids[bi]
    print('a_res_k_means: ', a_res_k_means)
    print('a_pred_boundaries: ', a_pred_boundaries)
    print('assigned_centroids: ', assigned_centroids)

# Compute inter-centroid distances
# Concatenate centroid matrices
all_centroid_concat = np.concatenate(all_centroids, axis=0)
# Compile sets of similar centroids
print('Computing inter-centroid distances ...')
all_distances = distance_matrix(all_centroid_concat, all_centroid_concat, 2)
all_distances = np.around(all_distances, decimals=0)
print('=== Centroid distances =============')
print('all_distances: ', all_distances)
# Note: this needs to be elaborated further with a distance threshold if we want to extract centroids
# that are potentially corresponding to the same species (this is not the priority yet (as of 07/05/22), since it is
# not part of the AI4E grant goals/requirements.)
