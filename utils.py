# pip install -i https://test.pypi.org/simple/ visualclustering
import numpy as np
from sklearn.metrics import pairwise_distances
import plotly.express as ex
import matplotlib.pyplot as plt
import random
from visualclustering.iVAT import iVAT
from visualclustering.VAT import VAT
import cv2
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import scipy
import time 

def display_image(image):
    fig = ex.imshow(image, color_continuous_scale='gray')
    fig.update_layout(  
    margin=dict(l=30, r=30, t=30, b=30),
    paper_bgcolor="LightSteelBlue",)
    fig.show()

def generate_data(n,max_clusters,min_clusters,target,mean_var):
    train_data=[]
    cut_values = []
    gt_num_clusters = []
    data_matrices = []
    reorderedindices =[]

    for _ in range(n):

        # generate a random number from min_clusters to max_clusters
        num = random.sample(range(min_clusters,max_clusters),1)[0]
        gt_num_clusters.append(num)

        # generate random num numbers such that they add upto target
        points = random.sample(range(int(0.1*target),target),num)
        mf = target/sum(points)
        points = [int(i*mf) for i in points]  # multiply by mf to get the actual points

        # if the sum of the points is less than target, add deficent value to the any random point
        if sum(points)!=target:
            points[random.sample(range(0,num),1)[0]]+=target-sum(points)


        data_matrix=[]
        full_data= []
        m_v = random.sample(range(0,len(mean_var)),num) # randomly select num clusters from mean_var
        
        for i in range(num):
            meanx,varx,meany,vary = mean_var[m_v[i]]
            data_points_without_labels = CS_2D_data_generate(meanx,varx,meany,vary,points[i])
            data_points_with_labels = np.concatenate((data_points_without_labels,np.ones((points[i],1))*i),axis=1)
            data_matrix.append(data_points_without_labels)
            if i==0: 
                full_data = data_points_with_labels
            else:
                full_data = np.concatenate((full_data,data_points_with_labels),axis=0)

        data_matrices.append(full_data)
        
        # concatenate all the matrices in data_matrix
        data_matrix = np.concatenate(data_matrix,axis=0)
        data_matrix = pairwise_distances(data_matrix,metric='euclidean') 
        res = iVAT(data_matrix)
        # reorderedindices.append(VAT(data_matrix)[2])
        reorderedindices.append(res[4])
        train_data.append(res[0])
        cut_values.append(res[3])

    return train_data, cut_values, gt_num_clusters, data_matrices, reorderedindices 


def MM(x, cp):
    n, p = x.shape
    m = np.ones(cp)
    # d = np.sqrt(np.sum((x-x[0])**2, axis=1))
    d = np.linalg.norm(x-x[0], axis=1, ord=2) # ord=2 is for euclidean distance
    Rp = np.zeros((n, cp))
    Rp[:,0] = d
    for t in range(1, cp):
        d = np.minimum(d, Rp[:,t-1])
        m[t] = np.argmax(d)
        # Rp[:,t] = np.sqrt(np.sum(((x[int(m[t])] - x)**2), axis=1))
        Rp[:,t] = np.linalg.norm(x[int(m[t])] - x, axis=1)
    return m, Rp



def MMRS(x, cp, ns):
    n, p = x.shape
    m, rp = MM(x, cp)
    i = np.argmin(rp, axis=1)
    smp = []
    for t in range(cp):
        s = np.where(i==t)[0]
        nt = (np.ceil(ns*len(s)/n)).astype('int')
        # randomly sample nt points from s
        ind = random.sample(range(len(s)), nt)
        # ind = np.random.choice(s, size=nt, replace=False)
        smp.append(s[ind])
    smp = [item for sublist in smp for item in sublist]
    smp = list(set(smp))
    return smp, rp, m


from sklearn.metrics.pairwise import euclidean_distances

def clusivat(x, cp, ns):
    """ 
    x: data
    cp: number of clusters (over-estimated)
    ns: number of samples required from data
    """
    # Sample data and obtain cluster information
    smp, rp, m = MMRS(x, cp, ns)
    # Compute pairwise distances between the sampled data
    rs = euclidean_distances(x[smp], x[smp], squared=True)
    # Run iVAT on the sampled data
    rv, C, I, ri, cut = iVAT(rs)
    return rv, C, I, ri, cut, smp


def myKNN(S, K, sigma=1.0):
    N = len(S)
    A = np.zeros((N,N))

    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(K+1)] # xi's k nearest neighbours

        for j in neighbours_id: # xj is xi's neighbour
            A[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
            A[j][i] = A[i][j] # mutually

    return A


def hough_transform(image,low_th = 100, high_th=200, apeture_size=3, L2gradient=True):

    min_cluster_size = int(0.01*image.shape[0])
    im1 = image.copy()
    im2 = image.copy()


    #smooth the image
    image = cv2.GaussianBlur(image,(3,3),0)

    # pad the image with white pixels to avoid boundary effects
    width = 5
    image = np.pad(image, ((width, width), (width, width), (0, 0)), mode='constant', constant_values= np.max(image))


    # Convert image to grayscale
    if len(image.shape)==2 : 
        gray = image
    else:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    gray =  255 * ((gray - np.min(gray)) / (np.max(gray) - np.min(gray)))
    gray = gray.astype('uint8')

    # Apply Canny edge detection
    edges = cv2.Canny(gray,low_th,high_th,apertureSize=apeture_size,L2gradient=L2gradient)
    
    # Threshold the edges to get a binary image
    edges = np.where(edges>0, 255, 0)
    edges = edges.astype(np.uint8)

    # display_image(edges) # display the edge map

    # Apply Hough Transform: This returns an array of r and theta values
    lines = cv2.HoughLines(edges, 0.1, np.pi/180, min_cluster_size)  

    # Filter lines output to get the indices where angle is between 89.5 and 90.5 degrees or less than 0.5 degrees
    indices = np.where((lines[:,:,1] > (89.5* np.pi / 180)) & (lines[:,:,1] < (90.5* np.pi / 180)) | (lines[:,:,1] < (0.5* np.pi / 180)))[0]
    lines = lines[indices]


    # # Draw lines on the image
    # for r_theta in lines:
    #     r, theta = r_theta[0]
    #     r= r - width
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a*r
    #     y0 = b*r
    #     x1 = int(x0 + 1000*(-b))
    #     y1 = int(y0 + 1000*(a))
    #     x2 = int(x0 - 1000*(-b))
    #     y2 = int(y0 - 1000*(a))

    #     cv2.line(im1,(x1,y1),(x2,y2),(0,255,0),2)

    # display_image(im1) # display the image with the baseline dark blocks identified

    labels= [int(r_theta[0][0] )for r_theta in lines]

    pred = np.array([1 if i in labels else 0 for i in range(image.shape[0])]) # create a binary array of 0s and 1s where 1s are the baseline dark blocks

    indices = np.where(pred==1)[0] # get the indices where the value is 1 (i.e. the baseline dark blocks corners)

    # # add 0 and image.shape[0] to the indices array if not present
    # if indices[0]!=0: indices = np.insert(indices, 0, 0)
    # if indices[-1]!= image.shape[0]-1: indices = np.insert(indices, len(indices),image.shape[0]-1)
    # print(indices)
    indices = indices - indices[0] - 1 # remove the padding from the indices
    indices[0] = 0


    # Draw lines on the image to display the baseline dark blocks identified
    for i in range(len(indices)-1):
        x,y  = indices[i], indices[i+1]
        if y-x> min_cluster_size:
            cv2.rectangle(im2, (x,x), (y,y), (255, 0, 0), 2)
    # display_image(im2)

    return indices



def cluster_hierarchy(image, cut, indice,min_clus=0.01):
    # max_score = -np.inf
    min_cluster_size = int(min_clus*image.shape[0])
    max_cluster_size = 0.5
    width = 5

    # pruning indices
    new=[]
    for i in range(len(indice)):
        idx = indice[i]
        a,b = max(idx-3,0),min(idx+3,image.shape[1])
        new.append(np.argmax(cut[a:b])+a)
    # remove duplicates
    new = list(dict.fromkeys(new))
    # sort the corners
    new =  sorted(new)
    new[0] = 0
    indices = new
    # print("new indices", indices)


    im = np.pad(image, ((width, width), (width, width), (0, 0)), mode='constant', constant_values= int(0.5*np.max(image)))
    # indices = merge_small_dark_blocks(image, indice, min_cluster_size, cut)

    sum = 0
    for i in range(1, len(indices)):
        x, y = indices[i-1], indices[i]
        if y - x > min_cluster_size:
            x, y = x+width, y+width
            left_part = im[x:y, y-width+1:y+1, 0]
            right_part = im[x:y, y+1:y+width+1, 0]
            sum += np.abs(left_part.mean() - right_part.mean())*min(max_cluster_size,((y-x)/image.shape[0]))
    
    max_score = abs(sum)
    max_indices = indices

    # print("Score for this partition: ", abs(sum))


    while len(indices) > 2:
        # Finding the index of the darkest block based on the cut
        res = np.inf
        for i in range(1, len(indices) - 1, 1):
            x = indices[i]
            a, b = max(x - 3, 0), min(x + 3, image.shape[0])
            cut_val = np.max(cut[a:b])
            if cut_val < res:
                res = cut_val
                idx = i


        new_indices = np.concatenate([indices[:idx], indices[idx+1:]])  # Merging dark blocks that are too close based on the cut
        # print("new_indices", new_indices)

        count = 0
        im_copy = image.copy()
        ind = []
        label = []
        for i in range(len(new_indices)-1):
            x,y  = new_indices[i], new_indices[i+1]
            if y-x > min_cluster_size:
                ind.append(np.arange(x,y))
                label.append(count)
                count += 1
        #         cv2.rectangle(im_copy, (x,x), (y,y), (255, 0, 0), 2)
        # display_image(im_copy)

        
        # Score calculation for each partition
        if len(np.unique(label)) > 1:
            sum = 0
            for i in range(1, len(new_indices)):
                x, y = new_indices[i-1], new_indices[i]
                if y - x > min_cluster_size:
                    x, y = x+width, y+width
                    left_part = im[x:y, y-width+1:y+1, 0]
                    right_part = im[x:y, y+1:y+width+1, 0]
                    sum += np.abs(left_part.mean() - right_part.mean())*min(max_cluster_size,((y-x)/image.shape[0]))
            # print("Score for this partition: ", abs(sum))
            # print("indices", new_indices)
            if abs(sum) > max_score: # capture the best score
                max_indices = new_indices
                max_score = abs(sum)
                # print("New max score: ", max_score)

        indices = new_indices


    # Optimal partitioning of the image
    final_indices = []
    cluster_count = 0
    for i in range(len(max_indices)-1):
        x, y = max_indices[i], max_indices[i+1]
        if y - x > min_cluster_size:
            final_indices.append(x)
            final_indices.append(y)
            cluster_count += 1
            cv2.rectangle(image, (x, x), (y, y), (255, 0, 0), 3)
    display_image(image)

    return image, final_indices, cluster_count



def f1(indices,gt,reodered_ind,data):
    pred = []
    act = []
    final_data = []
    count = 1
    for i in range(0,len(indices)-1,2):
        if i==0:
            a,b = indices[i],indices[i+1]
            id = reodered_ind[a:b+1]
            act.extend(gt[id]+1)
            pred.extend([count]*(len(gt[id]+1)))
            final_data.extend(data[id])
        else:
            a,b = indices[i],indices[i+1]
            id = reodered_ind[a+1:b+1]
            act.extend(gt[id]+1)
            pred.extend([count]*(len(gt[id]+1)))
            final_data.extend(data[id])
        count+=1

    pred = np.array(pred).T
    act = np.array(act).T
    final_data = np.array(final_data)
    return pred,act,final_data


def automated_evaluation(ivat_img,gt,reodered_ind,data,cut,low_th=10,high_th=100,min_cluster_size=0.01,flag=True):
    start = time.time()
    if flag: 
        pred = hough_transform(ivat_img,low_th=low_th,high_th=high_th)
    else:
        pred = np.arange(len(ivat_img))
    partitioned_img,indices,cluster_count = cluster_hierarchy(ivat_img, cut, pred, min_clus= min_cluster_size)
    end = time.time()
    pred,act,final_data = f1(indices,gt,reodered_ind,data)

    # calculate evaluation metrics 
    nmi = normalized_mutual_info_score(act,pred)
    ari = adjusted_rand_score(act,pred)

    print("NMI: {:.2f} | ARI: {:.2f} | Time taken: {:.2f} s | Number of clusters: {:.2f}".format(nmi, ari, end-start, cluster_count))

    return partitioned_img, cluster_count, indices, final_data, pred, act
