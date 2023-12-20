import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
import os
import torch
import json
from torch.autograd import Variable

#DEFINE YOUR DEVICE
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

a = 100 #positive and negative range for object points
num_points = 50
monte_carlo = 100
train_data_length = 100

cam1_origin = np.random.uniform(-50,50,(3,1))
alpha1 = np.random.uniform(-math.pi,math.pi)
beta1 = np.random.uniform(-math.pi,math.pi)
gama1 = np.random.uniform(-math.pi,math.pi)
#rotation matrices for each axis
rot1_x = np.array([[1, 0, 0],[0, math.cos(alpha1), -math.sin(alpha1)],[0, math.sin(alpha1), math.cos(alpha1)]])
rot1_y = np.array([[math.cos(beta1), 0, math.sin(beta1)],[0, 1, 0],[-math.sin(beta1), 0, math.cos(beta1)]])
rot1_z = np.array([[math.cos(gama1), -math.sin(gama1), 0],[math.sin(gama1), math.cos(gama1), 0],[0, 0 ,1]])
rot1_matrix = rot1_z @ (rot1_y @ rot1_x)
#world to camera frame transformation matrix
world_to_cam1_mat = np.hstack((rot1_matrix, cam1_origin))
f = np.random.uniform(0,10)  # focal length of the camera
cam_mat = np.array([[f,0,0],[0,f,0],[0,0,1]])
#overall transformation matrix
transform_matrix1 = cam_mat @ world_to_cam1_mat

sigmas = np.arange(0.0,2.0,0.05)
noise_length = len(sigmas)
alpha1deg = np.degrees(alpha1)
beta1deg = np.degrees(beta1)
gama1deg = np.degrees(gama1)

#random point homogenous coordinates
obj_points_homo = np.vstack((np.random.uniform(-a,a,(3,num_points)),np.ones((1,num_points))))
proj1_points = transform_matrix1 @ obj_points_homo
proj1_points = proj1_points/proj1_points[-1]
obj_points = obj_points_homo.T[:,:-1] #non-homogenous object points
img_points = proj1_points[:-1]
pose_errors = np.zeros((6,monte_carlo*noise_length*train_data_length))
all_points = np.zeros((5*num_points,monte_carlo*noise_length*train_data_length))
flat_obj = obj_points.flatten()

def add_gaussnoise(points, sigma):
    L = len(points[0])
    noisy_points = np.zeros([2,L])
    for i in range(L):
        noisy_points[:,i] = points[:,i] + np.random.normal(0, sigma, size = (2,))
    return noisy_points

for r in range(train_data_length):
    for k in range(monte_carlo):
        print("Monte Carlo is: "+str(k/monte_carlo)+" and Dataset is at: "+str(r/train_data_length))
        for i in range(noise_length):
            index = r*monte_carlo*noise_length+k*noise_length+i
            noisy_points = add_gaussnoise(img_points, sigmas[i])
            all_points[:3*num_points,index] = flat_obj
            all_points[3*num_points:,index] = noisy_points.T.flatten()

            _, rvec, tvec, inliers = cv2.solvePnPRansac(obj_points,noisy_points.T,cam_mat,distCoeffs=np.zeros((4, 1)),flags=cv2.SOLVEPNP_ITERATIVE)
            temp_rotation_mat, _ = cv2.Rodrigues(rvec)
            pose_mat = cv2.hconcat((temp_rotation_mat, tvec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            pose_errors[0,index] = euler_angles[0]-alpha1deg
            pose_errors[1,index] = euler_angles[1]-beta1deg
            pose_errors[2,index] = euler_angles[2]-gama1deg
            pose_errors[3,index] = tvec[0]-cam1_origin[0]
            pose_errors[4,index] = tvec[1]-cam1_origin[1]
            pose_errors[5,index] = tvec[2]-cam1_origin[2]


all_points = all_points/(abs(all_points).max())

extract = int(train_data_length*monte_carlo*noise_length*26/100)
permut = np.random.permutation(all_points.shape[1])
points = all_points[:,permut]
errors = pose_errors[:,permut]
sel_points = points[:,:extract]
sel_errors = errors[:,:extract]

train_data = torch.FloatTensor(sel_points[:,:int(train_data_length*monte_carlo*noise_length*25/100)])
train_labels = torch.FloatTensor(sel_errors[:,:int(train_data_length*monte_carlo*noise_length*25/100)])
val_data = torch.FloatTensor(sel_points[:,int(train_data_length*monte_carlo*noise_length*25/100):])
val_labels = torch.FloatTensor(sel_errors[:,int(train_data_length*monte_carlo*noise_length*25/100):])

# Alternative for mapped input data
# L = 5
# coords = np.zeros((5*num_points*2*L,extract))
# for k in range(extract):
#   print('{} over {} is done.'.format(str(k),str(extract)))
#   for i in range(5*num_points):
#     coords[i*2*L,k] = math.sin(2**0*math.pi*sel_points[i,k])
#     coords[i*2*L+1,k] = math.cos(2**0*math.pi*sel_points[i,k])
#     coords[i*2*L+2,k] = math.sin(2**1*math.pi*sel_points[i,k])
#     coords[i*2*L+3,k] = math.cos(2**1*math.pi*sel_points[i,k])
#     coords[i*2*L+4,k] = math.sin(2**2*math.pi*sel_points[i,k])
#     coords[i*2*L+5,k] = math.cos(2**2*math.pi*sel_points[i,k])
#     coords[i*2*L+6,k] = math.sin(2**3*math.pi*sel_points[i,k])
#     coords[i*2*L+7,k] = math.cos(2**3*math.pi*sel_points[i,k])
#     coords[i*2*L+8,k] = math.sin(2**4*math.pi*sel_points[i,k])
#     coords[i*2*L+9,k] = math.cos(2**4*math.pi*sel_points[i,k])

# train_data = torch.FloatTensor(coords[:,:int(train_data_length*monte_carlo*noise_length*25/100)])
# train_labels = torch.FloatTensor(sel_errors[:,:int(train_data_length*monte_carlo*noise_length*25/100)])
# val_data = torch.FloatTensor(coords[:,int(train_data_length*monte_carlo*noise_length*25/100):])
# val_labels = torch.FloatTensor(sel_errors[:,int(train_data_length*monte_carlo*noise_length*25/100):])