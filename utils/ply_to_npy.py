# VOXELIZATION PARAMETERS

# grid resolution
N=30
# path to reference table
orientation_classes_path = '/home/ubuntu/nndl-project/utils/orientation_classes.csv'

# root folder for the ply files, in the structure of modelnet dataset
ply_root = '/home/ubuntu/nndl-project/data/modelnet10/ply/'

import os
os.chdir('/home/ubuntu/nndl-project/')

import tqdm
import pandas as pd
import numpy as np

from src.dataset import load_voxel_grid # voxelization function


orientation_classes = pd.read_csv(orientation_classes_path,index_col=0)

pbar = tqdm.tqdm(list(os.walk(ply_root)))

for root,subdirs,files in pbar:
    for f in files:
        if f[-4:] != '.ply' : continue
        
        label = root.split('/')[-2]
        destination_root = root.replace("/ply/", "/npy/")

        for index,row in orientation_classes[orientation_classes['label']==label].iterrows():

            pbar.set_description("Processing {}, orientation class {}".format(f,row['class_id']))

            # make dir for orientation class
            destination_path = os.path.join(destination_root,str(row['class_id']))
            if not os.path.exists(destination_path): os.makedirs(destination_path)

            # grab rotations
            rot_xyz = row[['rot_x','rot_y','rot_z']].values

            # voxelize
            array=load_voxel_grid(
                os.path.join(root,f), # path of the original ply mesh
                N,                    # resolution
                *rot_xyz,             # rotation applied before voxelization
                add_channel_dim=True)
            
            # save array
            file_destination_path = os.path.join(destination_path,f.split('.')[0]+'.npy')
            np.save(file_destination_path,array)
        
