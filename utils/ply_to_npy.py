DATASET = 'modelnet10'

N=30

import os
os.chdir('/home/ubuntu/nndl-project/')

from src.dataset import load_voxel_grid
import tqdm
import numpy as np
import pandas as pd

path = '/home/ubuntu/nndl-project/data/'+DATASET
ply_path = path+'/ply/'
npy_path = path+'/npy/'

orientation_classes_path = '/home/ubuntu/nndl-project/data/'+DATASET+'/orientation_classes.csv'
orientation_classes = pd.read_csv(orientation_classes_path)

pbar = tqdm.tqdm(list(os.walk(ply_path)))

for root,subdirs,files in pbar:
    files = [f for f in files if f[-4:] == '.ply'] # only show .ply files
    if len(files) == 0: continue # skip paths with no files
    label,split = root.split('/')[-2:]
    
    for file in files:
        for idx,o_class in orientation_classes[orientation_classes['label_str']==label].iterrows():
            rot_xyz = o_class[['rot_x','rot_y','rot_z']]
            o_class_int = idx % 4

            pbar.set_description("Processing {}, orientation class {}".format(file,o_class_int))

            # shape (1,N,N)
            array=load_voxel_grid(root+'/'+file,N,*rot_xyz,add_channel_dim=True)

            destination_path = npy_path+label+'/'+split+'/'+str(o_class_int)+'/'
            
            if not os.path.exists(destination_path): os.makedirs(destination_path)

            filename,extension = file.split('.')
            np.save(destination_path+filename+'.npy',array)
