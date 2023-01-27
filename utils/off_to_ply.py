DATASET = 'modelnet10'

import os
import open3d as o3d
import tqdm

path = '/home/ubuntu/nndl-project/data/'+DATASET
off_path = path+'/off/'
ply_path = path+'/ply/'

pbar = tqdm.tqdm(os.walk(off_path))

for root, subdirs, files in pbar:
    files = [f for f in files if f[-4:] == '.off'] # only show .off files
    if len(files) == 0: continue # skip paths with no files
    label,split = root.split('/')[-2:]
    destination_path = ply_path+label+'/'+split+'/'
    if not os.path.exists(destination_path): os.makedirs(destination_path)
    
    pbar.set_description("Processing %s" % label)
     
    for file in files:
        if '.off' not in file: continue
        filename,extension = file.split('.')
        mesh = o3d.io.read_triangle_mesh(root+'/'+file)
        file_destination_path = destination_path+filename+'.ply'
        o3d.io.write_triangle_mesh(file_destination_path, mesh)
